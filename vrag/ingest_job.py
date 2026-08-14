"""Server-side ingest job: runs ingest.py as a subprocess and parses its log
stream into the structured state the ingest UI renders.

One job at a time.  The job lives in the server process; the browser only
polls /api/ingest/status, so closing the tab never touches the pipeline.
Every number shown in the UI comes from a real log line parsed here — the UI
invents nothing.
"""
from __future__ import annotations

import logging
import pathlib
import re
import subprocess
import sys
import threading
import time
from collections import deque

from vrag import config
from vrag.cache import read_json

log = logging.getLogger(__name__)

STAGES = [
    ("preflight", "Preflight"),
    ("audio", "Audio"),
    ("transcribe", "Transcription"),
    ("frames", "Frames & vision"),
    ("attribute", "Speaker attribution"),
    ("faceverify", "Face verification"),
    ("index", "Search index"),
    ("graph", "Knowledge graph"),
    ("proxy", "Browser proxy"),
]
IDX = {k: i for i, (k, _) in enumerate(STAGES)}
# ingest.py's "[N/8]" markers map to stages 1..8 here (0 is preflight).
MARKER_TO_STAGE = {1: "audio", 2: "transcribe", 3: "frames", 4: "attribute",
                   5: "faceverify", 6: "index", 7: "graph", 8: "proxy"}
# rough $ per call by namespace, for the "est. spend" ticker (final numbers
# come from the pipeline's own usage summary).
CALL_COST = {"transcribe": 0.012, "vision": 0.009, "faceverify": 0.007,
             "kg-extract": 0.007, "embed-batch": 0.0003, "roster-tamil": 0.004}

_TS = re.compile(r"^\d\d:\d\d:\d\d\s+(\w+)\s+(\S+)\s+(.*)$")


class IngestJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.state = "idle"          # idle | running | failed | done
        self.video_id = None
        self.video_path = None
        self.duration = None
        self.started_at = None
        self.ended_at = None
        self.error = None
        self.proc = None
        self.log = deque(maxlen=500)
        self.calls = 0
        self.est_cost = 0.0
        self.final = {}              # tokens / calls / cache from READY line
        self.warnings = []
        self.events = []             # notable moments (re-attributions etc.)
        self.stages = {k: {"key": k, "title": t, "status": "wait", "pct": 0.0,
                           "detail": {}} for k, t in STAGES}
        self.current = None
        self._prog_counts = {}       # namespace -> last-seen k

    # ------------------------------------------------------------------ run
    def start(self, video_path: pathlib.Path, video_id: str) -> None:
        with self.lock:
            if self.state == "running":
                raise RuntimeError("an ingest is already running")
            self.reset()
            self.state = "running"
            self.video_path = str(video_path)
            self.video_id = video_id
            self.started_at = time.time()
            self.stages["preflight"]["status"] = "run"
            self.current = "preflight"
        cmd = [sys.executable, "-u", str(config.ROOT / "ingest.py"),
               str(video_path), "--video-id", video_id]
        self.proc = subprocess.Popen(
            cmd, cwd=str(config.ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in self.proc.stdout:
                try:
                    self._parse(line.rstrip("\n"))
                except Exception:                          # noqa: BLE001
                    log.exception("ingest parser choked on: %r", line)
            code = self.proc.wait()
            with self.lock:
                if self.state == "running":
                    if code == 0:
                        self._finish_ok()
                    else:
                        self.state = "failed"
                        if not self.error:
                            tail = [e["txt"] for e in list(self.log)[-6:]]
                            self.error = "\n".join(tail) or f"exit code {code}"
                        if self.current:
                            self.stages[self.current]["status"] = "fail"
                self.ended_at = time.time()
        except Exception as exc:                           # noqa: BLE001
            with self.lock:
                self.state = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
                self.ended_at = time.time()

    def _finish_ok(self):
        self.state = "done"
        for s in self.stages.values():
            if s["status"] in ("run", "wait"):
                s["status"] = "done"
                s["pct"] = 100.0

    # ---------------------------------------------------------------- parse
    def _append_log(self, tag: str, txt: str):
        self.log.append({"t": round(time.time() - self.started_at, 1),
                         "tag": tag[:6].upper(), "txt": txt[:220]})

    def _enter(self, key: str):
        prev = self.current
        if prev and self.stages[prev]["status"] == "run":
            self.stages[prev]["status"] = "done"
            self.stages[prev]["pct"] = 100.0
        self.current = key
        st = self.stages[key]
        if st["status"] != "done":
            st["status"] = "run"

    def _parse(self, raw: str):
        m = _TS.match(raw)
        level, logger_name, msg = (m.group(1), m.group(2), m.group(3)) if m \
            else ("INFO", "raw", raw)
        with self.lock:
            self._append_log(logger_name.split(".")[-1], msg)
            if level in ("ERROR",) or "Traceback" in raw or raw.startswith("vrag."):
                # keep the most informative failure text
                self.error = ((self.error + "\n") if self.error else "") + msg[:400]
                self.error = self.error[-1200:]

            # stage markers from ingest.py: "[3/8] frames  ..."
            sm = re.match(r"\[(\d)/8\]\s+(\w+)\s+(.*)", msg)
            if sm:
                key = MARKER_TO_STAGE.get(int(sm.group(1)))
                if key:
                    self._enter(key)
                    if "SKIP" in sm.group(3):
                        self.stages[key]["status"] = "done"
                        self.stages[key]["pct"] = 100.0
                        self.stages[key]["detail"]["skipped"] = True
                return

            if "READY in" in msg:
                fm = re.search(r"network calls=(\d+) tokens=([\d,]+)", msg)
                if fm:
                    self.final = {"calls": int(fm.group(1)),
                                  "tokens": int(fm.group(2).replace(",", ""))}
                return

            # preflight detail (logged by ingest.py)
            pm = re.match(r"preflight: (.*)", msg)
            if pm:
                self.stages["preflight"]["detail"].setdefault("checks", []).append(pm.group(1))
                return
            if msg.startswith("INGEST "):
                self.stages["preflight"]["status"] = "run"
                return

            d_aud = self.stages["audio"]["detail"]
            d_asr = self.stages["transcribe"]["detail"]
            d_frm = self.stages["frames"]["detail"]
            d_att = self.stages["attribute"]["detail"]
            d_fv = self.stages["faceverify"]["detail"]
            d_kg = self.stages["graph"]["detail"]

            mm = re.search(r"audio: .* is ([\d.]+)s", msg)
            if mm:
                self.duration = float(mm.group(1))
            mm = re.search(r"audio: (\d+) silences, ([\d.]+)s total \((\d+)% of the clip is pause\)", msg)
            if mm:
                d_aud.update(silences=int(mm.group(1)),
                             speech_pct=100 - int(mm.group(3)))
                self.stages["audio"]["pct"] = max(self.stages["audio"]["pct"], 40)
            mm = re.search(r"audio: wrote .*\((\d+) chunks, (\d+) silences\)", msg)
            if mm:
                d_aud["chunks"] = int(mm.group(1))
                self.stages["audio"]["pct"] = 100.0
            mm = re.search(r"audio: chunk (\d+)", msg)
            if mm:
                d_aud["chunks"] = int(mm.group(1)) + 1
                self.stages["audio"]["pct"] = max(self.stages["audio"]["pct"], 60)

            # thread_map progress: "<desc> k/N (..%)"
            mm = re.search(r"^(transcribe|vision|faceverify|kg-extract|embed-batch|roster-tamil) (\d+)/(\d+) \(", msg)
            if mm:
                ns, k, n = mm.group(1), int(mm.group(2)), int(mm.group(3))
                last = self._prog_counts.get(ns, 0)
                if k > last:
                    self.calls += (k - last)
                    self.est_cost += CALL_COST.get(ns, 0.004) * (k - last)
                    self._prog_counts[ns] = k
                stage = {"transcribe": "transcribe", "vision": "frames",
                         "faceverify": "faceverify", "kg-extract": "graph",
                         "embed-batch": "index"}.get(ns)
                if stage:
                    self.stages[stage]["pct"] = 100.0 * k / n
                    self.stages[stage]["detail"]["batch"] = [k, n]
                return

            mm = re.search(r"span (\S+): (\d+) segments", msg)
            if mm:
                d_asr["segments"] = d_asr.get("segments", 0) + int(mm.group(2))
            mm = re.search(r"span \S+ last: (.*)", msg)
            if mm:
                d_asr["live_line"] = mm.group(1)
            if "clock ran" in msg or "overshoot" in msg:
                self.warnings.append(msg[:160])
                d_asr["warnings"] = len([w for w in self.warnings if "clock" in w or "overshoot" in w])
            mm = re.search(r"transcribe: (\d+) segments, (\d+) numeric flags", msg)
            if mm:
                d_asr.update(segments=int(mm.group(1)), numeric_flags=int(mm.group(2)))

            mm = re.search(r"frames: (\d+) scene cuts", msg)
            if mm:
                d_frm["cuts"] = int(mm.group(1))
            mm = re.search(r"frames: planned (\d+) keyframes", msg)
            if mm:
                d_frm["planned"] = int(mm.group(1))
            mm = re.search(r"frames: extracted (\d+) JPEGs", msg)
            if mm:
                d_frm["extracted"] = int(mm.group(1))
            mm = re.search(r"frames: (\d+)/(\d+) frames carry a chyron name, (\d+) show a standing", msg)
            if mm:
                d_frm.update(chyrons=int(mm.group(1)), standing=int(mm.group(3)))

            mm = re.search(r"attribute: (\d+) voice turns from (\d+) segments", msg)
            if mm:
                d_att["turns_raw"] = int(mm.group(1))
            mm = re.search(r"attribute: (\d+) introduction announcements", msg)
            if mm:
                d_att["announcements"] = int(mm.group(1))
            mm = re.search(r"attribute: (\d+)/(\d+) turns named \((\d+)% of speech time\), cast of (\d+)", msg)
            if mm:
                d_att.update(named=int(mm.group(1)), turns=int(mm.group(2)),
                             attributed_pct=int(mm.group(3)), cast=int(mm.group(4)))
                self.stages["attribute"]["pct"] = 90.0

            if "re-attributed" in msg:
                self.events.append(msg[:200])
                d_fv["reid"] = [e for e in self.events if "re-attributed" in e]
            mm = re.search(r"faceverify: (\{.*\})", msg)
            if mm:
                d_fv["verdicts"] = mm.group(1)

            mm = re.search(r"retrieve: indexed (\d+) segments \((\d+)d", msg)
            if mm:
                self.stages["index"]["detail"].update(vectors=int(mm.group(1)),
                                                      dim=int(mm.group(2)))
            mm = re.search(r"kg: (\d+) nodes, (\d+) edges", msg)
            if mm:
                d_kg.update(nodes=int(mm.group(1)), edges=int(mm.group(2)))

            # ffmpeg -progress lines during proxy
            mm = re.match(r"out_time_ms=(\d+)", msg)
            if mm and self.duration:
                pct = min(99.5, (int(mm.group(1)) / 1e6) / self.duration * 100)
                self.stages["proxy"]["pct"] = pct
                self.stages["proxy"]["detail"]["pct"] = round(pct, 1)
            mm = re.match(r"speed=\s*([\d.]+)x", msg)
            if mm:
                self.stages["proxy"]["detail"]["speed"] = float(mm.group(1))

    # --------------------------------------------------------------- status
    def status(self) -> dict:
        with self.lock:
            stages = [dict(self.stages[k]) for k, _ in STAGES]
            out = {
                "state": self.state,
                "video_id": self.video_id,
                "duration": self.duration,
                "elapsed": round(time.time() - self.started_at, 1) if self.started_at
                           and self.state == "running" else
                           round((self.ended_at or 0) - (self.started_at or 0), 1)
                           if self.started_at else 0,
                "stage_index": IDX.get(self.current, 0),
                "stages": stages,
                "calls": self.calls,
                "est_cost": round(self.est_cost, 2),
                "final": self.final,
                "warnings": self.warnings[-8:],
                "events": self.events[-8:],
                "error": self.error if self.state == "failed" else None,
                "log": list(self.log)[-40:],
            }
        # enrich from artifacts (outside the lock; reads are cheap and safe)
        vid = out["video_id"]
        if vid:
            try:
                att = self.stages["attribute"]
                if att["status"] == "done" and "cast_list" not in att["detail"]:
                    turns = read_json(config.artifact_dir(vid) / "turns.json")
                    att["detail"]["cast_list"] = [
                        {"name": m["name"], "party": m.get("party_code"),
                         "portrait": m.get("portrait")} for m in turns["cast"]]
                fr_dir = config.artifact_dir(vid) / "frames"
                if fr_dir.exists() and self.stages["frames"]["status"] in ("run", "done"):
                    names = sorted(p.name for p in fr_dir.glob("*.jpg"))
                    out["latest_frames"] = names[-7:]
                    self.stages["frames"]["detail"]["extracted_live"] = len(names)
            except Exception:                              # noqa: BLE001
                pass
            if out["state"] == "done":
                try:
                    seg = read_json(config.artifact_dir(vid) / "segments.json")
                    g = read_json(config.artifact_dir(vid) / "graph.json")
                    t = read_json(config.artifact_dir(vid) / "turns.json")
                    out["summary"] = {
                        "segments": seg["segment_count"],
                        "cast": len(t["cast"]),
                        "attributed_pct": round(100 * t["named_speech_ratio"]),
                        "nodes": g["node_count"], "edges": g["edge_count"],
                    }
                except Exception:                          # noqa: BLE001
                    pass
        return out


JOB = IngestJob()


# ---------------------------------------------------------------- candidates
def probe_video(p: pathlib.Path) -> dict:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height:format=duration",
             "-of", "default=noprint_wrappers=1", str(p)],
            capture_output=True, text=True, timeout=30)
        kv = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        return {"codec": kv.get("codec_name", "?"),
                "width": int(kv.get("width", 0) or 0),
                "height": int(kv.get("height", 0) or 0),
                "duration": float(kv.get("duration", 0) or 0)}
    except Exception:                                      # noqa: BLE001
        return {"codec": "?", "width": 0, "height": 0, "duration": 0}


_ARTIFACT_STAGES = ["audio.json", "segments.json", "frames.json", "turns.json",
                    "faceverify.json", "index_meta.json", "graph.json"]


def list_candidates() -> list[dict]:
    out = []
    for p in sorted(config.VIDEO_DIR.glob("*.mp4")) + sorted(config.VIDEO_DIR.glob("*.mkv")):
        vid = p.stem
        art = config.artifact_dir(vid)
        done = [n for n in _ARTIFACT_STAGES if (art / n).exists()]
        proxy = (config.PROXY_DIR / f"{vid}.mp4").exists()
        state = ("complete" if len(done) == len(_ARTIFACT_STAGES) and proxy
                 else "partial" if done else "new")
        info = probe_video(p)
        out.append({
            "video_id": vid, "file": p.name,
            "size_mb": round(p.stat().st_size / 1e6),
            "state": state, "stages_done": len(done),
            **info,
        })
    return out
