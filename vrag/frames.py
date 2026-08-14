"""P3 — keyframes + vision: read what the broadcast prints on screen.

Vision's honest job on this footage (see PLAN.md §3): it is NOT a retrieval
channel and it does NOT recognise faces.  It does three narrow things well:

1. **Chyron OCR** — the lower-third name/role graphic is authoritative,
   in-frame, written evidence of who is speaking.  This is the strongest
   speaker-attribution signal we have.
2. **Standing-speaker cue** — in an assembly the member speaking stands while
   the rest sit.  "Is someone standing at a mic?" needs no identity and
   corroborates the audio's voice-turn boundaries.
3. **Visual context** for evidence cards (setting, what is happening).

Frame selection is scene cuts PLUS a periodic ~15s grid: chyrons appear and
disappear on the broadcast's own schedule, independent of camera cuts, so
cut-only sampling would miss them.
"""
from __future__ import annotations

import logging
import pathlib
import re
import subprocess

from vrag import config, gemini
from vrag.audio import AudioError, _run, probe_duration
from vrag.cache import write_json
from vrag.parallel import thread_map

log = logging.getLogger(__name__)

SCENE_THRESHOLD = 0.30      # ffmpeg scene-change score; probed fine on this footage
GRID_INTERVAL_S = 15.0      # periodic sample so chyron appearances can't be missed
MIN_SPACING_S = 4.0         # merge frames closer than this (cut + grid collisions)
FRAMES_PER_CALL = 6         # batched vision: ~10-20 calls for a 27-min video
JPEG_QUALITY = 4            # ffmpeg -q:v (2 best; 4 ≈ 100-200KB at 720p)


class FrameError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Frame selection + extraction
# ---------------------------------------------------------------------------
def detect_scene_cuts(video_path: pathlib.Path, threshold: float = SCENE_THRESHOLD) -> list[float]:
    """Timestamps of camera cuts, via ffmpeg's scene filter."""
    proc = _run([
        "ffmpeg", "-hide_banner", "-nostdin", "-nostats",
        "-i", str(video_path),
        "-vf", f"select='gt(scene,{threshold})',metadata=print",
        "-fps_mode", "vfr", "-f", "null", "-",
    ], "scene detect")
    times = [float(m.group(1))
             for m in re.finditer(r"pts_time:([\d.]+)", proc.stderr or "")]
    times.sort()
    log.info("frames: %d scene cuts (threshold %.2f)", len(times), threshold)
    return times


def plan_frame_times(duration: float, cuts: list[float],
                     grid: float = GRID_INTERVAL_S,
                     min_spacing: float = MIN_SPACING_S) -> list[float]:
    """Merge scene cuts with a periodic grid, dropping near-duplicates.

    Cut frames are taken slightly AFTER the cut (+0.5s) so we sample the new
    shot, not the transition blur.
    """
    candidates = sorted({round(t + 0.5, 2) for t in cuts if t + 0.5 < duration}
                        | {round(t, 2) for t in _frange(grid / 2, duration, grid)})
    chosen: list[float] = []
    for t in candidates:
        if not chosen or t - chosen[-1] >= min_spacing:
            chosen.append(t)
    log.info("frames: planned %d keyframes (%d cuts + %.0fs grid, %.1fs min spacing)",
             len(chosen), len(cuts), grid, min_spacing)
    return chosen


def _frange(start: float, stop: float, step: float) -> list[float]:
    out, t = [], start
    while t < stop:
        out.append(t)
        t += step
    return out


def extract_frames(video_path: pathlib.Path, times: list[float],
                   out_dir: pathlib.Path, overwrite: bool = False) -> list[dict]:
    """Write one JPEG per timestamp.  Returns [{t, path}] records.

    One ffmpeg invocation per frame with `-ss` BEFORE `-i`: input seeking is
    keyframe-based and fast even on AV1; decoding the whole stream once with a
    select filter would be slower for ~100 sparse frames.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for t in times:
        out = out_dir / f"frame_{int(round(t * 10)):06d}.jpg"   # 0.1s resolution in name
        if not out.exists() or overwrite:
            _run([
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-ss", f"{t:.2f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", str(JPEG_QUALITY),
                str(out),
            ], f"extract frame @{t:.1f}s")
        if not out.exists() or out.stat().st_size == 0:
            raise FrameError(f"Frame extraction produced nothing at t={t:.1f}s")
        records.append({"t": t, "path": out})
    total_mb = sum(r["path"].stat().st_size for r in records) / 1e6
    log.info("frames: extracted %d JPEGs (%.1f MB) -> %s", len(records), total_mb, out_dir)
    return records


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
_VISION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "frames": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "frame_index": {"type": "INTEGER"},
                    "setting": {"type": "STRING"},
                    "onscreen_text": {"type": "STRING"},
                    "chyron_name": {"type": "STRING"},
                    "chyron_role": {"type": "STRING"},
                    "person_standing": {"type": "BOOLEAN"},
                    "standing_description": {"type": "STRING"},
                },
                "required": ["frame_index", "setting", "onscreen_text",
                             "chyron_name", "chyron_role", "person_standing",
                             "standing_description"],
            },
        }
    },
    "required": ["frames"],
}

_VISION_PROMPT = """These are {n} frames from a Tamil news broadcast of a Tamil Nadu \
Legislative Assembly session, in order. For EACH frame return one entry \
(frame_index 0-based, in the order given):

- setting: one short phrase — what the shot shows (e.g. "wide shot of assembly
  hall", "member speaking at seat", "Speaker's chair", "news studio").
- onscreen_text: ALL text visible in the frame, transcribed exactly as written
  (Tamil in Tamil script). Include tickers, captions, logos. Empty string if none.
- chyron_name: if a lower-third graphic names the person currently shown/speaking,
  that name EXACTLY as written. Empty string if no such graphic. Do NOT infer a
  name from the face — only from readable text.
- chyron_role: the role/title text on that graphic (e.g. அமைச்சர், எதிர்க்கட்சித்
  தலைவர்), exactly as written. Empty string if none.
- person_standing: true if a person in the assembly hall is standing (speaking
  posture) while others are seated. False for studio shots / wide shots where
  nobody stands out.
- standing_description: if person_standing, a short visual description of that
  person (clothing, position in hall) WITHOUT guessing who they are. Else empty.

Never guess or invent text that is not clearly legible. Reporting an empty string
is always better than a wrong reading."""


def describe_frames(client, records: list[dict],
                    per_call: int = FRAMES_PER_CALL) -> list[dict]:
    """Run batched vision over the extracted frames; returns merged records."""
    batches = [records[i:i + per_call] for i in range(0, len(records), per_call)]
    log.info("frames: describing %d frames in %d vision calls", len(records), len(batches))

    def run_batch(batch: list[dict]) -> list[dict]:
        parts = [gemini.text_part(_VISION_PROMPT.format(n=len(batch)))]
        for r in batch:
            parts.append(gemini.image_part(r["path"]))
        out = client.generate_json(
            parts, schema=_VISION_SCHEMA,
            timeout=config.TIMEOUT_VISION, namespace="vision",
            max_output_tokens=16384,
        )
        entries = out.get("frames") or []
        if len(entries) != len(batch):
            raise FrameError(
                f"vision: sent {len(batch)} frames, got {len(entries)} entries")
        merged = []
        for r, e in zip(batch, sorted(entries, key=lambda x: x["frame_index"])):
            merged.append({
                "t": r["t"],
                "path": r["path"].name,
                "setting": (e.get("setting") or "").strip(),
                "onscreen_text": (e.get("onscreen_text") or "").strip(),
                "chyron_name": (e.get("chyron_name") or "").strip(),
                "chyron_role": (e.get("chyron_role") or "").strip(),
                "person_standing": bool(e.get("person_standing")),
                "standing_description": (e.get("standing_description") or "").strip(),
            })
        return merged

    results = thread_map(run_batch, batches, workers=client.max_workers,
                         desc="vision")
    return [f for batch in results for f in batch]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def process_frames(client, video_path: pathlib.Path,
                   video_id: str | None = None, overwrite: bool = False) -> dict:
    video_path = pathlib.Path(video_path).resolve()
    video_id = video_id or video_path.stem
    art_dir = config.artifact_dir(video_id)
    frame_dir = art_dir / "frames"

    duration = probe_duration(video_path)
    cuts = detect_scene_cuts(video_path)
    times = plan_frame_times(duration, cuts)
    records = extract_frames(video_path, times, frame_dir, overwrite=overwrite)
    described = describe_frames(client, records)

    chyrons = [f for f in described if f["chyron_name"]]
    standing = [f for f in described if f["person_standing"]]
    log.info("frames: %d/%d frames carry a chyron name, %d show a standing speaker",
             len(chyrons), len(described), len(standing))
    if not chyrons:
        log.warning("frames: NO chyron names found anywhere — speaker attribution "
                    "will rest on transcript introductions alone")

    artifact = {
        "video_id": video_id,
        "duration": duration,
        "frame_count": len(described),
        "scene_cuts": [round(t, 2) for t in cuts],
        "chyron_frame_count": len(chyrons),
        "standing_frame_count": len(standing),
        "frames": described,
    }
    out = art_dir / "frames.json"
    write_json(out, artifact)
    log.info("frames: wrote %s", out)
    return artifact


def load_frames_artifact(video_id: str) -> dict:
    from vrag.cache import read_json
    p = config.artifact_dir(video_id) / "frames.json"
    if not p.exists():
        raise FrameError(
            f"No frames artifact for '{video_id}' at {p}. "
            "Run:  python scripts/p3_frames.py")
    return read_json(p)
