"""ONE COMMAND: turn a raw assembly video into a fully queryable, cited RAG.

    python ingest.py data/videos/MY_SESSION.mp4
    python ingest.py data/videos/MY_SESSION.mp4 --serve     # and start the UI
    python ingest.py                                        # auto-picks the only .mp4

Runs every pipeline phase in order, skipping any phase whose artifact already
exists (delete an artifact, or pass --force, to redo it).  Every Gemini call
is disk-cached, so re-running after a crash resumes where it stopped and
costs nothing for completed work.

Phases:
  1. audio      extract 16kHz WAV, map silences, cut 2-min chunks   (no API)
  2. transcribe Tamil verbatim + EN gloss, snapped timestamps       (~1 call / 2min)
  3. frames     keyframes + vision (on-screen text, standing spkr)  (~1 call / 6 frames)
  4. attribute  who is speaking — announcements + procedure rules   (1 call)
  5. faceverify photo-lineup verification of every attribution      (~2 calls / speaker)
  6. index      embeddings + BM25                                   (~1 call / 16 segs)
  7. graph      knowledge graph with segment provenance             (~1 call / 15 segs)
  8. proxy      H.264 rendition for the browser player              (no API, ffmpeg)

Cost scale: a 27-min video ≈ $1 of Gemini; ~4 hours ≈ $3–5. One-time; cached after.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from vrag import config
from vrag.logging_setup import setup_logging

log = logging.getLogger("ingest")


def art(video_id: str, name: str) -> pathlib.Path:
    return config.artifact_dir(video_id) / name


def find_video(arg: str | None) -> pathlib.Path:
    if arg:
        p = pathlib.Path(arg)
        if not p.exists():
            raise SystemExit(f"Video not found: {p}")
        return p.resolve()
    vids = sorted(config.VIDEO_DIR.glob("*.mp4"))
    if len(vids) == 1:
        return vids[0].resolve()
    raise SystemExit(
        f"Expected exactly one .mp4 in {config.VIDEO_DIR} (found "
        f"{[v.name for v in vids] or 'none'}). Put your video there or pass a path:\n"
        "    python ingest.py data/videos/MY_SESSION.mp4")


def preflight() -> None:
    """Fail fast, with fix-it instructions, before any work starts."""
    problems = []
    try:
        config.get_api_key()
    except Exception as exc:                              # noqa: BLE001
        problems.append(str(exc))
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        except Exception:                                 # noqa: BLE001
            problems.append(f"{tool} not found on PATH — install ffmpeg and reopen the shell.")
    from vrag.roster import ROSTER_JSON
    if not ROSTER_JSON.exists():
        problems.append(f"Roster missing at {ROSTER_JSON} — the repo ships it; "
                        "re-clone or run scripts/p15_roster.py.")
    if problems:
        raise SystemExit("Preflight failed:\n  - " + "\n  - ".join(problems))
    from vrag.roster import load_roster
    log.info("preflight: GEMINI_API_KEY present")
    log.info("preflight: ffmpeg / ffprobe found")
    log.info("preflight: roster loaded — %d members",
             load_roster()["sitting_members"])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", default=None,
                    help="path to the .mp4 (default: the only .mp4 in data/videos/)")
    ap.add_argument("--video-id", default=None, help="defaults to the filename stem")
    ap.add_argument("--force", action="store_true", help="redo every phase")
    ap.add_argument("--serve", action="store_true", help="start the demo UI afterwards")
    ap.add_argument("--skip-faceverify", action="store_true",
                    help="skip the face-lineup pass (attribution keeps text evidence only)")
    args = ap.parse_args()

    setup_logging()
    config.ensure_dirs()
    preflight()

    video = find_video(args.video)
    vid = args.video_id or video.stem
    t0 = time.time()
    log.info("=" * 74)
    log.info("INGEST %s  (video_id=%s)", video.name, vid)
    log.info("=" * 74)

    from vrag.gemini import GeminiClient

    def done(name: str) -> bool:
        if args.force:
            return False
        return art(vid, name).exists()

    with GeminiClient() as client:
        # 1 ─ audio
        if done("audio.json"):
            log.info("[1/8] audio        SKIP (audio.json exists)")
        else:
            log.info("[1/8] audio        extracting + silence map + chunks")
            from vrag.audio import prepare_audio
            prepare_audio(video, video_id=vid, overwrite=args.force)

        # 2 ─ transcribe
        if done("segments.json") and not args.force:
            log.info("[2/8] transcribe   SKIP (segments.json exists)")
        else:
            log.info("[2/8] transcribe   Tamil + EN + snapped timestamps")
            from vrag.transcribe import transcribe_video
            transcribe_video(client, vid)

        # 3 ─ frames + vision
        if done("frames.json"):
            log.info("[3/8] frames       SKIP (frames.json exists)")
        else:
            log.info("[3/8] frames       keyframes + vision")
            from vrag.frames import process_frames
            process_frames(client, video, video_id=vid, overwrite=args.force)

        # 4 ─ attribution
        if done("turns.json"):
            log.info("[4/8] attribute    SKIP (turns.json exists)")
        else:
            log.info("[4/8] attribute    resolving speakers against the roster")
            from vrag.attribute import attribute_video
            attribute_video(client, vid)

        # 5 ─ face verification
        if args.skip_faceverify:
            log.info("[5/8] faceverify   SKIP (--skip-faceverify)")
        elif done("faceverify.json"):
            log.info("[5/8] faceverify   SKIP (faceverify.json exists)")
        else:
            log.info("[5/8] faceverify   photo-lineup verification")
            from vrag.faceverify import verify_video
            verify_video(client, vid)

        # 6 ─ retrieval index
        if done("index_meta.json") and done("vectors.npy"):
            log.info("[6/8] index        SKIP (index exists)")
        else:
            log.info("[6/8] index        embeddings + BM25")
            from vrag.retrieve import build_index
            build_index(client, vid)

        # 7 ─ knowledge graph
        if done("graph.json"):
            log.info("[7/8] graph        SKIP (graph.json exists)")
        else:
            log.info("[7/8] graph        entity extraction with provenance")
            from vrag.graph import build_graph
            build_graph(client, vid)

        usage = client.usage_summary()

    # 8 ─ browser proxy (H.264 — the source is often AV1, which browsers seek badly)
    proxy = config.PROXY_DIR / f"{vid}.mp4"
    if proxy.exists() and not args.force:
        log.info("[8/8] proxy        SKIP (%s exists)", proxy.name)
    else:
        log.info("[8/8] proxy        transcoding H.264 rendition (longest step, no API)")
        # -progress pipe:1 streams machine-readable out_time_ms/speed lines to
        # stdout so the ingest UI can render a real percentage bar.
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats",
            "-progress", "pipe:1", "-y",
            "-i", str(video),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(proxy)], check=True)

    log.info("=" * 74)
    log.info("READY in %s  |  %s", f"{(time.time()-t0)/60:.1f} min", usage)
    log.info("Artifacts: %s", config.artifact_dir(vid))
    log.info("Start the demo:   python serve.py     then open http://127.0.0.1:8000")
    log.info("=" * 74)

    if args.serve:
        import uvicorn
        uvicorn.run("vrag.app:app", host="127.0.0.1", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
