"""P1 runner — extract audio, map silences, cut silence-aware chunks.

Usage:
    python scripts/p1_audio.py                          # default: data/videos/TNLA_1.mp4
    python scripts/p1_audio.py --video data/videos/X.mp4
    python scripts/p1_audio.py --overwrite              # redo even if artifacts exist
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag import config
from vrag.audio import prepare_audio, silences_from_artifact, snap_to_silence
from vrag.logging_setup import setup_logging

log = logging.getLogger("p1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=pathlib.Path,
                    default=config.VIDEO_DIR / "TNLA_1.mp4")
    ap.add_argument("--video-id", default=None,
                    help="defaults to the video filename stem")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    setup_logging()

    artifact = prepare_audio(args.video, video_id=args.video_id,
                             overwrite=args.overwrite)

    # ---- human-readable summary, so 'looks right' is checkable at a glance
    sil = artifact["silence"]
    log.info("=" * 70)
    log.info("P1 SUMMARY — %s", artifact["video_id"])
    log.info("  duration      %.1fs (%d:%02d)", artifact["duration"],
             int(artifact["duration"] // 60), int(artifact["duration"] % 60))
    log.info("  silences      %d pauses, %.1fs total, speech ratio %.1f%%",
             sil["count"], sil["total_seconds"], 100 * sil["speech_ratio"])
    log.info("  chunks        %d", len(artifact["chunks"]))
    for c in artifact["chunks"]:
        log.info("    chunk %02d  %7.1f -> %7.1f  (%5.1fs)  cut_at_silence=%s",
                 c["index"], c["start"], c["end"], c["duration"], c["cut_at_silence"])
    if artifact["chunk_boundaries_forced"]:
        log.warning("  FORCED boundaries (may clip a word): %s",
                    artifact["chunk_boundaries_forced"])

    # ---- sanity-check the snapper against the real silence map:
    # Gemini's probe timestamps were multiples of 5; show where each would snap.
    silences = silences_from_artifact(artifact)
    demo_points = [15.0, 60.0, 300.0, 900.0, 1500.0]
    log.info("  snap_to_silence spot-check (model-style guess -> snapped):")
    for t in demo_points:
        s = snap_to_silence(t, silences, prefer="start")
        e = snap_to_silence(t, silences, prefer="end")
        log.info("    %7.1f  -> start-snap %7.2f (%+.2f)   end-snap %7.2f (%+.2f)",
                 t, s, s - t, e, e - t)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
