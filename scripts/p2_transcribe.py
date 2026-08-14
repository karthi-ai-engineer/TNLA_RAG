"""P2 runner — transcribe all chunks, snap timestamps, run the numeric guard.

Usage:
    python scripts/p2_transcribe.py                  # default video TNLA_1
    python scripts/p2_transcribe.py --video-id X

Requires P1's audio.json (run scripts/p1_audio.py first).
API cost: one audio call per chunk (5 for TNLA_1); fully cached on re-run.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging
from vrag.transcribe import transcribe_video

log = logging.getLogger("p2")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_1")
    args = ap.parse_args()

    setup_logging()

    with GeminiClient() as c:
        result = transcribe_video(c, args.video_id)
        log.info(c.usage_summary())

    segs = result["segments"]
    log.info("=" * 74)
    log.info("P2 SUMMARY — %s", result["video_id"])
    log.info("  segments        %d covering %.0fs of %.0fs",
             result["segment_count"], result["covered_seconds"], result["duration"])
    log.info("  numeric flags   %d (see numeric_flags.json)",
             result["numeric_flag_count"])
    log.info("  gaps > 20s      %s", result["gaps_over_20s"] or "none")

    snap_moved = [abs(s["t_start"] - s["t_start_raw"]) for s in segs]
    moved = [d for d in snap_moved if d > 0.01]
    log.info("  snapping        %d/%d starts moved, median move %.2fs",
             len(moved), len(segs),
             sorted(moved)[len(moved) // 2] if moved else 0.0)

    voices = {s["voice_local"] for s in segs}
    turns = sum(1 for s in segs if s["new_speaker"])
    log.info("  voices          %d local labels, %d speaker turns", len(voices), turns)

    log.info("  first / middle / last segments:")
    for s in (segs[0], segs[len(segs) // 2], segs[-1]):
        log.info("    %s %7.1f-%7.1f [%s] ta: %s",
                 s["id"], s["t_start"], s["t_end"], s["voice_local"],
                 preview(s["text_ta"], 60))
        log.info("         %s en: %s", " " * 17, preview(s["text_en"], 60))
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
