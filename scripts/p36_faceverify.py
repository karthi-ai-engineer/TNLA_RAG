"""P3.6 runner — face-lineup verification of attributed turns.

Usage:
    python scripts/p36_faceverify.py --video-id TNLA_trimmed

Run AFTER p35_attribute.py.  Re-running p35 resets turns.json, so re-run this
after any attribution change.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.faceverify import verify_video
from vrag.gemini import GeminiClient
from vrag.logging_setup import setup_logging

log = logging.getLogger("p36")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    args = ap.parse_args()

    setup_logging()

    with GeminiClient() as c:
        out = verify_video(c, args.video_id)
        log.info(c.usage_summary())

    log.info("=" * 74)
    log.info("P3.6 SUMMARY — %s", out["video_id"])
    log.info("  verdicts: %s", out["counts"])
    for r in out["results"]:
        if r.get("verdict") == "no_frames":
            log.info("  turn %-3d no standing frames to check", r["turn_index"])
            continue
        log.info("  turn %-3d expected %s (%s) -> %s",
                 r["turn_index"], r["member_name"], r["expected_label"], r["verdict"])
        for v in r["votes"]:
            log.info("      frame@%-7.1f picked %-4s (%s) %s",
                     v["t"], v["picked"], v["model_confidence"], v["reason"][:80])
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
