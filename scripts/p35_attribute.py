"""P3.5 runner — fuse announcements + chyrons + voice turns into speaker names.

Usage:
    python scripts/p35_attribute.py --video-id TNLA_trimmed
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.attribute import attribute_video
from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging

log = logging.getLogger("p35")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    args = ap.parse_args()

    setup_logging()

    with GeminiClient() as c:
        result = attribute_video(c, args.video_id)
        log.info(c.usage_summary())

    log.info("=" * 74)
    log.info("P3.5 SUMMARY — %s", result["video_id"])
    log.info("  turns           %d (%d named, %.0f%% of speech time)",
             result["turn_count"], result["named_turn_count"],
             100 * result["named_speech_ratio"])
    log.info("  active cast     %d members", len(result["cast"]))
    for m in result["cast"]:
        log.info("    %-28s %-6s %s", m["name"], m["party_code"],
                 ", ".join(m["roles"]))
    log.info("  turn timeline:")
    for t in result["turns"]:
        sp = t["speaker"]
        who = sp["name"] or "— unknown —"
        ev = "+".join(sp.get("evidence_kinds", [])) or "none"
        log.info("    %7.1f-%7.1f  %-28s conf=%.2f  ev=%s%s",
                 t["t_start"], t["t_end"], who, sp["confidence"], ev,
                 "  CONFLICT" if sp.get("conflict") else "")
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
