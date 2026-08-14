"""P4 runner — build the hybrid index, then smoke-test retrieval quality.

Usage:
    python scripts/p4_index.py --video-id TNLA_trimmed
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging
from vrag.retrieve import HybridIndex, build_index

log = logging.getLogger("p4")

TEST_QUERIES = [
    "What was said about delimitation of constituencies?",
    "What did Udhayanidhi Stalin say about the all-party meeting?",
    "33 percent reservation for women",
    "What was the reply about America and its parliament?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    args = ap.parse_args()

    setup_logging()

    with GeminiClient() as c:
        build_index(c, args.video_id)
        idx = HybridIndex(args.video_id)

        log.info("=" * 74)
        log.info("P4 SUMMARY — retrieval smoke test")
        for q in TEST_QUERIES:
            hits = idx.search(c, q, k=3)
            log.info("  Q: %s", q)
            for h in hits:
                who = (h.get("speaker") or {}).get("name") or "?"
                log.info("     %s %7.1fs [%s] %s", h["id"], h["t_start"],
                         who, preview(h["text_en"], 70))
        log.info(c.usage_summary())
        log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
