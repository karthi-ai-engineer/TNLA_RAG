"""P6 runner — end-to-end Q&A with citation validation.

Usage:
    python scripts/p6_answer.py --video-id TNLA_trimmed
    python scripts/p6_answer.py -q "What was said about state autonomy?"
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.answer import answer_question
from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging
from vrag.retrieve import HybridIndex

log = logging.getLogger("p6")

DEMO_QUESTIONS = [
    "What was said about the delimitation of constituencies?",
    "What did Udhayanidhi Stalin say about women's reservation?",
    "What example was given about America's parliament?",
    "Who wrote letters to other Chief Ministers, and about what?",
    "What was said about the weather in Chennai today?",   # must refuse
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    ap.add_argument("-q", "--question", default=None)
    args = ap.parse_args()

    setup_logging()
    questions = [args.question] if args.question else DEMO_QUESTIONS

    with GeminiClient() as c:
        idx = HybridIndex(args.video_id)
        for q in questions:
            out = answer_question(c, args.video_id, q, index=idx)
            log.info("=" * 74)
            log.info("Q: %s", q)
            log.info("A: %s", out["answer"])
            if out["refused"]:
                log.info("   >>> refused: no valid citations <<<")
            for cit in out["citations"]:
                log.info("   [%s] %7.1fs %s (%s) conf=%.2s  %s",
                         cit["id"], cit["t_start"], cit["speaker"],
                         cit["party"], str(cit["confidence"]),
                         preview(cit["text_ta"], 55))
        log.info(c.usage_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
