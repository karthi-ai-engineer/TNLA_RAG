"""P1.5 runner — build the 234-member roster with Tamil aliases + portraits.

Usage:
    python scripts/p15_roster.py
    python scripts/p15_roster.py --source <path-to-members.json>

Runs once.  The Tamil-alias Gemini calls are cached, so re-running is free.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.gemini import GeminiClient
from vrag.logging_setup import setup_logging
from vrag.roster import (PORTRAIT_DIR, RosterIndex, add_tamil_aliases,
                         build_roster)

log = logging.getLogger("p15")

# Maintenance only: the built roster ships in assets/roster/, so a fresh clone
# never runs this. To rebuild from a new source dataset, pass --source.
DEFAULT_SOURCE = pathlib.Path("roster_source/members.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path, default=DEFAULT_SOURCE)
    args = ap.parse_args()

    setup_logging()

    roster = build_roster(args.source, args.source.parent / "images")

    with GeminiClient() as c:
        roster = add_tamil_aliases(c, roster)
        log.info(c.usage_summary())

    # ---- prove the index answers the questions later phases will ask -------
    idx = RosterIndex(roster)
    probes = [
        "UDHAYANIDHI STALIN",          # exact romanized
        "உதயநிதி ஸ்டாலின்",             # Tamil script (the whole point of P1.5)
        "Udhayanidhi",                  # partial / fuzzy
        "N. MARIE WILSON",              # a minister
        "TOTALLY FAKE PERSON",          # must be None — no guessing
    ]
    log.info("=" * 70)
    log.info("P1.5 SUMMARY")
    log.info("  sitting members  %d  (vacant seats: %d)",
             roster["sitting_members"], roster["vacant_seats"])
    n_tamil = sum(1 for m in roster["members"] if m.get("tamil_name"))
    log.info("  tamil names      %d/%d", n_tamil, roster["sitting_members"])
    n_port = sum(1 for m in roster["members"] if m.get("portrait"))
    log.info("  portraits        %d/%d in %s", n_port, roster["sitting_members"],
             PORTRAIT_DIR)
    log.info("  lookup spot-check:")
    for q in probes:
        m = idx.find(q)
        log.info("    %-28s -> %s", q,
                 f"{m['name']} ({m['party_code']}, {m['constituency_name']})" if m else "None")

    sample = next(m for m in roster["members"] if "UDHAYANIDHI" in m["name"])
    log.info("  sample entry: %s | %s | tamil=%s | aliases=%s",
             sample["name"], sample["current_roles"], sample["tamil_name"],
             sample["tamil_aliases"])
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
