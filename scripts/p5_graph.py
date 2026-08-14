"""P5 runner — build the knowledge graph with segment provenance.

Usage:
    python scripts/p5_graph.py --video-id TNLA_trimmed
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag.gemini import GeminiClient
from vrag.graph import build_graph, expand_query
from vrag.logging_setup import setup_logging

log = logging.getLogger("p5")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    args = ap.parse_args()

    setup_logging()

    with GeminiClient() as c:
        graph = build_graph(c, args.video_id)
        log.info(c.usage_summary())

    log.info("=" * 74)
    log.info("P5 SUMMARY — %s", graph["video_id"])
    log.info("  nodes %d, edges %d", graph["node_count"], graph["edge_count"])
    types = Counter(n["type"] for n in graph["nodes"])
    log.info("  node types: %s", dict(types))
    log.info("  most-connected nodes:")
    degree = Counter()
    for e in graph["edges"]:
        degree[e["source"]] += 1
        degree[e["target"]] += 1
    by_key = {n["key"]: n for n in graph["nodes"]}
    for key, deg in degree.most_common(10):
        n = by_key[key]
        log.info("    %-40s %-16s deg=%d segs=%d",
                 n["label"][:40], n["type"], deg, len(n["segment_ids"]))
    log.info("  expansion probe 'delimitation': %s",
             expand_query(graph, "what about delimitation"))
    log.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
