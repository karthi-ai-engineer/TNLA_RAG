"""Mirror a video's knowledge graph into Neo4j — FOR VIEWING ONLY.

The pipeline itself never reads Neo4j; retrieval and the web UI keep using
data/artifacts/<video_id>/graph.json.  This export exists because Neo4j
Browser's visualization looks good in front of stakeholders.  If Neo4j is
down, nothing else in the system notices.

Usage:
    python scripts/export_neo4j.py --video-id TNLA_trimmed --password <dbpass>
    # optional: --uri bolt://localhost:7687 --user neo4j

Then in Neo4j Browser:
    MATCH (n {video_id: 'TNLA_trimmed'}) RETURN n     -- everything
    MATCH (p:Person)-[r]->(t:Topic) RETURN p, r, t    -- who discussed what
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vrag import config
from vrag.cache import read_json
from vrag.logging_setup import setup_logging

log = logging.getLogger("neo4j-export")

# graph.json type -> Neo4j label (CamelCase looks right in the Browser).
LABELS = {
    "PERSON": "Person", "PARTY": "Party", "TOPIC": "Topic", "PLACE": "Place",
    "AMOUNT": "Amount", "LAW": "Law", "DATE": "Date", "SCHEME": "Scheme",
    "PERSON_EXTERNAL": "ExternalPerson",
}


def rel_type(predicate: str) -> str:
    """'wrote letter to' -> WROTE_LETTER_TO (Neo4j rel types can't be dynamic
    strings with spaces)."""
    t = re.sub(r"[^A-Za-z0-9]+", "_", predicate.strip().upper()).strip("_")
    return t or "RELATED_TO"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", default="TNLA_trimmed")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    setup_logging()

    graph_path = config.artifact_dir(args.video_id) / "graph.json"
    if not graph_path.exists():
        raise SystemExit(f"No graph at {graph_path} — run scripts/p5_graph.py first")
    graph = read_json(graph_path)

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    seg_art = read_json(config.artifact_dir(args.video_id) / "segments.json")
    t_start = {s["id"]: s["t_start"] for s in seg_art["segments"]}

    with driver.session() as sess:
        # Idempotent: wipe this video's previous mirror, keep anything else.
        sess.run("MATCH (n {video_id: $vid}) DETACH DELETE n",
                 vid=args.video_id)

        for n in graph["nodes"]:
            label = LABELS.get(n["type"], "Entity")
            segs = n.get("segment_ids") or []
            sess.run(
                f"CREATE (x:{label} {{key: $key, name: $name, tamil: $tamil, "
                f"party: $party, role: $role, video_id: $vid, "
                f"segment_ids: $segs, first_seen_s: $first}})",
                key=n["key"], name=n.get("label") or "",
                tamil=n.get("tamil") or "",
                party=n.get("party") or "",
                role=(n.get("roles") or [""])[0],
                vid=args.video_id, segs=segs,
                first=min((t_start.get(s, 0) for s in segs), default=None),
            )

        for e in graph["edges"]:
            sess.run(
                f"MATCH (a {{key: $src, video_id: $vid}}), "
                f"(b {{key: $dst, video_id: $vid}}) "
                f"CREATE (a)-[:{rel_type(e.get('predicate', ''))} "
                f"{{predicate: $pred, segment_ids: $segs, origin: $origin}}]->(b)",
                src=e["source"], dst=e["target"], vid=args.video_id,
                pred=e.get("predicate") or "", segs=e.get("segment_ids") or [],
                origin=e.get("origin") or "",
            )

        counts = sess.run(
            "MATCH (n {video_id: $vid}) "
            "OPTIONAL MATCH (n)-[r]->() "
            "RETURN count(DISTINCT n) AS nodes, count(r) AS rels",
            vid=args.video_id).single()

    driver.close()
    log.info("Exported %d nodes, %d relationships for %s",
             counts["nodes"], counts["rels"], args.video_id)
    log.info("Open Neo4j Browser and try:")
    log.info("  MATCH (n {video_id: '%s'}) RETURN n", args.video_id)
    log.info("  MATCH (p:Person)-[r]->(x) RETURN p, r, x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
