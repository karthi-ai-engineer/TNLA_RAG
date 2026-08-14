"""P5 — knowledge graph: entities and relations with segment provenance.

Every node and edge carries the segment ids it came from — provenance is the
whole point.  A graph fact that cannot cite a segment does not get in.

The graph is seeded from the ROSTER (members, parties, roles — facts we
already trust), then EXTENDED by per-segment Gemini extraction (schemes,
places, amounts, laws, dates, topics, and who-said-what edges).  Extraction
runs in batches with segment ids in the prompt, so provenance is structural.

Speaker→segment edges come from P3.5's attribution, not from the LLM — the
graph never re-decides who spoke.
"""
from __future__ import annotations

import logging

import networkx as nx

from vrag import config, gemini
from vrag.cache import read_json, write_json
from vrag.parallel import thread_map
from vrag.roster import RosterIndex

log = logging.getLogger(__name__)

BATCH = 15                      # segments per extraction call

ENTITY_TYPES = ["PARTY", "SCHEME", "PLACE", "AMOUNT", "LAW", "DATE",
                "TOPIC", "PERSON_EXTERNAL"]   # people NOT in the assembly

_EXTRACT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "entities": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": ENTITY_TYPES},
                    "name_ta": {"type": "STRING"},
                    "segment_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["name", "type", "name_ta", "segment_ids"],
            },
        },
        "relations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"},
                    "predicate": {"type": "STRING"},
                    "object": {"type": "STRING"},
                    "segment_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                },
                "required": ["subject", "predicate", "object", "segment_ids"],
            },
        },
    },
    "required": ["entities", "relations"],
}

_EXTRACT_PROMPT = """Extract entities and relations from these Tamil Nadu \
Legislative Assembly transcript segments (Tamil with English translation).

Entities — only these types:
- PARTY: political parties (DMK, AIADMK, BJP, TVK, Congress...)
- SCHEME: named government schemes/programmes/projects
- PLACE: states, districts, countries mentioned as subject matter
- AMOUNT: money amounts or significant quantities (quote EXACTLY as in the
  Tamil text — never convert or approximate)
- LAW: acts, bills, constitutional articles, resolutions
- DATE: specific dates or years tied to events
- TOPIC: substantive policy topics discussed (e.g. "delimitation",
  "women's reservation") — at most 2-3 per batch, only the central ones
- PERSON_EXTERNAL: people mentioned who are NOT members of this assembly
  (national politicians, historical figures)

Relations: subject/object are entity names from your list (or an assembly
member's name as spoken). Predicate is a short verb phrase in English
("criticised", "demanded", "announced", "wrote letter to", "compared with").

Rules:
- segment_ids: EVERY entity and relation must list the segment id(s) it
  appears in, from the ids given below. Never cite an id not shown.
- name: canonical English form. name_ta: as written in the Tamil text.
- Extract only what the text actually says. No outside knowledge, no
  inference beyond the words.

Segments:
{segments}"""


class GraphError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_batch(client, batch: list[dict]) -> dict:
    listing = "\n".join(
        f"[{s['id']}] ({(s.get('speaker') or {}).get('name') or 'unknown'}) "
        f"TA: {s['text_ta']}\n    EN: {s['text_en']}"
        for s in batch)
    out = client.generate_json(
        [gemini.text_part(_EXTRACT_PROMPT.format(segments=listing))],
        schema=_EXTRACT_SCHEMA,
        timeout=config.TIMEOUT_TEXT,
        namespace="kg_extract",
        max_output_tokens=16384,
    )
    valid_ids = {s["id"] for s in batch}
    # Provenance discipline: drop anything citing a segment we didn't send.
    ents = [e for e in out.get("entities", [])
            if e["segment_ids"] and set(e["segment_ids"]) <= valid_ids]
    rels = [r for r in out.get("relations", [])
            if r["segment_ids"] and set(r["segment_ids"]) <= valid_ids]
    dropped = (len(out.get("entities", [])) - len(ents)
               + len(out.get("relations", [])) - len(rels))
    if dropped:
        log.warning("kg: dropped %d extraction(s) with invalid segment ids", dropped)
    return {"entities": ents, "relations": rels}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def _node_key(name: str, ntype: str) -> str:
    return f"{ntype}:{name.strip().lower()}"


def build_graph(client, video_id: str) -> dict:
    art_dir = config.artifact_dir(video_id)
    seg_art = read_json(art_dir / "segments.json")
    segments = seg_art["segments"]
    idx = RosterIndex()

    g = nx.MultiDiGraph()

    # ---- Seed: attributed speakers, their parties, their roles (trusted).
    speakers_seen: set[str] = set()
    for s in segments:
        sp = s.get("speaker") or {}
        if not sp.get("member_id"):
            continue
        m = idx.by_id[sp["member_id"]]
        pk = _node_key(m["name"], "PERSON")
        if pk not in g:
            g.add_node(pk, label=m["name"], type="PERSON",
                       tamil=m.get("tamil_name"), party=m["party_code"],
                       roles=m["current_roles"], portrait=m.get("portrait"),
                       member_id=m["member_id"], segment_ids=[])
        g.nodes[pk]["segment_ids"].append(s["id"])
        speakers_seen.add(pk)
        party_k = _node_key(m["party_code"] or m["party"], "PARTY")
        if party_k not in g:
            g.add_node(party_k, label=m["party_code"] or m["party"],
                       type="PARTY", segment_ids=[])
        if not g.has_edge(pk, party_k):
            g.add_edge(pk, party_k, predicate="member of",
                       segment_ids=[], origin="roster")

    # ---- Extend: LLM extraction with provenance.
    batches = [segments[i:i + BATCH] for i in range(0, len(segments), BATCH)]
    log.info("kg: extracting from %d segments in %d calls", len(segments), len(batches))
    results = thread_map(lambda b: extract_batch(client, b), batches,
                         workers=client.max_workers, desc="kg-extract")

    seg_by_id = {s["id"]: s for s in segments}
    for res in results:
        for e in res["entities"]:
            # A "person" the extractor found may actually be a member —
            # resolve against the roster before creating a duplicate node.
            member = idx.find(e["name"]) or idx.find(e.get("name_ta", ""))
            if member:
                k = _node_key(member["name"], "PERSON")
                if k not in g:
                    g.add_node(k, label=member["name"], type="PERSON",
                               tamil=member.get("tamil_name"),
                               party=member["party_code"],
                               roles=member["current_roles"],
                               portrait=member.get("portrait"),
                               member_id=member["member_id"], segment_ids=[])
            else:
                k = _node_key(e["name"], e["type"])
                if k not in g:
                    g.add_node(k, label=e["name"], type=e["type"],
                               tamil=e.get("name_ta") or None, segment_ids=[])
            g.nodes[k]["segment_ids"].extend(e["segment_ids"])

        for r in res["relations"]:
            sk = _find_node(g, idx, r["subject"])
            ok = _find_node(g, idx, r["object"])
            if sk and ok and sk != ok:
                g.add_edge(sk, ok, predicate=r["predicate"],
                           segment_ids=r["segment_ids"], origin="extracted")

    # ---- "discussed" edges: speaker -> every TOPIC extracted from their segments.
    for tk, data in [(k, d) for k, d in g.nodes(data=True) if d["type"] == "TOPIC"]:
        for sid in data["segment_ids"]:
            sp = (seg_by_id.get(sid, {}).get("speaker") or {})
            if sp.get("name"):
                pk = _node_key(sp["name"], "PERSON")
                if pk in g and not any(
                        d.get("predicate") == "discussed" and v == tk
                        for _, v, d in g.out_edges(pk, data=True)):
                    g.add_edge(pk, tk, predicate="discussed",
                               segment_ids=[sid], origin="derived")

    # Dedup segment id lists.
    for _, data in g.nodes(data=True):
        data["segment_ids"] = sorted(set(data["segment_ids"]))

    payload = {
        "video_id": video_id,
        "node_count": g.number_of_nodes(),
        "edge_count": g.number_of_edges(),
        "nodes": [{"key": k, **d} for k, d in g.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d}
                  for u, v, d in g.edges(data=True)],
    }
    write_json(art_dir / "graph.json", payload)
    log.info("kg: %d nodes, %d edges -> graph.json",
             payload["node_count"], payload["edge_count"])
    return payload


def _find_node(g: nx.MultiDiGraph, idx: RosterIndex, name: str) -> str | None:
    member = idx.find(name)
    if member:
        k = _node_key(member["name"], "PERSON")
        return k if k in g else None
    for t in ("PARTY", "SCHEME", "PLACE", "AMOUNT", "LAW", "DATE", "TOPIC",
              "PERSON_EXTERNAL"):
        k = _node_key(name, t)
        if k in g:
            return k
    return None


# ---------------------------------------------------------------------------
# Graph-expansion retrieval (used by P6)
# ---------------------------------------------------------------------------
def load_graph(video_id: str) -> dict:
    p = config.artifact_dir(video_id) / "graph.json"
    if not p.exists():
        raise GraphError(f"No graph for '{video_id}'. Run scripts/p5_graph.py")
    return read_json(p)


def expand_query(graph: dict, query: str, limit: int = 6) -> list[str]:
    """Segment ids reachable from entities named in the query text.

    Dumb-and-transparent on purpose: substring match query terms against node
    labels (EN + TA), take those nodes' segments plus their neighbours'.
    """
    q = query.lower()
    hits: list[str] = []
    adj: dict[str, list[str]] = {}
    for e in graph["edges"]:
        adj.setdefault(e["source"], []).append(e["target"])
        adj.setdefault(e["target"], []).append(e["source"])
    by_key = {n["key"]: n for n in graph["nodes"]}

    for n in graph["nodes"]:
        label = (n.get("label") or "").lower()
        tamil = n.get("tamil") or ""
        if not label:
            continue
        if (len(label) > 2 and label in q) or (tamil and tamil in query):
            hits.extend(n["segment_ids"])
            for nb in adj.get(n["key"], [])[:4]:
                hits.extend(by_key[nb]["segment_ids"][:2])
    # Preserve first-seen order, cap.
    seen, out = set(), []
    for sid in hits:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out[:limit]
