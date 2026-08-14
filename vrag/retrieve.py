"""P4 — hybrid retrieval: dense embeddings + BM25, fused, LLM-reranked.

Corpus shape: ~100-200 segments per video.  At that size brute-force cosine
over a numpy matrix is microseconds; a vector DB would be theatre (PLAN.md
§1).  The `VectorStore` class keeps the swap point if that ever changes.

Retrieval text is the English gloss (cross-lingual sim Tamil↔EN verified at
0.858 in P0), prefixed with the attributed speaker so "what did Udhayanidhi
say about X" retrieves his segments preferentially.  BM25 indexes both the
English and Tamil tokens, so exact Tamil terms and names also hit.
"""
from __future__ import annotations

import logging
import re

import numpy as np
from rank_bm25 import BM25Okapi

from vrag import config, gemini
from vrag.cache import read_json, write_json

log = logging.getLogger(__name__)

RRF_K = 60               # standard reciprocal-rank-fusion constant
CANDIDATES = 20          # fused candidates handed to the reranker
TOP_K = 6                # what the answerer receives


class RetrieveError(RuntimeError):
    pass


def _embed_text(seg: dict) -> str:
    """What a segment 'is' for dense retrieval."""
    who = (seg.get("speaker") or {}).get("name") or "Unknown speaker"
    return f"{who}: {seg['text_en'] or seg['text_ta']}"


def _tokenize(text: str) -> list[str]:
    """Lowercased word tokens across scripts (Tamil words pass through)."""
    return re.findall(r"[a-z0-9]+|[஀-௿]+", text.lower())


# ---------------------------------------------------------------------------
# Index build (once per video, persisted)
# ---------------------------------------------------------------------------
def build_index(client, video_id: str) -> dict:
    art_dir = config.artifact_dir(video_id)
    seg_art = read_json(art_dir / "segments.json")
    segments = seg_art["segments"]
    if not segments:
        raise RetrieveError(f"{video_id}: segments.json has no segments")

    texts = [_embed_text(s) for s in segments]
    vecs = client.embed(texts, task_type=config.EMBED_TASK_DOCUMENT)
    arr = np.asarray(vecs, dtype=np.float32)
    np.save(art_dir / "vectors.npy", arr)
    write_json(art_dir / "index_meta.json", {
        "video_id": video_id,
        "segment_ids": [s["id"] for s in segments],
        "dim": int(arr.shape[1]),
        "count": int(arr.shape[0]),
    })
    log.info("retrieve: indexed %d segments (%dd vectors) for %s",
             arr.shape[0], arr.shape[1], video_id)
    return {"count": int(arr.shape[0])}


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
class HybridIndex:
    """Loads a video's persisted index and answers queries."""

    def __init__(self, video_id: str):
        art_dir = config.artifact_dir(video_id)
        self.video_id = video_id
        meta_p = art_dir / "index_meta.json"
        if not meta_p.exists():
            raise RetrieveError(
                f"No index for '{video_id}'. Run:  python scripts/p4_index.py")
        self.meta = read_json(meta_p)
        self.vectors = np.load(art_dir / "vectors.npy")
        seg_art = read_json(art_dir / "segments.json")
        self.segments = {s["id"]: s for s in seg_art["segments"]}
        self.ordered_ids = self.meta["segment_ids"]
        if len(self.ordered_ids) != self.vectors.shape[0]:
            raise RetrieveError(
                f"{video_id}: index/segment count mismatch "
                f"({len(self.ordered_ids)} ids vs {self.vectors.shape[0]} vectors) "
                "— rebuild the index.")
        corpus = [_tokenize(_embed_text(self.segments[i]) + " " +
                            self.segments[i]["text_ta"])
                  for i in self.ordered_ids]
        self.bm25 = BM25Okapi(corpus)

    def dense_ranking(self, qvec: np.ndarray) -> list[str]:
        sims = self.vectors @ qvec          # both sides L2-normalised
        order = np.argsort(-sims)
        return [self.ordered_ids[i] for i in order]

    def bm25_ranking(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        return [self.ordered_ids[i] for i in order]

    def search(self, client, query: str, k: int = TOP_K,
               rerank: bool = True) -> list[dict]:
        qvec = np.asarray(
            client.embed([query], task_type=config.EMBED_TASK_QUERY)[0],
            dtype=np.float32)
        dense = self.dense_ranking(qvec)
        sparse = self.bm25_ranking(query)

        rrf: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, sid in enumerate(ranking):
                rrf[sid] = rrf.get(sid, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused = sorted(rrf, key=rrf.get, reverse=True)[:CANDIDATES]

        if rerank and len(fused) > k:
            fused = self._llm_rerank(client, query, fused, k)
        return [self.segments[sid] for sid in fused[:k]]

    def _llm_rerank(self, client, query: str, candidate_ids: list[str],
                    k: int) -> list[str]:
        """Listwise rerank.  On any failure, fall back LOUDLY to RRF order —
        a degraded ranking is acceptable, a crashed demo is not."""
        listing = "\n".join(
            f"[{sid}] {(self.segments[sid].get('speaker') or {}).get('name') or '?'}: "
            f"{self.segments[sid]['text_en'][:300]}"
            for sid in candidate_ids)
        schema = {"type": "OBJECT",
                  "properties": {"ranked_ids": {"type": "ARRAY",
                                                "items": {"type": "STRING"}}},
                  "required": ["ranked_ids"]}
        prompt = (
            "Rank these transcript segments by how directly they help answer "
            f"the question. Return the {min(k + 2, len(candidate_ids))} most "
            "relevant segment ids, best first.\n\n"
            f"Question: {query}\n\nSegments:\n{listing}")
        try:
            out = client.generate_json(
                [gemini.text_part(prompt)], schema=schema,
                timeout=config.TIMEOUT_TEXT, namespace="rerank")
            valid = [sid for sid in out.get("ranked_ids", [])
                     if sid in set(candidate_ids)]
            if valid:
                # Anything the reranker dropped keeps its RRF position after.
                rest = [sid for sid in candidate_ids if sid not in set(valid)]
                return valid + rest
            log.warning("rerank returned no valid ids — using RRF order")
        except Exception as exc:                        # noqa: BLE001
            log.warning("rerank failed (%s: %s) — using RRF order",
                        type(exc).__name__, exc)
        return candidate_ids
