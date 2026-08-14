"""P6 — answer generation with structurally-validated citations.

The non-negotiable: every answer cites video evidence.  Mechanically:

1.  The model only ever cites SEGMENT IDS — never timestamps.  Timestamps are
    attached by code from the transcript data, so a citation can be wrong
    only if retrieval was wrong, never because a number was hallucinated.
2.  The validator drops any cited id not actually present in the provided
    context.  An answer left with zero valid citations is a refused answer,
    surfaced as such — never displayed as fact.
3.  Figures must come from the Tamil text (the "1,000 crore" rule).  The
    prompt enforces it; the numeric flags from P2 ride along on every
    evidence card so a flagged segment is visibly flagged.
"""
from __future__ import annotations

import logging
import re

from vrag import config, gemini
from vrag.graph import expand_query, load_graph
from vrag.retrieve import HybridIndex

log = logging.getLogger(__name__)

MAX_CONTEXT_SEGMENTS = 8

# Matches ids in both [S042] and [S042, S043] citation forms.
_BRACKET_RE = re.compile(r"\[([^\[\]]*S\d{3}[^\[\]]*)\]")
_ID_RE = re.compile(r"S\d{3}")


def _extract_citations(text: str) -> list[str]:
    out = []
    for group in _BRACKET_RE.findall(text):
        out.extend(_ID_RE.findall(group))
    return out

_ANSWER_PROMPT = """You are answering questions about a Tamil Nadu Legislative \
Assembly session, using ONLY the transcript segments provided below.

Rules — these are hard constraints:
- Answer in the language of the question.
- After EVERY factual claim, cite the segment id(s) it comes from, in square
  brackets: [S042]. Use only ids from the context below.
- Numbers, amounts and dates MUST be taken from the Tamil text (text_ta), and
  quoted exactly. The English translation is an aid, not a source.
- Name a speaker only as given in the segment metadata. If the context does
  not answer the question, say so plainly WITHOUT citing any segment — do not
  improvise.
- Be concise: 2-6 sentences, then stop.

Question: {question}

Context segments:
{context}"""


class AnswerError(RuntimeError):
    pass


def _format_segment(s: dict) -> str:
    sp = s.get("speaker") or {}
    who = sp.get("name") or "Unknown speaker"
    role = (sp.get("roles") or [""])[0]
    return (f"[{s['id']}] speaker: {who}"
            f"{f' ({role})' if role else ''} | {s['t_start']:.0f}s\n"
            f"  text_ta: {s['text_ta']}\n"
            f"  text_en: {s['text_en']}")


def answer_question(client, video_id: str, question: str,
                    index: HybridIndex | None = None) -> dict:
    """Retrieve, answer, validate.  Returns the full displayable payload."""
    index = index or HybridIndex(video_id)

    hits = index.search(client, question, k=6)
    hit_ids = [h["id"] for h in hits]

    # Graph expansion: entity-linked segments the dense/sparse pass may miss.
    try:
        graph = load_graph(video_id)
        extra_ids = [sid for sid in expand_query(graph, question)
                     if sid not in set(hit_ids)]
    except Exception as exc:                             # noqa: BLE001
        log.warning("graph expansion unavailable (%s) — vector+BM25 only", exc)
        extra_ids = []

    context_segs = hits + [index.segments[sid] for sid in extra_ids
                           if sid in index.segments]
    context_segs = context_segs[:MAX_CONTEXT_SEGMENTS]
    valid_ids = {s["id"] for s in context_segs}

    context = "\n\n".join(_format_segment(s) for s in context_segs)
    text = client.generate_text(
        [gemini.text_part(_ANSWER_PROMPT.format(question=question,
                                                context=context))],
        timeout=config.TIMEOUT_TEXT,
        namespace="answer",
        # Thinking tokens count against this cap on gemini-3.5-flash — 2048
        # caused live GeminiTruncated failures when the model thought long.
        max_output_tokens=16384,
    )

    # ---- structural citation validation --------------------------------
    cited = _extract_citations(text)
    good = [c for c in dict.fromkeys(cited) if c in valid_ids]
    bad = sorted({c for c in cited if c not in valid_ids})
    if bad:
        log.warning("answer cited ids outside context %s — stripping", bad)
        for b in bad:
            text = re.sub(rf"\[?{b},?\s*\]?", "", text)

    refused = not good
    if refused:
        log.warning("answer has no valid citations — surfacing as no-evidence")

    by_id = {s["id"]: s for s in context_segs}
    citations = []
    for cid in good:
        s = by_id[cid]
        sp = s.get("speaker") or {}
        citations.append({
            "id": cid,
            "t_start": s["t_start"],
            "t_end": s["t_end"],
            "speaker": sp.get("name"),
            "speaker_tamil": sp.get("tamil_name"),
            "party": sp.get("party_code"),
            "roles": sp.get("roles") or [],
            "confidence": sp.get("confidence"),
            "evidence_kinds": sp.get("evidence_kinds", []),
            "text_ta": s["text_ta"],
            "text_en": s["text_en"],
            "numeric_flags": s.get("numeric_flags", []),
        })

    return {
        "video_id": video_id,
        "question": question,
        "answer": text.strip(),
        "refused": refused,
        "citations": citations,
        "context_ids": sorted(valid_ids),
    }
