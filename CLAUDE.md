# CLAUDE.md — Multimodal Video RAG (Vector + Knowledge Graph)

## What we're building

A multimodal RAG system over video. Give it a video; it understands speech, on-screen text,
visual content, and scene structure; it stores that in a vector DB + knowledge graph; then
natural-language questions get a grounded answer **and the exact moment(s) in the video it
came from** — clickable timestamps that seek a player to that point, with the supporting
evidence shown alongside.

**Non-negotiable: every answer cites video evidence. No citation = a bug.**

**Audience:** a stakeholder demo, built in one day. It must feel like a product, not a
notebook — "startlingly accurate and clearly engineered," enough to fund the next phase.
Narrow and polished beats broad and half-working: perfect on 2–3 chosen videos, not
mediocre on everything.

## Domain

Tamil Nadu Legislative Assembly footage. Audio is Tamil. `data/videos/TNLA_1.mp4` is a
~27-minute clip already on hand; more may be added. Transcript is stored verbatim in Tamil
for display/citation; an English translation is generated per segment for embedding/search
only — **numbers, dates, and amounts shown to the user must always come from the Tamil
source, never the translation.** (Caught in testing: a capable model mistranslated "1,000
crore" as "1 lakh crore" — a 100x error. Translation is a search aid, not a source of truth.)
 this clip will be change nw and a new clip will be added ; so donsider this as main 
## LLM / model provider: Gemini

We use the **Gemini API** for everything — chat, vision (keyframe description), embeddings,
and audio transcription. One provider, reachable over normal internet, paid key already
available.

- **`gemini-3.5-flash`** for the accuracy-critical path: transcription, translation, visual
  description. Not `flash-lite` — we already caught a strong model making a bad numeric
  translation error, so this is the wrong place to economize; the cost difference at our
  volume (a handful of dozens of calls for one demo video) is pennies either way.
- **`gemini-embedding-001`** for text embeddings.
- **DECIDED 2026-08-14: everything uses Gemini's native API (`v1beta/models/...`), not the
  OpenAI-compatible endpoint.** An earlier draft of this file specified the compat layer;
  we deliberately do not use it. Reasons, all verified in P0: the compat layer cannot set
  embedding `taskType` (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT` — the smoke test proves
  these yield different vectors, and the distinction measurably improves retrieval), it
  exposes `responseSchema` less reliably, and it does not return `usageMetadata` for cost
  tracking. One transport also means one set of failure modes to harden. `vrag/gemini.py`
  is built and tested this way (15/15). **Do not "fix" this back to the compat endpoint.**
- Audio transcription likewise uses Gemini's native
  `generateContent`, sending audio inline as base64 (`inline_data`), with
  `responseMimeType: application/json` + a `responseSchema` to get back structured
  `{language, segments: [{start, end, text}]}` directly. Inline audio is capped at 100MB
  base64-encoded by the API — fine for clips up to roughly ~30 min at 16kHz mono; longer
  clips need the Files API (upload-then-reference), not yet needed.
- API key goes in `.env`, never hardcoded, never committed.

## Architecture

```
video.mp4
  ├─ audio ──► Gemini transcription ──► Tamil transcript, segmented with timestamps
  ├─ ffmpeg scene-detect ──► keyframes (a handful per segment, not every frame)
  │        └──► Gemini vision on keyframes ──► visual description (setting, on-screen text, action)
  └─ SEGMENT = fused unit: transcript slice (Tamil) + English gloss + visual description + timestamps
                │
       ┌────────┴─────────┐
       ▼                  ▼
  VECTOR + BM25        KNOWLEDGE GRAPH
  (segment embeddings)  (entities, relations, temporal order — provenance back to a segment id)
       └────────┬─────────┘
                ▼
     hybrid retrieval → rerank → top-k
                ▼
     answer generation, citing SEGMENT IDS (never raw timestamps —
     code maps segment id → timestamp from the transcript data, so a wrong
     citation is structurally impossible, not just unlikely)
                ▼
     UI: player + chat + citations that seek the video to that moment
```

The **segment** (not a raw frame, not the whole video) is the unit everything is built on —
small enough to embed, precise enough to cite, and the record every graph edge points back to
for provenance.

## Working agreement

- Be the pilot. The user is a capable beginner who wants expert direction, not just
  compliance — flag a bad idea plainly, recommend one better option, keep moving.
- Time is the scarcest resource today. Demoable beats architecturally pure.
- Verify, don't assume — run it, read the artifact, don't call something done without
  seeing it work.
- No silent fallbacks or invented data. Degrade loudly, never fake a result.
- Small modules, real functions, no premature abstraction. Boring and proven beats clever.
- Windows: no `&&` in PowerShell, `pathlib` always, release `cv2`/file handles.
- Never commit `.env`, `data/videos/*`, or any generated artifacts.
