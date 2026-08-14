# Implementation Plan — Multimodal Video RAG (TNLA)

Written after live probing against the real asset and the real API key.
Everything below marked **[VERIFIED]** was actually executed, not assumed.

---

## 0. What the probes established

| Check | Result | Consequence for the plan |
|---|---|---|
| Asset: `TNLA_1.mp4` | AV1 720p30, Opus 48k stereo, **1620.0s (27:00)**, 185 MB **[VERIFIED]** | Browser `<video>` gets an H.264 proxy; AV1 seek is unreliable |
| `gemini-3.5-flash` exists on this key | Yes (also 3.6/3.7-flash, `gemini-embedding-2`) **[VERIFIED]** | CLAUDE.md's model choice is valid, keep it |
| Tamil ASR quality | Excellent — coherent, correct proper nouns (Anna, Kalaignar, TVK, DMK), faithful EN gloss; 120s audio in **18.3s** **[VERIFIED]** | Transcription is *not* a project risk |
| ASR timestamps | Returned 0 / 15 / 35 / 60 / 85 / 100 — **all multiples of 5** **[VERIFIED]** | Model *estimates* boundaries. This is the #1 accuracy risk → silence-snapping (§2) |
| ffmpeg `silencedetect` | Dense, clean pauses; Gemini's "15.0" ↔ real pause at 14.27–15.61 **[VERIFIED]** | Snapping corrects boundaries by 0.5–1.5s. Cheap, deterministic |
| Scene detect on AV1 | 120s slice in **2.2s** → full video ≈ 30s; 4 cuts/2min → ~55 keyframes **[VERIFIED]** | Video decode is a non-issue |
| Vision on real keyframes | Only text found = channel logo "புதிய தலைமுறை". All 4 frames = "member in white shirt speaking" **[VERIFIED]** | **Vision is a thin signal on this footage.** Scope it down (§3) |
| Embeddings | ~0.4s single; **batch of 16 also ~0.5s** **[VERIFIED]** | Batch aggressively → all segments in ~15s |
| Embedding stall | Two calls hung at **exactly 300.4s** | Black-holed connection, not API. **Every call needs an explicit short timeout + retry** (§1) |
| Dep stack on py3.13 | numpy 2.5.1, networkx 3.6.1, fastapi, rank_bm25, pyvis, httpx all install clean **[VERIFIED]** | No GPU, no Docker, no Neo4j |
| `ffmpeg -vsync` | **Removed in ffmpeg 9.0** — errors out | Use `-fps_mode vfr`. Most tutorials/repos will break here |

---

## 1. Repo strategy — decision: **do not fork, build thin**

Researched candidates:

- **HKUDS/VideoRAG** (KDD'26) — needs ImageBind + MiniCPM checkpoints, RTX 3090, conda, Electron shell. Setup alone exceeds our budget. **Reject.**
- **Sh-31/Multimodal-GraphRAG** — Neo4j + Qdrant. Two servers to stand up on Windows. **Reject the infra**, borrow the schema idea.
- **LightRAG** (HKUDS) — genuinely good and pip-installable, but it owns its own chunk/storage abstraction and has no concept of a *timestamp-bearing* segment. We'd spend the day bending it to emit video provenance. **Reject as a dependency**, borrow its dual-level (entity-local + theme-global) retrieval idea.
- **LeDat98/NexusRAG** — closest in spirit (hybrid + graph + reranker + inline citations, Gemini-powered) but document-RAG, not video. **Borrow its citation UX.**

> **Rationale:** every candidate's integration cost is larger than writing the thing, and *none* of them do the one non-negotiable: `segment_id → timestamp` provenance. We write ~900 lines of boring, purpose-built code and borrow proven *designs*, not codebases.

**Stack:** Python 3.13 · ffmpeg 9 · Gemini (OpenAI-compat for chat/vision/embed, native `generateContent` for audio) · numpy (brute-force cosine) · rank_bm25 · networkx + pyvis · FastAPI + vanilla JS.

> On "vector DB": we have ~80 segments. Cosine over an 80×1536 numpy array is microseconds. A vector DB here is theatre, not engineering. Written behind a `VectorStore` class so Chroma can be dropped in later in ~20 lines if the stakeholder narrative demands the word.

---

## 2. Pipeline

```
video.mp4
  │
  ├─ ffmpeg ──► audio 16k mono WAV ──► silence map (silencedetect)
  │                  │
  │                  └─► split into ~6-min chunks AT SILENCE (never mid-word)
  │                          │
  │                          └─► Gemini 3.5-flash per chunk (inline base64, responseSchema)
  │                                    → {start,end,speaker,text_ta,text_en}
  │                                    → + chunk offset      (bounds drift to one chunk)
  │                                    → SNAP boundaries to nearest real silence  ◄── accuracy core
  │                                    → numeric guard: digits(text_ta) must match digits(text_en)
  │
  ├─ ffmpeg scene-detect ──► ~55 keyframes ──► Gemini vision, 6 frames/call (~10 calls)
  │                                    → {setting, onscreen_text_ta/en, speaker_name, action}
  │
  └─► SEGMENT[i] = { id "S042", t_start, t_end, text_ta, text_en, speaker,
                     visual[], keyframe_path, video_id }
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
      DENSE + BM25                KNOWLEDGE GRAPH
   embedding-001 @1536         Gemini structured extraction/segment
   cosine over numpy           Person/Party/Scheme/Place/Amount/Law/Date
   BM25 over EN gloss+TA       every node & edge carries segment_id
              └───────────┬────────────┘
                          ▼
      RRF fusion → LLM listwise rerank (top-20 → top-6)
                 + graph expansion (query entity → its segments)
                          ▼
      answer generation citing SEGMENT IDS ONLY  →  validator drops any id
      not present in context  →  code maps id → timestamp
                          ▼
      UI: player + chat + evidence cards + clickable seek + graph tab
```

**Timestamp accuracy strategy (the thing that makes or breaks citations):**
1. Chunk at ~6 min → progressive drift can never exceed one chunk.
2. Each chunk transcribed with local timestamps, then offset by a *known* chunk start.
3. Snap every boundary to the nearest `silencedetect` pause within ±2.5s → boundaries land on real audio events, not model guesses.
4. Player seeks to `t_start − 1.5s` so the cited line is always heard from its beginning.

**Numeric integrity (the "1,000 crore → 1 lakh crore" class of bug):**
- Same call emits `text_ta` + `text_en`, so they can't drift apart.
- `numeric_guard()` extracts numerals + Tamil number-words from both sides and flags mismatches into `data/artifacts/numeric_flags.json`.
- Answer prompt: figures **must** be quoted from `text_ta`; the English gloss is search-only.
- UI always shows Tamil verbatim in the evidence card next to the answer.

---

## 3. Honest scoping calls

- **Vision is deliberately down-scoped.** The probe proved this footage is talking heads + a channel logo. I will *not* spend hours making vision a retrieval channel it cannot be. It earns its place via shot structure, chyron/name-plate text *when present*, and visual context on the evidence card. Claiming more would be dishonest in the demo.
- **Speaker attribution** comes mostly from transcript context + chyrons, in a dedicated pass over the full transcript. It's what makes the KG interesting ("who said what about X").
- **Multi-video from hour one.** You said the clip changes and another is coming. Every artifact is keyed by `video_id`; nothing is hardcoded to TNLA_1.
- **Every artifact cached to `data/artifacts/<video_id>/*.json`.** Ingestion runs once; the demo reads JSON. The live demo never depends on the network for retrieval.

---

## 4. Schedule (≈8h, checkpointed)

| # | Phase | Est | Cut-if-behind |
|---|---|---|---|
| P0 | Skeleton + `gemini.py` (60s timeout, 3 retries, backoff, thread pool) | 45m | — critical |
| P1 | Audio extract, silence map, silence-aware chunking | 30m | — critical |
| P2 | Transcription + offset + snap + numeric guard | 60m | — critical |
| P3 | Keyframes + batched vision + fuse into segments | 45m | trim to 3 fields |
| P4 | Embeddings + BM25 + RRF + LLM rerank | 60m | drop rerank, keep RRF |
| P5 | KG extraction + networkx + graph-expansion retrieval | 60m | drop expansion, keep graph |
| P6 | Answer generation + citation validator | 45m | — critical |
| P7 | FastAPI + player + chat + evidence cards + graph tab | 90m | drop graph tab |
| P8 | Full run on TNLA_1, curate 6–8 demo questions, polish | 60m | — critical |

**Checkpoint after P2:** if the full-video transcript with snapped timestamps looks right, the project is essentially de-risked — everything after is well-trodden.

---

## 5. Demo script (what the stakeholder sees)

1. Player loaded with TNLA_1, transcript alongside in Tamil.
2. Ask in English: *"What was said about state autonomy?"* → grounded answer, 3 citation chips.
3. Click a chip → video seeks to that second, evidence card shows Tamil verbatim + English + the frame.
4. Ask a numeric question → answer quotes the Tamil figure; show the numeric guard is why it's trustworthy.
5. Switch to Knowledge Graph tab → entities and relations, click a node → its segments → seek.
6. Ask a cross-cutting question that only the graph can answer (entity appears in distant segments).

---

## 6. Success rating

### **82 / 100**

(demo quality only, deployment excluded, as requested)

**What earns it:** the hard part — Tamil ASR — is already proven excellent on the real file; citation integrity is *structural* (ids, not timestamps, validated against context) so it cannot silently break; corpus is small enough that hybrid retrieval will be near-perfect; no GPU/infra risk; everything verified on the actual machine.

**Deductions:**
- −6 **Vision is genuinely thin** on talking-head footage. Real limitation, not fixable today.
- −5 **KG risks being decorative** — graphs pay off across many videos; on one 27-min clip its unique lift is modest.
- −4 **UI polish is the tightest budget** (P7) and is what "feels like a product."
- −3 **Timestamps land ±1–2s**, not frame-exact, even after snapping.

**Rises to ~90** if the second video arrives early enough for cross-video graph queries — that is where this architecture actually shines.

**Biggest residual risk:** the intermittent 300s connection stall. Mitigated by explicit timeouts + retry + caching, but it is why P0 is non-negotiable.
