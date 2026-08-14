# How This System Works

A multimodal RAG system over Tamil Nadu Legislative Assembly video. You give it a
video; you ask questions in English or Tamil; it answers **with clickable citations
that seek the video to the exact moment the answer came from**. Every claim traces
to a segment; every segment traces to a second of video. No citation = a bug.

This document explains how each piece works and why it is built that way.

---

## The big picture

```
video.mp4
  │
  ├── AUDIO ──► silence map ──► 2-min chunks ──► Gemini transcription
  │                                   │  (Tamil verbatim + English gloss + voice turns)
  │                                   ▼
  │                     timestamp correction (snap + re-anchor)
  │                                   ▼
  ├── FRAMES ─► scene cuts + 15s grid ─► Gemini vision      SEGMENTS  ◄── the core unit
  │             (read on-screen text,      (setting,        S001…S120: id, t_start, t_end,
  │              spot standing speaker)     chyron, text)   text_ta, text_en, speaker
  │                                   │                          │
  │                                   ▼                          │
  └── ROSTER ──────────────► SPEAKER ATTRIBUTION ────────────────┤
      (234 MLAs, Tamil        announcements + procedure          │
       aliases, portraits)    + face-lineup verification         │
                                                                 ▼
                              ┌──────────────────┬───────────────────────┐
                              ▼                  ▼                       ▼
                        VECTOR INDEX          BM25 INDEX          KNOWLEDGE GRAPH
                        (embeddings)          (keywords)          (entities+relations,
                              │                  │                 every fact → segment id)
                              └───── RRF fusion ─┘                       │
                                        │                                │
                                        ▼                                │
                                  LLM rerank ◄── graph expansion ────────┘
                                        │
                                        ▼
                              ANSWER GENERATION
                        (cites segment IDs only; code maps
                         id → timestamp; validator drops any
                         id not actually in the context)
                                        │
                                        ▼
                          WEB UI: player + chat + evidence
                          cards + knowledge graph tab
```

**The segment is the unit everything is built on** — small enough to embed
(5–25 seconds of speech), precise enough to cite, and the record every graph
edge points back to. A citation is a segment id like `S042`; code, not the
model, converts that to a timestamp. This makes a hallucinated timestamp
*structurally impossible*, not just unlikely.

---

## 1. Audio → trustworthy timestamps (P1 + P2)

**The problem.** Gemini's transcript timestamps are estimates, not measurements.
Probing showed timestamps returned as neat multiples of 5, and worse: on long
audio chunks the model's internal clock ran at the wrong speed entirely — a
360-second chunk came back with segments claiming to end at t=600s (1.67× real
time). Timestamps like that would make every citation seek to the wrong moment.

**The fix is three deterministic layers, none of which trust the model:**

1. **Silence-aware chunking.** ffmpeg's `silencedetect` maps every real pause in
   the waveform (536 pauses in the 27-min demo clip — one per ~3s of speech).
   Audio is cut into ~2-minute chunks whose cut points sit in the *middle of a
   real pause*, never mid-word. Each chunk's start offset is a number **ffmpeg
   measured**, so model drift can never accumulate past one chunk.
2. **Calibration + split-and-reanchor.** Each transcription prompt states the
   chunk's exact duration ("this clip is EXACTLY 119.8 seconds") — that alone cut
   clock failures from 9/14 chunks to 2/14. A chunk whose timestamps still
   overshoot its real length is **split in half at a real silence and each half
   re-transcribed with its own ffmpeg-known offset** — re-anchoring time to
   ground truth. (Linear rescaling was tried first and *disproved by
   verification*: boundaries landed 5–10s off. The code keeps it only as a
   logged last resort for spans too short to split.)
3. **Snapping.** Every model-proposed boundary is moved to the nearest real
   pause within ±2.5s — a segment start snaps to the instant speech resumes, a
   segment end to the instant it stops. Median correction on the demo video:
   0.75s. If no pause is nearby, the timestamp is left alone — mid-sentence
   there is no better answer than the model's guess.

**How we know it works:** verification cuts a 5-second slice of audio at a
segment's claimed `t_start`, transcribes *just that slice* independently, and
compares with the segment's opening words. On the demo video the slices match
word-for-word ("20 லட்சம் 30 லட்சம் 50 லட்சம்…" heard exactly where claimed).

---

## 2. Who is speaking (P1.5 + P3 + P3.5 + P3.6)

Nobody is identified by face recognition or voice recognition. **The broadcast
itself carries the names**, and the system reads them, then verifies.

**The roster (P1.5).** All 234 assembly members from official sources: name,
constituency, party, current/former roles, official portrait. Two gaps were
fixed at build time: 7 vacant seats are excluded (their portraits show *former*
members — a trap), and the source has zero Tamil text, so Gemini generated Tamil
script forms of every name once (cached forever). Every name the pipeline ever
assigns must resolve against this closed list — a fabricated name cannot be
attributed. "Unknown speaker" is a legal, displayed value.

**The evidence, strongest first:**

| Evidence | How it works | Confidence |
|---|---|---|
| Chyron | Vision reads the name graphic printed on screen (OCR, not face-ID) | 0.85 (0.95 with a second source) |
| Introduction | Assembly procedure: the Speaker announces every member before they speak ("மாண்புமிகு எதிர்க்கட்சித் தலைவர் அவர்கள்…"). An LLM pass finds these announcements; *code* resolves the name/office against the roster. A portfolio like "பொதுப்பணித்துறை அமைச்சர்" resolves because exactly one member holds it | 0.75 |
| Procedure | The person doing the announcing *is* the presiding Speaker | 0.70 |
| Continuation | A turn split only by our own chunk boundary inherits the current floor-holder | prev − 0.1 |
| Face lineup | See below | +0.15 / −0.3 |

**Face-lineup verification (P3.6).** For each attributed turn, the model gets a
frame of the person standing and speaking (in an assembly, the speaker stands —
everyone else sits) plus **five official portraits: the expected member and four
decoys, shuffled**. Question: *which portrait, if any, shows the standing
person?* This is photo *comparison*, not open-set face recognition — a much
easier, much more reliable task. Rules:

- All votes match the expected portrait → confirmed, confidence rises.
- Votes go elsewhere or "none" → conflict flagged loudly, confidence cut.
  The printed/announced evidence is never silently overridden.
- If all votes converge on **one specific decoy**, a second lineup with fresh
  decoys tests that candidate. Two independent lineups agreeing is the bar for
  re-attribution.

On the demo video this pass confirmed both main speakers **and caught a real
error**: the opening interjection was announced as the Leader of the Opposition
but the lineup twice identified S. Regupathy (DMK) — a second independent
lineup agreed, and the turn was re-attributed automatically. The text pipeline
alone would have shipped that mistake.

**This footage has no chyrons** (raw floor feed, verified across 130 frames), so
on this video attribution rests on introductions + procedure + face lineups. The
chyron path exists and activates automatically on chyroned broadcasts.

---

## 3. Numbers can't lie in translation (P2)

A capable model was caught during planning translating "1,000 crore" as "1 lakh
crore" — a 100× error. Defenses:

- Tamil transcript and English translation come from the **same call**, so they
  cannot drift independently.
- A `numeric_guard` cross-checks digits and scale-words (ஆயிரம்/லட்சம்/கோடி vs
  thousand/lakh/crore) between the two. Mismatches are flagged into
  `numeric_flags.json` and ride along to the UI as a "check numbers" badge.
- The answer prompt requires figures to be **quoted from the Tamil text
  verbatim** — the English gloss is a search aid, never a source. Answers
  actually contain "ஏழு புள்ளி ஒன்று எட்டு" (7.18), not a re-translated number.
- The evidence card always shows the Tamil verbatim next to the answer.

---

## 4. Retrieval: three channels, each covering another's blind spot (P4 + P5)

| Channel | Great at | Blind to |
|---|---|---|
| **Vector** (Gemini embeddings, 1536-d, cosine over numpy) | meaning, paraphrase, cross-language — an English question finds Tamil-origin answers (verified similarity 0.858 across languages) | exact numbers, rare names |
| **BM25** (keyword match over English + Tamil tokens) | exact names, figures, Tamil terms | synonyms, translations |
| **Graph** (entities extracted per segment) | one entity across *distant* moments — all six Ambedkar mentions, minute 8 and minute 19 alike | anything not extracted as an entity |

Vector and BM25 rankings are blended with Reciprocal Rank Fusion (score =
Σ 1/(60+rank)); the top 20 go to an LLM reranker that picks the best 6 for the
actual question (falling back loudly to fusion order if the call fails). Graph
expansion then adds segments linked to any entity named in the question. The
final ~8 segments become the answer context.

Segments are embedded as `"SPEAKER NAME: text"`, so "what did Udhayanidhi say
about X" preferentially retrieves *his* segments.

There is deliberately **no vector database and no graph database in the
pipeline** — 120 segments is a numpy dot product (microseconds) and 100 nodes
is a dict lookup. Both live behind small classes so real stores can be swapped
in when the corpus is hundreds of sessions, not one.

---

## 5. Answering with citations that cannot be faked (P6)

1. The model receives the context segments and must cite **segment ids only** —
   `[S042]` — never timestamps.
2. A validator extracts every cited id and **drops any id not actually present
   in the context**. An answer left with zero valid citations is surfaced as a
   refusal ("the context does not contain this"), never displayed as fact.
3. Code maps each surviving id to its segment's timestamp, speaker, Tamil
   verbatim, portrait, and confidence — that becomes the evidence card, and the
   click seeks the player to `t_start − 1.5s` so the cited line is heard from
   its beginning.

Ask it something the video doesn't contain ("what was said about the weather?")
and it says so plainly, cites nothing, and the UI marks it as a no-evidence
answer.

---

## 6. The knowledge graph (P5 + Neo4j mirror)

Built in two layers:

- **Seeded from the roster** (trusted facts): every attributed speaker becomes a
  Person node carrying party, role, constituency, portrait; `MEMBER_OF` edges to
  Party nodes.
- **Extended by extraction**: per-segment Gemini extraction of parties, schemes,
  places, amounts, laws, dates, topics, and external persons, plus relations
  ("criticised", "wrote letter to", "compared with"). **Every node and edge
  carries the segment ids it came from**; extractions citing ids that weren't in
  the prompt are dropped. Extracted person-names are resolved against the
  roster first so "உதயநிதி", "Udhayanidhi" and "the Leader of the Opposition"
  are one node, not three. The graph never re-decides who spoke — speaker edges
  come from attribution, not extraction.

The pipeline reads `graph.json`. **Neo4j is a view-only mirror** for its
Browser visualization: `scripts/export_neo4j.py` wipes and recreates the video's
subgraph (typed labels, `DISCUSSED`/`CRITICISED`-style relationships, provenance
properties). If Neo4j is down, nothing else notices.

---

## 7. Engineering rules the whole build follows

- **Cache everything.** Every Gemini call is disk-cached, keyed on a hash of the
  full request. Ingestion runs once; re-runs are instant and free; a crash costs
  nothing; the live demo never depends on the network for retrieval. Changing a
  prompt automatically invalidates its cache entries.
- **Degrade loudly, never fake.** Blocked, truncated, empty, or unparseable
  model responses raise *named* exceptions instead of returning junk. A
  half-transcript refuses to write itself (coverage check). A failed reranker
  logs and falls back. An unidentifiable speaker displays as "unidentified".
- **Nothing is hardcoded to one video.** Every artifact lives under
  `data/artifacts/<video_id>/`. A new video — including the full 4-hour
  session — is the same commands with a different filename.
- **Verify, don't assume.** Chunk durations are re-probed after cutting;
  timestamps are checked against independently transcribed audio slices; the
  graph's most-connected nodes are eyeballed; face verification double-checks
  the text pipeline.

---

## 8. Running it

```bash
# one-time
python scripts/p0_smoke.py          # foundation self-test (expect 15/15)
python scripts/p15_roster.py        # build the 234-member roster

# per video (order matters)
python scripts/p1_audio.py --video data/videos/VIDEO.mp4
python scripts/p2_transcribe.py --video-id VIDEO
python scripts/p3_frames.py --video data/videos/VIDEO.mp4
python scripts/p35_attribute.py --video-id VIDEO
python scripts/p36_faceverify.py --video-id VIDEO
python scripts/p4_index.py --video-id VIDEO
python scripts/p5_graph.py --video-id VIDEO
python scripts/p6_answer.py --video-id VIDEO      # Q&A smoke test

# browser player needs an H.264 proxy (source is AV1, which browsers seek badly)
ffmpeg -i data/videos/VIDEO.mp4 -c:v libx264 -preset fast -crf 23 \
       -c:a aac -movflags +faststart data/proxy/VIDEO.mp4

# the demo
python -m uvicorn vrag.app:app --port 8000        # UI at 127.0.0.1:8000

# optional: mirror the graph into Neo4j for viewing
python scripts/export_neo4j.py --video-id VIDEO --password password
```

**Cost:** the entire 27-minute demo video — including every failed experiment —
cost under $1 of Gemini calls. A 4-hour video projects to roughly $3–5, once,
then free forever thanks to the cache.
