# TNLA Video RAG

**Ask questions about Tamil Nadu Legislative Assembly video — get answers with
clickable citations that seek the video to the exact moment they came from.**

Give it a session recording; it understands the Tamil speech, reads what's on
screen, works out *who* is speaking (and verifies it against official member
portraits), and builds a hybrid search index plus a knowledge graph. Then ask in
English or Tamil:

> *"What did Udhayanidhi Stalin say about delimitation?"*

…and every claim in the answer carries a chip like `▸ S012 · 3:46` — click it and
the player jumps to that second, with the Tamil verbatim and the speaker's
portrait alongside. **No citation = a bug**: answers may only cite segment ids
that exist, code (not the model) maps ids to timestamps, and an answer with no
valid evidence is shown as a refusal, never as fact.

## Quickstart

Requirements: **Python 3.11+**, **ffmpeg** on PATH, a **Gemini API key**.

```bash
git clone <this-repo> && cd video_rag
pip install -r requirements.txt

cp .env.example .env          # then put your GEMINI_API_KEY inside

# drop your video in data/videos/ and run ONE command:
python ingest.py data/videos/MY_SESSION.mp4 --serve
```

That runs the entire pipeline — audio → transcript → speaker attribution →
face verification → search index → knowledge graph → browser proxy — then opens
the UI at **http://127.0.0.1:8000**. A 27-minute video ingests in a few minutes
for under $1 of API cost; ~4 hours ≈ $3–5. Every call is disk-cached, so
re-running is free and a crash resumes where it stopped.

Already ingested? Just `python serve.py`.

## What's in the box

| | |
|---|---|
| `ingest.py` | the one-command pipeline (resumable, phase-skipping) |
| `serve.py` | the demo UI: player + live transcript + chat + knowledge graph + speakers |
| `vrag/` | the library — one module per pipeline stage |
| `scripts/` | individual phase runners + `export_neo4j.py` (mirror the graph into Neo4j for viewing) |
| `assets/roster/` | all 234 assembly members: names (English + Tamil), party, role, official portrait |
| `web/` | the single-file frontend (no build step, no CDN — works offline) |
| `docs/` | [HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) — full architecture walkthrough · PLAN.md · HANDOFF.md |
| `data/` | runtime only (videos, artifacts, cache, proxies) — never committed |

## How it stays trustworthy (short version)

- **Timestamps** are corrected against the audio itself: every boundary snaps to
  a real measured pause, and chunks whose model-clock drifts are split and
  re-anchored to ffmpeg-known offsets. Verified by independently transcribing
  5-second slices at claimed positions.
- **Speakers** are never guessed: names come from the Speaker's spoken
  announcements and on-screen graphics, resolved against the closed 234-member
  roster, then verified by photo lineups against official portraits.
  "Unidentified" is a legal, displayed value.
- **Numbers** shown to users always come from the Tamil source text — the
  English translation is a search aid, cross-checked by a numeric guard.

Full detail with diagrams: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Optional: Neo4j graph view

The pipeline needs no database. If you want the knowledge graph in Neo4j
Browser's visualization (`pip install neo4j`, a local Neo4j running):

```bash
python scripts/export_neo4j.py --video-id MY_SESSION --password <dbpass>
```
