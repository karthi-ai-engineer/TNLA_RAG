# HANDOFF — read this first

Session handoff for the Multimodal Video RAG build. Updated 2026-08-14, end of the
P0→P7 build session. The user is a capable beginner who wants an expert pilot —
flag bad ideas plainly, recommend one better option, keep moving, and EXPLAIN each
phase's design before running it (the user has asked for this repeatedly).

**Read in this order:** `CLAUDE.md` at repo root (spec + non-negotiables) →
`docs/PLAN.md` (original research) → this file (current state; supersedes
PLAN.md where they disagree). `docs/HOW_IT_WORKS.md` is the user-facing
architecture walkthrough. Since this file was first written the repo was
reorganized: `ingest.py` (one-command pipeline) + `serve.py` at root,
roster moved to `assets/roster/` (committed), docs into `docs/`.

---

## 1. Where the project stands

**P0–P7 are all COMPLETE and verified on `TNLA_trimmed.mp4`.** The full pipeline
runs end-to-end: audio → transcript → attribution → face verification → hybrid
index → knowledge graph → cited answers → web UI.

```
python scripts/p0_smoke.py                     # 15/15 (now hermetic — clears its
                                               #        own cache namespaces first)
python scripts/p1_audio.py --video data/videos/<f>.mp4    # audio + silence map
python scripts/p15_roster.py                   # once, video-independent
python scripts/p2_transcribe.py --video-id X   # transcript (accuracy core)
python scripts/p3_frames.py  --video ...       # keyframes + vision
python scripts/p35_attribute.py --video-id X   # speaker attribution
python scripts/p36_faceverify.py --video-id X  # face-lineup verification
python scripts/p4_index.py   --video-id X      # embeddings + BM25
python scripts/p5_graph.py   --video-id X      # knowledge graph
python scripts/p6_answer.py  --video-id X      # Q&A smoke test
python -m uvicorn vrag.app:app --port 8000     # the demo UI (needs H.264 proxy
                                               #  in data/proxy/<video_id>.mp4)
```

Everything is cached — re-running any phase is free and near-instant. Artifacts
per video in `data/artifacts/<video_id>/`: audio.json, segments.json, turns.json,
frames.json + frames/, faceverify.json, vectors.npy, index_meta.json, graph.json,
numeric_flags.json.

**Current demo video:** `data/videos/TNLA_trimmed.mp4` (27:00, AV1 — same footage
as the original TNLA_1). It is a **delimitation debate**: Udhayanidhi Stalin (LoO)
speaks ~38s–402s, PWD Minister Aadhav Arjuna replies ~415s–1599s, Speaker
Prabhakar J.C.D. presides, S. Regupathy (DMK) interjects at 14.5–33s.
**A ~4-hour full-session video is expected from the user** — the pipeline is
sized for it (2-min chunks ≈ 120 calls, frames ≈ 1000 → ~170 vision calls, hours
≈ nothing hardcoded), but the P7 proxy transcode will take ~15–30 min.

## 2. Discoveries this session (do NOT re-learn these the hard way)

- **Gemini's audio clock is broken on long chunks.** On 6-min chunks timestamps
  ran up to **1.67x real time**; on 2-min chunks ~2/14 still overshoot
  (×1.2–1.3). Discrete wrong "gears", not gradual drift. Mitigations now in
  `vrag/transcribe.py`, in order: (1) 2-min silence-cut chunks
  (`CHUNK_TARGET_S=120`), (2) exact clip duration stated in the prompt (cut
  failures from 9/14 to 2/14), (3) **split-and-reanchor**: an overshooting span
  is split at a real silence and each half re-transcribed with its own
  ffmpeg-known offset (recursive; rescale only as a logged last resort below
  90s). Linear rescale alone was tried and **falsified by verification** —
  boundaries landed 5–10s off.
- **Verify timestamps without ears:** cut a 5s slice at a segment's t_start with
  ffmpeg, transcribe just the slice, compare with the segment's opening words.
  Scratch script pattern in this session; worth promoting to a real script for
  P8 on the 4-hour video.
- **This footage has NO chyrons** (0/130 frames; visually confirmed — raw floor
  feed, only the புதிய தலைமுறை logo). Attribution rests on: Speaker's
  announcements in the transcript (LLM finds them, code resolves them against
  the roster) + procedure rules + face-lineup verification. Chyron code paths
  exist and will light up if a chyroned broadcast is ever ingested.
- **Face-lineup verification (the user's idea) works startlingly well.**
  Frame of the standing speaker + 5 portraits (expected + decoys, shuffled),
  "which matches, or none". It confirmed Udhayanidhi and Aadhav Arjuna, and
  **caught a mis-attribution**: turn 1 was announced as LoO but the lineup
  twice picked S. Regupathy; a second independent lineup confirmed → auto
  re-attributed with evidence kind "face". Two-lineup agreement is the bar for
  re-attribution (`vrag/faceverify.py`).
- **Roster role strings are messy**: "Speaker, Tamil Nadu Legislative Assembly",
  "Minister — Public Works & ...", and "Deputy Leader of the Opposition"
  substring-collides with "Leader of the Opposition". `_holds_role` +
  `PORTFOLIO_TAMIL` in `vrag/attribute.py` handle this — exact/prefix match
  only, portfolio-specific ministers resolvable, generic "Minister" never.
- **The numeric guard fires mostly on benign script differences** (Tamil spells
  numbers as words, EN uses digits). All 7 flags on this video are faithful
  translations. Flags ride along to the UI as "check numbers" badges.
- `TNLA_1.mp4` was deleted by the user; its artifacts linger in
  data/artifacts/TNLA_1 (harmless, ignorable).

## 3. Key decisions (settled — do not reopen)

- **Native Gemini API everywhere** — recorded in CLAUDE.md with rationale.
  The user chose this explicitly on 2026-08-14.
- **Video is current-assembly (2026, TVK govt) footage** — user confirmed.
  Display `current_roles` from the roster.
- Roster source: `C:\Users\10520\karthi-tech\Multimodal_rag\Multimodal-GraphRAG-main\data\roster\members.json`
  (234 members incl. 35 ministers' portfolios + portraits; the Excel adds
  nothing). Built into `data/roster/` by p15 — 227 sitting, 7 vacant excluded,
  Tamil aliases generated by Gemini (cached). Never match against vacant seats.
- Attribution confidence ladder: chyron+intro 0.95 / chyron 0.85 / intro 0.75 /
  procedure (Speaker announcing) 0.70 / face re-attribution 0.70 /
  continuation −0.1. Face verify: confirmed +0.15, contradicted −0.3 + conflict.
  **"Unknown" is a legal, displayed value — never guess.**

## 4. What remains (P8 + polish)

1. **Proxy transcode** for any new video: 
   `ffmpeg -i in.mp4 -c:v libx264 -preset fast -crf 23 -c:a aac -movflags +faststart data/proxy/<id>.mp4`
   (was still running for TNLA_trimmed when this file was written — verify with
   ffprobe before the demo; `moov atom not found` = still writing).
2. **Browser pass over the UI** (`web/index.html` — self-contained, no CDN):
   click chips → seek, transcript autoscroll, graph tab (hand-rolled canvas
   force layout), portraits on cards. Verified via API only, NOT yet eyeballed
   in a browser.
3. **Curate 6–8 demo questions** against the actual video; the working set in
   `scripts/p6_answer.py` DEMO_QUESTIONS all pass (incl. a must-refuse one).
4. When the **4-hour video** arrives: run P1→P6 (ingest is ~all-parallel; only
   transcode + transcription wall-time matter), spot-check timestamps with the
   slice-verification trick at 3–4 points, re-curate questions.
5. Stretch: unknown-turn face identification (open lineup over active cast) —
   code hooks exist; multi-video cross-referencing once a second video exists.

## 5. Environment gotchas (unchanged but still real)

- ffmpeg 9: `-vsync` removed → `-fps_mode vfr`. PowerShell 5.1: no `&&`.
- Windows cp1252: every file handle explicit UTF-8 (`cache.write_json` does).
  PowerShell console mojibake for Tamil is DISPLAY-only; the files are fine.
- AV1 source: browser gets the H.264 proxy; ffmpeg `-ss` before `-i` for fast
  frame seeks (used in `vrag/frames.py`).
- Never commit `.env`, `data/videos/`, `data/artifacts/`, `data/cache/`,
  `data/proxy/`, `data/roster/`.
