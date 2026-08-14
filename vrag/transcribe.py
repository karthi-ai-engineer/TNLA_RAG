"""P2 — transcription: the accuracy core.

Input:  P1's audio chunks (exact known offsets) + silence map.
Output: data/artifacts/<video_id>/segments.json — the SEGMENT list every
        later phase builds on.

The three mechanisms that make the timestamps and numbers trustworthy:

1.  *Offset, don't trust.*  Each chunk is transcribed with chunk-local
    timestamps, then shifted by the chunk's start offset — a number ffmpeg
    gave us, not the model.  Model drift is therefore bounded to one chunk.

2.  *Snap, don't hope.*  Every model-proposed boundary is moved to the
    nearest real pause in the waveform (P1's silence map).  The probe showed
    the model guesses in multiples of 5s; snapping corrects 0.5-1.5s.

3.  *Guard the numbers.*  text_ta and text_en come from the SAME call, and a
    numeric guard cross-checks digits between them.  Mismatches are flagged
    into numeric_flags.json and the segment carries a warning — the English
    gloss is a search aid and is never quoted for figures.

Speaker handling here is deliberately minimal: the model marks *voice turns*
(new_speaker + a within-chunk voice label).  Names are resolved later
(P3.5) against the roster; the transcription step is never asked to guess an
identity.
"""
from __future__ import annotations

import logging
import re

from vrag import config, gemini
from vrag.audio import (Silence, chunks_from_artifact, cut_wav,
                        load_audio_artifact, silences_from_artifact,
                        snap_to_silence)
from vrag.cache import write_json
from vrag.parallel import thread_map

log = logging.getLogger(__name__)


class TranscribeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Response schema — what each chunk call must return.
# ---------------------------------------------------------------------------
_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "text_ta": {"type": "STRING"},
                    "text_en": {"type": "STRING"},
                    "voice": {"type": "STRING"},
                    "new_speaker": {"type": "BOOLEAN"},
                },
                "required": ["start", "end", "text_ta", "text_en",
                             "voice", "new_speaker"],
            },
        }
    },
    "required": ["segments"],
}

_PROMPT = """Transcribe this Tamil audio from a Tamil Nadu Legislative Assembly \
session broadcast. Return a JSON list of segments.

IMPORTANT: this audio clip is EXACTLY {duration:.1f} seconds long. Every start and
end value must be between 0 and {duration:.1f}. Calibrate your timestamps against
this known total length — the final segment must end at or before {duration:.1f}s.

Rules:
- Each segment is one coherent utterance of roughly 5-25 seconds — a sentence or a
  few connected sentences by the same voice. Break at natural pauses.
- start/end are seconds from the beginning of THIS audio clip.
- text_ta: the exact Tamil words spoken, verbatim, in Tamil script. Do not
  paraphrase, do not translate into text_ta, do not omit filler or repetition.
- text_en: a faithful English translation of text_ta. Translate numbers, amounts
  and dates EXACTLY — "ஆயிரம் கோடி" is "one thousand crore", never approximate
  or convert units.
- voice: a label like "V1", "V2" — the same label whenever the same voice speaks
  within this clip. Labels do not need to be names.
- new_speaker: true when this segment's voice differs from the previous segment's.
- Cover ALL speech in the clip. Skip silence, applause, and desk-thumping; never
  invent words for inaudible passages — if a passage is unintelligible, skip it.

These people may be mentioned or speaking (spelling reference for proper nouns):
{names}"""


# ---------------------------------------------------------------------------
# Numeric guard
# ---------------------------------------------------------------------------
# Tamil number words that matter for the "1,000 crore" failure class.
_TA_NUMBER_WORDS = {
    "நூறு": 100, "ஆயிரம்": 1000, "லட்சம்": 100_000, "இலட்சம்": 100_000,
    "கோடி": 10_000_000,
}
_EN_NUMBER_WORDS = {
    "hundred": 100, "thousand": 1000, "lakh": 100_000,
    "crore": 10_000_000, "million": 1_000_000, "billion": 1_000_000_000,
}
_DIGITS_RE = re.compile(r"\d[\d,.]*")


def _digit_tokens(text: str) -> list[str]:
    """Digit runs, comma/point-stripped, order-insensitive compare basis."""
    return sorted(t.replace(",", "").rstrip(".") for t in _DIGITS_RE.findall(text))


def _scale_words(text: str, vocab: dict[str, int]) -> list[int]:
    found = []
    for w, v in vocab.items():
        if w.isascii():
            # Word-bounded and plural-tolerant: "lakhs" is one hit of "lakh",
            # never two hits of two different keys.
            pat = rf"\b{re.escape(w)}s?\b"
            found.extend([v] * len(re.findall(pat, text, re.IGNORECASE)))
        else:
            found.extend([v] * len(re.findall(re.escape(w), text)))
    return sorted(found)


def numeric_guard(text_ta: str, text_en: str) -> list[str]:
    """Return a list of human-readable mismatch descriptions (empty = clean).

    Deliberately conservative: it flags for a human eye rather than trying to
    be a full bilingual number parser.  False positives are cheap; a silent
    100x translation error is not.
    """
    problems = []
    d_ta, d_en = _digit_tokens(text_ta), _digit_tokens(text_en)
    if d_ta != d_en:
        problems.append(f"digits differ: ta={d_ta} en={d_en}")
    s_ta = _scale_words(text_ta, _TA_NUMBER_WORDS)
    s_en = _scale_words(text_en, _EN_NUMBER_WORDS)
    # Compare only when the Tamil side names a scale at all; the guard checks
    # that scale words were not inflated/deflated in translation.
    if s_ta and s_ta != s_en:
        problems.append(
            f"scale words differ: ta={s_ta} en={s_en} "
            f"(ta text: {text_ta[:80]}…)" if len(text_ta) > 80 else
            f"scale words differ: ta={s_ta} en={s_en} (ta text: {text_ta})")
    return problems


# ---------------------------------------------------------------------------
# Per-chunk transcription
# ---------------------------------------------------------------------------
def _roster_names_hint(max_names: int = 60) -> str:
    """Tamil + roman names of office-holders first, then a sample of members.

    The full 227-name list would bloat every prompt; office-holders are the
    people overwhelmingly likely to be speaking or addressed.
    """
    try:
        from vrag.roster import load_roster
        members = load_roster()["members"]
    except Exception as exc:  # roster is optional for P2 — degrade loudly, run anyway
        log.warning("transcribe: roster unavailable (%s); proceeding without name hints", exc)
        return "(no reference list available)"

    def line(m: dict) -> str:
        ta = m.get("tamil_name") or ""
        role = m["current_roles"][0] if m["current_roles"] else "MLA"
        return f"- {ta} ({m['name']}, {m['party_code']}, {role})"

    office = [m for m in members if m["current_roles"] and m["current_roles"] != ["MLA"]]
    plain = [m for m in members if m not in office]
    chosen = office[:max_names] + plain[: max(0, max_names - len(office))]
    return "\n".join(line(m) for m in chosen)


def transcribe_span(client, wav_path, abs_start: float, duration: float,
                    silences: list[Silence], names_hint: str,
                    label: str, spans_dir, depth: int = 0) -> list[dict]:
    """Transcribe one audio file; return segments with ABSOLUTE, SNAPPED times.

    Clock-failure policy, learned the hard way on this exact footage: the
    model's audio clock sometimes runs at a constant wrong speed (x1.2-x1.67,
    even after prompt calibration).  Verification proved a linear rescale is
    NOT accurate enough (boundaries land 5-10s off).  So a span whose
    timestamps overshoot is SPLIT IN HALF at a real silence and each half is
    re-transcribed with its own ffmpeg-known offset — re-anchoring time to
    ground truth.  Rescale survives only as a logged last resort for spans
    already too short to split.
    """
    prompt = _PROMPT.format(names=names_hint, duration=duration)
    out = client.generate_json(
        [gemini.text_part(prompt), gemini.audio_part(wav_path)],
        schema=_SCHEMA,
        timeout=config.TIMEOUT_AUDIO,
        namespace="transcribe",
        max_output_tokens=65536,
    )
    raw = out.get("segments") or []
    if not raw:
        raise TranscribeError(f"span {label}: model returned zero segments")

    max_end = max(float(s["end"]) for s in raw)
    scale = 1.0
    if max_end > duration * 1.05 + 2.0:
        factor = max_end / duration
        if duration >= 90.0 and depth < 3:
            split_abs = _pick_split(abs_start, duration, silences)
            if split_abs is not None:
                log.warning("span %s: clock ran %.2fx fast — splitting at %.1fs "
                            "and re-anchoring both halves", label, factor, split_abs)
                halves = []
                for sub_label, a, b in ((f"{label}a", abs_start, split_abs),
                                        (f"{label}b", split_abs, abs_start + duration)):
                    sub_wav = spans_dir / f"{sub_label}.wav"
                    cut_wav(wav_path, a - abs_start, b - abs_start, sub_wav)
                    halves.extend(transcribe_span(
                        client, sub_wav, a, b - a, silences, names_hint,
                        sub_label, spans_dir, depth + 1))
                return halves
        scale = duration / max_end
        log.warning("span %s: clock %.2fx fast and span too short to split — "
                    "rescaling by %.3f (LAST RESORT, boundaries approximate)",
                    label, factor, scale)

    segs = []
    for s in raw:
        start_local, end_local = float(s["start"]), float(s["end"])
        if end_local <= start_local:
            log.warning("span %s: dropping reversed segment %.1f-%.1f",
                        label, start_local, end_local)
            continue
        if not (s["text_ta"] or "").strip():
            continue

        # Local -> (rescale if last-resort) -> absolute (known offset) -> snap.
        t0 = abs_start + start_local * scale
        t1 = abs_start + end_local * scale
        t0s = snap_to_silence(t0, silences, prefer="start")
        t1s = snap_to_silence(t1, silences, prefer="end")
        if t1s <= t0s:               # snapping collapsed it — keep unsnapped ends
            t0s, t1s = t0, t1

        segs.append({
            "t_start": round(t0s, 2),
            "t_end": round(t1s, 2),
            "t_start_raw": round(t0, 2),        # kept for eyeballing snap deltas
            "t_end_raw": round(t1, 2),
            "text_ta": s["text_ta"].strip(),
            "text_en": (s["text_en"] or "").strip(),
            "chunk": label,
            "voice_local": f"{label}:{s['voice']}",   # voice labels don't
            "new_speaker": bool(s["new_speaker"]),    # cross spans
            "rescaled": scale != 1.0,
        })

    segs.sort(key=lambda x: x["t_start"])
    log.info("span %s: %d segments, %.1f-%.1fs covered",
             label, len(segs),
             segs[0]["t_start"] if segs else 0, segs[-1]["t_end"] if segs else 0)
    return segs


def _pick_split(abs_start: float, duration: float,
                silences: list[Silence]) -> float | None:
    """Silence midpoint nearest the span's centre (within the middle half)."""
    centre = abs_start + duration / 2
    lo, hi = abs_start + duration * 0.25, abs_start + duration * 0.75
    mids = [s.mid for s in silences if lo <= s.mid <= hi]
    return min(mids, key=lambda m: abs(m - centre)) if mids else None


# ---------------------------------------------------------------------------
# Whole-video driver
# ---------------------------------------------------------------------------
def transcribe_video(client, video_id: str) -> dict:
    art = load_audio_artifact(video_id)
    chunks = chunks_from_artifact(art)
    silences = silences_from_artifact(art)
    names_hint = _roster_names_hint()
    spans_dir = config.artifact_dir(video_id) / "audio" / "spans"

    per_chunk = thread_map(
        lambda ch: transcribe_span(client, ch.path, ch.start, ch.duration,
                                   silences, names_hint, f"C{ch.index}",
                                   spans_dir),
        chunks, workers=min(client.max_workers, len(chunks)), desc="transcribe")

    segments: list[dict] = []
    for segs in per_chunk:
        segments.extend(segs)
    segments.sort(key=lambda s: (s["t_start"], s["t_end"]))

    # Assign ids AFTER the global sort, so S001 < S002 in time, always.
    flags = []
    for i, s in enumerate(segments, 1):
        s["id"] = f"S{i:03d}"
        problems = numeric_guard(s["text_ta"], s["text_en"])
        s["numeric_flags"] = problems
        if problems:
            flags.append({"id": s["id"], "t_start": s["t_start"],
                          "problems": problems,
                          "text_ta": s["text_ta"], "text_en": s["text_en"]})

    # Coverage check: did the transcript actually span the video?
    dur = art["duration"]
    covered = sum(s["t_end"] - s["t_start"] for s in segments)
    speech = dur * art["silence"]["speech_ratio"]
    gaps = _find_gaps(segments, dur, min_gap=20.0)
    if covered < 0.5 * speech:
        raise TranscribeError(
            f"Transcript covers only {covered:.0f}s of ~{speech:.0f}s speech — "
            "a chunk probably failed silently. Not writing segments.json.")
    for g in gaps:
        log.warning("transcript gap %.1f-%.1fs (%.0fs) — check what's there",
                    g[0], g[1], g[1] - g[0])

    result = {
        "video_id": video_id,
        "duration": dur,
        "segment_count": len(segments),
        "covered_seconds": round(covered, 1),
        "numeric_flag_count": len(flags),
        "gaps_over_20s": [[round(a, 1), round(b, 1)] for a, b in gaps],
        "segments": segments,
    }
    out_dir = config.artifact_dir(video_id)
    write_json(out_dir / "segments.json", result)
    write_json(out_dir / "numeric_flags.json", flags)
    log.info("transcribe: %d segments, %d numeric flags -> %s",
             len(segments), len(flags), out_dir / "segments.json")
    return result


def _find_gaps(segments: list[dict], duration: float,
               min_gap: float) -> list[tuple[float, float]]:
    gaps, cursor = [], 0.0
    for s in segments:
        if s["t_start"] - cursor >= min_gap:
            gaps.append((cursor, s["t_start"]))
        cursor = max(cursor, s["t_end"])
    if duration - cursor >= min_gap:
        gaps.append((cursor, duration))
    return gaps
