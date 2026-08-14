"""P1 — audio extraction, silence mapping, and silence-aware chunking.

This module exists to make citations trustworthy.

The probe established the single biggest accuracy risk in this project: Gemini
returns transcript timestamps that are *estimates* — on the real file they came
back as 0/15/35/60/85/100, every one a multiple of 5.  A citation that seeks the
player 1.5s off is a citation the stakeholder does not believe.

Two deterministic mechanisms here fix that, and neither needs the model:

1. **Silence-aware chunking.**  Long audio is split into ~6-minute chunks that
   cut only inside a detected pause, never mid-word.  Each chunk's exact start
   offset is known, so any timestamp drift the model introduces is bounded to a
   single chunk instead of accumulating across 27 minutes.

2. **`snap_to_silence`.**  A real pause in the waveform is ground truth.  P2
   snaps every model-proposed boundary to the nearest one, which the probe
   showed corrects boundaries by 0.5-1.5s.

Everything is keyed by `video_id` — a second clip must require no code change.
"""
from __future__ import annotations

import bisect
import dataclasses
import logging
import pathlib
import re
import subprocess

from vrag import config
from vrag.cache import write_json

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables.
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000        # Gemini transcribes 16k mono happily; 1/6th the bytes of 48k stereo
CHANNELS = 1

# Verified good on this audio during probing: dense, clean boundaries that line
# up with real speech pauses.  Do not "tune" these without listening to the
# result — too aggressive and chunks cut mid-word, too lax and there is nothing
# to snap to.
SILENCE_NOISE_DB = -32.0
SILENCE_MIN_DUR = 0.35

# ~2 min chunks.  NOT a size limit — a model-accuracy limit, found the hard way:
# on ~6-min chunks Gemini's audio timestamps ran up to 1.67x real time (a 360s
# chunk came back with segments at t=600s) and segments ballooned to 60s.  At
# 120s (the probe regime) timestamps stay in range and land within ~1.5s.
CHUNK_TARGET_S = 120.0
CHUNK_SEARCH_S = 40.0      # how far from the ideal cut we will hunt for a pause
CHUNK_MIN_S = 45.0         # never emit a chunk shorter than this...
CHUNK_MIN_TAIL_S = 30.0    # ...and absorb a short final remainder into the last chunk

SNAP_WINDOW_S = 2.5        # P2 default: a boundary further than this from any pause is left alone

# The API caps inline audio at 100MB *base64-encoded*.  16kHz mono s16le is
# 32000 bytes/s, and base64 inflates by 4/3, so the real limit is ~2340s of
# audio per chunk.  We check against it rather than trusting the arithmetic.
INLINE_B64_LIMIT = 100 * 1024 * 1024


class AudioError(RuntimeError):
    """Audio preparation failed.  Never swallowed — a bad chunk map poisons P2."""


# --------------------------------------------------------------------------
# Small typed records.  These are what the rest of the pipeline passes around.
# --------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclasses.dataclass(frozen=True)
class Chunk:
    index: int
    path: pathlib.Path
    start: float          # absolute offset in the source video, in seconds
    end: float
    cut_at_silence: bool  # False means we had to force a cut mid-audio

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------
# ffmpeg plumbing.
# --------------------------------------------------------------------------
def _run(cmd: list[str], what: str) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command, failing loudly with real output attached.

    ffmpeg writes its progress *and* its errors to stderr, so we always capture
    both streams and only surface them when something actually went wrong.
    """
    log.debug("%s: %s", what, " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except FileNotFoundError as exc:
        raise AudioError(
            f"{cmd[0]} not found on PATH. Install ffmpeg 9 and reopen the shell."
        ) from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
        raise AudioError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return proc


def probe_duration(media_path: pathlib.Path) -> float:
    """Exact duration in seconds, straight from the container."""
    proc = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ], f"ffprobe {media_path.name}")
    raw = proc.stdout.strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise AudioError(f"ffprobe returned no duration for {media_path}: {raw!r}") from exc


def extract_audio(video_path: pathlib.Path, out_wav: pathlib.Path,
                  overwrite: bool = False) -> pathlib.Path:
    """Decode the video's audio to 16kHz mono PCM WAV.

    `-vn` means the AV1 video stream is never decoded, so this is fast despite
    the source being AV1.
    """
    video_path = pathlib.Path(video_path)
    if not video_path.exists():
        raise AudioError(f"Video not found: {video_path}")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if out_wav.exists() and not overwrite:
        log.info("audio: reusing existing %s", out_wav.name)
        return out_wav

    log.info("audio: extracting %s -> %dHz mono WAV", video_path.name, SAMPLE_RATE)
    _run([
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ], "audio extraction")

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise AudioError(f"Audio extraction produced nothing at {out_wav}")
    return out_wav


def cut_wav(src: pathlib.Path, start_local: float, end_local: float,
            out_wav: pathlib.Path) -> pathlib.Path:
    """Cut [start_local, end_local] (seconds within src) to a new WAV."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(src),
        "-ss", f"{start_local:.3f}", "-to", f"{end_local:.3f}",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ], f"cut {out_wav.name}")
    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise AudioError(f"cut_wav produced nothing at {out_wav}")
    return out_wav


# --------------------------------------------------------------------------
# Silence detection.
# --------------------------------------------------------------------------
_RE_SIL_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_RE_SIL_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def detect_silences(wav_path: pathlib.Path, duration: float,
                    noise_db: float = SILENCE_NOISE_DB,
                    min_dur: float = SILENCE_MIN_DUR) -> list[Silence]:
    """Map every pause in the audio.

    `silencedetect` reports to stderr as unpaired `silence_start:` /
    `silence_end:` lines.  If the file ends during a pause the final
    `silence_end` is never printed, so we close that interval at the known
    duration rather than dropping it — the end of the file is exactly where a
    final segment boundary wants to snap.
    """
    log.info("audio: detecting silences (noise=%gdB, min=%.2fs)", noise_db, min_dur)
    proc = _run([
        "ffmpeg", "-hide_banner", "-nostdin", "-nostats",
        "-i", str(wav_path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-",
    ], "silencedetect")

    silences: list[Silence] = []
    open_start: float | None = None
    for line in (proc.stderr or "").splitlines():
        m = _RE_SIL_START.search(line)
        if m:
            if open_start is not None:
                log.warning("silencedetect: two starts without an end at %.2fs; keeping the later",
                            float(m.group(1)))
            open_start = max(0.0, float(m.group(1)))
            continue
        m = _RE_SIL_END.search(line)
        if m and open_start is not None:
            end = min(duration, float(m.group(1)))
            if end > open_start:
                silences.append(Silence(open_start, end))
            open_start = None

    if open_start is not None and duration > open_start:
        silences.append(Silence(open_start, duration))

    silences.sort(key=lambda s: s.start)
    if not silences:
        raise AudioError(
            f"No silences detected in {wav_path.name}. Either the audio is one "
            f"continuous sound or noise={noise_db}dB is wrong for this material. "
            "Chunking and timestamp snapping both depend on this map."
        )

    total = sum(s.duration for s in silences)
    log.info("audio: %d silences, %.1fs total (%.0f%% of the clip is pause)",
             len(silences), total, 100.0 * total / duration if duration else 0.0)
    return silences


# --------------------------------------------------------------------------
# Snapping — used by P2 on every model-proposed boundary.
# --------------------------------------------------------------------------
def snap_to_silence(t: float, silences: list[Silence],
                    window: float = SNAP_WINDOW_S,
                    prefer: str | None = None) -> float:
    """Move a timestamp onto the nearest real pause in the audio.

    `prefer` encodes what the timestamp *means*, which changes what the correct
    landing point is:

      "start" — a segment's opening.  Snap to a `silence_end`: the instant
                speech resumes.  Landing on a silence_start would open the
                segment inside the preceding pause.
      "end"   — a segment's close.  Snap to a `silence_start`: the instant
                speech stops.
      None    — snap to whichever boundary of either kind is nearest.

    A timestamp with no pause within `window` is returned unchanged.  That is
    deliberate: mid-sentence, there is no better answer than the model's guess,
    and inventing one would be worse than leaving it.
    """
    if not silences:
        return t

    if prefer == "start":
        candidates = _boundary_index(silences, "end")
    elif prefer == "end":
        candidates = _boundary_index(silences, "start")
    elif prefer is None:
        candidates = _boundary_index(silences, "both")
    else:
        raise ValueError(f"prefer must be 'start', 'end' or None, got {prefer!r}")

    best = _nearest(candidates, t)
    if best is None or abs(best - t) > window:
        return t
    return best


_BOUNDARY_CACHE: dict[tuple[int, str], list[float]] = {}


def _boundary_index(silences: list[Silence], kind: str) -> list[float]:
    """Sorted boundary times, memoised per silence-list so P2 can call this ~80x."""
    key = (id(silences), kind)
    cached = _BOUNDARY_CACHE.get(key)
    if cached is not None and len(cached) >= len(silences):
        return cached

    if kind == "start":
        vals = [s.start for s in silences]
    elif kind == "end":
        vals = [s.end for s in silences]
    else:
        vals = [v for s in silences for v in (s.start, s.end)]
    vals.sort()
    _BOUNDARY_CACHE[key] = vals
    return vals


def _nearest(sorted_vals: list[float], t: float) -> float | None:
    """Closest value to t via binary search; ties go to the earlier value."""
    if not sorted_vals:
        return None
    i = bisect.bisect_left(sorted_vals, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_vals):
            v = sorted_vals[j]
            if best is None or abs(v - t) < abs(best - t):
                best = v
    return best


# --------------------------------------------------------------------------
# Chunk planning.
# --------------------------------------------------------------------------
def plan_chunk_spans(duration: float, silences: list[Silence],
                     target: float = CHUNK_TARGET_S,
                     search: float = CHUNK_SEARCH_S) -> list[tuple[float, float, bool]]:
    """Choose chunk boundaries that fall inside real pauses.

    Returns `(start, end, cut_at_silence)` triples covering [0, duration] with
    no gaps and no overlaps.

    For each cut we look for the pause whose midpoint is closest to the ideal
    position, within +/- `search`.  Cutting at the *midpoint* of a pause (rather
    than at its edge) leaves a margin of silence on both sides, so neither the
    chunk that ends nor the one that begins can clip a word.
    """
    if duration <= 0:
        raise AudioError(f"Refusing to chunk a {duration}s file")

    spans: list[tuple[float, float, bool]] = []
    mids = sorted(s.mid for s in silences)
    start = 0.0

    while duration - start > CHUNK_MIN_TAIL_S + 1e-6:
        ideal = start + target
        if ideal >= duration:
            break

        lo = max(start + CHUNK_MIN_S, ideal - search)
        hi = min(duration - CHUNK_MIN_TAIL_S, ideal + search)

        cut, at_silence = None, False
        if hi > lo:
            window_mids = mids[bisect.bisect_left(mids, lo):bisect.bisect_right(mids, hi)]
            if window_mids:
                cut = min(window_mids, key=lambda m: abs(m - ideal))
                at_silence = True

        if cut is None:
            # Degrade loudly: a forced cut may clip a word, and P2's transcript
            # will show it.  Better a logged imperfection than a silent one.
            cut = min(ideal, duration - CHUNK_MIN_TAIL_S)
            log.warning("audio: no pause within +/-%.0fs of %.1fs — forcing a cut at %.1fs "
                        "(a word may be clipped at this boundary)", search, ideal, cut)

        spans.append((start, cut, at_silence))
        start = cut

    spans.append((start, duration, False if not spans else spans[-1][2]))
    # The final span's flag describes its *opening* cut; its close is the end of
    # the file, which is a true boundary by definition.
    spans[-1] = (spans[-1][0], duration, True if len(spans) == 1 else spans[-1][2])

    _validate_spans(spans, duration)
    return spans


def _validate_spans(spans: list[tuple[float, float, bool]], duration: float) -> None:
    """Prove the chunk map loses no audio. A dropped second is a lost citation."""
    if not spans:
        raise AudioError("Chunk planning produced no spans")
    if abs(spans[0][0]) > 1e-6:
        raise AudioError(f"Chunks do not start at 0 (got {spans[0][0]})")
    if abs(spans[-1][1] - duration) > 1e-6:
        raise AudioError(f"Chunks end at {spans[-1][1]}, video is {duration}")
    for (a_start, a_end, _), (b_start, _, _) in zip(spans, spans[1:]):
        if a_end <= a_start:
            raise AudioError(f"Empty or reversed chunk [{a_start}, {a_end}]")
        if abs(b_start - a_end) > 1e-6:
            raise AudioError(f"Gap or overlap between chunks at {a_end} -> {b_start}")


def split_audio(wav_path: pathlib.Path, spans: list[tuple[float, float, bool]],
                out_dir: pathlib.Path, overwrite: bool = False) -> list[Chunk]:
    """Cut the master WAV into the planned chunks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Chunk] = []

    for i, (start, end, at_silence) in enumerate(spans):
        out = out_dir / f"chunk_{i:03d}.wav"
        if not out.exists() or overwrite:
            _run([
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-i", str(wav_path),
                "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
                "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
                str(out),
            ], f"split chunk {i}")

        # Verify rather than assume: check what ffmpeg actually wrote.
        actual = probe_duration(out)
        planned = end - start
        if abs(actual - planned) > 0.25:
            raise AudioError(
                f"chunk_{i:03d}.wav is {actual:.2f}s but was planned as {planned:.2f}s. "
                "Timestamp offsets would be wrong for every segment in this chunk."
            )

        b64_size = (out.stat().st_size * 4 + 2) // 3
        if b64_size > INLINE_B64_LIMIT:
            raise AudioError(
                f"chunk_{i:03d}.wav is {b64_size / 1e6:.0f}MB base64-encoded, over the "
                f"{INLINE_B64_LIMIT / 1e6:.0f}MB inline cap. Lower CHUNK_TARGET_S."
            )

        chunks.append(Chunk(i, out, start, end, at_silence))
        log.info("audio: chunk %02d  %7.1f -> %7.1f s  (%5.1f s, %4.1f MB, cut=%s)",
                 i, start, end, planned, out.stat().st_size / 1e6,
                 "silence" if at_silence else "FORCED")

    return chunks


# --------------------------------------------------------------------------
# Top-level entry point.
# --------------------------------------------------------------------------
def prepare_audio(video_path: pathlib.Path, video_id: str | None = None,
                  overwrite: bool = False) -> dict:
    """Run the whole of P1 for one video and persist `audio.json`.

    Returns the artifact dict.  P2 reads it back with `load_audio_artifact`.
    """
    video_path = pathlib.Path(video_path).resolve()
    video_id = video_id or video_path.stem
    art_dir = config.artifact_dir(video_id)
    audio_dir = art_dir / "audio"

    duration = probe_duration(video_path)
    log.info("audio: %s is %.1fs (%d:%02d)", video_path.name, duration,
             int(duration // 60), int(duration % 60))

    wav = extract_audio(video_path, audio_dir / "full.wav", overwrite=overwrite)
    wav_duration = probe_duration(wav)
    if abs(wav_duration - duration) > 1.0:
        log.warning("audio: WAV is %.1fs but the video is %.1fs — using the WAV duration, "
                    "since that is what the model will actually hear", wav_duration, duration)
        duration = wav_duration

    silences = detect_silences(wav, duration)
    spans = plan_chunk_spans(duration, silences)
    chunks = split_audio(wav, spans, audio_dir / "chunks", overwrite=overwrite)

    forced = [c.index for c in chunks if not c.cut_at_silence and c.index > 0]
    if forced:
        log.warning("audio: %d chunk boundary/ies were forced, not on a pause: %s",
                    len(forced), forced)

    total_silence = sum(s.duration for s in silences)
    artifact = {
        "video_id": video_id,
        "video_path": _rel(video_path),
        "duration": round(duration, 3),
        "wav_path": _rel(wav),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "silence": {
            "noise_db": SILENCE_NOISE_DB,
            "min_duration": SILENCE_MIN_DUR,
            "count": len(silences),
            "total_seconds": round(total_silence, 3),
            "speech_ratio": round(1.0 - total_silence / duration, 4) if duration else None,
            # Flat [start, end] pairs: compact on disk, trivial to rehydrate.
            "intervals": [[round(s.start, 3), round(s.end, 3)] for s in silences],
        },
        "chunks": [
            {
                "index": c.index,
                "path": _rel(c.path),
                "start": round(c.start, 3),
                "end": round(c.end, 3),
                "duration": round(c.duration, 3),
                "cut_at_silence": c.cut_at_silence,
            }
            for c in chunks
        ],
        "chunk_boundaries_forced": forced,
    }

    out = art_dir / "audio.json"
    write_json(out, artifact)
    log.info("audio: wrote %s (%d chunks, %d silences)", out, len(chunks), len(silences))
    return artifact


def load_audio_artifact(video_id: str) -> dict:
    """Read back `audio.json`, with an actionable error if P1 has not been run."""
    from vrag.cache import read_json
    p = config.artifact_dir(video_id) / "audio.json"
    if not p.exists():
        raise AudioError(
            f"No audio artifact for '{video_id}' at {p}. "
            f"Run:  python scripts/p1_audio.py --video data/videos/{video_id}.mp4"
        )
    return read_json(p)


def silences_from_artifact(artifact: dict) -> list[Silence]:
    """Rehydrate the silence map for `snap_to_silence`."""
    return [Silence(a, b) for a, b in artifact["silence"]["intervals"]]


def chunks_from_artifact(artifact: dict) -> list[Chunk]:
    return [
        Chunk(c["index"], config.ROOT / c["path"], c["start"], c["end"], c["cut_at_silence"])
        for c in artifact["chunks"]
    ]


def _rel(p: pathlib.Path) -> str:
    """Store paths relative to the project root so artifacts survive a move."""
    p = pathlib.Path(p).resolve()
    try:
        return p.relative_to(config.ROOT).as_posix()
    except ValueError:
        return p.as_posix()
