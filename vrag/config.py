"""Central configuration: paths, model names, and tunables.

Every value that a later phase might want to tweak lives here, so no magic
numbers get scattered through the pipeline.
"""
from __future__ import annotations

import os
import pathlib

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths.  ROOT is the project root (parent of the `vrag` package).
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
ARTIFACT_DIR = DATA_DIR / "artifacts"   # pipeline outputs, keyed by video_id
CACHE_DIR = DATA_DIR / "cache"          # content-addressed API response cache
PROXY_DIR = DATA_DIR / "proxy"          # browser-playable H.264 renditions
DEBUG_DIR = CACHE_DIR / "_debug"        # raw responses that failed to parse

# --------------------------------------------------------------------------
# Models.  Verified present on this API key (see PLAN.md §0).
# --------------------------------------------------------------------------
MODEL_SMART = "gemini-3.5-flash"        # transcription, translation, vision, answers
MODEL_EMBED = "gemini-embedding-001"
EMBED_DIM = 1536                        # reduced from native 3072 -> MUST L2-normalise
EMBED_BATCH = 16                        # probe: batch of 16 costs the same as 1

EMBED_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY = "RETRIEVAL_QUERY"

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# --------------------------------------------------------------------------
# Timeouts, in seconds, per class of call.
#
# These exist because of a real failure observed during probing: embedding
# calls that normally take 0.4s intermittently hung at *exactly* 300s — a
# black-holed connection.  An unbounded call would stall a whole ingest run,
# so every call gets an explicit, class-appropriate read timeout.
# --------------------------------------------------------------------------
TIMEOUT_EMBED = 30.0
TIMEOUT_TEXT = 120.0
TIMEOUT_VISION = 180.0
TIMEOUT_AUDIO = 420.0     # a ~6 min audio chunk, uploaded inline and transcribed
TIMEOUT_CONNECT = 10.0

# Retry policy: exponential backoff with full jitter.
MAX_ATTEMPTS = 4
BACKOFF_BASE = 1.5
BACKOFF_CAP = 20.0

# Client-side pacing, so we degrade gracefully instead of hammering into 429s.
RPM_LIMIT = 240
MAX_WORKERS = 6

# --------------------------------------------------------------------------
# Safety settings.
#
# This corpus is legislative debate: sharp political argument is the *subject
# matter*.  Default filters can refuse it, which would show up as a mysterious
# empty segment.  We disable them so that a block becomes impossible rather
# than silent.  HARM_CATEGORY_CIVIC_INTEGRITY is verified separately by the
# P0 smoke test before being relied upon.
# --------------------------------------------------------------------------
SAFETY_CATEGORIES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",  # verified supported by the P0 smoke test
]


def safety_settings() -> list[dict]:
    return [{"category": c, "threshold": "BLOCK_NONE"} for c in SAFETY_CATEGORIES]


class ConfigError(RuntimeError):
    """Raised for a misconfiguration the user must fix (never retried)."""


_loaded = False


def _ensure_env_loaded() -> None:
    global _loaded
    if _loaded:
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise ConfigError(
            f"No .env file at {env_path}.\n"
            "Create one containing:  GEMINI_API_KEY=AIza..."
        )
    # utf-8-sig tolerates a BOM, which Windows editors like to add.
    load_dotenv(env_path, encoding="utf-8-sig", override=False)
    _loaded = True


def get_api_key() -> str:
    """Return the Gemini API key, with an actionable error if it is unusable."""
    _ensure_env_loaded()
    key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"').strip("'")
    if not key:
        raise ConfigError(
            f"GEMINI_API_KEY is missing or empty in {ROOT / '.env'}.\n"
            "Expected a line like:  GEMINI_API_KEY=AIza..."
        )
    if not key.startswith("AIza"):
        # A warning, not a hard failure: key formats can change over time.
        import logging
        logging.getLogger(__name__).warning(
            "GEMINI_API_KEY does not start with 'AIza' (len=%d). "
            "If calls fail with 401/403, check the key.", len(key)
        )
    return key


def ensure_dirs() -> None:
    for d in (DATA_DIR, VIDEO_DIR, ARTIFACT_DIR, CACHE_DIR, PROXY_DIR, DEBUG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def artifact_dir(video_id: str) -> pathlib.Path:
    """Per-video artifact directory.

    Everything downstream is keyed by video_id: the second clip must not
    require touching any code.
    """
    d = ARTIFACT_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d
