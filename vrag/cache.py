"""Content-addressed cache for API responses.

Why it matters for this project: the demo must never depend on a live
ingestion run.  Every Gemini response is written to disk keyed by a hash of
the *exact* request, so re-running the pipeline is instant and free, and a
crash at phase 7 costs nothing to recover from.

The key is derived from the full request body (model, prompt, schema,
parameters), which means changing a prompt automatically invalidates the
cache — no stale results silently surviving an edit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import tempfile
from typing import Any

log = logging.getLogger(__name__)

# Strings longer than this are hashed rather than embedded in the cache key,
# so a 70 MB base64 audio payload doesn't get serialised twice per lookup.
_BIG_STRING = 4096


def _canonical(obj: Any) -> Any:
    """Recursively normalise an object into something cheap and stable to hash."""
    if isinstance(obj, str):
        if len(obj) > _BIG_STRING:
            return "sha256:" + hashlib.sha256(obj.encode("utf-8")).hexdigest()
        return obj
    if isinstance(obj, dict):
        return {k: _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def hash_payload(payload: Any) -> str:
    blob = json.dumps(_canonical(payload), sort_keys=True,
                      ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, root: pathlib.Path, enabled: bool = True):
        self.root = pathlib.Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, namespace: str, key: str) -> pathlib.Path:
        safe_ns = "".join(c if c.isalnum() or c in "-_." else "_" for c in namespace)
        return self.root / safe_ns / f"{key}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(namespace, key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with p.open("r", encoding="utf-8") as fh:
                value = json.load(fh)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # A truncated file from an interrupted run is a miss, not a crash.
            log.warning("Corrupt cache entry %s (%s); treating as miss", p.name, exc)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, namespace: str, key: str, value: Any) -> None:
        """Write atomically so an interrupted run can never leave half a file."""
        if not self.enabled:
            return
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(value, fh, ensure_ascii=False)
            os.replace(tmp, p)  # atomic on Windows and POSIX alike
        except BaseException:
            pathlib.Path(tmp).unlink(missing_ok=True)
            raise

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (100.0 * self.hits / total) if total else 0.0
        return f"cache hits={self.hits} misses={self.misses} ({rate:.0f}% hit)"


def write_json(path: pathlib.Path, value: Any) -> None:
    """Atomic, UTF-8, human-readable artifact write (used by later phases).

    ensure_ascii=False keeps Tamil readable in the file rather than storing it
    as \\uXXXX escapes — important when you need to eyeball a transcript.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: pathlib.Path) -> Any:
    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
