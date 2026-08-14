"""UTF-8-safe logging.

Windows consoles default to a legacy code page (cp1252 here).  Logging a line
of Tamil to such a stream raises UnicodeEncodeError and kills the run — which
would be an absurd way to lose a 20-minute ingest.  We reconfigure stdout to
UTF-8, and fall back to error-replacement if even that is unavailable.
"""
from __future__ import annotations

import logging
import sys

_configured = False


def setup_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        return

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # py3.7+
        except (AttributeError, OSError):
            pass  # not a real TTY (e.g. piped); handler below still guards us

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # httpx logs every request at INFO; far too chatty for a 1500-call ingest.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    _configured = True


def preview(text: str, limit: int = 90) -> str:
    """One-line, length-capped rendering of text for log messages."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
