"""Hardened Gemini client — the single doorway for every model call.

Design notes
------------
*Transport.*  We use Gemini's native `generateContent` / `batchEmbedContents`
endpoints for everything, rather than mixing in the OpenAI-compatible layer.
One transport, one set of failure modes.  The native API also exposes three
things the compatibility layer does not, all of which we need:
  - `responseSchema`, which gives reliable structured output;
  - embedding `taskType` (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY), which
    measurably improves retrieval quality;
  - `usageMetadata`, so token spend is observable.

*Authentication.*  The key goes in an `x-goog-api-key` header, never the query
string.  Gemini accepts `?key=`, but then every exception, log line and stack
trace containing a URL would leak the credential.

*Failure policy.*  Transient faults are retried with jittered backoff; user
errors (400/401/403/404) fail immediately with the response body, since
retrying a malformed request is just a slower bug.  Nothing ever returns a
silent None or an empty result — a blocked, truncated or unparseable response
raises a specific, named exception.
"""
from __future__ import annotations

import base64
import json
import logging
import pathlib
import random
import re
import threading
import time
from typing import Any, Sequence

import httpx

from . import config
from .cache import Cache, hash_payload
from .parallel import thread_map

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions — deliberately specific, so callers can tell "my prompt is wrong"
# from "the network flaked" from "the model refused".
# ---------------------------------------------------------------------------
class GeminiError(RuntimeError):
    """Base class for all Gemini failures."""


class GeminiAuthError(GeminiError):
    """401/403 — bad or unauthorised API key.  Never retried."""


class GeminiBadRequest(GeminiError):
    """400/404 — malformed request or unknown model.  Never retried."""


class GeminiBlocked(GeminiError):
    """Prompt or response suppressed by safety/recitation filters."""


class GeminiTruncated(GeminiError):
    """Hit maxOutputTokens; the JSON body is incomplete by definition."""


class GeminiEmptyResponse(GeminiError):
    """finishReason=STOP but no usable text came back."""


class GeminiParseError(GeminiError):
    """Response was not the JSON we asked for."""


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
FATAL_STATUS = {400, 401, 403, 404, 413}

# finishReason values that mean "the model declined", not "something broke".
BLOCKING_FINISH = {
    "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST",
    "SPII", "IMAGE_SAFETY", "LANGUAGE",
}


# ---------------------------------------------------------------------------
# Content part helpers
# ---------------------------------------------------------------------------
def text_part(text: str) -> dict:
    return {"text": text}


def inline_part(data: bytes, mime_type: str) -> dict:
    return {"inline_data": {"mime_type": mime_type,
                            "data": base64.b64encode(data).decode("ascii")}}


def image_part(path: pathlib.Path | str, mime_type: str = "image/jpeg") -> dict:
    return inline_part(pathlib.Path(path).read_bytes(), mime_type)


def audio_part(path: pathlib.Path | str, mime_type: str = "audio/wav") -> dict:
    """Inline audio.  The API caps inline payloads at 100 MB base64-encoded;
    we chunk long audio well below that in P1, so this stays safe."""
    raw = pathlib.Path(path).read_bytes()
    encoded_mb = len(raw) * 4 / 3 / 1e6
    if encoded_mb > 90:
        raise ValueError(
            f"Inline audio is ~{encoded_mb:.0f} MB base64-encoded, near the "
            "100 MB API cap. Chunk it smaller or use the Files API."
        )
    return inline_part(raw, mime_type)


# ---------------------------------------------------------------------------
# Client-side pacing
# ---------------------------------------------------------------------------
class RateLimiter:
    """Thread-safe token bucket that spaces calls to at most `rpm` per minute."""

    def __init__(self, rpm: int):
        self._interval = 60.0 / max(1, rpm)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            time.sleep(wait)


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        cache: Cache | None = None,
        use_cache: bool = True,
        rpm: int | None = None,
        max_workers: int | None = None,
    ):
        self._key = api_key or config.get_api_key()
        config.ensure_dirs()
        self.cache = cache if cache is not None else Cache(config.CACHE_DIR, enabled=use_cache)
        self.max_workers = max_workers or config.MAX_WORKERS
        self._limiter = RateLimiter(rpm or config.RPM_LIMIT)
        self._counter_lock = threading.Lock()
        self.request_count = 0          # network calls actually made (cache misses)
        self.total_tokens = 0
        self._client = httpx.Client(
            headers={"x-goog-api-key": self._key, "Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=self.max_workers * 2,
                                max_keepalive_connections=self.max_workers),
        )

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _redact(self, text: str) -> str:
        """Belt-and-braces: never let the key reach a log or traceback."""
        return text.replace(self._key, "***REDACTED***") if self._key else text

    # -- transport ---------------------------------------------------------
    def _post(
        self,
        path: str,
        body: dict,
        timeout: float,
        namespace: str,
        max_attempts: int | None = None,
    ) -> dict:
        """POST with cache, pacing, and bounded retry.  Returns the raw response."""
        key = hash_payload({"path": path, "body": body})
        cached = self.cache.get(namespace, key)
        if cached is not None:
            return cached

        attempts = max_attempts if max_attempts is not None else config.MAX_ATTEMPTS
        url = f"{config.API_BASE}/{path}"
        # Uploading tens of MB of inline audio needs a generous write timeout.
        to = httpx.Timeout(timeout, connect=config.TIMEOUT_CONNECT,
                           write=max(60.0, timeout), pool=config.TIMEOUT_CONNECT)

        last_exc: BaseException | None = None
        for attempt in range(1, attempts + 1):
            self._limiter.acquire()
            started = time.monotonic()
            try:
                resp = self._client.post(url, json=body, timeout=to)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # This is the 300s-stall guard doing its job.
                last_exc = exc
                log.warning("%s attempt %d/%d: %s after %.1fs",
                            namespace, attempt, attempts, type(exc).__name__,
                            time.monotonic() - started)
                self._sleep_backoff(attempt, attempts)
                continue

            elapsed = time.monotonic() - started

            if resp.status_code == 200:
                try:
                    data = resp.json()
                except ValueError as exc:
                    # 200 but not JSON => truncated/corrupted response body.
                    last_exc = GeminiParseError(f"200 response was not JSON: {exc}")
                    log.warning("%s attempt %d/%d: non-JSON 200 body",
                                namespace, attempt, attempts)
                    self._sleep_backoff(attempt, attempts)
                    continue

                with self._counter_lock:
                    self.request_count += 1
                    self.total_tokens += int(
                        data.get("usageMetadata", {}).get("totalTokenCount", 0) or 0)
                log.debug("%s ok in %.1fs", namespace, elapsed)
                self.cache.put(namespace, key, data)
                return data

            detail = self._redact(resp.text[:600])

            if resp.status_code in FATAL_STATUS:
                msg = f"{namespace}: HTTP {resp.status_code} — {detail}"
                if resp.status_code in (401, 403):
                    raise GeminiAuthError(
                        msg + "\nCheck GEMINI_API_KEY in .env is valid and enabled.")
                raise GeminiBadRequest(msg)

            if resp.status_code in RETRYABLE_STATUS:
                last_exc = GeminiError(f"HTTP {resp.status_code}: {detail}")
                # Honour server-directed pacing on 429 when offered.
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.strip().isdigit():
                    delay = min(float(retry_after), config.BACKOFF_CAP)
                    log.warning("%s attempt %d/%d: HTTP %d, Retry-After %.0fs",
                                namespace, attempt, attempts, resp.status_code, delay)
                    if attempt < attempts:
                        time.sleep(delay)
                else:
                    log.warning("%s attempt %d/%d: HTTP %d",
                                namespace, attempt, attempts, resp.status_code)
                    self._sleep_backoff(attempt, attempts)
                continue

            raise GeminiError(f"{namespace}: unexpected HTTP {resp.status_code} — {detail}")

        raise GeminiError(
            f"{namespace}: giving up after {attempts} attempts. Last error: {last_exc}"
        ) from last_exc

    @staticmethod
    def _sleep_backoff(attempt: int, attempts: int) -> None:
        if attempt >= attempts:
            return
        # Exponential with full jitter — avoids retry storms across the pool.
        ceiling = min(config.BACKOFF_CAP, config.BACKOFF_BASE * (2 ** (attempt - 1)))
        time.sleep(random.uniform(0.0, ceiling))

    # -- response interpretation -------------------------------------------
    @staticmethod
    def _extract_text(data: dict) -> str:
        """Pull the answer text out of a generateContent response.

        Handles the specific shapes that bite people:
          - the prompt itself being blocked (no candidates at all);
          - thinking models emitting `thought` parts that must be skipped;
          - MAX_TOKENS truncation, which silently yields invalid JSON;
          - STOP with an empty parts list.
        """
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise GeminiBlocked(
                f"Prompt blocked: {feedback['blockReason']} {feedback.get('safetyRatings', '')}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise GeminiEmptyResponse(f"No candidates returned. Raw keys: {list(data)}")

        cand = candidates[0]
        finish = cand.get("finishReason", "")

        if finish == "MAX_TOKENS":
            raise GeminiTruncated(
                "Response hit maxOutputTokens and is incomplete. "
                "Raise max_output_tokens or split the input.")
        if finish in BLOCKING_FINISH:
            raise GeminiBlocked(
                f"Response suppressed (finishReason={finish}). "
                f"safetyRatings={cand.get('safetyRatings')}")

        parts = (cand.get("content") or {}).get("parts") or []
        # `thought` parts are the model's reasoning, not its answer.
        chunks = [p["text"] for p in parts
                  if isinstance(p, dict) and "text" in p and not p.get("thought")]
        text = "".join(chunks).strip()

        if not text:
            raise GeminiEmptyResponse(
                f"finishReason={finish or 'unset'} but no text parts "
                f"({len(parts)} part(s) present)")
        return text

    @staticmethod
    def _parse_json(text: str, namespace: str) -> Any:
        """Parse structured output, tolerating stray markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            config.DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            dump = config.DEBUG_DIR / f"parse_fail_{namespace}_{int(time.time())}.txt"
            dump.write_text(text, encoding="utf-8")
            raise GeminiParseError(
                f"{namespace}: model output was not valid JSON ({exc}). "
                f"Raw response saved to {dump}"
            ) from exc

    # -- public API --------------------------------------------------------
    def generate(
        self,
        parts: Sequence[dict],
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = config.TIMEOUT_TEXT,
        max_output_tokens: int | None = None,
        response_schema: dict | None = None,
        system_instruction: str | None = None,
        namespace: str = "generate",
        max_attempts: int | None = None,
    ) -> dict:
        """Low-level call returning the raw response dict."""
        model = model or config.MODEL_SMART
        gen_config: dict[str, Any] = {"temperature": temperature}
        if max_output_tokens:
            gen_config["maxOutputTokens"] = max_output_tokens
        if response_schema is not None:
            gen_config["responseMimeType"] = "application/json"
            gen_config["responseSchema"] = response_schema

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": list(parts)}],
            "generationConfig": gen_config,
            "safetySettings": config.safety_settings(),
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        return self._post(f"models/{model}:generateContent", body,
                          timeout=timeout, namespace=namespace,
                          max_attempts=max_attempts)

    def generate_text(self, parts: Sequence[dict], **kw: Any) -> str:
        return self._extract_text(self.generate(parts, **kw))

    def generate_json(self, parts: Sequence[dict], schema: dict, **kw: Any) -> Any:
        """Structured generation.  Raises rather than ever returning junk."""
        namespace = kw.get("namespace", "generate_json")
        kw["response_schema"] = schema
        kw.setdefault("namespace", namespace)
        text = self._extract_text(self.generate(parts, **kw))
        return self._parse_json(text, namespace)

    # -- embeddings --------------------------------------------------------
    def embed(
        self,
        texts: Sequence[str],
        task_type: str = config.EMBED_TASK_DOCUMENT,
        model: str | None = None,
        dim: int | None = None,
        parallel: bool = True,
    ) -> list[list[float]]:
        """Embed texts, auto-batched and L2-normalised, order preserved.

        Normalisation is required, not cosmetic: gemini-embedding-001 returns
        unnormalised vectors at any outputDimensionality below its native
        3072, so raw dot products would be wrong.  Normalising here means
        downstream cosine similarity is a plain dot product.
        """
        import numpy as np

        model = model or config.MODEL_EMBED
        dim = dim or config.EMBED_DIM
        texts = list(texts)
        if not texts:
            return []

        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                # An empty segment is a pipeline bug; fail loudly at the source.
                raise ValueError(f"embed(): text at index {i} is empty or not a string")

        batches = [texts[i:i + config.EMBED_BATCH]
                   for i in range(0, len(texts), config.EMBED_BATCH)]

        def run_batch(batch: list[str]) -> list[list[float]]:
            body = {"requests": [
                {"model": f"models/{model}",
                 "content": {"parts": [{"text": t}]},
                 "taskType": task_type,
                 "outputDimensionality": dim}
                for t in batch]}
            data = self._post(f"models/{model}:batchEmbedContents", body,
                              timeout=config.TIMEOUT_EMBED, namespace="embed")
            embeddings = data.get("embeddings")
            if not embeddings or len(embeddings) != len(batch):
                raise GeminiError(
                    f"embed: expected {len(batch)} vectors, got "
                    f"{len(embeddings) if embeddings else 0}")
            out = []
            for e in embeddings:
                v = e.get("values")
                if not v or len(v) != dim:
                    raise GeminiError(
                        f"embed: expected dim {dim}, got {len(v) if v else 0}")
                out.append(v)
            return out

        if parallel and len(batches) > 1:
            results = thread_map(run_batch, batches,
                                 workers=self.max_workers, desc="embed-batch")
        else:
            results = [run_batch(b) for b in batches]

        vectors = [v for batch in results for v in batch]
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise GeminiError("embed: got a zero or non-finite embedding vector")
        return (arr / norms).tolist()

    # -- introspection -----------------------------------------------------
    def model_info(self, model: str | None = None) -> dict:
        model = model or config.MODEL_SMART
        resp = self._client.get(f"{config.API_BASE}/models/{model}",
                                timeout=httpx.Timeout(30.0, connect=config.TIMEOUT_CONNECT))
        if resp.status_code != 200:
            raise GeminiError(
                f"model_info({model}): HTTP {resp.status_code} — "
                f"{self._redact(resp.text[:300])}")
        return resp.json()

    def usage_summary(self) -> str:
        return (f"network calls={self.request_count} tokens={self.total_tokens:,} "
                f"| {self.cache.stats()}")
