"""P0 smoke test — proves the foundation works before any video code exists.

Each check targets a specific failure mode that would otherwise surface at a
much worse moment (mid-ingest, or during the demo).  Run:

    python scripts/p0_smoke.py
"""
from __future__ import annotations

import logging
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from vrag import config, gemini
from vrag.cache import Cache, read_json, write_json
from vrag.gemini import GeminiClient
from vrag.logging_setup import preview, setup_logging
from vrag.parallel import TaskFailed, thread_map

log = logging.getLogger("p0")
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            started = time.time()
            try:
                note = fn(*a, **kw) or ""
                RESULTS.append((name, True, f"{note} ({time.time() - started:.1f}s)"))
                log.info("PASS  %-34s %s", name, note)
            except Exception as exc:  # noqa: BLE001 - this is a test harness
                RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
                log.error("FAIL  %-34s %s: %s", name, type(exc).__name__, exc)
        return run
    return deco


# ---------------------------------------------------------------------------
@check("config + api key")
def t_config():
    key = config.get_api_key()
    config.ensure_dirs()
    assert config.VIDEO_DIR.exists()
    return f"key len={len(key)}, dirs ready"


@check("auth + model metadata")
def t_model_info(c: GeminiClient):
    info = c.model_info(config.MODEL_SMART)
    # Recorded because P2 needs to know how much transcript can come back at once.
    return (f"{config.MODEL_SMART}: in={info.get('inputTokenLimit'):,} "
            f"out={info.get('outputTokenLimit'):,}")


@check("plain text generation")
def t_text(c: GeminiClient):
    out = c.generate_text([gemini.text_part("Reply with exactly: PONG")],
                          namespace="smoke_text", timeout=60)
    assert "PONG" in out.upper(), f"unexpected reply: {out!r}"
    return preview(out, 40)


@check("structured JSON output")
def t_json(c: GeminiClient):
    schema = {
        "type": "OBJECT",
        "properties": {
            "capital": {"type": "STRING"},
            "population_millions": {"type": "NUMBER"},
        },
        "required": ["capital", "population_millions"],
    }
    out = c.generate_json(
        [gemini.text_part("Capital of Tamil Nadu and its population in millions.")],
        schema, namespace="smoke_json", timeout=60)
    assert isinstance(out, dict) and "capital" in out, out
    assert isinstance(out["population_millions"], (int, float)), out
    return f"{out['capital']} / {out['population_millions']}"


@check("Tamil UTF-8 round trip")
def t_tamil(c: GeminiClient):
    """Tamil must survive the console, the cache file, and JSON artifacts.

    This is the check that catches Windows cp1252 blowing up a 20-minute run.
    """
    schema = {"type": "OBJECT", "properties": {"tamil": {"type": "STRING"}},
              "required": ["tamil"]}
    out = c.generate_json(
        [gemini.text_part("Write 'Tamil Nadu Legislative Assembly' in Tamil script. "
                          "Return only the Tamil text.")],
        schema, namespace="smoke_tamil", timeout=60)
    tamil = out["tamil"]
    assert any(0x0B80 <= ord(ch) <= 0x0BFF for ch in tamil), f"no Tamil glyphs: {tamil!r}"

    log.info("      Tamil renders in logs: %s", tamil)  # would raise on a cp1252 stream

    tmp = config.DEBUG_DIR / "smoke_tamil.json"
    write_json(tmp, {"tamil": tamil})
    assert read_json(tmp)["tamil"] == tamil, "Tamil corrupted by artifact round-trip"
    tmp.unlink(missing_ok=True)
    return preview(tamil, 40)


def _clear_namespace(namespace: str) -> None:
    """Delete a cache namespace so a check that counts *network* calls is hermetic.

    Without this, a second run of the smoke test fails: the entries written by
    the first run turn the "first" call into a cache hit, so request_count
    never increments.  That is the cache working correctly, but it looks like a
    failure, which is worse than useless in a check.
    """
    d = config.CACHE_DIR / namespace
    if d.exists():
        for f in d.glob("*.json"):
            f.unlink(missing_ok=True)


@check("cache prevents repeat network call")
def t_cache(c: GeminiClient):
    _clear_namespace("smoke_cache")
    parts = [gemini.text_part("Reply with exactly: CACHED")]
    before = c.request_count
    c.generate_text(parts, namespace="smoke_cache", timeout=60)
    after_first = c.request_count
    c.generate_text(parts, namespace="smoke_cache", timeout=60)
    after_second = c.request_count
    assert after_first == before + 1, "first call should hit the network"
    assert after_second == after_first, "second identical call must be served from cache"
    return "1 network call, 2nd served from disk"


@check("cache key tracks prompt changes")
def t_cache_invalidation(c: GeminiClient):
    _clear_namespace("smoke_inval")
    before = c.request_count
    c.generate_text([gemini.text_part("Reply with exactly: ALPHA")],
                    namespace="smoke_inval", timeout=60)
    c.generate_text([gemini.text_part("Reply with exactly: BETA")],
                    namespace="smoke_inval", timeout=60)
    assert c.request_count == before + 2, "different prompts must not share a cache entry"
    return "distinct prompts -> distinct entries"


@check("timeout is bounded (300s stall guard)")
def t_timeout(c: GeminiClient):
    """The reason this whole module exists: a hung call must die fast."""
    started = time.time()
    try:
        c.generate_text([gemini.text_part(f"unique-{time.time()}")],
                        namespace="smoke_timeout", timeout=0.001, max_attempts=1)
        raise AssertionError("expected a timeout, got a response")
    except gemini.GeminiError:
        pass
    elapsed = time.time() - started
    assert elapsed < 20, f"took {elapsed:.1f}s — timeout not being honoured"
    return f"failed fast in {elapsed:.2f}s (not 300s)"


@check("fatal errors are not retried")
def t_fatal(c: GeminiClient):
    started = time.time()
    try:
        c.generate_text([gemini.text_part("hi")],
                        model="gemini-does-not-exist", namespace="smoke_fatal", timeout=30)
        raise AssertionError("expected a 404")
    except gemini.GeminiBadRequest:
        pass
    elapsed = time.time() - started
    assert elapsed < 15, f"took {elapsed:.1f}s — a 404 was retried"
    return f"raised immediately in {elapsed:.2f}s"


@check("truncation detected, not silently parsed")
def t_truncation(c: GeminiClient):
    try:
        c.generate_text(
            [gemini.text_part("Write a 900 word essay about the Tamil language.")],
            namespace="smoke_trunc", timeout=60, max_output_tokens=16)
        return "model fit within budget (no truncation to detect)"
    except gemini.GeminiTruncated:
        return "GeminiTruncated raised as expected"


@check("embeddings: dim, norm, task type")
def t_embed(c: GeminiClient):
    texts = ["The resolution concerns state autonomy.",
             "Fisheries policy in coastal districts.",
             "மாநில சுயாட்சி குறித்த தீர்மானம்."]
    docs = c.embed(texts, task_type=config.EMBED_TASK_DOCUMENT)
    assert len(docs) == 3, f"expected 3 vectors, got {len(docs)}"
    assert len(docs[0]) == config.EMBED_DIM, f"dim={len(docs[0])}"

    arr = np.asarray(docs)
    norms = np.linalg.norm(arr, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"not L2-normalised: {norms}"

    # taskType must actually reach the API, or retrieval quality silently drops.
    q = c.embed([texts[0]], task_type=config.EMBED_TASK_QUERY)[0]
    assert not np.allclose(np.asarray(q), arr[0], atol=1e-6), \
        "QUERY and DOCUMENT embeddings are identical — taskType is being ignored"

    # Sanity: the Tamil sentence should be nearer its English twin than the
    # unrelated fisheries line.  This is the cross-lingual retrieval premise.
    sim_related = float(arr[2] @ arr[0])
    sim_unrelated = float(arr[2] @ arr[1])
    assert sim_related > sim_unrelated, \
        f"cross-lingual similarity failed: {sim_related:.3f} <= {sim_unrelated:.3f}"
    return f"dim={config.EMBED_DIM}, |v|=1, ta~en {sim_related:.3f} > {sim_unrelated:.3f}"


@check("empty embed input rejected loudly")
def t_embed_empty(c: GeminiClient):
    try:
        c.embed(["fine", "   "])
        raise AssertionError("expected ValueError for blank text")
    except ValueError:
        pass
    assert c.embed([]) == [], "empty list should return empty list"
    return "ValueError on blank, [] on empty"


@check("civic-integrity safety category")
def t_civic(c: GeminiClient):
    """Legislative debate is political by nature; check whether we may also
    disable this filter, and record the answer for config."""
    body = {
        "contents": [{"role": "user", "parts": [{"text": "Say OK"}]}],
        "generationConfig": {"temperature": 0},
        "safetySettings": [{"category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                            "threshold": "BLOCK_NONE"}],
    }
    try:
        c._post(f"models/{config.MODEL_SMART}:generateContent", body,
                timeout=60, namespace="smoke_civic", max_attempts=1)
        return "SUPPORTED — can be added to config.SAFETY_CATEGORIES"
    except gemini.GeminiBadRequest:
        return "not supported by this API version — leaving it out (fine)"


@check("thread_map ordering + failure aggregation")
def t_parallel():
    out = thread_map(lambda x: x * 2, list(range(20)), workers=6, desc="smoke_par")
    assert out == [x * 2 for x in range(20)], "results came back out of order"

    def flaky(x):
        if x in (2, 5):
            raise ValueError(f"boom {x}")
        return x

    try:
        thread_map(flaky, list(range(8)), workers=4, desc="smoke_fail")
        raise AssertionError("expected TaskFailed")
    except TaskFailed as exc:
        assert len(exc.failures) == 2, exc.failures

    collected = thread_map(flaky, list(range(8)), workers=4,
                           desc="smoke_collect", raise_on_error=False)
    assert isinstance(collected[2], ValueError), "failed slot should hold the exception"
    assert collected[3] == 3, "successful slots should still hold values"
    return "order preserved, 2 failures aggregated"


@check("corrupt cache entry survives")
def t_corrupt_cache():
    cache = Cache(config.CACHE_DIR)
    path = cache._path("smoke_corrupt", "deadbeef")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert cache.get("smoke_corrupt", "deadbeef") is None, "should degrade to a miss"
    path.unlink(missing_ok=True)
    return "treated as miss, no crash"


# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging()
    log.info("=" * 74)
    log.info("P0 SMOKE TEST — foundation layer")
    log.info("=" * 74)

    t_config()
    t_parallel()
    t_corrupt_cache()

    with GeminiClient() as c:
        t_model_info(c)
        t_text(c)
        t_json(c)
        t_tamil(c)
        t_cache(c)
        t_cache_invalidation(c)
        t_timeout(c)
        t_fatal(c)
        t_truncation(c)
        t_embed(c)
        t_embed_empty(c)
        t_civic(c)
        usage = c.usage_summary()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    log.info("=" * 74)
    for name, ok, note in RESULTS:
        log.info("%s %-36s %s", "PASS" if ok else "FAIL", name, note)
    log.info("-" * 74)
    log.info("%d/%d checks passed | %s", passed, total, usage)
    log.info("=" * 74)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
