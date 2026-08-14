"""Bounded parallel map with ordered results and progress reporting."""
from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)


class TaskFailed(RuntimeError):
    def __init__(self, failures: list[tuple[int, BaseException]], total: int):
        self.failures = failures
        self.total = total
        detail = "; ".join(f"[{i}] {type(e).__name__}: {e}" for i, e in failures[:3])
        more = f" (+{len(failures) - 3} more)" if len(failures) > 3 else ""
        super().__init__(f"{len(failures)}/{total} tasks failed: {detail}{more}")


def thread_map(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    workers: int = 6,
    desc: str = "tasks",
    raise_on_error: bool = True,
) -> list[Any]:
    """Run `fn` over `items` across a thread pool, preserving input order.

    I/O-bound work only (every task here is an HTTPS call), so threads are the
    right tool and the GIL is irrelevant.

    Failure policy follows the project rule of degrading loudly: with
    raise_on_error=True (the default) *all* items are still attempted, then a
    single TaskFailed summarises every failure.  With it False, failed slots
    hold the exception object so the caller can decide — never a silent None.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    results: list[Any] = [None] * n
    failures: list[tuple[int, BaseException]] = []
    started = time.time()
    done = 0
    # Log roughly ten progress lines regardless of how big the batch is.
    step = max(1, n // 10)

    with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, n))) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        try:
            for fut in cf.as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
                    failures.append((i, exc))
                    results[i] = exc
                    log.error("%s[%d] failed: %s: %s", desc, i, type(exc).__name__, exc)
                done += 1
                if done % step == 0 or done == n:
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (n - done) / rate if rate > 0 else 0.0
                    log.info("%s %d/%d (%.0f%%) %.1fs elapsed, ~%.0fs left",
                             desc, done, n, 100.0 * done / n, elapsed, eta)
        except KeyboardInterrupt:
            log.warning("Interrupted — cancelling %s", desc)
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    log.info("%s complete: %d ok, %d failed in %.1fs",
             desc, n - len(failures), len(failures), time.time() - started)

    if failures and raise_on_error:
        raise TaskFailed(failures, n)
    return results
