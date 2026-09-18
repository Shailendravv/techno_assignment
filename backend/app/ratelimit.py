"""A per-client request limit for `POST /ask`.

Every request that clears the gate spends Groq tokens against a free tier
allowing roughly 8,000 a minute, shared by the API, the CLI and the evaluation
harness. The endpoint takes no credential, so without a limit a single script
can exhaust a day's quota in a minute and every other caller gets a 429 from
Groq instead of an answer.

**A fixed-window counter held in memory, and the trade is worth stating.** A
serverless deployment runs several instances, so the effective limit is the
configured rate times the number of warm instances, and a cold start forgets
everything. Redis would fix both and is a service to run, pay for and monitor
for a demo that does not otherwise need one - the same reasoning that put the
answer cache and the job queue in Postgres rather than in Redis. This is the
cheap ninety per cent: it stops the accidental loop and the casual abuser, and
it does not pretend to stop a distributed one.

The window is fixed rather than sliding, which permits a burst of up to twice
the limit across a boundary. At twenty a minute that is not worth a more
complex structure.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Fixed-window request counting, keyed on client identity."""

    def __init__(self, per_minute: int, window_s: float = 60.0):
        self.per_minute = per_minute
        self.window_s = window_s
        # FastAPI runs sync endpoints in a threadpool, so this is genuinely
        # concurrent. The lock is uncontended in the normal case and the
        # critical section is a dict update.
        self._lock = threading.Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """(allowed, seconds until the window resets).

        Counts the request it is checking, so a caller does not have to
        remember to record it separately.
        """
        if not self.enabled:
            return True, 0

        now = time.monotonic() if now is None else now
        with self._lock:
            started, count = self._windows.get(key, (now, 0))

            if now - started >= self.window_s:
                started, count = now, 0

            retry_after = max(1, int(self.window_s - (now - started)))

            if count >= self.per_minute:
                self._windows[key] = (started, count)
                return False, retry_after

            self._windows[key] = (started, count + 1)

            # Unbounded growth would be a slow memory leak on a long-lived
            # process, since every distinct client gets an entry. Expired
            # windows are dropped opportunistically rather than on a timer.
            if len(self._windows) > 2048:
                self._windows = {
                    k: v
                    for k, v in self._windows.items()
                    if now - v[0] < self.window_s
                }

            return True, retry_after

    def reset(self) -> None:
        """For tests, and for a process that has changed configuration."""
        with self._lock:
            self._windows.clear()
