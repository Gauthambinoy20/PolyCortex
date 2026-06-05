"""Async token-bucket rate limiter for API clients."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Async token-bucket rate limiter.

    Args:
        requests_per_minute: Maximum requests per minute.
        burst: Maximum burst size (defaults to requests_per_minute).
    """

    def __init__(self, requests_per_minute: float, burst: float | None = None) -> None:
        self.rate = requests_per_minute / 60.0
        self.burst = burst if burst is not None else requests_per_minute
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire one token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate
                logger.debug("Rate limit: waiting %.2fs", wait_time)
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1

    async def __aenter__(self) -> TokenBucketRateLimiter:
        await self.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


_gamma_limiter: TokenBucketRateLimiter | None = None
_clob_limiter: TokenBucketRateLimiter | None = None


def get_gamma_limiter(requests_per_minute: float = 60) -> TokenBucketRateLimiter:
    """Get or create the Gamma API rate limiter."""
    global _gamma_limiter
    if _gamma_limiter is None:
        _gamma_limiter = TokenBucketRateLimiter(requests_per_minute=requests_per_minute)
    return _gamma_limiter


def get_clob_limiter(requests_per_minute: float = 120) -> TokenBucketRateLimiter:
    """Get or create the CLOB API rate limiter."""
    global _clob_limiter
    if _clob_limiter is None:
        _clob_limiter = TokenBucketRateLimiter(requests_per_minute=requests_per_minute)
    return _clob_limiter
