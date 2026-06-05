import asyncio
import time

import pytest

from polymarket_agent.infra.rate_limiter import TokenBucketRateLimiter


@pytest.mark.asyncio
async def test_burst_allows_initial_requests_immediately():
    limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=5)
    t0 = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_sixth_request_waits_for_refill():
    limiter = TokenBucketRateLimiter(requests_per_minute=600, burst=2)  # 10 rps
    await limiter.acquire()
    await limiter.acquire()
    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.05


@pytest.mark.asyncio
async def test_context_manager_api():
    limiter = TokenBucketRateLimiter(requests_per_minute=120, burst=3)
    async with limiter:
        pass


@pytest.mark.asyncio
async def test_concurrent_acquire_serializes():
    limiter = TokenBucketRateLimiter(requests_per_minute=600, burst=1)

    async def _task():
        await limiter.acquire()

    await asyncio.gather(_task(), _task(), _task())
