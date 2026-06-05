"""Tests for infra.circuit_breaker."""

from __future__ import annotations

import time

import pytest

from polymarket_agent.infra.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    State,
)


async def _ok() -> int:
    return 42


async def _fail() -> int:
    raise RuntimeError("boom")


async def test_closed_passes_through():
    cb = CircuitBreaker(failure_threshold=2)
    assert await cb.call(_ok()) == 42
    assert cb.state == State.CLOSED


async def test_trips_after_threshold():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await cb.call(_fail())
    assert cb.state == State.OPEN
    with pytest.raises(CircuitBreakerOpen):
        await cb.call(_ok())


async def test_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05, success_threshold=2)
    with pytest.raises(RuntimeError):
        await cb.call(_fail())
    assert cb.state == State.OPEN
    time.sleep(0.06)
    assert cb.state == State.HALF_OPEN
    # One success in HALF_OPEN is not enough
    await cb.call(_ok())
    assert cb.state == State.HALF_OPEN
    # Second success closes
    await cb.call(_ok())
    assert cb.state == State.CLOSED


async def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
    with pytest.raises(RuntimeError):
        await cb.call(_fail())
    time.sleep(0.06)
    assert cb.state == State.HALF_OPEN
    with pytest.raises(RuntimeError):
        await cb.call(_fail())
    assert cb.state == State.OPEN


async def test_reset():
    cb = CircuitBreaker(failure_threshold=1)
    with pytest.raises(RuntimeError):
        await cb.call(_fail())
    assert cb.state == State.OPEN
    cb.reset()
    assert cb.state == State.CLOSED
    assert await cb.call(_ok()) == 42


async def test_success_reduces_failure_count():
    cb = CircuitBreaker(failure_threshold=3)
    with pytest.raises(RuntimeError):
        await cb.call(_fail())
    assert cb._failure_count == 1
    await cb.call(_ok())
    assert cb._failure_count == 0
