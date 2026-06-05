"""Tests for infra.backpressure."""

from __future__ import annotations

import asyncio

import pytest

from polymarket_agent.infra.backpressure import BoundedSignalQueue


async def test_put_and_get():
    q = BoundedSignalQueue(maxsize=3)
    assert await q.put("a") is True
    assert await q.put("b") is True
    assert q.size == 2
    assert await q.get() == "a"
    q.task_done()
    assert q.accepted_count == 2


async def test_drops_when_full():
    q = BoundedSignalQueue(maxsize=2)
    await q.put("a")
    await q.put("b")
    assert await q.put("c") is False
    assert q.dropped_count == 1
    assert q.size == 2


async def test_block_waits_when_full():
    q = BoundedSignalQueue(maxsize=1)
    await q.put("a")

    async def slow_consumer():
        await asyncio.sleep(0.05)
        await q.get()
        q.task_done()

    task = asyncio.create_task(slow_consumer())
    assert await q.put("b", block=True) is True
    await task
    assert q.dropped_count == 0


async def test_maxsize_exposed():
    q = BoundedSignalQueue(maxsize=7)
    assert q.maxsize == 7
    assert q.empty()


async def test_get_nowait_raises_when_empty():
    q = BoundedSignalQueue(maxsize=1)
    with pytest.raises(asyncio.QueueEmpty):
        q.get_nowait()
