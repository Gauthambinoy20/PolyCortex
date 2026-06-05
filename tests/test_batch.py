"""Tests for batch order manager."""

import pytest

from polymarket_agent.execution.batch import MAX_BATCH_SIZE, BatchOrderManager, PendingOrder


@pytest.fixture
def batcher():
    return BatchOrderManager(dry_run=True)


def make_order(i=0):
    return PendingOrder(token_id=f"tok_{i}", price=0.5, size=10, side="BUY")


class TestBatchOrderManager:
    def test_initial_state(self, batcher):
        assert batcher.pending_count == 0
        assert batcher.stats["batches_sent"] == 0

    def test_add_orders(self, batcher):
        batcher.add(make_order(0))
        batcher.add(make_order(1))
        assert batcher.pending_count == 2

    async def test_flush_single_batch(self, batcher):
        for i in range(5):
            batcher.add(make_order(i))
        results = await batcher.flush()
        assert len(results) == 1
        assert results[0].submitted_count == 5
        assert results[0].success_count == 5
        assert batcher.pending_count == 0

    async def test_flush_multiple_batches(self, batcher):
        for i in range(20):
            batcher.add(make_order(i))
        results = await batcher.flush()
        assert len(results) == 2
        assert results[0].submitted_count == 15
        assert results[1].submitted_count == 5

    async def test_flush_empty(self, batcher):
        results = await batcher.flush()
        assert results == []

    async def test_flush_if_full_below_threshold(self, batcher):
        batcher.add(make_order(0))
        results = await batcher.flush_if_full()
        assert results == []
        assert batcher.pending_count == 1

    async def test_flush_if_full_at_threshold(self, batcher):
        for i in range(MAX_BATCH_SIZE):
            batcher.add(make_order(i))
        results = await batcher.flush_if_full()
        assert len(results) >= 1
        assert batcher.pending_count == 0

    def test_clear(self, batcher):
        for i in range(5):
            batcher.add(make_order(i))
        cleared = batcher.clear()
        assert cleared == 5
        assert batcher.pending_count == 0

    async def test_stats_accumulate(self, batcher):
        for i in range(20):
            batcher.add(make_order(i))
        await batcher.flush()
        stats = batcher.stats
        assert stats["total_submitted"] == 20
        assert stats["total_succeeded"] == 20
        assert stats["total_failed"] == 0
        assert stats["batches_sent"] == 2

    def test_max_batch_size_cap(self):
        b = BatchOrderManager(max_batch_size=50, dry_run=True)
        assert b.max_batch_size == MAX_BATCH_SIZE
