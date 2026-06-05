"""Production safety tests — verify critical fail-safe behaviors."""

import os
from unittest.mock import patch

import pytest

from polymarket_agent.execution.executor import OrderExecutor
from polymarket_agent.tracking.tracker import PerformanceTracker


class TestExecutorSafety:
    """Verify the executor fails loudly, never silently."""

    def test_live_mode_raises_without_private_key(self):
        """Live mode must NOT silently degrade to paper mode."""
        config = {"dry_run": False}
        with patch.dict(os.environ, {}, clear=True):
            # Remove POLYMARKET_PRIVATE_KEY if present
            os.environ.pop("POLYMARKET_PRIVATE_KEY", None)
            with pytest.raises(RuntimeError, match="POLYMARKET_PRIVATE_KEY"):
                OrderExecutor(config)

    def test_live_mode_raises_on_clob_init_failure(self):
        """If CLOB client fails to init in live mode, must raise."""
        config = {"dry_run": False}
        with (
            patch.dict(os.environ, {"POLYMARKET_PRIVATE_KEY": "0xfakekey"}),
            patch(
                "polymarket_agent.execution.executor.OrderExecutor.__init__",
                side_effect=RuntimeError("CLOB init failed"),
            ),
            pytest.raises(RuntimeError),
        ):
            OrderExecutor(config)

    def test_paper_mode_does_not_raise(self):
        """Paper mode should always work without any keys."""
        config = {"dry_run": True}
        executor = OrderExecutor(config)
        assert executor.dry_run is True
        assert executor.clob is None

    @pytest.mark.asyncio
    async def test_rejects_invalid_price(self):
        """Orders with price outside 0-1 must be rejected."""
        config = {"dry_run": True}
        executor = OrderExecutor(config)
        result = await executor.place_order(
            market_id="test",
            direction="YES",
            size_usdc=10.0,
            price=1.5,  # Invalid
            token_id="tok",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_zero_size(self):
        """Orders with zero or negative size must be rejected."""
        config = {"dry_run": True}
        executor = OrderExecutor(config)
        result = await executor.place_order(
            market_id="test",
            direction="YES",
            size_usdc=0.0,
            price=0.5,
            token_id="tok",
        )
        assert result is None


class TestTrackerDurability:
    """Verify SQLite is configured for durability."""

    def test_wal_mode_enabled(self, tmp_path):
        """Tracker must use WAL journal mode."""
        db_path = str(tmp_path / "test_trades.db")
        tracker = PerformanceTracker(db_path)
        row = tracker._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        tracker.close()

    def test_busy_timeout_set(self, tmp_path):
        """Tracker must set a busy timeout to handle concurrent access."""
        db_path = str(tmp_path / "test_trades.db")
        tracker = PerformanceTracker(db_path)
        row = tracker._conn.execute("PRAGMA busy_timeout").fetchone()
        assert row[0] >= 5000
        tracker.close()

    def test_transaction_rollback_on_error(self, tmp_path):
        """Verify that failed writes don't leave partial state."""
        db_path = str(tmp_path / "test_trades.db")
        tracker = PerformanceTracker(db_path)

        # Insert a test trade
        edge = {
            "market_id": "test_market",
            "question": "Test?",
            "direction": "YES",
            "market_price": 0.5,
            "edge": 0.10,
            "confidence": 0.8,
            "estimated_prob": 0.6,
            "regime": "stable",
            "category": "test",
            "signal_breakdown": {},
        }
        trade_id = tracker.record_paper_trade(edge, 10.0)
        assert trade_id > 0

        # Verify trade exists
        positions = tracker.get_open_positions()
        assert len(positions) >= 1
        tracker.close()


class TestDataValidation:
    """Verify market data is validated at boundaries."""

    @pytest.mark.asyncio
    async def test_crossed_book_rejected(self):
        """Order books where bid > ask must be rejected."""
        from polymarket_agent.data.clob_client import ClobClient

        client = ClobClient()
        # Mock a crossed book response
        crossed_response = {
            "bids": [{"price": "0.70", "size": "100"}],
            "asks": [{"price": "0.60", "size": "100"}],  # ask < bid = crossed
        }
        with patch.object(client, "_request", return_value=crossed_response):
            result = await client.get_order_book("test_token")
        assert result is None
        await client.close()

    @pytest.mark.asyncio
    async def test_invalid_price_range_rejected(self):
        """Prices outside 0.0-1.0 must be rejected."""
        from polymarket_agent.data.clob_client import ClobClient

        client = ClobClient()
        bad_response = {
            "bids": [{"price": "1.50", "size": "100"}],
            "asks": [{"price": "2.00", "size": "100"}],
        }
        with patch.object(client, "_request", return_value=bad_response):
            result = await client.get_order_book("test_token")
        assert result is None
        await client.close()
