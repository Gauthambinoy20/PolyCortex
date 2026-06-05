from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from polymarket_agent.execution.executor import Order, OrderExecutor


class TestOrderExecution:
    async def test_dry_run_no_api_call(self, sample_config):
        config = {**sample_config, "dry_run": True, "paper_fill_on_place": False}
        executor = OrderExecutor(config)
        executor.clob = MagicMock()

        order = await executor.place_order(
            market_id="0xabc123",
            direction="YES",
            size_usdc=30.0,
            price=0.65,
            token_id="token_yes_123",
        )

        assert order is not None
        assert order.is_paper is True
        assert order.status == "placed"
        executor.clob.create_and_post_order.assert_not_called()

    async def test_order_splitting(self, sample_config):
        config = {**sample_config, "dry_run": True, "split_orders_above": 50}
        executor = OrderExecutor(config)

        order = await executor.place_order(
            market_id="0xabc123",
            direction="YES",
            size_usdc=120.0,
            price=0.65,
            token_id="token_yes_123",
        )

        assert order is not None
        assert order.chunks > 1
        assert order.size_usdc == 120.0

    async def test_cancel_order(self, sample_config):
        config = {**sample_config, "dry_run": True, "paper_fill_on_place": False}
        executor = OrderExecutor(config)

        order = await executor.place_order(
            market_id="0xabc123",
            direction="YES",
            size_usdc=30.0,
            price=0.65,
        )
        assert order is not None

        result = await executor.cancel_order(order.order_id)
        assert result is True
        assert executor.orders[order.order_id].status == "cancelled"

    async def test_cancel_expired_orders(self, sample_config):
        config = {
            **sample_config,
            "dry_run": True,
            "order_expiry_minutes": 60,
            "paper_fill_on_place": False,
        }
        executor = OrderExecutor(config)

        # Place a fresh order
        fresh = await executor.place_order(
            market_id="m1",
            direction="YES",
            size_usdc=20.0,
            price=0.60,
        )
        # Place an order and backdate it to make it expired
        old = await executor.place_order(
            market_id="m2",
            direction="NO",
            size_usdc=20.0,
            price=0.40,
        )
        assert fresh is not None and old is not None
        old.timestamp = datetime.now(UTC) - timedelta(minutes=120)

        cancelled = await executor.cancel_expired_orders()

        assert old.order_id in cancelled
        assert fresh.order_id not in cancelled
        assert executor.orders[old.order_id].status == "cancelled"
        assert executor.orders[fresh.order_id].status == "placed"

    def test_order_dataclass(self):
        now = datetime.now(UTC)
        order = Order(
            order_id="abc123",
            market_id="0xmarket",
            token_id="token1",
            direction="YES",
            side="BUY",
            size_usdc=50.0,
            price=0.65,
            status="placed",
            timestamp=now,
            chunks=1,
            is_paper=True,
        )
        assert order.order_id == "abc123"
        assert order.direction == "YES"
        assert order.size_usdc == 50.0
        assert isinstance(order.timestamp, datetime)
        assert order.is_paper is True

    async def test_paper_order_arms_brackets(self, sample_config):
        config = {**sample_config, "dry_run": True, "paper_fill_on_place": True}
        executor = OrderExecutor(config)

        order = await executor.place_order(
            market_id="m_bracket",
            direction="YES",
            size_usdc=40.0,
            price=0.65,
            token_id="token_yes_123",
        )

        assert order is not None
        assert order.status == "filled"
        assert order.bracket_state == "armed"
        assert len(order.bracket_order_ids) == 2
        child_orders = [executor.orders[child_id] for child_id in order.bracket_order_ids]
        assert {child.order_kind for child in child_orders} == {"take_profit", "stop_loss"}
        assert all(child.status == "armed" for child in child_orders)

    async def test_reconcile_triggers_take_profit_and_cancels_sibling(self, sample_config):
        config = {**sample_config, "dry_run": True, "paper_fill_on_place": True}
        executor = OrderExecutor(config)

        order = await executor.place_order(
            market_id="m_tp",
            direction="YES",
            size_usdc=40.0,
            price=0.65,
            token_id="token_yes_123",
        )
        assert order is not None

        events = await executor.reconcile_orders({"m_tp": {"midpoint": 0.71}})

        exit_events = [event for event in events if event.event_type == "exit_filled"]
        cancel_events = [event for event in events if event.event_type == "oco_cancelled"]
        assert len(exit_events) == 1
        assert exit_events[0].order_snapshot is not None
        assert exit_events[0].order_snapshot.order_kind == "take_profit"
        assert len(cancel_events) == 1

    async def test_partial_fill_reconciliation_for_paper_entry(self, sample_config):
        config = {
            **sample_config,
            "dry_run": True,
            "paper_fill_on_place": False,
            "paper_allow_partial_fills": True,
            "paper_partial_fill_ratio": 0.5,
            "enable_bracket_orders": False,
        }
        executor = OrderExecutor(config)

        order = await executor.place_order(
            market_id="m_partial",
            direction="YES",
            size_usdc=100.0,
            price=0.65,
            token_id="token_yes_123",
        )
        assert order is not None
        assert order.status == "placed"

        events_first = await executor.reconcile_orders({"m_partial": {"midpoint": 0.648}})
        assert any(event.event_type == "entry_partially_filled" for event in events_first)
        assert order.status == "partially_filled"
        assert order.filled_size_usdc == pytest.approx(50.0)
        assert order.remaining_size_usdc == pytest.approx(50.0)

        events_second = await executor.reconcile_orders({"m_partial": {"midpoint": 0.647}})
        assert any(event.event_type == "entry_filled" for event in events_second)
        assert order.status == "filled"
        assert order.filled_size_usdc == pytest.approx(100.0)
        assert order.remaining_size_usdc == pytest.approx(0.0)
