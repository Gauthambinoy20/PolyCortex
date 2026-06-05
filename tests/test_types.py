"""Tests for shared type definitions."""

from datetime import UTC, datetime

import pytest

from polymarket_agent.types import (
    VALID_TRANSITIONS,
    Event,
    EventType,
    MarketData,
    OrderBookLevel,
    OrderBookSnapshot,
    OrderState,
    RiskCheckResult,
    TradeSignal,
)


class TestOrderState:
    def test_all_states_have_transitions(self):
        for state in OrderState:
            assert state in VALID_TRANSITIONS, f"{state} missing from VALID_TRANSITIONS"

    def test_terminal_states_have_no_transitions(self):
        terminal = {OrderState.REDEEMED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED}
        for state in terminal:
            assert VALID_TRANSITIONS[state] == set(), f"{state} should have no transitions"

    def test_created_can_transition(self):
        assert OrderState.VALIDATED in VALID_TRANSITIONS[OrderState.CREATED]

    def test_state_string_values(self):
        assert OrderState.CREATED == "created"
        assert OrderState.FILLED == "filled"


class TestEventType:
    def test_event_types_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)


class TestEvent:
    def test_event_creation(self):
        e = Event(event_type=EventType.ORDER_PLACED, payload={"order_id": "abc"})
        assert e.event_type == EventType.ORDER_PLACED
        assert e.payload["order_id"] == "abc"
        assert isinstance(e.timestamp, datetime)

    def test_event_with_source(self):
        e = Event(event_type=EventType.TRADE_CLOSED, payload={}, source="executor")
        assert e.source == "executor"

    def test_event_default_source(self):
        e = Event(event_type=EventType.HEARTBEAT, payload={})
        assert e.source == "agent"

    def test_event_has_unique_id(self):
        e1 = Event(event_type=EventType.HEARTBEAT, payload={})
        e2 = Event(event_type=EventType.HEARTBEAT, payload={})
        assert e1.event_id != e2.event_id


class TestMarketData:
    def test_market_data_creation(self):
        md = MarketData(
            condition_id="cond123",
            question="Will X?",
            category="politics",
            clob_token_ids=["tok_yes", "tok_no"],
            volume_24h=50000.0,
            liquidity=100000.0,
        )
        assert md.condition_id == "cond123"
        assert md.active is True
        assert md.volume_24h == 50000.0

    def test_market_data_defaults(self):
        md = MarketData(
            condition_id="c1",
            question="Q?",
            category="sports",
            clob_token_ids=[],
        )
        assert md.volume_24h == 0.0
        assert md.liquidity == 0.0
        assert md.end_date is None
        assert md.active is True


class TestOrderBookLevel:
    def test_level(self):
        level = OrderBookLevel(price=0.55, size=100.0)
        assert level.price == 0.55
        assert level.size == 100.0


class TestOrderBookSnapshot:
    def test_empty_snapshot_defaults(self):
        snap = OrderBookSnapshot(token_id="tok1", bids=[], asks=[])
        assert snap.best_bid == 0.0
        assert snap.best_ask == 1.0
        assert snap.spread == pytest.approx(1.0)
        assert snap.midpoint == pytest.approx(0.5)

    def test_snapshot_with_levels(self):
        bids = [OrderBookLevel(0.50, 100), OrderBookLevel(0.49, 200)]
        asks = [OrderBookLevel(0.52, 100), OrderBookLevel(0.53, 200)]
        snap = OrderBookSnapshot(token_id="tok1", bids=bids, asks=asks)
        assert snap.best_bid == 0.50
        assert snap.best_ask == 0.52
        assert snap.spread == pytest.approx(0.02)
        assert snap.midpoint == pytest.approx(0.51)

    def test_snapshot_timestamp(self):
        now = datetime.now(UTC)
        snap = OrderBookSnapshot(token_id="t", bids=[], asks=[], timestamp=now)
        assert snap.timestamp == now


class TestTradeSignal:
    def test_signal(self):
        sig = TradeSignal(
            market_id="m1",
            question="Will X?",
            direction="YES",
            confidence=0.8,
            estimated_prob=0.7,
            edge=0.15,
            market_price=0.55,
            regime="trending",
            signal_breakdown={"sentiment": 0.6},
        )
        assert sig.direction == "YES"
        assert sig.category == "unknown"


class TestRiskCheckResult:
    def test_approved(self):
        r = RiskCheckResult(approved=True, position_size_usdc=100.0)
        assert r.reason == ""

    def test_rejected(self):
        r = RiskCheckResult(approved=False, position_size_usdc=0.0, reason="Too much exposure")
        assert not r.approved
        assert r.reason == "Too much exposure"

    def test_defaults(self):
        r = RiskCheckResult(approved=True, position_size_usdc=50.0)
        assert r.kelly_raw == 0.0
        assert r.kelly_adjusted == 0.0
        assert r.constraints_applied == []
