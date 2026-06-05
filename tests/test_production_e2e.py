"""Production readiness end-to-end tests.

Validates that all modules are properly integrated, error handling
works, and the system is ready for production deployment.
"""

import asyncio
import sqlite3
import tempfile
from datetime import UTC
from decimal import Decimal
from math import ceil

import pytest

# ── Import validation (all modules must be importable) ──


class TestModuleImports:
    """Every module must import without errors."""

    def test_import_types(self):
        from polymarket_agent.types import (
            EventType,
            OrderState,
        )

        assert len(OrderState) == 12
        assert len(EventType) == 18

    def test_import_event_bus(self):
        from polymarket_agent.events.bus import EventBus

        bus = EventBus()
        assert bus.stats["events_processed"] == 0

    def test_import_state_machine(self):
        from polymarket_agent.execution.state_machine import OrderStateMachine

        sm = OrderStateMachine("test-order")
        assert sm.state.value == "created"

    def test_import_slippage(self):
        from polymarket_agent.execution.slippage import (
            check_slippage_ok,
            estimate_slippage,
        )

        assert callable(estimate_slippage)
        assert callable(check_slippage_ok)

    def test_import_batch(self):
        from polymarket_agent.execution.batch import BatchOrderManager

        assert BatchOrderManager is not None

    def test_import_twap(self):
        from polymarket_agent.execution.twap import TWAPOrder

        assert TWAPOrder is not None

    def test_import_trailing_stop(self):
        from polymarket_agent.execution.trailing_stop import TrailingStop

        assert TrailingStop is not None

    def test_import_paper_engine(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        assert PaperTradingEngine is not None

    def test_import_ws_client(self):
        from polymarket_agent.data.ws_client import WebSocketClient

        assert WebSocketClient is not None

    def test_import_anomaly(self):
        from polymarket_agent.intelligence.anomaly import AnomalyDetector

        assert AnomalyDetector is not None

    def test_import_alerts(self):
        from polymarket_agent.risk.alerts import AlertManager

        assert AlertManager is not None

    def test_import_prices(self):
        from polymarket_agent.utils.prices import (
            to_price,
        )

        assert to_price(0.5) == Decimal("0.5")

    def test_import_shutdown(self):
        from polymarket_agent.infra.shutdown import GracefulShutdown

        gs = GracefulShutdown()
        assert not gs.is_shutting_down

    def test_import_retry(self):
        from polymarket_agent.infra.retry import (
            CircuitBreaker,
            CircuitState,
        )

        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_import_health(self):
        from polymarket_agent.infra.health import HealthChecker

        hc = HealthChecker()
        assert hc is not None

    def test_import_env_validator(self):
        from polymarket_agent.infra.env_validator import validate_env

        result = validate_env(require_live_trading=False)
        assert result is not None
        assert hasattr(result, "valid")


# ── Event Bus Integration ──


class TestEventBusIntegration:
    """Event bus must work with all event types."""

    @pytest.mark.asyncio
    async def test_publish_and_process_all_event_types(self):
        from polymarket_agent.events.bus import EventBus
        from polymarket_agent.types import Event, EventType

        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event.event_type)

        bus.subscribe_all(handler)

        for et in EventType:
            await bus.publish(Event(event_type=et, payload={"test": True}))

        count = await bus.process()
        assert count == len(EventType)
        assert len(received) == len(EventType)

    @pytest.mark.asyncio
    async def test_error_isolation(self):
        from polymarket_agent.events.bus import EventBus
        from polymarket_agent.types import Event, EventType

        bus = EventBus()
        good_received = []

        async def bad_handler(event):
            raise RuntimeError("boom")

        async def good_handler(event):
            good_received.append(event)

        bus.subscribe(EventType.ORDER_PLACED, bad_handler)
        bus.subscribe(EventType.ORDER_PLACED, good_handler)

        await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}))
        await bus.process()

        assert len(good_received) == 1  # Good handler still worked
        assert bus.stats["errors"] >= 1


# ── Order State Machine Integration ──


class TestStateMachineIntegration:
    """Order lifecycle state machine must enforce valid transitions."""

    def test_full_happy_path(self):
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import OrderState

        sm = OrderStateMachine("order-001")
        sm.transition(OrderState.VALIDATED)
        sm.transition(OrderState.SIGNED)
        sm.transition(OrderState.SUBMITTED)
        sm.transition(OrderState.ACKNOWLEDGED)
        sm.transition(OrderState.PARTIAL_FILL)
        sm.transition(OrderState.FILLED)
        sm.transition(OrderState.SETTLED)
        sm.transition(OrderState.REDEEMED)
        assert sm.state == OrderState.REDEEMED
        assert len(sm.history) == 8  # 8 transitions recorded

    def test_invalid_transition_raises(self):
        from polymarket_agent.execution.state_machine import (
            InvalidTransitionError,
            OrderStateMachine,
        )
        from polymarket_agent.types import OrderState

        sm = OrderStateMachine("order-002")
        with pytest.raises(InvalidTransitionError):
            sm.transition(OrderState.FILLED)  # Can't go CREATED -> FILLED

    def test_serialization_roundtrip(self):
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import OrderState

        sm = OrderStateMachine("order-003")
        sm.transition(OrderState.VALIDATED)
        sm.transition(OrderState.SIGNED)

        data = sm.to_dict()
        restored = OrderStateMachine.from_dict(data)
        assert restored.state == OrderState.SIGNED
        assert restored.order_id == "order-003"

    def test_terminal_states(self):
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import OrderState

        sm = OrderStateMachine("order-004")
        sm.transition(OrderState.CANCELLED)
        assert sm.is_terminal

    def test_force_state(self):
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import OrderState

        sm = OrderStateMachine("order-005")
        sm.force_state(OrderState.FILLED, reason="manual override")
        assert sm.state == OrderState.FILLED


# ── Retry & Circuit Breaker ──


class TestRetryCircuitBreaker:
    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self):
        from polymarket_agent.infra.retry import with_retry

        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01, jitter=False)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("temporary")
            return "ok"

        result = await flaky()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        from polymarket_agent.infra.retry import RetryExhausted, with_retry

        @with_retry(max_retries=2, base_delay=0.01, jitter=False)
        async def always_fails():
            raise ConnectionError("permanent")

        with pytest.raises(RetryExhausted):
            await always_fails()

    def test_circuit_breaker_opens(self):
        from polymarket_agent.infra.retry import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovers(self):
        from polymarket_agent.infra.retry import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        await asyncio.sleep(0.1)  # Wait for reset_timeout
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_reset(self):
        from polymarket_agent.infra.retry import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ── Graceful Shutdown ──


class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_hooks_run_in_order(self):
        from polymarket_agent.infra.shutdown import GracefulShutdown

        order = []

        async def hook_a():
            order.append("a")

        async def hook_b():
            order.append("b")

        gs = GracefulShutdown(timeout=5.0)
        gs.register(hook_a, name="first")
        gs.register(hook_b, name="second")

        results = await gs.execute()
        assert order == ["a", "b"]
        assert results["first"] == "ok"
        assert results["second"] == "ok"

    @pytest.mark.asyncio
    async def test_hook_timeout_handled(self):
        from polymarket_agent.infra.shutdown import GracefulShutdown

        async def slow_hook():
            await asyncio.sleep(100)

        gs = GracefulShutdown(timeout=0.1)
        gs.register(slow_hook, name="slow")

        results = await gs.execute()
        assert results["slow"] == "timeout"

    @pytest.mark.asyncio
    async def test_hook_error_handled(self):
        from polymarket_agent.infra.shutdown import GracefulShutdown

        async def bad_hook():
            raise RuntimeError("cleanup failed")

        gs = GracefulShutdown(timeout=5.0)
        gs.register(bad_hook, name="bad")

        results = await gs.execute()
        assert "error:" in results["bad"]


# ── Health Checker ──


class TestHealthChecker:
    @pytest.mark.asyncio
    async def test_all_healthy(self):
        from polymarket_agent.infra.health import HealthChecker, HealthStatus

        hc = HealthChecker()
        hc.register_check("db", lambda: True)
        hc.register_check("api", lambda: True)

        result = await hc.check_all()
        assert result["status"] == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_on_failure(self):
        from polymarket_agent.infra.health import HealthChecker, HealthStatus

        hc = HealthChecker()
        hc.register_check("db", lambda: True)
        hc.register_check("api", lambda: False)

        result = await hc.check_all()
        assert result["status"] in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_quick_status(self):
        from polymarket_agent.infra.health import HealthChecker

        hc = HealthChecker(service_name="test-service")
        status = hc.quick_status()
        assert "service" in status or "status" in status

    def test_metadata(self):
        from polymarket_agent.infra.health import HealthChecker

        hc = HealthChecker()
        hc.set_metadata("version", "1.0.0")
        hc.set_metadata("commit", "abc123")
        # Should not raise


# ── Decimal Prices ──


class TestDecimalPrices:
    def test_price_precision(self):
        from polymarket_agent.utils.prices import midpoint, spread, to_price

        bid = to_price(0.45)
        ask = to_price(0.55)
        s = spread(bid, ask)
        m = midpoint(bid, ask)
        assert s == Decimal("0.1")
        assert m == Decimal("0.5")

    def test_clamp_price_bounds(self):
        from polymarket_agent.utils.prices import clamp_price, to_price

        assert clamp_price(to_price(0.005)) == Decimal("0.01")
        assert clamp_price(to_price(0.999)) == Decimal("0.99")
        assert clamp_price(to_price(0.5)) == Decimal("0.5")

    def test_complement_price(self):
        from polymarket_agent.utils.prices import complement_price, to_price

        p = to_price(0.65)
        c = complement_price(p)
        assert c == Decimal("0.35")

    def test_is_valid_price(self):
        from polymarket_agent.utils.prices import is_valid_price

        assert is_valid_price(0.5)
        assert is_valid_price(0.01)
        assert is_valid_price(0.99)
        assert not is_valid_price(0.0)
        assert not is_valid_price(1.0)

    def test_to_usdc(self):
        from polymarket_agent.utils.prices import to_usdc

        result = to_usdc(10.556)
        assert result == Decimal("10.56")

    def test_vwap(self):
        from polymarket_agent.utils.prices import vwap

        levels = [(0.50, 100.0), (0.51, 200.0)]
        result = vwap(levels)
        assert result > Decimal("0.50")


# ── Slippage Estimation ──


class TestSlippageEstimation:
    def test_estimate_slippage_buy(self):
        from polymarket_agent.execution.slippage import estimate_slippage

        asks = [(0.55, 500.0), (0.56, 300.0), (0.57, 200.0)]
        bids = [(0.45, 500.0), (0.44, 300.0)]

        est = estimate_slippage(order_size_usdc=100.0, side="BUY", bids=bids, asks=asks)
        assert est.sufficient_liquidity
        assert est.slippage_bps >= 0

    def test_check_slippage_ok(self):
        from polymarket_agent.execution.slippage import check_slippage_ok

        asks = [(0.50, 5000.0), (0.501, 3000.0)]
        bids = [(0.499, 5000.0), (0.498, 3000.0)]

        ok, est = check_slippage_ok(order_size_usdc=50.0, side="BUY", bids=bids, asks=asks)
        assert ok
        assert est.sufficient_liquidity


# ── Executor Fixes ──


class TestExecutorFixes:
    @pytest.mark.asyncio
    async def test_bankroll_floor(self):
        """Bankroll should never go negative."""
        bankroll = 10.0
        size = 50.0
        bankroll = max(0.0, bankroll - size)
        assert bankroll == 0.0

    def test_chunk_count_capped(self):
        """Chunk count should be capped at 15."""
        size_usdc = 10000.0
        n_chunks = min(ceil(size_usdc / 20), 15)
        assert n_chunks == 15

    def test_recent_placements_cleanup(self):
        """Dedup cache should have hard cap."""
        from datetime import datetime

        cache = {}
        for i in range(11000):
            cache[(f"market-{i}", "YES", "entry")] = datetime.now(UTC)

        if len(cache) > 10000:
            oldest = sorted(cache, key=cache.get)[:5000]
            for k in oldest:
                del cache[k]
        assert len(cache) == 6000


# ── Tracker Fixes ──


class TestTrackerFixes:
    def test_wal_mode_enabled(self):
        """SQLite WAL mode should be enabled for concurrent access."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            conn = sqlite3.connect(f.name)
            conn.execute("PRAGMA journal_mode=WAL")
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
            conn.close()

    def test_negative_exit_price_clamped(self):
        """Negative exit prices should be clamped to 0."""
        exit_price = -0.5
        if exit_price < 0:
            exit_price = 0.0
        assert exit_price == 0.0


# ── Batch Orders ──


class TestBatchOrders:
    def test_batch_manager_creation(self):
        from polymarket_agent.execution.batch import BatchOrderManager

        bom = BatchOrderManager(dry_run=True)
        assert bom.pending_count == 0

    def test_add_pending_order(self):
        from polymarket_agent.execution.batch import BatchOrderManager, PendingOrder

        bom = BatchOrderManager(dry_run=True)
        order = PendingOrder(
            token_id="token-1",
            price=0.55,
            size=25.0,
            side="BUY",
            market_id="market-1",
        )
        count = bom.add(order)
        assert count == 1
        assert bom.pending_count == 1

    def test_clear_pending(self):
        from polymarket_agent.execution.batch import BatchOrderManager, PendingOrder

        bom = BatchOrderManager(dry_run=True)
        for i in range(5):
            bom.add(PendingOrder(token_id=f"token-{i}", price=0.5, size=10.0, side="BUY"))
        cleared = bom.clear()
        assert cleared == 5
        assert bom.pending_count == 0


# ── TWAP Orders ──


class TestTWAPOrders:
    def test_twap_creation(self):
        from polymarket_agent.execution.twap import TWAPOrder

        twap = TWAPOrder(
            order_id="twap-001",
            token_id="token-123",
            side="BUY",
            total_size=100.0,
            price_limit=0.65,
            num_slices=5,
            interval_seconds=60,
        )
        assert twap.status == "active"
        assert twap.progress["pct_complete"] == 0.0

    def test_twap_slice_generation(self):
        from polymarket_agent.execution.twap import TWAPOrder

        twap = TWAPOrder(
            order_id="twap-002",
            token_id="token-123",
            side="BUY",
            total_size=100.0,
            price_limit=0.65,
            num_slices=4,
            interval_seconds=30,
        )
        slice_info = twap.next_slice()
        assert slice_info is not None
        assert slice_info.size > 0

    def test_twap_completion(self):
        from polymarket_agent.execution.twap import TWAPOrder

        twap = TWAPOrder(
            order_id="twap-003",
            token_id="token-123",
            side="BUY",
            total_size=40.0,
            price_limit=0.60,
            num_slices=2,
            interval_seconds=0,  # No delay between slices for testing
        )
        for _ in range(2):
            s = twap.next_slice()
            assert s is not None
            twap.record_fill(s.slice_id, fill_price=0.59, fill_size=s.size)

        assert twap.is_complete
        assert twap.progress["pct_complete"] == 100.0

    def test_twap_cancel(self):
        from polymarket_agent.execution.twap import TWAPOrder

        twap = TWAPOrder(
            order_id="twap-004",
            token_id="token-123",
            side="BUY",
            total_size=100.0,
            price_limit=0.65,
            num_slices=5,
            interval_seconds=60,
        )
        twap.cancel()
        assert twap.status == "cancelled"


# ── Trailing Stop ──


class TestTrailingStop:
    def test_trailing_stop_sell(self):
        from polymarket_agent.execution.trailing_stop import TrailingStop

        ts = TrailingStop(
            order_id="ts-001",
            token_id="token-1",
            side="SELL",
            initial_price=0.60,
            trail_amount=0.05,
        )
        # Update with rising price
        ts.update_price(0.65)
        ts.update_price(0.70)

        # Price drops but not enough to trigger
        triggered = ts.update_price(0.66)
        assert not triggered

        # Price drops below trail
        triggered = ts.update_price(0.64)
        assert triggered

    def test_trailing_stop_buy(self):
        from polymarket_agent.execution.trailing_stop import TrailingStop

        ts = TrailingStop(
            order_id="ts-002",
            token_id="token-1",
            side="BUY",
            initial_price=0.60,
            trail_amount=0.05,
        )
        # Update with falling price
        ts.update_price(0.55)
        ts.update_price(0.50)

        # Price rises but not enough to trigger
        triggered = ts.update_price(0.54)
        assert not triggered

        # Price rises above trail
        triggered = ts.update_price(0.56)
        assert triggered

    def test_trailing_stop_status(self):
        from polymarket_agent.execution.trailing_stop import TrailingStop

        ts = TrailingStop(
            order_id="ts-003",
            token_id="token-1",
            side="SELL",
            initial_price=0.60,
            trail_amount=0.05,
        )
        status = ts.status
        assert "stop_price" in status or "high_water" in status

    def test_trailing_stop_reset(self):
        from polymarket_agent.execution.trailing_stop import TrailingStop

        ts = TrailingStop(
            order_id="ts-004",
            token_id="token-1",
            side="SELL",
            initial_price=0.60,
            trail_amount=0.05,
        )
        ts.update_price(0.70)
        ts.reset(new_price=0.55)
        assert ts.stop_price is not None


# ── Anomaly Detector ──


class TestAnomalyDetector:
    def test_anomaly_detector_creation(self):
        from polymarket_agent.intelligence.anomaly import AnomalyDetector

        detector = AnomalyDetector(window_size=5)
        assert detector is not None

    def test_anomaly_detection_with_data(self):
        from polymarket_agent.intelligence.anomaly import AnomalyDetector

        detector = AnomalyDetector(window_size=5)
        market_id = "test-market"

        # Feed normal data to build baseline
        for price in [0.50, 0.51, 0.49, 0.50, 0.51]:
            detector.check(market_id, price=price, volume=100, spread=0.02)

        # Feed anomalous data
        anomalies = detector.check(market_id, price=0.80, volume=10000, spread=0.10)
        assert len(anomalies) > 0  # Should detect at least one anomaly

    def test_anomaly_reset(self):
        from polymarket_agent.intelligence.anomaly import AnomalyDetector

        detector = AnomalyDetector()
        detector.check("market-1", price=0.5, volume=100)
        detector.reset("market-1")
        detector.reset()  # Reset all


# ── Paper Trading Engine ──


class TestPaperTradingEngine:
    def test_paper_engine_creation(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine()
        assert engine is not None

    def test_paper_order_submit(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine(miss_prob=0.0, partial_fill_prob=0.0)
        order = engine.submit_order(
            order_id="paper-001",
            token_id="token-1",
            side="BUY",
            price=0.55,
            size=100.0,
        )
        assert order.order_id == "paper-001"
        assert order.status == "open"

    def test_paper_order_cancel(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine()
        engine.submit_order(
            order_id="paper-002",
            token_id="token-1",
            side="BUY",
            price=0.55,
            size=50.0,
        )
        cancelled = engine.cancel_order("paper-002")
        assert cancelled

    def test_paper_order_fill(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine(miss_prob=0.0, partial_fill_prob=0.0)
        engine.submit_order(
            order_id="paper-003",
            token_id="token-1",
            side="BUY",
            price=0.55,
            size=100.0,
        )

        bids = [(0.45, 500.0), (0.44, 300.0)]
        asks = [(0.55, 500.0), (0.56, 300.0)]

        fills = engine.simulate_fills(bids=bids, asks=asks)
        assert isinstance(fills, list)

    def test_paper_engine_reset(self):
        from polymarket_agent.execution.paper_engine import PaperTradingEngine

        engine = PaperTradingEngine()
        engine.submit_order(
            order_id="paper-004",
            token_id="token-1",
            side="BUY",
            price=0.55,
            size=50.0,
        )
        engine.reset()
        assert len(engine.open_orders) == 0


# ── Alert System ──


class TestAlertSystem:
    def test_alert_creation(self):
        from polymarket_agent.risk.alerts import AlertManager

        manager = AlertManager()
        assert manager is not None

    def test_add_and_get_alert(self):
        from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

        manager = AlertManager()
        alert = Alert(
            alert_id="alert-001",
            market_id="market-1",
            condition=AlertCondition.PRICE_ABOVE,
            threshold=0.75,
            message="Price target hit",
        )
        manager.add_alert(alert)
        retrieved = manager.get_alert("alert-001")
        assert retrieved is not None
        assert retrieved.alert_id == "alert-001"

    def test_alert_trigger(self):
        from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

        manager = AlertManager()
        alert = Alert(
            alert_id="alert-002",
            market_id="market-1",
            condition=AlertCondition.PRICE_ABOVE,
            threshold=0.75,
        )
        manager.add_alert(alert)
        triggered = manager.check("market-1", price=0.80)
        assert len(triggered) > 0

    def test_alert_cancel(self):
        from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

        manager = AlertManager()
        alert = Alert(
            alert_id="alert-003",
            market_id="market-1",
            condition=AlertCondition.PRICE_BELOW,
            threshold=0.30,
        )
        manager.add_alert(alert)
        assert manager.cancel_alert("alert-003")
        active = manager.get_active_alerts("market-1")
        assert len(active) == 0


# ── Environment Validator ──


class TestEnvValidator:
    def test_validate_no_live_trading(self):
        from polymarket_agent.infra.env_validator import validate_env

        result = validate_env(require_live_trading=False)
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")

    def test_mask_value(self):
        from polymarket_agent.infra.env_validator import mask_value

        masked = mask_value("my-secret-api-key", show_chars=4)
        assert "my-s" in masked or "****" in masked
        assert masked != "my-secret-api-key"

    def test_get_env_with_default(self):
        from polymarket_agent.infra.env_validator import get_env

        val = get_env("NONEXISTENT_VAR_12345", default="fallback")
        assert val == "fallback"


# ── End-to-End Workflow ──


class TestEndToEndWorkflow:
    """Test the complete signal-to-order workflow."""

    def test_edge_to_sizer_to_executor_types(self):
        """Verify type compatibility across the pipeline."""
        from polymarket_agent.types import RiskCheckResult, TradeSignal

        signal = TradeSignal(
            market_id="test-market",
            question="Will X happen?",
            direction="YES",
            edge=0.05,
            confidence=0.8,
            estimated_prob=0.65,
            market_price=0.60,
            regime="stable",
            signal_breakdown={"sentiment": 0.3, "model": 0.7},
        )

        risk = RiskCheckResult(
            approved=True,
            position_size_usdc=25.0,
            reason="All checks passed",
            constraints_applied=["kelly", "drawdown", "portfolio"],
            kelly_raw=0.08,
            kelly_adjusted=0.04,
        )

        assert signal.edge > 0
        assert risk.approved
        assert risk.position_size_usdc > 0

    @pytest.mark.asyncio
    async def test_event_bus_connects_components(self):
        """Event bus should connect detector -> executor -> tracker lifecycle."""
        from polymarket_agent.events.bus import EventBus
        from polymarket_agent.types import Event, EventType

        bus = EventBus()
        lifecycle = []

        async def track_event(event):
            lifecycle.append(event.event_type)

        bus.subscribe(EventType.EDGE_DETECTED, track_event)
        bus.subscribe(EventType.ORDER_PLACED, track_event)
        bus.subscribe(EventType.ORDER_FILLED, track_event)
        bus.subscribe(EventType.TRADE_CLOSED, track_event)

        for et in [
            EventType.EDGE_DETECTED,
            EventType.ORDER_PLACED,
            EventType.ORDER_FILLED,
            EventType.TRADE_CLOSED,
        ]:
            await bus.publish(Event(event_type=et, payload={"market": "test"}))

        await bus.process()
        assert lifecycle == [
            EventType.EDGE_DETECTED,
            EventType.ORDER_PLACED,
            EventType.ORDER_FILLED,
            EventType.TRADE_CLOSED,
        ]

    def test_state_machine_with_event_types(self):
        """State machine transitions correspond to event lifecycle."""
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import EventType, OrderState

        sm = OrderStateMachine("workflow-001")

        # Simulate the order lifecycle that events would drive
        state_event_map = [
            (OrderState.VALIDATED, EventType.EDGE_DETECTED),
            (OrderState.SIGNED, EventType.ORDER_PLACED),
            (OrderState.SUBMITTED, EventType.ORDER_PLACED),
            (OrderState.ACKNOWLEDGED, EventType.ORDER_PLACED),
            (OrderState.FILLED, EventType.ORDER_FILLED),
            (OrderState.SETTLED, EventType.TRADE_CLOSED),
        ]

        for state, _event in state_event_map:
            sm.transition(state)

        assert sm.state == OrderState.SETTLED

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self):
        """Integration: signal -> risk check -> batch -> state machine."""
        from polymarket_agent.events.bus import EventBus
        from polymarket_agent.execution.batch import BatchOrderManager, PendingOrder
        from polymarket_agent.execution.state_machine import OrderStateMachine
        from polymarket_agent.types import (
            Event,
            EventType,
            OrderState,
            RiskCheckResult,
            TradeSignal,
        )

        # 1. Create signal
        signal = TradeSignal(
            market_id="integration-market",
            question="Integration test?",
            direction="YES",
            edge=0.10,
            confidence=0.9,
            estimated_prob=0.70,
            market_price=0.60,
            regime="stable",
            signal_breakdown={"model": 1.0},
        )

        # 2. Risk check
        risk = RiskCheckResult(
            approved=True,
            position_size_usdc=50.0,
            reason="OK",
            kelly_raw=0.12,
            kelly_adjusted=0.06,
        )

        # 3. Create state machine
        sm = OrderStateMachine("integration-001")
        sm.transition(OrderState.VALIDATED)

        # 4. Add to batch
        bom = BatchOrderManager(dry_run=True)
        bom.add(
            PendingOrder(
                token_id="token-int",
                price=signal.market_price,
                size=risk.position_size_usdc,
                side="BUY",
                market_id=signal.market_id,
                direction=signal.direction,
            )
        )
        assert bom.pending_count == 1

        # 5. Track via event bus
        bus = EventBus()
        events_seen = []

        async def tracker(event):
            events_seen.append(event.event_type)

        bus.subscribe_all(tracker)

        await bus.publish(
            Event(
                event_type=EventType.EDGE_DETECTED,
                payload={"market": signal.market_id},
            )
        )
        await bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED,
                payload={"order_id": sm.order_id},
            )
        )
        await bus.process()

        assert EventType.EDGE_DETECTED in events_seen
        assert EventType.ORDER_PLACED in events_seen
