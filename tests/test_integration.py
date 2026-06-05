"""Integration tests that wire multiple components together with mocks."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymarket_agent.data.clob_client import OrderBookSnapshot
from polymarket_agent.data.gamma_client import Market
from polymarket_agent.execution.executor import Order, OrderExecutor
from polymarket_agent.main import PolymarketTradingAgent, _resolve_config_path
from polymarket_agent.models.calibration import ProbabilityCalibrator
from polymarket_agent.models.edge import EdgeResult
from polymarket_agent.tracking.learner import SelfAdjuster
from polymarket_agent.tracking.tracker import PerformanceTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_market(condition_id: str, question: str, volume: float = 5000.0) -> Market:
    return Market(
        condition_id=condition_id,
        question=question,
        description="Test market",
        category="crypto",
        end_date=datetime.now(UTC) + timedelta(days=30),
        active=True,
        liquidity=10_000.0,
        volume_24h=volume,
        yes_price=0.65,
        no_price=0.35,
        clob_token_ids=["tok_yes_1", "tok_no_1"],
    )


def _make_snapshot(token_id: str = "tok_yes_1") -> OrderBookSnapshot:  # noqa: S107
    return OrderBookSnapshot(
        token_id=token_id,
        timestamp=datetime.now(UTC),
        midpoint=0.65,
        spread=0.04,
        bid_depth=1500.0,
        ask_depth=1200.0,
        best_bid=0.63,
        best_ask=0.67,
        bids=[(0.63, 500), (0.62, 300)],
        asks=[(0.67, 400), (0.68, 350)],
    )


def _make_edge_result(
    market_id: str,
    edge: float = 0.12,
    direction: str = "YES",
) -> EdgeResult:
    return EdgeResult(
        estimated_prob=0.72,
        market_price=0.65,
        edge=edge,
        direction=direction,
        confidence=0.75,
        regime="stable",
        market_id=market_id,
        question="Will BTC exceed $100k?",
        category="crypto",
        signal_breakdown={"order_book": 0.03, "momentum": 0.0, "sentiment": 0.04},
    )


def _make_paper_order(market_id: str) -> Order:
    return Order(
        order_id="abc123",
        market_id=market_id,
        token_id="tok_yes_1",
        direction="YES",
        side="BUY",
        size_usdc=10.0,
        price=0.649,
        status="placed",
        timestamp=datetime.now(UTC),
        chunks=1,
        is_paper=True,
    )


def _write_config(tmp_path, sample_config) -> str:
    """Write sample_config to a YAML file and return its path."""
    import yaml

    cfg_path = tmp_path / "settings.yaml"
    cfg_path.write_text(yaml.dump(sample_config))
    return str(cfg_path)


def _build_agent(tmp_path, sample_config) -> PolymarketTradingAgent:
    """Build a PolymarketTradingAgent from a temp config, with safe DB paths."""
    sample_config = dict(sample_config)
    sample_config["dry_run"] = True
    cfg_path = _write_config(tmp_path, sample_config)

    with patch.dict(os.environ, {}, clear=False):
        agent = PolymarketTradingAgent(cfg_path)

    # Redirect tracker & learner to temp paths so tests don't pollute real data
    agent.tracker = PerformanceTracker(str(tmp_path / "trades.db"))
    agent.learner = SelfAdjuster(
        sample_config,
        weights_path=str(tmp_path / "learned_weights.json"),
    )
    agent.calibrator = ProbabilityCalibrator(
        path=str(tmp_path / "probability_calibration.json"),
        enabled=bool(sample_config.get("enable_probability_calibration", True)),
        method=str(sample_config.get("probability_calibration_method", "isotonic")),
        min_samples=int(sample_config.get("probability_calibration_min_samples", 30)),
        refit_interval=int(sample_config.get("probability_calibration_refit_interval", 25)),
    )
    agent.edge_detector.set_probability_calibrator(agent.calibrator)
    return agent


# ---------------------------------------------------------------------------
# 1. Full cycle with mocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_cycle_with_mocks(tmp_path, sample_config):
    agent = _build_agent(tmp_path, sample_config)

    m1 = _make_market("cid_1", "Will BTC exceed $100k?", volume=5000)
    m2 = _make_market("cid_2", "Will ETH exceed $10k?", volume=3000)

    snapshot = _make_snapshot()

    # Edge > min_edge for market 1, below for market 2
    async def fake_estimate_edge(mdata, features, regime):
        if mdata["market_id"] == "cid_1":
            return _make_edge_result("cid_1", edge=0.12)
        return _make_edge_result("cid_2", edge=0.01)

    agent.gamma.get_all_markets = AsyncMock(return_value=[m1, m2])
    agent.clob.get_order_book = AsyncMock(return_value=snapshot)
    agent.history.save_snapshot = MagicMock()
    agent.history.load_history = MagicMock(return_value=None)
    agent.history.list_markets = MagicMock(return_value=[])
    agent.news.search_news = AsyncMock(return_value=[])
    agent.news.get_news_volume = MagicMock(return_value=(0, 0.0))
    agent.sentiment.get_cached = MagicMock(return_value={"score": 0.0})
    agent.sentiment.analyze_batch = AsyncMock()
    agent.edge_detector.estimate_edge = AsyncMock(side_effect=fake_estimate_edge)
    agent.executor.place_order = AsyncMock(return_value=_make_paper_order("cid_1"))
    agent.executor.cancel_expired_orders = AsyncMock(return_value=[])
    agent.executor.cancel_all_open_orders = AsyncMock(return_value=0)
    agent.arbitrage.scan = MagicMock(return_value=[])
    agent.drawdown.update = MagicMock()
    agent.drawdown.get_multiplier = MagicMock(return_value=1.0)
    agent.drawdown.should_close_all = MagicMock(return_value=False)
    agent.drawdown.get_drawdown = MagicMock(return_value=0.0)
    agent.regime.fitted = True

    await agent.run_cycle()

    # At least one order should have been placed (market 1 had high edge)
    agent.executor.place_order.assert_called()

    # Tracker should have at least one recorded trade
    metrics = agent.tracker.get_metrics()
    assert metrics["trade_count"] + metrics["open_count"] >= 1


# ---------------------------------------------------------------------------
# 2. Cycle skips when edge is below threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cycle_skips_low_edge(tmp_path, sample_config):
    agent = _build_agent(tmp_path, sample_config)

    m1 = _make_market("cid_1", "Low edge market")
    snapshot = _make_snapshot()

    # All edges below min_edge (0.08 from sample_config)
    async def fake_low_edge(mdata, features, regime):
        return _make_edge_result(mdata["market_id"], edge=0.02)

    agent.gamma.get_all_markets = AsyncMock(return_value=[m1])
    agent.clob.get_order_book = AsyncMock(return_value=snapshot)
    agent.history.save_snapshot = MagicMock()
    agent.history.load_history = MagicMock(return_value=None)
    agent.history.list_markets = MagicMock(return_value=[])
    agent.news.search_news = AsyncMock(return_value=[])
    agent.news.get_news_volume = MagicMock(return_value=(0, 0.0))
    agent.sentiment.get_cached = MagicMock(return_value={"score": 0.0})
    agent.sentiment.analyze_batch = AsyncMock()
    agent.edge_detector.estimate_edge = AsyncMock(side_effect=fake_low_edge)
    agent.executor.place_order = AsyncMock()
    agent.executor.cancel_expired_orders = AsyncMock(return_value=[])
    agent.executor.cancel_all_open_orders = AsyncMock(return_value=0)
    agent.arbitrage.scan = MagicMock(return_value=[])
    agent.drawdown.update = MagicMock()
    agent.drawdown.get_multiplier = MagicMock(return_value=1.0)
    agent.drawdown.should_close_all = MagicMock(return_value=False)
    agent.drawdown.get_drawdown = MagicMock(return_value=0.0)
    agent.regime.fitted = True

    await agent.run_cycle()

    # No orders should have been placed
    agent.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_cycle_handles_history_without_pandas_shadowing(
    tmp_path,
    sample_config,
    sample_price_history,
):
    agent = _build_agent(tmp_path, sample_config)

    market = _make_market("cid_hist", "History-backed market")
    snapshot = _make_snapshot()

    async def fake_edge(mdata, features, regime):
        return _make_edge_result("cid_hist", edge=0.12)

    agent.gamma.get_all_markets = AsyncMock(return_value=[market])
    agent.clob.get_order_book = AsyncMock(return_value=snapshot)
    agent.history.save_snapshot = MagicMock()
    agent.history.load_history = MagicMock(return_value=sample_price_history)
    agent.history.list_markets = MagicMock(return_value=[market.condition_id])
    agent.news.search_news = AsyncMock(return_value=[])
    agent.news.get_news_volume = MagicMock(return_value=(0, 0.0))
    agent.sentiment.get_cached = MagicMock(return_value={"score": 0.0})
    agent.sentiment.analyze_batch = AsyncMock()
    agent.edge_detector.estimate_edge = AsyncMock(side_effect=fake_edge)
    agent.edge_detector.get_cross_market_signals = MagicMock(return_value=[])
    agent.executor.place_order = AsyncMock(return_value=_make_paper_order("cid_hist"))
    agent.executor.cancel_expired_orders = AsyncMock(return_value=[])
    agent.executor.cancel_all_open_orders = AsyncMock(return_value=0)
    agent.executor.reconcile_orders = AsyncMock(return_value=[])
    agent.arbitrage.scan = MagicMock(return_value=[])
    agent.drawdown.update = MagicMock()
    agent.drawdown.get_multiplier = MagicMock(return_value=1.0)
    agent.drawdown.should_close_all = MagicMock(return_value=False)
    agent.drawdown.get_drawdown = MagicMock(return_value=0.0)
    agent.regime.fitted = True
    agent.regime.predict = MagicMock(return_value=("stable", {"stable": 1.0}))

    await agent.run_cycle()

    assert agent._consecutive_failures == 0
    agent.executor.place_order.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Error recovery — GammaClient raises on first call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_recovery(tmp_path, sample_config, caplog):
    agent = _build_agent(tmp_path, sample_config)

    # get_all_markets raises ConnectionError — run_cycle should catch it
    agent.gamma.get_all_markets = AsyncMock(side_effect=ConnectionError("network down"))

    # run_cycle wraps everything in try/except, so it should not crash
    await agent.run_cycle()

    assert agent._consecutive_failures == 1


# ---------------------------------------------------------------------------
# 4. PerformanceTracker records and retrieves a trade
# ---------------------------------------------------------------------------


def test_tracker_records_trade(tmp_path):
    tracker = PerformanceTracker(str(tmp_path / "test_trades.db"))

    edge_result = {
        "market_id": "0xabc",
        "question": "Will BTC hit $100k?",
        "direction": "YES",
        "market_price": 0.65,
        "edge": 0.10,
        "confidence": 0.80,
        "raw_estimated_prob": 0.78,
        "estimated_prob": 0.72,
        "calibrated_prob": 0.72,
        "regime": "stable",
        "category": "crypto",
        "signal_breakdown": {"order_book": 0.05, "sentiment": 0.05},
    }

    trade_id = tracker.record_paper_trade(edge_result, size=20.0)
    assert trade_id is not None and trade_id > 0

    # Close it so it appears in metrics
    pnl = tracker.close_position(trade_id, exit_price=0.70)
    assert isinstance(pnl, float)

    metrics = tracker.get_metrics()
    assert metrics["trade_count"] == 1


def test_tracker_reports_calibration_gain(tmp_path):
    tracker = PerformanceTracker(str(tmp_path / "test_calibration_metrics.db"))

    trade_id = tracker.record_paper_trade(
        {
            "market_id": "0xcal",
            "question": "Will calibration help?",
            "direction": "YES",
            "market_price": 0.65,
            "edge": 0.05,
            "confidence": 0.70,
            "raw_estimated_prob": 0.90,
            "estimated_prob": 0.60,
            "calibrated_prob": 0.60,
            "regime": "stable",
            "category": "crypto",
            "signal_breakdown": {"order_book": 0.05},
        },
        size=20.0,
    )
    tracker.close_position(trade_id, exit_price=0.60)

    metrics = tracker.get_metrics()
    assert metrics["raw_brier_score"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["calibration_gain"] is not None
    assert metrics["calibration_gain"] > 0


def test_tracker_syncs_entry_and_bracket_lifecycle(tmp_path, sample_config):
    tracker = PerformanceTracker(str(tmp_path / "test_trades_lifecycle.db"))
    executor = OrderExecutor(sample_config)

    edge_result = {
        "market_id": "0xlifecycle",
        "question": "Will lifecycle test pass?",
        "direction": "YES",
        "market_price": 0.65,
        "edge": 0.11,
        "confidence": 0.82,
        "estimated_prob": 0.76,
        "regime": "stable",
        "category": "crypto",
        "signal_breakdown": {"order_book": 0.05, "sentiment": 0.03},
    }

    order = asyncio.run(
        executor.place_order(
            market_id="0xlifecycle",
            direction="YES",
            size_usdc=20.0,
            price=0.65,
            token_id="tok_yes_1",
        )
    )
    assert order is not None

    trade_id = tracker.record_paper_trade(edge_result, size=20.0, order=order.__dict__)
    assert trade_id > 0

    tracker.sync_entry_order(order.__dict__)
    metrics = tracker.get_metrics()
    assert metrics["armed_bracket_count"] >= 1

    take_profit_order = executor.orders[order.bracket_order_ids[0]]
    if take_profit_order.order_kind != "take_profit":
        take_profit_order = executor.orders[order.bracket_order_ids[1]]
    take_profit_order.status = "filled"
    take_profit_order.average_fill_price = 0.71
    take_profit_order.last_fill_at = datetime.now(UTC)

    pnl = tracker.close_trade_for_order(
        take_profit_order.__dict__,
        exit_price=0.71,
        exit_reason="take_profit triggered",
    )
    assert isinstance(pnl, float)
    assert tracker.get_metrics()["trade_count"] == 1


# ---------------------------------------------------------------------------
# 5. SelfAdjuster raises min_edge on poor performance
# ---------------------------------------------------------------------------


def test_learner_adjusts_min_edge(tmp_path, sample_config):
    cfg = dict(sample_config)
    cfg["min_edge"] = 0.08
    cfg["brier_check_interval"] = 50

    learner = SelfAdjuster(cfg, weights_path=str(tmp_path / "weights.json"))
    original_min_edge = learner.min_edge

    # Simulate poor performance: high Brier + low win rate at check boundary
    poor_metrics = {
        "trade_count": 50,  # hits brier_check_interval
        "brier_score": 0.35,  # > 0.30 → should raise min_edge
        "win_rate": 0.45,  # < 0.52 with count > 20 → should raise more
        "per_signal_accuracy": {},
    }

    result = learner.check_and_adjust(poor_metrics)

    assert learner.min_edge > original_min_edge
    assert result["min_edge"] > original_min_edge


def test_agent_refits_probability_calibrator(tmp_path, sample_config):
    cfg = dict(sample_config)
    cfg["probability_calibration_min_samples"] = 6
    cfg["probability_calibration_refit_interval"] = 3
    agent = _build_agent(tmp_path, cfg)

    for idx in range(6):
        trade_id = agent.tracker.record_paper_trade(
            {
                "market_id": f"0xfit{idx}",
                "question": "Refit?",
                "direction": "YES",
                "market_price": 0.50,
                "edge": 0.05,
                "confidence": 0.70,
                "raw_estimated_prob": 0.75 if idx % 2 == 0 else 0.35,
                "estimated_prob": 0.75 if idx % 2 == 0 else 0.35,
                "calibrated_prob": 0.75 if idx % 2 == 0 else 0.35,
                "regime": "stable",
                "category": "crypto",
                "signal_breakdown": {"order_book": 0.02},
            },
            size=10.0,
        )
        exit_price = 0.70 if idx % 2 == 0 else 0.30
        agent.tracker.close_position(trade_id, exit_price=exit_price)

    metrics = agent.tracker.get_metrics()
    fit_metrics = agent._refresh_probability_calibration(metrics)

    assert fit_metrics is not None
    assert agent.calibrator.is_fitted
    assert agent.calibrator.path.exists()


# ---------------------------------------------------------------------------
# 6. Shutdown cleanup calls close on sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_cleanup(tmp_path, sample_config):
    agent = _build_agent(tmp_path, sample_config)

    # Replace the async cleanup targets with mocks
    agent.executor.cancel_all_open_orders = AsyncMock(return_value=0)
    agent.gamma.close = AsyncMock()
    agent.clob.close = AsyncMock()

    await agent.cleanup()

    agent.executor.cancel_all_open_orders.assert_awaited_once()
    agent.gamma.close.assert_awaited_once()
    agent.clob.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_once_always_cleans_up(tmp_path, sample_config):
    agent = _build_agent(tmp_path, sample_config)

    agent.run_cycle = AsyncMock()
    agent.cleanup = AsyncMock()

    await agent.run_once()

    agent.run_cycle.assert_awaited_once()
    agent.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_once_cleans_up_after_failure(tmp_path, sample_config):
    agent = _build_agent(tmp_path, sample_config)

    agent.run_cycle = AsyncMock(side_effect=RuntimeError("boom"))
    agent.cleanup = AsyncMock()

    with pytest.raises(RuntimeError, match="boom"):
        await agent.run_once()

    agent.cleanup.assert_awaited_once()


def test_resolve_default_config_from_any_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_config_path("config/settings.yaml")
    expected = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"

    assert resolved == expected
