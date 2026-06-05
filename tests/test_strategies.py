"""Tests for multi-strategy registry."""

from __future__ import annotations

import pytest

from polymarket_agent.strategies import Signal, Strategy, StrategyRegistry
from polymarket_agent.strategies.bayesian_tcn import BayesianTCNStrategy
from polymarket_agent.strategies.momentum import MomentumStrategy


class ConstStrategy(Strategy):
    name = "const"

    def __init__(self, p: float = 0.7, c: float = 0.5) -> None:
        self._p = p
        self._c = c

    def score(self, market: dict) -> Signal:
        return Signal(probability=self._p, confidence=self._c, metadata={"k": 1})


def _market(**overrides) -> dict:
    base = {
        "market_id": "m1",
        "midpoint": 0.55,
        "bid_depth": 1000.0,
        "ask_depth": 800.0,
        "price_roc_24h": 0.0,
        "sentiment_score": 0.1,
        "sentiment_confidence": 0.5,
    }
    base.update(overrides)
    return base


def test_signal_clamps_ranges():
    s = Signal(probability=1.5, confidence=-0.2)
    assert s.probability == 1.0
    assert s.confidence == 0.0


def test_bayesian_tcn_score():
    strat = BayesianTCNStrategy()
    sig = strat.score(_market())
    assert 0.0 <= sig.probability <= 1.0
    assert 0.0 <= sig.confidence <= 1.0
    assert "order_book" in sig.metadata


def test_bayesian_tcn_with_tcn_prob():
    strat = BayesianTCNStrategy()
    sig = strat.score(_market(tcn_prob=0.9))
    assert sig.metadata["tcn_used"] is True


def test_momentum_accumulates():
    strat = MomentumStrategy(lookback=5)
    s1 = strat.score(_market(midpoint=0.4))
    assert s1.confidence == 0.0  # first snapshot => no delta
    for price in (0.45, 0.5, 0.55, 0.6):
        sig = strat.score(_market(midpoint=price))
    assert sig.probability > 0.5
    assert sig.confidence > 0.0


def test_registry_empty_returns_neutral():
    reg = StrategyRegistry()
    sig = reg.get_ensemble_signal(_market())
    assert sig.probability == 0.5
    assert sig.confidence == 0.0


def test_registry_weighted_average():
    reg = StrategyRegistry()
    reg.register("a", ConstStrategy(p=0.8, c=1.0), weight=3.0)
    reg.register("b", ConstStrategy(p=0.4, c=0.5), weight=1.0)
    sig = reg.get_ensemble_signal(_market())
    # (0.8*3 + 0.4*1) / 4 = 0.7
    assert sig.probability == pytest.approx(0.7)
    assert sig.confidence == pytest.approx((1.0 * 3 + 0.5 * 1) / 4)
    assert "per_strategy" in sig.metadata
    assert set(sig.metadata["per_strategy"].keys()) == {"a", "b"}


def test_registry_skips_zero_weight():
    reg = StrategyRegistry()
    reg.register("a", ConstStrategy(p=0.9), weight=1.0)
    reg.register("b", ConstStrategy(p=0.1), weight=0.0)
    sig = reg.get_ensemble_signal(_market())
    assert sig.probability == pytest.approx(0.9)


def test_registry_handles_strategy_exception():
    class Broken(Strategy):
        name = "broken"

        def score(self, market: dict) -> Signal:
            raise RuntimeError("boom")

    reg = StrategyRegistry()
    reg.register("broken", Broken(), weight=1.0)
    reg.register("ok", ConstStrategy(p=0.6), weight=1.0)
    sig = reg.get_ensemble_signal(_market())
    assert sig.probability == pytest.approx(0.6)


def test_registry_negative_weight_rejected():
    reg = StrategyRegistry()
    with pytest.raises(ValueError):
        reg.register("a", ConstStrategy(), weight=-1.0)


def test_registry_unregister_and_contains():
    reg = StrategyRegistry()
    reg.register("a", ConstStrategy())
    assert "a" in reg
    assert len(reg) == 1
    assert "a" in reg.names
    reg.unregister("a")
    assert "a" not in reg


def test_registry_from_config_with_unknown():
    cfg = [
        {"name": "bayesian_tcn", "weight": 0.7},
        {"name": "momentum", "weight": 0.3, "lookback": 5},
        {"name": "nonexistent", "weight": 1.0},
    ]
    reg = StrategyRegistry.from_config(cfg)
    assert "bayesian_tcn" in reg
    assert "momentum" in reg
    assert "nonexistent" not in reg


def test_registry_from_config_empty():
    assert len(StrategyRegistry.from_config([])) == 0
    assert len(StrategyRegistry.from_config(None)) == 0
