"""Strategy registry with weighted ensemble voting."""

from __future__ import annotations

import logging
from typing import Any

from polymarket_agent.strategies.base import Signal, Strategy

logger = logging.getLogger(__name__)

_BUILTIN_STRATEGIES: dict[str, type[Strategy]] = {}


def _builtins() -> dict[str, type[Strategy]]:
    if not _BUILTIN_STRATEGIES:
        from polymarket_agent.strategies.bayesian_tcn import BayesianTCNStrategy
        from polymarket_agent.strategies.momentum import MomentumStrategy

        _BUILTIN_STRATEGIES["bayesian_tcn"] = BayesianTCNStrategy
        _BUILTIN_STRATEGIES["momentum"] = MomentumStrategy
    return _BUILTIN_STRATEGIES


class StrategyRegistry:
    """Holds registered strategies with per-strategy weights.

    The ensemble signal is a weight-weighted average of each strategy's
    ``probability`` and ``confidence``.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, tuple[Strategy, float]] = {}

    def register(self, name: str, strategy: Strategy, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self._strategies[name] = (strategy, float(weight))

    def unregister(self, name: str) -> None:
        self._strategies.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)

    @property
    def names(self) -> list[str]:
        return list(self._strategies.keys())

    def get_ensemble_signal(self, market: dict) -> Signal:
        """Weighted average of all strategy signals."""
        if not self._strategies:
            return Signal(probability=0.5, confidence=0.0, metadata={"empty_registry": True})

        total_w = 0.0
        p_sum = 0.0
        c_sum = 0.0
        per_strategy: dict[str, dict[str, Any]] = {}
        for name, (strat, weight) in self._strategies.items():
            if weight <= 0:
                continue
            try:
                sig = strat.score(market)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Strategy %s failed: %s", name, exc)
                continue
            total_w += weight
            p_sum += sig.probability * weight
            c_sum += sig.confidence * weight
            per_strategy[name] = {
                "probability": sig.probability,
                "confidence": sig.confidence,
                "weight": weight,
                "metadata": sig.metadata,
            }

        if total_w <= 0:
            return Signal(probability=0.5, confidence=0.0, metadata={"per_strategy": per_strategy})

        return Signal(
            probability=p_sum / total_w,
            confidence=c_sum / total_w,
            metadata={"per_strategy": per_strategy},
        )

    @classmethod
    def from_config(cls, config: list[dict]) -> StrategyRegistry:
        """Build a registry from a list of ``{"name": ..., "weight": ..., ...}`` dicts.

        Unknown strategies are skipped with a warning.
        """
        reg = cls()
        for item in config or []:
            name = item.get("name")
            weight = float(item.get("weight", 1.0))
            if not name:
                continue
            kwargs = {k: v for k, v in item.items() if k not in {"name", "weight"}}
            builtin = _builtins().get(name)
            if builtin is None:
                logger.warning("Unknown strategy %r, skipping", name)
                continue
            try:
                strat = builtin(**kwargs)
            except TypeError:
                strat = builtin()
            reg.register(name, strat, weight=weight)
        return reg
