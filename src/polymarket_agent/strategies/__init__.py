"""Multi-strategy scaffolding."""

from polymarket_agent.strategies.base import Signal, Strategy
from polymarket_agent.strategies.registry import StrategyRegistry

__all__ = ["Signal", "Strategy", "StrategyRegistry"]
