"""Strategy base class and Signal dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Signal:
    """Normalized trading signal emitted by a strategy.

    Attributes:
        probability: Estimated probability of YES outcome (0-1).
        confidence: Confidence in the estimate (0-1).
        metadata: Optional per-strategy diagnostic info.
    """

    probability: float
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probability = max(0.0, min(1.0, float(self.probability)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


class Strategy(ABC):
    """Abstract base for trading strategies that score markets."""

    name: str = "base"

    @abstractmethod
    def score(self, market: dict) -> Signal:
        """Score a market and return a trading signal."""
        ...
