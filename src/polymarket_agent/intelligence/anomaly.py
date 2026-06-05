"""Anomaly detector for unusual market activity.

Detects:
- Volume spikes (current volume > N * rolling average)
- Sudden price moves (price change > threshold in short window)
- Spread widening (bid-ask spread > N * average spread)
- Unusual activity patterns

Emits events via EventBus when anomalies are detected.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from polymarket_agent.types import Event, EventType

logger = logging.getLogger(__name__)


class AnomalyType(StrEnum):
    VOLUME_SPIKE = "volume_spike"
    PRICE_JUMP = "price_jump"
    SPREAD_WIDENING = "spread_widening"
    LIQUIDITY_DROP = "liquidity_drop"


@dataclass
class Anomaly:
    """A detected market anomaly."""

    anomaly_type: AnomalyType
    market_id: str
    severity: float  # 0.0 - 1.0
    current_value: float
    baseline_value: float
    ratio: float
    timestamp: float = field(default_factory=time.monotonic)
    message: str = ""

    def to_event(self) -> Event:
        return Event(
            event_type=EventType.ANOMALY_DETECTED,
            payload={
                "anomaly_type": self.anomaly_type,
                "market_id": self.market_id,
                "severity": round(self.severity, 3),
                "current": self.current_value,
                "baseline": self.baseline_value,
                "ratio": round(self.ratio, 2),
                "message": self.message,
            },
            source="anomaly_detector",
        )


class MarketTracker:
    """Tracks rolling statistics for a single market."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self._prices: deque[float] = deque(maxlen=window_size)
        self._volumes: deque[float] = deque(maxlen=window_size)
        self._spreads: deque[float] = deque(maxlen=window_size)
        self._last_price: float | None = None

    def update(self, price: float, volume: float = 0.0, spread: float = 0.0) -> None:
        self._last_price = price
        self._prices.append(price)
        if volume > 0:
            self._volumes.append(volume)
        if spread > 0:
            self._spreads.append(spread)

    @property
    def avg_price(self) -> float:
        return sum(self._prices) / len(self._prices) if self._prices else 0.0

    @property
    def avg_volume(self) -> float:
        return sum(self._volumes) / len(self._volumes) if self._volumes else 0.0

    @property
    def avg_spread(self) -> float:
        return sum(self._spreads) / len(self._spreads) if self._spreads else 0.0

    @property
    def price_std(self) -> float:
        if len(self._prices) < 3:
            return 0.0
        avg = self.avg_price
        variance = sum((p - avg) ** 2 for p in self._prices) / len(self._prices)
        return float(variance**0.5)

    @property
    def has_enough_data(self) -> bool:
        return len(self._prices) >= 5


class AnomalyDetector:
    """Monitors markets for anomalous behavior.

    Usage::

        detector = AnomalyDetector()
        anomalies = detector.check(
            market_id="0x123",
            price=0.75, volume=50000, spread=0.03,
        )
        for a in anomalies:
            await event_bus.publish(a.to_event())
    """

    def __init__(
        self,
        volume_spike_threshold: float = 3.0,
        price_jump_std_mult: float = 3.0,
        spread_widen_threshold: float = 2.5,
        liquidity_drop_threshold: float = 0.5,
        window_size: int = 50,
    ) -> None:
        self.volume_spike_threshold = volume_spike_threshold
        self.price_jump_std_mult = price_jump_std_mult
        self.spread_widen_threshold = spread_widen_threshold
        self.liquidity_drop_threshold = liquidity_drop_threshold
        self.window_size = window_size
        self._trackers: dict[str, MarketTracker] = {}
        self._anomaly_count: int = 0

    def _get_tracker(self, market_id: str) -> MarketTracker:
        if market_id not in self._trackers:
            self._trackers[market_id] = MarketTracker(self.window_size)
        return self._trackers[market_id]

    def check(
        self,
        market_id: str,
        price: float,
        volume: float = 0.0,
        spread: float = 0.0,
        liquidity: float = 0.0,
    ) -> list[Anomaly]:
        """Check a market update for anomalies.

        Args:
            market_id: Market identifier.
            price: Current market price.
            volume: Current volume (24h or recent).
            spread: Current bid-ask spread.
            liquidity: Current total liquidity.

        Returns:
            List of detected anomalies (may be empty).
        """
        tracker = self._get_tracker(market_id)
        anomalies: list[Anomaly] = []

        if tracker.has_enough_data:
            # Volume spike
            if volume > 0 and tracker.avg_volume > 0:
                ratio = volume / tracker.avg_volume
                if ratio >= self.volume_spike_threshold:
                    severity = min(
                        1.0,
                        (ratio - self.volume_spike_threshold) / self.volume_spike_threshold,
                    )
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.VOLUME_SPIKE,
                            market_id=market_id,
                            severity=severity,
                            current_value=volume,
                            baseline_value=tracker.avg_volume,
                            ratio=ratio,
                            message=f"Volume {ratio:.1f}x above average",
                        )
                    )

            # Price jump
            price_change = abs(price - tracker.avg_price)
            std = tracker.price_std
            if price_change > 0 and std >= 0:
                # Use a small floor so perfectly stable prices still trigger
                # on large moves.
                z_score = price_change / std if std > 1e-9 else float("inf")
                if z_score >= self.price_jump_std_mult:
                    severity = (
                        min(1.0, (z_score - self.price_jump_std_mult) / self.price_jump_std_mult)
                        if z_score < float("inf")
                        else 1.0
                    )
                    display_z = min(z_score, 999.9)
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.PRICE_JUMP,
                            market_id=market_id,
                            severity=severity,
                            current_value=price,
                            baseline_value=tracker.avg_price,
                            ratio=display_z,
                            message=(
                                f"Price {display_z:.1f}\u03c3 from mean ({tracker.avg_price:.3f} \u2192 {price:.3f})"
                            ),
                        )
                    )

            # Spread widening
            if spread > 0 and tracker.avg_spread > 0:
                ratio = spread / tracker.avg_spread
                if ratio >= self.spread_widen_threshold:
                    severity = min(
                        1.0,
                        (ratio - self.spread_widen_threshold) / self.spread_widen_threshold,
                    )
                    anomalies.append(
                        Anomaly(
                            anomaly_type=AnomalyType.SPREAD_WIDENING,
                            market_id=market_id,
                            severity=severity,
                            current_value=spread,
                            baseline_value=tracker.avg_spread,
                            ratio=ratio,
                            message=f"Spread {ratio:.1f}x wider than average",
                        )
                    )

        # Update tracker after checks
        tracker.update(price, volume, spread)

        if anomalies:
            self._anomaly_count += len(anomalies)
            for a in anomalies:
                logger.warning(
                    "ANOMALY [%s] %s: %s (severity=%.2f)",
                    a.anomaly_type,
                    market_id,
                    a.message,
                    a.severity,
                )

        return anomalies

    @property
    def stats(self) -> dict:
        return {
            "markets_tracked": len(self._trackers),
            "total_anomalies": self._anomaly_count,
        }

    def reset(self, market_id: str | None = None) -> None:
        """Reset tracker(s). If market_id is None, reset all."""
        if market_id:
            self._trackers.pop(market_id, None)
        else:
            self._trackers.clear()
            self._anomaly_count = 0
