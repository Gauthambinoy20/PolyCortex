"""Price momentum strategy."""

from __future__ import annotations

from collections import deque

from polymarket_agent.strategies.base import Signal, Strategy


class MomentumStrategy(Strategy):
    """Simple price-momentum strategy over last *lookback* snapshots.

    Maintains a per-market rolling price buffer and emits a probability that
    drifts toward 1.0 when price is trending up and 0.0 when trending down.
    """

    name = "momentum"

    def __init__(self, lookback: int = 10) -> None:
        self.lookback = max(2, int(lookback))
        self._history: dict[str, deque[float]] = {}

    def _record(self, market_id: str, price: float) -> deque[float]:
        buf = self._history.setdefault(market_id, deque(maxlen=self.lookback))
        buf.append(float(price))
        return buf

    def score(self, market: dict) -> Signal:
        market_id = str(market.get("market_id") or market.get("condition_id") or market.get("id") or "default")
        price = float(market.get("midpoint", market.get("best_bid", 0.5)))
        buf = self._record(market_id, price)

        if len(buf) < 2:
            return Signal(probability=price, confidence=0.0, metadata={"lookback": len(buf)})

        first = buf[0]
        last = buf[-1]
        delta = last - first
        # Scale delta to a probability nudge; clip to keep well within (0,1)
        nudge = max(-0.3, min(0.3, delta * 2.0))
        probability = max(0.01, min(0.99, price + nudge))
        # Confidence increases with both buffer size and signal magnitude
        confidence = min(1.0, (len(buf) / self.lookback) * min(1.0, abs(delta) * 5.0))

        return Signal(
            probability=probability,
            confidence=confidence,
            metadata={"delta": delta, "lookback": len(buf), "nudge": nudge},
        )
