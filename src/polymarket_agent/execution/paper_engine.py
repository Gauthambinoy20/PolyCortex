"""Paper trading simulation engine.

Simulates order fills against real orderbook snapshots with:
- Realistic fill simulation (walk the book, partial fills)
- Configurable latency simulation
- Fee modeling (maker 0%, taker varies by tier)
- Slippage simulation based on order book depth
- Fill probability based on price distance from midpoint
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


class FillType(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


@dataclass
class SimulatedFill:
    """Result of a simulated order fill."""

    order_id: str
    fill_type: FillType
    fill_price: float
    fill_size: float
    remaining_size: float
    fee_usdc: float
    is_maker: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def total_cost(self) -> float:
        return self.fill_size * self.fill_price + self.fee_usdc


@dataclass
class PaperOrder:
    """An order in the paper trading engine."""

    order_id: str
    token_id: str
    side: str  # 'BUY' or 'SELL'
    price: float
    size: float
    filled_size: float = 0.0
    status: str = "open"
    created_at: float = field(default_factory=time.monotonic)
    fills: list[SimulatedFill] = field(default_factory=list)


class PaperTradingEngine:
    """Simulates order execution for paper trading.

    Usage:
        engine = PaperTradingEngine()
        order = engine.submit_order("buy1", "token123", "BUY", price=0.55, size=100)

        # Simulate against a book snapshot
        bids = [(0.54, 500), (0.53, 300)]
        asks = [(0.55, 200), (0.56, 300)]
        fills = engine.simulate_fills(bids, asks)
    """

    def __init__(
        self,
        taker_fee_rate: float = 0.02,
        maker_fee_rate: float = 0.0,
        min_latency_ms: float = 50.0,
        max_latency_ms: float = 500.0,
        partial_fill_prob: float = 0.15,
        miss_prob: float = 0.05,
    ) -> None:
        self.taker_fee_rate = taker_fee_rate
        self.maker_fee_rate = maker_fee_rate
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.partial_fill_prob = partial_fill_prob
        self.miss_prob = miss_prob
        self._orders: dict[str, PaperOrder] = {}
        self._fill_count: int = 0
        self._total_volume: float = 0.0
        self._total_fees: float = 0.0

    def submit_order(
        self,
        order_id: str,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> PaperOrder:
        """Submit a new paper order."""
        order = PaperOrder(
            order_id=order_id,
            token_id=token_id,
            side=side.upper(),
            price=price,
            size=size,
        )
        self._orders[order_id] = order
        logger.debug(
            "Paper order submitted: %s %s %.0f @ %.3f [%s]",
            side,
            token_id[:8] if token_id else "?",
            size,
            price,
            order_id,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order."""
        order = self._orders.get(order_id)
        if order and order.status == "open":
            order.status = "cancelled"
            return True
        return False

    def get_order(self, order_id: str) -> PaperOrder | None:
        return self._orders.get(order_id)

    def simulate_fills(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> list[SimulatedFill]:
        """Simulate fills for all open orders against current book.

        Args:
            bids: [(price, size), ...] sorted descending by price.
            asks: [(price, size), ...] sorted ascending by price.

        Returns:
            List of fills generated this cycle.
        """
        fills: list[SimulatedFill] = []

        for order in list(self._orders.values()):
            if order.status != "open":
                continue

            remaining = order.size - order.filled_size
            if remaining <= 0:
                order.status = "filled"
                continue

            fill = self._try_fill(order, remaining, bids, asks)
            if fill:
                order.fills.append(fill)
                order.filled_size += fill.fill_size
                self._fill_count += 1
                self._total_volume += fill.fill_size
                self._total_fees += fill.fee_usdc

                if order.filled_size >= order.size:
                    order.status = "filled"

                fills.append(fill)

        return fills

    def _try_fill(
        self,
        order: PaperOrder,
        remaining: float,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> SimulatedFill | None:
        """Try to fill a single order against the book."""
        latency = random.uniform(self.min_latency_ms, self.max_latency_ms)  # noqa: S311

        # Random miss (simulates network/timing issues)
        if random.random() < self.miss_prob:  # noqa: S311
            return None

        if order.side == "BUY":
            levels = sorted(asks, key=lambda x: x[0])
            is_maker = order.price < levels[0][0] if levels else True
        else:
            levels = sorted(bids, key=lambda x: x[0], reverse=True)
            is_maker = order.price > levels[0][0] if levels else True

        if not levels:
            return None

        # Check if order price is marketable
        if order.side == "BUY":
            if order.price < levels[0][0]:
                # Limit order resting — may fill if price moves
                return None
        elif order.price > levels[0][0]:
            return None

        # Walk the book to determine fill
        fill_size = 0.0
        fill_cost = 0.0

        for level_price, level_size in levels:
            if order.side == "BUY" and level_price > order.price:
                break
            if order.side == "SELL" and level_price < order.price:
                break

            available = min(remaining - fill_size, level_size)
            fill_size += available
            fill_cost += available * level_price

            if fill_size >= remaining:
                break

        if fill_size <= 0:
            return None

        # Partial fill simulation
        if random.random() < self.partial_fill_prob and fill_size > 10:  # noqa: S311
            fill_size *= random.uniform(0.3, 0.8)  # noqa: S311
            fill_cost = fill_size * (fill_cost / max(fill_size, 0.01))

        vwap = fill_cost / fill_size if fill_size > 0 else order.price
        fee_rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        fee = fill_size * fee_rate

        return SimulatedFill(
            order_id=order.order_id,
            fill_type=FillType.FULL if fill_size >= remaining else FillType.PARTIAL,
            fill_price=round(vwap, 4),
            fill_size=round(fill_size, 2),
            remaining_size=round(remaining - fill_size, 2),
            fee_usdc=round(fee, 4),
            is_maker=is_maker,
            latency_ms=round(latency, 1),
        )

    @property
    def open_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.status == "open"]

    @property
    def stats(self) -> dict:
        return {
            "total_orders": len(self._orders),
            "open_orders": len(self.open_orders),
            "total_fills": self._fill_count,
            "total_volume": round(self._total_volume, 2),
            "total_fees": round(self._total_fees, 4),
        }

    def reset(self) -> None:
        """Clear all orders and stats."""
        self._orders.clear()
        self._fill_count = 0
        self._total_volume = 0.0
        self._total_fees = 0.0
