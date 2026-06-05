"""TWAP (Time-Weighted Average Price) order execution.

Splits a large order into smaller child orders placed at regular intervals.
This minimizes market impact by spreading execution over time.

Usage:
    twap = TWAPOrder(
        order_id="twap-1", token_id="0xabc", side="BUY",
        total_size=1000, price_limit=0.55,
        num_slices=10, interval_seconds=30,
    )

    # In your execution loop:
    while not twap.is_complete:
        child = twap.next_slice()
        if child:
            await execute(child)
            twap.record_fill(child.slice_id, fill_price=0.54, fill_size=100)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TWAPSlice:
    """A single child order in a TWAP execution."""

    slice_id: str
    slice_number: int
    token_id: str
    side: str
    size: float
    price_limit: float
    status: str = "pending"  # pending, submitted, filled, failed
    fill_price: float | None = None
    fill_size: float = 0.0
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class TWAPOrder:
    """TWAP order that splits execution across time intervals.

    Args:
        order_id: Unique identifier for this TWAP order.
        token_id: Market token to trade.
        side: 'BUY' or 'SELL'.
        total_size: Total order size in USDC.
        price_limit: Maximum (BUY) or minimum (SELL) price.
        num_slices: Number of child orders to create.
        interval_seconds: Time between child order placements.
        randomize_size: If True, randomize slice sizes ±20% to avoid detection.
    """

    order_id: str
    token_id: str
    side: str
    total_size: float
    price_limit: float
    num_slices: int = 10
    interval_seconds: float = 30.0
    randomize_size: bool = False

    # Internal state
    _slices: list[TWAPSlice] = field(default_factory=list, repr=False)
    _next_slice_idx: int = field(default=0, repr=False)
    _last_slice_time: float = field(default=0.0, repr=False)
    _created_at: float = field(default_factory=time.monotonic, repr=False)
    _status: str = field(default="active", repr=False)

    def __post_init__(self) -> None:
        if not self._slices:
            self._build_slices()

    def _build_slices(self) -> None:
        """Pre-compute slice sizes."""
        import random as _rand

        base_size = self.total_size / self.num_slices

        for i in range(self.num_slices):
            if self.randomize_size:
                # ±20% randomization
                factor = 0.8 + _rand.random() * 0.4  # noqa: S311
                size = base_size * factor
            else:
                size = base_size

            self._slices.append(
                TWAPSlice(
                    slice_id=f"{self.order_id}-s{i}",
                    slice_number=i,
                    token_id=self.token_id,
                    side=self.side,
                    size=round(size, 2),
                    price_limit=self.price_limit,
                )
            )

        # Adjust last slice to match total exactly
        current_total = sum(s.size for s in self._slices)
        diff = self.total_size - current_total
        if self._slices:
            self._slices[-1].size = round(self._slices[-1].size + diff, 2)

    @property
    def status(self) -> str:
        if self._status == "cancelled":
            return "cancelled"
        if all(s.status == "filled" for s in self._slices):
            return "completed"
        if any(s.status in ("submitted", "filled") for s in self._slices):
            return "in_progress"
        return "active"

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "cancelled")

    @property
    def progress(self) -> dict:
        filled = sum(1 for s in self._slices if s.status == "filled")
        submitted = sum(1 for s in self._slices if s.status == "submitted")
        total_filled_size = sum(s.fill_size for s in self._slices)
        avg_fill = (
            sum(s.fill_price * s.fill_size for s in self._slices if s.fill_price) / total_filled_size
            if total_filled_size > 0
            else 0.0
        )
        return {
            "order_id": self.order_id,
            "status": self.status,
            "slices_filled": filled,
            "slices_submitted": submitted,
            "slices_total": self.num_slices,
            "size_filled": round(total_filled_size, 2),
            "size_total": self.total_size,
            "avg_fill_price": round(avg_fill, 4),
            "pct_complete": round(filled / self.num_slices * 100, 1),
            "elapsed_seconds": round(time.monotonic() - self._created_at, 1),
        }

    def next_slice(self) -> TWAPSlice | None:
        """Get the next slice if interval has elapsed.

        Returns:
            Next TWAPSlice to execute, or None if not time yet / complete.
        """
        if self.is_complete:
            return None

        if self._next_slice_idx >= len(self._slices):
            return None

        now = time.monotonic()
        if self._last_slice_time > 0 and (now - self._last_slice_time) < self.interval_seconds:
            return None

        slice_ = self._slices[self._next_slice_idx]
        slice_.status = "submitted"
        self._next_slice_idx += 1
        self._last_slice_time = now

        logger.info(
            "TWAP %s: slice %d/%d — %s %.0f @ limit %.3f",
            self.order_id,
            slice_.slice_number + 1,
            self.num_slices,
            self.side,
            slice_.size,
            self.price_limit,
        )

        return slice_

    def record_fill(self, slice_id: str, fill_price: float, fill_size: float) -> bool:
        """Record a fill for a child order.

        Returns:
            True if the slice was found and updated.
        """
        for s in self._slices:
            if s.slice_id == slice_id:
                s.status = "filled"
                s.fill_price = fill_price
                s.fill_size = fill_size
                logger.info(
                    "TWAP %s: slice %s filled %.0f @ %.4f",
                    self.order_id,
                    slice_id,
                    fill_size,
                    fill_price,
                )
                return True
        return False

    def record_failure(self, slice_id: str) -> bool:
        """Record a failed slice."""
        for s in self._slices:
            if s.slice_id == slice_id:
                s.status = "failed"
                return True
        return False

    def cancel(self) -> None:
        """Cancel the TWAP order. Pending slices are cancelled."""
        self._status = "cancelled"
        for s in self._slices:
            if s.status in ("pending", "submitted"):
                s.status = "failed"
        logger.info("TWAP %s cancelled", self.order_id)

    @property
    def slices(self) -> list[TWAPSlice]:
        return list(self._slices)
