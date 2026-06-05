"""Trailing stop order: dynamic stop-loss that follows price movement.

A trailing stop moves the stop price upward as the market price increases,
but never moves it downward. This locks in profits while allowing further gains.

Usage:
    stop = TrailingStop(
        order_id="ts-1", token_id="0xabc", side="SELL",
        initial_price=0.60, trail_amount=0.05,
    )

    # Feed price updates
    stop.update_price(0.65)  # stop moves to 0.60
    stop.update_price(0.70)  # stop moves to 0.65
    stop.update_price(0.63)  # stop stays at 0.65 → TRIGGERED!
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TrailingStopMode:
    AMOUNT = "amount"  # Trail by fixed amount (e.g., 0.05)
    PERCENT = "percent"  # Trail by percentage (e.g., 5%)


@dataclass
class TrailingStop:
    """A trailing stop that adjusts with price movement.

    For long positions (SELL stop): trails below the high water mark.
    For short positions (BUY stop): trails above the low water mark.
    """

    order_id: str
    token_id: str
    side: str  # 'SELL' for longs (stop sells), 'BUY' for shorts
    initial_price: float
    trail_amount: float = 0.05
    trail_mode: str = TrailingStopMode.AMOUNT

    # State
    _high_water: float = field(default=0.0, repr=False)
    _low_water: float = field(default=float("inf"), repr=False)
    _stop_price: float = field(default=0.0, repr=False)
    _triggered: bool = field(default=False, repr=False)
    _trigger_price: float | None = field(default=None, repr=False)
    _created_at: float = field(default_factory=time.monotonic, repr=False)
    _update_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.side.upper() == "SELL":
            self._high_water = self.initial_price
            self._stop_price = self._calculate_stop_sell(self.initial_price)
        else:
            self._low_water = self.initial_price
            self._stop_price = self._calculate_stop_buy(self.initial_price)

    def _calculate_stop_sell(self, price: float) -> float:
        """Calculate stop price for a SELL trailing stop (long protection)."""
        if self.trail_mode == TrailingStopMode.PERCENT:
            return round(price * (1 - self.trail_amount / 100), 4)
        return round(price - self.trail_amount, 4)

    def _calculate_stop_buy(self, price: float) -> float:
        """Calculate stop price for a BUY trailing stop (short protection)."""
        if self.trail_mode == TrailingStopMode.PERCENT:
            return round(price * (1 + self.trail_amount / 100), 4)
        return round(price + self.trail_amount, 4)

    @property
    def stop_price(self) -> float:
        return self._stop_price

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def high_water(self) -> float:
        return self._high_water

    @property
    def low_water(self) -> float:
        return self._low_water

    def update_price(self, current_price: float) -> bool:
        """Update with latest market price.

        Returns:
            True if the stop was triggered.
        """
        if self._triggered:
            return True

        self._update_count += 1

        if self.side.upper() == "SELL":
            return self._update_sell(current_price)
        return self._update_buy(current_price)

    def _update_sell(self, price: float) -> bool:
        """Update for SELL trailing stop (long position protection)."""
        # New high → move stop up
        if price > self._high_water:
            self._high_water = price
            new_stop = self._calculate_stop_sell(price)
            if new_stop > self._stop_price:
                old = self._stop_price
                self._stop_price = new_stop
                logger.debug(
                    "Trailing stop %s: price %.4f → new high, stop %.4f → %.4f",
                    self.order_id,
                    price,
                    old,
                    new_stop,
                )

        # Check trigger: price dropped to or below stop
        if price <= self._stop_price:
            self._triggered = True
            self._trigger_price = price
            logger.info(
                "TRAILING STOP TRIGGERED [%s]: price %.4f <= stop %.4f (high=%.4f)",
                self.order_id,
                price,
                self._stop_price,
                self._high_water,
            )
            return True

        return False

    def _update_buy(self, price: float) -> bool:
        """Update for BUY trailing stop (short position protection)."""
        # New low → move stop down
        if price < self._low_water:
            self._low_water = price
            new_stop = self._calculate_stop_buy(price)
            if new_stop < self._stop_price:
                old = self._stop_price
                self._stop_price = new_stop
                logger.debug(
                    "Trailing stop %s: price %.4f → new low, stop %.4f → %.4f",
                    self.order_id,
                    price,
                    old,
                    new_stop,
                )

        # Check trigger: price rose to or above stop
        if price >= self._stop_price:
            self._triggered = True
            self._trigger_price = price
            logger.info(
                "TRAILING STOP TRIGGERED [%s]: price %.4f >= stop %.4f (low=%.4f)",
                self.order_id,
                price,
                self._stop_price,
                self._low_water,
            )
            return True

        return False

    @property
    def status(self) -> dict:
        return {
            "order_id": self.order_id,
            "side": self.side,
            "triggered": self._triggered,
            "stop_price": self._stop_price,
            "high_water": self._high_water,
            "low_water": self._low_water if self._low_water != float("inf") else None,
            "trigger_price": self._trigger_price,
            "trail_amount": self.trail_amount,
            "trail_mode": self.trail_mode,
            "updates": self._update_count,
        }

    def reset(self, new_price: float) -> None:
        """Reset the trailing stop with a new reference price."""
        self._triggered = False
        self._trigger_price = None
        self._update_count = 0
        if self.side.upper() == "SELL":
            self._high_water = new_price
            self._stop_price = self._calculate_stop_sell(new_price)
        else:
            self._low_water = new_price
            self._stop_price = self._calculate_stop_buy(new_price)
