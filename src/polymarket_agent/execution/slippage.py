"""Slippage protection: estimate fill price from order book depth before placing orders."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Result of slippage analysis for a proposed order."""

    estimated_vwap: float
    midpoint: float
    slippage_bps: float
    total_depth_usdc: float
    levels_consumed: int
    sufficient_liquidity: bool

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10_000


def estimate_slippage(
    order_size_usdc: float,
    side: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    max_slippage_bps: float = 100.0,
) -> SlippageEstimate:
    """Estimate fill price by walking the order book.

    For BUY orders, we consume the ask side (ascending price).
    For SELL orders, we consume the bid side (descending price).

    Args:
        order_size_usdc: Total order size in USDC.
        side: 'BUY' or 'SELL'.
        bids: List of (price, size_usdc) tuples, sorted descending by price.
        asks: List of (price, size_usdc) tuples, sorted ascending by price.
        max_slippage_bps: Maximum acceptable slippage in basis points.

    Returns:
        SlippageEstimate with VWAP, slippage, and liquidity info.
    """
    if not bids or not asks:
        return SlippageEstimate(
            estimated_vwap=0.0,
            midpoint=0.0,
            slippage_bps=float("inf"),
            total_depth_usdc=0.0,
            levels_consumed=0,
            sufficient_liquidity=False,
        )

    # Determine midpoint
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    midpoint = (best_bid + best_ask) / 2.0

    # Select the side to consume
    if side.upper() == "BUY":
        # Consume asks (ascending price — already sorted)
        levels = sorted(asks, key=lambda x: x[0])
    else:
        # Consume bids (descending price — already sorted)
        levels = sorted(bids, key=lambda x: x[0], reverse=True)

    # Walk the book
    remaining = order_size_usdc
    total_cost = 0.0
    total_filled = 0.0
    levels_consumed = 0
    total_depth_usdc = sum(p * s for p, s in levels)

    for price, size in levels:
        if remaining <= 0:
            break
        levels_consumed += 1
        # size is in token units, cost in USDC = price * tokens
        # But for prediction markets, size is typically in USDC
        fill_amount = min(remaining, size)
        total_cost += fill_amount * price
        total_filled += fill_amount
        remaining -= fill_amount

    sufficient = remaining <= 0

    if total_filled <= 0:
        return SlippageEstimate(
            estimated_vwap=0.0,
            midpoint=midpoint,
            slippage_bps=float("inf"),
            total_depth_usdc=total_depth_usdc,
            levels_consumed=0,
            sufficient_liquidity=False,
        )

    vwap = total_cost / total_filled

    # Slippage relative to midpoint
    if side.upper() == "BUY":
        slippage = (vwap - midpoint) / midpoint if midpoint > 0 else 0
    else:
        slippage = (midpoint - vwap) / midpoint if midpoint > 0 else 0

    slippage_bps = slippage * 10_000

    if slippage_bps > max_slippage_bps:
        logger.warning(
            "Slippage %.1f bps exceeds max %.1f bps for $%.2f %s order (VWAP=%.4f, mid=%.4f, %d levels consumed)",
            slippage_bps,
            max_slippage_bps,
            order_size_usdc,
            side,
            vwap,
            midpoint,
            levels_consumed,
        )

    return SlippageEstimate(
        estimated_vwap=vwap,
        midpoint=midpoint,
        slippage_bps=round(slippage_bps, 2),
        total_depth_usdc=round(total_depth_usdc, 2),
        levels_consumed=levels_consumed,
        sufficient_liquidity=sufficient,
    )


def check_slippage_ok(
    order_size_usdc: float,
    side: str,
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    max_slippage_bps: float = 100.0,
) -> tuple[bool, SlippageEstimate]:
    """Convenience: returns (is_ok, estimate). Use before placing market orders."""
    est = estimate_slippage(order_size_usdc, side, bids, asks, max_slippage_bps)
    ok = est.sufficient_liquidity and est.slippage_bps <= max_slippage_bps
    return ok, est
