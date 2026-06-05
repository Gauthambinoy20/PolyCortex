"""Precise price arithmetic using Decimal.

Prediction market prices MUST be between 0.01 and 0.99.
Using float can cause subtle rounding errors:
    >>> 0.1 + 0.2
    0.30000000000000004

This module provides safe price operations with proper rounding.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# Polymarket price constraints
MIN_PRICE = Decimal("0.01")
MAX_PRICE = Decimal("0.99")
PRICE_TICK = Decimal("0.001")  # Minimum price increment
USDC_PRECISION = Decimal("0.01")  # 2 decimal places for USDC amounts


def to_price(value: float | str | Decimal) -> Decimal:
    """Convert a value to a Decimal price, quantized to tick size.

    Args:
        value: Price as float, string, or Decimal.

    Returns:
        Decimal price quantized to 0.001.

    Raises:
        ValueError: If price is not a valid number.
    """
    try:
        d = Decimal(str(value)).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid price value: {value!r}") from exc
    return d


def clamp_price(value: float | str | Decimal) -> Decimal:
    """Convert and clamp price to valid Polymarket range [0.01, 0.99].

    Args:
        value: Price value.

    Returns:
        Clamped Decimal price.
    """
    d = to_price(value)
    if d < MIN_PRICE:
        return MIN_PRICE
    if d > MAX_PRICE:
        return MAX_PRICE
    return d


def is_valid_price(value: float | str | Decimal) -> bool:
    """Check if a price is within valid Polymarket range."""
    try:
        d = to_price(value)
    except ValueError:
        return False
    return MIN_PRICE <= d <= MAX_PRICE


def complement_price(price: float | str | Decimal) -> Decimal:
    """Get the complementary price (YES price ↔ NO price).

    For a YES price of 0.65, the NO price is 0.35.
    """
    d = to_price(price)
    return (Decimal("1.0") - d).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)


def to_usdc(value: float | str | Decimal) -> Decimal:
    """Convert to USDC amount with 2 decimal places."""
    try:
        d = Decimal(str(value)).quantize(USDC_PRECISION, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid USDC amount: {value!r}") from exc
    return d


def price_to_float(price: Decimal) -> float:
    """Convert Decimal price to float for APIs that require float."""
    return float(price)


def spread(bid: float | str | Decimal, ask: float | str | Decimal) -> Decimal:
    """Calculate bid-ask spread."""
    return to_price(ask) - to_price(bid)


def midpoint(bid: float | str | Decimal, ask: float | str | Decimal) -> Decimal:
    """Calculate midpoint price between bid and ask."""
    b = to_price(bid)
    a = to_price(ask)
    return ((b + a) / 2).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)


def vwap(levels: list[tuple[float, float]]) -> Decimal:
    """Calculate volume-weighted average price from order book levels.

    Args:
        levels: List of (price, size) tuples.

    Returns:
        VWAP as Decimal, or Decimal("0") if empty.
    """
    if not levels:
        return Decimal("0")

    total_cost = Decimal("0")
    total_size = Decimal("0")

    for price, size in levels:
        p = Decimal(str(price))
        s = Decimal(str(size))
        total_cost += p * s
        total_size += s

    if total_size == 0:
        return Decimal("0")

    return (total_cost / total_size).quantize(PRICE_TICK, rounding=ROUND_HALF_UP)
