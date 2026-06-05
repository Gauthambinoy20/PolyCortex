"""Shared type definitions used across all PolymarketTrader modules.

This module contains enums, dataclasses, and constants that form the
common vocabulary for the agent's domain model.  It is intentionally
free of business logic — only simple computed properties are included.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "OrderState",
    "VALID_TRANSITIONS",
    "EventType",
    "Event",
    "MarketData",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "TradeSignal",
    "RiskCheckResult",
]


# ---------------------------------------------------------------------------
# Order lifecycle
# ---------------------------------------------------------------------------


class OrderState(StrEnum):
    """Lifecycle states an order passes through from creation to settlement."""

    CREATED = "created"
    VALIDATED = "validated"
    SIGNED = "signed"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIAL_FILL = "partial_fill"
    FILLED = "filled"
    SETTLED = "settled"
    REDEEMED = "redeemed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.CANCELLED},
    OrderState.VALIDATED: {OrderState.SIGNED, OrderState.REJECTED},
    OrderState.SIGNED: {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {
        OrderState.ACKNOWLEDGED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
    },
    OrderState.ACKNOWLEDGED: {
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL_FILL: {
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
    },
    OrderState.FILLED: {OrderState.SETTLED},
    OrderState.SETTLED: {OrderState.REDEEMED},
    # Terminal states
    OrderState.REDEEMED: set(),
    OrderState.REJECTED: set(),
    OrderState.CANCELLED: set(),
    OrderState.EXPIRED: set(),
}
"""Maps each ``OrderState`` to the set of states it may legally transition to."""


# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------


class EventType(StrEnum):
    """Canonical event types flowing through the agent's event bus."""

    MARKET_SCANNED = "market.scanned"
    ORDERBOOK_UPDATED = "orderbook.updated"
    EDGE_DETECTED = "edge.detected"
    TRADE_SIGNAL = "trade.signal"
    ORDER_PLACED = "order.placed"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    TRADE_CLOSED = "trade.closed"
    POSITION_UPDATED = "position.updated"
    DRAWDOWN_ALERT = "drawdown.alert"
    KILL_SWITCH_ACTIVATED = "kill_switch.activated"
    ANOMALY_DETECTED = "anomaly.detected"
    SENTIMENT_UPDATED = "sentiment.updated"
    REGIME_CHANGED = "regime.changed"
    CYCLE_STARTED = "cycle.started"
    CYCLE_COMPLETED = "cycle.completed"
    HEARTBEAT = "heartbeat"
    ALERT_TRIGGERED = "alert.triggered"


@dataclass(frozen=True)
class Event:
    """An immutable event dispatched on the agent's event bus.

    Attributes:
        event_type: Categorises the event.
        payload: Arbitrary data carried by the event.
        event_id: Short unique identifier (12-char hex string).
        timestamp: UTC time the event was created.
        source: Name of the component that emitted the event.
    """

    event_type: EventType
    payload: dict
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "agent"


# ---------------------------------------------------------------------------
# Market & order-book data
# ---------------------------------------------------------------------------


@dataclass
class MarketData:
    """Snapshot of a Polymarket prediction-market listing.

    Attributes:
        condition_id: On-chain condition identifier.
        question: Human-readable market question.
        category: Market category (e.g. "politics", "sports").
        clob_token_ids: Token IDs for each outcome on the CLOB.
        end_date: Market resolution date, if known.
        volume_24h: Rolling 24-hour USDC volume.
        liquidity: Current liquidity in USDC.
        active: Whether the market is open for trading.
    """

    condition_id: str
    question: str
    category: str
    clob_token_ids: list[str]
    end_date: datetime | None = None
    volume_24h: float = 0.0
    liquidity: float = 0.0
    active: bool = True


@dataclass
class OrderBookLevel:
    """A single price/size level on one side of an order book.

    Attributes:
        price: Price of the level (0.0–1.0 for binary markets).
        size: Total size available at this price.
    """

    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    """Point-in-time snapshot of an order book for a single token.

    Attributes:
        token_id: The CLOB token this snapshot describes.
        bids: Bid levels sorted best (highest) first.
        asks: Ask levels sorted best (lowest) first.
        timestamp: UTC time the snapshot was captured.
    """

    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def best_bid(self) -> float:
        """Highest bid price, or ``0.0`` if the book is empty."""
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Lowest ask price, or ``1.0`` if the book is empty."""
        return self.asks[0].price if self.asks else 1.0

    @property
    def midpoint(self) -> float:
        """Mid-price between best bid and best ask."""
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        """Absolute spread between best ask and best bid."""
        return self.best_ask - self.best_bid


# ---------------------------------------------------------------------------
# Trading signals & risk
# ---------------------------------------------------------------------------


@dataclass
class TradeSignal:
    """A directional trade recommendation produced by the signal pipeline.

    Attributes:
        market_id: Condition ID of the target market.
        question: Human-readable market question.
        direction: ``"YES"`` or ``"NO"``.
        edge: Estimated edge (estimated_prob − market_price).
        confidence: Composite confidence score (0.0–1.0).
        estimated_prob: Model's probability estimate for the outcome.
        market_price: Current market-implied probability.
        regime: Current market regime label.
        signal_breakdown: Per-signal-source contribution scores.
        category: Market category.
    """

    market_id: str
    question: str
    direction: str
    edge: float
    confidence: float
    estimated_prob: float
    market_price: float
    regime: str
    signal_breakdown: dict[str, float]
    category: str = "unknown"


@dataclass
class RiskCheckResult:
    """Outcome of a pre-trade risk evaluation.

    Attributes:
        approved: Whether the trade passed all risk checks.
        position_size_usdc: Approved position size in USDC.
        reason: Human-readable rejection/approval reason.
        constraints_applied: List of constraint names that were evaluated.
        kelly_raw: Raw Kelly criterion fraction.
        kelly_adjusted: Kelly fraction after adjustment (e.g. half-Kelly).
    """

    approved: bool
    position_size_usdc: float
    reason: str = ""
    constraints_applied: list[str] = field(default_factory=list)
    kelly_raw: float = 0.0
    kelly_adjusted: float = 0.0
