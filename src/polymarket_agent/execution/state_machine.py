"""Order lifecycle state machine with transition validation and audit logging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from polymarket_agent.types import VALID_TRANSITIONS, OrderState

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Record of a single state transition."""

    from_state: OrderState
    to_state: OrderState
    timestamp: datetime
    reason: str = ""


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: OrderState, target: OrderState) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"Invalid transition: {current.value} → {target.value}. "
            f"Valid targets: {[s.value for s in VALID_TRANSITIONS.get(current, set())]}"
        )


class OrderStateMachine:
    """Manages the lifecycle of a single order with validated transitions.

    Enforces that only valid state transitions occur, logs every transition
    with timestamp and optional reason, and provides a complete audit trail.

    Usage:
        sm = OrderStateMachine(order_id="order-123")
        sm.transition(OrderState.VALIDATED, reason="price check passed")
        sm.transition(OrderState.SIGNED, reason="signature generated")
        sm.transition(OrderState.SUBMITTED, reason="sent to CLOB")

        print(sm.state)          # OrderState.SUBMITTED
        print(sm.history)        # list of StateTransition objects
        print(sm.is_terminal)    # False

        # Invalid transition raises:
        sm.transition(OrderState.REDEEMED)  # raises InvalidTransitionError
    """

    TERMINAL_STATES = {
        OrderState.REDEEMED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    }

    def __init__(
        self,
        order_id: str,
        initial_state: OrderState = OrderState.CREATED,
    ) -> None:
        self._order_id = order_id
        self._state = initial_state
        self._history: list[StateTransition] = []
        self._created_at = datetime.now(UTC)

    @property
    def order_id(self) -> str:
        return self._order_id

    @property
    def state(self) -> OrderState:
        return self._state

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._state in self.TERMINAL_STATES

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def last_transition_at(self) -> datetime | None:
        return self._history[-1].timestamp if self._history else None

    @property
    def time_in_state(self) -> float:
        """Seconds since last transition (or creation if no transitions)."""
        ref = self.last_transition_at or self._created_at
        return (datetime.now(UTC) - ref).total_seconds()

    def can_transition(self, target: OrderState) -> bool:
        """Check if a transition to target state is valid without performing it."""
        valid_targets = VALID_TRANSITIONS.get(self._state, set())
        return target in valid_targets

    def transition(self, target: OrderState, *, reason: str = "") -> None:
        """Transition to a new state. Raises InvalidTransitionError if invalid."""
        if self.is_terminal:
            raise InvalidTransitionError(self._state, target)

        if not self.can_transition(target):
            raise InvalidTransitionError(self._state, target)

        prev = self._state
        now = datetime.now(UTC)

        self._history.append(
            StateTransition(
                from_state=prev,
                to_state=target,
                timestamp=now,
                reason=reason,
            )
        )
        self._state = target

        logger.info(
            "Order %s: %s → %s%s",
            self._order_id,
            prev.value,
            target.value,
            f" ({reason})" if reason else "",
        )

    def force_state(self, target: OrderState, *, reason: str = "forced") -> None:
        """Force a state transition bypassing validation (for recovery/admin use).

        This logs a warning and should only be used for error recovery.
        """
        prev = self._state
        now = datetime.now(UTC)

        logger.warning(
            "FORCED Order %s: %s → %s (%s)",
            self._order_id,
            prev.value,
            target.value,
            reason,
        )

        self._history.append(
            StateTransition(
                from_state=prev,
                to_state=target,
                timestamp=now,
                reason=f"FORCED: {reason}",
            )
        )
        self._state = target

    def to_dict(self) -> dict:
        """Serialize state machine to dict for persistence."""
        return {
            "order_id": self._order_id,
            "state": self._state.value,
            "created_at": self._created_at.isoformat(),
            "history": [
                {
                    "from": t.from_state.value,
                    "to": t.to_state.value,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                }
                for t in self._history
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> OrderStateMachine:
        """Deserialize state machine from dict."""
        sm = cls(
            order_id=data["order_id"],
            initial_state=OrderState(data.get("state", "created")),
        )
        sm._created_at = datetime.fromisoformat(data["created_at"])
        sm._history = [
            StateTransition(
                from_state=OrderState(t["from"]),
                to_state=OrderState(t["to"]),
                timestamp=datetime.fromisoformat(t["timestamp"]),
                reason=t.get("reason", ""),
            )
            for t in data.get("history", [])
        ]
        return sm

    def __repr__(self) -> str:
        return f"OrderStateMachine(id={self._order_id!r}, state={self._state.value}, transitions={len(self._history)})"
