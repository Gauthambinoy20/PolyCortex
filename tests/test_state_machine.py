"""Tests for order lifecycle state machine."""

import pytest

from polymarket_agent.execution.state_machine import InvalidTransitionError, OrderStateMachine
from polymarket_agent.types import OrderState


class TestOrderStateMachine:
    def test_initial_state(self):
        sm = OrderStateMachine(order_id="test")
        assert sm.state == OrderState.CREATED

    def test_valid_transition_path(self):
        sm = OrderStateMachine(order_id="test")
        sm.transition(OrderState.VALIDATED)
        sm.transition(OrderState.SIGNED)
        sm.transition(OrderState.SUBMITTED)
        sm.transition(OrderState.ACKNOWLEDGED)
        sm.transition(OrderState.FILLED)
        assert sm.state == OrderState.FILLED

    def test_invalid_transition_raises(self):
        sm = OrderStateMachine(order_id="test")
        with pytest.raises(InvalidTransitionError):
            sm.transition(OrderState.FILLED)

    def test_history(self):
        sm = OrderStateMachine(order_id="test")
        sm.transition(OrderState.VALIDATED)
        sm.transition(OrderState.SIGNED)
        history = sm.history
        assert len(history) == 2
        assert history[0].to_state == OrderState.VALIDATED
        assert history[1].to_state == OrderState.SIGNED

    def test_serialization_roundtrip(self):
        sm = OrderStateMachine(order_id="test")
        sm.transition(OrderState.VALIDATED)
        sm.transition(OrderState.SIGNED)
        data = sm.to_dict()
        restored = OrderStateMachine.from_dict(data)
        assert restored.state == OrderState.SIGNED
        assert restored.order_id == "test"
        assert len(restored.history) == 2

    def test_force_state(self):
        sm = OrderStateMachine(order_id="test")
        sm.force_state(OrderState.FILLED, reason="manual recovery")
        assert sm.state == OrderState.FILLED

    def test_terminal_state_no_transition(self):
        sm = OrderStateMachine(order_id="t1")
        sm.force_state(OrderState.REJECTED)
        with pytest.raises(InvalidTransitionError):
            sm.transition(OrderState.VALIDATED)

    def test_is_terminal(self):
        sm = OrderStateMachine(order_id="t1")
        assert not sm.is_terminal
        sm.force_state(OrderState.EXPIRED)
        assert sm.is_terminal

    def test_can_transition(self):
        sm = OrderStateMachine(order_id="t1")
        assert sm.can_transition(OrderState.VALIDATED)
        assert not sm.can_transition(OrderState.FILLED)

    def test_to_dict_structure(self):
        sm = OrderStateMachine(order_id="o1")
        sm.transition(OrderState.VALIDATED)
        d = sm.to_dict()
        assert d["order_id"] == "o1"
        assert d["state"] == "validated"
        assert len(d["history"]) == 1
        assert d["history"][0]["from"] == "created"
        assert d["history"][0]["to"] == "validated"

    def test_time_in_state(self):
        sm = OrderStateMachine(order_id="t1")
        assert sm.time_in_state >= 0
