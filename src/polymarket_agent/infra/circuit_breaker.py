"""Circuit breaker with CLOSED / OPEN / HALF_OPEN states."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable
from enum import Enum
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted on an OPEN circuit."""


class CircuitBreaker:
    """Async-friendly circuit breaker.

    Args:
        failure_threshold: Consecutive failures required to trip the circuit.
        recovery_timeout: Seconds to wait in OPEN before transitioning to
            HALF_OPEN.
        success_threshold: Number of successful HALF_OPEN calls required to
            close the circuit.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> State:
        if self._state == State.OPEN and time.monotonic() - self._last_failure_time >= self.recovery_timeout:
            self._state = State.HALF_OPEN
            self._success_count = 0
            logger.info("Circuit breaker transitioned to HALF_OPEN")
        return self._state

    def record_success(self) -> None:
        if self.state == State.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = State.CLOSED
                self._failure_count = 0
                logger.info("Circuit breaker CLOSED after recovery")
        elif self.state == State.CLOSED:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            if self._state != State.OPEN:
                logger.warning("Circuit breaker OPEN after %d failures", self._failure_count)
            self._state = State.OPEN

    async def call(self, coro: Awaitable[T]) -> T:
        if self.state == State.OPEN:
            raise CircuitBreakerOpen("Circuit breaker is OPEN")
        try:
            result = await coro
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def reset(self) -> None:
        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
