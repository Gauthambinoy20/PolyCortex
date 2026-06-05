"""Retry with exponential backoff and circuit breaker.

Wrap any async function with retry logic:
    @with_retry(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def call_api(): ...

Or use the circuit breaker directly:
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
    if breaker.allow_request():
        try:
            result = await call_api()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(StrEnum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing — reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Simple circuit breaker to avoid hammering a degraded service.

    States:
        CLOSED: all requests pass. After `failure_threshold` consecutive
                failures, transitions to OPEN.
        OPEN: all requests rejected for `reset_timeout` seconds,
              then transitions to HALF_OPEN.
        HALF_OPEN: allows one request. On success → CLOSED, on failure → OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        name: str = "default",
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count: int = 0
        self._total_rejected: int = 0

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker '%s' → HALF_OPEN (testing)", self.name)
        return self._state

    @property
    def stats(self) -> dict:
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_rejected": self._total_rejected,
        }

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return True
        self._total_rejected += 1
        return False

    def record_success(self) -> None:
        """Record a successful request."""
        self._success_count += 1
        if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
            logger.info("Circuit breaker '%s' → CLOSED (recovered)", self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            if self._state != CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker '%s' → OPEN after %d failures",
                    self.name,
                    self._failure_count,
                )
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"Retry exhausted after {attempts} attempts: {last_error}")


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    circuit_breaker: CircuitBreaker | None = None,
) -> Callable[[F], F]:
    """Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds (doubles each retry).
        max_delay: Maximum delay cap in seconds.
        jitter: Add random jitter to delay to avoid thundering herd.
        retry_on: Tuple of exception types to retry on.
        circuit_breaker: Optional CircuitBreaker instance.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None

            for attempt in range(max_retries + 1):
                # Circuit breaker check
                if circuit_breaker and not circuit_breaker.allow_request():
                    raise RetryExhausted(
                        attempt,
                        RuntimeError(f"Circuit breaker '{circuit_breaker.name}' is OPEN"),
                    )

                try:
                    result = await func(*args, **kwargs)
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    return result
                except retry_on as exc:
                    last_exc = exc
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    if attempt >= max_retries:
                        break

                    delay = min(base_delay * (2**attempt), max_delay)
                    if jitter:
                        delay *= 0.5 + random.random()  # noqa: S311

                    logger.warning(
                        "Retry %d/%d for %s after %.1fs: %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)

            raise RetryExhausted(max_retries + 1, last_exc)  # type: ignore[arg-type]

        return wrapper  # type: ignore[return-value]

    return decorator
