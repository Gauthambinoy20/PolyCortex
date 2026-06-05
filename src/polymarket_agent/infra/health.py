"""Health check for the trading agent.

Provides a health status report including:
- System uptime
- API connectivity status
- Open orders count
- Current P&L
- Memory usage
- Component status

Can be used standalone or integrated into a lightweight HTTP server.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float | None = None
    last_checked: float = field(default_factory=time.monotonic)


class HealthChecker:
    """Aggregates health from multiple components.

    Usage:
        health = HealthChecker(service_name="polymarket-trader")
        health.register_check("clob_api", check_clob_connectivity)
        health.register_check("database", check_db_connection)

        report = await health.check_all()
        print(report)
    """

    def __init__(self, service_name: str = "polymarket-trader") -> None:
        self.service_name = service_name
        self._start_time = time.monotonic()
        self._checks: dict[str, Callable[..., Any]] = {}
        self._last_results: dict[str, ComponentHealth] = {}
        self._metadata: dict[str, str | float | int] = {}

    @property
    def uptime_seconds(self) -> float:
        return round(time.monotonic() - self._start_time, 1)

    def set_metadata(self, key: str, value: str | float | int) -> None:
        """Set a metadata value included in health reports."""
        self._metadata[key] = value

    def register_check(self, name: str, check_fn) -> None:
        """Register a health check function.

        The function should be an async callable that returns a ComponentHealth.
        If it raises, the component is marked unhealthy.
        """
        self._checks[name] = check_fn
        logger.debug("Registered health check: %s", name)

    async def check_all(self) -> dict:
        """Run all health checks and return a report."""
        import asyncio

        results: dict[str, ComponentHealth] = {}

        for name, check_fn in self._checks.items():
            start = time.monotonic()
            try:
                if asyncio.iscoroutinefunction(check_fn):
                    result = await asyncio.wait_for(check_fn(), timeout=10.0)
                else:
                    result = check_fn()

                if isinstance(result, ComponentHealth):
                    result.latency_ms = round((time.monotonic() - start) * 1000, 1)
                    results[name] = result
                else:
                    results[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                        latency_ms=round((time.monotonic() - start) * 1000, 1),
                    )
            except TimeoutError:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Health check timed out (>10s)",
                )
            except Exception as exc:
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(exc),
                )

        self._last_results = results

        # Determine overall status
        statuses = [r.status for r in results.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses) or not statuses:
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        # Build report
        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = round(process.memory_info().rss / 1024 / 1024, 1)
        except (ImportError, Exception):
            memory_mb = None

        return {
            "status": overall,
            "service": self.service_name,
            "uptime_seconds": self.uptime_seconds,
            "memory_mb": memory_mb,
            "components": {
                name: {
                    "status": r.status,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                }
                for name, r in results.items()
            },
            "metadata": dict(self._metadata),
        }

    def quick_status(self) -> dict:
        """Return cached status without re-running checks."""
        if not self._last_results:
            return {"status": HealthStatus.HEALTHY, "message": "No checks run yet"}

        statuses = [r.status for r in self._last_results.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        else:
            overall = HealthStatus.DEGRADED

        return {
            "status": overall,
            "uptime_seconds": self.uptime_seconds,
            "components_checked": len(self._last_results),
        }
