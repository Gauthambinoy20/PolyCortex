"""Prometheus metrics exporter for the trading agent."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, start_http_server

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False
    logger.debug("prometheus_client not installed; metrics disabled")


class TradingMetrics:
    """Prometheus metrics for the trading agent."""

    def __init__(self) -> None:
        if not _HAS_PROMETHEUS:
            self._enabled = False
            return
        self._enabled = True
        self.orders_submitted_total = Counter(
            "orders_submitted_total",
            "Total number of orders submitted",
            ["market_id", "side"],
        )
        self.orders_filled_total = Counter(
            "orders_filled_total",
            "Total number of orders filled",
            ["market_id", "side"],
        )
        self.orders_failed_total = Counter(
            "orders_failed_total",
            "Total number of order failures",
            ["market_id", "reason"],
        )
        self.api_errors_total = Counter(
            "api_errors_total",
            "Total API errors",
            ["client", "endpoint"],
        )
        self.current_drawdown_usdc = Gauge(
            "current_drawdown_usdc",
            "Current drawdown in USDC",
        )

    def start_server(self, port: int = 8080) -> None:
        """Start Prometheus HTTP server on given port."""
        if not self._enabled:
            logger.warning("prometheus_client not installed; metrics server not started")
            return
        try:
            start_http_server(port)
            logger.info("Prometheus metrics server started on port %d", port)
        except Exception as exc:
            logger.warning("Failed to start Prometheus metrics server: %s", exc)

    def inc_orders_submitted(self, market_id: str = "", side: str = "") -> None:
        if self._enabled:
            self.orders_submitted_total.labels(market_id=market_id, side=side).inc()

    def inc_orders_filled(self, market_id: str = "", side: str = "") -> None:
        if self._enabled:
            self.orders_filled_total.labels(market_id=market_id, side=side).inc()

    def inc_orders_failed(self, market_id: str = "", reason: str = "") -> None:
        if self._enabled:
            self.orders_failed_total.labels(market_id=market_id, reason=reason).inc()

    def inc_api_errors(self, client: str = "", endpoint: str = "") -> None:
        if self._enabled:
            self.api_errors_total.labels(client=client, endpoint=endpoint).inc()

    def set_drawdown(self, drawdown_usdc: float) -> None:
        if self._enabled:
            self.current_drawdown_usdc.set(drawdown_usdc)


_metrics: TradingMetrics | None = None


def get_metrics() -> TradingMetrics:
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics
