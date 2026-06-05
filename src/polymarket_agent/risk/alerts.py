"""Configurable alert system for market monitoring.

Supports:
- Price threshold alerts (price crosses above/below a target)
- Volume alerts (volume exceeds threshold)
- Custom condition alerts (arbitrary callable)

Alerts fire once, then deactivate unless set to repeating.
Integrates with EventBus for notification dispatch.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from polymarket_agent.types import Event, EventType

logger = logging.getLogger(__name__)


class AlertCondition(StrEnum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    VOLUME_ABOVE = "volume_above"
    SPREAD_ABOVE = "spread_above"


class AlertStatus(StrEnum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class Alert:
    """A single alert definition."""

    alert_id: str
    market_id: str
    condition: AlertCondition
    threshold: float
    status: AlertStatus = AlertStatus.ACTIVE
    repeating: bool = False
    cooldown_seconds: float = 300.0  # Min time between repeated triggers
    message: str = ""
    created_at: float = field(default_factory=time.monotonic)
    last_triggered_at: float | None = None
    trigger_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event(self, current_value: float) -> Event:
        return Event(
            event_type=EventType.ALERT_TRIGGERED,
            payload={
                "alert_id": self.alert_id,
                "market_id": self.market_id,
                "condition": self.condition,
                "threshold": self.threshold,
                "current_value": current_value,
                "message": self.message or f"{self.condition} {self.threshold} (current: {current_value:.4f})",
                "trigger_count": self.trigger_count,
            },
            source="alert_system",
        )


class AlertManager:
    """Manages alert rules and checks them against market data.

    Usage:
        mgr = AlertManager()
        mgr.add_alert(Alert(
            alert_id="btc-drop",
            market_id="0x123",
            condition=AlertCondition.PRICE_BELOW,
            threshold=0.40,
            message="Price dropped below 40%",
        ))

        # Check on each market update
        triggered = mgr.check("0x123", price=0.38, volume=5000)
        for alert, value in triggered:
            await event_bus.publish(alert.to_event(value))
    """

    def __init__(self) -> None:
        self._alerts: dict[str, Alert] = {}
        self._total_triggered: int = 0

    def add_alert(self, alert: Alert) -> None:
        """Register a new alert."""
        self._alerts[alert.alert_id] = alert
        logger.info(
            "Alert registered: %s — %s %s %.4f for market %s",
            alert.alert_id,
            alert.condition,
            "threshold",
            alert.threshold,
            alert.market_id,
        )

    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert. Returns True if found."""
        if alert_id in self._alerts:
            del self._alerts[alert_id]
            return True
        return False

    def cancel_alert(self, alert_id: str) -> bool:
        """Mark an alert as cancelled."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.CANCELLED
            return True
        return False

    def get_alert(self, alert_id: str) -> Alert | None:
        return self._alerts.get(alert_id)

    def get_active_alerts(self, market_id: str | None = None) -> list[Alert]:
        """Get all active alerts, optionally filtered by market."""
        alerts = [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]
        if market_id:
            alerts = [a for a in alerts if a.market_id == market_id]
        return alerts

    def check(
        self,
        market_id: str,
        price: float = 0.0,
        volume: float = 0.0,
        spread: float = 0.0,
    ) -> list[tuple[Alert, float]]:
        """Check all active alerts for this market.

        Returns:
            List of (alert, current_value) tuples for triggered alerts.
        """
        triggered: list[tuple[Alert, float]] = []
        now = time.monotonic()

        for alert in list(self._alerts.values()):
            if alert.status != AlertStatus.ACTIVE:
                continue
            if alert.market_id != market_id:
                continue

            # Cooldown check for repeating alerts
            if alert.last_triggered_at and alert.repeating and now - alert.last_triggered_at < alert.cooldown_seconds:
                continue

            fired = False
            current_value = 0.0

            if alert.condition == AlertCondition.PRICE_ABOVE and price > 0:
                if price >= alert.threshold:
                    fired = True
                    current_value = price

            elif alert.condition == AlertCondition.PRICE_BELOW and price > 0:
                if price <= alert.threshold:
                    fired = True
                    current_value = price

            elif alert.condition == AlertCondition.VOLUME_ABOVE and volume > 0:
                if volume >= alert.threshold:
                    fired = True
                    current_value = volume

            elif alert.condition == AlertCondition.SPREAD_ABOVE and spread > 0 and spread >= alert.threshold:
                fired = True
                current_value = spread

            if fired:
                alert.trigger_count += 1
                alert.last_triggered_at = now
                self._total_triggered += 1

                if not alert.repeating:
                    alert.status = AlertStatus.TRIGGERED

                triggered.append((alert, current_value))
                logger.info(
                    "ALERT TRIGGERED [%s]: %s (value=%.4f, threshold=%.4f)",
                    alert.alert_id,
                    alert.condition,
                    current_value,
                    alert.threshold,
                )

        return triggered

    @property
    def stats(self) -> dict:
        statuses: dict[str, int] = {}
        for a in self._alerts.values():
            statuses[a.status] = statuses.get(a.status, 0) + 1
        return {
            "total_alerts": len(self._alerts),
            "active": statuses.get(AlertStatus.ACTIVE, 0),
            "triggered": statuses.get(AlertStatus.TRIGGERED, 0),
            "total_trigger_count": self._total_triggered,
        }
