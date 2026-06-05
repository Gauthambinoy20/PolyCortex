"""Unified alert router — writes to JSONL and optionally to Telegram."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymarket_agent.utils.telegram import TelegramNotifier

logger = logging.getLogger(__name__)

# Severity ordering used for threshold filtering.
_SEVERITY_ORDER: dict[str, int] = {
    "info": 0,
    "trade": 1,
    "warning": 2,
    "critical": 3,
}


class AlertRouter:
    """Fan-out alerts to a local JSONL file and (optionally) Telegram.

    The JSONL sink is unconditional — every alert is persisted regardless of
    severity.  Telegram delivery respects a configurable severity threshold so
    low-priority alerts don't cause notification spam.

    No method ever raises; errors are logged and swallowed so the trading
    loop is never disrupted.
    """

    def __init__(
        self,
        alerts_path: Path,
        telegram: TelegramNotifier | None = None,
    ) -> None:
        self._alerts_path = alerts_path
        self._telegram = telegram
        self._telegram_threshold: str = "warning"

        # Ensure parent directory exists.
        self._alerts_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_telegram_threshold(self, level: str) -> None:
        """Set the minimum severity that triggers a Telegram notification.

        Args:
            level: ``"info"``, ``"warning"``, or ``"critical"``.
        """
        if level not in _SEVERITY_ORDER:
            logger.warning("Unknown severity %r — threshold unchanged", level)
            return
        self._telegram_threshold = level

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    async def alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
    ) -> None:
        """Route an alert to JSONL and (conditionally) Telegram.

        1. Always append to the JSONL file.
        2. Forward to Telegram when *severity* meets or exceeds the current
           threshold **or** severity is ``"trade"`` (always forwarded).
        3. Never raises.

        Args:
            alert_type: Short label, e.g. ``"TRADE"`` or ``"RISK"``.
            message: Human-readable alert body.
            severity: ``"info"``, ``"trade"``, ``"warning"``, or
                ``"critical"``.
        """
        ts = datetime.now(tz=UTC).isoformat()
        record = {
            "timestamp": ts,
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
        }

        # --- JSONL sink (synchronous, always) ---
        try:
            with self._alerts_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            logger.error("Failed to write alert to JSONL", exc_info=True)

        # --- Telegram sink (async, conditional) ---
        if self._telegram is not None and self._telegram.configured:
            sev_val = _SEVERITY_ORDER.get(severity, 0)
            threshold_val = _SEVERITY_ORDER.get(self._telegram_threshold, 2)
            if severity == "trade" or sev_val >= threshold_val:
                try:
                    await self._telegram.send_alert(alert_type, message, severity)
                except Exception:
                    logger.error("Telegram alert delivery failed", exc_info=True)
