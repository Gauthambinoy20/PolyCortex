"""Async Telegram alert sender for production monitoring."""

import logging
import os
from datetime import UTC, datetime

import aiohttp

logger = logging.getLogger(__name__)

_EMOJI = {
    "critical": "\U0001f6a8",
    "warning": "\u26a0\ufe0f",
    "info": "\u2139\ufe0f",
    "trade": "\U0001f4b0",
}

_MAX_MESSAGE_LENGTH = 4096
_TIMEOUT_SECONDS = 10
_MAX_RETRIES = 2


class TelegramNotifier:
    """Sends alerts to a Telegram chat via the Bot API.

    If credentials are missing every send call silently becomes a no-op so
    the trading loop is never disrupted.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.configured:
            logger.debug("TelegramNotifier not configured — sends will be no-ops")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        """Return ``True`` when both token and chat_id are available."""
        return bool(self._token and self._chat_id)

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Post *message* to the configured Telegram chat.

        * Truncates messages longer than 4 096 characters.
        * Retries up to 2 times on transient failures.
        * Never raises — returns ``False`` on any error.
        """
        if not self.configured:
            logger.debug("Telegram not configured — skipping send")
            return False

        text = message[:_MAX_MESSAGE_LENGTH]
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS)

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session, session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        return True
                    body = await resp.text()
                    logger.warning(
                        "Telegram send attempt %d/%d failed: %s %s",
                        attempt,
                        _MAX_RETRIES,
                        resp.status,
                        body,
                    )
            except Exception:
                logger.warning(
                    "Telegram send attempt %d/%d exception",
                    attempt,
                    _MAX_RETRIES,
                    exc_info=True,
                )

        logger.error("Telegram send failed after %d attempts", _MAX_RETRIES)
        return False

    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
    ) -> bool:
        """Send a formatted alert with emoji, type header, and timestamp.

        Args:
            alert_type: Short label such as ``"TRADE"`` or ``"RISK"``.
            message: Free-form alert body.
            severity: One of ``"critical"``, ``"warning"``, ``"info"``,
                ``"trade"``.  Controls the emoji prefix.

        Returns:
            ``True`` if the message was delivered, ``False`` otherwise.
        """
        emoji = _EMOJI.get(severity, _EMOJI["info"])
        ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        formatted = f"{emoji} <b>{alert_type}</b>\n{message}\n<i>{ts}</i>"
        return await self.send(formatted)
