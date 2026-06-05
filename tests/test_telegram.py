"""Tests for telegram, alerting, and heartbeat modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymarket_agent.utils.alerting import AlertRouter
from polymarket_agent.utils.heartbeat import HeartbeatMonitor
from polymarket_agent.utils.telegram import TelegramNotifier

# ── TelegramNotifier ──────────────────────────────────────────────────


class TestTelegramNotifier:
    """TelegramNotifier without real credentials."""

    def test_not_configured_by_default(self) -> None:
        notifier = TelegramNotifier()
        assert notifier.configured is False

    def test_configured_when_both_set(self) -> None:
        notifier = TelegramNotifier(bot_token="tok", chat_id="123")
        assert notifier.configured is True

    def test_not_configured_missing_chat_id(self) -> None:
        notifier = TelegramNotifier(bot_token="tok")
        assert notifier.configured is False

    @pytest.mark.asyncio
    async def test_send_noop_when_not_configured(self) -> None:
        notifier = TelegramNotifier()
        result = await notifier.send("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_noop_when_not_configured(self) -> None:
        notifier = TelegramNotifier()
        result = await notifier.send_alert("TEST", "body")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_success_mocked(self) -> None:
        notifier = TelegramNotifier(bot_token="tok", chat_id="123")

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await notifier.send("hello")

        assert result is True

    @pytest.mark.asyncio
    async def test_send_truncates_long_messages(self) -> None:
        notifier = TelegramNotifier(bot_token="tok", chat_id="123")

        captured_payload: dict = {}

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()

        def capture_post(url: str, json: dict) -> AsyncMock:
            captured_payload.update(json)
            return mock_resp

        mock_session.post = MagicMock(side_effect=capture_post)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await notifier.send("x" * 5000)

        assert len(captured_payload["text"]) == 4096

    @pytest.mark.asyncio
    async def test_send_retries_on_failure(self) -> None:
        notifier = TelegramNotifier(bot_token="tok", chat_id="123")

        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="err")
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await notifier.send("hello")

        assert result is False
        assert mock_session.post.call_count == 2  # 2 retries


# ── AlertRouter ───────────────────────────────────────────────────────


class TestAlertRouter:
    """AlertRouter JSONL + Telegram routing."""

    @pytest.mark.asyncio
    async def test_writes_jsonl(self, tmp_path: Path) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        router = AlertRouter(alerts_path=alerts_file)

        await router.alert("TRADE", "Bought YES", severity="trade")
        await router.alert("RISK", "Drawdown high", severity="warning")

        lines = alerts_file.read_text().strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["alert_type"] == "TRADE"
        assert rec["severity"] == "trade"
        assert rec["message"] == "Bought YES"

    @pytest.mark.asyncio
    async def test_telegram_skipped_for_info(self, tmp_path: Path) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        tg = AsyncMock(spec=TelegramNotifier)
        tg.configured = True
        router = AlertRouter(alerts_path=alerts_file, telegram=tg)

        await router.alert("SYS", "routine check", severity="info")

        tg.send_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_telegram_sent_for_warning(self, tmp_path: Path) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        tg = AsyncMock(spec=TelegramNotifier)
        tg.configured = True
        router = AlertRouter(alerts_path=alerts_file, telegram=tg)

        await router.alert("RISK", "drawdown", severity="warning")

        tg.send_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_telegram_sent_for_trade(self, tmp_path: Path) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        tg = AsyncMock(spec=TelegramNotifier)
        tg.configured = True
        router = AlertRouter(alerts_path=alerts_file, telegram=tg)

        await router.alert("TRADE", "filled", severity="trade")

        tg.send_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_telegram_threshold(self, tmp_path: Path) -> None:
        alerts_file = tmp_path / "alerts.jsonl"
        tg = AsyncMock(spec=TelegramNotifier)
        tg.configured = True
        router = AlertRouter(alerts_path=alerts_file, telegram=tg)

        router.set_telegram_threshold("info")
        await router.alert("SYS", "now delivered", severity="info")

        tg.send_alert.assert_awaited_once()


# ── HeartbeatMonitor ──────────────────────────────────────────────────


class TestHeartbeatMonitor:
    """HeartbeatMonitor beat / stale detection."""

    def test_beat_creates_file(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "hb.json")
        hb.beat()
        assert (tmp_path / "hb.json").exists()

    def test_beat_with_metrics(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "hb.json")
        hb.beat(cycle_metrics={"trades_placed": 2})
        data = json.loads((tmp_path / "hb.json").read_text())
        assert data["cycle_metrics"]["trades_placed"] == 2
        assert "pid" in data
        assert "timestamp" in data

    def test_is_stale_when_no_file(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "missing.json")
        stale, mins = hb.is_stale()
        assert stale is True
        assert mins == float("inf")

    def test_is_not_stale_after_beat(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "hb.json", max_stale_minutes=30)
        hb.beat()
        stale, mins = hb.is_stale()
        assert stale is False
        assert mins < 1.0

    def test_get_last_beat_none_when_missing(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "nope.json")
        assert hb.get_last_beat() is None

    def test_get_last_beat_returns_dict(self, tmp_path: Path) -> None:
        hb = HeartbeatMonitor(heartbeat_path=tmp_path / "hb.json")
        hb.beat()
        data = hb.get_last_beat()
        assert isinstance(data, dict)
        assert "timestamp" in data
