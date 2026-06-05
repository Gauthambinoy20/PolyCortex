"""Real-time WebSocket client for Polymarket CLOB order book updates."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import UTC, datetime

import aiohttp

from polymarket_agent.types import Event, EventType, OrderBookLevel, OrderBookSnapshot

logger = logging.getLogger(__name__)


class WebSocketClient:
    """Maintains WebSocket connections to Polymarket CLOB for real-time book updates.

    Features:
    - Subscribe/unsubscribe to token_id order book streams
    - Automatic reconnection with exponential backoff (1s → 2s → 4s → ... → 60s max)
    - Local order book state maintained per token_id
    - Publishes OrderBookSnapshot updates to an optional EventBus
    - Falls back gracefully if WebSocket connection fails
    - Thread-safe token management

    Usage:
        ws = WebSocketClient()
        await ws.start()
        await ws.subscribe("token_id_123")
        # snapshots auto-update
        snapshot = ws.get_snapshot("token_id_123")
        await ws.stop()
    """

    WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    def __init__(
        self,
        url: str | None = None,
        max_reconnect_delay: float = 60.0,
        event_bus=None,  # Optional EventBus instance
    ) -> None:
        self._url = url or self.WS_URL
        self._max_reconnect_delay = max_reconnect_delay
        self._event_bus = event_bus

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._running = False
        self._reconnect_delay = 1.0

        # Token subscriptions and local book state
        self._subscribed_tokens: set[str] = set()
        self._snapshots: dict[str, OrderBookSnapshot] = {}
        self._lock = asyncio.Lock()

        # Tasks
        self._listen_task: asyncio.Task | None = None

        # Stats
        self._messages_received = 0
        self._reconnect_count = 0
        self._last_message_at: datetime | None = None

    async def start(self) -> None:
        """Start the WebSocket client and begin listening."""
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession()
        self._listen_task = asyncio.create_task(self._connection_loop())
        logger.info("WebSocket client started, connecting to %s", self._url)

    async def stop(self) -> None:
        """Gracefully stop the WebSocket client."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._listen_task:
            self._listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listen_task
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info(
            "WebSocket client stopped. Messages: %d, Reconnects: %d",
            self._messages_received,
            self._reconnect_count,
        )

    async def subscribe(self, token_id: str) -> None:
        """Subscribe to order book updates for a token."""
        async with self._lock:
            self._subscribed_tokens.add(token_id)
        if self._ws and not self._ws.closed:
            await self._send_subscribe(token_id)
        logger.info("Subscribed to token %s", token_id[:16])

    async def unsubscribe(self, token_id: str) -> None:
        """Unsubscribe from a token's order book updates."""
        async with self._lock:
            self._subscribed_tokens.discard(token_id)
            self._snapshots.pop(token_id, None)
        logger.info("Unsubscribed from token %s", token_id[:16])

    def get_snapshot(self, token_id: str) -> OrderBookSnapshot | None:
        """Get the latest order book snapshot for a token (non-blocking)."""
        return self._snapshots.get(token_id)

    def get_all_snapshots(self) -> dict[str, OrderBookSnapshot]:
        """Get all current order book snapshots."""
        return dict(self._snapshots)

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    @property
    def stats(self) -> dict:
        return {
            "connected": self.is_connected,
            "subscriptions": len(self._subscribed_tokens),
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            "last_message_at": (self._last_message_at.isoformat() if self._last_message_at else None),
        }

    # ── Internal methods ──────────────────────────────────────────

    async def _connection_loop(self) -> None:
        """Main connection loop with auto-reconnect."""
        while self._running:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("WebSocket connection error")

            if not self._running:
                break

            # Exponential backoff
            logger.info(
                "Reconnecting in %.1fs (attempt %d)",
                self._reconnect_delay,
                self._reconnect_count + 1,
            )
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            self._reconnect_count += 1

    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and process messages."""
        assert self._session is not None

        self._ws = await self._session.ws_connect(
            self._url,
            heartbeat=30.0,
            timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
        )
        logger.info("WebSocket connected to %s", self._url)

        # Reset backoff on successful connection
        self._reconnect_delay = 1.0

        # Re-subscribe to all tokens
        async with self._lock:
            tokens = list(self._subscribed_tokens)
        for token_id in tokens:
            await self._send_subscribe(token_id)

        # Listen for messages
        async for msg in self._ws:
            if not self._running:
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._handle_message(msg.data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error("WebSocket error: %s", self._ws.exception())
                break
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                logger.warning("WebSocket closed by server")
                break

    async def _send_subscribe(self, token_id: str) -> None:
        """Send subscription message for a token."""
        if not self._ws or self._ws.closed:
            return
        msg = json.dumps(
            {
                "type": "market",
                "assets_id": token_id,
            }
        )
        try:
            await self._ws.send_str(msg)
            logger.debug("Sent subscribe for %s", token_id[:16])
        except Exception:
            logger.exception("Failed to send subscribe for %s", token_id[:16])

    async def _handle_message(self, raw: str) -> None:
        """Parse a WebSocket message and update local order book state."""
        self._messages_received += 1
        self._last_message_at = datetime.now(UTC)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from WebSocket: %s", raw[:100])
            return

        # Extract token_id from message
        # Polymarket sends various message formats; handle the common ones
        token_id = data.get("asset_id") or data.get("market") or data.get("token_id")
        if not token_id:
            logger.debug("WS message without token_id: %s", str(data)[:100])
            return

        # Parse order book data
        raw_bids = data.get("bids") or []
        raw_asks = data.get("asks") or []

        bids = self._parse_levels(raw_bids)
        asks = self._parse_levels(raw_asks)

        if not bids and not asks:
            return

        # Build snapshot
        snapshot = OrderBookSnapshot(
            token_id=token_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(UTC),
        )

        # Store locally
        self._snapshots[token_id] = snapshot

        # Publish to event bus if available
        if self._event_bus is not None:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.ORDERBOOK_UPDATED,
                    payload={
                        "token_id": token_id,
                        "midpoint": snapshot.midpoint,
                        "spread": snapshot.spread,
                        "best_bid": snapshot.best_bid,
                        "best_ask": snapshot.best_ask,
                        "bid_levels": len(bids),
                        "ask_levels": len(asks),
                    },
                    source="ws_client",
                )
            )

    @staticmethod
    def _parse_levels(raw_levels: list[dict]) -> list[OrderBookLevel]:
        """Parse raw price/size dicts into OrderBookLevel objects."""
        levels: list[OrderBookLevel] = []
        for entry in raw_levels:
            try:
                price = float(entry.get("price", 0))
                size = float(entry.get("size", 0))
                if 0 < price <= 1.0 and size > 0:
                    levels.append(OrderBookLevel(price=price, size=size))
            except (ValueError, TypeError):
                continue
        # Sort bids descending, asks ascending
        levels.sort(key=lambda lv: lv.price, reverse=True)
        return levels
