"""Async client for the Polymarket CLOB API (order book and trades)."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    token_id: str
    timestamp: datetime
    midpoint: float
    spread: float
    bid_depth: float  # total USDC on bid side (top 10 levels)
    ask_depth: float
    best_bid: float
    best_ask: float
    bids: list[tuple[float, float]]  # (price, size) top 10
    asks: list[tuple[float, float]]  # (price, size) top 10


class ClobClient:
    """Async client for https://clob.polymarket.com."""

    def __init__(
        self,
        base_url: str = "https://clob.polymarket.com",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._max_retries: int = 3
        self._base_delay: float = 1.0

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def _request(self, method: str, url: str) -> dict | list | None:
        session = await self._get_session()
        for attempt in range(self._max_retries):
            try:
                async with session.request(method, url) as resp:
                    if resp.status == 200:
                        return cast("dict | list | None", await resp.json())
                    if resp.status == 429:
                        retry_after_hdr = resp.headers.get("Retry-After")
                        if retry_after_hdr is not None:
                            try:
                                delay = float(retry_after_hdr)
                            except ValueError:
                                delay = self._base_delay * (2**attempt)
                        else:
                            delay = self._base_delay * (2**attempt)
                        logger.warning(
                            "CLOB API %s rate limited (429), waiting %.1fs (attempt %d/%d)",
                            url,
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if resp.status >= 500:
                        delay = self._base_delay * (2**attempt)
                        logger.warning(
                            "CLOB API %s returned %d, retrying in %.1fs (attempt %d/%d)",
                            url,
                            resp.status,
                            delay,
                            attempt + 1,
                            self._max_retries,
                        )
                        await asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
            except aiohttp.ClientError as exc:
                delay = self._base_delay * (2**attempt)
                if attempt < self._max_retries - 1:
                    logger.warning(
                        "CLOB API request error: %s, retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning("CLOB API request failed after %d retries: %s", self._max_retries, exc)
                    return None
        return None

    @staticmethod
    def _parse_levels(raw_levels: list[dict]) -> list[tuple[float, float]]:
        parsed: list[tuple[float, float]] = []
        for level in raw_levels:
            try:
                price = float(level.get("price", 0))
                size = float(level.get("size", 0))
                parsed.append((price, size))
            except (ValueError, TypeError):
                continue
        return parsed

    async def get_order_book(self, token_id: str) -> OrderBookSnapshot | None:
        url = f"{self._base_url}/book?token_id={token_id}"
        resp = await self._request("GET", url)
        if not resp or not isinstance(resp, dict):
            return None

        raw_bids = resp.get("bids") or []
        raw_asks = resp.get("asks") or []

        if not raw_bids or not raw_asks:
            return None

        bids = self._parse_levels(raw_bids)
        asks = self._parse_levels(raw_asks)

        if not bids or not asks:
            return None

        # Sort bids descending by price, asks ascending by price
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        # Take top 10
        bids_top10 = bids[:10]
        asks_top10 = asks[:10]

        best_bid = bids_top10[0][0]
        best_ask = asks_top10[0][0]
        midpoint = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        if not (0.0 <= best_bid <= 1.0) or not (0.0 <= best_ask <= 1.0):
            logger.warning(
                "Invalid price bounds for %s: bid=%.4f ask=%.4f — skipping",
                token_id,
                best_bid,
                best_ask,
            )
            return None
        if best_bid > best_ask:
            logger.warning(
                "Crossed book for %s: bid=%.4f > ask=%.4f — skipping",
                token_id,
                best_bid,
                best_ask,
            )
            return None

        bid_depth = sum(size for _, size in bids_top10)
        ask_depth = sum(size for _, size in asks_top10)

        return OrderBookSnapshot(
            token_id=token_id,
            timestamp=datetime.now(UTC),
            midpoint=midpoint,
            spread=spread,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            best_bid=best_bid,
            best_ask=best_ask,
            bids=bids_top10,
            asks=asks_top10,
        )

    async def get_trades(self, token_id: str, limit: int = 100) -> list[dict]:
        url = f"{self._base_url}/trades?token_id={token_id}&limit={limit}"
        resp = await self._request("GET", url)
        if not resp or not isinstance(resp, list):
            return []
        return resp

    async def __aenter__(self) -> "ClobClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
