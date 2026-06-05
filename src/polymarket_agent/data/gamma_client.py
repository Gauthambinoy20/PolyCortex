"""Async client for the Polymarket Gamma API."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from typing import cast

import aiohttp

logger = logging.getLogger(__name__)


class _TTLCache:
    """Simple in-memory cache with per-key TTL expiration.

    Thread-safe via asyncio (single-threaded event loop).
    Not designed for multi-process use.
    """

    def __init__(self, default_ttl: float = 300.0, max_size: int = 1000) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._store: dict[str, tuple[float, object]] = {}  # key → (expires_at, value)
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> object | None:
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        expires_at, value = entry
        if monotonic() > expires_at:
            del self._store[key]
            self._misses += 1
            return None
        self._hits += 1
        return value

    def set(self, key: str, value: object, ttl: float | None = None) -> None:
        if len(self._store) >= self._max_size:
            self._evict_expired()
        expires_at = monotonic() + (ttl if ttl is not None else self._default_ttl)
        self._store[key] = (expires_at, value)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def _evict_expired(self) -> None:
        now = monotonic()
        expired = [k for k, (exp, _) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._store),
        }


@dataclass
class Market:
    condition_id: str
    question: str
    description: str
    category: str
    end_date: datetime | None
    active: bool
    liquidity: float
    volume_24h: float
    yes_price: float
    no_price: float
    clob_token_ids: list[str] = field(default_factory=list)


class GammaClient:
    """Async client for https://gamma-api.polymarket.com."""

    def __init__(
        self,
        base_url: str = "https://gamma-api.polymarket.com",
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._session_lock = asyncio.Lock()
        self._max_retries: int = 3
        self._base_delay: float = 1.0
        self._cache = _TTLCache(default_ttl=300.0)  # 5 min default

    async def __aenter__(self) -> "GammaClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

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
                            "Gamma API %s rate limited (429), waiting %.1fs (attempt %d/%d)",
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
                            "Gamma API %s returned %d, retrying in %.1fs (attempt %d/%d)",
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
                        "Gamma API request error: %s, retrying in %.1fs (attempt %d/%d)",
                        exc,
                        delay,
                        attempt + 1,
                        self._max_retries,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.warning("Gamma API request failed after %d retries: %s", self._max_retries, exc)
                    return None
        return None

    @staticmethod
    def _parse_json_field(raw: str | list | None) -> list:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        try:
            return cast("list", json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _parse_outcome_prices(raw: str | list | None) -> tuple[float, float]:
        prices = GammaClient._parse_json_field(raw)
        if len(prices) >= 2:
            try:
                return float(prices[0]), float(prices[1])
            except (ValueError, TypeError):
                pass
        return 0.0, 0.0

    @staticmethod
    def _parse_market(data: dict) -> Market | None:
        clob_token_ids = GammaClient._parse_json_field(data.get("clobTokenIds"))
        if not clob_token_ids:
            return None

        end_date: datetime | None = None
        raw_end = data.get("endDate")
        if raw_end:
            from contextlib import suppress

            with suppress(ValueError, AttributeError):
                end_date = datetime.fromisoformat(raw_end.replace("Z", "+00:00"))

        yes_price, no_price = GammaClient._parse_outcome_prices(data.get("outcomePrices"))

        category = data.get("groupItemTitle") or data.get("category") or ""

        try:
            liquidity = float(data.get("liquidity") or 0)
        except (ValueError, TypeError):
            liquidity = 0.0

        volume_raw = data.get("volume24hr") or data.get("volume") or 0
        try:
            volume_24h = float(volume_raw)
        except (ValueError, TypeError):
            volume_24h = 0.0

        return Market(
            condition_id=data.get("conditionId", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            category=category,
            end_date=end_date,
            active=bool(data.get("active", False)),
            liquidity=liquidity,
            volume_24h=volume_24h,
            yes_price=yes_price,
            no_price=no_price,
            clob_token_ids=[str(t) for t in clob_token_ids],
        )

    async def get_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        cache_key = f"markets:{active}:{closed}:{limit}:{offset}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Gamma cache hit for %s", cache_key)
            return cast("list[Market]", cached)

        url = (
            f"{self._base_url}/markets"
            f"?active={str(active).lower()}&closed={str(closed).lower()}"
            f"&limit={limit}&offset={offset}"
        )
        resp = await self._request("GET", url)
        if not resp or not isinstance(resp, list):
            return []

        markets: list[Market] = []
        for item in resp:
            market = self._parse_market(item)
            if market is not None:
                markets.append(market)

        self._cache.set(cache_key, markets, ttl=300.0)
        return markets

    async def get_all_markets(
        self,
        *,
        active: bool = True,
        closed: bool = False,
    ) -> list[Market]:
        cache_key = f"all_markets:{active}:{closed}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Gamma cache hit for %s", cache_key)
            return cast("list[Market]", cached)

        all_markets: list[Market] = []
        offset = 0
        limit = 100
        while True:
            batch = await self.get_markets(
                active=active,
                closed=closed,
                limit=limit,
                offset=offset,
            )
            all_markets.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
            await asyncio.sleep(0.5)

        self._cache.set(cache_key, all_markets, ttl=300.0)
        return all_markets

    async def get_market(self, condition_id: str) -> Market | None:
        url = f"{self._base_url}/markets/{condition_id}"
        resp = await self._request("GET", url)
        if not resp or not isinstance(resp, dict):
            return None
        return self._parse_market(resp)

    @property
    def cache_stats(self) -> dict:
        """Return cache hit/miss statistics."""
        return self._cache.stats

    def clear_cache(self) -> None:
        """Force clear all cached data."""
        self._cache.clear()
        logger.info("Gamma API cache cleared")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
