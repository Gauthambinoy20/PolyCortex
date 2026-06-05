"""Polygon RPC failover with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import cast

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_RPC_URLS = [
    "https://polygon-rpc.com",
    "https://rpc.ankr.com/polygon",
    "https://polygon.drpc.org",
]


def get_rpc_urls() -> list[str]:
    """Get list of Polygon RPC URLs from environment or defaults."""
    env_val = os.environ.get("POLYGON_RPC_URLS", "")
    if env_val:
        return [url.strip() for url in env_val.split(",") if url.strip()]
    single = os.environ.get("POLYGON_RPC_URL", "")
    if single:
        return [single] + [u for u in DEFAULT_RPC_URLS if u != single]
    return DEFAULT_RPC_URLS


async def eth_call_with_failover(payload: dict, timeout: float = 5.0) -> dict:
    """Execute an eth JSON-RPC call with RPC failover and exponential backoff.

    Args:
        payload: JSON-RPC payload dict.
        timeout: Per-request timeout in seconds.

    Returns:
        JSON-RPC response dict.

    Raises:
        RuntimeError: If all RPCs fail.
    """
    rpc_urls = get_rpc_urls()
    last_exc: Exception | None = None

    for attempt, rpc_url in enumerate(rpc_urls):
        delay = min(2**attempt, 30)
        if attempt > 0:
            logger.warning(
                "RPC failover: trying %s (attempt %d/%d) after %.1fs",
                rpc_url,
                attempt + 1,
                len(rpc_urls),
                delay,
            )
            await asyncio.sleep(delay)
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp,
            ):
                if resp.status == 200:
                    return cast("dict", await resp.json())
                logger.warning("RPC %s returned HTTP %d", rpc_url, resp.status)
        except Exception as exc:
            last_exc = exc
            logger.warning("RPC %s error: %s", rpc_url, exc)

    raise RuntimeError(f"All Polygon RPC endpoints failed. Last error: {last_exc}")
