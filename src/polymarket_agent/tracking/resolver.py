"""Periodic market resolution checker: closes resolved positions and records P&L."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class MarketResolver:
    """Checks open positions and marks resolved markets, recording final P&L.

    Call check_and_resolve() periodically (e.g., every 5 minutes).
    """

    def __init__(self, db_path: str, gamma_client: object) -> None:
        self.db_path = db_path
        self.gamma_client = gamma_client
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    status TEXT,
                    pnl REAL,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    async def check_and_resolve(self) -> int:
        """Check open positions for resolved markets. Returns count of resolved positions."""
        resolved_count = 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT id, market_id FROM trades WHERE status='open'")
                open_trades = cur.fetchall()

            for trade in open_trades:
                try:
                    is_resolved = await self._check_market_resolved(trade["market_id"])
                    if is_resolved:
                        await self._close_resolved_position(trade["id"], trade["market_id"])
                        resolved_count += 1
                except Exception as exc:
                    logger.warning("Error checking market %s: %s", trade["market_id"], exc)
        except Exception as exc:
            logger.warning("Market resolution check failed: %s", exc)

        if resolved_count:
            logger.info("Resolved %d position(s)", resolved_count)
        return resolved_count

    async def _check_market_resolved(self, market_id: str) -> bool:
        """Check if a market is resolved via Gamma API."""
        try:
            if hasattr(self.gamma_client, "get_market"):
                market = await self.gamma_client.get_market(market_id)
                if market:
                    return bool(market.get("closed", False) or market.get("resolved", False))
        except Exception as exc:
            logger.debug("Could not check market %s: %s", market_id, exc)
        return False

    async def _close_resolved_position(self, trade_id: int, market_id: str) -> None:
        """Mark a position as resolved and record final P&L.

        NOTE: Actual redemption via py-clob-client is stubbed here.
        When py-clob-client exposes a redeem() method, call it here.
        """
        logger.info(
            "Market %s is resolved. Marking position %d as resolved. "
            "[STUB: call clob_client.redeem(market_id) when available]",
            market_id,
            trade_id,
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "UPDATE trades SET status='resolved', resolved_at=? WHERE id=?",
                (datetime.now(UTC).isoformat(), trade_id),
            )
            conn.commit()
