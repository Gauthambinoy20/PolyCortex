"""Price history storage using parquet files."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from polymarket_agent.data.clob_client import ClobClient, OrderBookSnapshot
    from polymarket_agent.data.gamma_client import GammaClient

logger = logging.getLogger(__name__)


class HistoryStore:
    """Persists order book snapshots as daily parquet files."""

    def __init__(self, data_dir: str = "data/price_history") -> None:
        self._data_dir = Path(data_dir)

    def save_snapshot(
        self,
        condition_id: str,
        snapshot: OrderBookSnapshot,
        volume: float,
    ) -> None:
        market_dir = self._data_dir / condition_id
        market_dir.mkdir(parents=True, exist_ok=True)

        date_str = snapshot.timestamp.strftime("%Y-%m-%d")
        file_path = market_dir / f"{date_str}.parquet"

        bid_depth = snapshot.bid_depth
        ask_depth = snapshot.ask_depth
        book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)

        new_row = pd.DataFrame(
            [
                {
                    "timestamp": snapshot.timestamp,
                    "midpoint": snapshot.midpoint,
                    "spread": snapshot.spread,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "volume": volume,
                    "book_imbalance": book_imbalance,
                }
            ]
        )

        if file_path.exists():
            try:
                existing = pd.read_parquet(file_path)
                combined = pd.concat([existing, new_row], ignore_index=True)
            except Exception as exc:
                logger.error(
                    "Failed to read existing parquet %s: %s — backing up corrupted file",
                    file_path,
                    exc,
                )
                backup_path = file_path.with_suffix(f".corrupted.{int(datetime.now(UTC).timestamp())}.parquet")
                try:
                    file_path.rename(backup_path)
                    logger.info("Moved corrupted file to %s", backup_path)
                except OSError as rename_exc:
                    logger.error("Failed to backup corrupted file: %s", rename_exc)
                combined = new_row
        else:
            combined = new_row

        combined.to_parquet(file_path, index=False)

    def load_history(
        self,
        condition_id: str,
        lookback_hours: int | None = None,
    ) -> pd.DataFrame | None:
        market_dir = self._data_dir / condition_id
        if not market_dir.is_dir():
            return None

        parquet_files = sorted(market_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        frames: list[pd.DataFrame] = []
        for f in parquet_files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", f, exc)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        if lookback_hours is not None:
            cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(hours=lookback_hours)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
            if df.empty:
                return None

        return df[["timestamp", "midpoint", "spread", "bid_depth", "ask_depth", "volume", "book_imbalance"]]

    async def collect_all_active(
        self,
        gamma: GammaClient,
        clob: ClobClient,
    ) -> int:
        markets = await gamma.get_all_markets(active=True)
        collected = 0

        for market in markets:
            if not market.clob_token_ids:
                continue
            token_id = market.clob_token_ids[0]
            try:
                snapshot = await clob.get_order_book(token_id)
                if snapshot is None:
                    continue
                self.save_snapshot(market.condition_id, snapshot, market.volume_24h)
                collected += 1
            except Exception as exc:
                logger.warning(
                    "Failed to collect snapshot for %s: %s",
                    market.condition_id,
                    exc,
                )
            await asyncio.sleep(0.3)

        logger.info("Collected snapshots for %d / %d active markets", collected, len(markets))
        return collected

    def list_markets(self) -> list[str]:
        if not self._data_dir.is_dir():
            return []
        return [d.name for d in sorted(self._data_dir.iterdir()) if d.is_dir()]
