"""Load historical Polymarket data for backtesting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from polymarket_agent.data.history import HistoryStore

if TYPE_CHECKING:
    from polymarket_agent.data.gamma_client import GammaClient

logger = logging.getLogger(__name__)

CATEGORIES = ["politics", "crypto", "sports", "science", "entertainment"]


@dataclass
class BacktestMarket:
    condition_id: str
    question: str
    category: str
    resolution: float  # 1.0 for YES, 0.0 for NO, 0.5 for unknown
    price_history: pd.DataFrame  # columns: timestamp, midpoint, spread, bid_depth, ask_depth, volume, book_imbalance
    resolution_date: datetime


class BacktestDataLoader:
    """Loads market data from local history, Gamma API, or synthetic generation."""

    def __init__(self, data_dir: str = "data/price_history") -> None:
        self._data_dir = data_dir
        self._history_store = HistoryStore(data_dir=data_dir)

    def load_from_local(self, min_rows: int = 50) -> list[BacktestMarket]:
        """Load markets from locally stored parquet history files.

        Resolution is unknown for local data, so we set resolution=0.5
        and resolution_date to last timestamp + 30 days as placeholders.
        """
        market_ids = self._history_store.list_markets()
        if not market_ids:
            logger.warning("No local markets found in %s", self._data_dir)
            return []

        markets: list[BacktestMarket] = []
        for condition_id in market_ids:
            df = self._history_store.load_history(condition_id)
            if df is None or len(df) < min_rows:
                logger.debug(
                    "Skipping %s: %d rows (need %d)",
                    condition_id,
                    0 if df is None else len(df),
                    min_rows,
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            last_ts = df["timestamp"].max().to_pydatetime()
            resolution_date = last_ts + timedelta(days=30)

            # Infer resolution from final price trajectory
            final_price = df["midpoint"].iloc[-1]
            if final_price >= 0.85:
                resolution = 1.0
            elif final_price <= 0.15:
                resolution = 0.0
            else:
                resolution = round(final_price, 2)  # Use final price as best estimate

            markets.append(
                BacktestMarket(
                    condition_id=condition_id,
                    question=f"Market {condition_id[:12]}...",
                    category="unknown",
                    resolution=resolution,
                    price_history=df,
                    resolution_date=resolution_date,
                )
            )

        logger.info(
            "Loaded %d local markets (filtered from %d with min_rows=%d)",
            len(markets),
            len(market_ids),
            min_rows,
        )
        return markets

    async def load_resolved_markets(
        self,
        gamma: GammaClient,
        min_trades: int = 50,
        limit: int = 100,
    ) -> list[BacktestMarket]:
        """Fetch resolved markets from Gamma API and build BacktestMarket objects.

        Uses active=false&closed=true to get resolved markets.
        Tries local history first, falls back to Gamma API data.
        """
        raw_markets = await gamma.get_markets(active=False, closed=True, limit=limit)
        if not raw_markets:
            logger.warning("No resolved markets returned from Gamma API")
            return []

        markets: list[BacktestMarket] = []
        for market in raw_markets:
            # Try local history first
            df = self._history_store.load_history(market.condition_id)
            if df is None or len(df) < min_trades:
                logger.debug(
                    "Skipping resolved market %s: insufficient history (%d rows)",
                    market.condition_id,
                    0 if df is None else len(df),
                )
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

            # Determine resolution from final price
            final_price = df["midpoint"].iloc[-1]
            if final_price >= 0.95:
                resolution = 1.0
            elif final_price <= 0.05:
                resolution = 0.0
            else:
                resolution = 0.5  # ambiguous

            resolution_date = (
                market.end_date
                if market.end_date is not None
                else df["timestamp"].max().to_pydatetime()
            )

            markets.append(
                BacktestMarket(
                    condition_id=market.condition_id,
                    question=market.question,
                    category=market.category or "unknown",
                    resolution=resolution,
                    price_history=df,
                    resolution_date=resolution_date,
                )
            )

        logger.info(
            "Loaded %d resolved markets from Gamma API (filtered from %d)",
            len(markets),
            len(raw_markets),
        )
        return markets

    def load_synthetic(
        self,
        n_markets: int = 20,
        n_points: int = 200,
    ) -> list[BacktestMarket]:
        """Generate synthetic markets for testing the backtest pipeline.

        Creates random-walk prices bounded [0.05, 0.95] with realistic
        spread, volume, and book imbalance. Prices drift toward the
        final resolution.
        """
        rng = np.random.default_rng(seed=42)
        markets: list[BacktestMarket] = []

        for i in range(n_markets):
            resolution = float(rng.choice([0.0, 1.0]))
            category = rng.choice(CATEGORIES)

            # Random walk with drift toward resolution
            drift = 0.001 if resolution == 1.0 else -0.001
            prices = np.zeros(n_points)
            prices[0] = rng.uniform(0.3, 0.7)
            for t in range(1, n_points):
                noise = rng.normal(0, 0.02)
                prices[t] = prices[t - 1] + drift + noise
                prices[t] = np.clip(prices[t], 0.05, 0.95)

            # Realistic spread: 2%–5%
            spreads = rng.uniform(0.02, 0.05, size=n_points)

            # Bid/ask depths
            bid_depth = rng.exponential(500, size=n_points)
            ask_depth = rng.exponential(500, size=n_points)

            # Volume: log-normal
            volume = rng.lognormal(mean=6.0, sigma=1.0, size=n_points)

            # Book imbalance derived from depths
            book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)

            # Timestamps: hourly over n_points hours
            base_time = datetime(2025, 1, 1, tzinfo=UTC)
            timestamps = [base_time + timedelta(hours=h) for h in range(n_points)]
            resolution_date = timestamps[-1] + timedelta(days=7)

            df = pd.DataFrame({
                "timestamp": timestamps,
                "midpoint": prices,
                "spread": spreads,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "volume": volume,
                "book_imbalance": book_imbalance,
            })

            markets.append(
                BacktestMarket(
                    condition_id=f"synthetic_{i:04d}",
                    question=f"Synthetic market #{i} ({category})",
                    category=category,
                    resolution=resolution,
                    price_history=df,
                    resolution_date=resolution_date,
                )
            )

        logger.info("Generated %d synthetic markets with %d points each", n_markets, n_points)
        return markets
