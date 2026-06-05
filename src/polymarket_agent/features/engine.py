import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PolymarketFeatureEngine:
    """Generates 10 features for prediction market data."""

    FEATURE_NAMES: list[str] = [
        "logit_return",
        "price_extremity",
        "rsi",
        "price_vs_sma",
        "volume_ratio",
        "volume_momentum",
        "spread",
        "book_imbalance",
        "time_remaining_pct",
        "urgency",
    ]

    def __init__(self, window_size: int = 64) -> None:
        self.window_size: int = window_size

    def create_features(
        self,
        history_df: pd.DataFrame,
        resolution_date: datetime | None = None,
    ) -> np.ndarray | None:
        """Build a (timesteps, 10) feature matrix from market history.

        Returns None if insufficient data remains after dropping NaNs.
        """
        df = history_df.copy()

        # Ensure we have a datetime index / column
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No timestamp column or DatetimeIndex found")
            return None

        df = df.sort_index()

        if len(df) < 2:
            logger.warning("History too short (%d rows)", len(df))
            return None

        midpoint: pd.Series = df["midpoint"]
        volume: pd.Series = df["volume"]

        # --- Feature 0: logit_return ---
        clipped: pd.Series = midpoint.clip(0.01, 0.99)
        logit: pd.Series = np.log(clipped / (1.0 - clipped))
        logit_return: pd.Series = logit.diff()

        # --- Feature 1: price_extremity ---
        price_extremity: pd.Series = 2.0 * (midpoint - 0.5).abs()

        # --- Feature 2: RSI(14) — standard Wilder RSI ---
        delta: pd.Series = midpoint.diff()
        gain: pd.Series = delta.clip(lower=0).rolling(14).mean()
        loss: pd.Series = (-delta.clip(upper=0)).rolling(14).mean()
        rs: pd.Series = gain / (loss + 1e-8)
        rsi: pd.Series = 1.0 - 1.0 / (1.0 + rs)  # Normalized [0, 1]

        # --- Feature 3: price_vs_sma ---
        price_vs_sma: pd.Series = midpoint - midpoint.rolling(20).mean()

        # --- Feature 4: volume_ratio ---
        vol_sma_20: pd.Series = volume.rolling(20).mean()
        if (volume == 0).all():
            volume_ratio: pd.Series = pd.Series(1.0, index=df.index)
        else:
            volume_ratio = volume / (vol_sma_20 + 1e-8)

        # --- Feature 5: volume_momentum ---
        vol_sma_5: pd.Series = volume.rolling(5).mean()
        if (volume == 0).all():
            volume_momentum: pd.Series = pd.Series(1.0, index=df.index)
        else:
            volume_momentum = vol_sma_5 / (vol_sma_20 + 1e-8)

        # --- Feature 6: spread ---
        spread: pd.Series = df["spread"]

        # --- Feature 7: book_imbalance ---
        book_imbalance: pd.Series = df["book_imbalance"]

        # --- Feature 8: time_remaining_pct ---
        if resolution_date is not None:
            # Ensure resolution_date is timezone-aware
            res_date = resolution_date
            if res_date.tzinfo is None:
                res_date = res_date.replace(tzinfo=UTC)
            first_ts = df.index[0].to_pydatetime()
            if first_ts.tzinfo is None:
                first_ts = first_ts.replace(tzinfo=UTC)
            total_days: float = (res_date - first_ts).days
            if total_days <= 0:
                time_remaining_pct: pd.Series = pd.Series(0.0, index=df.index)
            else:
                days_remaining: pd.Series = pd.Series(
                    [
                        (
                            res_date
                            - (
                                ts.to_pydatetime().replace(tzinfo=UTC)
                                if ts.to_pydatetime().tzinfo is None
                                else ts.to_pydatetime()
                            )
                        ).days
                        for ts in df.index
                    ],
                    index=df.index,
                    dtype=float,
                )
                time_remaining_pct = (days_remaining / total_days).clip(0.0, 1.0)
        else:
            time_remaining_pct = pd.Series(1.0, index=df.index)

        # --- Feature 9: urgency ---
        if resolution_date is not None:
            res_date_u = resolution_date
            if res_date_u.tzinfo is None:
                res_date_u = res_date_u.replace(tzinfo=UTC)
            days_rem: pd.Series = pd.Series(
                [
                    (
                        res_date_u
                        - (
                            ts.to_pydatetime().replace(tzinfo=UTC)
                            if ts.to_pydatetime().tzinfo is None
                            else ts.to_pydatetime()
                        )
                    ).days
                    for ts in df.index
                ],
                index=df.index,
                dtype=float,
            )
            urgency: pd.Series = np.exp(-days_rem / 30.0)
        else:
            urgency = pd.Series(0.0, index=df.index)

        # Assemble feature DataFrame
        features_df = pd.DataFrame(
            {
                "logit_return": logit_return,
                "price_extremity": price_extremity,
                "rsi": rsi,
                "price_vs_sma": price_vs_sma,
                "volume_ratio": volume_ratio,
                "volume_momentum": volume_momentum,
                "spread": spread,
                "book_imbalance": book_imbalance,
                "time_remaining_pct": time_remaining_pct,
                "urgency": urgency,
            },
            index=df.index,
        )

        # Rolling normalize features 0–7, leave 8–9 as-is
        norm_cols: list[str] = self.FEATURE_NAMES[:8]
        roll_win: int = min(50, len(features_df))
        for col in norm_cols:
            roll_mean: pd.Series = features_df[col].rolling(roll_win, min_periods=1).mean()
            roll_std: pd.Series = features_df[col].rolling(roll_win, min_periods=1).std()
            features_df[col] = (features_df[col] - roll_mean) / (roll_std + 1e-8)

        # Drop NaN rows
        features_df = features_df.dropna()

        if len(features_df) < 2:
            logger.warning("Fewer than 2 rows after NaN drop (%d)", len(features_df))
            return None

        result: np.ndarray = features_df.values.astype(np.float32)
        logger.info("Created feature matrix with shape %s", result.shape)
        return result
