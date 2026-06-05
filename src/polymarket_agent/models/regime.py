"""3-state HMM regime detector for market microstructure regimes."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

REGIMES = ("stable", "trending", "volatile")


class RegimeDetector:
    """Detects market regime (stable / trending / volatile) via a Gaussian HMM."""

    def __init__(self, *, buffer_maxlen: int = 50) -> None:
        self.model: GaussianHMM = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        self.fitted: bool = False
        self.regime_labels: dict[int, str] = {}
        self._history_buffer: list[list[float]] = []
        self._buffer_maxlen: int = buffer_maxlen

    def fit(
        self,
        volatility: pd.Series,
        volume_ratio: pd.Series,
        spread: pd.Series,
    ) -> RegimeDetector:
        """Fit the HMM on stacked (volatility, volume_ratio, spread) features."""
        features = np.column_stack(
            [
                volatility.values,
                volume_ratio.values,
                spread.values,
            ]
        )

        mask = ~np.isnan(features).any(axis=1)
        features = features[mask]

        if len(features) < 100:
            logger.warning("Only %d valid samples (need >=100). Skipping HMM fit.", len(features))
            return self

        self.model.fit(features)

        means = self.model.means_[:, 0]
        order = np.argsort(means)
        self.regime_labels = {int(state_idx): REGIMES[rank] for rank, state_idx in enumerate(order)}
        self.fitted = True
        self.reset_buffer()

        logger.info(
            "HMM fitted on %d samples. State mapping: %s",
            len(features),
            self.regime_labels,
        )
        return self

    def fit_from_history(self, history_df: pd.DataFrame) -> RegimeDetector:
        """Derive features from market history and fit the HMM."""
        volatility = history_df["midpoint"].pct_change().rolling(20).std()
        volume_ratio = history_df["volume"] / history_df["volume"].rolling(20).mean().clip(lower=1e-8)
        spread = history_df["spread"]
        return self.fit(volatility, volume_ratio, spread)

    def predict(
        self,
        volatility: float,
        volume_ratio: float,
        spread: float,
    ) -> tuple[str, dict[str, float]]:
        """Return the latest regime label and probability dictionary."""
        if not self.fitted:
            return ("stable", {"stable": 0.5, "trending": 0.3, "volatile": 0.2})

        observation = [float(volatility), float(volume_ratio), float(spread)]
        self._history_buffer.append(observation)
        if len(self._history_buffer) > self._buffer_maxlen:
            self._history_buffer = self._history_buffer[-self._buffer_maxlen :]

        sequence = np.asarray(self._history_buffer, dtype=float)
        try:
            states = self.model.predict(sequence)
            probs = self.model.predict_proba(sequence)[-1]
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Degenerate covariance / non-PD matrix — fall back to neutral prior
            # rather than crash the trading loop. Log once per failure type.
            logger.warning("Regime predict failed (%s) — returning neutral prior", exc)
            return ("stable", {"stable": 0.5, "trending": 0.3, "volatile": 0.2})

        state = int(states[-1])
        label = self.regime_labels.get(state, "stable")
        prob_dict = {self.regime_labels.get(i, f"state_{i}"): float(prob) for i, prob in enumerate(probs)}
        return (label, prob_dict)

    def reset_buffer(self) -> None:
        """Clear the sliding observation buffer used for sequential prediction."""
        self._history_buffer = []

    def save(self, path: str) -> None:
        """Persist model state, labels, and buffer settings."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": self.model,
            "fitted": self.fitted,
            "regime_labels": self.regime_labels,
            "history_buffer": self._history_buffer,
            "buffer_maxlen": self._buffer_maxlen,
        }
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        logger.info("RegimeDetector saved to %s", out_path)

    def load(self, path: str) -> bool:
        """Load model state from disk.

        Supports both the current dict payload and the older tuple payload.
        """
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)  # noqa: S301  # nosec B301 - local trusted artifact from scripts/train_tcn.py

            if isinstance(payload, dict):
                self.model = payload["model"]
                self.fitted = bool(payload.get("fitted", False))
                self.regime_labels = dict(payload.get("regime_labels", {}))
                self._history_buffer = [list(row) for row in payload.get("history_buffer", [])]
                self._buffer_maxlen = int(payload.get("buffer_maxlen", self._buffer_maxlen))
            elif isinstance(payload, tuple) and len(payload) >= 3:
                self.model = payload[0]
                self.fitted = bool(payload[1])
                self.regime_labels = dict(payload[2])
                self.reset_buffer()
            else:
                raise ValueError("Unsupported regime payload format")

            logger.info("RegimeDetector loaded from %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to load RegimeDetector from %s: %s", path, exc)
            return False
