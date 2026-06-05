"""Probability calibration for trading signals."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CalibrationFitMetrics:
    sample_count: int
    positive_rate: float
    brier_before: float
    brier_after: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ProbabilityCalibrator:
    """Calibrates raw model probabilities using isotonic regression."""

    SUPPORTED_METHODS = {"isotonic"}

    def __init__(
        self,
        *,
        path: str = "data/models/probability_calibration.json",
        enabled: bool = True,
        method: str = "isotonic",
        min_samples: int = 30,
        refit_interval: int = 25,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported calibration method: {method}")

        self.path = Path(path)
        self.enabled = enabled
        self.method = method
        self.min_samples = max(int(min_samples), 2)
        self.refit_interval = max(int(refit_interval), 1)
        self.x_thresholds: list[float] = []
        self.y_thresholds: list[float] = []
        self.sample_count: int = 0
        self.last_fit_trade_count: int = 0
        self.fitted_at: str | None = None

    @property
    def is_fitted(self) -> bool:
        return len(self.x_thresholds) >= 2 and len(self.x_thresholds) == len(self.y_thresholds)

    def should_refit(self, trade_count: int) -> bool:
        if not self.enabled or trade_count < self.min_samples:
            return False
        if not self.is_fitted:
            return True
        return trade_count >= self.last_fit_trade_count + self.refit_interval

    def calibrate(self, probability: float) -> float:
        clipped = self._clip_probability(probability)
        if not self.enabled or not self.is_fitted:
            return clipped

        calibrated = float(
            np.interp(
                clipped,
                self.x_thresholds,
                self.y_thresholds,
                left=self.y_thresholds[0],
                right=self.y_thresholds[-1],
            )
        )
        return self._clip_probability(calibrated)

    def fit(
        self,
        predictions: Iterable[float],
        outcomes: Iterable[float],
        *,
        trade_count: int | None = None,
    ) -> CalibrationFitMetrics | None:
        probs = np.asarray(list(predictions), dtype=float)
        labels = np.asarray(list(outcomes), dtype=float)

        mask = np.isfinite(probs) & np.isfinite(labels)
        probs = probs[mask]
        labels = labels[mask]

        if len(probs) < self.min_samples:
            logger.info(
                "Skipping probability calibration fit: only %d samples (need >= %d)",
                len(probs),
                self.min_samples,
            )
            return None

        probs = np.clip(probs, 0.01, 0.99)
        labels = np.clip(labels, 0.0, 1.0)

        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            logger.info("Skipping probability calibration fit: need both positive and negative outcomes")
            return None

        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        calibrated = model.fit_transform(probs, labels)

        self.x_thresholds = [float(v) for v in model.X_thresholds_]
        self.y_thresholds = [self._clip_probability(v) for v in model.y_thresholds_]
        self.sample_count = int(len(probs))
        self.last_fit_trade_count = int(trade_count if trade_count is not None else len(probs))
        self.fitted_at = datetime.now(UTC).isoformat()

        return CalibrationFitMetrics(
            sample_count=self.sample_count,
            positive_rate=float(labels.mean()),
            brier_before=float(np.mean((probs - labels) ** 2)),
            brier_after=float(np.mean((calibrated - labels) ** 2)),
        )

    def fit_from_trades(
        self,
        trades: Iterable[dict],
        *,
        trade_count: int | None = None,
    ) -> CalibrationFitMetrics | None:
        predictions: list[float] = []
        outcomes: list[float] = []

        for trade in trades:
            raw_prob = trade.get("raw_estimated_prob")
            if raw_prob is None:
                raw_prob = trade.get("estimated_prob")
            pnl = trade.get("pnl")
            if raw_prob is None or pnl is None:
                continue

            predictions.append(float(raw_prob))
            outcomes.append(1.0 if float(pnl) > 0 else 0.0)

        return self.fit(predictions, outcomes, trade_count=trade_count)

    def save(self) -> bool:
        if not self.enabled or not self.is_fitted:
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "method": self.method,
            "x_thresholds": self.x_thresholds,
            "y_thresholds": self.y_thresholds,
            "sample_count": self.sample_count,
            "last_fit_trade_count": self.last_fit_trade_count,
            "fitted_at": self.fitted_at,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True

    def load(self) -> bool:
        if not self.enabled or not self.path.exists():
            return False

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            method = str(payload["method"])
            x_thresholds = [float(v) for v in payload["x_thresholds"]]
            y_thresholds = [float(v) for v in payload["y_thresholds"]]
            if method != self.method:
                raise ValueError(
                    f"Calibration method mismatch: expected {self.method}, found {method}",
                )
            if len(x_thresholds) < 2 or len(x_thresholds) != len(y_thresholds):
                raise ValueError("Calibration thresholds are invalid")

            self.x_thresholds = [self._clip_probability(v) for v in x_thresholds]
            self.y_thresholds = [self._clip_probability(v) for v in y_thresholds]
            self.sample_count = int(payload.get("sample_count", len(self.x_thresholds)))
            self.last_fit_trade_count = int(
                payload.get("last_fit_trade_count", self.sample_count),
            )
            fitted_at = payload.get("fitted_at")
            self.fitted_at = str(fitted_at) if fitted_at is not None else None
            return True
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            logger.warning("Failed to load probability calibration from %s: %s", self.path, exc)
            return False

    @staticmethod
    def _clip_probability(probability: float) -> float:
        return float(np.clip(float(probability), 0.01, 0.99))
