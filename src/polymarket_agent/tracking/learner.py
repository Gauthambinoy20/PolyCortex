import json
import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class SelfAdjuster:
    MIN_EDGE_FLOOR = 0.04
    MIN_EDGE_CEILING = 0.20
    DECAY_RATE = 0.01  # Slow drift back toward config baseline each check

    def __init__(self, config: dict, weights_path: str = "data/learned_weights.json") -> None:
        self.min_edge: float = float(config.get("min_edge", 0.08))
        self._baseline_min_edge: float = self.min_edge
        self.signal_weights: dict[str, float] = dict(config.get("signal_weights", {}))
        self.brier_check_interval: int = int(config.get("brier_check_interval", 50))
        self.weights_path: str = weights_path
        self.adjustment_history: list[dict] = []

        self.load_weights()

    def check_and_adjust(self, metrics: dict) -> dict:
        trade_count: int = metrics.get("trade_count", 0)
        brier: float | None = metrics.get("brier_score")
        win_rate: float | None = metrics.get("win_rate")

        if trade_count > 0 and trade_count % self.brier_check_interval == 0:
            # Brier-based edge adjustment
            if brier is not None and brier > 0.30:
                self.min_edge = min(self.min_edge + 0.02, 0.20)
                logger.warning(
                    "High Brier score (%.4f) — raised min_edge to %.4f",
                    brier,
                    self.min_edge,
                )
            if brier is not None and brier < 0.15:
                self.min_edge = max(self.min_edge - 0.01, 0.05)
                logger.info(
                    "Low Brier score (%.4f) — lowered min_edge to %.4f",
                    brier,
                    self.min_edge,
                )

            # Win-rate-based edge adjustment
            if win_rate is not None and win_rate < 0.52 and trade_count > 20:
                self.min_edge = min(self.min_edge + 0.01, 0.20)
                logger.warning(
                    "Low win rate (%.2f%%) — raised min_edge to %.4f",
                    win_rate * 100,
                    self.min_edge,
                )
            if win_rate is not None and win_rate > 0.60 and trade_count > 20:
                self.min_edge = max(self.min_edge - 0.005, 0.05)
                logger.info(
                    "High win rate (%.2f%%) — lowered min_edge to %.4f",
                    win_rate * 100,
                    self.min_edge,
                )

        # Adjust signal weights based on per-signal accuracy
        per_signal_accuracy: dict[str, dict] = metrics.get("per_signal_accuracy", {})
        weights_changed = False
        for signal, stats in per_signal_accuracy.items():
            if signal not in self.signal_weights:
                continue
            count = stats.get("count", 0)
            accuracy = stats.get("accuracy", 0.5)

            if count > 20 and accuracy < 0.45:
                old_w = self.signal_weights[signal]
                self.signal_weights[signal] = old_w * 0.5
                weights_changed = True
                logger.warning(
                    "Signal '%s' accuracy %.2f%% — halved weight %.4f -> %.4f",
                    signal,
                    accuracy * 100,
                    old_w,
                    self.signal_weights[signal],
                )
            elif count > 20 and accuracy > 0.60:
                old_w = self.signal_weights[signal]
                self.signal_weights[signal] = min(old_w * 1.25, 0.40)
                weights_changed = True
                logger.info(
                    "Signal '%s' accuracy %.2f%% — boosted weight %.4f -> %.4f",
                    signal,
                    accuracy * 100,
                    old_w,
                    self.signal_weights[signal],
                )

        # Renormalize weights to sum to 1.0
        if weights_changed and self.signal_weights:
            total = sum(self.signal_weights.values())
            if total > 0:
                self.signal_weights = {k: v / total for k, v in self.signal_weights.items()}

        # Enforce hard bounds and decay toward baseline to prevent death spiral
        self.min_edge = max(self.MIN_EDGE_FLOOR, min(self.min_edge, self.MIN_EDGE_CEILING))
        self.min_edge = self.min_edge * (1.0 - self.DECAY_RATE) + self._baseline_min_edge * self.DECAY_RATE

        # Record adjustment
        self.adjustment_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "min_edge": self.min_edge,
                "signal_weights": dict(self.signal_weights),
            }
        )

        self.save_weights()

        return {
            "min_edge": self.min_edge,
            "signal_weights": dict(self.signal_weights),
        }

    def save_weights(self) -> None:
        os.makedirs(os.path.dirname(self.weights_path) or ".", exist_ok=True)
        data = {
            "min_edge": self.min_edge,
            "signal_weights": self.signal_weights,
        }
        with open(self.weights_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_weights(self) -> bool:
        if not os.path.exists(self.weights_path):
            return False
        try:
            with open(self.weights_path) as f:
                data = json.load(f)
            if "min_edge" in data:
                self.min_edge = float(data["min_edge"])
            if "signal_weights" in data and isinstance(data["signal_weights"], dict):
                self.signal_weights = {str(k): float(v) for k, v in data["signal_weights"].items()}
            logger.info("Loaded learned weights from %s", self.weights_path)
            return True
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("Failed to load weights from %s: %s", self.weights_path, exc)
            return False

    def get_current_settings(self) -> dict:
        return {
            "min_edge": self.min_edge,
            "signal_weights": dict(self.signal_weights),
        }
