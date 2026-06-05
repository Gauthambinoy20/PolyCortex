"""Unified edge detector combining Bayesian signals with optional TCN."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import log1p
from typing import TYPE_CHECKING, cast

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from polymarket_agent.models.calibration import ProbabilityCalibrator
from polymarket_agent.models.tcn import PolymarketEnsemble

if TYPE_CHECKING:
    import pandas as pd

    from polymarket_agent.data.gamma_client import Market
    from polymarket_agent.data.history import HistoryStore
    from polymarket_agent.features.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    estimated_prob: float
    market_price: float
    edge: float
    direction: str
    confidence: float
    regime: str
    market_id: str
    question: str
    category: str
    signal_breakdown: dict[str, float]
    raw_estimated_prob: float | None = None
    calibrated_prob: float | None = None


class UnifiedEdgeDetector:
    """Combines order-book, momentum, sentiment, cross-market, and TCN signals."""

    def __init__(
        self,
        config: dict,
        sentiment_analyzer: SentimentAnalyzer | None = None,
    ) -> None:
        self.min_edge: float = config.get("min_edge", 0.08)
        self.use_llm_sentiment: bool = config.get("use_llm_sentiment", True)
        self.regime_weights: dict[str, dict[str, float]] = config.get("regime_weights", {})
        self.signal_weights: dict[str, float] = config.get(
            "signal_weights",
            {
                "order_book": 0.15,
                "momentum": 0.10,
                "sentiment": 0.30,
                "news_volume": 0.10,
                "cross_market": 0.10,
                "tcn_model": 0.25,
            },
        )
        self.sentiment: SentimentAnalyzer | None = sentiment_analyzer
        self.tcn_model: PolymarketEnsemble | None = None
        self.probability_calibrator: ProbabilityCalibrator | None = None

    def set_tcn_model(self, model: PolymarketEnsemble) -> None:
        self.tcn_model = model

    def set_probability_calibrator(self, calibrator: ProbabilityCalibrator) -> None:
        self.probability_calibrator = calibrator

    async def estimate_edge(
        self,
        market_data: dict,
        features: np.ndarray | None,
        regime: str,
    ) -> EdgeResult:
        market_price: float = market_data["midpoint"]
        signals: dict[str, float] = {}
        w = self.signal_weights

        # 1. Order book signal
        bid_depth: float = market_data.get("bid_depth", 0.0)
        ask_depth: float = market_data.get("ask_depth", 0.0)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
        ob_adj = imbalance * w.get("order_book", 0.15)
        signals["order_book"] = ob_adj

        # 2. Momentum signal (extreme markets only)
        mom_adj = 0.0
        if market_price > 0.85 or market_price < 0.15:
            price_roc_24h: float = market_data.get("price_roc_24h", 0.0)
            mom_adj = float(np.clip(price_roc_24h * w.get("momentum", 0.10) * 5.0, -0.03, 0.03))
        signals["momentum"] = mom_adj

        # 3. Sentiment signal
        sent_adj = 0.0
        if self.sentiment and self.use_llm_sentiment:
            cached = self.sentiment.get_cached(market_data.get("market_id", ""))
            if cached is not None:
                sent_adj = cached.score * cached.confidence * w.get("sentiment", 0.30)
        signals["sentiment"] = sent_adj

        # 4. News volume amplification
        news_volume_ratio: float = market_data.get("news_volume_ratio", 1.0)
        if news_volume_ratio > 2.0:
            amplifier = min(news_volume_ratio / 2.0, 2.0)
            sent_adj *= amplifier
            signals["sentiment"] = sent_adj

        # 5. Cross-market signal
        related_markets: list[dict] = market_data.get("related_markets", [])
        cross_signal = self._calc_cross_signal(market_price, related_markets)
        cross_signal *= w.get("cross_market", 0.10)
        signals["cross_market"] = cross_signal

        # 6. TCN signal
        tcn_prob: float | None = None
        if torch is not None and self.tcn_model is not None and features is not None and len(features) >= 64:
            try:
                self.tcn_model.eval()
                with torch.no_grad():
                    tensor = torch.tensor(
                        features[-64:],
                        dtype=torch.float32,
                    ).unsqueeze(0)
                    prob, _ = self.tcn_model(tensor)
                    tcn_prob = float(prob.item())
                signals["tcn"] = tcn_prob
            except Exception as exc:
                logger.warning("TCN inference failed: %s", exc)
                tcn_prob = None

        # 7. Combine Bayesian
        bayesian_prob = float(np.clip(market_price + ob_adj + mom_adj + sent_adj + cross_signal, 0.01, 0.99))

        # 8. Blend with TCN using regime weights
        if tcn_prob is not None:
            rw = self.regime_weights.get(regime, {"bayesian": 0.6, "tcn": 0.4})
            w_bayes = rw.get("bayesian", 0.6)
            w_tcn = rw.get("tcn", 0.4)
            combined = bayesian_prob * w_bayes + tcn_prob * w_tcn
        else:
            combined = bayesian_prob

        # 9. Volume-based confidence scaling
        # High-volume markets are more efficient, so we require a higher signal
        # magnitude to trust our estimate — but we do NOT regress toward the
        # market price (which would suppress all edges in liquid markets).
        volume_confidence = min(log1p(float(market_data.get("volume_24h", 0.0))) / 15.0, 0.3)
        raw_prob = combined
        calibrated_prob = raw_prob
        if self.probability_calibrator is not None:
            calibrated_prob = self.probability_calibrator.calibrate(raw_prob)

        # 10. Edge and direction
        edge = calibrated_prob - market_price
        direction = "YES" if edge > 0 else "NO"

        # 11. Confidence — magnitude-weighted signal agreement
        nonzero_signals = {k: v for k, v in signals.items() if v != 0.0}
        if nonzero_signals:
            total_magnitude = sum(abs(v) for v in nonzero_signals.values())
            agreement_magnitude = sum(abs(v) for v in nonzero_signals.values() if (v > 0) == (edge > 0))
            confidence = agreement_magnitude / (total_magnitude + 1e-8)
        else:
            confidence = 0.0
        confidence *= max(0.7, 1.0 - volume_confidence)

        return EdgeResult(
            estimated_prob=calibrated_prob,
            market_price=market_price,
            edge=abs(edge),
            direction=direction,
            confidence=confidence,
            regime=regime,
            market_id=market_data.get("market_id", ""),
            question=market_data.get("question", ""),
            category=market_data.get("category", ""),
            signal_breakdown=signals,
            raw_estimated_prob=raw_prob,
            calibrated_prob=calibrated_prob,
        )

    def _calc_cross_signal(
        self,
        market_price: float,
        related_markets: list[dict],
    ) -> float:
        if not related_markets:
            return 0.0
        signals: list[float] = []
        for rm in related_markets:
            price_change = rm.get("price_change_24h", 0.0)
            correlation = rm.get("correlation", 0.0)
            signals.append(price_change * correlation)
        return float(np.clip(np.mean(signals), -0.05, 0.05))

    def get_cross_market_signals(
        self,
        market: Market,
        all_markets: list[Market],
        history_store: HistoryStore,
    ) -> list[dict]:
        same_category = [
            m for m in all_markets if m.category == market.category and m.condition_id != market.condition_id
        ]

        our_history = history_store.load_history(
            market.condition_id,
            lookback_hours=7 * 24,
        )
        if our_history is None or our_history.empty:
            return []

        our_prices = our_history["midpoint"]
        results: list[dict] = []

        for other in same_category:
            try:
                other_history = history_store.load_history(
                    other.condition_id,
                    lookback_hours=7 * 24,
                )
                if other_history is None or other_history.empty:
                    continue

                other_prices = other_history["midpoint"]
                min_len = min(len(our_prices), len(other_prices))
                if min_len < 5:
                    continue

                corr = float(
                    np.corrcoef(
                        our_prices.iloc[-min_len:].values,
                        other_prices.iloc[-min_len:].values,
                    )[0, 1]
                )
                if np.isnan(corr):
                    continue

                price_change_24h = 0.0
                if len(other_prices) >= 2:
                    lookback = min(len(other_prices), 24)
                    price_change_24h = float(other_prices.iloc[-1] - other_prices.iloc[-lookback])

                results.append(
                    {
                        "condition_id": other.condition_id,
                        "price_change_24h": price_change_24h,
                        "correlation": corr,
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Error computing cross-signal for %s: %s",
                    other.condition_id,
                    exc,
                )

        results.sort(key=lambda r: abs(r["correlation"]), reverse=True)
        return results[:5]

    def precompute_cross_signals(
        self,
        markets: list[Market],
        history_store: HistoryStore,
    ) -> dict[str, list[dict]]:
        """Batch-compute cross-market signals for all markets at once.

        Loads each market's price history only once, avoiding the O(N²)
        Parquet reads of calling ``get_cross_market_signals`` per market.
        """
        # Load histories once
        histories: dict[str, object] = {}
        for m in markets:
            h = history_store.load_history(m.condition_id, lookback_hours=7 * 24)
            if h is not None and not h.empty:
                histories[m.condition_id] = h["midpoint"]

        # Group by category
        by_category: dict[str, list[Market]] = {}
        for m in markets:
            by_category.setdefault(m.category, []).append(m)

        result: dict[str, list[dict]] = {}
        for _category, cat_markets in by_category.items():
            for market in cat_markets:
                if market.condition_id not in histories:
                    result[market.condition_id] = []
                    continue
                our_prices = cast("pd.Series", histories[market.condition_id])
                signals: list[dict] = []
                for other in cat_markets:
                    if other.condition_id == market.condition_id:
                        continue
                    if other.condition_id not in histories:
                        continue
                    other_prices = cast("pd.Series", histories[other.condition_id])
                    min_len = min(len(our_prices), len(other_prices))
                    if min_len < 5:
                        continue
                    try:
                        corr = float(
                            np.corrcoef(
                                our_prices.iloc[-min_len:].values,
                                other_prices.iloc[-min_len:].values,
                            )[0, 1]
                        )
                        if np.isnan(corr):
                            continue
                        price_change_24h = 0.0
                        if len(other_prices) >= 2:
                            lookback = min(len(other_prices), 24)
                            price_change_24h = float(other_prices.iloc[-1] - other_prices.iloc[-lookback])
                        signals.append(
                            {
                                "condition_id": other.condition_id,
                                "price_change_24h": price_change_24h,
                                "correlation": corr,
                            }
                        )
                    except Exception:
                        logger.debug(
                            "Skipping cross-signal correlation failure for %s vs %s",
                            market.condition_id,
                            other.condition_id,
                            exc_info=True,
                        )
                        continue
                signals.sort(key=lambda r: abs(r["correlation"]), reverse=True)
                result[market.condition_id] = signals[:5]
        return result
