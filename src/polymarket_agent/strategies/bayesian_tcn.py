"""Bayesian + optional TCN strategy (wraps UnifiedEdgeDetector-style logic)."""

from __future__ import annotations

from polymarket_agent.strategies.base import Signal, Strategy


class BayesianTCNStrategy(Strategy):
    """Bayesian signal combiner producing probability estimates.

    This is a lightweight strategy wrapper; the full Bayesian/TCN pipeline lives
    in :class:`polymarket_agent.models.edge.UnifiedEdgeDetector`.  This wrapper
    computes a simple ensemble of order-book imbalance, momentum, and a
    sentiment nudge so the strategy can be used standalone or registered.
    """

    name = "bayesian_tcn"

    def __init__(
        self,
        *,
        ob_weight: float = 0.15,
        momentum_weight: float = 0.10,
        sentiment_weight: float = 0.30,
        tcn_weight: float = 0.25,
    ) -> None:
        self.ob_weight = ob_weight
        self.momentum_weight = momentum_weight
        self.sentiment_weight = sentiment_weight
        self.tcn_weight = tcn_weight

    def score(self, market: dict) -> Signal:
        price = float(market.get("midpoint", market.get("best_bid", 0.5)))
        bid_depth = float(market.get("bid_depth", 0.0))
        ask_depth = float(market.get("ask_depth", 0.0))
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
        ob_adj = imbalance * self.ob_weight

        mom_adj = 0.0
        price_roc = float(market.get("price_roc_24h", 0.0))
        if price > 0.85 or price < 0.15:
            mom_adj = max(-0.03, min(0.03, price_roc * self.momentum_weight * 5.0))

        sentiment = float(market.get("sentiment_score", 0.0))
        sent_conf = float(market.get("sentiment_confidence", 0.0))
        sent_adj = sentiment * sent_conf * self.sentiment_weight

        tcn_prob = market.get("tcn_prob")
        combined = price + ob_adj + mom_adj + sent_adj
        if isinstance(tcn_prob, (int, float)):
            combined = combined * (1.0 - self.tcn_weight) + float(tcn_prob) * self.tcn_weight

        probability = max(0.01, min(0.99, combined))

        signals = [abs(ob_adj), abs(mom_adj), abs(sent_adj)]
        magnitude = sum(signals)
        if isinstance(tcn_prob, (int, float)):
            magnitude += abs(float(tcn_prob) - price) * self.tcn_weight
        confidence = min(1.0, magnitude * 5.0)

        return Signal(
            probability=probability,
            confidence=confidence,
            metadata={
                "order_book": ob_adj,
                "momentum": mom_adj,
                "sentiment": sent_adj,
                "tcn_used": tcn_prob is not None,
            },
        )
