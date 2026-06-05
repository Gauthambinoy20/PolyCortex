import logging

logger = logging.getLogger(__name__)


class UnifiedPositionSizer:
    REGIME_MULTIPLIERS: dict[str, float] = {
        "stable": 1.0,
        "trending": 0.8,
        "volatile": 0.5,
    }

    def __init__(self, config: dict) -> None:
        self.bankroll: float = config.get("bankroll", 100)
        self.kelly_fraction: float = config.get("kelly_fraction", 0.25)
        self.max_position_pct: float = config.get("max_position_pct", 0.02)
        self.max_portfolio_pct: float = config.get("max_portfolio_pct", 0.20)
        self.max_category_exposure_pct: float = config.get(
            "max_category_exposure_pct",
            self.max_portfolio_pct,
        )
        self.max_positions: int = config.get("max_positions", 10)
        self.max_per_category: int = config.get("max_per_category", 3)
        self.min_confidence: float = config.get("min_confidence", 0.0)
        self.use_uncertainty_bands: bool = config.get("use_uncertainty_bands", True)
        self.uncertainty_z_score: float = config.get("uncertainty_z_score", 1.5)
        self.min_position: float = 5.0
        self.max_single_loss_usdc: float | None = config.get("max_single_loss_usdc")

    def update_bankroll(self, new_bankroll: float) -> None:
        self.bankroll = new_bankroll

    def calculate_position(
        self,
        edge_result: dict,
        current_positions: list[dict],
        drawdown_multiplier: float = 1.0,
    ) -> float:
        estimated_prob: float = edge_result["estimated_prob"]
        market_price: float = edge_result["market_price"]
        direction: str = edge_result["direction"]
        confidence: float = edge_result["confidence"]
        regime: str = edge_result["regime"]
        category: str = edge_result["category"]
        signal_breakdown: dict = edge_result["signal_breakdown"]
        uncertainty: float = edge_result.get("uncertainty", 0.0)

        if confidence < self.min_confidence:
            return 0.0

        # 1. Kelly criterion — clamp price to avoid division by zero
        market_price = max(0.01, min(market_price, 0.99))
        if direction == "YES":
            b = (1 - market_price) / market_price
            p = estimated_prob
        else:
            b = market_price / (1 - market_price)
            p = 1 - estimated_prob
        q = 1 - p
        kelly = (b * p - q) / (b + 1e-8)
        if kelly <= 0:
            return 0.0

        # Uncertainty-adjusted Kelly: use lower bound of edge confidence interval
        if self.use_uncertainty_bands and uncertainty > 0:
            # Adjust probability estimate by uncertainty
            prob_lower = max(0.01, p - self.uncertainty_z_score * uncertainty)
            kelly_lower = (b * prob_lower - (1 - prob_lower)) / (b + 1e-8)
            if kelly_lower <= 0:
                logger.info(
                    "Kelly lower bound ≤ 0 (kelly=%.4f, kelly_lower=%.4f, uncertainty=%.4f) — skipping",
                    kelly,
                    kelly_lower,
                    uncertainty,
                )
                return 0.0
            # Use the more conservative lower-bound Kelly
            kelly = kelly_lower
            logger.debug(
                "Uncertainty-adjusted Kelly: %.4f (z=%.1f, uncertainty=%.4f)",
                kelly,
                self.uncertainty_z_score,
                uncertainty,
            )

        # 2. Fractional Kelly
        position_frac = kelly * self.kelly_fraction

        # 3. Confidence scaling
        position_frac *= confidence

        # 4. Spread-based liquidity scaling — wider spreads reduce size
        spread: float = signal_breakdown.get("spread", 0.0) if isinstance(signal_breakdown, dict) else 0.0
        if spread > 0.05:
            position_frac *= max(0.5, 1.0 - (spread - 0.05) * 5.0)

        # 5. Regime multiplier
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 0.5)
        position_frac *= regime_mult

        # 6. Convert to USDC
        position_usdc = position_frac * self.bankroll

        # 7. Max per trade
        position_usdc = min(position_usdc, self.bankroll * self.max_position_pct)

        # 8. Portfolio capacity
        total = sum(p.get("size", p.get("size_usdc", 0.0)) for p in current_positions)
        remaining = self.bankroll * self.max_portfolio_pct - total
        position_usdc = min(position_usdc, max(remaining, 0))

        # 9. Max positions
        if len(current_positions) >= self.max_positions:
            return 0.0

        # 10. Category limit
        category_count = sum(1 for p in current_positions if p.get("category") == category)
        if category_count >= self.max_per_category:
            return 0.0

        category_exposure = sum(
            p.get("size", p.get("size_usdc", 0.0)) for p in current_positions if p.get("category") == category
        )
        category_remaining = self.bankroll * self.max_category_exposure_pct - category_exposure
        position_usdc = min(position_usdc, max(category_remaining, 0.0))

        # 11. Max single-loss hard cap (default: 5% of bankroll)
        effective_max_loss = (
            self.max_single_loss_usdc if self.max_single_loss_usdc is not None else self.bankroll * 0.05
        )
        if position_usdc > effective_max_loss:
            if self.max_single_loss_usdc is not None:
                logger.warning(
                    "Position $%.2f clipped to max single loss cap $%.2f",
                    position_usdc,
                    effective_max_loss,
                )
                position_usdc = effective_max_loss
            else:
                logger.warning(
                    "Position $%.2f exceeds recommended max single loss $%.2f "
                    "(set max_single_loss_usdc in config to enforce)",
                    position_usdc,
                    effective_max_loss,
                )

        # 12. Drawdown
        position_usdc *= drawdown_multiplier

        # 13. Minimum
        if position_usdc < self.min_position:
            return 0.0

        constraints = []
        if self.use_uncertainty_bands and uncertainty > 0:
            constraints.append("uncertainty_bands")
        if regime_mult < 1.0:
            constraints.append(f"regime_{regime}")
        if spread > 0.05:
            constraints.append("wide_spread")
        if drawdown_multiplier < 1.0:
            constraints.append("drawdown_scaling")
        if constraints:
            logger.debug(
                "Position $%.2f constrained by: %s",
                position_usdc,
                ", ".join(constraints),
            )

        return round(position_usdc, 2)
