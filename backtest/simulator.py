"""Walk-forward backtester for the Polymarket trading agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from backtest.data_loader import BacktestMarket
from polymarket_agent.features.engine import PolymarketFeatureEngine
from polymarket_agent.models.edge import UnifiedEdgeDetector
from polymarket_agent.models.regime import RegimeDetector
from polymarket_agent.risk.drawdown import DrawdownController
from polymarket_agent.risk.sizer import UnifiedPositionSizer

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series  # bankroll over time
    trades: pd.DataFrame  # all trades with columns below
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_edge: float
    brier_score: float
    profit_factor: float  # gross_profit / gross_loss (inf if no losses)
    n_trades: int
    by_category: dict  # {category: {n_trades, total_pnl, win_rate}}


class WalkForwardBacktester:
    """Walk-forward backtest engine.

    Splits each market's history into train/test portions, fits
    the regime detector on training data, then steps through the
    test portion generating trades.
    """

    def __init__(self, config: dict) -> None:
        self.train_window_days: int = config.get("train_window_days", 30)
        self.test_window_days: int = config.get("test_window_days", 7)
        self._config = config

    async def run(
        self,
        markets: list[BacktestMarket],
        config: dict | None = None,
    ) -> BacktestResult:
        """Execute the walk-forward simulation across all provided markets."""
        cfg = {**self._config, **(config or {})}
        bankroll: float = cfg.get("bankroll", 100)
        min_edge: float = cfg.get("min_edge", 0.08)
        apply_costs: bool = bool(cfg.get("backtest_apply_costs", True))
        fee_rate: float = float(cfg.get("backtest_fee_bps", 10)) / 10_000.0
        slippage_rate: float = float(cfg.get("backtest_slippage_bps", 5)) / 10_000.0
        spread_impact: float = float(cfg.get("backtest_spread_impact", 0.35))

        # Disable LLM sentiment — too expensive for backtest
        edge_cfg = {**cfg, "use_llm_sentiment": False}

        feature_engine = PolymarketFeatureEngine()
        regime_detector = RegimeDetector()
        edge_detector = UnifiedEdgeDetector(config=edge_cfg, sentiment_analyzer=None)
        sizer = UnifiedPositionSizer(cfg)
        drawdown_ctrl = DrawdownController(
            reduce_at=cfg.get("drawdown_reduce", 0.08),
            stop_at=cfg.get("drawdown_stop", 0.15),
            emergency_at=cfg.get("drawdown_emergency", 0.20),
        )

        drawdown_ctrl.update(bankroll)
        all_trades: list[dict[str, Any]] = []
        equity_points: list[tuple[datetime, float]] = []

        # Sort by resolution date for temporal ordering
        sorted_markets = sorted(markets, key=lambda m: m.resolution_date)

        for market in sorted_markets:
            df = market.price_history.copy()
            if len(df) < 64:
                logger.debug(
                    "Skipping %s: only %d rows (need 64)",
                    market.condition_id,
                    len(df),
                )
                continue

            # Split: 80% train, 20% test
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            if len(test_df) < 2:
                continue

            # Fit regime detector on training data
            if len(train_df) >= 100:
                try:
                    regime_detector.fit_from_history(train_df)
                except Exception as exc:
                    logger.warning(
                        "Regime fit failed for %s: %s", market.condition_id, exc
                    )

            # Walk through test portion
            open_positions: list[dict[str, Any]] = []

            for row_idx in range(len(test_df)):
                # Build features from all data up to this point
                history_so_far = pd.concat(
                    [train_df, test_df.iloc[: row_idx + 1]], ignore_index=True
                )
                features = feature_engine.create_features(
                    history_so_far, resolution_date=market.resolution_date
                )

                row = test_df.iloc[row_idx]
                midpoint = float(row["midpoint"])
                spread = float(row["spread"])
                bid_depth = float(row["bid_depth"])
                ask_depth = float(row["ask_depth"])
                volume = float(row["volume"])
                book_imbalance = float(row["book_imbalance"])

                # Regime prediction
                if regime_detector.fitted:
                    vol_20 = history_so_far["midpoint"].pct_change().rolling(20).std()
                    vol_sma = history_so_far["volume"].rolling(20).mean().clip(lower=1e-8)
                    vol_ratio = volume / (float(vol_sma.iloc[-1]) + 1e-8)
                    regime, _ = regime_detector.predict(
                        volatility=float(vol_20.iloc[-1]) if not np.isnan(vol_20.iloc[-1]) else 0.01,
                        volume_ratio=vol_ratio,
                        spread=spread,
                    )
                else:
                    regime = "stable"

                # Build market_data dict for edge detector
                market_data = {
                    "midpoint": midpoint,
                    "spread": spread,
                    "bid_depth": bid_depth,
                    "ask_depth": ask_depth,
                    "volume": volume,
                    "book_imbalance": book_imbalance,
                    "volume_24h": volume,
                    "market_id": market.condition_id,
                    "question": market.question,
                    "category": market.category,
                    "price_roc_24h": 0.0,
                    "related_markets": [],
                }

                # Calculate edge
                try:
                    edge_result = await edge_detector.estimate_edge(
                        market_data=market_data,
                        features=features,
                        regime=regime,
                    )
                except Exception as exc:
                    logger.warning("Edge estimation failed: %s", exc)
                    continue

                if edge_result.edge < min_edge:
                    continue

                # Check drawdown
                dd_mult = drawdown_ctrl.get_multiplier(bankroll)
                if dd_mult <= 0:
                    continue

                # Position sizing
                edge_dict = {
                    "estimated_prob": edge_result.estimated_prob,
                    "market_price": edge_result.market_price,
                    "direction": edge_result.direction,
                    "confidence": edge_result.confidence,
                    "regime": regime,
                    "category": market.category,
                    "signal_breakdown": edge_result.signal_breakdown,
                }
                size = sizer.calculate_position(
                    edge_result=edge_dict,
                    current_positions=open_positions,
                    drawdown_multiplier=dd_mult,
                )
                if size <= 0:
                    continue

                depth = max(bid_depth + ask_depth, 1.0)
                liquidity_impact = min(size / depth, 1.0) * spread * spread_impact
                execution_slippage = slippage_rate + liquidity_impact if apply_costs else 0.0
                entry_price = midpoint
                if edge_result.direction == "YES":
                    entry_price = min(0.99, midpoint + execution_slippage)
                else:
                    entry_price = max(0.01, midpoint - execution_slippage)
                trade_cost = size * (fee_rate + spread * spread_impact * 0.25) if apply_costs else 0.0

                ts = row.get("timestamp", row.name) if "timestamp" in test_df.columns else row_idx
                open_positions.append({
                    "market_id": market.condition_id,
                    "direction": edge_result.direction,
                    "entry_price": entry_price,
                    "size": size,
                    "trade_cost": trade_cost,
                    "slippage": execution_slippage,
                    "edge": edge_result.edge,
                    "confidence": edge_result.confidence,
                    "regime": regime,
                    "category": market.category,
                    "estimated_prob": edge_result.estimated_prob,
                    "timestamp": ts,
                })

            # Resolve all open positions for this market
            for pos in open_positions:
                if pos["direction"] == "YES":
                    pnl = (
                        (market.resolution - pos["entry_price"]) * pos["size"] / pos["entry_price"]
                        if pos["entry_price"]
                        else 0.0
                    )
                else:
                    pnl = (
                        (pos["entry_price"] - market.resolution) * pos["size"] / (1 - pos["entry_price"])
                        if (1 - pos["entry_price"])
                        else 0.0
                    )
                pnl -= pos.get("trade_cost", 0.0)

                bankroll += pnl
                drawdown_ctrl.update(bankroll)
                sizer.update_bankroll(bankroll)

                all_trades.append({
                    "market_id": pos["market_id"],
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": market.resolution,
                    "pnl": pnl,
                    "size": pos["size"],
                    "trade_cost": pos.get("trade_cost", 0.0),
                    "slippage": pos.get("slippage", 0.0),
                    "edge": pos["edge"],
                    "confidence": pos["confidence"],
                    "regime": pos["regime"],
                    "category": pos["category"],
                    "estimated_prob": pos["estimated_prob"],
                    "timestamp": pos["timestamp"],
                })

                equity_points.append((market.resolution_date, bankroll))

        # Build result dataframes
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
        else:
            trades_df = pd.DataFrame(
                columns=[
                    "market_id", "direction", "entry_price", "exit_price",
                    "pnl", "size", "edge", "confidence", "regime", "category",
                    "estimated_prob", "timestamp",
                ]
            )

        initial_bankroll = cfg.get("bankroll", 100)
        if equity_points:
            eq_dates, eq_vals = zip(*equity_points, strict=True)
            equity_curve = pd.Series(data=list(eq_vals), index=pd.DatetimeIndex(eq_dates))
            # Prepend initial bankroll
            equity_curve = pd.concat([
                pd.Series([initial_bankroll], index=[equity_curve.index[0]]),
                equity_curve,
            ])
        else:
            equity_curve = pd.Series([initial_bankroll], index=[pd.Timestamp.now(tz="UTC")])

        total_return = (bankroll - initial_bankroll) / initial_bankroll
        sharpe = self._compute_sharpe(equity_curve)
        max_dd = self._compute_max_drawdown(equity_curve)

        n_trades = len(all_trades)
        win_rate = 0.0
        avg_edge = 0.0
        profit_factor = 0.0

        if n_trades > 0:
            wins = sum(1 for t in all_trades if t["pnl"] > 0)
            win_rate = wins / n_trades
            avg_edge = float(np.mean([t["edge"] for t in all_trades]))

            gross_profit = sum(t["pnl"] for t in all_trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in all_trades if t["pnl"] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Brier score
        market_map = {m.condition_id: m.resolution for m in markets}
        brier = self._compute_brier(trades_df, market_map) if n_trades > 0 else 0.0

        # Per-category breakdown
        by_category: dict[str, dict] = {}
        if n_trades > 0:
            for cat, group in trades_df.groupby("category"):
                cat_wins = (group["pnl"] > 0).sum()
                by_category[str(cat)] = {
                    "n_trades": len(group),
                    "total_pnl": float(group["pnl"].sum()),
                    "win_rate": float(cat_wins / len(group)) if len(group) > 0 else 0.0,
                }

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades_df,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            avg_edge=avg_edge,
            brier_score=brier,
            profit_factor=profit_factor,
            n_trades=n_trades,
            by_category=by_category,
        )

    @staticmethod
    def _compute_sharpe(equity_curve: pd.Series, risk_free: float = 0.0) -> float:
        """Annualized Sharpe ratio using calendar-day resampled returns."""
        if len(equity_curve) < 2:
            return 0.0
        # Resample to daily to get consistent time intervals
        daily = equity_curve.resample("1D").last().ffill().dropna()
        if len(daily) < 2:
            return 0.0
        returns = daily.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float((returns.mean() - risk_free) / returns.std() * np.sqrt(252))

    @staticmethod
    def _compute_max_drawdown(equity_curve: pd.Series) -> float:
        """Peak-to-trough drawdown as a positive fraction."""
        if len(equity_curve) < 2:
            return 0.0
        cummax = equity_curve.cummax()
        drawdown = (cummax - equity_curve) / cummax
        return float(drawdown.max())

    @staticmethod
    def _compute_brier(
        trades_df: pd.DataFrame,
        market_resolutions: dict[str, float],
    ) -> float:
        """Mean Brier score: (estimated_prob - actual_resolution)^2."""
        if trades_df.empty or "estimated_prob" not in trades_df.columns:
            return 0.0

        scores: list[float] = []
        for _, trade in trades_df.iterrows():
            actual = market_resolutions.get(trade["market_id"])
            if actual is None:
                continue
            scores.append((trade["estimated_prob"] - actual) ** 2)

        return float(np.mean(scores)) if scores else 0.0
