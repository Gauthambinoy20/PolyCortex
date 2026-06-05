#!/usr/bin/env python3
"""CLI script to run the Polymarket backtesting pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import yaml

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Ensure the project root is importable (for backtest package)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtest.data_loader import BacktestDataLoader
from backtest.report import BacktestReporter
from backtest.simulator import WalkForwardBacktester

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    """Load config from YAML file, returning empty dict on failure."""
    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s — using defaults", config_path)
        return {}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Polymarket backtest")
    parser.add_argument(
        "--bankroll",
        type=float,
        default=100,
        help="Starting bankroll in USDC (default: 100)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic generated data",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use locally stored price history",
    )
    parser.add_argument(
        "--n-markets",
        type=int,
        default=20,
        help="Number of synthetic markets to generate (default: 20)",
    )
    parser.add_argument("--fee-bps", type=float, default=None, help="Override backtest fee bps")
    parser.add_argument("--slippage-bps", type=float, default=None, help="Override backtest slippage bps")
    args = parser.parse_args()

    if not args.synthetic and not args.local:
        parser.error("Specify --synthetic or --local")

    # Load config and override bankroll
    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(project_root, "config", "settings.yaml")
    config = load_config(config_path)
    config["bankroll"] = args.bankroll
    if args.fee_bps is not None:
        config["backtest_fee_bps"] = args.fee_bps
    if args.slippage_bps is not None:
        config["backtest_slippage_bps"] = args.slippage_bps

    # Load data
    data_dir = os.path.join(project_root, "data", "price_history")
    loader = BacktestDataLoader(data_dir=data_dir)

    if args.synthetic:
        markets = loader.load_synthetic(n_markets=args.n_markets)
        logger.info("Loaded %d synthetic markets", len(markets))
    else:
        markets = loader.load_from_local()
        logger.info("Loaded %d local markets", len(markets))

    if not markets:
        print("No markets loaded. Check data directory or use --synthetic.")
        sys.exit(1)

    # Run backtest
    backtester = WalkForwardBacktester(config)
    result = await backtester.run(markets, config)

    if result.n_trades == 0:
        print("No trades generated — check min_edge threshold or data quality")
        sys.exit(0)

    # Generate report
    output_dir = os.path.join(project_root, "data", "backtest_results")
    reporter = BacktestReporter(output_dir=output_dir)
    report_path = reporter.generate(result)

    # Print summary
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Total Return:   {result.total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:   {result.max_drawdown * 100:.1f}%")
    print(f"  Win Rate:       {result.win_rate * 100:.1f}%")
    print(f"  Brier Score:    {result.brier_score:.4f}")
    print(f"  Avg Edge:       {result.avg_edge * 100:.2f}%")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor != float("inf") else "∞"
    print(f"  Profit Factor:  {pf_str}")
    print(f"  Total Trades:   {result.n_trades}")

    if result.by_category:
        print("\n  By Category:")
        for cat, stats in sorted(result.by_category.items()):
            print(
                f"    {cat:15s}  trades={stats['n_trades']:3d}  "
                f"pnl={stats['total_pnl']:+8.2f}  "
                f"wr={stats['win_rate'] * 100:.0f}%"
            )

    print(f"\n  Report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
