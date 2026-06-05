#!/usr/bin/env python3
"""Collect price snapshots for all active Polymarket markets."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import yaml

# Ensure package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_agent.data.clob_client import ClobClient
from polymarket_agent.data.gamma_client import GammaClient
from polymarket_agent.data.history import HistoryStore
from polymarket_agent.utils.logger import setup_logging

logger = logging.getLogger(__name__)


async def collect_round(
    gamma: GammaClient,
    clob: ClobClient,
    history: HistoryStore,
) -> int:
    """Run one collection round and return the number of snapshots saved."""
    return await history.collect_all_active(gamma, clob)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect price history snapshots for active Polymarket markets",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Seconds between collection rounds (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of collection rounds (default: 1 for one-off)",
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO", json_output=not args.verbose)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gamma = GammaClient(
        base_url=config.get("gamma_api_url", "https://gamma-api.polymarket.com"),
    )
    clob = ClobClient(
        base_url=config.get("clob_api_url", "https://clob.polymarket.com"),
    )
    history = HistoryStore("data/price_history")

    total_collected = 0
    for rnd in range(1, args.rounds + 1):
        logger.info("Collection round %d/%d", rnd, args.rounds)
        start = time.monotonic()
        count = await collect_round(gamma, clob, history)
        elapsed = time.monotonic() - start
        total_collected += count
        logger.info("Collected %d market snapshots in %.1fs", count, elapsed)

        if rnd < args.rounds:
            logger.info("Sleeping %d seconds until next round...", args.interval)
            await asyncio.sleep(args.interval)

    await gamma.close()
    await clob.close()

    print(f"\nDone. Collected {total_collected} total snapshots across {args.rounds} round(s).")
    print(f"Markets with history: {len(history.list_markets())}")


if __name__ == "__main__":
    asyncio.run(main())
