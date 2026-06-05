#!/usr/bin/env python3
"""Seed demo trading data for the dashboard and local walkthroughs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_agent.demo import seed_demo_workspace


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo Polymarket agent data")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--trades", type=int, default=36)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()

    bankroll = 1000.0
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}
        bankroll = float(config.get("bankroll", bankroll))
    except FileNotFoundError:
        pass

    result = seed_demo_workspace(
        root_dir=args.root,
        reset=not args.no_reset,
        trade_count=args.trades,
        bankroll=bankroll,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
