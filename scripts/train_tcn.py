#!/usr/bin/env python3
"""Train the TCN ensemble model on collected price history."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Ensure package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from polymarket_agent.data.history import HistoryStore
from polymarket_agent.features.engine import PolymarketFeatureEngine
from polymarket_agent.models.tcn import PolymarketEnsemble, train_tcn
from polymarket_agent.utils.logger import setup_logging

logger = logging.getLogger(__name__)

WINDOW_SIZE = 64
TARGET_OFFSET = 1  # predict price at step window+1


def build_dataset(
    history: HistoryStore,
    engine: PolymarketFeatureEngine,
    min_length: int = 100,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Build sliding-window arrays from all markets with enough history."""
    all_windows: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    used_markets: list[str] = []

    for cid in history.list_markets():
        hist = history.load_history(cid)
        if hist is None or len(hist) < min_length:
            continue

        features = engine.create_features(hist)
        if features is None or len(features) < WINDOW_SIZE + TARGET_OFFSET:
            continue

        midpoints = hist["midpoint"].values
        # Align midpoints with feature rows (features drop some leading rows due to rolling)
        n_features = len(features)
        midpoints_aligned = midpoints[len(midpoints) - n_features :]

        # Create sliding windows
        for i in range(n_features - WINDOW_SIZE - TARGET_OFFSET + 1):
            window = features[i : i + WINDOW_SIZE]
            target = midpoints_aligned[i + WINDOW_SIZE]
            if np.isnan(window).any() or np.isnan(target):
                continue
            all_windows.append(window)
            all_targets.append(np.array([target]))

        used_markets.append(cid)

    return all_windows, all_targets, used_markets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the TCN ensemble on collected price history",
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--min-markets",
        type=int,
        default=5,
        help="Minimum markets with sufficient data (default: 5)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(level="DEBUG" if args.verbose else "INFO", json_output=not args.verbose)

    history = HistoryStore("data/price_history")
    engine = PolymarketFeatureEngine()

    logger.info("Building dataset from collected history...")
    windows, targets, used_markets = build_dataset(history, engine)

    if len(used_markets) < args.min_markets:
        print(
            f"Error: Only {len(used_markets)} markets have enough data "
            f"(need {args.min_markets}). Collect more history first."
        )
        sys.exit(1)

    if not windows:
        print("Error: No valid training windows could be created.")
        sys.exit(1)

    X = np.stack(windows).astype(np.float32)  # (N, 64, 10)
    y = np.concatenate(targets).astype(np.float32)  # (N,)

    logger.info(
        "Dataset: %d windows from %d markets, feature shape=%s",
        len(X), len(used_markets), X.shape,
    )

    # Time-based split: earlier 80% train, later 20% val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    logger.info("Train: %d samples, Val: %d samples", len(X_train), len(X_val))

    # Create model
    n_features = X.shape[2]
    model = PolymarketEnsemble(n_features=n_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Training on %s", device)

    save_path = "data/models/tcn_latest.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    results = train_tcn(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        save_path=save_path,
    )

    print("\nTraining complete!")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Best val loss (Brier): {results['best_val_loss']:.6f}")
    print(f"  Final train loss: {results['train_losses'][-1]:.6f}")
    print(f"  Final val loss: {results['val_losses'][-1]:.6f}")
    print(f"  Model saved to: {save_path}")
    print(f"  Markets used: {len(used_markets)}")


if __name__ == "__main__":
    main()
