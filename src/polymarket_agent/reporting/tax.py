"""Tax reporting: IRS Form 8949 + generic trades CSV.

Reads closed trades from the tracker SQLite database and produces two
CSV files:

* ``form_8949_<year>.csv`` — IRS-compatible Form 8949 short-term schedule
* ``trades_<year>.csv`` — a generic trades export useful for spreadsheets

Cost basis is computed using the requested method (default FIFO) per
``market_id`` + ``direction``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)

FORM_8949_HEADER = [
    "description",
    "date_acquired",
    "date_sold",
    "proceeds",
    "cost_basis",
    "gain_loss",
]

TRADES_CSV_HEADER = [
    "id",
    "market_id",
    "question",
    "direction",
    "entry_price",
    "exit_price",
    "size_usdc",
    "pnl",
    "timestamp",
    "closed_at",
    "is_paper",
]


def _connect(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _iter_closed_trades(conn: sqlite3.Connection, year: int) -> list[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, market_id, question, direction, entry_price, size_usdc,
               timestamp, status, exit_price, pnl, closed_at, is_paper, category
        FROM trades
        WHERE status = 'closed'
          AND closed_at IS NOT NULL
          AND substr(closed_at, 1, 4) = ?
        ORDER BY closed_at ASC
        """,
        (str(year),),
    )
    return list(cur.fetchall())


def generate_form_8949(
    year: int,
    method: str = "fifo",
    db_path: str = "data/trades.db",
    output_dir: str = "reports",
) -> Path:
    """Generate IRS Form 8949 CSV for the given tax year.

    Args:
        year: Tax year (e.g. 2026).
        method: Cost-basis method. ``fifo`` or ``lifo`` supported.
        db_path: SQLite database path.
        output_dir: Directory to write CSV into.
    """
    method = method.lower()
    if method not in {"fifo", "lifo"}:
        raise ValueError(f"Unsupported method: {method}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"form_8949_{year}.csv"

    with _connect(db_path) as conn:
        rows = _iter_closed_trades(conn, year)

    # Group by (market_id, direction) and build cost-basis lots
    lots: dict[tuple[str, str], deque] = defaultdict(deque)
    entries: list[dict] = []

    for row in rows:
        key = (row["market_id"] or "", row["direction"] or "")
        size = float(row["size_usdc"] or 0.0)
        entry_price = float(row["entry_price"] or 0.0)
        exit_price = float(row["exit_price"] or 0.0)

        # Every row is a full round-trip in our schema; treat entry as acquire
        # and exit as sell.  This keeps FIFO/LIFO meaningful if the same market
        # was traded multiple times.
        lots[key].append(
            {
                "date_acquired": (row["timestamp"] or "")[:10],
                "cost_basis": size,
                "entry_price": entry_price,
            }
        )

        # Pop lot using chosen method
        if lots[key]:
            lot = lots[key].popleft() if method == "fifo" else lots[key].pop()
        else:
            lot = {"date_acquired": (row["timestamp"] or "")[:10], "cost_basis": size}

        proceeds = size * (exit_price / entry_price) if entry_price > 0 else size + float(row["pnl"] or 0.0)
        cost_basis = lot["cost_basis"]
        gain_loss = proceeds - cost_basis

        entries.append(
            {
                "description": f"{row['market_id']} {row['direction']}: {(row['question'] or '')[:60]}",
                "date_acquired": lot["date_acquired"],
                "date_sold": (row["closed_at"] or "")[:10],
                "proceeds": round(proceeds, 4),
                "cost_basis": round(cost_basis, 4),
                "gain_loss": round(gain_loss, 4),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FORM_8949_HEADER)
        writer.writeheader()
        writer.writerows(entries)

    logger.info("Wrote %d rows to %s", len(entries), out_path)
    return out_path


def generate_trades_csv(
    year: int,
    db_path: str = "data/trades.db",
    output_dir: str = "reports",
) -> Path:
    """Generate a generic trades export for *year*."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trades_{year}.csv"

    with _connect(db_path) as conn:
        rows = _iter_closed_trades(conn, year)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRADES_CSV_HEADER)
        writer.writeheader()
        for row in rows:
            row_keys = row.keys()
            writer.writerow({k: row[k] for k in TRADES_CSV_HEADER if k in row_keys})

    logger.info("Wrote %d rows to %s", len(rows), out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tax reports from trade history")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--method", default="fifo", choices=["fifo", "lifo"])
    parser.add_argument("--db", default="data/trades.db")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--format", default="both", choices=["8949", "trades", "both"])
    args = parser.parse_args()

    if args.format in ("8949", "both"):
        generate_form_8949(args.year, args.method, args.db, args.output_dir)
    if args.format in ("trades", "both"):
        generate_trades_csv(args.year, args.db, args.output_dir)


if __name__ == "__main__":
    main()
