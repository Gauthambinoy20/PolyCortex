#!/usr/bin/env python3
"""Export trading data to CSV or Parquet for analysis.

Usage:
    python scripts/export_data.py --format csv --output exports/
    python scripts/export_data.py --format parquet --output exports/ --table trades
    python scripts/export_data.py --format csv --output exports/ --since 2025-01-01
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = "data/trades.db"
EXPORTABLE_TABLES = ["trades", "snapshots"]


def export_table(
    db_path: str,
    table: str,
    output_dir: Path,
    fmt: str = "csv",
    since: str | None = None,
) -> Path | None:
    """Export a single table to file.

    Args:
        db_path: Path to SQLite database.
        table: Table name to export.
        output_dir: Directory for output files.
        fmt: 'csv' or 'parquet'.
        since: Optional ISO date string to filter by timestamp.

    Returns:
        Path to exported file, or None if table is empty.
    """
    if table not in EXPORTABLE_TABLES:
        logger.error("Unknown table '%s'. Available: %s", table, EXPORTABLE_TABLES)
        return None

    conn = sqlite3.connect(db_path)

    query = f"SELECT * FROM {table}"  # noqa: S608 — table name is validated above
    params: list = []

    if since:
        query += " WHERE timestamp >= ?"
        params.append(since)

    query += " ORDER BY id ASC"

    try:
        df = pd.read_sql_query(query, conn, params=params)
    except Exception as exc:
        logger.error("Failed to read table '%s': %s", table, exc)
        conn.close()
        return None

    conn.close()

    if df.empty:
        logger.warning("Table '%s' is empty (no data to export)", table)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fmt == "parquet":
        filename = f"{table}_{timestamp_str}.parquet"
        filepath = output_dir / filename
        df.to_parquet(filepath, index=False)
    else:
        filename = f"{table}_{timestamp_str}.csv"
        filepath = output_dir / filename
        df.to_csv(filepath, index=False)

    logger.info("Exported %d rows from '%s' to %s", len(df), table, filepath)
    return filepath


def compute_pnl_curve(db_path: str) -> pd.DataFrame:
    """Compute cumulative P&L curve from trades."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT timestamp, pnl, direction, category, status FROM trades WHERE pnl IS NOT NULL ORDER BY id ASC",
        conn,
    )
    conn.close()

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["trade_number"] = range(1, len(df) + 1)

    # Running stats
    df["running_win_rate"] = (df["pnl"] > 0).expanding().mean()
    df["max_cumulative_pnl"] = df["cumulative_pnl"].expanding().max()
    df["drawdown"] = df["max_cumulative_pnl"] - df["cumulative_pnl"]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trading data")
    parser.add_argument(
        "--db", default=DEFAULT_DB, help=f"SQLite database path (default: {DEFAULT_DB})"
    )
    parser.add_argument(
        "--format", choices=["csv", "parquet"], default="csv", dest="fmt",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--output", default="exports/", help="Output directory (default: exports/)",
    )
    parser.add_argument(
        "--table", choices=EXPORTABLE_TABLES + ["all", "pnl_curve"], default="all",
        help="Table to export (default: all)",
    )
    parser.add_argument(
        "--since", default=None, help="Only export data after this date (ISO format: YYYY-MM-DD)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    db_path = args.db
    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    output_dir = Path(args.output)
    exported: list[Path] = []

    if args.table in ("all", "pnl_curve"):
        # Export P&L curve
        pnl_df = compute_pnl_curve(db_path)
        if not pnl_df.empty:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.fmt == "parquet":
                path = output_dir / f"pnl_curve_{ts}.parquet"
                pnl_df.to_parquet(path, index=False)
            else:
                path = output_dir / f"pnl_curve_{ts}.csv"
                pnl_df.to_csv(path, index=False)
            exported.append(path)
            logger.info("Exported P&L curve (%d rows) to %s", len(pnl_df), path)

    if args.table == "all":
        tables = EXPORTABLE_TABLES
    elif args.table == "pnl_curve":
        tables = []
    else:
        tables = [args.table]

    for table in tables:
        path = export_table(db_path, table, output_dir, args.fmt, args.since)
        if path:
            exported.append(path)

    if exported:
        print(f"\n✅ Exported {len(exported)} file(s) to {output_dir}/")
        for p in exported:
            print(f"   {p}")
    else:
        print("⚠️ No data exported (database may be empty)")


if __name__ == "__main__":
    main()
