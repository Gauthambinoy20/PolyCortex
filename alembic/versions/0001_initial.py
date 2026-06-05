"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-21 00:00:00.000000

"""
from __future__ import annotations

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT,
    question TEXT,
    direction TEXT,
    entry_price REAL,
    size_usdc REAL,
    timestamp TEXT,
    status TEXT DEFAULT 'open',
    exit_price REAL,
    pnl REAL,
    edge_at_entry REAL,
    confidence_at_entry REAL,
    raw_estimated_prob REAL,
    estimated_prob REAL,
    calibrated_prob REAL,
    regime_at_entry TEXT,
    category TEXT,
    signal_breakdown TEXT,
    is_paper INTEGER DEFAULT 1,
    closed_at TEXT,
    entry_reason TEXT,
    exit_reason TEXT,
    order_type TEXT DEFAULT 'limit',
    source TEXT DEFAULT 'agent',
    exchange_order_ids TEXT,
    realized_roi REAL,
    notes TEXT,
    local_order_id TEXT,
    entry_order_status TEXT DEFAULT 'filled',
    filled_size_usdc REAL,
    remaining_size_usdc REAL,
    average_fill_price REAL,
    parent_order_id TEXT,
    oco_group_id TEXT,
    trigger_price REAL,
    trigger_condition TEXT,
    take_profit_price REAL,
    stop_loss_price REAL,
    bracket_order_ids TEXT,
    bracket_state TEXT DEFAULT 'inactive',
    last_fill_at TEXT,
    reduce_only INTEGER DEFAULT 0,
    order_kind TEXT DEFAULT 'entry'
)
"""

_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    bankroll REAL,
    open_positions INTEGER,
    daily_pnl REAL,
    metadata TEXT
)
"""


def upgrade() -> None:
    # Postgres uses SERIAL instead of AUTOINCREMENT
    dialect = op.get_bind().dialect.name
    if dialect == "postgresql":
        trades = _TRADES_SQL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
        snapshots = _SNAPSHOTS_SQL.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    else:
        trades = _TRADES_SQL
        snapshots = _SNAPSHOTS_SQL
    op.execute(trades)
    op.execute(snapshots)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS snapshots")
    op.execute("DROP TABLE IF EXISTS trades")
