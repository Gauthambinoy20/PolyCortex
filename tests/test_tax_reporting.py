"""Tests for reporting.tax."""

from __future__ import annotations

import csv
import sqlite3

import pytest

from polymarket_agent.reporting import tax


@pytest.fixture
def seeded_db(tmp_path):
    db = tmp_path / "trades.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT, question TEXT, direction TEXT,
            entry_price REAL, size_usdc REAL, timestamp TEXT,
            status TEXT DEFAULT 'open', exit_price REAL, pnl REAL,
            closed_at TEXT, is_paper INTEGER DEFAULT 1, category TEXT
        )
        """
    )
    rows = [
        ("m1", "Q1?", "YES", 0.5, 100, "2026-01-10T12:00:00", "closed", 0.6, 20, "2026-02-01T12:00:00", 0, "cat"),
        ("m1", "Q1?", "YES", 0.55, 50, "2026-02-05T12:00:00", "closed", 0.45, -5, "2026-03-01T12:00:00", 0, "cat"),
        ("m2", "Q2?", "NO", 0.3, 80, "2026-04-01T12:00:00", "closed", 0.35, 13, "2026-04-20T12:00:00", 0, "cat"),
        ("m3", "Open", "YES", 0.5, 50, "2026-05-01T12:00:00", "open", None, None, None, 0, "cat"),
        # Different year - should be excluded
        ("m4", "Q4?", "YES", 0.5, 30, "2025-01-01T12:00:00", "closed", 0.7, 6, "2025-02-01T12:00:00", 0, "cat"),
    ]
    conn.executemany(
        "INSERT INTO trades (market_id, question, direction, entry_price, size_usdc, timestamp, status, exit_price, pnl, closed_at, is_paper, category) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def test_generate_form_8949_fifo(seeded_db, tmp_path):
    out = tax.generate_form_8949(2026, method="fifo", db_path=seeded_db, output_dir=str(tmp_path))
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # three closed trades in 2026
    for row in rows:
        assert set(row.keys()) == set(tax.FORM_8949_HEADER)
        float(row["proceeds"])
        float(row["cost_basis"])
        float(row["gain_loss"])


def test_generate_form_8949_lifo(seeded_db, tmp_path):
    out = tax.generate_form_8949(2026, method="lifo", db_path=seeded_db, output_dir=str(tmp_path))
    assert out.exists()


def test_invalid_method(seeded_db, tmp_path):
    with pytest.raises(ValueError):
        tax.generate_form_8949(2026, method="hifo", db_path=seeded_db, output_dir=str(tmp_path))


def test_missing_db(tmp_path):
    with pytest.raises(FileNotFoundError):
        tax.generate_form_8949(2026, db_path=str(tmp_path / "nope.db"), output_dir=str(tmp_path))


def test_generate_trades_csv(seeded_db, tmp_path):
    out = tax.generate_trades_csv(2026, db_path=seeded_db, output_dir=str(tmp_path))
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert "market_id" in rows[0]


def test_generate_trades_csv_year_filter(seeded_db, tmp_path):
    out = tax.generate_trades_csv(2025, db_path=seeded_db, output_dir=str(tmp_path))
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["market_id"] == "m4"
