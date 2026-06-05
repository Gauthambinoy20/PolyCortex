import json
import logging
import math
import os
import sqlite3
from datetime import UTC, datetime
from typing import cast

logger = logging.getLogger(__name__)


class PerformanceTracker:
    TRADE_MIGRATIONS: dict[str, str] = {
        "raw_estimated_prob": "REAL",
        "calibrated_prob": "REAL",
        "closed_at": "TEXT",
        "entry_reason": "TEXT",
        "exit_reason": "TEXT",
        "order_type": "TEXT DEFAULT 'limit'",
        "source": "TEXT DEFAULT 'agent'",
        "exchange_order_ids": "TEXT",
        "realized_roi": "REAL",
        "notes": "TEXT",
        "local_order_id": "TEXT",
        "entry_order_status": "TEXT DEFAULT 'filled'",
        "filled_size_usdc": "REAL",
        "remaining_size_usdc": "REAL",
        "average_fill_price": "REAL",
        "parent_order_id": "TEXT",
        "oco_group_id": "TEXT",
        "trigger_price": "REAL",
        "trigger_condition": "TEXT",
        "take_profit_price": "REAL",
        "stop_loss_price": "REAL",
        "bracket_order_ids": "TEXT",
        "bracket_state": "TEXT DEFAULT 'inactive'",
        "last_fill_at": "TEXT",
        "reduce_only": "INTEGER DEFAULT 0",
        "order_kind": "TEXT DEFAULT 'entry'",
        "fee_usdc": "REAL DEFAULT 0.0",
        "is_maker": "INTEGER",
        "gas_cost_usdc": "REAL DEFAULT 0.0",
    }
    SNAPSHOT_MIGRATIONS: dict[str, str] = {
        "brier": "REAL",
        "win_rate": "REAL",
        "profit_factor": "REAL",
        "open_exposure": "REAL",
    }

    def __init__(self, db_path: str = "data/trades.db") -> None:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()
        self._migrate_tables()

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
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
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bankroll REAL,
                total_pnl REAL,
                open_positions_count INTEGER,
                drawdown REAL,
                brier REAL,
                win_rate REAL,
                profit_factor REAL,
                open_exposure REAL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_local_order_id ON trades(local_order_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_market_id ON trades(market_id)")
        self._conn.commit()

    def _migrate_tables(self) -> None:
        self._ensure_columns("trades", self.TRADE_MIGRATIONS)
        self._ensure_columns("snapshots", self.SNAPSHOT_MIGRATIONS)

    def _ensure_columns(self, table: str, columns: dict[str, str]) -> None:
        cur = self._conn.cursor()
        existing = {row["name"] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
        changed = False
        for name, column_type in columns.items():
            if name in existing:
                continue
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")
            changed = True
        if changed:
            self._conn.commit()

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _iso_or_none(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if hasattr(value, "isoformat"):
            return str(value.isoformat())
        return str(value)

    @staticmethod
    def _json_dumps(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value)

    @staticmethod
    def _parse_iso(value: str | None) -> datetime:
        if not value:
            return datetime.min.replace(tzinfo=UTC)
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)

    @staticmethod
    def _effective_size_usdc(row: dict) -> float:
        filled = row.get("filled_size_usdc")
        if filled is not None:
            try:
                filled_float = float(filled)
            except (TypeError, ValueError):
                filled_float = 0.0
            if filled_float > 0.0:
                return filled_float
        return float(row.get("size_usdc") or 0.0)

    @staticmethod
    def _entry_yes_price(row: dict) -> float:
        entry_price = row.get("entry_price")
        average_fill_price = row.get("average_fill_price")
        direction = row.get("direction")
        if average_fill_price is None:
            return float(entry_price or 0.0)
        avg = float(average_fill_price or 0.0)
        if direction == "NO":
            return float(max(0.0, min(1.0, 1.0 - avg)))
        return avg

    @staticmethod
    def _exit_yes_price(row: dict, exit_price: float) -> float:
        if row.get("direction") == "NO":
            return float(max(0.0, min(1.0, 1.0 - exit_price)))
        return float(exit_price)

    @staticmethod
    def _net_pnl(pnl: float, row: dict) -> float:
        fees = float(row.get("fee_usdc") or 0.0)
        gas = float(row.get("gas_cost_usdc") or 0.0)
        return pnl - fees - gas

    @staticmethod
    def _derive_entry_reason(signal_breakdown) -> str:
        if not isinstance(signal_breakdown, dict) or not signal_breakdown:
            return "system_signal"
        numeric = {key: float(value) for key, value in signal_breakdown.items() if isinstance(value, (int, float))}
        if not numeric:
            return "system_signal"
        strongest = max(numeric.items(), key=lambda item: abs(item[1]))[0]
        return f"strongest_signal:{strongest}"

    def _insert_trade(
        self,
        edge_result: dict,
        size: float,
        is_paper: int,
        *,
        order: dict | None = None,
        meta: dict | None = None,
    ) -> int:
        order = order or {}
        meta = meta or {}
        signal_bd = edge_result.get("signal_breakdown")
        entry_reason = meta.get("entry_reason") or self._derive_entry_reason(signal_bd)
        filled_size = order.get("filled_size_usdc")
        if filled_size is None:
            filled_size = size if order.get("status") == "filled" else 0.0
        remaining_size = order.get("remaining_size_usdc")
        if remaining_size is None:
            remaining_size = max(size - float(filled_size or 0.0), 0.0)

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO trades
                (market_id, question, direction, entry_price, size_usdc,
                 timestamp, status, edge_at_entry, confidence_at_entry,
                 raw_estimated_prob, estimated_prob, calibrated_prob,
                 regime_at_entry, category, signal_breakdown,
                 is_paper, entry_reason, order_type, source, exchange_order_ids, notes,
                 local_order_id, entry_order_status, filled_size_usdc, remaining_size_usdc,
                 average_fill_price, parent_order_id, oco_group_id, trigger_price,
                 trigger_condition, take_profit_price, stop_loss_price,
                 bracket_order_ids, bracket_state, last_fill_at, reduce_only, order_kind)
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                edge_result.get("market_id"),
                edge_result.get("question"),
                edge_result.get("direction"),
                edge_result.get("market_price"),
                size,
                meta.get("timestamp", self._now()),
                meta.get("status", "open"),
                edge_result.get("edge"),
                edge_result.get("confidence"),
                edge_result.get("raw_estimated_prob", edge_result.get("estimated_prob")),
                edge_result.get("estimated_prob"),
                edge_result.get("calibrated_prob", edge_result.get("estimated_prob")),
                edge_result.get("regime"),
                edge_result.get("category"),
                self._json_dumps(signal_bd),
                is_paper,
                entry_reason,
                meta.get("order_type", order.get("order_type", "limit")),
                meta.get("source", "agent"),
                self._json_dumps(order.get("exchange_order_ids")),
                meta.get("notes"),
                order.get("order_id"),
                order.get("status", "filled"),
                filled_size,
                remaining_size,
                order.get("average_fill_price"),
                order.get("parent_order_id"),
                order.get("oco_group_id"),
                order.get("trigger_price"),
                order.get("trigger_condition"),
                order.get("take_profit_price"),
                order.get("stop_loss_price"),
                self._json_dumps(order.get("bracket_order_ids")),
                order.get("bracket_state", "inactive"),
                self._iso_or_none(order.get("last_fill_at")),
                int(bool(order.get("reduce_only", False))),
                order.get("order_kind", "entry"),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def record_trade(
        self,
        edge_result: dict,
        size: float,
        order: dict,
        meta: dict | None = None,
    ) -> int:
        return self._insert_trade(edge_result, size, is_paper=0, order=order, meta=meta)

    def record_paper_trade(
        self,
        edge_result: dict,
        size: float,
        order: dict | None = None,
        meta: dict | None = None,
    ) -> int:
        return self._insert_trade(edge_result, size, is_paper=1, order=order, meta=meta)

    def get_trade_id_for_order(self, order_id: str) -> int | None:
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT id FROM trades WHERE local_order_id = ?",
            (order_id,),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def get_trade(self, trade_id: int) -> dict | None:
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        return dict(row) if row is not None else None

    def sync_entry_order(self, order: dict, *, trade_id: int | None = None) -> bool:
        trade_id = trade_id or self.get_trade_id_for_order(str(order.get("order_id") or ""))
        if trade_id is None:
            return False

        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE trades
            SET entry_order_status = ?,
                filled_size_usdc = ?,
                remaining_size_usdc = ?,
                average_fill_price = ?,
                exchange_order_ids = ?,
                take_profit_price = ?,
                stop_loss_price = ?,
                bracket_order_ids = ?,
                bracket_state = ?,
                last_fill_at = ?
            WHERE id = ?
            """,
            (
                order.get("status"),
                order.get("filled_size_usdc"),
                order.get("remaining_size_usdc"),
                order.get("average_fill_price"),
                self._json_dumps(order.get("exchange_order_ids")),
                order.get("take_profit_price"),
                order.get("stop_loss_price"),
                self._json_dumps(order.get("bracket_order_ids")),
                order.get("bracket_state", "inactive"),
                self._iso_or_none(order.get("last_fill_at")),
                trade_id,
            ),
        )
        self._conn.commit()
        return True

    def sync_bracket_state(self, order: dict) -> bool:
        parent_order_id = order.get("parent_order_id")
        if not parent_order_id:
            return False

        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE trades
            SET bracket_state = ?,
                last_fill_at = COALESCE(?, last_fill_at),
                notes = COALESCE(notes, ?)
            WHERE local_order_id = ?
            """,
            (
                order.get("status"),
                self._iso_or_none(order.get("last_fill_at")),
                f"bracket:{order.get('order_kind', 'gtt')}",
                parent_order_id,
            ),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def update_position(self, trade_id: int, current_price: float) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT direction, entry_price, average_fill_price, size_usdc, filled_size_usdc, fee_usdc, gas_cost_usdc FROM trades WHERE id = ?",
            (trade_id,),
        )
        row = cur.fetchone()
        if row is None:
            logger.warning("Trade %d not found for update", trade_id)
            return

        direction = row["direction"]
        entry_price = self._entry_yes_price(dict(row))
        size_usdc = row["filled_size_usdc"] or row["size_usdc"]

        if direction == "YES":
            pnl = (current_price - entry_price) * size_usdc / entry_price if entry_price else 0.0
        else:
            pnl = (entry_price - current_price) * size_usdc / (1 - entry_price) if (1 - entry_price) else 0.0

        pnl = self._net_pnl(pnl, dict(row))
        cur.execute("UPDATE trades SET pnl = ? WHERE id = ?", (pnl, trade_id))
        self._conn.commit()

    def close_position(
        self,
        trade_id: int,
        exit_price: float,
        *,
        status: str = "closed",
        exit_reason: str = "manual_close",
        closed_at: str | None = None,
    ) -> float:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT direction, entry_price, average_fill_price, size_usdc, filled_size_usdc, fee_usdc, gas_cost_usdc FROM trades WHERE id = ?",
            (trade_id,),
        )
        row = cur.fetchone()
        if row is None:
            logger.warning("Trade %d not found for close", trade_id)
            return 0.0

        direction = row["direction"]
        entry_price = self._entry_yes_price(dict(row))
        size_usdc = row["filled_size_usdc"] or row["size_usdc"]
        exit_price = self._exit_yes_price(dict(row), exit_price)

        if direction == "YES":
            pnl = (exit_price - entry_price) * size_usdc / entry_price if entry_price else 0.0
        else:
            pnl = (entry_price - exit_price) * size_usdc / (1 - entry_price) if (1 - entry_price) else 0.0

        pnl = self._net_pnl(pnl, dict(row))
        realized_roi = pnl / size_usdc if size_usdc else 0.0
        cur.execute(
            """
            UPDATE trades
            SET status = ?, exit_price = ?, pnl = ?, closed_at = ?,
                exit_reason = ?, realized_roi = ?, bracket_state = ?,
                entry_order_status = COALESCE(entry_order_status, 'filled')
            WHERE id = ?
            """,
            (
                status,
                exit_price,
                pnl,
                closed_at or self._now(),
                exit_reason,
                realized_roi,
                "closed",
                trade_id,
            ),
        )
        self._conn.commit()
        return pnl

    def _sync_bracket_state_no_commit(self, order: dict, cur: sqlite3.Cursor) -> None:
        parent_order_id = order.get("parent_order_id")
        if not parent_order_id:
            return
        cur.execute(
            """
            UPDATE trades
            SET bracket_state = ?,
                last_fill_at = COALESCE(?, last_fill_at),
                notes = COALESCE(notes, ?)
            WHERE local_order_id = ?
            """,
            (
                order.get("status"),
                self._iso_or_none(order.get("last_fill_at")),
                f"bracket:{order.get('order_kind', 'gtt')}",
                parent_order_id,
            ),
        )

    def _close_position_no_commit(
        self,
        trade_id: int,
        exit_price: float,
        *,
        status: str = "closed",
        exit_reason: str = "manual_close",
        closed_at: str | None = None,
        cur: sqlite3.Cursor | None = None,
    ) -> float:
        if cur is None:
            cur = self._conn.cursor()
        cur.execute(
            "SELECT direction, entry_price, average_fill_price, size_usdc, filled_size_usdc, fee_usdc, gas_cost_usdc FROM trades WHERE id = ?",
            (trade_id,),
        )
        row = cur.fetchone()
        if row is None:
            logger.warning("Trade %d not found for close", trade_id)
            return 0.0

        direction = row["direction"]
        entry_price = self._entry_yes_price(dict(row))
        size_usdc = row["filled_size_usdc"] or row["size_usdc"]
        exit_price = self._exit_yes_price(dict(row), exit_price)

        if direction == "YES":
            pnl = (exit_price - entry_price) * size_usdc / entry_price if entry_price else 0.0
        else:
            pnl = (entry_price - exit_price) * size_usdc / (1 - entry_price) if (1 - entry_price) else 0.0

        pnl = self._net_pnl(pnl, dict(row))
        realized_roi = pnl / size_usdc if size_usdc else 0.0
        cur.execute(
            """
            UPDATE trades
            SET status = ?, exit_price = ?, pnl = ?, closed_at = ?,
                exit_reason = ?, realized_roi = ?, bracket_state = ?,
                entry_order_status = COALESCE(entry_order_status, 'filled')
            WHERE id = ?
            """,
            (
                status,
                exit_price,
                pnl,
                closed_at or self._now(),
                exit_reason,
                realized_roi,
                "closed",
                trade_id,
            ),
        )
        return pnl

    def close_trade_for_order(
        self,
        order: dict,
        *,
        exit_price: float | None = None,
        exit_reason: str = "bracket_exit",
        status: str = "closed",
    ) -> float:
        if exit_price is not None and exit_price < 0:
            logger.warning("close_trade_for_order: negative exit_price %.4f, clamping to 0", exit_price)
            exit_price = 0.0
        order_id = order.get("parent_order_id") or order.get("order_id")
        if not order_id:
            return 0.0
        cur = self._conn.cursor()
        row = cur.execute(
            "SELECT id FROM trades WHERE local_order_id = ?",
            (order_id,),
        ).fetchone()
        if row is None:
            return 0.0

        try:
            cur.execute("BEGIN IMMEDIATE")
            self._sync_bracket_state_no_commit(order, cur)
            pnl = self._close_position_no_commit(
                int(row[0]),
                exit_price=float(
                    exit_price
                    if exit_price is not None
                    else order.get("average_fill_price") or order.get("price") or 0.0
                ),
                status=status,
                exit_reason=exit_reason,
                closed_at=self._iso_or_none(order.get("last_fill_at")),
                cur=cur,
            )
            self._conn.commit()
            return pnl
        except Exception:
            self._conn.rollback()
            logger.exception("Failed to close trade for order %s", order_id)
            return 0.0

    def get_open_positions(self) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC")
        return [dict(row) for row in cur.fetchall()]

    def get_closed_trades(self, limit: int | None = 100) -> list[dict]:
        cur = self._conn.cursor()
        if limit is None:
            cur.execute(
                """
                SELECT * FROM trades
                WHERE status IN ('closed', 'resolved')
                ORDER BY COALESCE(closed_at, timestamp) DESC
                """
            )
        else:
            cur.execute(
                """
                SELECT * FROM trades
                WHERE status IN ('closed', 'resolved')
                ORDER BY COALESCE(closed_at, timestamp) DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in cur.fetchall()]

    def reset(self) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM trades")
        cur.execute("DELETE FROM snapshots")
        self._conn.commit()

    def get_metrics(self) -> dict:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM trades")
        rows = [dict(r) for r in cur.fetchall()]

        closed_rows = [r for r in rows if r.get("status") in ("closed", "resolved")]
        open_rows = [r for r in rows if r.get("status") == "open"]
        ordered_closed = sorted(
            closed_rows,
            key=lambda r: self._parse_iso(r.get("closed_at") or r.get("timestamp")),
        )

        total_pnl = sum(r.get("pnl") or 0.0 for r in closed_rows)
        trade_count = len(closed_rows)
        wins = sum(1 for r in closed_rows if (r.get("pnl") or 0.0) > 0)
        win_rate = wins / trade_count if trade_count > 0 else 0.0

        gross_profit = sum((r.get("pnl") or 0.0) for r in closed_rows if (r.get("pnl") or 0.0) > 0)
        gross_loss = abs(sum((r.get("pnl") or 0.0) for r in closed_rows if (r.get("pnl") or 0.0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        avg_win = gross_profit / wins if wins > 0 else 0.0
        losses = sum(1 for r in closed_rows if (r.get("pnl") or 0.0) < 0)
        avg_loss = -gross_loss / losses if losses > 0 else 0.0
        expectancy = total_pnl / trade_count if trade_count > 0 else 0.0

        edges = [r.get("edge_at_entry") for r in closed_rows if r.get("edge_at_entry") is not None]
        avg_edge = sum(edges) / len(edges) if edges else 0.0
        confidences = [r.get("confidence_at_entry") for r in closed_rows if r.get("confidence_at_entry") is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_trade_size = (
            sum(self._effective_size_usdc(r) for r in closed_rows) / trade_count if trade_count > 0 else 0.0
        )
        best_trade = max((r.get("pnl") or 0.0 for r in closed_rows), default=0.0)
        worst_trade = min((r.get("pnl") or 0.0 for r in closed_rows), default=0.0)

        current_exposure = sum(self._effective_size_usdc(r) for r in open_rows)
        paper_count = sum(1 for r in rows if int(r.get("is_paper") or 0) == 1)
        live_count = len(rows) - paper_count
        paper_open_count = sum(1 for r in open_rows if int(r.get("is_paper") or 0) == 1)
        live_open_count = len(open_rows) - paper_open_count

        brier_terms: list[float] = []
        raw_brier_terms: list[float] = []
        calibration_buckets: dict[int, dict[str, float]] = {}
        for row in closed_rows:
            prob = row.get("estimated_prob")
            if prob is None:
                continue
            outcome = 1.0 if (row.get("pnl") or 0.0) > 0 else 0.0
            prob = float(prob)
            brier_terms.append((prob - outcome) ** 2)
            raw_prob = row.get("raw_estimated_prob")
            if raw_prob is None:
                raw_prob = prob
            raw_brier_terms.append((float(raw_prob) - outcome) ** 2)
            bucket = min(int(prob * 10), 9)
            stats = calibration_buckets.setdefault(bucket, {"predicted": 0.0, "actual": 0.0, "count": 0.0})
            stats["predicted"] += prob
            stats["actual"] += outcome
            stats["count"] += 1
        brier_score = sum(brier_terms) / len(brier_terms) if brier_terms else None
        raw_brier_score = sum(raw_brier_terms) / len(raw_brier_terms) if raw_brier_terms else None

        per_category_stats: dict[str, dict] = {}
        exposure_by_category: dict[str, float] = {}
        for row in closed_rows:
            category = row.get("category") or "unknown"
            per_category_stats.setdefault(category, {"count": 0, "total_pnl": 0.0, "wins": 0})
            per_category_stats[category]["count"] += 1
            per_category_stats[category]["total_pnl"] += row.get("pnl") or 0.0
            if (row.get("pnl") or 0.0) > 0:
                per_category_stats[category]["wins"] += 1
        for _category, stats in per_category_stats.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0.0
            del stats["wins"]
        for row in open_rows:
            category = row.get("category") or "unknown"
            exposure_by_category[category] = exposure_by_category.get(category, 0.0) + self._effective_size_usdc(row)

        direction_stats: dict[str, dict] = {}
        exposure_by_direction: dict[str, float] = {}
        for direction in ("YES", "NO"):
            direction_rows = [r for r in closed_rows if r.get("direction") == direction]
            direction_wins = sum(1 for r in direction_rows if (r.get("pnl") or 0.0) > 0)
            direction_stats[direction] = {
                "count": len(direction_rows),
                "total_pnl": sum(r.get("pnl") or 0.0 for r in direction_rows),
                "win_rate": direction_wins / len(direction_rows) if direction_rows else 0.0,
            }
            exposure_by_direction[direction] = sum(
                self._effective_size_usdc(r) for r in open_rows if r.get("direction") == direction
            )

        regime_stats: dict[str, dict] = {}
        for row in closed_rows:
            regime = row.get("regime_at_entry") or "unknown"
            regime_stats.setdefault(regime, {"count": 0, "total_pnl": 0.0, "wins": 0})
            regime_stats[regime]["count"] += 1
            regime_stats[regime]["total_pnl"] += row.get("pnl") or 0.0
            if (row.get("pnl") or 0.0) > 0:
                regime_stats[regime]["wins"] += 1
        for _regime, stats in regime_stats.items():
            stats["win_rate"] = stats["wins"] / stats["count"] if stats["count"] > 0 else 0.0
            del stats["wins"]

        per_signal_accuracy: dict[str, dict] = {}
        signal_stats: dict[str, dict[str, int]] = {}
        for row in closed_rows:
            raw = row.get("signal_breakdown")
            if not raw:
                continue
            try:
                breakdown = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(breakdown, dict):
                continue
            trade_won = (row.get("pnl") or 0.0) > 0
            for signal_name, signal_val in breakdown.items():
                try:
                    val = float(signal_val)
                except (TypeError, ValueError):
                    continue
                stats = cast(
                    "dict[str, float]",
                    signal_stats.setdefault(signal_name, {"correct": 0, "total": 0}),
                )
                stats["total"] += 1
                signal_positive = val > 0
                if (signal_positive and trade_won) or (not signal_positive and not trade_won):
                    stats["correct"] += 1
        for signal_name, signal_stat in signal_stats.items():
            stats = cast("dict[str, float]", signal_stat)
            per_signal_accuracy[signal_name] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "count": stats["total"],
            }

        longest_win_streak = 0
        longest_loss_streak = 0
        current_type = "flat"
        current_count = 0
        win_streak = 0
        loss_streak = 0
        for row in ordered_closed:
            trade_won = (row.get("pnl") or 0.0) > 0
            if trade_won:
                win_streak += 1
                loss_streak = 0
                longest_win_streak = max(longest_win_streak, win_streak)
                current_type = "win"
                current_count = win_streak
            elif (row.get("pnl") or 0.0) < 0:
                loss_streak += 1
                win_streak = 0
                longest_loss_streak = max(longest_loss_streak, loss_streak)
                current_type = "loss"
                current_count = loss_streak
            else:
                win_streak = 0
                loss_streak = 0
                current_type = "flat"
                current_count = 0

        recent_rows = ordered_closed[-10:]
        recent_trade_count = len(recent_rows)
        recent_pnl = sum(r.get("pnl") or 0.0 for r in recent_rows)
        recent_win_rate = (
            sum(1 for r in recent_rows if (r.get("pnl") or 0.0) > 0) / recent_trade_count
            if recent_trade_count > 0
            else 0.0
        )

        calibration = []
        for bucket in sorted(calibration_buckets):
            stats = calibration_buckets[bucket]
            count = int(stats["count"])
            calibration.append(
                {
                    "bucket": f"{bucket / 10:.1f}-{(bucket + 1) / 10:.1f}",
                    "predicted": stats["predicted"] / count if count > 0 else 0.0,
                    "actual": stats["actual"] / count if count > 0 else 0.0,
                    "count": count,
                }
            )

        status_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        entry_status_counts: dict[str, int] = {}
        for row in rows:
            status = row.get("status") or "unknown"
            source = row.get("source") or "agent"
            entry_status = row.get("entry_order_status") or "unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            entry_status_counts[entry_status] = entry_status_counts.get(entry_status, 0) + 1

        return {
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "trade_count": trade_count,
            "closed_count": trade_count,
            "brier_score": brier_score,
            "raw_brier_score": raw_brier_score,
            "calibration_gain": (
                raw_brier_score - brier_score if raw_brier_score is not None and brier_score is not None else None
            ),
            "avg_edge": avg_edge,
            "avg_confidence": avg_confidence,
            "avg_trade_size": avg_trade_size,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "per_category_stats": per_category_stats,
            "per_signal_accuracy": per_signal_accuracy,
            "direction_stats": direction_stats,
            "regime_stats": regime_stats,
            "open_count": len(open_rows),
            "current_exposure": current_exposure,
            "exposure_by_category": exposure_by_category,
            "exposure_by_direction": exposure_by_direction,
            "paper_count": paper_count,
            "live_count": live_count,
            "paper_open_count": paper_open_count,
            "live_open_count": live_open_count,
            "longest_win_streak": longest_win_streak,
            "longest_loss_streak": longest_loss_streak,
            "current_streak": {"type": current_type, "count": current_count},
            "recent_trade_count": recent_trade_count,
            "recent_pnl": recent_pnl,
            "recent_win_rate": recent_win_rate,
            "calibration": calibration,
            "status_counts": status_counts,
            "source_counts": source_counts,
            "entry_status_counts": entry_status_counts,
            "pending_entry_count": entry_status_counts.get("placed", 0),
            "partial_fill_count": entry_status_counts.get("partially_filled", 0),
            "armed_bracket_count": sum(1 for r in rows if (r.get("bracket_state") or "") == "armed"),
        }

    def record_snapshot(self, bankroll: float, drawdown: float) -> None:
        metrics = self.get_metrics()
        cur = self._conn.cursor()
        profit_factor = metrics["profit_factor"]
        cur.execute(
            """
            INSERT INTO snapshots (
                timestamp, bankroll, total_pnl, open_positions_count,
                drawdown, brier, win_rate, profit_factor, open_exposure
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._now(),
                bankroll,
                metrics["total_pnl"],
                metrics["open_count"],
                drawdown,
                metrics["brier_score"],
                metrics["win_rate"],
                profit_factor if math.isfinite(profit_factor) else None,
                metrics["current_exposure"],
            ),
        )
        self._conn.commit()

    def log_summary(self) -> None:
        metrics = self.get_metrics()
        profit_factor = metrics["profit_factor"]
        pf_label = f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf"
        logger.info(
            "Performance | PnL: %.4f | Win Rate: %.2f%% | Open: %d | Exposure: %.2f | PF: %s | Brier: %s | Trades: %d",
            metrics["total_pnl"],
            metrics["win_rate"] * 100,
            metrics["open_count"],
            metrics["current_exposure"],
            pf_label,
            f"{metrics['brier_score']:.4f}" if metrics["brier_score"] is not None else "N/A",
            metrics["trade_count"],
        )

    def record_fee(self, trade_id: int, fee_usdc: float, is_maker: bool, gas_cost_usdc: float = 0.0) -> None:
        """Record fee information for a trade."""
        self._conn.execute(
            """
            UPDATE trades
            SET fee_usdc = COALESCE(fee_usdc, 0.0) + ?,
                is_maker = ?,
                gas_cost_usdc = COALESCE(gas_cost_usdc, 0.0) + ?
            WHERE id=?
            """,
            (fee_usdc, int(is_maker), gas_cost_usdc, trade_id),
        )
        self._conn.commit()
        logger.debug(
            "Recorded fee $%.4f (maker=%s, gas=$%.4f) for trade %d", fee_usdc, is_maker, gas_cost_usdc, trade_id
        )

    def get_fee_summary(self) -> dict:
        """Return aggregate fee statistics."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) AS total_trades,
                COALESCE(SUM(fee_usdc), 0.0) AS total_fees,
                COALESCE(SUM(gas_cost_usdc), 0.0) AS total_gas,
                COALESCE(SUM(CASE WHEN is_maker=1 THEN 1 ELSE 0 END), 0) AS maker_count,
                COALESCE(SUM(CASE WHEN is_maker=0 THEN 1 ELSE 0 END), 0) AS taker_count
            FROM trades
            WHERE fee_usdc IS NOT NULL AND fee_usdc > 0
        """)
        row = cur.fetchone()
        total = (row["maker_count"] or 0) + (row["taker_count"] or 0)
        return {
            "total_fees_usdc": round(row["total_fees"], 4),
            "total_gas_usdc": round(row["total_gas"], 4),
            "total_cost_usdc": round(row["total_fees"] + row["total_gas"], 4),
            "maker_count": row["maker_count"],
            "taker_count": row["taker_count"],
            "maker_ratio": round(row["maker_count"] / total, 4) if total > 0 else 0.0,
        }

    def close(self) -> None:
        self._conn.close()
