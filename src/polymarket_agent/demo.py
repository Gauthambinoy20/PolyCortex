"""Demo data seeding for dashboards, testing, and operator walkthroughs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from polymarket_agent.tracking.tracker import PerformanceTracker

DEMO_DB_PATH = "data/demo_trades.db"

QUESTIONS = [
    "Will BTC trade above $120k by year-end?",
    "Will the Fed cut rates twice this year?",
    "Will ETH ETF inflows exceed expectations this quarter?",
    "Will SOL outperform ETH this month?",
    "Will US CPI print below 3.0% next release?",
    "Will an AI mega-cap beat earnings guidance?",
    "Will gold close higher this week?",
    "Will Nasdaq finish the month positive?",
    "Will the next payrolls report beat consensus?",
    "Will oil stay above $80 next month?",
]
REGIMES = ["stable", "trending", "volatile"]
CATEGORIES = ["crypto", "macro", "equities", "commodities", "rates"]


def _signal_breakdown(index: int, direction: str) -> dict[str, float]:
    sign = 1.0 if direction == "YES" else -1.0
    return {
        "order_book": round(sign * (0.02 + (index % 3) * 0.01), 4),
        "momentum": round(sign * (0.005 + (index % 2) * 0.006), 4),
        "sentiment": round(sign * (0.015 + (index % 4) * 0.008), 4),
        "news_volume": round(sign * (0.004 + (index % 5) * 0.003), 4),
        "cross_market": round(sign * (0.003 + (index % 3) * 0.004), 4),
        "tcn": round(sign * (0.01 + (index % 4) * 0.005), 4),
        "spread": round(0.018 + (index % 4) * 0.006, 4),
    }


def _seed_snapshots(tracker: PerformanceTracker, bankroll: float, events: list[dict]) -> int:
    tracker._conn.execute("DELETE FROM snapshots")

    ordered_events = sorted(events, key=lambda event: event["timestamp"])
    available_bankroll = bankroll
    realized_pnl = 0.0
    open_exposure = 0.0
    open_count = 0
    peak = bankroll
    closed_pnls: list[float] = []

    snapshot_rows: list[tuple] = []
    for event in ordered_events:
        if event["kind"] == "open":
            open_exposure += event["size"]
            open_count += 1
        else:
            open_exposure = max(0.0, open_exposure - event["size"])
            open_count = max(0, open_count - 1)
            realized_pnl += event["pnl"]
            closed_pnls.append(event["pnl"])

        available_bankroll = bankroll + realized_pnl - open_exposure
        peak = max(peak, available_bankroll)
        drawdown = (peak - available_bankroll) / peak if peak else 0.0

        trade_count = len(closed_pnls)
        win_rate = sum(1 for pnl in closed_pnls if pnl > 0) / trade_count if trade_count > 0 else 0.0
        gross_profit = sum(pnl for pnl in closed_pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in closed_pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        brier = max(0.08, 0.27 - (trade_count * 0.003)) if trade_count > 0 else 0.24

        snapshot_rows.append(
            (
                event["timestamp"].isoformat(),
                round(available_bankroll, 2),
                round(realized_pnl, 2),
                open_count,
                round(drawdown, 4),
                round(brier, 4),
                round(win_rate, 4),
                round(profit_factor, 4) if profit_factor is not None else None,
                round(open_exposure, 2),
            )
        )

    if not snapshot_rows:
        snapshot_rows.append((datetime.now(UTC).isoformat(), bankroll, 0.0, 0, 0.0, 0.24, 0.0, None, 0.0))

    tracker._conn.executemany(
        """
        INSERT INTO snapshots (
            timestamp, bankroll, total_pnl, open_positions_count,
            drawdown, brier, win_rate, profit_factor, open_exposure
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        snapshot_rows,
    )
    tracker._conn.commit()
    return len(snapshot_rows)


def seed_demo_workspace(
    root_dir: str | Path = ".",
    *,
    reset: bool = True,
    trade_count: int = 36,
    bankroll: float = 1000.0,
) -> dict[str, int | str]:
    """Seed realistic demo data for the dashboard and CLI walkthroughs."""
    root = Path(root_dir)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "demo_trades.db"
    if reset and db_path.exists():
        db_path.unlink()

    tracker = PerformanceTracker(str(db_path))
    if reset:
        tracker.reset()

    now = datetime.now(UTC)
    open_slots = max(5, trade_count // 6)
    events: list[dict] = []
    seeded_trade_ids: list[int] = []

    for index in range(trade_count):
        category = CATEGORIES[index % len(CATEGORIES)]
        regime = REGIMES[index % len(REGIMES)]
        question = QUESTIONS[index % len(QUESTIONS)]
        direction = "YES" if index % 2 == 0 else "NO"
        entry_price = round(0.34 + ((index % 6) * 0.07), 3)
        edge = round(0.05 + ((index % 5) * 0.015), 3)
        estimated_prob = min(0.97, entry_price + edge) if direction == "YES" else max(0.03, entry_price - edge)
        confidence = round(0.58 + ((index % 4) * 0.08), 2)
        size = float(18 + (index % 6) * 6)
        timestamp = now - timedelta(hours=(trade_count - index) * 6)
        order_type = "basket" if index % 9 == 0 else "limit"
        source = "demo_live" if index % 4 == 0 else "demo"
        signal_breakdown = _signal_breakdown(index, direction)
        status = "open" if index >= trade_count - open_slots else ("resolved" if index % 3 == 0 else "closed")

        edge_result = {
            "market_id": f"demo_{index:03d}",
            "question": question,
            "direction": direction,
            "market_price": entry_price,
            "edge": edge,
            "confidence": confidence,
            "estimated_prob": estimated_prob,
            "regime": regime,
            "category": category,
            "signal_breakdown": signal_breakdown,
        }
        meta = {
            "timestamp": timestamp.isoformat(),
            "source": source,
            "order_type": order_type,
            "entry_reason": f"signal_stack:{category}:{regime}",
            "notes": "seeded_demo_trade",
            "status": "open",
        }

        if source == "demo_live":
            trade_id = tracker.record_trade(
                edge_result,
                size,
                {"exchange_order_ids": [f"demo-exchange-{index:04d}"]},
                meta=meta,
            )
        else:
            trade_id = tracker.record_paper_trade(edge_result, size, meta=meta)
        seeded_trade_ids.append(trade_id)
        events.append({"kind": "open", "timestamp": timestamp, "size": size, "pnl": 0.0})

        if status == "open":
            mark_price = entry_price + (0.025 if direction == "YES" else -0.02)
            mark_price = min(0.99, max(0.01, mark_price))
            tracker.update_position(trade_id, mark_price)
            continue

        winning_trade = index % 4 != 1
        if direction == "YES":
            exit_price = entry_price + (edge * 1.1 if winning_trade else -edge * 0.7)
        else:
            exit_price = entry_price - (edge * 1.1 if winning_trade else -edge * 0.7)
        exit_price = round(min(0.99, max(0.01, exit_price)), 3)
        closed_at = timestamp + timedelta(hours=8 + (index % 5) * 3)
        pnl = tracker.close_position(
            trade_id,
            exit_price,
            status=status,
            exit_reason="demo_resolution" if status == "resolved" else "demo_take_profit",
            closed_at=closed_at.isoformat(),
        )
        tracker._conn.execute(
            "UPDATE trades SET notes = ?, source = ? WHERE id = ?",
            (
                "seeded_demo_trade_closed",
                source,
                trade_id,
            ),
        )
        events.append({"kind": "close", "timestamp": closed_at, "size": size, "pnl": pnl})

    tracker._conn.commit()
    snapshot_count = _seed_snapshots(tracker, bankroll, events)

    alerts = [
        {
            "timestamp": (now - timedelta(hours=12)).isoformat(),
            "type": "demo",
            "severity": "info",
            "message": "Demo workspace seeded successfully",
        },
        {
            "timestamp": (now - timedelta(hours=9)).isoformat(),
            "type": "arbitrage",
            "severity": "info",
            "message": "Two synthetic cross-book gaps exceeded cost threshold",
        },
        {
            "timestamp": (now - timedelta(hours=6)).isoformat(),
            "type": "nudge",
            "severity": "warning",
            "message": "Three markets show widening spreads above the configured nudge threshold",
        },
        {
            "timestamp": (now - timedelta(hours=4)).isoformat(),
            "type": "health",
            "severity": "info",
            "message": "All data clients healthy in demo mode",
        },
        {
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "type": "calibration",
            "severity": "warning",
            "message": "Recent Brier trend improved but remains above target",
        },
    ]
    alerts_path = data_dir / "alerts.jsonl"
    alerts_path.write_text("\n".join(json.dumps(item) for item in alerts) + "\n", encoding="utf-8")

    learned_weights = {
        "min_edge": 0.09,
        "signal_weights": {
            "order_book": 0.18,
            "momentum": 0.12,
            "sentiment": 0.24,
            "news_volume": 0.12,
            "cross_market": 0.14,
            "tcn_model": 0.20,
        },
    }
    (data_dir / "learned_weights.json").write_text(
        json.dumps(learned_weights, indent=2),
        encoding="utf-8",
    )

    scan_summary = {
        "timestamp": now.isoformat(),
        "total_markets": 86,
        "filtered_markets": 18,
        "collected_order_books": 18,
        "arbitrage_count": 2,
        "regime_mix": {"stable": 7, "trending": 6, "volatile": 5},
        "category_mix": {"crypto": 5, "macro": 4, "equities": 4, "commodities": 3, "rates": 2},
        "top_markets": [
            {
                "market_id": f"watch_{idx:02d}",
                "question": QUESTIONS[idx % len(QUESTIONS)],
                "category": CATEGORIES[idx % len(CATEGORIES)],
                "midpoint": round(0.38 + idx * 0.04, 3),
                "spread": round(0.018 + idx * 0.003, 3),
                "volume_24h": 1400 + idx * 350,
                "bid_depth": 900 + idx * 120,
                "ask_depth": 850 + idx * 110,
            }
            for idx in range(8)
        ],
        "top_opportunities": [
            {
                "market_id": f"demo_{idx:03d}",
                "question": QUESTIONS[idx % len(QUESTIONS)],
                "direction": "YES" if idx % 2 == 0 else "NO",
                "edge": round(0.07 + idx * 0.01, 3),
                "confidence": round(0.62 + idx * 0.04, 2),
                "regime": REGIMES[idx % len(REGIMES)],
                "category": CATEGORIES[idx % len(CATEGORIES)],
                "estimated_prob": round(0.55 + idx * 0.03, 3),
                "market_price": round(0.48 + idx * 0.025, 3),
            }
            for idx in range(6)
        ],
        "top_arbitrage": [
            {
                "market_id": "arb_demo_01",
                "question": "Synthetic paired market spread dislocation",
                "gap": 0.044,
                "profit_after_costs": 0.017,
            },
            {
                "market_id": "arb_demo_02",
                "question": "Complement book mismatch on macro market",
                "gap": 0.039,
                "profit_after_costs": 0.012,
            },
        ],
        "top_spreads": [
            {
                "market_id": f"spread_{idx:02d}",
                "question": QUESTIONS[(idx + 3) % len(QUESTIONS)],
                "spread": round(0.045 + idx * 0.006, 3),
                "category": CATEGORIES[idx % len(CATEGORIES)],
            }
            for idx in range(5)
        ],
    }
    (data_dir / "scan_summary.json").write_text(
        json.dumps(scan_summary, indent=2),
        encoding="utf-8",
    )

    tracker.close()

    return {
        "db_path": str(db_path),
        "trade_count": len(seeded_trade_ids),
        "open_count": open_slots,
        "snapshot_count": snapshot_count,
        "alert_count": len(alerts),
    }
