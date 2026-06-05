import sqlite3

import pytest

from polymarket_agent.tracking.resolver import MarketResolver


class _FakeGamma:
    def __init__(self, resolved_ids: set[str]) -> None:
        self._resolved_ids = resolved_ids

    async def get_market(self, market_id: str) -> dict:
        return {"closed": market_id in self._resolved_ids}


def _seed_trades(db_path: str, rows: list[tuple[str, str]]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        for market_id, status in rows:
            conn.execute(
                "INSERT INTO trades (market_id, status) VALUES (?, ?)",
                (market_id, status),
            )
        conn.commit()


@pytest.mark.asyncio
async def test_resolver_closes_resolved_markets(tmp_path):
    db = str(tmp_path / "t.db")
    gamma = _FakeGamma(resolved_ids={"m1", "m3"})
    resolver = MarketResolver(db_path=db, gamma_client=gamma)
    _seed_trades(
        db,
        [("m1", "open"), ("m2", "open"), ("m3", "open"), ("m4", "closed")],
    )

    count = await resolver.check_and_resolve()
    assert count == 2

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT market_id, status FROM trades ORDER BY market_id"))
    statuses = {r["market_id"]: r["status"] for r in rows}
    assert statuses["m1"] == "resolved"
    assert statuses["m2"] == "open"
    assert statuses["m3"] == "resolved"


@pytest.mark.asyncio
async def test_resolver_no_op_when_nothing_open(tmp_path):
    db = str(tmp_path / "t.db")
    resolver = MarketResolver(db_path=db, gamma_client=_FakeGamma(set()))
    count = await resolver.check_and_resolve()
    assert count == 0
