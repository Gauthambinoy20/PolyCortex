"""Tests for Gamma, CLOB, news clients, and history store."""

import time

import pytest
from aioresponses import aioresponses

from polymarket_agent.data.clob_client import ClobClient, OrderBookSnapshot
from polymarket_agent.data.gamma_client import GammaClient, Market
from polymarket_agent.data.history import HistoryStore
from polymarket_agent.data.news_client import NewsClient

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


# ── Gamma Client ─────────────────────────────────────────────────────────────


async def test_gamma_get_markets(sample_market_dict):
    gamma = GammaClient(base_url=GAMMA_BASE, timeout=5.0)
    url = f"{GAMMA_BASE}/markets?active=true&closed=false&limit=100&offset=0"

    with aioresponses() as mocked:
        mocked.get(url, payload=[sample_market_dict])
        markets = await gamma.get_markets()

    await gamma.close()

    assert isinstance(markets, list)
    assert len(markets) == 1
    m = markets[0]
    assert isinstance(m, Market)
    assert m.condition_id == "0xabc123"
    assert m.question == "Will Bitcoin exceed $100k by end of 2026?"
    assert m.liquidity == pytest.approx(15000.50)
    assert m.yes_price == pytest.approx(0.65)
    assert m.no_price == pytest.approx(0.35)


async def test_gamma_handles_api_error():
    gamma = GammaClient(base_url=GAMMA_BASE, timeout=5.0)
    gamma._base_delay = 0.0  # skip retry delays in tests
    url = f"{GAMMA_BASE}/markets?active=true&closed=false&limit=100&offset=0"

    with aioresponses() as mocked:
        mocked.get(url, status=500, repeat=True)
        markets = await gamma.get_markets()

    await gamma.close()

    assert markets == []


async def test_gamma_retry_on_429(sample_market_dict):
    gamma = GammaClient(base_url=GAMMA_BASE, timeout=5.0)
    gamma._base_delay = 0.0  # skip retry delays in tests
    url = f"{GAMMA_BASE}/markets?active=true&closed=false&limit=100&offset=0"

    with aioresponses() as mocked:
        mocked.get(url, status=429)
        mocked.get(url, payload=[sample_market_dict])
        markets = await gamma.get_markets()

    await gamma.close()

    assert len(markets) == 1
    assert markets[0].condition_id == "0xabc123"


# ── CLOB Client ──────────────────────────────────────────────────────────────


async def test_clob_get_order_book(sample_order_book_response):
    clob = ClobClient(base_url=CLOB_BASE, timeout=5.0)
    url = f"{CLOB_BASE}/book?token_id=token_yes_123"

    with aioresponses() as mocked:
        mocked.get(url, payload=sample_order_book_response)
        snap = await clob.get_order_book("token_yes_123")

    await clob.close()

    assert snap is not None
    assert isinstance(snap, OrderBookSnapshot)
    assert snap.midpoint == pytest.approx(0.65)
    assert snap.spread == pytest.approx(0.04)
    assert snap.best_bid == pytest.approx(0.63)
    assert snap.best_ask == pytest.approx(0.67)
    assert snap.bid_depth > 0
    assert snap.ask_depth > 0


async def test_clob_handles_empty_book():
    clob = ClobClient(base_url=CLOB_BASE, timeout=5.0)
    url = f"{CLOB_BASE}/book?token_id=token_yes_123"

    with aioresponses() as mocked:
        mocked.get(url, payload={"bids": [], "asks": []})
        snap = await clob.get_order_book("token_yes_123")

    await clob.close()

    assert snap is None


# ── History Store ────────────────────────────────────────────────────────────


async def test_history_round_trip(tmp_path, sample_order_book):
    store = HistoryStore(data_dir=str(tmp_path))
    condition_id = "0xabc123"

    store.save_snapshot(condition_id, sample_order_book, volume=1234.0)

    df = store.load_history(condition_id)
    assert df is not None
    assert len(df) == 1
    expected_cols = {"timestamp", "midpoint", "spread", "bid_depth", "ask_depth", "volume", "book_imbalance"}
    assert expected_cols.issubset(set(df.columns))
    assert df["midpoint"].iloc[0] == pytest.approx(sample_order_book.midpoint)


# ── News Client ──────────────────────────────────────────────────────────────


async def test_news_client_caching():
    client = NewsClient(cache_minutes=30)
    query = "Bitcoin price prediction"
    cached_results = [
        {"title": "BTC surges", "snippet": "...", "url": "https://example.com", "published_date": None},
    ]

    # Pre-populate cache with a fresh timestamp
    client._cache[query] = (time.time(), cached_results)

    # Should return cached results without making any API call
    results = await client.search_news(query)
    assert results == cached_results
    assert len(results) == 1
    assert results[0]["title"] == "BTC surges"
