import inspect as _inspect
from datetime import UTC, datetime

import aioresponses.core as _aioresponses_core
import numpy as np
import pandas as pd
import pytest
from aiohttp.client_reqrep import ClientResponse as _ClientResponse

# --- aiohttp >= 3.14 / aioresponses 0.7.8 compatibility shim ---
# aiohttp 3.14 made ClientResponse require a keyword-only ``stream_writer``
# argument, but aioresponses 0.7.8 (its latest release) still constructs mocked
# responses without it. Until aioresponses gains 3.14 support we inject the
# missing argument so the HTTP-mocking tests keep working on the
# security-patched aiohttp. The guard keeps this a no-op on older aiohttp.
if "stream_writer" in _inspect.signature(_ClientResponse.__init__).parameters:

    class _NoopStreamWriter:
        """Minimal stand-in: aiohttp 3.14 reads ``stream_writer.output_size``
        on the already-sent (``writer is None``) path that aioresponses uses."""

        output_size = 0

    class _CompatClientResponse(_ClientResponse):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("stream_writer", _NoopStreamWriter())
            super().__init__(*args, **kwargs)

    _aioresponses_core.ClientResponse = _CompatClientResponse


# --- Mock Market Data ---


@pytest.fixture
def sample_market_dict():
    """Raw market dict as returned by Gamma API (before parsing)."""
    return {
        "conditionId": "0xabc123",
        "question": "Will Bitcoin exceed $100k by end of 2026?",
        "description": "Resolves YES if BTC >= $100,000 on Dec 31, 2026.",
        "groupItemTitle": "Crypto",
        "endDate": "2026-12-31T23:59:59Z",
        "active": True,
        "liquidity": "15000.50",
        "volume": "8500.00",
        "outcomePrices": '["0.65", "0.35"]',
        "clobTokenIds": '["token_yes_123", "token_no_123"]',
        "outcomes": '["Yes", "No"]',
    }


@pytest.fixture
def sample_market():
    """Parsed Market dataclass."""
    from polymarket_agent.data.gamma_client import Market

    return Market(
        condition_id="0xabc123",
        question="Will Bitcoin exceed $100k by end of 2026?",
        description="Resolves YES if BTC >= $100,000 on Dec 31, 2026.",
        category="Crypto",
        end_date=datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC),
        active=True,
        liquidity=15000.50,
        volume_24h=8500.00,
        yes_price=0.65,
        no_price=0.35,
        clob_token_ids=["token_yes_123", "token_no_123"],
    )


@pytest.fixture
def sample_order_book_response():
    """Raw order book response from CLOB API."""
    return {
        "bids": [
            {"price": "0.63", "size": "500"},
            {"price": "0.62", "size": "300"},
            {"price": "0.61", "size": "200"},
            {"price": "0.60", "size": "1000"},
            {"price": "0.59", "size": "800"},
        ],
        "asks": [
            {"price": "0.67", "size": "400"},
            {"price": "0.68", "size": "350"},
            {"price": "0.69", "size": "250"},
            {"price": "0.70", "size": "600"},
            {"price": "0.71", "size": "500"},
        ],
    }


@pytest.fixture
def sample_order_book():
    """Parsed OrderBookSnapshot."""
    from polymarket_agent.data.clob_client import OrderBookSnapshot

    return OrderBookSnapshot(
        token_id="token_yes_123",
        timestamp=datetime.now(UTC),
        midpoint=0.65,
        spread=0.04,
        bid_depth=1500.0,
        ask_depth=1200.0,
        best_bid=0.63,
        best_ask=0.67,
        bids=[(0.63, 500), (0.62, 300), (0.61, 200), (0.60, 1000), (0.59, 800)],
        asks=[(0.67, 400), (0.68, 350), (0.69, 250), (0.70, 600), (0.71, 500)],
    )


@pytest.fixture
def sample_price_history():
    """DataFrame of price history for one market."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2026-01-01", periods=n, freq="1h", tz=UTC)
    # Random walk bounded [0.1, 0.9]
    prices = [0.5]
    for _ in range(n - 1):
        prices.append(np.clip(prices[-1] + np.random.normal(0, 0.01), 0.1, 0.9))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "midpoint": prices,
            "spread": np.random.uniform(0.01, 0.04, n),
            "bid_depth": np.random.uniform(500, 2000, n),
            "ask_depth": np.random.uniform(500, 2000, n),
            "volume": np.random.uniform(100, 5000, n),
            "book_imbalance": np.random.uniform(-0.3, 0.3, n),
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_edge_result():
    """Edge result dict for testing sizer and tracker."""
    return {
        "estimated_prob": 0.72,
        "market_price": 0.65,
        "edge": 0.07,
        "direction": "YES",
        "confidence": 0.75,
        "regime": "stable",
        "market_id": "0xabc123",
        "question": "Will Bitcoin exceed $100k?",
        "category": "crypto",
        "signal_breakdown": {
            "order_book": 0.03,
            "momentum": 0.0,
            "sentiment": 0.04,
            "news_volume": 0.0,
            "cross_market": 0.0,
            "tcn": None,
            "efficiency_penalty": 0.15,
        },
    }


@pytest.fixture
def sample_config():
    """Standard test config dict."""
    return {
        "bankroll": 1000,
        "dry_run": True,
        "demo_seed_on_start": False,
        "demo_trade_count": 24,
        "min_edge": 0.08,
        "kelly_fraction": 0.25,
        "min_confidence": 0.55,
        "max_position_pct": 0.02,
        "max_portfolio_pct": 0.20,
        "max_category_exposure_pct": 0.08,
        "max_positions": 10,
        "max_per_category": 3,
        "drawdown_reduce": 0.08,
        "drawdown_stop": 0.15,
        "drawdown_emergency": 0.20,
        "enable_probability_calibration": True,
        "probability_calibration_method": "isotonic",
        "probability_calibration_min_samples": 30,
        "probability_calibration_refit_interval": 25,
        "scan_interval_minutes": 15,
        "market_watch_top_n": 12,
        "opportunity_top_n": 10,
        "news_top_n": 20,
        "min_liquidity": 500,
        "min_volume_24h": 100,
        "min_days_to_resolution": 1,
        "max_days_to_resolution": 90,
        "order_expiry_minutes": 60,
        "split_orders_above": 50,
        "enable_bracket_orders": True,
        "bracket_take_profit_pct": 0.06,
        "bracket_stop_loss_pct": 0.035,
        "gtt_ttl_minutes": 1440,
        "paper_fill_on_place": True,
        "paper_allow_partial_fills": True,
        "paper_partial_fill_ratio": 0.50,
        "paper_fill_tolerance": 0.002,
        "live_reconcile_enabled": True,
        "risk_nudge_spread": 0.06,
        "risk_nudge_book_depth": 300,
        "use_llm_sentiment": False,  # Disable for tests
        "sentiment_cache_minutes": 30,
        "backtest_apply_costs": True,
        "backtest_fee_bps": 10,
        "backtest_slippage_bps": 5,
        "backtest_spread_impact": 0.35,
        "gamma_api_url": "https://gamma-api.polymarket.com",
        "clob_api_url": "https://clob.polymarket.com",
        "api_timeout": 10,
        "llm_timeout": 30,
        "signal_weights": {
            "order_book": 0.15,
            "momentum": 0.10,
            "sentiment": 0.30,
            "news_volume": 0.10,
            "cross_market": 0.10,
            "tcn_model": 0.25,
        },
        "regime_weights": {
            "stable": {"bayesian": 0.70, "tcn": 0.30},
            "trending": {"bayesian": 0.40, "tcn": 0.60},
            "volatile": {"bayesian": 0.80, "tcn": 0.20},
        },
    }
