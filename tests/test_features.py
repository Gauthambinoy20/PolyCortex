import os
import sqlite3
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from polymarket_agent.features.engine import PolymarketFeatureEngine
from polymarket_agent.features.sentiment import SentimentAnalyzer


def test_feature_shape(sample_price_history):
    engine = PolymarketFeatureEngine()
    features = engine.create_features(sample_price_history)
    assert features is not None
    assert features.ndim == 2
    assert features.shape[1] == len(PolymarketFeatureEngine.FEATURE_NAMES)


def test_no_nans_in_features(sample_price_history):
    engine = PolymarketFeatureEngine()
    features = engine.create_features(sample_price_history)
    assert features is not None
    assert not np.isnan(features).any()


def test_known_rsi_values():
    """After 14 consecutive up moves, RSI should be close to 1.0 (all gains)."""
    # 15 price points = 14 up diffs, then 14 down diffs
    prices_up = [0.50 + i * 0.01 for i in range(15)]
    prices_down = [prices_up[-1] - i * 0.01 for i in range(1, 15)]
    prices = prices_up + prices_down

    midpoint = pd.Series(prices)
    delta = midpoint.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = gain / (gain + loss + 1e-8)

    # At index 14 (after exactly 14 up moves), RSI ≈ 1.0
    assert rsi.iloc[14] == pytest.approx(1.0, abs=0.01)
    # After the down moves, RSI should drop below 0.5
    assert rsi.iloc[-1] < 0.5


def test_features_normalized(sample_price_history):
    engine = PolymarketFeatureEngine()
    features = engine.create_features(sample_price_history)
    assert features is not None
    assert features.dtype == np.float32
    assert np.all(np.abs(features) < 1000)


async def test_sentiment_cache_hit(tmp_path):
    cache_db = str(tmp_path / "sentiment_cache.db")
    analyzer = SentimentAnalyzer(cache_db=cache_db, cache_minutes=30)

    # Manually insert a cached result
    conn = sqlite3.connect(cache_db)
    conn.execute(
        "INSERT INTO cache (market_id, timestamp, score, confidence, reasoning) VALUES (?, ?, ?, ?, ?)",
        ("mkt_001", time.time(), 0.8, 0.9, "cached reason"),
    )
    conn.commit()
    conn.close()

    # Mock _call_api to raise if invoked — proves cache was used
    with patch.object(analyzer, "_call_api", side_effect=RuntimeError("should not be called")):
        result = await analyzer.analyze_single("mkt_001", "Will X happen?")

    assert result.score == pytest.approx(0.8)
    assert result.confidence == pytest.approx(0.9)
    assert result.reasoning == "cached reason"


async def test_sentiment_batch_size(tmp_path):
    cache_db = str(tmp_path / "sentiment_cache.db")
    analyzer = SentimentAnalyzer(cache_db=cache_db, cache_minutes=30)

    markets = [(f"mkt_{i}", f"Question {i}?", f"Description {i}") for i in range(5)]
    fake_response = [{"score": 0.1 * i, "confidence": 0.5, "reasoning": f"reason {i}"} for i in range(5)]

    with (
        patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
        patch.object(
            analyzer,
            "_call_api",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_api,
    ):
        results = await analyzer.analyze_batch(markets)

    # All 5 uncached markets should be sent in a single API call
    assert mock_api.call_count == 1
    assert len(results) == 5
