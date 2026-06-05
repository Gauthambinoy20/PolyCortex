from unittest.mock import MagicMock

import numpy as np
import torch

from polymarket_agent.models.calibration import ProbabilityCalibrator
from polymarket_agent.models.edge import UnifiedEdgeDetector


async def test_edge_known_scenario(sample_config):
    detector = UnifiedEdgeDetector(config=sample_config)
    market_data = {
        "midpoint": 0.50,
        "bid_depth": 2000.0,
        "ask_depth": 500.0,
        "volume_24h": 0.0,
        "market_id": "test_001",
        "question": "Test?",
        "category": "test",
    }
    result = await detector.estimate_edge(market_data, features=None, regime="stable")
    assert result.edge > 0
    assert result.direction == "YES"


async def test_edge_no_signal(sample_config):
    detector = UnifiedEdgeDetector(config=sample_config)
    market_data = {
        "midpoint": 0.50,
        "bid_depth": 1000.0,
        "ask_depth": 1000.0,
        "volume_24h": 100_000.0,
        "market_id": "test_002",
        "question": "Test?",
        "category": "test",
    }
    result = await detector.estimate_edge(market_data, features=None, regime="stable")
    assert result.edge < sample_config["min_edge"]


async def test_regime_weighting(sample_config):
    detector = UnifiedEdgeDetector(config=sample_config)

    # Mock TCN model returning a fixed probability
    mock_tcn = MagicMock()
    mock_tcn.return_value = (torch.tensor(0.70), None)
    detector.set_tcn_model(mock_tcn)

    features = np.random.rand(64, 10).astype(np.float32)
    market_data = {
        "midpoint": 0.50,
        "bid_depth": 1500.0,
        "ask_depth": 1000.0,
        "volume_24h": 0.0,
        "market_id": "test_003",
        "question": "Test?",
        "category": "test",
    }

    result_stable = await detector.estimate_edge(market_data, features, regime="stable")
    result_volatile = await detector.estimate_edge(market_data, features, regime="volatile")

    # Different regime weights (stable: 0.70/0.30 vs volatile: 0.80/0.20)
    # should produce different edge values for the same inputs
    assert result_stable.edge != result_volatile.edge


async def test_efficiency_penalty(sample_config):
    detector = UnifiedEdgeDetector(config=sample_config)
    base_data = {
        "midpoint": 0.50,
        "bid_depth": 2000.0,
        "ask_depth": 500.0,
        "market_id": "test_004",
        "question": "Test?",
        "category": "test",
    }

    low_vol = {**base_data, "volume_24h": 0.0}
    high_vol = {**base_data, "volume_24h": 1_000_000.0}

    r_low = await detector.estimate_edge(low_vol, features=None, regime="stable")
    r_high = await detector.estimate_edge(high_vol, features=None, regime="stable")

    # After removing the efficiency penalty (Issue #5), volume no longer
    # regresses the edge toward market price — both should produce equal edges.
    assert r_low.edge == r_high.edge


async def test_cross_market_signals(sample_config):
    detector = UnifiedEdgeDetector(config=sample_config)
    market_data = {
        "midpoint": 0.50,
        "bid_depth": 1000.0,
        "ask_depth": 1000.0,
        "volume_24h": 0.0,
        "market_id": "test_005",
        "question": "Test?",
        "category": "test",
        "related_markets": [
            {"price_change_24h": 0.10, "correlation": 0.8},
            {"price_change_24h": 0.05, "correlation": 0.6},
        ],
    }
    result = await detector.estimate_edge(market_data, features=None, regime="stable")
    assert result.signal_breakdown["cross_market"] != 0.0


async def test_probability_calibration_applied(sample_config, tmp_path):
    detector = UnifiedEdgeDetector(config=sample_config)
    calibrator = ProbabilityCalibrator(path=str(tmp_path / "probability_calibration.json"), min_samples=2)
    calibrator.x_thresholds = [0.01, 0.99]
    calibrator.y_thresholds = [0.01, 0.40]
    calibrator.last_fit_trade_count = 2
    detector.set_probability_calibrator(calibrator)

    market_data = {
        "midpoint": 0.50,
        "bid_depth": 2000.0,
        "ask_depth": 500.0,
        "volume_24h": 0.0,
        "market_id": "test_006",
        "question": "Test calibrated?",
        "category": "test",
    }

    result = await detector.estimate_edge(market_data, features=None, regime="stable")
    assert result.raw_estimated_prob is not None
    assert result.estimated_prob == result.calibrated_prob
    assert result.raw_estimated_prob > result.estimated_prob
