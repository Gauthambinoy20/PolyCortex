import numpy as np
import pandas as pd

from polymarket_agent.models.regime import REGIMES, RegimeDetector


def _make_three_regime_data(n=150, seed=42):
    """Generate 3-regime synthetic features (stable, trending, volatile)."""
    np.random.seed(seed)
    volatility = pd.Series(
        np.concatenate(
            [
                np.random.normal(0.005, 0.001, n),
                np.random.normal(0.05, 0.01, n),
                np.random.normal(0.15, 0.03, n),
            ]
        )
    )
    volume_ratio = pd.Series(
        np.concatenate(
            [
                np.random.normal(1.0, 0.05, n),
                np.random.normal(1.5, 0.2, n),
                np.random.normal(3.0, 0.5, n),
            ]
        )
    )
    spread = pd.Series(
        np.concatenate(
            [
                np.random.normal(0.01, 0.002, n),
                np.random.normal(0.03, 0.005, n),
                np.random.normal(0.08, 0.02, n),
            ]
        )
    )
    return volatility, volume_ratio, spread


def test_hmm_fit_synthetic():
    volatility, volume_ratio, spread = _make_three_regime_data()
    detector = RegimeDetector()
    detector.fit(volatility, volume_ratio, spread)

    assert detector.fitted

    label, probs = detector.predict(0.01, 1.0, 0.02)
    assert label in REGIMES
    assert isinstance(probs, dict)
    assert all(isinstance(v, float) for v in probs.values())


def test_regime_stable_detection():
    volatility, volume_ratio, spread = _make_three_regime_data()
    detector = RegimeDetector()
    detector.fit(volatility, volume_ratio, spread)

    # Predict with stable-like inputs repeatedly
    np.random.seed(99)
    predictions = [
        detector.predict(
            volatility=0.005 + np.random.normal(0, 0.001),
            volume_ratio=1.0 + np.random.normal(0, 0.05),
            spread=0.01 + np.random.normal(0, 0.002),
        )[0]
        for _ in range(20)
    ]
    stable_count = predictions.count("stable")
    assert stable_count >= 15


def test_regime_save_load(tmp_path):
    volatility, volume_ratio, spread = _make_three_regime_data()
    detector = RegimeDetector()
    detector.fit(volatility, volume_ratio, spread)

    save_path = str(tmp_path / "regime_model.pkl")
    detector.save(save_path)

    loaded = RegimeDetector()
    assert loaded.load(save_path)
    assert loaded.fitted

    # Predictions from loaded model must match original
    test_points = [(0.01, 1.0, 0.02), (0.05, 1.5, 0.04), (0.12, 2.5, 0.07)]
    for v, vr, s in test_points:
        label_orig, _ = detector.predict(v, vr, s)
        label_loaded, _ = loaded.predict(v, vr, s)
        assert label_orig == label_loaded


def test_regime_labels():
    volatility, volume_ratio, spread = _make_three_regime_data()
    detector = RegimeDetector()
    detector.fit(volatility, volume_ratio, spread)

    # All three regime labels should be assigned
    assert set(detector.regime_labels.values()) == set(REGIMES)

    # predict should return one of the expected labels
    label, probs = detector.predict(0.05, 1.5, 0.04)
    assert label in REGIMES
