from polymarket_agent.models.calibration import ProbabilityCalibrator


def test_calibrator_returns_identity_without_fit(tmp_path):
    calibrator = ProbabilityCalibrator(path=str(tmp_path / "probability_calibration.json"))
    assert calibrator.calibrate(0.73) == 0.73


def test_calibrator_fit_save_and_load(tmp_path):
    calibrator = ProbabilityCalibrator(
        path=str(tmp_path / "probability_calibration.json"),
        min_samples=6,
    )
    metrics = calibrator.fit(
        predictions=[0.80, 0.75, 0.70, 0.45, 0.40, 0.35],
        outcomes=[1, 1, 1, 0, 0, 0],
        trade_count=6,
    )

    assert metrics is not None
    assert calibrator.is_fitted
    assert metrics.brier_after <= metrics.brier_before
    assert calibrator.save() is True

    loaded = ProbabilityCalibrator(
        path=str(tmp_path / "probability_calibration.json"),
        min_samples=6,
    )
    assert loaded.load() is True
    assert loaded.is_fitted
    assert loaded.calibrate(0.78) >= loaded.calibrate(0.42)


def test_calibrator_skips_fit_without_class_balance(tmp_path):
    calibrator = ProbabilityCalibrator(
        path=str(tmp_path / "probability_calibration.json"),
        min_samples=4,
    )
    metrics = calibrator.fit(
        predictions=[0.7, 0.75, 0.8, 0.85],
        outcomes=[1, 1, 1, 1],
        trade_count=4,
    )
    assert metrics is None
    assert calibrator.is_fitted is False
