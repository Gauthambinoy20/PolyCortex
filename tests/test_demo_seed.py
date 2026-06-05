from pathlib import Path

from polymarket_agent.demo import seed_demo_workspace
from polymarket_agent.tracking.tracker import PerformanceTracker


def test_seed_demo_workspace(tmp_path):
    result = seed_demo_workspace(tmp_path, reset=True, trade_count=12, bankroll=500.0)

    data_dir = Path(tmp_path) / "data"
    assert result["trade_count"] == 12
    assert (data_dir / "demo_trades.db").exists()
    assert (data_dir / "alerts.jsonl").exists()
    assert (data_dir / "scan_summary.json").exists()
    assert (data_dir / "learned_weights.json").exists()

    tracker = PerformanceTracker(str(data_dir / "demo_trades.db"))
    metrics = tracker.get_metrics()
    tracker.close()

    assert metrics["open_count"] > 0
    assert metrics["trade_count"] > 0
    assert "crypto" in metrics["exposure_by_category"] or metrics["exposure_by_category"]
