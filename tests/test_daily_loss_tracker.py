import pytest

from polymarket_agent.risk.drawdown import DailyLossTracker


def test_constructor_rejects_non_positive_limit():
    with pytest.raises(ValueError):
        DailyLossTracker(0.0)
    with pytest.raises(ValueError):
        DailyLossTracker(-5.0)


def test_record_accumulates_pnl():
    t = DailyLossTracker(100.0)
    t.record_pnl(10.0)
    t.record_pnl(-5.0)
    assert t.daily_pnl == pytest.approx(5.0)


def test_positive_pnl_never_triggers_limit():
    t = DailyLossTracker(50.0)
    t.record_pnl(200.0)
    assert t.is_limit_hit is False
    assert t.daily_loss == 0.0
    t.assert_trading_allowed()


def test_limit_hit_at_threshold():
    t = DailyLossTracker(50.0)
    t.record_pnl(-30.0)
    assert t.is_limit_hit is False
    t.record_pnl(-25.0)
    assert t.is_limit_hit is True
    assert t.daily_loss == pytest.approx(55.0)
    with pytest.raises(RuntimeError):
        t.assert_trading_allowed()


def test_midnight_reset(monkeypatch):
    t = DailyLossTracker(50.0)
    t.record_pnl(-40.0)
    t._reset_date = "1999-01-01"  # simulate crossing midnight
    assert t.daily_pnl == 0.0
    assert t.is_limit_hit is False
