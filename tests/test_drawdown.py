"""Tests for drawdown controller with daily limits."""

import pytest

from polymarket_agent.risk.drawdown import DrawdownController


class TestDrawdownController:
    def test_no_drawdown(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.update(1000)
        assert dc.get_multiplier(1000) == 1.0

    def test_reduce_tier(self):
        dc = DrawdownController(reduce_at=0.08, stop_at=0.15, initial_bankroll=1000)
        dc.update(1000)
        mult = dc.get_multiplier(900)  # 10% drawdown
        assert 0 < mult < 1.0

    def test_stop_tier(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.update(1000)
        assert dc.get_multiplier(800) == 0.0  # 20% drawdown

    def test_emergency_close_all(self):
        dc = DrawdownController(emergency_at=0.20, initial_bankroll=1000)
        dc.update(1000)
        assert dc.should_close_all(790)

    def test_status_includes_tier(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.update(1000)
        status = dc.get_status(1000)
        assert status["tier"] == "normal"
        assert status["multiplier"] == 1.0

    def test_get_drawdown(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.update(1000)
        assert dc.get_drawdown(900) == pytest.approx(0.10)

    def test_reset_peak(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.update(1000)
        dc.reset_peak(800)
        assert dc.peak == 800
        assert dc.get_multiplier(800) == 1.0

    def test_threshold_properties(self):
        dc = DrawdownController(reduce_at=0.05, stop_at=0.10, emergency_at=0.25)
        assert dc.reduce_threshold == 0.05
        assert dc.stop_threshold == 0.10
        assert dc.emergency_threshold == 0.25


class TestDailyLimits:
    def test_no_limit_by_default(self):
        dc = DrawdownController(initial_bankroll=1000)
        assert not dc.daily_limit_hit

    def test_daily_pnl_tracking(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.daily_loss_limit_usdc = 50.0
        dc.record_trade_pnl(-20)
        dc.record_trade_pnl(-10)
        assert dc.daily_pnl == pytest.approx(-30.0)
        assert not dc.daily_limit_hit

    def test_daily_limit_breached(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.daily_loss_limit_usdc = 50.0
        dc.record_trade_pnl(-30)
        dc.record_trade_pnl(-25)
        assert dc.daily_limit_hit
        assert dc.get_multiplier(950) == 0.0

    def test_positive_pnl_no_limit(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.daily_loss_limit_usdc = 50.0
        dc.record_trade_pnl(100)
        assert dc.daily_pnl == pytest.approx(100.0)
        assert not dc.daily_limit_hit

    def test_status_includes_daily(self):
        dc = DrawdownController(initial_bankroll=1000)
        dc.daily_loss_limit_usdc = 50.0
        dc.record_trade_pnl(-10)
        status = dc.get_status(1000)
        assert "daily_pnl" in status
        assert "daily_limit_hit" in status
        assert status["daily_pnl"] == -10.0
