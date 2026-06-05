"""Tests for slippage protection."""

import pytest

from polymarket_agent.execution.slippage import check_slippage_ok, estimate_slippage


class TestEstimateSlippage:
    def test_empty_book(self):
        est = estimate_slippage(100, "BUY", [], [])
        assert not est.sufficient_liquidity
        assert est.slippage_bps == float("inf")

    def test_small_buy_single_level(self):
        bids = [(0.50, 1000)]
        asks = [(0.51, 1000)]
        est = estimate_slippage(100, "BUY", bids, asks)
        assert est.sufficient_liquidity
        assert est.estimated_vwap == pytest.approx(0.51)
        assert est.levels_consumed == 1

    def test_large_buy_multiple_levels(self):
        bids = [(0.50, 500)]
        asks = [(0.51, 200), (0.52, 200), (0.53, 200)]
        est = estimate_slippage(500, "BUY", bids, asks)
        assert est.levels_consumed >= 2
        assert est.estimated_vwap > 0.51

    def test_insufficient_liquidity(self):
        bids = [(0.50, 100)]
        asks = [(0.51, 50)]
        est = estimate_slippage(200, "BUY", bids, asks)
        assert not est.sufficient_liquidity

    def test_sell_side(self):
        bids = [(0.50, 500), (0.49, 300)]
        asks = [(0.51, 500)]
        est = estimate_slippage(100, "SELL", bids, asks)
        assert est.sufficient_liquidity
        assert est.estimated_vwap == pytest.approx(0.50)

    def test_slippage_increases_with_size(self):
        bids = [(0.50, 500), (0.49, 300)]
        asks = [(0.51, 500), (0.52, 300)]
        small = estimate_slippage(100, "BUY", bids, asks)
        large = estimate_slippage(700, "BUY", bids, asks)
        assert large.slippage_bps >= small.slippage_bps

    def test_slippage_pct_property(self):
        bids = [(0.50, 1000)]
        asks = [(0.51, 1000)]
        est = estimate_slippage(100, "BUY", bids, asks)
        assert est.slippage_pct == pytest.approx(est.slippage_bps / 10_000)

    def test_midpoint_calculation(self):
        bids = [(0.40, 1000)]
        asks = [(0.60, 1000)]
        est = estimate_slippage(100, "BUY", bids, asks)
        assert est.midpoint == pytest.approx(0.50)


class TestCheckSlippageOk:
    def test_ok_within_threshold(self):
        bids = [(0.50, 1000)]
        asks = [(0.51, 1000)]
        ok, est = check_slippage_ok(100, "BUY", bids, asks, max_slippage_bps=500)
        assert ok

    def test_rejected_exceeds_threshold(self):
        bids = [(0.50, 50)]
        asks = [(0.51, 50), (0.60, 50)]
        ok, est = check_slippage_ok(90, "BUY", bids, asks, max_slippage_bps=10)
        assert not ok
