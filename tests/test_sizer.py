from polymarket_agent.risk.drawdown import DrawdownController
from polymarket_agent.risk.sizer import UnifiedPositionSizer


class TestKellySizing:
    def test_kelly_basic(self, sample_config, sample_edge_result):
        sizer = UnifiedPositionSizer(sample_config)
        position = sizer.calculate_position(
            edge_result=sample_edge_result,
            current_positions=[],
        )
        max_allowed = sample_config["bankroll"] * sample_config["max_position_pct"]
        assert 0 < position <= max_allowed
        assert isinstance(position, float)

    def test_kelly_respects_max_position(self, sample_config, sample_edge_result):
        sizer = UnifiedPositionSizer(sample_config)
        # Inflate estimated_prob to create very high edge
        big_edge = {**sample_edge_result, "estimated_prob": 0.99, "confidence": 1.0}
        position = sizer.calculate_position(
            edge_result=big_edge,
            current_positions=[],
        )
        max_allowed = sample_config["bankroll"] * sample_config["max_position_pct"]
        assert position <= max_allowed

    def test_kelly_zero_edge(self, sample_config, sample_edge_result):
        sizer = UnifiedPositionSizer(sample_config)
        # Edge <= 0: estimated_prob below market_price for YES direction
        no_edge = {**sample_edge_result, "estimated_prob": 0.30, "market_price": 0.65}
        position = sizer.calculate_position(
            edge_result=no_edge,
            current_positions=[],
        )
        assert position == 0.0

    def test_kelly_fraction_applied(self, sample_config, sample_edge_result):
        # Full Kelly config
        full_config = {
            **sample_config,
            "kelly_fraction": 1.0,
            "max_position_pct": 1.0,
            "max_category_exposure_pct": 1.0,
            "min_confidence": 0.0,
        }
        full_sizer = UnifiedPositionSizer(full_config)
        full_pos = full_sizer.calculate_position(
            edge_result=sample_edge_result,
            current_positions=[],
        )

        # Fractional Kelly config (0.25)
        frac_config = {
            **sample_config,
            "kelly_fraction": 0.25,
            "max_position_pct": 1.0,
            "max_category_exposure_pct": 1.0,
            "min_confidence": 0.0,
        }
        frac_sizer = UnifiedPositionSizer(frac_config)
        frac_pos = frac_sizer.calculate_position(
            edge_result=sample_edge_result,
            current_positions=[],
        )

        # Fractional should be approximately kelly_fraction * full
        if full_pos > 0 and frac_pos > 0:
            ratio = frac_pos / full_pos
            assert abs(ratio - 0.25) < 0.05

    def test_open_positions_accept_size_usdc_key(self, sample_config, sample_edge_result):
        sizer = UnifiedPositionSizer(sample_config)
        current_positions = [{"size_usdc": 50.0, "category": "crypto"}]
        position = sizer.calculate_position(
            edge_result=sample_edge_result,
            current_positions=current_positions,
        )
        assert position >= 0.0

    def test_min_confidence_blocks_trade(self, sample_config, sample_edge_result):
        guarded = {**sample_config, "min_confidence": 0.8}
        sizer = UnifiedPositionSizer(guarded)
        low_conf = {**sample_edge_result, "confidence": 0.6}
        position = sizer.calculate_position(
            edge_result=low_conf,
            current_positions=[],
        )
        assert position == 0.0


class TestDrawdown:
    def test_drawdown_multiplier(self, sample_config):
        ctrl = DrawdownController(
            reduce_at=sample_config["drawdown_reduce"],  # 0.08
            stop_at=sample_config["drawdown_stop"],  # 0.15
            emergency_at=sample_config["drawdown_emergency"],  # 0.20
        )
        peak = 1000.0
        ctrl.update(peak)

        # 0% drawdown
        assert ctrl.get_multiplier(1000.0) == 1.0

        # 5% drawdown — below reduce threshold
        assert ctrl.get_multiplier(950.0) == 1.0

        # 10% drawdown — between reduce (8%) and stop (15%)
        mult = ctrl.get_multiplier(900.0)
        assert 0 < mult < 1.0

        # 15% drawdown — at stop threshold
        assert ctrl.get_multiplier(850.0) == 0.0

    def test_drawdown_should_close_all(self, sample_config):
        ctrl = DrawdownController(
            reduce_at=sample_config["drawdown_reduce"],
            stop_at=sample_config["drawdown_stop"],
            emergency_at=sample_config["drawdown_emergency"],
        )
        ctrl.update(1000.0)

        # 20% drawdown (emergency) → close all
        assert ctrl.should_close_all(800.0) is True

        # 5% drawdown → do not close
        assert ctrl.should_close_all(950.0) is False
