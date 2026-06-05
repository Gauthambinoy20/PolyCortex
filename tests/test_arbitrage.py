from polymarket_agent.execution.arbitrage import ArbitrageScanner


class TestArbitrageScanner:
    def test_profitable_arbitrage(self):
        scanner = ArbitrageScanner(fee=0.02)
        markets = [
            {
                "market_id": "arb1",
                "question": "Will X happen?",
                "yes_ask": 0.35,
                "no_ask": 0.35,
                "spread": 0.01,
            },
        ]
        opps = scanner.scan(markets)
        assert len(opps) == 1
        assert opps[0].profit_after_costs > 0
        assert opps[0].market_id == "arb1"

    def test_no_arbitrage(self):
        scanner = ArbitrageScanner(fee=0.02)
        markets = [
            {
                "market_id": "no_arb",
                "question": "Will Y happen?",
                "yes_ask": 0.55,
                "no_ask": 0.50,
                "spread": 0.02,
            },
        ]
        opps = scanner.scan(markets)
        assert len(opps) == 0

    def test_borderline_arbitrage(self):
        scanner = ArbitrageScanner(fee=0.02)
        # Sum exactly 1.0 → gap=0.0, after fees not profitable
        markets = [
            {
                "market_id": "border",
                "question": "Borderline?",
                "yes_ask": 0.50,
                "no_ask": 0.50,
                "spread": 0.02,
            },
        ]
        opps = scanner.scan(markets)
        assert len(opps) == 0

    def test_scan_multiple_markets(self):
        scanner = ArbitrageScanner(fee=0.02)
        markets = [
            # Profitable pair
            {
                "market_id": "m1",
                "question": "Q1",
                "yes_ask": 0.30,
                "no_ask": 0.30,
                "spread": 0.01,
            },
            # Not profitable
            {
                "market_id": "m2",
                "question": "Q2",
                "yes_ask": 0.55,
                "no_ask": 0.50,
                "spread": 0.02,
            },
            # Not profitable
            {
                "market_id": "m3",
                "question": "Q3",
                "yes_ask": 0.48,
                "no_ask": 0.53,
                "spread": 0.02,
            },
            # Not profitable
            {
                "market_id": "m4",
                "question": "Q4",
                "yes_ask": 0.60,
                "no_ask": 0.45,
                "spread": 0.02,
            },
        ]
        opps = scanner.scan(markets)
        assert len(opps) == 1
        assert opps[0].market_id == "m1"
