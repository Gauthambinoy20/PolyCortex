import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ArbitrageOpportunity:
    market_id: str
    question: str
    yes_price: float
    no_price: float
    gap: float
    spread_cost: float
    fee: float
    profit_after_costs: float


class ArbitrageScanner:
    def __init__(self, fee: float = 0.02) -> None:
        self.fee: float = fee

    def scan(self, markets: list[dict]) -> list[ArbitrageOpportunity]:
        opportunities: list[ArbitrageOpportunity] = []
        for market in markets:
            yes_ask: float | None = market.get("yes_ask")
            no_ask: float | None = market.get("no_ask")
            if yes_ask is None or no_ask is None or yes_ask <= 0 or no_ask <= 0:
                continue
            total = yes_ask + no_ask
            gap = 1.0 - total
            spread_cost = market.get("spread", 0.02) * 2
            cost = spread_cost + self.fee
            if gap > cost:
                profit = gap - cost
                opp = ArbitrageOpportunity(
                    market_id=market.get("market_id", ""),
                    question=market.get("question", ""),
                    yes_price=yes_ask,
                    no_price=no_ask,
                    gap=gap,
                    spread_cost=spread_cost,
                    fee=self.fee,
                    profit_after_costs=profit,
                )
                opportunities.append(opp)
                logger.info(
                    "Arbitrage found: %s gap=%.4f profit=%.4f",
                    opp.market_id,
                    opp.gap,
                    opp.profit_after_costs,
                )
        opportunities.sort(key=lambda o: o.profit_after_costs, reverse=True)
        return opportunities

    def check_single(self, market_data: dict) -> ArbitrageOpportunity | None:
        yes_ask: float | None = market_data.get("yes_ask")
        no_ask: float | None = market_data.get("no_ask")
        if yes_ask is None or no_ask is None or yes_ask <= 0 or no_ask <= 0:
            return None
        total = yes_ask + no_ask
        gap = 1.0 - total
        spread_cost = market_data.get("spread", 0.02) * 2
        cost = spread_cost + self.fee
        if gap > cost:
            profit = gap - cost
            opp = ArbitrageOpportunity(
                market_id=market_data.get("market_id", ""),
                question=market_data.get("question", ""),
                yes_price=yes_ask,
                no_price=no_ask,
                gap=gap,
                spread_cost=spread_cost,
                fee=self.fee,
                profit_after_costs=profit,
            )
            logger.info(
                "Arbitrage found: %s gap=%.4f profit=%.4f",
                opp.market_id,
                opp.gap,
                opp.profit_after_costs,
            )
            return opp
        return None
