import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


class DrawdownController:
    def __init__(
        self,
        reduce_at: float = 0.08,
        stop_at: float = 0.15,
        emergency_at: float = 0.20,
        initial_bankroll: float = 0.0,
    ) -> None:
        self.reduce_at: float = reduce_at
        self.stop_at: float = stop_at
        self.emergency_at: float = emergency_at
        self.peak: float = initial_bankroll
        self.daily_loss_limit_usdc: float | None = None
        self._daily_pnl: float = 0.0
        self._daily_reset_date: str = datetime.now(UTC).strftime("%Y-%m-%d")

        if not (0 < self.reduce_at < self.stop_at):
            logger.warning(
                "Drawdown thresholds may be misconfigured: reduce=%.2f stop=%.2f",
                self.reduce_at,
                self.stop_at,
            )

    def update(self, current_bankroll: float) -> None:
        self.peak = max(self.peak, current_bankroll)

    def reset_peak(self, current_bankroll: float) -> None:
        """Reset the high-water mark after manual intervention (e.g. kill switch recovery).

        Without this, the drawdown controller stays in emergency mode
        permanently after a recovery because peak remains at the pre-drawdown
        high.
        """
        self.peak = current_bankroll
        logger.warning("Drawdown peak reset to $%.2f", current_bankroll)

    def record_trade_pnl(self, pnl: float) -> None:
        """Record a trade's P&L for daily tracking. Resets at UTC midnight."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            logger.info(
                "Daily P&L reset: previous day %s had P&L $%.2f",
                self._daily_reset_date,
                self._daily_pnl,
            )
            self._daily_pnl = 0.0
            self._daily_reset_date = today
        self._daily_pnl += pnl

    @property
    def daily_limit_hit(self) -> bool:
        """Check if daily loss limit has been breached."""
        if self.daily_loss_limit_usdc is None:
            return False
        return self._daily_pnl <= -abs(self.daily_loss_limit_usdc)

    @property
    def daily_pnl(self) -> float:
        """Current day's cumulative P&L."""
        return self._daily_pnl

    def get_multiplier(self, current_bankroll: float) -> float:
        if self.peak <= 0:
            return 1.0
        if self.daily_limit_hit:
            logger.warning(
                "Daily loss limit hit ($%.2f): pausing trading",
                self._daily_pnl,
            )
            return 0.0
        drawdown = (self.peak - current_bankroll) / self.peak
        if drawdown >= self.emergency_at:
            return 0.0
        if drawdown >= self.stop_at:
            return 0.0
        if drawdown >= self.reduce_at:
            return 1.0 - (drawdown - self.reduce_at) / (self.stop_at - self.reduce_at)
        return 1.0

    def should_close_all(self, current_bankroll: float) -> bool:
        if self.peak <= 0:
            return False
        drawdown = (self.peak - current_bankroll) / self.peak
        return drawdown >= self.emergency_at

    def get_drawdown(self, current_bankroll: float) -> float:
        """Return the current drawdown as a fraction (0.0–1.0)."""
        if self.peak <= 0:
            return 0.0
        return (self.peak - current_bankroll) / self.peak

    def get_status(self, current_bankroll: float) -> dict:
        """Return a status dict with drawdown %, multiplier, and tier name.

        Returns:
            A dict with keys ``drawdown_pct``, ``multiplier``, and ``tier``.
            The tier is one of ``"normal"``, ``"reducing"``, ``"stopped"``,
            or ``"emergency"``.
        """
        dd = self.get_drawdown(current_bankroll)
        mult = self.get_multiplier(current_bankroll)

        if dd >= self.emergency_at:
            tier = "emergency"
        elif dd >= self.stop_at:
            tier = "stopped"
        elif dd >= self.reduce_at:
            tier = "reducing"
        else:
            tier = "normal"

        return {
            "drawdown_pct": round(dd * 100, 2),
            "multiplier": round(mult, 4),
            "tier": tier,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_limit_hit": self.daily_limit_hit,
        }

    # --- Threshold properties ---

    @property
    def reduce_threshold(self) -> float:
        """Drawdown fraction at which position sizes start reducing."""
        return self.reduce_at

    @property
    def stop_threshold(self) -> float:
        """Drawdown fraction at which new trading stops."""
        return self.stop_at

    @property
    def emergency_threshold(self) -> float:
        """Drawdown fraction at which all positions should be closed."""
        return self.emergency_at


class DailyLossTracker:
    """Tracks daily P&L with UTC midnight reset. Halts trading when daily loss limit is hit.

    Args:
        daily_loss_limit_usdc: Maximum allowed daily loss in USDC (positive number).
    """

    def __init__(self, daily_loss_limit_usdc: float) -> None:
        if daily_loss_limit_usdc <= 0:
            raise ValueError("daily_loss_limit_usdc must be positive")
        self.daily_loss_limit_usdc = daily_loss_limit_usdc
        self._daily_pnl: float = 0.0
        self._reset_date: str = self._today_utc()

    @staticmethod
    def _today_utc() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _maybe_reset(self) -> None:
        today = self._today_utc()
        if today != self._reset_date:
            logger.info(
                "DailyLossTracker: UTC midnight reset. Previous day %s P&L: $%.2f",
                self._reset_date,
                self._daily_pnl,
            )
            self._daily_pnl = 0.0
            self._reset_date = today

    def record_pnl(self, pnl: float) -> None:
        """Record a trade P&L (can be positive or negative)."""
        self._maybe_reset()
        self._daily_pnl += pnl

    @property
    def daily_pnl(self) -> float:
        self._maybe_reset()
        return self._daily_pnl

    @property
    def daily_loss(self) -> float:
        """Current daily loss as a positive number (0 if net positive)."""
        self._maybe_reset()
        return max(0.0, -self._daily_pnl)

    @property
    def is_limit_hit(self) -> bool:
        """Returns True if daily loss limit has been reached."""
        self._maybe_reset()
        if self._daily_pnl >= 0:
            return False
        return abs(self._daily_pnl) >= self.daily_loss_limit_usdc

    def assert_trading_allowed(self) -> None:
        """Raise RuntimeError if daily loss limit is hit."""
        if self.is_limit_hit:
            raise RuntimeError(
                f"Daily loss limit hit: lost ${self.daily_loss:.2f} USDC today "
                f"(limit: ${self.daily_loss_limit_usdc:.2f}). Trading halted until UTC midnight."
            )
