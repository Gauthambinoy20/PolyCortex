"""Position reconciliation between executor state and tracker DB.

Detects and resolves inconsistencies such as:
- Phantom positions: DB says open but executor has no matching order
- Stale positions: open longer than a configurable threshold with no updates
- Orphan orders: executor has orders not tracked in DB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polymarket_agent.execution.executor import OrderExecutor
    from polymarket_agent.tracking.tracker import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationIssue:
    """A single inconsistency found during reconciliation."""

    issue_type: str  # phantom_position, stale_position, orphan_order, size_mismatch
    market_id: str
    description: str
    severity: str  # info, warning, critical
    auto_resolved: bool = False
    resolution: str = ""


class PositionReconciler:
    """Cross-checks executor order state against the tracker database.

    Call ``reconcile()`` once per trading cycle to detect and optionally
    auto-resolve inconsistencies.  Issues are returned so the caller can
    route them through the alert system.
    """

    def __init__(
        self,
        *,
        stale_hours: float = 48.0,
        auto_close_phantoms: bool = False,
    ) -> None:
        self._stale_hours = stale_hours
        self._auto_close_phantoms = auto_close_phantoms

    def reconcile(
        self,
        tracker: PerformanceTracker,
        executor: OrderExecutor,
    ) -> list[ReconciliationIssue]:
        """Run all reconciliation checks and return any issues found.

        Args:
            tracker: The performance tracker with DB state.
            executor: The order executor with in-memory order state.

        Returns:
            List of ``ReconciliationIssue`` objects.  Empty means clean.
        """
        issues: list[ReconciliationIssue] = []
        issues.extend(self._check_phantom_positions(tracker, executor))
        issues.extend(self._check_stale_positions(tracker))
        issues.extend(self._check_orphan_orders(tracker, executor))
        issues.extend(self._check_size_mismatches(tracker, executor))

        if issues:
            logger.warning(
                "Reconciliation found %d issue(s): %s",
                len(issues),
                ", ".join(f"{i.issue_type}({i.market_id})" for i in issues),
            )
        else:
            logger.debug("Reconciliation clean — no issues found")

        return issues

    def _check_phantom_positions(
        self,
        tracker: PerformanceTracker,
        executor: OrderExecutor,
    ) -> list[ReconciliationIssue]:
        """Positions the DB thinks are open but the executor has no record of."""
        issues: list[ReconciliationIssue] = []
        open_positions = tracker.get_open_positions()

        executor_market_ids = {
            order.market_id for order in executor.orders.values() if order.status not in ("cancelled", "expired")
        }

        for pos in open_positions:
            market_id = pos.get("market_id", "")
            local_order_id = pos.get("local_order_id")

            if local_order_id and local_order_id in executor.orders:
                continue
            if market_id in executor_market_ids:
                continue

            issue = ReconciliationIssue(
                issue_type="phantom_position",
                market_id=market_id,
                description=(
                    f"Trade #{pos.get('id')} open in DB for {market_id} "
                    f"(${pos.get('size_usdc', 0):.2f}) but no matching "
                    f"executor order found"
                ),
                severity="warning",
            )

            if self._auto_close_phantoms:
                trade_id = pos.get("id")
                if trade_id is not None:
                    entry_price = pos.get("entry_price", 0.0)
                    tracker.close_position(
                        trade_id,
                        exit_price=entry_price,
                        status="closed",
                        exit_reason="phantom_reconciliation",
                    )
                    issue.auto_resolved = True
                    issue.resolution = "Auto-closed at entry price (zero PnL)"
                    logger.info(
                        "Auto-closed phantom position #%d for %s",
                        trade_id,
                        market_id,
                    )

            issues.append(issue)

        return issues

    def _check_stale_positions(
        self,
        tracker: PerformanceTracker,
    ) -> list[ReconciliationIssue]:
        """Open positions with no update beyond the staleness threshold."""
        issues: list[ReconciliationIssue] = []
        open_positions = tracker.get_open_positions()
        now = datetime.now(UTC)
        cutoff = now - timedelta(hours=self._stale_hours)

        for pos in open_positions:
            last_update = pos.get("last_fill_at") or pos.get("timestamp")
            if not last_update:
                continue

            try:
                if isinstance(last_update, str):
                    ts = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                else:
                    ts = last_update
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
            except (ValueError, AttributeError):
                continue

            if ts < cutoff:
                hours_stale = (now - ts).total_seconds() / 3600
                issues.append(
                    ReconciliationIssue(
                        issue_type="stale_position",
                        market_id=pos.get("market_id", ""),
                        description=(
                            f"Trade #{pos.get('id')} open for {hours_stale:.1f}h "
                            f"with no updates (threshold: {self._stale_hours}h)"
                        ),
                        severity="info" if hours_stale < self._stale_hours * 2 else "warning",
                    )
                )

        return issues

    def _check_orphan_orders(
        self,
        tracker: PerformanceTracker,
        executor: OrderExecutor,
    ) -> list[ReconciliationIssue]:
        """Executor orders with no matching DB trade record."""
        issues: list[ReconciliationIssue] = []

        tracked_order_ids: set[str] = set()
        for pos in tracker.get_open_positions():
            oid = pos.get("local_order_id")
            if oid:
                tracked_order_ids.add(oid)

        for order_id, order in executor.orders.items():
            if order.status in ("cancelled", "expired"):
                continue
            if order.order_kind != "entry":
                continue
            if order_id in tracked_order_ids:
                continue

            issues.append(
                ReconciliationIssue(
                    issue_type="orphan_order",
                    market_id=order.market_id,
                    description=(
                        f"Executor order {order_id} ({order.direction} "
                        f"${order.size_usdc:.2f} @ {order.price:.3f}) "
                        f"has no matching DB trade"
                    ),
                    severity="warning",
                )
            )

        return issues

    def _check_size_mismatches(
        self,
        tracker: PerformanceTracker,
        executor: OrderExecutor,
    ) -> list[ReconciliationIssue]:
        """DB trade size doesn't match executor filled size."""
        issues: list[ReconciliationIssue] = []

        for pos in tracker.get_open_positions():
            local_order_id = pos.get("local_order_id")
            if not local_order_id or local_order_id not in executor.orders:
                continue

            order = executor.orders[local_order_id]
            db_filled = pos.get("filled_size_usdc") or pos.get("size_usdc") or 0.0
            exec_filled = order.filled_size_usdc

            if exec_filled <= 0:
                continue

            diff = abs(float(db_filled) - exec_filled)
            if diff > 0.01:
                issues.append(
                    ReconciliationIssue(
                        issue_type="size_mismatch",
                        market_id=order.market_id,
                        description=(
                            f"Trade #{pos.get('id')} DB filled=${float(db_filled):.2f} "
                            f"vs executor filled=${exec_filled:.2f} "
                            f"(diff=${diff:.2f})"
                        ),
                        severity="warning" if diff > 1.0 else "info",
                    )
                )

        return issues
