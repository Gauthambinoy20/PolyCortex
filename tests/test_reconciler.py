"""Tests for the position reconciliation module."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from polymarket_agent.tracking.reconciler import PositionReconciler


def _make_executor(orders=None):
    executor = MagicMock()
    executor.orders = orders or {}
    return executor


def _make_order(
    order_id="ord1",
    market_id="mkt1",
    direction="YES",
    size_usdc=10.0,
    price=0.5,
    status="filled",
    filled_size_usdc=10.0,
    order_kind="entry",
):
    order = MagicMock()
    order.order_id = order_id
    order.market_id = market_id
    order.direction = direction
    order.size_usdc = size_usdc
    order.price = price
    order.status = status
    order.filled_size_usdc = filled_size_usdc
    order.order_kind = order_kind
    return order


def _make_tracker(open_positions=None):
    tracker = MagicMock()
    tracker.get_open_positions.return_value = open_positions or []
    return tracker


class TestPhantomPositions:
    def test_no_issues_when_synced(self):
        order = _make_order(order_id="ord1", market_id="mkt1")
        executor = _make_executor({"ord1": order})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "ord1", "size_usdc": 10.0},
            ]
        )

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        phantoms = [i for i in issues if i.issue_type == "phantom_position"]
        assert len(phantoms) == 0

    def test_detects_phantom(self):
        executor = _make_executor({})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "missing_ord", "size_usdc": 10.0},
            ]
        )

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        phantoms = [i for i in issues if i.issue_type == "phantom_position"]
        assert len(phantoms) == 1
        assert phantoms[0].market_id == "mkt1"
        assert not phantoms[0].auto_resolved

    def test_auto_closes_phantom(self):
        executor = _make_executor({})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "missing_ord", "size_usdc": 10.0, "entry_price": 0.5},
            ]
        )

        reconciler = PositionReconciler(auto_close_phantoms=True)
        issues = reconciler.reconcile(tracker, executor)
        phantoms = [i for i in issues if i.issue_type == "phantom_position"]
        assert len(phantoms) == 1
        assert phantoms[0].auto_resolved
        tracker.close_position.assert_called_once()


class TestStalePositions:
    def test_no_stale_when_recent(self):
        now = datetime.now(UTC)
        tracker = _make_tracker(
            [
                {
                    "id": 1,
                    "market_id": "mkt1",
                    "local_order_id": "ord1",
                    "timestamp": now.isoformat(),
                    "last_fill_at": None,
                },
            ]
        )
        order = _make_order(order_id="ord1", market_id="mkt1")
        executor = _make_executor({"ord1": order})

        reconciler = PositionReconciler(stale_hours=48.0)
        issues = reconciler.reconcile(tracker, executor)
        stale = [i for i in issues if i.issue_type == "stale_position"]
        assert len(stale) == 0

    def test_detects_stale(self):
        old = datetime.now(UTC) - timedelta(hours=72)
        tracker = _make_tracker(
            [
                {
                    "id": 1,
                    "market_id": "mkt1",
                    "local_order_id": "ord1",
                    "timestamp": old.isoformat(),
                    "last_fill_at": None,
                },
            ]
        )
        order = _make_order(order_id="ord1", market_id="mkt1")
        executor = _make_executor({"ord1": order})

        reconciler = PositionReconciler(stale_hours=48.0)
        issues = reconciler.reconcile(tracker, executor)
        stale = [i for i in issues if i.issue_type == "stale_position"]
        assert len(stale) == 1


class TestOrphanOrders:
    def test_no_orphans_when_tracked(self):
        order = _make_order(order_id="ord1", market_id="mkt1")
        executor = _make_executor({"ord1": order})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "ord1"},
            ]
        )

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        orphans = [i for i in issues if i.issue_type == "orphan_order"]
        assert len(orphans) == 0

    def test_detects_orphan(self):
        order = _make_order(order_id="ord_untracked", market_id="mkt2")
        executor = _make_executor({"ord_untracked": order})
        tracker = _make_tracker([])

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        orphans = [i for i in issues if i.issue_type == "orphan_order"]
        assert len(orphans) == 1
        assert orphans[0].market_id == "mkt2"

    def test_ignores_cancelled_orders(self):
        order = _make_order(order_id="ord1", market_id="mkt1", status="cancelled")
        executor = _make_executor({"ord1": order})
        tracker = _make_tracker([])

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        orphans = [i for i in issues if i.issue_type == "orphan_order"]
        assert len(orphans) == 0


class TestSizeMismatches:
    def test_no_mismatch_when_equal(self):
        order = _make_order(order_id="ord1", market_id="mkt1", filled_size_usdc=10.0)
        executor = _make_executor({"ord1": order})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "ord1", "filled_size_usdc": 10.0, "size_usdc": 10.0},
            ]
        )

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        mismatches = [i for i in issues if i.issue_type == "size_mismatch"]
        assert len(mismatches) == 0

    def test_detects_mismatch(self):
        order = _make_order(order_id="ord1", market_id="mkt1", filled_size_usdc=8.0)
        executor = _make_executor({"ord1": order})
        tracker = _make_tracker(
            [
                {"id": 1, "market_id": "mkt1", "local_order_id": "ord1", "filled_size_usdc": 10.0, "size_usdc": 10.0},
            ]
        )

        reconciler = PositionReconciler()
        issues = reconciler.reconcile(tracker, executor)
        mismatches = [i for i in issues if i.issue_type == "size_mismatch"]
        assert len(mismatches) == 1
        assert "diff=$2.00" in mismatches[0].description


class TestCleanReconciliation:
    def test_empty_state_is_clean(self):
        reconciler = PositionReconciler()
        issues = reconciler.reconcile(_make_tracker([]), _make_executor({}))
        assert issues == []
