from polymarket_agent.execution.executor import InsufficientBalanceError, OrderExecutor


def test_submitted_orders_table_created(tmp_path):
    db = tmp_path / "trades.db"
    OrderExecutor(config={"dry_run": True, "db_path": str(db)})
    assert db.exists()


def test_persist_order_intent_returns_false_on_duplicate(tmp_path):
    db = tmp_path / "trades.db"
    execu = OrderExecutor(config={"dry_run": True, "db_path": str(db)})
    assert execu._persist_order_intent("abc", "market1", "YES", 10.0) is True
    assert execu._persist_order_intent("abc", "market1", "YES", 10.0) is False


def test_update_order_status(tmp_path):
    db = tmp_path / "trades.db"
    execu = OrderExecutor(config={"dry_run": True, "db_path": str(db)})
    execu._persist_order_intent("id1", "m", "YES", 5.0)
    execu._update_order_status_db("id1", "filled")

    import sqlite3

    with sqlite3.connect(str(db)) as conn:
        (status,) = conn.execute("SELECT status FROM submitted_orders WHERE client_order_id=?", ("id1",)).fetchone()
    assert status == "filled"


def test_insufficient_balance_error_is_exception():
    assert issubclass(InsufficientBalanceError, Exception)
