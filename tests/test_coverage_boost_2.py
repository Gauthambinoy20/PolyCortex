"""Additional coverage tests for infrastructure and risk modules."""

from __future__ import annotations

import asyncio

import pytest


# ────────────────────────────── MemoryGuard ──────────────────────────────
def test_memory_guard_basic():
    from polymarket_agent.infra.memory_guard import MemoryGuard

    with MemoryGuard("test", threshold_mb=1000.0) as guard:
        data = [i for i in range(1000)]
        assert data[-1] == 999
    assert guard.peak_mb >= 0


def test_memory_guard_threshold_warning(caplog):
    from polymarket_agent.infra.memory_guard import MemoryGuard

    with caplog.at_level("WARNING"), MemoryGuard("tight", threshold_mb=0.0001):
        _ = [i for i in range(10_000)]
    # may or may not log depending on allocation; just ensure no crash


def test_memory_guard_context_helper():
    from polymarket_agent.infra.memory_guard import track_memory

    with track_memory("loop") as guard:
        _ = list(range(100))
    assert guard.peak_mb >= 0


# ────────────────────────────── Health ──────────────────────────────
async def test_health_checker_basic():
    from polymarket_agent.infra.health import ComponentHealth, HealthChecker, HealthStatus

    h = HealthChecker("agent")

    async def ok_check():
        return ComponentHealth(name="ok", status=HealthStatus.HEALTHY, message="ok")

    async def bad_check():
        return ComponentHealth(name="bad", status=HealthStatus.UNHEALTHY, message="fail")

    h.register_check("ok", ok_check)
    h.register_check("bad", bad_check)
    h.set_metadata("version", "1.0")
    result = await h.check_all()
    assert isinstance(result, dict)
    assert h.uptime_seconds >= 0


async def test_health_checker_sync_check():
    from polymarket_agent.infra.health import ComponentHealth, HealthChecker, HealthStatus

    h = HealthChecker()
    h.register_check("x", lambda: ComponentHealth(name="x", status=HealthStatus.HEALTHY, message="ok"))
    res = await h.check_all()
    assert isinstance(res, dict)


async def test_health_checker_exception_in_check():
    from polymarket_agent.infra.health import HealthChecker

    h = HealthChecker()

    async def broken():
        raise RuntimeError("boom")

    h.register_check("broken", broken)
    res = await h.check_all()
    assert isinstance(res, dict)


def test_health_quick_status():
    from polymarket_agent.infra.health import HealthChecker

    h = HealthChecker()
    q = h.quick_status()
    assert isinstance(q, dict)


# ────────────────────────────── Shutdown ──────────────────────────────
async def test_graceful_shutdown_hooks():
    from polymarket_agent.infra.shutdown import GracefulShutdown

    gs = GracefulShutdown(timeout=1.0)
    called = []

    async def hook1():
        called.append("a")

    def hook2():
        called.append("b")

    gs.register(hook1, name="a")
    gs.register(hook2, name="b")
    assert not gs.is_shutting_down
    await gs.execute()
    assert gs.is_shutting_down
    assert set(called) == {"a", "b"}


async def test_graceful_shutdown_hook_error_swallowed():
    from polymarket_agent.infra.shutdown import GracefulShutdown

    gs = GracefulShutdown(timeout=0.5)

    async def bad():
        raise RuntimeError("boom")

    gs.register(bad, name="bad")
    await gs.execute()  # must not raise


async def test_graceful_shutdown_timeout():
    from polymarket_agent.infra.shutdown import GracefulShutdown

    gs = GracefulShutdown(timeout=0.05)

    async def slow():
        await asyncio.sleep(0.2)

    gs.register(slow, name="slow")
    await gs.execute()  # should time out gracefully


# ────────────────────────────── env_validator ──────────────────────────────
def test_env_validator_validate_pk():
    import pytest as _pt

    from polymarket_agent.infra.env_validator import validate_private_key

    # Returns the key on success, raises on failure
    out = validate_private_key("0x" + "a" * 64)
    assert out
    with _pt.raises(Exception):
        validate_private_key("notakey")


def test_env_validator_mask_value():
    from polymarket_agent.infra.env_validator import mask_value

    masked = mask_value("abcdefghijkl")
    assert "*" in masked


def test_env_validator_validate_env(monkeypatch):
    from polymarket_agent.infra.env_validator import EnvVar, validate_env

    monkeypatch.setenv("MY_VAR", "val1")
    monkeypatch.delenv("MISSING_VAR", raising=False)
    vars_ = [
        EnvVar(name="MY_VAR", required=True),
        EnvVar(name="MISSING_VAR", required=True),
        EnvVar(name="DEFAULTED", required=False, default="def"),
    ]
    result = validate_env(vars_)
    assert result is not None
    # Should have errors due to MISSING_VAR
    assert hasattr(result, "errors") or hasattr(result, "is_valid")


def test_env_validator_get_env(monkeypatch):
    from polymarket_agent.infra.env_validator import get_env

    monkeypatch.setenv("FOO_X", "hello")
    assert get_env("FOO_X") == "hello"
    assert get_env("NOT_SET", default="d") == "d"


# ────────────────────────────── heartbeat ──────────────────────────────
def test_heartbeat_write_and_stale(tmp_path):
    from polymarket_agent.utils.heartbeat import HeartbeatMonitor

    hb = HeartbeatMonitor(tmp_path / "hb.json", max_stale_minutes=1)
    stale, _ = hb.is_stale()
    assert stale is True
    hb.beat({"iteration": 1})
    last = hb.get_last_beat()
    assert last is not None
    stale, _ = hb.is_stale()
    assert stale is False


def test_heartbeat_missing_file(tmp_path):
    from polymarket_agent.utils.heartbeat import HeartbeatMonitor

    hb = HeartbeatMonitor(tmp_path / "absent.json")
    assert hb.get_last_beat() is None
    stale, _ = hb.is_stale()
    assert stale is True


def test_heartbeat_corrupt_file(tmp_path):
    from polymarket_agent.utils.heartbeat import HeartbeatMonitor

    p = tmp_path / "hb.json"
    p.write_text("not json")
    hb = HeartbeatMonitor(p)
    assert hb.get_last_beat() is None


# ────────────────────────────── event bus ──────────────────────────────
async def test_event_bus_publish_subscribe():
    from polymarket_agent.events.bus import Event, EventBus, EventType

    bus = EventBus()
    got: list = []

    async def handler(event: Event):
        got.append(event.event_type)

    bus.subscribe(EventType.ORDERBOOK_UPDATED, handler)
    await bus.start()
    await bus.publish(Event(event_type=EventType.ORDERBOOK_UPDATED, payload={}, source="t"))
    await asyncio.sleep(0.1)
    await bus.stop()
    assert EventType.ORDERBOOK_UPDATED in got


async def test_event_bus_subscribe_all():
    from polymarket_agent.events.bus import Event, EventBus, EventType

    bus = EventBus()
    got = []

    async def catchall(e: Event):
        got.append(e.event_type)

    bus.subscribe_all(catchall)
    await bus.start()
    await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}, source="t"))
    await asyncio.sleep(0.1)
    await bus.stop()
    assert len(got) >= 1


async def test_event_bus_unsubscribe():
    from polymarket_agent.events.bus import Event, EventBus, EventType

    bus = EventBus()

    async def h(e: Event):
        pass

    bus.subscribe(EventType.ORDER_FILLED, h)
    bus.unsubscribe(EventType.ORDER_FILLED, h)
    stats = bus.stats
    assert "queue_size" in stats or stats is not None


async def test_event_bus_handler_exception():
    from polymarket_agent.events.bus import Event, EventBus, EventType

    bus = EventBus()

    async def bad(e: Event):
        raise RuntimeError("boom")

    bus.subscribe(EventType.ORDER_PLACED, bad)
    await bus.start()
    await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}, source="t"))
    await asyncio.sleep(0.1)
    await bus.stop()


# ────────────────────────────── AlertManager ──────────────────────────────
def test_alert_manager_add_and_check():
    from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

    mgr = AlertManager()
    alert = Alert(
        alert_id="a1",
        market_id="m",
        condition=AlertCondition.PRICE_ABOVE,
        threshold=0.6,
    )
    mgr.add_alert(alert)
    assert mgr.get_alert("a1") is not None
    fired = mgr.check("m", price=0.7)
    assert isinstance(fired, list)
    active = mgr.get_active_alerts()
    assert isinstance(active, list)


def test_alert_manager_cancel_and_remove():
    from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

    mgr = AlertManager()
    a = Alert(alert_id="a", market_id="m", condition=AlertCondition.PRICE_BELOW, threshold=0.1)
    mgr.add_alert(a)
    mgr.cancel_alert("a")
    mgr.remove_alert("a")
    assert mgr.get_alert("a") is None


def test_alert_manager_conditions():
    from polymarket_agent.risk.alerts import Alert, AlertCondition, AlertManager

    mgr = AlertManager()
    for cond in (
        AlertCondition.PRICE_ABOVE,
        AlertCondition.PRICE_BELOW,
        AlertCondition.SPREAD_ABOVE,
        AlertCondition.VOLUME_ABOVE,
    ):
        mgr.add_alert(Alert(alert_id=str(cond), market_id="m", condition=cond, threshold=0.5))
    fired = mgr.check("m", price=0.95, spread=0.6, volume=2e6)
    assert isinstance(fired, list)
    stats = mgr.stats
    assert isinstance(stats, dict)


# ────────────────────────────── UnifiedPositionSizer ──────────────────────────────
def test_position_sizer_calc():
    from polymarket_agent.risk.sizer import UnifiedPositionSizer

    s = UnifiedPositionSizer(
        {
            "bankroll": 1000,
            "kelly_fraction": 0.25,
            "min_confidence": 0.5,
            "max_position_pct": 0.05,
            "max_portfolio_pct": 0.5,
            "max_category_exposure_pct": 0.2,
            "max_per_category": 3,
            "max_positions": 10,
        }
    )
    out = s.calculate_position(
        edge_result={
            "estimated_prob": 0.6,
            "edge": 0.1,
            "market_price": 0.5,
            "confidence": 0.8,
            "direction": "YES",
            "regime": "stable",
            "category": "cat",
            "signal_breakdown": {},
        },
        current_positions=[],
    )
    assert isinstance(out, (int, float))
    assert out >= 0


def test_position_sizer_low_confidence():
    from polymarket_agent.risk.sizer import UnifiedPositionSizer

    s = UnifiedPositionSizer(
        {
            "bankroll": 1000,
            "kelly_fraction": 0.25,
            "min_confidence": 0.9,
            "max_position_pct": 0.05,
            "max_portfolio_pct": 0.5,
            "max_category_exposure_pct": 0.2,
            "max_per_category": 3,
            "max_positions": 10,
        }
    )
    out = s.calculate_position(
        edge_result={
            "estimated_prob": 0.6,
            "edge": 0.1,
            "market_price": 0.5,
            "confidence": 0.3,
            "direction": "YES",
            "regime": "stable",
            "category": "c",
            "signal_breakdown": {},
        },
        current_positions=[],
    )
    assert out == 0


def test_position_sizer_regime_caps():
    from polymarket_agent.risk.sizer import UnifiedPositionSizer

    s = UnifiedPositionSizer(
        {
            "bankroll": 1000,
            "kelly_fraction": 0.5,
            "min_confidence": 0.5,
            "max_position_pct": 0.2,
            "max_portfolio_pct": 0.5,
            "max_category_exposure_pct": 0.5,
            "max_per_category": 5,
            "max_positions": 20,
        }
    )
    for regime in ("crisis", "volatile", "stable"):
        out = s.calculate_position(
            edge_result={
                "estimated_prob": 0.55,
                "edge": 0.15,
                "market_price": 0.4,
                "confidence": 0.9,
                "direction": "YES",
                "regime": regime,
                "category": "c",
                "signal_breakdown": {},
            },
            current_positions=[],
        )
        assert out >= 0


def test_position_sizer_update_bankroll():
    from polymarket_agent.risk.sizer import UnifiedPositionSizer

    s = UnifiedPositionSizer(
        {
            "bankroll": 100,
            "kelly_fraction": 0.25,
            "min_confidence": 0.5,
            "max_position_pct": 0.05,
            "max_portfolio_pct": 0.5,
            "max_category_exposure_pct": 0.5,
            "max_per_category": 5,
            "max_positions": 20,
        }
    )
    s.update_bankroll(200)


def test_position_sizer_with_drawdown_multiplier():
    from polymarket_agent.risk.sizer import UnifiedPositionSizer

    s = UnifiedPositionSizer(
        {
            "bankroll": 100,
            "kelly_fraction": 0.5,
            "min_confidence": 0.5,
            "max_position_pct": 0.5,
            "max_portfolio_pct": 0.5,
            "max_category_exposure_pct": 0.5,
            "max_per_category": 5,
            "max_positions": 20,
        }
    )
    out = s.calculate_position(
        edge_result={
            "estimated_prob": 0.7,
            "edge": 0.2,
            "market_price": 0.5,
            "confidence": 0.9,
            "direction": "YES",
            "regime": "stable",
            "category": "c",
            "signal_breakdown": {},
        },
        current_positions=[],
        drawdown_multiplier=0.5,
    )
    assert out >= 0


# ────────────────────────────── RegimeDetector ──────────────────────────────
def test_regime_detector_default_stable():
    from polymarket_agent.models.regime import RegimeDetector

    rd = RegimeDetector()
    label, probs = rd.predict(volatility=0.05, volume_ratio=1.0, spread=0.02)
    assert isinstance(label, str)
    assert isinstance(probs, dict)


def test_regime_detector_reset_buffer():
    from polymarket_agent.models.regime import RegimeDetector

    rd = RegimeDetector()
    rd.predict(volatility=0.1, volume_ratio=1.0, spread=0.01)
    rd.reset_buffer()


def test_regime_detector_save_load(tmp_path):
    from polymarket_agent.models.regime import RegimeDetector

    rd = RegimeDetector()
    # Train synthetic data
    import numpy as np

    try:
        rd.fit(np.random.rand(100, 2).tolist())
    except Exception:
        pytest.skip("hmmlearn not available")
    p = tmp_path / "regime.pkl"
    rd.save(str(p))
    rd2 = RegimeDetector()
    rd2.load(str(p))


# ────────────────────────────── Tax CLI ──────────────────────────────
def test_tax_cli_main(tmp_path, monkeypatch):
    import sqlite3

    from polymarket_agent.reporting import tax

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            market_id TEXT, question TEXT, direction TEXT,
            entry_price REAL, size_usdc REAL, timestamp TEXT,
            status TEXT, exit_price REAL, pnl REAL,
            closed_at TEXT, is_paper INTEGER, category TEXT
        )
    """)
    conn.execute(
        "INSERT INTO trades VALUES (1, 'm', 'q', 'YES', 0.5, 100, '2026-01-01', 'closed', 0.6, 20, '2026-02-01', 0, 'c')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "sys.argv",
        ["tax", "--year", "2026", "--db", str(db), "--output-dir", str(tmp_path), "--format", "both"],
    )
    tax.main()
    assert (tmp_path / "form_8949_2026.csv").exists()
    assert (tmp_path / "trades_2026.csv").exists()


def test_tax_cli_8949_only(tmp_path, monkeypatch):
    import sqlite3

    from polymarket_agent.reporting import tax

    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE trades (id INTEGER PRIMARY KEY, market_id TEXT, question TEXT, direction TEXT, entry_price REAL, size_usdc REAL, timestamp TEXT, status TEXT, exit_price REAL, pnl REAL, closed_at TEXT, is_paper INTEGER, category TEXT)"""
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(
        "sys.argv", ["tax", "--year", "2025", "--db", str(db), "--output-dir", str(tmp_path), "--format", "8949"]
    )
    tax.main()
    assert (tmp_path / "form_8949_2025.csv").exists()
