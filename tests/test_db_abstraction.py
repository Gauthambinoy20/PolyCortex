"""Tests for infra.db abstraction."""

from __future__ import annotations

import pytest

from polymarket_agent.infra import db


def test_default_sqlite_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert db.get_database_url() == db.DEFAULT_SQLITE_URL
    assert db.is_sqlite()
    assert not db.is_postgres()


def test_postgres_detection(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@h/d")
    assert db.is_postgres()
    assert not db.is_sqlite()


def test_custom_url_direct():
    assert db.is_postgres("postgresql://x")
    assert db.is_postgres("postgres://x")
    assert db.is_sqlite("sqlite:///tmp/x.db")


def test_create_engine_sqlite(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")  # optional "postgres" extra
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/x.db")
    db.reset_engine_cache()
    engine = db.create_engine()
    assert engine is not None
    # Cached
    engine2 = db.create_engine()
    assert engine is engine2
    db.reset_engine_cache()


def test_session_scope_rollback(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")  # optional "postgres" extra
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path}/x.db"
    monkeypatch.setenv("DATABASE_URL", url)
    db.reset_engine_cache()
    with db.session_scope() as session:
        session.execute(text("CREATE TABLE t (id INTEGER)"))
        session.execute(text("INSERT INTO t VALUES (1)"))

    with db.session_scope() as session:
        result = session.execute(text("SELECT COUNT(*) FROM t")).scalar()
    assert result == 1
    db.reset_engine_cache()


def test_session_scope_exception_rolls_back(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")  # optional "postgres" extra
    from sqlalchemy import text

    url = f"sqlite:///{tmp_path}/x.db"
    monkeypatch.setenv("DATABASE_URL", url)
    db.reset_engine_cache()
    with db.session_scope() as session:
        session.execute(text("CREATE TABLE t (id INTEGER)"))

    with pytest.raises(RuntimeError), db.session_scope() as session:
        session.execute(text("INSERT INTO t VALUES (1)"))
        raise RuntimeError("abort")

    with db.session_scope() as session:
        count = session.execute(text("SELECT COUNT(*) FROM t")).scalar()
    assert count == 0
    db.reset_engine_cache()
