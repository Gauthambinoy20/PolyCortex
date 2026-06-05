"""Database abstraction supporting SQLite and PostgreSQL via SQLAlchemy."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

DEFAULT_SQLITE_URL = "sqlite:///data/trades.db"


def get_database_url() -> str:
    """Return database URL from ``DATABASE_URL`` env var, defaulting to SQLite."""
    return os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL)


def is_postgres(url: str | None = None) -> bool:
    """Return True if *url* points at a PostgreSQL server."""
    target = url or get_database_url()
    return target.startswith(("postgresql://", "postgres://", "postgresql+"))


def is_sqlite(url: str | None = None) -> bool:
    """Return True if *url* points at a SQLite database."""
    target = url or get_database_url()
    return target.startswith("sqlite")


_engine_cache: dict[str, Engine] = {}


def create_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    """Create or reuse a SQLAlchemy engine for *url*."""
    try:
        from sqlalchemy import create_engine as sa_create_engine
    except ImportError as exc:  # pragma: no cover - sqlalchemy is a soft dep
        raise ImportError("sqlalchemy required: pip install sqlalchemy") from exc

    target = url or get_database_url()
    if target in _engine_cache:
        return _engine_cache[target]

    connect_args: dict = {}
    if is_sqlite(target):
        connect_args["check_same_thread"] = False

    engine = sa_create_engine(target, echo=echo, connect_args=connect_args, future=True)
    _engine_cache[target] = engine
    return engine


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Context-managed SQLAlchemy session."""
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(url)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Dispose of cached engines (used in tests)."""
    for engine in _engine_cache.values():
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            logger.debug("Engine dispose failed", exc_info=True)
    _engine_cache.clear()
