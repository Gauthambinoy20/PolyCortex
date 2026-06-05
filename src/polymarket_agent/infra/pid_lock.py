"""PID-file lock to prevent multiple concurrent agent instances on same DB."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import TracebackType
from typing import IO

logger = logging.getLogger(__name__)

try:
    import fcntl  # type: ignore[import-not-found]

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]
    _HAS_FCNTL = False


class PidLock:
    """Idempotent startup guard using ``fcntl`` advisory lock."""

    def __init__(self, lock_file: str = "data/.polymarket.lock") -> None:
        self._path = Path(lock_file)
        self._file: IO[str] | None = None
        self._acquired = False

    def acquire(self) -> bool:
        """Return True if lock acquired, False if another instance holds it."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", encoding="utf-8")  # noqa: SIM115  (lifetime is explicit)
        if not _HAS_FCNTL:
            # Best-effort on platforms without fcntl
            self._file.write(str(os.getpid()))
            self._file.flush()
            self._acquired = True
            return True
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            self._file = None
            return False
        self._file.write(str(os.getpid()))
        self._file.flush()
        self._acquired = True
        return True

    def release(self) -> None:
        import contextlib

        if self._file is not None:
            try:
                if _HAS_FCNTL:
                    fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            except OSError:
                logger.debug("flock LOCK_UN failed", exc_info=True)
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
        if self._acquired:
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()
            self._acquired = False

    def __enter__(self) -> PidLock:
        if not self.acquire():
            raise RuntimeError(f"Another instance is already running (lock={self._path})")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
