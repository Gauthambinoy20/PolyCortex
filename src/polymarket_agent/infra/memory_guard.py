"""Memory-tracking helper wrapping :mod:`tracemalloc`."""

from __future__ import annotations

import logging
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryGuard:
    """Lightweight wrapper around :mod:`tracemalloc`.

    Usage::

        with MemoryGuard("scan-cycle") as guard:
            run_cycle()
        print(guard.peak_mb)
    """

    def __init__(self, label: str = "unnamed", threshold_mb: float | None = None) -> None:
        self.label = label
        self.threshold_mb = threshold_mb
        self.current_mb = 0.0
        self.peak_mb = 0.0
        self._was_tracing = False

    def __enter__(self) -> MemoryGuard:
        self._was_tracing = tracemalloc.is_tracing()
        if not self._was_tracing:
            tracemalloc.start()
        tracemalloc.clear_traces()
        return self

    def __exit__(self, *exc: object) -> None:
        current, peak = tracemalloc.get_traced_memory()
        self.current_mb = current / (1024 * 1024)
        self.peak_mb = peak / (1024 * 1024)
        if not self._was_tracing:
            tracemalloc.stop()
        if self.threshold_mb is not None and self.peak_mb > self.threshold_mb:
            logger.warning(
                "MemoryGuard[%s]: peak %.1fMB exceeded threshold %.1fMB",
                self.label,
                self.peak_mb,
                self.threshold_mb,
            )
        else:
            logger.info(
                "MemoryGuard[%s]: current=%.1fMB peak=%.1fMB",
                self.label,
                self.current_mb,
                self.peak_mb,
            )


@contextmanager
def track_memory(label: str = "block", threshold_mb: float | None = None) -> Iterator[MemoryGuard]:
    """Context-manager form for quick blocks."""
    guard = MemoryGuard(label, threshold_mb)
    with guard:
        yield guard
