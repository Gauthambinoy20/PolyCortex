"""Bounded asyncio queue with drop-on-full backpressure."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BoundedSignalQueue:
    """Bounded queue for scanner → decision engine handoff.

    When full and non-blocking, the newest signal is dropped and the counter
    incremented.  This keeps the decision engine from falling behind the
    scanner during bursts.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0
        self._accepted = 0

    async def put(self, signal: Any, block: bool = False) -> bool:
        """Enqueue *signal*.  Returns True if accepted, False if dropped."""
        if block:
            await self._queue.put(signal)
            self._accepted += 1
            return True
        try:
            self._queue.put_nowait(signal)
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "Signal queue full, dropping stale signal. Total dropped: %d",
                self._dropped,
            )
            return False
        self._accepted += 1
        return True

    async def get(self) -> Any:
        return await self._queue.get()

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def accepted_count(self) -> int:
        return self._accepted

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def empty(self) -> bool:
        return self._queue.empty()
