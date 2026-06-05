"""Graceful shutdown handler for the trading agent.

Handles SIGINT (Ctrl+C) and SIGTERM: cancels open orders, flushes logs,
saves state, then exits cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from collections.abc import Callable, Coroutine
from typing import Any, cast

logger = logging.getLogger(__name__)

ShutdownHook = Callable[[], Coroutine[Any, Any, None]]


class GracefulShutdown:
    """Manages graceful shutdown of the trading agent.

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register(cancel_all_orders)
        shutdown.register(save_state)
        shutdown.register(flush_logs)
        shutdown.install()  # hooks SIGINT/SIGTERM

        # ... run your agent ...
        # On Ctrl+C or kill, registered hooks run in order.
    """

    def __init__(self, timeout: float | None = None) -> None:
        self.timeout = timeout if timeout is not None else float(os.environ.get("SHUTDOWN_TIMEOUT_SEC", "60"))
        self._hooks: list[tuple[str, ShutdownHook]] = []
        self._shutting_down = False
        self._installed = False

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register(self, hook: ShutdownHook, name: str = "") -> None:
        """Register an async shutdown hook. Hooks run in registration order."""
        hook_name = name or getattr(hook, "__name__", repr(hook))
        self._hooks.append((hook_name, hook))
        logger.debug("Registered shutdown hook: %s", hook_name)

    def install(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Install signal handlers for SIGINT and SIGTERM."""
        if self._installed:
            return

        target_loop = loop or asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                target_loop.add_signal_handler(
                    sig,
                    cast(
                        "Callable[[], Any]",
                        lambda s=sig: asyncio.ensure_future(self._handle_signal(s)),
                    ),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: asyncio.ensure_future(self._handle_signal(s)))

        self._installed = True
        logger.info(
            "Graceful shutdown installed (%d hooks, %.0fs timeout)",
            len(self._hooks),
            self.timeout,
        )

    async def _handle_signal(self, sig: int) -> None:
        """Handle shutdown signal."""
        sig_name = signal.Signals(sig).name
        if self._shutting_down:
            logger.warning("Force shutdown requested (%s) — exiting immediately", sig_name)
            sys.exit(1)

        self._shutting_down = True
        logger.warning("Received %s — initiating graceful shutdown...", sig_name)

        await self.execute()
        sys.exit(0)

    async def execute(self) -> dict[str, str]:
        """Run all shutdown hooks in order with timeout.

        Returns:
            Dict mapping hook name to 'ok', 'error: ...', or 'timeout'.
        """
        self._shutting_down = True
        results: dict[str, str] = {}

        for name, hook in self._hooks:
            logger.info("Running shutdown hook: %s", name)
            try:
                await asyncio.wait_for(hook(), timeout=self.timeout / max(len(self._hooks), 1))
                results[name] = "ok"
                logger.info("Shutdown hook completed: %s ✓", name)
            except TimeoutError:
                results[name] = "timeout"
                logger.error("Shutdown hook timed out: %s ✗", name)
            except Exception as exc:
                results[name] = f"error: {exc}"
                logger.error("Shutdown hook failed: %s — %s", name, exc)

        succeeded = sum(1 for v in results.values() if v == "ok")
        logger.info(
            "Graceful shutdown complete: %d/%d hooks succeeded",
            succeeded,
            len(results),
        )
        return results
