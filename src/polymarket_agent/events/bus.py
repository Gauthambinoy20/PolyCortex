"""Async publish/subscribe event bus for internal component communication."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from polymarket_agent.types import Event, EventType

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Async event bus with topic-based publish/subscribe.

    Components subscribe to specific EventTypes and receive Events
    asynchronously. Supports multiple subscribers per topic, wildcard
    subscriptions (receive ALL events), and handler error isolation
    (one failing handler doesn't break others).

    Usage:
        bus = EventBus()

        async def on_order(event: Event):
            print(f"Order event: {event.payload}")

        bus.subscribe(EventType.ORDER_PLACED, on_order)
        await bus.publish(Event(
            event_type=EventType.ORDER_PLACED,
            payload={"order_id": "abc123", "size": 100.0}
        ))

        # Process all pending events
        await bus.process()
    """

    def __init__(self, max_queue_size: int = 10000) -> None:
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._wildcard_subscribers: list[EventHandler] = []
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._process_task: asyncio.Task | None = None
        self._event_count = 0
        self._error_count = 0

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to a specific event type."""
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", handler.__name__, event_type.value)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe a handler to ALL event types (wildcard)."""
        self._wildcard_subscribers.append(handler)
        logger.debug("Subscribed %s to all events", handler.__name__)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a handler from a specific event type."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus. Non-blocking (queued)."""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "Event bus queue full (%d), dropping event %s",
                self._queue.maxsize,
                event.event_type.value,
            )
            self._error_count += 1

    async def process(self) -> int:
        """Process all pending events in the queue. Returns count processed."""
        processed = 0
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            await self._dispatch(event)
            processed += 1
        return processed

    async def start(self) -> None:
        """Start continuous background event processing."""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._run_loop())
        logger.info("Event bus started")

    async def stop(self) -> None:
        """Stop background processing and drain remaining events."""
        self._running = False
        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task
        # Drain remaining
        await self.process()
        logger.info(
            "Event bus stopped. Total events: %d, errors: %d",
            self._event_count,
            self._error_count,
        )

    async def _run_loop(self) -> None:
        """Continuous processing loop."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in event bus loop")
                self._error_count += 1

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all matching subscribers."""
        self._event_count += 1
        handlers = list(self._subscribers.get(event.event_type, []))
        handlers.extend(self._wildcard_subscribers)

        if not handlers:
            logger.debug("No handlers for %s", event.event_type.value)
            return

        # Run all handlers concurrently but isolate errors
        results = await asyncio.gather(
            *(self._safe_call(h, event) for h in handlers),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                self._error_count += 1

    async def _safe_call(self, handler: EventHandler, event: Event) -> None:
        """Call handler with error isolation."""
        try:
            await handler(event)
        except Exception:
            logger.exception(
                "Handler %s failed for event %s",
                handler.__name__,
                event.event_type.value,
            )
            raise

    @property
    def stats(self) -> dict:
        """Return bus statistics."""
        return {
            "events_processed": self._event_count,
            "errors": self._error_count,
            "queue_size": self._queue.qsize(),
            "subscriber_count": sum(len(h) for h in self._subscribers.values()) + len(self._wildcard_subscribers),
            "running": self._running,
        }
