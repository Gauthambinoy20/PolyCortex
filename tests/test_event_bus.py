"""Tests for async event bus."""

import pytest

from polymarket_agent.events.bus import EventBus
from polymarket_agent.types import Event, EventType


@pytest.fixture
def bus():
    return EventBus()


class TestEventBus:
    async def test_subscribe_and_publish(self, bus):
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.ORDER_PLACED, handler)
        event = Event(event_type=EventType.ORDER_PLACED, payload={"id": "1"})
        await bus.publish(event)
        await bus.process()
        assert len(received) == 1
        assert received[0].payload["id"] == "1"

    async def test_multiple_subscribers(self, bus):
        results = []

        async def h1(e):
            results.append("h1")

        async def h2(e):
            results.append("h2")

        bus.subscribe(EventType.TRADE_CLOSED, h1)
        bus.subscribe(EventType.TRADE_CLOSED, h2)
        await bus.publish(Event(event_type=EventType.TRADE_CLOSED, payload={}))
        await bus.process()
        assert set(results) == {"h1", "h2"}

    async def test_no_cross_topic_delivery(self, bus):
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe(EventType.ORDER_PLACED, handler)
        await bus.publish(Event(event_type=EventType.TRADE_CLOSED, payload={}))
        await bus.process()
        assert len(received) == 0

    async def test_unsubscribe(self, bus):
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe(EventType.ORDER_PLACED, handler)
        bus.unsubscribe(EventType.ORDER_PLACED, handler)
        await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}))
        await bus.process()
        assert len(received) == 0

    async def test_handler_error_isolation(self, bus):
        results = []

        async def bad_handler(e):
            raise ValueError("boom")

        async def good_handler(e):
            results.append("ok")

        bus.subscribe(EventType.ORDER_PLACED, bad_handler)
        bus.subscribe(EventType.ORDER_PLACED, good_handler)
        await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}))
        await bus.process()
        assert results == ["ok"]

    async def test_wildcard_subscriber(self, bus):
        received = []

        async def handler(e):
            received.append(e.event_type)

        bus.subscribe_all(handler)
        await bus.publish(Event(event_type=EventType.ORDER_PLACED, payload={}))
        await bus.publish(Event(event_type=EventType.HEARTBEAT, payload={}))
        await bus.process()
        assert EventType.ORDER_PLACED in received
        assert EventType.HEARTBEAT in received

    async def test_stats(self, bus):
        async def noop(e):
            pass

        bus.subscribe(EventType.HEARTBEAT, noop)
        await bus.publish(Event(event_type=EventType.HEARTBEAT, payload={}))
        await bus.process()
        s = bus.stats
        assert s["events_processed"] == 1
        assert s["subscriber_count"] == 1

    async def test_publish_multiple_then_process(self, bus):
        count = 0

        async def handler(e):
            nonlocal count
            count += 1

        bus.subscribe(EventType.HEARTBEAT, handler)
        for _ in range(5):
            await bus.publish(Event(event_type=EventType.HEARTBEAT, payload={}))
        processed = await bus.process()
        assert processed == 5
        assert count == 5
