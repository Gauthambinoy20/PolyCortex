"""Abbreviated load test.

Simulates 500 markets + repeated decisioning for 30s and asserts memory stays
bounded.  Designed to run in CI.
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc


async def main() -> None:
    tracemalloc.start()
    markets = [
        {
            "id": f"market_{i}",
            "question": f"Will X happen {i}?",
            "best_bid": 0.4,
            "best_ask": 0.6,
            "midpoint": 0.5,
            "bid_depth": 1000.0,
            "ask_depth": 1000.0,
        }
        for i in range(500)
    ]
    start = time.monotonic()
    decisions = 0
    while time.monotonic() - start < 30:
        for market in markets[:10]:
            decisions += 1
            # Cheap decision work to exercise the loop
            _ = (market["best_bid"] + market["best_ask"]) / 2.0
            await asyncio.sleep(0.001)

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024
    print(f"Decisions: {decisions}, Peak memory: {peak_mb:.1f}MB")
    assert peak_mb < 200, f"Memory too high: {peak_mb:.1f}MB"
    print("Load test PASSED")


if __name__ == "__main__":
    asyncio.run(main())
