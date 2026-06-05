# Performance

## Methodology

Profiling is done with:

* `cProfile` for CPU hotspots
* `tracemalloc` for peak memory
* `py-spy` (optional) for live sampling

A representative load test lives at `scripts/loadtest.py`:

```bash
python scripts/loadtest.py
```

The script simulates 500 markets and a steady decision loop for 30 seconds,
asserting peak RSS stays under 200 MB.

## Key hotspots addressed

| Area                         | Before    | After     |
|------------------------------|-----------|-----------|
| Cross-market correlation     | O(N²) reads | O(N) via `precompute_cross_signals` |
| Gamma API calls              | uncached  | 5-min TTL cache |
| Order book fetch             | serial    | async parallelized |
| Learner JSON writes          | every trade | only on threshold change |

## Async HTTP

All external HTTP uses `aiohttp` with shared `ClientSession` instances per
client.  Sessions are reused across calls; do not create per-call sessions.

## Memory

* The TCN model is loaded once at startup; inference reuses the same tensors.
* DataFrames are sliced with `.iloc[-N:]` rather than `.tail()` to avoid
  unnecessary copies.
* `MemoryGuard` (see `infra/memory_guard.py`) wraps hot loops to log peak
  memory without persistent overhead.

## CI

`scripts/loadtest.py` runs in CI on every PR; failure if peak > 200 MB.
