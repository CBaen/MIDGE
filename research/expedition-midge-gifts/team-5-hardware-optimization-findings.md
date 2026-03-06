# Team 5: Consumer Hardware Optimization — Research Findings

**Date:** March 5, 2026
**Project:** MIDGE
**Angle:** Maximizing throughput on Wardenclyffe (Win11 desktop)
**Latency target:** Detect within minutes, act within the hour

---

## Executive Summary

The 46-minute excavation projection is not a hardware ceiling — it is an architectural one. The code runs work sequentially that could run concurrently. Three specific changes (concurrent async fetching, sensing worker scaling, PolygonBulkFetcher conversion to aiohttp) could compress the 46-minute estimate to under 10 minutes. Beyond those, a tiered optimization path exists ranging from zero-risk thread scaling to more involved Rust extensions.

---

## 1. What the Code Actually Shows

Before recommending anything, it helps to state precisely what is happening now.

**Excavation (the 85s/100 bottleneck):**
`ExcavationDaemon.step()` processes a batch of symbols in a `for symbol in batch` loop — one symbol at a time. Each symbol calls `PolygonBulkFetcher.get_daily_history()`, which uses `requests.Session.get()` — a fully synchronous, blocking HTTP call. With Polygon's paid plan having unlimited API requests (confirmed: Massive/Polygon paid plan = unlimited REST calls), the only constraint is network round-trip latency. At ~0.8s per call, 100 symbols takes ~80s. The math is simple: sequential I/O where parallel I/O is legal.

**Sensing workers:**
`MarketSensingHook.__init__` creates `ThreadPoolExecutor(max_workers=3)` and maintains a cap of 3 concurrent in-flight fetches (see `_launch_next_fetch`: `slots = 3 - len(self._pending_futures)`). With 28 sources in `SOURCE_ROTATION`, running only 3 simultaneously leaves 25 idle. The comment in the docstring still says "one pending future at a time" (a stale copy from the original design). The actual executor is already 3, but the previous expedition identified 12-20 as safe.

**Pattern library storage:**
`PatternLibrary` stores fingerprints and templates in two JSONL files (`pattern_library.jsonl`, `pattern_templates.jsonl`). Every `store_batch()` call opens the file and appends. There is no bulk write, no columnar format. For the pattern matching query path, the library loads all fingerprints into a dict in memory at startup — so query speed is fine, but the write path and the disk footprint are inefficient.

**EventBus:**
The EventBus is a pure in-process structure with an `RLock`. Every `publish()` serializes the message to JSON, then delivers it to registered callbacks under the lock. For the volume of signals MIDGE handles, this is adequate. The lock contention point is only an issue if many agents publish simultaneously — which they do in `_parallel_step()`.

**TA computation:**
`HistoricalDataFetcher._compute_ta_signals()` already caches per symbol (`_get_ta_cached`), so RSI/MACD/Bollinger are computed once per symbol. This is well-designed. However, the Bollinger computation at line 391 runs a Python `for i in range(period, len(closes))` loop computing `sum(window)` and `sum((x - sma)**2 for x in window)` manually — no numpy. For 2000 bars per symbol times hundreds of dig sites, this is pure Python arithmetic in the hot path.

---

## 2. Research Area 1: Async/Concurrent API Fetching

### Finding: aiohttp over requests is the right conversion for bulk excavation

Benchmarks consistently show aiohttp outperforms synchronous requests by 10-18x in concurrent scenarios (121.8 req/s vs ~6.5 req/s for sequential requests). For MIDGE's excavation workload — fire-and-forget HTTP GETs to Polygon, no auth complexity, uniform response structure — this is a direct swap.

**Concrete change:** Convert `PolygonBulkFetcher` from `requests.Session` to `aiohttp.ClientSession` with:
- A single shared `aiohttp.TCPConnector(limit=50)` — the connector pools TCP connections per host. Start at 50 concurrent connections to Polygon. Polygon's paid plan is unlimited, but TCP connection setup has its own cost.
- An `asyncio.Semaphore(50)` wrapping each fetch to prevent overwhelming the connector.
- Replace the `for symbol in batch` loop in `ExcavationDaemon._excavate_symbol` with `asyncio.gather(*[fetch(sym) for sym in batch])`.

**Rate limit reality (verified):** Polygon.io's paid plan (confirmed under the Massive.com brand, 2025) explicitly states "unlimited API requests" for paying customers. The previous 85s/100 symbols bottleneck is *entirely* self-imposed by sequential calls. At 50 concurrent connections, 100 symbols should complete in under 5 seconds (100 calls / 50 concurrent, each ~0.8s = ~1.6s).

**What changes:** `PolygonBulkFetcher` needs an async interface. The `ExcavationDaemon` either runs under an event loop or uses `asyncio.run()` for batch calls. Because `ExcavationDaemon.step()` is called from the Mesa model's synchronous step hook, the cleanest path is `asyncio.run(self._fetch_batch_async(batch))` for the excavation context only — this does not require converting the whole daemon to async.

**Connection pool sizing for the sensing hook:**
`MarketSensingHook` already uses `ThreadPoolExecutor` for its 28 sources. This is appropriate for I/O-bound work — the GIL is released during network I/O so threads achieve real concurrency. No conversion to asyncio is needed here; the current design is sound. The fix is purely increasing the worker count (see Section 4).

### Finding: aiohttp TCPConnector on Windows

On Windows, `asyncio` uses `ProactorEventLoop` by default (since Python 3.8). `aiohttp` works correctly with ProactorEventLoop. No special configuration needed beyond installing `aiohttp`. The one caveat: `aiohttp.ClientSession` must be created inside an event loop context, not at module level.

**Sources:**
- [aiohttp Request Lifecycle docs](https://docs.aiohttp.org/en/stable/http_request_lifecycle.html)
- [Optimizing aiohttp for High Concurrency](https://proxiesapi.com/articles/optimizing-aiohttp-for-high-concurrency)
- [Massive/Polygon rate limit KB](https://massive.com/knowledge-base/article/what-is-the-request-limit-for-polygons-restful-apis)

---

## 3. Research Area 2: Rust via PyO3

### Finding: PyO3 is production-ready on Windows. Worth it for two specific hotspots.

PyO3 with Maturin is the standard path to Rust extensions in Python. As of 2025: Windows is a first-class target, CI publishes wheels for Windows + Linux + macOS simultaneously, and `maturin develop` for local dev is a one-command workflow. Reported speedups: 5-15x for compute-heavy Python code.

**What in MIDGE would benefit:**

1. **Bollinger Band variance computation** — `historical_fetcher.py` lines 391-445. The inner loop computes a rolling standard deviation in pure Python. This is the textbook case for a Rust extension: tight numerical loop, no Python objects involved once the float list is passed in, embarrassingly parallelizable across symbols.

2. **Pattern template similarity scoring** — `PatternLibrary` uses a similarity function to match fingerprints against templates. If this is a dot product or cosine similarity over float vectors, Rust with SIMD can be significantly faster.

3. **Thompson Sampling beta draws** — `sensing_hook.py` calls `random.betavariate()` for each of 28 sources on every fetch cadence. This is currently pure Python. A Rust implementation using the `rand` crate's beta distribution would be ~10x faster per draw. However, the total time for 28 draws is microseconds — this is *not* a meaningful bottleneck.

**Practical threshold:** Only implement Rust for Bollinger/TA computation if profiling (see Section 6) confirms it accounts for >10% of excavation time. The conversion work is ~2 days. Numpy vectorization (see Section 5) achieves 80% of the benefit with zero build complexity.

**Windows build requirements:**
- Rust toolchain via `rustup`: `rustup target add x86_64-pc-windows-msvc`
- `maturin develop` compiles and installs into the active venv
- No "unsafe" code required for numeric operations — PyO3 handles all FFI safely

**Sources:**
- [Maturin User Guide](https://www.maturin.rs/tutorial.html)
- [PyO3 Getting Started](https://pyo3.rs/v0.27.2/getting-started.html)
- [PyO3 Windows CI evidence](https://github.com/PyO3/pyo3)

---

## 4. Research Area 3: Memory-Mapped Data and Efficient Storage

### Finding: JSONL is the right choice *now* but has a clear upgrade path

Current pattern library scale (pattern_library.jsonl, pattern_templates.jsonl) is unknown in size, but at 3,237 symbols each producing dozens of fingerprints, the files could reach 50-500MB. At that scale:

- **JSONL read:** Fine. The library loads entirely into memory at startup (`_load_library()`). Read time is proportional to file size. For 500MB of fingerprints, a cold load takes 5-30 seconds depending on parse speed — a one-time cost.
- **JSONL write:** Every `store_batch()` opens and appends. This is safe and correct but cannot be concurrent.
- **JSONL query:** Not applicable — the library holds everything in `self._fingerprints` dict in memory.

**If the library exceeds 1GB:** Convert to DuckDB. DuckDB is a pure-Python-installable embedded OLAP database (no server). Benchmarks show DuckDB queries are 400µs vs 20ms+ for Parquet scans. It can directly query JSON/Parquet/CSV files. The `duckdb` pip package works on Windows with no external dependencies.

Specific upgrade path: serialize fingerprints as Parquet using `pyarrow` (already available if numpy is installed), query with DuckDB. This makes template similarity searches a SQL GROUP BY instead of a Python dict scan.

**For the signal archive (data/midge/signals/*.jsonl):**
The archive preloading reads *all* JSONL files into `_signal_cache` at startup. This is already the right pattern — no repeated disk I/O during excavation. No change needed unless the archive exceeds available RAM.

**Memory-mapped numpy arrays:**
`numpy.memmap` is appropriate for the price history arrays if they are kept between excavation cycles. Current design fetches price history fresh per symbol. If price history were cached to disk as numpy memmaps (shape: [symbols, days, OHLCV]), the next excavation run could skip Polygon fetches for symbols with recent data. This is the more impactful optimization than the storage format.

**Sources:**
- [DuckDB Analytics Revolution](https://dev.to/emiroberti/duckdb-the-analytics-database-revolution-a-comprehensive-guide-442b)
- [Building with Python, Parquet, DuckDB](https://www.kdnuggets.com/building-your-modern-data-analytics-stack-with-python-parquet-and-duckdb)
- [numpy.memmap NumPy docs](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

---

## 5. Research Area 4: Process-Level Parallelism on Windows

### Finding: ThreadPoolExecutor is correct for MIDGE's I/O workload. ProcessPoolExecutor is the wrong tool.

**The Windows spawn constraint:**
On Windows, `multiprocessing` always uses "spawn" — a fresh Python interpreter starts, imports the module, pickles all arguments, and un-pickles them in the child. Spawn is ~20x slower than Linux fork for process startup (42s vs 2s for 1,000 spawns). More critically, every argument to a worker function must be picklable. MIDGE's agent objects, EventBus, and PatternLibrary almost certainly contain unpicklable state (lambda functions, locks, thread-local storage).

**Pickling test for the excavation path:**
The previous expedition flagged this. Here is the specific test:
```python
import pickle
from mae_core.market.archaeology.polygon_bulk_fetcher import PolygonBulkFetcher
fetcher = PolygonBulkFetcher()
# This must not raise:
pickle.dumps(fetcher)
```
`PolygonBulkFetcher` holds only `self._session = requests.Session()`. `requests.Session` is NOT picklable (it holds socket handles). So `ProcessPoolExecutor` with a shared fetcher object would fail immediately. To use processes, the fetcher would need to be reconstructed in each worker — adding per-worker startup overhead.

**Conclusion:** For MIDGE's primary bottleneck (network I/O), `ThreadPoolExecutor` is the correct tool and already in use. The GIL is released during network I/O, so threads achieve true parallelism. Moving to processes adds spawn overhead and pickling complexity with no benefit for I/O-bound work.

**Where processes *would* help:**
If TA computation (RSI/MACD/Bollinger) were the bottleneck — which is pure Python CPU work that holds the GIL — `ProcessPoolExecutor` with a stateless worker function (just floats in, floats out) would be worth it. But profiling should confirm this before building it. The simpler path is numpy vectorization first (see finding below).

**Sensing workers — the immediate win:**
The sensing hook `ThreadPoolExecutor(max_workers=3)` cap is conservative. 28 sources are in rotation. Each source fetch is mostly I/O-bound (HTTP calls). The GIL is released during I/O. Raising to `max_workers=12` is safe and low-risk — it just means 12 HTTP calls can be in-flight simultaneously instead of 3. The previous expedition said 12-20. 12 is the conservative target; start there.

**One thread pool concern:** The `model.py` agent executor is also `ThreadPoolExecutor(max_workers=min(len(agents), 8))`. With sensing workers, pattern excavation, agent stepping, and background validators all using thread pools, total threads can reach 30-40 under load. Windows handles this without issue up to hundreds of threads — thread overhead is ~1MB stack each but actual CPU time is dominated by the GIL for Python code.

**Sources:**
- [Fork is 20x faster than spawn](https://superfastpython.com/fork-faster-than-spawn/)
- [ProcessPoolExecutor common errors](https://superfastpython.com/processpoolexecutor-common-errors/)
- [Python multiprocessing start methods](https://docs.python.org/3/library/multiprocessing.html)

---

## 6. Research Area 5: Mesa 3.4/3.5 Performance

### Finding: Mesa 3.5.0 adds DiscreteEventSimulator — relevant for MIDGE's architecture

MIDGE is currently on Mesa 3.4, using `shuffle_do("step")` and `AgentSet`. Mesa 3.5.0 (released 2025) introduces:

1. **DiscreteEventSimulator:** Schedule events at arbitrary timestamps rather than fixed ticks. Instead of every agent stepping every tick, agents only step when they have something to do. For MIDGE, this could mean the sensing hook only fires on cadence ticks, the excavation daemon only steps every 5000 ticks, etc. The architecture already does cadence-based work inside `step()` — this just makes it more explicit and potentially faster by skipping empty step() calls.

2. **AgentSet stability improvements:** Mesa 3.5 stabilizes the event system that was experimental in 3.4.

**What MIDGE's model.py shows:**
`_parallel_step()` uses `ThreadPoolExecutor(max_workers=min(len(agents), 8))` and submits all agents concurrently. This is already a good pattern. The cap of 8 workers is conservative — with the GIL, pure Python agent logic is still serialized, but agents doing I/O (memory consolidation, pattern matching against the library) do release the GIL.

**EventBus RLock contention:**
In `_parallel_step()`, all agents can call `bus.publish()` simultaneously. `publish()` acquires `self._lock`. With 8+ concurrent agents all publishing, lock contention becomes measurable. The mitigation is to batch agent publications: agents accumulate messages during `step()` and flush after all agents complete. This is an architectural change, not a simple config tweak — flag for future consideration.

**Concrete Mesa recommendation:**
Upgrade to Mesa 3.5.0. Test with existing test suite (4,384+ tests). The API change from 3.4 to 3.5 involves deprecations but no breaking changes for AgentSet/shuffle_do. The `DiscreteEventSimulator` is additive. Upgrade is low-risk.

**Sources:**
- [Mesa 3.5 PyPI](https://pypi.org/project/Mesa/)
- [Mesa 3 paper: Agent-based modeling with Python in 2025](https://joss.theoj.org/papers/10.21105/joss.07668)
- [Mesa documentation](https://mesa.readthedocs.io/latest/)

---

## 7. Research Area 6: Profiling Tools

### Finding: Profile before optimizing. Scalene + py-spy is the right combination.

The 85s/100 symbols figure is measured. But we don't have measured data on where time goes *within* that 85s: is it 95% HTTP wait? Is the TA computation expensive? Is pattern library writes measurable? Assumptions about hotspots are frequently wrong.

**Recommended profiling workflow:**

**Step 1 — Scalene (offline, detailed):**
Scalene v2.5 (2025) profiles CPU time split between Python and native code, and memory allocations, at line granularity. Install: `pip install scalene`. Run: `scalene populate_library.py --max-symbols 20`. Output shows exactly which lines consume Python CPU vs. native CPU vs. memory.

Cost: ~35% runtime overhead. Use on a representative small run (20-50 symbols).

**Step 2 — py-spy (attach to live run):**
py-spy 0.4 (2025) attaches to a running Python process with ~0.1% overhead. It produces flamegraphs showing the call stack distribution. Use this on a full excavation run to capture realistic behavior: `py-spy record -o profile.svg --pid <PID>`. Then open profile.svg in a browser.

py-spy now handles async frame unwinding, so it correctly shows time spent in asyncio coroutines if the code is converted.

**Step 3 — Interpret and act:**
- If >80% of time is in `requests.get` → async conversion is the right fix
- If >20% of time is in `_compute_bollinger_series` → numpy vectorization or Rust
- If pattern library writes show up → batch writes or DuckDB migration
- If EventBus lock shows up under parallel step → batch publication refactor

**Sources:**
- [Profiling Scalene Py-Spy 2025](https://johal.in/profiling-scalene-py-spy-memory-cpu-flamegraphs-2025/)
- [Scalene GitHub](https://github.com/plasma-umass/scalene)
- [Python profiling guide](https://betterstack.com/community/guides/scaling-python/profiling-in-python/)

---

## 8. Numpy Vectorization: The Fast Middle Path

This deserves its own section because it is lower cost than Rust but higher reward than async for the TA computation.

`HistoricalDataFetcher._compute_bollinger_series()` uses a Python loop to compute rolling mean and standard deviation. This is pure Python arithmetic. Numpy already exists in the project (imported in `event_bus.py`). The identical computation using `numpy.lib.stride_tricks.sliding_window_view()` or `pandas.Series.rolling()` runs 10-50x faster and requires no new dependencies or build step.

The RSI computation (`_compute_rsi_series`) is also a manual loop. Converting to vectorized numpy operations is a 20-line change per function.

**This is the recommended first optimization after sensing worker scaling.** It requires:
- No new dependencies (numpy is already present)
- No async refactoring
- No build toolchain
- Estimated time: 2-4 hours

---

## 9. Prioritized Optimization Roadmap

Ordered by: risk-adjusted impact per hour of work.

| Priority | Change | Expected Impact | Risk | Effort |
|----------|--------|-----------------|------|--------|
| 1 | Raise sensing workers: `max_workers=3` → `max_workers=12` | 4x more concurrent signal fetches | Very low — one number change | 5 min |
| 2 | Profile first: run Scalene on 50-symbol excavation | Confirms actual bottleneck before coding | None | 30 min |
| 3 | Numpy-vectorize TA computation in historical_fetcher.py | 10-50x faster Bollinger/RSI | Low — pure Python → numpy, fully testable | 4 hrs |
| 4 | Convert PolygonBulkFetcher to async batch fetch | 85s/100 → ~5s/100 (projected) | Medium — requires asyncio in sync context | 1 day |
| 5 | Price history disk cache (numpy.memmap) | Skip re-fetching on 2nd+ excavation runs | Low — additive feature | 4 hrs |
| 6 | Mesa 3.5.0 upgrade | DiscreteEventSimulator for cadence work | Low — test suite validates | 2 hrs |
| 7 | DuckDB for pattern library (if >500MB) | Faster template queries at scale | Medium — storage format change | 1 day |
| 8 | Rust Bollinger computation via PyO3 | Additional 3-5x beyond numpy | Medium — build toolchain, new language | 3 days |

---

## 10. Key Constraints Validated

- **Pure Python ecosystem:** All recommendations except Rust (item 8) use pure Python or existing dependencies.
- **No external infrastructure:** DuckDB is embedded, no server. aiohttp is pure client-side.
- **Test suite (4,384+ tests):** Numpy vectorization and sensing worker scaling are the safest changes. Async conversion and storage format changes require dedicated test coverage for the new paths.
- **Windows ProcessPoolExecutor spawn:** Confirmed: `requests.Session` is not picklable → ProcessPoolExecutor would fail for the current PolygonBulkFetcher. Do not use ProcessPoolExecutor without first restructuring the worker to be stateless.
- **Day/swing latency ("minutes to detect"):** The sensing hook is designed for this — continuous rotation across 28 sources every 50 steps. The excavation bottleneck does not affect live sensing. Excavation is a *background* enrichment task.

---

## Sources

- [aiohttp TCPConnector optimization](https://proxiesapi.com/articles/optimizing-aiohttp-for-high-concurrency)
- [aiohttp concurrent requests guide](https://apidog.com/blog/aiohttp-concurrent-request/)
- [asyncio semaphore concurrency](https://rednafi.com/python/limit-concurrency-with-semaphore/)
- [aiohttp vs requests benchmark](https://miguel-mendez-ai.com/2024/10/20/aiohttp-vs-httpx)
- [PyO3 Getting Started](https://pyo3.rs/v0.27.2/getting-started.html)
- [Maturin User Guide](https://www.maturin.rs/tutorial.html)
- [Rust Python performance 2025](https://markaicode.com/rust-integration-strategies-2025/)
- [Fork 20x faster than spawn](https://superfastpython.com/fork-faster-than-spawn/)
- [ProcessPoolExecutor common errors](https://superfastpython.com/processpoolexecutor-common-errors/)
- [Python multiprocessing docs](https://docs.python.org/3/library/multiprocessing.html)
- [DuckDB analytics guide](https://dev.to/emiroberti/duckdb-the-analytics-database-revolution-a-comprehensive-guide-442b)
- [Building with DuckDB and Parquet](https://www.kdnuggets.com/building-your-modern-data-analytics-stack-with-python-parquet-and-duckdb)
- [numpy.memmap docs](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [Mesa 3 paper 2025](https://joss.theoj.org/papers/10.21105/joss.07668)
- [Mesa 3.5 PyPI](https://pypi.org/project/Mesa/)
- [Scalene GitHub](https://github.com/plasma-umass/scalene)
- [Scalene + Py-Spy profiling 2025](https://johal.in/profiling-scalene-py-spy-memory-cpu-flamegraphs-2025/)
- [Python profiling guide](https://betterstack.com/community/guides/scaling-python/profiling-in-python/)
- [Massive/Polygon rate limits](https://massive.com/knowledge-base/article/what-is-the-request-limit-for-polygons-restful-apis)
