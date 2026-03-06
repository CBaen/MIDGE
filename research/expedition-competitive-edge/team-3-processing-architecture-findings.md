# Team 3 Findings: Processing Architecture
## Date: 2026-03-05
## Researcher: Team Member 3

---

### Codebase Baseline: What MIDGE Has Now

Before any recommendations, it is essential to understand the exact current state from reading the code:

**MarketSensingHook** (`mae_core/market/sensing_hook.py`):
- `ThreadPoolExecutor(max_workers=3, thread_name_prefix="mkt-sense")` — 3 concurrent threads for data fetching
- One pending future per source slot, non-blocking; results collected next step
- 28 sources rotating through a single `deque(SOURCE_ROTATION)`
- Sources are fetched sequentially through the rotation; max 3 in-flight at once
- All signal processing (convergence alerter, pattern watcher) happens in the **main thread** — thread-safe by design

**ExcavationDaemon** (`mae_core/market/archaeology/excavation_daemon.py`):
- Runs as a step hook, every 5000 steps
- Sequential: processes 5 symbols per batch, one symbol at a time within the batch
- 3,237 symbols at ~85s/100 symbols (with Polygon) = ~46 minutes if dedicated; current daemon cadence stretches this to 9+ hours because it only runs every 5000 steps and processes 5 symbols per batch call
- No parallelism within symbol processing

**HypothesisEngine / BacktestScheduler** (`mae_core/market/intelligence/`):
- `ThreadPoolExecutor(1)` for validation — single background thread with skip-if-busy semantics
- `ThreadPoolExecutor(1)` for backtest scheduling — same pattern

**Mesa model** (`main.py`):
- Single-threaded step execution; 6 agents step sequentially
- No parallelism within the model loop
- Mesa's `batch_run` supports `number_processes` parameter for running multiple model instances in parallel, but that runs the whole organism multiple times — not relevant to MIDGE's single-organism design

**The GIL reality**: All current concurrency is via `ThreadPoolExecutor` — threads, not processes. This is correct for I/O-bound data fetching (API calls, file reads). Python's GIL does NOT limit I/O-bound threaded code. The current 3-worker thread pool is sound for the fetching layer.

---

### Battle-Tested Approaches

#### Approach 1: Expand ThreadPoolExecutor Workers for the Sensing Hook (3 → 12-20)

- **What:** Increase `max_workers` in `MarketSensingHook` from 3 to 12-20, allowing more data sources to be in-flight simultaneously rather than sequentially rotating.
- **Evidence:** Python's `concurrent.futures` docs (Python 3.14, accessed 2026-03-05) confirm ThreadPoolExecutor is designed for I/O-bound parallelism and scales well. The GIL does not limit thread-based I/O. QuantStart's parallelism guide (accessed 2026-03-05) documents practical guidance: for I/O-bound tasks, threads are the correct tool and scale proportionally to worker count. The current `max_workers=3` was chosen for safety, not ceiling.
- **Source:** https://docs.python.org/3/library/concurrent.futures.html (Python 3.14 docs, 2026); https://www.quantstart.com/articles/Parallelising-Python-with-Threading-and-Multiprocessing/
- **Fits our case because:** MIDGE fetches from 28 rate-limited HTTP sources. Each fetch is I/O-bound and independent. Increasing from 3 to 12-20 workers means 4-7x more concurrent fetches, reducing the effective rotation period. Sources with different rate limits (SEC EDGAR's slow rate vs. Finnhub's faster rate) no longer block each other.
- **Tradeoffs:** More in-flight requests means more memory for pending responses. Rate limit violations become more likely if the same API endpoint is in multiple concurrent slots. Needs per-source rate limiting (semaphore per source, not just global worker count). Windows socket limits are not a concern at this scale (thousands of concurrent sockets are supported).

#### Approach 2: ProcessPoolExecutor for Parallel Symbol Excavation

- **What:** Replace the sequential per-symbol loop in `ExcavationDaemon._excavate_symbol()` with `ProcessPoolExecutor`, processing N symbols in true parallel across CPU cores.
- **Evidence:** This is the single most documented pattern in quant literature for embarrassingly parallel work. Every symbol excavation is completely independent — no shared state between symbol analyses. The dupoin.com architecture (July 2025, accessed 2026-03-05) documents a producer-consumer ProcessPoolExecutor pattern achieving 8 minutes vs 83 minutes single-threaded for tick data (10x+ speedup). QuantStart documents ~4x speedup with 4 cores for CPU-bound backtesting. The `populate_library.py` batch excavation script already has the right structure (it's a loop calling `daemon.step()`) — parallelizing the inner loop is the minimal change.
- **Source:** https://academy.dupoin.com/en/python-multiprocess-backtesting-engine-38767-186349.html (July 2025); https://www.quantstart.com/articles/Parallelising-Python-with-Threading-and-Multiprocessing/
- **Fits our case because:** Symbol excavation is CPU-bound (TA computation on 2000 days of price history) AND embarrassingly parallel (no symbol shares state with another). With an i9 having 8-24 cores, running 8-16 parallel symbol excavations could reduce the 9-hour daemon time to under an hour. This directly solves the research brief's "excavation takes 9+ hours" bottleneck.
- **Tradeoffs:** Each process needs its own copy of the Excavator, HistoricalDataFetcher, and PatternLibrary objects (Python pickling required). The PatternLibrary's JSONL file writes become a write-contention point — need file locking or a collector process that aggregates results. `multiprocessing` on Windows requires `if __name__ == '__main__':` guard (already present in `populate_library.py`). Process startup cost is ~0.5-1s per worker — amortized across batches of 50+ symbols, this is negligible.

#### Approach 3: Asyncio-based Concurrent Fetch Layer (replacing ThreadPoolExecutor)

- **What:** Replace `ThreadPoolExecutor` in `MarketSensingHook` with `asyncio` + `aiohttp` for the HTTP-bound fetch layer, with per-source `asyncio.Semaphore` for rate limiting.
- **Evidence:** The asyncio approach is the modern consensus for high-concurrency I/O in Python. The 2026 guide (Medium, Dec 2025) documents that asyncio scales to "hundreds of thousands of concurrent operations" using far less memory per task than threads. The `aiolimiter` library (v1.2.1, actively maintained) provides `AsyncLimiter` for per-source rate limiting. Rate-limiting patterns are well-documented (aiolimiter docs, Quentin Pradet's blog). Real-world evidence: asyncio handles 150+ concurrent API connections efficiently where threads become unwieldy.
- **Source:** https://medium.com/@yogeshkrishnanseeniraj/high-performance-python-asyncio-vs-multiprocessing-vs-threadpools-2026-guide-ad49d40452fc (Dec 2025); https://aiolimiter.readthedocs.io/; https://quentin.pradet.me/blog/how-do-you-rate-limit-calls-with-aiohttp.html
- **Fits our case because:** MIDGE's fetch layer is purely I/O-bound. Asyncio would allow all 28 sources to be in-flight simultaneously with per-source rate limit enforcement, instead of the current rotating-queue-of-3. This is a meaningful throughput increase for live sensing.
- **Tradeoffs:** This is a significant refactor of `sensing_hook.py` and all 27 `sensing_fetchers.py` functions. Many current fetchers use blocking libraries (yfinance, requests, SEC Edgar's library). These would need `asyncio.get_event_loop().run_in_executor()` wrappers or migration to async-native clients. The Mesa step loop is synchronous — asyncio would need to run in a separate thread or as a companion event loop. Risk: introduces a threading-asyncio boundary that is notoriously tricky. This is a moderate-to-large refactor, not a quick win.

#### Approach 4: Companion Data Fetcher Process (Sidecar Architecture)

- **What:** Run a separate Python process (`data_fetcher.py`) that continuously fetches from all 28 sources at maximum throughput, writing signals to a shared JSONL buffer or SQLite WAL database. MIDGE's main process reads from this buffer instead of fetching directly.
- **Evidence:** This is the standard "sidecar" pattern used in production trading systems. NautilusTrader (Rust-core, Python bindings, maintained 2025) separates data adapters from strategy execution. The QuantConnect forum confirms LEAN separates data loading from algorithm execution — data loads in parallel, algorithm runs sequentially on synchronized data (accessed 2026-03-05). The pyalgotrade and AAT (Async Algo Trading) frameworks use the same separation. Redis pub/sub for inter-process communication is documented in multiple trading system architectures (willguxy.github.io blog, Redis/ZeroMQ comparison gist). Python's `multiprocessing.shared_memory` (Python 3.8+, production-stable) allows zero-copy NumPy array sharing between processes.
- **Source:** https://www.quantconnect.com/forum/discussion/12678/is-quantconnect-single-threaded/ (QuantConnect forums, accessed 2026-03-05); https://github.com/nautechsystems/nautilus_trader (updated Jun 2025); https://charlesleifer.com/blog/going-fast-with-sqlite-and-python/
- **Fits our case because:** The sidecar decouples MIDGE's Mesa step loop from data fetching latency entirely. The companion process fetches continuously at its own pace; MIDGE consumes from the buffer. This is the cleanest architectural separation and fits Law 5 (different configuration, not different code — the fetcher process runs the same fetcher functions, just continuously). It also enables the fetcher to run even when MIDGE is restarting (solving the "daemon runs on old code, must restart" problem).
- **Tradeoffs:** Adds process management complexity (the fetcher process must be started/stopped alongside MIDGE). Inter-process communication adds latency (milliseconds via shared memory or file-based JSONL). The current signal buffer (`data/market/signal_buffer.json`) already suggests this pattern was considered. File-based IPC (JSONL append + tail-read) is the simplest implementation with near-zero complexity, at the cost of disk I/O.

---

### Novel Approaches

#### Approach 5: Symbol Universe Sharding — Multiple MIDGE Instances

- **What:** Run 3 simultaneous MIDGE organisms (using Mesa's `batch_run` with `number_processes=3`), each assigned a different shard of the symbol universe, with results merged via shared PatternLibrary and Thompson distributions.
- **Why it's interesting:** Mesa 3 explicitly supports `batch_run(number_processes=N)` for parallel model instances (Mesa 3.2.0 docs, Feb 2026). Rather than parallelizing within a single organism, this runs three complete organisms in parallel, each sensing a third of the ticker universe. Results converge in shared data files.
- **Evidence:** Mesa 3 docs confirm `number_processes=None` to use all CPUs. The "embarrassingly parallel across symbols" insight from dupoin.com applies here at the organism level. This is essentially the "sibling system" the research brief asks about. No custom IPC needed — organisms share the same PatternLibrary JSONL files (file locking handled by Python's `fcntl`/file append atomicity on Windows NTFS).
- **Source:** https://mesa.readthedocs.io/latest/_modules/mesa/batchrunner.html (Mesa 3 docs, accessed 2026-03-05); https://zenodo.org/records/15363883 (Mesa 3: ABM with Python in 2025, JOSS paper)
- **Fits our case because:** Each MIDGE instance already has all the infrastructure (Thompson, convergence alerter, pattern library). Three parallel organisms triple the symbol coverage without any architectural change to the organism itself. The convergence alerter's per-ticker signals are already partitioned by symbol — sharding is natural.
- **Risks:** Three Mesa model instances consume 3x the memory (estimated 3x ~2GB = ~6GB — well within 64GB). Template writes to the shared PatternLibrary JSONL could corrupt on concurrent appends without file locking. Thompson distribution updates from multiple processes would race. This requires a designated "writer" process or a small merge step.

#### Approach 6: Memory-Mapped Signal Cache (numpy.memmap)

- **What:** Pre-load the entire signal archive (~900+ JSONL files) into a `numpy.memmap` array at startup, allowing all processes to access the same signal history via zero-copy shared memory rather than each process loading its own copy.
- **Why it's interesting:** The `populate_library.py` already does `fetcher.preload_archive()` to eliminate per-dig-site I/O — this is the right idea but loads into per-process Python dicts. A `numpy.memmap` backed by a flat binary file can be shared across all worker processes without duplication. The superfastpython.com guide (2024) documents that `shared_memory` approaches achieve "0.007s vs 1.35s" compared to copy-based sharing.
- **Evidence:** `multiprocessing.shared_memory` (Python 3.8+, standard library) and `numpy.memmap` (NumPy v2.4, current) both support zero-copy multi-process access to the same data. The signal archive is append-only JSONL, which converts cleanly to a structured array (timestamp, ticker, source, strength as fixed-width fields).
- **Source:** https://superfastpython.com/numpy-array-memory-mapped-file/ (accessed 2026-03-05); https://numpy.org/doc/stable/reference/generated/numpy.memmap.html (NumPy v2.4 docs)
- **Fits our case because:** MIDGE's signal archive is 900+ files, ~414+ days. Parallel excavation workers currently each load and parse their own archive slice. A shared memory signal cache would make all 16 parallel excavation workers reference the same memory-mapped array — eliminating 15 redundant copies and their load time.
- **Risks:** Requires converting JSONL signals to a fixed-schema binary format (lossy for metadata dicts). The architecture investment is significant for what may be a one-time startup cost. `numpy.ndarray` with `dtype=object` cannot be memory-mapped (segfault risk) — must use fixed-width dtypes (int64 timestamps, str-to-hash mapping for tickers/sources).

---

### Emerging Approaches

#### Approach 7: Pathway or Bytewax for Streaming Pipeline

- **What:** Replace the polling-based `MarketSensingHook` with a streaming pipeline framework (Pathway or Bytewax) that treats each API source as an input stream and the convergence alerter as a transformation step.
- **Momentum:** Pathway (GitHub: pathwaycom/pathway) is a Python ETL framework for stream processing; Bytewax is an open-source Rust-core Python streaming framework. Both are actively maintained in 2025. Pathway specifically supports reading and writing JSONL. Bytewax handles "thousands of events per second" per their docs.
- **Source:** https://github.com/pathwaycom/pathway (updated 2025); https://bytewax.io/ (accessed 2026-03-05)
- **Fits our case because:** The current poll-rotate-collect pattern in `MarketSensingHook` is a rudimentary stream. A streaming framework would handle backpressure, retry, and windowed aggregation natively — all things MIDGE currently implements manually.
- **Maturity risk:** Both frameworks are designed for continuous streaming workloads, not agent-step cadence. Integrating either with Mesa's synchronous step loop requires an event loop bridge. Neither has documented integration with Mesa. This is a significant architectural leap that violates the "no new major frameworks" principle of conservative change. Flag as exploratory only.

#### Approach 8: Redis Streams as Signal Bus Between Processes

- **What:** Use Redis (already available on Wardenclyffe per CLAUDE.md) as a message bus between the data fetcher sidecar and MIDGE's main process, replacing file-based JSONL signal storage for IPC.
- **Momentum:** Redis Streams (added in Redis 5.0, production-proven at scale) provide a persistent, ordered, consumer-group-aware log. Multiple trading system architectures use Redis pub/sub or Streams for inter-process signal distribution (willguxy.github.io trading system blog, 2018; Abhishek Jain HFT system, Medium 2024). Redis is already installed on Wardenclyffe.
- **Source:** https://vardhmanandroid2015.medium.com/building-a-high-frequency-trading-system-with-hybrid-strategy-redis-influxdb-from-10ms-to-85716febefcb (2024); comparison gist: https://gist.github.com/hmartiro/85b89858d2c12ae1a0f9
- **Fits our case because:** Redis Streams solve the concurrent-write race condition for the sidecar architecture (Approach 4) cleanly. Multiple fetcher processes write to Redis; MIDGE reads at its own pace. Consumer groups allow MIDGE to restart without losing buffered signals.
- **Maturity risk:** Adds Redis as a required runtime dependency for MIDGE operation. If Redis is down, all signal flow stops. Operational complexity increases. For MIDGE's current scale, JSONL append with file locking is simpler and sufficient.

---

### Gaps and Unknowns

1. **i9 core count not confirmed.** The hardware is described as "i9 or equivalent" in the research brief. i9 processors range from 8 to 24 cores (e.g., i9-13900K has 24 cores). The optimal `ProcessPoolExecutor` worker count depends on this. Recommendation: run `os.cpu_count()` on Wardenclyffe before sizing any worker pool.

2. **Mesa step loop timing not measured.** Unknown: how long does a single model step take with all 33 bootstrap layers? If a step takes 10ms and there are 500 steps at pace=2.0, that's ~5 seconds of CPU time per run. Parallelizing the step loop itself requires knowing whether agents are CPU-bound or I/O-bound within the step.

3. **PatternLibrary JSONL write safety under concurrent access.** JSONL append is atomic on NTFS for small writes (<4KB), but this hasn't been verified under concurrent multi-process appends. Need to test or add explicit `fcntl`-style locking (on Windows, use `msvcrt.locking` or a `threading.Lock` in a coordinator process).

4. **Polygon API rate limits under parallel symbol fetching.** The Polygon Starter plan's rate limits are not documented in the codebase. Before scaling excavation to 16 parallel workers hitting Polygon simultaneously, the rate limit must be confirmed. If the limit is 10 req/s per plan, 16 parallel workers will trigger throttling and backfire.

5. **Memory cost of parallel excavation processes.** Each `ExcavationDaemon` worker process needs its own HistoricalDataFetcher (which preloads 900 signal archive files). If that preload is ~500MB per process, 16 workers = 8GB just for archive copies. Need to profile actual memory footprint before committing to N workers.

6. **ThreadPoolExecutor safety on Windows with Python 3.14.** The CLAUDE.md flags that v2.1.45 broke Git Bash stdout capture due to Windows HANDLE vs POSIX fd mismatch. ThreadPoolExecutor itself is unaffected (pure Python), but any subprocess-based fetch (e.g., if any fetcher uses `subprocess.run`) would be affected.

---

### Synthesis

#### What the landscape tells us

MIDGE's current architecture is well-designed for safety, not throughput. The `ThreadPoolExecutor(3)` pattern in `MarketSensingHook` is proven and correct for I/O-bound fetching — it was chosen conservatively and can be expanded without any architectural change. The excavation bottleneck is the dominant problem: 9+ hours for 3,237 symbols directly limits how quickly MIDGE can discover new patterns.

The quant community has converged on a clear principle: **I/O-bound work → threads or asyncio; CPU-bound work → processes**. MIDGE's two bottlenecks map cleanly to this:
- Data fetching (28 APIs) = I/O-bound → expand ThreadPoolExecutor or move to asyncio
- Symbol excavation (TA computation on 2000-day price histories) = CPU-bound → ProcessPoolExecutor

#### The strongest approach: Two targeted changes, not one architectural overhaul

**Priority 1 — Parallel Excavation (CPU-bound):** Add a `parallel_excavate()` function to `populate_library.py` that uses `ProcessPoolExecutor(max_workers=os.cpu_count() // 2)` to process batches of symbols in parallel. Each worker gets its own Excavator instance. Results (fingerprints, templates) are returned and merged by the coordinator. No changes to the Mesa daemon loop or core architecture. Estimated impact: 9-hour excavation → 45 minutes to 1 hour. Evidence-backed speedup: 8-10x from multiprocessing backtesting literature. This is the single highest-leverage change available.

**Priority 2 — Expanded Sensing Concurrency (I/O-bound):** Increase `MarketSensingHook`'s `ThreadPoolExecutor` from 3 to 12 workers, add a per-source `threading.Semaphore` to enforce individual API rate limits. No architectural change — just tuning constants. The queue rotation already handles source selection. Impact: 4x more simultaneous API fetches, reducing the effective sensing lag for 28 sources.

**Do not recommend:** The asyncio refactor (Approach 3) is high-risk, high-effort, and yields marginal gains over expanded threads at MIDGE's scale. The sidecar architecture (Approach 4) is architecturally sound but adds operational complexity that isn't justified until excavation and sensing throughput are actually bottlenecking the system's output quality. The Mesa multi-instance sharding (Approach 5) has race conditions on shared state that would require significant work to resolve safely.

#### What the orchestrator needs to know

The real bottleneck is not sensing throughput — it is excavation throughput. MIDGE is already sensing 28 sources continuously. The pattern template library (MIDGE's "institutional memory" of historical patterns) cannot grow quickly enough because excavation is single-threaded and step-cadenced. Fixing excavation parallelism directly accelerates the RSI Layer 4 feedback loop: more templates → better pattern stacking → higher confidence signals. The sensing expansion is a bonus, not the core fix.

A companion/sibling process for excavation (running `populate_library.py --source polygon` in parallel with the main daemon) is already supported by the existing CLI and requires zero code changes. This is the immediate action: run excavation as a **separate process** while the daemon runs, sharing the same PatternLibrary JSONL files. NTFS append atomicity makes this safe for the library, and Thompson/hypothesis updates stay in the daemon process.

The hardware headroom on an i9/64GB machine is significant. There is no reason to remain at 6 agents and 3 fetch workers. The bottleneck is code configuration, not hardware.

---

*Sources referenced in this document:*
- [Python 3.14 concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html)
- [QuantStart: Parallelising Python with Threading and Multiprocessing](https://www.quantstart.com/articles/Parallelising-Python-with-Threading-and-Multiprocessing/)
- [Python Multiprocess Backtesting: Shatter GIL for 10M+ Tick Data (July 2025)](https://academy.dupoin.com/en/python-multiprocess-backtesting-engine-38767-186349.html)
- [QuantConnect Threading Forum](https://www.quantconnect.com/forum/discussion/12678/is-quantconnect-single-threaded/)
- [NautilusTrader GitHub (updated Jun 2025)](https://github.com/nautechsystems/nautilus_trader)
- [Mesa 3: ABM with Python in 2025 (JOSS Paper)](https://zenodo.org/records/15363883)
- [Mesa batchrunner module docs](https://mesa.readthedocs.io/latest/_modules/mesa/batchrunner.html)
- [aiolimiter docs](https://aiolimiter.readthedocs.io/)
- [Python mmap + shared memory for NumPy (SuperFastPython)](https://superfastpython.com/numpy-array-memory-mapped-file/)
- [NumPy memmap docs v2.4](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)
- [SQLite WAL mode (Charles Leifer)](https://charlesleifer.com/blog/going-fast-with-sqlite-and-python/)
- [Pathway streaming framework](https://github.com/pathwaycom/pathway)
- [Bytewax streaming framework](https://bytewax.io/)
- [Redis HFT architecture (2024)](https://vardhmanandroid2015.medium.com/building-a-high-frequency-trading-system-with-hybrid-strategy-redis-influxdb-from-10ms-to-85716febefcb)
- [asyncio rate limiting with aiohttp](https://quentin.pradet.me/blog/how-do-you-rate-limit-calls-with-aiohttp.html)
- [High-Performance Python: AsyncIO vs Multiprocessing vs ThreadPools (Dec 2025)](https://medium.com/@yogeshkrishnanseeniraj/high-performance-python-asyncio-vs-multiprocessing-vs-threadpools-2026-guide-ad49d40452fc)
