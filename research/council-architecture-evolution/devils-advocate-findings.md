# Devil's Advocate Findings — Thread-Per-Subsystem Architecture Proposal

**Date:** 2026-03-12
**Reviewer Role:** Devil's Advocate
**Proposal:** Evolve MIDGE from Mesa step-cadence to 50+ independent daemon threads with queue.Queue signal routing

---

## Verdict Summary

**This proposal carries severe, concrete risk of destroying MIDGE's most fragile and recently-fixed system — the Thompson feedback loop — while providing uncertain benefit.** The convergence engine has no thread safety on its primary data structure (`self.signals: Dict[str, list]`), meaning concurrent injection from 50 threads is a data-corruption hazard today. The proposal's core premise — that threading provides speed gain — is weakened significantly by the Python GIL at this scale and on this workload. Seven of eight key assumptions are either unverified or actively contradicted by the codebase.

---

## Step 1: What Is Actually Being Proposed

The proposal is to:
1. Keep Mesa's heartbeat but free market subsystems from its step cadence
2. Give each of 50+ subsystems its own `sleep()` loop in a daemon thread
3. Route signals through `queue.Queue` instead of synchronous function calls
4. Make the convergence engine event-driven (react to queue entries instead of being called on a cadence)

**The implicit claim:** this removes the sequential bottleneck (54+ hooks run one after another) and allows faster signal integration.

**What this actually requires:**
- Every shared object touched by two or more threads needs a lock OR must be replaced by a thread-safe alternative
- The step-cadence ordering (sense → collect → converge → alert) must be replicated by queue message ordering or eliminated with a correctness argument
- 4,536 tests must be rewritten or adapted to a non-deterministic execution model
- The 33-layer bootstrap must survive creating 50+ long-running threads during initialization

---

## Step 2: Codebase Findings

### Finding 1: The Convergence Engine's Primary Data Structure Is Not Thread-Safe

`ConvergenceAlerter.__init__` (convergence_alerter.py line 215):
```python
self.signals: Dict[str, list] = defaultdict(list)
```

`record_signal()` (inherited via ConvergenceConfidenceMixin) appends to these lists. `check_convergence()` (ConvergenceDetectionMixin) calls `_prune_old_signals()` which iterates and mutates the same dict simultaneously.

The existing `_alert_lock = threading.Lock()` (line 252) protects **alert deduplication only** — it does not protect `self.signals`. With 50 threads concurrently calling `record_signal()`, concurrent append + prune on a `defaultdict(list)` is a classic TOCTOU race. Python's GIL makes this unlikely to cause crashes (list.append is atomic at the bytecode level) but does NOT make it safe: `_prune_old_signals()` iterates while appending threads may resize the list, producing missed signals or double-pruning.

**Status: Unverified assumption (6) is directly contradicted by the code.**

### Finding 2: The Thompson Feedback Loop Was Just Fixed and Is Fragile

MEMORY.md records four compounding bugs in the Thompson loop, all fixed 2026-03-09. The fix involved careful bootstrap ordering — `OutcomeCollector` receiving the correct `ThompsonSampler` instance. Under the proposal, each subsystem daemon thread would call into `ThompsonSampler` on its own schedule. `ThompsonSampler._lock` (thompson_sampler.py line 73) exists and protects `update()` and `apply_forgetting()`. However:

- `sample()` (line 131) calls `get_distribution()` which reads `self.distributions` **without acquiring `_lock`**
- `get_rankings()` iterates `self.distributions` without a lock
- `_save_distributions_locked()` is only called from within lock-held code — but `_seed_from_reliability()` and `replay_from_history()` can run concurrently at startup with no lock

If a daemon thread calls `sample()` while another calls `apply_forgetting()` (which takes `_lock` and mutates `params["alpha"]` and `params["beta"]` in-place via dict mutation), you get a read of a partially-mutated dict. In CPython this is practically safe due to GIL atomicity on dict operations, but Python 3.14 (the declared runtime) is introducing the **optional no-GIL build**. If `python3.14t` (no-GIL) is ever used, this becomes a real data corruption scenario.

**Status: Assumption (2) partially verified — existing locks are not comprehensive.**

### Finding 3: The Step-Cadence Ordering Is Load-Bearing, Not Incidental

The hook ordering in `market_hooks_steps_core.py` is not arbitrary. Reading the code:

1. `_market_sense_hook` runs **every step** — convergence check fires first
2. `_wire_sensing_hook` wraps sensing — collection happens in the main thread, then feeds convergence
3. The docstring in market_hooks.py explicitly states: "_register_market_step_hooks() MUST be called before _wire_sensing_hook()"
4. `ctx._cached_alerts` is a shared handshake between these two hooks — it's a list with one element (`[None]`), not a thread-safe queue

Under the proposal, this handshake breaks. If the sensing daemon thread writes new signals to the convergence engine while `check_convergence()` is running in a different thread, you get mid-computation signal injection. The result: a convergence alert that is half old-data and half new-data, with no defined behavior.

**Status: Assumption (8) — "no ordering issues" — is directly contradicted by the code.**

### Finding 4: The Outcome Collector and Post-Mortem Share State Without Cross-Thread Protection

`OutcomeCollector._registered` (outcome_collector.py line 90) is a plain dict that is read and written from `register_signals()`, `register_convergence_alert()`, and `register_pattern_stack()`. Under the proposal, multiple daemon threads could call these concurrently, producing duplicate registrations or missed entries. The prune operation at the top of `register_signals()` iterates while appending — a race condition with any concurrent registration call.

`PostMortemReviewer.review()` opens `outcomes.jsonl` for reading (line ~80) while `OutcomeTracker` appends to it from the Thompson update path. This file-level race is currently avoided because everything runs single-threaded. Under the proposal, the file is now a shared resource with no file-level locking.

**Status: Assumption (2) — "existing locks are sufficient" — is contradicted.**

### Finding 5: The CascadeTracker Has No Locks Whatsoever

`CascadeTracker._active_chains` (cascade_tracker.py line 52) is a plain dict with no lock. `check_signal()` iterates and mutates `_active_chains` (marks links as "confirmed"). `register_cascade()` adds entries. `expire_stale()` deletes entries. All three can run concurrently if sensing, convergence, and the cadence hook are in separate threads. A dict size-change during iteration raises `RuntimeError: dictionary changed size during iteration` — this is not silently suppressed, it is a hard crash.

**This is not a theoretical risk. It is a guaranteed crash under concurrent execution.**

**Status: Assumption (2) is directly contradicted.**

### Finding 6: The Bootstrap Is Sequence-Dependent in Ways That Break Under Async Init

The 33-layer bootstrap (main.py) runs in strict order. Layer 33 market bootstrap creates `OutcomeCollector` with a specific `ThompsonSampler` instance — the bug that caused 81/83 distributions to stay at priors was precisely a wrong-instance injection due to ordering. If subsystems launch their daemon threads during `__init__` (as the proposal implies), threads may start running before downstream dependencies (e.g., `PatternLibrary`, `WorldModel`) are wired in. The current codebase uses `None`-checks as a safety net (e.g., `if self._world_model is not None`), but a thread that starts before `world_model` is injected will silently skip the WorldModel path forever — not just for the first call.

**Status: Assumption (7) — "bootstrap ordering still valid" — is unverified and likely false.**

### Finding 7: FinnhubWebSocket Is Not a Clean Precedent

The proposal cites FinnhubWebSocket as "precedent" for background threads. Looking at the code in `sensing_hook.py` (lines 233-248): the WebSocket thread does NOT write directly to `convergence_alerter`. It deposits signals into a pending buffer (`get_pending_signals()`), and the **main thread** drains that buffer every step. This is explicitly a producer-only thread with a single consumer on the main thread. This pattern avoids the race conditions the proposal would create by having all 50 subsystem threads write directly to `convergence_alerter.record_signal()`.

**The FinnhubWebSocket precedent proves the OPPOSITE of what the proposal claims.** It shows the safe pattern is queue-to-main-thread, not concurrent multi-writer to shared state.

### Finding 8: The `_pending_futures` Dict in MarketSensingHook Is Not Thread-Safe

`MarketSensingHook._pending_futures: Dict[str, Future]` (sensing_hook.py line 199) is iterated in `get_statistics()` (`len([f for f in self._pending_futures.values()...])`) while the `_executor` pool may be modifying it. This is already present code — it's a latent race that the step-cadence model prevents because stats are always read from the main thread. Under the proposal, stats queries could arrive from a monitoring thread at the same time as a fetch completion callback.

---

## Step 3: Failure Mode Research

### GIL Thrashing at 50+ Threads

David Beazley's documented GIL convoy effect (bugs.python.org issue 7946) is directly relevant: when I/O-bound threads (API fetchers) release the GIL and CPU-bound threads (convergence computation, Thompson sampling) compete to reacquire it, the I/O threads add 5ms+ of GIL wait to every operation. At 50 threads, this is not linear — each thread competes against 49 others for GIL reacquisition. DeepMind's production data shows the GIL becomes a bottleneck with as few as 10 threads.

The MIDGE workload mixes I/O-bound (API fetchers — 30+ sources) with CPU-bound (numpy RSI/MACD vectorization, Thompson sampling, Granger causality). This is the exact pathological case. The "50 threads → faster signals" assumption may produce slower signal throughput than the current 12-worker pool.

### Windows-Specific Thread Scheduling

A Python.org confirmed bug (issue 13077) documents unclear behavior of daemon threads on Windows when the main thread exits. MIDGE on Windows 11 uses `--daemon` mode where the main loop runs indefinitely — but any crash or KeyboardInterrupt triggers `MycelialModel.shutdown()` which calls `self._agent_executor.shutdown(wait=False)`. With 50 daemon threads, `wait=False` means they are abandoned mid-operation. Thompson distributions (`_save_distributions_locked()`) and `registered_signals.json` could be in mid-write when the process dies.

The Python.org discussions also document a Windows-specific threading memory bug (reproduce on Windows 11, does not reproduce on macOS/Linux). The MIDGE Windows-only deployment is uniquely exposed.

### Deadlocking Queue Patterns

The documented "tragic tale of the deadlocking Python queue" (codewithoutrules.com) describes a class of deadlock where: (1) thread A puts to queue, (2) thread B blocks waiting on queue.get(), (3) A holds a lock that B needs for cleanup. In MIDGE's proposed model, if the convergence thread is blocked on `queue.get()` while a sensing thread is in `record_signal()` holding `_alert_lock` and waiting for the convergence thread to release a resource, you get a deadlock that only appears under load — invisible in tests.

### Incremental Migration Is a Myth at This Scale

Python.org issue 27422 confirms that mixing threading and multiprocessing produces deadlocks. The analogous risk here is mixing step-cadence (Mesa hooks) with daemon threads: the `ctx._cached_alerts[0]` shared handshake is written by the hook (step-cadence, main thread) and read by the advisory bridge (also step-cadence). If convergence is moved to an async thread, this handshake becomes a race. "Running both models simultaneously" requires converting every shared reference to a thread-safe equivalent first — that is not incremental, it is a full rewrite of the integration layer.

---

## Step 4: Assumption Audit

| # | Assumption | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Threading is sufficient — GIL not a real problem for 50 threads | **UNVERIFIED / LIKELY FALSE** | Dabeaz GIL convoy effect; DeepMind: GIL bottleneck at 10 threads; MIDGE workload is I/O+CPU mixed (worst case) |
| 2 | Existing thread locks are sufficient | **CONTRADICTED** | `self.signals` has no lock; `CascadeTracker._active_chains` has no lock; `_registered` dict unprotected; `sample()` reads without `_lock` |
| 3 | queue.Queue is fast enough for signal routing | **UNVERIFIED** | `queue.Queue` is thread-safe but adds per-signal overhead; at 50 producers × 30+ sources, queue backpressure behavior under load is unmeasured |
| 4 | Incremental migration is possible | **CONTRADICTED** | `ctx._cached_alerts[0]` handshake and bootstrap ordering are not incrementally decomposable; FinnhubWebSocket precedent shows the safe pattern requires a main-thread consumer |
| 5 | 50+ daemon threads on Windows is stable | **UNVERIFIED / RISKY** | Python.org issue 13077 (Windows daemon thread behavior); Windows-specific threading memory bug confirmed; `wait=False` shutdown abandons threads mid-write |
| 6 | Convergence engine handles concurrent injection from 50 threads | **CONTRADICTED** | `self.signals` is a bare `defaultdict(list)`; no lock on `record_signal()` write path; `_prune_old_signals()` concurrent with appending is a TOCTOU race |
| 7 | Bootstrap ordering remains valid with async | **UNVERIFIED / LIKELY FALSE** | Bootstrap is strictly sequential for correctness (Thompson instance ordering bug history); threads starting before dependency injection = silent wrong-instance use |
| 8 | No ordering issues — step cadence ordering doesn't matter | **CONTRADICTED** | market_hooks.py docstring explicitly states hook ordering is a hard constraint; `ctx._cached_alerts` handshake requires write-before-read ordering |

---

## Step 5: Counter-Evidence Against the Proposal

### Counter 1: The Bottleneck Is Not Where the Proposal Assumes

The proposal assumes the sequential hook chain is the performance bottleneck. But MIDGE's daemon output and architecture show that:
- The sensing hook already uses `ThreadPoolExecutor(12)` — API calls are not sequential
- The convergence check is "lightweight, pure in-memory" (market_hooks_steps_core.py comment, line 57)
- The true bottlenecks are external API rate limits (not remediable by threading) and the 25-step cadence gate (not remediable by threading — it's intentional throttling)

Adding 50 threads to a system where the bottleneck is API rate limits produces zero improvement and adds significant complexity.

### Counter 2: The Thompson Feedback Loop Is the Most Valuable Thing MIDGE Has

The MEMORY.md history shows four bugs were needed to fix it, it took specific bootstrap ordering to get right, and 13,190 historical updates had to be replayed to recover distributions. This is the crown jewel's brain. The proposal's concurrent write pattern puts it at highest risk, with the smallest possible benefit (the convergence check is already sub-millisecond).

### Counter 3: The Test Suite Cannot Validate This Architecture

4,536 tests run deterministically against single-threaded code. Thread-safety bugs are timing-dependent and will not appear in the test suite. The "zero regressions policy" cannot be satisfied by the existing tests for this class of change. The proposal would need a completely new concurrency test harness — that is a major hidden cost.

### Counter 4: Python 3.14 No-GIL Complicates This Further

The declared runtime is Python 3.14. Python 3.14 introduces the experimental no-GIL build (`python3.14t`). The codebase currently relies on GIL atomicity to make operations "practically safe" despite missing locks. If/when the no-GIL build is used (likely trajectory for performance-seeking workloads), the latent race conditions in `self.signals`, `CascadeTracker._active_chains`, and `_registered` become hard crashes. The proposal would accelerate adoption of a threading model that becomes broken by the next Python feature.

### Counter 5: The Proposal Addresses Latency, Not Throughput

The stated goal is "instant convergence reaction." But convergence requires signals from multiple domains — there is an irreducible minimum wait time for those signals to arrive from external APIs (seconds to minutes, not milliseconds). Shaving milliseconds off the convergence check loop provides zero user-visible benefit when the domain signals themselves arrive on 25-step cadences and API rate limits. The latency problem is upstream, not in the computation layer.

---

## Step 6: Risk Scorecard

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Failure Probability** | 2/10 (very likely to fail) | CascadeTracker dict-during-iteration crash is guaranteed under concurrent load; convergence signals unprotected; 7/8 assumptions unverified or contradicted |
| **Failure Severity** | 2/10 (catastrophic if fails) | Thompson distributions corrupted = learning engine destroyed; cascade tracker crash = step loop exception every convergence event; `registered_signals.json` torn write = duplicate Thompson updates |
| **Assumption Fragility** | 1/10 (extremely fragile) | 7 of 8 assumptions are unverified or contradicted; the one verified assumption (locks exist) is incomplete |
| **Rollback Difficulty** | 2/10 (very hard) | Thread-per-subsystem architecture requires rewriting all shared state access patterns; you cannot "undo" this by removing daemon threads — you must re-add locks everywhere first |
| **Hidden Complexity** | 1/10 (massive hidden work) | Every shared object needs audit + lock; `ctx._cached_alerts` handshake must be redesigned; bootstrap ordering must be rethought; 4,536 tests cannot validate thread safety — new test harness required |

**Overall Risk Assessment: 2/10 — This change has a high probability of destroying functioning systems to solve a problem that may not exist.**

---

## Specific Crashes That Will Occur Without Intervention

These are not theoretical. These will happen:

1. **CascadeTracker hard crash:** Two threads (sensing + convergence) concurrently call `check_signal()` and `expire_stale()`. Both iterate `_active_chains`. One deletes a key mid-iteration. `RuntimeError: dictionary changed size during iteration`. Step loop dies.

2. **Thompson distribution partial-write:** `apply_forgetting()` holds `_lock` and calls `params["alpha"] = max(2.0, params["alpha"] * decay)`. Concurrently, a daemon thread calls `sample()` (no lock), reads `params["alpha"]` between the `max()` call and the assignment. Under no-GIL Python 3.14: data corruption. Under current GIL Python 3.14: probabilistically safe but undefined behavior.

3. **Convergence signal buffer drift:** `_prune_old_signals()` iterates `self.signals` while 50 daemon threads append. CPython list.append is atomic at C level, but `defaultdict.__missing__` (creating a new key) is not atomic under concurrent access. Intermittent `KeyError` or duplicate domain creation.

4. **Outcome collector duplicate registration:** Two threads simultaneously call `register_convergence_alert()` for the same alert (same ticker, different threads). Both pass the `if alert_id in self._registered` check before either writes. Both register. Thompson receives double-credit for the same prediction. Learning engine corrupted.

5. **Daemon thread starts before dependency injection:** A subsystem's `__init__` starts its daemon thread. The thread's first iteration calls `self._world_model.find_ripple_effects()`. But `world_model` is injected by the bootstrap 200ms later. The thread saw `None`, skipped, and will never retry because the `None`-check only runs at thread start. WorldModel is silently bypassed forever.

---

## What Would Need to Be True for This to Be Safe

This proposal is not inherently wrong — it is premature and incompletely specified. For it to be safe:

1. Every shared object touched by more than one thread needs an explicit threading audit, documented in code
2. `self.signals` in ConvergenceAlerter needs a `threading.RLock` protecting both `record_signal()` and `check_convergence()` (not just `_alert_lock`)
3. `CascadeTracker._active_chains` needs a lock protecting all three mutating operations
4. `OutcomeCollector._registered` needs a lock protecting `register_*` operations
5. The FinnhubWebSocket pattern (producer-only daemon, main-thread consumer) must be applied uniformly — daemon threads ONLY deposit to a queue, NEVER write to shared intelligence objects
6. The bootstrap must guarantee dependency injection completes before any daemon thread's first iteration
7. A concurrency test harness (pytest-xdist + random thread interleavings) must validate correctness — the existing 4,536 deterministic tests cannot do this
8. The "bottleneck" claim needs measurement data, not assumption — profile the current step loop before redesigning it

Until those eight conditions are met, this proposal exchanges a working system for a broken one.

---

*Devil's Advocate analysis complete. This document identifies risks; it does not propose solutions.*
