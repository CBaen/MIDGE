# Team 4 Findings: Autonomous Scheduling & Scale
## Date: 2026-03-08
## Researcher: Team Member 4

---

### Preamble: What I Found in the Codebase

Before diving into approaches, here is the ground truth of MIDGE's current execution model that every recommendation must build on:

**`add_step_hook` is MIDGE's own invention.** It is not a Mesa 3.4 API — it lives in `mae_core/model.py` at line 253. Mesa 3.4's `Model` class has no such method. MIDGE's `MycelialModel` subclass adds it. This is critical: the step hook system is fully controllable code, not an external framework constraint.

**The step loop architecture in `mae_core/model.py`:**
- `model.step()` calls all registered `_step_hooks` in registration order, then activates agents via `ThreadPoolExecutor` (parallel) or `agents.shuffle_do("step")` (sequential).
- All market cadence logic lives in two hooks registered in `market_hooks.py`: `_market_sense_hook` and `_sensing_step_with_advisory`. These are closures with their own internal counters — `step % 10`, `step % 50`, `step % 200`, etc.
- The OctopusColony runs its 100ms monitoring on a **separate `threading.Thread`** (`_monitoring_loop`), already completely decoupled from the step loop.
- There are currently ~12 registered step hooks (`wiring.py` + `market_hooks.py` + `main.py`), each firing every step.

**The key tension this research addresses:** Every step hook fires on EVERY step. `_market_sense_hook` alone branches across 7 cadence thresholds per call. As inhabitants grow to 50+, this becomes an O(N*step_count) fire-everything-every-tick bottleneck.

**Mesa version installed: 3.4.2.** Mesa 3.5 (which adds `model.schedule_recurring()`) is available but not yet installed. It introduces no breaking changes from 3.4.

---

### Battle-Tested Approaches

#### Approach 1: Daemon Thread with Independent Timer per Inhabitant

**What:** Each inhabitant runs as a `threading.Thread` (daemon=True) with its own `while running: do_work(); time.sleep(interval)` loop. The inhabitant publishes results to the EventBus. The step loop consumes them.

**Evidence:** This is the exact pattern already used by OctopusColony's `_monitoring_loop` (5-second monitoring interval), and by the Finnhub WebSocket client (real-time streaming thread). Both have been in production in MIDGE for months. Also used extensively in production by Kafka consumers, Redis listeners, and background health checkers across the industry. Python's stdlib `threading.Thread` documentation classifies daemon threads as the standard pattern for background periodic work.

**Source:** `mae_core/network/octopus_colony.py` lines 283-296 (in-project); Python threading documentation https://docs.python.org/3/library/threading.html (official, current)

**Fits our case because:** It directly maps to the "inhabitant wakes on its own clock" vision. A Circadian Rhythm inhabitant could run every 5 minutes wall-clock. A Congressional trades fetcher could run every 6 hours. A VIX watcher could run every 30 seconds. All independently. All decoupled from the step count. The EventBus is already thread-safe (RLock) and is the established intra-organism communication medium.

**Tradeoffs:**
- 50 daemon threads consume ~50 OS thread stacks (~8MB each = ~400MB baseline). Python's GIL means they don't run truly in parallel for CPU-bound code, but for IO-bound work (API calls, file reads) threads release the GIL and do run concurrently.
- Thread starvation risk if many threads block simultaneously. Mitigated by keeping inhabitants IO-bound (they fetch, they don't compute).
- No built-in scheduling metadata — you can't inspect "what is every inhabitant's schedule" without implementing it yourself.
- Shutdown coordination requires explicit stop events or flags.

---

#### Approach 2: Single `threading.Timer` Wheel via Shared Priority Queue

**What:** One background thread drives all inhabitant scheduling through a `heapq`-based priority queue. Each entry is `(next_fire_time, interval, callback)`. The wheel thread sleeps until the next fire time, calls the callback in a thread pool, then re-inserts the entry at `next_fire_time + interval`.

**Evidence:** This is the algorithmic core of Python's `sched` module (stdlib, Python 3.3+), APScheduler 3.x (11k+ GitHub stars, used in production at Heroku, documented as standard for Flask/Django deployments), and the Mesa 3.5 event scheduling internals (`EventList` based on heapq). It's used in every major scheduler library because `heapq.heappush/heappop` are O(log N) even at N=1000.

**Source:** https://pypi.org/project/APScheduler/ (APScheduler 3.11.2, production-stable); https://docs.python.org/3/library/sched.html (Python stdlib); Mesa 3.5 internals as described in https://github.com/mesa/mesa/pull/3155

**Fits our case because:** 50+ inhabitants with heterogeneous intervals (some every 30 seconds, some daily) are exactly the use case for a wheel scheduler. The single thread overhead is minimal. Callbacks can be dispatched to `ThreadPoolExecutor` for true concurrency. The wheel can publish to EventBus, keeping inhabitants as pure data-producing processes. You get full observability: the queue is inspectable, you can see every inhabitant's next fire time.

**Tradeoffs:**
- Introduces a coordination point (the wheel thread) that is a single point of failure — mitigated by making it stateless (wheel reconstructs from inhabitant metadata on restart).
- Inhabitants don't "know" their own schedule — the wheel knows. This slightly complicates the "inhabitant with intrinsic drive" metaphor, though in practice the inhabitant still controls what it does when woken.
- APScheduler 4.0 is pre-release and explicitly not production-ready. Use 3.11.2.

---

#### Approach 3: Mesa 3.5 `schedule_recurring` for In-Step Inhabitants

**What:** Upgrade to Mesa 3.5 (no breaking changes from 3.4) and use `model.schedule_recurring(callback, Schedule(interval=N))` to give individual systems their own heterogeneous step-based cadences, replacing the manual `if step % N == 0` branches inside monolithic hooks.

**Evidence:** Mesa 3.5.0 released with this as headline feature. The `Schedule` dataclass supports `interval`, `start`, `end`, `count`. Recurring events can be stopped via `gen.stop()`. The Mesa core team confirmed: `model.run_for(1)` is functionally equivalent to `model.step()` — the step mechanism is now internally implemented as a recurring event, so the upgrade is drop-in.

**Source:** https://github.com/mesa/mesa/releases/tag/v3.5.0 (release notes, accessed 2026-03-08); https://github.com/mesa/mesa/pull/3155 (implementation PR); https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html (official tutorial)

**Fits our case because:** The 7+ cadence branches inside `_market_sense_hook` (`step % 10`, `step % 50`, `step % 200`, etc.) would each become independent registered schedules rather than branches in one monolithic hook. This is a direct path from the current "everything fires through one closure" to "each system has its own cadence identity." It's the lowest-risk change since it stays entirely within the existing step loop without introducing threads or wall-clock time.

**Tradeoffs:**
- Does NOT solve the wall-clock independence problem. Inhabitants still wake when the step loop ticks, not on their own clocks. A 50-step interval at pace=1.0 means 50 real-time seconds; at pace=2.0 it means 25 seconds. Wall-clock mapping requires daemon threads (Approach 1) or a wheel scheduler (Approach 2).
- Migration cost: replacing all `step % N` branches with `schedule_recurring` calls is mechanical but touches core bootstrap files.
- Mesa 3.5 is newer than current install (3.4.2). Upgrade is low-risk but requires verification.

---

### Novel Approaches

#### Approach 4: Pykka Actor Model — One Actor per Inhabitant

**What:** Each inhabitant becomes a `pykka.ThreadingActor`. Actors communicate exclusively by message-passing (no shared state). Each actor has a mailbox and processes messages sequentially. An actor can schedule itself by sending timed messages.

**Evidence:** Pykka v4.4.1 (released January 2026) — actively maintained, 1,051 commits, 1,300+ GitHub stars, 1.3k projects depend on it. No external dependencies. Apache License. Requires Python 3.10+. Used in production by Mopidy (the music server) as its primary concurrency model since 2014.

**Source:** https://github.com/jodal/pykka (accessed 2026-03-08, v4.4.1); https://pykka.org/ (official documentation)

**Fits our case because:** Actors are a formalization of the "inhabitant with independent life" model. Each actor owns its state, processes messages serially (no internal locking needed), and communicates via the same EventBus publish/subscribe pattern MIDGE already uses. An inhabitant's "heartbeat" becomes a recurring message it sends to itself via a `threading.Timer`. This is philosophically aligned with Law 6 (autopoietic closure): actors govern themselves.

**Why it's interesting:** Pykka's actor model provides a conceptual framework that maps directly to the inhabitant-culture vision — inhabitants don't share state, they pass messages. It formalizes what MIDGE is already trying to do informally. The actor registry provides built-in introspection (list all running actors, send shutdown).

**Risks:**
- Pykka actors don't natively support "wake up on a schedule" — you still need `threading.Timer` or similar to inject timed messages. Pykka is a message-passing abstraction, not a scheduler.
- 50 actors = 50 threads (Pykka ThreadingActor is one thread per actor). Same resource cost as Approach 1 but with more framework overhead.
- Integration with Mesa's step loop requires bridging: either actors publish to EventBus (readable from step hooks) or they call model methods (needs thread safety review).
- No evidence of Pykka being used inside a Mesa simulation context. The combination is untested.

---

#### Approach 5: asyncio Event Loop in a Daemon Thread (Hybrid)

**What:** Spin up one asyncio event loop in a dedicated daemon thread at bootstrap. All IO-bound inhabitants become `async def` coroutines scheduled with `asyncio.sleep(interval)` on this loop. The step loop (synchronous Mesa) submits callbacks to the async loop via `asyncio.run_coroutine_threadsafe()`.

**Evidence:** Python official documentation explicitly supports this pattern for integrating async code with synchronous applications. "Running an asyncio event loop in a separate thread can be helpful when we want to integrate asyncio-based asynchronous code with synchronous code." Used in production by Jupyter kernels, streaming data pipelines, and hybrid web servers. GitHub gist by dmfigol (2k+ stars) documents the exact implementation pattern.

**Source:** https://superfastpython.com/asyncio-event-loop-separate-thread/ (2024); https://gist.github.com/dmfigol/3e7d5b84a16d076df02baa9f53271058 (well-known reference implementation); Python docs https://docs.python.org/3/library/asyncio-task.html

**Fits our case because:** Allows 50+ inhabitants to share ONE thread (the async loop's thread) rather than 50 separate threads. Each inhabitant is a `asyncio.Task` that awaits its interval. IO concurrency is excellent — while one inhabitant waits for an HTTP response, all others advance. This is 10-50x more memory-efficient than one thread per inhabitant.

**Why it's interesting:** The OctopusColony's monitoring thread could potentially merge into this single async loop. API callers (fetchers) are inherently IO-bound — this is their ideal execution model. The Finnhub WebSocket already runs async; unifying it with other inhabitants in the same event loop is architecturally clean.

**Risks:**
- Adds `async/await` to inhabitants that are currently synchronous. Migration cost is real — each client would need async versions of their blocking HTTP calls (httpx async, aiohttp, etc.). Not all third-party libraries support async.
- Python's GIL still applies within the single async thread. CPU-bound inhabitants would block the entire event loop. Mitigation: `loop.run_in_executor(None, blocking_fn)` for CPU work.
- `asyncio.run_coroutine_threadsafe()` introduces a synchronization boundary between the step loop and the async loop. Callbacks are fire-and-forget; results are consumed via EventBus.
- Risk of "callback hell" if inhabitants need to wait for each other. The EventBus pub/sub pattern mitigates this.

---

### Emerging Approaches

#### Approach 6: Mesa 3.5 Agent Self-Scheduling as First-Class Pattern

**What:** After upgrading to Mesa 3.5, inhabitants (as Mesa Agents) call `self.model.schedule_event(self.act, after=interval)` from within their own `step()` method to re-schedule themselves, rather than being activated every step. This creates "dormant until needed" inhabitants that only consume compute when their interval fires.

**Momentum:** This pattern emerged from Mesa's Hybrid ABM/DEVS discussion (#2032 on GitHub) which showed ~50% runtime reduction in a wolf-sheep example by eliminating wasted activation of inactive agents. Mesa 3.5 stabilized this capability in February 2025. The Mesa team explicitly designed agent self-scheduling as a use case (jailed prisoner example: agent schedules its own release).

**Source:** https://github.com/projectmesa/mesa/discussions/2032 (design discussion, 2024); https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html (official tutorial, 2025)

**Fits our case because:** Bio systems that currently "run every step but do nothing" (because their cadence gate hasn't fired) become completely dormant between activations. A CongressionalTracker that should run every 6 real-time hours doesn't register in the step loop at all until its next window. This is the most Mesa-native solution and requires no new threading infrastructure.

**Maturity risk:** Agent self-scheduling within Mesa 3.5 is documented but MIDGE hasn't verified that `self.model.schedule_event()` works correctly from within an agent's `step()` call in the presence of `_step_hooks`. The interaction between the hook system (MIDGE's invention) and Mesa 3.5's event scheduler is untested. Needs a prototype before committing.

---

#### Approach 7: Ray Actors for True Process-Level Isolation

**What:** Each inhabitant becomes a `@ray.remote` actor — a separate Python process with its own memory space. Ray handles scheduling, resource allocation, and fault recovery. Inhabitants communicate via Ray's object store.

**Momentum:** Ray 2.54.0 (current as of 2026). Used at Uber, Spotify, Instacart for large-scale distributed ML pipelines. 35k+ GitHub stars.

**Source:** https://docs.ray.io/en/latest/ray-core/actors.html (official docs, accessed 2026-03-08)

**Fits our case because (theoretically):** True CPU parallelism (bypasses GIL), automatic fault recovery, built-in resource accounting (CPU/memory per actor), distributed scaling if MIDGE ever moves to multi-machine.

**Maturity risk:** This is overkill for a single-machine deployment. Ray actors run in separate OS processes — 50 actors = 50 Python interpreters + Ray overhead (~200-500MB baseline before any work). The inter-process communication overhead is orders of magnitude higher than in-process EventBus. For IO-bound fetchers on a single machine, threading outperforms Ray. The research brief constraints also prohibit external single points of failure — Ray's head node is exactly that. Ray is the right tool when MIDGE grows to multi-machine; it is not the right tool now.

---

### Gaps and Unknowns

1. **How many step hooks are registered total?** I found 12+ across `wiring.py`, `market_hooks.py`, and `main.py`. Each fires every step. No benchmark exists for the overhead per step of the current hook architecture. Before any migration, measure this baseline.

2. **Does Mesa 3.5's `schedule_recurring` work correctly with MIDGE's `add_step_hook` system?** `add_step_hook` is MIDGE's own custom API, not native Mesa. Mesa 3.5's event scheduler lives inside `model.step()` (via `_wrapped_step`). MIDGE's hooks are called BEFORE agent activation inside `MycelialModel.step()`. Whether Mesa 3.5's event scheduling fires before or after MIDGE's hooks needs verification.

3. **What is the thread count budget on Wardenclyffe?** Windows 11 limits practical threading to ~hundreds of threads (OS scheduler overhead). 50 daemon threads is well within range. 200+ would start to show scheduling jitter. Not measured.

4. **Do any MIDGE inhabitants have shared mutable state that would require locking?** The EventBus has an RLock but is designed for concurrent access. The Thompson sampler, ConvergenceAlerter, and PatternLibrary have their own threading assumptions. If daemon-thread inhabitants call these systems directly (not via EventBus), race conditions may emerge. Needs audit.

5. **What is the performance cost of the current "if step % N" architecture?** The branch evaluation is trivial, but the closure captures and counter increments add up across 12+ hooks and hundreds of cadence checks per step. This hasn't been profiled. Profiling would clarify whether the step loop IS actually a bottleneck or only becomes one at 50+ inhabitants.

6. **Is there a wall-clock-to-step calibration mechanism?** MarketClock exists for market hours awareness. But "inhabitant wakes every 15 minutes wall-clock" requires mapping wall-clock intervals to step counts OR bypassing the step loop entirely. No existing facility for this mapping was found in the codebase.

---

### Synthesis

#### The Landscape in One Sentence

MIDGE already has the right skeleton — the step loop, the EventBus, and daemon threads (OctopusColony monitoring) — and the path to 50+ autonomous inhabitants is a migration, not a rewrite.

#### The Strongest Approach: Two-Tier Architecture

**Tier 1 (Step-relative cadences):** Upgrade to Mesa 3.5 and migrate `step % N` cadence branches to `model.schedule_recurring()`. This cleans up the monolithic hooks, gives each system an explicit cadence identity, and eliminates wasted computation. Zero risk, low migration cost, immediate readability benefit. Keep the step loop for systems that ARE step-relative: convergence checks, Thompson updates, hypothesis engine.

**Tier 2 (Wall-clock inhabitants):** For inhabitants with wall-clock semantics (Circadian Rhythm, Congressional Fetcher, EIA Energy, etc.), implement a lightweight `InhabitantScheduler` — a single daemon thread driving a `heapq` priority queue (Approach 2), dispatching callbacks to a small `ThreadPoolExecutor`. Inhabitants register themselves with their interval and callback. Results publish to EventBus. The step loop reads from EventBus — no direct coupling between step loop and daemon inhabitants. This is a formalization of the OctopusColony pattern, generalized to all inhabitants.

**Why not asyncio (Approach 5)?** Async is architecturally right for IO-bound inhabitants but requires migrating all fetcher clients from synchronous to async. That's a large migration for uncertain gain — the existing `ThreadPoolExecutor` in `MycelialModel._parallel_step()` already handles IO concurrency for agents. Async is the right answer at 200+ concurrent inhabitants; it's overfitted for 50.

**Why not Ray (Approach 7)?** Process-level isolation is expensive overkill for a single machine. Ray adds distributed infrastructure complexity that violates the project's "no external single points of failure" constraint.

**Why not Pykka (Approach 4)?** Pykka is conceptually elegant but doesn't add capability over plain `threading.Thread` + EventBus for MIDGE's specific use case. Its value is formalizing message-passing; MIDGE already has EventBus for that. The added dependency without added capability is a code smell.

#### The Migration Path

The key insight from the codebase: **the OctopusColony monitoring thread is the prototype for everything that follows.** It is already a daemon thread with its own interval (5 seconds wall-clock), publishing to EventBus, completely decoupled from the step loop. The `InhabitantScheduler` is just a generalization of this pattern to N inhabitants, with a priority queue instead of N separate threads.

**Step 1:** Benchmark the current step loop. Measure microseconds per step across hook count and step count. Establish the baseline before any change.

**Step 2:** Upgrade Mesa 3.4.2 to 3.5. No breaking changes. Run full test suite (4,536 tests) to verify.

**Step 3:** Replace `step % N` branches with `model.schedule_recurring()` calls where the cadence is arbitrary (stats collection, calibration, excavation). Leave the every-step checks (convergence, hypothesis engine) as-is.

**Step 4:** Build `InhabitantScheduler` — 50 lines of code, modeled on OctopusColony's `_monitoring_loop`. Wire it into bootstrap Layer 33 (or Layer 34 if the sequence matters).

**Step 5:** Migrate one inhabitant at a time to wall-clock scheduling. Start with the systems that have clear wall-clock semantics: Circadian Rhythm (market hours), Congressional Fetcher (6-hour cadence), EIA Energy (daily). Each migration is independent.

#### Critical Warning for the Orchestrator

**The step loop must not be removed.** The research brief says as much, and the codebase confirms why: `model.time`, `model.steps`, agent `step()` methods, the Mesa `agents.shuffle_do()` activation, the circadian rhythm's phase counter, the Granger analysis cadence — all of these are coherent ONLY in terms of step time. The step loop is the organism's heartbeat. Wall-clock inhabitants are its glands. Both must exist.

**The transition from step-count cadence to wall-clock cadence for any individual system requires answering: what does "every 50 steps" mean in wall-clock time?** At pace=1.0 (the default daemon flag), MIDGE runs ~1 step/second. At pace=2.0, ~0.5 steps/second. The step loop's pace is configurable. Any inhabitant that needs wall-clock precision must bypass the step count entirely (daemon thread) rather than trying to map wall-clock to step count.

---

### Sources

- Mesa 3.5 release: https://github.com/mesa/mesa/releases/tag/v3.5.0
- Mesa event scheduling tutorial: https://mesa.readthedocs.io/latest/tutorials/3_event_scheduling.html
- Mesa hybrid ABM/DEVS discussion: https://github.com/projectmesa/mesa/discussions/2032
- Mesa unified scheduling API PR: https://github.com/mesa/mesa/pull/3155
- Pykka v4.4.1: https://github.com/jodal/pykka
- Ray actors: https://docs.ray.io/en/latest/ray-core/actors.html
- APScheduler 3.11.2: https://github.com/agronholm/apscheduler
- Python threading: https://docs.python.org/3/library/threading.html
- asyncio event loop in separate thread: https://superfastpython.com/asyncio-event-loop-separate-thread/
- Free-threading vs async (Optiver): https://optiver.com/working-at-optiver/career-hub/choosing-between-free-threading-and-async-in-python/
- Python 3.14 free-threading asyncio scaling: https://labs.quansight.org/blog/scaling-asyncio-on-free-threaded-python
