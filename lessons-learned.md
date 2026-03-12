# Lessons Learned — Project-Specific

Reviewed by every instance on arrival. Append-only. Keep entries atomic and actionable.
Universal lessons go in `C:\Users\baenb\.claude\lessons-learned.md` instead.

## How to Use

**On arrival:** Scan this file for patterns relevant to your current task.
**After correction:** Append a new entry if the lesson is specific to THIS project.
**Format:** Use the template below. One pattern per entry. No narrative.

---

### Never assume stored data is read back
- **Pattern**: Built 25 raw_store methods across 6 mixins. Only 1 had a `get_*` read method. 24 SQLite databases were write-only black holes for months. Nobody checked.
- **Rule**: Every `store_*` method MUST have a corresponding `get_*` read method. Every read method MUST have at least one caller in the live pipeline. Verify both before marking storage work complete.
- **Why**: Data that's stored but never read is worse than not storing it — it creates false confidence that MIDGE is "learning" when she's actually blind.

### Audit API data extraction, not just API connectivity
- **Pattern**: 27 API clients were "wired" and "working" but most threw away 50-90% of fetched data. yfinance: 80+ fields fetched, 7 used. COT: 300 contracts fetched, 10 processed. Congress.gov: sponsors/committees/subjects all discarded. Nobody audited what was extracted vs available.
- **Rule**: When building or reviewing an API client, compare the API response schema against the dataclass fields. Document what's extracted and what's discarded. Flag high-value discarded fields.
- **Why**: MIDGE's intelligence is limited by what data reaches her brain, not by how many APIs she calls.

### Silent `except: pass` hides critical failures
- **Pattern**: `store_binance_funding()` and `store_kalshi_markets()` were called but the methods didn't exist. Wrapped in `except: pass`, they silently failed every single daemon run. Nobody knew.
- **Rule**: Never use bare `except: pass` on storage operations. At minimum use `logger.debug(exc_info=True)`. For new store methods, verify the method actually exists by checking `hasattr()` or running a test that calls it.
- **Why**: Silent failures compound. MIDGE ran for days thinking she was storing data when she wasn't.

### Use agents for audits, not solo work
- **Pattern**: Guiding Light corrected: "Stop doing things on your own. Use agents." Single-instance work touches 2-3 files and compacts context. Parallel agents audited 27 clients in 3 minutes.
- **Rule**: Any task touching 5+ files or requiring cross-codebase analysis MUST use parallel agents. Reserve main context for orchestration and user communication.
- **Why**: Context is precious. Solo deep-dives eat it. Agents are disposable and parallelizable.

### SQLite is for storage, not for thinking
- **Pattern**: MIDGE stored data in SQLite and assumed that was sufficient for pattern recognition. SQLite can't do semantic search, graph traversal, or cross-domain correlation at scale.
- **Rule**: SQLite = raw data landing zone. DuckDB = analytical queries. Neo4j = causal relationships. Qdrant = semantic similarity. Use each tool for what it's built for.
- **Why**: "Does COT positioning predict EIA surprises?" requires cross-domain temporal analysis. SQLite can't answer that. DuckDB can.

### Verify the full pipeline, not just individual components
- **Pattern**: Every individual system "worked" — API clients fetched data, adapters created signals, convergence engine fired alerts. But the full pipeline had gaps: data entered SQLite and died, cascade chains were never expired, sweep bypass skipped risk gates.
- **Rule**: Before declaring anything "done," trace the full pipeline: data source → storage → analysis → signal → convergence → outcome → feedback. If any link is broken, the system doesn't work.
- **Why**: A chain of working components with broken connections is indistinguishable from a system that does nothing.

### Level-based economic series need delta computation
- **Pattern**: 6 FRED series (PCEPI, PCEPILFE, RSXFS, M2SL, DFF, CPIAUCSL) permanently returned direction="neutral" because they fetched limit=1 and couldn't compute a delta.
- **Rule**: Series that represent levels (not rates) must fetch limit=2 minimum. Direction comes from the change between observations, not from thresholds on the level itself.
- **Why**: Inflation at 3.2% means nothing without knowing it was 2.8% last month. The delta IS the signal.

### Update HANDOFF.md and lessons-learned.md every session
- **Pattern**: 5 sessions of work happened before lessons-learned.md got a single entry. HANDOFF.md was stale for multiple sessions. Future instances repeated the same mistakes.
- **Rule**: Every session MUST update HANDOFF.md with session notes AND append relevant lessons to this file. Check both before closing.
- **Why**: Without persistent lessons, every instance starts from zero. Guiding Light has to re-teach the same things. That's unacceptable.

### Build the analyst FIRST, not the plumbing
- **Pattern**: 8+ sessions built API clients, bootstrap layers, connections, signals, templates — 287K signals accumulated, 43 templates, 56 correlations. But NOTHING synthesized them into "what's most likely to happen." Guiding Light asked for this repeatedly in different words across multiple sessions. Every instance treated it as a coding problem and built more infrastructure instead.
- **Rule**: Before building ANY new data source, connection, or system, verify that MIDGE can answer: "Based on everything I know, what are the top 5 most likely near-term moves?" If she can't, fix THAT first. The analyst (`deep_analyst.py`) exists for this. Run it. If its output is weak, improve it. Don't add more plumbing.
- **Why**: Data without analysis is a filing cabinet nobody reads. MIDGE's value is in synthesis, not collection. Guiding Light doesn't care how many API clients exist — they care whether MIDGE can tell them something actionable.

### MIDGE must start from knowledge, not from zero
- **Pattern**: Convergence buffer started with 131 signals in 2 domains after restart. Needed 3+ domains to fire. MIDGE sat blind until live signals dripped in over hours. Meanwhile, 297K historical signals existed in the archive, unread.
- **Rule**: On startup, warm the convergence buffer from the signal archive (`startup_warmup.py`). Run `archive_scanner.py` to log what MIDGE knows. Run `DeepAnalyst.analyze()` to produce immediate findings. MIDGE should never start from zero when she has data.
- **Why**: Every restart that discards accumulated knowledge is a waste. The daemon should start smarter than it was yesterday, not amnesia every boot.

### "Built" is not "wired" — check for disconnected systems
- **Pattern**: DeepAnalyst was built in Session 3 (474 lines, tested, working). Two sessions later it had ZERO bootstrap references — never constructed, never called during daemon operation. PatternWatcher existed but only ran on a 10-step cadence, not reactively. WorldModel had `add_discovered_edge()` but nobody called it from Granger or lag findings. 3 of 15 intelligence systems were built but disconnected.
- **Rule**: After building any system, grep for its class name in `market_systems.py` (bootstrap), `market_hooks*.py` (step loop), and `sensing_collector.py` (signal pipeline). If it appears in zero of these, it's dead code. A system that isn't bootstrapped AND called from a step hook is a system that doesn't exist.
- **Why**: Waking these 3 systems produced immediate measurable activity: 649 reactive pattern checks, 20 ranked inevitabilities, 10 auto-discovered WorldModel edges — all in the first 1.5 hours. The intelligence was there all along; it just wasn't connected.

### Reactive beats polling for pattern detection
- **Pattern**: PatternWatcher ran every 10 steps via `_run_sensing_archaeology`. By adding a reactive check in `_collect_one()` (fires on every signal ingestion), pattern stacks are detected the moment signals arrive instead of up to 10 steps later.
- **Rule**: For detection systems (PatternWatcher, convergence checks), trigger reactively on signal arrival AND keep the cadence tick as a safety net. The cadence tick catches anything the reactive path missed; the reactive path catches things instantly.
- **Why**: In a fast market, 10 steps of delay (25+ seconds) can mean missing a stacking pattern as it forms. The reactive path caught 649 checks in 1.5 hours. The cadence safety net adds redundancy without cost (alerter deduplication prevents double-fires).

### Auto-discovery turns curated graphs into living knowledge
- **Pattern**: WorldModel had 102 hand-curated edges. Granger analyzer found 3 causal relationships per cycle. Lag correlator found 70+ correlations with |r| >= 0.6. But neither fed discoveries into the WorldModel — the graph never grew.
- **Rule**: When statistical methods discover relationships (Granger causality, lag correlations), feed them into the WorldModel via `add_discovered_edge()`. Set a quality threshold (|r| >= 0.6 for lag, all significant for Granger). The method handles both new edges and strengthening existing ones on re-discovery.
- **Why**: A static causal graph is a human's best guess. A growing graph is the system learning its own domain. Within 1.5 hours, the WorldModel grew from 102 to 112 edges autonomously — and will keep growing every 500 steps.
