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
