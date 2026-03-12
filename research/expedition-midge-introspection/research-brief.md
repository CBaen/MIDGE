# Research Brief: MIDGE Internal Introspection

## Date: 2026-03-12
## Project: MIDGE (Mae for trading)

### Problem Statement

MIDGE is a world of systems, data flows, memory layers, and intelligence engines. She looks outward at markets to find inevitabilities — patterns of convergence where multiple forces stack to make an outcome structurally inevitable. But nobody has applied that same lens INWARD. What internal inevitabilities exist inside MIDGE? What systems complement each other but don't know it? What data flows die before reaching the systems that need them? What capabilities would emerge from connecting things that are currently separate?

Guiding Light: "like the inevitabilities she's looking for, what are the inevitabilities we should be seeing inside her?"

### Expected Outcome

A complete map of MIDGE's internal ecosystem showing:
- Every system classified by liveness (alive/dormant/decorative/zombie)
- Every data flow traced from source to terminal use, with dead ends identified
- Natural complementary pairs — systems whose outputs match another's inputs
- Memory layer utilization gaps — write-only stores, starving readers
- Emergent capabilities that would arise from connecting currently-separate systems

The output should read like MIDGE's own convergence analysis, but applied to herself.

### Current State

MIDGE has 149 systems (92 core + 57 market), 37 data sources, 13 domains, 5+ memory layers (Qdrant, SQLite/raw_store, JSONL cold storage, ctx runtime, EventBus real-time), 29/30 bio-systems wired, OctopusColony bootstrapped, 3 execution venues (Alpaca, FTMO, Kalshi).

Recent Session 6 work wired:
- DeepAnalyst output → persist JSONL, embed Qdrant, format plain language, register outcomes, full EventBus
- HypothesisEngine ← archaeological_analyzer adapter + Granger findings + CH_HYPOTHESIS_FIRED subscriber
- 3 edge detectors (cluster_detector, politician_tracker, contract_predictor) wired into live sensing
- Law 7 enforcement gate (3/4 validators required before paper trade)

But this session's work was targeted at known gaps. The expedition should find what we DON'T know about — the unknown unknowns.

### Project Direction

MIDGE is an inevitability-surfacing organism. Every system exists to detect, validate, remember, or act on patterns of convergence. The goal is autonomous trading across all markets (Alpaca/FTMO/Kalshi/crypto) based on high-confidence inevitabilities.

### Constraints

- Mae's 8 Mathematical Laws govern all changes (see CLAUDE.md)
- 500-line file cap enforced
- No monoliths — one job per file
- No breaking existing tests (4700+ tests, zero regression policy)
- Advisory enforcement — triads observe/report, never block
- Bio-systems should be ACTIVATED with real market jobs, not deleted

### Destructive Boundaries

- Do NOT suggest deleting bio-systems or core Mae infrastructure
- Do NOT suggest replacing the convergence engine or Thompson sampler
- Do NOT suggest changing the 33-layer bootstrap architecture
- Do NOT suggest new external dependencies or tools

### Research Angles

1. **System Liveness Census** (Team 1) — Classify every bootstrapped system as alive (producing meaningful output), dormant (built but never called), decorative (called but output ignored), or zombie (called but silently erroring). Include all 29 bio-systems. For each non-alive system, what would it take to wake it up?

2. **Data Flow Topology** (Team 2) — Trace every data path from API source through processing to terminal use (paper trade, memory storage, human alert, Thompson update). Find dead ends where data enters and dies. Find missing feedback loops. Find one-way streets that should be bidirectional.

3. **Complementary Convergence** (Team 3) — Find systems whose outputs naturally match another system's inputs but aren't connected. These are MIDGE's "internal inevitabilities" — connections that are structurally obvious once you see both sides. Rank by impact.

4. **Memory & Learning Completeness** (Team 4) — Map which systems write to which memory layers (Qdrant, SQLite, JSONL, ctx, EventBus, plain_language). Which read? Where are write-only stores? What stored data could feed a system that's currently data-starved?

5. **Emergent Capability Analysis** (Team 5) — What qualitative leaps would emerge from connecting 2-3 currently-separate systems? Not incremental improvements — genuine new capabilities that don't exist in any individual system but would emerge from their combination.

### Team Size: 5

Five independent angles, each requiring deep codebase reading across different file sets. Minimal overlap between angles ensures maximum coverage.

### Failed Approaches

Previous ecosystem audit (Session 6) found and fixed the most obvious disconnections. This expedition goes deeper — not "what's broken" but "what's possible."
