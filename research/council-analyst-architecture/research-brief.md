# Research Council Brief: Communicating Multi-Analyst Architecture
## Date: 2026-03-13
## Project: MIDGE (Mae for trading)

### The Question
MIDGE has 28 live intelligence systems funneling through ONE analyst (DeepAnalyst) that runs every 200 steps. Guiding Light wants multiple high-quality analysts that communicate with each other — not silos. Pattern stacking (the core value proposition) requires analysts that share findings, challenge each other, and build composite pictures of inevitability. How should this be architected?

### Expected Outcome
A design where three specialized analysts each see what the others found and build on it — like a war room, not separate offices. The output should be human-readable (Guiding Light is a designer, not a coder). The analysts must understand temporal relationships: "Patterns are sequences with time, rest periods, and energy transfer."

### Current State
- DeepAnalyst (482 lines) runs every 200 steps, reads 7 data sources, produces ranked Inevitabilities with a 6-component scoring model
- 3 tiered ConvergenceAlerters (tactical/strategic/thematic) are BUILT, RUNNING, and producing output — but their output goes to a dict (`ctx._market_advisory`) that nothing downstream reads. This is the most obvious existing skeleton.
- Plain language formatter exists and works (zero-jargon output)
- EventBus provides message passing between all systems
- WorldModel causal graph: 114 nodes, 102 edges
- PatternLibrary: 44 templates from archaeology (273K fingerprints)
- PostMortemReviewer: analyzes WHY predictions succeed/fail
- HypothesisEngine: RSI Layer 2 hypothesis lifecycle (2,829 hypotheses)
- CorrelationTracker + GrangerAnalyzer + LagCorrelationAnalyzer: temporal relationship detection
- Signal archive: 733K signals, 49 days, 354 tickers with 3+ domains
- Signal buffer: 32,580 signals, 72% technical analysis (domain imbalance)
- Thompson Sampler: 83 core + 20 combo distributions (Bayesian reliability)
- CascadeTracker: watches causal chain confirmations (energy transfer tracking)

### Project Fingerprint
- **Runtime:** Python 3.14.2, Mesa 3.4.2 (agent-based modeling framework)
- **Key dependencies:** NumPy 2.3.5, Pandas 2.3.3, statsmodels, stumpy, river, networkx, yfinance, httpx, alpaca-py
- **Architecture:** 33-layer bootstrap, 149 systems (92 core + 57 market), fractal triadic structure
- **State management:** In-memory signal buffer + JSONL persistence + SQLite raw stores + JSON state files
- **Database/Storage:** 19 SQLite databases (WAL mode), JSONL signal archive, JSON state files, Qdrant vector store
- **Known constraints:** Mae's 8 Mathematical Laws govern ALL changes. Law 2 (Triadic Generator) means 3 analysts, not 2 or 4. Law 7 (Rule of 3/5) means minimum 3 validators. Law 1 (No Bare Dyads) means every analyst connection needs a witness. Under 500 lines per file. One job per file.
- **Prior approaches:** DeepAnalyst exists but is a solo generalist. Three tiered alerters were built but never wired to output.
- **Active boundaries:** Must not break the existing convergence engine, Thompson feedback loop, or hypothesis lifecycle. Must not add new data collection — focus on analyzing what exists.

### Constraints
- Mae's 8 Mathematical Laws (especially Laws 1, 2, 4, 7)
- Under 500 lines per file (monolith prevention)
- Must use EventBus for inter-system communication (Transfractal Compromise)
- Must produce human-readable output via plain_language.py patterns
- Must bootstrap into Layer 33 market systems
- Must not duplicate DeepAnalyst — each analyst must have genuinely different analytical capability
- Analysts must communicate (no silos) — pattern stacking requires shared context

### Destructive Boundaries
- Do NOT modify the convergence engine (convergence_alerter.py + mixins)
- Do NOT modify the Thompson feedback loop (thompson_sampler.py + outcome_collector.py)
- Do NOT modify the hypothesis lifecycle (hypothesis_engine.py + registry + validator)
- Do NOT suggest adding new data sources — the brief is about analyzing existing data better

### Failed Approaches
- DeepAnalyst as sole synthesizer: works but is one overworked generalist. Can't specialize.
- Three tiered alerters: built but disconnected. Output goes to a dict nobody reads. The tiered approach (tactical/strategic/thematic) may be the right skeleton but needs validation.

### Codebase Files for Analysis
- `mae_core/market/intelligence/deep_analyst.py` — current sole analyst
- `mae_core/market/intelligence/convergence_alerter.py` — crown jewel, how convergence works
- `mae_core/market/intelligence/convergence_detection.py` — detection mixin
- `mae_core/bootstrap/market_hooks_steps_core.py` — step hook cadences
- `mae_core/bootstrap/market_hooks_steps.py` — slow cadence ops (200/500/2000)
- `mae_core/bootstrap/market_hooks_sensing_setup.py` — tiered alerter construction
- `mae_core/bootstrap/market_systems.py` — bootstrap Layer 33
- `mae_core/market/intelligence/world_model.py` — causal graph
- `mae_core/market/intelligence/cascade_tracker.py` — energy transfer tracking
- `mae_core/market/intelligence/post_mortem.py` — prediction failure analysis
- `mae_core/market/archaeology/pattern_watcher.py` — live pattern matching
- `mae_core/market/intelligence/correlation_tracker.py` — cross-domain correlation
- `mae_core/market/intelligence/hypothesis_engine.py` — hypothesis lifecycle
- `mae_core/market/plain_language.py` — human-readable output formatting
- `mae_core/market/intelligence/learning_config.py` — self-modifiable learning parameters

### External Research Angles
1. How do professional trading desks organize their analyst teams? (Domain vs. temporal horizon vs. analytical method specialization)
2. What patterns exist for multi-agent communication in ABM systems — blackboard architecture, publish-subscribe, shared memory, stigmergy?
3. How do ensemble methods in ML combine multiple models that see the same data differently? (Boosting, stacking, mixture of experts — as architectural inspiration, not literal implementation)
