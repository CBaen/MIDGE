# Research Brief: MIDGE Operational Efficiency
## Date: 2026-03-12
## Project: MIDGE (Mae for trading)

### Problem Statement
MIDGE is running live with all 5 arcs of her circular intelligence architecture wired and growth sprint cadences active. She's ingesting 89K+ signals from 31 data sources, but producing zero learning and zero alerts. She's eating but not digesting. Three symptoms:
1. Thompson distributions: 146 total, ALL at prior (alpha=1, beta=1) — zero learning
2. Convergence engine: zero alerts firing, despite DeepAnalyst finding defense sector inevitabilities (NOC/LMT/GD with 5 domains stacking)
3. Hypothesis engine: 30 generated, 0 promoted, 4 active — stuck in limbo

### Expected Outcome
Understand WHY each subsystem is silently failing. Find the specific code paths, threshold values, timing issues, or broken handoffs that prevent data from flowing through the full loop. The result should be a prioritized list of fixes that, once applied, would make MIDGE actually learn from her data and produce convergence alerts.

### Current State
- Daemon running at step 4800+, 24 agents, pace 2.0
- Growth sprint cadences: fetch/10, outcome/75, Thompson forget/75, slow cadence/200/500/2000
- 89K+ signals ingested across: TA indicators (620/batch), OpenInsider (107), SEC EFTS, FRED macro, congress legislation, economic calendar, EIA energy, Google Trends, order flow, crypto, institutional 13F, hiring, session sweeps
- Predictions being registered for outcome tracking (hiring, TA signals)
- DeepAnalyst producing inevitabilities (NOC bullish 0.655, LMT bullish 0.648, GD bullish 0.640)
- Convergence state: regime=sideways, zero ticker alerts, zero global/tactical/strategic alerts
- Hypotheses: 4 active, 30 generated, 0 promoted
- Thompson: 146 distributions, 0 learned (all at prior)
- WitnessNotifier: 570K+ witnessed, 0 failures, 150 channels, ~40 witnesses active
- Running after market hours (9:30 PM ET) — most data sources return stale/cached data

### Project Direction
MIDGE is an inevitability surfacer — she finds where converging forces across financial/economic/social/political domains make outcomes structurally inevitable. She should be discovering pattern stacks, learning which sources are reliable, and producing convergence alerts for human review.

### Constraints
- 8 Mathematical Laws govern all changes (Law 1: no bare dyads, Law 7: rule of 3/5, etc.)
- Zero regression policy — all 1074 tests must pass
- Advisory enforcement — triads observe/report, never block
- No monoliths — one job per file, flag files over 500 lines
- Growth sprint cadences are intentional — do not recommend reverting them

### Destructive Boundaries
- Do NOT suggest removing or restructuring the circular architecture (5 arcs)
- Do NOT suggest removing bio-systems or EventBus wiring
- Do NOT suggest changing Mae's Mathematical Laws
- Do NOT suggest reverting growth sprint cadences

### Research Angles

**Team 1: Thompson Feedback Loop** — Trace the exact path from signal ingestion → prediction registration → outcome window expiry → outcome grading → Thompson update. Find where the chain breaks. Check: Is OutcomeCollector receiving the right ThompsonSampler instance? Are predictions being registered with correct source keys? Are outcome windows expiring and being graded? Is the grading calling thompson_sampler.update()?

**Team 2: Convergence Silence** — DeepAnalyst finds NOC/LMT/GD with 5 domain stacks but convergence fires nothing. Investigate: signal buffer contents (what domains are actually in the buffer per ticker), domain window expiry times, min_domains threshold, after-hours effects on signal freshness, whether convergence_check() is being called, and the confidence calculation pipeline.

**Team 3: Hypothesis Stagnation** — 30 generated, 0 promoted. Investigate: promotion criteria in hypothesis_engine.py, validation thresholds in hypothesis_validator.py, DSR requirements, what "active" vs "probation" means operationally, whether the engine has enough graded outcomes to promote anything, and the cadence at which validation runs.

**Team 4: Cadence Conflicts & Timing** — The growth sprint halved all cadences. Investigate: Is Thompson forgetting running before any learning happens (forgetting at prior = no-op, or destructive)? Are outcome windows (3-30 days) incompatible with step-based cadences? Is rate limiting (yfinance, Finnhub) starving specific domains? Is the after-hours timing causing all signals to arrive in one domain window and expire together?

### Team Size: 4
Four distinct, independent failure modes requiring deep code-level investigation.

### Failed Approaches
- Thompson feedback loop was fixed once before (Session 6, 2026-03-09) — 4 compounding bugs found. May have regressed or new bugs introduced.
- Convergence was working during replay_history.py runs (288 alerts in Feb 2026 replay) but may not work in live daemon mode.
