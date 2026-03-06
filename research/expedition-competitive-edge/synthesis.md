# Expedition Synthesis: Competitive Edge — How MIDGE Beats Enterprise Trading AI

## Date: 2026-03-05
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief

---

## High Confidence (teams converged, validators confirmed)

### 1. MIDGE's Cross-Domain Synthesis Is a Genuine Structural Moat

All 5 teams independently confirmed: no competitor — Kensho, QuantConnect, Numerai, Alpaca, Man Group's AlphaGPT — provides automated cross-domain convergence detection. Every platform treats data domains as separate silos. MIDGE is the only system that:
- Ingests 28+ sources across 11 domains simultaneously
- Detects when 3+ independent domains converge on the same ticker/direction
- Learns signal reliability via Bayesian Thompson Sampling
- Reverse-engineers historical moves to find the domain patterns that preceded them

Man Group's AlphaGPT is the closest analog but focuses on signal discovery within financial data, not across domains. Numerai validates the independence principle ($500M valuation, $30M Series C) but only uses traditional equity features. MIDGE's combination moat gets stronger as single-domain alpha decays — hedge funds now all use the same satellite, credit card, and sentiment data.

### 2. Domain Independence Is Unverified — The Foundation Assumption

Three teams (2, 4, 5) and all 3 validators independently flagged: MIDGE's stacking confidence math (`1 - (1-a)(1-b)(1-c)`) is only valid when domains are genuinely independent. Nobody has verified this empirically. If "technical" and "events" domains both fire on earnings days, they may be 70%+ correlated — meaning a "5-domain convergence" might actually represent 2-3 independent signals.

The infrastructure to test this already exists: `CorrelationTracker.get_correlation_matrix()`. This check requires zero new code — just running the function on existing data.

### 3. The Payoff Ratio Is The Most Important Unknown Number

All 3 validators flagged this unanimously: nobody computed the actual payoff ratio (average win magnitude / average loss magnitude) from `outcomes.jsonl`. This single number determines whether MIDGE is already profitable at 19.9% win rate:
- At 4:1 payoff ratio → break-even at 20% win rate (MIDGE is right there)
- At 5:1 → profitable at 16.7% (MIDGE is comfortably above)
- At 3:1 → need 25% win rate (MIDGE is below)

Every Kelly calculation, every position sizing discussion, every win rate optimization recommendation is operating on this missing input. Measuring it from `outcomes.jsonl` is an hour of Python analysis.

### 4. Parallel Excavation Is the Highest-Leverage Throughput Change

Teams 3 and 4 converge: ProcessPoolExecutor for CPU-bound symbol excavation is the correct bottleneck fix. Validator 2 confirmed Polygon.io Starter plan has unlimited API calls (eliminates rate limit concern). The immediate zero-code-change action: run `populate_library.py` as a separate companion process alongside the daemon.

Validator 3 flags a risk: Windows ProcessPoolExecutor uses "spawn" not "fork" — MIDGE's 33-layer bootstrap objects must be picklable. This needs testing before implementing parallel workers.

### 5. EIA Energy Supply Is the Best New Domain

Teams 1 and 2 converge, validators confirm: EIA weekly energy data (natural gas storage, petroleum inventories) is the highest-priority new domain.
- Free government API (registration required)
- Thursday EIA report routinely moves nat gas futures 2-5% on surprises
- Genuinely orthogonal to all 11 existing domains (physical supply ≠ insider behavior ≠ technical patterns)
- Low engineering effort (thin REST wrapper, weekly data)

### 6. Combo-Specific Kelly Sizing Using Existing Data

Team 5's recommendation, uncontested by validators: ComboThompson already tracks per-combination Beta distributions. Using the distribution mean as a continuous multiplier on Kelly sizing is architecturally consistent and requires only plumbing changes. The worst combo (8.3% WR) has a negative Kelly fraction — it should never be traded. The best volume combo (31.2% WR, n=32) warrants ~14% Kelly (half-Kelly: 7%).

Critical guard: require n>=15 samples before using combo-specific sizing. Combos at n=3 are statistically meaningless.

---

## Battle-Tested Approaches (proven, ready to implement)

### New Data Domains (Priority Order)

| Domain | Sources | Cost | Evidence | Effort | Validator Notes |
|--------|---------|------|----------|--------|-----------------|
| **energy_supply** | EIA nat gas storage, petroleum inventory | Free | Strong — Thursday EIA report moves futures 2-5% | Low | Confirmed by all |
| **legislative** | Congress.gov API + LegiScan free tier | Free | Moderate — GovGreed building commercial service validates thesis | Medium — needs NLP bill classification, start with keywords | Congress.gov is batch-oriented, not real-time |
| **agriculture/weather** | USDA NASS crop progress + NOAA drought | Free | Strong — published research on crop condition vs commodity prices | Medium — seasonal (April-November), needs sector mapping | Valid but seasonal limitation noted |
| **logistics** | Baltic Dry Index via FRED (not FBX) | Free | Strong — Nature 2023: freight predicts stock returns in 26/29 countries | Low (BDI proxy) to High (raw AIS) | FBX API is enterprise-only (Team 2's "free" claim debunked) |

### Pattern Discovery Upgrades (Priority Order)

| Method | Library | What It Does | Effort | Validator Notes |
|--------|---------|-------------|--------|-----------------|
| **Granger causality** | statsmodels (already installed) | Directed lag-based causal testing between domain pairs | Low — drop-in for LagCorrelationAnalyzer | Confirmed, well-established |
| **Transfer entropy** | infomeasure (NOT ordpy) | Directed nonlinear information flow between domains | Medium — new library, but CPU-only, milliseconds per pair | ordpy does NOT implement TE (Validator 3 caught this) |
| **RMT denoising** | skfolio | Removes noise from correlation matrix before anomaly detection | Low — apply to CorrelationTracker output | Requires 90+ day window (current 30-day is too short) |
| **PCMCI+ causal graph** | Tigramite | Discovers causal graph conditional on all other domains | Medium — minutes per daily run, cadenced | Python 3.14/numba compatibility unverified |
| **FP-Growth pattern mining** | mlxtend | Discovers multi-domain sequential patterns from signal archive | Medium — treat daily signals as baskets | Confirmed, actively maintained |

### Processing Architecture (Priority Order)

| Change | Code Impact | Expected Benefit |
|--------|-------------|------------------|
| **Run populate_library.py as companion process** | Zero code changes | Immediate: excavation runs alongside daemon |
| **Expand sensing workers 3 → 12-20** | One-line config change + per-source semaphores | 4-7x more concurrent API fetches |
| **ProcessPoolExecutor for parallel excavation** | Moderate — add to populate_library.py | 8-10x excavation speedup (test pickling first) |

---

## Novel Approaches (worth investigating, not yet proven for MIDGE)

### BOCPD for Cross-Domain Coupling Detection
When two previously uncorrelated domains suddenly couple, Bayesian Online Changepoint Detection fires an alert at the 2nd or 3rd observation. Library: `changepoint` (not `bocpd` which is abandoned). Fits MIDGE's deception detection architecture.

### Cross-Asset Ensemble Validation (WorldQuant/Numerai paradigm)
Test every pattern template against 3+ instruments from different sectors before promotion. A pattern that works on NVDA AND MSFT AND SPY is structural; one that only works on NVDA is overfit. MIDGE's PatternWatcher already requires 3+ symbols — this extends that to the hypothesis loop.

### Signal Neutralization for Independence Enforcement
After computing a new hypothesis, regress it against existing patterns. Only register the residual — the genuinely new information. Validators flag: this needs careful adaptation for MIDGE's convergence alerts (which are structural detections, not independent predictions like Numerai models).

---

## Synthesized Recommendation: The Action Sequence

**Phase 0 — Measure before you build (1 day)**
1. Parse `outcomes.jsonl` → compute payoff ratio (average win / average loss magnitude)
2. Run `CorrelationTracker.get_correlation_matrix()` → verify domain independence
3. These two numbers determine whether MIDGE needs optimization or is already profitable

**Phase 1 — Use what you have (1 week)**
4. Raise combo sample size gate from n≥5 to n≥15
5. Apply combo-specific Kelly fraction using ComboThompson mean
6. Run `populate_library.py` as companion process alongside daemon
7. Expand sensing workers from 3 to 12-20

**Phase 2 — Add new eyes (2-3 weeks)**
8. Build EIA energy supply client (thin REST wrapper)
9. Build Congress.gov legislative tracker (keyword-based bill classification)
10. Add Granger causality to LagCorrelationAnalyzer (statsmodels, drop-in)

**Phase 3 — Sharpen the brain (3-4 weeks)**
11. Add transfer entropy via infomeasure library
12. Implement RMT denoising on CorrelationTracker matrix
13. Build USDA + NOAA agriculture/weather domain (seasonal)
14. ProcessPoolExecutor for parallel excavation (after pickling test)

**Phase 4 — Advanced intelligence (6-8 weeks)**
15. PCMCI+ conditional causal graph (if domain correlation confirmed)
16. FP-Growth sequential pattern mining on signal archive
17. BOCPD for real-time cross-domain coupling detection
18. Logistics domain via BDI proxy

---

## Disagreements

### Excavation vs. Pattern Quality as Primary Bottleneck
- Team 3: "The real bottleneck is excavation throughput"
- Team 4: "MIDGE's biggest gap is that all correlation is bivariate Pearson"
- Both are correct in different dimensions. Excavation limits how fast the template library grows. Correlation quality limits how accurate the stacking confidence is. The action sequence addresses both.

### Independence Enforcement vs. Payoff Ratio as First Priority
- Team 4: Fix domain independence first (PCMCI+)
- Team 5: Measure payoff ratio first
- Validators agree with Team 5: the payoff ratio is a 1-hour measurement that could prove MIDGE is already profitable. Domain independence is a longer investigation. Measure first, then fix.

---

## Filtered Out

| Recommendation | Why Removed |
|---|---|
| Two Sigma "18% improvement" | Unverifiable — industry benchmark, not Two Sigma disclosure. All 3 validators flagged. |
| FBX as "free" logistics source | API is enterprise-only. Validators confirmed via Freightos documentation. |
| ordpy for transfer entropy | Does NOT implement TE (Validator 3). Use infomeasure instead. |
| mlfinlab for MST/PMFG | Closed-source, not freely available. Use scipy instead. |
| bocpd for changepoint detection | Abandoned since Oct 2023. Use changepoint library instead. |
| ApeWisdom as "quick win" | Already fully wired into convergence engine (Validator 3 confirmed via codebase). |
| Mesa batch_run for sharding | Misapplication of parameter-sweep API. Thompson race condition unsolved. |
| Telegram/Discord sentiment | Legal risk (TOS violation), no free access path. |
| Kelly-VIX "23.1% returns" | Tested on index options put-writing, not equity predictions. Not transferable. |
| Wikipedia page views | 13-year-old research with no 2024-2025 replication. Alpha likely decayed. |
| Copula modeling | Requires far more data than MIDGE's 414-day archive provides. |

---

## Risks

1. **Windows ProcessPoolExecutor pickling** — MIDGE's 33-layer bootstrap objects may not be picklable. Must test before committing to parallel excavation.
2. **Tigramite/numba compatibility with Python 3.14** — Unverified. Must pip install and test before building PCMCI+ integration.
3. **New domains require 90-180 day ramp-up** — Team 4's advanced analytics can't operate on new domains until sufficient history accumulates.
4. **Mathematical Law compliance overhead** — Adding new domains requires triadic connections, ConnectionRegistry wiring, HolonRegistry integration, and bootstrap layer updates. "Low effort" ratings don't account for this.
5. **Congressional trade alpha decay** — NANC/GOP ETFs commoditize the raw signal. MIDGE's edge must be the cross-domain correlation layer, not congressional trades in isolation.
6. **Man Group's AlphaGPT roadmap** — Bloomberg reporting hints at expansion to cross-domain connections. Monitor as the closest institutional competitor.
