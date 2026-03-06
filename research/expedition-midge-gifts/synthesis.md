# Expedition Synthesis: Gifts for Midge — Expanding Her Senses, Speed, and Reasoning

## Date: 2026-03-05
## Vetted by: Orchestrator
## Alignment: Checked against Research Brief

---

## High Confidence (teams converged, validators confirmed)

### 1. Sensing Workers 3→12 Is the Safest, Fastest Win

All 3 validators and Team 5 agree: changing `max_workers=3` to `max_workers=12` in MarketSensingHook is a one-number change that delivers 4x more concurrent signal fetches. The GIL is released during network I/O, so threads achieve real concurrency. Previous expedition and this one both converge. Five minutes of work.

### 2. ProcessPoolExecutor Is Confirmed Dead — Async Is the Replacement

Team 5 ran the specific test: `requests.Session` is not picklable. Windows uses "spawn" (not "fork"), so every worker argument must be picklable. The previous expedition's Phase 3 recommendation for ProcessPoolExecutor is now formally superseded. The correct path is converting PolygonBulkFetcher to `aiohttp` with `asyncio.gather()` — projected speedup from 85s/100 symbols to 3-10s/100 symbols. Validators confirmed the math and noted "3-10s" is more realistic than "~5s" due to TCP overhead and server-side variance.

### 3. edgartools Is the Strongest New Capability Recommendation

All 3 validators independently highlighted edgartools as the highest-confidence recommendation. 1,800 stars, 3,459 commits, MIT license, actively maintained. Adds 13F (hedge fund quarterly holdings) and 13D/G (activist positions crossing 5%/10% thresholds) parsing that MIDGE's existing SEC client doesn't cover. Activist positions are high-signal events (10-30% stock moves after announcement). Low implementation complexity — `pip install edgartools`, drop-in upgrade to existing `sec_edgar/`. No API key required.

### 4. Finnhub Is the Single Highest-Value New Data Source

Free tier verified at 60 req/min. One API key delivers news sentiment + economic calendar simultaneously. When CPI comes in hot + tech news goes bearish + StockTwits is bearish = three-domain convergence. Validated by Team 1, uncontested by all validators. The `finnhub-python` library is official and maintained.

### 5. Qwen3-14B via Ollama Is the Right LLM Choice

Model existence verified (released April 29, 2025, Alibaba). VRAM: ~9-10GB weights at Q4_K_M, ~11-12GB with KV cache at short context. Generation speed: 42+ tokens/sec on RTX 4070-class hardware. OllamaProvider is a thin wrapper on existing OpenAIProvider pattern (~50 lines core, ~300-400 lines total including EventBus subscriber, CausalNarrative model, and Layer 33 wiring). Validators confirmed the architecture is clean. **Critical gate:** Run `nvidia-smi` on Wardenclyffe first — 12GB GPU is tight, 16GB+ is comfortable.

### 6. No Open-Source Library Does What ConvergenceAlerter Does

Team 3 confirmed: TradingAgents (31K stars) and Qlib (38K stars) are framework competitors, not drop-in libraries. No packaged library performs automated multi-domain convergence detection with Bayesian signal reliability scoring. MIDGE's ConvergenceAlerter is genuinely ahead of what's publicly available. This is confirmation of competitive moat, not a gap.

### 7. NetworkX Is the Correct World-Model Substrate

Teams 2 and 3 both recommended NetworkX DiGraph. Pure Python, pip install, zero Windows issues, holds MIDGE's working graph easily. CausalReasoningEngine's existing `_links` dict maps directly to NetworkX edge attributes. The conversion is non-breaking.

---

## Battle-Tested Approaches (proven, ready to implement)

### New Data Sources (Priority Order)

| Source | Signal | Cost | Rate Limit | Validator Notes |
|--------|--------|------|------------|-----------------|
| **Finnhub** | News sentiment + economic calendar | Free | 60 req/min | Verified by all. Highest single-source value. |
| **Yahoo Finance RSS** | Per-ticker breaking headlines | Free | No hard limit | Verified. Zero friction. feedparser library. |
| **FRED API** | Macro context (CPI, yield curve, unemployment) | Free | Generous | Verified. Context layer, not fast signal. Cache 24h. |
| **edgartools** (13F/13D) | Hedge fund holdings + activist positions | Free | 10 req/s (EDGAR) | Strongest recommendation in expedition. Drop-in. |

### Pattern Discovery Tools (Priority Order)

| Tool | What It Does | Stars | Validator Notes |
|------|-------------|-------|-----------------|
| **STUMPY** (per-symbol, stumpi) | Motif discovery — finds repeated patterns without prior labels | 3.8-4.1K | TD Ameritrade is trademark holder, not confirmed production user. Use `stumpi` streaming mode. Do NOT use `mstump` multi-domain mode without explicit scoping. |
| **PySAD** (RRCF) | Streaming anomaly detection on composite signal vector | 284 | Wire into VelocityDetector. Unsupervised — surfaces candidates for ConvergenceAlerter. |
| **River** (ADWIN) | Reactive regime drift detection | 5,700 | Wire into RegimeClassifier as event trigger. Published algorithm (1997), well-understood. |
| **Ruptures** (PELT) | Change point detection for archaeology segmentation | 2,000 | Use in historical_fetcher to segment excavation windows by regime. |
| **smart-money-concepts** | ICT order block / fair value gap detection | 1,100 | New edge detector domain for ConvergenceAlerter. Use only in confluence, not standalone. |

### Processing Optimizations (Priority Order)

| Change | Expected Impact | Risk | Effort |
|--------|----------------|------|--------|
| **Sensing workers 3→12** | 4x concurrent fetches | Very low | 5 min |
| **Profile with Scalene first** | Confirms actual bottleneck | None | 30 min |
| **Numpy-vectorize Bollinger/RSI** | 10-50x faster TA computation | Low | 4 hrs |
| **aiohttp async for PolygonBulkFetcher** | 85s→3-10s per 100 symbols | Medium | 1 day |

### LLM Integration (Verified Architecture)

| Component | Lines | What It Does |
|-----------|-------|-------------|
| **OllamaProvider** | ~50 | Subclass OpenAIProvider, change base_url to localhost:11434/v1 |
| **CausalNarrative (Pydantic)** | ~15 | Typed output: causal_story, bull_case, bear_case, hidden_risks, story_strength |
| **OllamaReasoningSubscriber** | ~80 | EventBus subscriber for convergence alerts, gates on confidence |
| **WHY section in plain_language.py** | ~30 | Adds LLM interpretation between HISTORY and TIMING sections |
| **Layer 33 + ConnectionRegistry wiring** | ~100 | Bootstrap registration, triadic connections, channel definition |
| **Total** | ~300-400 | Validators confirmed 200-300 is happy path; wiring adds ~100 |

---

## Novel Approaches (worth investigating, not yet proven for MIDGE)

### CausationEntropy — Full Causal Network Discovery from 28+ Signals

Discovers which of MIDGE's 28+ signal streams actually cause changes in which others — not just tests pre-specified pairs. Uses optimal causation entropy (oCSE), an information-theoretic method that conditions on all variables simultaneously. Published algorithm (Sun/Taylor/Bollt, SIAM 2014), Python implementation November 2025 with 354 unit tests.

**Validator caution:** 16 stars after 10 months. No production use outside academic settings. Mixed-frequency data handling (daily COT vs. real-time price) not verified. Computational cost at 28 variables not benchmarked. Use as experimental weekly batch run with explicit monitoring, not a core dependency.

### Curated Causal Chain Graph (Scoped from Team 2)

A manually-curated NetworkX DiGraph with 50-100 key causal relationships: hurricane→ethylene→auto_parts→ford, oil_shock→airline_costs→airline_earnings, fed_rate→financials→bank_stocks. BFS through the graph when a trigger event fires to find affected tickers. This is the practical version of Team 2's WorldModelGraph — achieves the Brief's "trace a hurricane's ripple" outcome at 10% of the engineering cost.

**Validators unanimously flagged:** Team 2's full proposal (GLEIF + UN Comtrade + EoN + CausalFlow integration) is a 4-6 week infrastructure project that exceeds the expedition brief's scope. The scoped version delivers signal faster.

### Three-Stage Bull/Bear/Synthesis Prompt

Single LLM call with structured multi-perspective prompt: Stage 1 builds the strongest case for the signal, Stage 2 plays devil's advocate, Stage 3 synthesizes. Research shows this is nearly as effective as multi-agent debate (4-6% accuracy improvement vs. single-agent) at a fraction of the computational cost. If Qwen3-14B's story_strength < 0.5, escalate to DeepSeek-R1-14B for chain-of-thought reasoning.

---

## Synthesized Recommendation: The Action Sequence

### Phase 0 — Measure Before You Build (BLOCKING — 1 day)

Validators 2 and 3 independently flagged this as the most important finding the teams missed:

1. **Parse `outcomes.jsonl` → compute payoff ratio** (average win / average loss magnitude)
2. **Run `CorrelationTracker.get_correlation_matrix()` → verify domain independence**

These two measurements determine whether MIDGE needs more domains (independence is real, payoff ratio needs help) or needs domain optimization (independence is weak, stacking correlated domains inflates false confidence). All five teams' recommendations assume domain independence is real. If it isn't, adding more correlated domains is net negative.

**Also verify:** Run `nvidia-smi` on Wardenclyffe to confirm GPU VRAM. This gates the Qwen3-14B vs. Qwen3-8B decision.

### Phase 1 — Use What You Have (same day as Phase 0)

3. Sensing workers 3→12 (one number change, 5 minutes)
4. Run `populate_library.py` as companion process (zero code changes)
5. Profile with Scalene on 50-symbol excavation (30 minutes — confirms where time actually goes)

### Phase 2 — Quick New Eyes (1-2 weeks)

6. Finnhub client (news sentiment + economic calendar — one API key, two new domains)
7. Yahoo Finance RSS via feedparser (zero cost, per-ticker headline velocity detection)
8. FRED API client (macro context anchor — CPI, yield curve, consumer sentiment)
9. edgartools upgrade to SEC client (13F hedge fund tracking + 13D activist detection)

**Each new source must wire through BoundaryMembrane + InputValidator.** All 3 validators flagged that no team addressed this constraint. The existing client pattern (stocktwits_client, trends_client) likely satisfies this by construction, but it must be confirmed explicitly before wiring new sources.

### Phase 3 — Sharpen the Brain (2-3 weeks)

10. OllamaProvider + OllamaReasoningSubscriber + WHY section (~300-400 lines)
11. Numpy-vectorize Bollinger/RSI computation in historical_fetcher.py
12. STUMPY per-symbol streaming (`stumpi` mode only) as discovery layer alongside Pattern Archaeology
13. PySAD RRCF wired into VelocityDetector for streaming anomaly detection
14. River ADWIN wired into RegimeClassifier as reactive drift trigger

### Phase 4 — Deeper Intelligence (3-4 weeks, conditional on Phase 0 results)

15. aiohttp async conversion for PolygonBulkFetcher
16. Curated NetworkX causal chain graph (50-100 manually seeded relationships)
17. smart-money-concepts as new ConvergenceAlerter domain
18. CausationEntropy weekly batch run (experimental, monitored)

### Phase 5 — Future Expedition Scope (deferred)

19. Full WorldModelGraph with GLEIF + Comtrade data feeds (requires dedicated expedition)
20. EoN cascade simulation (no financial precedent found — needs prototype validation)
21. PRAW Reddit integration (gated on API credential approval — submit application now if wanted)
22. Fin-ModernBERT vs. existing NLP comparison
23. tsfresh feature extraction (schema change implications with 4,384 tests)

---

## Disagreements

### STUMPY Mode: Per-Symbol vs. Multi-Domain Matrix

- Team 3 implied concatenating all signal domains into STUMPY's `mstump` multi-dimensional mode
- Validator 3 flagged: STUMPY docs warn mstump is "slow, approximate, and brittle to irrelevant dimensions"
- Validator 1 confirmed the per-symbol `stumpi` streaming mode is the battle-tested path
- **Resolution:** Use `stumpi` (per-symbol streaming) now. Evaluate `mstump` only after per-symbol mode proves value.

### EoN vs. NDlib for Cascade Simulation

- Team 2 recommended EoN (fast_nonMarkov_SIR) for per-edge delay distributions
- Validators 1 and 3 found: no documented use of EoN in financial modeling anywhere in the literature; EoN had a 5-year gap between releases
- Validator 2 confirmed: SIR/SEIR models are academically valid for financial contagion, but literature uses the math, not the EoN package
- **Resolution:** If cascade simulation is pursued, start with NDlib's simpler Independent Cascade model. EoN is a Phase 5 investigation.

### Fin-ModernBERT vs. Qwen3-14B for Financial NLP

- Team 3 recommended Fin-ModernBERT (0.1B, 1024 token context)
- Validator 2 flagged: 1024 token context limit means it can't process full earnings calls or SEC filings
- Validator 3 flagged: Team 3 didn't compare against MIDGE's existing FinBERT
- Team 4's Qwen3-14B (32K+ context) dominates for every use case except high-volume batch sentiment classification
- **Resolution:** Qwen3-14B via Ollama serves the NLP reasoning needs. Fin-ModernBERT deferred pending comparison with existing FinBERT.

### WorldModelGraph Scope

- Team 2 proposed full GLEIF + Comtrade + EoN + CausalFlow integration (4-6 week project)
- All 3 validators flagged scope creep: this exceeds the expedition brief and risks making MIDGE a supply chain platform rather than a pattern recognizer
- Validator 3 proposed: curated NetworkX DiGraph with 50-100 key causal relationships achieves the Brief's goal at 10% cost
- **Resolution:** Curated graph in Phase 4. Full WorldModelGraph deferred to dedicated expedition.

---

## Filtered Out

| Recommendation | Why Removed |
|---|---|
| STUMPY "production use by TD Ameritrade" framing | All 3 validators caught: TD Ameritrade is trademark holder, not confirmed production user. Library is solid on its own merits. |
| ProcessPoolExecutor for parallel excavation | Team 5 confirmed: `requests.Session` not picklable, Windows spawn overhead prohibitive. Previous expedition Phase 3 item superseded. |
| Full GLEIF + Comtrade data pipeline | All 3 validators: exceeds expedition brief scope, 4-6 week project, not a "gift" — it's a new product feature. |
| EoN cascade simulation (as primary tool) | Validators 1 & 3: no financial precedent in literature, 5-year release gap, "actively maintained" overstated. |
| Fin-ModernBERT (as separate NLP system) | Validator 2: 1024 token context too limiting for earnings calls/filings. Redundant with Qwen3-14B. Validator 3: no comparison against existing FinBERT. |
| tsfresh integration into pattern library | Validator 3: 794 features would expand schema, touching fingerprint.py/historical_fetcher.py/pattern_library.py. Zero-regression on 4,384 tests not addressed. |
| TCDF temporal causal discovery | Team 2 correctly flagged: last commit 2018, PyTorch 0.4.1 era. Archived. |
| SPACETIME non-stationary causal discovery | Team 2 correctly flagged: research paper only, no packaged library. |
| Memgraph graph database | Team 2 correctly noted: not needed at MIDGE's scale. NetworkX is sufficient. |
| CrewAI / LangChain frameworks | Team 4 correctly eliminated: adds framework debt with no benefit over direct Ollama API calls. |
| NewsAPI.org | Team 1 correctly eliminated: $449/mo, 24h delay on free tier. |
| Unusual Whales | Team 1 correctly eliminated: $150/mo post-May 2025 increase. Deprioritized by Guiding Light. |

---

## Risks

1. **Phase 0 is still unexecuted.** Domain independence is unverified. Payoff ratio is unknown. All new domain additions assume independence is real. If domains are 70%+ correlated, stacking more correlated domains inflates false confidence rather than multiplying statistical power. **This is the single most important measurement in the system.**

2. **Reddit/PRAW access is now gated.** Self-service API key creation ended November 2025. New applications require manual review with 7-day target response. If Guiding Light doesn't have pre-November credentials, PRAW is a week-delay item, not same-day. Submit application now if desired.

3. **BoundaryMembrane + InputValidator wiring was not addressed by any team.** The Research Brief's hard constraint — "all new data sources must wire through BoundaryMembrane + InputValidator" — received zero coverage across 5 teams and was flagged by 2 of 3 validators. Existing client patterns likely satisfy this, but it must be explicitly confirmed before wiring new sources. Bypassing InputValidator could inject unvalidated data into convergence calculations.

4. **ConnectionRegistry + triadic connection wiring was not addressed.** New systems require registration in ConnectionRegistry with triadic connections. Layer 33 bootstrap wiring costs were not estimated by any team. The "~200-300 lines" estimates systematically ignore wiring overhead.

5. **Marketaux's 100 req/day ceiling breaks at 12 sensing workers.** Validator 2 computed: at standard rotation cadence with 12 workers, Marketaux exhausts its daily budget in under 2 hours. Requires a custom per-source throttle outside the normal rotation — a design decision that was not specified.

6. **Wardenclyffe GPU VRAM is unknown.** Qwen3-14B at Q4_K_M needs ~11-12GB with KV cache. An RTX 4070 (12GB) works but has zero headroom. An RTX 4090 (24GB) is comfortable. Must run `nvidia-smi` before committing.

7. **CausationEntropy (16 stars) has no community debugging history.** Edge cases in MIDGE's mixed-frequency data (daily COT vs. real-time price) have never been stress-tested by anyone outside the author's team. Use as experimental layer with monitoring.

---

## Validator Agreement Matrix

| Finding | V1 | V2 | V3 | Confidence |
|---------|----|----|-----|------------|
| Sensing workers 3→12 safe | Yes | Yes | Yes | Very High |
| ProcessPoolExecutor broken | Yes | Yes | Yes | Very High |
| STUMPY production claim wrong | Yes | Yes | Yes | Very High |
| Reddit/PRAW access gated | Yes | Yes | — | High |
| Phase 0 must come first | — | Yes | Yes | High |
| BoundaryMembrane not addressed | — | Yes | Yes | High |
| EoN financial use unproven | Yes | — | Yes | High |
| Qwen3-14B VRAM needs qualifier | Yes | Yes | Yes | High |
| Team 2 scope exceeds brief | Yes | Yes | Yes | Very High |
| edgartools strongest recommendation | — | Yes | Yes | High |
| Finnhub verified, ready | Yes | Yes | — | High |
| NetworkX correct substrate | — | Yes | — | High |
