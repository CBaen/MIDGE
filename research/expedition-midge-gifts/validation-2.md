# Expedition Validator Report — Midge Gifts
**Date:** 2026-03-05
**Validator Role:** Cross-validation, evidence challenge, alignment check
**Write target:** validation-2.md

---

## Validation Order (Divergence-First Protocol)

1. Evidence Challenges
2. Contradictions
3. Alignment Drift
4. Missing Angles
5. Agreements
6. Surprises

---

## 1. Evidence Challenges — What Doesn't Hold Up

### CRITICAL: PRAW/Reddit API — Team 1 misrepresents access state

**Claim (Team 1):** "60 requests/minute with OAuth (read-only). No cost for non-commercial use. Requires Reddit app registration (client_id + client_secret, free)."

**Reality (verified via WebSearch):** In November 2025, Reddit discontinued self-service API key creation. New developers cannot self-serve credentials. The "Create App" button now routes to a manual ticket-based approval process with a 7-day review window. Applications must describe use case, data needs, target subreddits, and expected volume.

Existing pre-November 2025 OAuth credentials still function. But Team 1's framing — "requires Reddit app registration (free)" — implies this is a straightforward sign-up. It is not. New credential acquisition requires an application and approval decision by Reddit.

**Impact:** PRAW remains valid IF Guiding Light already has credentials from before November 2025. If not, this is a blocked dependency requiring application submission and approval wait. Team 1's Phase 2 placement is appropriate but the access friction is materially understated.

**Rate limit accuracy:** The 60 req/min figure is disputed in sources. Authenticated OAuth can be up to 100 QPM per some sources, with 60 QPM cited as the rolling baseline. The stated figure is conservative and defensible — not wrong, but may underestimate available headroom.

**Verdict:** Team 1's PRAW recommendation is conditionally valid. Gate on credential availability. If credentials don't exist, this is a multi-day approval process, not a same-day setup.

---

### SIGNIFICANT: STUMPY star count inflated in Team 3 findings

**Claim (Team 3):** "Stars: 4,100"

**Reality (verified via WebSearch):** The GitHub repository (TDAmeritrade/stumpy) shows 3.8k stars with 329 forks. The 4,100 figure overstates by approximately 7-8%. Not a fabrication — likely a rounding or staleness artifact — but the citation quality is lower than claimed.

**Production use claim (Team 3):** "Used in production by TD Ameritrade (now Schwab)." The library was open-sourced by TD Ameritrade, and is maintained under their GitHub org. However, active 2025 production use at Schwab post-merger was not confirmed by search results — only that the library originates from TD Ameritrade. The distinction matters: "created by" vs. "currently used by" are different claims.

**Verdict:** Star count is a minor inflation. Production use claim is directionally correct but not current-verified. The library itself is legitimate and the recommendation stands.

---

### SIGNIFICANT: Fin-ModernBERT "0.1B" claim is confirmed but context is missing

**Claim (Team 3):** "Fin-ModernBERT (0.1B financial LM)" — listed without further elaboration in the brief's summary of findings.

**Reality (verified via WebFetch of HuggingFace):** Confirmed 0.1B parameters, BF16, released 2025 by clapAI. Trained on ~20M deduplicated financial records from 8+ sources. Context length: 1024 tokens. Outperforms FinBERT on CIKM (54.89 vs 42.77 F1) and PhraseBank (88.09 vs 86.33 F1).

**What's missing:** The 1024 token context limit is not flagged anywhere in Team 3's report. For earnings call transcripts (which often run 8,000-20,000 words), this is a binding constraint. The model cannot process a full transcript — chunking is required. SEC 10-K filings are similarly out of scope without chunking strategy.

**Verdict:** Model exists, specs are accurate, but the constraint that makes it appropriate only for short text (news headlines, social posts) was not surfaced. For Team 4's integration use case (financial NLP on earnings calls and SEC filings), Qwen3-14B is a better fit due to its 32K+ context window.

---

### SIGNIFICANT: Qwen3-14B VRAM estimates are internally inconsistent across teams

**Claim (Team 4):** "VRAM requirement: ~10-12GB at Q4_K_M"

**Reality (verified via WebSearch):** Multiple sources show Q4_K_M model weights require approximately 7GB. However, inference requires additional VRAM for KV cache (context-dependent). At 4K context: fits in ~9-10GB. At longer contexts (32K): KV cache alone may consume 8-12GB additional VRAM.

The 10-12GB figure from Team 4 matches "base weights + minimal context" — this is accurate for short inference passes (200-token responses). But Team 4 simultaneously recommends using Qwen3-14B for "Hidden Risk Scanner" tasks with injected SEC filings and news context. At those context lengths, VRAM may reach 18-22GB.

Without knowing Wardenclyffe's exact GPU VRAM, this is a real risk. Team 4's hardware check says "safe assumption is 12-24GB VRAM" — which spans two different outcomes. The lower end (12GB RTX 4070 Ti) would not sustain long-context inference.

**Verdict:** VRAM numbers are accurate for short-context use. Long-context use cases (SEC filing analysis) may not fit on a 12GB GPU. Team 4 should have explicitly gated the SEC analysis patterns on GPU VRAM verification.

---

### MINOR: Bluesky "getTrends" endpoints are unspecced — Team 1 appropriately flags this

**Claim (Team 1):** "The `getTrends` and `getTrendingTopics` endpoints are marked `unspecced` — they are undocumented and subject to change without notice."

**Verification:** Bluesky's official rate limits documentation confirms authenticated vs. unauthenticated access tiers. The unspecced endpoint warning is verified as accurate. Team 1 appropriately flags this and recommends Phase 3 placement with graceful degradation. No issue here.

---

### MINOR: EoN library version recency claim should be checked

**Claim (Team 2):** "Version 1.2 released June 2024 — actively maintained."

**Verification attempted:** The EoN docs confirm version 1.2rc1 is the documented version. The "rc1" (release candidate) designation was not surfaced in Team 2's description, which described it as "Version 1.2 released June 2024." An rc1 tag signals pre-release status. Whether a final 1.2 release exists is unclear from available sources.

**Verdict:** The "actively maintained" claim may be slightly optimistic. The library is functional and the JOSS paper is peer-reviewed, but the rc1 designation should be verified before using in production.

---

## 2. Contradictions — Where Teams Disagree

### Contradiction: STUMPY star count (3.8k vs. 4.1k)

Team 3 states 4,100 stars. Verified count is approximately 3,800. Minor discrepancy but confirms star count was not checked at research time.

### Contradiction: CausalFlow positioning — Teams 2 and 3 both recommend causal discovery but from different angles

Team 2 recommends CausalFlow (F-PCMCI, CAnDOIT) for auto-discovering causal chains from MIDGE's historical data as part of WorldModelGraph population. Team 3 recommends CausationEntropy / mlcausality for discovering "which variables cause which other variables automatically from 28+ signal streams."

These are addressing overlapping needs with different tools. There is no direct contradiction, but there is an unmapped dependency: if MIDGE builds both, the outputs would need to be reconciled. Two causal discovery systems running in parallel on overlapping data will produce different graphs. No team proposed an integration architecture for this overlap.

### Contradiction: ProcessPoolExecutor assessment between previous expedition synthesis and Team 5

The previous expedition's synthesis (expedition-competitive-edge) included "ProcessPoolExecutor for parallel excavation" as Phase 3 work, citing "8-10x speedup." Team 5 (this expedition) explicitly debunks ProcessPoolExecutor for MIDGE's current architecture: `requests.Session` is not picklable, Windows spawn overhead is prohibitive, and ThreadPoolExecutor is correct for I/O-bound work.

Team 5's analysis is more rigorous. The previous expedition's recommendation was not tested against pickling constraints. **Team 5 wins this contradiction** — and the previous expedition's Phase 3 item 14 (ProcessPoolExecutor) should be marked superseded.

---

## 3. Alignment Drift — Where Findings Stray from the Brief

### Alignment Check: Does this solve the stated problem?

**Brief asks for:** External input, processing power, and deeper reasoning. The stated outcome is: precursor detection, causal chain tracing, confident timeline predictions.

**Assessment:**
- Team 1 (data feeds): Directly aligned. Adds external input across news, social, macro. No drift.
- Team 2 (causal chain): Directly aligned. Builds the causal chain machinery the brief explicitly requests.
- Team 3 (pattern recognition): Partially aligned. STUMPY, tsfresh, PySAD, and edgartools are on-target. The Fin-ModernBERT recommendation drifts slightly toward NLP infrastructure that Team 4 already covers better.
- Team 4 (LLM reasoning): Directly aligned. The "internal dialogue" and WHY section are exactly what the brief describes as "reason internally about why patterns are forming."
- Team 5 (hardware): Directly aligned. Addresses throughput constraints.

### Alignment Drift: Team 3's Fin-ModernBERT recommendation is superseded by Team 4

Team 3 recommends Fin-ModernBERT (0.1B, 1024 token context) for financial NLP. Team 4 recommends Qwen3-14B (14B, 32K+ context) via Ollama. For MIDGE's use cases — earnings call analysis, SEC filing parsing, convergence narrative generation — Qwen3-14B dominates on every relevant dimension except inference speed. Team 3's recommendation would only add value for batch sentiment classification tasks at very high volume where a tiny model is needed. The brief does not describe that use case.

**Verdict:** Fin-ModernBERT is not wrong, but it is mostly redundant given Team 4's recommendation. The brief asks for deeper reasoning, not faster shallow classification.

### Alignment Drift: Team 2's WorldModelGraph is architecturally ambitious relative to Phases 0-1 not yet executed

The brief states "Previous expedition produced 4-phase action plan — Phases 0-1 not yet executed." Phase 0 was: measure payoff ratio and verify domain independence. Phase 1 was: raise combo sample gate to n≥15, apply combo-specific Kelly, run populate_library.py as companion process, and expand sensing workers 3→12-20.

Team 2's WorldModelGraph proposal (GLEIF + Comtrade + EoN + CausalFlow integration) is Phase 4-equivalent complexity. Building a world-model graph before completing Phase 0 measurements creates a risk: if the payoff ratio measurement reveals MIDGE is already above break-even and domain independence is already sufficient, the causal chain engine may be elaborating a working system rather than fixing a broken one. Phases 0-1 are not prerequisites in a strict sense — they run in parallel — but they are prerequisite to understanding *how urgently* the new capabilities are needed.

**This is the most important alignment finding in this report.**

---

## 4. Missing Angles — What Wasn't Researched

### Missing: BoundaryMembrane + InputValidator wiring was not addressed by any team

The brief's constraint: "All new data sources must wire through BoundaryMembrane + InputValidator."

Team 1 provides a client pattern (XxxClient class with XxxSignal dataclass) that describes the output format, but does not address BoundaryMembrane or InputValidator explicitly. None of the five teams mention BoundaryMembrane or InputValidator by name. The constraint was stated as a hard requirement, and it received zero coverage.

This is the most significant gap in the research findings. Before any new data source is wired, the integration points through BoundaryMembrane and InputValidator must be understood. The existing pattern (stocktwits_client, trends_client) likely satisfies this by construction — but that needs to be stated explicitly, not assumed.

**Risk:** If a new source is wired incorrectly (bypassing InputValidator), it could inject unvalidated data into convergence calculations, corrupting signal quality and violating Mae's Mathematical Laws. This is high blast radius.

### Missing: API rate limit budget at 12 sensing workers is not computed for the combined stack

The brief specifically asks whether combined rate limits support sensing worker scaling to 12. This analysis was not performed.

Let me compute it based on stated limits:

| Source | Rate Limit | Daily Budget |
|--------|-----------|-------------|
| Finnhub | 60 req/min | 86,400 req/day |
| Yahoo Finance RSS | None stated | Effectively unlimited |
| Marketaux | 100 req/day | 100 req/day |
| PRAW | 60-100 req/min OAuth | 86,400+ req/day |
| FRED | No hard limit stated | ~2,000 req/day practical |
| Bluesky | ~30 req/5min unspecced | ~8,640 req/day |
| Polygon.io (existing) | Unlimited (paid) | Unlimited |
| Existing 28 sources | Varies | Varies |

**Critical finding:** At 12 sensing workers, the binding constraint is not Finnhub (86,400/day is ample) or PRAW (similar) — it is **Marketaux at 100 req/day**. At 12 workers cycling through all sources, Marketaux will exhaust its daily quota in under 2 hours if it is included in the rotation at the same frequency as other sources.

Team 1 acknowledges this ("100 req/day is limiting") and proposes targeted batching (15-ticker watchlist = 15 requests, 6×/day = 90 requests). That strategy is incompatible with the sensing hook's current source rotation design, which cycles through all registered sources. Marketaux must be registered with a custom cadence limiter or excluded from the standard rotation — this architectural point was not specified.

**The sensing worker scaling to 12 is rate-limit-safe for all sources EXCEPT Marketaux, which requires a custom throttle outside the normal rotation cadence.**

### Missing: ConnectionRegistry registration requirement not addressed

The brief states: "New systems must register in ConnectionRegistry with triadic connections." None of the five teams addressed this. For a 146-system organism with 425 connections, the ConnectionRegistry integration is not optional — it affects Mae's Mathematical Laws compliance. The research teams should have noted this as a per-item integration requirement.

### Missing: The previous expedition's Phase 0 payoff ratio measurement has still not been done

The HANDOFF.md shows the excavation was running and pattern archaeology is active, but there is no record of the Phase 0 payoff ratio measurement being completed. The synthesis from expedition-competitive-edge said "measuring payoff ratio from outcomes.jsonl is an hour of Python analysis." If this still hasn't been done, the entire Kelly sizing and combo-selection framework in Phase 1 is operating blind.

Before any new data source adds complexity, this measurement is the highest-leverage 1-hour task in the system.

---

## 5. Agreements — High-Confidence Zone

### Agreement 1: NetworkX as world-model graph substrate

Team 2 and the overall architecture are aligned: NetworkX DiGraph with probability/delay edge attributes is the correct foundation. The mapping from CausalReasoningEngine's existing `_links` dict is direct. This is the right choice — pure Python, Windows-native, zero infrastructure, sufficient for MIDGE's scale.

### Agreement 2: No LangChain/CrewAI frameworks

Team 4 explicitly recommends against LangChain and CrewAI. The previous expedition's synthesis did not address this, but the recommendation is consistent with MIDGE's established pattern (custom EventBus, direct API calls, no framework overhead). This is high-confidence correct for MIDGE's architecture.

### Agreement 3: ProcessPoolExecutor is wrong for MIDGE's current architecture

Team 5's debunking of ProcessPoolExecutor (requests.Session not picklable, Windows spawn overhead) is correct and consistent with the earlier expedition-competitive-edge validation note ("Windows ProcessPoolExecutor uses 'spawn' not 'fork' — MIDGE's 33-layer bootstrap objects must be picklable. This needs testing."). Team 5 ran the test and confirmed failure. This is now settled.

### Agreement 4: EoN's non-Markovian SIR is valid for financial cascade modeling — with caveats

The concern about applying epidemiology models to financial supply chains is legitimate, but web research confirms this is an established academic approach. Multiple peer-reviewed papers (2022-2025) apply SIR/SEIR epidemic models to supply chain risk propagation: supply chain network contagion amplifies expected loss, value at risk, and expected shortfall by factors of 4-7x in empirical studies. The mathematical mapping is valid: "infected" = affected by a disruption, propagation probability = supply relationship strength, recovery = inventory buffer/alternative sourcing. EoN is an appropriate tool.

**The caveat Team 2 missed:** SIR models assume a node can only transition susceptible→infected→recovered. In financial supply chains, a company can be affected multiple times (multiple shocks, recovery, new shock). For multi-shock modeling, an SIS (no permanent recovery) or custom model may be more appropriate. This is a nuance, not a blocker.

### Agreement 5: Sensing workers 3→12 is safe and high-leverage

Teams 3 and 5 from the previous expedition plus Team 5 from this expedition all converge: expanding to 12 workers is a one-line change with 4x throughput gain and very low risk. This is the highest leverage/lowest risk change in the entire recommendation set.

### Agreement 6: Qwen3-14B is the right LLM choice given stated hardware

The VRAM calculation (7GB model + KV cache = ~10-12GB for short contexts) fits a 12-16GB+ GPU. Team 4's choice is well-reasoned and the alternatives are correctly ranked. DeepSeek-R1-14B for chain-of-thought is a valid secondary.

---

## 6. Surprises — What Changed My Thinking

### Surprise 1: Reddit API access is now a gated, approval-based process

This is a meaningful infrastructure change from the previous expedition's research. PRAW has been treated as a straightforward dependency in both expeditions. The November 2025 policy change makes it a conditional dependency. If Guiding Light does not already have pre-November 2025 credentials, adding PRAW is a week-delay item, not a same-day one.

### Surprise 2: The rate limit analysis reveals Marketaux is the only binding constraint

The expedition brief specifically asked about rate limit compatibility with 12 sensing workers. The answer is counterintuitive: Finnhub (60/min) and PRAW (60-100/min) are not the problems — they have thousands of requests per day available. The 100 req/day ceiling on Marketaux is the binding constraint. This is not a reason to reject Marketaux, but it means it cannot be plugged into the standard sensing rotation without special handling.

### Surprise 3: BoundaryMembrane constraint received zero coverage across all five teams

A hard constraint stated in the research brief was not addressed by any of the five research teams. This is a systematic gap, not an individual miss. It suggests the teams optimized for "find good libraries" rather than "verify integration path through MIDGE's existing constraints."

### Surprise 4: Phase 0 measurement (payoff ratio) may still be unexecuted

The HANDOFF.md confirms excavation was running but does not confirm Phase 0 completion. This 1-hour measurement is the gating condition for understanding whether the new capabilities are urgent or optional. Its absence means this entire expedition's priority ordering is operating on an unverified assumption about MIDGE's current profitability status.

---

## Summary Scorecard

| Team | Core Claims | Evidence Quality | Alignment | Integration Coverage | Verdict |
|------|------------|-----------------|-----------|---------------------|---------|
| 1 (Zeitgeist) | Verified | Good | Aligned | BoundaryMembrane: missing. PRAW access state: understated. | Conditionally solid. Flag Reddit access. |
| 2 (Causal Chain) | Verified | Good | Aligned but Phase-jumping | ConnectionRegistry: missing. EoN rc1: flag. | Solid architecture, sequencing risk. |
| 3 (Pattern Recognition) | Mostly verified | Star count inflated | Partially aligned | BoundaryMembrane: missing | Fin-ModernBERT redundant with Team 4. |
| 4 (LLM Reasoning) | Verified | Good | Aligned | GPU VRAM gating for long context: missing | Strong. Add VRAM gate for SEC analysis use case. |
| 5 (Hardware) | Verified | Best of five | Aligned | Processess constraint well-documented | Strongest report. Correctly debunks ProcessPoolExecutor. |

---

## Action Flags for Orchestrator

**Stop and check before building:**
1. Verify whether Guiding Light has pre-November 2025 Reddit OAuth credentials. If not, PRAW is a gated item.
2. Execute Phase 0 payoff ratio measurement from `outcomes.jsonl` before proceeding with new complexity. This is the highest-leverage 1-hour task in the system.
3. Clarify how each new source will wire through BoundaryMembrane + InputValidator. The existing client pattern likely satisfies this, but it must be confirmed explicitly.
4. Verify GPU VRAM on Wardenclyffe before committing to long-context SEC analysis with Qwen3-14B.

**Architecture corrections:**
5. Marketaux cannot be added to the standard sensing rotation at 12 workers without a custom per-source cadence limiter (daily budget exhaustion in <2 hours otherwise).
6. Supersede the previous expedition's Phase 3 "ProcessPoolExecutor" recommendation — Team 5 has confirmed it will fail without architectural restructuring.
7. Team 2's WorldModelGraph is Phase 4 complexity, not Phase 2. Do not build it before Phase 0-1 are complete.

**Verified and ready to implement (no flags):**
- Sensing workers 3→12 (one-line change, verified safe)
- Finnhub integration (rate limits verified, library confirmed)
- Yahoo Finance RSS via feedparser (zero cost, verified)
- FRED API (free, confirmed)
- NetworkX as causal graph substrate (correct choice, confirmed)
- Qwen3-14B via OllamaProvider pattern (confirmed, ~200-300 lines)
- STUMPY for motif discovery (library confirmed, appropriate)
- edgartools for SEC filings (active, no API key needed, confirmed)
- EoN for cascade simulation with per-edge delays (valid for financial modeling, literature confirms)
