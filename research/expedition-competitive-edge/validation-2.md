# Expedition Validation Report — Competitive Edge
## Date: 2026-03-05
## Validator: Expedition Validator (Second Pass)

---

## Validation Scope

Five research teams returned findings on: competitive landscape (T1), alternative data sources (T2), processing architecture (T3), pattern discovery methods (T4), and win rate optimization (T5). This report follows the Divergence-First Protocol: challenges and contradictions are assessed before agreements.

---

## Section 1: Evidence Challenges

### Team 1 — Competitive Landscape

**Challenge 1.1 — Numerai "$500M valuation" is attributed to JPMorgan investment, but the mechanism is imprecise.**
Team 1 states "JPMorgan invested $500M through Numerai" and "$500M valuation." Verified against primary sources: JPMorgan secured *$500M capacity allocation* (fund AUM commitment), not an equity investment. The *$500M valuation* came separately from the November 2025 $30M Series C led by university endowments. These are two distinct events. Team 1's framing conflates them ("JPMorgan invested $500M through Numerai" on line 73) in a way that implies equity stake. The $30M Series C and the $500M capacity allocation are both verified and real, but the $500M valuation preceded the $500M JPMorgan allocation. This is a minor factual imprecision, not fabrication, but matters for interpretation: Numerai is valued at $500M with $550M AUM — not because JPMorgan invested $500M in equity.

**Challenge 1.2 — Two Sigma's "18% improvement" from satellite data is not attributed to Two Sigma specifically.**
Team 1 claims "Two Sigma's multimodal approach improved signal quality by 18% (satellite + traditional factors combined)." The verification search found the 18% figure is a widely cited industry benchmark for satellite imagery enhancement of earnings estimates generally — it does not appear to be attributable to Two Sigma specifically. The source Team 1 cites is a secondary investment analysis publication (investmentists.com), not a Two Sigma disclosure. This claim is presented with unwarranted specificity. The broader point (cross-domain combination improves signal quality) is directionally correct, but attributing a precise 18% figure to Two Sigma is not supported by the evidence cited.

**Challenge 1.3 — Man Group AlphaGPT's cross-domain scope is asserted more definitively than evidence supports.**
Team 1 states AlphaGPT has "no mention of cross-domain stacking" and is "constrained to single-domain financial signals." The verification found Bloomberg (July 2025) quotes Man Group saying the system addresses "a growing challenge in quantitative investing: the surge of available data and possible market relationships that outstrip human bandwidth" and that "early tests show the system can identify connections that researchers may overlook." The system's "modular design should allow it to expand to other asset classes." This is ambiguous — AlphaGPT may already be exploring cross-domain connections. Team 1 correctly flags this as unknowable from public sources (line 131), but then makes the definitive assertion at line 61 that it focuses "on finding signals within financial data, not across domains." This is overclaiming certainty.

**Challenge 1.4 — Alpaca futures claim is unverified and contradicted.**
Team 1 states "No futures" for Alpaca. Verification: Alpaca's 2025 review (cited by Team 1 itself) documents "multi-leg options." Whether Alpaca offers futures directly or requires a futures-specific broker add-on is not confirmed by the search. However, the critical gap the research brief identifies is options flow (Unusual Whales) — Team 1 does not specifically address whether Alpaca's options data would help close this gap for MIDGE. This is an alignment miss more than an evidence challenge.

---

### Team 2 — Alternative Data Sources

**Challenge 2.1 — ApeWisdom "already in the codebase" claim is correct but requires caveat.**
Team 2 states at line 36: "ApeWisdom is already in the codebase as `mae_core/market/apis/apewisdom.py` — this is zero integration cost if it isn't already wired into the convergence engine." The file is confirmed to exist at `C:\Users\baenb\projects\MIDGE\mae_core\market\apis\apewisdom.py`. However, Team 2 does not confirm whether it is wired into the convergence engine. If it already is, the "zero integration cost quick win" has already been captured. Team 2 raises this as a question but frames it as a recommendation without checking the answer. This is the team's own stated gap (line 222): "Check whether it's wired to the convergence alerter and domain-mapped." A validator cannot confirm this without code inspection, but the recommendation to "check" is weaker than the framing implies.

**Challenge 2.2 — Freightos Baltic Index "free API" claim is contradicted by verified evidence.**
Team 2 states FBX is "Free" in the Tier 1 table and describes it as accessible via a "free Freightos Terminal account." Direct verification found that FBX API access is part of an enterprise package — only CSV downloads are available on the personal/free tier. Programmatic API access requires an enterprise subscription. The Freightos documentation page explicitly asks "Is there API access to the Freightos Baltic Index?" and the answer references enterprise pricing. Team 2 acknowledges at line 163 that "API access may require a Freightos Terminal account (free registration) or may be rate-limited. No confirmed free API endpoint found." This contradicts listing it as "Free" in the table and calling it a Tier 1 implementation. FBX should be in the "Evaluate Carefully" tier, not Tier 1, or scraping from Trading Economics BDI as a proxy should be listed as the actual free path.

**Challenge 2.3 — Wikipedia page view alpha decay is a real concern inadequately addressed.**
Team 2 notes the original research is from 2013 and "no 2024-2025 replication study was found specifically for Wikipedia page views as a standalone trading signal." Team 2 then recommends it as a Tier 2 enhancement. This is logically inconsistent: if standalone alpha has decayed since 2013 and no replication evidence exists, recommending it as an actionable Tier 2 addition requires more justification. The 2024 J.P. Morgan alternative data survey reference is via LuxAlgo (a secondary source) and does not confirm Wikipedia's alpha specifically. This source chain is too weak to anchor a recommendation.

**Challenge 2.4 — The AISStream.io "beta" status understated.**
Team 2 presents AISStream.io as a free, confirmed alternative. Verification found AISStream.io is explicitly "currently in BETA." Beta status for a data source MIDGE would depend on is a material reliability risk Team 2 does not surface adequately. Raw AIS data processing complexity is acknowledged, but the beta risk is not.

---

### Team 3 — Processing Architecture

**Challenge 3.1 — Polygon.io Starter plan rate limits are now documented as "unlimited."**
Team 3 flags at line 123: "Polygon API rate limits under parallel symbol fetching... If the limit is 10 req/s per plan, 16 parallel workers will trigger throttling." Verification found: the Polygon.io Starter plan ($29/month) offers *unlimited API calls*. This eliminates the rate-limit constraint Team 3 treats as a potentially blocking unknown. The excavation parallelism recommendation is strengthened, not weakened, by this finding. Team 3's caution was appropriate uncertainty — but the answer is now available and resolves in favor of the recommendation.

**Challenge 3.2 — Mesa `batch_run` multiprocessing on Windows has documented issues.**
Team 3 recommends running 3 simultaneous MIDGE organisms via Mesa's `batch_run(number_processes=3)` (Approach 5). Verification found a GitHub issue (#952) explicitly titled "Runtime Error when using Multi-Processing using batch runner on Windows machine." Mesa's own docs note multiprocessing is "not recommended in Jupyter notebooks" (MIDGE runs `.py` files, so this is partially OK), but Windows-specific issues with Python's `multiprocessing` module on `batch_run` are documented. Team 3 does not mention Windows-specific risks for this approach. The `if __name__ == '__main__':` guard is mentioned, but the Windows HANDLE vs POSIX fd issue from CLAUDE.md is directly relevant here. This warrants explicit flagging before recommending Approach 5.

**Challenge 3.3 — "Companion process" recommendation has a documented current analog that Team 3 misses.**
Team 3 recommends at line 153 that "running excavation as a separate process while the daemon runs" requires "zero code changes." This is partially correct — but the research brief identifies "daemon runs on old code (must restart to pick up changes)" as a current pain point. Team 3 does not address this constraint at all. A companion excavation process running `populate_library.py` in parallel doesn't solve the daemon code-staleness problem, and combining these two recommendations requires clarifying that the companion process gets the benefits of fresh code while the daemon still needs a restart mechanism.

---

### Team 4 — Pattern Discovery Methods

**Challenge 4.1 — PCMCI+ Python 3.14 compatibility is an unverified risk.**
Team 4 states Tigramite works with "Python ≥3.10 (MIDGE uses Python 3.14)." Verification found tigramite's PyPI page does not explicitly list Python 3.14 as a tested version. A critical note: tigramite has "incompatibility issues between numba and numpy" where "soft dependencies on the versions are currently enforced." MIDGE is running Python 3.14, which is a very recent release. numba's compatibility with Python 3.14 is not confirmed in the search results. If tigramite's numba dependency cannot be satisfied on Python 3.14, the entire PCMCI+ recommendation falls. Team 4 does not flag this as a risk.

**Challenge 4.2 — The T/N ratio problem for RMT denoising is flagged but the proposed remedy is wrong.**
Team 4 states the T/N ratio for MIDGE with 30-day window and 28 sources is T/N ≈ 1.07 — "at the marginal edge" — and suggests using a "90+ day" window instead. This remedy is correct directionally but Team 4 does not acknowledge that using a 90-day window for RMT denoising would reduce the sensitivity to recent regime changes. The MIDGE signal archive has 414+ days, but many source pairs may have substantially fewer observations where both sources were active simultaneously. The effective T/N for sparse source pairs could be far lower than 90/28. This limitation should gate the RMT recommendation more strongly than Team 4 does.

**Challenge 4.3 — "Ordinal TE runs in milliseconds" for 756 pairs requires qualification.**
Team 4 cites the `infomeasure` paper: "time series of 10^5 elements can be analysed in less than a minute in a standard computer." But this is for a *single pair* of 100,000-element time series. MIDGE's signal pairs are daily data over 90-day windows = 90 data points per pair, with 28 sources = 756 pairs. The claim is directionally correct (milliseconds per pair at MIDGE's actual data volume) but citing the 10^5 benchmark to support it is slightly misleading — it proves the library can handle large datasets, not specifically that it's fast at MIDGE's 90-element scale (which it almost certainly is, but that's a different claim than what's cited).

---

### Team 5 — Win Rate Optimization

**Challenge 5.1 — Kelly calculation at the "near zero" point contains a mathematical error.**
Team 5 computes at line 43: "At 19.9% win rate with 4:1 payoff: f* = (4 × 0.199 - 0.801) / 4 = (0.796 - 0.801) / 4 ≈ near zero." The Kelly numerator is (bp - q) where b = payoff ratio, p = win probability, q = 1-p. At b=4, p=0.199: f* = (4 × 0.199 - 0.801) / 4 = (0.796 - 0.801) / 4 = -0.005/4 = -0.00125. This is technically *negative* Kelly (don't trade), not "near zero." The difference matters: Team 5 uses this to argue that "Kelly says bet nothing," but the actual implication is that at exactly 4:1 R:R with 19.9% WR, Kelly prescribes *not trading at all* — a slightly stronger statement than "bet nothing" which implies optionality. This is a nuance, not a gross error, but it affects the framing.

**Challenge 5.2 — The "replay_results.json was essentially empty" finding needs verification.**
Team 5 states at line 210: "The replay_results.json file was essentially empty ({"alerts": [], "phase": "replay"})." This is a critical finding — without payoff ratio data, all Kelly calculations are blind. But if this file was just recently reset (as suggested), the historical payoff data should exist in `outcomes.jsonl`. Team 5 recommends pulling from `outcomes.jsonl` (line 213) but doesn't report whether they actually inspected it. The validation cannot confirm whether payoff data is genuinely missing from the system or merely in a different file. This is the single most critical unknown in the entire expedition.

**Challenge 5.3 — The Hybrid Kelly-VIX arxiv paper citation is suspicious.**
Team 5 cites "arXiv:2508.16598v1" from a "2024 arxiv study" for the hybrid Kelly-VIX result showing 23.1% annualized returns. Paper ID 2508 would indicate the paper was submitted in *August 2025* (YYMM format: 25=2025, 08=August). If this paper was submitted August 2025 and Team 5 describes it as a "2024 arxiv study," either the date attribution is wrong or the paper ID format means something different. The specific performance numbers (23.1% returns, 18.5% volatility, 11% max drawdown) are suspiciously precise for an options-strategy paper being applied to MIDGE's equity multi-domain setup. The options-strategy caveat is acknowledged (line 89) but the precision of the numbers lends false confidence to what is a loosely analogous result.

---

## Section 2: Contradictions Between Teams

### Contradiction A — "FBX is free" (Team 2) vs. architectural caution about API costs (Research Brief)

Team 2 places FBX in Tier 1 as "Free" while the actual verification shows FBX API access requires enterprise subscription. The Research Brief's constraint is "Budget-conscious API costs — free/cheap data sources preferred." If FBX requires scraping as Team 2's own Gaps section acknowledges, this introduces a fragile data dependency the Brief explicitly discourages. Teams 3 and 4 have no architectural accommodation for scraping-based data ingestion (no retry, no HTML parsing layer). This is a multi-team gap: Team 2 proposes it, Team 3 doesn't factor its complexity, and no team addresses the maintenance burden.

### Contradiction B — Team 3 says "excavation is the bottleneck"; Team 4 recommends adding more compute-heavy analytics on top

Team 3's synthesis states explicitly: "The real bottleneck is not sensing throughput — it is excavation throughput." Team 4 then recommends PCMCI+ (minutes to tens of minutes per run per line 82), rolling Granger (computationally heavier per line 38), BOCPD (lightweight per line 104), RMT denoising (seconds per line 196), and FP-Growth on the full archive (seconds per line 196). If excavation is already consuming the machine's CPU budget, and PCMCI+ cadenced runs take "minutes to tens of minutes," these compete for CPU time. Team 3 does not flag Team 4's analytics as additional compute load. Team 4 asserts CPU feasibility (line 196) without accounting for the parallel excavation workload Team 3 recommends. Concurrently running PCMCI+ and parallel excavation with ProcessPoolExecutor on the same i9 needs explicit core budget planning that neither team does.

**Stronger evidence:** Team 3's bottleneck diagnosis is more grounded — it has specific measured numbers (9+ hours, 85s/100 symbols, measured ThreadPoolExecutor config). Team 4's CPU feasibility claims are theoretical.

### Contradiction C — Team 1 says congressional trade alpha is being "commoditized"; Team 5 doesn't factor this into win rate projections

Team 1 states at line 136: "Now that ETFs like NANC and GOP directly track congressional trades, and Quiver makes the data widely available, the alpha from congressional trades alone may be decaying." Team 5's analysis of combo win rates includes combos with insider signals (correlated with congressional trades in the institutional domain) but does not incorporate the alpha decay risk. If the best-performing combos (events+macro+price at 31.2% WR, n=32) don't heavily rely on congressional/insider signals, this may not matter. But if MIDGE's edge is partially built on congressional signal alpha that is now decaying, the forward-looking win rate estimates are optimistic.

**Stronger evidence:** Team 1's alpha decay point is documented (NANC/GOP ETFs tracking congressional trades is verifiable, Quiver's widespread availability is confirmed). Team 5 uses historical replay data from Feb 2026 which may not yet reflect the decay.

### Contradiction D — Team 2 says "four new domains achievable for free or near-free"; Team 4's methods require those same domains to have sufficient time-series history

Team 4's PCMCI+, transfer entropy, and RMT denoising all require adequate time series observations. Team 4 notes at line 155: "some source pairs may have substantially fewer observations where both sources were active simultaneously." If MIDGE adds four new domains tomorrow (energy_supply, agriculture, logistics, legislative), those domains will have zero historical observations in the signal archive. Team 4's methods cannot be applied to new domains for months. Team 2 presents new domains as immediately value-additive. Team 4's statistical methods reveal that new domains require a data accumulation period before producing statistically meaningful correlation findings.

**Stronger evidence:** Both are correct in their own timeframes. Neither team explicitly reconciles the timeline mismatch. Team 4's statistical requirements implicitly mean new domains proposed by Team 2 would have a 90-180 day ramp-up period before Team 4's methods can operate on them.

---

## Section 3: Alignment Drift

### Drift 3.1 — Team 3's Approach 5 (Mesa multi-instance sharding) conflicts with a Destructive Boundary

The Research Brief's Destructive Boundaries state: "Do NOT suggest replacing Mesa agent framework" and "Do NOT suggest cloud migration or distributed computing." Team 3's Approach 5 — running 3 simultaneous MIDGE organisms via `batch_run(number_processes=3)` — is not technically distributed computing, but it runs multiple Mesa model instances simultaneously which approaches the spirit of the "distributed computing" boundary. More critically, Team 3 notes "Thompson distribution updates from multiple processes would race" and "template writes to the shared PatternLibrary JSONL could corrupt on concurrent appends." These are architectural integrity risks that directly affect the organism's Bayesian learning state — a core system. Team 3's own synthesis correctly deprioritizes this (line 147: "Do not recommend... Mesa multi-instance sharding"), but the approach is still presented in the findings which risks being actioned.

### Drift 3.2 — Team 4's "Signal Neutralization" approach goes beyond the research scope

The Research Brief asks how MIDGE can *discover* non-obvious multi-domain patterns. Team 4's Signal Neutralization approach (Approach 12) answers a different question: how to ensure newly discovered patterns are orthogonal to existing ones. This is a hypothesis management problem, not a discovery problem. The Numerai Signals platform framing is intellectually interesting but the Brief explicitly states "Pattern stacking is the goal — not individual signal accuracy." Signal neutralization is a refinement of existing signals, not a new discovery method. It is the most misaligned recommendation in the pattern discovery findings relative to the Brief's stated scope.

### Drift 3.3 — Team 5 repeatedly recommends waiting for more data instead of optimizing

The Research Brief's expected outcome is clear: "Achieves win rates and position sizing that generate real financial returns." Team 5's Tier 1 recommendation is "Continue current paper trading at current gates while collecting more outcome data." The "wait" recommendation is intellectually honest but drifts from the expected outcome. The Brief is not asking "when will we have enough data to optimize?" — it is asking "what separates 20% from 50-60% win rates?" Team 5 partially answers this but uses substantial space on data insufficiency caveats that the Brief's question doesn't require. The key insight (fix payoff ratio measurement, apply combo-specific Kelly) is buried in Synthesis.

### Drift 3.4 — Team 2 proposes Telegram/Discord sentiment knowing it violates a constraint

The Brief's constraint states: "Budget-conscious API costs — free/cheap data sources preferred." Telegram scraping also has an explicit legal risk Team 2 acknowledges: "TOS gray area... needs legal review before implementation." Yet the team still catalogs it as finding #10. A finding with acknowledged legal risk and no clear free-access path should not be in the findings at all — it creates implementation risk if actioned without the legal review. Team 2's own Tier 3 note says to "deprioritize" Telegram, but surfacing a legally questionable approach in a research document creates forward risk.

---

## Section 4: Missing Angles

### Missing Angle A — None of the teams calculated MIDGE's actual current payoff ratio

Team 5 correctly identifies the payoff ratio as the most critical unknown but does not actually compute it from the existing `outcomes.jsonl` data. This is the single highest-value action in the entire expedition and it was *not taken*. Team 5 recommends it be done — but could have done it. The `outcomes.jsonl` file is specified in the CLAUDE.md as existing and containing graded outcomes. A validator cannot access this file, but the fact that five research teams produced extensive findings without calculating whether MIDGE is already profitable is a significant gap. If the payoff ratio is above 4:1, significant portions of Team 5's recommendations (chasing win rate improvement) are moot.

### Missing Angle B — No team assessed domain correlation in MIDGE's existing 11 domains

Teams 1, 4, and 5 all note that MIDGE's stacking confidence model (`1 - (1-a)(1-b)(1-c)`) assumes domain independence. Team 5 notes at line 234: "If MIDGE is counting 3 correlated domains as '3 independent confirmations,' the actual probability improvement from stacking is much less than the theoretical maximum." But none of the five teams computed the actual cross-domain correlation from MIDGE's `lag_correlations.json` or `CorrelationTracker` output. The 30+ bivariate findings in `lag_correlations.json` are documented in Team 4's contextual grounding. If "technical" and "price" domains (both derived from price history) are 70%+ correlated, the effective domain count in a "5-domain convergence" might be 2-3 independent signals, not 5. This would explain the confidence gap (winners 0.560, losers 0.565 from the replay data) — confidence does not discriminate because it overcounts correlated domains.

### Missing Angle C — No team addressed the "daemon runs on old code" problem

The Research Brief explicitly identifies this as a current known problem: "Daemon runs on old code (must restart to pick up changes)." Not one of the five teams addresses this. Team 3 mentions it only in passing at line 68 when describing sidecar architecture benefits. A system running autonomously 24/7 that requires a manual restart to pick up code changes is a fundamental operational limitation. This was in the Brief and was ignored.

### Missing Angle D — No team assessed whether MIDGE's outcome grading is correctly calibrated

Team 5 identifies that `outcomes.jsonl` grading uses a 5% minimum move threshold. But none of the teams assessed whether the 14-30 day outcome windows in MIDGE match the actual lead times of the signals generating convergence alerts. Team 1 notes at line 194 that "Congressional trade signals may have 7-45 day disclosure lag. Government contract signals may have 30-90 day lead time before stock reaction." If MIDGE grades outcomes at 14 days for signals with 30-90 day lead times, the system is systematically mislabeling winning signals as losses. This would directly depress the measured win rate below the true win rate.

### Missing Angle E — No team addressed the interaction between new domains (Team 2) and the triadic architecture (Mathematical Laws)

The Research Brief states "all changes must respect these laws — particularly Law 1 (no bare dyads), Law 2 (triadic generator)." Adding four new domains (energy_supply, agriculture, logistics, legislative) means adding new signal sources that must be integrated into the ConnectionRegistry with triadic witnesses, wired into the HolonRegistry, and bootstrapped within the 33-layer structure. Team 2 does not acknowledge any of this architectural integration cost. The "low effort" ratings in the Tier 1 table (EIA: "Low," USDA: "Medium") do not account for the Mae Mathematical Law compliance overhead that all MIDGE additions require.

---

## Section 5: Agreements (High-Confidence Convergent Findings)

Teams 1, 2, 4, and 5 all independently arrive at the same structural insight: **MIDGE's edge comes from domain independence and stacking, not from individual signal accuracy**. This convergence from four independent perspectives is strong validation that the expedition understood the problem correctly.

Teams 1 and 5 independently identify that **congressional trade alpha is decaying as a standalone signal** (T1: commoditization by NANC/GOP ETFs; T5: mentions it implicitly via combo filtering). The implication — MIDGE's edge must be the correlation layer, not raw congressional data — is agreed upon.

Teams 3 and 4 independently identify that **excavation parallelism is the highest-leverage architectural change**. Team 3 recommends it directly; Team 4's analytics recommendations implicitly require a larger, faster-growing template library to function. Both are working toward the same bottleneck.

Teams 1 and 2 independently validate that **government/legislative data is underused relative to its alpha potential**. Team 1 cites TenderAlpha's 5.4-7.1% annual alpha from government contract signals; Team 2 identifies four new government-related domains (energy_supply via EIA, legislative via Congress.gov, agriculture via USDA). Both are pointing at the same opportunity.

Teams 4 and 5 independently identify that **the independence assumption in MIDGE's convergence model may be violated**. T4 in the false convergence problem (line 193-194); T5 in the domain correlation analysis gap (line 234-238). This is a structural problem with the confidence calculation that both teams surface from different angles.

---

## Section 6: Surprises

**Surprise 1 — Polygon.io Starter has unlimited API calls.** Team 3 treats the Polygon rate limit as an unknown blocking factor for parallel excavation. It is not. The Starter plan is unlimited. This removes the most significant constraint against scaling excavation to 16 parallel workers.

**Surprise 2 — ApeWisdom is confirmed alive and free in 2026.** The search results confirm ApeWisdom's API is functioning, free, unauthenticated, and actively maintained. Team 2's recommendation to verify it is confirmed valid — and it is already in the MIDGE codebase. If it is not wired to the convergence engine, this is a zero-cost win requiring one connection in the bootstrap.

**Surprise 3 — The "18% improvement" figure attributed to Two Sigma is an industry benchmark, not a Two Sigma disclosure.** Team 1 uses this number with institutional authority. Its actual origin is the alternative data industry broadly. This weakens the evidentiary weight of the cross-domain stacking validation, though the directional claim (cross-domain combination improves signal quality) is multiply confirmed from other sources.

**Surprise 4 — FBX has no free programmatic API.** Team 2 lists it as Tier 1 "Free." It is not. The Freightos help center explicitly documents that API access is enterprise-only. This changes the logistics domain entry point: the Baltic Dry Index via Trading Economics (scraping) or simply BDI as proxy (freely available on multiple financial data sites) should replace FBX as the Tier 1 logistics entry.

**Surprise 5 — Man Group's AlphaGPT may already be exploring cross-domain connections.** Bloomberg's reporting includes language about "connections that researchers may overlook" and addressing "the surge of available data and possible market relationships." This is more ambiguous than Team 1 presents. Man Group is the closest institutional competitor, and their roadmap ("modular design should allow expansion to other asset classes") deserves closer monitoring than Team 1's confident dismissal.

---

## Overall Assessment

**Strongest findings (high evidence, directly aligned):**
- Team 3's Priority 1 (parallel excavation via ProcessPoolExecutor): grounded in measured codebase data, excavation timing confirmed, Polygon rate limit now confirmed unlimited. Highest-leverage, lowest-risk change.
- Team 5's payoff ratio audit recommendation: mathematically necessary before any Kelly optimization. The finding that `replay_results.json` appears reset is alarming and should be confirmed against `outcomes.jsonl` immediately.
- Team 1's structural gap identification: no competitor combines Bayesian learning, multi-domain convergence, and pattern archaeology in one system. This is the real differentiator and is well-evidenced.
- Team 4's ordinal transfer entropy recommendation (ordpy): confirmed library exists, is fast, CPU-only, and is a genuine upgrade over bivariate Pearson. The `infomeasure` paper's performance claims are verified.

**Weakest findings (evidence gaps or alignment drift):**
- Team 2's FBX as Tier 1 "free" logistics source: contradicted by verified evidence. Needs revision.
- Team 4's PCMCI+ on Python 3.14: numba compatibility risk not addressed. Flag before implementing.
- Team 3's Approach 5 (Mesa multi-instance sharding): Windows multiprocessing issues documented, shared-state race conditions acknowledged by Team 3 itself. Do not action.
- Team 2's Telegram/Discord sentiment: legal risk, no clear free path, team's own synthesis deprioritizes it.
- Team 1's attribution of "18% improvement" specifically to Two Sigma: not verifiable from primary sources.

**The critical question none of the teams answered:**
What is MIDGE's actual payoff ratio from the existing `outcomes.jsonl` data? This one number determines whether MIDGE is already profitable (if above 4:1) or structurally requires a different instrument class (if below 3:1). All five teams produced research around this question without computing the answer. It should be the first action after this expedition closes.

---

*Validation conducted 2026-03-05. Web verification performed on key claims per Evidence Testing protocol. All source checks noted inline.*
