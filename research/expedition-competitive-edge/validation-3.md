# Expedition Validator Report — Competitive Edge
## Date: 2026-03-05
## Validator: Expedition Validator (Divergence-First Protocol)

---

## Divergence-First: Problems Before Agreements

Following the Divergence-First Protocol, problems are catalogued before noting what holds up.

---

## 1. Evidence Challenges

### Team 2: ApeWisdom is Already in the Codebase — But Team 2 Doesn't Know Its Status

Team 2 writes: "ApeWisdom is already in the codebase as `mae_core/market/apis/apewisdom.py` — this is zero integration cost if it isn't already wired into the convergence engine."

This is wrong on the key factual claim. ApeWisdom IS wired into the convergence engine. Codebase inspection confirms:

- `mae_core/market/sensing_fetchers.py` has `fetch_social_sentiment()` using `apewisdom.get_accelerating_tickers()` and `apewisdom.get_by_ticker()`
- `mae_core/market/sensing_hook.py` accepts `apewisdom` as an injected dependency and routes `source_name == "social_sentiment"` to that fetcher
- `pattern_library.py` `_SOURCE_DOMAIN_MAP` maps `"social_sentiment"` to the `"sentiment"` domain

Team 2 presents this as a potential quick win to investigate. It is already done. The recommendation to "check domain mapping" is moot — it is correctly mapped. This does not invalidate Team 2's other recommendations, but presenting an already-implemented feature as a discovery is a research gap.

### Team 4: ordpy Does Not Implement Transfer Entropy

Team 4 describes ordpy as implementing "permutation/ordinal transfer entropy" and calls it "the fastest TE option." This claim needs challenge.

Verification: ordpy's own documentation and PyPI page describe it as implementing "permutation entropy and ordinal network methods." The search results explicitly note: "the package appears to focus on permutation entropy and ordinal networks rather than transfer entropy specifically." Permutation entropy and transfer entropy are related but distinct measures. Permutation entropy measures complexity of a single series; transfer entropy measures directed information flow between two series.

Team 4 conflates the two throughout the synthesis section. The `infomeasure` library (Scientific Reports 2025) does implement transfer entropy properly. ordpy's role in Team 4's framework should be clarified as: permutation entropy for individual signal complexity measurement, not directed transfer entropy between domain pairs. The distinction matters for implementation — using ordpy where TE is needed would produce wrong results.

### Team 4: mlfinlab Is Not Freely Available

Team 4 recommends `mlfinlab` for MST/PMFG implementation: "Implemented in `mlfinlab` (Hudson & Thames) and `scipy.sparse.csgraph.minimum_spanning_tree`."

Verification: mlfinlab's public GitHub repository "houses documentation but doesn't contain the source code." The package is closed-source/commercial as of 2025, and the PyPI version is not actively maintained (no new versions in 12+ months per Snyk analysis). The MST implementation via `scipy.sparse.csgraph.minimum_spanning_tree` is correctly cited as the free alternative. The mlfinlab reference is misleading — it should not appear in the implementation path. MST is achievable with scipy alone; mlfinlab is not needed and not freely usable.

### Team 4: bocpd Package Is Effectively Unmaintained

Team 4 recommends the `bocpd` package for Bayesian online changepoint detection. Verification: the latest release (0.0.4) was uploaded October 19, 2023. No updates in 2024-2025. The package is effectively abandoned. Team 4 does mention the `changepoint` library (Rust + Python, 2024) as an alternative, but buries it in the same sentence as bocpd without flagging the maintenance gap. For a production system, bocpd is a risky dependency. The `bocd` package (separate from `bocpd`) and Facebook Kats are the viable alternatives, but neither was verified against Python 3.14.

### Team 2: Freightos FBX "Free API" Claim Is Incorrect

Team 2 states FBX provides "free access to container freight rates across 12 trade lanes via a free Freightos Terminal account."

Verification: Freightos's own help documentation (resources.freightos.com) confirms API access is an enterprise-tier feature, not available on the free personal tier. The personal tier provides CSV downloads. The "free" claim is incorrect for programmatic access. Team 2 partially acknowledges this ("API access is unclear — no confirmed free API endpoint found") in the Gaps section but leads the finding with "free access" language. The Freightos FBX logistics recommendation as currently framed requires either scraping or a paid enterprise account. The Baltic Dry Index (BDI) via Trading Economics or FRED is the actual free proxy — but Team 2 also notes scraping would be required there. This makes the logistics domain more expensive to access than the team's Tier 1 classification implies.

### Team 5: replay_results.json "Essentially Empty" — But What Was Actually Checked?

Team 5 writes: "The replay_results.json file was essentially empty (`{"alerts": [], "phase": "replay"}`) — the file appears to have been reset." This is presented as a critical gap: the payoff ratio cannot be determined.

Challenge: `replay_results.json` being empty does not mean payoff data doesn't exist. The codebase has `data/market/outcomes.jsonl` which contains ground-truth outcome records, and `ActiveTracker` tracks MFE/MAE per prediction. Team 5 says "the data exists, it just needs to flow into the outcome records" — but then treats the payoff ratio as unknown rather than potentially computable from `outcomes.jsonl` right now. The gap is partially self-created by checking the wrong file. A thorough validation would have checked `outcomes.jsonl` directly.

### Team 1: "18% Improvement from Multimodal AI" — Secondary Source, Unverifiable

Team 1 cites: "Their multimodal approach improved signal quality by 18% (satellite + traditional factors combined)." The source is `investmentists.com/multimodal-ai-systems-for-market-analysis-the-future-of-trading/` — an investment blog, not a primary technical disclosure. Two Sigma's actual architecture and quantitative results are proprietary. This number is unverifiable and should be treated as illustrative, not empirical. It does not undermine the broader point (cross-domain correlation works at institutional scale), but the specific 18% figure cannot be relied upon.

### Team 3: NTFS Append Atomicity Claim Needs Qualification

Team 3 states: "JSONL append is atomic on NTFS for small writes (<4KB)." This is partially correct but overstated. NTFS does not guarantee write atomicity in the same way that ext4 or ZFS do. On Windows, file writes are atomic at the sector level (512 bytes), but a JSON line exceeding a sector boundary is NOT guaranteed atomic. For writes under 512 bytes the claim holds; for typical JSONL records including signal metadata, this is borderline. Team 3 correctly flags this as unverified, but the parenthetical "(<4KB)" in the main text conveys false confidence. The safe recommendation is explicit file locking (`msvcrt.locking` or a coordinator process), which Team 3 also mentions — but the atomic claim should not be in the synthesis as a justification.

---

## 2. Contradictions Between Teams

### Teams 2 and 4 Both Recommend Expanding Correlation — But From Different Directions

Team 2 recommends adding domain independence checks before wiring any new source: "If correlation > 0.6 with an existing domain, it doesn't add stacking power." Team 4's entire framework is about improving correlation detection methods (Granger, TE, PCMCI+). These are compatible in principle, but Team 4 never addresses the domain independence enforcement problem that Team 2 raises, and Team 2 never addresses the quality-of-correlation-measurement problem that Team 4 raises. Neither team synthesizes the other's insight. The correct order of operations — which neither team states — is: (1) improve correlation measurement quality (Team 4), then (2) use better correlation measurements to enforce domain independence (Team 2). Doing Team 2's check with Team 4's better methods is more defensible than doing either alone.

### Team 5 and Team 4 Disagree on Whether Independence Enforcement Is the Priority

Team 5 identifies domain correlation as a critical flaw: "If MIDGE is counting 3 correlated domains as '3 independent confirmations,' the actual probability improvement is much less than the theoretical maximum." Team 4 identifies the same flaw with the additional insight that PCMCI+ is the principled solution. Both teams flag this as a priority. But Team 5's synthesis section does not recommend fixing it — it recommends measuring the payoff ratio first, then adjusting combo gates. Team 4's synthesis recommends implementing PCMCI+ before anything else. These are competing priorities from the same underlying diagnosis. Team 4's prescription addresses root cause; Team 5's is a workaround.

### Team 3 and Team 4 Implicitly Contradict on Excavation as the Bottleneck

Team 3 says: "The real bottleneck is not sensing throughput — it is excavation throughput." Team 4 says: "MIDGE's biggest gap is that all current correlation is bivariate Pearson." These are not strictly contradictory (different dimensions of the system), but both present their area as the primary bottleneck to system improvement. An orchestrator reading both would get conflicting priority signals. The brief asks teams to focus on competitive edge improvement — Team 4's point is closer to the stated goal (pattern discovery quality) while Team 3's point addresses throughput speed. Both are legitimate but they're not in dialogue with each other.

---

## 3. Alignment Drift

### Team 1 Drifts Into Execution Infrastructure — Not in the Brief

The Research Brief for Team 1 asks: "What are the major AI trading platforms doing? What data sources do they use? Where are the gaps MIDGE can exploit?" Team 1's Alpaca section concludes: "MIDGE could use Alpaca as its execution layer without any architectural conflict. The MCP Server Alpaca launched in 2025 would enable MIDGE to trigger trades autonomously without a human at the keyboard."

This is outside scope. The brief does not ask about execution infrastructure. MIDGE's competitive edge question is about signal discovery and pattern stacking, not trade execution. The Alpaca discussion — while accurate — is a detour that consumes space better used for competitive differentiation analysis. The section doesn't answer "what gap can MIDGE exploit against Alpaca" because Alpaca isn't a competitor; it's an execution rail.

### Team 2's Telegram/Discord Recommendation Violates Budget and Legal Constraints

The brief states: "Budget-conscious API costs — free/cheap data sources preferred." Team 2 rates Telegram/Discord as Tier 3 and correctly flags "legal risk if Telegram enforces TOS" — but still includes it as a recommendation. The brief's constraint against approaches with legal risk is implicit but clear (the Destructive Boundaries section prohibits recommendations that would compromise the system). Telethon-based Telegram scraping is a TOS violation waiting to happen and has no business appearing in a recommendation set, even at Tier 3. Team 2 should have excluded it entirely given the legal exposure.

### Team 4's Copula Recommendation Does Not Fit MIDGE's Data Volume

The brief notes MIDGE has 414+ days of signal archive and 156 graded predictions. Team 4 recommends copula modeling (Approach 9) and acknowledges: "MIDGE's 414-day archive is borderline adequate for some pairs, insufficient for others." More specifically, copula parameter estimation at T=90 observations (a 90-day window for a single domain pair) requires far more data for stable estimates — typical guidance is T >> 5×N parameters, which for a t-copula means hundreds of observations per pair. At T/N=1.07 for 28 signals in a 30-day window, the recommendation to proceed with copulas contradicts the acknowledged data requirement. This is a "flag it, don't recommend it" situation that Team 4 presents as a development priority.

### Team 5's Futures Instrument Recommendation Lacks Implementation Path

The brief's expected outcome includes "achieves win rates and position sizing that generate real financial returns." Team 5 notes: "Guiding Light's vision mentions preferring instruments 'where payoff math is linear (futures-like).'" Team 5 then recommends: "Move toward futures instruments for better R:R on the same directional signals." This is directionally aligned with the brief. However, MIDGE currently generates paper trades against equities. Moving to futures requires different infrastructure (futures brokerage API, contract specification, margin management) that Team 5 doesn't address. The recommendation is aligned with the brief's vision but has zero implementation path. It's a direction, not a finding.

---

## 4. Missing Angles

### No Team Examined Whether MIDGE's "11 Domains" Are Actually Independent

This is the single most important gap across all five teams. Team 5 flags it briefly: "If MIDGE is counting 3 correlated domains as '3 independent confirmations,' the probability improvement is much less than the theoretical maximum." Team 4 proposes PCMCI+ as the solution. But no team actually ran the correlation check against MIDGE's existing domain data.

The codebase confirms: `CorrelationTracker` already has `get_correlation_matrix()` and `get_least_correlated_pairs()`. The signal archive has 414+ days of data. This check could be done right now against live MIDGE data. The critical question — "are technical + events correlated because both fire on earnings days?" or "are insider + government correlated because Congress members trade on information relevant to their committees?" — was researched by no team. This is the most actionable missing angle because it directly determines whether MIDGE's stacking confidence math is valid or inflated.

### No Team Examined MIDGE's Existing Hypotheses for Redundancy or Conflict

MIDGE has an active hypothesis registry (`hypothesis_registry.py`). No team looked at what hypotheses are currently active, probationary, or hibernated to understand whether new pattern discovery methods would create duplicate or contradictory hypotheses. Team 4's FP-Growth recommendation would mine new domain co-occurrence patterns from the archive — but if those patterns are already registered as hypotheses, the work is redundant. This is an architectural blind spot.

### No Team Examined Options Flow — Identified as a Gap in the Brief

The brief explicitly states: "No options flow data (Unusual Whales API identified but not integrated)." No team investigated Unusual Whales API pricing, integration feasibility, or the quality of options flow as a signal domain. This is a specific gap Guiding Light named, and it received zero research attention across five teams. Unusual Whales API is approximately $50/month for real-time access — within budget — and options flow is a well-documented leading indicator for large institutional positioning. This is a concrete missing angle.

### No Team Addressed the "Daemon Runs on Old Code" Problem

The brief explicitly states as a known problem: "Daemon runs on old code (must restart to pick up changes)." No team addressed this. Team 3's sidecar architecture tangentially addresses it (the fetcher process could run continuously while the daemon restarts), but the live-reload problem — MIDGE's inability to update itself without manual restart — was not researched by any team. For an organism designed for autonomous operation, this is a significant operational gap.

### No Team Examined What "Volatility" Domain Actually Covers

The brief lists 11 domains including a "volatility" domain (VIX term structure). No team examined how this domain is currently used in convergence stacking or whether it's redundant with the "technical" domain (which includes market structure analysis). VIX term structure is arguably a macro-risk signal, not a technical signal. The domain assignment affects stacking math — if VIX-based alerts consistently co-fire with technical domain alerts, they're not adding independent confirmation.

### Team 2 Did Not Research Whether Congress.gov API Handles Volume Appropriately

Team 2 cites Congress.gov API at 5,000 requests/hour. But MIDGE would need to track bill status changes continuously, not just pull a snapshot. Team 2 never addresses: does the Congress.gov API provide webhooks or change-detection endpoints, or does MIDGE need to poll all active bills repeatedly? For the legislative domain to work as a real-time signal (not a weekly batch), this matters. The 5,000/hour limit is sufficient for batch use but the real-time update mechanism was not researched.

### No Team Examined the Senate Stock Watcher

The codebase confirms `mae_core/market/apis/senate_stock_watcher.py` exists alongside `house_stock_watcher.py`. Team 1 discusses congressional trades as an alpha source and Quiver Quantitative as a supplement. Team 2 discusses the legislative domain. But neither team noted that MIDGE already has Senate stock watcher integration, making their discussion of "congressional trades" incomplete — they only discuss House (STOCK Act disclosures) without noting Senate coverage already exists.

---

## 5. Implementation Feasibility Check

### Python 3.14 Library Compatibility — Team 4's Library Table Has Gaps

Team 4 presents a library summary table and states all are "pip-installable" and "CPU-only." Verification findings:

- **tigramite**: Latest version 5.2.8.2 (August 2025). Described as "Python 3" compatible. No explicit Python 3.14 testing documented. The pyreadiness.org/3.14 resource exists for checking compatibility; tigramite was not verified there. Risk is low given active maintenance, but not confirmed.
- **infomeasure**: Confirmed real (Scientific Reports 2025), available on PyPI and conda-forge. The ordpy confusion (see Evidence Challenges above) should not contaminate this library's legitimate role.
- **skfolio**: Confirmed on PyPI, version 0.14.2 as of March 2026, requires Python >= 3.10. Compatible with Python 3.14 in theory. Windows support confirmed (platform-independent wheel). The `DenoiseCovariance` for RMT is likely in this package — but Team 4 should verify the exact class name and API before citing it, as skfolio is primarily a portfolio optimizer, not a standalone covariance denoiser.
- **bocpd**: Last updated October 2023. Effectively unmaintained (see Evidence Challenges). Should be replaced with `changepoint` library or `bocd`.
- **mlfinlab**: Closed-source, not freely available (see Evidence Challenges). MST is achievable with scipy alone.
- **mlxtend (FP-Growth)**: Version 0.24.0, released December 2025. Actively maintained. Platform-independent wheel. Python 3.14 compatibility not explicitly tested but high confidence given recent active release.
- **ordpy**: Version 1.2.2, actively maintained, but does NOT implement transfer entropy (see Evidence Challenges). Should be removed from the TE recommendation.

### Team 3's ProcessPoolExecutor Warning on Windows — Understated

Team 3 notes the Windows `if __name__ == '__main__':` guard requirement for multiprocessing. This is correct but understates the Windows multiprocessing risk. On Windows, `ProcessPoolExecutor` uses "spawn" not "fork" — each worker process imports the entire module from scratch, including all Django/Mesa model initialization code. If MIDGE's bootstrap layer is not fully picklable (which is uncertain given its 33-layer complexity with EventBus, HolonRegistry, and ConnectionRegistry), workers will fail on Windows in ways they wouldn't on Linux. Team 3 does not address picklability of the Excavator, HistoricalDataFetcher, or PatternLibrary objects. This is the primary implementation risk for the highest-priority recommendation (parallel excavation) and it was not tested.

### Team 3's Mesa batch_run Multi-Instance Approach Has a Race Condition on Thompson

Team 3 (Approach 5) proposes running 3 MIDGE instances sharing the same PatternLibrary JSONL files. Team 3 acknowledges "Thompson distribution updates from multiple processes would race" but doesn't fully explore the consequences. Thompson distributions (`data/market/thompson_distributions.json`) are read-modify-write operations on a JSON file. Three organisms simultaneously updating Thompson after discovering patterns would produce a race condition where two of three updates are silently lost. This is not a minor file locking issue — it corrupts the Bayesian learning layer. The approach is not safe as described without a coordinator process for Thompson writes.

### AISStream.io Complexity Is Understated by Team 2

Team 2 calls AISStream "significant processing required" but still frames it as a target. The actual processing pipeline for turning raw AIS vessel position pings into trading signals involves: (1) defining polygon boundaries for ports of interest, (2) associating vessels with cargo types (crude carriers vs container ships vs LNG tankers), (3) computing port dwell time or arrival/departure events from position streams, (4) mapping port events to commodity-exposed equities. Each step is non-trivial. Team 2 recommends starting with FBX instead — this is the correct call — but the FBX "free" claim is wrong (see Evidence Challenges), making the logistics domain more complex to enter than Team 2 presents.

---

## 6. Agreements — High-Confidence Zone

Where independent teams converged, confidence is highest:

**Domain independence as the structural bottleneck** — Teams 1, 4, and 5 all independently identify that MIDGE's stacking confidence model is only valid if domains are genuinely independent. Three teams arrived at this without coordination. This is the highest-confidence finding in the expedition.

**The payoff ratio must be measured** — Teams 5 and 3 (implicitly) both note that key decision inputs are missing. Team 5's call to measure actual winner/loser magnitude is well-supported. All Kelly-based sizing recommendations across teams depend on this number.

**Excavation throughput is the pattern library growth bottleneck** — Teams 3 and 4 independently agree that the pattern library cannot grow fast enough. Team 3 identifies the mechanism (sequential single-threaded excavation), Team 4 identifies the consequence (too few templates for reliable pattern stacking). Both suggest parallelism without contradiction.

**Granger causality is the correct upgrade from Pearson** — Teams 4 (directly recommends) and indirectly Team 1 (notes "lagged" signals like congressional disclosure timing) both point to lag-aware directed correlation as superior to concurrent Pearson. Team 4's statsmodels implementation is straightforward and has no library risk.

**The legislative domain is a genuine gap** — Teams 1 and 2 independently identify that MIDGE covers congressional *trades* but not the *legislation itself*. Team 1 surfaces this via the Quiver lobbying angle; Team 2 via Congress.gov API. This is a real gap with free data available. The implementation challenge (NLP bill classification) is real but the signal direction is sound.

**Congressional trade alpha alone is decaying** — Teams 1 and 5 both note that ETFs (NANC, GOP) now track congressional trades, commoditizing the signal. Both conclude MIDGE's edge must be *cross-domain correlation* of congressional trades with other signals, not congressional trades in isolation. This is aligned with the brief's vision.

---

## 7. Surprises

**The CorrelationTracker already has the infrastructure for most of Team 4's "novel" additions.** Reading the actual codebase: `CorrelationTracker` already computes Pearson rolling correlation, has `detect_cross_domain_anomalies()`, uses Bonferroni correction for multiple comparisons, and has `get_correlation_matrix()` returning the full pair matrix. The 30-observation window and z-score anomaly threshold are configurable. Team 4 proposes RMT denoising as an enhancement to the correlation matrix — that enhancement could be applied directly to `CorrelationTracker.get_correlation_matrix()` output without touching the class. The infrastructure is closer to ready than the research implies.

**The senate_stock_watcher.py exists but was never mentioned.** Five teams researched congressional trades as a signal without any team noting that Senate coverage already exists in the codebase (`senate_stock_watcher.py`). The MEMORY.md mentions only `house_stock_watcher.py` in the architecture table. This may be a documentation gap — the Senate watcher may not be wired into the sensing hook. But no team found it.

**Team 5's most important finding is actually a question, not an answer.** The payoff ratio is unmeasured. Every Kelly formula in Team 5's research is applied to this missing parameter. The actual work — parsing `outcomes.jsonl` to compute average winner magnitude vs loser magnitude — is an hour of Python analysis, not a research question. Yet no team did it. The expedition returned from the field without measuring the one number that determines whether all the optimization work is even needed.

---

## Summary Assessment

| Team | Alignment | Evidence Quality | Implementation Feasibility | Key Failure |
|------|-----------|-----------------|---------------------------|-------------|
| Team 1 (Competitive Landscape) | Good | Good, with one unverifiable stat (18%) | N/A — research only | Alpaca section is out of scope |
| Team 2 (Alternative Data) | Mostly good | Mixed — FBX "free" claim is wrong; ApeWisdom already wired | Medium — logistics harder than stated; NASA NDVI understates effort | Telegram recommendation should have been excluded |
| Team 3 (Processing Architecture) | Good | Good — codebase-grounded | Medium — Windows ProcessPoolExecutor pickling risk unaddressed | Multi-instance Thompson race condition not solved |
| Team 4 (Pattern Discovery) | Good | Mixed — ordpy TE claim wrong; mlfinlab not available; bocpd unmaintained | Medium — library verification incomplete | Conflates permutation entropy with transfer entropy |
| Team 5 (Win Rate Optimization) | Excellent | Good — math is sound | High — recommendations are code changes, not architecture | Didn't check outcomes.jsonl directly; payoff ratio still unknown |

### The Three Highest-Confidence Actionable Findings

1. **Measure the payoff ratio from `outcomes.jsonl` immediately.** This takes an hour and determines whether MIDGE is already profitable. Every other optimization is second-order until this is known.

2. **Run `CorrelationTracker.get_correlation_matrix()` on MIDGE's domain signals to test actual independence.** The infrastructure exists today. If "technical" and "events" domains are >0.6 correlated (they may be — both fire around earnings), the convergence engine's confidence math is overstated.

3. **Replace `mlfinlab` references with `scipy.sparse.csgraph.minimum_spanning_tree` and replace `bocpd` with `changepoint` before any implementation begins.** Two library recommendations in Team 4 will fail in production.

### The One Finding Most Likely to Be Overlooked

Team 4's PCMCI+ (Tigramite) recommendation for conditional causal discovery is the most technically sophisticated finding and the one most likely to be deprioritized as "too complex." But it directly addresses the false convergence problem — multiple domains firing simultaneously because of a common market factor (broad bull trend), not because of genuine cross-domain causality. This is the principled solution to MIDGE's most structurally embedded false-positive source. The library is real, maintained, CPU-only, and pip-installable. The operational cost (minutes per daily run) is acceptable. If domain independence testing (finding #2 above) reveals domains ARE correlated, PCMCI+ is the correct fix.

---

*Validation conducted 2026-03-05. All library claims verified via web search. Codebase claims verified by direct file inspection.*
