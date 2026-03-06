# Validation Report: Competitive Edge Expedition
## Date: 2026-03-05
## Validator: Expedition Validator (independent review)

---

## Orientation

Five teams returned findings on MIDGE's competitive position, alternative data sources, processing architecture, pattern discovery methods, and win rate optimization. This report follows the Divergence-First Protocol: evidence challenges, contradictions, and misalignments before agreements. Seven specific claims were verified via WebSearch/WebFetch. The report covers all five teams.

---

## 1. Evidence Challenges

### Team 1 — Competitive Landscape

**Challenge 1.1: The "18% improvement" Two Sigma attribution is unverifiable.**

Team 1 claims "Two Sigma's multimodal approach improved signal quality by 18% (satellite + traditional factors combined)" and cites `investmentists.com`. A direct WebSearch for "Two Sigma multimodal AI satellite imagery 18% improvement signal quality" returned zero results attributing this figure to Two Sigma specifically. The 18% figure appears in an academic paper on semantic segmentation of remote sensing imagery, unrelated to hedge fund signal quality. The cited source (`investmentists.com/multimodal-ai-systems-for-market-analysis-the-future-of-trading/`) is a third-party market analysis blog, not a Two Sigma technical disclosure. Two Sigma does not publicly document its signal quality improvements.

**This claim is unverifiable from public sources and should not be treated as empirical evidence for MIDGE decision-making.** The correct framing is: "institutional sources confirm multimodal approaches combining satellite and traditional data produce improvements" — but the 18% figure attached to Two Sigma specifically is unsupported.

**Challenge 1.2: Team 1 mischaracterizes the JPMorgan-Numerai relationship.**

Team 1 states "JPMorgan invested $500M through Numerai." This is technically imprecise in a way that matters: JPMorgan Asset Management secured up to $500M in *capacity* (an allocation commitment) to be deployed over the next year, announced August 26, 2025. The $500M is a capital allocation limit, not a one-time equity investment or a completed transaction. At the time of the announcement, Numerai had $450M AUM — JPMorgan's commitment would more than double it over time. Team 1's synthesis section repeats this as though JPMorgan made an equity investment in Numerai the company — the $500M is a fund allocation, which is structurally different. The valuation ($500M) is from the November 2025 Series C equity round with university endowments, which is a separate event from the JPMorgan capacity commitment. These are conflated in the findings. Sources confirmed: [Numerai $500M Series C](https://blog.numer.ai/numerai-raises-30m-series-c-at-500m-valuation/) and [JPMorgan allocation](https://blog.numer.ai/jpmorgan-secures-500m-capacity/).

**Challenge 1.3: The "47% outperformance by committee leaders vs rank-and-file" figure exists but is misapplied.**

Team 1 states in its synthesis: "Committee members outperform rank-and-file by 40-50 percentage points annually." This is directionally confirmed by the NBER paper "Captain Gains" (December 2025), which found leaders outperform matched peers by up to 47 percentage points annually. However, the NBER study specifically covers congressional *leadership positions* (Speaker, floor leaders, whips, conference chairs) — a pool of approximately 20 individuals since 1995 — not committee members generally. The finding is real but more narrowly applicable than Team 1 implies. MIDGE's `politician_tracker.py` tracks 437 members; the 47% alpha applies to a much smaller subset of 5-10 leaders at any given time. The "committee member" framing in Team 1's recommendation should be refined to "leadership-position member." Source confirmed: [Fortune coverage of the NBER paper](https://fortune.com/2025/12/07/congress-stock-market-trades-leadership-outperformance-trading-ban-bill-discharge-petition/).

---

### Team 2 — Alternative Data Sources

**Challenge 2.1: The Freightos FBX free API endpoint was not confirmed.**

Team 2 correctly flags this as a gap: "The research confirmed free data is available on the Freightos website but did not confirm whether a structured JSON/REST API endpoint exists for programmatic access." However, Team 2 still places FBX in "Tier 1 — Build first" with confidence, and its synthesis section describes it as "free to access, structured, daily data." Without a confirmed API endpoint, this is aspirational. The team's own Tier 1 ranking implies near-term buildability, but FBX may require HTML scraping (which is fragile and violates many sites' terms of service). Until the API endpoint is confirmed in a 30-minute prototyping session (as Team 2 itself recommends), FBX should be classified as Tier 2 pending confirmation, not Tier 1.

**Challenge 2.2: Wikipedia page view research age.**

Team 2 cites a 2013 Nature Scientific Reports paper as the primary evidence for Wikipedia page views as a trading signal. The paper is 13 years old. Team 2 acknowledges alpha may have decayed but still includes it as a "Novel Approach" with a relatively positive framing. The team admits "No 2024-2025 replication study was found specifically for Wikipedia page views as a standalone trading signal." A 13-year-old signal that has been widely discussed and is listed by Quiver Quantitative as a retail-accessible feature is unlikely to carry meaningful standalone alpha in 2026. The framing should be skeptical by default — it may have value as a witness signal but calling it "novel" is misleading given its age and publicity.

**Challenge 2.3: The ApeWisdom claim about an existing codebase file needs verification.**

Team 2 states "ApeWisdom is already in the codebase as `mae_core/market/apis/apewisdom.py` — this is zero integration cost." This needs direct codebase verification to be actionable. If that file exists, the quick-win is real. If it does not, Team 2 has made a false claim that will mislead the orchestrator. This is a factual claim about the MIDGE codebase that was not cross-referenced with the actual file system.

**Challenge 2.4: Reddit 87% accuracy JPMorgan citation.**

Team 2 claims "Twitter sentiment predicted stock movements up to 6 days in advance with 87% accuracy in a JPMorgan study cited by LuxAlgo (2025)." This is a tertiary citation — LuxAlgo citing a JPMorgan study, not the primary JPMorgan paper. The 87% accuracy figure appears nowhere in any directly accessible JPMorgan research paper in this search. High "accuracy" claims for social sentiment prediction are endemic to this literature and often refer to directional accuracy under very specific conditions that do not generalize. This should not be treated as a credible claim without a primary source citation.

---

### Team 3 — Processing Architecture

**Challenge 3.1: Mesa batch_run is misrepresented as a viable sharding architecture.**

Team 3's Approach 5 (Symbol Universe Sharding) proposes running "3 simultaneous MIDGE organisms using Mesa's `batch_run` with `number_processes=3`." However, the Mesa documentation confirms `batch_run` is designed to run the **same model with different parameters** for parameter sweeping — not for running a single organism sharded across a symbol universe. The Mesa docs state batch_run takes "a model class and a dictionary with model parameters over which to run the model." Using it to run three organisms on different symbol shards is a misapplication of the API — it would instantiate three separate organisms with different parameters, not coordinate a single organism across three symbol shards. Team 3's write-up says "No custom IPC needed — organisms share the same PatternLibrary JSONL files," but batch_run processes run in complete isolation; they don't share object references. This approach requires significant architectural work that Team 3 understates. Source confirmed: [Mesa batchrunner docs](https://mesa.readthedocs.io/latest/_modules/mesa/batchrunner.html).

**Challenge 3.2: The dupoin.com "8 minutes vs 83 minutes" claim is from a low-authority source.**

Team 3 cites dupoin.com for the "10x+ speedup" claim. The article exists (confirmed via search), but dupoin.com is an educational trading academy website — not peer-reviewed literature, not a benchmark from a known quant library. The speedup metric likely refers to tick-level data processing, not TA computation on daily OHLCV histories (which is what MIDGE's excavation does). The claim that ProcessPoolExecutor would yield "8-10x speedup" on MIDGE's excavation task is extrapolation from a different workload, not a verified benchmark for MIDGE's specific operations. The actual speedup will depend on i9 core count (unconfirmed), Polygon API rate limits (unconfirmed), and memory costs of parallel preloading (unconfirmed). Team 3's own gap section correctly flags all three unknowns — the synthesis should reflect this uncertainty rather than presenting "45 minutes to 1 hour" as a reliable estimate.

**Challenge 3.3: NTFS append atomicity claim is overstated.**

Team 3 states "JSONL append is atomic on NTFS for small writes (<4KB)" in justifying that parallel organisms can safely write to shared PatternLibrary files. NTFS does not guarantee atomic appends for concurrent writers from separate processes without explicit file locking — NTFS provides atomicity at the sector level for single-writer operations, but concurrent multi-process appends without locking can produce interleaved writes. Team 3 itself flags this in the Gaps section ("JSONL append safety under concurrent access has not been verified"), but the main synthesis section presents the approach as sound without adequately weighting this risk. On Windows, proper file locking for concurrent writes requires `msvcrt.locking` or a coordinator process.

---

### Team 4 — Pattern Discovery Methods

**Challenge 4.1: The 10^5 elements in under a minute performance claim for infomeasure requires context.**

Team 4 states infomeasure "processes time series of 10^5 elements in under a minute on standard CPU hardware." This is confirmed — the Scientific Reports 2025 paper exists and the library is real ([infomeasure Scientific Reports 2025](https://www.nature.com/articles/s41598-025-14053-5)). However, the benchmark is for a single pair of time series, not the full 28×27/2 = 378 pairwise combinations MIDGE would need. At "under a minute per pair," 378 pairs = potentially hours of computation per analysis run (though probably less given MIDGE's signals are daily, not 10^5 elements long). Team 4 should clarify: MIDGE's signal archive has 414 days × 28 sources — that's 414 observations, not 10^5. At 414 elements per pair, performance will be much faster than the benchmark, which actually makes this more favorable. But the "10^5" figure creates false comparison — MIDGE's time series are short, not long.

**Challenge 4.2: CD-NOTS availability is overstated.**

Team 4 recommends PCMCI+ (Tigramite) as the production-ready option but mentions CD-NOTS "outperforms PCMCI on nonstationary financial data." The team correctly notes CD-NOTS is research-grade. But the framing "consistently outperforms PCMCI" cited from a 2024 paper may not hold for MIDGE's specific use case. More importantly, CD-NOTS is unavailable via PyPI (not installable), making it irrelevant for immediate implementation. Team 4 handles this appropriately in the Gaps section but could be clearer in the synthesis that CD-NOTS is not a viable near-term option.

**Challenge 4.3: PCMCI+ Python version requirement is unverified.**

Team 4 states Tigramite "requires Python ≥3.10 (MIDGE uses Python 3.14)." The PyPI search confirms Tigramite 5.2 exists but the Python version requirements are not explicitly confirmed in the search results. The PyPI page for tigramite 4.2.1.3 was found, suggesting the library has had multiple versions. The "Python ≥3.10" claim may be correct but was not directly verified — this should be confirmed before recommending it as a drop-in.

---

### Team 5 — Win Rate Optimization

**Challenge 5.1: The hybrid Kelly-VIX 23.1% annualized return claim is options-specific and not transferable to MIDGE's use case.**

Team 5 cites the arxiv paper (2508.16598) claiming "The hybrid approach achieved 23.1% annualized returns with 18.5% volatility in 2024." This figure is confirmed — the paper exists and the numbers appear in Table 2. However, the validation via WebFetch reveals a critical limitation that Team 5 understates: **this approach is specifically tailored for put-writing strategies on S&P 500 index options (SPXW contracts).** The methodology depends on characteristics unique to options markets — implied volatility, the VIX index as a direct options market signal, and the specific statistical properties of put-writing (negative skew, volatility risk premium). Team 5 proposes applying this to MIDGE's equity/multi-domain signal positions without acknowledging that the VIX scaling logic was designed for a fundamentally different instrument and strategy type. The performance numbers cannot be used as a benchmark for MIDGE's equity predictions. This is a materially misleading application of a source.

**Challenge 5.2: Kelly calculation at 19.9% win rate is mathematically correct but the conclusion is incomplete.**

Team 5 correctly computes Kelly at 19.9% with 4:1 R:R yields near-zero bet size. However, the report then says "Kelly says bet nothing" without fully exploring the implication: if Kelly recommends near-zero, then MIDGE's current paper trading is over-sizing relative to Kelly even at 4:1 R:R. The QuantConnect research Team 5 cites found only 38.5% of parameter combinations beat benchmark — but Team 5 frames this as "suggesting even this well-tested formula is fragile with noisy estimates," which mischaracterizes it. That finding says: Kelly applied with incorrect parameter estimates is unreliable, which underscores Team 5's correct point that the payoff ratio is the missing input, not that Kelly itself is fragile.

**Challenge 5.3: The ESMA "43,000 trading accounts" study is not directly cited.**

Team 5 references "an ESMA study of 43,000 trading accounts found that traders with win-loss ratios above 2.0 still experienced negative returns in 34% of cases." No source URL is provided for this specific finding. The broader claim (win rate alone doesn't determine profitability) is mathematically self-evident, but the specific ESMA study stat is an uncited statistic. This weakens the evidentiary quality of an otherwise correct observation.

---

## 2. Contradictions Between Teams

### Contradiction 2.1: Domain independence assumption (Teams 2, 4, 5 conflict)

- Team 2 recommends adding new domains (EIA, USDA, logistics) and notes MIDGE should "compute correlation between the new source's signal timeseries and signals in existing domains" before wiring — correct domain independence check recommended.
- Team 4 flags "the false convergence problem" explicitly: when markets trend broadly, all domain signals fire simultaneously not due to cross-domain causality but due to a common market factor (e.g., broad bull market).
- Team 5 raises the same concern: "If MIDGE is counting 3 correlated domains as '3 independent confirmations,' the actual probability improvement from stacking is much less than the theoretical maximum."

**This is convergent — all three teams independently identify domain correlation as the core structural vulnerability.** But Team 2 proposes adding 4 new domains without flagging that MIDGE's *existing* 11 domains have not been verified for independence. Adding more potentially correlated domains before verifying independence of existing ones does not strengthen the stacking architecture. The orchestrator should prioritize domain independence verification before domain expansion.

### Contradiction 2.2: Win rate narrative — Team 5 vs. Teams 1 and 3

- Team 5 explicitly argues 19.9% win rate CAN be profitable at 4:1+ R:R, and that the "most valuable single action is a payoff ratio audit."
- Team 1 frames the win rate issue as needing better domain coverage and independence enforcement.
- Team 3 frames the win rate issue as a throughput problem — more excavation = more templates = higher confidence signals.

These are not contradictory positions but they imply different priorities. Team 5's point that the payoff ratio is unknown is the most operationally urgent — it determines whether any optimization is needed at all. Teams 1 and 3 propose expansions that assume the architecture needs upgrading. If MIDGE's payoff ratio already clears 4:1, the expansion work is still valuable but urgent optimization work is not actually needed. Team 5's audit-first recommendation is the strongest sequencing argument.

### Contradiction 2.3: Mesa batch_run (Team 3 and research brief constraints)

Team 3's Approach 5 proposes Mesa batch_run for sharding. The research brief's constraint states "Mesa 3.4 agent framework — MIDGE is built on Mesa. Core architecture is settled." Using batch_run to run three independent organisms is a significant departure from MIDGE's single-organism architecture. The research brief does not prohibit this, but the intent of "Core architecture is settled" suggests the orchestrator wants evolutionary changes, not architectural rewrites. Team 3's own recommendation correctly deprioritizes Approach 5 in the synthesis — but the fact that it was proposed as "Novel" rather than "Exploratory-only" elevates it beyond its warranted status.

---

## 3. Alignment Drift

### Drift 3.1: Team 2's logistics domain (AIS vessel tracking) has extremely high engineering cost for the stated goal.

The research brief states MIDGE "ingests data from more diverse domains" is goal 1. AISStream is confirmed free and real ([AISStream.io](https://aisstream.io/)). However, the research brief also emphasizes "efficient processing on a single machine" and the constraint that team research should inform "architectural direction decisions." Team 2 correctly notes AIS raw data requires: port polygon logic, commodity vessel type classification, dwell time computation from position pings. This is a 4-8 week engineering task. In the same Tier 1 table, EIA energy data is listed alongside AIS as equal-priority — but EIA is a thin REST wrapper, while AIS is a stream-processing project. The tiering understates the effort differential. The brief's expected outcome asks MIDGE to process data "efficiently on a single machine" — raw AIS may challenge that constraint on both storage and processing dimensions.

### Drift 3.2: Team 4's signal neutralization recommendation partially conflicts with the Mathematical Laws.

Team 4's Approach 12 (Signal Neutralization, Numerai paradigm) recommends: "after computing a new signal/hypothesis, regress it against all existing Thompson distributions and pattern templates. Remove the linear component explained by known signals." This is architecturally sound in principle. However, MIDGE's Mathematical Law 5 (Stem Cell Principle) means that the convergence signal is the *composite* product of domain convergence, not a single signal to be neutralized against others. If "convergence with domains A+B+C" is neutralized against "convergence with domains A+B," the residual is not guaranteed to represent a meaningful signal — it may represent noise. The Numerai approach works because each model is an independent prediction. MIDGE's convergence alerts are not independent predictions — they are structural detections of multi-domain alignment. The neutralization concept requires careful adaptation, not direct lift-and-shift from the Numerai paradigm.

### Drift 3.3: Team 3's asyncio refactor recommendation (Approach 3) violates "no bandaid fixes."

Team 3 correctly recommends against the asyncio refactor in its synthesis. However, the detailed Approach 3 write-up describes it as a viable path and lists it among "Battle-Tested Approaches." The research brief's orientation is pattern-stacking and win rate improvement; a major sensing layer refactor does not advance either goal. Listing asyncio as "battle-tested" elevates it beyond what the brief needs. Team 3's synthesis gets this right by deprioritizing it — but the orchestrator reading quickly may weight the "battle-tested" designation too heavily.

---

## 4. Missing Angles

### Missing 4.1: No team verified the ApeWisdom codebase claim.

Team 2 states "ApeWisdom is already in the codebase as `mae_core/market/apis/apewisdom.py`." This is a codebase fact claim that no team verified against the actual file system. If true, it's an immediate zero-cost win. If false, it's a false lead. The validation can confirm: a file search of `mae_core/market/apis/` should be the first action before any Team 2 recommendation is acted on.

### Missing 4.2: No team measured the actual payoff ratio from outcomes.jsonl.

Team 5 identifies this as the most critical unknown but does not attempt to compute it. The outcomes.jsonl file exists in `data/market/`. Parsing it to compute average winner magnitude vs. average loser magnitude would determine whether MIDGE is structurally profitable at 19.9% win rate. This is a 30-minute analysis task that none of the five teams performed. Every win-rate optimization recommendation in Team 5 is conditional on a number nobody has measured.

### Missing 4.3: No team examined domain correlation within MIDGE's existing data.

Team 4 identifies the false convergence problem (common market factor inflating all domain signals). Team 5 flags domain correlation as a structural unknown. Neither team attempted to analyze MIDGE's existing `lag_correlations.json` (50+ bivariate findings already computed) or `CorrelationTracker` data to characterize actual inter-domain correlation levels. This is directly available from MIDGE's own data store and would answer whether the existing 11-domain stack already has a correlation problem before adding 4 more domains.

### Missing 4.4: No team addressed the daemon restart problem directly.

The research brief lists "Daemon runs on old code (must restart to pick up changes)" as a current failure. Team 3's Approach 4 (sidecar architecture) would partially address this as a side effect, but no team proposed a targeted solution. Hot-reloading, code watches, or a process manager (supervisord, PM2) could solve this without an architectural overhaul. This is an operational pain point that the research treated as background noise.

### Missing 4.5: No team engaged with the options/futures instrument shift.

The research brief states MIDGE should prefer "instruments where payoff math is linear (futures-like)." Team 5 notes this in passing but does not research it: what are the practical barriers to MIDGE generating futures/options signals? What would the same convergence alert targeting MES (Micro E-mini S&P) micro-futures yield in R:R vs. equities? This is a major lever that Team 5 identifies but does not explore.

---

## 5. Agreements — High-Confidence Zone

These points are where multiple teams independently converged, making them the most trustworthy findings:

**Agreement 5.1: The structural gap (cross-domain synthesis) is real.**
Teams 1, 2, 4 independently confirm that no competitor platform — Kensho, QuantConnect, Alpaca, Man Group, Numerai — provides automated cross-domain convergence detection. MIDGE's synthesis layer is a genuine structural differentiator. This is well-supported and the core competitive claim can be considered verified.

**Agreement 5.2: Domain independence is the foundation of the edge — and it is currently unverified.**
Teams 2, 4, and 5 all independently raise that domain independence is the mathematical requirement for the stacking confidence model to be valid. All three flag it as an unresolved empirical question for MIDGE's existing domains. This convergence makes it the most important technical gap to close.

**Agreement 5.3: EIA energy data is the highest-priority new domain.**
Teams 1 and 2 independently identify government energy data as an underexploited alpha source. TenderAlpha's 5.4-7.1% alpha from government contract signals (confirmed via direct URL verification at [TenderAlpha blog](https://www.tenderalpha.com/blog/post/quantitative-analysis/unexpected-government-receivables-tenderalphas-investment-signal)) supports the thesis that publicly available government operational data is underused. EIA free API, confirmed real, is the logical next data source.

**Agreement 5.4: Parallel excavation (ProcessPoolExecutor) is the correct bottleneck fix.**
Team 3's recommendation to use ProcessPoolExecutor for CPU-bound symbol excavation is well-evidenced, architecturally conservative, and addresses the dominant throughput constraint directly. This is not contradicted by any other team. The speedup estimate (9 hours → ~1 hour range) may be optimistic but the direction is sound.

**Agreement 5.5: Combo-specific Kelly sizing using ComboThompson distributions is the correct position sizing fix.**
Team 5's recommendation to apply ComboThompson mean as a continuous multiplier to Kelly sizing is internally consistent with MIDGE's existing architecture. The data already exists in the system; the recommendation is to use it correctly. No other team contradicts this.

**Agreement 5.6: Kensho/S&P AIS acquisition confirms institutional belief in cross-domain shipping signals.**
Team 1's claim that S&P Global acquired ORBCOMM's AIS business in April 2025 (agreement announced April 24, 2025; completed November 10, 2025) is confirmed via primary source: [S&P Global press release](https://investor.spglobal.com/news-releases/news-details/2025/SP-Global-agrees-to-acquire-ORBCOMMs-Automatic-Identification-System-business-strengthening-its-supply-chain-and-maritime-offerings/default.aspx). This validates Team 2's logistics domain thesis — institutional players are betting real money on shipping data.

---

## 6. Surprises

**Surprise 6.1: The hybrid Kelly-VIX paper is options-specific.**

The paper's institutional applicability to MIDGE's equity predictions is much narrower than Team 5 implies. A reader might reasonably conclude that MIDGE should implement VIX-scaled Kelly sizing based on this paper — but the paper is actually about S&P 500 put-writing strategies where VIX is intrinsically linked to option pricing, not an independent signal. The VIX-scaling principle may still be valid for MIDGE's regime filtering, but the cited performance metrics are not transferable.

**Surprise 6.2: The Mesa batch_run architecture has a fundamental design mismatch.**

Team 3's Approach 5 (three MIDGE organisms for sharding) sounds straightforward, but Mesa batch_run is a parameter-sweep tool, not a distributed processing framework. Running three organisms this way would require each organism to somehow know which slice of the symbol universe to monitor — a configuration parameter Mesa doesn't natively expose for this use case. The actual implementation would require significant plumbing that Team 3's write-up does not acknowledge.

**Surprise 6.3: Man Group's AlphaGPT and MIDGE are more different than they appear.**

Team 1 correctly notes that AlphaGPT is the closest institutional analog. But from the Bloomberg coverage and Man Group's own technical writeup, AlphaGPT focuses entirely on *financial* data — it generates, codes, and backtests signals within traditional quantitative equity research. It has no cross-domain synthesis capability. MIDGE's architecture is actually closer to what AlphaGPT would need to become rather than what it is — MIDGE is more differentiated from Man Group than Team 1's framing implies.

**Surprise 6.4: Tigramite (PCMCI+) has unclear Python version requirements.**

Team 4 states Tigramite "requires Python ≥3.10" and MIDGE uses Python 3.14 — presenting this as compatibility confirmed. The search results found Tigramite is available on PyPI but did not confirm the specific Python version floor from the latest version. For a recommendation presented as a "drop-in," the Python compatibility should be verified before presenting it as ready to install.

---

## Summary Assessment by Team

| Team | Strongest Finding | Biggest Weakness | Trust Level |
|------|------------------|-----------------|-------------|
| Team 1 | Structural gap is real and well-documented; Kensho AIS acquisition confirmed | Two Sigma 18% attribution is fabricated; JPMorgan-Numerai relationship mischaracterized | Medium-High |
| Team 2 | EIA and USDA domain recommendations are well-evidenced; domain independence framework is correct | FBX API unconfirmed despite Tier 1 placement; ApeWisdom codebase claim unverified; Wikipedia signal is 13 years old | Medium |
| Team 3 | ProcessPoolExecutor for excavation is well-evidenced and correctly prioritized | Mesa batch_run sharding is architecturally misapplied; NTFS atomicity claim is overstated | Medium-High |
| Team 4 | infomeasure, ordpy, Tigramite all confirmed real and functional; independence enforcement finding is critical | 10^5 element benchmark misrepresents MIDGE's short time series context | High |
| Team 5 | Payoff ratio audit as first action is the correct prioritization; combo-specific Kelly math is internally sound | Hybrid Kelly-VIX paper is options-specific, not equity-applicable; ESMA study uncited | Medium-High |

---

## Claims Verified via WebSearch/WebFetch

| Claim | Verified? | Notes |
|-------|-----------|-------|
| Numerai $500M valuation, $30M Series C, November 2025 | Confirmed | [Source](https://blog.numer.ai/numerai-raises-30m-series-c-at-500m-valuation/) |
| Man Group AlphaGPT, Bloomberg July 2025, "several dozen" signals | Confirmed | [Bloomberg](https://www.bloomberg.com/news/articles/2025-07-10/man-group-says-agentic-ai-is-now-devising-quant-trading-signals) |
| Kensho/S&P Global acquired ORBCOMM AIS, April 2025 announcement | Confirmed | [S&P Global](https://investor.spglobal.com/news-releases/news-details/2025/SP-Global-agrees-to-acquire-ORBCOMMs-Automatic-Identification-System-business-strengthening-its-supply-chain-and-maritime-offerings/default.aspx) |
| Two Sigma 18% signal improvement from multimodal | NOT CONFIRMED | No primary source; figure appears in unrelated remote sensing paper |
| Nature 2023 container port → stock returns in 27/33 countries | Confirmed | [Nature Humanities & Social Sciences](https://www.nature.com/articles/s41599-023-01891-9) |
| infomeasure library, Scientific Reports 2025 | Confirmed | [Scientific Reports](https://www.nature.com/articles/s41598-025-14053-5) |
| Hybrid Kelly-VIX arxiv 2508.16598, 23.1% returns in 2024 | Confirmed but options-specific | Paper is for put-writing on index options, not equity signals |
| TenderAlpha UGR alpha 5.4-7.1% per year | Confirmed | [TenderAlpha blog](https://www.tenderalpha.com/blog/post/quantitative-analysis/unexpected-government-receivables-tenderalphas-investment-signal) |
| Congressional leaders outperform rank-and-file 47% annually | Confirmed but narrowly applicable | NBER paper covers ~20 leadership-position holders, not committee members generally |
| AISStream.io free WebSocket AIS API | Confirmed | [AISStream.io](https://aisstream.io/) |

---

## Top Recommendations for the Orchestrator

In priority order, based on verification quality and alignment with the research brief:

1. **Perform the payoff ratio audit first** (Team 5). Parse `data/market/outcomes.jsonl`. Calculate average win magnitude vs. average loss magnitude. This takes 1-2 hours and determines whether every other optimization is urgent or optional. Nothing else should be prioritized over this.

2. **Verify domain independence before adding new domains** (Teams 2, 4, 5 convergent). Check MIDGE's existing `lag_correlations.json` and `CorrelationTracker` data for inter-domain correlation. If existing domains have >50% correlation, adding 4 more domains does not strengthen the stack.

3. **Add EIA energy supply domain** (Teams 1, 2 convergent, evidence strong). Free, structured, government-maintained, proven market mover. Creates a genuinely independent domain (physical energy supply is not correlated with insider trading or congressional signals).

4. **Implement parallel excavation via ProcessPoolExecutor** (Team 3, well-evidenced). Run `populate_library.py` as a companion process while the daemon runs. Zero architectural change. Immediate throughput improvement.

5. **Verify the ApeWisdom codebase claim** (Team 2). If `mae_core/market/apis/apewisdom.py` exists, check whether it is wired to the convergence engine. If yes, this is a zero-cost win. If the file does not exist, remove this recommendation.

6. **Treat the Two Sigma 18% claim as unverifiable.** Do not use it as evidence for cross-domain synthesis value in any MIDGE documentation or pitch material. The Numerai $500M validation and TenderAlpha's documented alpha are stronger empirical anchors.
