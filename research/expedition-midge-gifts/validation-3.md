# Validation Report 3 — Expedition: Gifts for Midge

**Date:** 2026-03-05
**Validator Role:** Expedition Validator — Divergence-First Protocol
**Write target:** C:\Users\baenb\projects\MIDGE\research\expedition-midge-gifts\validation-3.md

---

## SECTION 1: EVIDENCE CHALLENGES

### Challenge 1 — STUMPY at 3,237-symbol scale: The computational framing is wrong

Team 3 presents STUMPY as "radar for patterns Midge hasn't been trained to see" and claims moderate implementation complexity. This framing conceals a serious scaling problem.

**What the evidence actually shows:**

STUMPY's matrix profile is O(n²) in time complexity where n is the length of a single time series. The spatial complexity is O(n) — that part is fine. The time complexity is the issue.

For a single symbol with 2000 daily bars at window length 20, the computation is fast — milliseconds. STUMPY's parallelization with Numba handles this well on a single machine.

The problem is the 3,237-symbol scope. Team 3's integration proposal ("run MIDGE's concatenated multi-domain signal matrix") does not explain:

1. **How they plan to run this across 3,237 symbols.** If STUMPY runs per-symbol on daily bars, that is 3,237 separate matrix profile computations. Each takes milliseconds individually. Total wall time is manageable for a background daemon — this is actually fine.

2. **The multidimensional (mstump) path is where the risk is.** Team 3's proposal to "feed STUMPY the concatenated multi-domain signal matrix" implies `mstump` — the multidimensional variant. mstump takes a matrix of (d dimensions × n timepoints) and computes a motif across all dimensions simultaneously. Across 28 signal streams, this is computationally heavier. The STUMPY documentation itself warns that multidimensional motif discovery algorithms are "slow, approximate, and brittle to irrelevant dimensions." The brute-force baseline for high-dimensional data is described as requiring "11.1 PB of memory" — a dramatized illustration of O(n²m) scaling, but the warning is real.

3. **No benchmark evidence provided** for mstump on MIDGE's actual data shape (28 dimensions × 2000 bars × 3,237 symbols). The claim of "moderate implementation complexity" does not account for dimension selection, normalization strategy, or whether the computational budget allows daily vs. weekly refresh.

**Verdict:** STUMPY per-symbol on daily bars is fine and the streaming `stumpi` is well-suited for incremental updates. The "concatenated multi-domain matrix" path for mstump is undersupported. Team 3 should have specified which STUMPY mode they mean. The streaming single-symbol path is Battle-Tested; the multi-domain mstump path is Novel and needs explicit scoping.

---

### Challenge 2 — Qwen3-14B VRAM claim needs precision

Team 4 states "~10-12GB VRAM at Q4_K_M — fits on common consumer GPUs (RTX 4070, 4080, 4090)." Independent verification confirms the model exists (released April 29, 2025, Alibaba) and the VRAM claims are broadly correct.

However, one precision gap: the RTX 4070 has 12GB VRAM. At Q4_K_M, Qwen3-14B sits at approximately 9-10GB for model weights. The KV cache for a 200-token response at the default context length eats additional VRAM — potentially pushing total usage to 11-12GB. This leaves 0-1GB of headroom on a 12GB card.

The claim "fits on common consumer GPUs (RTX 4070)" is technically true but leaves no margin. If Wardenclyffe's GPU is an RTX 4070 (12GB), the model will fit — but any other process competing for VRAM (Qdrant, another Ollama model cached in memory) could cause OOM errors.

**What Team 4 missed:** They did not check Wardenclyffe's actual GPU spec. The HANDOFF.md states "Win11 desktop, Docker/Qdrant/Ollama installed" but does not specify the GPU. Without knowing whether it's a 12GB or 16GB+ card, the "fits comfortably" claim cannot be confirmed.

**Evidence supporting Team 4:** Generation speed benchmarks confirm Qwen3-14B at Q4 produces 42+ tokens/second on RTX 4070-class hardware. The 30-second latency budget is achievable. The model itself is verified production software.

**Verdict:** Model existence confirmed, VRAM claims broadly correct, but headroom analysis on 12GB cards was glossed over. This is a yellow flag, not a red one.

---

### Challenge 3 — CausationEntropy (16 stars): Production-quality versus academic-only

Team 3 recommends CausationEntropy for weekly causal network discovery. They pre-empt the obvious concern ("Low star count reflects novelty (published November 2025), not quality") but this pre-emption does not fully resolve the concern.

**What independent investigation found:**

- v1.1.0 released November 12, 2025. Active maintenance confirmed (144 commits, CI workflows, ReadTheDocs).
- Associated with Clarkson Center for Complex Systems Science (C3S2) — legitimate academic institution.
- arXiv paper published January 2025 (arXiv:2601.13365) — formally reviewed.
- DOI on Zenodo: 10.5281/zenodo.17047565.
- 354 unit tests, 100% code coverage (as claimed in the findings).
- **Zero evidence of production deployment outside academic settings.**

The 16-star count is not explained by novelty alone — the paper appeared January 2025 (10 months before this validation). Libraries that solve real problems at MIDGE's level of sophistication tend to accumulate more than 16 stars in 10 months if they're genuinely useful.

**The actual risk:** This is not a data integrity risk. The code appears correct and tested. The risk is:
1. Edge cases in MIDGE's data format (mixed-frequency signals: daily COT vs. real-time price) that no one outside the author's team has stress-tested.
2. No community debugging history — issues that arise in MIDGE will require going directly to the paper authors.
3. The oCSE algorithm's computational complexity scales with number of variables × time series length. With 28 signals, this may be slow for weekly batch runs — no benchmark evidence provided.

**Verdict:** Academic-quality code, not production-validated. Usable as an experimental layer with explicit monitoring. Team 3's "Moderate" effort estimate needs a footnote: debugging unsupported edge cases is the real cost. Not a blocker, but not a "low risk" recommendation either.

---

### Challenge 4 — EoN fast_nonMarkov_SIR: The financial application is unproven

Team 2 recommends EoN's `fast_nonMarkov_SIR` as the cascade simulation engine specifically because it supports per-edge delay distributions. This is technically accurate — `fast_nonMarkov_SIR` allows user-defined transmission rules that encode varying delay distributions per edge.

**The stretch that needs flagging:**

No evidence was found of EoN or `fast_nonMarkov_SIR` being used for financial supply chain modeling. Every citation in the literature uses EoN for actual epidemics, social contagion, or opinion spread. The financial contagion papers found (ScienceDirect 2024, Springer Nature 2025) use custom network models or MATLAB-based frameworks — not EoN.

This does not mean EoN is wrong for this use case. The mathematical isomorphism between epidemic spread and supply chain shock propagation is real — SIR models are increasingly used for financial contagion in the academic literature. But the literature uses the mathematical framework, not the EoN Python package specifically.

**The practical risk:** EoN was designed for epidemiology. Its node states (Susceptible/Infected/Recovered) map awkwardly onto "supply chain disrupted/affected/recovered." Team 2 proposes encoding financial delay distributions into the transmission rules, but this mapping requires domain-specific engineering that the EoN documentation does not address. The per-edge delay encoding Team 2 cites is real functionality, but the engineering effort to adapt it to financial graphs is higher than "pip install EoN" implies.

**Verdict:** Theoretically sound, engineering effort understated. The recommendation is valid as an exploration — but the finding presents it as validated when it's Novel at best. NDlib's Independent Cascade model (simpler, less delay granularity) may be more practical for an initial implementation.

---

## SECTION 2: CONTRADICTIONS

### Contradiction 1 — Team 5 flags ProcessPoolExecutor as broken; the previous expedition synthesis recommended it

Previous expedition synthesis (Phase 3): "ProcessPoolExecutor for parallel excavation" as a recommended Phase 3 action.

Team 5 (this expedition): "ProcessPoolExecutor wrong (unpicklable)" — confirmed via specific test that `requests.Session` is not picklable.

**Assessment:** Team 5 is correct and supersedes the prior recommendation. This is not a contradiction between teams in this expedition — it is a contradiction between this expedition and the previous one. Team 5 validated the specific failure mode (spawn semantics on Windows, unpicklable `requests.Session`). The HANDOFF.md still lists "ProcessPoolExecutor" as a future optimization, which is now confirmed incorrect without restructuring the worker to be stateless.

**Action required:** The HANDOFF.md `What's Next` section recommends "Concurrent symbol processing (ProcessPoolExecutor) for another ~3-4x speedup." This is now known to be incorrect as stated. The async aiohttp path (Team 5's priority 4 recommendation) achieves similar or better speedup and avoids the pickling problem entirely.

---

### Contradiction 2 — Team 4 defers Instructor as "optional"; Team 3 recommends CausationEntropy's DAG output integrates with "matplotlib and networkx"

These are not in conflict, but they reveal an integration assumption gap. Team 3 says CausationEntropy outputs a "causal adjacency matrix + DAG visualization" compatible with NetworkX. Team 2 says causal-learn/CausalFlow outputs graphs compatible with NetworkX. But neither team clarifies whether these two outputs (CausationEntropy's DAG and Team 2's causal chain graph) would be stored in the same WorldModelGraph or in separate structures.

If both feed the same NetworkX DiGraph, there is a node namespace collision risk — CausationEntropy discovers causal links between MIDGE's 28 signal streams, while Team 2's WorldModelGraph stores supply chain company/commodity relationships. These are different graph vocabularies merging into one data structure. This was not addressed.

---

### Contradiction 3 — Team 5 says sensing workers should go to 12; HANDOFF.md says 12-20

Minor contradiction in number, not direction. Both agree increase is safe. Previous expedition validator confirmed 12-20 is safe range. Team 5 recommends 12 as conservative start. This is alignment, not contradiction — the underlying finding is consistent.

---

## SECTION 3: ALIGNMENT DRIFT

### Drift 1 — Team 2's scope significantly exceeds the problem statement

The problem statement asks for tools to "trace cascading effects across domains" and "detect precursors, stack patterns, trace causal chains."

Team 2 proposes: GLEIF (2.6M entity corporate ownership graph), UN Comtrade (country-level commodity flows), Open Supply Hub (apparel supply chains), SupplyGraph benchmark (Bangladesh FMCG), Wikidata SPARQL, NetworkX DiGraph, NDlib, EoN, CausalFlow, causal-learn, TCDF, SPACETIME, Memgraph.

This is an entire new engineering domain grafted onto MIDGE. The expected outcome was "Midge becomes the ultimate pattern recognizer." Team 2's recommendation would make Midge a supply chain intelligence platform — a different product.

The WorldModelGraph + CascadeSimulator proposal would require:
- Populating a graph with GLEIF + Comtrade data (significant data engineering)
- Maintaining the graph as ownership and trade relationships change (ongoing maintenance)
- Encoding per-edge delay distributions (domain expert input MIDGE doesn't have)
- Integrating with ConvergenceAlerter as a new "causal_chain" domain (new signal domain + test coverage)

This is a 4-6 week minimum project, not a gift for Midge. It is a new product feature. The alignment check fails: this exceeds the scope of "add capability" and enters "build new system."

**What was actually asked for:** Tools that let Midge trace causal chains from events to tickers. The simpler path — maintaining a small, manually-curated NetworkX DiGraph with 50-100 key causal relationships (hurricane→ethylene→auto, oil_shock→airline, fed_rate→financials) and using BFS to find affected tickers — achieves the expected outcome at 10% of the engineering cost. Team 2 skipped this scoped option entirely.

---

### Drift 2 — Team 3 includes tsfresh's 794 features without addressing MIDGE's 4,384-test zero-regression constraint

tsfresh's automated feature extraction generates 794 features per time series. Team 3 recommends integrating this into the pattern library as "auto-discovered precursor features." However, these features would expand the pattern library's data schema significantly and could touch `fingerprint.py`, `historical_fetcher.py`, and `pattern_library.py`.

Any changes to these files require passing 4,384 existing tests with zero regression. Team 3's "Low (offline)" effort estimate does not account for the test coverage implications of schema changes. This is not blocking — tsfresh can run offline without modifying core data structures — but Team 3's framing as a simple offline batch job understates the integration cost.

---

### Drift 3 — Both Teams 3 and 4 treat "causal narrative" as a new capability; MIDGE already has CausalReasoningEngine

Team 4 proposes building `OllamaReasoningSubscriber` to generate causal narratives. Team 3 proposes CausationEntropy to discover causal relationships. But MIDGE already has `CausalReasoningEngine` with `explain_causation()`, `_links`, `_causes`, `_effects` dicts, and a `find_causal_path()` method (confirmed via HANDOFF.md and Team 4's own findings).

Neither team audited what CausalReasoningEngine currently does before recommending new causal capabilities. Team 4 did note the integration point ("enriching `explain_causation()` with LLM-generated language") but Team 3 did not acknowledge the existing engine at all. This creates risk of duplication — implementing CausationEntropy's DAG discovery while the existing CausalReasoningEngine already maintains causal links.

---

## SECTION 4: MISSING ANGLES

### Missing Angle 1 — Phase 0-1 dependency analysis (the most important missing piece)

The research brief explicitly states "Previous expedition Phases 0-1 not yet executed." This validator checked what Phases 0 and 1 require:

**Phase 0 (1 day):**
1. Parse `outcomes.jsonl` → compute payoff ratio
2. Run `CorrelationTracker.get_correlation_matrix()` → verify domain independence

**Phase 1 (1 week):**
3. Raise combo sample size gate from n≥5 to n≥15
4. Apply combo-specific Kelly fraction using ComboThompson mean
5. Run `populate_library.py` as companion process
6. Expand sensing workers 3 → 12-20

None of the five teams researched whether their recommendations interact with or depend on Phase 0-1 findings. This matters:

- **Phase 0 measures domain independence.** If domains are 70%+ correlated, the entire premise of adding more domains (Teams 1, 2, 3) is undermined — stacking correlated domains does not multiply statistical power, it inflates false confidence. All five teams' recommendations assume domain independence is real.

- **Phase 0 measures payoff ratio.** If the payoff ratio is already ≥4:1, MIDGE may be profitable at its current win rate without any of these additions. Adding complexity to a system that's already working is expensive maintenance.

- **Phase 1 raises the combo gate to n≥15.** Several of Team 3's recommendations (CausationEntropy weekly runs, Thompson Sampling prior updates) would generate new signal pathways. If the combo gate isn't raised first, new pathways compound the existing problem of low-n combo reliance.

**Verdict:** No team addressed this dependency. All recommendations should be sequenced AFTER Phase 0 measurement. The findings are useful, but the sequencing priority is inverted — teams recommended new capabilities before the existing system's effectiveness is even measured.

---

### Missing Angle 2 — Bootstrap Layer 33 wiring cost was not researched

The constraint states "New systems must register in ConnectionRegistry with triadic connections" and "Bootstrap layer 33 (market) handles all market module wiring."

No team investigated what triadic connection registration actually requires in MIDGE's architecture. The HANDOFF.md references `market_systems.py` as the Layer 33 wiring file. Without reading that file, no team could estimate whether wiring 5-6 new systems (WorldModelGraph, CascadeSimulator, OllamaProvider, CausationEntropy runner, PySAD instance, River ADWIN) into Layer 33 would require architectural changes to the bootstrap sequence itself.

This is a non-trivial concern: Layer 33 already has 54 market systems wired. Adding 5-6 new systems with triadic connections (each triadic connection requires 3 components) means potentially 15-18 new ConnectionRegistry entries. If Layer 33 has ordering dependencies (some systems must initialize before others), adding new systems in the wrong order could silently break bootstrap.

No team accounted for this cost. The "~200-300 lines total" estimates across teams systematically ignore wiring overhead.

---

### Missing Angle 3 — Bluesky as "unspecced" in Team 1 is underresearched

Team 1 lists Bluesky as a source with "(unspecced)" noted. This is a significant gap given that Bluesky has grown substantially as an X/Twitter alternative and is one of the few open social platforms with an accessible API. The AT Protocol (Bluesky's underlying protocol) has a public firehose. This deserved explicit research rather than "(unspecced)" dismissal.

---

### Missing Angle 4 — Fin-ModernBERT vs. MIDGE's existing sentiment stack

Team 3 recommends Fin-ModernBERT (0.1B parameter BERT-class model) to "dramatically improve" NLP signals. But Team 3 did not audit MIDGE's existing NLP stack. MIDGE has `FinBERT`, VADER, and (from previous expedition) `infomeasure` for transfer entropy between text signals. Adding another NLP model without documenting how it differs from existing FinBERT coverage duplicates infrastructure.

Fin-ModernBERT vs. FinBERT is a legitimate comparison that Team 3 should have made. The claim of "dramatically improve" is unsupported without a comparison to what's already there.

---

## SECTION 5: AGREEMENTS — WHERE TEAMS CONVERGED

### Agreement 1 — Async aiohttp conversion is the correct bottleneck fix

Teams 5 confirmed: sequential requests.Session calls are the bottleneck, aiohttp is the correct solution, and the speedup projection (85s → ~5s per 100 symbols) is based on sound math (network I/O parallelism, Polygon unlimited API). This is actionable, well-evidenced, and consistent with the HANDOFF.md baseline measurement.

### Agreement 2 — Qwen3-14B is the right model choice

Teams 4's model selection is well-researched. Qwen3-14B's existence is confirmed (released April 29, 2025). VRAM requirements verified at ~9-10GB at Q4_K_M. Generation speed confirmed at 42+ tokens/sec on RTX 4070-class hardware. The Ollama OpenAI-compatible endpoint is confirmed functional. The "~50 lines for OllamaProvider" estimate aligns with the pattern (subclass OpenAIProvider, change base_url). This is Battle-Tested.

### Agreement 3 — edgartools for 13F/13D is a high-confidence recommendation

Team 3's edgartools recommendation is the strongest recommendation in the entire expedition. 1,800 stars, 3,459 commits, actively maintained (MIT license), adds genuinely new capabilities (13F hedge fund tracking, 13D activist detection) that MIDGE's existing SEC client doesn't cover. The r/algotrading community validation adds real-world adoption evidence. Low implementation complexity. No licensing risk.

### Agreement 4 — ThreadPoolExecutor is correct; ProcessPoolExecutor is wrong for MIDGE's current architecture

Team 5 independently confirmed the ProcessPoolExecutor failure mode. Previous expedition validation also flagged it (Validation 1, "pickling test required"). Both reach the same conclusion from different angles. The evidence is strong.

### Agreement 5 — River ADWIN for regime drift detection is well-supported

River has 5,700 stars, BSD-3 license, active maintenance. ADWIN is a published algorithm (1997, Bifet & Gavaldà) with well-understood statistical properties. The integration with RegimeClassifier is architecturally clean — event-driven rather than polling. Low risk, medium value. Multiple teams referenced River without contradiction.

---

## SECTION 6: SURPRISES

### Surprise 1 — Team 4's "~200-300 lines" estimate is plausible but carries hidden architectural debt

The "~200-300 lines total" for full Ollama integration is a genuine surprise on the low end — and it holds up under scrutiny for the core plumbing (OllamaProvider, ReasoningPayloadBuilder, OllamaReasoningSubscriber, WHY section in plain_language.py).

What makes it potentially accurate: the existing OpenAIProvider + ApiGateway infrastructure is genuinely reusable. An OllamaProvider that subclasses OpenAIProvider and changes base_url is ~30-50 lines. The Pydantic CausalNarrative model is ~15 lines. The subscriber is ~60-80 lines. The plain_language.py WHY section is ~20-30 lines. The math works.

What's hidden: the ConnectionRegistry + triadic connection wiring for the new OllamaReasoningSubscriber agent was not counted. Adding it to Layer 33 bootstrap, registering it in ConnectionRegistry with 3 triadic connections, and ensuring the new `CH_LLM_REASONING_COMPLETE` channel is defined all add 50-100 lines. The real estimate is ~300-400 lines including wiring — still very manageable, but the 200-300 number is the happy path, not the complete path.

### Surprise 2 — EoN's fast_nonMarkov_SIR has no documented use in financial modeling anywhere in the literature

This was unexpected. Given the mathematical isomorphism between epidemic and financial contagion, one would expect at least some Python research implementations using EoN for financial network simulation. None were found. All financial network contagion papers from 2024-2025 use custom implementations or MATLAB. This is a stronger caution against Team 2's EoN recommendation than initially expected.

### Surprise 3 — CausationEntropy has an arXiv paper from January 2025 with formal mathematical backing

The 16-star count was suspicious enough to investigate the arXiv paper directly. arXiv:2601.13365 ("CausationEntropy: Pythonic Optimal Causation Entropy for Causal Network Discovery") was published January 2025 and presents formally derived mathematical backing for the oCSE algorithm. The algorithm itself was originally published in 2014 (Sun, Taylor, Bollt — SIAM Journal) — this is a Python implementation of an established algorithm, not speculative research. This is more reassuring than the star count implies.

---

## SUMMARY SCORECARD

| Team | Core Finding | Evidence Quality | Alignment | Action-Ready? |
|------|-------------|-----------------|-----------|---------------|
| Team 1: Zeitgeist Feeds | Finnhub/Yahoo/PRAW/FRED practical | Good — specific rate limits, free tiers verified | Yes — direct new domains | Yes, after BoundaryMembrane wiring |
| Team 2: Causal Chain Engine | NetworkX + EoN/NDlib stack | Mixed — EoN financial use unproven, WorldModelGraph scope bloated | Drift — exceeds brief scope | No — start with curated NetworkX DiGraph only |
| Team 3: Novel Pattern Recognition | edgartools + STUMPY + PySAD lead | Strong for edgartools; STUMPY mstump path understated | Mostly aligned | Yes for top 3 (edgartools, STUMPY per-symbol, River ADWIN) |
| Team 4: LLM Reasoning | Qwen3-14B + OllamaProvider | Strong — model verified, latency confirmed, architecture clean | Aligned | Yes — but account for Layer 33 wiring |
| Team 5: Hardware Optimization | aiohttp async > ProcessPoolExecutor | Strong — specific benchmarks, confirmed pickling failure | Aligned | Yes — priority 1 (sensing workers) executable immediately |

---

## BLOCKING ISSUES

1. **Phase 0-1 unexecuted — creates risk for all teams.** Domain independence not verified. Payoff ratio unknown. All new domain additions assume independence is real. Measure first.

2. **ProcessPoolExecutor in HANDOFF.md `What's Next` is now a known bad path.** It should be corrected before it misleads the next instance. Async aiohttp is the correct replacement.

3. **Team 2's WorldModelGraph/CascadeSimulator scope exceeds the expedition brief.** Implementing it as proposed would take 4-6 weeks and risks bloating MIDGE beyond the "pure Python, no external servers" constraint. Recommend scoping to: manually-curated NetworkX DiGraph with 50-100 key causal relationships, no EoN simulation, no GLEIF/Comtrade bulk data pipeline. Add simulation and data feeds in a future dedicated expedition.

---

## RECOMMENDED SEQUENCING (revised)

**Before any new build:**
- Execute Phase 0 (measure payoff ratio + domain independence) — 1 day
- Execute Phase 1 items 5+6 (sensing workers to 12, companion process for excavation) — same day

**High confidence, execute in order:**
1. edgartools 13F/13D integration (Team 3, Priority 1) — low risk, highest new signal value
2. OllamaProvider + OllamaReasoningSubscriber (Team 4) — verified architecture, confirmed hardware fit
3. River ADWIN wired to RegimeClassifier (Team 3) — low complexity, well-supported
4. aiohttp async conversion for PolygonBulkFetcher (Team 5) — largest throughput gain

**Conditional (after Phase 0 confirms domain independence):**
5. PRAW Reddit + Finnhub zeitgeist feeds (Team 1) — only if independence verified (new correlated domains are net negative)
6. STUMPY per-symbol streaming (Team 3, stumpi only — not mstump) — scoped to single-symbol streaming
7. CausationEntropy weekly batch run (Team 3) — experimental layer, monitor for edge cases

**Defer for dedicated expedition:**
- Team 2's WorldModelGraph + CascadeSimulator — out of scope for this expedition's size
- tsfresh feature extraction — schema change implications require dedicated test coverage planning
- Fin-ModernBERT — needs comparison against existing FinBERT before adding another NLP model
- EoN financial cascade simulation — no production precedent found; theoretical only

---

*Validation conducted 2026-03-05. WebSearch used to independently verify: Qwen3-14B existence and VRAM specs, CausationEntropy repository status, EoN financial use precedent, STUMPY computational complexity, DeepSeek-R1-14B hardware requirements, ProcessPoolExecutor Windows spawn limitations, Instructor + Ollama integration status.*
