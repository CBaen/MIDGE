# Expedition Validation Report: Gifts for Midge
**Date:** 2026-03-05
**Validator Role:** Cross-validation, Divergence-First Protocol

---

## Divergence-First: Problems Before Agreement

The protocol requires finding what does not hold up before noting what does. This section leads.

---

## 1. Evidence Challenges — Claims That Do Not Hold Up Under Scrutiny

### CRITICAL: Team 3 — STUMPY "Production Use by TD Ameritrade" is Trademark, Not Adoption

**The claim:** "Used in production by TD Ameritrade (now Schwab)."

**What is actually true:** TD Ameritrade IP Company, Inc. is the trademark holder of the name STUMPY. The GitHub repository is hosted under the `TDAmeritrade` org. Trademark ownership does not equal production deployment. No public documentation, press release, or verified source confirms that TD Ameritrade (or Schwab) runs STUMPY in production trading systems.

**Why this matters:** This is the strongest social proof claim in Team 3's entire report. If it does not hold, the battle-tested framing is weakened. STUMPY is still a solid, well-maintained library (4,100 stars, active, BSD-3 licensed, SciPy 2024 presentation). But the production adoption claim should be retracted and not used to justify implementation priority.

**Verification method:** Fetched the GitHub repo directly. The only TD Ameritrade reference is in the license/trademark notice. Searched specifically for Schwab/TD Ameritrade STUMPY production deployment — no confirming sources found.

**Verdict:** Claim is misattributed. Do not rely on it. STUMPY's value stands on its merits; it does not need this claim.

---

### SERIOUS: Team 1 — Reddit/PRAW Integration Is Gated by Manual Review Since November 2025

**The claim:** Team 1 treats PRAW as a straightforward Phase 2 addition requiring "Reddit app registration (client_id + client_secret, free)." The finding notes pre-approval "may be required for applications with significant volume."

**What is actually true:** Reddit ended self-service API key creation in November 2025. All new OAuth applications now require submission to Reddit's Developer Support form and manual review, with a stated seven-day target response time. This is not a volume concern — it applies to all new applications regardless of scale.

**Why this matters:** Team 1's Phase 2 plan treats PRAW as low-friction. The actual friction is: write an application explaining your use case, subreddits, and expected volume, then wait up to a week for approval. If Reddit denies or delays, the whole Phase 2 Reddit integration stalls. This significantly changes the implementation risk profile. PRAW itself works fine if approved; the bottleneck is getting credentials.

**Verification method:** Two separate searches confirmed November 2025 policy change. The molehill.io and replydaddy.com sources directly confirm the self-service closure.

**Verdict:** Risk flag must be elevated from "minor note" to "blocking pre-condition." Phase 2 should begin by submitting the Reddit developer application now, not when the code is ready.

---

### MODERATE: Team 2 — NetworkX Claim That It "Holds Millions of Nodes Efficiently" Is Optimistic for GLEIF's 2.6M Entity Dataset

**The claim:** "NetworkX holds millions of nodes in memory efficiently. For MIDGE's use case — a few thousand companies, commodities, and event types — NetworkX in-memory is the right choice."

**The contradiction:** Team 2 then recommends GLEIF Level 2 data as the backbone for the world model, citing 2.6 million entities. The "few thousand nodes" framing and the "2.6M entity" data source are not reconciled.

**What verification shows:** NetworkX's memory overhead is roughly 100 bytes per node/edge in pure Python dict structures. A 2.6M entity graph (even partially ingested) at moderate edge density (say 5M relationships) = hundreds of MB to several GB. This is not "efficient" — it is workable but heavy. The team's own "When to use a graph DB instead" threshold is "100k+ nodes" — yet GLEIF provides 2.6M entities.

**Mitigating factor:** MIDGE would not load all 2.6M LEI entities — only those intersecting its ticker universe. If the watchlist is 500-5,000 tickers and their ownership chains, the actual working set may be a few hundred thousand nodes — within NetworkX's comfortable range. But this filtering logic is not specified.

**Verdict:** The recommendation is probably correct in practice, but the reasoning has an internal gap. Team 2 should explicitly state: "Load only the ownership subgraph for tickers in MIDGE's universe, not the full GLEIF dataset."

---

### MODERATE: Team 4 — VRAM Estimate for Qwen3-14B Q4_K_M Is Internally Inconsistent

**The claim:** "~10-12GB VRAM" in the executive summary. The recommendation says fits on "RTX 4070, 4080, 4090."

**What verification shows:** Multiple sources confirm Qwen3-14B Q4_K_M base weights require approximately 7-8GB VRAM for weights alone. With a context window and KV cache at 4K tokens, total VRAM rises to roughly 10-12GB. The 10-12GB figure is defensible but context-length dependent. An RTX 4070 (12GB VRAM) has essentially zero headroom for context beyond ~2K tokens while also running the model. An RTX 4070 Ti (12GB) and 4080 (16GB) are the practical floor.

**Wardenclyffe GPU unknown:** The team explicitly notes "without knowing the exact GPU" — this is fine, it is an honest limitation. But the claim "fits on RTX 4070" needs the qualification "at minimal context length." For reasoning tasks producing 200-300 token outputs with 1K-2K token prompts, the 4070's 12GB is right at the limit.

**Verdict:** The model choice is sound. The VRAM estimate needs the qualifier "at minimal context windows (~2K tokens); 16GB+ recommended for comfortable operation." Not a blocking error, but an accuracy issue.

---

### MINOR: Team 2 — EoN's "Version 1.2 Released June 2024 — Actively Maintained" Overstates Momentum

**The claim:** "Version 1.2 released June 2024 — actively maintained."

**What verification shows:** EoN's version history shows a 5-year gap between v1.1 (December 2019) and v1.2 (June 2024). The maintainers themselves wrote "We hope to move on to v1.3 fairly quickly after this long gap." GitHub shows 165 stars, 50 open issues, 301 commits total. This is a lightly-staffed academic library that had one release after a multi-year gap.

**Why this matters for MIDGE:** `fast_nonMarkov_SIR` is the specific function Team 2 recommends for per-edge delay distributions. The function exists and is documented (verified at epidemicsonnetworks.readthedocs.io). But if EoN returns to a maintenance gap and has Python version compatibility issues, MIDGE would be dependent on a fragile library for a core capability.

**Mitigating factor:** The function itself is stable and the API is unlikely to change. The risk is Python version compatibility in future environments, not immediate breakage.

**Verdict:** "Actively maintained" is an overstatement. "Recently re-activated after a long gap" is accurate. Treat as a working-but-watch dependency. NDlib (which Team 2 also recommends) has a more active maintainer profile and is simpler for the initial implementation.

---

### MINOR: Team 3 — CausationEntropy Star Count (16) Is Treated as a Signal of Novelty, But It Is Also a Signal of Risk

**The claim:** "Low star count reflects novelty (published November 2025), not quality — the paper is formally reviewed and the implementation has full test coverage."

**The challenge:** 16 stars 4 months after publication is genuinely low, even for a niche academic library. The team's defense is reasonable, but the risk profile is real: one maintainer, no visible community, unknown behavior on mixed-frequency data (the team themselves flag this). If the library has a bug in an edge case MIDGE triggers, there is no community to surface it.

**This is not an elimination.** The library is interesting and appropriate for weekly offline runs. But the "354 unit tests, 100% code coverage" claim is difficult to verify independently and the team did not do so.

**Verdict:** Keep in the priority list but at its current priority-5 position, not elevated. Do not treat it as battle-tested.

---

### MINOR: Team 5 — "85s → ~5s" Async Projection Is a Model, Not a Measurement

**The claim:** "85s/100 → ~5s/100 (projected)" for the async conversion.

**The math:** "100 calls / 50 concurrent, each ~0.8s = ~1.6s." This math is correct under ideal conditions but ignores: TCP connection setup overhead, Polygon's server-side response latency variance, potential backpressure from 50 simultaneous connections to a single host, and asyncio event loop overhead per coroutine. The actual result will likely be 3-10s, not 1.6s.

**Why this is minor:** Team 5 explicitly recommends profiling first (Step 1 in the roadmap). The number is a projection to motivate the optimization, not a committed deliverable. The directional claim (async will dramatically reduce excavation time) is solid.

**Verdict:** The projection is reasonable as a ceiling estimate. Label it as "projected best case" in implementation planning.

---

## 2. Contradictions — Where Teams Diverge

### Team 1 vs. Team 3 on NLP Approach

Team 1 recommends VADER sentiment for Reddit, Bluesky, and Yahoo Finance RSS text. Team 3 recommends Fin-ModernBERT as a local model for news NLP, which would dramatically improve financial-domain sentiment accuracy over VADER.

**Which has stronger evidence:** Team 3's recommendation is better supported. VADER is a general-purpose sentiment tool; it has documented weaknesses on financial text (industry jargon, ironic market commentary, hedged guidance language). Fin-ModernBERT is purpose-trained on financial text and produces NER + entity-level sentiment, which is directly what Team 1's news signals need.

**Resolution:** VADER as a fast first pass is reasonable for Team 1's Phase 1 wiring (get data flowing, validate integration). Fin-ModernBERT should replace VADER as the NLP layer in Phase 2, not be deferred to an independent workstream. The two teams' findings should be sequenced, not treated as alternatives.

---

### Team 2 vs. Team 3 on Causal Discovery (Minor)

Team 2 recommends CausalFlow (F-PCMCI) as the primary discovery tool. Team 3 independently identifies CausationEntropy (oCSE) for the same use case. Both are sound choices for different workloads: CausalFlow is better for pairwise lag discovery with known structure; CausationEntropy is better for full-network discovery from 28+ simultaneous signals where MIDGE does not know which pairs to test.

**No true contradiction** — these solve slightly different problems. The findings should be read as complementary. Team 3's framing of oCSE as doing something CausalFlow does not (simultaneous multi-variable causal graph discovery) is accurate.

---

### Team 4 vs. Team 2 on World Model Reasoning

Team 4 proposes using the LLM to generate "hidden risks the statistical engine cannot see." Team 2 builds a deterministic causal chain engine (WorldModelGraph + CascadeSimulator). These are not in conflict architecturally, but neither team flags the integration question: should the LLM reasoning be *upstream* of the cascade simulation (helping to define the world model), or *downstream* (interpreting the simulation output)?

**This is a gap, not a contradiction.** See Section 4.

---

### Team 5 vs. Previous Expedition on ProcessPoolExecutor

Team 5 confirms what the previous expedition flagged: ProcessPoolExecutor will fail for MIDGE's current PolygonBulkFetcher because `requests.Session` is not picklable on Windows spawn. This is consistent and validated. Team 5 goes further by suggesting the correct alternative (async) rather than just eliminating the wrong one. No contradiction, and this is one of the strongest aligned findings.

---

## 3. Alignment Drift — Where Findings Drift From the Research Brief

### Partial Drift: Team 2's World Model Graph Is an Infrastructure Project, Not a Pattern Recognizer

The Brief asks for Midge to "trace a hurricane's ripple through supply chains to a factory in Ohio to a stock price in New York" and "detect precursors, stack patterns, trace causal chains." Team 2's answer — build WorldModelGraph, build CascadeSimulator, wire to ConvergenceAlerter — is architecturally correct but is a multi-month infrastructure project that produces no new signal until all components are built.

The Brief's constraint "Zero regression policy on 4,384+ tests" is respected by Team 2's additive design. But the build order (five sequential steps, the last being CausalFlow integration) means no capability improvement lands until steps 1-4 complete.

**This is not a rejection of Team 2's work.** The architecture is right. The implementation should consider whether a lighter first step (add NetworkX to CausalReasoningEngine as a wrapper, manually seed 10-20 known commodity-to-sector links) delivers real signal faster while the GLEIF/Comtrade pipeline is built.

---

### Minor Drift: Team 3's Recommendation Priority Table Ranks edgartools #1 Over STUMPY

The Brief's problem statement centers on detecting precursors, stacking patterns, and tracing causal chains. edgartools (13F hedge fund holdings) is quarterly-lagged data with 45-day disclosure delay. STUMPY (motif discovery) directly addresses "find patterns Midge hasn't been trained to see." For the Brief's stated goal of day/swing trading pattern recognition, STUMPY more directly solves the problem.

edgartools as #1 may reflect general strategic value but it does not address the core brief as directly as STUMPY. Team 3's ranking logic ("low effort + high signal") is defensible, but the Brief's emphasis on real-time pattern recognition is better served by STUMPY.

---

### No Drift Found: Teams 1, 4, 5

Team 1's phased API rollout directly addresses the Brief's "more external input" requirement with free/affordable sources. Team 4's OllamaReasoningSubscriber directly addresses "deeper reasoning capability" and "reason internally about why patterns are forming." Team 5's optimization roadmap directly addresses "more concurrent processing." All three are well-aligned.

---

## 4. Missing Angles — Research Not Done, Questions Not Asked

### No Integration Ordering Across All Five Teams

The five teams produced independent findings with independent priority lists. No team was asked to synthesize across all five. As a result, there is no cross-team implementation sequence that answers: "Given all 21 recommendations across 5 teams, what is the optimal order that delivers the most capability per week of engineering time?"

Specific sequencing questions unresolved:
- Does Fin-ModernBERT (Team 3) replace VADER before or after Team 1's Phase 1 APIs are wired?
- Does the aiohttp async conversion (Team 5 Priority 4) happen before or after the 12-worker sensing scaling (Team 5 Priority 1), which takes 5 minutes?
- Does the LLM reasoning layer (Team 4) land before or after the causal chain engine (Team 2), since the LLM benefits from having world model data to reason about?

---

### No Wardenclyffe GPU Specification

Team 4 explicitly flags "without knowing the exact GPU" as a limitation. No team attempted to determine Wardenclyffe's GPU spec (which is discoverable: NVIDIA System Management Interface output, or `nvidia-smi`, or simply asking). Without this, the Qwen3-14B recommendation is conditional on hardware that has not been verified. The difference between an RTX 4070 (12GB, marginal) and an RTX 4090 (24GB, comfortable) significantly changes the model recommendation and context budget.

**Easy fix:** Run `nvidia-smi` on Wardenclyffe before pulling any model.

---

### Team 1 Did Not Flag Yahoo Finance RSS Terms of Service Risk

Team 1 correctly notes "Terms of service say personal use only" but does not investigate whether Yahoo Finance has historically enforced this or what the scraping risk profile is. The Brief asks for tools Midge can actually use. If Yahoo Finance RSS is technically against ToS and Yahoo has sent cease-and-desist notices to scrapers (which has happened), this is a risk worth flagging more prominently.

---

### Team 3 Did Not Verify CausationEntropy's Mixed-Frequency Data Handling

Team 3 recommends CausationEntropy for running across MIDGE's 28+ signal streams, which include daily COT data, real-time price ticks, and monthly FRED series. The team itself flags: "Verify the library handles the mixed-frequency data MIDGE has." This verification was not done. Mixed-frequency causal inference is a known hard problem; naive interpolation can introduce spurious causal links. This should be verified before adoption.

---

### Team 2 Did Not Assess CausalFlow's Computational Cost at MIDGE's Scale

CausalFlow (F-PCMCI) was recommended for weekly offline discovery runs. The team did not estimate runtime for the actual scale: 28 signal streams, daily observations over 3+ years (~750 data points), symbol-by-symbol runs across 3,237+ symbols. F-PCMCI is not computationally cheap at this scale. A runtime estimate or benchmark citation would strengthen the recommendation.

---

## 5. Agreements — High-Confidence Convergence Zones

The following findings are mutually reinforcing across teams or are supported by direct verification:

**Finnhub free tier: 60 req/min, includes news sentiment and economic calendar.**
Verified directly against Finnhub's own documentation and multiple third-party API guides. Team 1's claim is accurate.

**Marketaux free tier: 100 requests/day, 3 articles per request.**
Verified directly by fetching the Marketaux pricing page. Team 1's claim matches exactly.

**GLEIF Level 2 bulk download is genuinely free with no registration required.**
Verified against GLEIF's own documentation. The "Who Owns Whom" dataset is a free, open public good. Team 2's claim is accurate.

**ProcessPoolExecutor will fail for PolygonBulkFetcher on Windows due to requests.Session not being picklable.**
Both Team 5 and the prior expedition agree on this. The technical reasoning (Windows spawn behavior, Session socket handles) is sound. Not pursued further.

**aiohttp dramatically outperforms sequential requests for concurrent I/O workloads.**
Verified across multiple benchmark sources. The directional claim (10x+ improvement in concurrent scenarios) is consistent. Team 5's recommendation to convert PolygonBulkFetcher is well-grounded.

**Qwen3-14B exists, is available on Ollama, and was released April 2025.**
Verified directly via Ollama library page and Hugging Face. Team 4's model recommendation is real and available.

**DeepSeek-R1's MMLU score of 90.8% is accurate.**
Confirmed from the DeepSeek-R1 technical report (arxiv.org/abs/2501.12948). Team 4's benchmark claim is accurate.

**CausalFlow (py-causalflow) is actively maintained — v4.0.5 released May 2025.**
Verified against PyPI. Team 2's "active development" claim is accurate.

**STUMPY has 4,100 stars, is BSD-3 licensed, and has a stumpi streaming module.**
Verified via GitHub fetch. Star count matches Team 3's stated "4,100" (Team 3 summary says 4.1k). The library is real, active, and appropriately characterized. The production use claim is the only problem.

**smart-money-concepts is actively maintained — v0.0.26 released March 3, 2025.**
Verified against PyPI. Team 3's maintenance claim is accurate.

**Sensing worker scaling from 3→12 is a one-number change that is safe and immediate.**
Teams 5's analysis of MarketSensingHook's ThreadPoolExecutor is internally consistent and technically sound. This is the lowest-risk, highest-speed win in the entire expedition.

---

## 6. Surprises — What Changed the Validator's Thinking

**Reddit's API closure is more consequential than any team acknowledged.** Team 1 lists PRAW as a straightforward Phase 2 addition. The actual state is: you cannot create a new Reddit OAuth application without applying through a developer support form and waiting up to a week for manual review. If Reddit denies the application or takes longer than expected, Phase 2 Reddit integration does not happen. This should be the first thing done if PRAW is wanted — not the last.

**The "holy grail" gap finding (Team 3) strengthens MIDGE's identity.** Team 3 explicitly confirms that no open-source library does what MIDGE's ConvergenceAlerter does. TradingAgents (31K stars) and Qlib (38K stars) are framework competitors, not drop-in libraries. This is not just a gap finding — it is confirmation that MIDGE's core architecture is genuinely ahead of what is publicly packaged. This should factor into how Guiding Light prioritizes hardening existing systems vs. adding new ones.

**EoN's maintenance gap is a risk that Team 2 glossed over.** A library that went 5 years between releases before posting v1.2 in June 2024 is not "actively maintained" in the conventional sense. Given that Team 2's cascade simulation architecture depends specifically on `fast_nonMarkov_SIR`, and given that NDlib (which also does cascade simulation) is more actively maintained, there is a pragmatic case for starting with NDlib's simpler Independent Cascade model and treating EoN's per-edge delay distributions as a Phase 2 refinement. The initial cascade simulation does not require non-Markovian delay accuracy — it requires proving the concept first.

---

## Summary Scorecard

| Team | Findings Quality | Critical Issues | Alignment | Confidence |
|------|-----------------|-----------------|-----------|------------|
| Team 1: Zeitgeist Feeds | Strong | Reddit API access gated (serious) | High | High with corrections |
| Team 2: Causal Chain | Strong | EoN maintenance overstated (minor); NetworkX/GLEIF scale mismatch (moderate) | Moderate (multi-month infra) | High with scoping note |
| Team 3: GitHub/Reddit | Strong | STUMPY production claim misattributed (critical) | High for most items | High with one correction |
| Team 4: LLM Reasoning | Strong | VRAM estimate needs context qualifier (minor) | High | High |
| Team 5: Optimization | Strong | Async projection is a model not a measurement (minor) | High | High |

**Overall expedition quality:** High. Five teams produced internally consistent, well-sourced findings. The problems found are correctable. No finding needs to be abandoned entirely — each correction is a qualification, not an elimination.

---

## Action Items for Guiding Light Before Implementation Begins

1. **Run `nvidia-smi` on Wardenclyffe** to confirm GPU VRAM. This gates the Qwen3-14B vs. 8B decision.
2. **Submit Reddit developer application now** if PRAW is wanted for Phase 2 — before writing any code.
3. **Correct STUMPY framing** in any planning documents: remove "production use by TD Ameritrade." Use "trademark holder is TD Ameritrade; library is battle-tested in academic and community contexts."
4. **Add GLEIF subgraph filter note** to Team 2's WorldModelGraph design: load ownership subgraph for MIDGE's ticker universe only, not all 2.6M LEI entities.
5. **Sequence cross-team implementation** in a single ordered roadmap before starting any coding. The five independent priority lists need synthesis.
