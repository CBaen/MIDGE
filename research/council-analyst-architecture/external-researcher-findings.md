# External Researcher Findings
## Multi-Analyst Architecture for Trading Intelligence Pattern Stacking

**Date:** 2026-03-13
**Role:** External Researcher
**Question:** How should multiple analysts in a trading intelligence system communicate and share findings to enable pattern stacking?

---

## Search Strategy

The research was decomposed into four sub-intents, searched in parallel:

1. **Trading desk organization** — how do real quant trading firms specialize analysts?
2. **Multi-agent communication patterns** — blackboard, publish-subscribe, stigmergy, BDI for Python/Mesa systems
3. **Ensemble intelligence** — how do MoE, TradingAgents, FinCon, MASFIN, QuantAgent assemble composite pictures from specialists?
4. **Temporal pattern communication** — how do CEP (Flink), SRMT, and orchestration frameworks track "sequence X% complete"?

Sources consulted: context7 (not applicable — no library API question), then WebSearch → WebFetch chain on each sub-intent. Twelve external sources fetched, eight search chains completed.

---

## Sub-Intent 1: How Do Real Trading Desks Organize Analysts?

### What Industry Practice Shows

Real-world quant trading firms do NOT organize by domain (macro analyst, insider analyst, technical analyst). They organize by **role in the decision pipeline**, with specialization by **data modality and temporal horizon**.

**The pod model (Citadel, Millennium, Point72):**
Each pod is a self-contained P&L unit with one Portfolio Manager plus specialized support. The PM synthesizes all signals. Pods compete for capital via Sharpe ratio — information flows UP as quantitative performance metrics, not across as narrative. Risk management is centralized, separate from the pods. Source: [Millennium, Citadel & Point72 Pod Structure](https://navnoorbawa.substack.com/p/how-millennium-citadel-and-point72)

**The domain modality model (TradingAgents, FinCon):**
Four to seven specialists each own ONE information domain:
- Fundamental Analyst: company financials, valuation
- Sentiment Analyst: social media, public mood
- News Analyst: macro economic indicators, events
- Technical Analyst: price chart patterns, indicators
- (FinCon adds) Audio Analyst: earnings call recordings
- (FinCon adds) Quantitative Analyst: computed metrics, CVaR

All analyst outputs flow to a single Manager Agent who synthesizes the composite view. Analysts do NOT talk to each other — all cross-agent communication goes through the manager/synthesizer. This prevents quadratic message explosion and keeps each analyst pure to its domain. Sources: [TradingAgents](https://tradingagents-ai.github.io/) | [FinCon (NeurIPS 2024)](https://arxiv.org/html/2407.06567v2)

**The temporal horizon model (MASFIN):**
Five sequential crews, each owning ONE phase of the analysis lifecycle:
1. Postmortem Crew: what failed recently? (bias removal)
2. Screening Crew: filter candidates (sentiment + rules)
3. Analysis Crew: compute metrics on candidates
4. Timing Crew: buy/hold/sell decisions + macro context
5. Portfolio Crew: weight and construct final output

Each crew produces a structured document that becomes the next crew's input. No backward information flow. Temporal alignment is enforced by using only data available up to time t-1. Source: [MASFIN (arXiv:2512.21878)](https://arxiv.org/html/2512.21878v1)

### The Key Structural Insight for MIDGE

Real firms use ONE of two primary axes for analyst specialization:
- **By data domain** (what information type): insider, macro, technical, events, positioning
- **By analysis phase** (what question): "what patterns exist?" → "which are developing?" → "which are inevitable?"

MIDGE's existing architecture (ConvergenceAlerter, PatternWatcher, DeepAnalyst) already implicitly uses the analysis phase axis. The gap is that these three systems operate as independent monoliths rather than as analysts who read each other's findings and build composite pictures.

**Scoring:**
- Relevance: 9/10 — directly answers how to specialize
- Maturity: 9/10 — this is how the most successful quant firms work
- Community Health: 9/10 — actively published and implemented
- Integration Effort: 7/10 — requires restructuring analyst communication contracts

---

## Sub-Intent 2: Multi-Agent Communication Architecture Patterns

### Pattern A: Blackboard Architecture

**What it is:** A shared read/write working space where specialist agents post their findings without knowing who will consume them. All agents observe the blackboard and act on what they find relevant.

**Key data structures:**
- Central Blackboard (β): shared key-value or document store
- Response Board (βr): isolated area where agent responses accumulate without overwriting each other
- Agents post with "varying strengths" and signals "fade over time using decay curves"

**How it works:** No task assignment. Agents monitor the blackboard, self-select when they can contribute, post findings. The orchestrator (or a synthesis agent) reads the accumulated picture. "There is no task assignment; instead, requests are broadcast on the blackboard, and each agent retains full autonomy to decide whether to participate."

**Performance:** Recent benchmarks (Google Research, 2025) show blackboard architectures outperform master-slave by 13-57% on complex discovery tasks. Source: [Google Research: Blackboard Multi-Agent Systems](https://research.google/pubs/blackboard-multi-agent-systems-for-information-discovery-in-data-science/) | [arXiv:2510.01285](https://arxiv.org/html/2510.01285v1)

**Fit for MIDGE:** High. The MIDGE signal buffer is already a de facto blackboard — signals accumulate and decay (72h window). The missing piece is that specialists don't POST their synthesized findings back to a shared space for other analysts to read.

**Scoring:**
- Relevance: 10/10 — exact pattern needed
- Maturity: 9/10 — production-proven at Google scale
- Community Health: 9/10 — active 2025 research
- Integration Effort: 6/10 — requires a structured "analyst findings" shared structure layered on top of signal buffer

### Pattern B: Stigmergic Blackboard Protocol (Digital Pheromones)

**What it is:** Agents leave "traces" in a shared environment. Other agents read the accumulated traces and act. No direct agent-to-agent messaging. The trace structure itself IS the communication.

**Key data structures:**
- Signal entries with: source_agent, signal_type, strength, timestamp
- Decay function: strength decreases over time (prevents stale data accumulation)
- Accumulation: multiple agents posting on the same topic causes strength to add (pheromone reinforcement = implicit consensus)

**How this maps to trading:** When three analysts independently post "bullish: AAPL" to the stigmergic board (from three different analytical perspectives), the accumulated strength of those traces is itself a pattern — stronger than any individual finding. The ConvergenceAlerter is already detecting this, but signals are raw source signals, not analyst-synthesized findings.

**Implementation:** The `sbp-client` Python package (PyPI) implements this. In-memory store by default, extensible to Redis/SQLite. Source: [Stigmergic Blackboard Protocol](https://dev.to/naveentvelu/introducing-sbp-multi-agent-coordination-via-digital-pheromones-2j4e)

**Fit for MIDGE:** Medium-high. The MIDGE signal buffer already implements a form of this — signals accumulate in a 72h window and drive convergence detection. The upgrade would be to add a second stigmergic layer where ANALYST CONCLUSIONS (not raw signals) accumulate with decay, enabling the ConvergenceAlerter to detect convergence of analyst perspectives rather than raw source signals.

**Scoring:**
- Relevance: 8/10 — directly applicable to signal accumulation
- Maturity: 7/10 — production-proven in concept, newer Python SDK
- Community Health: 7/10 — active but niche
- Integration Effort: 7/10 — layerable on existing signal buffer pattern

### Pattern C: BDI (Belief-Desire-Intention) with Shared Belief Base

**What it is:** Each analyst maintains a private belief base updated by perceptions. A shared belief base holds what all analysts have agreed is true. Belief revision (adding, removing, contracting beliefs) happens when new evidence conflicts with existing beliefs.

**Temporal aspect:** "A temporal component allows one to represent the dynamics of how agents and their environments change over time." BDI with continual temporal planning interleaves online search with execution, "extracting and revising partial plans in response to environmental change."

**Fit for MIDGE:** Medium. The `Inevitability` dataclass in `DeepAnalyst` is effectively a belief statement. The `hypothesis_registry.py` event-sourced lifecycle (probation → active → hibernated → retired) IS a belief revision system. What's missing is sharing these beliefs across analysts so each can condition on what the others believe.

**Scoring:**
- Relevance: 7/10 — good theoretical fit, adds formal belief revision
- Maturity: 8/10 — well-established theory, implementations in Jason/AgentSpeak
- Community Health: 7/10 — active research, new ML-BDI integration 2025
- Integration Effort: 5/10 — formal BDI is heavy; selective application is practical

### Pattern D: Global Workspace Theory (SRMT)

**What it is:** Independent specialist modules broadcast their working memory states to a global workspace. Other specialists read the global workspace alongside their own local observations. No direct peer-to-peer messaging.

**Key mechanism (from SRMT paper):** Each agent has a personal memory vector. At each step:
1. Self-attention: agent processes its personal history
2. Cross-attention: agent reads all other agents' current memory vectors from the global shared memory
3. Memory head: updates personal memory for next step

"Coordination emerges through asynchronous memory observation rather than active messaging." Source: [SRMT (arXiv:2501.13200)](https://arxiv.org/html/2501.13200v1)

**Fit for MIDGE:** High. MIDGE already has a `GlobalWorkspace` system. The missing connection is that the market intelligence pipeline (ConvergenceAlerter, PatternWatcher, DeepAnalyst) never writes its findings INTO the GlobalWorkspace — and specialist analysts never READ each other's outputs from there. This is the exact "two disconnected pipelines" problem identified in the Evolution Blueprint.

**Scoring:**
- Relevance: 9/10 — matches the identified pipeline disconnection
- Maturity: 8/10 — well-grounded in neuroscience, implemented in 2025
- Community Health: 8/10 — active research in both cognitive science and ML
- Integration Effort: 7/10 — GlobalWorkspace already exists, needs wiring

---

## Sub-Intent 3: Ensemble Intelligence for Composite Assessment

### Pattern E: The Manager-Analyst Hub (FinCon)

**The most production-proven pattern for trading intelligence.**

**Architecture:**
- Analysts are uni-modal specialists (one domain each)
- All outputs flow to a Manager who synthesizes
- Analysts NEVER communicate directly with each other
- Manager has three memory types: working memory (real-time), procedural memory (ranked past decisions), episodic memory (full trajectory history)

**Timeliness-aware memory decay by analyst type:**
- Annual filings: low decay (persistent relevance)
- Earnings calls: medium decay
- Daily news: rapid decay (highest recency weight)

**Belief propagation (CVRF mechanism):** When the composite decision is wrong, the Manager traces which analysts contributed to the failure and sends targeted belief-correction messages ONLY to those analysts. Not broadcast — surgical. Source: [FinCon NeurIPS 2024](https://arxiv.org/html/2407.06567v2)

**Fit for MIDGE:** High. The ConvergenceAlerter already acts as the Manager. The upgrade: each "analyst" (domain specialist) synthesizes its domain's signals into a structured opinion before sending to ConvergenceAlerter. ConvergenceAlerter then aggregates analyst opinions, not raw signals. When an alert fails (outcome_collector records loss), targeted feedback goes to the analysts whose domains drove the bad alert.

### Pattern F: The Dialectical Researcher Team (TradingAgents)

**Architecture:** After analysts produce structured reports, a Researcher Team conducts a "dialectical process involving bullish and bearish perspectives" — two researchers debate. One argues the bull case, one the bear case. The debate record is what flows to the trader/decision agent.

**Why this works:** Forced disagreement surfaces the weakest parts of a thesis. The bull researcher's job is to make the strongest possible case. The bear researcher's job is to find the flaws. The synthesis is more robust than either alone.

**Fit for MIDGE:** Medium-high. MIDGE's `hypothesis_validator.py` already uses adversarial validation (DSR anti-overfitting). The upgrade: formalize this into a two-perspective structure before converging. When ConvergenceAlerter detects 3+ domains aligning bullish, spawn a "bear case analyst" that specifically looks for counter-signals in the same time window. Source: [TradingAgents](https://tradingagents-ai.github.io/) | [arXiv:2412.20138](https://arxiv.org/abs/2412.20138)

### Pattern G: Decomposed Sequential Pipeline (MASFIN, QuantAgent)

**QuantAgent's four-agent signal consensus model:**
- Each agent produces a structured JSON finding: `{signal_state, pattern_detection, trend_info, confidence}`
- DecisionAgent proceeds "only when majority align and are reinforced by confirmations"
- Conflicting signals are down-weighted; "choose the side with stronger, more recent confirmation"
- Tie-breaking: fall back to the dominant trendline slope

**Message schema between agents (directly applicable):**
```
{
  "source_agent": "IndicatorAgent|PatternAgent|TrendAgent",
  "timestamp": ISO,
  "finding": {
    "direction": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "evidence": [domain-specific structured evidence],
    "domain": "technical|insider|macro|..."
  }
}
```

Source: [QuantAgent (arXiv:2509.09995)](https://arxiv.org/html/2509.09995v3)

**MASFIN's data flow insight:** "Numerical arrays are not sent directly. Instead, they are stored in data files or tables and referred to by identifiers." Agents exchange lightweight reference messages, not heavy payloads. This is critical for an EventBus system — prevent message bloat.

Source: [Financial Agent Orchestration (arXiv:2512.02227)](https://arxiv.org/html/2512.02227v1)

### Pattern H: Mixture of Experts for Adaptive Gating

**TradExpert (2025):** Four specialist LLMs analyze distinct data types; a General Expert synthesizes. Critical finding: "The Market Analyst and News Analyst emerged as the most critical experts, while the Fundamental Analyst had the smallest effect on daily trading metrics but provided essential long-term stability." This implies different analysts should have different weights — not equal voting.

**Adaptive MoE for volatility regimes:** A gating network dynamically weights which expert's output to trust most based on current market regime. In volatile regimes, momentum experts dominate. In sideways markets, fundamentals experts get more weight. "MoE model with static gating demonstrated superior overall accuracy, reducing MSE to 0.001105 for volatile firms." Source: [TradExpert (arXiv:2411.00782)](https://arxiv.org/html/2411.00782v2) | [Adaptive MoE (arXiv:2508.02686)](https://arxiv.org/abs/2508.02686)

**Fit for MIDGE:** High. MIDGE already has regime-aware Thompson Sampling and `REGIME_DECAY_RATES`. The upgrade: not just decay rates, but ANALYST WEIGHTS per regime. In volatile regimes, weight the technical analyst's view more. In trend regimes, weight the macro analyst more. Thompson already learns which domains perform — extend this to learn which analysts' domain views perform per regime.

---

## Sub-Intent 4: Temporal Pattern Communication

### Pattern I: Complex Event Processing (FlinkCEP) State Model

FlinkCEP tracks partial pattern completion using a `SharedBuffer` that maintains named event lists for each pattern stage. Crucially, the system does NOT use a "60% complete" representation — it uses **discrete named stages with boolean completion flags per stage**.

**Pattern state representation:**
```
{
  "pattern_name": "bull_setup",
  "stages": {
    "stage_1_accumulation": [events...],  # complete if list non-empty
    "stage_2_momentum_shift": [events...], # complete if list non-empty
    "stage_3_breakout": []                # empty = not yet triggered
  },
  "window_expires_at": timestamp,
  "skip_strategy": "SKIP_PAST_LAST_EVENT"
}
```

**Partial match handling:** When a time window expires before all stages complete, Flink fires a "timeout handler" — it reports the partial match as a timed-out sequence, not just discards it. This is the key insight: **partial completions are first-class events, not failures**. Source: [FlinkCEP Docs](https://nightlies.apache.org/flink/flink-docs-master/docs/libs/cep/)

**Fit for MIDGE:** High. The `CascadeTracker` already tracks multi-stage causal sequences. The upgrade: apply the same stage-completion model to ANALYST SEQUENCES — "insider domain fired" → "macro domain fired" → "technical domain fired" is a 3-stage pattern. Tracking partial completion tells you whether a pattern is 1/3, 2/3, or 3/3 complete, enabling proactive watch before the alert threshold.

### Pattern J: Temporal Orchestration with Memory Agent

From the financial agent orchestration paper (arXiv:2512.02227): "Memory stores only structural summaries, not evaluation-window labels." The memory agent is the situation board. Its schema:
```
{
  "task_id": UUID,
  "agent_role": "analyst_type",
  "time_window": {"start": ISO, "end": ISO},
  "finding_summary": "text",
  "regime": "current_market_regime",
  "confidence": 0.0-1.0
}
```

**Walk-forward temporal ordering:** Each finding references the time window it analyzed. The synthesizer can then see WHEN each analyst's evidence was gathered, enabling temporal ordering ("macro fired 3 days before technical — this is the expected sequence").

**Energy transfer model from peer coordination protocol:** "All peer exchanges are time-stamped and stored in memory for replay and audit." Timestamp deltas between analysts' findings = measured energy transfer time. Source: [Financial Agent Orchestration (arXiv:2512.02227)](https://arxiv.org/html/2512.02227v1)

### Pattern K: Timeliness-Aware Decay in Multi-Layer Memory (FinCon)

FinCon's memory decay rates, mapped to MIDGE domains:

| Domain | Decay Rate | Rationale |
|--------|-----------|-----------|
| Annual filings / COT | Very low | Structural, persists weeks |
| Earnings / macro | Medium | Event-driven, 7-14 days |
| Congressional trades | Medium-low | Political cycle |
| News / technical | High | Hours to days |
| Real-time sentiment | Very high | Minutes to hours |

This maps directly to MIDGE's existing `_domain_windows` in ConvergenceAlerter (positioning=14d, government=7d, contracts=7d, default=72h). The upgrade: analyst OPINIONS about their domain should decay at domain-appropriate rates, independently of the raw signal decay.

---

## Novel Approaches

### Novel A: The Situation Report Pattern

Rather than raw signals flowing between analysts, each analyst produces a "situation report" — a structured summary of what their domain currently says about a ticker. The report becomes the unit of inter-analyst communication.

```python
@dataclass
class SituationReport:
    analyst_id: str           # "insider_analyst", "macro_analyst", "temporal_analyst"
    ticker: str
    direction: str            # "bullish" | "bearish" | "neutral"
    confidence: float         # 0-1 analyst-level confidence
    domain: str               # which domain this analyst owns
    key_evidence: list[dict]  # top 3-5 evidence items (not all signals)
    stage_completion: float   # 0.0-1.0: how far into the expected pattern sequence
    temporal_position: str    # "early" | "developing" | "mature" | "fading"
    gestation_days_elapsed: int  # how many days since first signal
    expected_window_days: int    # analyst's estimate of remaining time
    timestamp: str
    decay_rate: float         # how fast this report loses relevance
```

The `SituationBoard` (shared blackboard) holds one active `SituationReport` per analyst-ticker-direction combination. The ConvergenceAlerter reads SituationReports, not raw signals. When 3+ analysts post SituationReports on the same ticker+direction, convergence fires.

**Why this is novel in this context:** No existing system in the literature applies the SituationReport pattern to MIDGE's existing domain structure. This is a direct synthesis of blackboard architecture + FinCon's analyst outputs + FlinkCEP's stage tracking + MIDGE's domain-aware convergence.

### Novel B: Temporal Analyst — Sequencing Specialist

No existing framework specializes an analyst for TEMPORAL ORDER. All existing multi-agent trading systems specialize by DATA DOMAIN. But Guiding Light's energy wave concept ("patterns are sequences with gestation periods and rest periods") calls for an analyst who specifically tracks:

- Which domain fired FIRST in the current developing situation
- How much time has elapsed between domain firings
- Whether the time gaps match historical lag correlations
- Whether energy is accelerating (dominos faster than predicted) or decaying (stalling out)

The `TemporalAnalyst` would consume the outputs of all other analysts (their SituationReports) and produce a single meta-finding:
```python
@dataclass
class TemporalSituationReport(SituationReport):
    domain_sequence: list[str]      # order in which domains fired
    time_gaps_days: list[float]     # time between each domain firing
    expected_gaps_days: list[float] # from lag_correlations.json
    sequence_match_score: float     # 0-1 how well this matches known sequences
    energy_ratio: float             # >1.0 = accelerating, <1.0 = decaying
    gestation_phase: str            # "early" | "active" | "peak" | "rest"
```

This temporal analyst does NOT fetch new data — it reads the SituationBoard and produces meta-analysis about the developing pattern's temporal dynamics.

### Novel C: The Bear Desk (Adversarial Analyst)

Inspired by TradingAgents' dialectical researcher team and MIDGE's existing `hypothesis_validator.py` adversarial validation. When any analyst posts a SituationReport with confidence > 0.70, automatically spawn a counter-analyst that specifically searches the same time window for bearish counter-signals in the same domain.

The Bear Desk doesn't have to be a separate agent — it's a mode flag. The same domain analyst re-runs its analysis in "adversarial mode" looking for the opposite case. The result is a `CounterReport` posted to the SituationBoard. The final convergence strength is adjusted by: `net_confidence = bullish_confidence * (1 - bearish_confidence)`.

---

## Emerging Approaches

### Emerging A: Shared Recurrent Memory Transformer applied to Multi-Domain Trading

The SRMT paper (January 2025) shows a production approach where each agent maintains a personal memory vector that is pooled and broadcast globally. In trading context: each analyst's "state of belief about current market conditions" is its memory vector. The global broadcast ensures every analyst can condition on what every other analyst currently believes — without direct messaging.

This differs from the hub-and-spoke manager pattern: there's no single synthesizer. Instead, EACH analyst reads all others' beliefs and updates its own. Convergence emerges from mutual conditioning rather than central aggregation.

**Risk:** Circular reasoning — analyst A conditions on analyst B who already conditioned on analyst A. Mitigation: use tick-lagged memory (analyst sees OTHER analysts' beliefs from t-1, not t).

Source: [SRMT arXiv:2501.13200](https://arxiv.org/html/2501.13200v1)

### Emerging B: LLM-as-Synthesizer for Human-Readable Composite Picture

Several 2025 implementations use a small fast LLM (GPT-4.1-nano or equivalent) as the synthesis layer — the only component that consumes ALL analyst reports and produces human-readable narrative. The LLM sees structured JSON from each analyst and outputs plain-language. This is not MIDGE's path (MIDGE uses `plain_language.py` templates), but the insight is valuable: the human-readable summary should be generated AFTER synthesis, from the composite picture, not from individual domain signals.

Source: [MASFIN arXiv:2512.21878](https://arxiv.org/html/2512.21878v1)

---

## Gaps in External Literature

1. **No published framework handles Guiding Light's "energy wave" concept** (sequences with gestation periods, rest periods, and energy transfer between phases). FlinkCEP is closest but uses stage flags, not energy continuums. The TemporalAnalyst (Novel B) fills this gap.

2. **No published framework specializes one analyst for temporal ordering** of other analysts' findings. All specialize by data domain. The temporal meta-analyst is a genuine architectural gap that MIDGE's existing `lag_correlations.json` and `CascadeTracker.energy_ratio` are uniquely positioned to fill.

3. **Domain imbalance (72% technical signals) is not addressed by any existing MoE system.** Existing MoE gating adapts to market regime but not to domain imbalance. A domain-aware gating mechanism that deliberately downweights the dominant domain (technical) when rare domains (energy, government) appear would be novel.

4. **Partial-completion alerting** (fire when pattern is 60% complete, before full convergence) exists in CEP systems (FlinkCEP's timeout handler for partial matches) but is not implemented in any trading intelligence framework. MIDGE's `CascadeTracker` and `PatternWatcher` have the data to implement this.

---

## Synthesis

### The Core Pattern: Three Analysts + One Situation Board

The strongest synthesis from external research is a **three-analyst architecture** that maps to MIDGE's existing analytical dimensions, communicating through a shared `SituationBoard`:

**Analyst 1: Domain Analyst (what is each domain saying right now?)**
- Reads raw signals from signal buffer
- Produces one `SituationReport` per active ticker per domain
- Decay rate varies by domain (72h technical, 7d government, 14d positioning)
- Closest to current `ConvergenceAlerter` behavior

**Analyst 2: Pattern Analyst (does the current signal combination match historical templates?)**
- Reads `SituationBoard` (analyst 1's reports, not raw signals)
- Queries `PatternLibrary` for template matches
- Produces `SituationReport` enriched with: template_match, template_win_rate, stage_completion
- Closest to current `PatternWatcher` behavior

**Analyst 3: Temporal Analyst (what is the timing structure of this developing situation?)**
- Reads `SituationBoard` (both analysts' reports)
- Consumes `lag_correlations.json`, `CascadeTracker.energy_ratio`
- Produces `TemporalSituationReport` with: domain_sequence, time_gaps, energy_ratio, gestation_phase
- This is NEW — no current component does this

**The `SituationBoard` (shared blackboard):**
- Replaces/augments the raw `signal_buffer` as the primary inter-analyst medium
- Stores `SituationReport` objects keyed by `(ticker, direction, analyst_id)`
- Each report decays at domain-appropriate rate
- The ConvergenceAlerter reads the SituationBoard to detect when 3 analysts all have active reports on the same ticker+direction

**The Synthesizer (ConvergenceAlerter, enhanced):**
- Reads SituationBoard, not signal buffer directly
- Fires when: 3+ analysts have confidence > threshold on same ticker+direction
- Composite confidence = weighted geometric mean of analyst confidences
- Composite report includes: what each analyst found, temporal sequence, template match, energy state

**The Counter-Analyst (adversarial mode):**
- Triggered when composite confidence > 0.70
- Re-runs domain analyst in "find bearish evidence" mode
- Posts CounterReport to SituationBoard
- Net confidence adjusted by counter-evidence strength

### Communication Flow

```
Raw signals → Domain Analyst → SituationBoard
                                    ↓
              Pattern Analyst reads SituationBoard → SituationBoard (enriched)
                                    ↓
              Temporal Analyst reads SituationBoard → TemporalSituationReport
                                    ↓
              ConvergenceAlerter reads all three analysts → ConvergenceAlert
                                    ↓
              (if high confidence) Counter-Analyst → CounterReport → adjusted confidence
                                    ↓
              plain_language.py → human-readable alert
```

**EventBus integration:** Each analyst posts via EventBus topic `CH_SITUATION_REPORT`. ConvergenceAlerter subscribes to this topic. When a SituationReport arrives, ConvergenceAlerter checks: "do I now have 3 analysts on the same ticker+direction?" If yes, fire. This replaces the current polling cadence with reactive convergence.

### Why This Architecture Fits MIDGE's Laws

- **Law 1 (No Bare Dyads):** Three analysts + SituationBoard + ConvergenceAlerter forms a triadic K3. No analyst finding stands alone.
- **Law 2 (Triadic Generator):** Domain Analyst → Pattern Analyst → Temporal Analyst is the three-node structure.
- **Law 3 (Holon Protocol):** Each analyst implements sense (read signals/board), decide (produce report), act (post to board), learn (update from outcome feedback).
- **Law 5 (Stem Cell):** All three analysts share the same `BaseAnalyst` mixin; specialization is epigenome configuration.
- **Law 8 (Consciousness):** Shared SituationBoard enables integration across analysts; differentiated specialization; self-reference via temporal analyst reading all others.

---

## Scored Approaches

| Approach | Relevance | Maturity | Community | Integration | Risk | Reversibility | Evidence |
|----------|-----------|----------|-----------|-------------|------|---------------|----------|
| Blackboard / SituationBoard | 10/10 | 9/10 | 9/10 | 6/10 | 4/10 | 8/10 | 9/10 |
| Manager-Analyst Hub (FinCon) | 9/10 | 9/10 | 9/10 | 7/10 | 3/10 | 9/10 | 9/10 |
| Temporal Analyst (Novel B) | 9/10 | 5/10 | 4/10 | 7/10 | 5/10 | 9/10 | 6/10 |
| Dialectical Counter-Analyst | 8/10 | 8/10 | 8/10 | 7/10 | 3/10 | 9/10 | 8/10 |
| Stigmergic Pheromone Decay | 7/10 | 6/10 | 6/10 | 6/10 | 4/10 | 8/10 | 7/10 |
| CEP Stage-Completion Model | 8/10 | 9/10 | 8/10 | 6/10 | 3/10 | 8/10 | 8/10 |
| SRMT Global Broadcast Memory | 7/10 | 7/10 | 7/10 | 5/10 | 6/10 | 6/10 | 7/10 |
| Adaptive MoE Gating per Regime | 8/10 | 8/10 | 8/10 | 6/10 | 4/10 | 8/10 | 8/10 |

**Top recommendation:** Blackboard / SituationBoard + Manager-Analyst Hub (FinCon-style) + Temporal Analyst (Novel B). This combination is highest relevance, production-proven, and maps directly onto MIDGE's existing architectural components.

---

## Source List

- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://tradingagents-ai.github.io/) — accessed 2026-03-13
- [TradingAgents GitHub](https://github.com/TauricResearch/TradingAgents) — accessed 2026-03-13
- [FinCon NeurIPS 2024 (arXiv:2407.06567)](https://arxiv.org/html/2407.06567v2) — accessed 2026-03-13
- [MASFIN (arXiv:2512.21878)](https://arxiv.org/html/2512.21878v1) — accessed 2026-03-13
- [QuantAgent (arXiv:2509.09995)](https://arxiv.org/html/2509.09995v3) — accessed 2026-03-13
- [Financial Agent Orchestration (arXiv:2512.02227)](https://arxiv.org/html/2512.02227v1) — accessed 2026-03-13
- [Blackboard Multi-Agent Systems (Google Research / arXiv:2510.01285)](https://arxiv.org/html/2510.01285v1) — accessed 2026-03-13
- [SRMT: Shared Memory for Multi-Agent (arXiv:2501.13200)](https://arxiv.org/html/2501.13200v1) — accessed 2026-03-13
- [TradExpert: MoE for Trading (arXiv:2411.00782)](https://arxiv.org/html/2411.00782v2) — accessed 2026-03-13
- [Adaptive MoE for Volatility Regimes (arXiv:2508.02686)](https://arxiv.org/abs/2508.02686) — accessed 2026-03-13
- [FlinkCEP Complex Event Processing](https://nightlies.apache.org/flink/flink-docs-master/docs/libs/cep/) — accessed 2026-03-13
- [Stigmergic Blackboard Protocol](https://dev.to/naveentvelu/introducing-sbp-multi-agent-coordination-via-digital-pheromones-2j4e) — accessed 2026-03-13
- [Millennium/Citadel/Point72 Pod Structure](https://navnoorbawa.substack.com/p/how-millennium-citadel-and-point72) — accessed 2026-03-13
- [Situation Awareness Wikipedia](https://en.wikipedia.org/wiki/Situation_awareness) — accessed 2026-03-13
- [Delphi Method Structured Debate (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8374446/) — accessed 2026-03-13
