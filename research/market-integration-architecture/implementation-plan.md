# MIDGE Market Intelligence Integration — Implementation Plan
**Phase 4: Triadic Construction Blueprint**
**Date: 2026-02-22**
**Authority: Lead synthesis of Lead + Alpha + Beta findings across all six phases**

---

## 0. How to Read This Plan

This plan is organized into **Tiers** (sequential) and **Rounds** within each tier (parallel work by three builders). A Tier must be fully verified before the next Tier begins. Within a Tier, Rounds assign non-overlapping work to the three builder roles:

- **Forge (Signal Processing):** Owns data shapes, normalization, adapters, OutcomeTracker, TradeSignal
- **Anvil (Bootstrap Wiring):** Owns Layer 33, EventBus, ConnectionRegistry, HolonRegistry, fractal, stem cell roles
- **Crucible (Bug Fixes + Adversarial):** Owns all pre-condition bug fixes, parameter tuning, thread safety, validation

No builder writes code that a Tier gate requires from another builder. All Tiers except Tier 0 produce runnable, testable code. Zero regressions policy holds at every gate.

---

## 1. Resolved Architecture Decisions

These decisions were disputed across investigations and are now settled. Every builder must use these exact specifications.

### 1.1 EventBus Channel Names (Beta found mismatch, Lead resolved)

The authoritative channel namespace uses the subsystem prefix that ConnectionRegistry will register. Lead's revised channels (from lead-revision.md Section 3):

```
# Published by market_edge (ClusterDetector, PoliticianTracker, FilingTimeAnalyzer, ContractPredictor)
market.edge.cluster_detected         # ClusterSignal published
market.edge.politician_trade         # CorrelationSignal from PoliticianTracker
market.edge.filing_anomaly           # FilingTimeSignal with suspicious timing
market.edge.contract_predicted       # ContractPrediction published

# Published by market_intel (VelocityDetector, ConvergenceAlerter)
market.intel.velocity_anomaly        # VelocityDetector flags anomalous velocity
market.intel.convergence             # ConvergenceAlerter fires ConvergenceAlert
market.intel.actionable              # Final TradeSignal with Thompson weighting
market.intel.thompson_stats          # Periodic Thompson Sampler stats (monitoring only)

# Published by market_sensing (Signal ingest, PriceFetcher-based feedback)
market.sensing.signal_received       # Raw normalized MarketSignal arrives
market.sensing.outcome_observed      # Price checked, outcome determined
market.sensing.prediction_result     # ThompsonSampler.update() called, result logged
```

**Rule:** These strings must be defined as string constants in `mae_core/market/channels.py`. No magic strings anywhere else. Forge writes this file in Tier 2.

### 1.2 MarketSignal Dataclass (Lead's design, revised)

`mae_core/market/signal.py` — canonical data contract between all publishers and all consumers.

```python
@dataclass
class MarketSignal:
    # Identity
    signal_id: str          # UUID or "{source}:{symbol}:{timestamp}"
    source: str             # "sec_form4", "congressional", "hiring_tracker", etc.
    symbol: str             # Ticker. Empty string ("") for macro/pre-ticker signals.
    asset_class: str        # "stock", "crypto", "futures", "commodities", "macro"

    # Classification — ConvergenceAlerter.record_signal() inputs
    domain: str             # "insider", "congress", "contracts", "government",
                            # "technical", "sentiment", "news", "institutional"
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0-1.0 normalized

    # Reliability — ThompsonSampler key
    confidence: float       # Source reliability estimate 0.0-1.0
    decay_rate: float       # Per-day decay (from learning_config.decay_rates)
                            # NOTE: downstream consumer (_apply_forgetting) not yet built

    # Time — VelocityDetector timestamp series
    timestamp: datetime     # When the UNDERLYING EVENT occurred. NOT MIDGE's receive time.
                            # For CongressionalTrade: use transaction_date, NOT disclosure_date.
    received_at: datetime   # When MIDGE received/detected the signal.

    # Velocity — populated by VelocityDetector before passing to ConvergenceAlerter
    velocity: float         # Per-DAY velocity (not per-second). Default 0.0 before wiring.

    # Feedback loop
    outcome_symbol: str     # Ticker to check for price outcome (usually == symbol)
    outcome_window_days: int # How many days forward to measure outcome

    # Audit trail — lightweight reference, not full payload (avoids Qdrant bloat)
    raw_id: str             # Identifier of original record in source system
    raw_type: str           # "InsiderTrade", "Form8KEvent", "CongressionalTrade", etc.

    # Pattern discovery context
    metadata: dict          # Sector, committee name, NAICS code, etc.
```

**Field rules:**
- `raw_payload: dict` is NOT included. Use `raw_id + raw_type` to retrieve source records. (Beta 5.7 — full payload bloats Qdrant and defeats normalization.)
- `velocity` must be in per-day units, not per-second.
- `timestamp` for congressional trades MUST be `transaction_date`, not `disclosure_date` (45-day STOCK Act lag — Lead Part 10.5, confirmed by Alpha cross-review).

### 1.3 Source-to-Domain Mapping

| Source Module | `source` | `domain` | `strength` derivation |
|---|---|---|---|
| InsiderTrade (acquire) | `"sec_form4"` | `"insider"` | `min(1.0, total_value / 1_000_000)` |
| InsiderTrade (dispose) | `"sec_form4"` | `"insider"` | `min(1.0, total_value / 500_000)` |
| Form8KEvent | `"sec_form8k"` | `"events"` | `confidence` field (0.50-0.70) |
| CongressionalTrade | `"congressional"` | `"congress"` | `min(1.0, amount_high / 500_000)` |
| HiringSignal | `"hiring_tracker"` | `"institutional"` | `min(1.0, spike_ratio / 5.0)` |
| GovernmentContract | `"contract_award"` | `"contracts"` | `min(1.0, award_amount / 100_000_000)` |
| ContractOpportunity | `"sam_gov"` | `"contracts"` | `0.3` (opportunity only) |
| ClusterSignal | `"insider_cluster"` | `"insider"` | `confidence` field |
| ContractPrediction | `"contract_prediction"` | `"institutional_synthesis"` | `confidence` field |
| CorrelationSignal | `"politician_correlation"` | `"congress"` | `confidence` field |
| FilingTimeSignal | modifier only — not a primary signal | — | applied to co-occurring signals |

**Note on ContractPrediction domain:** Uses `"institutional_synthesis"` — a new category that does not overlap with `"insider"` or `"contracts"` in `domain_categories`. This prevents ContractPredictor + ConvergenceAlerter double-counting defense signals (Beta cross-review 5.4). One-line addition to `domain_categories` in `convergence_alerter.py`.

### 1.4 Bootstrap Layer 33 Structure

`mae_core/bootstrap/market.py` — eight private functions, one public entry point:

```
bootstrap_market(ctx)
├── _instantiate_market_systems(ctx)      # 14 market objects + OutcomeTracker (15 total)
├── _register_market_somatic(ctx)         # SomaticMap registration (must precede connections)
├── _register_market_holons(ctx)          # HolonRegistry + proxy injection
├── _register_market_fractal(ctx)         # generate_triad() for K3 repair
├── _register_market_connections(ctx)     # 23 triadic connections — Group 14
├── _register_market_stem_roles(ctx)      # SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST
├── _register_market_step_hooks(ctx)      # Fibonacci cadence with deduplication
└── _register_market_eventbus(ctx)        # Channel wiring (publish + subscribe)
```

### 1.5 Triadic Connections — Group 14 (23 connections)

Alpha's design from alpha-findings.md Part 9, channel names updated to canonical namespace.

**Market Sensing K3:**
```
sec_edgar_client -> price_fetcher,    witnesses=[job_tracker, auditor]
price_fetcher    -> job_tracker,      witnesses=[sec_edgar_client, auditor]
job_tracker      -> sec_edgar_client, witnesses=[price_fetcher, auditor]
```

**Market Edge K3:**
```
cluster_detector   -> politician_tracker, witnesses=[contract_predictor, auditor]
politician_tracker -> contract_predictor, witnesses=[cluster_detector, auditor]
contract_predictor -> cluster_detector,   witnesses=[politician_tracker, auditor]
```

**Market Learning K3:**
```
thompson_sampler    -> convergence_alerter, witnesses=[velocity_detector, auditor]
convergence_alerter -> velocity_detector,   witnesses=[thompson_sampler, auditor]
velocity_detector   -> thompson_sampler,    witnesses=[convergence_alerter, auditor]
```

**Cross-subsystem (EventBus publish — type=EVENTBUS_PUBSUB):**
```
cluster_detector   -> event_bus, channel="market.edge.cluster_detected",   witnesses=[threat_detector, auditor]
politician_tracker -> event_bus, channel="market.edge.politician_trade",    witnesses=[threat_detector, auditor]
contract_predictor -> event_bus, channel="market.edge.contract_predicted",  witnesses=[threat_detector, auditor]
filing_time_analyzer->event_bus, channel="market.edge.filing_anomaly",      witnesses=[threat_detector, auditor]
```

**Cross-subsystem (EventBus subscribe — type=CALLBACK_REGISTRATION):**
```
convergence_alerter -> event_bus, channel="market.edge.cluster_detected",  witnesses=[thompson_sampler, auditor]
convergence_alerter -> event_bus, channel="market.edge.politician_trade",   witnesses=[thompson_sampler, auditor]
convergence_alerter -> event_bus, channel="market.edge.contract_predicted", witnesses=[thompson_sampler, auditor]
```

**Integration connections (type=DIRECT_REFERENCE):**
```
convergence_alerter -> thompson_sampler,   witnesses=[velocity_detector, knowledge_base]
velocity_detector   -> convergence_alerter, witnesses=[thompson_sampler, auditor]
convergence_alerter -> knowledge_base,      witnesses=[thompson_sampler, auditor]
sec_edgar_client    -> boundary_membrane,   witnesses=[input_validator, threat_detector]
price_fetcher       -> boundary_membrane,   witnesses=[input_validator, threat_detector]
```

**Step hook + periodic (type=STEP_HOOK):**
```
convergence_alerter -> model, witnesses=[auditor, connection_registry]
thompson_sampler    -> event_bus, channel="market.intel.thompson_stats", witnesses=[auditor, connection_registry]
```

**Endocrine coupling:**
```
convergence_alerter -> event_bus, channel="market.intel.convergence", witnesses=[endocrine, auditor]
```

**Compliance note:** All 23 connections satisfy Law 1 structurally (source + target + 2 witnesses each). Connections through Qdrant (cluster_detector, filing_time_analyzer, contract_predictor) become operationally honest only after Tier 1 replaces hardcoded localhost URLs with `ctx.qdrant_url`. Until then, the registrations are structurally valid but describe the intended path, not the current path. (Alpha revision 5.)

### 1.6 Fractal Placement (Alpha's Option A — confirmed by Lead, Beta)

Market intelligence completes the bare dyad in `organ-cluster-cognitive` (currently only cognitive-system and sensory-system — a Law 1 violation at organ level). Adding `market-intelligence-system` as the third child repairs it.

```
organ-cluster-cognitive (K3 after repair)
├── cognitive-system
├── sensory-system
└── market-intelligence-system (new)
    ├── market-sensing  (K3: sec_edgar_client, price_fetcher, job_tracker)
    ├── market-edge     (K3: cluster_detector, politician_tracker, contract_predictor)
    └── market-learning (K3: thompson_sampler, convergence_alerter, velocity_detector)
```

Remaining 5 systems (house_stock_watcher, filing_time_analyzer, usa_spending_client, sam_gov_client, correlation_tracker) register individually under `market-intelligence-system` — advisory non-triadic warning is acceptable.

Implementation: use `ctx.fractal_generator.generate_triad()` at Layer 33 runtime. Do NOT modify `FRACTAL_GROUPING` in the source file (that would alter all future bootstrap calls).

### 1.7 Stem Cell Roles (Alpha's design)

Three new profiles appended to `ROLE_PROFILES` in `mae_core/agents/stem_cell.py`:

```python
"SEC_WATCHER": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.6,
    "replay_enabled": True,
    "consolidation_enabled": True,
    "semantic_search_enabled": True,
    "sensing_radius": 10.0,
    "exploration_bonus": 0.15,
    "world_model_enabled": True,
    "planning_horizon": 7,
    "capabilities": frozenset({"market_sense", "insider_track", "sec_watch"}),
},
"CONTRACT_TRACKER": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.6,
    "quorum_sensing_enabled": True,
    "world_model_enabled": True,
    "planning_horizon": 14,
    "replay_enabled": True,
    "transfer_enabled": True,
    "capabilities": frozenset({"market_sense", "contract_track", "govt_monitor"}),
},
"MARKET_ANALYST": {
    "api_call_enabled": True,
    "llm_prompt_quality": 0.85,
    "world_model_enabled": True,
    "planning_horizon": 10,
    "replay_enabled": True,
    "consolidation_enabled": True,
    "semantic_search_enabled": True,
    "generative_memory_enabled": True,
    "transfer_enabled": True,
    "maml_enabled": True,
    "quorum_sensing_enabled": True,
    "capabilities": frozenset({"market_analyze", "convergence_detect", "signal_synthesize"}),
},
```

### 1.8 Step Hook Design (Alpha's revision — with deduplication)

```python
_market_step_counter = [0]
_last_convergence_state = [None]  # {"direction": str, "strength": float}

def _market_sense_hook():
    _market_step_counter[0] += 1
    step = _market_step_counter[0]

    # Every step: check convergence (lightweight, pure in-memory)
    if hasattr(ctx, "convergence_alerter") and ctx.convergence_alerter is not None:
        try:
            alerts = ctx.convergence_alerter.check_convergence()
            for alert in alerts:
                last = _last_convergence_state[0]
                is_new_direction = (last is None or last["direction"] != alert.direction)
                is_material_change = (last is not None
                                      and abs(last["strength"] - alert.strength) > 0.1)
                if is_new_direction or is_material_change:
                    ctx.bus.publish("market.intel.convergence", alert.to_dict())
                    _last_convergence_state[0] = {
                        "direction": alert.direction,
                        "strength": alert.strength,
                    }
        except Exception:
            logger.debug("Convergence alerter step failed", exc_info=True)

    # Every 10 steps: Thompson stats
    if step % 10 == 0:
        if hasattr(ctx, "thompson_sampler") and ctx.thompson_sampler is not None:
            try:
                stats = ctx.thompson_sampler.get_stats()
                ctx.bus.publish("market.intel.thompson_stats", stats)
            except Exception:
                logger.debug("Thompson sampler stats step failed", exc_info=True)

    # Every 50 steps: Velocity anomaly scan
    if step % 50 == 0:
        if hasattr(ctx, "velocity_detector") and ctx.velocity_detector is not None:
            try:
                anomalies = ctx.velocity_detector.detect_velocity_anomalies()
                if anomalies:
                    ctx.bus.publish("market.intel.velocity_anomaly",
                                    {"anomalies": len(anomalies)})
            except Exception:
                logger.debug("Velocity detector step failed", exc_info=True)
```

**Why deduplication is mandatory:** Without it, a persistent convergence condition publishes one alert per simulation step. At 30 steps per run, that is 30 dopamine releases into the EndocrineSystem from a single market observation. Behavioral state becomes a function of run length, not information content. (Alpha revision 3, Beta cross-review 5.2.)

### 1.9 OutcomeTracker Design (Lead revision 3 — required for learning to occur)

`mae_core/market/outcome_tracker.py` — new file, approximately 70-90 lines.

Responsibilities:
1. Read `predictions.jsonl` (entries written when TradeSignals are generated)
2. For each prediction where `signal_timestamp + outcome_window_days <= today`, call `PriceFetcher.price_fetcher_for_outcomes(outcome_symbol, signal_date, window_days)` — this function already exists at `price_fetcher.py:263`
3. If price moved in predicted direction by >= 2.0%, call `ThompsonSampler.update(source, success=True, regime="default")`; otherwise `update(source, success=False, regime="default")`
4. Append result to `outcomes.jsonl`
5. Publish `market.sensing.prediction_result` on EventBus

**Where it fits:** ctx.outcome_tracker = 15th system in `_instantiate_market_systems()`. This is Phase 1 work, not Phase 2, because the two hardest pieces already exist (price_fetcher_for_outcomes, ThompsonSampler.update). Without OutcomeTracker, the Thompson Sampler never updates from real market outcomes and MIDGE never learns. (Beta cross-review 5.3, Lead revision 3.)

### 1.10 HolonProxy Adapter Methods

Add to each affected market system — additive, non-breaking:

**ThompsonSampler** (`thompson_sampler.py`):
```python
def get_statistics(self) -> dict:
    """Alias for HolonProxy.sense() delegation."""
    return self.get_stats()
```

**ConvergenceAlerter** (`convergence_alerter.py`):
```python
def get_statistics(self) -> dict:
    return {
        "domain_count": len(self.signals),
        "alert_count": len(self.alerts),
        "recent_alerts": [a.to_dict() for a in list(self.alerts)[-3:]],
    }

def step(self) -> None:
    """Step hook for HolonProxy.act() delegation. Does not publish — bootstrap hook handles that."""
    self.check_convergence()
```

**VelocityDetector** (`velocity_detector.py`):
```python
def get_statistics(self) -> dict:
    return {
        "signal_count": len(self.signals),
        "anomalous_count": sum(1 for s in self.signals.values() if s.is_anomalous),
    }
```

**Note on ConvergenceAlerter.step():** Returns `None`, not `len(alerts)`. The proxy step hook pattern ignores return values. (Alpha revision 2.)

---

## 2. Pre-Conditions — ALL Must Be Done Before Any Tier 1 Work

These are bugs that cause guaranteed crashes or permanent silent failures. They are all assigned to **Crucible**. Forge and Anvil are blocked until these pass verification.

### Pre-condition table

| ID | File | Line | Bug | Fix | Confirmed By |
|----|------|------|-----|-----|--------------|
| PC-1 | `mae_core/market/edge/contract_predictor.py` | 232 | `trade.is_purchase` — AttributeError on first real data | Add `@property is_purchase` to `InsiderTrade`: `return self.transaction_type in ("A", "P", "buy", "purchase")` | Lead + Beta independently |
| PC-2 | `mae_core/market/edge/politician_tracker.py` | 276 | `trade.shares_traded` — AttributeError; field is named `shares` | Change `trade.shares_traded` to `trade.shares` | Lead + Beta independently |
| PC-3 | `mae_core/market/intelligence/thompson_sampler.py` | 32 | `DEFAULT_PRIOR_SCALE = 10` — seeds overconfident distributions before any real data (Beta(9.5, 0.5) for sec_edgar = variance 0.004, tighter than 10 real observations) | Set `DEFAULT_PRIOR_SCALE = 2` | Beta + Alpha cross-review |
| PC-4 | `mae_core/market/intelligence/thompson_sampler.py` | 311 | `min_variance=0.01` — excludes seeded signals from exploration queue despite zero real validation | Set default to `min_variance=0.001` | Beta + Alpha cross-review (must change with PC-3) |
| PC-5 | `data/market/thompson_distributions.json` | — | File has 22 entries; `learning_config.py` seeds only 12. 10 manually-added entries with contradictory values (`rsi` at mean=0.167, `technical_rsi` at mean=0.857). Non-reproducible from codebase. | Delete file. Regenerate from seeding logic with PC-3 applied. The 10 extra entries (`rsi`, `bollinger`, `insider_cluster`, `options_flow`, `congress_trade`, `contract_award`, `technical_macd`, `technical_rsi`, `insider_form4`, `polygon`) are not in `learning_config.py` and will not be seeded — correct, as they were corrupted manual additions. Commit the clean 12-entry file. | Beta audit + Lead cross-review Surprise 2 |
| PC-6 | `mae_core/market/apis/job_tracker.py` | 302-307 | `daily_avg = signal.jobs_30d / 30` — `jobs_30d` is populated from a 7-day API call. Denominator is 4x too small. Nearly all hiring rates appear as spikes. Breaks SAM → hiring → contract leading indicator chain. | Change to `daily_avg = signal.jobs_7d / 7` | Beta + Lead cross-review Surprise 3 |
| PC-7 | `mae_core/market/intelligence/velocity_detector.py` | 132-136 | Velocity computed in units per second. At daily frequency: velocity ≈ 0.00007/sec. ConvergenceAlerter urgency threshold is `> 0.1`. Urgency permanently reads "days" regardless of acceleration. | Change `dt = total_seconds()` to `dt = total_seconds() / 86400` | Beta + Lead revision 1 |
| PC-8 | `mae_core/market/edge/cluster_detector.py` | 21 | Qdrant URL hardcoded `"http://localhost:6333"` | Replace with `qdrant_url` constructor parameter, default `"http://localhost:6333"` | Beta + Lead + Alpha cross-review |
| PC-9 | `mae_core/market/edge/filing_time_analyzer.py` | 102 | Same Qdrant hardcode | Same fix | Beta |
| PC-10 | `mae_core/market/edge/contract_predictor.py` | 32 | Same Qdrant hardcode | Same fix | Beta |
| PC-11 | `mae_core/market/edge/cluster_detector.py` | 620 | `abs(hash(cluster_id)) % (10**18)` — Python hash() is randomized per process restart. Same cluster gets different Qdrant ID on each restart. Duplicate accumulation. | Replace with `uuid.UUID(cluster_id).int` | Beta |
| PC-12 | All 16 market module files | — | All error/status output uses `print()`. No structured logging. Unobservable in Mae's multi-agent context. | Replace with `import logging; logger = logging.getLogger(__name__)` and `logger.debug/info/warning` as appropriate | Alpha + Beta |
| PC-13 | `mae_core/market/intelligence/convergence_alerter.py` | 89 | `min_domains=2` default allows two correlated signals from same company + two domains to trigger convergence. Violates Law 2 (Triadic Generator requires 3 for genuine independence). | Change default to `min_domains=3`. Also: bootstrap instantiation must pass this explicitly (do not rely on default). | Beta + Lead revision 4 + Alpha cross-review |

### Pre-condition verification command

```bash
python -m pytest tests/ -v    # Must still pass before any Tier 1 work begins
```

---

## 3. Tier 1: Signal Format and Adapter Layer
**Gate: PC-1 through PC-13 all fixed and test suite passing**

This tier establishes the shared data contract. Forge owns all work in this tier.

### Round 1-A (Forge only)

**File: `mae_core/market/channels.py`** — NEW FILE

Define all EventBus channel names as string constants. No magic strings anywhere else.

```python
# Market Edge channels (published by edge detectors)
CH_CLUSTER_DETECTED = "market.edge.cluster_detected"
CH_POLITICIAN_TRADE = "market.edge.politician_trade"
CH_FILING_ANOMALY = "market.edge.filing_anomaly"
CH_CONTRACT_PREDICTED = "market.edge.contract_predicted"

# Market Intel channels (published by intelligence layer)
CH_VELOCITY_ANOMALY = "market.intel.velocity_anomaly"
CH_CONVERGENCE = "market.intel.convergence"
CH_ACTIONABLE = "market.intel.actionable"
CH_THOMPSON_STATS = "market.intel.thompson_stats"

# Market Sensing channels (ingest and feedback)
CH_SIGNAL_RECEIVED = "market.sensing.signal_received"
CH_OUTCOME_OBSERVED = "market.sensing.outcome_observed"
CH_PREDICTION_RESULT = "market.sensing.prediction_result"
```

**File: `mae_core/market/signal.py`** — NEW FILE

Contains:
1. `MarketSignal` dataclass (exact schema from Section 1.2)
2. `TradeSignal` dataclass (exact schema from Lead Part 9)
3. Source adapter functions — one per raw type:
   - `insider_trade_to_signal(trade: InsiderTrade) -> MarketSignal`
   - `form8k_to_signal(event: Form8KEvent) -> MarketSignal`
   - `congressional_trade_to_signal(trade: CongressionalTrade) -> MarketSignal`
   - `hiring_signal_to_signal(signal: HiringSignal) -> MarketSignal`
   - `government_contract_to_signal(contract: GovernmentContract) -> MarketSignal`
   - `contract_opportunity_to_signal(opp: ContractOpportunity) -> MarketSignal`
   - `cluster_signal_to_market_signal(cluster: ClusterSignal) -> MarketSignal`
   - `contract_prediction_to_signal(pred: ContractPrediction) -> MarketSignal`
   - `correlation_signal_to_signal(corr: CorrelationSignal) -> MarketSignal`

**Congressional trade adapter critical rule:**
```python
def congressional_trade_to_signal(trade: CongressionalTrade) -> MarketSignal:
    # MUST use transaction_date, NOT disclosure_date
    # disclosure_date can be up to 45 days later than the actual trade
    # Using disclosure_date would: (a) make old trades appear fresh for decay,
    # (b) cause VelocityDetector to see a burst of "new" activity from delayed filings
    return MarketSignal(
        timestamp=trade.transaction_date,  # The actual event date
        received_at=trade.disclosure_date,  # When MIDGE saw it
        ...
    )
```

**ContractPrediction adapter critical rule:**
```python
def contract_prediction_to_signal(pred: ContractPrediction) -> MarketSignal:
    # domain MUST be "institutional_synthesis" to prevent double-counting
    # ContractPredictor already synthesized the insider + hiring + contract signals.
    # If those raw signals ALSO flow through ConvergenceAlerter, the insider buy
    # appears once as "insider" domain and again embedded in "contracts" domain.
    # "institutional_synthesis" maps to its own category in domain_categories,
    # preventing the composite from inflating cross_domain_count artificially.
    return MarketSignal(
        domain="institutional_synthesis",
        ...
    )
```

**File: `mae_core/market/intelligence/convergence_alerter.py`** — MODIFY

Add `"institutional_synthesis": "institutional"` to `domain_categories` dict. This maps the ContractPrediction's synthetic domain to the "institutional" category without overlapping with the individual "insider" or "contracts" domains.

**File: `mae_core/market/__init__.py`** — MODIFY (or create if missing)

Export `MarketSignal`, `TradeSignal`, and all adapter functions from the package.

### Round 1-A Verification

```bash
python -c "from mae_core.market.signal import MarketSignal, TradeSignal; print('Signal types OK')"
python -c "from mae_core.market.channels import CH_CONVERGENCE; print('Channels OK')"
python -m pytest tests/ -v    # Zero regressions
```

---

## 4. Tier 2: HolonProxy Adapters and Stem Cell Roles
**Gate: Tier 1 complete and verified**

These are additive changes — no behavior changes, only new methods. All three builders can work in parallel.

### Round 2-A (Forge)

**File: `mae_core/market/outcome_tracker.py`** — NEW FILE

```python
class OutcomeTracker:
    """
    Closes the Bayesian feedback loop.
    Reads predictions.jsonl daily, checks prices, updates ThompsonSampler.
    price_fetcher_for_outcomes() already exists at price_fetcher.py:263.
    This class provides the scheduling and routing.
    """
    def __init__(self, price_fetcher, thompson_sampler, data_dir: Path):
        self.price_fetcher = price_fetcher
        self.thompson_sampler = thompson_sampler
        self.predictions_path = data_dir / "predictions.jsonl"
        self.outcomes_path = data_dir / "outcomes.jsonl"
        self.min_price_move_pct = 2.0  # 2% move = successful prediction
        self._logger = logging.getLogger(__name__)

    def check_pending_outcomes(self) -> int:
        """
        Check all predictions where outcome_window_days has elapsed.
        Returns count of outcomes evaluated this call.
        """
        # 1. Read predictions.jsonl
        # 2. Filter to entries where timestamp + outcome_window_days <= today
        # 3. For each: call price_fetcher.price_fetcher_for_outcomes(...)
        # 4. Evaluate success/failure against min_price_move_pct and direction
        # 5. Call thompson_sampler.update(source, success, regime="default")
        # 6. Append to outcomes.jsonl
        # 7. Return count of evaluated predictions

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "pending_predictions": self._count_pending(),
            "total_evaluated": self._count_evaluated(),
        }
```

### Round 2-B (Anvil)

**File: `mae_core/agents/stem_cell.py`** — MODIFY

Append SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST to `ROLE_PROFILES` dict (exact specs from Section 1.7). No other changes to this file.

### Round 2-C (Crucible)

**Files: `mae_core/market/intelligence/thompson_sampler.py`, `convergence_alerter.py`, `velocity_detector.py`** — MODIFY (additive only)

Add `get_statistics()` methods per Section 1.10 spec.

Fix `ConvergenceAlerter.step()` to return `None`, not `len(alerts)`.

Also address in this round:

**File: `mae_core/market/intelligence/convergence_alerter.py`** — MODIFY

```python
# In check_convergence(): cap self.alerts to prevent memory leak
self.alerts.extend(new_alerts)
if len(self.alerts) > 1000:
    self.alerts = self.alerts[-1000:]  # Keep most recent 1000
```

**File: `mae_core/market/intelligence/convergence_alerter.py`** — MODIFY

In `record_signal()`: add input validation at top of method:
```python
strength = max(0.0, min(1.0, strength))
confidence = max(0.0, min(1.0, confidence))
```

### Round 2 Verification

```bash
python -c "from mae_core.agents.stem_cell import ROLE_PROFILES; assert 'SEC_WATCHER' in ROLE_PROFILES"
python -c "from mae_core.market.outcome_tracker import OutcomeTracker; print('OutcomeTracker OK')"
python -m pytest tests/ -v    # Zero regressions
```

---

## 5. Tier 3: Bootstrap Layer 33
**Gate: Tier 2 complete and verified**

Anvil owns the bootstrap module. Forge owns EventBus adapter wiring. Crucible owns document parity and test suite updates (which must be done ATOMICALLY with this tier — see verification gate below).

### Round 3-A (Anvil)

**File: `mae_core/bootstrap/market.py`** — NEW FILE

Implements `bootstrap_market(ctx)` and all eight private sub-functions per Section 1.4.

**Systems instantiated (15 total):**

| ctx attribute | Class | Module | Notes |
|---|---|---|---|
| `ctx.sec_edgar_client` | `SECEdgarClient` | `mae_core.market.apis.sec_edgar.client` | |
| `ctx.price_fetcher` | `PriceFetcher` | `mae_core.market.apis.price_fetcher` | |
| `ctx.house_stock_watcher` | `HouseStockWatcher` | `mae_core.market.apis.house_stock_watcher` | |
| `ctx.job_tracker` | `JobTracker` | `mae_core.market.apis.job_tracker` | |
| `ctx.usa_spending_client` | `USASpendingClient` | `mae_core.market.apis.usa_spending` | |
| `ctx.sam_gov_client` | `SAMGovClient` | `mae_core.market.apis.sam_gov` | |
| `ctx.cluster_detector` | `ClusterDetector` | `mae_core.market.edge.cluster_detector` | pass `qdrant_url=qdrant_url` |
| `ctx.politician_tracker` | `PoliticianTracker` | `mae_core.market.edge.politician_tracker` | wrap constructor in try/except |
| `ctx.filing_time_analyzer` | `FilingTimeAnalyzer` | `mae_core.market.edge.filing_time_analyzer` | pass `qdrant_url=qdrant_url` |
| `ctx.contract_predictor` | `ContractPredictor` | `mae_core.market.edge.contract_predictor` | pass `qdrant_url=qdrant_url`; wrap constructor in try/except |
| `ctx.thompson_sampler` | `ThompsonSampler` | `mae_core.market.intelligence.thompson_sampler` | |
| `ctx.convergence_alerter` | `ConvergenceAlerter` | `mae_core.market.intelligence.convergence_alerter` | MUST pass `min_domains=3` explicitly |
| `ctx.velocity_detector` | `VelocityDetector` | `mae_core.market.intelligence.velocity_detector` | |
| `ctx.correlation_tracker` | `CorrelationTracker` | `mae_core.market.intelligence.correlation_tracker` | |
| `ctx.outcome_tracker` | `OutcomeTracker` | `mae_core.market.outcome_tracker` | pass `price_fetcher=ctx.price_fetcher`, `thompson_sampler=ctx.thompson_sampler` |

**Critical instantiation rules:**
1. Every instantiation wrapped in try/except with `ctx.{name} = None` fallback
2. Qdrant URL sourced from ctx: `qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")`
3. `ConvergenceAlerter` MUST receive `min_domains=3` — this enforces Law 2 at the bootstrap layer. Do not rely on the default.
4. Market sources registered with BoundaryMembrane via trust scoring (Alpha's `_trust_provider()` pattern):
   ```python
   market_sources = [
       ("sec_edgar", 0.90), ("yfinance", 0.75), ("alpha_vantage", 0.80),
       ("rapidapi", 0.65),  ("usa_spending", 0.85), ("sam_gov", 0.80),
   ]
   ```
5. Log message must distinguish construction-time vs. operational-time degradation:
   ```
   Layer 33a - Market systems instantiated: 15 systems (construction failures: N)
               Operational dependencies: Qdrant, RAPIDAPI_KEY, ALPHA_VANTAGE_KEY, SAM_GOV_API_KEY
               — operational failures deferred to first use
   ```

**Fractal registration using generate_triad():**
```python
# Subsystem triads (called in _register_market_fractal)
ctx.fractal_generator.generate_triad(
    name="market-sensing",
    holon_type="subsystem",
    children_ids=["sec_edgar_client", "price_fetcher", "job_tracker"],
    parent_id="market-intelligence-system",
)
ctx.fractal_generator.generate_triad(
    name="market-edge",
    holon_type="subsystem",
    children_ids=["cluster_detector", "politician_tracker", "contract_predictor"],
    parent_id="market-intelligence-system",
)
ctx.fractal_generator.generate_triad(
    name="market-learning",
    holon_type="subsystem",
    children_ids=["thompson_sampler", "convergence_alerter", "velocity_detector"],
    parent_id="market-intelligence-system",
)
# Organ — completes the bare dyad in organ-cluster-cognitive
ctx.fractal_generator.generate_triad(
    name="market-intelligence-system",
    holon_type="organ",
    children_ids=["market-sensing", "market-edge", "market-learning"],
    parent_id="organ-cluster-cognitive",
)
```

**Endocrine coupling in `_register_market_eventbus()`:**
```python
def _on_market_convergence(channel, serialized):
    msg = json.loads(serialized) if isinstance(serialized, str) else serialized
    strength = msg.get("strength", 0.0)
    direction = msg.get("direction", "neutral")
    if direction == "bullish" and strength > 0.7:
        if hasattr(ctx, "endocrine") and ctx.endocrine is not None:
            ctx.endocrine.release_hormone(HormoneType.DOPAMINE,
                                          min(0.4, strength * 0.4),
                                          "market_opportunity")
    elif direction == "bearish" and strength > 0.7:
        if hasattr(ctx, "endocrine") and ctx.endocrine is not None:
            ctx.endocrine.release_hormone(HormoneType.ADRENALINE,
                                          min(0.5, strength * 0.5),
                                          "market_threat")

ctx.bus.register_callback(CH_CONVERGENCE, _on_market_convergence)
```

**Step hook:** Use exact design from Section 1.8.

**Expected Layer 33 log output:**
```
Layer 33a - Market systems instantiated: 15 systems (construction failures: 0)
            Operational dependencies: Qdrant, RAPIDAPI_KEY, ALPHA_VANTAGE_KEY, SAM_GOV_API_KEY — failures deferred to first use
Layer 33b - Market holons registered: 20 holons (15 systems + 3 subsystems + organ + outcome_tracker)
Layer 33c - Market fractal: organ-cluster-cognitive repaired from dyad to K3 (3 children confirmed)
Layer 33d - Market connections: 23 triadic connections registered (Group 14)
Layer 33e - Market EventBus: 12 channels wired (4 publish, 3 subscribe, 1 endocrine, 4 step hooks)
Layer 33f - Market step hooks: 1 sense hook registered (cadence: convergence/1 steps, stats/10 steps, velocity/50 steps)
Layer 33  - Market Intelligence organ complete: 15 systems, 20 holons, 23 connections
            NOTE: OutcomeTracker active (Bayesian feedback from real market outcomes — Phase 1).
            NOTE: Full ApiGateway routing deferred to Phase 2.
```

### Round 3-B (Forge)

**File: `main.py`** — MODIFY

Add import and call between bootstrap_external and bootstrap_audit:
```python
from mae_core.bootstrap.market import bootstrap_market

# in run_bootstrap():
bootstrap_external(ctx)      # Layer 31
bootstrap_market(ctx)        # Layer 33: market intelligence organ
bootstrap_audit(ctx)         # Layer 32: triadic bootstrap audit
```

Update `_build_systems_dict(ctx)` — add all 15 market systems (each tolerates None for graceful degradation):
```python
# Market Intelligence (Layer 33) — all must tolerate None
"sec_edgar_client":    getattr(ctx, "sec_edgar_client", None),
"price_fetcher":       getattr(ctx, "price_fetcher", None),
"house_stock_watcher": getattr(ctx, "house_stock_watcher", None),
"job_tracker":         getattr(ctx, "job_tracker", None),
"usa_spending_client": getattr(ctx, "usa_spending_client", None),
"sam_gov_client":      getattr(ctx, "sam_gov_client", None),
"cluster_detector":    getattr(ctx, "cluster_detector", None),
"politician_tracker":  getattr(ctx, "politician_tracker", None),
"filing_time_analyzer":getattr(ctx, "filing_time_analyzer", None),
"contract_predictor":  getattr(ctx, "contract_predictor", None),
"thompson_sampler":    getattr(ctx, "thompson_sampler", None),
"convergence_alerter": getattr(ctx, "convergence_alerter", None),
"velocity_detector":   getattr(ctx, "velocity_detector", None),
"correlation_tracker": getattr(ctx, "correlation_tracker", None),
"outcome_tracker":     getattr(ctx, "outcome_tracker", None),
```

### Round 3-C (Crucible) — ATOMIC with Round 3-A and 3-B

**CRITICAL:** test_integration.py must be updated in the SAME commit that adds bootstrap_market(). If test_integration.py is not updated atomically, the test suite fails on first run after adding Layer 33. (Beta cross-review 5.8.)

**File: `tests/test_integration.py`** — MODIFY

Update bootstrap docstring and expected_keys list to include the 15 new market systems.

Also in this round: full document parity sweep. Every file in the parity table must be updated.

---

## 6. Document Parity — Counts After Tier 3

Per CLAUDE.md document parity rule, update ALL files simultaneously:

| Metric | Before | After | Files to Update |
|--------|--------|-------|-----------------|
| Systems | 85 | ~100 (+15 market systems) | CLAUDE.md, HANDOFF.md, README.md, main.py |
| Bootstrap layers | 32 | 33 | CLAUDE.md, HANDOFF.md, README.md, CONNECTIONS.md |
| Fractal organs | 5 | 6 | data/MAES-MATHEMATICAL-IDENTITY.md |
| Triadic connections | 313 | ~336 (+23 Group 14) | CLAUDE.md, HANDOFF.md, CONNECTIONS.md, data/MAES-MATHEMATICAL-IDENTITY.md |
| Holons | 107 | ~127 (+15 systems + 3 subsystems + organ + outcome_tracker = +20) | CLAUDE.md, HANDOFF.md |
| Stem cell roles | 9 | 12 (+3 market roles) | CLAUDE.md, HANDOFF.md |
| Market modules wired | 0 | 16 | CLAUDE.md |

**Files to update (7 total):**
1. `C:\Users\baenb\projects\MIDGE\CLAUDE.md`
2. `C:\Users\baenb\projects\MIDGE\HANDOFF.md`
3. `C:\Users\baenb\projects\MIDGE\README.md`
4. `C:\Users\baenb\projects\MIDGE\mae_core\CONNECTIONS.md`
5. `C:\Users\baenb\projects\MIDGE\data\MAES-MATHEMATICAL-IDENTITY.md`
6. `C:\Users\baenb\projects\MIDGE\tests\test_integration.py`
7. `C:\Users\baenb\projects\MIDGE\main.py` (log messages with hardcoded counts)

---

## 7. Tier 4: Calibration and Safety
**Gate: Tier 3 complete, test suite passing, Layer 33 log output verified**

These changes make the integrated system safe for sustained operation. All three builders can work in parallel.

### Round 4-A (Forge)

**File: `mae_core/market/intelligence/convergence_alerter.py`** — MODIFY

Add alert deduplication (separate from step hook deduplication — defense in depth):
```python
# Add to __init__:
self._last_alert_direction = None
self._last_alert_time = None
self._min_alert_interval_hours = 4.0  # Suppress re-alert within 4 hours

# Add at start of check_convergence() before generating new alerts:
now = datetime.now()
if (self._last_alert_direction == current_direction
        and self._last_alert_time is not None
        and (now - self._last_alert_time).total_seconds() / 3600 < self._min_alert_interval_hours):
    return []  # Suppress — same condition, too recent

# Update after generating alerts:
if new_alerts:
    self._last_alert_direction = new_alerts[0].direction
    self._last_alert_time = now
```

**File: `mae_core/market/intelligence/convergence_alerter.py`** — MODIFY (continued)

Fix `get_actionable_summary()` confidence formula (Beta Section 1.3):
```python
# Before (strength contributes almost nothing due to always hitting cap):
confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.05 * bullish_strength)

# After (normalize strength by domain count):
avg_strength = bullish_strength / max(1, len(bullish_domains))
confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.1 * avg_strength)
```

### Round 4-B (Anvil)

**File: `mae_core/market/intelligence/velocity_detector.py`** — MODIFY

Fix sample variance (Bessel's correction, Beta Section 1.4):
```python
# Before:
variance = sum((v - state.velocity_mean) ** 2 for v in velocities) / len(velocities)

# After:
n = len(velocities)
variance = sum((v - state.velocity_mean) ** 2 for v in velocities) / max(1, n - 1)
```

**File: `mae_core/market/apis/filing_time_analyzer.py`** — MODIFY

Add timezone handling (Beta Section 2.4):
```python
from zoneinfo import ZoneInfo  # Python 3.9+

EASTERN = ZoneInfo("America/New_York")

# In _classify_filing_time(), before comparing against MARKET_OPEN/MARKET_CLOSE:
if filing_datetime.tzinfo is None:
    # Assume UTC for naive datetimes (SEC EDGAR returns UTC)
    from datetime import timezone
    filing_datetime = filing_datetime.replace(tzinfo=timezone.utc)
filing_et = filing_datetime.astimezone(EASTERN)
filing_time = filing_et.time()
```

**File: `mae_core/market/intelligence/correlation_tracker.py`** — MODIFY

Increase `min_observations` from 10 to 30 (Beta Section 1.5). With 10 observations, Pearson correlation confidence intervals span almost the full [-1, 1] range — statistically meaningless.

### Round 4-C (Crucible)

**File: `mae_core/market/intelligence/thompson_sampler.py`** — MODIFY

Add thread locking to file writes (Beta Section 3.3):
```python
import threading

class ThompsonSampler:
    def __init__(self, ...):
        ...
        self._lock = threading.Lock()

    def _save_distributions(self) -> None:
        """Atomic write with lock."""
        import tempfile, os
        with self._lock:
            tmp = self.persistence_path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(self.distributions, indent=2))
                os.replace(tmp, self.persistence_path)  # Atomic on POSIX and Windows
            except Exception:
                logger.warning("Failed to persist Thompson distributions", exc_info=True)
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
```

**File: `mae_core/market/intelligence/learning_config.py`** — MODIFY

Fix history log path (Beta Section 1.2). Move from `Path(__file__).parent / "config_history.jsonl"` to use `DATA_DIR` (consistent with thompson_sampler.py's path resolution):
```python
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# In update_config(), change:
_HISTORY_PATH = _DATA_DIR / "config_history.jsonl"
```

**File: `mae_core/market/apis/sec_edgar/client.py`** — MODIFY

Replace placeholder email in user agent string (Beta Section 2.1):
```python
# Before:
SEC_USER_AGENT = "MIDGE Trading Research contact@example.com"

# After (use a real address before any production/live testing):
# This is a compliance requirement — SEC blocks IPs with fake contact info
SEC_USER_AGENT = "MIDGE Trading Research midge@wardenclyffe.local"
# NOTE: Replace with real email before any live EDGAR queries
```

### Tier 4 Verification

```bash
python -m pytest tests/ -v    # Zero regressions
python main.py --agents 3 --steps 30    # Smoke test — Layer 33 log output must appear
```

---

## 8. Tier 5: Live Operation Safety
**Gate: Tier 4 complete and verified**

Lower priority items that are required before real market data is connected.

### Round 5 (All builders, assign by file)

**Crucible:**

- `mae_core/market/intelligence/thompson_sampler.py`: Implement Bayesian forgetting via `_apply_forgetting(decay_factor)`. Connect `learning_config.decay_rates` (currently dead config) to actual decay. Apply daily via step hook (every ~100 steps in simulation time). Mechanism: `alpha *= decay_factor; beta *= decay_factor` before the new observation update. This preserves the mean direction while shrinking total weight toward the prior over time.

- `mae_core/market/intelligence/velocity_detector.py`: Increase `min_observations` for anomaly detection from 5 to 10 in `detect_velocity_anomalies()`. 5 is insufficient for stable z-score estimation.

**Anvil:**

- `mae_core/market/intelligence/convergence_alerter.py`: Increase `convergence_window_hours` from 48 to 72 (Beta tuning table). Insider buy clusters can unfold over a week — 48h window is too tight for slow-moving institutional signals.

**Forge:**

- `mae_core/market/outcome_tracker.py`: Add JSONL writer to track predictions when TradeSignals are generated. Currently the OutcomeTracker can read predictions.jsonl but something must write to it. The TradeSignal publisher (wherever TradeSignal is emitted) must also write the prediction record with `signal_id`, `source`, `direction`, `timestamp`, `outcome_symbol`, `outcome_window_days`.

---

## 9. Deferred — Phase 2

These are explicitly not part of this integration. They require larger architectural changes or external dependencies not yet available.

1. **Full ApiGateway routing** for the six market-specific API clients (SEC EDGAR, HouseStockWatcher, JobTracker, USASpending, SAM.gov, PriceFetcher). Currently they call HTTP directly. Phase 2 routes them through BoundaryMembrane + InputValidator + ApiGateway. (Lead + Alpha consensus.)

2. **TickerResolver service** (`mae_core/market/apis/ticker_resolver.py`). Maps company names (from USASpending and SAM.gov) to ticker symbols. Required before government contract signals can carry valid `symbol` fields. (Lead Part 10.2, Alpha agreement.)

3. **Regime-aware Thompson Sampling.** Separate Beta distributions per market regime (bull/bear/sideways). Architecture is present (the `regime` parameter in `ThompsonSampler.update()`) but all calls currently use `regime="default"`. Requires regime detection module.

4. **CorrelationTracker deque persistence.** On restart, correlation history deque is empty but summary stats are loaded from disk. The first few updates post-restart compare new readings against stale means — producing false anomalies. Fix requires persisting deque contents.

5. **ContractPredictor decomposition evaluation.** ContractPredictor is architecturally isomorphic to ConvergenceAlerter for the defense sector — it already synthesizes SAM + hiring + insider signals into a probability. Should it be retired in favor of routing its inputs through ConvergenceAlerter directly (Option A), or retained as a domain-specific sub-convergence detector that publishes only its final prediction (Option B)? Option B with `"institutional_synthesis"` domain (implemented in Tier 1) is the interim mitigation. Full evaluation deferred.

6. **KNOWN_POLITICIANS expansion.** Currently 4 entries. PoliticianTracker is functionally hollow for 535 of 539 Congress members. Consult govtrack.us API or ProPublica Congress API for full member list.

7. **discovery_log.jsonl reader.** ConvergenceAlerter writes novel discoveries to this file; nothing reads it. Either build a reader that uses discovered patterns to seed future Thompson distributions, or document the file as audit-only.

---

## 10. Verification: End-to-End Integration

After Tier 4 is complete, verify that market intelligence actually flows through the organism.

### Test 1: Bootstrap sanity

```bash
python main.py --agents 3 --steps 30
```

Expected output (check for all these strings):
```
Layer 33a - Market systems instantiated: 15 systems
Layer 33b - Market holons registered
Layer 33c - Market fractal: organ-cluster-cognitive repaired from dyad to K3
Layer 33d - Market connections: 23 triadic connections registered (Group 14)
Layer 33  - Market Intelligence organ complete
```

### Test 2: Signal flow (dry run with synthetic signals)

```python
# In a Python shell after bootstrapping:
from mae_core.market.signal import MarketSignal
from mae_core.market.apis.sec_edgar.models import InsiderTrade
from mae_core.market.signal import insider_trade_to_signal
from datetime import datetime

# Synthesize a test insider trade
trade = InsiderTrade(
    filer_name="Test CEO",
    filer_title="Chief Executive Officer",
    filer_relationship="Officer",
    ticker_symbol="TEST",
    transaction_date=datetime(2026, 2, 15),
    transaction_type="A",
    shares=10000,
    price_per_share=100.0,
    total_value=1000000.0,
    shares_owned_after=50000,
)
signal = insider_trade_to_signal(trade)
assert signal.strength > 0.0, "Strength normalization failed"
assert signal.direction == "bullish", "Direction mapping failed"
assert signal.domain == "insider", "Domain mapping failed"
assert signal.velocity == 0.0, "Velocity should be 0.0 before VelocityDetector wiring"
print("Signal adapter: OK")

# Feed into ConvergenceAlerter
alerter = ctx.convergence_alerter
alerter.record_signal(
    signal_id=signal.signal_id,
    strength=signal.strength,
    domain=signal.domain,
    direction=signal.direction,
    confidence=signal.confidence,
    velocity=signal.velocity,
    timestamp=signal.timestamp,
    metadata=signal.metadata,
)
print("ConvergenceAlerter record_signal: OK")
```

### Test 3: Parameter verification

```python
# Verify critical parameter fixes from Pre-conditions
from mae_core.market.intelligence.thompson_sampler import DEFAULT_PRIOR_SCALE
assert DEFAULT_PRIOR_SCALE == 2, f"Prior scale is {DEFAULT_PRIOR_SCALE}, expected 2"

from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
alerter = ConvergenceAlerter()
assert alerter.min_domains == 3, f"min_domains is {alerter.min_domains}, expected 3"

from mae_core.market.intelligence.velocity_detector import VelocityDetector
from datetime import datetime, timedelta
vd = VelocityDetector()
t0 = datetime(2026, 1, 1, 12, 0, 0)
t1 = t0 + timedelta(days=1)
vd.record("test_signal", 2.0, t0)
vd.record("test_signal", 8.0, t1)
state = vd.signals.get("test_signal")
# Velocity should be in per-day units: 8-2 = 6 per day
assert abs(state.current_velocity - 6.0) < 0.01, f"Velocity {state.current_velocity} is not per-day"
print("Parameter fixes: OK")
```

### Test 4: Regression suite

```bash
python -m pytest tests/ -v --tb=short
```

All 2425 existing tests must pass. Zero regressions.

### Test 5: No alert storm

```bash
python main.py --agents 3 --steps 100
```

Grep log output for `market.intel.convergence` publication count. Should be 0-3 (not 100). If the deduplication logic is working, a persistent (or absent) convergence condition should not produce one alert per step.

---

## 11. Builder Role Summary

| Role | Tiers | Primary Files | Cannot Start Until |
|------|-------|--------------|-------------------|
| **Crucible** | Pre-conditions (sole owner), 2-C, 3-C, 4-C, 5 | `contract_predictor.py`, `politician_tracker.py`, `thompson_sampler.py` (PC-1,2,3,4,5), `job_tracker.py` (PC-6), `velocity_detector.py` (PC-7,4-B), `cluster_detector.py` (PC-8,11), `filing_time_analyzer.py` (PC-9), all 16 files for print→logging (PC-12), `convergence_alerter.py` (PC-13, 2-C, 4-A), `learning_config.py`, `correlation_tracker.py`, all doc parity files | Nothing (starts immediately) |
| **Forge** | 1-A (sole owner), 2-A, 3-B, 4-A, 5 | `mae_core/market/channels.py` (new), `mae_core/market/signal.py` (new), `mae_core/market/outcome_tracker.py` (new), `main.py` | Pre-conditions complete |
| **Anvil** | 2-B, 3-A (sole owner), 4-B, 5 | `mae_core/agents/stem_cell.py`, `mae_core/bootstrap/market.py` (new), `velocity_detector.py`, `filing_time_analyzer.py` | Tier 1 complete (needs `signal.py` to exist first) |

**Crucible begins today.** Forge begins after all pre-conditions pass. Anvil begins after Tier 1 clears.

---

## 12. Mae's Laws Compliance Map

| Law | How This Integration Satisfies It |
|-----|----------------------------------|
| **Law 1: No Bare Dyads** | 23 triadic connections in Group 14, each with 2 witnesses. Fractal repair completes organ-cluster-cognitive from dyad to K3. All channel registrations have source + target + 2 witnesses. |
| **Law 2: Triadic Generator** | Three K3 subsystems (market-sensing, market-edge, market-learning) with exactly 3 members each. min_domains=3 enforced at ConvergenceAlerter constructor — convergence requires triadic domain confirmation. |
| **Law 3: Holon Protocol** | All 15 market systems receive all 10 capabilities via HolonProxy injection. get_statistics() adapters enable meaningful sense() output. |
| **Law 4: Fractal Self-Similarity** | market-intelligence-system (organ) → 3 subsystems → 3 systems each. Fits within organ-cluster-cognitive → mae. Same K3 pattern at every level. |
| **Law 5: Stem Cell** | Three new market-specific ROLE_PROFILES (SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST). Agents differentiate via epigenome, not different code classes. |
| **Law 6: Autopoietic Closure** | ConvergenceAlerter publishes to EventBus → EndocrineSystem modulates agents → agents take market API actions → outcomes update OutcomeTracker → OutcomeTracker updates ThompsonSampler → ThompsonSampler weights future convergence → circular closure. |
| **Law 7: Rule of 3/5** | All three K3 subsystems have exactly 3 members. No bare dyads within market organ. |
| **Law 8: Eight Properties of Consciousness** | Integration: market organ feeds into Mae's decision cascade. Differentiation: 3 distinct subsystems with specialized functions. Self-reference: ThompsonSampler learns from its own prediction outcomes. Recurrence/feedback: OutcomeTracker → ThompsonSampler → ConvergenceAlerter weighting → new predictions → OutcomeTracker. Multi-scale hierarchy: fractal placement at organ/subsystem/system levels. Self-produced boundary: BoundaryMembrane + ApiGateway trust registration. Competition/selection: Thompson Sampler explore/exploit arms compete. Prediction/error-correction: OutcomeTracker explicitly measures prediction correctness and adjusts reliability. |

---

## 13. Priority Summary (Flat List for Quick Reference)

### Must-Do Before Any Code (Crucible — all blocking):
1. Fix `trade.is_purchase` crash (PC-1)
2. Fix `trade.shares_traded` crash (PC-2)
3. Set `DEFAULT_PRIOR_SCALE = 2` (PC-3)
4. Set `min_variance = 0.001` (PC-4) — same commit as PC-3
5. Rebuild `thompson_distributions.json` from seeding (PC-5) — same commit as PC-3/4
6. Fix `jobs_30d / 30` → `jobs_7d / 7` (PC-6)
7. Fix velocity per-second → per-day (PC-7)
8. Replace all hardcoded `localhost:6333` (PC-8, 9, 10)
9. Fix hash() Qdrant IDs → uuid.UUID (PC-11)
10. Replace all `print()` → logging (PC-12)
11. Change `min_domains` default to 3 (PC-13)

### Tier 1 (Forge — all blocking for Anvil):
12. Create `mae_core/market/channels.py` with channel constants
13. Create `mae_core/market/signal.py` with MarketSignal + adapters
14. Add `"institutional_synthesis"` to `domain_categories` in convergence_alerter.py

### Tier 2 (parallel):
15. Create `mae_core/market/outcome_tracker.py` (Forge)
16. Add stem cell roles SEC_WATCHER, CONTRACT_TRACKER, MARKET_ANALYST (Anvil)
17. Add `get_statistics()` adapters to Thompson, ConvergenceAlerter, VelocityDetector (Crucible)
18. Fix ConvergenceAlerter.step() to return None (Crucible)
19. Cap self.alerts at 1000 in convergence_alerter.py (Crucible)
20. Add input validation to record_signal() (Crucible)

### Tier 3 (sequential — Anvil primary, must be atomic):
21. Create `mae_core/bootstrap/market.py` (Anvil)
22. Update `main.py` with bootstrap_market() call (Forge)
23. Update `test_integration.py` (Crucible — SAME COMMIT as 21+22)
24. Update all 7 document parity files (Crucible — SAME COMMIT as 21+22)

### Tier 4 (parallel):
25. Add alert deduplication to convergence_alerter.py (Forge)
26. Fix get_actionable_summary() strength formula (Forge)
27. Fix velocity sample variance with Bessel's correction (Anvil)
28. Add timezone handling to filing_time_analyzer.py (Anvil)
29. Increase correlation_tracker min_observations to 30 (Anvil)
30. Add thread locking + atomic writes to thompson_sampler.py (Crucible)
31. Fix learning_config.py history log path to DATA_DIR (Crucible)
32. Replace placeholder email in SEC_USER_AGENT (Crucible)

### Tier 5 (before live data):
33. Implement Bayesian forgetting in ThompsonSampler (Crucible)
34. Increase velocity anomaly min_observations from 5 to 10 (Crucible)
35. Increase convergence_window_hours from 48 to 72 (Anvil)
36. Wire TradeSignal publisher to write predictions.jsonl (Forge)

---

*This plan supersedes all intermediate priority orderings from individual investigation phases. It is the authoritative implementation specification for triadic-construction.*
