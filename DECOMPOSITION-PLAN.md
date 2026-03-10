I now have a thorough understanding of the codebase structure, split points, dependencies, and test relationships. Let me produce the comprehensive decomposition plan.

---

# MIDGE Codebase Decomposition Plan

## Overview

This plan covers 58 source files and 61 test files across 12 parallel agent teams, organized in two waves. Wave 1 handles source files only. Wave 2 handles the corresponding test files and pytest infrastructure. Each team owns a non-overlapping domain. No team touches another team's files.

The memory root cause: the test suite loads `create_mae()` from `main.py` in four test files, which instantiates the full 33-layer organism including market intelligence. The `HistoricalDataFetcher.preload_archive()` loads 130 MB of signal archives into RAM in object instances held alive for test duration. Additionally, `test_integration.py`, `test_wave1_wave2.py`, `test_pattern_ecosystem.py`, and `test_octopus_bootstrap.py` each call `create_mae()` independently — these live in pytest's single process and accumulate. The structural decomposition enables `pytest-xdist` parallelism which isolates these in separate processes, the primary memory fix.

---

## Wave 1: Source File Decomposition (Teams 1–10)

Teams 1–10 work simultaneously. No file is owned by more than one team.

---

### Team 1: Market Hooks Bootstrap

**Domain:** The `market_hooks.py` monolith — the largest file at 2,107 lines.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/market_hooks.py` — 2,107 lines

**Why it's splittable:** The file already has exactly 9 top-level functions with clean interfaces. Three functions are massive (351, 547, 603 lines) and contain nested closures with shared state passed via `ctx` — but they do not share imports with each other beyond `SimpleNamespace` and logging.

**Split strategy:**

The function boundaries are the split lines. Create three new sub-modules alongside `market_hooks.py`, then reduce `market_hooks.py` to a thin dispatcher that re-exports the three entry points.

1. `market_hooks_trades.py` — contains `_check_sweep_bypass`, `_write_paper_trade`, `_translate_and_log_executable_signal`, `_submit_to_alpaca`. These four functions (lines 33–468, ~435 lines) are all concerned with converting a convergence alert into a trade record or Alpaca submission. No shared closure state.

2. `market_hooks_eventbus.py` — contains `_register_market_eventbus` (lines 469–820, ~351 lines). Registers all endocrine coupling, hypothesis lifecycle, cascade confirmation, and backward cascade EventBus subscribers. Self-contained.

3. `market_hooks_steps.py` — contains `_register_market_step_hooks`, `_write_convergence_heartbeat`, `_run_drift_detector` (lines 821–1503, ~682 lines). This remains slightly over cap so split further: extract the heartbeat helper and drift detector into `market_hooks_steps.py` (~200 lines) and keep the main `_register_market_step_hooks` closure in `market_hooks_steps_core.py` (~500 lines).

4. `market_hooks_sensing.py` — contains `_wire_sensing_hook` (lines 1504–2107, ~603 lines). Builds the `MarketSensingHook`, all tiered alerters, the `_market_sense_hook` closure, the `_sensing_step_with_advisory` closure, and the advisory bridge. This is self-contained once `ctx._cached_alerts` exists.

5. `market_hooks.py` becomes ~50 lines: imports from all four sub-modules and re-exports `_register_market_eventbus`, `_register_market_step_hooks`, `_wire_sensing_hook` — preserving all existing import paths without change.

**Expected files:** 5 (was 1)
**Expected line counts:** 435 / 351 / ~200 / 490 / ~500 / 50 — all under 500.
**Critical constraint:** `ctx._cached_alerts` is written by `_register_market_step_hooks` and read by `_wire_sensing_hook`. The call-order comment in `market_hooks.py` must be preserved in the new dispatcher. The `ctx` object carries all shared state between functions so no hidden coupling exists.
**Dependencies on other teams:** None. All imports in this file are lazy (inside function bodies) and reference stable public APIs.

---

### Team 2: Market Intelligence Core

**Domain:** The convergence engine and related intelligence files.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/convergence_alerter.py` — 1,912 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/hypothesis_engine.py` — 885 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/hypothesis_generator.py` — 605 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/hypothesis_validator.py` — 656 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/thompson_sampler.py` — 645 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/world_model.py` — 597 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/correlation_tracker.py` — 551 lines

**Split strategies:**

`convergence_alerter.py` (1,912 lines): Three distinct responsibilities exist:

1. `convergence_models.py` — dataclasses `Signal` and `ConvergenceAlert` (lines 48–109, ~60 lines) plus the `read_discoveries` module-level function (lines 1827–1912, ~85 lines). These have zero dependencies on `ConvergenceAlerter`.

2. `convergence_confidence.py` — methods `_compute_confidence`, `_compute_effective_domain_count`, `_max_domain_correlation`, `_compute_coherence_score`, `_apply_quorum_boost`, `_get_thompson_weight`, `_compute_freshness`, `_resolve_thompson_key`, `_get_regime`, `_prune_old_signals`, `record_signal` and the two static lookup dicts `_SOURCE_TO_THOMPSON_KEY` and `_DOMAIN_SOURCES` (approximately lines 135–920, ~785 lines). Extract as a `ConvergenceConfidenceEngine` mixin that `ConvergenceAlerter` inherits. This group handles all signal intake and weight computation. Target ~480 lines.

3. `convergence_detection.py` — `check_convergence`, `_check_direction_convergence`, `check_ticker_convergence`, `check_ticker_convergence_for`, `_compute_ripple_effects`, `get_domain_status`, `get_convergence_matrix`, `get_actionable_summary`, `get_statistics`, `step`, `to_dict`, `save`, `_log_discovery` (approximately lines 1096–1912, ~816 lines). Extract as `ConvergenceDetectionEngine` mixin (~490 lines after models extracted).

4. `convergence_alerter.py` becomes the coordinator (~80 lines): defines `ConvergenceAlerter` inheriting both mixins, wires `__init__`, re-exports `Signal`, `ConvergenceAlert`, `read_discoveries` for backward compatibility.

`hypothesis_engine.py` (885 lines): Split at the meta-learning / lifecycle boundary.

1. `hypothesis_lifecycle.py` — `step`, `request_generation`, `request_validation`, `on_signal_ingested`, `_run_generation`, `_launch_validation`, `_collect_validation_results`, `_run_validation`, `_promote`, `_retire`, `get_persistence_stats` (lines 134–542, ~408 lines).

2. `hypothesis_meta_learning.py` — `_save_retirement_window`, `_load_retirement_window`, `_seed_retirement_window_from_registry`, `_check_regime`, `_review_gates`, `_run_meta_learning`, `get_statistics` (lines 477–885, ~408 lines).

3. `hypothesis_engine.py` becomes `__init__` + class shell that imports from both (~70 lines).

`hypothesis_generator.py` (605 lines): Split at composite vs. simple hypothesis generation.

1. `hypothesis_causal.py` — `_get_gen_threshold`, `_auto_generate_causal_story`, `_get_causal_story` module-level functions (lines 46–280, ~234 lines).

2. `hypothesis_generator.py` keeps `HypothesisGenerator` class (~370 lines after extraction). Still under cap.

`hypothesis_validator.py` (656 lines): Split by validation pathway.

1. `hypothesis_dsr.py` — DSR/Sharpe computation: `_compute_sharpe`, `_compute_dsr`, `_load_dsr_trials`, `_save_dsr_trials` (~100 lines).

2. `hypothesis_event_search.py` — `_find_trigger_events`, `_find_composite_trigger_events`, `_check_event_outcome` (~280 lines).

3. `hypothesis_validator.py` keeps `HypothesisValidator.__init__`, `validate`, `_validate_from_precomputed`, `get_statistics` — importing from the two new files (~280 lines).

`thompson_sampler.py` (645 lines): Split persistence from sampling logic.

1. `thompson_persistence.py` — `_load_distributions`, `_save_distributions_locked`, `_save_distributions`, `_seed_from_reliability`, `replay_from_history`, `seed_combo_distributions`, `_log_update` plus dataclasses `BetaDistribution`, `SamplingResult`, `UpdateResult` (~300 lines).

2. `thompson_sampler.py` keeps `ThompsonSampler` class body minus persistence internals + `main()` (~350 lines), importing dataclasses from persistence module.

`world_model.py` (597 lines): Split causal graph data from graph operations.

1. `world_model_chains.py` — `_seed_curated_chains`, `_get_curated_chains` (the 176-entry static chain list, lines 96–287, ~190 lines).

2. `world_model.py` keeps `WorldModel` class + dataclasses (~407 lines), importing chains via `_get_curated_chains`.

`correlation_tracker.py` (551 lines): Minor trim possible — the `seed_from_lag_data` method contains a large inline loop that could move to `correlation_persistence.py` along with `save` and `_load_state`. But at 551 lines with no natural class boundary, prefer trimming: move the `CorrelationPair` dataclass and its ~13 lines to a `correlation_models.py` that `correlation_tracker.py` imports, reducing it to ~538 lines. Still barely over — also move `find_leading_pairs` (40 lines) and its helper into `correlation_analysis.py`. Result: ~498 lines.

**Expected files:** 20 (was 7). All under 500 lines.
**Dependencies on other teams:** None — all cross-references are lazy imports or stable stable paths.

---

### Team 3: Raw Store

**Domain:** The SQLite persistence layer for raw market data.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/market/raw_store.py` — 1,835 lines

**Why it's cleanly splittable:** `RawStore` is a single class with 30 independent `store_*` methods, each 50–100 lines. Every method is a completely independent table creation + upsert. They share only `_get_conn()` and `close()`. There is zero logical coupling between `store_vix_daily` and `store_sam_opportunities`.

**Split strategy:** Group methods by data domain into 6 sub-stores. The base infrastructure (connection management) stays in `raw_store.py`. Each sub-store file defines a mixin that `RawStore` inherits.

1. `raw_store_base.py` — `RawStore.__init__`, `_get_conn`, `close` (~50 lines). This is the only file that opens SQLite connections.

2. `raw_store_market_data.py` — stores for VIX, COT, EIA, Trends, Price snapshots, FRED observations, FRED yields, USDA (~430 lines, 8 methods). Economic/price/commodities domain.

3. `raw_store_social.py` — StockTwits messages, Finnhub sentiment, Finnhub earnings, Finnhub economic events, Yahoo headlines, ApeWisdom sentiment, Finnhub WebSocket ticks (~450 lines, 7 methods). Sentiment/news domain.

4. `raw_store_government.py` — Congressional trades, Congress bills, FINRA short volume (~280 lines, 3 methods). Government/regulatory domain.

5. `raw_store_insider.py` — SEC Form 4, OpenInsider purchases, FinViz insider trades, FinViz unusual volume, EDGAR filings (~420 lines, 5 methods). Insider/institutional domain.

6. `raw_store_operational.py` — Massive bars, CoinGecko prices, CoinCap assets, Job postings, SAM opportunities (~450 lines, 5 methods). Crypto/hiring/contracts domain.

7. `raw_store.py` becomes `class RawStore(MarketDataMixin, SocialMixin, GovernmentMixin, InsiderMixin, OperationalMixin)` importing from all 5 mixins, re-exporting `RawStore` from the same path (~50 lines). Existing `from mae_core.market.raw_store import RawStore` imports unchanged.

**Expected files:** 7 (was 1). All under 500 lines.
**Dependencies on other teams:** None.

---

### Team 4: Sensing Pipeline

**Domain:** Data ingestion from all 31+ sources.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/market/sensing_hook.py` — 1,248 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/sensing_fetchers.py` — 904 lines

**Split strategies:**

`sensing_hook.py` (1,248 lines): Three distinct responsibilities.

1. `sensing_scheduler.py` — `_launch_next_fetch`, `_launch_thompson_guided`, `_launch_round_robin` (lines 676–801, ~125 lines). The scheduling logic is purely mechanical: pick next sources, fill slots.

2. `sensing_collector.py` — `_collect_results`, `_collect_one`, `_process_realtime_signals` (lines 617–973, ~356 lines). The result processing pipeline: receive future results, convert to signals, record in alerter.

3. `sensing_reactive.py` — `_trigger_reactive_convergence` (lines 974–1045, ~71 lines) and `_fetch_source` (lines 1046–1218, ~172 lines). The fetch dispatch table and reactive convergence trigger.

4. `sensing_hook.py` becomes `MarketSensingHook` class with `__init__`, `step`, `_evaluate_outcomes`, `get_statistics`, `shutdown` — importing helpers from the three new files (~340 lines). The two module-level helpers `_absence_source_to_domain` and `_build_domain_to_sources` stay in `sensing_hook.py` as they're used by `MarketSensingHook.__init__`.

`sensing_fetchers.py` (904 lines): Contains 35+ module-level `fetch_*` functions, each 20–60 lines. Clean split by data domain:

1. `fetchers_insider.py` — `fetch_sec_form4`, `fetch_sec_form8k`, `fetch_openinsider`, `fetch_13f_holdings`, `fetch_finviz` (~180 lines). Insider/institutional domain.

2. `fetchers_government.py` — `fetch_congressional`, `fetch_senate`, `fetch_congress_legislation`, `fetch_usa_spending`, `fetch_sam_gov` (~120 lines). Government domain.

3. `fetchers_market_data.py` — `fetch_fred`, `fetch_fred_yields`, `fetch_cot`, `fetch_vix`, `fetch_eia`, `fetch_massive_snapshot`, `fetch_usda` (~200 lines). Economic/price domain.

4. `fetchers_technical.py` — `fetch_ta_indicators`, `fetch_session_sweep`, `fetch_fractal_resonance`, `fetch_order_flow` (~130 lines). Technical analysis domain.

5. `fetchers_social.py` — `fetch_social_sentiment`, `fetch_stocktwits`, `fetch_trends`, `fetch_social_text`, `fetch_yahoo_rss`, `fetch_finnhub`, `fetch_finnhub_extras` (~200 lines). Sentiment/social domain.

6. `fetchers_crypto.py` — `fetch_crypto_prices`, `fetch_crypto_exchange`, `fetch_finra_short`, `fetch_economic_calendar` (~80 lines). Crypto/market-structure domain.

7. `sensing_fetchers.py` becomes ~50 lines: imports all `fetch_*` functions from the 6 sub-modules, re-exports them. Callers use `from mae_core.market.sensing_fetchers import fetch_*` unchanged.

**Expected files:** 11 (was 2). All under 500 lines.
**Dependencies on other teams:** None.

---

### Team 5: Backbone Infrastructure

**Domain:** Core architectural backbone systems.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/connection_registrations.py` — 1,208 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/fractal_act.py` — 962 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/connection_registry.py` — 741 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/holon_protocol.py` — 704 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/integration_meter.py` — 652 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/triad_enforcer.py` — 541 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/backbone/triad_registry.py` — 550 lines

**Split strategies:**

`connection_registrations.py` (1,208 lines): A single `register_all_connections` function containing 14 labeled groups. Split by group clusters:

1. `connection_registrations_core.py` — Groups 1–5 (EventBus pub/sub + Groups 1-5, lines 69–840, ~771 lines). Still over — split: Groups 1–3 in `connection_registrations_bio.py` (~430 lines), Groups 4–5 in `connection_registrations_agent.py` (~300 lines).

2. `connection_registrations_patterns.py` — Groups 6–9 (cross-system wiring, patterns, biological step hooks, GNN handlers, lines 841–1060, ~220 lines).

3. `connection_registrations_advanced.py` — Groups 10–14 (auto-redifferentiation, mitosis, unregistered channels, autopoietic closure, emergent circuits, lines 1061–1208, ~148 lines).

4. `connection_registrations.py` becomes a dispatcher function (~60 lines) that calls all three sub-module registration functions in order, presenting the same `register_all_connections` public API.

`fractal_act.py` (962 lines): Four classes with natural boundaries.

1. `fractal_act_subsystem.py` — `SubsystemAction` (lines 31–221, ~190 lines) and `OrganClusterAction` (lines 222–408, ~186 lines). Both operate at sub-organ scale. ~376 lines total.

2. `fractal_act_organ.py` — `OrganAction` (lines 409–582, ~173 lines). Mid-scale.

3. `fractal_act_organism.py` — `OrganismAction` (lines 583–887, ~304 lines) and `build_fractal_action` factory (lines 888–962, ~74 lines). Top scale. ~378 lines total.

4. `fractal_act.py` becomes a re-export hub: `from .fractal_act_subsystem import SubsystemAction, OrganClusterAction` etc. (~20 lines). All existing import paths unchanged.

`connection_registry.py` (741 lines): Split data model from operations.

1. `connection_registry_models.py` — `ConnectionType`, `ConnectionCriticality`, `EnforcementMode`, `ConnectionTriad` (~110 lines). Pure dataclasses/enums.

2. `connection_registry.py` keeps `ConnectionRegistry` (~630 lines) — still over. Extract query/reporting methods: `verify_all`, `get_coverage_report`, `get_euler_statistics`, `check_euler_invariant`, `get_bare_dyads`, `get_unhealthy_connections` into `connection_registry_analysis.py` (~200 lines). `connection_registry.py` drops to ~430 lines.

3. `connection_registry.py` imports from `connection_registry_models.py` and `connection_registry_analysis.py`, re-exports everything.

`holon_protocol.py` (704 lines): Four natural classes.

1. `holon_registry.py` — `HolonEntry`, `HolonRegistry` (lines 34–271, ~237 lines).

2. `holon_proxy.py` — `HolonProxy` (lines 273–593, ~320 lines).

3. `awareness_pulse.py` — `AwarenessPulse` (lines 594–704, ~110 lines).

4. `holon_protocol.py` becomes re-export hub: `from .holon_registry import HolonEntry, HolonRegistry` etc. (~15 lines). All existing imports unchanged.

`integration_meter.py` (652 lines): Split computation from reporting.

1. `integration_meter_phi.py` — `_enumerate_bipartitions`, `_compute_phi`, `_compute_entropy`, `_compute_joint_entropy`, `_state_to_scalar` (~160 lines). Pure math.

2. `integration_meter_blanket.py` — `_compute_markov_blanket` (~87 lines). Separate algorithm.

3. `integration_meter.py` keeps `IntegrationMeter` class with `__init__`, `step`, `_collect_states`, `_compute_and_publish`, `get_latest_report`, `get_state`, `get_statistics` — importing computation from the two new files (~405 lines). Under cap.

`triad_enforcer.py` (541 lines): Minor split.

1. `triad_enforcer_models.py` — `ProcessCriticality`, `ValidatorType`, `Validator`, `ProcessTriad`, `VoteResult` (lines 49–160, ~111 lines). Dataclasses/enums.

2. `triad_enforcer.py` keeps `TriadEnforcer` — imports models from above (~430 lines). Under cap.

`triad_registry.py` (550 lines): Clean split of data vs. wiring.

1. `triad_registry.py` keeps `register_all_triads` function (lines 33–508, ~475 lines). The main registration logic. Under cap.

2. `triad_wiring.py` — `wire_triad_systems` (lines 509–550, ~41 lines). Wiring function is entirely separate. Import and re-export from `triad_registry.py` for backward compat.

**Expected files:** 23 (was 7). All under 500 lines.
**Dependencies on other teams:** None. All cross-references are stable public paths.

---

### Team 6: Agent and Coordination Systems

**Domain:** Agent lifecycle, organism coordination, and biological systems.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/agents/mixins/episodic_memory.py` — 1,149 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/agents/lifecycle_decision.py` — 705 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/coordination/organism_state.py` — 802 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/coordination/endocrine_system.py` — 674 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/learning/haven.py` — 502 lines

**Split strategies:**

`episodic_memory.py` (1,149 lines): The mixin has five distinct memory mechanisms, each marked by docstring sections:

1. `episodic_memory_core.py` — `_init_episodic_memory`, `store_experience`, `should_consolidate`, `consolidate_memory`, `learn_from_memory`, `_learn_from_batch`, `_use_generative_memory`, `get_episodic_memory_statistics`, `_serialize_episodic_memory`, `_restore_episodic_memory` (~450 lines). The foundational storage and retrieval.

2. `episodic_memory_reconsolidation.py` — `_check_reconsolidation`, `_reconsolidate_memory`, `_tick_reconsolidation` (lines 499–674, ~175 lines). Memory reconsolidation mechanism.

3. `episodic_memory_activation.py` — `_spread_activation`, `_apply_activation_boost`, `_decay_spreading_activation` (lines 675–889, ~214 lines). Spreading activation mechanism.

4. `episodic_memory_search.py` — `tick_episodic_memory`, `search_similar_experiences`, `get_counterfactual_experiences`, `verify_recall` (~210 lines). Search and verification.

5. `episodic_memory.py` becomes the final `EpisodicMemoryMixin` class that inherits all four sub-mixins (~25 lines). Backward compatible.

`lifecycle_decision.py` (705 lines): The mixin handles three act phases.

1. `lifecycle_inhibit_decide.py` — `_inhibit`, `_decide`, `_route_with_advisory` (lines 24–397, ~373 lines). Decision phase logic.

2. `lifecycle_act.py` — `_act`, `_act_explore`, `_act_exploit`, `_act_communicate`, `_act_rest`, `_act_api_call` (lines 398–705, ~307 lines). Action phase logic.

3. `lifecycle_decision.py` becomes `DecisionActionLifecycleMixin(InhibitDecideMixin, ActMixin)` (~25 lines).

`organism_state.py` (802 lines): The class aggregates dozens of EventBus callbacks.

1. `organism_state_subscriptions.py` — `_subscribe_all` and all `_on_*` callbacks (lines 187–481, ~294 lines). Pure event wiring, no state.

2. `organism_state_outputs.py` — `get_body_state`, `get_reflex_override`, `get_decision_context`, `report_action_outcome`, `get_statistics`, `serialize`, `restore`, `_parse_message` (lines 500–802, ~302 lines). State query and serialization.

3. `organism_state.py` keeps `OrganismState.__init__` and `step` — imports from both above (~210 lines). Clean because subscriptions just call `self.subscribe()` and outputs read `self._*` attributes.

`endocrine_system.py` (674 lines): Split consumer registration from core hormone logic.

1. `endocrine_consumers.py` — the 14 `register_*` methods plus `subscribe` (lines 360–503, ~143 lines). Pure dependency injection.

2. `endocrine_step.py` — `step`, `_apply_cascades`, `set_circadian_phase`, internal callbacks `_on_phi_measurement`, `_on_healing_phase_changed` (lines 536–661, ~125 lines).

3. `endocrine_system.py` keeps `HormoneType`, `HormoneConfig`, `EndocrineSystem.__init__`, the main hormone accessors (`release_hormone`, `suppress_hormone`, `get_level`, `get_global_state`, `is_stressed`, `get_exploration_bias`, etc.), `get_statistics` — imports from the two new files (~410 lines). Under cap.

`haven.py` (502 lines): Minor split.

1. `haven_validators.py` — `validate_decision`, `validate_modification`, `validate_healing`, `validate_policy`, `validate_threat` (lines 295–497, ~202 lines). The validator interface methods.

2. `haven.py` keeps `RiskLevel`, `InterventionType`, `ContagionStatus`, `RiskAssessment`, `ContagionReport`, `HavenRiskCoordinator.__init__`, risk assessment, detection, and execution methods (~300 lines). Under cap.

**Expected files:** 18 (was 5). All under 500 lines.
**Dependencies on other teams:** None.

---

### Team 7: Market Edge Detectors

**Domain:** Technical analysis, sweep detection, cluster detection, and archaeology.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/market/edge/sweep_backtest.py` — 953 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/edge/session_sweep_detector.py` — 800 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/edge/ta_indicators.py` — 754 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/edge/cluster_detector.py` — 730 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/archaeology/historical_fetcher.py` — 764 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/archaeology/pattern_library.py` — 555 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/edge/politician_tracker.py` — 505 lines

**Split strategies:**

`sweep_backtest.py` (953 lines): Backtest engine vs. reporting vs. CLI.

1. `sweep_backtest_models.py` — `Level`, `FVGZone`, `SweepEvent`, `Trade` (lines 70–130, ~60 lines). Dataclasses.

2. `sweep_backtest_engine.py` — `SweepBacktester` class (lines 131–680, ~549 lines). Still over — split: `fetch_candles`, `get_session_levels`, `detect_sweeps`, `_find_sweep`, `find_fvgs`, `find_ifvg`, scoring helpers in `sweep_backtest_engine.py` (~350 lines); `simulate_trade`, `backtest_symbol`, `run` in `sweep_backtest_runner.py` (~200 lines).

3. `sweep_backtest_report.py` — `report` function (lines 681–851, ~170 lines).

4. `sweep_backtest.py` becomes re-export hub + `main()` CLI entry point (~70 lines). `main()` stays in the original file since it's rarely imported.

`session_sweep_detector.py` (800 lines): Separate signal building from geometry detection.

1. `session_sweep_models.py` — `SessionLevel`, `FairValueGap`, `SessionSweepSignal` (lines 66–151, ~85 lines).

2. `session_sweep_geometry.py` — `_mark_session_levels`, `_extract_session_levels`, `_find_fvg`, `_find_ifvg`, `_score_displacement`, `_compute_atr`, `_get_kill_zone_score`, `_compute_quality`, `_is_kill_zone` (lines 270–799, ~529 lines). Split: geometry in `session_sweep_geometry.py` (~300 lines), scoring/quality in `session_sweep_scoring.py` (~230 lines).

3. `session_sweep_detector.py` keeps `SessionSweepDetector.__init__`, `detect_sweeps`, `_fetch_1m_candles`, `_detect_sweep`, `_confirm_sweep_reversal`, `_build_signal`, `_calculate_stop_and_target`, `_deduplicate` (~370 lines). Imports from models and geometry. Under cap.

`ta_indicators.py` (754 lines): Split by indicator type.

1. `ta_models.py` — `TASignalBase`, `RSISignal`, `MACDSignal`, `BollingerSignal`, `StructureSignal`, `CandleSignal`, `_extract_ohlcv` (lines 39–123, ~84 lines).

2. `ta_oscillators.py` — `compute_rsi`, `compute_macd`, `compute_bollinger` (lines 124–370, ~246 lines). Oscillators.

3. `ta_structure.py` — `compute_market_structure`, `compute_candlestick_patterns`, `compute_atr` (lines 371–660, ~289 lines). Structure/candle analysis.

4. `ta_indicators.py` keeps `compute_all` dispatcher (~50 lines) + imports from all three. All existing callers use `from mae_core.market.edge.ta_indicators import compute_*` unchanged.

`cluster_detector.py` (730 lines): Two natural classes.

1. `cluster_detector.py` keeps `InsiderInCluster`, `InsiderRelationship`, `ClusterSignal`, `ClusterDetector` (lines 43–421, ~378 lines). Core detection.

2. `relationship_tracker.py` — `RelationshipTracker` (lines 422–616, ~194 lines) and module-level helpers `store_cluster_signal`, `scan_all_symbols` (lines 617–730, ~113 lines). ~307 lines total.

`historical_fetcher.py` (764 lines): Split TA computation from API fetching.

1. `historical_ta.py` — `_compute_ta_signals`, `_compute_rsi_series`, `_compute_macd_series`, `_compute_bollinger_series`, `_compute_volume_signals`, `_get_ta_cached` (lines 116–490, ~374 lines). All pure computation on price data.

2. `historical_fetcher.py` keeps `HistoricalDataFetcher.__init__`, `preload_archive`, `fetch_all`, `_fetch_sec_signals`, `_fetch_fred_signals`, `_fetch_cot_signals`, `_fetch_congressional_signals`, `_load_archive_signals`, `_load_signals_for_date`, `_make_signal`, `clear_cache`, `clear_all_caches` (~390 lines). Imports TA functions from `historical_ta.py`.

`pattern_library.py` (555 lines): Split fingerprint I/O from template operations.

1. `pattern_library_io.py` — `_persist_fingerprints_dict`, `_persist_templates`, `rebuild_templates`, `_load`, `_load_fingerprints`, `store`, `store_batch`, `store_template`, `store_templates`, `get`, `get_template`, `size`, `template_count` (~270 lines). Storage operations.

2. `pattern_library.py` keeps `PatternMatch`, `PatternLibrary.__init__`, `query_similar`, `update_outcome`, `get_statistics`, `clopper_pearson_ci`, `_update_template`, `_source_domain`, `_sources_to_domains` (~285 lines). Query and analysis.

`politician_tracker.py` (505 lines): Minor split.

1. `politician_models.py` — `PoliticianProfile`, `CorrelationSignal`, `_load_congress_members` (lines 30–151, ~121 lines). Data model + static member list loader.

2. `politician_tracker.py` keeps `PoliticianTracker` + module-level helpers (~384 lines). Under cap.

**Expected files:** 22 (was 7). All under 500 lines.
**Dependencies on other teams:** None. `historical_ta.py` uses only numpy/pandas (already imported in original).

---

### Team 8: Emergent and Pattern Systems

**Domain:** Self-organization, pattern recognition, global workspace, somatic map, auto-healer.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/emergent/auto_healer.py` — 861 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/emergent/somatic_map.py` — 754 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/patterns/pattern_consolidator.py` — 797 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/patterns/global_workspace.py` — 669 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/patterns/pattern_cortex.py` — 525 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/defense/threat_detector.py` — 529 lines

**Split strategies:**

`auto_healer.py` (861 lines): Split healing phases from monitoring.

1. `auto_healer_models.py` — `HealingPhase`, `FailureType`, `FailureReport`, `HealingAction`, `HealingRecord` (lines 51–108, ~57 lines).

2. `auto_healer_phases.py` — `_phase_isolate`, `_phase_assess`, `_phase_restore`, `_phase_verify`, `_execute_healing`, `_register_defaults`, `register_recovery` (lines 534–769, ~235 lines). The four healing phases as a standalone unit.

3. `auto_healer.py` keeps `AutoHealer.__init__`, `step`, `_self_monitor`, `register_self_healing_triad`, `report_failure`, `_on_risk_alert`, `_on_starvation`, `set_cortisol_priority`, `set_hormone_level`, `_publish_phase`, `get_statistics`, `get_active_healings`, `get_healing_history` (~565 lines). Still over — extract `_self_monitor` (lines 300–404, ~104 lines) into `auto_healer_monitor.py`. `auto_healer.py` drops to ~465 lines. Imports phases and monitor from sub-modules.

`somatic_map.py` (754 lines): Split blast-radius analysis from system registry.

1. `somatic_map_models.py` — `SystemCriticality`, `ModificationVerdict`, `SystemNode`, `BlastRadiusReport`, `ModificationRecord` (lines 55–121, ~66 lines).

2. `somatic_map_blast.py` — `analyze_blast_radius`, `_compute_risk`, `_determine_verdict`, `propose_modification`, `execute_modification`, `complete_modification`, `rollback_modification`, `_rollback` (lines 286–573, ~287 lines). Modification safety analysis.

3. `somatic_map.py` keeps `SomaticMap.__init__`, `register_system`, `register_all_systems`, `add_dependency`, `register_snapshot_provider`, `heartbeat`, `get_system_info`, `get_all_systems`, `get_dependency_chain`, `get_critical_path`, `get_unhealthy_systems`, `register_all_bootstrap_systems`, EventBus callbacks, `get_statistics`, `get_body_map` (~400 lines). Imports blast analysis.

`pattern_consolidator.py` (797 lines): Split pattern extraction from the main consolidation loop.

1. `pattern_consolidator_extractors.py` — `_extract_trend_patterns`, `_extract_meta_patterns`, `_extract_insight_patterns`, `_extract_distilled_patterns`, `_store_pattern`, `_get_contributing_agents`, `_get_co_occurring_domains` (lines 555–785, ~230 lines). Pure extraction logic.

2. `pattern_consolidator_selection.py` — `_competitive_select`, `_compute_fitness`, `_get_emotional_weight`, `_apply_synaptic_downscaling` (lines 316–553, ~237 lines). Selection algorithm.

3. `pattern_consolidator.py` keeps `_SignalExperience`, `PatternConsolidator.__init__`, `consolidate`, `get_statistics`, `__repr__` (~330 lines). Imports from both.

`global_workspace.py` (669 lines): Split candidate management from ignition.

1. `global_workspace_models.py` — `WorkspaceCandidate`, `WorkspaceItem`, `IgnitionResult` (lines 106–171, ~65 lines).

2. `global_workspace_competition.py` — `compete`, `_add_to_workspace`, `_attempt_chunking`, `_update_candidates`, `_check_corroboration`, `_get_activation_map`, `_get_refractory_domains` (~280 lines). Competition mechanism.

3. `global_workspace.py` keeps `GlobalWorkspace.__init__`, `get_workspace_contents`, `_get_workspace_content_names`, `_on_phi_measurement`, `get_statistics`, `__repr__` (~325 lines). Imports from models and competition sub-modules.

`pattern_cortex.py` (525 lines): Minor split.

1. `pattern_cortex_detection.py` — `_detect_trends`, `_detect_meta_patterns`, `_recall_ancestral`, `_compute_domain_level`, `_update_domain_streaks` (lines 219–364, ~145 lines). Pattern detection internals.

2. `pattern_cortex.py` keeps `PatternAdvisory`, `PatternCortex.__init__`, `window_size`, `process_digest`, `_generate_insights`, `_recommend_tier`, `_compute_confidence`, `_publish_gwt_broadcast`, `get_statistics`, `get_recent_advisories`, `__repr__` (~380 lines). Imports detection from sub-module.

`threat_detector.py` (529 lines): Minor split.

1. `threat_detector_models.py` — `ThreatLevel`, `DefenseStrategy`, `Threat`, `DefenseResponse` (lines 39–91, ~52 lines).

2. `threat_detector.py` keeps `ThreatDetector` (~477 lines). Imports models. Under cap.

**Expected files:** 20 (was 6). All under 500 lines.
**Dependencies on other teams:** None.

---

### Team 9: Bootstrap Orchestration

**Domain:** Core bootstrap files that exceeded the cap.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/wiring.py` — 814 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/bio_market_wiring.py` — 661 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/bio_market_wiring_extended.py` — 711 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/organs.py` — 593 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/patterns.py` — 518 lines

**Split strategies:**

`wiring.py` (814 lines): The single `bootstrap_wiring` function contains layers 14–21. Split by layer cluster:

1. `wiring_layers_14_16.py` — Layer 14 step hooks, Layer 15 EventBus cross-wiring, Layer 15b HAVEN, Layer 16 endocrine consumers (lines 47–461, ~414 lines).

2. `wiring_layers_17_21.py` — Layer 17 SomaticMap registration, Layer 18 ConnectionRegistry, Layer 18b WitnessNotifier, Layer 19 Bidirectional Awareness, Layer 20 Fractal Generator, Layer 21 Stem Cell Registry (lines 462–814, ~352 lines).

3. `wiring.py` becomes `bootstrap_wiring` as a two-line caller + `_register_somatic_systems` helper (~45 lines).

`bio_market_wiring.py` (661 lines): Split into two files of ~7 wiring functions each.

1. `bio_market_wiring_a.py` — `_wire_emotional_system`, `_wire_homeostasis`, `_wire_arousal`, `_wire_curiosity`, `_wire_nociception`, `_wire_metacognition`, `_wire_threat_detector` (lines 107–338, ~231 lines).

2. `bio_market_wiring_b.py` — `_wire_quorum`, `_wire_circadian`, `_wire_haven`, `_wire_inhibition`, `_wire_memory_consolidator`, `_wire_collective_dream`, `_wire_stigmergy` (lines 395–661, ~266 lines).

3. `bio_market_wiring.py` keeps `_parse`, `wire_bio_systems_to_market` orchestrator + `_register_somatic_systems` (~80 lines), calling functions from both sub-modules.

`bio_market_wiring_extended.py` (711 lines): Same pattern.

1. `bio_market_wiring_extended_a.py` — `_wire_digestive`, `_wire_circulatory`, `_wire_lymphatic`, `_wire_microbiome`, `_wire_renal_filter`, `_wire_senescence`, `_wire_morphogenesis`, `_wire_reproductive` (lines 83–449, ~366 lines).

2. `bio_market_wiring_extended_b.py` — `_wire_pearl_defense`, `_wire_respiratory`, `_wire_thermoregulation`, `_wire_vestibular`, `_wire_proprioception`, `_wire_energy_reserve`, `_wire_predictive_field` (lines 451–711, ~260 lines).

3. `bio_market_wiring_extended.py` keeps `_parse`, `wire_bio_systems_extended` orchestrator (~50 lines).

`organs.py` (593 lines): Split by layer.

1. `organs_layers_26_27.py` — Layer 26 (metabolic systems) + Layer 27 (social cognition + sensory), lines 65–237, ~172 lines.

2. `organs_layers_28_30.py` — Layer 28 (maintenance/growth/boundary) + Layer 29 (organism state + deep integration) + Layer 30 (lifecycle step systems), lines 238–593, ~355 lines.

3. `organs.py` becomes `bootstrap_organs` orchestrator + `_register_somatic_systems` (~50 lines).

`patterns.py` (518 lines): Split by layer.

1. `patterns_layers_22_25.py` — Layers 22–25d, lines 41–480, ~439 lines.

2. `patterns.py` becomes `bootstrap_patterns` orchestrator (~80 lines) calling sub-module function. Import-compatible.

**Expected files:** 17 (was 5). All under 500 lines.
**Dependencies on other teams:** None. All cross-references use lazy imports inside function bodies.

---

### Team 10: Remaining Source Files

**Domain:** All remaining source files over 500 lines not covered by teams 1–9.

**Files to decompose:**
- `/c/Users/baenb/projects/MIDGE/mae_core/substrate/mycelial_substrate.py` — 618 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/morphogenesis/organ_builder.py` — 609 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/memory/deep_memory.py` — 607 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/cognition/decision_router.py` — 605 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/network/octopus_colony.py` — 560 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/planning/temporal_memory.py` — 559 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/apis/finnhub_client.py` — 787 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/apis/fred_client.py` — 558 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/apis/house_stock_watcher.py` — 550 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/apis/sec_edgar/client.py` — 589 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/apis/price_fetcher.py` — 521 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/network/market_task_handlers.py` — 519 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/signal_adapters/wave2_3.py` — 564 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/substrate/circulatory_system.py` — 534 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/planning/worldline_planner.py` — 546 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/market_actions.py` — 508 lines
- `/c/Users/baenb/projects/MIDGE/mae_core/morphogenesis/reproductive_system.py` — 509 lines

**Split strategies:**

`mycelial_substrate.py` (618 lines): Split topology building from signaling.

1. `mycelial_topology.py` — `_build_topology`, `_find_or_create_node`, `_auto_connect`, `grow_node`, `prune_node`, `isolate_region`, `restore_region` (~230 lines). Structural operations.

2. `mycelial_substrate.py` keeps `MycelialSubstrate.__init__`, `register_agent`, `deregister_agent`, `propagate_signal`, `get_signal_path`, `get_peers`, `get_agent_position`, `get_all_agent_positions`, `get_topology_graph`, `step`, `_publish_health_report`, `get_health_report`, `set_phase` (~390 lines). Under cap.

`organ_builder.py` (609 lines): Split design from build/lifecycle.

1. `organ_builder_design.py` — `design_organ`, `_on_system_senescent` (lines 274–548, ~274 lines). Design algorithm + dataclasses `ProblemSignature`, `OrganBlueprint`, `Organ`.

2. `organ_builder.py` keeps `OrganBuilder.__init__`, `grow_organ`, `dissolve_organ`, `prune_organs`, `get_organ`, `get_all_organs`, `active_organ_count`, `get_statistics`, `_connect_organ_agents` plus models (~335 lines). Imports design from above.

`deep_memory.py` (607 lines): Split embedding/search from storage.

1. `deep_memory_search.py` — `search`, `search_with_filter`, `embed_text`, `embed_texts_batch`, `embed_sparse` (~175 lines). Query/embedding operations.

2. `deep_memory.py` keeps `QdrantConfig`, `SearchResult`, `DeepMemoryStore.__init__`, `ping`, `is_available`, `ensure_collections`, `_ensure_indexes`, `get_collection_stats`, `store_point`, `store_points_batch`, `compute_witness_hash`, `__repr__`, `_sanitize_payload`, `_sparse_embed`, `hash_token` (~432 lines). Imports search methods.

`decision_router.py` (605 lines): Split reflexes from planning.

1. `decision_reflexes.py` — `_check_reflex`, `_check_habit`, `_track_for_habit_formation`, `_register_default_reflexes`, `register_reflex`, `register_habit` (~180 lines). Reflex/habit subsystem.

2. `decision_router.py` keeps `DecisionTier`, `ReflexPattern`, `Habit`, `RouterDecision`, `DecisionRouter.__init__`, `route_decision`, `executive_override`, `set_reflex_bias`, `get_performance_metrics`, `get_tier_health`, `_invoke_prefrontal`, `_force_tier`, `_create_decision`, `__repr__` (~425 lines). Imports reflexes.

`octopus_colony.py` (560 lines): Split monitoring from colony management.

1. `octopus_monitoring.py` — `_monitoring_loop`, `_check_health`, `_check_workload_scaling`, `_publish_health_report` (~85 lines).

2. `octopus_connections.py` — `_establish_peer_connections` (~33 lines).

3. `octopus_colony.py` keeps core class (~442 lines). Under cap.

`temporal_memory.py` (559 lines): Split causal discovery from event storage.

1. `temporal_causal.py` — `_discover_causal_links`, `_check_patterns`, `trace_causal_chain`, `find_common_causes`, `predict_next_event_type` (~190 lines). Causal reasoning algorithms.

2. `temporal_memory.py` keeps `EventType`, `FourDEvent`, `TemporalPattern`, `CausalChain`, `TemporalMemory.__init__`, `record_event`, `_link_temporal_neighbors`, `_evict_oldest`, `get_event`, `query_*`, `get_recent`, `get_patterns`, `get_statistics`, `get_timeline` (~370 lines). Under cap.

`finnhub_client.py` (787 lines): Split data models from client.

1. `finnhub_models.py` — `NewsSentiment`, `EconomicEvent`, `AnalystRec`, `EarningsEvent` dataclasses with their methods (lines 41–197, ~156 lines).

2. `finnhub_parsers.py` — `_parse_sentiment`, `_parse_earnings_calendar`, `_parse_economic_calendar`, `_parse_analyst_recommendations` (lines 550–720, ~170 lines). Parse-only logic.

3. `finnhub_client.py` keeps `FinnhubClient` with `__init__`, all public `get_*` methods, `_rate_limit`, `_get`, plus module-level helpers — imports models and parsers (~462 lines). Under cap.

`fred_client.py` (558 lines): Split series definitions from client.

1. `fred_series.py` — `_determine_direction`, module-level `SERIES_CONFIG` dict (lines 54–162, ~108 lines). Pure data.

2. `fred_client.py` keeps `MacroIndicator`, `FREDClient`, `get_macro_snapshot` module helper — imports from `fred_series.py` (~450 lines). Under cap.

`house_stock_watcher.py` (550 lines): Split data normalization from client.

1. `house_stock_normalizer.py` — `_normalize_trade`, `parse_amount_range` (lines 84–250, ~166 lines). Transform logic.

2. `house_stock_watcher.py` keeps `CongressionalTrade`, `HouseStockWatcherClient`, query methods, `get_recent_trades` module helper — imports normalizer (~384 lines). Under cap.

`sec_edgar/client.py` (589 lines): Split parsing from fetching.

1. `sec_edgar_parsers.py` — `_parse_form4_html`, `_parse_transaction`, `parse_form8k` (lines 333–589, ~256 lines). HTML/XML parsing.

2. `sec_edgar/client.py` keeps `_ResponseShim`, `SECEdgarClient.__init__`, `_rate_limit`, `_get`, `get_company_cik`, `get_company_filings`, `parse_form4` — imports parsers (~333 lines). Under cap.

`price_fetcher.py` (521 lines): Minor split.

1. `price_fetcher_helpers.py` — `get_price`, `get_prices`, `price_fetcher_for_outcomes` module-level functions + `_fetch_yfinance`, `_fetch_alpha_vantage` (~145 lines).

2. `price_fetcher.py` keeps `PriceData`, `PriceFetcher` with core methods (~376 lines). Imports helpers. Under cap.

`market_task_handlers.py` (519 lines): Split task factories from handler injection.

1. `market_task_factories.py` — `_make_investigate_partial`, `_make_archaeology_lookup`, `_make_situation_check` (lines 285–519, ~234 lines). Factory functions.

2. `market_task_handlers.py` keeps `select_preferred_role`, `inject_market_handlers`, `patch_new_arm`, `_patch_arm` (~285 lines). Imports factories.

`wave2_3.py` (564 lines): Split by adapter group.

1. `wave2_3_insider.py` — `from_openinsider`, `from_13f_holding`, `from_activist_filing`, `from_finviz_insider` (~145 lines).

2. `wave2_3_technical.py` — `from_finviz_unusual_volume`, `from_finviz_short_squeeze`, `from_massive_snapshot`, `from_suppression_event` (~155 lines).

3. `wave2_3.py` keeps `from_crypto_signal`, `from_finnhub_realtime` + re-exports all for backward compat (~90 lines, plus imports from 2 new files). All `from .wave2_3 import` paths unchanged.

`circulatory_system.py` (534 lines): Minor split.

1. `circulatory_models.py` — `ResourcePacket`, `DemandSignal` (~16 lines).

2. `circulatory_system.py` keeps `CirculatorySystem` — imports models (~518 lines). Still tight. Also extract `_distribute` logic into `circulatory_distribution.py` (~77 lines). `circulatory_system.py` drops to ~441 lines.

`worldline_planner.py` (546 lines): Split projection from planning.

1. `worldline_projection.py` — `_project_worldline`, `_select_action`, `_score_worldlines`, `plan_multi_horizon` (lines 265–459, ~194 lines). Projection algorithms.

2. `worldline_planner.py` keeps `WorldlineStatus`, `WorldlinePoint`, `Worldline`, `PlanningResult`, `WorldlinePlanner.__init__`, `plan`, `begin_execution`, `check_divergence`, `complete_worldline`, `abandon_worldline`, `get_temporal_context`, `_on_pattern_detected`, `replan_pending`, `acknowledge_replan`, `get_statistics` (~352 lines). Imports projection.

`market_actions.py` (508 lines): Minor split.

1. `market_actions_hypothesis.py` — `_hypothesis_generate`, `_hypothesis_deepen`, `_hypothesis_sample`, `_hypothesis_validate`, `_log_hypothesis_activity` (lines 307–481, ~174 lines). Hypothesis action group.

2. `market_actions.py` keeps routing dispatch, SEC/contract/convergence/broadcast actions, logging helper (~334 lines). Imports hypothesis actions.

`reproductive_system.py` (509 lines): Minor split.

1. `reproductive_system.py` is 509 lines — only 9 lines over. The `_parse_message` static method (lines 498–509) plus `serialize`/`restore` (lines 450–497) can move to `reproductive_persistence.py` (~60 lines), dropping the main file to ~449 lines.

**Expected files:** 47 (was 17). All under 500 lines.
**Dependencies on other teams:** None.

---

## Barely-Over Files (500–560 lines): Full Split vs. Trim Decision

For the following files at the boundary, the decision is **full split** for every file. Trimming (removing docstrings, collapsing imports) is unreliable and creates hidden tech debt. Full structural splits are mechanical, testable, and permanent. At 9 lines over, `reproductive_system.py` earns only a micro-split to avoid unnecessary churn on a stable file.

The key principle: use `__init__.py` re-exports or a thin "hub" file to preserve all existing import paths. Callers do not change. Tests do not change. Only the file that contained the code changes.

---

## Wave 2: Test File Decomposition (Teams 11–12)

Wave 2 begins after Wave 1 source splits are merged and all tests pass. Test splits MUST happen after source splits because tests reference the original module paths — and those paths must be confirmed stable first.

---

### Team 11: Critical Test Files (1,000+ lines)

**Files to decompose (7 files, 7,613 lines):**

- `test_session_sweep_detector.py` — 1,296 lines
- `test_new_sources.py` — 1,181 lines
- `test_wave2_3_integration.py` — 1,118 lines
- `test_absence_monitor.py` — 947 lines
- `test_signal_freshness_and_combination.py` — 924 lines
- `test_holon_protocol.py` — 869 lines
- `test_integration.py` — 854 lines

**Strategy:** Each test file contains multiple `TestCase` classes with clear subject boundaries. Split by test class groups:

`test_session_sweep_detector.py` → 3 files by functional area:
- `test_session_sweep_geometry.py` — `TestGracefulDegradation`, `TestFetchCandles`, `TestSessionLevelMarking` (~280 lines).
- `test_session_sweep_detection.py` — `TestSweepDetection`, `TestFVGDetection`, `TestKillZoneDetection`, `TestClassifySweepQuality`, `TestStopAndTarget`, `TestDeduplication` (~495 lines).
- `test_session_sweep_adapters.py` — `TestFromSessionSweepAdapter`, `TestPlainLanguage`, `TestFullPipelineIntegration`, `TestIFVGDetection`, `TestPatternStackingScores`, `TestIFVGSignalAdapter` (~520 lines). Split further into `test_session_sweep_pipeline.py` (~270) + `test_session_sweep_ifvg.py` (~250).

`test_new_sources.py` → 3 files by source type:
- `test_new_sources_cot_vix_trends.py` — COT, VIX, Trends tests (~450 lines).
- `test_new_sources_finnhub.py` — Finnhub economic/analyst/calendar tests (~380 lines).
- `test_new_sources_adapters.py` — Adapter tests for all new sources (~350 lines).

`test_wave2_3_integration.py` → 3 files:
- `test_wave2_3_crypto_insider.py` — `TestCryptoAdapters`, `TestInsiderAdapters` (~280 lines).
- `test_wave2_3_institutional.py` — `TestInstitutionalAdapters`, `TestTechnicalAdapters` (~280 lines).
- `test_wave2_3_integration.py` — `TestSuppressionAdapter`, `TestFetchFunctions`, `TestSensingHookIntegration`, `TestMarketClockAvailability`, `TestThompsonKeys`, `TestSuppressionIntegration`, `TestWebSocketIntegration` (~560 lines) → split to `test_wave2_3_hooks.py` (~280) + `test_wave2_3_suppression.py` (~280).

`test_absence_monitor.py` → 2 files:
- `test_absence_monitor_core.py` — `TestCadenceLearning`, `TestAbsenceDetection`, `TestEventBusPublishing`, `TestArchiveBootstrap` (~385 lines).
- `test_absence_monitor_integration.py` — `TestSensingHookIntegration`, `TestBootstrapWiring`, `TestGetStatistics`, `TestAbsenceToConvergenceFeed`, `TestAbsenceSourceToDomain` (~562 lines). Split: first 3 classes into `test_absence_monitor_wiring.py` (~280), last 2 into `test_absence_monitor_domain.py` (~282).

`test_signal_freshness_and_combination.py` → 3 files:
- `test_signal_temporal_freshness.py` — `TestTemporalFreshness` (~227 lines).
- `test_signal_intra_domain.py` — `TestIntraDomainCombination` (~243 lines).
- `test_signal_freshness_integration.py` — `TestFreshnessAndCombinationIntegration` (~454 lines). Split: `TestFreshnessAndCombinationIntegration` has many test methods; split at the halfway point into two `TestFreshnessIntegrationA` and `TestFreshnessIntegrationB` files.

`test_holon_protocol.py` → 3 files:
- `test_holon_registry.py` — `TestHolonRegistryRegistration` through `TestHolonRegistryProxy` (~295 lines).
- `test_holon_proxy.py` — `TestHolonProxy` through `TestHolonProxyHeal` (~340 lines).
- `test_awareness_pulse.py` — `TestAwarenessPulse` (~234 lines).

`test_integration.py` → 3 files (this one is special — it calls `create_mae`):
- `test_integration_bootstrap.py` — `TestBootstrap`, `TestAgentLifecycle`, `TestCircadianCycle`, `TestCrossSystems`, `TestPersistence` (~337 lines).
- `test_integration_fullrun.py` — `TestFullRun`, `TestPhysarumOptimizer`, `TestPearlDefense`, `TestOctopusMemory` (~330 lines).
- `test_integration_architecture.py` — `TestHolonProtocol`, `TestConnectionTriads`, `TestBidirectionalAwareness`, `TestFractalGenerator`, `TestStemCellRefactor`, `TestTier2Persistence` (~330 lines).

**Rule:** Every split test file must copy the module-level imports from the original file. Each split file is fully independent (no shared fixtures between split files of the same origin, unless a fixture was already in `conftest.py`).

**Expected files:** 25 (was 7).

---

### Team 12: Large Test Files (500–900 lines)

**Files to decompose (54 remaining test files, all between 500–915 lines):**

The strategy is consistent across all: split by `TestCase` class groups. Each new file gets ~200–450 lines. Teams do NOT share files.

High-priority splits (these files affect memory most, as they contain `create_mae` calls or heavy fixture setup):

`test_phase57_emergence_defense.py` (845) → 3 files:
- `test_auto_healer_tests.py` — `TestAutoHealer`, `TestCapabilityDiscovery` (~250 lines).
- `test_threat_input_tests.py` — `TestThreatDetector`, `TestInputValidator` (~260 lines).
- `test_emergence_defense_integration.py` — `TestCrossSystemIntegration`, `TestSomaticMap` (~335 lines).

`test_fractal_act.py` (844) → 3 files per scale:
- `test_fractal_act_subsystem.py` — `TestHolonProxyAct`, `TestSubsystemAction` (~230 lines).
- `test_fractal_act_organ.py` — `TestOrganAction`, `TestOrganismAction` (~260 lines).
- `test_fractal_act_capabilities.py` — `TestBuildFractalAction`, `TestFractalDelegationChain`, `TestFractalSelfSimilarity`, `TestSubsystemCapabilities`, `TestOrganCapabilities`, `TestOrganismCapabilities`, `TestModuleLevel` (~354 lines).

`test_phase56_growth_coordination.py` (815) → 3 files:
- `test_substrate_topology.py` — `TestSubstrateTopology`, `TestNutrientFlow`, `TestMycelialSubstrate`, `TestProblemSignature`, `TestNoveltyDetector` (~265 lines).
- `test_organ_builder_tests.py` — `TestOrganBuilder`, `TestMorphogenesisCoordinator` (~195 lines).
- `test_growth_endocrine.py` — `TestEndocrineSystem`, `TestCircadianRhythm`, `TestPredictiveField`, `TestSpatialConsensus`, `TestCrossSystemIntegration` (~355 lines).

For the remaining 51 files (500–815 lines), the rule is: split at the `TestCase` class boundary that brings each new file to 200–480 lines. Use the class list discovered during exploration to identify split points. The file name carries the primary `TestCase` class name.

All splits follow the same mechanical pattern: copy module imports, pick contiguous TestCase classes, create new file. No logic changes.

**Expected files:** 120+ (was 54). All under 500 lines.
**Dependencies:** Depends on Wave 1 (source splits) being complete and all import paths verified.

---

## Team 13: Pytest Infrastructure

**Domain:** Memory optimization and test execution infrastructure. Works independently of Teams 1–12.

**No files to split.** This team makes targeted modifications to existing files.

### Problem diagnosis

The 13.8 GB figure has three root causes:

1. **Four test files call `create_mae()`** — the full 33-layer organism. Each instantiation holds: the OctopusColony thread pool, MarketSensingHook with 12 worker threads, ExcavationDaemon thread, ConvergenceAlerter with full signal buffer, and a `HistoricalDataFetcher` with a 130 MB signal archive preloaded into a `dict[str, list[dict]]`. When these four test files run in the same process, each `create_mae()` call's cleanup is garbage-collected but thread pools may stay alive.

2. **`deque` and `dict` signal buffers** in `ConvergenceAlerter` and `MarketSensingHook` are not bounded in tests; they accumulate across test methods that call `record_signal()`.

3. **No test parallelism:** all 174 test files share one process, so module-level caches (like `HistoricalDataFetcher._signal_cache` and TA cache) that are populated by one test class persist for the lifetime of the process.

### Changes required

**`pyproject.toml`:** Add `pytest-xdist` to dev dependencies. Add `[tool.pytest.ini_options]` section with `addopts = "-n auto --dist worksteal"`. This runs test files in separate worker processes. Each `create_mae()` call gets its own process. When the process ends, its memory is returned to the OS.

**`tests/conftest.py`:** Add a `gc_after_test` autouse fixture at function scope that calls `gc.collect()` after each test. This ensures circular references in large objects are collected between tests in the same worker. Also add an explicit `clear_signal_caches` fixture that resets known module-level caches:
```python
# In the existing _isolate_market_state fixture, append:
import gc
yield
gc.collect()
```

**`tests/conftest.py` — `HistoricalDataFetcher` guard:** Any test that instantiates `HistoricalDataFetcher` should use `tmp_path` as the signal directory, not the production `data/midge/signals/` directory with 911 files and 130 MB of data. Add to the autouse fixture:
```python
try:
    import mae_core.market.archaeology.historical_fetcher as hf_mod
    monkeypatch.setattr(hf_mod, "SIGNAL_ARCHIVE_DIR", tmp_path / "signals")
except (ImportError, AttributeError):
    pass
```

**`pyproject.toml` — test marks:** Add `markers = ["slow: marks tests as slow (deselect with -m 'not slow')"]`. Mark the four `create_mae()` test files with `@pytest.mark.slow`. In CI, run slow tests separately: `pytest -m slow -n 2` (limits parallelism for these heavy tests) and `pytest -m "not slow" -n auto` (full parallelism for unit tests).

### Expected memory outcome

With `pytest-xdist` and default auto workers (typically 4–8 on modern hardware), each worker process handles ~40 test files. Peak memory per worker: ~1–1.5 GB. Total: 4–8 × 1.5 GB = 6–12 GB across all workers, but this is distributed across processes — the OS can swap or handle this without OOM pressure. The critical change is that `create_mae()` runs once per worker process, not 4+ times in the same process.

The `conftest.py` `HistoricalDataFetcher` guard alone drops memory per worker by ~130 MB for tests that touch archaeology features.

---

## Execution Order and Wave Dependencies

```
WAVE 1 (parallel, no dependencies between teams):
  Teams 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 → all run simultaneously

GATE: Run full test suite after Wave 1. All 4,500+ tests must pass.

WAVE 2 (after gate):
  Team 11 and Team 12 → run simultaneously
  Team 13 → runs in parallel with Wave 1 (no source dependencies)

GATE: Run full test suite after Wave 2. Verify <4 GB memory peak.
```

---

## Handling the Public API Contract

Every source split MUST preserve import compatibility. The pattern is:

```python
# Original: mae_core/market/some_file.py has ClassA and ClassB
# After split: ClassA → some_file_a.py, ClassB → some_file_b.py
# some_file.py becomes:
from .some_file_a import ClassA  # noqa: F401  (re-export)
from .some_file_b import ClassB  # noqa: F401  (re-export)
__all__ = ["ClassA", "ClassB"]
```

Tests that import `from mae_core.market.some_file import ClassA` continue to work. Nothing changes from the caller's perspective.

The `# noqa: F401` comment prevents linters from flagging the re-export as unused.

---

## Document Parity Impact

The splits themselves add no new systems, tests, holons, or connections. The module count increases significantly (322 → ~500 source files) but the CLAUDE.md and HANDOFF.md count tables track systems/tests/layers/connections, not file count. No document parity update is required for the structural decomposition itself.

However, the test count in the Document Parity table tracks test functions — not test files. Splitting test files does not add or remove test functions. The count remains 4,536.

After Wave 2, update `CLAUDE.md` comment block: `Market modules: ~500 files (up from 119, decomposed for 500-line cap compliance)`.

---

### Critical Files for Implementation

- `/c/Users/baenb/projects/MIDGE/mae_core/bootstrap/market_hooks.py` - Most urgent source split (2,107 lines); the `_register_market_eventbus`, `_register_market_step_hooks`, and `_wire_sensing_hook` function boundaries are the split lines
- `/c/Users/baenb/projects/MIDGE/mae_core/market/intelligence/convergence_alerter.py` - Second largest (1,912 lines); split into `convergence_models.py`, `convergence_confidence.py`, `convergence_detection.py` plus thin hub
- `/c/Users/baenb/projects/MIDGE/mae_core/market/raw_store.py` - Cleanest split (1,835 lines, 30 independent methods) — domain groupings are mechanical; also the primary architectural reference for the mixin-based split pattern used throughout this plan
- `/c/Users/baenb/projects/MIDGE/tests/conftest.py` - The pytest infrastructure hub; Team 13 modifies this to add the `HistoricalDataFetcher` signal archive guard and `gc.collect()` — this alone significantly reduces memory consumption before any structural file splits occur
- `/c/Users/baenb/projects/MIDGE/pyproject.toml` - The `pytest-xdist` addition and test configuration changes that enable process-isolated test execution; this is the primary mechanism for the memory target