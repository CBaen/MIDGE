"""Bootstrap Layer 33d: Market triadic connections.

One job: register all 106 triadic connections for the market intelligence organ
(Groups 14-35). No bare dyads — every A<->B has witness C.

Groups:
  14 - Market K3 subsystems + EventBus pub/sub + TA indicators
  15 - Phase 4: Signal archive, lag-correlation, calibration, Kelly
  16 - Hypothesis loop (RSI Layer 2)
  17 - Backtest bridge (Bridge 1 + Bridge 3)
  18 - Efficiency: StepTimer performance metabolism
  19 - Layer 6 new source clients
  20 - Coherence scoring (narrative conflict detection — Capability 3)
  21 - Causal Bridge (CausalReasoningEngine wired to market layer)
  22 - Absence Monitor (detecting unexpectedly silent sources)
  23 - Portfolio Tracker (Gift 1: portfolio state awareness)
  24 - Order Flow Detector (Gift 2: intraday volume imbalance)
  25 - Catalyst Calendar (Gift 3: timing context for signals)
  26 - Cross-Asset Confirmer (Gift 4: cross-asset regime confirmation)
  27 - Deception Detector (Gift 5: manipulation immune response)
  28 - Consolidation Engine (Gift 6: sleep cycle / memory pruning)
  29 - Fractal Resonance (Gift 7: multi-timeframe structure alignment)
  30 - Pattern Archetypes (Gift 8: canonical pattern templates)
  31 - Somatic Anticipation (Gift 9: pre-conscious pattern response)
  32 - Pattern Completion Engine (Gift 10: active pattern seeking)
  33 - Pattern Archaeology (reverse-engineered historical patterns)
  34 - OctopusColony (ecosystem bridge)
  35 - ResourceGovernor (Law 6 autopoietic self-governance)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_market_connections(ctx: SimpleNamespace) -> None:
    """Register 106 triadic connections for market systems (Group 14-35)."""
    from mae_core.backbone.connection_registry import (
        ConnectionType,
        ConnectionCriticality,
    )

    reg = ctx.connection_registry.register_connection
    dr = ConnectionType.DIRECT_REFERENCE
    eb = ConnectionType.EVENTBUS_PUBSUB
    cb = ConnectionType.CALLBACK_REGISTRATION
    sh = ConnectionType.STEP_HOOK

    # --- Market Sensing K3 ---
    reg("sec_edgar_client", "price_fetcher", dr,
        witnesses=["job_tracker", "auditor"],
        description="SEC signals correlate with price movement")
    reg("price_fetcher", "job_tracker", dr,
        witnesses=["sec_edgar_client", "auditor"],
        description="Price data validates hiring signal predictions")
    reg("job_tracker", "sec_edgar_client", dr,
        witnesses=["price_fetcher", "auditor"],
        description="Hiring data cross-references insider filings")

    # --- Market Edge K3 ---
    reg("cluster_detector", "politician_tracker", dr,
        witnesses=["contract_predictor", "auditor"],
        description="Insider clusters may correlate with political trades")
    reg("politician_tracker", "contract_predictor", dr,
        witnesses=["cluster_detector", "auditor"],
        description="Political trades may precede contract awards")
    reg("contract_predictor", "cluster_detector", dr,
        witnesses=["politician_tracker", "auditor"],
        description="Contract predictions validated by insider clusters")

    # --- Market Learning K3 ---
    reg("thompson_sampler", "convergence_alerter", dr,
        witnesses=["velocity_detector", "auditor"],
        description="Thompson reliability weights convergence decisions")
    reg("convergence_alerter", "velocity_detector", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alerts carry velocity context")
    reg("velocity_detector", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Velocity anomalies feed signal exploration priority")

    # --- Cross-subsystem: EventBus publish ---
    reg("cluster_detector", "event_bus", eb,
        channel="market.edge.cluster_detected",
        witnesses=["threat_detector", "auditor"],
        description="Insider cluster signal published")
    reg("politician_tracker", "event_bus", eb,
        channel="market.edge.politician_trade",
        witnesses=["threat_detector", "auditor"],
        description="Political trade correlation published")
    reg("contract_predictor", "event_bus", eb,
        channel="market.edge.contract_predicted",
        witnesses=["threat_detector", "auditor"],
        description="Contract prediction published")
    reg("filing_time_analyzer", "event_bus", eb,
        channel="market.edge.filing_anomaly",
        witnesses=["threat_detector", "auditor"],
        description="Filing time anomaly published")
    reg("session_sweep_detector", "event_bus", eb,
        channel="market.edge.session_sweep",
        witnesses=["threat_detector", "auditor"],
        description="Session sweep signal published")

    # --- Cross-subsystem: EventBus subscribe ---
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.cluster_detected",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to cluster signals")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.politician_trade",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to political trade signals")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.contract_predicted",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to contract predictions")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.session_sweep",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to session sweep signals")

    # --- Integration connections ---
    reg("convergence_alerter", "thompson_sampler", dr,
        witnesses=["velocity_detector", "knowledge_base"],
        description="Convergence queries Thompson for signal reliability")
    reg("velocity_detector", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Velocity enriches convergence signal context")
    reg("convergence_alerter", "knowledge_base", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence stores discoveries to knowledge base")
    reg("sec_edgar_client", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="SEC client validated through boundary membrane")
    reg("price_fetcher", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="Price fetcher validated through boundary membrane")
    reg("session_sweep_detector", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Session sweep reads intraday price data via yfinance")

    # --- TA Indicators connections ---
    reg("ta_indicators", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="TA indicators compute RSI/MACD/Bollinger from daily OHLCV")
    reg("convergence_alerter", "ta_indicators", dr,
        channel="market.edge.ta_indicators",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to TA indicator signals")
    reg("ta_indicators", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Thompson learns TA indicator reliability from outcomes")

    # --- Step hook + periodic ---
    reg("convergence_alerter", "model", sh,
        witnesses=["auditor", "connection_registry"],
        description="Convergence alerter runs each step")
    reg("thompson_sampler", "event_bus", eb,
        channel="market.intel.thompson_stats",
        witnesses=["auditor", "connection_registry"],
        description="Thompson stats published periodically")

    # --- Group 15: Phase 4 — Signal Archive, Lag-Correlation, Calibration, Kelly ---
    reg("signal_archive_reader", "lag_correlation_analyzer", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Archive reader provides historical series to lag analyzer")
    reg("lag_correlation_analyzer", "event_bus", eb,
        channel="market.intel.lag_finding",
        witnesses=["correlation_tracker", "auditor"],
        description="Lag findings published to EventBus")
    reg("thompson_calibrator", "signal_archive_reader", dr,
        witnesses=["lag_correlation_analyzer", "auditor"],
        description="Calibrator reads archives to validate distributions")

    reg("thompson_calibrator", "thompson_sampler", dr,
        witnesses=["signal_archive_reader", "auditor"],
        description="Calibrator seeds and diagnoses Thompson distributions")
    reg("thompson_calibrator", "event_bus", eb,
        channel="market.intel.thompson_stats",
        witnesses=["convergence_alerter", "auditor"],
        description="Calibration report published to EventBus")
    reg("convergence_alerter", "thompson_calibrator", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alerter uses calibration for confidence")

    reg("kelly_position_sizer", "thompson_sampler", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Kelly sizer queries Thompson mean for p parameter")
    reg("convergence_alerter", "kelly_position_sizer", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alert triggers Kelly sizing recommendation")
    reg("kelly_position_sizer", "event_bus", eb,
        channel="market.intel.kelly_sizing",
        witnesses=["convergence_alerter", "auditor"],
        description="Kelly sizing recommendation published on convergence")

    reg("lag_correlation_analyzer", "correlation_tracker", dr,
        witnesses=["signal_archive_reader", "auditor"],
        description="Lag findings feed leading-indicator pairs to tracker")
    reg("lag_correlation_analyzer", "convergence_alerter", dr,
        witnesses=["correlation_tracker", "auditor"],
        description="Significant lag pairs update convergence domain weights")
    reg("signal_archive_reader", "outcome_tracker", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Archive reader provides historical signals to outcome tracker")

    # --- Group 16: Hypothesis loop (RSI Layer 2) ---
    reg("hypothesis_generator", "hypothesis_registry", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Generator registers new hypotheses in the registry")
    reg("hypothesis_validator", "hypothesis_registry", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Validator reads hypotheses for adversarial testing")
    reg("hypothesis_engine", "hypothesis_generator", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Engine drives generation on cadence")
    reg("hypothesis_engine", "hypothesis_validator", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine drives validation on cadence")
    reg("hypothesis_engine", "hypothesis_registry", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine manages lifecycle transitions (promote/retire)")
    reg("hypothesis_generator", "lag_correlation_analyzer", dr,
        witnesses=["hypothesis_registry", "auditor"],
        description="Generator reads lag findings to create hypotheses")
    reg("hypothesis_engine", "thompson_sampler", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Engine registers Thompson key on hypothesis promotion")
    reg("hypothesis_engine", "event_bus", eb,
        channel="market.hypothesis.discovered",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine publishes hypothesis discoveries")
    reg("hypothesis_engine", "event_bus", eb,
        channel="market.hypothesis.promoted",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine publishes hypothesis promotions")
    reg("hypothesis_engine", "event_bus", cb,
        channel="market.sensing.signal_ingested",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine subscribes to ingested signals for trigger matching")

    # --- Group 17: Backtest bridge (Bridge 1 — RSI Layer 2 self-reading) ---
    reg("backtest_analyzer", "hypothesis_registry", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Backtest analyzer registers derived hypotheses in registry")
    reg("backtest_analyzer", "hypothesis_validator", dr,
        witnesses=["hypothesis_registry", "auditor"],
        description="Backtest hypotheses flow to validator for DSR evaluation")
    reg("hypothesis_engine", "backtest_analyzer", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine drives backtest analysis on generation cadence")

    # --- Bridge 3: Autonomous backtest scheduling ---
    reg("backtest_scheduler", "backtest_analyzer", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="Scheduler triggers backtest rerun and refreshes analyzer")
    reg("backtest_scheduler", "event_bus", eb,
        channel="market.intel.backtest_refreshed",
        witnesses=["hypothesis_engine", "auditor"],
        description="Scheduler publishes refresh completion event")

    # --- Efficiency: Step timer (performance metabolism) ---
    reg("step_timer", "convergence_alerter", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="StepTimer wraps convergence check to measure latency")
    reg("step_timer", "event_bus", eb,
        channel="market.intel.step_timing",
        witnesses=["convergence_alerter", "auditor"],
        description="StepTimer publishes timing statistics for health monitoring")

    # --- Group 19: Layer 6 new source clients ---
    for client_id in ("cot_client", "stocktwits_client", "vix_client", "trends_client"):
        if getattr(ctx, client_id, None) is not None:
            reg(client_id, "convergence_alerter", dr,
                witnesses=["thompson_sampler", "auditor"],
                description=f"{client_id} feeds signals to convergence")
            reg(client_id, "boundary_membrane", dr,
                witnesses=["input_validator", "threat_detector"],
                criticality=ConnectionCriticality.IMPORTANT,
                description=f"{client_id} validated through boundary membrane")

    # --- Group 20: Coherence scoring (narrative conflict detection) ---
    reg("convergence_alerter", "event_bus", eb,
        channel="market.intel.contradiction_detected",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence publishes contradiction when signals conflict")

    # --- Group 21: Causal Bridge (completing Cap 3 — contradiction → causation) ---
    reg("shared_causal_engine", "convergence_alerter", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Causal engine receives domain pair correlations from convergence")
    reg("shared_causal_engine", "hypothesis_validator", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Causal engine checks confounding before hypothesis promotion")
    reg("shared_causal_engine", "event_bus", eb,
        channel="market.intel.contradiction_detected",
        witnesses=["convergence_alerter", "auditor"],
        description="Causal engine context enriches contradiction events")

    # --- Group 22: Absence Monitor (detecting unexpectedly silent sources) ---
    reg("absence_monitor", "event_bus", eb,
        channel="market.intel.absence_detected",
        witnesses=["convergence_alerter", "auditor"],
        description="AbsenceMonitor publishes silence signals to EventBus")
    reg("absence_monitor", "sensing_hook", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Sensing hook records arrivals for cadence tracking")
    reg("absence_monitor", "convergence_alerter", dr,
        witnesses=["sensing_hook", "auditor"],
        description="Absence signals feed into convergence pipeline")

    # --- Group 23: Portfolio Tracker (Gift 1 — portfolio state awareness) ---
    reg("portfolio_tracker", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Portfolio tracker marks positions to market via price fetcher")
    reg("portfolio_tracker", "convergence_alerter", dr,
        witnesses=["price_fetcher", "auditor"],
        description="Exit signals from portfolio tracker feed into convergence")
    reg("portfolio_tracker", "event_bus", eb,
        channel="market.intel.exit_signal",
        witnesses=["convergence_alerter", "auditor"],
        description="Portfolio tracker publishes exit signals on EventBus")

    # --- Group 24: Order Flow Detector (Gift 2 — intraday volume imbalance) ---
    reg("order_flow_detector", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Order flow detector reads intraday candles from price fetcher")
    reg("order_flow_detector", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Order flow imbalance signals feed into convergence pipeline")
    reg("order_flow_detector", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Thompson learns order flow signal reliability from outcomes")

    # --- Group 25: Catalyst Calendar (Gift 3 — timing context for signals) ---
    reg("catalyst_calendar", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Catalyst calendar provides timing modifiers to convergence")
    reg("catalyst_calendar", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="Catalyst calendar API calls validated through boundary membrane")
    reg("catalyst_calendar", "event_bus", cb,
        channel="market.sensing.signal_ingested",
        witnesses=["convergence_alerter", "auditor"],
        description="Catalyst calendar listens for signals to enrich with timing context")

    # --- Group 26: Cross-Asset Confirmer (Gift 4 — cross-asset regime confirmation) ---
    reg("cross_asset_confirmer", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Cross-asset confirmer reads prices for correlated assets")
    reg("cross_asset_confirmer", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Cross-asset confirmation scores adjust convergence confidence")
    reg("cross_asset_confirmer", "regime_classifier", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Cross-asset confirmer uses regime context for pair weighting")

    # --- Group 27: Deception Detector (Gift 5 — manipulation immune response) ---
    reg("deception_detector", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Deception assessments reduce convergence confidence on flagged tickers")
    reg("deception_detector", "event_bus", eb,
        channel="market.edge.deception_detected",
        witnesses=["convergence_alerter", "auditor"],
        description="Deception detector publishes manipulation pattern alerts")
    reg("deception_detector", "correlation_tracker", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Deception detector monitors cross-domain signal composition")

    # --- Group 28: Consolidation Engine (Gift 6 — sleep cycle / memory pruning) ---
    reg("consolidation_engine", "thompson_sampler", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="Consolidation engine prunes thin/stale Thompson distributions")
    reg("consolidation_engine", "hypothesis_registry", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="Consolidation engine archives old retired hypotheses")
    reg("consolidation_engine", "event_bus", eb,
        channel="market.intel.consolidation_complete",
        witnesses=["thompson_sampler", "auditor"],
        description="Consolidation engine publishes completion reports on EventBus")

    # --- Group 29: Fractal Resonance (Gift 7 — multi-timeframe structure) ---
    reg("fractal_resonance_detector", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Fractal resonance reads daily/weekly/monthly price history")
    reg("fractal_resonance_detector", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Fractal resonance signals feed into convergence pipeline")
    reg("fractal_resonance_detector", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Thompson learns fractal resonance signal reliability")

    # --- Group 30: Pattern Archetypes (Gift 8 — canonical pattern templates) ---
    reg("pattern_archetype_engine", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Archetype matches boost convergence confidence on matching tickers")
    reg("pattern_archetype_engine", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Archetype engine reads price data for structure analysis")
    reg("pattern_archetype_engine", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Thompson learns archetype match signal reliability")

    # --- Group 31: Somatic Anticipation (Gift 9 — pre-conscious pattern response) ---
    reg("somatic_anticipation", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Somatic anticipation fires before convergence — preparatory state")
    reg("somatic_anticipation", "event_bus", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Somatic anticipation publishes hormone release events")
    reg("somatic_anticipation", "correlation_tracker", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Somatic anticipation tracks cross-domain signal accumulation")

    # --- Group 32: Pattern Completion Engine (Gift 10 — active pattern seeking) ---
    reg("pattern_completion_engine", "pattern_archetype_engine", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Completion engine reads partial matches from archetype engine")
    reg("pattern_completion_engine", "convergence_alerter", dr,
        witnesses=["pattern_archetype_engine", "auditor"],
        description="Completed patterns boost convergence confidence (+0.15)")
    reg("pattern_completion_engine", "event_bus", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Completion events published for hypothesis engine consumption")

    # --- Group 33: Pattern Archaeology (reverse-engineered historical patterns) ---
    if getattr(ctx, "pattern_library", None) is not None:
        reg("pattern_library", "pattern_watcher", dr,
            witnesses=["convergence_alerter", "auditor"],
            description="Watcher queries library for fingerprint matches")
        reg("pattern_watcher", "convergence_alerter", dr,
            witnesses=["pattern_library", "auditor"],
            description="Pattern stacks cross-validate with live convergence")
        reg("pattern_library", "hypothesis_engine", dr,
            witnesses=["pattern_watcher", "auditor"],
            description="Excavated patterns feed hypothesis generation pipeline")

    # --- Group 34: OctopusColony pipeline bridge ---
    if getattr(ctx, "octopus_colony", None) is not None:
        reg("octopus_colony", "convergence_alerter", dr,
            witnesses=["pattern_watcher", "auditor"],
            description="Colony investigates partial convergences from alerter")
        reg("octopus_colony", "pattern_watcher", dr,
            witnesses=["convergence_alerter", "auditor"],
            description="Colony queries archaeology for partial templates")
        reg("octopus_colony", "event_bus", eb,
            witnesses=["convergence_alerter", "pattern_watcher"],
            description="Colony emits investigation results via EventBus")

    # --- Group 35: ResourceGovernor (Law 6 autopoietic self-governance) ---
    if getattr(ctx, "resource_governor", None) is not None:
        reg("resource_governor", "convergence_alerter", eb,
            channel="market.resource.throttle",
            witnesses=["event_bus", "auditor"],
            description="ResourceGovernor publishes throttle events; convergence holds back on throttled sources")
        reg("resource_governor", "thompson_sampler", dr,
            witnesses=["event_bus", "auditor"],
            description="ResourceGovernor budget state visible to Thompson for source-level exploration gating")
        reg("resource_governor", "event_bus", eb,
            channel="market.resource.budget_warning",
            witnesses=["convergence_alerter", "auditor"],
            description="ResourceGovernor publishes budget warnings via EventBus for organism awareness")

    logger.info("Layer 33d - Market connections: triadic connections registered (Group 14-35)")
