"""Bootstrap Layer 33b-e: Market system registration.

One job: register all market systems with the organism's awareness infrastructure.
Four sub-tasks, each a distinct registry:
  - SomaticMap (dependency graph / body awareness)
  - HolonRegistry (fractal self-awareness proxies)
  - FractalGenerator (K3 subsystem hierarchy)
  - StemCell ROLE_PROFILES (verify market roles are available)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_market_somatic(ctx: SimpleNamespace) -> None:
    """Register market systems with SomaticMap (must precede connections)."""
    somatic = getattr(ctx, "somatic_map", None)
    if somatic is None or not hasattr(somatic, "register_system"):
        return

    market_systems = {
        "sec_edgar_client": ("SECEdgarClient", []),
        "price_fetcher": ("PriceFetcher", []),
        "house_stock_watcher": ("HouseStockWatcher", []),
        "job_tracker": ("JobTracker", []),
        "usa_spending_client": ("USASpendingClient", []),
        "sam_gov_client": ("SAMGovClient", []),
        "cluster_detector": ("ClusterDetector", []),
        "politician_tracker": ("PoliticianTracker", ["sec_edgar_client", "usa_spending_client"]),
        "filing_time_analyzer": ("FilingTimeAnalyzer", []),
        "contract_predictor": ("ContractPredictor", ["job_tracker", "sam_gov_client"]),
        "session_sweep_detector": ("SessionSweepDetector", ["price_fetcher"]),
        "thompson_sampler": ("ThompsonSampler", []),
        "convergence_alerter": ("ConvergenceAlerter", ["thompson_sampler", "velocity_detector", "regime_classifier"]),
        "velocity_detector": ("VelocityDetector", []),
        "correlation_tracker": ("CorrelationTracker", []),
        "outcome_tracker": ("OutcomeTracker", ["price_fetcher", "thompson_sampler"]),
        "regime_classifier": ("RegimeClassifier", ["price_fetcher"]),
        "signal_archive_reader": ("SignalArchiveReader", []),
        "lag_correlation_analyzer": ("LagCorrelationAnalyzer", ["signal_archive_reader"]),
        "granger_analyzer": ("GrangerAnalyzer", ["signal_archive_reader"]),
        "thompson_calibrator": ("ThompsonCalibrator", ["thompson_sampler"]),
        "kelly_position_sizer": ("KellyPositionSizer", ["thompson_sampler"]),
        "hypothesis_registry": ("HypothesisRegistry", []),
        "hypothesis_generator": ("HypothesisGenerator", ["hypothesis_registry"]),
        "hypothesis_validator": ("HypothesisValidator", []),
        "hypothesis_engine": ("HypothesisEngine", ["hypothesis_registry", "hypothesis_generator", "hypothesis_validator"]),
        "backtest_analyzer": ("BacktestAnalyzer", ["hypothesis_registry"]),
        "backtest_scheduler": ("BacktestScheduler", ["backtest_analyzer"]),
        "ta_indicators": ("TAIndicators", ["price_fetcher"]),
        "step_timer": ("StepTimer", []),
        "absence_monitor": ("AbsenceMonitor", ["signal_archive_reader"]),
        # Ten Gifts: Wave 1
        "portfolio_tracker": ("PortfolioTracker", ["price_fetcher"]),
        "order_flow_detector": ("OrderFlowDetector", ["price_fetcher"]),
        "catalyst_calendar": ("CatalystCalendar", ["finnhub_client"]),
        "cross_asset_confirmer": ("CrossAssetConfirmer", ["price_fetcher"]),
        # Ten Gifts: Wave 2
        "deception_detector": ("DeceptionDetector", []),
        "consolidation_engine": ("ConsolidationEngine", ["thompson_sampler", "hypothesis_registry"]),
        "fractal_resonance_detector": ("FractalResonanceDetector", ["price_fetcher"]),
        "pattern_archetype_engine": ("PatternArchetypeEngine", ["price_fetcher"]),
        # Ten Gifts: Wave 3
        "somatic_anticipation": ("SomaticAnticipation", []),
        "pattern_completion_engine": ("PatternCompletionEngine", ["pattern_archetype_engine"]),
        # Layer 6 clients
        "cot_client": ("COTClient", []),
        "stocktwits_client": ("StockTwitsClient", []),
        "vix_client": ("VIXClient", []),
        "trends_client": ("TrendsClient", []),
        # Always-On Wave 2+3: Real-Time + Data Enrichment
        "coingecko_client": ("CoinGeckoClient", []),
        "coincap_client": ("CoinCapClient", []),
        "openinsider_client": ("OpenInsiderClient", []),
        "edgar_enhanced_client": ("EdgarEnhancedClient", []),
        "finviz_client": ("FinVizClient", []),
        "economic_calendar_client": ("EconomicCalendarClient", []),
        "finnhub_websocket": ("FinnhubWebSocket", []),
        "massive_client": ("MassiveClient", []),
        # Pattern Archaeology
        "pattern_library": ("PatternLibrary", []),
        "pattern_watcher": ("PatternWatcher", ["pattern_library", "convergence_alerter"]),
        # Ecosystem Bridge
        "octopus_colony": ("OctopusColony", ["convergence_alerter", "pattern_watcher"]),
        # Resource Governance (Law 6 autopoiesis — self-monitoring API budget)
        "resource_governor": ("ResourceGovernor", []),
    }

    for sys_id, (desc, deps) in market_systems.items():
        if getattr(ctx, sys_id, None) is not None:
            try:
                somatic.register_system(
                    system_id=sys_id,
                    description=desc,
                    depends_on=deps,
                )
            except Exception:
                logger.debug("Could not register %s with SomaticMap", sys_id)


def _register_market_holons(ctx: SimpleNamespace) -> None:
    """Register all market systems as holons with HolonProxy injection."""
    registry = ctx.holon_registry

    market_systems = [
        "sec_edgar_client", "price_fetcher", "house_stock_watcher",
        "job_tracker", "usa_spending_client", "sam_gov_client",
        "cluster_detector", "politician_tracker", "filing_time_analyzer",
        "contract_predictor", "session_sweep_detector",
        "thompson_sampler", "convergence_alerter",
        "velocity_detector", "correlation_tracker", "outcome_tracker",
        "regime_classifier",
        "signal_archive_reader", "lag_correlation_analyzer",
        "thompson_calibrator", "kelly_position_sizer",
        "hypothesis_registry", "hypothesis_generator",
        "hypothesis_validator", "hypothesis_engine",
        "backtest_analyzer", "backtest_scheduler",
        "ta_indicators", "step_timer", "absence_monitor",
        # Ten Gifts: Wave 1
        "portfolio_tracker", "order_flow_detector",
        "catalyst_calendar", "cross_asset_confirmer",
        # Ten Gifts: Wave 2
        "deception_detector", "consolidation_engine",
        "fractal_resonance_detector", "pattern_archetype_engine",
        # Ten Gifts: Wave 3
        "somatic_anticipation", "pattern_completion_engine",
        # Layer 6
        "cot_client", "stocktwits_client", "vix_client", "trends_client",
        # Always-On Wave 2+3
        "coingecko_client", "coincap_client", "openinsider_client",
        "edgar_enhanced_client", "finviz_client", "economic_calendar_client",
        "finnhub_websocket", "massive_client",
        # Pattern Archaeology
        "pattern_library", "pattern_watcher",
        # Ecosystem Bridge
        "octopus_colony",
        # Resource Governance (Law 6 autopoiesis)
        "resource_governor",
    ]

    registered = 0
    for sys_id in market_systems:
        system_obj = getattr(ctx, sys_id, None)
        if system_obj is None:
            continue

        try:
            registry.register(sys_id, holon_type="system", parent_id="mae")
            proxy = registry.get_proxy(sys_id)
            if proxy is not None:
                proxy.set_system_ref(system_obj)
            registered += 1
        except Exception:
            logger.debug("Could not register holon: %s", sys_id)

    logger.info("Layer 33b - Market holons registered: %d holons", registered)


def _register_market_fractal(ctx: SimpleNamespace) -> None:
    """Register fractal hierarchy — three K3 subsystems under one organ."""
    fg = ctx.fractal_generator

    # Subsystem triads
    fg.generate_triad(
        name="market-sensing",
        holon_type="subsystem",
        children_ids=["sec_edgar_client", "price_fetcher", "job_tracker"],
        parent_id="market-intelligence-system",
    )
    fg.generate_triad(
        name="market-edge",
        holon_type="subsystem",
        children_ids=["cluster_detector", "politician_tracker", "contract_predictor"],
        parent_id="market-intelligence-system",
    )
    fg.generate_triad(
        name="market-learning",
        holon_type="subsystem",
        children_ids=["thompson_sampler", "convergence_alerter", "velocity_detector"],
        parent_id="market-intelligence-system",
    )

    # Organ — completes the bare dyad in organ-cluster-cognitive
    fg.generate_triad(
        name="market-intelligence-system",
        holon_type="organ",
        children_ids=["market-sensing", "market-edge", "market-learning"],
        parent_id="organ-cluster-cognitive",
    )

    # Hypothesis K3 subsystem (RSI Layer 2: generator + validator + engine)
    fg.generate_triad(
        name="market-hypothesis",
        holon_type="subsystem",
        children_ids=["hypothesis_generator", "hypothesis_validator", "hypothesis_engine"],
        parent_id="market-intelligence-system",
    )

    # Register remaining systems individually (advisory non-triadic)
    extras = [
        "house_stock_watcher", "filing_time_analyzer",
        "session_sweep_detector",
        "usa_spending_client", "sam_gov_client", "correlation_tracker",
        "outcome_tracker", "regime_classifier",
        "signal_archive_reader", "lag_correlation_analyzer",
        "thompson_calibrator", "kelly_position_sizer",
        "hypothesis_registry", "backtest_analyzer", "backtest_scheduler",
        "ta_indicators", "step_timer", "absence_monitor",
        # Ten Gifts: Wave 1
        "portfolio_tracker", "order_flow_detector",
        "catalyst_calendar", "cross_asset_confirmer",
        # Ten Gifts: Wave 2
        "deception_detector", "consolidation_engine",
        "fractal_resonance_detector", "pattern_archetype_engine",
        # Ten Gifts: Wave 3
        "somatic_anticipation", "pattern_completion_engine",
        # Layer 6
        "cot_client", "stocktwits_client", "vix_client", "trends_client",
        # Always-On Wave 2+3
        "coingecko_client", "coincap_client", "openinsider_client",
        "edgar_enhanced_client", "finviz_client", "economic_calendar_client",
        "finnhub_websocket", "massive_client",
        # Pattern Archaeology
        "pattern_library", "pattern_watcher",
        # Ecosystem Bridge
        "octopus_colony",
        # Resource Governance (Law 6 autopoiesis)
        "resource_governor",
    ]
    for sys_id in extras:
        if ctx.holon_registry.get_entry(sys_id) is not None:
            try:
                ctx.holon_registry.reparent(sys_id, "market-intelligence-system")
            except Exception:
                logger.debug("Could not reparent %s under market-intelligence-system", sys_id)

    # Verify repair
    children = ctx.holon_registry.get_children("organ-cluster-cognitive")
    logger.info(
        "Layer 33c - Market fractal: organ-cluster-cognitive now has %d children "
        "(K3 requires 3)",
        len(children),
    )


def _register_market_stem_roles(ctx: SimpleNamespace) -> None:
    """Verify market stem cell roles are available."""
    from mae_core.agents.stem_cell import ROLE_PROFILES

    market_roles = [
        "SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST",
        "HYPOTHESIS_EXPLORER", "HYPOTHESIS_VALIDATOR",
    ]
    for role in market_roles:
        if role not in ROLE_PROFILES:
            logger.warning("Market stem cell role %s not found in ROLE_PROFILES", role)

    logger.info("Layer 33e - Market stem cell roles verified: %s", ", ".join(market_roles))
