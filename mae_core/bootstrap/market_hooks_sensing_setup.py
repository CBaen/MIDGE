"""Sensing infrastructure setup helpers for MIDGE market hooks.

Extracted from market_hooks_sensing.py — purely structural split.
Contains:
  - _build_sensing_infrastructure(ctx) — builds all sensing dependencies
  - _wire_octopus_colony(ctx) — wires OctopusColony market handlers
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _build_sensing_infrastructure(ctx: SimpleNamespace):
    """Build OutcomeCollector, SignalMemory, form8k_sentiment, market_clock,
    tiered alerters, and MarketSensingHook.

    Returns (outcome_collector, memory, form8k_sentiment, market_clock,
             tiered_alerters, hook) where hook may be None if construction failed.
    Extracted from _wire_sensing_hook to keep that function under 500 lines.
    """
    from mae_core.market.sensing_hook import MarketSensingHook
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    tiered_alerters: dict = {}
    try:
        tiered_alerters["tactical"] = ConvergenceAlerter(min_domains=2, convergence_window_hours=48)
        tiered_alerters["strategic"] = ConvergenceAlerter(min_domains=2, convergence_window_hours=21 * 24)
        tiered_alerters["thematic"] = ConvergenceAlerter(min_domains=2, convergence_window_hours=90 * 24)
    except Exception:
        logger.debug("Tiered alerter construction failed", exc_info=True)

    outcome_collector = None
    try:
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector
        _ts = getattr(ctx, "thompson_sampler", None)
        if _ts is None:
            logger.error(
                "OutcomeCollector skipped — ThompsonSampler not found on ctx. "
                "Ensure ThompsonSampler is bootstrapped before OutcomeCollector."
            )
        else:
            logger.info("OutcomeCollector using ThompsonSampler id=%d", id(_ts))
            outcome_collector = OutcomeCollector(
                price_fetcher=getattr(ctx, "price_fetcher", None),
                thompson_sampler=_ts,
                regime_classifier=getattr(ctx, "regime_classifier", None),
            )
    except Exception:
        logger.debug("OutcomeCollector construction failed", exc_info=True)

    if outcome_collector is not None:
        ctx.outcome_collector = outcome_collector
        _plib = getattr(ctx, "pattern_library", None)
        if _plib is not None:
            outcome_collector.set_pattern_library(_plib)

    memory = None
    try:
        from mae_core.market.memory import SignalMemory
        memory = SignalMemory(qdrant_url=getattr(ctx, "qdrant_url", "http://localhost:6333"))
    except Exception:
        logger.debug("SignalMemory construction failed", exc_info=True)

    form8k_sentiment = None
    try:
        from mae_core.market.edge.form8k_sentiment import Form8KSentimentAnalyzer
        form8k_sentiment = Form8KSentimentAnalyzer()
    except Exception:
        logger.debug("Form8KSentimentAnalyzer construction failed", exc_info=True)

    market_clock = None
    try:
        from mae_core.market.market_clock import MarketClock
        market_clock = MarketClock()
        ctx.market_clock = market_clock
        logger.info("Market clock initialized (timezone: US/Eastern)")
    except Exception:
        logger.debug("Market clock initialization failed", exc_info=True)

    hook = None
    try:
        hook = MarketSensingHook(
            sec_client=getattr(ctx, "sec_edgar_client", None),
            price_fetcher=getattr(ctx, "price_fetcher", None),
            congress_client=getattr(ctx, "house_stock_watcher", None),
            senate_client=getattr(ctx, "senate_stock_watcher", None),
            job_tracker=getattr(ctx, "job_tracker", None),
            usa_spending=getattr(ctx, "usa_spending_client", None),
            sam_gov=getattr(ctx, "sam_gov_client", None),
            apewisdom=getattr(ctx, "apewisdom_client", None),
            finra_client=getattr(ctx, "finra_client", None),
            sec_efts=getattr(ctx, "sec_efts_client", None),
            finnhub=getattr(ctx, "finnhub_client", None),
            fred=getattr(ctx, "fred_client", None),
            convergence_alerter=getattr(ctx, "convergence_alerter", None),
            velocity_detector=getattr(ctx, "velocity_detector", None),
            filing_analyzer=getattr(ctx, "filing_time_analyzer", None),
            form8k_sentiment=form8k_sentiment,
            session_sweep_detector=getattr(ctx, "session_sweep_detector", None),
            ta_indicators=getattr(ctx, "ta_indicators", None),
            cot_client=getattr(ctx, "cot_client", None),
            stocktwits_client=getattr(ctx, "stocktwits_client", None),
            vix_client=getattr(ctx, "vix_client", None),
            trends_client=getattr(ctx, "trends_client", None),
            outcome_collector=outcome_collector,
            memory=memory,
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            tiered_alerters=tiered_alerters,
            order_flow_detector=getattr(ctx, "order_flow_detector", None),
            portfolio_tracker=getattr(ctx, "portfolio_tracker", None),
            catalyst_calendar=getattr(ctx, "catalyst_calendar", None),
            deception_detector=getattr(ctx, "deception_detector", None),
            consolidation_engine=getattr(ctx, "consolidation_engine", None),
            fractal_resonance_detector=getattr(ctx, "fractal_resonance_detector", None),
            pattern_archetype_engine=getattr(ctx, "pattern_archetype_engine", None),
            somatic_anticipation=getattr(ctx, "somatic_anticipation", None),
            pattern_completion_engine=getattr(ctx, "pattern_completion_engine", None),
            market_clock=market_clock,
            coingecko_client=getattr(ctx, "coingecko_client", None),
            coincap_client=getattr(ctx, "coincap_client", None),
            openinsider_client=getattr(ctx, "openinsider_client", None),
            edgar_enhanced_client=getattr(ctx, "edgar_enhanced_client", None),
            finviz_client=getattr(ctx, "finviz_client", None),
            economic_calendar_client=getattr(ctx, "economic_calendar_client", None),
            finnhub_websocket=getattr(ctx, "finnhub_websocket", None),
            massive_client=getattr(ctx, "massive_client", None),
            eia_client=getattr(ctx, "eia_client", None),
            congress_gov_client=getattr(ctx, "congress_gov_client", None),
            social_text_analyzer=getattr(ctx, "social_text_analyzer", None),
            yahoo_rss_client=getattr(ctx, "yahoo_rss_client", None),
            usda_client=getattr(ctx, "usda_client", None),
        )
    except Exception:
        logger.warning(
            "MarketSensingHook construction failed — agents will not sense market data",
            exc_info=True,
        )

    return outcome_collector, memory, form8k_sentiment, market_clock, tiered_alerters, hook


def _wire_octopus_colony(ctx: SimpleNamespace) -> None:
    """Wire OctopusColony market handlers, start monitoring, and register subscribers.

    Extracted from _wire_sensing_hook to keep that function under 500 lines.
    Idempotent: no-ops if octopus_colony not on ctx.
    """
    colony = getattr(ctx, "octopus_colony", None)
    if colony is None:
        return

    try:
        from mae_core.network.market_task_handlers import inject_market_handlers, patch_new_arm
        inject_market_handlers(
            colony=colony,
            convergence_alerter=getattr(ctx, "convergence_alerter", None),
            pattern_watcher=getattr(ctx, "pattern_watcher", None),
            event_bus=getattr(ctx, "bus", None),
            pattern_library=getattr(ctx, "pattern_library", None),
            world_model=getattr(ctx, "world_model", None),
        )
        logger.info("OctopusColony: market handlers injected")
    except Exception:
        logger.debug("OctopusColony handler injection failed", exc_info=True)

    try:
        colony.start_monitoring()
        logger.info("OctopusColony: monitoring started")
    except Exception:
        logger.debug("OctopusColony monitoring start failed", exc_info=True)

    try:
        from mae_core.network.market_task_handlers import patch_new_arm as _pna

        def _on_octopus_spawn(channel, data):
            msg = data if isinstance(data, dict) else {}
            oct_id = msg.get("octopus_id", "")
            oct_obj = colony.octopuses.get(oct_id)
            if oct_obj is None:
                return
            cognition = getattr(oct_obj, "cognition", oct_obj)
            for arm in getattr(cognition, "arms", {}).values():
                try:
                    _pna(colony, arm)
                except Exception:
                    logger.debug("Failed to patch arm on spawned %s", oct_id, exc_info=True)

        bus = getattr(ctx, "bus", None)
        if bus is not None:
            bus.register_callback("octopus.spawn", _on_octopus_spawn)
    except Exception:
        logger.debug("OctopusColony spawn subscriber failed", exc_info=True)

    try:
        from mae_core.network.market_task_handlers import CH_OCTOPUS_INVESTIGATION

        def _on_octopus_investigation(channel, data):
            msg = data if isinstance(data, dict) else {}
            ticker = msg.get("ticker", "?")
            source = msg.get("source", "?")
            check_count = msg.get("check_count", 0)
            priority_created = msg.get("priority_request_created", False)
            historical = msg.get("historical_templates", [])
            logger.info(
                "OctopusInvestigation[%s] ticker=%s check=%d templates=%d%s",
                source, ticker, check_count, len(historical),
                " [FOCUSED-ATTENTION ENGAGED]" if priority_created else "",
            )

        bus_obj = getattr(ctx, "bus", None)
        if bus_obj is not None:
            bus_obj.register_callback(CH_OCTOPUS_INVESTIGATION, _on_octopus_investigation)
            logger.info("OctopusColony: investigation subscriber wired")
    except Exception:
        logger.debug("OctopusColony investigation subscriber failed", exc_info=True)
