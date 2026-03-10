"""Sensing pipeline wiring for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _wire_sensing_hook (with all nested closures).
Heavy sub-blocks extracted to module-level helpers:
  - _run_sensing_archaeology(ctx, step, _shm_sensing)
  - _run_active_tracker_check(ctx)
  - _run_synergy_detection(ctx)
  - _run_paper_trading_gate(ctx, alerts, step)

Critical constraint: ctx._cached_alerts must already exist when this runs.
_register_market_step_hooks() MUST be called before _wire_sensing_hook().
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _run_sensing_archaeology(ctx: SimpleNamespace, step: int, _shm_sensing) -> None:
    """Pattern archaeology stacking detection + active tracker price check.

    Called every 10 steps from _sensing_step_with_advisory.
    Builds active signal map, runs PatternWatcher.check(), registers stacks,
    writes plain-language alerts, and updates ActiveTracker.
    """
    if getattr(ctx, "pattern_watcher", None) is not None:
        try:
            _alerter = getattr(ctx, "convergence_alerter", None)
            if _alerter is not None and hasattr(_alerter, "signals"):
                _active: dict = {}
                for _domain, _sigs in _alerter.signals.items():
                    for _sig in _sigs:
                        _sym = getattr(_sig, "metadata", {}).get("symbol", "")
                        _dir = getattr(_sig, "direction", "")
                        _src = getattr(_sig, "source", "")
                        if not _sym or not _dir or not _src:
                            continue
                        if _sym not in _active:
                            _active[_sym] = {}
                        if _dir not in _active[_sym]:
                            _active[_sym][_dir] = set()
                        _active[_sym][_dir].add(_src)
                if _active:
                    _stacks = ctx.pattern_watcher.check(_active)
                    ctx._cached_pattern_stacks = _stacks or []
                    if _shm_sensing:
                        _shm_sensing.record_success("pattern_watcher")
                    _oc = getattr(ctx, "outcome_collector", None)
                    if _stacks:
                        for _stack in _stacks:
                            if _oc is not None:
                                try:
                                    _oc.register_pattern_stack(_stack, _stack.symbol)
                                except Exception:
                                    logger.debug("Pattern stack registration failed", exc_info=True)
                            try:
                                from mae_core.market.plain_language import (
                                    format_pattern_stack_alert, write_plain_alert,
                                )
                                _tmpl_windows = []
                                for _act in _stack.activations:
                                    _tw = getattr(_act.template, "expected_move_window_days", None)
                                    if _tw is not None:
                                        _tmpl_windows.append(_tw)
                                if _tmpl_windows:
                                    _tmpl_windows.sort()
                                    _win_days = _tmpl_windows[len(_tmpl_windows) // 2]
                                    _win_src = "dynamic"
                                else:
                                    _win_days = 14
                                    _win_src = "fallback"
                                _msg = format_pattern_stack_alert(
                                    _stack, window_days=_win_days, window_source=_win_src,
                                )
                                write_plain_alert(
                                    _msg, _stack.symbol, _stack.direction,
                                    source="pattern_stack",
                                    metadata={"tier": _stack.tier,
                                              "confidence": _stack.stack_confidence,
                                              "n_patterns": len(_stack.activations)},
                                )
                            except Exception:
                                logger.debug("Plain-language alert failed", exc_info=True)
                            _at = getattr(ctx, "active_tracker", None)
                            if _at is not None:
                                try:
                                    _pf = getattr(ctx, "price_fetcher", None)
                                    _entry = 0.0
                                    if _pf is not None:
                                        _pd = _pf.get_current_price(_stack.symbol)
                                        if _pd and _pd.price > 0:
                                            _entry = _pd.price
                                    if _entry > 0:
                                        _at.register(_stack, _entry)
                                except Exception:
                                    logger.debug("Active tracker registration failed", exc_info=True)
        except Exception as exc:
            logger.debug("Pattern watcher check failed", exc_info=True)
            if _shm_sensing:
                _shm_sensing.record_error("pattern_watcher", exc)


def _run_active_tracker_check(ctx: SimpleNamespace) -> None:
    """Active tracker price check — writes plain-language status updates.

    Called every 20 steps from _sensing_step_with_advisory.
    """
    _at = getattr(ctx, "active_tracker", None)
    if _at is None or _at.count <= 0:
        return
    try:
        _events = _at.check_prices()
        if _events:
            try:
                from mae_core.market.plain_language import write_plain_alert
                for _ev in _events:
                    _status = _ev["new_status"]
                    _sym = _ev["symbol"]
                    _pct = _ev.get("current_pct", 0)
                    if _status == "confirmed":
                        _msg = (
                            f"UPDATE: {_sym} prediction CONFIRMED. "
                            f"Price moved {abs(_pct):.1f}% in the expected direction. "
                            f"MIDGE was right on this one."
                        )
                    elif _status == "failed":
                        _msg = (
                            f"UPDATE: {_sym} prediction DID NOT PLAY OUT. "
                            f"Price moved {abs(_pct):.1f}% in the wrong direction. "
                            f"MIDGE is learning from this outcome."
                        )
                    elif _status == "expired":
                        _msg = (
                            f"UPDATE: {_sym} prediction window expired. "
                            f"Final move: {_pct:+.1f}%. "
                            f"The expected move did not materialize in time."
                        )
                    elif _status == "confirming":
                        _msg = (
                            f"UPDATE: {_sym} is showing early signs of the expected move "
                            f"({_pct:+.1f}% so far). MIDGE is watching closely."
                        )
                    else:
                        continue
                    write_plain_alert(
                        _msg, _sym, _ev.get("direction", ""),
                        source="active_tracking",
                        metadata={"status": _status, "pct_change": _pct},
                    )
            except Exception:
                logger.debug("Active tracking alert write failed", exc_info=True)
    except Exception:
        logger.debug("Active tracker check failed", exc_info=True)


def _run_synergy_detection(ctx: SimpleNamespace) -> None:
    """Detect dual confirmation: convergence alert + pattern stack on same ticker.

    Called every step from _sensing_step_with_advisory after pattern watcher runs.
    Publishes CH_DUAL_CONFIRMATION when both fire on the same ticker + direction.
    """
    from mae_core.market.channels import CH_DUAL_CONFIRMATION

    _conv_alerts = ctx._cached_alerts[0] or []
    _p_stacks = getattr(ctx, "_cached_pattern_stacks", [])
    if not _conv_alerts or not _p_stacks:
        return
    try:
        _stack_keys = set()
        for _ps in _p_stacks:
            _stack_keys.add((_ps.symbol, _ps.direction))

        for _alert in _conv_alerts:
            _alert_sym = None
            for _sig in getattr(_alert, "signals", []):
                _alert_sym = getattr(_sig, "metadata", {}).get("symbol", "")
                if _alert_sym:
                    break
            if not _alert_sym:
                _alert_sym = getattr(_alert, "ticker", "")
            _alert_dir = getattr(_alert, "direction", "")
            if _alert_sym and (_alert_sym, _alert_dir) in _stack_keys:
                _match_stack = next(
                    (s for s in _p_stacks
                     if s.symbol == _alert_sym and s.direction == _alert_dir),
                    None,
                )
                if _match_stack is not None:
                    logger.info(
                        "DUAL CONFIRMATION: %s %s — convergence (%.2f) + "
                        "pattern stack (%d patterns, %.2f confidence)",
                        _alert_sym, _alert_dir.upper(),
                        _alert.confidence, len(_match_stack.activations),
                        _match_stack.stack_confidence,
                    )
                    if getattr(ctx, "bus", None) is not None:
                        ctx.bus.publish(
                            CH_DUAL_CONFIRMATION,
                            {
                                "symbol": _alert_sym,
                                "direction": _alert_dir,
                                "convergence_confidence": _alert.confidence,
                                "convergence_strength": _alert.strength,
                                "convergence_domains": getattr(_alert, "domains_converging", []),
                                "stack_confidence": _match_stack.stack_confidence,
                                "stack_patterns": len(_match_stack.activations),
                                "stack_tier": _match_stack.tier,
                                "stack_independent_pairs": _match_stack.independent_pairs,
                                "combined_confidence": min(
                                    0.99,
                                    1.0 - (1.0 - _alert.confidence) * (1.0 - _match_stack.stack_confidence),
                                ),
                            },
                        )
    except Exception:
        logger.debug("Synergy detection failed", exc_info=True)


def _wire_sensing_hook(ctx: SimpleNamespace) -> None:
    """Wire the MarketSensingHook into the step lifecycle.

    This is the critical missing link: data never flowed into the
    convergence_alerter during agent runs. Everything downstream was
    already wired (endocrine coupling, body state, advisory). This
    function connects the sensory organs to the nervous system.

    Biological analogy: Wiring the optic nerve — the eyes existed,
    the visual cortex existed, but signals never traveled between them.

    Reads ctx._cached_alerts (written by _market_sense_hook in
    _register_market_step_hooks). That function MUST be called first.
    """
    from mae_core.market.sensing_hook import MarketSensingHook
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
    from mae_core.bootstrap.market_hooks_trades import (
        _write_paper_trade, _translate_and_log_executable_signal,
    )

    # --- Tiered ConvergenceAlerters (tactical/strategic/thematic) ---
    tiered_alerters = {}
    try:
        tiered_alerters["tactical"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=48,
        )
        tiered_alerters["strategic"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=21 * 24,
        )
        tiered_alerters["thematic"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=90 * 24,
        )
    except Exception:
        logger.debug("Tiered alerter construction failed", exc_info=True)

    # --- OutcomeCollector (signal → prediction → Thompson feedback) ---
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

    # --- SignalMemory (Qdrant persistence) ---
    memory = None
    try:
        from mae_core.market.memory import SignalMemory
        qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
        memory = SignalMemory(qdrant_url=qdrant_url)
    except Exception:
        logger.debug("SignalMemory construction failed", exc_info=True)

    # --- 8-K text sentiment via Ollama ---
    form8k_sentiment = None
    try:
        from mae_core.market.edge.form8k_sentiment import Form8KSentimentAnalyzer
        form8k_sentiment = Form8KSentimentAnalyzer()
    except Exception:
        logger.debug("Form8KSentimentAnalyzer construction failed", exc_info=True)

    # --- Market clock for time-aware source selection (WP-B) ---
    market_clock = None
    try:
        from mae_core.market.market_clock import MarketClock
        market_clock = MarketClock()
        ctx.market_clock = market_clock
        logger.info("Market clock initialized (timezone: US/Eastern)")
    except Exception:
        logger.debug("Market clock initialization failed", exc_info=True)

    # --- Instantiate the sensing hook ---
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
        logger.warning("MarketSensingHook construction failed — agents will not sense market data", exc_info=True)
        return

    hook._correlation_tracker = getattr(ctx, "correlation_tracker", None)
    hook._absence_monitor = getattr(ctx, "absence_monitor", None)

    somatic = getattr(ctx, "somatic_anticipation", None)
    if somatic is not None:
        somatic._endocrine_system = getattr(ctx, "endocrine", None)

    ctx._tiered_alerters = tiered_alerters
    ctx._market_advisory = {
        "alert": None,
        "updated_step": 0,
        "active_hypotheses": 0,
        "tactical": None,
        "strategic": None,
        "thematic": None,
        "ticker_alerts": [],
    }
    ctx._ticker_alerts = []
    ctx._latest_kelly = {}
    ctx._paper_trade_dedup = {}
    ctx._bypass_dedup = {}

    _sensing_step_counter = [0]
    original_step = hook.step

    def _sensing_step_with_advisory():
        """Wrap the sensing hook step to also update the market advisory."""
        _sensing_step_counter[0] += 1
        step = _sensing_step_counter[0]
        _shm = getattr(ctx, "system_health_monitor", None)
        try:
            original_step()
            if _shm:
                _shm.record_success("sensing")
            if _shm and step % 200 == 0:
                _shm.record_success("outcome_evaluation")
        except Exception as exc:
            logger.debug("Sensing hook step failed", exc_info=True)
            if _shm:
                _shm.record_error("sensing", exc)
                if step % 200 == 0:
                    _shm.record_error("outcome_evaluation", exc)

        alerts = ctx._cached_alerts[0] or []
        if alerts:
            try:
                strongest = max(alerts, key=lambda a: a.strength)
                ctx._market_advisory["alert"] = (
                    strongest.to_dict() if hasattr(strongest, "to_dict")
                    else {"direction": strongest.direction, "strength": strongest.strength}
                )
                ctx._market_advisory["updated_step"] = step
            except Exception:
                logger.debug("Advisory bridge failed", exc_info=True)

            try:
                from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
                _pt_conf = LEARNING_CONFIG.get("paper_trade_min_confidence", 0.45)
                _pt_str = LEARNING_CONFIG.get("paper_trade_min_strength", 0.65)
                _pt_combo = LEARNING_CONFIG.get("paper_trade_min_combo_mean", 0.25)
                for alert in alerts:
                    if (
                        hasattr(alert, "confidence")
                        and hasattr(alert, "strength")
                        and alert.confidence > _pt_conf
                        and alert.strength > _pt_str
                    ):
                        _pass_combo = True
                        _ts = getattr(ctx, "thompson_sampler", None)
                        _raw_domains = getattr(alert, "domains_converging", None)
                        if _raw_domains and _ts is not None:
                            _domains = sorted(_raw_domains)
                            if len(_domains) >= 2:
                                _combo_key = "combo:" + "+".join(_domains)
                                _cd = _ts.get_distribution(_combo_key)
                                if _cd.samples >= 3 and _cd.mean < _pt_combo:
                                    _pass_combo = False
                        if _pass_combo:
                            _dm = getattr(ctx, "drawdown_monitor", None)
                            if _dm and _dm.is_trading_halted():
                                logger.info("Paper trade BLOCKED — drawdown circuit breaker active")
                            else:
                                _sm = getattr(ctx, "self_monitor", None)
                                if _sm:
                                    _sm.record_alert(
                                        direction=getattr(alert, "direction", "unknown"),
                                        confidence=getattr(alert, "confidence", 0.0),
                                        ticker=getattr(alert, "ticker", ""),
                                        step=step,
                                    )
                                    if _sm.is_alerting_suppressed():
                                        logger.warning("Paper trade BLOCKED — behavioral anomaly detected: %s", _sm._anomaly_flags)
                                    else:
                                        _write_paper_trade(alert, ctx)
                                        _translate_and_log_executable_signal(alert, ctx)
                                else:
                                    _write_paper_trade(alert, ctx)
                                    _translate_and_log_executable_signal(alert, ctx)
            except Exception:
                logger.debug("Paper trading gate failed", exc_info=True)

        # Every 10 steps: query tiered alerters
        if step % 10 == 0 and ctx._tiered_alerters:
            for tier_name, tier_alerter in ctx._tiered_alerters.items():
                try:
                    tier_alerts = tier_alerter.check_convergence()
                    if tier_alerts:
                        strongest = max(tier_alerts, key=lambda a: a.strength)
                        ctx._market_advisory[tier_name] = (
                            strongest.to_dict() if hasattr(strongest, "to_dict")
                            else {"direction": strongest.direction, "strength": strongest.strength}
                        )
                    else:
                        ctx._market_advisory[tier_name] = None
                except Exception:
                    logger.debug("Tiered alerter %s query failed", tier_name, exc_info=True)

        # Every 10 steps: pattern archaeology stacking detection
        _shm_sensing = getattr(ctx, "system_health_monitor", None)
        if step % 10 == 0:
            _run_sensing_archaeology(ctx, step, _shm_sensing)

        # Active tracker price check (every 20 steps)
        if step % 20 == 0:
            _run_active_tracker_check(ctx)

        # Synergy detection: convergence alerts + pattern stacks on same ticker
        _run_synergy_detection(ctx)

    ctx.model.add_step_hook(_sensing_step_with_advisory)

    hook._bus = ctx.bus

    _ws = getattr(ctx, "finnhub_websocket", None)
    if _ws is not None:
        try:
            _ws.start()
            logger.info("Layer 33h - FinnhubWebSocket background thread started")
        except Exception:
            logger.debug("FinnhubWebSocket start() failed", exc_info=True)

    if _ws is not None:
        try:
            original_shutdown = getattr(hook, "shutdown", None)

            def _shutdown_with_ws():
                try:
                    _ws.stop()
                    logger.info("FinnhubWebSocket stopped")
                except Exception:
                    pass
                if original_shutdown is not None:
                    original_shutdown()

            hook.shutdown = _shutdown_with_ws
        except Exception:
            logger.debug("FinnhubWebSocket shutdown hook registration failed", exc_info=True)

    colony = getattr(ctx, "octopus_colony", None)
    if colony is not None:
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

        def _on_octopus_spawn(channel, data):
            msg = data if isinstance(data, dict) else {}
            oct_id = msg.get("octopus_id", "")
            oct_obj = colony.octopuses.get(oct_id)
            if oct_obj is None:
                return
            cognition = getattr(oct_obj, "cognition", oct_obj)
            for arm in getattr(cognition, "arms", {}).values():
                try:
                    patch_new_arm(colony, arm)
                except Exception:
                    logger.debug("Failed to patch arm on spawned %s", oct_id, exc_info=True)

        bus = getattr(ctx, "bus", None)
        if bus is not None:
            bus.register_callback("octopus.spawn", _on_octopus_spawn)

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

    _sched = getattr(ctx, "inhabitant_scheduler", None)
    if _sched is not None:
        try:
            _sched.start()
            logger.info("InhabitantScheduler: daemon thread started")
        except Exception:
            logger.debug("InhabitantScheduler start() failed", exc_info=True)

    ctx._market_sensing_hook = hook

    logger.info(
        "Layer 33h - Market sensing hook wired: "
        "async fetch (cadence=25, slots=8), outcome tracking (cadence=200), "
        "tiered alerters (%d), advisory bridge active",
        len(tiered_alerters),
    )
