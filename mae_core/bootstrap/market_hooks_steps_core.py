"""Main step hook registration for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _register_market_step_hooks (with all nested closures).
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_market_step_hooks(ctx: SimpleNamespace) -> None:
    """Register step hooks with cadence and deduplication."""
    from mae_core.market.channels import (
        CH_CONVERGENCE, CH_DUAL_CONFIRMATION, CH_THOMPSON_STATS,
        CH_VELOCITY_ANOMALY,
    )
    from mae_core.bootstrap.market_hooks_trades import _check_sweep_bypass
    from mae_core.bootstrap.market_hooks_steps import (
        _write_convergence_heartbeat, _run_drift_detector,
    )

    _step_counter = [0]
    _last_convergence_state = [None]  # {"direction": str, "strength": float}
    ctx._cached_alerts = [None]  # Shared: written by _market_sense_hook, read by advisory bridge
    ctx._cached_pattern_stacks = []  # Written by pattern watcher, read by synergy detector
    # Bug 3 fix: track last evaluated outcome count so forgetting gate can compare.
    # Forgetting is skipped if no new outcomes have been graded since the last
    # forgetting event — prevents systematic erosion during quiet learning periods.
    _last_evaluated_count = [0]

    def _get_regime():
        """Get current market regime (cached daily, essentially free)."""
        rc = getattr(ctx, "regime_classifier", None)
        if rc is not None:
            try:
                return rc.classify()
            except Exception:
                logger.debug("Regime classifier failed, using default", exc_info=True)
        return "default"

    _timer = getattr(ctx, "step_timer", None)

    def _market_sense_hook():
        _step_counter[0] += 1
        step = _step_counter[0]

        _shm = getattr(ctx, "system_health_monitor", None)

        # Every step: check convergence (lightweight, pure in-memory)
        alerter = getattr(ctx, "convergence_alerter", None)
        if alerter is not None:
            try:
                if _timer is not None:
                    with _timer.track("convergence_check"):
                        alerts = alerter.check_convergence()
                else:
                    alerts = alerter.check_convergence()
                ctx._cached_alerts[0] = alerts  # Cache for advisory bridge (avoid duplicate call)
                if _shm:
                    _shm.record_success("convergence_check")
                for alert in alerts:
                    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
                    last = _last_convergence_state[0]
                    direction = alert_dict.get("direction", "neutral")
                    strength = alert_dict.get("strength", 0.0)

                    is_new_direction = (last is None or last["direction"] != direction)
                    is_material_change = (
                        last is not None
                        and abs(last["strength"] - strength) > 0.1
                    )

                    if is_new_direction or is_material_change:
                        ctx.bus.publish(CH_CONVERGENCE, alert_dict)
                        _last_convergence_state[0] = {
                            "direction": direction,
                            "strength": strength,
                        }

                    # Register combo-level prediction for Thompson feedback loop.
                    # Extracts primary symbol from contributing signals' metadata,
                    # falling back to alert.ticker if signals lack symbol metadata.
                    _oc = getattr(ctx, "outcome_collector", None)
                    if _oc is not None:
                        _sym = None
                        for _sig in (alert.signals if hasattr(alert, "signals") else []):
                            _sym = getattr(_sig, "metadata", {}).get("symbol", "")
                            if _sym:
                                break
                        if not _sym:
                            _sym = getattr(alert, "ticker", None) or ""
                        if _sym:
                            try:
                                _oc.register_convergence_alert(alert, _sym)
                            except Exception:
                                logger.debug("Combo registration failed", exc_info=True)
            except Exception as exc:
                logger.debug("Convergence alerter step failed", exc_info=True)
                if _shm:
                    _shm.record_error("convergence_check", exc)

        # Every 10 steps: Thompson stats (regime-aware)
        if step % 10 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    if _timer is not None:
                        with _timer.track("thompson_stats"):
                            regime = _get_regime()
                            stats = sampler.get_stats(regime)
                    else:
                        regime = _get_regime()
                        stats = sampler.get_stats(regime)
                    stats["regime"] = regime
                    ctx.bus.publish(CH_THOMPSON_STATS, stats)
                    if _shm:
                        _shm.record_success("thompson")
                except Exception as exc:
                    logger.debug("Thompson sampler stats step failed", exc_info=True)
                    if _shm:
                        _shm.record_error("thompson", exc)

        # Every 50 steps: stigmergy evaporation (triggers global pheromone decay)
        # StigmergicEnvironment._apply_decay() is lazy — it only fires when
        # sense_markers() is called. Without this, old convergence:ticker markers
        # accumulate indefinitely and task routing stays biased to stale positions.
        if step % 50 == 0:
            if hasattr(ctx, "stigmergy") and ctx.stigmergy is not None:
                try:
                    ctx.stigmergy.sense_markers(
                        position=(0.0, 0.0, 0.0),
                        radius=float("inf"),
                        marker_types=None,
                    )
                    logger.debug("Stigmergy evaporation triggered (step %d)", step)
                except Exception:
                    logger.debug("Stigmergy evaporation step failed", exc_info=True)

        # Every 50 steps: velocity anomaly scan
        if step % 50 == 0:
            vd = getattr(ctx, "velocity_detector", None)
            if vd is not None:
                try:
                    if _timer is not None:
                        with _timer.track("velocity_scan"):
                            anomalies = vd.detect_velocity_anomalies()
                    else:
                        anomalies = vd.detect_velocity_anomalies()
                    if anomalies:
                        ctx.bus.publish(CH_VELOCITY_ANOMALY,
                                        {"anomalies": len(anomalies)})
                except Exception:
                    logger.debug("Velocity detector step failed", exc_info=True)

            # Session sweep bypass — direct output for backtest-validated signals
            if alerter is not None:
                try:
                    _check_sweep_bypass(alerter, ctx)
                except Exception:
                    logger.debug("Session sweep bypass step failed", exc_info=True)

            # Drift detector: track regime signal distributions every 50 steps
            # Feeds price returns, volume proxy, VIX, and sentiment from cached alerts
            dd = getattr(ctx, "drift_detector", None)
            if dd is not None:
                try:
                    _run_drift_detector(dd, ctx)
                except Exception:
                    logger.debug("Drift detector step failed", exc_info=True)

        # Every 200 steps: Bayesian forgetting (decay old evidence)
        # Cadence matches outcome evaluation (sensing_hook._outcome_cadence=200)
        # so forgetting never outpaces learning.
        # Bug 3 fix: only forget if at least one outcome was graded since the
        # last forgetting event.  During quiet periods (no mature predictions)
        # forgetting would otherwise drive all distributions to the floor (2.0,
        # 2.0) with no offsetting learning signal.
        if step % 200 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    # Check if any new outcomes have been graded since last forget
                    _oc_for_gate = getattr(ctx, "outcome_collector", None)
                    current_evaluated = 0
                    if _oc_for_gate is not None:
                        try:
                            current_evaluated = (
                                _oc_for_gate.get_statistics().get("total_evaluated", 0)
                            )
                        except Exception:
                            pass

                    if current_evaluated > _last_evaluated_count[0]:
                        # New outcomes graded — safe to apply forgetting
                        regime_clf = getattr(ctx, "regime_classifier", None)
                        regime = regime_clf.classify() if regime_clf is not None else "default"
                        sampler.regime_aware_forget(regime)
                        _last_evaluated_count[0] = current_evaluated
                    else:
                        logger.debug(
                            "Skipping Thompson forget — no outcomes graded since last forget "
                            "(step=%d, total_evaluated=%d)",
                            step, current_evaluated,
                        )
                except Exception as exc:
                    logger.debug("Thompson forgetting step failed", exc_info=True)
                    if _shm:
                        _shm.record_error("thompson", exc)

            # Convergence heartbeat: overwrite data/midge/convergence_state.json
            _write_convergence_heartbeat(ctx, step)

        # Every 20 steps: OctopusColony coordination cycle
        if step % 20 == 0:
            colony = getattr(ctx, "octopus_colony", None)
            if colony is not None:
                try:
                    for oct_id, oct in list(colony.octopuses.items()):
                        oct.cognition.run_coordination_cycle()
                except Exception:
                    logger.debug("OctopusColony coordination failed", exc_info=True)

                # Dispatch investigation tasks for developing situations.
                # Each partial convergence gets an arm to check if new signals
                # have completed the picture.  Cap at 5 submissions per cycle
                # to avoid flooding the colony.
                try:
                    lock = getattr(colony, "_situations_lock", None)
                    situations_snapshot = {}
                    if lock is not None:
                        with lock:
                            situations_snapshot = dict(colony._developing_situations)
                    else:
                        situations_snapshot = dict(
                            getattr(colony, "_developing_situations", {})
                        )

                    task_budget = 5
                    for key, sit in situations_snapshot.items():
                        if task_budget <= 0:
                            break
                        check_count = sit.get("check_count", 0)
                        if check_count >= 20:
                            continue  # Will be evicted by situation_check

                        # Compute role affinity so the colony can route to a
                        # specialist octopus when one exists (soft preference).
                        preferred_role = None
                        try:
                            from mae_core.network.market_task_handlers import (
                                select_preferred_role,
                            )
                            preferred_role = select_preferred_role(
                                domains_seen=sit.get("domains_seen", []),
                                missing_domains=sit.get("missing_domains", []),
                                causal_predictions=sit.get("causal_predictions", []),
                            )
                        except Exception:
                            pass  # Non-critical — fall back to workload routing

                        task_data_inv: dict = {
                            "ticker": sit["ticker"],
                            "direction": sit["direction"],
                            "domains_seen": sit.get("domains_seen", []),
                            "missing_domains": sit.get("missing_domains", []),
                        }
                        if preferred_role is not None:
                            task_data_inv["preferred_role"] = preferred_role

                        colony.submit_task(
                            task_data_inv,
                            "investigate_partial",
                        )
                        task_budget -= 1

                        # Every 5th check: submit a situation_check for eviction
                        if check_count > 0 and check_count % 5 == 0:
                            if task_budget > 0:
                                colony.submit_task(
                                    {
                                        "ticker": sit["ticker"],
                                        "direction": sit["direction"],
                                    },
                                    "situation_check",
                                )
                                task_budget -= 1
                except Exception:
                    logger.debug(
                        "Investigation dispatcher failed", exc_info=True
                    )

        # Every 500 steps: lag-correlation analysis
        if step % 500 == 0:
            lag = getattr(ctx, "lag_correlation_analyzer", None)
            if lag is not None:
                try:
                    if _timer is not None:
                        with _timer.track("lag_correlation"):
                            findings = lag.analyze(lookback_days=90)
                    else:
                        findings = lag.analyze(lookback_days=90)
                    if findings and hasattr(ctx, "bus"):
                        ctx.bus.publish("market.intel.lag_finding", {
                            "count": len(findings),
                            "top": [
                                {"a": f.source_a, "b": f.source_b,
                                 "lag": f.lag_days, "r": f.correlation}
                                for f in findings[:3]
                            ],
                        })
                    # Task 3: Feed lag findings into convergence alerter for sequence scoring.
                    # Future alerts will be scored by whether their domain firing order
                    # matches known lead-lag relationships from the archive analysis.
                    if findings:
                        _ca = getattr(ctx, "convergence_alerter", None)
                        if _ca is not None and hasattr(_ca, "set_lag_findings"):
                            try:
                                _ca.set_lag_findings(findings)
                            except Exception:
                                logger.debug("set_lag_findings failed", exc_info=True)
                except Exception:
                    logger.debug("Lag correlation step failed", exc_info=True)

            granger = getattr(ctx, "granger_analyzer", None)
            if granger is not None:
                try:
                    if _timer is not None:
                        with _timer.track("granger_causality"):
                            g_findings = granger.analyze(lookback_days=180)
                    else:
                        g_findings = granger.analyze(lookback_days=180)
                    if g_findings and hasattr(ctx, "bus"):
                        ctx.bus.publish("market.intel.granger_finding", {
                            "count": len(g_findings),
                            "top": [
                                {"cause": f.cause_source, "effect": f.effect_source,
                                 "lag": f.best_lag, "p": f.p_value}
                                for f in g_findings[:3]
                            ],
                        })
                except Exception:
                    logger.debug("Granger causality step failed", exc_info=True)

            # Post-mortem review: periodic retrospective analysis of graded outcomes.
            # Runs at 500-step cadence alongside Granger/Lag to keep learning in sync.
            post_mortem = getattr(ctx, "post_mortem_reviewer", None)
            if post_mortem is not None:
                try:
                    if _timer is not None:
                        with _timer.track("post_mortem"):
                            pm_summary = post_mortem.review()
                    else:
                        pm_summary = post_mortem.review()
                    if pm_summary.get("outcomes_reviewed", 0) > 0:
                        logger.info(
                            "PostMortem: reviewed %d outcomes (%d combos, %d sequences)",
                            pm_summary.get("outcomes_reviewed", 0),
                            pm_summary.get("combos_analyzed", 0),
                            pm_summary.get("sequences_analyzed", 0),
                        )
                    if _shm:
                        _shm.record_success("post_mortem")
                except Exception as exc:
                    logger.debug("Post-mortem review step failed", exc_info=True)
                    if _shm:
                        _shm.record_error("post_mortem", exc)

        # Every 1000 steps: Thompson calibration diagnostic
        if step % 1000 == 0:
            calibrator = getattr(ctx, "thompson_calibrator", None)
            if calibrator is not None:
                try:
                    if _timer is not None:
                        with _timer.track("thompson_calibration"):
                            calibrator.calibrate()
                    else:
                        calibrator.calibrate()
                except Exception:
                    logger.debug("Thompson calibration step failed", exc_info=True)

        # Every 5000 steps: backtest scheduler staleness check + excavation
        if step % 5000 == 0:
            scheduler = getattr(ctx, "backtest_scheduler", None)
            if scheduler is not None:
                try:
                    scheduler.check_and_schedule()
                except Exception:
                    logger.debug("Backtest scheduler check failed", exc_info=True)

            # Excavation daemon: continuous background pattern discovery
            # Submitted to sensing hook's thread pool so it never blocks the main step loop.
            daemon = getattr(ctx, "excavation_daemon", None)
            if daemon is not None:
                sensing_hook = getattr(ctx, "_market_sensing_hook", None)
                executor = getattr(sensing_hook, "_executor", None) if sensing_hook is not None else None
                if executor is not None:
                    def _run_excavation(d=daemon):
                        try:
                            summary = d.step()
                            if summary.get("fingerprints_found", 0) > 0:
                                logger.info(
                                    "Excavation: %d fingerprints, %d new templates (%d/%d symbols done)",
                                    summary.get("fingerprints_found", 0),
                                    summary.get("new_templates", 0),
                                    summary.get("symbols_done", 0),
                                    summary.get("symbols_done", 0) + summary.get("symbols_remaining", 0),
                                )
                        except Exception:
                            logger.debug("Excavation daemon step failed", exc_info=True)
                    executor.submit(_run_excavation)
                else:
                    # Fallback: run synchronously if sensing hook executor unavailable
                    try:
                        summary = daemon.step()
                        if summary.get("fingerprints_found", 0) > 0:
                            logger.info(
                                "Excavation: %d fingerprints, %d new templates (%d/%d symbols done)",
                                summary.get("fingerprints_found", 0),
                                summary.get("new_templates", 0),
                                summary.get("symbols_done", 0),
                                summary.get("symbols_done", 0) + summary.get("symbols_remaining", 0),
                            )
                    except Exception:
                        logger.debug("Excavation daemon step failed", exc_info=True)

        # Every step: hypothesis engine (manages its own cadence internally)
        hyp_engine = getattr(ctx, "hypothesis_engine", None)
        if hyp_engine is not None:
            try:
                if _timer is not None:
                    with _timer.track("hypothesis_engine"):
                        hyp_engine.step()
                else:
                    hyp_engine.step()
                if _shm:
                    _shm.record_success("hypothesis_engine")
            except Exception as exc:
                logger.debug("Hypothesis engine step failed", exc_info=True)
                if _shm:
                    _shm.record_error("hypothesis_engine", exc)

        # Kelly sizing: fires on per-ticker convergence alerts
        if step % 50 == 0:
            sizer = getattr(ctx, "kelly_position_sizer", None)
            alerter = getattr(ctx, "convergence_alerter", None)
            if sizer is not None and alerter is not None:
                try:
                    alerts = alerter.check_ticker_convergence(min_domains=2)
                    # Store per-ticker alerts for agent access (Phase 1c)
                    ctx._ticker_alerts = alerts
                    ctx._market_advisory["ticker_alerts"] = [
                        a.to_dict() if hasattr(a, "to_dict") else {"direction": a.direction, "strength": a.strength}
                        for a in alerts
                    ]
                    for alert in alerts:
                        # Extract symbol from alert signals' metadata
                        symbols = set()
                        for sig in alert.signals:
                            sym = sig.metadata.get("symbol", "")
                            if sym:
                                symbols.add(sym)
                        source = alert.signals[0].source if alert.signals else "unknown"
                        for symbol in symbols:
                            rec = sizer.recommend(source, symbol)
                            if rec.kelly_capped > 0 and hasattr(ctx, "bus"):
                                ctx.bus.publish("market.intel.kelly_sizing", {
                                    "symbol": symbol,
                                    "source": source,
                                    "kelly_capped": rec.kelly_capped,
                                    "p_win": rec.p_win,
                                    "confidence": rec.confidence_in_sizing,
                                })
                except Exception:
                    logger.debug("Kelly sizing step failed", exc_info=True)

        # Every 100 steps: motif detection + streaming anomaly across tracked tickers
        if step % 100 == 0:
            md = getattr(ctx, "motif_detector", None)
            sad = getattr(ctx, "streaming_anomaly", None)
            pf = getattr(ctx, "price_fetcher", None)
            if (md is not None or sad is not None) and pf is not None:
                try:
                    # Collect tickers from recent ticker alerts (already cached)
                    ticker_alerts = getattr(ctx, "_ticker_alerts", [])
                    tickers = set()
                    for a in ticker_alerts:
                        for sig in getattr(a, "signals", []):
                            sym = getattr(sig, "metadata", {}).get("symbol", "")
                            if sym:
                                tickers.add(sym)

                    now = datetime.now()
                    for sym in list(tickers)[:10]:   # cap at 10 per cycle
                        try:
                            price_data = pf.get_current_price(sym)
                            if price_data is None:
                                continue
                            price = float(price_data.price)
                            change_pct = float(getattr(price_data, "change_pct", 0.0) or 0.0)

                            # Motif detection — feed into convergence alerter
                            if md is not None:
                                motif_sigs = md.update(sym, price, now)
                                for ms in motif_sigs:
                                    if hasattr(ctx, "bus"):
                                        ctx.bus.publish("market.intel.motif_detected", {
                                            "symbol": sym,
                                            "type": ms.signal_type,
                                            "strength": round(ms.strength, 4),
                                            "mp_value": round(ms.mp_value, 4),
                                        })
                                    # Wire into convergence alerter as technical signal.
                                    # Direction: discords are contrarian (anomaly → reversal),
                                    # motifs follow recent trend.
                                    if hasattr(ctx, "convergence_alerter"):
                                        source = "motif_match" if ms.signal_type == "motif" else "price_discord"
                                        direction = ("bearish" if change_pct > 0 else "bullish") if ms.signal_type == "discord" else ("bullish" if change_pct > 0 else "bearish")
                                        ctx.convergence_alerter.record_signal(
                                            signal_id=f"{source}_{sym}",
                                            strength=ms.strength,
                                            domain="technical",
                                            direction=direction,
                                            source=source,
                                            metadata={"symbol": sym},
                                        )

                            # Streaming anomaly: [price_change, volume_ratio, 0, 0]
                            # volume_ratio and sentiment default to 0 (unknown at this point)
                            if sad is not None:
                                vol = float(getattr(price_data, "volume", 0) or 0)
                                vec = [change_pct, min(vol / 1e6, 10.0), 0.0, 0.0]
                                score = sad.update(vec)
                                if score >= sad.threshold:
                                    if hasattr(ctx, "bus"):
                                        ctx.bus.publish("market.intel.streaming_anomaly", {
                                            "symbol": sym,
                                            "score": round(score, 4),
                                        })
                                    # Wire into convergence alerter — anomaly is contrarian
                                    if hasattr(ctx, "convergence_alerter"):
                                        direction = "bearish" if change_pct > 0 else "bullish"
                                        ctx.convergence_alerter.record_signal(
                                            signal_id=f"streaming_anomaly_{sym}",
                                            strength=min(score / 2.0, 1.0),
                                            domain="technical",
                                            direction=direction,
                                            source="streaming_anomaly",
                                            metadata={"symbol": sym},
                                        )
                        except Exception:
                            logger.debug("Pattern discovery failed for %s", sym, exc_info=True)
                except Exception:
                    logger.debug("Pattern discovery step failed", exc_info=True)

    ctx.model.add_step_hook(_market_sense_hook)
    logger.info(
        "Layer 33g - Market step hooks: 1 sense hook registered "
        "(cadence: convergence/1, stats/10, velocity/50, stigmergy-evap/50, forgetting/200, "
        "motif+anomaly/100, drift/50, lag/500, calibration/1000, backtest/5000)"
    )
