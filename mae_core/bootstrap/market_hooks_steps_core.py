"""Main step hook registration for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _register_market_step_hooks (with all nested closures).
Heavy sub-blocks live in market_hooks_steps.py:
  - _run_octopus_dispatch
  - _run_slow_cadence_ops
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_market_step_hooks(ctx: SimpleNamespace) -> None:
    """Register step hooks with cadence and deduplication."""
    from mae_core.market.channels import (
        CH_CONVERGENCE, CH_THOMPSON_STATS, CH_VELOCITY_ANOMALY,
    )
    from mae_core.bootstrap.market_hooks_trades import _check_sweep_bypass
    from mae_core.bootstrap.market_hooks_steps import (
        _write_convergence_heartbeat,
        _run_drift_detector,
        _run_octopus_dispatch,
        _run_slow_cadence_ops,
    )

    _step_counter = [0]
    _last_convergence_state = [None]  # {"direction": str, "strength": float}
    ctx._cached_alerts = [None]  # Shared: written by _market_sense_hook, read by advisory bridge
    ctx._cached_pattern_stacks = []  # Written by pattern watcher, read by synergy detector
    # Bug 3 fix: track last evaluated outcome count so forgetting gate can compare.
    _last_evaluated_count = [0]

    # Shared ecosystem whiteboard — lazy singleton so import errors don't break boot
    _shared_attention = [None]

    def _get_shared_attention():
        if _shared_attention[0] is None:
            try:
                from mae_core.market.ecosystem.shared_attention import SharedAttention
                _shared_attention[0] = SharedAttention()
            except Exception:
                logger.debug("SharedAttention unavailable", exc_info=True)
        return _shared_attention[0]

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

    def _market_sense_hook():  # noqa: C901
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
                ctx._cached_alerts[0] = alerts
                if _shm:
                    _shm.record_success("convergence_check")
                for alert in alerts:
                    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
                    last = _last_convergence_state[0]
                    direction = alert_dict.get("direction", "neutral")
                    strength = alert_dict.get("strength", 0.0)
                    is_new_direction = (last is None or last["direction"] != direction)
                    is_material_change = (
                        last is not None and abs(last["strength"] - strength) > 0.1
                    )
                    if is_new_direction or is_material_change:
                        ctx.bus.publish(CH_CONVERGENCE, alert_dict)
                        _last_convergence_state[0] = {"direction": direction, "strength": strength}
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
                ctx._cached_alerts[0] = []  # Clear stale alerts on failure
                logger.debug("Convergence alerter step failed", exc_info=True)
                if _shm:
                    _shm.record_error("convergence_check", exc)

        # Per-ticker convergence — surfaces ticker-specific inevitabilities
        if alerter is not None and hasattr(alerter, "check_ticker_convergence"):
            try:
                ticker_alerts = alerter.check_ticker_convergence(min_domains=3)
                for ta in ticker_alerts:
                    ta_dict = ta.to_dict() if hasattr(ta, "to_dict") else {}
                    ctx.bus.publish(CH_CONVERGENCE, ta_dict)
                    logger.info(
                        "TICKER CONVERGENCE: %s %s (%d domains, confidence=%.3f)",
                        getattr(ta, "ticker", ta_dict.get("primary_ticker", "?")),
                        ta_dict.get("direction", "?"),
                        len(ta_dict.get("domains_converging", [])),
                        ta_dict.get("confidence", 0),
                    )
                    _oc = getattr(ctx, "outcome_collector", None)
                    _sym = getattr(ta, "ticker", None) or ta_dict.get("primary_ticker", "")
                    if _oc and _sym:
                        try:
                            _oc.register_convergence_alert(ta, _sym)
                        except Exception:
                            pass
                    # Whiteboard: advertise hot ticker to other processes
                    _sa = _get_shared_attention()
                    if _sa is not None and _sym:
                        try:
                            _sa.update_hot_ticker(
                                ticker=_sym,
                                reason="convergence_building",
                                confidence=ta_dict.get("confidence", 0.0),
                                domains=len(ta_dict.get("domains_converging", [])),
                            )
                        except Exception:
                            logger.debug("SharedAttention hot ticker update failed", exc_info=True)
            except Exception:
                logger.debug("Per-ticker convergence failed", exc_info=True)

        # Semantic memory: embed new convergence alerts (non-blocking, fire-and-forget)
        _pm = getattr(ctx, "pattern_memory", None)
        _new_alerts = ctx._cached_alerts[0] or []
        if _pm is not None and _pm.is_available and _new_alerts:
            for _alert in _new_alerts:
                try:
                    _pm.remember_convergence_alert(_alert)
                except Exception:
                    logger.debug("PatternMemory embed failed for alert", exc_info=True)

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

        # Every 50 steps: stigmergy evaporation
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
                        ctx.bus.publish(CH_VELOCITY_ANOMALY, {"anomalies": len(anomalies)})
                except Exception:
                    logger.debug("Velocity detector step failed", exc_info=True)

            if alerter is not None:
                try:
                    _check_sweep_bypass(alerter, ctx)
                except Exception:
                    logger.debug("Session sweep bypass step failed", exc_info=True)

            dd = getattr(ctx, "drift_detector", None)
            if dd is not None:
                try:
                    _run_drift_detector(dd, ctx)
                except Exception:
                    logger.debug("Drift detector step failed", exc_info=True)

        # Every 200 steps: Bayesian forgetting
        if step % 75 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    _oc_for_gate = getattr(ctx, "outcome_collector", None)
                    current_evaluated = 0
                    if _oc_for_gate is not None:
                        try:
                            current_evaluated = (
                                _oc_for_gate.get_statistics().get("total_evaluated", 0)
                            )
                        except Exception:
                            pass
                    # Fix: require at least 10 new graded outcomes before permitting
                    # forgetting. The old gate (> 0, i.e. just 1 new outcome) allowed
                    # forgetting to outpace learning — one graded outcome unlocked decay
                    # that erased far more accumulated signal than that one outcome added.
                    MIN_NEW_OUTCOMES_FOR_FORGET = 10
                    if current_evaluated >= _last_evaluated_count[0] + MIN_NEW_OUTCOMES_FOR_FORGET:
                        regime_clf = getattr(ctx, "regime_classifier", None)
                        regime = regime_clf.classify() if regime_clf is not None else "default"
                        sampler.regime_aware_forget(regime)
                        _last_evaluated_count[0] = current_evaluated
                    else:
                        logger.debug(
                            "Skipping Thompson forget — fewer than %d new outcomes since last forget "
                            "(step=%d, total_evaluated=%d, last=%d)",
                            MIN_NEW_OUTCOMES_FOR_FORGET, step,
                            current_evaluated, _last_evaluated_count[0],
                        )
                except Exception as exc:
                    logger.debug("Thompson forgetting step failed", exc_info=True)
                    if _shm:
                        _shm.record_error("thompson", exc)
            _write_convergence_heartbeat(ctx, step)
            # Whiteboard: daemon heartbeat so parallel processes know main loop is alive
            _sa = _get_shared_attention()
            if _sa is not None:
                try:
                    _rc = getattr(ctx, "regime_classifier", None)
                    _regime = _rc.classify() if _rc is not None else "unknown"
                    _sa.heartbeat("daemon", last_step=step, regime=_regime)
                except Exception:
                    logger.debug("SharedAttention daemon heartbeat failed", exc_info=True)

        # Every 20 steps: OctopusColony coordination + investigation dispatch
        if step % 20 == 0:
            colony = getattr(ctx, "octopus_colony", None)
            if colony is not None:
                _run_octopus_dispatch(colony, step)

        # Every 500/1000/5000 steps: slow cadence ops
        if step % 200 == 0 or step % 500 == 0 or step % 2000 == 0:
            _run_slow_cadence_ops(ctx, step, _shm, _timer)

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

        # Every 50 steps: Kelly sizing on per-ticker convergence alerts
        if step % 50 == 0:
            sizer = getattr(ctx, "kelly_position_sizer", None)
            alerter = getattr(ctx, "convergence_alerter", None)
            if sizer is not None and alerter is not None:
                try:
                    alerts = alerter.check_ticker_convergence(min_domains=2)
                    ctx._ticker_alerts = alerts
                    ctx._market_advisory["ticker_alerts"] = [
                        a.to_dict() if hasattr(a, "to_dict") else {"direction": a.direction, "strength": a.strength}
                        for a in alerts
                    ]
                    for alert in alerts:
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
                    ticker_alerts = getattr(ctx, "_ticker_alerts", [])
                    tickers = set()
                    for a in ticker_alerts:
                        for sig in getattr(a, "signals", []):
                            sym = getattr(sig, "metadata", {}).get("symbol", "")
                            if sym:
                                tickers.add(sym)
                    now = datetime.now()
                    for sym in list(tickers)[:10]:
                        try:
                            price_data = pf.get_current_price(sym)
                            if price_data is None:
                                continue
                            price = float(price_data.price)
                            change_pct = float(getattr(price_data, "change_pct", 0.0) or 0.0)
                            if md is not None:
                                motif_sigs = md.update(sym, price, now)
                                for ms in motif_sigs:
                                    if hasattr(ctx, "bus"):
                                        ctx.bus.publish("market.intel.motif_detected", {
                                            "symbol": sym, "type": ms.signal_type,
                                            "strength": round(ms.strength, 4),
                                            "mp_value": round(ms.mp_value, 4),
                                        })
                                    if hasattr(ctx, "convergence_alerter"):
                                        source = "motif_match" if ms.signal_type == "motif" else "price_discord"
                                        direction = ("bearish" if change_pct > 0 else "bullish") if ms.signal_type == "discord" else ("bullish" if change_pct > 0 else "bearish")
                                        ctx.convergence_alerter.record_signal(
                                            signal_id=f"{source}_{sym}", strength=ms.strength,
                                            domain="technical", direction=direction,
                                            source=source, metadata={"symbol": sym},
                                        )
                            if sad is not None:
                                vol = float(getattr(price_data, "volume", 0) or 0)
                                vec = [change_pct, min(vol / 1e6, 10.0), 0.0, 0.0]
                                score = sad.update(vec)
                                if score >= sad.threshold:
                                    if hasattr(ctx, "bus"):
                                        ctx.bus.publish("market.intel.streaming_anomaly", {
                                            "symbol": sym, "score": round(score, 4),
                                        })
                                    if hasattr(ctx, "convergence_alerter"):
                                        direction = "bearish" if change_pct > 0 else "bullish"
                                        ctx.convergence_alerter.record_signal(
                                            signal_id=f"streaming_anomaly_{sym}",
                                            strength=min(score / 2.0, 1.0),
                                            domain="technical", direction=direction,
                                            source="streaming_anomaly", metadata={"symbol": sym},
                                        )
                        except Exception:
                            logger.debug("Pattern discovery failed for %s", sym, exc_info=True)
                except Exception:
                    logger.debug("Pattern discovery step failed", exc_info=True)

            # Refill curiosity investigation budget every 100 steps
            if hasattr(ctx, "_curiosity_budget"):
                ctx._curiosity_budget = 5
                ctx._curiosity_step_counter = 0

    ctx.model.add_step_hook(_market_sense_hook)
    logger.info(
        "Layer 33g - Market step hooks: 1 sense hook registered "
        "(cadence: convergence/1, stats/10, velocity/50, stigmergy-evap/50, forgetting/200, "
        "motif+anomaly/100, drift/50, lag/500, calibration/1000, backtest/5000)"
    )
