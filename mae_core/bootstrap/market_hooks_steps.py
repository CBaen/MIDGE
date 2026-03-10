"""Step hook helpers for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _write_convergence_heartbeat, _run_drift_detector,
          _run_slow_cadence_ops (every-500/1000/5000-step analysis).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _write_convergence_heartbeat(ctx: SimpleNamespace, step: int) -> None:
    """Overwrite data/midge/convergence_state.json with current snapshot.

    Called every 100 steps. Single JSON file (not append) — always
    current, cheap to read for monitoring. Never blocks step loop.
    """
    import json as _json
    from datetime import datetime as _dt

    try:
        heartbeat: dict = {"step": step, "ts": _dt.now().isoformat(timespec="seconds")}

        # Regime
        rc = getattr(ctx, "regime_classifier", None)
        heartbeat["regime"] = rc.classify() if rc is not None else "unknown"

        # Global convergence
        adv = getattr(ctx, "_market_advisory", None)
        if adv is not None:
            alert = adv.get("alert")
            if alert is not None and isinstance(alert, dict):
                heartbeat["global"] = {
                    "direction": alert.get("direction", "neutral"),
                    "strength": alert.get("strength", 0.0),
                    "domains": len(alert.get("domains", [])),
                }
            else:
                heartbeat["global"] = None

            heartbeat["tactical"] = adv.get("tactical")
            heartbeat["strategic"] = adv.get("strategic")
        else:
            heartbeat["global"] = None

        # Per-ticker alerts
        ticker_alerts = getattr(ctx, "_ticker_alerts", [])
        heartbeat["ticker_alerts"] = {}
        for ta in ticker_alerts:
            if hasattr(ta, "to_dict"):
                td = ta.to_dict()
                # Try to extract symbol from signals
                symbols = set()
                for sig in getattr(ta, "signals", []):
                    sym = getattr(sig, "metadata", {}).get("symbol", "")
                    if sym:
                        symbols.add(sym)
                for sym in symbols:
                    heartbeat["ticker_alerts"][sym] = td.get("direction", "neutral")

        # Hypothesis stats
        hyp_engine = getattr(ctx, "hypothesis_engine", None)
        if hyp_engine is not None:
            try:
                hyp_stats = hyp_engine.get_statistics()
                heartbeat["hypotheses"] = {
                    "active": hyp_stats.get("active_count", 0),
                    "probation": hyp_stats.get("probation_count", 0),
                    "generated": hyp_stats.get("hypotheses_generated", 0),
                    "promoted": hyp_stats.get("hypotheses_promoted", 0),
                }
            except Exception:
                heartbeat["hypotheses"] = None
        else:
            heartbeat["hypotheses"] = None

        # Kelly
        heartbeat["kelly"] = getattr(ctx, "_latest_kelly", {})

        # Write
        out_dir = Path("data/midge")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "convergence_state.json"
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(heartbeat, f, indent=2, default=str)

    except Exception:
        logger.debug("Convergence heartbeat write failed", exc_info=True)


def _run_drift_detector(dd, ctx: SimpleNamespace) -> None:
    """Feed current market signal values into DriftDetector.

    Pulls regime-relevant scalars from live system state:
      - price_returns: SPY daily return (from regime classifier prices if available)
      - vix: VIX level (from vix_client state if available)
      - sentiment: mean sentiment from cached convergence signals
      - volume: placeholder (uses convergence signal count as proxy)

    When drift is detected in any stream, publishes market.intel.drift_detected
    on the bus so downstream systems can react (e.g. RegimeClassifier re-run).
    """
    try:
        # Price returns: read from regime classifier's reference symbol history
        rc = getattr(ctx, "regime_classifier", None)
        if rc is not None:
            try:
                prices = rc._get_recent_prices()
                if prices and len(prices) >= 2:
                    ret = (prices[-1] - prices[-2]) / max(prices[-2], 1e-9)
                    drift, old_m, new_m = dd.update("price_returns", ret)
                    if drift and hasattr(ctx, "bus"):
                        ctx.bus.publish("market.intel.drift_detected", {
                            "stream": "price_returns",
                            "old_mean": round(old_m, 6),
                            "new_mean": round(new_m, 6),
                        })
            except Exception:
                pass

        # VIX: read from vix_client if available
        vix_c = getattr(ctx, "vix_client", None)
        if vix_c is not None:
            vix_val = getattr(vix_c, "_last_vix", None) or getattr(vix_c, "last_vix_level", None)
            if vix_val is not None:
                drift, old_m, new_m = dd.update("vix", float(vix_val))
                if drift and hasattr(ctx, "bus"):
                    ctx.bus.publish("market.intel.drift_detected", {
                        "stream": "vix",
                        "old_mean": round(old_m, 4),
                        "new_mean": round(new_m, 4),
                    })

        # Convergence signal count as volume proxy
        cached = getattr(ctx, "_cached_alerts", [None])
        alerts = cached[0] if cached else None
        if alerts is not None:
            n_signals = sum(
                len(getattr(a, "signals", [])) for a in alerts
            ) if isinstance(alerts, list) else 0
            dd.update("signal_volume", float(n_signals))

    except Exception:
        logger.debug("_run_drift_detector failed", exc_info=True)


def _run_octopus_dispatch(colony, step: int) -> None:  # noqa: C901
    """Coordinate OctopusColony and dispatch investigation tasks.

    Called every 20 steps from _market_sense_hook.
    Runs coordination cycles + submits investigate_partial / situation_check tasks.
    """
    try:
        for oct_id, oct in list(colony.octopuses.items()):
            oct.cognition.run_coordination_cycle()
    except Exception:
        logger.debug("OctopusColony coordination failed", exc_info=True)

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

            preferred_role = None
            try:
                from mae_core.network.market_task_handlers import select_preferred_role
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

            colony.submit_task(task_data_inv, "investigate_partial")
            task_budget -= 1

            if check_count > 0 and check_count % 5 == 0:
                if task_budget > 0:
                    colony.submit_task(
                        {"ticker": sit["ticker"], "direction": sit["direction"]},
                        "situation_check",
                    )
                    task_budget -= 1
    except Exception:
        logger.debug("Investigation dispatcher failed", exc_info=True)


def _run_slow_cadence_ops(ctx: SimpleNamespace, step: int, _shm, _timer) -> None:
    """Run every-500-step analysis: lag-correlation, Granger causality, post-mortem.

    Also handles every-1000-step Thompson calibration and
    every-5000-step backtest scheduler + excavation daemon.
    """
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

    if step % 5000 == 0:
        scheduler = getattr(ctx, "backtest_scheduler", None)
        if scheduler is not None:
            try:
                scheduler.check_and_schedule()
            except Exception:
                logger.debug("Backtest scheduler check failed", exc_info=True)

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
