"""Bootstrap Layer 33f-h: Market EventBus callbacks, step hooks, and sensing hook.

One job: wire all runtime signal pathways.
Three sub-tasks:
  - EventBus callbacks: endocrine coupling for convergence + hypothesis lifecycle
  - Step hooks: cadenced operations (convergence/1, stats/10, velocity/50, etc.)
  - Sensing hook: MarketSensingHook + advisory bridge

Critical constraint: ctx._cached_alerts is the shared handshake between
_register_market_step_hooks() (writer) and _wire_sensing_hook() (reader).
_register_market_step_hooks() MUST be called before _wire_sensing_hook()
to ensure ctx._cached_alerts exists when the sensing hook wraps it.
Hook registration order also matters: _market_sense_hook runs before
_sensing_step_with_advisory because step hooks run in registration order.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")

# Sources eligible for the bypass path (backtest-validated, single-domain actionable).
# Min 1 domain instead of 3, quality gate replaces domain diversity requirement.
BYPASS_ELIGIBLE_SOURCES = {"session_sweep_ifvg"}


def _check_sweep_bypass(alerter, ctx: SimpleNamespace) -> None:
    """Direct output path for high-quality session sweep signals.

    Unlike standard convergence (min_domains=3, writes paper_trades.jsonl),
    this path uses min_domains=1 ticker convergence and applies a quality gate.
    Writes to data/midge/paper_trades_bypass.jsonl — a SEPARATE file.

    Gate: any contributing signal.source in BYPASS_ELIGIBLE_SOURCES
          AND alert quality >= 0.65 AND alert confidence >= 0.55.

    Dedup: ctx._bypass_dedup {"{direction}:{ticker}" -> datetime}, 4-hour window.
    """
    try:
        ticker_alerts = alerter.check_ticker_convergence(min_domains=1)
    except Exception:
        logger.debug("Sweep bypass ticker convergence failed", exc_info=True)
        return

    bypass_dedup = getattr(ctx, "_bypass_dedup", {})
    now = datetime.now()

    for alert in ticker_alerts:
        # Check that at least one contributing signal comes from an eligible source
        signals = getattr(alert, "signals", [])
        has_eligible = any(
            getattr(sig, "source", "") in BYPASS_ELIGIBLE_SOURCES
            for sig in signals
        )
        if not has_eligible:
            continue

        # Quality gate: pull from alert metadata or signals
        # Quality is stored in signal metadata from session_sweep_detector
        quality = 0.0
        for sig in signals:
            q = getattr(sig, "metadata", {}).get("quality", 0.0)
            if q > quality:
                quality = q

        confidence = getattr(alert, "confidence", 0.0)

        if quality < 0.65 or confidence < 0.55:
            logger.debug(
                "Sweep bypass rejected: quality=%.2f confidence=%.2f",
                quality, confidence,
            )
            continue

        # Resolve ticker and direction
        direction = getattr(alert, "direction", "neutral")
        if direction not in ("bullish", "bearish"):
            continue

        ticker = "UNKNOWN"
        for sig in signals:
            sym = getattr(sig, "metadata", {}).get("symbol", "")
            if sym:
                ticker = sym
                break

        # Dedup gate: same direction+ticker within 4 hours → skip
        dedup_key = f"{direction}:{ticker}"
        last_written = bypass_dedup.get(dedup_key)
        if last_written is not None and (now - last_written) < timedelta(hours=4):
            logger.debug("Bypass dedup suppressed: %s", dedup_key)
            continue

        # Write to separate bypass file
        try:
            alert_id = getattr(alert, "alert_id", None)
            signal_id = (
                f"BYP-{alert_id}" if alert_id
                else f"BYP-{now.strftime('%Y%m%d%H%M%S')}-{direction}"
            )
            domains = getattr(alert, "domains_converging", [])
            summary = getattr(alert, "summary", "")
            record = {
                "signal_id": signal_id,
                "asset": ticker,
                "asset_class": "futures",
                "direction": "buy" if direction == "bullish" else "sell",
                "confidence": round(float(confidence), 4),
                "quality": round(float(quality), 4),
                "bypass_reason": "backtest_validated",
                "contributing_signals": [
                    getattr(sig, "signal_id", "") for sig in signals
                    if getattr(sig, "signal_id", "")
                ],
                "domains": domains,
                "summary": summary,
                "generated_at": now.isoformat(),
            }
            bypass_path = Path("data/midge/paper_trades_bypass.jsonl")
            bypass_path.parent.mkdir(parents=True, exist_ok=True)
            with open(bypass_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            bypass_dedup[dedup_key] = now
            ctx._bypass_dedup = bypass_dedup

            logger.info(
                "Sweep bypass trade written: %s %s quality=%.2f confidence=%.2f",
                direction.upper(), ticker, quality, confidence,
            )
        except Exception:
            logger.debug("Sweep bypass write failed", exc_info=True)


def _write_paper_trade(alert, ctx: SimpleNamespace) -> None:
    """Convert a high-confidence ConvergenceAlert into a TradeSignal and persist it.

    Called when alert passes confidence + strength + combo gates (see learning_config).

    Dedup gate: same direction+ticker combination is suppressed for 4 hours.
    Writes to data/midge/paper_trades.jsonl (atomic append).
    Optionally registers with OutcomeCollector to close the Thompson feedback loop.
    """
    try:
        from mae_core.market.signal import TradeSignal, MarketSignal

        # --- Resolve ticker from alert signals ---
        ticker = "MULTI"
        asset_class = "stock"
        for sig in getattr(alert, "signals", []):
            sym = getattr(sig, "metadata", {}).get("symbol", "")
            if sym:
                ticker = sym
                break
            # Session sweep signals carry asset_class=futures
            if getattr(sig, "source", "") in ("session_sweep", "session_sweep_ifvg"):
                asset_class = "futures"

        # Determine asset class from signal metadata if available
        for sig in getattr(alert, "signals", []):
            ac = getattr(sig, "metadata", {}).get("asset_class", "")
            if ac:
                asset_class = ac
                break

        # --- Dedup gate: same direction+ticker within 4h → skip ---
        dedup_key = f"{alert.direction}:{ticker}"
        dedup = getattr(ctx, "_paper_trade_dedup", {})
        now = datetime.now()
        last_written = dedup.get(dedup_key)
        if last_written is not None and (now - last_written) < timedelta(hours=4):
            logger.debug(
                "Paper trade dedup suppressed: %s (last: %s)",
                dedup_key, last_written.isoformat(timespec="seconds"),
            )
            return

        # --- Resolve direction (ConvergenceAlert uses bullish/bearish) ---
        raw_direction = getattr(alert, "direction", "neutral")
        if raw_direction == "bullish":
            trade_direction = "buy"
        elif raw_direction == "bearish":
            trade_direction = "sell"
        else:
            return  # Neutral alerts are not actionable

        # --- Resolve catalyst text ---
        summary = getattr(alert, "summary", None)
        domains_converging = getattr(alert, "domains_converging", [])
        if summary:
            catalyst = summary
        else:
            catalyst = (
                f"{raw_direction} convergence across "
                f"{len(domains_converging)} domains: {', '.join(domains_converging)}"
            )

        # --- Build contributing signal IDs ---
        contributing_signals = [
            getattr(sig, "signal_id", "") for sig in getattr(alert, "signals", [])
            if getattr(sig, "signal_id", "")
        ]

        # --- Generate signal_id ---
        alert_id = getattr(alert, "alert_id", None)
        if alert_id:
            signal_id = f"PT-{alert_id}"
        else:
            signal_id = f"PT-{now.strftime('%Y%m%d%H%M%S')}-{trade_direction}"

        # --- Kelly fraction (best-effort) ---
        kelly_fraction: float | None = None
        latest_kelly = getattr(ctx, "_latest_kelly", {}) or {}
        if isinstance(latest_kelly, dict) and latest_kelly.get("symbol") == ticker:
            kelly_fraction = latest_kelly.get("kelly_capped")

        # --- Instantiate TradeSignal ---
        trade_signal = TradeSignal(
            signal_id=signal_id,
            asset=ticker,
            asset_class=asset_class,
            direction=trade_direction,
            confidence=round(float(alert.confidence), 4),
            timeframe_days=5,
            catalyst=catalyst,
            contributing_signals=contributing_signals,
            hit_rate=0.0,
            generated_at=now,
        )

        # --- Serialize to JSONL (generated_at → ISO string) ---
        record = asdict(trade_signal)
        record["generated_at"] = trade_signal.generated_at.isoformat()
        if kelly_fraction is not None:
            record["kelly_fraction"] = round(float(kelly_fraction), 4)

        trade_path = Path("data/midge/paper_trades.jsonl")
        trade_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trade_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # --- Update dedup dict (evict entries older than 24h to prevent unbounded growth) ---
        dedup[dedup_key] = now
        _cutoff = now - timedelta(hours=24)
        ctx._paper_trade_dedup = {k: v for k, v in dedup.items() if v > _cutoff}

        logger.info(
            "Paper trade written: %s %s %s (confidence=%.2f, strength=%.2f, kelly=%s)",
            trade_direction.upper(), ticker, asset_class,
            alert.confidence, alert.strength,
            f"{kelly_fraction:.3f}" if kelly_fraction is not None else "n/a",
        )

        # --- Register with OutcomeCollector (closes Thompson feedback loop) ---
        outcome_collector = getattr(ctx, "_market_sensing_hook", None)
        if outcome_collector is not None:
            outcome_collector = getattr(outcome_collector, "_outcome_collector", None)
        if outcome_collector is None:
            # Fallback: check ctx directly (some setups store it there)
            outcome_collector = getattr(ctx, "outcome_collector", None)

        if outcome_collector is not None:
            try:
                # Synthesize a minimal MarketSignal for outcome tracking
                synthetic = MarketSignal(
                    signal_id=signal_id,
                    source="convergence_alert",
                    symbol=ticker,
                    asset_class=asset_class,
                    domain="convergence",
                    direction=raw_direction,
                    strength=float(alert.strength),
                    confidence=float(alert.confidence),
                    decay_rate=0.05,
                    timestamp=now,
                    received_at=now,
                    outcome_symbol=ticker,
                    outcome_window_days=trade_signal.timeframe_days,
                    metadata={"alert_id": getattr(alert, "alert_id", ""), "paper_trade": True},
                )
                outcome_collector.register_signals([synthetic])
                logger.debug("Paper trade %s registered with OutcomeCollector", signal_id)
            except Exception:
                logger.debug("OutcomeCollector registration for paper trade failed", exc_info=True)

        # --- Write plain-language alert for convergence-based paper trade ---
        try:
            from mae_core.market.plain_language import (
                format_convergence_alert, write_plain_alert,
            )
            _msg = format_convergence_alert(
                alert, ticker,
                window_days=trade_signal.timeframe_days,
            )
            write_plain_alert(
                _msg, ticker, raw_direction,
                source="convergence_alert",
                metadata={"confidence": float(alert.confidence),
                          "strength": float(alert.strength)},
            )
        except Exception:
            logger.debug("Plain-language convergence alert failed", exc_info=True)

    except Exception:
        logger.debug("_write_paper_trade failed", exc_info=True)


def _register_market_eventbus(ctx: SimpleNamespace) -> None:
    """Wire EventBus channels — endocrine coupling for convergence alerts."""
    from mae_core.market.channels import CH_CONVERGENCE

    def _on_market_convergence(channel, data):
        msg = json.loads(data) if isinstance(data, str) else data
        strength = msg.get("strength", 0.0)
        direction = msg.get("direction", "neutral")

        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return

        try:
            from mae_core.coordination.endocrine_system import HormoneType

            if direction == "bullish" and strength > 0.7:
                endocrine.release_hormone(
                    HormoneType.DOPAMINE,
                    min(0.4, strength * 0.4),
                    "market_opportunity",
                )
            elif direction == "bearish" and strength > 0.7:
                endocrine.release_hormone(
                    HormoneType.ADRENALINE,
                    min(0.5, strength * 0.5),
                    "market_threat",
                )
        except Exception:
            logger.debug("Endocrine coupling failed for convergence alert", exc_info=True)

    ctx.bus.register_callback(CH_CONVERGENCE, _on_market_convergence)

    # --- Hypothesis lifecycle endocrine coupling ---
    from mae_core.market.channels import CH_HYPOTHESIS_PROMOTED, CH_HYPOTHESIS_RETIRED

    def _on_hypothesis_promoted(channel, data):
        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return
        try:
            from mae_core.coordination.endocrine_system import HormoneType
            endocrine.release_hormone(HormoneType.DOPAMINE, 0.3, "hypothesis_promoted")
        except Exception:
            pass

    def _on_hypothesis_retired(channel, data):
        msg = json.loads(data) if isinstance(data, str) else data
        if not msg.get("was_active", False):
            return  # Only cortisol if it was actively being used
        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return
        try:
            from mae_core.coordination.endocrine_system import HormoneType
            endocrine.release_hormone(HormoneType.CORTISOL, 0.15, "hypothesis_retired_unexpectedly")
        except Exception:
            pass

    ctx.bus.register_callback(CH_HYPOTHESIS_PROMOTED, _on_hypothesis_promoted)
    ctx.bus.register_callback(CH_HYPOTHESIS_RETIRED, _on_hypothesis_retired)

    # --- Hypothesis engine signal ingestion subscription ---
    engine = getattr(ctx, "hypothesis_engine", None)
    if engine is not None:
        from mae_core.market.channels import CH_SIGNAL_INGESTED
        ctx.bus.register_callback(CH_SIGNAL_INGESTED, engine.on_signal_ingested)

    # --- Kelly sizing subscriber (store latest recommendation on ctx) ---
    def _on_kelly_sizing(channel, data):
        msg = json.loads(data) if isinstance(data, str) else data
        ctx._latest_kelly = msg

    ctx.bus.register_callback("market.intel.kelly_sizing", _on_kelly_sizing)

    logger.info("Layer 33f - Market EventBus: convergence + hypothesis -> endocrine coupling wired")


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


def _register_market_step_hooks(ctx: SimpleNamespace) -> None:
    """Register step hooks with cadence and deduplication."""
    from mae_core.market.channels import (
        CH_CONVERGENCE, CH_DUAL_CONFIRMATION, CH_THOMPSON_STATS,
        CH_VELOCITY_ANOMALY,
    )

    _step_counter = [0]
    _last_convergence_state = [None]  # {"direction": str, "strength": float}
    ctx._cached_alerts = [None]  # Shared: written by _market_sense_hook, read by advisory bridge
    ctx._cached_pattern_stacks = []  # Written by pattern watcher, read by synergy detector

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
            except Exception:
                logger.debug("Convergence alerter step failed", exc_info=True)

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
                except Exception:
                    logger.debug("Thompson sampler stats step failed", exc_info=True)

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

        # Every 200 steps: Bayesian forgetting (decay old evidence)
        # Cadence matches outcome evaluation (sensing_hook._outcome_cadence=200)
        # so forgetting never outpaces learning.
        if step % 200 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    sampler.apply_forgetting(decay_factor=0.99)
                except Exception:
                    logger.debug("Thompson forgetting step failed", exc_info=True)

            # Convergence heartbeat: overwrite data/midge/convergence_state.json
            _write_convergence_heartbeat(ctx, step)

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
            except Exception:
                logger.debug("Hypothesis engine step failed", exc_info=True)

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

    ctx.model.add_step_hook(_market_sense_hook)
    logger.info(
        "Layer 33g - Market step hooks: 1 sense hook registered "
        "(cadence: convergence/1, stats/10, velocity/50, forgetting/200, "
        "lag/500, calibration/1000, backtest/5000)"
    )


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
    from mae_core.market.channels import CH_CONVERGENCE

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
        outcome_collector = OutcomeCollector(
            price_fetcher=getattr(ctx, "price_fetcher", None),
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            regime_classifier=getattr(ctx, "regime_classifier", None),
        )
    except Exception:
        logger.debug("OutcomeCollector construction failed", exc_info=True)

    # Store on ctx so combo Thompson feedback can find it (lines 265, 509)
    if outcome_collector is not None:
        ctx.outcome_collector = outcome_collector
        # Wire pattern library for template outcome feedback
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

    # --- Instantiate the sensing hook (with optional CorrelationTracker) ---
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
        )
    except Exception:
        logger.warning("MarketSensingHook construction failed — agents will not sense market data", exc_info=True)
        return

    # Inject CorrelationTracker (already bootstrapped in Layer 33a, receives no
    # data until now — Package C of "Completing the Circle" wiring)
    hook._correlation_tracker = getattr(ctx, "correlation_tracker", None)

    # Inject AbsenceMonitor (Package B of "Completing the Circle")
    hook._absence_monitor = getattr(ctx, "absence_monitor", None)

    # Wire endocrine system into somatic anticipation (Gift 9 — two-phase init)
    somatic = getattr(ctx, "somatic_anticipation", None)
    if somatic is not None:
        somatic._endocrine_system = getattr(ctx, "endocrine", None)

    # --- Store tiered alerters on ctx for agent access ---
    ctx._tiered_alerters = tiered_alerters

    # --- Market advisory dict (Channel B: supplements endocrine Channel A) ---
    # Separate from _latest_advisory which PatternCortex overwrites every step.
    # Market-role agents read this in their decision cascade.
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
    ctx._paper_trade_dedup = {}  # {"{direction}:{ticker}" -> datetime} dedup for paper trades
    ctx._bypass_dedup = {}  # {"{direction}:{ticker}" -> datetime} dedup for bypass trades

    # Wire convergence alerts into the advisory dict
    _sensing_step_counter = [0]
    original_step = hook.step

    def _sensing_step_with_advisory():
        """Wrap the sensing hook step to also update the market advisory."""
        _sensing_step_counter[0] += 1
        step = _sensing_step_counter[0]
        original_step()

        # Reuse cached convergence alerts (written by _market_sense_hook)
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

            # Paper trading gate — convert high-confidence convergence to TradeSignal
            # Thresholds from learning_config (replay-proven edge at 0.45)
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
                        # Combo filter: block combos with poor historical WR
                        _pass_combo = True
                        _ts = getattr(ctx, "thompson_sampler", None)
                        _raw_domains = getattr(alert, "domains_converging", None)
                        if _raw_domains and _ts is not None:
                            _domains = sorted(_raw_domains)
                            if len(_domains) >= 2:
                                _combo_key = "combo:" + "+".join(_domains)
                                _cd = _ts.get_distribution(_combo_key)
                                # Let unseen combos through (samples < 3), block known losers
                                if _cd.samples >= 3 and _cd.mean < _pt_combo:
                                    _pass_combo = False
                        if _pass_combo:
                            _write_paper_trade(alert, ctx)
            except Exception:
                logger.debug("Paper trading gate failed", exc_info=True)

        # Every 10 steps: query tiered alerters (tactical/strategic/thematic)
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
        if step % 10 == 0 and getattr(ctx, "pattern_watcher", None) is not None:
            try:
                # Build active signals from convergence alerter's signal buffer
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
                        # Register stacks for outcome tracking (Thompson feedback)
                        # + write plain-language alerts
                        _oc = getattr(ctx, "outcome_collector", None)
                        if _stacks:
                            for _stack in _stacks:
                                if _oc is not None:
                                    try:
                                        _oc.register_pattern_stack(_stack, _stack.symbol)
                                    except Exception:
                                        logger.debug("Pattern stack registration failed", exc_info=True)
                                # Plain-language alert for each pattern stack
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
                                # Register with Active Tracker for continuous monitoring
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
            except Exception:
                logger.debug("Pattern watcher check failed", exc_info=True)

        # Active tracker price check (every 20 steps)
        _at = getattr(ctx, "active_tracker", None)
        if _at is not None and step % 20 == 0 and _at.count > 0:
            try:
                _events = _at.check_prices()
                if _events:
                    # Write plain-language updates for status changes
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

        # Synergy detection: convergence alerts + pattern stacks on same ticker
        _conv_alerts = ctx._cached_alerts[0] or []
        _p_stacks = getattr(ctx, "_cached_pattern_stacks", [])
        if _conv_alerts and _p_stacks:
            try:
                # Build lookup of pattern stack symbols+directions
                _stack_keys = set()
                for _ps in _p_stacks:
                    _stack_keys.add((_ps.symbol, _ps.direction))

                for _alert in _conv_alerts:
                    # Extract primary symbol from the convergence alert
                    _alert_sym = None
                    for _sig in getattr(_alert, "signals", []):
                        _alert_sym = getattr(_sig, "metadata", {}).get("symbol", "")
                        if _alert_sym:
                            break
                    if not _alert_sym:
                        _alert_sym = getattr(_alert, "ticker", "")
                    _alert_dir = getattr(_alert, "direction", "")
                    if _alert_sym and (_alert_sym, _alert_dir) in _stack_keys:
                        # Find the matching stack
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
                            # Publish dual confirmation event
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

    # Register the wrapped step hook
    ctx.model.add_step_hook(_sensing_step_with_advisory)

    # Inject EventBus for signal bridge (Phase 1 of hypothesis loop)
    hook._bus = ctx.bus

    # Start FinnhubWebSocket background thread now that sensing hook is wired
    _ws = getattr(ctx, "finnhub_websocket", None)
    if _ws is not None:
        try:
            _ws.start()
            logger.info("Layer 33h - FinnhubWebSocket background thread started")
        except Exception:
            logger.debug("FinnhubWebSocket start() failed", exc_info=True)

    # Register FinnhubWebSocket stop() as a shutdown hook so the background
    # thread is cleaned up when the model terminates
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

    # Store hook reference on ctx for monitoring
    ctx._market_sensing_hook = hook

    logger.info(
        "Layer 33h - Market sensing hook wired: "
        "async fetch (cadence=25, slots=8), outcome tracking (cadence=200), "
        "tiered alerters (%d), advisory bridge active",
        len(tiered_alerters),
    )
