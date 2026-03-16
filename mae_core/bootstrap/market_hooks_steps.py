"""Step hook helpers for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _write_convergence_heartbeat, _run_drift_detector,
          _run_slow_cadence_ops (every-500/1000/5000-step analysis),
          _ingest_granger_bridge (continuous-Granger → WorldModel bridge reader).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
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

        # SituationBoard snapshot
        sb = getattr(ctx, "situation_board", None)
        if sb is not None:
            try:
                heartbeat["situation_board"] = sb.get_snapshot()
                sb.save(Path("data/midge/situation_board.json"))
            except Exception:
                logger.debug("SituationBoard snapshot/save failed", exc_info=True)

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


def _run_raw_analyst(ctx: SimpleNamespace, step: int) -> None:
    """Run RawDataAnalyst every 100 steps and inject enriched signals.

    RawDataAnalyst reads across all SQLite raw stores, computes cross-domain
    insights (insider price context, FRED macro regime, pre-convergence detection,
    funding rate squeeze), and returns MarketSignal objects that are fed directly
    into the convergence engine via record_signal().
    """
    analyst = getattr(ctx, "raw_data_analyst", None)
    if analyst is None:
        return
    if step % 50 != 0:
        return
    try:
        enriched_signals = analyst.analyze(step)
        if not enriched_signals:
            return
        alerter = getattr(ctx, "convergence_alerter", None)
        if alerter is None:
            return
        for sig in enriched_signals:
            try:
                alerter.record_signal(
                    signal_id=sig.signal_id,
                    strength=sig.strength,
                    domain=sig.domain,
                    direction=sig.direction,
                    confidence=sig.confidence,
                    timestamp=sig.timestamp,
                    metadata={**sig.metadata, "symbol": sig.symbol,
                               "asset_class": sig.asset_class},
                    source=sig.source,
                )
            except Exception:
                logger.debug(
                    "RawDataAnalyst: failed to record signal %s", sig.signal_id,
                    exc_info=True,
                )
    except Exception:
        logger.debug("_run_raw_analyst failed", exc_info=True)


def _run_circular_health_check(ctx: SimpleNamespace, step: int) -> None:
    """Verify all 5 arcs of the circular architecture carried data.

    Observability, not enforcement — tells us which arcs are active vs dormant.
    """
    arcs = {}

    # Arc 1: Outcomes → Advisors (CH_PREDICTION_RESULT published?)
    _bus = getattr(ctx, "bus", None)
    if _bus is not None:
        # Check if outcome collector has bus wired
        _oc = getattr(ctx, "outcome_collector", None)
        arcs["arc1_outcomes_to_advisors"] = _oc is not None and getattr(_oc, "_bus", None) is not None
    else:
        arcs["arc1_outcomes_to_advisors"] = False

    # Arc 2: Advisors → Decisions (bio caution or HAVEN flags set?)
    _caution = getattr(ctx, "_market_caution", None)
    _haven = getattr(ctx, "_haven_market_flags", None)
    arcs["arc2_advisors_to_decisions"] = _caution is not None or _haven is not None

    # Arc 3: Memory → Observer (pattern_memory available?)
    _pmem = getattr(ctx, "pattern_memory", None)
    arcs["arc3_memory_to_observer"] = _pmem is not None and getattr(_pmem, "is_available", False)

    # Arc 4: Agents ↔ Market (track records populated?)
    _tracks = getattr(ctx, "_agent_track_records", None)
    arcs["arc4_agents_market"] = _tracks is not None and len(_tracks) > 0

    # Arc 5: Risk → Decisions (risk channels wired?)
    arcs["arc5_risk_to_decisions"] = hasattr(ctx, "_risk_halt") or hasattr(ctx, "_drawdown_warning")

    active = sum(1 for v in arcs.values() if v)
    total = len(arcs)

    logger.info(
        "Circular health: %d/%d arcs active — %s",
        active, total,
        ", ".join(f"{k}={'OK' if v else 'DORMANT'}" for k, v in arcs.items()),
    )

    # Store for external monitoring
    ctx._circular_health = arcs


_GRANGER_BRIDGE_PATH = Path(__file__).resolve().parents[2] / "data" / "market" / "granger_bridge.json"
_REPLAY_BRIDGE_PATH  = Path(__file__).resolve().parents[2] / "data" / "market" / "replay_bridge.json"
_BRIDGE_DIR          = Path(__file__).resolve().parents[2] / "data" / "market"


def _ingest_jsonl_bridge(ctx: SimpleNamespace, bridge_name: str) -> None:
    """Read a JSONL bridge file and inject signals into the convergence engine.

    Generic bridge reader for any ecosystem process that writes MarketSignal-format
    JSONL. Reads new lines since last offset, injects via convergence_alerter.record_signal().

    Used for: raw_miner_signals.jsonl, cross_market_signals.jsonl
    """
    bridge_path = _BRIDGE_DIR / f"{bridge_name}.jsonl"
    if not bridge_path.exists():
        return

    offset_attr = f"_bridge_offset_{bridge_name}"
    last_offset = getattr(ctx, offset_attr, 0)

    alerter = getattr(ctx, "convergence_alerter", None)
    if alerter is None:
        return

    try:
        file_size = bridge_path.stat().st_size
        if file_size <= last_offset:
            return  # No new data

        injected = 0
        with open(bridge_path, "r", encoding="utf-8") as f:
            f.seek(last_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sig = json.loads(line)
                    alerter.record_signal(
                        signal_id=sig.get("signal_id", f"{bridge_name}_{injected}"),
                        strength=sig.get("strength", 0.5),
                        domain=sig.get("domain", "unknown"),
                        direction=sig.get("direction", ""),
                        confidence=sig.get("confidence", 0.5),
                        timestamp=sig.get("timestamp", ""),
                        metadata={
                            "symbol": sig.get("symbol", ""),
                            "asset_class": sig.get("asset_class", ""),
                            "signal_source": sig.get("source", bridge_name),
                            **(sig.get("metadata", {})),
                        },
                        source=sig.get("source", bridge_name),
                    )
                    injected += 1
                except Exception:
                    pass  # Skip malformed lines
            new_offset = f.tell()

        setattr(ctx, offset_attr, new_offset)
        if injected > 0:
            logger.info(
                "Bridge %s: injected %d signals into convergence engine",
                bridge_name, injected,
            )
    except Exception:
        logger.debug("_ingest_jsonl_bridge(%s) failed", bridge_name, exc_info=True)


def _ingest_granger_bridge(ctx: SimpleNamespace) -> None:
    """Read granger_bridge.json (written by feed_granger_to_worldmodel.py) and inject
    any new/updated domain-level causal edges into the live WorldModel.

    Designed for cross-process safety:
      - Parallel Granger process writes granger_continuous.json
      - feed_granger_to_worldmodel.py converts it to granger_bridge.json (atomic write)
      - This function reads granger_bridge.json; the daemon never touches the source file

    Deduplication: ctx._granger_bridge_ts tracks the last-seen 'last_written' timestamp.
    If the file hasn't changed since the last call, we skip all I/O.

    Edge injection mirrors the existing in-process Granger path (lines ~392-408 above):
      world_model.add_discovered_edge(cause, effect, strength, lag_days, evidence)
    """
    _wm = getattr(ctx, "world_model", None)
    if _wm is None:
        return

    if not _GRANGER_BRIDGE_PATH.exists():
        return

    try:
        raw = json.loads(_GRANGER_BRIDGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("_ingest_granger_bridge: could not read bridge file", exc_info=True)
        return

    last_written = raw.get("last_written", "")
    # Skip if nothing new since last read
    if last_written and last_written == getattr(ctx, "_granger_bridge_ts", ""):
        return

    edges = raw.get("edges", [])
    if not edges:
        ctx._granger_bridge_ts = last_written
        return

    injected = 0
    for edge in edges:
        cause = edge.get("cause", "")
        effect = edge.get("effect", "")
        strength = float(edge.get("strength", 0.5))
        lag_days = float(edge.get("lag_days", 3.0))
        if not cause or not effect:
            continue
        try:
            _wm.add_discovered_edge(
                cause=cause,
                effect=effect,
                strength=strength,
                lag_days=lag_days,
                evidence="granger_continuous",
            )
            injected += 1
        except Exception:
            logger.debug(
                "_ingest_granger_bridge: add_discovered_edge failed for %s→%s",
                cause, effect, exc_info=True,
            )

    ctx._granger_bridge_ts = last_written

    if injected:
        logger.info(
            "WorldModel: injected %d domain-level edges from continuous Granger "
            "(source: %s)",
            injected, raw.get("source_last_updated", "?"),
        )
        # Persist WorldModel so discovered edges survive daemon restart
        try:
            _wm.persist()
        except Exception:
            logger.debug("WorldModel persist after bridge inject failed", exc_info=True)


def _ingest_replay_bridge(ctx: SimpleNamespace) -> None:
    """Read replay_bridge.json (written by feed_replay_to_thompson.py) and push
    aggregated replay outcomes into the live Thompson combo distributions.

    Designed for cross-process safety:
      - continuous_replay.py runs in a separate OS process, grades historical
        convergence alerts against known price outcomes, and appends to
        continuous_replay_results.jsonl.
      - feed_replay_to_thompson.py aggregates those results and writes
        replay_bridge.json atomically (write-to-tmp, then replace).
      - This function reads replay_bridge.json; the daemon never touches the
        source JSONL directly.

    Deduplication: ctx._replay_bridge_ts tracks the last-seen 'last_updated'
    timestamp.  If the file hasn't changed since the last call we skip all I/O.

    Thompson update:
      For each (combo, regime) entry with wins + losses >= MIN_OUTCOMES_GATE,
      calls thompson_sampler.update("combo:<combo>", success=True/False, regime)
      once per win and once per loss.  This is additive across bridge reads —
      calling update() N times is equivalent to N Bayesian observations.

    The regime in the bridge is always "default" (replay has no live regime
    classifier).  If a future version adds regime information, it flows through
    automatically.
    """
    ts = getattr(ctx, "thompson_sampler", None)
    if ts is None:
        return

    if not _REPLAY_BRIDGE_PATH.exists():
        return

    try:
        raw = json.loads(_REPLAY_BRIDGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("_ingest_replay_bridge: could not read bridge file", exc_info=True)
        return

    last_updated = raw.get("last_updated", "")
    if last_updated and last_updated == getattr(ctx, "_replay_bridge_ts", ""):
        return  # Nothing new since last read

    outcomes = raw.get("replay_outcomes", [])
    if not outcomes:
        ctx._replay_bridge_ts = last_updated
        return

    min_gate = int(raw.get("min_outcomes_gate", 5))
    combos_updated = 0

    for entry in outcomes:
        combo  = entry.get("combo", "")
        regime = entry.get("regime", "default") or "default"
        wins   = int(entry.get("wins", 0))
        losses = int(entry.get("losses", 0))

        if not combo:
            continue
        if wins + losses < min_gate:
            continue  # Too few samples — skip to avoid noise

        signal_id = f"combo:{combo}"

        # Apply each win/loss as a Bayesian observation.
        # update() is thread-safe (ThompsonSampler uses an RLock internally).
        for _ in range(wins):
            try:
                ts.update(signal_id, success=True, regime=regime)
            except Exception:
                logger.debug(
                    "_ingest_replay_bridge: update failed for %s (win)", signal_id,
                    exc_info=True,
                )

        for _ in range(losses):
            try:
                ts.update(signal_id, success=False, regime=regime)
            except Exception:
                logger.debug(
                    "_ingest_replay_bridge: update failed for %s (loss)", signal_id,
                    exc_info=True,
                )

        combos_updated += 1

    ctx._replay_bridge_ts = last_updated

    if combos_updated:
        logger.info(
            "Thompson: injected replay outcomes for %d combo distributions "
            "(source: %d lines, %d total outcomes)",
            combos_updated,
            raw.get("source_line_count", 0),
            raw.get("outcome_count", 0),
        )
        # Persist so updated combo distributions survive a daemon restart
        try:
            ts.save_distributions()
        except Exception:
            logger.debug("Thompson persist after replay bridge inject failed", exc_info=True)


def _run_slow_cadence_ops(ctx: SimpleNamespace, step: int, _shm, _timer) -> None:
    """Run every-500-step analysis: lag-correlation, Granger causality, post-mortem.

    Also handles every-100-step raw data analysis, every-1000-step Thompson
    calibration, and every-5000-step backtest scheduler + excavation daemon.
    """
    # --- Every 100 steps: raw data cross-domain analysis ---
    _run_raw_analyst(ctx, step)

    if step % 200 == 0:
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
                    # Auto-discover WorldModel edges from strong lag correlations
                    _wm = getattr(ctx, "world_model", None)
                    if _wm is not None:
                        _lag_added = 0
                        for f in findings:
                            if abs(f.correlation) >= 0.6:
                                try:
                                    _wm.add_discovered_edge(
                                        cause=f.source_a,
                                        effect=f.source_b,
                                        strength=abs(f.correlation),
                                        lag_days=float(f.lag_days),
                                        evidence="lag_correlation",
                                    )
                                    _lag_added += 1
                                except Exception:
                                    pass
                        if _lag_added:
                            logger.info(
                                "WorldModel: auto-discovered %d edges from lag correlations",
                                _lag_added,
                            )
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
                # Auto-discover WorldModel edges from Granger findings
                _wm = getattr(ctx, "world_model", None)
                if _wm is not None and g_findings:
                    for f in g_findings:
                        try:
                            _wm.add_discovered_edge(
                                cause=f.cause_source,
                                effect=f.effect_source,
                                strength=min(1.0, max(0.1, 1.0 - f.p_value)),
                                lag_days=float(f.best_lag),
                                evidence="granger",
                            )
                        except Exception:
                            pass
                    logger.info(
                        "WorldModel: auto-discovered %d edges from Granger",
                        len(g_findings),
                    )
                # FIX 7b: KnowledgeGraph — persist Granger findings to Neo4j
                _kg = getattr(ctx, "knowledge_graph", None)
                if _kg is not None:
                    try:
                        for _gf in g_findings:
                            _kg.store_granger_finding(
                                cause_domain=_gf.cause_source,
                                effect_domain=_gf.effect_source,
                                lag=int(_gf.best_lag),
                                f_stat=float(getattr(_gf, "f_statistic", 1.0)),
                                p_value=float(_gf.p_value),
                            )
                    except Exception:
                        logger.debug("KnowledgeGraph Granger write failed", exc_info=True)
            except Exception:
                logger.debug("Granger causality step failed", exc_info=True)

        # FIX 5: CorrelationTracker cross-domain anomalies → convergence alerter
        _ct_tracker = getattr(ctx, "correlation_tracker", None)
        _alerter_ct = getattr(ctx, "convergence_alerter", None)
        if _ct_tracker is not None and _alerter_ct is not None:
            try:
                _ct_anomalies = _ct_tracker.detect_cross_domain_anomalies()
                for _pair, _domain_a, _domain_b in _ct_anomalies[:10]:
                    _ticker = getattr(_pair, "signal_a", "").split(":")[-1] if ":" in getattr(_pair, "signal_a", "") else ""
                    _direction = "neutral"
                    _corr = getattr(_pair, "correlation", 0.0)
                    _strength = min(1.0, abs(_corr))
                    _alerter_ct.record_signal(
                        signal_id=f"corr_anomaly_{_pair.signal_a}_{_pair.signal_b}",
                        strength=_strength,
                        domain=_domain_a,
                        direction=_direction,
                        confidence=_strength * 0.7,
                        metadata={"signal_a": _pair.signal_a, "signal_b": _pair.signal_b,
                                  "domain_a": _domain_a, "domain_b": _domain_b,
                                  "correlation": round(_corr, 4),
                                  "symbol": _ticker},
                        source="correlation_tracker",
                    )
                if _ct_anomalies:
                    logger.info("CorrelationTracker: %d cross-domain anomalies → convergence", len(_ct_anomalies))
            except Exception:
                logger.debug("CorrelationTracker anomaly injection failed", exc_info=True)

        # --- Continuous Granger bridge: inject domain-level edges from parallel process ---
        # The parallel continuous_granger.py discovers domain-level causal relationships
        # (e.g. institutional → insider, macro → institutional) from the full 874K signal
        # archive.  feed_granger_to_worldmodel.py converts its output to granger_bridge.json.
        # We read and inject those edges here — the daemon never touches granger_continuous.json
        # directly, keeping the two processes cleanly decoupled.
        try:
            _ingest_granger_bridge(ctx)
        except Exception:
            logger.debug("_ingest_granger_bridge failed", exc_info=True)

        # --- Continuous replay bridge: push graded replay outcomes into Thompson combos ---
        # continuous_replay.py (separate OS process) grades historical convergence alerts
        # against known price outcomes and writes continuous_replay_results.jsonl.
        # feed_replay_to_thompson.py aggregates those into replay_bridge.json.
        # We read and inject those combo distributions here — same decoupled pattern
        # as the Granger bridge above.
        try:
            _ingest_replay_bridge(ctx)
        except Exception:
            logger.debug("_ingest_replay_bridge failed", exc_info=True)

        # --- Raw data miner bridge: ingest signals from DuckDB-powered extraction ---
        try:
            _ingest_jsonl_bridge(ctx, "raw_miner_signals")
        except Exception:
            logger.debug("_ingest_jsonl_bridge(raw_miner) failed", exc_info=True)

        # --- Cross-market hunter bridge: ingest cross-market anomaly signals ---
        try:
            _ingest_jsonl_bridge(ctx, "cross_market_signals")
        except Exception:
            logger.debug("_ingest_jsonl_bridge(cross_market) failed", exc_info=True)

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
                    # FailureExplainer — explain recent failed predictions.
                    # Bug fix: was passing (outcome, outcome) as the pair — outcome dicts
                    # don't have a "confidence" field, so every failure was misclassified.
                    # Now joins predictions.jsonl by prediction_id so the explainer receives
                    # (prediction_dict, outcome_dict) as intended.
                    _fe = getattr(ctx, "failure_explainer", None)
                    if _fe is not None:
                        try:
                            import json as _jfe
                            from pathlib import Path as _Pfe
                            _pred_path = _Pfe("data/market/predictions.jsonl")
                            _out_path = _Pfe("data/market/outcomes.jsonl")
                            # Build prediction lookup: prediction_id -> prediction_dict
                            _pred_lookup: dict = {}
                            if _pred_path.exists():
                                with open(_pred_path, "r") as _f:
                                    for _ln in _f:
                                        try:
                                            _p = _jfe.loads(_ln)
                                            _pid = _p.get("prediction_id") or _p.get("id", "")
                                            if _pid:
                                                _pred_lookup[_pid] = _p
                                        except Exception:
                                            pass
                            _failed_pairs: list = []
                            if _out_path.exists():
                                with open(_out_path, "r") as _f:
                                    for _ln in _f:
                                        try:
                                            _o = _jfe.loads(_ln)
                                            if not _o.get("success", True):
                                                _pid = _o.get("prediction_id", "")
                                                _pred = _pred_lookup.get(_pid)
                                                # Pass (prediction_dict, outcome_dict).
                                                # If prediction not found, pass None so
                                                # explainer can handle gracefully.
                                                _failed_pairs.append((_pred, _o))
                                        except Exception:
                                            pass
                            if _failed_pairs:
                                _fe.batch_explain(_failed_pairs[-50:])  # cap at 50 per cycle
                                logger.debug("FailureExplainer: explained %d failures", len(_failed_pairs[-50:]))
                        except Exception:
                            logger.debug("FailureExplainer batch_explain failed", exc_info=True)
                if _shm:
                    _shm.record_success("post_mortem")
            except Exception as exc:
                logger.debug("Post-mortem review step failed", exc_info=True)
                if _shm:
                    _shm.record_error("post_mortem", exc)

        # DeepAnalyst: synthesize ranked inevitabilities from all data sources
        _da = getattr(ctx, "deep_analyst", None)
        if _da is not None:
            try:
                if _timer is not None:
                    with _timer.track("deep_analyst"):
                        _inevitabilities = _da.analyze(lookback_days=30, top_n=20)
                else:
                    _inevitabilities = _da.analyze(lookback_days=30, top_n=20)
                if _inevitabilities:
                    # Store on ctx for Law 7 validator access
                    ctx.inevitabilities = _inevitabilities
                    # Publish top 5 inevitabilities to SituationBoard
                    _sb = getattr(ctx, "situation_board", None)
                    if _sb is not None:
                        try:
                            from mae_core.market.intelligence.situation_board import AnalystFinding
                            for _iv in _inevitabilities[:5]:
                                _sb.publish(AnalystFinding(
                                    analyst_id="deep_analyst",
                                    ticker=_iv.ticker,
                                    direction=_iv.direction,
                                    confidence=_iv.score,
                                    summary=(_iv.evidence_summary[:200]
                                             if _iv.evidence_summary else ""),
                                    evidence=[],
                                    domains=_iv.domains,
                                    timestamp=datetime.now().isoformat(),
                                    step=step,
                                    decay_hours=4.0,
                                ))
                        except Exception:
                            logger.debug("SituationBoard publish (deep_analyst) failed",
                                         exc_info=True)
                    logger.info(
                        "DeepAnalyst: top %d inevitabilities (best: %s %s score=%.3f)",
                        len(_inevitabilities),
                        _inevitabilities[0].ticker,
                        _inevitabilities[0].direction,
                        _inevitabilities[0].score,
                    )
                    # Persist to JSONL
                    try:
                        import json as _json
                        _inv_path = os.path.join("data", "midge", "inevitabilities.jsonl")
                        os.makedirs(os.path.dirname(_inv_path), exist_ok=True)
                        with open(_inv_path, "a", encoding="utf-8") as _f:
                            for _iv in _inevitabilities[:10]:
                                _f.write(_json.dumps({
                                    "ticker": _iv.ticker,
                                    "direction": _iv.direction,
                                    "score": _iv.score,
                                    "domains": _iv.domains,
                                    "evidence_summary": _iv.evidence_summary,
                                    "expected_window_days": _iv.expected_window_days,
                                    "template_match": _iv.template_match,
                                    "template_win_rate": _iv.template_win_rate,
                                    "world_model_chain": str(_iv.world_model_chain) if _iv.world_model_chain else None,
                                    "signal_count": _iv.signal_count,
                                    "timestamp": datetime.now().isoformat(),
                                }) + "\n")
                    except Exception:
                        logger.debug("Failed to persist inevitabilities", exc_info=True)
                    # Embed top 5 in Qdrant (long-term semantic memory)
                    _pmem = getattr(ctx, "pattern_memory", None)
                    if _pmem is not None:
                        for _iv in _inevitabilities[:5]:
                            try:
                                _pmem.remember_inevitability(_iv)
                            except Exception:
                                logger.debug("Failed to embed inevitability", exc_info=True)
                    # Format top 5 for humans
                    try:
                        from mae_core.market.plain_language import format_inevitability, write_plain_alert
                        for _iv in _inevitabilities[:5]:
                            try:
                                _alert_data = format_inevitability(_iv)
                                write_plain_alert(_alert_data)
                            except Exception:
                                logger.debug("Failed to format inevitability", exc_info=True)
                    except Exception:
                        logger.debug("Plain language formatting unavailable", exc_info=True)
                    # Register top 10 for outcome grading
                    _oc = getattr(ctx, "outcome_collector", None)
                    if _oc is not None:
                        _registered = 0
                        for _iv in _inevitabilities[:10]:
                            try:
                                if _oc.register_inevitability(_iv):
                                    _registered += 1
                            except Exception:
                                pass
                        if _registered:
                            logger.info("DeepAnalyst: registered %d inevitabilities for outcome tracking", _registered)
                    # Publish full payload to EventBus
                    if hasattr(ctx, "bus"):
                        ctx.bus.publish("market.intel.deep_analysis", {
                            "count": len(_inevitabilities),
                            "top": [
                                {"ticker": iv.ticker, "direction": iv.direction,
                                 "score": iv.score, "domains": iv.domains,
                                 "evidence_summary": iv.evidence_summary,
                                 "expected_window_days": iv.expected_window_days}
                                for iv in _inevitabilities[:5]
                            ],
                        })
            except Exception:
                logger.debug("DeepAnalyst step failed", exc_info=True)

        # Expire stale cascade chains so WorldModel learns from failures
        _ct = getattr(ctx, "cascade_tracker", None)
        if _ct is not None:
            try:
                expired = _ct.expire_stale()
                if expired:
                    logger.info("CascadeTracker: expired %d stale chains", len(expired))
            except Exception as exc:
                logger.debug("CascadeTracker expire_stale failed", exc_info=True)

            # Persist cascade snapshot for daily narrative (file-based reader)
            try:
                _chains = _ct.get_active_chains()
                _stats = _ct.get_statistics()
                _snap: list[dict] = []
                for _chain_id, _chain in _chains.items():
                    _confirmed = [lk for lk in _chain.get("links", []) if lk.get("status") == "confirmed"]
                    _pending = [lk for lk in _chain.get("links", []) if lk.get("status") == "pending"]
                    if not _chain.get("links"):
                        continue
                    _snap.append({
                        "trigger": _chain.get("trigger", "?"),
                        "direction": _chain.get("direction", "?"),
                        "total_links": len(_chain.get("links", [])),
                        "confirmed_count": len(_confirmed),
                        "pending_count": len(_pending),
                        "confirmed_tickers": [lk.get("ticker") for lk in _confirmed],
                        "next_dominoes": [lk.get("ticker") for lk in _pending[:3]],
                        "registered_at": _chain.get("registered_at", ""),
                        "mean_energy_ratio": (
                            round(sum(lk["energy_ratio"] for lk in _confirmed if "energy_ratio" in lk)
                                  / max(len([lk for lk in _confirmed if "energy_ratio" in lk]), 1), 3)
                            if any("energy_ratio" in lk for lk in _confirmed) else None
                        ),
                    })
                # Sort by confirmed_count descending — most-confirmed chains first
                _snap.sort(key=lambda x: x["confirmed_count"], reverse=True)
                _cascade_path = os.path.join("data", "market", "cascade_snapshot.json")
                os.makedirs(os.path.dirname(_cascade_path), exist_ok=True)
                import json as _json_cas
                with open(_cascade_path, "w", encoding="utf-8") as _cf:
                    _json_cas.dump({"updated_at": datetime.now().isoformat(),
                                    "stats": _stats, "chains": _snap[:10]}, _cf)
            except Exception:
                logger.debug("CascadeTracker snapshot failed", exc_info=True)

        # Arc 5: Circular flow health check (growth sprint: every 200 steps)
        _run_circular_health_check(ctx, step)

    if step % 500 == 0:
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

    if step % 2000 == 0:
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

    # ── Daily narrative letter + stats dashboard ──────────────────
    # Check once every 100 steps whether we've crossed into a new calendar day.
    # If so, generate the morning letter and stats dashboard.
    if step % 100 == 0:
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            last_narrative_day = getattr(ctx, "_last_narrative_day", "")
            if today_str != last_narrative_day:
                ctx._last_narrative_day = today_str
                logger.info("Daily narrative: generating letter for %s", today_str)
                try:
                    from mae_core.market.intelligence.daily_narrative import (
                        generate_daily_narrative,
                    )
                    narrative_text = generate_daily_narrative(today_str)
                    # Email if notifier is configured
                    _notifier = getattr(ctx, "email_notifier", None)
                    if _notifier is not None:
                        try:
                            sent = _notifier.send_daily_narrative(narrative_text)
                            if sent:
                                logger.info("Daily narrative emailed to Guiding Light")
                        except Exception:
                            logger.debug("Daily narrative email failed", exc_info=True)
                    else:
                        logger.info("Daily narrative ready (no email notifier configured)")
                except Exception:
                    logger.debug("Daily narrative generation failed", exc_info=True)

                # Daily stats dashboard (Mermaid charts)
                try:
                    from mae_core.market.intelligence.daily_stats import (
                        generate_daily_stats,
                    )
                    stats_path = generate_daily_stats(today_str)
                    if stats_path:
                        logger.info("Daily stats dashboard written: %s", stats_path)
                except Exception:
                    logger.debug("Daily stats generation failed", exc_info=True)
        except Exception:
            logger.debug("Daily narrative date check failed", exc_info=True)
