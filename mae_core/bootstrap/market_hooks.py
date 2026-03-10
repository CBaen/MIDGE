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

    # Subscribe to partial convergences — register as developing situations
    def _on_partial_convergence(channel, data):
        colony = getattr(ctx, "octopus_colony", None)
        if colony is None:
            return
        msg = data if isinstance(data, dict) else {}
        direction = msg.get("direction", "neutral")
        # Extract ticker from signal symbol fields (global convergence
        # partials don't carry a top-level ticker).
        ticker = msg.get("ticker")
        if not ticker:
            for sig in msg.get("signals", []):
                t = sig.get("symbol", "") or sig.get("metadata", {}).get("symbol", "")
                if t:
                    ticker = t
                    break
        if ticker is None:
            return
        key = f"{direction}:{ticker}"
        lock = getattr(colony, "_situations_lock", None)
        if lock:
            with lock:
                # Cap at 200 developing situations to prevent unbounded growth
                # if situation_check handler isn't running (e.g. zero octopuses).
                if key not in colony._developing_situations and len(colony._developing_situations) < 200:
                    colony._developing_situations[key] = {
                        "ticker": ticker, "direction": direction,
                        "domains_seen": msg.get("domains_seen", []),
                        "missing_domains": msg.get("missing_domains", []),
                        "causal_predictions": msg.get("causal_predictions", []),
                        "first_seen": __import__("time").time(),
                        "check_count": 0,
                    }

    ctx.bus.register_callback("market.intel.partial_convergence", _on_partial_convergence)

    # --- Proactive causal watch: signal → WorldModel → downstream predictions ---
    # When any signal maps to a world model trigger, trace the causal chain
    # forward and emit predictions about what should move next. This is the
    # "inevitability detection" layer — noticing dominoes before they fall.
    world_model = getattr(ctx, "world_model", None)
    if world_model is not None:
        from mae_core.market.channels import CH_SIGNAL_INGESTED

        import time as _time

        def _on_signal_causal_watch(channel, data):
            try:
                msg = data if isinstance(data, dict) else {}
                source = msg.get("source", "")
                metadata = msg.get("metadata", {})
                ticker = (
                    msg.get("symbol", "")
                    or metadata.get("symbol", "")
                )

                # --- Forward: map signal → world model trigger → downstream predictions ---
                trigger = world_model.map_signal_to_trigger(source, metadata)
                if trigger:
                    effects = world_model.find_ripple_effects(trigger, min_strength=0.4)
                    if effects:
                        ctx.bus.publish("market.intel.causal_watch", {
                            "trigger": trigger,
                            "source": source,
                            "effects": [{
                                "ticker": e.ticker,
                                "direction": e.direction,
                                "strength": round(e.strength, 3),
                                "lag_days": e.total_lag_days,
                                "path": e.path,
                            } for e in effects[:10]],
                        })

                # --- Backward: does this ticker appear as a downstream world-model node? ---
                # If so, trace back to find what genesis event would cause this.
                # This handles mid-pattern discovery: we see a domino fall and ask
                # "what caused THIS, and what else should we expect?"
                if ticker and ticker in world_model._graph:
                    root_causes = world_model.find_root_causes(ticker, min_strength=0.3)
                    _ct = getattr(ctx, "cascade_tracker", None)
                    for rc in root_causes[:3]:  # cap to top 3 root causes
                        try:
                            active_chains = _ct.get_active_chains() if _ct is not None else {}
                            # Check if any active chain already tracks this trigger
                            existing = any(
                                c.get("trigger") == rc.trigger
                                for c in active_chains.values()
                            )

                            if not existing and _ct is not None:
                                # Mid-pattern discovery: register a late-joining cascade
                                # from the genesis trigger forward
                                forward_effects = world_model.find_ripple_effects(
                                    rc.trigger, min_strength=0.3
                                )
                                if forward_effects:
                                    ripple_dicts = [{
                                        "ticker": e.ticker,
                                        "direction": e.direction,
                                        "strength": round(e.strength, 3),
                                        "lag_days": e.total_lag_days,
                                    } for e in forward_effects[:10]]
                                    _ct.register_cascade(
                                        alert_id=f"backward_{ticker}_{rc.trigger}",
                                        trigger=rc.trigger,
                                        ripple_effects=ripple_dicts,
                                        direction=rc.direction,
                                    )
                                    logger.info(
                                        "Backward discovery: %s is downstream of '%s' "
                                        "(strength=%.2f) — late-joining cascade registered",
                                        ticker, rc.trigger, rc.strength,
                                    )

                            # Populate priority requests so focused attention can
                            # investigate the genesis domain
                            _prio = getattr(ctx, "_priority_requests", None)
                            if _prio is None:
                                ctx._priority_requests = {}
                                _prio = ctx._priority_requests

                            # Cap at 50 to prevent unbounded growth
                            if len(_prio) < 50:
                                # Determine domain of genesis trigger
                                genesis_domain = "macro"  # sensible default
                                if "energy" in rc.trigger or "eia" in rc.trigger:
                                    genesis_domain = "energy"
                                elif "fed" in rc.trigger or "rate" in rc.trigger or "cpi" in rc.trigger:
                                    genesis_domain = "macro"
                                elif "defense" in rc.trigger or "geopolit" in rc.trigger:
                                    genesis_domain = "government"
                                elif "crypto" in rc.trigger:
                                    genesis_domain = "crypto"
                                elif "vix" in rc.trigger:
                                    genesis_domain = "volatility"

                                _prio[f"{ticker}_{rc.trigger}"] = {
                                    "ticker": ticker,
                                    "domains_needed": [genesis_domain],
                                    "priority": "high",
                                    "expires": _time.time() + 3600,
                                    "source": "backward_discovery",
                                    "root_cause_trigger": rc.trigger,
                                    "root_cause_strength": round(rc.strength, 3),
                                }
                        except Exception:
                            logger.debug(
                                "Backward cascade discovery failed for ticker %s trigger %s",
                                ticker, rc.trigger, exc_info=True,
                            )
            except Exception:
                pass  # Never block signal ingestion

        ctx.bus.register_callback(CH_SIGNAL_INGESTED, _on_signal_causal_watch)
        logger.info("Layer 33f - Causal watch: signal → WorldModel → downstream + backward root-cause wired")

    # --- Cascade tracking: watch dominoes fall, confirm chain links ---
    # When signals arrive, check if they confirm any predicted cascade.
    # When convergence alerts fire, register new cascades from ripple_effects.
    cascade_tracker = getattr(ctx, "cascade_tracker", None)
    if cascade_tracker is not None:
        from mae_core.market.channels import CH_SIGNAL_INGESTED, CH_CONVERGENCE

        def _on_signal_cascade_check(channel, data):
            _shm_ct = getattr(ctx, "system_health_monitor", None)
            try:
                msg = data if isinstance(data, dict) else {}
                ticker = msg.get("symbol", "") or msg.get("metadata", {}).get("symbol", "")
                direction = msg.get("direction", "")
                if ticker and direction in ("bullish", "bearish"):
                    cascade_tracker.check_signal(ticker, direction)
                    if _shm_ct:
                        _shm_ct.record_success("cascade_tracker")
            except Exception as exc:
                if _shm_ct:
                    _shm_ct.record_error("cascade_tracker", exc)

        def _on_convergence_register_cascade(channel, data):
            _shm_ct = getattr(ctx, "system_health_monitor", None)
            try:
                msg = data if isinstance(data, dict) else {}
                alert_id = msg.get("alert_id", "")
                ripples = msg.get("ripple_effects", [])
                if not alert_id or not ripples:
                    return
                # Find the trigger by mapping contributing signals
                wm = getattr(ctx, "world_model", None)
                if wm is None:
                    return
                trigger = None
                for sig in msg.get("signals", []):
                    source = sig.get("source", "") if isinstance(sig, dict) else getattr(sig, "source", "")
                    metadata = sig.get("metadata", {}) if isinstance(sig, dict) else getattr(sig, "metadata", {})
                    trigger = wm.map_signal_to_trigger(source, metadata)
                    if trigger:
                        break
                if trigger:
                    cascade_tracker.register_cascade(
                        alert_id, trigger, ripples, msg.get("direction", "neutral"),
                    )
            except Exception as exc:
                if _shm_ct:
                    _shm_ct.record_error("cascade_tracker", exc)

        ctx.bus.register_callback(CH_SIGNAL_INGESTED, _on_signal_cascade_check)
        ctx.bus.register_callback(CH_CONVERGENCE, _on_convergence_register_cascade)
        logger.info("Layer 33f - Cascade tracker: domino confirmation + WorldModel feedback wired")

    # --- Forward Chain Boost: inject synthetic signals for remaining dominoes ---
    # When a cascade link is confirmed, we boost the remaining predicted dominoes
    # by injecting them as "cascade" domain signals into the convergence alerter.
    # This creates a feedback loop: confirmed dominoes raise confidence that the
    # rest will fall — pushing them toward the convergence threshold faster.
    _alerter_ref = getattr(ctx, "convergence_alerter", None)
    if _alerter_ref is not None and getattr(ctx, "cascade_tracker", None) is not None:
        from mae_core.market.channels import CH_CASCADE_CONFIRMED

        def _on_cascade_confirmed(channel, data):
            try:
                msg = data if isinstance(data, dict) else {}
                chain_id = msg.get("chain_id", "")
                trigger = msg.get("trigger", "")
                confirmed_count = msg.get("confirmed_count", 0)
                total_links = msg.get("total_links", 1)
                remaining = msg.get("remaining", [])

                if not remaining or total_links == 0:
                    return  # Nothing left to boost

                confirmed_ratio = confirmed_count / max(total_links, 1)

                injected = 0
                for domino in remaining:
                    domino_ticker = domino.get("ticker", "")
                    domino_direction = domino.get("direction", "neutral")
                    domino_strength = domino.get("strength", 0.5)
                    domino_lag_days = domino.get("lag_days", 0)

                    if not domino_ticker or domino_direction not in ("bullish", "bearish"):
                        continue

                    try:
                        _alerter_ref.record_signal(
                            signal_id=f"cascade_{chain_id}_{domino_ticker}",
                            strength=confirmed_ratio * domino_strength,
                            domain="cascade",
                            direction=domino_direction,
                            confidence=confirmed_ratio,
                            metadata={
                                "cascade_boosted": True,
                                "chain_id": chain_id,
                                "trigger": trigger,
                                "remaining_lag_days": domino_lag_days,
                                "symbol": domino_ticker,
                            },
                        )
                        injected += 1
                    except Exception:
                        logger.debug(
                            "Cascade boost signal injection failed for %s", domino_ticker,
                            exc_info=True,
                        )

                if injected > 0:
                    logger.info(
                        "Cascade boost: injected %d synthetic signals for chain %s",
                        injected, chain_id,
                    )
            except Exception:
                logger.debug("_on_cascade_confirmed handler failed", exc_info=True)

        ctx.bus.register_callback(CH_CASCADE_CONFIRMED, _on_cascade_confirmed)
        logger.info("Layer 33f - Forward chain boost: cascade confirmed → synthetic signal injection wired")

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
        if step % 200 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    regime_clf = getattr(ctx, "regime_classifier", None)
                    regime = regime_clf.classify() if regime_clf is not None else "default"
                    sampler.regime_aware_forget(regime)
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
            social_text_analyzer=getattr(ctx, "social_text_analyzer", None),
            yahoo_rss_client=getattr(ctx, "yahoo_rss_client", None),
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
        _shm = getattr(ctx, "system_health_monitor", None)
        try:
            original_step()
            if _shm:
                _shm.record_success("sensing")
            # outcome_evaluation runs inside original_step() on a 200-step cadence.
            # Record a success proxy on the same cadence so the health tier reflects
            # that outcome evaluation is functioning whenever sensing succeeds.
            if _shm and step % 200 == 0:
                _shm.record_success("outcome_evaluation")
        except Exception as exc:
            logger.debug("Sensing hook step failed", exc_info=True)
            if _shm:
                _shm.record_error("sensing", exc)
                if step % 200 == 0:
                    _shm.record_error("outcome_evaluation", exc)

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
                            # Risk architecture gates
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
                                else:
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
        _shm_sensing = getattr(ctx, "system_health_monitor", None)
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
                        if _shm_sensing:
                            _shm_sensing.record_success("pattern_watcher")
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
            except Exception as exc:
                logger.debug("Pattern watcher check failed", exc_info=True)
                if _shm_sensing:
                    _shm_sensing.record_error("pattern_watcher", exc)

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

    # --- Wire market task handlers into OctopusColony ---
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

        # Start monitoring in a separate try so injection failure doesn't
        # silently prevent the colony from running at all.
        try:
            colony.start_monitoring()
            logger.info("OctopusColony: monitoring started")
        except Exception:
            logger.debug("OctopusColony monitoring start failed", exc_info=True)

        # Subscribe to spawn events so newly auto-scaled arms get handlers.
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

        # Subscribe to investigation results so they are visible in the log
        # and can trigger focused-attention escalation.
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
                    source,
                    ticker,
                    check_count,
                    len(historical),
                    " [FOCUSED-ATTENTION ENGAGED]" if priority_created else "",
                )

            bus_obj = getattr(ctx, "bus", None)
            if bus_obj is not None:
                bus_obj.register_callback(
                    CH_OCTOPUS_INVESTIGATION, _on_octopus_investigation
                )
                logger.info("OctopusColony: investigation subscriber wired")
        except Exception:
            logger.debug(
                "OctopusColony investigation subscriber failed", exc_info=True
            )

    # Start InhabitantScheduler daemon thread
    _sched = getattr(ctx, "inhabitant_scheduler", None)
    if _sched is not None:
        try:
            _sched.start()
            logger.info("InhabitantScheduler: daemon thread started")
        except Exception:
            logger.debug("InhabitantScheduler start() failed", exc_info=True)

    # Store hook reference on ctx for monitoring
    ctx._market_sensing_hook = hook

    logger.info(
        "Layer 33h - Market sensing hook wired: "
        "async fetch (cadence=25, slots=8), outcome tracking (cadence=200), "
        "tiered alerters (%d), advisory bridge active",
        len(tiered_alerters),
    )
