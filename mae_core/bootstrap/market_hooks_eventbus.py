"""EventBus subscriber registration for MIDGE market hooks.

Extracted from market_hooks.py — purely structural split.
Contains: _register_market_eventbus (with all nested closures).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_market_eventbus(ctx: SimpleNamespace) -> None:
    """Wire EventBus channels — endocrine coupling for convergence alerts."""
    import json
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

                    # Stage-gating: only boost watchable links (prior stage confirmed)
                    if not domino.get("watchable", True):
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

    # --- Hypothesis fired: boost focused attention for the predicted effect ---
    from mae_core.market.channels import CH_HYPOTHESIS_FIRED

    def _on_hypothesis_fired(channel, data):
        try:
            msg = data if isinstance(data, dict) else {}
            trigger_source = msg.get("trigger_source", "")
            trigger_symbol = msg.get("trigger_symbol", "")
            expected_direction = msg.get("expected_direction", "")
            if trigger_source and hasattr(ctx, "sensing_hook"):
                hook = ctx.sensing_hook
                if hasattr(hook, "_priority_requests"):
                    from mae_core.market.sensing_constants import _DOMAIN_TO_SOURCES
                    from datetime import datetime, timedelta
                    for domain, sources in _DOMAIN_TO_SOURCES.items():
                        if trigger_source in sources:
                            for src in sources:
                                hook._priority_requests[src] = {
                                    "reason": f"hypothesis_fired:{trigger_symbol}",
                                    "boost": 2.0,
                                    "expires": datetime.now() + timedelta(hours=1),
                                }
                            break
            if trigger_symbol:
                if not hasattr(ctx, "_recent_hypothesis_fires"):
                    ctx._recent_hypothesis_fires = {}
                from datetime import datetime
                ctx._recent_hypothesis_fires[f"{trigger_symbol}:{expected_direction}"] = datetime.now()
        except Exception:
            logger.debug("Hypothesis fire handler failed", exc_info=True)

    ctx.bus.register_callback(CH_HYPOTHESIS_FIRED, _on_hypothesis_fired)
    logger.info("Layer 33f - Hypothesis fired: focused attention boost + fire tracker wired")

    # --- Arc 2/5: Risk channel subscribers ---
    from mae_core.market.channels import (
        CH_DRAWDOWN_WARNING, CH_TRADING_HALTED, CH_TRADING_RESUMED,
    )

    def _on_drawdown_warning(channel, data):
        msg = data if isinstance(data, dict) else {}
        ctx._drawdown_warning = msg
        logger.info("Risk: drawdown warning received — %s", msg.get("message", ""))

    def _on_trading_halted(channel, data):
        ctx._risk_halt = True
        logger.warning("Risk: TRADING HALTED via EventBus")

    def _on_trading_resumed(channel, data):
        ctx._risk_halt = False
        logger.info("Risk: trading resumed via EventBus")

    ctx.bus.register_callback(CH_DRAWDOWN_WARNING, _on_drawdown_warning)
    ctx.bus.register_callback(CH_TRADING_HALTED, _on_trading_halted)
    ctx.bus.register_callback(CH_TRADING_RESUMED, _on_trading_resumed)

    # --- Arc 5: Wire orphaned publisher channels ---
    from mae_core.market.channels import (
        CH_CONTRADICTION_DETECTED, CH_ABSENCE_DETECTED,
        CH_SOMATIC_ANTICIPATION,
    )

    def _on_contradiction_detected(channel, data):
        msg = data if isinstance(data, dict) else {}
        logger.info(
            "Contradiction: %s vs %s (coherence=%.2f)",
            msg.get("dominant_direction", "?"),
            msg.get("minority_direction", "?"),
            msg.get("coherence", 0.0),
        )
        ctx._last_contradiction = msg

    def _on_absence_detected(channel, data):
        msg = data if isinstance(data, dict) else {}
        logger.info(
            "Absence: expected %s signal missing for %s",
            msg.get("expected_source", "?"),
            msg.get("ticker", "?"),
        )
        colony = getattr(ctx, "octopus_colony", None)
        if colony is not None:
            try:
                colony.submit_task({
                    "ticker": msg.get("ticker", ""),
                    "direction": "neutral",
                    "domains_seen": [],
                    "missing_domains": [msg.get("expected_domain", "")],
                    "source": "absence_detection",
                }, "investigate_partial")
            except Exception:
                pass

    def _on_somatic_anticipation(channel, data):
        msg = data if isinstance(data, dict) else {}
        ticker = msg.get("ticker", "")
        if ticker and hasattr(ctx, "sensing_hook"):
            hook = ctx.sensing_hook
            if hasattr(hook, "_priority_requests"):
                import time as _time
                hook._priority_requests[f"somatic_{ticker}"] = {
                    "ticker": ticker,
                    "domains_needed": msg.get("anticipated_domains", []),
                    "priority": "high",
                    "expires": _time.time() + 3600,
                    "source": "somatic_anticipation",
                }

    ctx.bus.register_callback(CH_CONTRADICTION_DETECTED, _on_contradiction_detected)
    ctx.bus.register_callback(CH_ABSENCE_DETECTED, _on_absence_detected)
    ctx.bus.register_callback(CH_SOMATIC_ANTICIPATION, _on_somatic_anticipation)

    # Wire drift → regime recalculation
    def _on_drift_detected(channel, data):
        msg = data if isinstance(data, dict) else {}
        logger.info("Drift: %s stream shifted (%.4f → %.4f)",
                    msg.get("stream", "?"), msg.get("old_mean", 0), msg.get("new_mean", 0))
        rc = getattr(ctx, "regime_classifier", None)
        if rc is not None:
            try:
                rc.classify()
            except Exception:
                pass

    ctx.bus.register_callback("market.intel.drift_detected", _on_drift_detected)

    # Wire granger findings → hypothesis generator
    def _on_granger_finding(channel, data):
        msg = data if isinstance(data, dict) else {}
        count = msg.get("count", 0)
        if count > 0:
            logger.info("Granger: %d causal findings — hypothesis generation triggered", count)
            gen = getattr(ctx, "hypothesis_generator", None)
            if gen is not None:
                try:
                    gen.generate()
                except Exception:
                    pass

    ctx.bus.register_callback("market.intel.granger_finding", _on_granger_finding)

    logger.info("Layer 33f - Market EventBus: convergence + hypothesis -> endocrine coupling wired")
