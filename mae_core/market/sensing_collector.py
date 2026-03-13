"""Sensing collector mixin — result processing and signal pipeline.

Provides _collect_results and _collect_one.
Mixed into MarketSensingHook via SensingCollectorMixin.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger("midge.market.sensing")


class SensingCollectorMixin:
    """Mixin providing signal collection and pipeline processing.

    Requires the host class to have:
      self._pending_futures, self._convergence_alerter, self._tiered_alerters,
      self._deception_detector, self._somatic_anticipation,
      self._pattern_completion_engine, self._absence_monitor,
      self._correlation_tracker, self._bus, self._memory,
      self._outcome_collector, self._recent_domains,
      self._total_signals_fed, self._last_fetch_source,
      self._trigger_reactive_convergence
    Plus module-level constants: TIER_ROUTING
    """

    def _collect_results(self):
        """Check for completed futures and process their signals."""
        done = [k for k, f in self._pending_futures.items() if f.done()]
        for source_name in done:
            self._collect_one(source_name)

    def _collect_one(self, source_name: str):
        """Process signals from a single completed fetch."""
        from mae_core.market.sensing_constants import TIER_ROUTING
        from mae_core.market.sensing_lifecycle import store_signals

        future = self._pending_futures.pop(source_name, None)
        if future is None:
            return

        _cb = getattr(self, "_circuit_breaker", None)
        try:
            signals = future.result()
        except Exception as _exc:
            logger.warning("Market sensing: fetch [%s] failed", source_name, exc_info=True)
            if _cb is not None:
                _cb.record_failure(source_name, str(_exc))
            signals = []

        if not signals:
            # Empty result counts as a soft failure for circuit breaker purposes
            # (avoids tripping on legitimately empty responses like after-hours data)
            # Only record failure if an exception was thrown (handled above).
            return

        if _cb is not None:
            _cb.record_success(source_name)

        self._last_fetch_source = source_name

        # Signals arrive pre-enriched from background thread (_fetch_source)

        # Feed into convergence engine (global + tiered)
        for sig in signals:
            sig_kwargs = dict(
                signal_id=sig.signal_id,
                strength=sig.strength,
                domain=sig.domain,
                direction=sig.direction,
                confidence=sig.confidence,
                velocity=sig.velocity,
                timestamp=sig.timestamp,
                metadata={**sig.metadata, "symbol": sig.symbol},
                source=sig.source,
            )
            # Global alerter
            if self._convergence_alerter is not None:
                try:
                    self._convergence_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Failed to feed signal to global alerter", exc_info=True)

            # Route to tier
            tier = TIER_ROUTING.get(sig.source, "strategic")
            tier_alerter = self._tiered_alerters.get(tier)
            if tier_alerter is not None:
                try:
                    tier_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Failed to feed signal to %s alerter", tier, exc_info=True)

        # Track per-ticker signal domains for archetype scanning (Gift 8)
        for sig in signals:
            sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
            if sym:
                if sym not in self._recent_domains:
                    self._recent_domains[sym] = set()
                self._recent_domains[sym].add(sig.domain)

        # Feed deception detector (Gift 5) — track signal patterns for manipulation detection
        if self._deception_detector is not None:
            for sig in signals:
                try:
                    sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
                    if sym:
                        self._deception_detector.record_signal(
                            ticker=sym,
                            domain=sig.domain,
                            direction=sig.direction,
                            strength=sig.strength,
                            timestamp=sig.timestamp,
                        )
                except Exception:
                    logger.debug("Deception detector record failed", exc_info=True)

        # Feed somatic anticipation (Gift 9) — accumulate per-ticker signal state
        if self._somatic_anticipation is not None:
            for sig in signals:
                try:
                    sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
                    if sym:
                        self._somatic_anticipation.record_signal(
                            ticker=sym,
                            domain=sig.domain,
                            direction=sig.direction,
                            strength=sig.strength,
                            timestamp=sig.timestamp,
                        )
                except Exception:
                    logger.debug("Somatic anticipation record failed", exc_info=True)

        # Check pattern completions (Gift 10) — match signals against active hunts
        if self._pattern_completion_engine is not None:
            try:
                completion_events = self._pattern_completion_engine.check_completions(signals)
                if completion_events:
                    logger.info(
                        "Pattern completion: %d hunts matched by incoming signals",
                        len(completion_events),
                    )
            except Exception:
                logger.debug("Pattern completion check failed", exc_info=True)

        self._total_signals_fed += len(signals)
        logger.info(
            "Market sensing: fed %d signals from [%s] (total: %d)",
            len(signals), source_name, self._total_signals_fed,
        )

        # Record signal arrival for AbsenceMonitor cadence tracking
        if self._absence_monitor is not None and signals:
            try:
                self._absence_monitor.record_arrival(source_name, datetime.now())
            except Exception:
                logger.debug("AbsenceMonitor record_arrival failed", exc_info=True)

        # Feed CorrelationTracker (cross-source anomaly detection)
        if self._correlation_tracker is not None and signals:
            try:
                max_strength = max(sig.strength for sig in signals)
                domain = signals[0].domain if hasattr(signals[0], "domain") else None
                self._correlation_tracker.record(
                    signal_id=source_name,
                    value=max_strength,
                    timestamp=datetime.now(),
                    domain=domain,
                )
            except Exception:
                logger.debug("CorrelationTracker record failed", exc_info=True)

        # Publish each signal to EventBus (hypothesis engine subscribes here)
        if self._bus is not None:
            from mae_core.market.channels import CH_SIGNAL_INGESTED
            for sig in signals:
                try:
                    self._bus.publish(CH_SIGNAL_INGESTED, {
                        "signal_id": sig.signal_id,
                        "source": sig.source,
                        "symbol": sig.symbol,
                        "domain": sig.domain,
                        "direction": sig.direction,
                        "strength": sig.strength,
                        "confidence": sig.confidence,
                        "velocity": sig.velocity,
                        "timestamp": sig.timestamp.isoformat(),
                    })
                except Exception:
                    logger.debug("Failed to publish signal to EventBus", exc_info=True)

        # Store to Qdrant + JSONL
        store_signals(signals, self._memory)

        # Register with outcome collector
        if self._outcome_collector is not None:
            try:
                registered = self._outcome_collector.register_signals(signals)
                if registered:
                    logger.info("Market sensing: registered %d predictions for outcome tracking", registered)
            except Exception:
                logger.debug("Outcome registration failed", exc_info=True)

        # --- Reactive convergence check ---
        # Run convergence immediately after signals are ingested rather than
        # waiting for the next step tick. The step-tick check in market_hooks.py
        # remains as a safety net; alerter deduplication (cooldown window) prevents
        # duplicate alerts from firing on both paths.
        self._trigger_reactive_convergence(signals)

        # --- Reactive PatternWatcher check ---
        # Run pattern watcher immediately after signals arrive instead of
        # waiting for the 10-step cadence. Builds active signal map from
        # convergence alerter buffer (same as _run_sensing_archaeology).
        pw = getattr(self, "_pattern_watcher", None)
        if pw is not None and self._convergence_alerter is not None:
            try:
                _active: dict = {}
                for _domain, _sigs in self._convergence_alerter.signals.items():
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
                    _stacks = pw.check(_active)
                    if _stacks:
                        logger.info(
                            "Reactive PatternWatcher: %d stacks after [%s]",
                            len(_stacks), source_name,
                        )
            except Exception:
                logger.debug("Reactive pattern watcher check failed", exc_info=True)
