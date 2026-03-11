"""Sensing step operations mixin — periodic cadence-based step operations.

Provides _run_step_portfolio, _run_step_absence, _run_step_archetype
and related cadence helpers called from MarketSensingHook.step().
Mixed into MarketSensingHook via SensingStepOpsMixin.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger("midge.market.sensing")


class SensingStepOpsMixin:
    """Mixin providing periodic cadence operations for the step loop.

    Requires the host class to have all the relevant client attributes.
    """

    def _run_step_portfolio(self):
        """Portfolio tracker: mark-to-market + exit signal check (cadence 50)."""
        if self._portfolio_tracker is None:
            return
        try:
            self._portfolio_tracker.update_prices()
            exits = self._portfolio_tracker.check_exits()
            if exits and self._convergence_alerter is not None:
                for exit_sig in exits:
                    try:
                        self._convergence_alerter.record_signal(
                            signal_id=f"exit:{exit_sig.ticker}:{exit_sig.reason}",
                            strength=exit_sig.strength,
                            domain="portfolio",
                            direction=exit_sig.direction,
                            source="portfolio_exit",
                        )
                    except Exception:
                        logger.debug("Portfolio exit signal feed failed", exc_info=True)
        except Exception:
            logger.debug("Portfolio tracker step failed", exc_info=True)

    def _run_step_absence(self):
        """Absence detection — check for unexpectedly silent sources (cadence 100)."""
        if self._absence_monitor is None:
            return
        from mae_core.market.sensing_constants import _absence_source_to_domain
        try:
            absences = self._absence_monitor.check_absences()
            if absences:
                logger.info(
                    "AbsenceMonitor: %d sources unexpectedly silent",
                    len(absences),
                )
                if self._convergence_alerter is not None:
                    for absence in absences:
                        try:
                            from mae_core.market.intelligence.convergence_alerter import Signal as CASignal
                            absence_signal = CASignal(
                                signal_id=f"absence:{absence.source}",
                                strength=min(1.0, 0.3 + 0.1 * (absence.silence_ratio - self._absence_monitor._absence_threshold)),
                                domain=_absence_source_to_domain(absence.source),
                                direction=absence.direction,
                                timestamp=datetime.now(),
                                confidence=0.5,
                            )
                            self._convergence_alerter.record_signal(
                                absence_signal.signal_id,
                                absence_signal.strength,
                                absence_signal.domain,
                                direction=absence_signal.direction,
                                source=f"absence_{absence.source}",
                            )
                        except Exception:
                            logger.debug("Absence signal feed failed", exc_info=True)
        except Exception:
            logger.debug("AbsenceMonitor check failed", exc_info=True)

    def _run_step_correlation(self):
        """Cross-source correlation anomaly scan (cadence 200)."""
        if self._correlation_tracker is None:
            return
        try:
            anomalies = self._correlation_tracker.detect_cross_domain_anomalies()
            if anomalies:
                logger.info(
                    "CorrelationTracker: %d cross-domain anomalies detected",
                    len(anomalies),
                )
        except Exception:
            logger.debug("CorrelationTracker anomaly scan failed", exc_info=True)

    def _run_step_consolidation(self):
        """Memory pruning via consolidation engine (cadence 5000)."""
        if self._consolidation_engine is None:
            return
        try:
            if self._consolidation_engine.should_consolidate(self._step_counter):
                self._consolidation_engine.consolidate()
        except Exception:
            logger.debug("Consolidation engine step failed", exc_info=True)

    def _run_step_somatic(self):
        """Somatic anticipation: pre-conscious pattern response (cadence 25)."""
        if self._somatic_anticipation is None:
            return
        try:
            events = self._somatic_anticipation.check_anticipation()
            if events:
                logger.info(
                    "Somatic anticipation: %d tickers building pre-convergence",
                    len(events),
                )
        except Exception:
            logger.debug("Somatic anticipation check failed", exc_info=True)

    def _run_step_archetypes(self):
        """Pattern archetype scanning: check watchlist tickers (cadence 100)."""
        if self._pattern_archetype_engine is None:
            return
        try:
            for ticker in list(self._watchlist.get("tickers", []))[:5]:
                self._pattern_archetype_engine.scan_for_archetypes(
                    ticker, signal_domains=list(self._recent_domains.get(ticker, [])),
                )
        except Exception:
            logger.debug("Pattern archetype scan failed", exc_info=True)

    def _run_step_pattern_completion(self):
        """Pattern completion: review partial archetype matches (cadence 100)."""
        if self._pattern_completion_engine is None:
            return
        try:
            new_hunts = self._pattern_completion_engine.review_partial_matches()
            if new_hunts:
                logger.info(
                    "Pattern completion: %d new hunts created",
                    len(new_hunts),
                )
            # Prune expired hunts
            self._pattern_completion_engine._prune_expired()
        except Exception:
            logger.debug("Pattern completion review failed", exc_info=True)
