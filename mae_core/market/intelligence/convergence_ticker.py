"""ConvergenceTickerMixin — per-ticker convergence detection.

Used by ConvergenceDetectionMixin as a mixin. Expects the following instance
attributes and methods (provided by ConvergenceAlerter and ConvergenceConfidenceMixin):
    self.signals, self.min_domains, self.domain_categories,
    self._thompson, self._deception_detector,
    self._prune_old_signals(), self._compute_confidence(),
    self._get_regime(), self._build_domain_sequence(),
    self._compute_sequence_score(), self._apply_quorum_boost(),
    self._compute_ripple_effects(), self._alert_counter
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

from mae_core.market.intelligence.convergence_models import ConvergenceAlert, Signal

logger = logging.getLogger(__name__)

# Minimum interval between ticker alerts for the same ticker+direction.
_TICKER_DEDUP_HOURS = 1.0


class ConvergenceTickerMixin:
    """Per-ticker convergence detection."""

    # Populated on first use. Keyed by "TICKER:direction" → datetime.
    _last_ticker_alert_times: Dict[str, datetime]

    def check_ticker_convergence_for(self, ticker: str) -> list:
        """Check convergence for a single ticker symbol."""
        return [
            a for a in self.check_ticker_convergence(min_domains=self.min_domains)
            if any(
                s.metadata.get("symbol") == ticker
                for s in getattr(a, "signals", [])
            )
        ]

    def check_ticker_convergence(self, min_domains: int = 2) -> List[ConvergenceAlert]:
        """Per-ticker convergence — more actionable than global domain convergence.

        Groups signals by symbol, then checks if multiple domains converge
        on the same ticker. A ticker with insider buying + hiring bullish +
        contract award across 3 domains is far stronger than 3 domains
        globally bullish on different stocks.

        Args:
            min_domains: Minimum different domains with signals on the same ticker

        Returns:
            List of ConvergenceAlert objects, one per ticker with sufficient convergence
        """
        with self._alert_lock:
            return self._check_ticker_convergence_locked(min_domains)

    def _check_ticker_convergence_locked(self, min_domains: int = 2) -> List[ConvergenceAlert]:
        """Inner per-ticker convergence — caller must hold self._alert_lock."""
        self._prune_old_signals()

        # Group all signals by symbol
        by_ticker: Dict[str, Dict[str, List[Signal]]] = defaultdict(lambda: defaultdict(list))
        for domain, signals in self.signals.items():
            for sig in signals:
                symbol = sig.metadata.get("symbol", "")
                if symbol:
                    by_ticker[symbol][domain].append(sig)

        alerts = []
        for ticker, domain_signals in by_ticker.items():
            if len(domain_signals) < min_domains:
                continue

            for direction in ("bullish", "bearish"):
                converging = []
                domains_seen = set()
                categories_seen = set()

                # Track seen enrichment groups to prevent a single fanned-out
                # event from counting as multiple domain contributions.
                seen_enrichment_groups: set = set()

                for domain, sigs in domain_signals.items():
                    matching = [s for s in sigs if s.direction == direction]
                    if matching:
                        strongest = max(matching, key=lambda s: s.strength)
                        # If this signal is an enriched copy, check the group.
                        eg = strongest.metadata.get("enrichment_group", "")
                        if eg and eg in seen_enrichment_groups:
                            logger.debug(
                                "Ticker %s: skipping domain [%s] — enrichment_group "
                                "%s already counted", ticker, domain, eg[:20],
                            )
                            continue
                        if eg:
                            seen_enrichment_groups.add(eg)
                        converging.append(strongest)
                        domains_seen.add(domain)
                        categories_seen.add(self.domain_categories.get(domain, domain))

                if len(domains_seen) < min_domains:
                    continue

                avg_strength = sum(s.strength for s in converging) / len(converging)
                cross_domain_count = len(categories_seen)
                final_confidence = self._compute_confidence(converging, cross_domain_count)

                # Combo Thompson — same logic as global convergence path.
                combo_key = "combo:" + "+".join(sorted(domains_seen))
                if self._thompson is not None:
                    try:
                        regime = self._get_regime()
                        combo_dist = self._thompson.get_distribution(combo_key, regime)
                        if combo_dist.samples >= 5:
                            combo_multiplier = 0.5 + combo_dist.mean
                            final_confidence = max(0.05, min(0.95, final_confidence * combo_multiplier))
                    except Exception:
                        pass  # Graceful degradation

                # Temporal ordering — same logic as global convergence path.
                domain_sequence = []
                sequence_score = 1.0
                try:
                    domain_sequence = self._build_domain_sequence(converging)
                    sequence_score = self._compute_sequence_score(domain_sequence)
                    if sequence_score != 1.0:
                        final_confidence = max(0.05, min(0.95, final_confidence * sequence_score))
                except Exception:
                    pass  # Graceful degradation

                # Deception detection — same logic as global convergence path.
                if self._deception_detector is not None:
                    try:
                        deception_penalty = self._deception_detector.check(
                            ticker, direction, [s.source for s in converging]
                        )
                        if deception_penalty < 1.0:
                            final_confidence = max(0.05, final_confidence * deception_penalty)
                    except Exception:
                        pass  # Graceful degradation

                # Quorum boost — collective agent agreement amplifies confidence.
                final_confidence = self._apply_quorum_boost(
                    final_confidence, ticker, direction
                )

                avg_velocity = sum(abs(s.velocity) for s in converging) / len(converging)
                if avg_velocity > 0.1:
                    urgency = "immediate"
                elif avg_velocity > 0.05:
                    urgency = "hours"
                else:
                    urgency = "days"

                # Dedup: suppress same ticker+direction within the dedup window.
                if not hasattr(self, "_last_ticker_alert_times"):
                    self._last_ticker_alert_times = {}
                _dedup_key = f"{ticker}:{direction}"
                _now = datetime.now()
                _last_fire = self._last_ticker_alert_times.get(_dedup_key)
                if _last_fire and (_now - _last_fire) < timedelta(hours=_TICKER_DEDUP_HOURS):
                    logger.debug(
                        "Ticker convergence dedup: %s suppressed (%.0fs remaining)",
                        _dedup_key,
                        (_last_fire + timedelta(hours=_TICKER_DEDUP_HOURS) - _now).total_seconds(),
                    )
                    continue
                self._last_ticker_alert_times[_dedup_key] = _now

                domain_list = ", ".join(sorted(domains_seen))
                seq_note = f", seq={sequence_score:.2f}" if sequence_score != 1.0 else ""
                summary = (
                    f"TICKER {ticker} {direction.upper()}: {len(domains_seen)} domains "
                    f"({domain_list}) | strength={avg_strength:.2f}, "
                    f"confidence={final_confidence:.2f}, urgency={urgency}{seq_note}"
                )

                self._alert_counter += 1
                alert = ConvergenceAlert(
                    alert_id=f"TCKR-{ticker}-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
                    timestamp=datetime.now(),
                    direction=direction,
                    strength=avg_strength,
                    confidence=final_confidence,
                    domains_converging=sorted(domains_seen),
                    signals=converging,
                    cross_domain_count=cross_domain_count,
                    summary=summary,
                    urgency=urgency,
                    combo_key=combo_key,
                    domain_sequence=domain_sequence,
                    sequence_score=sequence_score,
                    ripple_effects=self._compute_ripple_effects(converging),
                )
                alerts.append(alert)

        return alerts
