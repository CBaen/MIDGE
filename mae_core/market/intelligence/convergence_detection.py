"""ConvergenceDetectionMixin — convergence checks, ticker convergence, reporting.

Used by ConvergenceAlerter as a mixin. Expects the following instance attributes
(provided by ConvergenceAlerter.__init__ and ConvergenceConfidenceMixin):
    self.signals, self.alerts, self.min_domains, self.min_strength,
    self.convergence_window, self._domain_windows, self.domain_categories,
    self.persistence_path, self._alert_counter, self._last_alert_times,
    self._alert_lock, self._min_alert_interval_hours,
    self._thompson, self._world_model, self._catalyst_calendar,
    self._cross_asset_confirmer, self._deception_detector,
    self._economic_calendar, self._pattern_archetype_engine,
    self._causal_engine, self._bus,
    -- and all methods from ConvergenceConfidenceMixin --
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mae_core.market.intelligence.convergence_models import ConvergenceAlert, Signal

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DISCOVERY_LOG = _DATA_DIR / "discovery_log.jsonl"


class ConvergenceDetectionMixin:
    """Convergence checking, ticker-level alerts, reporting, and persistence."""

    def check_convergence(
        self,
        direction_filter: str = None
    ) -> List[ConvergenceAlert]:
        """
        Check for convergence and generate alerts.

        Args:
            direction_filter: Only check for specific direction (bullish/bearish)

        Returns:
            List of ConvergenceAlert objects
        """
        self._prune_old_signals()

        # Compute coherence once across all signals — shared by both directions.
        # Bearish signals dampen bullish confidence and vice versa.
        coherence_data = self._compute_coherence_score()

        # Publish CH_CONTRADICTION_DETECTED when signals materially disagree.
        # Completes Cap 3 (narrative coherence) — the sensor existed but never
        # published to the EventBus. Also feeds CausalReasoningEngine if wired.
        coherence = coherence_data.get("coherence", 1.0)
        if coherence < 0.7 and self._bus is not None:
            try:
                from mae_core.market.channels import CH_CONTRADICTION_DETECTED
                self._bus.publish(CH_CONTRADICTION_DETECTED, {
                    "coherence": coherence,
                    "dominant_direction": coherence_data.get("dominant_direction"),
                    "bullish_count": coherence_data.get("bullish_count", 0),
                    "bearish_count": coherence_data.get("bearish_count", 0),
                    "contradiction_details": coherence_data.get("contradiction_details", []),
                })
            except Exception:
                logger.debug("CH_CONTRADICTION_DETECTED publish failed", exc_info=True)

        # Feed domain pair correlations into CausalReasoningEngine.
        # This builds the causal graph that hypothesis_validator can later query.
        if self._causal_engine is not None:
            try:
                active_domains = [d for d, sigs in self.signals.items() if sigs]
                for i, domain_a in enumerate(active_domains):
                    for domain_b in active_domains[i + 1:]:
                        # Compute strength as product of max signal strengths
                        str_a = max((s.strength for s in self.signals[domain_a]), default=0)
                        str_b = max((s.strength for s in self.signals[domain_b]), default=0)
                        corr_strength = str_a * str_b
                        if corr_strength > 0.3:
                            self._causal_engine.observe_correlation(
                                domain_a, domain_b, corr_strength,
                                context={"source": "convergence_alerter"},
                            )
            except Exception:
                logger.debug("CausalReasoningEngine observation failed", exc_info=True)

        alerts = []

        # Check bullish convergence
        if direction_filter in (None, "bullish"):
            alert = self._check_direction_convergence("bullish", coherence_data)
            if alert:
                alerts.append(alert)

        # Check bearish convergence
        if direction_filter in (None, "bearish"):
            alert = self._check_direction_convergence("bearish", coherence_data)
            if alert:
                alerts.append(alert)

        # Alert deduplication — suppress re-alert within interval, per direction.
        # Using a dict keyed by direction so bullish and bearish each maintain
        # their own independent suppression window.  A threading.Lock guards the
        # dict so concurrent step-hook calls can't race each other.
        with self._alert_lock:
            now = datetime.now()
            filtered = []
            for alert in alerts:
                direction = alert.direction if hasattr(alert, "direction") else "neutral"
                last_time = self._last_alert_times.get(direction)
                if (last_time is not None
                        and (now - last_time).total_seconds() / 3600
                        < self._min_alert_interval_hours):
                    continue  # Suppress — same direction, too recent
                filtered.append(alert)
                self._last_alert_times[direction] = now
            alerts = filtered

        # Store alerts (capped to prevent unbounded growth)
        self.alerts.extend(alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

        # Log novel cross-domain discoveries
        for alert in alerts:
            if alert.cross_domain_count >= 3:
                self._log_discovery(alert)

        return alerts

    def _check_direction_convergence(
        self,
        direction: str,
        coherence_data: dict = None,
    ) -> Optional[ConvergenceAlert]:
        """Check for convergence in a specific direction.

        Directional signals (bullish/bearish) contribute domain presence, direction
        voting, strength, and confidence.

        Neutral signals (e.g. finra_short with short_ratio < threshold) contribute
        domain presence and confidence weighting ONLY — they do not cast a direction
        vote and are not counted toward avg_strength.  This ensures that a reliable
        volatility predictor like FINRA short-interest counts toward min_domains
        without polluting the directional signal.

        Capability 3 — Narrative coherence: if coherence_data is provided, the
        final confidence is multiplied by a coherence_multiplier derived from the
        global directional vote ratio.  Perfect agreement (coherence=1.0) leaves
        confidence unchanged; evenly split signals (coherence=0.5) halve it.
        Formula: coherence_multiplier = 0.5 + 0.5 * coherence → range [0.5, 1.0].
        """
        directional_signals = []   # Cast direction vote + strength + confidence
        neutral_signals = []       # Domain presence + confidence only
        domains_seen = set()
        categories_seen = set()
        effective_strengths = {}   # id(signal) -> effective strength (Cap 5 + 6)

        for domain, signals in self.signals.items():
            # Directional signals for the requested direction
            matching = [
                s for s in signals
                if s.direction == direction and s.strength >= self.min_strength
            ]

            # Neutral signals from this domain (direction-agnostic predictors)
            neutral = [
                s for s in signals
                if s.direction == "neutral" and s.strength >= self.min_strength
            ]

            if matching:
                # Cap 5: Select strongest by freshness-weighted strength.
                # A 2-hour-old signal at 0.8 beats a 70-hour-old signal at 0.85.
                strongest = max(
                    matching,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                max_eff = strongest.strength * self._compute_freshness(strongest, domain)

                # Cap 6: Intra-domain count boost — multiple confirming signals
                # from the same domain IS additional evidence.
                # 1 signal: 1.0x, 2: ~1.07x, 3: ~1.11x (log-saturating)
                count = len(matching)
                if count > 1:
                    max_eff *= (1 + 0.1 * math.log(count))

                directional_signals.append(strongest)
                effective_strengths[id(strongest)] = max_eff
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))
            elif neutral:
                # Cap 5: Select strongest neutral by freshness-weighted strength.
                strongest_neutral = max(
                    neutral,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                neutral_signals.append(strongest_neutral)
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))

        # Need minimum domains (directional + neutral combined)
        if len(domains_seen) < self.min_domains:
            # Emit partial convergence for ecosystem investigation
            if directional_signals and self._bus is not None:
                try:
                    signal_dicts = [{"source": s.source, "strength": s.strength,
                                     "symbol": getattr(s, "symbol", ""),
                                     "metadata": s.metadata} for s in directional_signals[:5]]
                    # Causal predictions: what the WorldModel says should follow
                    causal_predictions = self._compute_ripple_effects(directional_signals)
                    self._bus.publish("market.intel.partial_convergence", {
                        "direction": direction,
                        "domains_seen": list(domains_seen),
                        "missing_domains": self._compute_missing_domains(domains_seen),
                        "signals": signal_dicts,
                        "min_domains_required": self.min_domains,
                        "causal_predictions": causal_predictions,
                    })
                except Exception:
                    pass  # Never block convergence check
            return None

        # Confidence is computed over ALL contributing signals so that
        # Thompson-weighted neutral sources (e.g. high-reliability FINRA short)
        # improve the estimate without inflating strength or direction.
        all_contributing = directional_signals + neutral_signals

        # Calculate cross-domain count (different categories)
        cross_domain_count = len(categories_seen)

        # Cap 5+6: Strength uses effective (freshness-weighted, count-boosted) values.
        # Neutral signals don't vote on strength — they only add domain presence.
        if directional_signals:
            avg_strength = sum(
                effective_strengths.get(id(s), s.strength)
                for s in directional_signals
            ) / len(directional_signals)
        else:
            # All domains are neutral — no directional conviction; don't alert
            return None

        final_confidence = self._compute_confidence(all_contributing, cross_domain_count)

        # Capability 3 — Apply coherence multiplier.
        # Bearish signals dampen bullish confidence and vice versa.
        # coherence_multiplier = 0.5 + 0.5 * coherence → range [0.5, 1.0].
        # Perfect agreement (coherence=1.0): unchanged.  Evenly split: halved.
        coherence = 1.0
        contradiction_details: list = []
        if coherence_data is not None:
            coherence = coherence_data.get("coherence", 1.0)
            contradiction_details = coherence_data.get("contradiction_details", [])
            coherence_multiplier = 0.5 + 0.5 * coherence
            final_confidence = max(0.05, min(0.95, final_confidence * coherence_multiplier))

        # Gift 3: Catalyst Calendar modifier — timing context affects confidence.
        # Extract primary ticker from signals' metadata for catalyst + cross-asset.
        primary_ticker = None
        primary_domain = None
        for sig in directional_signals:
            sym = sig.metadata.get("symbol", "")
            if sym:
                primary_ticker = sym
                primary_domain = sig.domain
                break

        if primary_ticker and self._catalyst_calendar is not None:
            try:
                modifier = self._catalyst_calendar.compute_catalyst_modifier(
                    primary_ticker, signal_domain=primary_domain or "",
                )
                final_confidence = max(0.05, min(0.95, final_confidence * modifier))
            except Exception:
                pass  # Graceful degradation

        # Gift 4: Cross-Asset Confirmation — correlated assets agree/disagree.
        # Score range: -1 to 1. Mapped to confidence multiplier: 0.4x to 1.2x.
        if primary_ticker and self._cross_asset_confirmer is not None:
            try:
                result = self._cross_asset_confirmer.check_confirmation(
                    primary_ticker, direction,
                )
                if result is not None:
                    score = getattr(result, "confirmation_score", 0.0)
                    cross_multiplier = 0.8 + 0.4 * score
                    final_confidence = max(0.05, min(0.95, final_confidence * cross_multiplier))
            except Exception:
                pass  # Graceful degradation

        # Gift 5: Deception detection — penalize confidence on suspected manipulation.
        if primary_ticker and self._deception_detector is not None:
            try:
                assessment = self._deception_detector.assess_signal_authenticity(primary_ticker)
                if assessment is not None and assessment.deception_probability > 0.5:
                    deception_penalty = 1.0 - assessment.deception_probability
                    final_confidence = max(0.05, final_confidence * deception_penalty)
            except Exception:
                pass  # Graceful degradation

        # Economic calendar suppression — reduce confidence during high-impact events.
        # FOMC, CPI, NFP create scheduled volatility that makes normal signals unreliable.
        # Multiplier: 0.5x during suppression windows (halves confidence).
        if self._economic_calendar is not None:
            try:
                if self._economic_calendar.is_in_suppression_window():
                    final_confidence = max(0.05, final_confidence * 0.5)
                    logger.info(
                        "Convergence: suppression window active — confidence reduced to %.2f",
                        final_confidence,
                    )
            except Exception:
                pass  # Graceful degradation

        # Gift 8: Pattern Archetype context — boost confidence when signals match a known archetype.
        if primary_ticker and self._pattern_archetype_engine is not None:
            try:
                matches = self._pattern_archetype_engine.scan_for_archetypes(
                    primary_ticker, signal_domains=list(domains_seen),
                )
                if matches:
                    best = max(matches, key=lambda m: m.match_score)
                    if best.match_score > 0.7:
                        final_confidence = min(0.95, final_confidence + 0.10)
            except Exception:
                pass  # Graceful degradation

        # Combo Thompson — historical domain combination reliability.
        # Different domain combos have vastly different win rates (8% to 67%).
        # The combo key tracks per-combination Thompson distributions.
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

        # Determine urgency based on velocity (directional signals only)
        avg_velocity = sum(abs(s.velocity) for s in directional_signals) / len(directional_signals)
        if avg_velocity > 0.1:
            urgency = "immediate"
        elif avg_velocity > 0.05:
            urgency = "hours"
        else:
            urgency = "days"

        # Temporal ordering — sort contributing signals by when they fired.
        # domain_sequence: domains ordered earliest-to-latest.
        # sequence_score: multiplier based on how well order matches known lags.
        try:
            domain_sequence = self._build_domain_sequence(all_contributing)
            sequence_score = self._compute_sequence_score(domain_sequence)
            # Apply sequence score to confidence (within existing [0.05, 0.95] bounds)
            if sequence_score != 1.0:
                final_confidence = max(0.05, min(0.95, final_confidence * sequence_score))
        except Exception:
            domain_sequence = []
            sequence_score = 1.0
            logger.debug("Temporal ordering failed gracefully", exc_info=True)

        # Quorum boost — collective agent agreement amplifies confidence.
        # Uses primary_ticker extracted earlier in this method.
        if primary_ticker:
            final_confidence = self._apply_quorum_boost(
                final_confidence, primary_ticker, direction
            )

        # Generate summary — note neutral signal count and coherence for transparency
        domain_list = ", ".join(sorted(domains_seen))
        neutral_note = f", +{len(neutral_signals)} neutral" if neutral_signals else ""
        coherence_note = f", coherence={coherence:.2f}" if coherence < 1.0 else ""
        seq_note = f", seq={sequence_score:.2f}" if sequence_score != 1.0 else ""
        summary = (
            f"{direction.upper()}: {len(domains_seen)} domains converging "
            f"({domain_list}) | {len(directional_signals)} directional"
            f"{neutral_note} | strength={avg_strength:.2f}, "
            f"confidence={final_confidence:.2f}, urgency={urgency}{coherence_note}{seq_note}"
        )

        # Create alert — signals list includes both directional and neutral
        # so callers can inspect the full contributing set
        self._alert_counter += 1
        alert = ConvergenceAlert(
            alert_id=f"CONV-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
            timestamp=datetime.now(),
            direction=direction,
            strength=avg_strength,
            confidence=final_confidence,
            domains_converging=sorted(domains_seen),
            signals=all_contributing,
            cross_domain_count=cross_domain_count,
            summary=summary,
            urgency=urgency,
            coherence=coherence,
            contradiction_details=contradiction_details,
            combo_key=combo_key,
            domain_sequence=domain_sequence,
            sequence_score=sequence_score,
            ripple_effects=self._compute_ripple_effects(all_contributing),
        )

        return alert

    def get_domain_status(self) -> Dict[str, Dict]:
        """
        Get current status by domain.

        Returns:
            Dict mapping domain -> {direction, strength, signal_count}
        """
        self._prune_old_signals()

        status = {}
        for domain, signals in self.signals.items():
            if not signals:
                continue

            # Determine dominant direction
            bullish = [s for s in signals if s.direction == "bullish"]
            bearish = [s for s in signals if s.direction == "bearish"]

            if len(bullish) > len(bearish):
                dominant = "bullish"
                strength = sum(s.strength for s in bullish) / len(bullish) if bullish else 0
            elif len(bearish) > len(bullish):
                dominant = "bearish"
                strength = sum(s.strength for s in bearish) / len(bearish) if bearish else 0
            else:
                dominant = "neutral"
                strength = 0.5

            status[domain] = {
                "direction": dominant,
                "strength": round(strength, 3),
                "signal_count": len(signals),
                "bullish_count": len(bullish),
                "bearish_count": len(bearish),
                "category": self.domain_categories.get(domain, "unknown")
            }

        return status

    def _compute_missing_domains(self, domains_seen: set) -> list:
        """Return domains with current signals that haven't fired for this direction."""
        all_domains = set(self.signals.keys())
        return sorted(all_domains - domains_seen)

    def _compute_ripple_effects(self, signals) -> List[dict]:
        """Trace downstream causal cascade from alert signals via WorldModel.

        For each signal, maps it to a world model trigger (e.g. an EIA crude
        draw maps to 'eia_crude_draw'), then traces all downstream effects
        (tickers that should move if the thesis is correct).

        This is what makes MIDGE an inevitability surfacer — not just detecting
        convergence, but predicting what should follow.
        """
        if self._world_model is None:
            return []

        seen_tickers = set()
        ripples = []

        for signal in signals:
            source = signal.source if hasattr(signal, "source") else signal.get("source", "")
            metadata = signal.metadata if hasattr(signal, "metadata") else signal.get("metadata", {})
            trigger = self._world_model.map_signal_to_trigger(source, metadata)
            if not trigger:
                continue

            effects = self._world_model.find_ripple_effects(trigger)
            for effect in effects:
                if effect.ticker not in seen_tickers:
                    seen_tickers.add(effect.ticker)
                    ripples.append({
                        "ticker": effect.ticker,
                        "direction": effect.direction,
                        "strength": round(effect.strength, 3),
                        "lag_days": effect.total_lag_days,
                        "path": effect.path,
                        "confidence": round(effect.confidence, 3),
                    })

        ripples.sort(key=lambda r: r["strength"], reverse=True)
        return ripples[:20]

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

            # Check convergence for this ticker in each direction
            for direction in ("bullish", "bearish"):
                converging = []
                domains_seen = set()
                categories_seen = set()

                for domain, sigs in domain_signals.items():
                    matching = [s for s in sigs if s.direction == direction]
                    if matching:
                        strongest = max(matching, key=lambda s: s.strength)
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
                # ticker is explicit in this per-ticker path.
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

    def get_convergence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Get matrix of which domains agree with each other.

        Returns:
            Dict[domain_a][domain_b] = 1 (agree), -1 (disagree), 0 (no data)
        """
        status = self.get_domain_status()
        domains = list(status.keys())
        matrix = {}

        for domain_a in domains:
            matrix[domain_a] = {}
            dir_a = status[domain_a]["direction"]

            for domain_b in domains:
                if domain_a == domain_b:
                    matrix[domain_a][domain_b] = 1
                    continue

                dir_b = status[domain_b]["direction"]

                if dir_a == "neutral" or dir_b == "neutral":
                    matrix[domain_a][domain_b] = 0
                elif dir_a == dir_b:
                    matrix[domain_a][domain_b] = 1
                else:
                    matrix[domain_a][domain_b] = -1

        return matrix

    def get_actionable_summary(self) -> Dict:
        """
        Get actionable summary for trading decisions.

        Returns:
            Dict with direction recommendation, confidence, and reasoning
        """
        status = self.get_domain_status()

        bullish_domains = [d for d, s in status.items() if s["direction"] == "bullish"]
        bearish_domains = [d for d, s in status.items() if s["direction"] == "bearish"]

        bullish_strength = sum(status[d]["strength"] for d in bullish_domains) if bullish_domains else 0
        bearish_strength = sum(status[d]["strength"] for d in bearish_domains) if bearish_domains else 0

        # Count unique categories
        bullish_categories = set(status[d]["category"] for d in bullish_domains)
        bearish_categories = set(status[d]["category"] for d in bearish_domains)

        if len(bullish_domains) > len(bearish_domains) and len(bullish_categories) >= 2:
            recommendation = "bullish"
            avg_strength = bullish_strength / max(1, len(bullish_domains))
            n_cats = len(bullish_categories)
            diversity = 1.0 + 0.10 * math.log1p(n_cats)
            confidence = min(0.85, avg_strength * diversity)
            reasoning = f"{len(bullish_domains)} domains ({n_cats} categories) bullish"
        elif len(bearish_domains) > len(bullish_domains) and len(bearish_categories) >= 2:
            recommendation = "bearish"
            avg_strength = bearish_strength / max(1, len(bearish_domains))
            n_cats = len(bearish_categories)
            diversity = 1.0 + 0.10 * math.log1p(n_cats)
            confidence = min(0.85, avg_strength * diversity)
            reasoning = f"{len(bearish_domains)} domains ({n_cats} categories) bearish"
        else:
            recommendation = "neutral"
            confidence = 0.3
            reasoning = "Insufficient convergence - signals mixed or single-category"

        return {
            "recommendation": recommendation,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "bullish_domains": bullish_domains,
            "bearish_domains": bearish_domains,
            "total_signals": sum(s["signal_count"] for s in status.values())
        }

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "domain_count": len(self.signals),
            "alert_count": len(self.alerts),
            "recent_alerts": [a.to_dict() for a in list(self.alerts)[-3:]],
        }

    def step(self) -> None:
        """Step hook for HolonProxy.act() delegation.

        Does not publish — bootstrap hook handles EventBus publishing
        with deduplication logic.
        """
        self.check_convergence()

    def to_dict(self) -> dict:
        """Export state for API/persistence."""
        return {
            "config": {
                "min_domains": self.min_domains,
                "min_strength": self.min_strength,
                "convergence_window_hours": self.convergence_window.total_seconds() / 3600
            },
            "domain_status": self.get_domain_status(),
            "actionable_summary": self.get_actionable_summary(),
            "recent_alerts": [a.to_dict() for a in self.alerts[-10:]]
        }

    def _log_discovery(self, alert: ConvergenceAlert) -> None:
        """Log a novel cross-domain convergence pattern to discovery_log.jsonl."""
        record = {
            "timestamp": alert.timestamp.isoformat(),
            "alert_id": alert.alert_id,
            "direction": alert.direction,
            "strength": round(alert.strength, 3),
            "confidence": round(alert.confidence, 3),
            "domains": alert.domains_converging,
            "cross_domain_count": alert.cross_domain_count,
            "signal_sources": [s.signal_id for s in alert.signals],
            "summary": alert.summary,
        }
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            with open(_DISCOVERY_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Failed to write discovery log: %s", e)

    def save(self):
        """Persist state to disk."""
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
