"""ConvergenceDetectionMixin — global convergence checks, reporting, persistence.

Inherits:
  ConvergenceTickerMixin (convergence_ticker.py) — per-ticker convergence

Provides:
  check_convergence, _check_direction_convergence,
  get_domain_status, _compute_missing_domains, _compute_ripple_effects,
  get_convergence_matrix, get_actionable_summary, get_statistics,
  step, to_dict, save, _log_discovery

Expects the following instance attributes
(provided by ConvergenceAlerter.__init__ and ConvergenceConfidenceMixin):
    self.signals, self.alerts, self.min_domains, self.min_strength,
    self.convergence_window, self._domain_windows, self.domain_categories,
    self.persistence_path, self._alert_counter, self._last_alert_times,
    self._alert_lock, self._min_alert_interval_hours,
    self._thompson, self._world_model, self._catalyst_calendar,
    self._cross_asset_confirmer, self._deception_detector,
    self._economic_calendar, self._pattern_archetype_engine,
    self._causal_engine, self._bus
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mae_core.market.intelligence.convergence_models import ConvergenceAlert, Signal
from mae_core.market.intelligence.convergence_ticker import ConvergenceTickerMixin

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DISCOVERY_LOG = _DATA_DIR / "discovery_log.jsonl"


class ConvergenceDetectionMixin(ConvergenceTickerMixin):
    """Global convergence checking, domain status, reporting, and persistence."""

    def check_convergence(
        self,
        direction_filter: str = None
    ) -> List[ConvergenceAlert]:
        """Check for convergence and generate alerts.

        Args:
            direction_filter: Only check for specific direction (bullish/bearish)

        Returns:
            List of ConvergenceAlert objects
        """
        self._prune_old_signals()

        coherence_data = self._compute_coherence_score()

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
        if self._causal_engine is not None:
            try:
                active_domains = [d for d, sigs in self.signals.items() if sigs]
                for i, domain_a in enumerate(active_domains):
                    for domain_b in active_domains[i + 1:]:
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

        if direction_filter in (None, "bullish"):
            alert = self._check_direction_convergence("bullish", coherence_data)
            if alert:
                alerts.append(alert)

        if direction_filter in (None, "bearish"):
            alert = self._check_direction_convergence("bearish", coherence_data)
            if alert:
                alerts.append(alert)

        # Alert deduplication — per-direction independent suppression windows.
        with self._alert_lock:
            now = datetime.now()
            filtered = []
            for alert in alerts:
                direction = alert.direction if hasattr(alert, "direction") else "neutral"
                last_time = self._last_alert_times.get(direction)
                if (last_time is not None
                        and (now - last_time).total_seconds() / 3600
                        < self._min_alert_interval_hours):
                    continue
                filtered.append(alert)
                self._last_alert_times[direction] = now
            alerts = filtered

        self.alerts.extend(alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

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

        Directional signals contribute domain presence, direction voting,
        strength, and confidence. Neutral signals contribute domain presence
        and confidence weighting only (no direction vote, no strength vote).

        Capability 3 — Narrative coherence: if coherence_data is provided, the
        final confidence is multiplied by a coherence_multiplier in [0.5, 1.0].
        """
        directional_signals = []
        neutral_signals = []
        domains_seen = set()
        categories_seen = set()
        effective_strengths = {}   # id(signal) -> effective strength

        for domain, signals in self.signals.items():
            matching = [
                s for s in signals
                if s.direction == direction and s.strength >= self.min_strength
            ]
            neutral = [
                s for s in signals
                if s.direction == "neutral" and s.strength >= self.min_strength
            ]

            if matching:
                strongest = max(
                    matching,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                max_eff = strongest.strength * self._compute_freshness(strongest, domain)

                count = len(matching)
                if count > 1:
                    max_eff *= (1 + 0.1 * math.log(count))

                directional_signals.append(strongest)
                effective_strengths[id(strongest)] = max_eff
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))
            elif neutral:
                strongest_neutral = max(
                    neutral,
                    key=lambda s: s.strength * self._compute_freshness(s, domain)
                )
                neutral_signals.append(strongest_neutral)
                domains_seen.add(domain)
                categories_seen.add(self.domain_categories.get(domain, domain))

        if len(domains_seen) < self.min_domains:
            if directional_signals and self._bus is not None:
                try:
                    signal_dicts = [{"source": s.source, "strength": s.strength,
                                     "symbol": getattr(s, "symbol", ""),
                                     "metadata": s.metadata} for s in directional_signals[:5]]
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
                    pass
            return None

        all_contributing = directional_signals + neutral_signals
        cross_domain_count = len(categories_seen)

        if directional_signals:
            avg_strength = sum(
                effective_strengths.get(id(s), s.strength)
                for s in directional_signals
            ) / len(directional_signals)
        else:
            return None

        final_confidence = self._compute_confidence(all_contributing, cross_domain_count)

        coherence = 1.0
        contradiction_details: list = []
        if coherence_data is not None:
            coherence = coherence_data.get("coherence", 1.0)
            contradiction_details = coherence_data.get("contradiction_details", [])
            coherence_multiplier = 0.5 + 0.5 * coherence
            final_confidence = max(0.05, min(0.95, final_confidence * coherence_multiplier))

        primary_ticker = None
        primary_domain = None
        for sig in directional_signals:
            sym = sig.metadata.get("symbol", "")
            if sym:
                primary_ticker = sym
                primary_domain = sig.domain
                break

        combo_key = "combo:" + "+".join(sorted(domains_seen))
        final_confidence = self._apply_confidence_modifiers(
            final_confidence, direction, domains_seen, primary_ticker,
            primary_domain, combo_key,
        )

        avg_velocity = sum(abs(s.velocity) for s in directional_signals) / len(directional_signals)
        if avg_velocity > 0.1:
            urgency = "immediate"
        elif avg_velocity > 0.05:
            urgency = "hours"
        else:
            urgency = "days"

        try:
            domain_sequence = self._build_domain_sequence(all_contributing)
            sequence_score = self._compute_sequence_score(domain_sequence)
            if sequence_score != 1.0:
                final_confidence = max(0.05, min(0.95, final_confidence * sequence_score))
        except Exception:
            domain_sequence = []
            sequence_score = 1.0
            logger.debug("Temporal ordering failed gracefully", exc_info=True)

        if primary_ticker:
            final_confidence = self._apply_quorum_boost(
                final_confidence, primary_ticker, direction
            )

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
        """Get current status by domain.

        Returns:
            Dict mapping domain -> {direction, strength, signal_count}
        """
        self._prune_old_signals()

        status = {}
        for domain, signals in self.signals.items():
            if not signals:
                continue

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
        """Trace downstream causal cascade from alert signals via WorldModel."""
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

    def get_convergence_matrix(self) -> Dict[str, Dict[str, int]]:
        """Get matrix of which domains agree with each other.

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
        """Get actionable summary for trading decisions."""
        status = self.get_domain_status()

        bullish_domains = [d for d, s in status.items() if s["direction"] == "bullish"]
        bearish_domains = [d for d, s in status.items() if s["direction"] == "bearish"]

        bullish_strength = sum(status[d]["strength"] for d in bullish_domains) if bullish_domains else 0
        bearish_strength = sum(status[d]["strength"] for d in bearish_domains) if bearish_domains else 0

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
        """Step hook for HolonProxy.act() delegation."""
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
