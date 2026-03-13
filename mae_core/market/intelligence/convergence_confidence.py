"""ConvergenceConfidenceMixin — Thompson weighting, confidence scoring, coherence.

Inherits:
  ConvergenceLagScoringMixin (convergence_lag_scoring.py)
  ConvergenceBufferMixin     (convergence_buffer.py)

Provides:
  _resolve_thompson_key, _get_thompson_weight, _compute_freshness,
  _apply_quorum_boost, _compute_confidence, _compute_effective_domain_count,
  _max_domain_correlation, _compute_coherence_score

Expects the following instance attributes:
    self._thompson, self._quorum_space, self._correlation_tracker,
    self.signals, self.convergence_window, self._domain_windows,
    self._SWEEP_SOURCES, self._SOURCE_TO_THOMPSON_KEY, self._DOMAIN_SOURCES
    (last 3 provided as class attributes on ConvergenceAlerter)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Dict, List

from mae_core.market.intelligence.convergence_lag_scoring import ConvergenceLagScoringMixin
from mae_core.market.intelligence.convergence_buffer import ConvergenceBufferMixin
from mae_core.market.intelligence.convergence_models import Signal

logger = logging.getLogger(__name__)

# Independence correction thresholds — controls how correlated domains
# are discounted in the effective domain count calculation.
# Phase 0 found macro+technical at r=0.73 — these thresholds gate that fix.
STRONG_CORRELATION_THRESHOLD = 0.5   # |r| above this → half credit (+0.5)
MODERATE_CORRELATION_THRESHOLD = 0.3  # |r| above this → partial credit (+0.7)


class ConvergenceConfidenceMixin(ConvergenceLagScoringMixin, ConvergenceBufferMixin):
    """Thompson weighting, freshness, quorum boost, confidence scoring, and coherence."""

    def _resolve_thompson_key(self, signal: Signal) -> str:
        """Resolve the Thompson distribution key for a signal.

        For sweep sources, runs a most-specific-wins cascade through
        granular backtest-derived keys (Bridge 2):
          sweep_bt:{symbol}:{direction} → sweep_bt:{symbol} →
          sweep_bt:{direction} → generic session_sweep fallback.

        For all other sources with metadata (Capability 2 — contextual Thompson):
          {source}:{role}:{sector} → {source}:{role} → {source} fallback.
          The "size" dimension (large/small) is appended when role is present.
          Size is derived: "large" if transaction_value > $500K, else "small".

        Only selects a granular key if it has >= 5 samples (mature data).
        Final fallback: static _SOURCE_TO_THOMPSON_KEY map (existing behavior).
        """
        # --- Sweep sources: Bridge 2 cascade ---
        if signal.source in self._SWEEP_SOURCES:
            if self._thompson is None:
                return self._SOURCE_TO_THOMPSON_KEY.get(
                    signal.source, signal.source or "unknown"
                )

            symbol = signal.metadata.get("symbol", "")
            direction = signal.direction or ""
            regime = self._get_regime()

            candidates = []
            if symbol and direction:
                candidates.append(f"sweep_bt:{symbol}:{direction}")
            if symbol:
                candidates.append(f"sweep_bt:{symbol}")
            if direction:
                candidates.append(f"sweep_bt:{direction}")

            for key in candidates:
                try:
                    dist = self._thompson.get_distribution(key, regime)
                    if dist.samples >= 5:
                        logger.debug("Thompson cascade: %s → %s (n=%d)",
                                     signal.source, key, dist.samples)
                        return key
                except Exception:
                    continue

            return self._SOURCE_TO_THOMPSON_KEY.get(
                signal.source, signal.source or "unknown"
            )

        # --- All other sources: contextual cascade (Capability 2) ---
        if self._thompson is not None and signal.metadata:
            base = self._SOURCE_TO_THOMPSON_KEY.get(
                signal.source, signal.source or "unknown"
            )
            role = signal.metadata.get("role", "")
            sector = signal.metadata.get("sector", "")
            tx_value = signal.metadata.get("transaction_value", 0)
            size = "large" if tx_value > 500_000 else "small" if tx_value > 0 else ""

            regime = self._get_regime()

            contextual_candidates = []
            if role and sector and size:
                contextual_candidates.append(f"{base}:{role}:{sector}:{size}")
            if role and sector:
                contextual_candidates.append(f"{base}:{role}:{sector}")
            if role and size:
                contextual_candidates.append(f"{base}:{role}:{size}")
            if role:
                contextual_candidates.append(f"{base}:{role}")

            for key in contextual_candidates:
                try:
                    dist = self._thompson.get_distribution(key, regime)
                    if dist.samples >= 5:
                        logger.debug("Contextual Thompson cascade: %s → %s (n=%d)",
                                     signal.source, key, dist.samples)
                        return key
                except Exception:
                    continue

        # --- Static map fallback (existing behavior) ---
        return self._SOURCE_TO_THOMPSON_KEY.get(
            signal.source, signal.source or "unknown"
        )

    def _get_thompson_weight(self, signal: Signal, regime: str) -> float:
        """Return Thompson-blended reliability weight for a signal.

        Returns a weight in [0.5, 1.5]:
        - 1.0 = neutral (trust signal's own confidence as-is)
        - >1.0 = Thompson says this source is more reliable than average
        - <1.0 = Thompson says this source is less reliable than average

        When Thompson is not configured or has no data, returns 1.0 (neutral).
        """
        if self._thompson is None:
            return 1.0

        thompson_key = self._resolve_thompson_key(signal)

        try:
            dist = self._thompson.get_distribution(thompson_key, regime)
            observations = dist.samples  # alpha + beta - 2

            if observations < 5:
                # Thin data: blend toward 1.0 (neutral) proportional to obs count
                blend = observations / 5.0
                raw_weight = 0.5 + dist.mean  # map [0,1] -> [0.5, 1.5]
                return 1.0 * (1.0 - blend) + raw_weight * blend
            else:
                # Mature data: use Thompson posterior mean as weight
                return 0.5 + dist.mean
        except Exception:
            return 1.0

    def _compute_freshness(self, signal: Signal, domain: str) -> float:
        """Compute temporal freshness weight for a signal (Capability 5).

        Recent signals are weighted more heavily than stale ones within
        their convergence window.  Uses sqrt decay (concave curve) —
        aggressive early dropoff so yesterday's signal is noticeably
        dimmer, but a 0.3 floor preserves domain presence.

        Examples (72h global window):
          2h old  → 0.83
          24h old → 0.42
          70h old → 0.30 (floored)
        """
        now = datetime.now()
        age_hours = max(0, (now - signal.timestamp).total_seconds() / 3600)
        window = self._domain_windows.get(domain, self.convergence_window)
        window_hours = window.total_seconds() / 3600

        if window_hours <= 0:
            return 1.0

        freshness = 1.0 - (age_hours / window_hours) ** 0.5
        return max(0.3, freshness)

    def _apply_quorum_boost(
        self,
        confidence: float,
        ticker: str,
        direction: str,
    ) -> float:
        """Apply quorum contributor count as a confidence multiplier.

        When 3+ independent agents all deposit signals for the same
        ticker+direction in the QuorumSpace, that collective agreement
        boosts our confidence in the convergence.

        Multiplier schedule:
            count < 3:  1.0x (no boost)
            count == 3: 1.1x
            count == 4: 1.2x
            count >= 5: 1.3x (capped)
        """
        if self._quorum_space is None or not ticker:
            return confidence
        try:
            signal_key = f"{direction}:{ticker}"
            count = self._quorum_space.get_contributor_count(signal_key)
            if count < 3:
                return confidence
            multiplier = 1.0 + min(0.3, (count - 2) * 0.1)
            boosted = min(1.0, confidence * multiplier)
            logger.debug(
                "Quorum boost: %d contributors on %s:%s → %.2f multiplier",
                count, direction, ticker, multiplier,
            )
            return boosted
        except Exception:
            logger.debug("Quorum boost failed gracefully", exc_info=True)
            return confidence

    def _compute_confidence(
        self,
        signals: list,
        cross_domain_count: int,
    ) -> float:
        """Compute convergence confidence via Thompson-weighted geometric mean.

        The geometric mean correctly handles correlated signals without blowing
        up confidence the way additive or log-odds combination does.

        The diversity bonus uses an effective domain count that discounts
        correlated domains (e.g. institutional+macro at r=0.57 count as ~1.5
        domains, not 2). Falls back to raw count when CorrelationTracker absent.
        """
        if not signals:
            return 0.5

        regime = self._get_regime()

        log_sum = 0.0
        weight_sum = 0.0

        for sig in signals:
            c = max(0.01, min(0.99, sig.confidence))
            w = self._get_thompson_weight(sig, regime)
            log_sum += w * math.log(c)
            weight_sum += w

        geo_mean = math.exp(log_sum / weight_sum) if weight_sum > 0 else 0.5

        domain_list = sorted({sig.domain for sig in signals})
        if self._correlation_tracker:
            effective_count = self._compute_effective_domain_count(domain_list)
        else:
            effective_count = cross_domain_count

        # Cross-domain diversity bonus (multiplicative, log-saturating)
        # 1 domain=0%, 2=~8%, 3=~13%, 4=~17%, saturates near 25%
        diversity_factor = 1.0 + 0.12 * math.log1p(max(0, effective_count - 1))
        boosted = geo_mean * diversity_factor

        return min(0.95, max(0.05, boosted))

    def _compute_effective_domain_count(self, domains: List[str]) -> float:
        """Compute effective number of independent domains, accounting for correlations.

        Strongly correlated domain pairs reduce the effective count — two domains
        that move together carry less independent information than two that don't.
        """
        if not domains:
            return 0.0
        if len(domains) == 1:
            return 1.0

        counted_domains: List[str] = [domains[0]]
        effective_count: float = 1.0

        for domain in domains[1:]:
            max_abs_corr = self._max_domain_correlation(domain, counted_domains)
            if max_abs_corr > STRONG_CORRELATION_THRESHOLD:
                effective_count += 0.5
            elif max_abs_corr > MODERATE_CORRELATION_THRESHOLD:
                effective_count += 0.7
            else:
                effective_count += 1.0
            counted_domains.append(domain)

        return effective_count

    def _max_domain_correlation(self, domain_a: str, other_domains: List[str]) -> float:
        """Return the maximum absolute correlation between domain_a and any other domain.

        Looks up correlations at source level via CorrelationTracker + _DOMAIN_SOURCES.
        Returns 0.0 if no data available (treated as independent).
        """
        sources_a = self._DOMAIN_SOURCES.get(domain_a, [])
        if not sources_a:
            return 0.0

        max_corr = 0.0
        for other_domain in other_domains:
            sources_b = self._DOMAIN_SOURCES.get(other_domain, [])
            if not sources_b:
                continue
            for src_a in sources_a:
                for src_b in sources_b:
                    corr = self._correlation_tracker.get_correlation(src_a, src_b)
                    if corr is not None:
                        max_corr = max(max_corr, abs(corr))

        return max_corr

    def _apply_confidence_modifiers(
        self,
        confidence: float,
        direction: str,
        domains_seen: set,
        primary_ticker: str,
        primary_domain: str,
        combo_key: str,
    ) -> float:
        """Apply external-system confidence modifiers (catalyst, cross-asset, deception,
        economic calendar, archetype, combo Thompson).

        Each modifier is applied independently; failures are silently ignored
        so a broken optional component never blocks alert generation.

        Returns:
            Final confidence clamped to [0.05, 0.95].
        """
        # Gift 3: Catalyst Calendar modifier
        if primary_ticker and self._catalyst_calendar is not None:
            try:
                modifier = self._catalyst_calendar.compute_catalyst_modifier(
                    primary_ticker, signal_domain=primary_domain or "",
                )
                confidence = max(0.05, min(0.95, confidence * modifier))
            except Exception:
                pass

        # Gift 4: Cross-Asset Confirmation
        if primary_ticker and self._cross_asset_confirmer is not None:
            try:
                result = self._cross_asset_confirmer.check_confirmation(
                    primary_ticker, direction,
                )
                if result is not None:
                    score = getattr(result, "confirmation_score", 0.0)
                    confidence = max(0.05, min(0.95, confidence * (0.8 + 0.4 * score)))
            except Exception:
                pass

        # Gift 5: Deception detection
        if primary_ticker and self._deception_detector is not None:
            try:
                assessment = self._deception_detector.assess_signal_authenticity(primary_ticker)
                if assessment is not None and assessment.deception_probability > 0.5:
                    confidence = max(0.05, confidence * (1.0 - assessment.deception_probability))
            except Exception:
                pass

        # Economic calendar suppression — FOMC/CPI/NFP create scheduled volatility.
        if self._economic_calendar is not None:
            try:
                if self._economic_calendar.is_in_suppression_window():
                    confidence = max(0.05, confidence * 0.5)
                    logger.info(
                        "Convergence: suppression window active — confidence reduced to %.2f",
                        confidence,
                    )
            except Exception:
                pass

        # Gift 8: Pattern Archetype context
        if primary_ticker and self._pattern_archetype_engine is not None:
            try:
                matches = self._pattern_archetype_engine.scan_for_archetypes(
                    primary_ticker, signal_domains=list(domains_seen),
                )
                if matches:
                    best = max(matches, key=lambda m: m.match_score)
                    if best.match_score > 0.7:
                        confidence = min(0.95, confidence + 0.10)
            except Exception:
                pass

        # Combo Thompson — per-combination historical win rate adjustment.
        if self._thompson is not None:
            try:
                regime = self._get_regime()
                combo_dist = self._thompson.get_distribution(combo_key, regime)
                if combo_dist.samples >= 5:
                    confidence = max(0.05, min(0.95, confidence * (0.5 + combo_dist.mean)))
            except Exception:
                pass

        # Arc 2: HAVEN bio-system source suspicion — downweight flagged sources.
        if self._haven_flags:
            try:
                max_suspicion = 0.0
                for src, score in self._haven_flags.items():
                    if score > max_suspicion:
                        max_suspicion = score
                if max_suspicion > 0.5:
                    confidence = max(0.05, confidence * (1.0 - max_suspicion * 0.2))
            except Exception:
                pass

        return confidence

    def _compute_coherence_score(self) -> dict:
        """Compute agreement ratio between directional signals across ALL domains.

        Used by Capability 3 (narrative coherence scoring) to detect when the
        market is sending contradictory signals.

        Returns a dict with:
          coherence: float in [0.5, 1.0] — 1.0 = all agree, 0.5 = evenly split.
          dominant_direction: "bullish" or "bearish" (or "neutral" if no votes).
          bullish_count, bearish_count: domain vote counts.
          contradiction_details: list of (domain, direction) for minority domains.
        """
        bullish_domains = []
        bearish_domains = []

        for domain, signals in self.signals.items():
            has_bullish = any(s.direction == "bullish" for s in signals)
            has_bearish = any(s.direction == "bearish" for s in signals)
            if has_bullish:
                bullish_domains.append(domain)
            if has_bearish:
                bearish_domains.append(domain)

        total = len(bullish_domains) + len(bearish_domains)
        if total == 0:
            return {
                "coherence": 1.0,
                "dominant_direction": "neutral",
                "bullish_count": 0,
                "bearish_count": 0,
                "contradiction_details": [],
            }

        bullish_count = len(bullish_domains)
        bearish_count = len(bearish_domains)
        majority_count = max(bullish_count, bearish_count)
        coherence = majority_count / total  # range [0.5, 1.0]

        if bullish_count >= bearish_count:
            dominant = "bullish"
            minority_domains = [(d, "bearish") for d in bearish_domains]
        else:
            dominant = "bearish"
            minority_domains = [(d, "bullish") for d in bullish_domains]

        return {
            "coherence": coherence,
            "dominant_direction": dominant,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "contradiction_details": minority_domains,
        }
