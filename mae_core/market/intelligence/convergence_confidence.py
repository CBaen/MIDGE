"""ConvergenceConfidenceMixin — signal recording, Thompson weighting, coherence scoring.

Used by ConvergenceAlerter as a mixin. Expects the following instance attributes:
    self.signals, self.convergence_window, self._domain_windows,
    self._thompson, self._regime_classifier, self._causal_engine,
    self._correlation_tracker, self._quorum_space,
    self._cached_regime, self._lag_findings,
    self._alert_counter, self._last_alert_times, self._alert_lock,
    self._min_alert_interval_hours, self.min_strength,
    self._SOURCE_TO_THOMPSON_KEY, _SWEEP_SOURCES, _DOMAIN_SOURCES
    (last 3 provided as class attributes on ConvergenceAlerter)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from mae_core.market.intelligence.convergence_models import Signal

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# Independence correction thresholds — controls how correlated domains
# are discounted in the effective domain count calculation.
# Phase 0 found macro+technical at r=0.73 — these thresholds gate that fix.
STRONG_CORRELATION_THRESHOLD = 0.5   # |r| above this → half credit (+0.5)
MODERATE_CORRELATION_THRESHOLD = 0.3  # |r| above this → partial credit (+0.7)


class ConvergenceConfidenceMixin:
    """Signal recording, Thompson weighting, coherence scoring, buffer persistence."""

    # ------------------------------------------------------------------
    # Lag findings integration (Task 1 + Task 3)
    # ------------------------------------------------------------------

    def set_lag_findings(self, findings: List) -> None:
        """Accept lag findings from LagCorrelationAnalyzer.

        Called by the step hook after every lag analysis run (every 500 steps).
        Findings shape the sequence_score on future alerts.

        Args:
            findings: List of LagFinding objects (source_a leads source_b by lag_days)
        """
        self._lag_findings = findings if findings else []
        logger.info(
            "ConvergenceAlerter: received %d lag findings for sequence scoring",
            len(self._lag_findings),
        )

    def _load_lag_findings(self) -> None:
        """Load persisted lag findings on startup so sequence scoring works immediately."""
        lag_path = _DATA_DIR / "lag_correlations.json"
        if not lag_path.exists():
            return
        try:
            from mae_core.market.intelligence.lag_correlation_analyzer import LagFinding
            raw = json.loads(lag_path.read_text(encoding="utf-8"))
            self._lag_findings = [LagFinding(**r) for r in raw]
            logger.debug(
                "ConvergenceAlerter: loaded %d lag findings from disk",
                len(self._lag_findings),
            )
        except Exception:
            logger.debug("ConvergenceAlerter: failed to load lag findings", exc_info=True)
            self._lag_findings = []

    def _build_domain_sequence(self, signals: list) -> List[str]:
        """Sort contributing signals by timestamp (earliest first) and return domain order.

        Returns a list of domain names ordered from first-to-fire to last-to-fire.
        Domains with multiple signals use the timestamp of the earliest signal.
        """
        try:
            domain_first_ts: Dict[str, datetime] = {}
            for sig in signals:
                ts = getattr(sig, "timestamp", None)
                if ts is None:
                    continue
                domain = getattr(sig, "domain", "")
                if domain and (domain not in domain_first_ts or ts < domain_first_ts[domain]):
                    domain_first_ts[domain] = ts
            if not domain_first_ts:
                return []
            return [d for d, _ in sorted(domain_first_ts.items(), key=lambda x: x[1])]
        except Exception:
            logger.debug("domain sequence build failed", exc_info=True)
            return []

    def _compute_sequence_score(self, domain_sequence: List[str]) -> float:
        """Score how well the observed domain firing order matches known lag relationships.

        For each pair (A, B) in domain_sequence where A fired before B:
        - If LagFinding says A leads B → matches → contributes a boost (+).
        - If LagFinding says B leads A → reversed → contributes a discount (-).
        - No data → neutral.

        Score formula:
            score = 1.0 + sum(matched_boosts) - sum(reversed_penalties)
        Clamped to [0.5, 1.3] so it never destroys or artificially doubles confidence.

        Returns:
            float in [0.5, 1.3]. 1.0 = neutral (no data or no matches).
        """
        if len(domain_sequence) < 2 or not self._lag_findings:
            return 1.0

        try:
            # Build a domain-level lag lookup: domain_a leads domain_b → lag_days
            # Uses _DOMAIN_SOURCES to map sources → domains.
            # Format: (leading_domain, lagging_domain) → max |correlation|
            domain_lead: Dict = {}  # (lead_domain, lag_domain) → strength

            for finding in self._lag_findings:
                src_a = finding.source_a
                src_b = finding.source_b
                corr = abs(finding.correlation)
                # Find which domains these sources belong to
                dom_a = self._source_to_domain(src_a)
                dom_b = self._source_to_domain(src_b)
                if dom_a and dom_b and dom_a != dom_b:
                    key = (dom_a, dom_b)
                    # source_a leads source_b (A fires first, B follows)
                    if corr > domain_lead.get(key, 0.0):
                        domain_lead[key] = corr

            score_delta = 0.0
            pair_count = 0

            for i, dom_a in enumerate(domain_sequence):
                for dom_b in domain_sequence[i + 1:]:
                    # dom_a fired before dom_b in this alert
                    pair_count += 1
                    forward_key = (dom_a, dom_b)   # matches lag data
                    reverse_key = (dom_b, dom_a)   # contradicts lag data

                    forward_strength = domain_lead.get(forward_key, 0.0)
                    reverse_strength = domain_lead.get(reverse_key, 0.0)

                    if forward_strength > 0.3:
                        # Alert ordering matches known lag → slight boost
                        score_delta += 0.05 * forward_strength
                    elif reverse_strength > 0.3:
                        # Alert ordering REVERSES known lag → slight discount
                        score_delta -= 0.05 * reverse_strength

            if pair_count == 0:
                return 1.0

            raw_score = 1.0 + score_delta
            return max(0.5, min(1.3, raw_score))

        except Exception:
            logger.debug("sequence score computation failed", exc_info=True)
            return 1.0

    def _source_to_domain(self, source: str) -> str:
        """Map a source name to its domain using _DOMAIN_SOURCES (reverse lookup)."""
        for domain, sources in self._DOMAIN_SOURCES.items():
            if source in sources:
                return domain
        return ""

    # ------------------------------------------------------------------
    # Signal buffer persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _json_default(obj):
        """JSON serializer for objects not handled by default encoder."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def save_signal_buffer(self) -> int:
        """Persist in-memory signal buffer and alerter state to disk.

        Saves only signals that are still within their domain convergence
        window — expired signals are not worth carrying across restarts.
        Uses atomic writes (write to .tmp then rename) for thread safety,
        matching the pattern used by ThompsonSampler.

        Returns the number of signals saved (0 if buffer was empty).
        """
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        total_in_memory = sum(len(v) for v in self.signals.values())
        logger.info(
            "Signal buffer save starting: %d signals across %d domains in memory",
            total_in_memory,
            len(self.signals),
        )

        # Build serialisable buffer — only keep live (non-expired) signals.
        buffer: Dict[str, list] = {}
        for domain, sigs in list(self.signals.items()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            live = []
            for s in sigs:
                try:
                    if s.timestamp >= cutoff:
                        live.append({
                            "signal_id": s.signal_id,
                            "strength": s.strength,
                            "domain": s.domain,
                            "direction": s.direction,
                            "timestamp": s.timestamp.isoformat(),
                            "metadata": s.metadata,
                            "velocity": s.velocity,
                            "confidence": s.confidence,
                            "source": s.source,
                        })
                except Exception:
                    logger.debug("Skipping non-serializable signal in %s", domain)
            if live:
                buffer[domain] = live

        # Atomically write signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        tmp_buf = buf_path.with_suffix(".tmp")
        try:
            tmp_buf.write_text(json.dumps(buffer, indent=2, default=self._json_default))
            os.replace(tmp_buf, buf_path)
        except Exception:
            logger.warning("Failed to save signal buffer", exc_info=True)
            if tmp_buf.exists():
                tmp_buf.unlink(missing_ok=True)
            return 0

        # Atomically write alerter state (counter + dedup times).
        state = {
            "alert_counter": self._alert_counter,
            "last_alert_times": {
                k: (v.isoformat() if isinstance(v, datetime) else str(v))
                for k, v in self._last_alert_times.items()
            },
        }
        state_path = _DATA_DIR / "alerter_state.json"
        tmp_state = state_path.with_suffix(".tmp")
        try:
            tmp_state.write_text(json.dumps(state, indent=2))
            os.replace(tmp_state, state_path)
        except Exception:
            logger.warning("Failed to save alerter state", exc_info=True)
            if tmp_state.exists():
                tmp_state.unlink(missing_ok=True)
            return 0

        total_signals = sum(len(v) for v in buffer.values())
        logger.info(
            "Signal buffer saved: %d signals across %d domains",
            total_signals,
            len(buffer),
        )
        return total_signals

    def load_signal_buffer(self) -> int:
        """Restore signal buffer and alerter state from disk.

        Signals older than their domain convergence window are discarded
        on load — only signals that are still relevant are restored.
        Missing or corrupt files are handled gracefully.

        Returns the number of signals restored (0 if none found).
        """
        now = datetime.now()

        # Restore signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        if buf_path.exists():
            try:
                raw: Dict[str, list] = json.loads(buf_path.read_text())
                loaded = 0
                for domain, entries in raw.items():
                    window = self._domain_windows.get(domain, self.convergence_window)
                    cutoff = now - window
                    for entry in entries:
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts < cutoff:
                            continue  # expired — skip
                        self.signals[domain].append(
                            Signal(
                                signal_id=entry["signal_id"],
                                strength=entry["strength"],
                                domain=entry["domain"],
                                direction=entry["direction"],
                                timestamp=ts,
                                metadata=entry.get("metadata", {}),
                                velocity=entry.get("velocity", 0.0),
                                confidence=entry.get("confidence", 0.5),
                                source=entry.get("source", ""),
                            )
                        )
                        loaded += 1
                domain_count = len(self.signals)
                logger.info(
                    "Signal buffer loaded: %d signals across %d domains",
                    loaded,
                    domain_count,
                )
            except Exception:
                logger.warning("Failed to load signal buffer — starting fresh", exc_info=True)
                loaded = 0
        else:
            logger.debug("No signal buffer file found — starting with empty buffer")
            loaded = 0

        # Restore alerter state.
        state_path = _DATA_DIR / "alerter_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self._alert_counter = int(state.get("alert_counter", 0))
                raw_times = state.get("last_alert_times", {})
                self._last_alert_times = {
                    k: datetime.fromisoformat(v) for k, v in raw_times.items()
                }
                logger.debug(
                    "Alerter state loaded: counter=%d, dedup_entries=%d",
                    self._alert_counter,
                    len(self._last_alert_times),
                )
            except Exception:
                logger.warning("Failed to load alerter state — using defaults", exc_info=True)
        else:
            logger.debug("No alerter state file found — using defaults")

        return loaded

    def record_signal(
        self,
        signal_id: str,
        strength: float,
        domain: str,
        direction: str = "neutral",
        confidence: float = 0.5,
        velocity: float = 0.0,
        timestamp: datetime = None,
        metadata: dict = None,
        source: str = "",
    ):
        """
        Record a signal observation.

        Args:
            signal_id: Unique signal identifier
            strength: Signal strength 0-1 (higher = stronger)
            domain: Domain category (insider, crypto, sentiment, etc.)
            direction: bullish, bearish, or neutral
            confidence: Reliability estimate 0-1
            velocity: Rate of change (from VelocityDetector)
            timestamp: When observed
            metadata: Additional context
            source: Original source type (e.g. "sec_form4") for Thompson lookup
        """
        # Input validation — clamp to valid ranges
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        if direction not in ("bullish", "bearish", "neutral"):
            direction = "neutral"

        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        signal = Signal(
            signal_id=signal_id,
            strength=strength,
            domain=domain,
            direction=direction,
            timestamp=timestamp,
            metadata=metadata,
            velocity=velocity,
            confidence=confidence,
            source=source,
        )

        self.signals[domain].append(signal)

        # Prune old signals
        self._prune_old_signals()

    def _prune_old_signals(self):
        """Remove signals outside the convergence window.

        Each domain may have its own extended window (self._domain_windows).
        Domains not in the dict use the global convergence_window (72h default).
        Domains whose signal list empties after pruning are removed entirely to
        keep self.signals clean.
        """
        now = datetime.now()
        for domain in list(self.signals.keys()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            self.signals[domain] = [
                s for s in self.signals[domain]
                if s.timestamp >= cutoff
            ]
            if not self.signals[domain]:
                del self.signals[domain]

    def _get_regime(self) -> str:
        """Get current market regime with 60s cache to avoid repeated calls."""
        if self._regime_classifier is None:
            return "default"
        now = time.monotonic()
        regime, cached_at = self._cached_regime
        if now - cached_at < 60.0:
            return regime
        try:
            regime = self._regime_classifier.classify()
        except Exception:
            regime = "default"
        self._cached_regime = (regime, now)
        return regime

    def _resolve_thompson_key(self, signal: Signal) -> str:
        """Resolve the Thompson distribution key for a signal.

        For sweep sources, runs a most-specific-wins cascade through
        granular backtest-derived keys (Bridge 2):
          sweep_bt:{symbol}:{direction} → sweep_bt:{symbol} →
          sweep_bt:{direction} → generic session_sweep fallback.

        For all other sources with metadata (Capability 2 — contextual Thompson):
          {source}:{role}:{sector} → {source}:{role} → {source} fallback.
          The "size" dimension (large/small) is appended when role is present:
          {source}:{role}:{size} → {source}:{role} as the final intermediate.
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

            # Build candidates most-specific → least-specific
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

        Final confidence is capped at 1.0.

        Args:
            confidence: Current confidence value before quorum adjustment.
            ticker: Instrument symbol (e.g. "AAPL", "CL=F").
            direction: "bullish" or "bearish".

        Returns:
            Adjusted confidence value in [0.0, 1.0].
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
                count,
                direction,
                ticker,
                multiplier,
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

        Replaces the additive formula (Alpha's standing dissent). The geometric
        mean correctly handles correlated signals without blowing up confidence
        the way additive or log-odds combination does.

        With thin Thompson data, weights are ~1.0 so the result approximates
        arithmetic mean. As Thompson accumulates outcomes, unreliable sources
        are down-weighted and reliable sources are up-weighted automatically.

        The diversity bonus now uses an effective domain count that discounts
        correlated domains (e.g. institutional+macro at r=0.57 count as ~1.5 domains,
        not 2). When CorrelationTracker is not available, falls back to the raw
        domain count for backward compatibility.
        """
        if not signals:
            return 0.5

        regime = self._get_regime()

        log_sum = 0.0
        weight_sum = 0.0

        for sig in signals:
            # Clamp confidence to prevent log(0)
            c = max(0.01, min(0.99, sig.confidence))
            w = self._get_thompson_weight(sig, regime)

            log_sum += w * math.log(c)
            weight_sum += w

        # Weighted geometric mean of per-signal confidences
        geo_mean = math.exp(log_sum / weight_sum) if weight_sum > 0 else 0.5

        # Compute the effective number of independent domains.
        # Correlated domains (e.g. institutional+macro r=0.57) receive partial credit.
        # Falls back to raw cross_domain_count when no CorrelationTracker is wired.
        # Sorted for deterministic effective count (order-dependent algorithm).
        domain_list = sorted({sig.domain for sig in signals})
        if self._correlation_tracker:
            effective_count = self._compute_effective_domain_count(domain_list)
        else:
            effective_count = cross_domain_count  # backward-compatible fallback

        # Cross-domain diversity bonus (multiplicative, log-saturating)
        # 1 domain=0%, 2=~8%, 3=~13%, 4=~17%, saturates near 25%
        diversity_factor = 1.0 + 0.12 * math.log1p(max(0, effective_count - 1))
        boosted = geo_mean * diversity_factor

        return min(0.95, max(0.05, boosted))

    def _compute_effective_domain_count(self, domains: List[str]) -> float:
        """Compute effective number of independent domains, accounting for correlations.

        Strongly correlated domain pairs reduce the effective count — two domains
        that move together carry less independent information than two that don't.

        Algorithm:
        - First domain always counts fully (1.0).
        - Each additional domain adds credit based on its max |correlation| with
          all previously counted domains:
            |r| > 0.5  →  +0.5  (strongly correlated — half credit)
            |r| > 0.3  →  +0.7  (moderately correlated — partial credit)
            no data or |r| <= 0.3 →  +1.0  (independent — full credit)

        Correlation is looked up at the source level via CorrelationTracker and
        _DOMAIN_SOURCES, taking the maximum |correlation| across all source pairs
        for the two domains.

        Args:
            domains: List of domain names present in the convergence (e.g. ["macro", "insider"])

        Returns:
            Effective domain count as a float (always >= 1.0 if domains is non-empty)
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
                effective_count += 0.5  # strongly correlated — half credit
            elif max_abs_corr > MODERATE_CORRELATION_THRESHOLD:
                effective_count += 0.7  # moderately correlated — partial credit
            else:
                effective_count += 1.0  # independent or no data — full credit
            counted_domains.append(domain)

        return effective_count

    def _max_domain_correlation(self, domain_a: str, other_domains: List[str]) -> float:
        """Return the maximum absolute correlation between domain_a and any domain in other_domains.

        Looks up correlations at source level using _DOMAIN_SOURCES and CorrelationTracker.
        Returns 0.0 if no correlation data is available (treated as independent).
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

    def _compute_coherence_score(self) -> dict:
        """Compute agreement ratio between directional signals across ALL domains.

        Used by Capability 3 (narrative coherence scoring) to detect when the
        market is sending contradictory signals — e.g. 5 bullish and 4 bearish
        domains simultaneously.  This does NOT look at signal strength; it is a
        simple directional vote count across all currently-held signals.

        Returns a dict with:
          coherence: float in [0.5, 1.0] — 1.0 = all agree, 0.5 = evenly split.
          dominant_direction: "bullish" or "bearish" (or "neutral" if no votes).
          bullish_count: number of domains with at least one bullish signal.
          bearish_count: number of domains with at least one bearish signal.
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

        # coherence = fraction of votes going to the majority side
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
