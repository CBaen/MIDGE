#!/usr/bin/env python3
"""
Convergence Alerter - Generates alerts when patterns converge across domains.

CORE OUTPUT for trading decisions:
- Detects when multiple signals from different domains align
- "crypto whale + congress trade + hiring surge = actionable signal"
- Combines velocity, correlation, and confidence into single alert

Usage:
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    alerter = ConvergenceAlerter()
    alerter.record_signal("insider_buys", 0.8, "insider", direction="bullish")
    alerter.record_signal("crypto_whales", 0.7, "crypto", direction="bullish")

    alerts = alerter.check_convergence()
    for alert in alerts:
        print(alert.summary)
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# Discovery log path — same resolution as thompson_sampler.py
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DISCOVERY_LOG = _DATA_DIR / "discovery_log.jsonl"


@dataclass
class Signal:
    """Single signal observation."""
    signal_id: str
    strength: float  # 0-1 normalized
    domain: str
    direction: str  # bullish, bearish, neutral
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    velocity: float = 0.0  # Rate of change
    confidence: float = 0.5  # Reliability estimate
    source: str = ""  # Original source type for Thompson key lookup


@dataclass
class ConvergenceAlert:
    """Alert generated when multiple domains converge."""
    alert_id: str
    timestamp: datetime
    direction: str  # bullish, bearish
    strength: float  # Overall convergence strength 0-1
    confidence: float  # Reliability estimate 0-1
    domains_converging: List[str]
    signals: List[Signal]
    cross_domain_count: int
    summary: str
    urgency: str  # immediate, hours, days

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "strength": round(self.strength, 3),
            "confidence": round(self.confidence, 3),
            "domains": self.domains_converging,
            "signal_count": len(self.signals),
            "cross_domain_count": self.cross_domain_count,
            "summary": self.summary,
            "urgency": self.urgency
        }


class ConvergenceAlerter:
    """
    Detects convergence across domains and generates actionable alerts.

    Key insight: Single-domain signals are noisy. When 3+ domains
    from different categories all point the same direction, that's
    a much stronger signal.

    Example convergence:
    - Insider domain: Executives buying (bullish)
    - Crypto domain: Whales accumulating (bullish)
    - Government domain: Contract awarded (bullish)
    → CONVERGENCE ALERT: 3 domains bullish = high confidence

    Confidence is computed via Thompson-weighted geometric mean when
    a ThompsonSampler is provided. This replaces the original additive
    formula (Alpha's standing dissent — see deliverable.md).
    """

    # Maps MarketSignal.source values -> ThompsonSampler distribution keys.
    # signal.py uses descriptive source names, Thompson uses shorter keys
    # from learning_config.py source_reliability dict.
    _SOURCE_TO_THOMPSON_KEY = {
        "sec_form4": "sec_form4",
        "sec_form8k": "sec_form8k",
        "sec_efts": "sec_efts",
        "congressional": "congressional",
        "senate": "senate",
        "insider_cluster": "insider_cluster",
        "politician_correlation": "congressional",
        "contract_award": "contract_award",
        "contract_prediction": "contract_prediction",
        "hiring_tracker": "hiring_tracker",
        "sam_gov": "sam_gov",
        "social_sentiment": "social_sentiment",
        "finra_short": "finra_short",
        "finnhub_news": "finnhub_news",
        "finnhub_earnings": "finnhub_earnings",
        "fred_macro": "fred_macro",
        "session_sweep": "session_sweep",
    }

    def __init__(
        self,
        min_domains: int = 3,
        min_strength: float = 0.6,
        convergence_window_hours: int = 72,
        persistence_path: str = None,
        thompson_sampler=None,
        regime_classifier=None,
    ):
        """
        Initialize convergence alerter.

        Args:
            min_domains: Minimum different domains to trigger alert
            min_strength: Minimum average signal strength
            convergence_window_hours: How recent signals must be
            persistence_path: Path for alert history
            thompson_sampler: Optional ThompsonSampler for reliability weights
            regime_classifier: Optional RegimeClassifier for regime-aware Thompson queries
        """
        self.min_domains = min_domains
        self.min_strength = min_strength
        self.convergence_window = timedelta(hours=convergence_window_hours)
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._thompson = thompson_sampler
        self._regime_classifier = regime_classifier
        self._cached_regime = ("default", 0.0)  # (regime_str, timestamp)

        # Recent signals by domain
        self.signals: Dict[str, List[Signal]] = defaultdict(list)

        # Alert history
        self.alerts: List[ConvergenceAlert] = []

        # Domain categories for cross-domain verification
        self.domain_categories = {
            "insider": "behavioral",
            "congress": "behavioral",
            "crypto": "market",
            "technical": "market",
            "volume": "market",
            "sentiment": "social",
            "reddit": "social",
            "news": "information",
            "events": "information",
            "fundamentals": "financial",
            "macro": "financial",
            "government": "institutional",
            "contracts": "institutional",
            "institutional_synthesis": "institutional"
        }

        self._alert_counter = 0

        # Alert deduplication (defense in depth — step hook also deduplicates)
        self._last_alert_direction = None
        self._last_alert_time = None
        self._min_alert_interval_hours = 4.0

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
        """Remove signals outside the convergence window."""
        cutoff = datetime.now() - self.convergence_window

        for domain in self.signals:
            self.signals[domain] = [
                s for s in self.signals[domain]
                if s.timestamp >= cutoff
            ]

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

        thompson_key = self._SOURCE_TO_THOMPSON_KEY.get(
            signal.source, signal.source or "unknown"
        )

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

        # Cross-domain diversity bonus (multiplicative, log-saturating)
        # 1 domain=0%, 2=~8%, 3=~13%, 4=~17%, saturates near 25%
        diversity_factor = 1.0 + 0.12 * math.log1p(max(0, cross_domain_count - 1))
        boosted = geo_mean * diversity_factor

        return min(0.95, max(0.05, boosted))

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

        alerts = []

        # Check bullish convergence
        if direction_filter in (None, "bullish"):
            alert = self._check_direction_convergence("bullish")
            if alert:
                alerts.append(alert)

        # Check bearish convergence
        if direction_filter in (None, "bearish"):
            alert = self._check_direction_convergence("bearish")
            if alert:
                alerts.append(alert)

        # Alert deduplication — suppress re-alert within interval
        now = datetime.now()
        filtered = []
        for alert in alerts:
            direction = alert.direction if hasattr(alert, "direction") else "neutral"
            if (self._last_alert_direction == direction
                    and self._last_alert_time is not None
                    and (now - self._last_alert_time).total_seconds() / 3600
                    < self._min_alert_interval_hours):
                continue  # Suppress — same condition, too recent
            filtered.append(alert)
            self._last_alert_direction = direction
            self._last_alert_time = now
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

    def _check_direction_convergence(self, direction: str) -> Optional[ConvergenceAlert]:
        """Check for convergence in a specific direction."""
        converging_signals = []
        domains_seen = set()
        categories_seen = set()

        for domain, signals in self.signals.items():
            # Get recent signals matching direction
            matching = [
                s for s in signals
                if s.direction == direction and s.strength >= self.min_strength
            ]

            if matching:
                # Take strongest recent signal from this domain
                strongest = max(matching, key=lambda s: s.strength)
                converging_signals.append(strongest)
                domains_seen.add(domain)

                # Track category
                category = self.domain_categories.get(domain, domain)
                categories_seen.add(category)

        # Need minimum domains
        if len(domains_seen) < self.min_domains:
            return None

        # Calculate cross-domain count (different categories)
        cross_domain_count = len(categories_seen)

        # Calculate overall strength and confidence
        avg_strength = sum(s.strength for s in converging_signals) / len(converging_signals)
        final_confidence = self._compute_confidence(converging_signals, cross_domain_count)

        # Determine urgency based on velocity
        avg_velocity = sum(abs(s.velocity) for s in converging_signals) / len(converging_signals)
        if avg_velocity > 0.1:
            urgency = "immediate"
        elif avg_velocity > 0.05:
            urgency = "hours"
        else:
            urgency = "days"

        # Generate summary
        domain_list = ", ".join(sorted(domains_seen))
        summary = (
            f"{direction.upper()}: {len(domains_seen)} domains converging "
            f"({domain_list}) | strength={avg_strength:.2f}, "
            f"confidence={final_confidence:.2f}, urgency={urgency}"
        )

        # Create alert
        self._alert_counter += 1
        alert = ConvergenceAlert(
            alert_id=f"CONV-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
            timestamp=datetime.now(),
            direction=direction,
            strength=avg_strength,
            confidence=final_confidence,
            domains_converging=sorted(domains_seen),
            signals=converging_signals,
            cross_domain_count=cross_domain_count,
            summary=summary,
            urgency=urgency
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

                avg_velocity = sum(abs(s.velocity) for s in converging) / len(converging)
                if avg_velocity > 0.1:
                    urgency = "immediate"
                elif avg_velocity > 0.05:
                    urgency = "hours"
                else:
                    urgency = "days"

                domain_list = ", ".join(sorted(domains_seen))
                summary = (
                    f"TICKER {ticker} {direction.upper()}: {len(domains_seen)} domains "
                    f"({domain_list}) | strength={avg_strength:.2f}, "
                    f"confidence={final_confidence:.2f}, urgency={urgency}"
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
                    urgency=urgency
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


def read_discoveries(
    max_entries: int = 100,
    min_strength: float = 0.0,
    direction: str = None,
) -> List[dict]:
    """
    Read recent discoveries from discovery_log.jsonl.

    Useful for:
    - Surfacing novel patterns to a dashboard
    - Seeding Thompson distributions with discovered signal combinations
    - Auditing what convergence patterns MIDGE has detected

    Args:
        max_entries: Maximum entries to return (most recent first)
        min_strength: Only return discoveries above this strength
        direction: Filter by "bullish" or "bearish" (None = all)

    Returns:
        List of discovery dicts, most recent first
    """
    if not _DISCOVERY_LOG.exists():
        return []

    entries = []
    try:
        with open(_DISCOVERY_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("strength", 0) < min_strength:
                        continue
                    if direction and entry.get("direction") != direction:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.debug("Failed to read discovery log: %s", e)

    # Most recent first, capped
    return entries[-max_entries:][::-1]


if __name__ == "__main__":
    from datetime import datetime, timedelta

    alerter = ConvergenceAlerter(min_domains=2, min_strength=0.5)

    # Simulate signals from multiple domains
    now = datetime.now()

    # Bullish signals from different domains
    alerter.record_signal("exec_buy_1", 0.8, "insider", "bullish", confidence=0.7)
    alerter.record_signal("whale_move_1", 0.75, "crypto", "bullish", confidence=0.65)
    alerter.record_signal("contract_award", 0.7, "government", "bullish", confidence=0.6)
    alerter.record_signal("reddit_hype", 0.6, "sentiment", "bullish", confidence=0.5)

    # One bearish signal
    alerter.record_signal("tech_indicator", 0.65, "technical", "bearish", confidence=0.55)

    print("=== Domain Status ===")
    for domain, status in alerter.get_domain_status().items():
        print(f"  {domain}: {status['direction']} (strength={status['strength']})")

    print("\n=== Convergence Check ===")
    alerts = alerter.check_convergence()
    for alert in alerts:
        print(f"  {alert.summary}")

    print("\n=== Actionable Summary ===")
    summary = alerter.get_actionable_summary()
    print(f"  Recommendation: {summary['recommendation']}")
    print(f"  Confidence: {summary['confidence']}")
    print(f"  Reasoning: {summary['reasoning']}")

    print("\n=== Convergence Matrix ===")
    matrix = alerter.get_convergence_matrix()
    domains = list(matrix.keys())
    print("       " + " ".join(f"{d[:6]:>7}" for d in domains))
    for d1 in domains:
        row = " ".join(f"{matrix[d1].get(d2, 0):>7}" for d2 in domains)
        print(f"{d1[:6]:>6} {row}")
