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
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


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
    """

    def __init__(
        self,
        min_domains: int = 3,
        min_strength: float = 0.6,
        convergence_window_hours: int = 48,
        persistence_path: str = None
    ):
        """
        Initialize convergence alerter.

        Args:
            min_domains: Minimum different domains to trigger alert
            min_strength: Minimum average signal strength
            convergence_window_hours: How recent signals must be
            persistence_path: Path for alert history
        """
        self.min_domains = min_domains
        self.min_strength = min_strength
        self.convergence_window = timedelta(hours=convergence_window_hours)
        self.persistence_path = Path(persistence_path) if persistence_path else None

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
        metadata: dict = None
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
            confidence=confidence
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

        # Store alerts (capped to prevent unbounded growth)
        self.alerts.extend(alerts)
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]

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
        avg_confidence = sum(s.confidence for s in converging_signals) / len(converging_signals)

        # Boost confidence based on cross-domain agreement
        confidence_boost = min(0.2, 0.05 * (cross_domain_count - 1))
        final_confidence = min(0.95, avg_confidence + confidence_boost)

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
            confidence = min(0.9, 0.5 + 0.1 * len(bullish_categories) + 0.05 * bullish_strength)
            reasoning = f"{len(bullish_domains)} domains ({len(bullish_categories)} categories) bullish"
        elif len(bearish_domains) > len(bullish_domains) and len(bearish_categories) >= 2:
            recommendation = "bearish"
            confidence = min(0.9, 0.5 + 0.1 * len(bearish_categories) + 0.05 * bearish_strength)
            reasoning = f"{len(bearish_domains)} domains ({len(bearish_categories)} categories) bearish"
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

    def save(self):
        """Persist state to disk."""
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)


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
