#!/usr/bin/env python3
"""
Cross-Domain Correlation Tracker - Detects which domains move together.

CRITICAL for leading indicator detection:
- Correlation spikes BEFORE patterns become obvious
- Cross-domain correlation breaks (divergence) signal regime change
- Unusual correlations may indicate informed trading

Usage:
    from mae_core.market.intelligence.correlation_tracker import CorrelationTracker

    ct = CorrelationTracker()
    ct.record("crypto_whales", 0.8, timestamp)
    ct.record("insider_buys", 0.7, timestamp)

    corr = ct.get_correlation("crypto_whales", "insider_buys")
    anomalies = ct.detect_correlation_anomalies()
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class CorrelationPair:
    """Tracks correlation between two signals."""
    signal_a: str
    signal_b: str
    current_correlation: float = 0.0
    historical_mean: float = 0.0
    historical_std: float = 0.1
    correlation_zscore: float = 0.0
    is_anomalous: bool = False
    observation_count: int = 0
    last_updated: datetime = None


class CorrelationTracker:
    """
    Tracks cross-domain correlations between signals.

    Key insight: When normally uncorrelated domains suddenly correlate,
    someone may be trading on information not yet public.

    Example: Crypto whale movements correlating with congressional trades
    could indicate coordinated or informed activity.
    """

    def __init__(
        self,
        window_size: int = 30,
        anomaly_threshold: float = 2.5,
        min_observations: int = 30,
        persistence_path: str = None
    ):
        """
        Initialize correlation tracker.

        Args:
            window_size: Rolling window for correlation calculation
            anomaly_threshold: Z-score for flagging unusual correlations
            min_observations: Minimum data points before computing correlation
            persistence_path: Path to save/load state
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.min_observations = min_observations
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Signal history: signal_id -> deque of (timestamp, value)
        self.history: Dict[str, deque] = {}

        # Correlation pairs: (signal_a, signal_b) -> CorrelationPair
        self.correlations: Dict[Tuple[str, str], CorrelationPair] = {}

        # Historical correlations for anomaly detection
        self.correlation_history: Dict[Tuple[str, str], deque] = {}

        # Domain mappings for cross-domain analysis
        self.signal_domains: Dict[str, str] = {}

        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

    def register_signal(self, signal_id: str, domain: str):
        """
        Register a signal with its domain for cross-domain analysis.

        Domains: crypto, insider, government, sentiment, technical, etc.
        """
        self.signal_domains[signal_id] = domain
        if signal_id not in self.history:
            self.history[signal_id] = deque(maxlen=self.window_size)

    def record(
        self,
        signal_id: str,
        value: float,
        timestamp: datetime = None,
        domain: str = None
    ):
        """
        Record a signal observation.

        Args:
            signal_id: Signal identifier
            value: Normalized signal value (recommend 0-1 scale)
            timestamp: When observed
            domain: Optional domain (auto-registers signal)
        """
        timestamp = timestamp or datetime.now()

        if domain:
            self.register_signal(signal_id, domain)
        elif signal_id not in self.history:
            self.history[signal_id] = deque(maxlen=self.window_size)

        self.history[signal_id].append((timestamp, value))

    def compute_correlation(self, signal_a: str, signal_b: str) -> Optional[float]:
        """
        Compute Pearson correlation between two signals.

        Aligns observations by timestamp proximity (within 1 hour window).

        Returns:
            Correlation coefficient (-1 to 1) or None if insufficient data
        """
        if signal_a not in self.history or signal_b not in self.history:
            return None

        history_a = list(self.history[signal_a])
        history_b = list(self.history[signal_b])

        if len(history_a) < self.min_observations or len(history_b) < self.min_observations:
            return None

        # Align observations by timestamp (within 1 hour)
        aligned_a = []
        aligned_b = []

        for ts_a, val_a in history_a:
            # Find closest observation in signal_b within 1 hour
            best_match = None
            best_diff = timedelta(hours=1)

            for ts_b, val_b in history_b:
                diff = abs(ts_a - ts_b)
                if diff < best_diff:
                    best_diff = diff
                    best_match = val_b

            if best_match is not None:
                aligned_a.append(val_a)
                aligned_b.append(best_match)

        if len(aligned_a) < self.min_observations:
            return None

        # Compute Pearson correlation
        n = len(aligned_a)
        mean_a = sum(aligned_a) / n
        mean_b = sum(aligned_b) / n

        var_a = sum((x - mean_a) ** 2 for x in aligned_a)
        var_b = sum((x - mean_b) ** 2 for x in aligned_b)

        if var_a == 0 or var_b == 0:
            return 0.0

        covariance = sum((aligned_a[i] - mean_a) * (aligned_b[i] - mean_b)
                        for i in range(n))

        correlation = covariance / math.sqrt(var_a * var_b)
        return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]

    def update_correlations(self):
        """
        Update all tracked correlation pairs.

        Call this periodically to refresh correlation estimates.
        """
        signals = list(self.history.keys())

        for i, sig_a in enumerate(signals):
            for sig_b in signals[i+1:]:
                pair_key = (sig_a, sig_b) if sig_a < sig_b else (sig_b, sig_a)

                corr = self.compute_correlation(sig_a, sig_b)
                if corr is None:
                    continue

                # Initialize pair if new
                if pair_key not in self.correlations:
                    self.correlations[pair_key] = CorrelationPair(
                        signal_a=pair_key[0],
                        signal_b=pair_key[1]
                    )
                    self.correlation_history[pair_key] = deque(maxlen=self.window_size)

                pair = self.correlations[pair_key]
                pair.current_correlation = corr
                pair.last_updated = datetime.now()
                pair.observation_count += 1

                # Track history for anomaly detection
                self.correlation_history[pair_key].append(corr)

                # Update rolling stats
                history = list(self.correlation_history[pair_key])
                if len(history) >= 3:
                    pair.historical_mean = sum(history) / len(history)
                    variance = sum((c - pair.historical_mean) ** 2 for c in history) / len(history)
                    pair.historical_std = math.sqrt(variance) if variance > 0 else 0.1

                    # Compute z-score
                    if pair.historical_std > 0:
                        pair.correlation_zscore = (corr - pair.historical_mean) / pair.historical_std
                        pair.is_anomalous = abs(pair.correlation_zscore) >= self.anomaly_threshold

    def get_correlation(self, signal_a: str, signal_b: str) -> Optional[float]:
        """Get current correlation between two signals."""
        pair_key = (signal_a, signal_b) if signal_a < signal_b else (signal_b, signal_a)
        pair = self.correlations.get(pair_key)
        return pair.current_correlation if pair else None

    def get_pair_state(self, signal_a: str, signal_b: str) -> Optional[CorrelationPair]:
        """Get full correlation state for a pair."""
        pair_key = (signal_a, signal_b) if signal_a < signal_b else (signal_b, signal_a)
        return self.correlations.get(pair_key)

    def detect_correlation_anomalies(self) -> List[CorrelationPair]:
        """
        Find all pairs with anomalous correlation.

        Anomalous = correlation differs significantly from historical norm.
        """
        self.update_correlations()

        anomalies = [
            pair for pair in self.correlations.values()
            if pair.is_anomalous and pair.observation_count >= self.min_observations
        ]

        # Sort by z-score magnitude
        anomalies.sort(key=lambda p: abs(p.correlation_zscore), reverse=True)
        return anomalies

    def detect_cross_domain_anomalies(self) -> List[Tuple[CorrelationPair, str, str]]:
        """
        Find anomalous correlations specifically between different domains.

        These are the most valuable for leading indicator detection:
        normally uncorrelated domains suddenly correlating.

        Returns:
            List of (pair, domain_a, domain_b) tuples
        """
        anomalies = self.detect_correlation_anomalies()

        cross_domain = []
        for pair in anomalies:
            domain_a = self.signal_domains.get(pair.signal_a, "unknown")
            domain_b = self.signal_domains.get(pair.signal_b, "unknown")

            # Only include if different domains
            if domain_a != domain_b:
                cross_domain.append((pair, domain_a, domain_b))

        return cross_domain

    def get_correlation_matrix(self, signals: List[str] = None) -> Dict[Tuple[str, str], float]:
        """
        Get correlation matrix for specified signals (or all).

        Returns:
            Dict mapping (signal_a, signal_b) -> correlation
        """
        self.update_correlations()

        signals = signals or list(self.history.keys())
        matrix = {}

        for i, sig_a in enumerate(signals):
            for sig_b in signals[i+1:]:
                corr = self.get_correlation(sig_a, sig_b)
                if corr is not None:
                    matrix[(sig_a, sig_b)] = corr

        return matrix

    def get_most_correlated_pairs(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Get the most strongly correlated signal pairs."""
        self.update_correlations()

        pairs = [
            (p.signal_a, p.signal_b, p.current_correlation)
            for p in self.correlations.values()
            if p.observation_count >= self.min_observations
        ]

        # Sort by absolute correlation
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs[:top_n]

    def get_least_correlated_pairs(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Get the most uncorrelated (independent) signal pairs."""
        self.update_correlations()

        pairs = [
            (p.signal_a, p.signal_b, p.current_correlation)
            for p in self.correlations.values()
            if p.observation_count >= self.min_observations
        ]

        # Sort by absolute correlation (lowest first)
        pairs.sort(key=lambda x: abs(x[2]))
        return pairs[:top_n]

    def find_leading_pairs(
        self,
        correlation_threshold: float = 0.7,
        anomaly_only: bool = True
    ) -> List[Tuple[CorrelationPair, str]]:
        """
        Find pairs that may indicate leading indicator relationships.

        A leading pair shows:
        - High current correlation (signals moving together)
        - This correlation is unusual (wasn't there before)

        Returns:
            List of (pair, reason) tuples
        """
        self.update_correlations()

        leading = []
        for pair in self.correlations.values():
            if pair.observation_count < self.min_observations:
                continue

            reasons = []

            # High correlation that's anomalously high
            if abs(pair.current_correlation) >= correlation_threshold:
                if pair.correlation_zscore > self.anomaly_threshold:
                    reasons.append(f"unusual positive correlation ({pair.current_correlation:.2f}, z={pair.correlation_zscore:.2f})")
                elif pair.correlation_zscore < -self.anomaly_threshold:
                    reasons.append(f"unusual negative correlation ({pair.current_correlation:.2f}, z={pair.correlation_zscore:.2f})")
                elif not anomaly_only:
                    reasons.append(f"strong correlation ({pair.current_correlation:.2f})")

            if reasons:
                leading.append((pair, "; ".join(reasons)))

        # Sort by z-score
        leading.sort(key=lambda x: abs(x[0].correlation_zscore), reverse=True)
        return leading

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "tracked_signals": len(self.history),
            "correlation_pairs": len(self.correlations),
            "anomalous_pairs": sum(1 for p in self.correlations.values() if p.is_anomalous),
        }

    def to_dict(self) -> dict:
        """Export state for API/persistence."""
        return {
            "window_size": self.window_size,
            "anomaly_threshold": self.anomaly_threshold,
            "signal_domains": self.signal_domains,
            "correlations": {
                f"{p.signal_a}:{p.signal_b}": {
                    "signal_a": p.signal_a,
                    "signal_b": p.signal_b,
                    "current_correlation": p.current_correlation,
                    "historical_mean": p.historical_mean,
                    "historical_std": p.historical_std,
                    "correlation_zscore": p.correlation_zscore,
                    "is_anomalous": p.is_anomalous,
                    "observation_count": p.observation_count
                }
                for p in self.correlations.values()
            }
        }

    def save(self):
        """Persist state to disk."""
        if self.persistence_path:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

    def _load_state(self):
        """Load persisted state."""
        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            self.window_size = data.get("window_size", self.window_size)
            self.anomaly_threshold = data.get("anomaly_threshold", self.anomaly_threshold)
            self.signal_domains = data.get("signal_domains", {})

            for key, corr_data in data.get("correlations", {}).items():
                pair_key = (corr_data["signal_a"], corr_data["signal_b"])
                self.correlations[pair_key] = CorrelationPair(
                    signal_a=corr_data["signal_a"],
                    signal_b=corr_data["signal_b"],
                    current_correlation=corr_data.get("current_correlation", 0),
                    historical_mean=corr_data.get("historical_mean", 0),
                    historical_std=corr_data.get("historical_std", 0.1),
                    correlation_zscore=corr_data.get("correlation_zscore", 0),
                    is_anomalous=corr_data.get("is_anomalous", False),
                    observation_count=corr_data.get("observation_count", 0)
                )
                self.correlation_history[pair_key] = deque(maxlen=self.window_size)
        except Exception as e:
            logger.warning(f"Could not load correlation state: {e}")


if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta

    ct = CorrelationTracker(window_size=20, min_observations=5)

    # Register signals with domains
    ct.register_signal("crypto_whales", "crypto")
    ct.register_signal("btc_price", "crypto")
    ct.register_signal("insider_buys", "insider")
    ct.register_signal("congress_trades", "government")
    ct.register_signal("reddit_sentiment", "sentiment")

    # Simulate data - crypto signals correlated, others independent
    base_time = datetime.now() - timedelta(days=20)

    for i in range(20):
        ts = base_time + timedelta(days=i)

        # Crypto signals correlated
        crypto_base = random.uniform(0.3, 0.7)
        ct.record("crypto_whales", crypto_base + random.uniform(-0.1, 0.1), ts)
        ct.record("btc_price", crypto_base + random.uniform(-0.1, 0.1), ts)

        # Other signals independent
        ct.record("insider_buys", random.uniform(0, 1), ts)
        ct.record("congress_trades", random.uniform(0, 1), ts)
        ct.record("reddit_sentiment", random.uniform(0, 1), ts)

    # Update correlations
    ct.update_correlations()

    print("=== Most Correlated Pairs ===")
    for sig_a, sig_b, corr in ct.get_most_correlated_pairs(5):
        print(f"  {sig_a} <-> {sig_b}: {corr:.3f}")

    print("\n=== Correlation Anomalies ===")
    anomalies = ct.detect_correlation_anomalies()
    for pair in anomalies[:3]:
        print(f"  {pair.signal_a} <-> {pair.signal_b}: corr={pair.current_correlation:.3f}, z={pair.correlation_zscore:.2f}")

    print("\n=== Cross-Domain Anomalies ===")
    cross = ct.detect_cross_domain_anomalies()
    for pair, dom_a, dom_b in cross[:3]:
        print(f"  {pair.signal_a} ({dom_a}) <-> {pair.signal_b} ({dom_b}): z={pair.correlation_zscore:.2f}")
