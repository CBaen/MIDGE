#!/usr/bin/env python3
"""
Velocity Detector - Tracks rate of change in signals.

CRITICAL for leading indicator detection:
- Signals accelerating often precede major moves
- Rate of change is more predictive than absolute levels
- Velocity divergence across domains signals regime change

Usage:
    from mae_core.market.intelligence.velocity_detector import VelocityDetector

    vd = VelocityDetector()
    vd.record("insider_buys", 5, timestamp)
    vd.record("insider_buys", 8, timestamp + 1day)

    velocity = vd.get_velocity("insider_buys")
    acceleration = vd.get_acceleration("insider_buys")
    anomalies = vd.detect_velocity_anomalies()
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SignalPoint:
    """Single observation of a signal."""
    value: float
    timestamp: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class VelocityState:
    """Velocity analysis state for a signal."""
    signal_id: str
    current_velocity: float = 0.0  # First derivative (rate of change)
    current_acceleration: float = 0.0  # Second derivative
    velocity_zscore: float = 0.0  # How unusual is current velocity
    is_accelerating: bool = False
    is_anomalous: bool = False
    last_value: float = 0.0
    last_timestamp: Optional[datetime] = None

    # Historical stats for anomaly detection
    velocity_mean: float = 0.0
    velocity_std: float = 1.0
    observation_count: int = 0


class VelocityDetector:
    """
    Tracks velocity (rate of change) and acceleration of signals.

    Key insight: Leading indicators often show VELOCITY changes
    before the actual signal levels become noteworthy.

    Example: Insider buying velocity increasing from 2/week to 8/week
    is more significant than the absolute count of 8.
    """

    def __init__(
        self,
        window_size: int = 20,
        anomaly_threshold: float = 2.5,
        persistence_path: str = None
    ):
        """
        Initialize velocity detector.

        Args:
            window_size: Number of observations for rolling statistics
            anomaly_threshold: Z-score threshold for velocity anomalies
            persistence_path: Path to save/load state (optional)
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.persistence_path = Path(persistence_path) if persistence_path else None

        # Signal history: signal_id -> deque of SignalPoints
        self.history: Dict[str, deque] = {}

        # Velocity history for anomaly detection
        self.velocity_history: Dict[str, deque] = {}

        # Current state per signal
        self.states: Dict[str, VelocityState] = {}

        # Load persisted state if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

    def record(
        self,
        signal_id: str,
        value: float,
        timestamp: datetime = None,
        metadata: dict = None
    ) -> VelocityState:
        """
        Record a new signal observation and compute velocity.

        Args:
            signal_id: Unique identifier for this signal type
            value: Current signal value
            timestamp: When observed (defaults to now)
            metadata: Optional context about this observation

        Returns:
            Updated VelocityState for this signal
        """
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        # Initialize if new signal
        if signal_id not in self.history:
            self.history[signal_id] = deque(maxlen=self.window_size)
            self.velocity_history[signal_id] = deque(maxlen=self.window_size)
            self.states[signal_id] = VelocityState(signal_id=signal_id)

        point = SignalPoint(value=value, timestamp=timestamp, metadata=metadata)
        state = self.states[signal_id]

        # Compute velocity if we have prior observations
        if state.last_timestamp is not None:
            dt = (timestamp - state.last_timestamp).total_seconds() / 86400
            if dt > 0:
                # Velocity = change in value / change in time (per day)
                dv = value - state.last_value
                velocity = dv / dt

                # Compute acceleration if we have prior velocity
                if len(self.velocity_history[signal_id]) > 0:
                    prev_velocity = self.velocity_history[signal_id][-1]
                    acceleration = (velocity - prev_velocity) / dt
                    state.current_acceleration = acceleration
                    state.is_accelerating = acceleration > 0

                state.current_velocity = velocity
                self.velocity_history[signal_id].append(velocity)

                # Update rolling stats for anomaly detection
                self._update_velocity_stats(signal_id)

                # Check for anomaly
                if state.velocity_std > 0:
                    state.velocity_zscore = (velocity - state.velocity_mean) / state.velocity_std
                    state.is_anomalous = abs(state.velocity_zscore) >= self.anomaly_threshold

        # Store observation
        self.history[signal_id].append(point)
        state.last_value = value
        state.last_timestamp = timestamp
        state.observation_count += 1

        return state

    def _update_velocity_stats(self, signal_id: str):
        """Update rolling mean and std for velocity."""
        velocities = list(self.velocity_history[signal_id])
        if len(velocities) < 2:
            return

        state = self.states[signal_id]
        state.velocity_mean = sum(velocities) / len(velocities)

        variance = sum((v - state.velocity_mean) ** 2 for v in velocities) / len(velocities)
        state.velocity_std = math.sqrt(variance) if variance > 0 else 1.0

    def get_velocity(self, signal_id: str) -> Optional[float]:
        """Get current velocity for a signal."""
        if signal_id in self.states:
            return self.states[signal_id].current_velocity
        return None

    def get_acceleration(self, signal_id: str) -> Optional[float]:
        """Get current acceleration for a signal."""
        if signal_id in self.states:
            return self.states[signal_id].current_acceleration
        return None

    def get_state(self, signal_id: str) -> Optional[VelocityState]:
        """Get full velocity state for a signal."""
        return self.states.get(signal_id)

    def detect_velocity_anomalies(
        self,
        min_observations: int = 5
    ) -> List[Tuple[str, VelocityState]]:
        """
        Find all signals with anomalous velocity.

        Args:
            min_observations: Minimum observations before flagging anomalies

        Returns:
            List of (signal_id, state) tuples for anomalous signals
        """
        anomalies = []
        for signal_id, state in self.states.items():
            if state.observation_count >= min_observations and state.is_anomalous:
                anomalies.append((signal_id, state))

        # Sort by z-score magnitude (most anomalous first)
        anomalies.sort(key=lambda x: abs(x[1].velocity_zscore), reverse=True)
        return anomalies

    def detect_acceleration_shifts(
        self,
        min_observations: int = 5
    ) -> List[Tuple[str, VelocityState]]:
        """
        Find signals where acceleration direction changed.

        Acceleration shifts often precede trend changes.
        """
        shifts = []
        for signal_id, state in self.states.items():
            if state.observation_count < min_observations:
                continue

            # Check if recent acceleration differs from historical pattern
            if len(self.velocity_history[signal_id]) >= 3:
                recent = list(self.velocity_history[signal_id])[-3:]

                # Detect inflection: velocity was increasing, now decreasing (or vice versa)
                if len(recent) >= 3:
                    accel_1 = recent[1] - recent[0]
                    accel_2 = recent[2] - recent[1]

                    # Sign change = inflection point
                    if (accel_1 > 0 and accel_2 < 0) or (accel_1 < 0 and accel_2 > 0):
                        shifts.append((signal_id, state))

        return shifts

    def get_velocity_divergence(
        self,
        signal_a: str,
        signal_b: str
    ) -> Optional[float]:
        """
        Compute velocity divergence between two signals.

        Divergence (signals moving opposite directions) often
        precedes market regime changes.

        Returns:
            Divergence score (-1 to 1, negative = diverging)
        """
        state_a = self.states.get(signal_a)
        state_b = self.states.get(signal_b)

        if not state_a or not state_b:
            return None

        if state_a.velocity_std == 0 or state_b.velocity_std == 0:
            return None

        # Normalize velocities
        norm_a = state_a.current_velocity / state_a.velocity_std
        norm_b = state_b.current_velocity / state_b.velocity_std

        # Same direction = positive, opposite = negative
        if (norm_a > 0 and norm_b > 0) or (norm_a < 0 and norm_b < 0):
            return min(abs(norm_a), abs(norm_b)) / max(abs(norm_a), abs(norm_b), 0.001)
        else:
            return -min(abs(norm_a), abs(norm_b)) / max(abs(norm_a), abs(norm_b), 0.001)

    def get_leading_signals(
        self,
        lookback_hours: int = 24
    ) -> List[Tuple[str, VelocityState, str]]:
        """
        Identify signals showing leading indicator behavior.

        Leading indicators:
        - High velocity magnitude (changing fast)
        - Acceleration in same direction as velocity (trend building)
        - Anomalous velocity (unusual rate of change)

        Returns:
            List of (signal_id, state, reason) tuples
        """
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        leaders = []

        for signal_id, state in self.states.items():
            if state.last_timestamp and state.last_timestamp < cutoff:
                continue  # Skip stale signals

            if state.observation_count < 3:
                continue

            reasons = []

            # Check for anomalous velocity
            if state.is_anomalous:
                direction = "accelerating" if state.velocity_zscore > 0 else "decelerating"
                reasons.append(f"anomalous velocity ({direction}, z={state.velocity_zscore:.2f})")

            # Check for acceleration in same direction as velocity
            if state.is_accelerating and state.current_velocity > 0:
                reasons.append("positive momentum building")
            elif not state.is_accelerating and state.current_velocity < 0:
                reasons.append("negative momentum building")

            if reasons:
                leaders.append((signal_id, state, "; ".join(reasons)))

        # Sort by absolute velocity z-score
        leaders.sort(key=lambda x: abs(x[1].velocity_zscore), reverse=True)
        return leaders

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "signal_count": len(self.states),
            "anomalous_count": sum(1 for s in self.states.values() if s.is_anomalous),
        }

    def to_dict(self) -> dict:
        """Export state for persistence or API."""
        return {
            "window_size": self.window_size,
            "anomaly_threshold": self.anomaly_threshold,
            "states": {
                k: {
                    "signal_id": v.signal_id,
                    "current_velocity": v.current_velocity,
                    "current_acceleration": v.current_acceleration,
                    "velocity_zscore": v.velocity_zscore,
                    "is_accelerating": v.is_accelerating,
                    "is_anomalous": v.is_anomalous,
                    "last_value": v.last_value,
                    "velocity_mean": v.velocity_mean,
                    "velocity_std": v.velocity_std,
                    "observation_count": v.observation_count
                }
                for k, v in self.states.items()
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

            for signal_id, state_data in data.get("states", {}).items():
                self.states[signal_id] = VelocityState(
                    signal_id=state_data["signal_id"],
                    current_velocity=state_data.get("current_velocity", 0),
                    current_acceleration=state_data.get("current_acceleration", 0),
                    velocity_zscore=state_data.get("velocity_zscore", 0),
                    is_accelerating=state_data.get("is_accelerating", False),
                    is_anomalous=state_data.get("is_anomalous", False),
                    last_value=state_data.get("last_value", 0),
                    velocity_mean=state_data.get("velocity_mean", 0),
                    velocity_std=state_data.get("velocity_std", 1),
                    observation_count=state_data.get("observation_count", 0)
                )
                self.history[signal_id] = deque(maxlen=self.window_size)
                self.velocity_history[signal_id] = deque(maxlen=self.window_size)
        except Exception as e:
            logger.warning(f"Could not load velocity state: {e}")


if __name__ == "__main__":
    # Demo usage
    from datetime import datetime, timedelta

    vd = VelocityDetector(window_size=10, anomaly_threshold=2.0)

    # Simulate insider buying pattern
    base_time = datetime.now() - timedelta(days=10)

    # Normal period: 2-3 buys per day
    for i in range(5):
        vd.record("insider_buys", 2 + (i % 2), base_time + timedelta(days=i))

    # Acceleration: buys increasing rapidly
    for i in range(5, 10):
        vd.record("insider_buys", 3 + (i - 5) * 2, base_time + timedelta(days=i))

    # Check state
    state = vd.get_state("insider_buys")
    print(f"Insider buys velocity: {state.current_velocity:.4f}/sec")
    print(f"Velocity z-score: {state.velocity_zscore:.2f}")
    print(f"Is anomalous: {state.is_anomalous}")
    print(f"Is accelerating: {state.is_accelerating}")

    # Get leading signals
    leaders = vd.get_leading_signals(lookback_hours=240)
    print(f"\nLeading signals: {len(leaders)}")
    for signal_id, state, reason in leaders:
        print(f"  {signal_id}: {reason}")
