#!/usr/bin/env python3
"""
learning_loop.py - Self-Improvement Through Outcome Feedback

The brain that learns from its own predictions:
1. Analyze outcomes by signal source
2. Update reliability scores (Bayesian)
3. Adjust decay rates based on signal half-life
4. Log all changes for transparency
5. Report findings

This is what makes MIDGE self-evolving.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from trading.outcome_tracker import OutcomeTracker, PredictionOutcome
from trading.storage import DECAY_RATES, SOURCE_RELIABILITY


# Config files
CONFIG_DIR = Path("C:/Users/baenb/projects/MIDGE/trading/config")
HISTORY_FILE = CONFIG_DIR / "config_history.jsonl"
RELIABILITY_FILE = CONFIG_DIR / "learned_reliability.json"


@dataclass
class SignalPerformance:
    """Performance metrics for a signal type."""
    signal_type: str
    predictions: int
    correct: int
    accuracy: float
    avg_confidence: float
    avg_return: float
    calibration_error: float  # |accuracy - avg_confidence|
    old_reliability: float
    new_reliability: float
    reliability_change: float


@dataclass
class LearningCycleResult:
    """Result of a learning cycle."""
    cycle_id: str
    timestamp: str
    predictions_analyzed: int
    signal_performances: List[SignalPerformance]
    reliability_updates: Dict[str, float]
    decay_updates: Dict[str, float]
    summary: str


class LearningLoop:
    """
    Bayesian learning system for signal reliability.

    The core insight: A signal's reliability should increase when it makes
    confident predictions that turn out correct, and decrease when confident
    predictions are wrong. Low-confidence predictions should have minimal impact.
    """

    def __init__(self, learning_rate: float = 0.1):
        """
        Initialize learning loop.

        Args:
            learning_rate: How fast to update weights (0.01 = conservative, 0.2 = aggressive)
        """
        self.learning_rate = learning_rate
        self.outcome_tracker = OutcomeTracker()

        # Load current reliability scores (or use defaults)
        self.reliability_scores = self._load_reliability_scores()

        # Ensure config directory exists
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    def _load_reliability_scores(self) -> Dict[str, float]:
        """Load learned reliability scores or return defaults."""
        if RELIABILITY_FILE.exists():
            try:
                return json.loads(RELIABILITY_FILE.read_text())
            except:
                pass
        return SOURCE_RELIABILITY.copy()

    def _save_reliability_scores(self):
        """Persist learned reliability scores."""
        RELIABILITY_FILE.write_text(json.dumps(self.reliability_scores, indent=2))

    def run_learning_cycle(self, min_predictions: int = 5) -> LearningCycleResult:
        """
        Run a learning cycle: analyze outcomes and update weights.

        Args:
            min_predictions: Minimum predictions needed for a signal type to be updated

        Returns:
            LearningCycleResult with all changes
        """
        cycle_id = f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().isoformat()

        # Get signal performance from outcome tracker
        signal_stats = self.outcome_tracker.get_signal_performance()
        overall_stats = self.outcome_tracker.get_overall_stats()

        signal_performances = []
        reliability_updates = {}
        decay_updates = {}

        for signal_id, stats in signal_stats.items():
            if stats["predictions"] < min_predictions:
                continue

            # Get current reliability
            old_reliability = self.reliability_scores.get(signal_id, 0.5)

            # Calculate new reliability using Bayesian update
            accuracy = stats["accuracy"]
            avg_confidence = stats["avg_confidence"]

            # The update formula:
            # - If accuracy > confidence: signal was under-confident, boost reliability
            # - If accuracy < confidence: signal was over-confident, penalize reliability
            # - Scale by learning rate and number of predictions (more data = more weight)

            prediction_weight = min(1.0, stats["predictions"] / 50)  # Cap at 50 predictions
            update_magnitude = self.learning_rate * prediction_weight

            # Bayesian-ish update
            calibration_error = accuracy - avg_confidence
            new_reliability = old_reliability + update_magnitude * calibration_error

            # Clamp to valid range
            new_reliability = max(0.1, min(0.99, new_reliability))

            # Store performance
            perf = SignalPerformance(
                signal_type=signal_id,
                predictions=stats["predictions"],
                correct=stats["correct"],
                accuracy=accuracy,
                avg_confidence=avg_confidence,
                avg_return=stats["avg_return"],
                calibration_error=abs(calibration_error),
                old_reliability=old_reliability,
                new_reliability=new_reliability,
                reliability_change=new_reliability - old_reliability
            )
            signal_performances.append(perf)

            # Update reliability if significant change
            if abs(new_reliability - old_reliability) > 0.01:
                self.reliability_scores[signal_id] = new_reliability
                reliability_updates[signal_id] = new_reliability

        # Potentially adjust decay rates based on signal half-life observed
        # (This is more complex - for now, just log if signals are being effective longer/shorter than expected)

        # Save updated reliability scores
        if reliability_updates:
            self._save_reliability_scores()

        # Generate summary
        summary = self._generate_summary(overall_stats, signal_performances)

        # Create result
        result = LearningCycleResult(
            cycle_id=cycle_id,
            timestamp=timestamp,
            predictions_analyzed=overall_stats["total_predictions"],
            signal_performances=signal_performances,
            reliability_updates=reliability_updates,
            decay_updates=decay_updates,
            summary=summary
        )

        # Log to history
        self._log_cycle(result)

        return result

    def _generate_summary(self,
                         overall_stats: Dict,
                         performances: List[SignalPerformance]) -> str:
        """Generate human-readable summary of learning cycle."""
        lines = []

        lines.append(f"Analyzed {overall_stats['total_predictions']} predictions")
        lines.append(f"Overall accuracy: {overall_stats['accuracy']:.1%}")

        if performances:
            # Best performing signals
            sorted_by_accuracy = sorted(performances, key=lambda x: x.accuracy, reverse=True)
            best = sorted_by_accuracy[0]
            lines.append(f"Best signal: {best.signal_type} ({best.accuracy:.1%} accuracy)")

            # Most improved
            sorted_by_change = sorted(performances, key=lambda x: x.reliability_change, reverse=True)
            most_improved = sorted_by_change[0]
            if most_improved.reliability_change > 0:
                lines.append(
                    f"Most improved: {most_improved.signal_type} "
                    f"(+{most_improved.reliability_change:.2f} reliability)"
                )

            # Needs attention (high calibration error)
            needs_attention = [p for p in performances if p.calibration_error > 0.2]
            if needs_attention:
                lines.append(
                    f"Needs calibration: {', '.join(p.signal_type for p in needs_attention[:3])}"
                )

        return " | ".join(lines)

    def _log_cycle(self, result: LearningCycleResult):
        """Log learning cycle to history file."""
        # Convert to dict, handling nested dataclasses
        result_dict = {
            "cycle_id": result.cycle_id,
            "timestamp": result.timestamp,
            "predictions_analyzed": result.predictions_analyzed,
            "signal_performances": [asdict(p) for p in result.signal_performances],
            "reliability_updates": result.reliability_updates,
            "decay_updates": result.decay_updates,
            "summary": result.summary
        }

        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(result_dict) + "\n")

    def get_signal_reliability(self, signal_type: str) -> float:
        """Get current reliability score for a signal type."""
        return self.reliability_scores.get(signal_type, 0.5)

    def get_all_reliabilities(self) -> Dict[str, float]:
        """Get all current reliability scores."""
        return self.reliability_scores.copy()

    def get_learning_history(self, limit: int = 10) -> List[Dict]:
        """Get recent learning cycle results."""
        if not HISTORY_FILE.exists():
            return []

        cycles = []
        for line in HISTORY_FILE.read_text().strip().split("\n"):
            if line:
                try:
                    cycles.append(json.loads(line))
                except:
                    continue

        return cycles[-limit:]

    def get_reliability_trend(self, signal_type: str) -> List[Tuple[str, float]]:
        """Get historical reliability values for a signal type."""
        history = self.get_learning_history(limit=100)
        trend = []

        for cycle in history:
            if signal_type in cycle.get("reliability_updates", {}):
                trend.append((
                    cycle["timestamp"],
                    cycle["reliability_updates"][signal_type]
                ))

        return trend

    def explain_signal_performance(self, signal_type: str) -> str:
        """
        Generate a plain-language explanation of a signal's performance.

        For Guiding Light's understanding.
        """
        reliability = self.get_signal_reliability(signal_type)
        signal_stats = self.outcome_tracker.get_signal_performance().get(signal_type, {})

        if not signal_stats:
            return f"No data yet for {signal_type}. Keep tracking to build confidence."

        accuracy = signal_stats.get("accuracy", 0)
        predictions = signal_stats.get("predictions", 0)
        avg_return = signal_stats.get("avg_return", 0)

        # Build explanation
        explanation = []
        explanation.append(f"Signal: {signal_type}")
        explanation.append(f"Reliability score: {reliability:.2f} (0-1 scale)")
        explanation.append(f"Based on {predictions} predictions")
        explanation.append(f"Accuracy: {accuracy:.1%}")
        explanation.append(f"Average return: {avg_return:+.1f}%")

        # Add interpretation
        if accuracy >= 0.7:
            explanation.append("This signal has been reliable. Trust it more.")
        elif accuracy >= 0.5:
            explanation.append("This signal is about average. Use with other confirmation.")
        else:
            explanation.append("This signal has been underperforming. Be cautious.")

        return "\n".join(explanation)


def run_weekly_review() -> LearningCycleResult:
    """
    Run the weekly self-improvement cycle.

    This is meant to be called periodically (e.g., every Sunday)
    to review the week's predictions and update weights.
    """
    loop = LearningLoop(learning_rate=0.1)
    result = loop.run_learning_cycle(min_predictions=3)

    print(f"Learning cycle complete: {result.cycle_id}")
    print(f"Predictions analyzed: {result.predictions_analyzed}")
    print(f"Summary: {result.summary}")

    if result.reliability_updates:
        print("\nReliability updates:")
        for signal, new_score in result.reliability_updates.items():
            print(f"  {signal}: {new_score:.3f}")

    return result


def get_signal_health() -> Dict[str, str]:
    """
    Get health status for all signal types.

    Returns dict mapping signal_type -> health status
    """
    loop = LearningLoop()
    signal_stats = loop.outcome_tracker.get_signal_performance()

    health = {}
    for signal_id, stats in signal_stats.items():
        if stats["predictions"] < 3:
            health[signal_id] = "INSUFFICIENT_DATA"
        elif stats["accuracy"] >= 0.7:
            health[signal_id] = "HEALTHY"
        elif stats["accuracy"] >= 0.5:
            health[signal_id] = "MODERATE"
        else:
            health[signal_id] = "NEEDS_ATTENTION"

    return health


if __name__ == "__main__":
    print("MIDGE Learning Loop - Self-Improvement System")
    print("=" * 50)

    loop = LearningLoop()

    # Show current reliability scores
    print("\nCurrent Signal Reliability Scores:")
    for signal, score in loop.get_all_reliabilities().items():
        print(f"  {signal}: {score:.3f}")

    # Run a learning cycle
    print("\n" + "=" * 50)
    print("Running learning cycle...")
    result = run_weekly_review()

    # Show history
    print("\n" + "=" * 50)
    print("Recent Learning History:")
    history = loop.get_learning_history(limit=5)
    for cycle in history:
        print(f"  {cycle['timestamp']}: {cycle['summary']}")
