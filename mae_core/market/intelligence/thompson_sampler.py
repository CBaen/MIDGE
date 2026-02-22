#!/usr/bin/env python3
"""
thompson_sampler.py - Exploration/Exploitation Balance via Beta Distributions

Manages Beta(α, β) distributions for each signal type.
Samples to balance trying new patterns vs exploiting proven ones.

Thompson Sampling algorithm:
1. Each signal maintains Beta(α, β) representing reliability belief
2. Sample from each distribution to get a score
3. Select signal with highest sampled score
4. After observing outcome, update: success → α += 1, failure → β += 1

This creates automatic exploration-exploitation balance without tuning.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional


# Persistence path
# Resolve data directory relative to project root
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
DISTRIBUTIONS_FILE = DATA_DIR / "thompson_distributions.json"
HISTORY_FILE = DATA_DIR / "thompson_history.jsonl"

# Default prior scale (higher = more confident in prior)
DEFAULT_PRIOR_SCALE = 2


@dataclass
class BetaDistribution:
    """Beta distribution parameters for a signal."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def samples(self) -> int:
        """Approximate number of observations."""
        return int(self.alpha + self.beta - 2)  # Subtract initial prior


@dataclass
class SamplingResult:
    """Result of a Thompson sampling selection."""
    timestamp: str
    signal_id: str
    sampled_score: float
    distribution: BetaDistribution
    regime: str


@dataclass
class UpdateResult:
    """Result of a distribution update."""
    timestamp: str
    signal_id: str
    success: bool
    regime: str
    old_alpha: float
    old_beta: float
    new_alpha: float
    new_beta: float
    old_mean: float
    new_mean: float


class ThompsonSampler:
    """
    Beta distribution manager for Thompson Sampling.

    Maintains separate distributions per signal (and optionally per regime).
    Supports seeding from existing reliability scores.
    """

    def __init__(
        self,
        persistence_path: Optional[Path] = None,
        prior_scale: float = DEFAULT_PRIOR_SCALE,
        seed_from_reliability: bool = True
    ):
        """
        Initialize Thompson Sampler.

        Args:
            persistence_path: Where to save/load distributions (default: DATA_DIR)
            prior_scale: How confident to be in prior (higher = more confident)
            seed_from_reliability: Whether to seed from existing reliability scores
        """
        self.persistence_path = persistence_path or DISTRIBUTIONS_FILE
        self.prior_scale = prior_scale

        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Load or initialize distributions
        # Format: {signal_id: {regime: {"alpha": float, "beta": float}}}
        self.distributions: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._load_distributions()

        # Seed from reliability scores if empty and requested
        if seed_from_reliability and not self.distributions:
            self._seed_from_reliability()

    def _load_distributions(self) -> None:
        """Load distributions from disk."""
        if self.persistence_path.exists():
            try:
                self.distributions = json.loads(self.persistence_path.read_text())
            except (json.JSONDecodeError, OSError):
                self.distributions = {}

    def _save_distributions(self) -> None:
        """Persist distributions to disk."""
        self.persistence_path.write_text(json.dumps(self.distributions, indent=2))

    def _seed_from_reliability(self) -> None:
        """
        Seed distributions from existing reliability scores.

        Converts reliability r to Beta(α, β) where:
        α = r × scale
        β = (1-r) × scale
        """
        try:
            from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
            SOURCE_RELIABILITY = LEARNING_CONFIG.get("source_reliability", {})

            for signal_id, reliability in SOURCE_RELIABILITY.items():
                alpha = reliability * self.prior_scale
                beta = (1 - reliability) * self.prior_scale

                self.distributions[signal_id] = {
                    "default": {"alpha": alpha, "beta": beta}
                }

            self._save_distributions()
        except ImportError:
            # No reliability scores available
            pass

    def get_distribution(
        self,
        signal_id: str,
        regime: str = "default"
    ) -> BetaDistribution:
        """
        Get the Beta distribution for a signal.

        Args:
            signal_id: The signal identifier
            regime: Market regime (default, bull, bear, sideways)

        Returns:
            BetaDistribution with current parameters
        """
        if signal_id not in self.distributions:
            # Initialize with uninformative prior Beta(1, 1)
            self.distributions[signal_id] = {}

        if regime not in self.distributions[signal_id]:
            # Initialize regime with uninformative prior
            self.distributions[signal_id][regime] = {"alpha": 1.0, "beta": 1.0}

        params = self.distributions[signal_id][regime]
        return BetaDistribution(alpha=params["alpha"], beta=params["beta"])

    def sample(self, signal_id: str, regime: str = "default") -> float:
        """
        Sample from a signal's Beta distribution.

        Args:
            signal_id: The signal identifier
            regime: Market regime

        Returns:
            Sampled value in [0, 1] representing estimated reliability
        """
        dist = self.get_distribution(signal_id, regime)
        return float(np.random.beta(dist.alpha, dist.beta))

    def update(
        self,
        signal_id: str,
        success: bool,
        regime: str = "default"
    ) -> UpdateResult:
        """
        Update a signal's distribution based on outcome.

        Args:
            signal_id: The signal identifier
            success: Whether the prediction was correct
            regime: Market regime

        Returns:
            UpdateResult with change details
        """
        dist = self.get_distribution(signal_id, regime)
        old_alpha, old_beta = dist.alpha, dist.beta
        old_mean = dist.mean

        # Bayesian update
        if success:
            new_alpha = old_alpha + 1
            new_beta = old_beta
        else:
            new_alpha = old_alpha
            new_beta = old_beta + 1

        # Store updated distribution
        self.distributions[signal_id][regime] = {
            "alpha": new_alpha,
            "beta": new_beta
        }

        new_dist = BetaDistribution(alpha=new_alpha, beta=new_beta)

        # Create result
        result = UpdateResult(
            timestamp=datetime.now().isoformat(),
            signal_id=signal_id,
            success=success,
            regime=regime,
            old_alpha=old_alpha,
            old_beta=old_beta,
            new_alpha=new_alpha,
            new_beta=new_beta,
            old_mean=old_mean,
            new_mean=new_dist.mean
        )

        # Log to history
        self._log_update(result)

        # Persist changes
        self._save_distributions()

        return result

    def _log_update(self, result: UpdateResult) -> None:
        """Append update to history file."""
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def select_top_n(
        self,
        signal_ids: List[str],
        n: int,
        regime: str = "default"
    ) -> List[Tuple[str, float]]:
        """
        Select top N signals by Thompson Sampling.

        Args:
            signal_ids: List of signal identifiers to choose from
            n: Number of signals to select
            regime: Market regime

        Returns:
            List of (signal_id, sampled_score) tuples, sorted by score descending
        """
        samples = []
        for signal_id in signal_ids:
            score = self.sample(signal_id, regime)
            samples.append((signal_id, score))

        # Sort by sampled score, descending
        samples.sort(key=lambda x: x[1], reverse=True)

        return samples[:n]

    def get_rankings(self, regime: str = "default") -> List[Tuple[str, float]]:
        """
        Get all signals ranked by mean reliability.

        Args:
            regime: Market regime

        Returns:
            List of (signal_id, mean) tuples, sorted by mean descending
        """
        rankings = []
        for signal_id in self.distributions:
            if regime in self.distributions[signal_id]:
                dist = self.get_distribution(signal_id, regime)
                rankings.append((signal_id, dist.mean))

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def get_uncertain_signals(
        self,
        regime: str = "default",
        min_variance: float = 0.001
    ) -> List[Tuple[str, float]]:
        """
        Get signals with high uncertainty (worth exploring).

        Args:
            regime: Market regime
            min_variance: Minimum variance threshold

        Returns:
            List of (signal_id, variance) tuples, sorted by variance descending
        """
        uncertain = []
        for signal_id in self.distributions:
            if regime in self.distributions[signal_id]:
                dist = self.get_distribution(signal_id, regime)
                if dist.variance >= min_variance:
                    uncertain.append((signal_id, dist.variance))

        uncertain.sort(key=lambda x: x[1], reverse=True)
        return uncertain

    def get_stats(self, regime: str = "default") -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict with signal count, average mean, exploration candidates
        """
        if not self.distributions:
            return {
                "signal_count": 0,
                "average_reliability": 0,
                "exploration_candidates": 0,
                "top_signals": [],
                "bottom_signals": []
            }

        rankings = self.get_rankings(regime)
        uncertain = self.get_uncertain_signals(regime)

        means = [r[1] for r in rankings]

        return {
            "signal_count": len(rankings),
            "average_reliability": sum(means) / len(means) if means else 0,
            "exploration_candidates": len(uncertain),
            "top_signals": rankings[:5],
            "bottom_signals": rankings[-5:] if len(rankings) >= 5 else rankings
        }

    def get_statistics(self) -> Dict:
        """Alias for HolonProxy.sense() delegation."""
        return self.get_stats()


def main():
    """Demo and test the Thompson Sampler."""
    print("Thompson Sampler Demo")
    print("=" * 50)

    sampler = ThompsonSampler()

    # Show initial stats
    stats = sampler.get_stats()
    print(f"\nInitial stats:")
    print(f"  Signals: {stats['signal_count']}")
    print(f"  Avg reliability: {stats['average_reliability']:.3f}")

    # Simulate some outcomes
    test_signals = ["sec_edgar", "reddit", "unknown"]
    outcomes = [
        ("sec_edgar", True),
        ("sec_edgar", True),
        ("sec_edgar", False),
        ("reddit", False),
        ("reddit", False),
        ("unknown", True),
    ]

    print(f"\nSimulating {len(outcomes)} outcomes...")
    for signal_id, success in outcomes:
        result = sampler.update(signal_id, success)
        print(f"  {signal_id}: {'success' if success else 'failure'} "
              f"-> mean {result.old_mean:.3f} -> {result.new_mean:.3f}")

    # Show updated rankings
    print("\nTop signals after updates:")
    for signal_id, mean in sampler.get_rankings()[:5]:
        print(f"  {signal_id}: {mean:.3f}")

    # Thompson sampling selection
    print("\nThompson sampling selection from test signals:")
    for _ in range(3):
        selected = sampler.select_top_n(test_signals, n=1)
        print(f"  Selected: {selected[0][0]} (score: {selected[0][1]:.3f})")


if __name__ == "__main__":
    main()
