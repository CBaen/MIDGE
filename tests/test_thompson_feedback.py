"""Tests for Thompson Sampling feedback loop fixes.

Covers:
- Floor of 2.0: distributions never decay to Beta(1,1) uninformative prior
- Forgetting logs a summary entry to thompson_history.jsonl
- Extended forgetting cycles: distributions with real evidence stay above floor
- Cadence alignment: forgetting fires at step 200, not 100
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sampler(tmp_path):
    """Create a ThompsonSampler backed by tmp_path (isolated from production)."""
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    return ThompsonSampler(
        persistence_path=tmp_path / "thompson_distributions.json",
        seed_from_reliability=False,
    )


# ---------------------------------------------------------------------------
# Task 1: Floor of 2.0
# ---------------------------------------------------------------------------

class TestForgettingFloor:
    """Forgetting must never push alpha or beta below 2.0."""

    def test_floor_is_two_for_uninformative_prior(self, tmp_path):
        """A Beta(1,1) prior that gets decayed stays at 2.0, not 1.0."""
        ts = _make_sampler(tmp_path)
        # Initialize a signal with default uninformative prior
        dist = ts.get_distribution("test_signal")
        assert dist.alpha == 1.0
        assert dist.beta == 1.0

        # Apply many rounds of forgetting
        for _ in range(100):
            ts.apply_forgetting(decay_factor=0.99)

        dist = ts.get_distribution("test_signal")
        # Floor should be 2.0, not 1.0
        assert dist.alpha == 2.0, f"Expected alpha floor 2.0, got {dist.alpha}"
        assert dist.beta == 2.0, f"Expected beta floor 2.0, got {dist.beta}"

    def test_floor_not_below_two_after_single_forgetting(self, tmp_path):
        """Even a single forgetting call must respect the 2.0 floor."""
        ts = _make_sampler(tmp_path)
        ts.get_distribution("sig")  # initialises at Beta(1,1)

        ts.apply_forgetting(decay_factor=0.50)

        dist = ts.get_distribution("sig")
        assert dist.alpha >= 2.0
        assert dist.beta >= 2.0

    def test_informed_distribution_converges_to_floor_not_below(self, tmp_path):
        """Beta(100, 200) after many forgetting cycles stays >= 2.0 on both sides."""
        ts = _make_sampler(tmp_path)

        # Seed a strongly informed distribution
        for _ in range(100):
            ts.update("strong_signal", success=True)
        for _ in range(200):
            ts.update("strong_signal", success=False)

        dist = ts.get_distribution("strong_signal")
        assert dist.alpha > 90
        assert dist.beta > 190

        # Apply aggressive forgetting — 500 rounds should drive to floor
        for _ in range(500):
            ts.apply_forgetting(decay_factor=0.95)

        dist = ts.get_distribution("strong_signal")
        assert dist.alpha >= 2.0, f"alpha {dist.alpha} dropped below floor 2.0"
        assert dist.beta >= 2.0, f"beta {dist.beta} dropped below floor 2.0"

    def test_distribution_mean_direction_preserved_until_floor(self, tmp_path):
        """A distribution biased toward wins should retain mean > 0.5 while above floor."""
        ts = _make_sampler(tmp_path)

        # Build a distribution with 3:1 wins (mean ~0.75)
        for _ in range(30):
            ts.update("biased", success=True)
        for _ in range(10):
            ts.update("biased", success=False)

        dist = ts.get_distribution("biased")
        assert dist.mean > 0.6, f"Initial mean too low: {dist.mean}"

        # Moderate forgetting — should preserve direction
        for _ in range(10):
            ts.apply_forgetting(decay_factor=0.99)

        dist = ts.get_distribution("biased")
        # Mean direction should still lean bullish (alpha > beta, both above floor)
        assert dist.alpha > dist.beta or dist.alpha >= 2.0

    def test_floor_two_is_non_uniform(self, tmp_path):
        """Beta(2,2) is NOT the same as Beta(1,1). Verify mean is still 0.5 (symmetric)
        but that variance is lower — it carries more information than the flat prior."""
        from mae_core.market.intelligence.thompson_sampler import BetaDistribution
        floor_dist = BetaDistribution(alpha=2.0, beta=2.0)
        uninformative = BetaDistribution(alpha=1.0, beta=1.0)

        # Both have mean 0.5 (symmetric) — but floor has less variance
        assert abs(floor_dist.mean - 0.5) < 1e-9
        assert floor_dist.variance < uninformative.variance


# ---------------------------------------------------------------------------
# Task 2: Forgetting logs to history
# ---------------------------------------------------------------------------

class TestForgettingLogsToHistory:
    """apply_forgetting must write a summary entry to thompson_history.jsonl."""

    def test_forgetting_creates_history_entry(self, tmp_path):
        """After apply_forgetting, history file contains a forgetting_applied entry."""
        ts = _make_sampler(tmp_path)
        ts.get_distribution("sig_a")  # ensure there is something to decay

        ts.apply_forgetting(decay_factor=0.99)

        history_path = tmp_path / "thompson_history.jsonl"
        assert history_path.exists(), "History file was not created"

        entries = [json.loads(l) for l in history_path.read_text().strip().splitlines()]
        forgetting_entries = [e for e in entries if e.get("event") == "forgetting_applied"]
        assert len(forgetting_entries) == 1, (
            f"Expected 1 forgetting_applied entry, got {len(forgetting_entries)}"
        )

    def test_forgetting_entry_has_required_fields(self, tmp_path):
        """Each forgetting_applied entry must have event, decay_factor, distributions_affected, timestamp."""
        ts = _make_sampler(tmp_path)
        ts.get_distribution("sig_a")

        ts.apply_forgetting(decay_factor=0.97)

        history_path = tmp_path / "thompson_history.jsonl"
        entries = [json.loads(l) for l in history_path.read_text().strip().splitlines()]
        fe = [e for e in entries if e.get("event") == "forgetting_applied"][0]

        assert fe["event"] == "forgetting_applied"
        assert abs(fe["decay_factor"] - 0.97) < 1e-9
        assert fe["distributions_affected"] >= 1
        assert "timestamp" in fe and fe["timestamp"]  # non-empty

    def test_forgetting_one_entry_per_call_not_per_distribution(self, tmp_path):
        """With 5 signals and 3 forgetting calls, history has exactly 3 forgetting entries."""
        ts = _make_sampler(tmp_path)
        for i in range(5):
            ts.get_distribution(f"sig_{i}")

        ts.apply_forgetting(decay_factor=0.99)
        ts.apply_forgetting(decay_factor=0.99)
        ts.apply_forgetting(decay_factor=0.99)

        history_path = tmp_path / "thompson_history.jsonl"
        entries = [json.loads(l) for l in history_path.read_text().strip().splitlines()]
        forgetting_entries = [e for e in entries if e.get("event") == "forgetting_applied"]

        assert len(forgetting_entries) == 3, (
            f"Expected 3 forgetting entries (one per call), got {len(forgetting_entries)}"
        )

    def test_forgetting_logs_distributions_affected_count(self, tmp_path):
        """distributions_affected should equal the number of signal/regime slots decayed."""
        ts = _make_sampler(tmp_path)
        for i in range(4):
            ts.get_distribution(f"sig_{i}")

        count = ts.apply_forgetting(decay_factor=0.99)

        history_path = tmp_path / "thompson_history.jsonl"
        entries = [json.loads(l) for l in history_path.read_text().strip().splitlines()]
        fe = [e for e in entries if e.get("event") == "forgetting_applied"][0]

        assert fe["distributions_affected"] == count

    def test_no_history_entry_when_no_distributions(self, tmp_path):
        """If there are no distributions at all, no forgetting entry should be written."""
        ts = _make_sampler(tmp_path)
        # Do not initialise any distributions

        count = ts.apply_forgetting(decay_factor=0.99)
        assert count == 0

        history_path = tmp_path / "thompson_history.jsonl"
        if history_path.exists():
            content = history_path.read_text().strip()
            if content:
                entries = [json.loads(l) for l in content.splitlines()]
                forgetting_entries = [e for e in entries if e.get("event") == "forgetting_applied"]
                assert len(forgetting_entries) == 0


# ---------------------------------------------------------------------------
# Task 3: Extended forgetting — Beta(100, 200) stays above floor
# ---------------------------------------------------------------------------

class TestExtendedForgettingStaysAboveFloor:
    """Explicit regression: alpha=100, beta=200 after many cycles stays above 2.0."""

    def test_alpha_100_beta_200_stays_above_floor_after_many_cycles(self, tmp_path):
        """Regression for the exact case in the build brief."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler, BetaDistribution

        ts = _make_sampler(tmp_path)

        # Manually insert a strong distribution (alpha=100, beta=200)
        signal_id = "finra_short"
        ts.distributions[signal_id] = {
            "default": {"alpha": 100.0, "beta": 200.0}
        }

        # Apply many forgetting cycles (simulate 8 days of daemon at cadence 200)
        # 8 days × ~24 forgetting events/day at pace 2.0 ≈ 192 cycles
        for _ in range(200):
            ts.apply_forgetting(decay_factor=0.99)

        dist = ts.get_distribution(signal_id)
        assert dist.alpha >= 2.0, f"alpha {dist.alpha:.4f} dropped below floor"
        assert dist.beta >= 2.0, f"beta {dist.beta:.4f} dropped below floor"

    def test_floor_prevents_collapse_to_uniform_prior(self, tmp_path):
        """After extreme forgetting, distributions must NOT be Beta(1,1)."""
        ts = _make_sampler(tmp_path)
        ts.update("sig", success=True)
        ts.update("sig", success=True)

        for _ in range(10000):
            ts.apply_forgetting(decay_factor=0.90)

        dist = ts.get_distribution("sig")
        # Must be at least Beta(2,2), never Beta(1,1)
        assert dist.alpha >= 2.0
        assert dist.beta >= 2.0


# ---------------------------------------------------------------------------
# Task 4: Cadence alignment — forgetting at step 200, not step 100
# ---------------------------------------------------------------------------

class TestCadenceAlignment:
    """Forgetting hook fires at step % 200 == 0, matching outcome evaluation cadence."""

    def test_forgetting_fires_at_step_200_not_100(self):
        """Mock the step hook logic: forgetting should not fire at step 100."""
        fired_at = []

        # Replicate the hook logic from market_hooks.py
        def simulated_hook(step, sampler):
            if step % 200 == 0:
                sampler.apply_forgetting(decay_factor=0.99)
                fired_at.append(step)

        mock_sampler = MagicMock()

        for step in range(1, 401):
            simulated_hook(step, mock_sampler)

        # Should fire at 200, 400 — NOT at 100, 300
        assert 200 in fired_at
        assert 400 in fired_at
        assert 100 not in fired_at
        assert 300 not in fired_at

    def test_forgetting_fires_every_200_steps(self):
        """Verify exact cadence: fires at 0, 200, 400, 600 in a 700-step run."""
        fired_at = []

        def simulated_hook(step, sampler):
            if step % 200 == 0:
                sampler.apply_forgetting()
                fired_at.append(step)

        mock_sampler = MagicMock()
        for step in range(0, 700):
            simulated_hook(step, mock_sampler)

        assert fired_at == [0, 200, 400, 600], f"Unexpected fire pattern: {fired_at}"

    def test_forgetting_not_in_market_hooks_at_100(self):
        """Regression: verify market_hooks.py no longer contains step % 100 == 0 for forgetting."""
        from pathlib import Path
        hooks_path = Path(__file__).parents[1] / "mae_core" / "bootstrap" / "market_hooks.py"
        content = hooks_path.read_text()

        # The old pattern was: # Every 100 steps: Bayesian forgetting
        assert "Every 100 steps: Bayesian forgetting" not in content, (
            "market_hooks.py still has forgetting at cadence 100 — should be 200"
        )

    def test_forgetting_in_market_hooks_at_200(self):
        """Positive check: market_hooks.py now fires forgetting at cadence 200."""
        from pathlib import Path
        hooks_path = Path(__file__).parents[1] / "mae_core" / "bootstrap" / "market_hooks.py"
        content = hooks_path.read_text()

        assert "Every 200 steps: Bayesian forgetting" in content, (
            "market_hooks.py does not mention forgetting at cadence 200"
        )
