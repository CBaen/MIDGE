"""test_foundation_fixes.py — Crucible Round 1: Foundation Fix Verification

Tests that verify Forge's Round 1 changes are correctly applied:
  1. Thompson Sampler thread safety (lock acquisition on mutation paths)
  2. Atomic writes (tmp+replace pattern on all JSON persistence)
  3. Source reliability defaults (calibrated to observed accuracy, not aspirational)
  4. Data purity (no test fixture keys in production data files)

These tests are written to PASS after Forge's changes. They will FAIL
against the pre-fix codebase — that is their purpose.
"""

import json
import re
import threading
import concurrent.futures
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Section 1: Thompson Sampler Thread Safety
# ---------------------------------------------------------------------------

class TestThompsonSamplerThreadSafety:
    """Verify the lock is correctly acquired on all distribution-mutation paths."""

    def test_update_acquires_lock(self, tmp_path):
        """update() must acquire the lock before mutating and release after saving."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        dist_file = tmp_path / "thompson_distributions.json"
        sampler = ThompsonSampler(
            persistence_path=dist_file,
            seed_from_reliability=False,
        )

        # Seed one distribution so update() has something to mutate
        sampler.distributions["test_signal"] = {"default": {"alpha": 1.0, "beta": 1.0}}

        acquire_calls = []
        release_calls = []
        real_lock = sampler._lock

        class TrackingLock:
            """Proxy that records acquire/release calls in order."""
            def acquire(self, *args, **kwargs):
                acquire_calls.append("acquire")
                return real_lock.acquire(*args, **kwargs)

            def release(self):
                release_calls.append("release")
                return real_lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        sampler._lock = TrackingLock()
        sampler.update("test_signal", success=True, regime="default")

        assert len(acquire_calls) >= 1, "Lock must be acquired at least once during update()"
        assert len(release_calls) >= 1, "Lock must be released after update() completes"

    def test_concurrent_updates_no_corruption(self, tmp_path):
        """10 concurrent update() calls must each add exactly 1 to alpha or beta."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        dist_file = tmp_path / "thompson_distributions.json"
        sampler = ThompsonSampler(
            persistence_path=dist_file,
            seed_from_reliability=False,
        )

        signal_id = "concurrent_test"
        sampler.distributions[signal_id] = {"default": {"alpha": 1.0, "beta": 1.0}}
        initial_sum = 1.0 + 1.0  # alpha + beta = 2.0

        def do_update(i):
            # Alternate success/failure to exercise both branches
            sampler.update(signal_id, success=(i % 2 == 0), regime="default")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(do_update, i) for i in range(10)]
            concurrent.futures.wait(futures)
            # Propagate any exceptions from workers
            for f in futures:
                f.result()

        final = sampler.distributions[signal_id]["default"]
        final_sum = final["alpha"] + final["beta"]

        assert final_sum == pytest.approx(initial_sum + 10, abs=1e-9), (
            f"Expected alpha+beta={initial_sum + 10} after 10 updates, "
            f"got {final_sum} (corruption detected)"
        )

    def test_save_distributions_locked_exists(self, tmp_path):
        """ThompsonSampler must expose _save_distributions_locked inner/helper method.

        Forge adds this as the lock-holding inner function so the outer
        _save_distributions() can acquire the lock once and call the inner
        function to do the actual disk write. Its presence confirms the
        refactor was applied.
        """
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        dist_file = tmp_path / "thompson_distributions.json"
        sampler = ThompsonSampler(
            persistence_path=dist_file,
            seed_from_reliability=False,
        )

        assert hasattr(sampler, "_save_distributions_locked"), (
            "ThompsonSampler must have _save_distributions_locked method "
            "after Forge's thread-safety refactor"
        )
        assert callable(sampler._save_distributions_locked), (
            "_save_distributions_locked must be callable"
        )

    def test_apply_forgetting_under_lock(self, tmp_path):
        """apply_forgetting() must acquire the lock before mutating distributions."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        dist_file = tmp_path / "thompson_distributions.json"
        sampler = ThompsonSampler(
            persistence_path=dist_file,
            seed_from_reliability=False,
        )

        sampler.distributions["sig_a"] = {"default": {"alpha": 5.0, "beta": 3.0}}
        sampler.distributions["sig_b"] = {"default": {"alpha": 2.0, "beta": 8.0}}

        acquire_calls = []
        release_calls = []
        real_lock = sampler._lock

        class TrackingLock:
            def acquire(self, *args, **kwargs):
                acquire_calls.append("acquire")
                return real_lock.acquire(*args, **kwargs)

            def release(self):
                release_calls.append("release")
                return real_lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        sampler._lock = TrackingLock()
        count = sampler.apply_forgetting(decay_factor=0.95)

        assert count == 2, f"Expected 2 distributions decayed, got {count}"
        assert len(acquire_calls) >= 1, (
            "apply_forgetting() must acquire the lock before mutating distributions"
        )
        assert len(release_calls) >= 1, (
            "apply_forgetting() must release the lock after finishing"
        )


# ---------------------------------------------------------------------------
# Section 2: Atomic Writes (tmp + os.replace pattern)
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    """Verify all JSON persistence paths use atomic tmp+replace writes."""

    def test_learning_config_atomic_write(self, tmp_path):
        """save_snapshot() failure on second call must not corrupt the first file."""
        from mae_core.market.intelligence.learning_config import save_snapshot

        snapshot_path = tmp_path / "config_snapshot.json"

        # First call must succeed and produce a valid file
        save_snapshot(path=snapshot_path)
        assert snapshot_path.exists(), "First save_snapshot() must create the file"
        content_after_first = snapshot_path.read_text()
        parsed_first = json.loads(content_after_first)  # Must be valid JSON
        assert "version" in parsed_first

        # Simulate a failure on os.replace during the second call.
        # The tmp file write itself succeeds; the atomic rename fails.
        # The original file must remain intact.
        original_replace = __import__("os").replace
        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated rename failure")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            # Should not raise — save_snapshot() swallows errors
            save_snapshot(path=snapshot_path)

        # Original file must still be valid
        content_after_failure = snapshot_path.read_text()
        parsed_after = json.loads(content_after_failure)
        assert parsed_after == parsed_first, (
            "After a failed second save, the file must equal the first good save. "
            "A non-atomic write would have truncated or corrupted it."
        )

    def test_hypothesis_engine_atomic_write(self, tmp_path):
        """_save_retirement_window() must survive an os.replace failure gracefully."""
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine

        data_dir = tmp_path / "hyp_data"
        data_dir.mkdir()

        registry = HypothesisRegistry(data_dir=data_dir)
        generator = HypothesisGenerator(registry=registry)
        validator = HypothesisValidator(data_dir=data_dir)
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=validator,
        )

        # Seed the retirement window with some data
        engine._retirement_window = ["promoted", "retired", "promoted"]
        engine._meta_promoted_total = 2
        engine._meta_retired_after_active = 1

        # First save must succeed
        engine._save_retirement_window()
        rw_path = engine._retirement_window_path
        assert rw_path.exists(), "_save_retirement_window() must create the file"
        first_content = json.loads(rw_path.read_text())
        assert first_content["meta_promoted_total"] == 2

        # Simulate os.replace failing on the second save
        original_replace = __import__("os").replace
        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated rename failure")
            return original_replace(src, dst)

        engine._retirement_window.append("promoted")
        engine._meta_promoted_total = 3

        with patch("os.replace", side_effect=failing_replace):
            engine._save_retirement_window()  # Must not raise

        # Original file must still be valid and unchanged
        content_after = json.loads(rw_path.read_text())
        assert content_after["meta_promoted_total"] == 2, (
            "After failed atomic write, file must still contain the first good snapshot"
        )

    def test_hypothesis_generator_atomic_write(self, tmp_path):
        """save_pair_outcomes() must survive an os.replace failure gracefully."""
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator

        data_dir = tmp_path / "gen_data"
        data_dir.mkdir()

        registry = HypothesisRegistry(data_dir=data_dir)
        generator = HypothesisGenerator(registry=registry)

        # Seed pair outcomes
        generator._pair_outcomes = {
            ("sec_form4", "finnhub_earnings"): {"promoted": 3, "retired": 0}
        }

        # First save must succeed
        generator.save_pair_outcomes()
        po_path = generator._pair_outcomes_path
        assert po_path.exists(), "save_pair_outcomes() must create the file"
        first_data = json.loads(po_path.read_text())
        assert "sec_form4|finnhub_earnings" in first_data

        # Simulate os.replace failure on second save
        original_replace = __import__("os").replace
        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated rename failure")
            return original_replace(src, dst)

        generator._pair_outcomes[("sec_form4", "finnhub_earnings")]["promoted"] = 99

        with patch("os.replace", side_effect=failing_replace):
            generator.save_pair_outcomes()  # Must not raise

        # Original file must be intact
        data_after = json.loads(po_path.read_text())
        assert data_after["sec_form4|finnhub_earnings"]["promoted"] == 3, (
            "After failed atomic write, file must still contain the first good snapshot"
        )

    def test_step_timer_atomic_write(self, tmp_path):
        """save_session_stats() must survive an os.replace failure gracefully."""
        from mae_core.market.step_timer import StepTimer

        timer = StepTimer()

        # Inject some timing data
        timer._timings["convergence_check"].append(0.001)
        timer._timings["convergence_check"].append(0.002)

        stats_path = tmp_path / "step_timer_stats.json"

        # First save must succeed
        timer.save_session_stats(stats_path)
        assert stats_path.exists(), "save_session_stats() must create the file"
        first_data = json.loads(stats_path.read_text())
        assert "stats" in first_data
        assert "convergence_check" in first_data["stats"]

        # Simulate os.replace failure on second save
        original_replace = __import__("os").replace
        call_count = {"n": 0}

        def failing_replace(src, dst):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise OSError("simulated rename failure")
            return original_replace(src, dst)

        timer._timings["new_operation"].append(0.050)

        with patch("os.replace", side_effect=failing_replace):
            timer.save_session_stats(stats_path)  # Must not raise

        # Original file must be intact — no "new_operation" key
        data_after = json.loads(stats_path.read_text())
        assert "new_operation" not in data_after["stats"], (
            "After failed atomic write, file must still contain the first good snapshot. "
            "new_operation must not appear since that save failed."
        )


# ---------------------------------------------------------------------------
# Section 3: Source Reliability Defaults
# ---------------------------------------------------------------------------

class TestSourceReliabilityDefaults:
    """Verify source_reliability values reflect observed accuracy, not aspirational priors.

    Forge calibrates these from the Thompson distribution means after replaying
    12,544 outcomes. Congressional trades empirically hit ~16-20%, not 75%.
    finra_short and sec_form4 both empirically hit ~35-36%.
    """

    def _get_reliability(self):
        """Fresh import to avoid module-level caching from other tests."""
        import importlib
        import mae_core.market.intelligence.learning_config as lc
        importlib.reload(lc)
        return lc.LEARNING_CONFIG["source_reliability"]

    def test_source_reliability_matches_thompson(self):
        """Key source reliability values must reflect empirically observed accuracy."""
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG

        rel = LEARNING_CONFIG["source_reliability"]

        # Congressional: 53 samples, 16.4% observed accuracy
        assert abs(rel["congressional"] - 0.20) <= 0.02, (
            f"congressional reliability should be ~0.20 (empirical), got {rel['congressional']}"
        )

        # finra_short: 1987 samples, 35.8% observed accuracy
        assert abs(rel["finra_short"] - 0.36) <= 0.02, (
            f"finra_short reliability should be ~0.36 (empirical), got {rel['finra_short']}"
        )

        # sec_form4: 18 samples, 36.0% observed accuracy
        assert abs(rel["sec_form4"] - 0.36) <= 0.02, (
            f"sec_form4 reliability should be ~0.36 (empirical), got {rel['sec_form4']}"
        )

        # finnhub_earnings: 110 samples, 30.9% observed accuracy
        assert abs(rel["finnhub_earnings"] - 0.27) <= 0.06, (
            f"finnhub_earnings reliability should be ~0.27-0.33 (empirical), "
            f"got {rel['finnhub_earnings']}"
        )

        # contract_award: very low observed accuracy
        assert abs(rel["contract_award"] - 0.15) <= 0.05, (
            f"contract_award reliability should be ~0.15 (empirical), got {rel['contract_award']}"
        )

        # yfinance_price: low baseline signal
        assert abs(rel["yfinance_price"] - 0.23) <= 0.05, (
            f"yfinance_price reliability should be ~0.23 (empirical), got {rel['yfinance_price']}"
        )

    def test_source_reliability_no_extreme_divergence(self):
        """No source reliability value should be at pathological extremes."""
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG

        rel = LEARNING_CONFIG["source_reliability"]

        for key, value in rel.items():
            assert 0.05 <= value <= 0.95, (
                f"source_reliability['{key}'] = {value} is outside [0.05, 0.95] sanity bounds. "
                "Values at extremes (0 or 1) indicate hardcoded constants, not learned beliefs."
            )


# ---------------------------------------------------------------------------
# Section 4: Data Purity
# ---------------------------------------------------------------------------

class TestDataPurity:
    """Verify production data files contain no test fixture artifacts."""

    PAIR_OUTCOMES_PATH = Path(
        r"C:\Users\baenb\projects\MIDGE\data\market\pair_outcomes.json"
    )
    PREDICTIONS_PATH = Path(
        r"C:\Users\baenb\projects\MIDGE\data\market\predictions.jsonl"
    )

    # Pattern that matches test fixture keys like "a|b", "a0|b0", "a1|b1", ...
    _FIXTURE_KEY_PATTERN = re.compile(r"^a\d*\|b\d*$")

    def test_pair_outcomes_no_test_fixtures(self):
        """pair_outcomes.json must not contain synthetic test fixture keys.

        Keys matching 'a|b' or 'a<N>|b<N>' are artifacts from unit tests that
        wrote directly to the production file. They corrupt the pair quality
        memory that guides hypothesis generation priority.
        """
        assert self.PAIR_OUTCOMES_PATH.exists(), (
            f"pair_outcomes.json not found at {self.PAIR_OUTCOMES_PATH}"
        )

        data = json.loads(self.PAIR_OUTCOMES_PATH.read_text())

        fixture_keys = [
            key for key in data.keys()
            if self._FIXTURE_KEY_PATTERN.match(key)
        ]

        assert len(fixture_keys) == 0, (
            f"pair_outcomes.json contains test fixture keys that must be removed: "
            f"{fixture_keys}. These are synthetic data from unit tests that leaked "
            "into the production file."
        )

    def test_predictions_no_future_dates(self):
        """predictions.jsonl entries must not reference years beyond 2026.

        Any timestamp with year > 2026 indicates a test that injected synthetic
        future-dated data, or a clock/timezone bug.
        """
        assert self.PREDICTIONS_PATH.exists(), (
            f"predictions.jsonl not found at {self.PREDICTIONS_PATH}"
        )

        # Fields that may carry date/timestamp information
        date_fields = [
            "predicted_at", "outcome_due", "timestamp",
            "outcome_date", "last_modified",
        ]

        # Regex to extract a 4-digit year from ISO-format strings
        year_pattern = re.compile(r"(\d{4})-\d{2}-\d{2}")

        violations = []
        with open(self.PREDICTIONS_PATH) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                for field in date_fields:
                    val = entry.get(field, "")
                    if not isinstance(val, str) or not val:
                        continue
                    match = year_pattern.search(val)
                    if match:
                        year = int(match.group(1))
                        if year > 2026:
                            violations.append(
                                f"Line {line_num}: {field}={val!r} (year={year})"
                            )

        assert len(violations) == 0, (
            f"predictions.jsonl contains {len(violations)} entries with dates "
            f"beyond 2026:\n" + "\n".join(violations[:10])
        )
