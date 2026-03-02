"""Tests for Layer 7 meta-learning logic fixes.

Covers:
1. _seed_retirement_window_from_registry — cold-start seeding from registry state
2. Starvation loop fix — Case B can fire because window can be seeded from registry
3. Case C — retire gate loosens when fp_rate is very low
4. Case D — promote_dsr is coupled to promote_win_rate direction
5. Case E — min_observations raised when too many retirements happen at the boundary
6. BacktestScheduler._compute_results_fingerprint — change detection
7. ThompsonCalibrator.get_calibration_feedback — sample_count < 20 guard
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
from mae_core.market.intelligence.backtest_scheduler import BacktestScheduler
from mae_core.market.intelligence.thompson_calibrator import ThompsonCalibrator


# ---------------------------------------------------------------------------
# Shared fixtures (mirrors test_dynamic_gates.py pattern)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


@pytest.fixture
def engine(registry):
    return HypothesisEngine(
        registry=registry,
        generator=MagicMock(generate=MagicMock(return_value=[])),
        validator=MagicMock(),
    )


def _make_hypothesis(name, status, source_a="a", source_b="b", lag_days=10,
                     total_observations=20, min_obs_gate=None):
    """Helper to create a hypothesis in the desired state."""
    hyp = Hypothesis(
        name=name,
        trigger=TriggerPattern(source_a=source_a, source_b=source_b, lag_days=lag_days),
        causal_story="Test causal story.",
    )
    hyp.status = status
    if min_obs_gate is not None:
        hyp.stats.total_observations = min_obs_gate
    else:
        hyp.stats.total_observations = total_observations
    return hyp


# ---------------------------------------------------------------------------
# Test 1: _seed_retirement_window_from_registry
# ---------------------------------------------------------------------------


class TestSeedRetirementWindowFromRegistry:
    def test_seeds_promoted_and_retired_entries(self, tmp_data_dir):
        """ACTIVE → 'promoted', RETIRED → 'retired', PROBATION → excluded."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        active_hyp = _make_hypothesis("active_one", HypothesisStatus.ACTIVE)
        retired_hyp = _make_hypothesis("retired_one", HypothesisStatus.RETIRED)
        probation_hyp = _make_hypothesis("probation_one", HypothesisStatus.PROBATION)

        # Inject directly into registry internal dict (bypass lifecycle guards
        # because promote/retire require specific prior states and we only need
        # get_all() to return these objects for the seed method).
        reg._hypotheses[active_hyp.hypothesis_id] = active_hyp
        reg._hypotheses[retired_hyp.hypothesis_id] = retired_hyp
        reg._hypotheses[probation_hyp.hypothesis_id] = probation_hyp

        eng = HypothesisEngine(
            registry=reg,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
        )

        # Window should contain one "promoted" (ACTIVE) and one "retired" (RETIRED).
        # Entries are now dicts with seeded=True (P5C cold-start guard format).
        outcomes = [e.get("outcome") for e in eng._retirement_window]
        seeded_flags = [e.get("seeded") for e in eng._retirement_window]
        assert "promoted" in outcomes
        assert "retired" in outcomes
        # All seeded entries carry seeded=True so Wire 2 can exclude them
        assert all(s is True for s in seeded_flags)
        # PROBATION must not produce any entry
        assert outcomes.count("promoted") == 1
        assert outcomes.count("retired") == 1
        assert len(eng._retirement_window) == 2

    def test_probation_and_hibernated_excluded(self, tmp_data_dir):
        """PROBATION and HIBERNATED hypotheses must not appear in the seeded window."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        probation_hyp = _make_hypothesis("p1", HypothesisStatus.PROBATION)
        hibernated_hyp = _make_hypothesis("h1", HypothesisStatus.HIBERNATED)

        reg._hypotheses[probation_hyp.hypothesis_id] = probation_hyp
        reg._hypotheses[hibernated_hyp.hypothesis_id] = hibernated_hyp

        eng = HypothesisEngine(
            registry=reg,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
        )

        assert eng._retirement_window == []


# ---------------------------------------------------------------------------
# Test 2: Starvation loop is broken — Case B can now fire
# ---------------------------------------------------------------------------


class TestStarvationLoopBroken:
    def test_case_b_loosens_when_window_non_empty_via_seed(self, tmp_data_dir):
        """With seeding in place, Case B (zero promotions + candidates exist)
        loosens promote_win_rate. Before the fix, an empty retirement window
        could mask the starvation state. After the fix, the window is populated
        from registry history, so Wire 2 of meta-learning has signal.

        This test verifies that the engine with a seeded window (via registry
        RETIRED hypotheses) still allows Case B to fire when meta_promoted_total
        is 0 and probation candidates exist.
        """
        reg = HypothesisRegistry(data_dir=tmp_data_dir)

        # Pre-populate registry with RETIRED hypotheses (these seed the window)
        for i in range(2):
            rh = _make_hypothesis(f"retired_{i}", HypothesisStatus.RETIRED)
            reg._hypotheses[rh.hypothesis_id] = rh

        # Add 3 probation hypotheses so the gate recognises available candidates
        for i in range(3):
            ph = Hypothesis(
                name=f"probation_{i}",
                trigger=TriggerPattern(source_a=f"src_{i}", source_b=f"tgt_{i}", lag_days=5),
                causal_story="Causal story for starvation test.",
            )
            reg.register(ph)

        eng = HypothesisEngine(
            registry=reg,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
        )

        # Confirm the window was seeded (proving the starvation loop fix is active)
        assert len(eng._retirement_window) > 0, (
            "Retirement window should be seeded from RETIRED hypotheses in registry"
        )

        # Set up conditions for Case B
        eng._meta_promoted_total = 0
        eng._step_counter = 5000
        eng._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.55,
                    "promote_dsr": 0.5,
                    "_bounds": {
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.0, 1.0],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            eng._review_gates()

        # Case B should have fired: promote_win_rate loosened by -0.01
        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.promote_win_rate" in call_keys
        win_rate_call = next(
            c for c in mock_update.call_args_list
            if c[0][0] == "hypothesis_gates.promote_win_rate"
        )
        assert win_rate_call[0][1] == pytest.approx(0.54)


# ---------------------------------------------------------------------------
# Test 3: Case C — retire gate loosens when fp_rate is very low
# ---------------------------------------------------------------------------


class TestCaseCRetireGateLookens:
    def test_case_c_appends_retire_win_rate_loosen(self, engine):
        """fp_rate=0 with meta_promoted_total>=5 triggers Case C.

        Case C should append ("retire_win_rate", -0.01) to adjustments.
        No Case A or Case B fires here because fp_rate < 0.30 and promotions > 0.
        """
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 0  # fp_rate = 0.0 < 0.05
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "promote_dsr": 0.5,
                    "retire_win_rate": 0.45,
                    "_bounds": {
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                        "retire_win_rate": [0.35, 0.50],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.retire_win_rate" in call_keys

        retire_call = next(
            c for c in mock_update.call_args_list
            if c[0][0] == "hypothesis_gates.retire_win_rate"
        )
        # 0.45 - 0.01 = 0.44
        assert retire_call[0][1] == pytest.approx(0.44)

    def test_case_c_does_not_fire_when_fp_rate_above_threshold(self, engine):
        """Case C must NOT fire when fp_rate >= 0.05."""
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 1  # fp_rate = 0.10, above 0.05
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "promote_dsr": 0.5,
                    "retire_win_rate": 0.45,
                    "_bounds": {
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                        "retire_win_rate": [0.35, 0.50],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.retire_win_rate" not in call_keys


# ---------------------------------------------------------------------------
# Test 4: Case D — promote_dsr coupled to promote_win_rate direction
# ---------------------------------------------------------------------------


class TestCaseDCouplesDsrToWinRate:
    def test_case_d_tightens_dsr_when_win_rate_tightens(self, engine):
        """When Case A fires (high FP rate → tighten promote_win_rate),
        Case D must also tighten promote_dsr in the same direction (+0.05).
        """
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 5  # fp_rate = 50% > 30%
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "promote_dsr": 0.5,
                    "_bounds": {
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.promote_win_rate" in call_keys
        assert "hypothesis_gates.promote_dsr" in call_keys

        dsr_call = next(
            c for c in mock_update.call_args_list
            if c[0][0] == "hypothesis_gates.promote_dsr"
        )
        # 0.5 + 0.05 = 0.55
        assert dsr_call[0][1] == pytest.approx(0.55)

    def test_case_d_loosens_dsr_when_win_rate_loosens(self, engine, registry):
        """When Case B fires (zero promotions → loosen promote_win_rate),
        Case D must also loosen promote_dsr in the same direction (-0.05).
        """
        engine._meta_promoted_total = 0
        engine._step_counter = 5000
        engine._gate_review_cadence = 2000

        # 3 probation hypotheses to satisfy Case B condition
        for i in range(3):
            hyp = Hypothesis(
                name=f"caseD_{i}",
                trigger=TriggerPattern(source_a=f"x{i}", source_b=f"y{i}", lag_days=7),
                causal_story="Case D coupling test.",
            )
            registry.register(hyp)

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.55,
                    "promote_dsr": 0.5,
                    "_bounds": {
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.promote_win_rate" in call_keys
        assert "hypothesis_gates.promote_dsr" in call_keys

        dsr_call = next(
            c for c in mock_update.call_args_list
            if c[0][0] == "hypothesis_gates.promote_dsr"
        )
        # 0.5 - 0.05 = 0.45
        assert dsr_call[0][1] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Test 5: Case E — min_observations raised when boundary retirements > 40%
# ---------------------------------------------------------------------------


class TestCaseEMinObservationsRaises:
    def test_case_e_raises_min_observations_when_boundary_fraction_high(
        self, tmp_data_dir
    ):
        """When >40% of retired hypotheses have total_observations exactly equal
        to the min_observations gate, Case E raises the gate by +5.
        """
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        min_obs_gate = 20

        # 5 retired at exactly min_obs_gate, 5 retired at different counts → 50%
        for i in range(5):
            h = _make_hypothesis(
                f"boundary_ret_{i}", HypothesisStatus.RETIRED,
                min_obs_gate=min_obs_gate
            )
            reg._hypotheses[h.hypothesis_id] = h

        for i in range(5):
            h = _make_hypothesis(
                f"nonboundary_ret_{i}", HypothesisStatus.RETIRED,
                total_observations=35
            )
            reg._hypotheses[h.hypothesis_id] = h

        eng = HypothesisEngine(
            registry=reg,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
        )
        eng._meta_promoted_total = 0
        eng._step_counter = 0
        eng._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "min_observations": min_obs_gate,
                    "promote_win_rate": 0.52,
                    "promote_dsr": 0.5,
                    "_bounds": {
                        "min_observations": [10, 50],
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            eng._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.min_observations" in call_keys

        min_obs_call = next(
            c for c in mock_update.call_args_list
            if c[0][0] == "hypothesis_gates.min_observations"
        )
        # 20 + 5 = 25
        assert min_obs_call[0][1] == 25

    def test_case_e_does_not_fire_below_threshold(self, tmp_data_dir):
        """When boundary fraction <= 40%, Case E must not fire."""
        reg = HypothesisRegistry(data_dir=tmp_data_dir)
        min_obs_gate = 20

        # 2 boundary, 8 non-boundary → 20% boundary fraction
        for i in range(2):
            h = _make_hypothesis(
                f"bound_{i}", HypothesisStatus.RETIRED,
                min_obs_gate=min_obs_gate
            )
            reg._hypotheses[h.hypothesis_id] = h

        for i in range(8):
            h = _make_hypothesis(
                f"nonbound_{i}", HypothesisStatus.RETIRED,
                total_observations=40
            )
            reg._hypotheses[h.hypothesis_id] = h

        eng = HypothesisEngine(
            registry=reg,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
        )
        eng._meta_promoted_total = 0
        eng._step_counter = 0
        eng._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "min_observations": min_obs_gate,
                    "promote_win_rate": 0.52,
                    "promote_dsr": 0.5,
                    "_bounds": {
                        "min_observations": [10, 50],
                        "promote_win_rate": [0.50, 0.65],
                        "promote_dsr": [0.2, 0.8],
                    },
                    "_regime_deltas": {"default": {}},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            eng._review_gates()

        call_keys = [c[0][0] for c in mock_update.call_args_list]
        assert "hypothesis_gates.min_observations" not in call_keys


# ---------------------------------------------------------------------------
# Test 6: BacktestScheduler._compute_results_fingerprint
# ---------------------------------------------------------------------------


class TestBacktestFingerprint:
    def test_identical_data_produces_identical_fingerprints(self, tmp_path):
        """Two reads of the same results file produce the same fingerprint."""
        results_path = tmp_path / "sweep_backtest_results.json"
        results_path.write_text(json.dumps({
            "run_time": "2026-02-28T10:00:00",
            "summary": {
                "total_trades": 100,
                "win_rate": 44.5,
            },
        }))

        scheduler = BacktestScheduler(
            backtest_analyzer=MagicMock(),
            results_path=results_path,
        )

        fp1 = scheduler._compute_results_fingerprint()
        fp2 = scheduler._compute_results_fingerprint()

        assert fp1 == fp2
        assert fp1 != ""

    def test_changed_data_produces_different_fingerprint(self, tmp_path):
        """Modifying win_rate or total_trades changes the fingerprint."""
        results_path = tmp_path / "sweep_backtest_results.json"
        results_path.write_text(json.dumps({
            "run_time": "2026-02-28T10:00:00",
            "summary": {
                "total_trades": 100,
                "win_rate": 44.5,
            },
        }))

        scheduler = BacktestScheduler(
            backtest_analyzer=MagicMock(),
            results_path=results_path,
        )

        fp_before = scheduler._compute_results_fingerprint()

        # Update win_rate — fingerprint must change
        results_path.write_text(json.dumps({
            "run_time": "2026-02-28T11:00:00",
            "summary": {
                "total_trades": 100,
                "win_rate": 51.3,
            },
        }))

        fp_after = scheduler._compute_results_fingerprint()

        assert fp_before != fp_after

    def test_missing_file_returns_empty_string(self, tmp_path):
        """A missing results file returns an empty fingerprint without raising."""
        results_path = tmp_path / "nonexistent.json"

        scheduler = BacktestScheduler(
            backtest_analyzer=MagicMock(),
            results_path=results_path,
        )

        fp = scheduler._compute_results_fingerprint()
        assert fp == ""


# ---------------------------------------------------------------------------
# Test 7: ThompsonCalibrator.get_calibration_feedback — sample_count < 20 guard
# ---------------------------------------------------------------------------


class TestCalibratorMinSampleGuard:
    def _make_calibrator_with_report(self, tmp_path, thin_source="thin_source",
                                     thin_count=10, thick_source="thick_source",
                                     thick_count=25):
        """Build a ThompsonCalibrator with a mocked _last_report.

        thin_source has sample_count < 20 and should be excluded.
        thick_source has sample_count >= 20 and should appear in feedback
        if it is overconfident.
        """
        from mae_core.market.intelligence.thompson_calibrator import (
            ThompsonCalibrator, SourceCalibration, CalibrationBucket,
        )

        # Build a minimal mock sampler — calibrator only calls _seed_missing_keys
        mock_sampler = MagicMock()
        mock_sampler.distributions = {}
        mock_sampler.prior_scale = 10.0

        # Patch LEARNING_CONFIG so _seed_missing_keys is a no-op
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {thick_source: 0.55}},
        ):
            calibrator = ThompsonCalibrator(
                thompson_sampler=mock_sampler,
                data_dir=tmp_path,
            )

        # Inject a mock calibration report directly
        thin_cal = SourceCalibration(
            source=thin_source,
            brier_score=0.25,
            mean_predicted_confidence=0.70,
            mean_observed_hit_rate=0.40,
            sample_count=thin_count,
            is_overconfident=True,
            is_underconfident=False,
            recommendation="Overconfident",
            buckets=[],
        )
        thick_cal = SourceCalibration(
            source=thick_source,
            brier_score=0.20,
            mean_predicted_confidence=0.75,
            mean_observed_hit_rate=0.40,
            sample_count=thick_count,
            is_overconfident=True,
            is_underconfident=False,
            recommendation="Overconfident",
            buckets=[],
        )
        calibrator._last_report = [thin_cal, thick_cal]
        return calibrator

    def test_thin_source_excluded_from_feedback(self, tmp_path):
        """Source with sample_count=10 must NOT appear in feedback (< 20 guard)."""
        calibrator = self._make_calibrator_with_report(
            tmp_path,
            thin_source="thin_src",
            thin_count=10,
            thick_source="thick_src",
            thick_count=25,
        )

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"thin_src": 0.55, "thick_src": 0.55}},
        ):
            feedback = calibrator.get_calibration_feedback()

        overconfident_keys = [item["key"] for item in feedback["overconfident"]]
        assert "thin_src" not in overconfident_keys

    def test_thick_source_included_in_feedback(self, tmp_path):
        """Source with sample_count=25 MUST appear in overconfident feedback."""
        calibrator = self._make_calibrator_with_report(
            tmp_path,
            thin_source="thin_src",
            thin_count=10,
            thick_source="thick_src",
            thick_count=25,
        )

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"thin_src": 0.55, "thick_src": 0.55}},
        ):
            feedback = calibrator.get_calibration_feedback()

        overconfident_keys = [item["key"] for item in feedback["overconfident"]]
        assert "thick_src" in overconfident_keys

    def test_exactly_20_samples_is_excluded(self, tmp_path):
        """Boundary: sample_count == 19 is excluded; == 20 passes the guard."""
        from mae_core.market.intelligence.thompson_calibrator import (
            ThompsonCalibrator, SourceCalibration,
        )

        mock_sampler = MagicMock()
        mock_sampler.distributions = {}
        mock_sampler.prior_scale = 10.0

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"src_19": 0.55, "src_20": 0.55}},
        ):
            calibrator = ThompsonCalibrator(
                thompson_sampler=mock_sampler,
                data_dir=tmp_path,
            )

        # Two overconfident sources: one at 19, one at 20
        cal_19 = SourceCalibration(
            source="src_19",
            brier_score=0.25,
            mean_predicted_confidence=0.80,
            mean_observed_hit_rate=0.40,
            sample_count=19,
            is_overconfident=True,
            is_underconfident=False,
            recommendation="Overconfident",
            buckets=[],
        )
        cal_20 = SourceCalibration(
            source="src_20",
            brier_score=0.25,
            mean_predicted_confidence=0.80,
            mean_observed_hit_rate=0.40,
            sample_count=20,
            is_overconfident=True,
            is_underconfident=False,
            recommendation="Overconfident",
            buckets=[],
        )
        calibrator._last_report = [cal_19, cal_20]

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"src_19": 0.55, "src_20": 0.55}},
        ):
            feedback = calibrator.get_calibration_feedback()

        overconfident_keys = [item["key"] for item in feedback["overconfident"]]
        assert "src_19" not in overconfident_keys
        assert "src_20" in overconfident_keys
