"""Tests for Bridge 5 — Meta-Learning (RSI Layer 3).

Tests the feedback loops that let MIDGE improve how she discovers patterns:
  Wire 1: Calibration → source_reliability
  Wire 2: Retirement rate → generator thresholds
  Wire 3: Pair quality tracking
"""

import pytest
from unittest.mock import MagicMock, patch

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import (
    HypothesisGenerator,
    _get_gen_threshold,
)
from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
from mae_core.market.intelligence.thompson_calibrator import (
    ThompsonCalibrator,
    SourceCalibration,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


@pytest.fixture
def generator(registry, tmp_data_dir):
    return HypothesisGenerator(
        registry=registry,
        lag_data_path=tmp_data_dir / "lag_correlations.json",
    )


@pytest.fixture
def engine(registry, generator):
    return HypothesisEngine(
        registry=registry,
        generator=generator,
        validator=MagicMock(),
    )


# ── Wire 1: Calibration feedback → source_reliability ────────────────


class TestCalibrationFeedback:
    def test_get_calibration_feedback_empty_without_report(self):
        sampler = MagicMock()
        sampler.distributions = {}
        sampler.prior_scale = 10.0
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {}},
        ):
            calibrator = ThompsonCalibrator(sampler, data_dir=MagicMock())
            result = calibrator.get_calibration_feedback()
            assert result == {"overconfident": [], "underconfident": []}

    def test_get_calibration_feedback_detects_overconfident(self):
        sampler = MagicMock()
        sampler.distributions = {}
        sampler.prior_scale = 10.0

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"sec_form4": 0.70}},
        ):
            calibrator = ThompsonCalibrator(sampler, data_dir=MagicMock())
            calibrator._last_report = [
                SourceCalibration(
                    source="sec_form4",
                    brier_score=0.18,
                    mean_predicted_confidence=0.75,
                    mean_observed_hit_rate=0.55,
                    sample_count=30,
                    is_overconfident=True,
                    is_underconfident=False,
                    recommendation="overconfident",
                    buckets=[],
                )
            ]
            result = calibrator.get_calibration_feedback()
            assert len(result["overconfident"]) == 1
            item = result["overconfident"][0]
            assert item["key"] == "sec_form4"
            assert item["gap"] > 0.10
            assert item["delta"] < 0  # Should reduce

    def test_get_calibration_feedback_detects_underconfident(self):
        sampler = MagicMock()
        sampler.distributions = {}
        sampler.prior_scale = 10.0

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"finra_short": 0.65}},
        ):
            calibrator = ThompsonCalibrator(sampler, data_dir=MagicMock())
            calibrator._last_report = [
                SourceCalibration(
                    source="finra_short",
                    brier_score=0.12,
                    mean_predicted_confidence=0.50,
                    mean_observed_hit_rate=0.70,
                    sample_count=25,
                    is_overconfident=False,
                    is_underconfident=True,
                    recommendation="underconfident",
                    buckets=[],
                )
            ]
            result = calibrator.get_calibration_feedback()
            assert len(result["underconfident"]) == 1
            item = result["underconfident"][0]
            assert item["key"] == "finra_short"
            assert item["gap"] < -0.10
            assert item["delta"] > 0  # Should increase

    def test_calibration_feedback_ignores_unknown_sources(self):
        sampler = MagicMock()
        sampler.distributions = {}
        sampler.prior_scale = 10.0

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {}},
        ):
            calibrator = ThompsonCalibrator(sampler, data_dir=MagicMock())
            calibrator._last_report = [
                SourceCalibration(
                    source="unknown_source",
                    brier_score=0.30,
                    mean_predicted_confidence=0.80,
                    mean_observed_hit_rate=0.40,
                    sample_count=10,
                    is_overconfident=True,
                    is_underconfident=False,
                    recommendation="overconfident",
                    buckets=[],
                )
            ]
            result = calibrator.get_calibration_feedback()
            assert len(result["overconfident"]) == 0

    def test_delta_capped_at_005(self):
        sampler = MagicMock()
        sampler.distributions = {}
        sampler.prior_scale = 10.0

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"sec_form4": 0.70}},
        ):
            calibrator = ThompsonCalibrator(sampler, data_dir=MagicMock())
            # Gap of 0.30 → raw delta = 0.30 * -0.3 = -0.09, capped to -0.05
            calibrator._last_report = [
                SourceCalibration(
                    source="sec_form4",
                    brier_score=0.25,
                    mean_predicted_confidence=0.85,
                    mean_observed_hit_rate=0.55,
                    sample_count=40,
                    is_overconfident=True,
                    is_underconfident=False,
                    recommendation="overconfident",
                    buckets=[],
                )
            ]
            result = calibrator.get_calibration_feedback()
            assert result["overconfident"][0]["delta"] == pytest.approx(-0.05)


# ── Wire 1 in _run_meta_learning ─────────────────────────────────────


class TestMetaLearningWire1:
    def test_adjusts_source_reliability_from_calibration(self, registry, generator):
        mock_calibrator = MagicMock()
        mock_calibrator.get_calibration_feedback.return_value = {
            "overconfident": [{"key": "sec_form4", "gap": 0.20, "delta": -0.05}],
            "underconfident": [],
        }

        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
            thompson_calibrator=mock_calibrator,
        )

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"sec_form4": 0.70}, "generator_thresholds": {}},
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_called_once()
            args = mock_update.call_args[0]
            assert args[0] == "source_reliability.sec_form4"
            assert args[1] == pytest.approx(0.65)

    def test_skips_when_no_calibrator(self, engine):
        engine._thompson_calibrator = None
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {}, "generator_thresholds": {}},
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_not_called()


# ── Wire 2: Retirement rate → generator thresholds ───────────────────


class TestMetaLearningWire2:
    def test_tightens_min_correlation_on_high_retirement(self, registry, generator):
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
        )
        # Fill retirement window with 80% retirements
        engine._retirement_window = ["retired"] * 16 + ["promoted"] * 4

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "source_reliability": {},
                "generator_thresholds": {
                    "min_correlation": 0.60,
                    "_bounds": {"min_correlation": [0.4, 0.85]},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_called_once()
            args = mock_update.call_args[0]
            assert args[0] == "generator_thresholds.min_correlation"
            assert args[1] == pytest.approx(0.62)

    def test_loosens_min_correlation_on_low_retirement(self, registry, generator):
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
        )
        # Fill retirement window with 10% retirements
        engine._retirement_window = ["retired"] * 2 + ["promoted"] * 18

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "source_reliability": {},
                "generator_thresholds": {
                    "min_correlation": 0.65,
                    "_bounds": {"min_correlation": [0.4, 0.85]},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_called_once()
            args = mock_update.call_args[0]
            assert args[0] == "generator_thresholds.min_correlation"
            assert args[1] == pytest.approx(0.64)

    def test_no_change_in_healthy_retirement_range(self, registry, generator):
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
        )
        # 40% retirements — healthy range
        engine._retirement_window = ["retired"] * 8 + ["promoted"] * 12

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "source_reliability": {},
                "generator_thresholds": {
                    "min_correlation": 0.60,
                    "_bounds": {"min_correlation": [0.4, 0.85]},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_not_called()

    def test_skips_wire2_with_small_window(self, registry, generator):
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
        )
        # Only 5 entries — below 20 threshold
        engine._retirement_window = ["retired"] * 5

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "source_reliability": {},
                "generator_thresholds": {
                    "min_correlation": 0.60,
                    "_bounds": {"min_correlation": [0.4, 0.85]},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_not_called()

    def test_clamps_to_bounds(self, registry, generator):
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=MagicMock(),
        )
        engine._retirement_window = ["retired"] * 16 + ["promoted"] * 4

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "source_reliability": {},
                "generator_thresholds": {
                    "min_correlation": 0.84,  # Near upper bound
                    "_bounds": {"min_correlation": [0.4, 0.85]},
                },
            },
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._run_meta_learning()
            mock_update.assert_called_once()
            assert mock_update.call_args[0][1] == pytest.approx(0.85)

    def test_meta_learning_cadence_fires_in_step(self, engine):
        engine._meta_learning_cadence = 10
        ml_count = [0]
        original = engine._run_meta_learning

        def counting():
            ml_count[0] += 1
            original()

        engine._run_meta_learning = counting
        for _ in range(20):
            engine.step()
        assert ml_count[0] == 2


# ── Wire 3: Pair quality tracking ────────────────────────────────────


class TestPairQuality:
    def test_promote_records_outcome_on_generator(self, registry):
        mock_gen = MagicMock()
        engine = HypothesisEngine(
            registry=registry,
            generator=mock_gen,
            validator=MagicMock(),
        )
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="sec_form4", source_b="finnhub_earnings", lag_days=45),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._promote(hyp)
        mock_gen.record_outcome.assert_called_once_with(
            "sec_form4", "finnhub_earnings", "promoted"
        )

    def test_retire_records_outcome_on_generator(self, registry):
        mock_gen = MagicMock()
        engine = HypothesisEngine(
            registry=registry,
            generator=mock_gen,
            validator=MagicMock(),
        )
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._retire(hyp, "Bad")
        mock_gen.record_outcome.assert_called_once_with("a", "b", "retired")

    def test_generator_pair_priority_bonus(self, registry, tmp_data_dir):
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        gen.record_outcome("a", "b", "promoted")
        gen.record_outcome("a", "b", "promoted")
        gen.record_outcome("a", "b", "retired")
        # a→b has 2 promoted, 1 retired → should get bonus
        finding = {"source_a": "a", "source_b": "b", "correlation": 0.70}
        priority = gen._finding_priority(finding)
        assert priority == pytest.approx(0.70 + 0.1)

    def test_generator_no_bonus_for_bad_pair(self, registry, tmp_data_dir):
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        gen.record_outcome("a", "b", "retired")
        gen.record_outcome("a", "b", "retired")
        finding = {"source_a": "a", "source_b": "b", "correlation": 0.70}
        priority = gen._finding_priority(finding)
        assert priority == pytest.approx(0.70)  # No bonus


# ── Retirement window ────────────────────────────────────────────────


class TestRetirementWindow:
    def test_promote_appends_to_window(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._promote(hyp)
        assert engine._retirement_window == ["promoted"]

    def test_retire_appends_to_window(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._retire(hyp, "Bad")
        assert engine._retirement_window == ["retired"]

    def test_window_caps_at_max(self, engine, registry):
        engine._retirement_window_max = 5
        for i in range(7):
            hyp = Hypothesis(
                name=f"test_{i}",
                trigger=TriggerPattern(source_a=f"a{i}", source_b=f"b{i}", lag_days=10),
                causal_story="Test.",
            )
            registry.register(hyp)
            engine._promote(hyp)
        assert len(engine._retirement_window) == 5

    def test_publishes_ch_meta_adjusted(self, registry):
        bus = MagicMock()
        mock_calibrator = MagicMock()
        mock_calibrator.get_calibration_feedback.return_value = {
            "overconfident": [{"key": "sec_form4", "gap": 0.20, "delta": -0.05}],
            "underconfident": [],
        }

        engine = HypothesisEngine(
            registry=registry,
            generator=MagicMock(),
            validator=MagicMock(),
            bus=bus,
            thompson_calibrator=mock_calibrator,
        )

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"source_reliability": {"sec_form4": 0.70}, "generator_thresholds": {}},
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ):
            engine._run_meta_learning()
            meta_calls = [
                c for c in bus.publish.call_args_list
                if "meta_adjusted" in str(c)
            ]
            assert len(meta_calls) == 1


# ── Generator threshold config read ──────────────────────────────────


class TestGenThreshold:
    def test_reads_from_config(self):
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {"generator_thresholds": {"min_correlation": 0.70}},
        ):
            assert _get_gen_threshold("min_correlation") == 0.70

    def test_falls_back_to_hardcoded(self):
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {},
        ):
            assert _get_gen_threshold("min_correlation") == 0.6
            assert _get_gen_threshold("min_pairs") == 10

    def test_generator_statistics_include_live_thresholds(self, registry, tmp_data_dir):
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        stats = gen.get_statistics()
        assert "live_min_correlation" in stats
        assert "live_min_pairs" in stats
        assert "pair_outcomes_tracked" in stats


# ── Channel constants exist ──────────────────────────────────────────


class TestChannels:
    def test_gate_adjusted_channel_exists(self):
        from mae_core.market.channels import CH_GATE_ADJUSTED
        assert "gate_adjusted" in CH_GATE_ADJUSTED

    def test_meta_adjusted_channel_exists(self):
        from mae_core.market.channels import CH_META_ADJUSTED
        assert "meta_adjusted" in CH_META_ADJUSTED
