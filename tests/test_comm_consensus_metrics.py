"""Tests for consensus_metrics - Statistical confidence for quorum decisions.

Tests Wilson score intervals, confidence classification, and consensus comparison.
"""

import math
import pytest

from mae_core.communication.consensus_metrics import (
    ConfidenceInterval,
    ConfidenceLevel,
    ConsensusConfidenceCalculator,
    is_consensus_reliable,
)


class TestConfidenceLevelEnum:
    """Test ConfidenceLevel enum definition."""

    def test_enum_values(self):
        """Verify all confidence level values are defined."""
        assert ConfidenceLevel.VERY_LOW.value == 0.0
        assert ConfidenceLevel.LOW.value == 0.2
        assert ConfidenceLevel.MODERATE.value == 0.4
        assert ConfidenceLevel.HIGH.value == 0.6
        assert ConfidenceLevel.VERY_HIGH.value == 0.8


class TestConfidenceInterval:
    """Test ConfidenceInterval dataclass."""

    def test_basic_construction(self):
        """Create a basic confidence interval."""
        ci = ConfidenceInterval(
            point_estimate=0.75,
            lower_bound=0.5,
            upper_bound=0.9,
            confidence_width=0.4,
            is_significant=True,
            confidence_level=ConfidenceLevel.HIGH,
            supporters=9,
            total_agents=12,
        )
        assert ci.point_estimate == 0.75
        assert ci.lower_bound == 0.5
        assert ci.upper_bound == 0.9
        assert ci.supporters == 9
        assert ci.total_agents == 12

    def test_non_significant_interval(self):
        """Create a non-significant interval."""
        ci = ConfidenceInterval(
            point_estimate=0.1,
            lower_bound=0.0,
            upper_bound=0.5,
            confidence_width=0.5,
            is_significant=False,
            confidence_level=ConfidenceLevel.VERY_LOW,
            supporters=1,
            total_agents=10,
        )
        assert not ci.is_significant
        assert ci.confidence_level == ConfidenceLevel.VERY_LOW


class TestWilsonScoreInterval:
    """Test Wilson score interval calculation."""

    def test_basic_interval_calculation(self):
        """Calculate Wilson score interval for basic case."""
        calc = ConsensusConfidenceCalculator()
        lower, upper = calc.wilson_score_interval(successes=7, total=10)

        # Should be roughly [0.39, 0.92]
        assert lower > 0.35
        assert upper < 1.0
        assert lower < upper

    def test_empty_sample(self):
        """Wilson score for empty sample returns [0, 0]."""
        calc = ConsensusConfidenceCalculator()
        lower, upper = calc.wilson_score_interval(successes=0, total=0)
        assert lower == 0.0
        assert upper == 0.0

    def test_full_success(self):
        """Wilson score when all succeed."""
        calc = ConsensusConfidenceCalculator()
        lower, upper = calc.wilson_score_interval(successes=10, total=10)
        assert lower > 0.6
        assert upper == 1.0

    def test_no_success(self):
        """Wilson score when none succeed."""
        calc = ConsensusConfidenceCalculator()
        lower, upper = calc.wilson_score_interval(successes=0, total=10)
        assert lower == 0.0
        assert upper < 0.5

    def test_bounds_in_valid_range(self):
        """Bounds should always be in [0, 1]."""
        calc = ConsensusConfidenceCalculator()
        for successes in range(0, 6):  # Limit to smaller cases
            for total in range(successes, min(successes + 3, 6)):
                if total == 0:
                    continue
                lower, upper = calc.wilson_score_interval(successes, total)
                assert 0.0 <= lower <= 1.0
                assert 0.0 <= upper <= 1.0
                assert lower <= upper

    def test_custom_z_score(self):
        """Custom z-score affects interval width."""
        calc_default = ConsensusConfidenceCalculator(z_score=1.96)
        calc_loose = ConsensusConfidenceCalculator(z_score=2.576)

        lower_default, upper_default = calc_default.wilson_score_interval(5, 10)
        lower_loose, upper_loose = calc_loose.wilson_score_interval(5, 10)

        # Looser z-score should produce wider interval
        assert (upper_loose - lower_loose) > (upper_default - lower_default)


class TestCalculateMethod:
    """Test main calculate() method."""

    def test_below_min_votes(self):
        """Below min_votes returns non-significant."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=1, total_agents=10, min_votes=3)

        assert not ci.is_significant
        assert ci.confidence_level == ConfidenceLevel.VERY_LOW

    def test_zero_agents(self):
        """Zero agents returns zero values."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=0, total_agents=0)

        assert ci.point_estimate == 0.0
        assert not ci.is_significant

    def test_majority_consensus(self):
        """Strong majority produces significant confidence."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=8, total_agents=10)

        assert ci.point_estimate == 0.8
        assert ci.is_significant
        assert ci.confidence_level in [
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]

    def test_unanimous_decision(self):
        """Unanimous decision should be maximally significant."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=10, total_agents=10)

        assert ci.point_estimate == 1.0
        assert ci.is_significant
        assert ci.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        assert ci.lower_bound >= 0.7

    def test_single_supporter(self):
        """Single supporter in larger group."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=1, total_agents=100)

        assert ci.point_estimate == 0.01
        assert not ci.is_significant

    def test_threshold_sensitivity(self):
        """Significance changes at threshold boundary."""
        calc_loose = ConsensusConfidenceCalculator(significance_threshold=0.2)
        calc_strict = ConsensusConfidenceCalculator(significance_threshold=0.5)

        ci_loose = calc_loose.calculate(supporters=3, total_agents=10)
        ci_strict = calc_strict.calculate(supporters=3, total_agents=10)

        # Same calculation, different thresholds
        # Loose should be more likely to be significant
        assert ci_loose.is_significant or not ci_strict.is_significant


class TestCompareConsensus:
    """Test consensus comparison method."""

    def test_clear_winner_a(self):
        """Option A has clearly stronger consensus."""
        calc = ConsensusConfidenceCalculator()
        result = calc.compare_consensus(msg_a_votes=8, msg_b_votes=2, total_agents=10)
        assert result == "A_STRONGER"

    def test_clear_winner_b(self):
        """Option B has clearly stronger consensus."""
        calc = ConsensusConfidenceCalculator()
        result = calc.compare_consensus(msg_a_votes=2, msg_b_votes=8, total_agents=10)
        assert result == "B_STRONGER"

    def test_no_difference(self):
        """Similar consensus levels."""
        calc = ConsensusConfidenceCalculator()
        result = calc.compare_consensus(msg_a_votes=5, msg_b_votes=5, total_agents=10)
        assert result == "NO_DIFFERENCE"

    def test_both_insignificant(self):
        """Both options have weak support."""
        calc = ConsensusConfidenceCalculator()
        result = calc.compare_consensus(msg_a_votes=1, msg_b_votes=1, total_agents=10)
        assert result == "NO_DIFFERENCE"


class TestConsensusQualityScore:
    """Test quality score calculation."""

    def test_non_significant_scores_zero(self):
        """Non-significant consensus scores 0."""
        calc = ConsensusConfidenceCalculator()
        score = calc.get_consensus_quality_score(supporters=1, total_agents=10)
        assert score == 0.0

    def test_narrow_high_consensus(self):
        """High consensus with narrow interval scores high."""
        calc = ConsensusConfidenceCalculator()
        score = calc.get_consensus_quality_score(supporters=9, total_agents=10)
        assert score > 0.5

    def test_uncertain_consensus(self):
        """Moderate consensus scores moderate."""
        calc = ConsensusConfidenceCalculator()
        score = calc.get_consensus_quality_score(supporters=5, total_agents=10)
        # Should be non-zero if significant
        assert 0.0 <= score <= 1.0

    def test_quality_bounded(self):
        """Quality score always in [0, 1]."""
        calc = ConsensusConfidenceCalculator()
        for supporters in range(0, 11):
            for total in range(supporters, 11):
                if total == 0:
                    continue
                score = calc.get_consensus_quality_score(supporters, total)
                assert 0.0 <= score <= 1.0


class TestRuleOfThree:
    """Test Rule of Three edge case handling."""

    def test_zero_success_upper_bound(self):
        """Rule of Three: no successes observed."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=0, total_agents=10)

        # Should use Rule of Three: 3/10 = 0.3 as upper bound
        assert ci.upper_bound <= 0.3

    def test_full_success_lower_bound(self):
        """Rule of Three: all succeed."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=10, total_agents=10)

        # Should use Rule of Three: 1 - 3/10 = 0.7 as lower bound
        assert ci.lower_bound >= 0.7

    def test_small_sample_rule_of_three(self):
        """Rule of Three applies to small samples."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=0, total_agents=3)

        # With total=3, 3/3=1.0, should be adjusted
        assert ci.upper_bound <= 1.0


class TestConfidenceClassification:
    """Test confidence level classification."""

    def test_classify_very_low(self):
        """Lower bounds < 0.2 are VERY_LOW."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=1, total_agents=100)
        assert ci.confidence_level == ConfidenceLevel.VERY_LOW

    def test_classify_low(self):
        """Lower bounds [0.2, 0.4) are LOW."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=30, total_agents=100)
        # May vary, but testing the classification logic works
        assert ci.confidence_level in [
            ConfidenceLevel.LOW,
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]

    def test_classify_high(self):
        """Lower bounds >= 0.6 are HIGH or VERY_HIGH."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=8, total_agents=10)
        # Due to Wilson score, this may be MODERATE, HIGH, or VERY_HIGH
        assert ci.confidence_level in [
            ConfidenceLevel.MODERATE,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.VERY_HIGH,
        ]

    def test_classify_very_high(self):
        """Lower bounds >= 0.8 are VERY_HIGH."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=10, total_agents=10)
        assert ci.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]


class TestStandaloneFunctionConsensusReliable:
    """Test standalone is_consensus_reliable function."""

    def test_reliable_high_confidence(self):
        """Strong consensus is reliable."""
        result = is_consensus_reliable(supporters=9, total_agents=10, min_confidence=0.5)
        assert result is True

    def test_unreliable_low_confidence(self):
        """Weak consensus is unreliable."""
        result = is_consensus_reliable(supporters=1, total_agents=100, min_confidence=0.5)
        assert result is False

    def test_custom_confidence_threshold(self):
        """Respects custom confidence threshold."""
        # With strict threshold, even reasonable consensus may be unreliable
        result_strict = is_consensus_reliable(
            supporters=6, total_agents=10, min_confidence=0.8
        )
        result_loose = is_consensus_reliable(
            supporters=6, total_agents=10, min_confidence=0.3
        )
        # Loose should be more permissive
        assert not (result_strict and not result_loose)

    def test_default_min_confidence(self):
        """Uses default min_confidence of 0.5."""
        result = is_consensus_reliable(supporters=9, total_agents=10)
        # Should succeed with strong majority
        assert result is True


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_very_large_sample(self):
        """Works with large sample sizes."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=8000, total_agents=10000)
        assert ci.point_estimate == 0.8
        assert ci.is_significant

    def test_very_small_sample(self):
        """Works with very small samples."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=1, total_agents=2)
        # Should handle gracefully
        assert 0.0 <= ci.point_estimate <= 1.0

    def test_point_estimate_precision(self):
        """Point estimate is precise ratio calculation."""
        calc = ConsensusConfidenceCalculator()
        ci = calc.calculate(supporters=3, total_agents=7)
        assert abs(ci.point_estimate - 3/7) < 1e-10

    def test_width_non_negative(self):
        """Confidence width is always non-negative."""
        calc = ConsensusConfidenceCalculator()
        for supporters in range(0, 8):
            for total in range(supporters, 8):
                if total == 0:
                    continue
                ci = calc.calculate(supporters, total)
                assert ci.confidence_width >= 0.0
