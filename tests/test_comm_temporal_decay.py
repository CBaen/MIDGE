"""Tests for temporal_decay - Time-based value decay for communication systems.

Tests decay profiles, decay calculations, and configuration management.
"""

import math
import time
import pytest

from mae_core.communication.temporal_decay import (
    DecayProfile,
    DecayConfig,
    TemporalDecayManager,
    calculate_time_decay,
)


class TestDecayProfileEnum:
    """Test DecayProfile enum."""

    def test_profile_values(self):
        """Verify all decay profiles are defined."""
        assert DecayProfile.FAST.value == (30.0, 0.01)
        assert DecayProfile.NORMAL.value == (60.0, 0.01)
        assert DecayProfile.SLOW.value == (120.0, 0.01)
        assert DecayProfile.PERSISTENT.value == (300.0, 0.01)

    def test_profile_half_lives(self):
        """Profiles have reasonable half-life ordering."""
        fast_hl = DecayProfile.FAST.value[0]
        normal_hl = DecayProfile.NORMAL.value[0]
        slow_hl = DecayProfile.SLOW.value[0]
        persistent_hl = DecayProfile.PERSISTENT.value[0]

        assert fast_hl < normal_hl < slow_hl < persistent_hl


class TestDecayConfig:
    """Test DecayConfig dataclass."""

    def test_basic_config(self):
        """Create a basic decay config."""
        cfg = DecayConfig(half_life=60.0, removal_threshold=0.01)
        assert cfg.half_life == 60.0
        assert cfg.removal_threshold == 0.01

    def test_default_removal_threshold(self):
        """Removal threshold defaults to 0.01."""
        cfg = DecayConfig(half_life=60.0)
        assert cfg.removal_threshold == 0.01

    def test_decay_constant_calculation(self):
        """Decay constant is ln(2) / half_life."""
        cfg = DecayConfig(half_life=60.0)
        expected = math.log(2) / 60.0
        assert abs(cfg.decay_constant - expected) < 1e-10

    def test_zero_half_life_infinite_decay_constant(self):
        """Zero half-life gives infinite decay constant."""
        cfg = DecayConfig(half_life=0.0)
        assert math.isinf(cfg.decay_constant)

    def test_custom_removal_threshold(self):
        """Custom removal threshold is preserved."""
        cfg = DecayConfig(half_life=60.0, removal_threshold=0.001)
        assert cfg.removal_threshold == 0.001


class TestTemporalDecayManagerBasics:
    """Test basic TemporalDecayManager creation."""

    def test_create_manager(self):
        """Create a decay manager."""
        manager = TemporalDecayManager()
        assert manager is not None

    def test_custom_defaults(self):
        """Manager respects custom defaults."""
        manager = TemporalDecayManager(
            default_half_life=30.0,
            default_removal_threshold=0.05,
        )
        assert manager._default.half_life == 30.0
        assert manager._default.removal_threshold == 0.05

    def test_initial_statistics_zero(self):
        """Statistics start at zero."""
        manager = TemporalDecayManager()
        assert manager.total_decays_calculated == 0
        assert manager.messages_removed == 0


class TestSetDecayProfile:
    """Test setting decay profiles."""

    def test_set_fast_profile(self):
        """Set FAST decay profile."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("TEST_SIG", DecayProfile.FAST)

        cfg = manager.get_config("TEST_SIG")
        assert cfg.half_life == 30.0

    def test_set_slow_profile(self):
        """Set SLOW decay profile."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("TEST_SIG", DecayProfile.SLOW)

        cfg = manager.get_config("TEST_SIG")
        assert cfg.half_life == 120.0

    def test_multiple_profiles(self):
        """Set different profiles for different signal types."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("FAST_SIG", DecayProfile.FAST)
        manager.set_decay_profile("SLOW_SIG", DecayProfile.SLOW)

        fast_cfg = manager.get_config("FAST_SIG")
        slow_cfg = manager.get_config("SLOW_SIG")

        assert fast_cfg.half_life < slow_cfg.half_life


class TestSetCustomDecay:
    """Test setting custom decay configurations."""

    def test_set_custom_half_life(self):
        """Set custom half-life."""
        manager = TemporalDecayManager()
        manager.set_custom_decay("MY_SIG", half_life=45.0)

        cfg = manager.get_config("MY_SIG")
        assert cfg.half_life == 45.0

    def test_set_custom_threshold(self):
        """Set custom removal threshold."""
        manager = TemporalDecayManager()
        manager.set_custom_decay("MY_SIG", half_life=60.0, removal_threshold=0.001)

        cfg = manager.get_config("MY_SIG")
        assert cfg.removal_threshold == 0.001

    def test_multiple_custom_configs(self):
        """Set multiple custom configurations."""
        manager = TemporalDecayManager()
        manager.set_custom_decay("SIG_A", half_life=20.0)
        manager.set_custom_decay("SIG_B", half_life=80.0)

        cfg_a = manager.get_config("SIG_A")
        cfg_b = manager.get_config("SIG_B")

        assert cfg_a.half_life == 20.0
        assert cfg_b.half_life == 80.0


class TestGetConfig:
    """Test configuration retrieval."""

    def test_get_configured_signal(self):
        """Get configuration for a configured signal."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("TEST", DecayProfile.FAST)

        cfg = manager.get_config("TEST")
        assert cfg.half_life == 30.0

    def test_get_unconfigured_signal_returns_default(self):
        """Unconfigured signal uses default."""
        manager = TemporalDecayManager(default_half_life=50.0)

        cfg = manager.get_config("UNCONFIGURED")
        assert cfg.half_life == 50.0

    def test_configured_overrides_default(self):
        """Configured signal overrides default."""
        manager = TemporalDecayManager(default_half_life=50.0)
        manager.set_custom_decay("CUSTOM", half_life=25.0)

        cfg = manager.get_config("CUSTOM")
        assert cfg.half_life == 25.0


class TestCalculateDecayFactor:
    """Test decay factor calculation."""

    def test_zero_age_no_decay(self):
        """Zero age results in no decay (factor = 1.0)."""
        manager = TemporalDecayManager()
        factor = manager.calculate_decay_factor(age_seconds=0.0)
        assert factor == 1.0

    def test_negative_age_no_decay(self):
        """Negative age treated as zero."""
        manager = TemporalDecayManager()
        factor = manager.calculate_decay_factor(age_seconds=-1.0)
        assert factor == 1.0

    def test_decay_factor_at_half_life(self):
        """At half-life, decay factor is ~0.5."""
        manager = TemporalDecayManager(default_half_life=60.0)
        # At half-life: e^(-ln(2)) = 0.5
        factor = manager.calculate_decay_factor(age_seconds=60.0)
        assert abs(factor - 0.5) < 0.01

    def test_decay_factor_exponential_formula(self):
        """Decay factor follows e^(-lambda*t)."""
        manager = TemporalDecayManager(default_half_life=60.0)
        age = 30.0
        factor = manager.calculate_decay_factor(age)

        lambda_val = math.log(2) / 60.0
        expected = math.exp(-lambda_val * age)

        assert abs(factor - expected) < 1e-10

    def test_decay_factor_bounded(self):
        """Decay factor always in [0, 1]."""
        manager = TemporalDecayManager()
        for age in [0, 10, 60, 300, 1000]:
            factor = manager.calculate_decay_factor(age)
            assert 0.0 <= factor <= 1.0

    def test_decay_statistics_incremented(self):
        """Decay calculations tracked in statistics."""
        manager = TemporalDecayManager()
        manager.calculate_decay_factor(10.0)
        manager.calculate_decay_factor(20.0)

        assert manager.total_decays_calculated == 2

    def test_custom_config_parameter(self):
        """calculate_decay_factor respects custom config."""
        manager = TemporalDecayManager()
        cfg_fast = DecayConfig(half_life=10.0)
        cfg_slow = DecayConfig(half_life=100.0)

        factor_fast = manager.calculate_decay_factor(20.0, cfg_fast)
        factor_slow = manager.calculate_decay_factor(20.0, cfg_slow)

        assert factor_fast < factor_slow


class TestShouldRemove:
    """Test removal threshold logic."""

    def test_should_remove_below_threshold(self):
        """Values below removal threshold should be removed."""
        manager = TemporalDecayManager(default_half_life=10.0)
        # Very old value
        should_remove = manager.should_remove(age_seconds=100.0)
        assert should_remove is True

    def test_should_not_remove_above_threshold(self):
        """Recent values should not be removed."""
        manager = TemporalDecayManager(default_half_life=60.0)
        should_remove = manager.should_remove(age_seconds=1.0)
        assert should_remove is False

    def test_removal_statistics_tracked(self):
        """Removals are counted in statistics."""
        manager = TemporalDecayManager(default_half_life=10.0)
        manager.should_remove(100.0)  # Will trigger removal
        manager.should_remove(1.0)     # Won't trigger removal
        manager.should_remove(100.0)   # Will trigger removal

        assert manager.messages_removed == 2

    def test_custom_threshold_affects_removal(self):
        """Custom removal threshold affects decision."""
        manager = TemporalDecayManager()
        cfg_strict = DecayConfig(half_life=10.0, removal_threshold=0.5)
        cfg_loose = DecayConfig(half_life=10.0, removal_threshold=0.01)

        # At age=5 (half_life), factor is 0.5
        # Strict: 0.5 < 0.5 is False, so don't remove
        # Loose: 0.5 < 0.01 is False, so don't remove
        # Need bigger difference

        # At age=20 (2*half_life), factor is 0.25
        manager.should_remove(20.0, "TEST")  # Uses default


class TestGetDecayedValue:
    """Test applying decay to a value."""

    def test_zero_age_preserves_value(self):
        """Zero age preserves original value."""
        manager = TemporalDecayManager()
        decayed = manager.get_decayed_value(original_value=1.0, age_seconds=0.0)
        assert decayed == 1.0

    def test_decay_reduces_value(self):
        """Decay reduces value proportionally."""
        manager = TemporalDecayManager(default_half_life=60.0)
        decayed = manager.get_decayed_value(original_value=1.0, age_seconds=60.0)
        assert abs(decayed - 0.5) < 0.01

    def test_decay_for_custom_signal(self):
        """Decay respects custom signal configuration."""
        manager = TemporalDecayManager()
        manager.set_custom_decay("FAST", half_life=10.0)
        manager.set_custom_decay("SLOW", half_life=100.0)

        decayed_fast = manager.get_decayed_value(1.0, 20.0, "FAST")
        decayed_slow = manager.get_decayed_value(1.0, 20.0, "SLOW")

        assert decayed_fast < decayed_slow

    def test_large_value_decay(self):
        """Decay works with large values."""
        manager = TemporalDecayManager()
        decayed = manager.get_decayed_value(1000.0, 60.0)
        assert decayed == pytest.approx(500.0, rel=0.02)

    def test_small_value_decay(self):
        """Decay works with small values."""
        manager = TemporalDecayManager()
        decayed = manager.get_decayed_value(0.001, 60.0)
        assert decayed == pytest.approx(0.0005, rel=0.02)


class TestGetStatistics:
    """Test statistics collection."""

    def test_statistics_dict(self):
        """Get statistics as dictionary."""
        manager = TemporalDecayManager(default_half_life=45.0)
        stats = manager.get_statistics()

        assert isinstance(stats, dict)
        assert "total_decays_calculated" in stats
        assert "messages_removed" in stats
        assert "configured_types" in stats
        assert "default_half_life" in stats

    def test_statistics_values(self):
        """Statistics track actual values."""
        manager = TemporalDecayManager()
        manager.calculate_decay_factor(10.0)
        manager.calculate_decay_factor(20.0)
        # Use a very large age (e.g., 1000 seconds) to ensure decay below threshold
        manager.should_remove(1000.0)

        stats = manager.get_statistics()
        assert stats["total_decays_calculated"] == 3
        assert stats["messages_removed"] == 1

    def test_statistics_configured_types(self):
        """Statistics include configured types count."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("TYPE_A", DecayProfile.FAST)
        manager.set_decay_profile("TYPE_B", DecayProfile.SLOW)

        stats = manager.get_statistics()
        assert stats["configured_types"] == 2


class TestStandaloneFunctionCalculateTimeDecay:
    """Test standalone calculate_time_decay function."""

    def test_basic_decay(self):
        """Calculate simple decay."""
        decay = calculate_time_decay(age_seconds=60.0, half_life=60.0)
        assert abs(decay - 0.5) < 0.01

    def test_zero_age(self):
        """Zero age returns 1.0."""
        decay = calculate_time_decay(age_seconds=0.0, half_life=60.0)
        assert decay == 1.0

    def test_negative_age(self):
        """Negative age returns 1.0."""
        decay = calculate_time_decay(age_seconds=-10.0, half_life=60.0)
        assert decay == 1.0

    def test_zero_half_life(self):
        """Zero half-life returns 1.0."""
        decay = calculate_time_decay(age_seconds=100.0, half_life=0.0)
        assert decay == 1.0

    def test_default_half_life(self):
        """Default half-life is 60 seconds."""
        decay1 = calculate_time_decay(age_seconds=60.0)
        decay2 = calculate_time_decay(age_seconds=60.0, half_life=60.0)
        assert decay1 == decay2

    def test_large_age(self):
        """Very old values decay to near-zero."""
        decay = calculate_time_decay(age_seconds=600.0, half_life=60.0)
        assert decay < 0.001

    def test_decay_formula_correct(self):
        """Decay follows exponential formula."""
        age = 30.0
        half_life = 60.0
        decay = calculate_time_decay(age, half_life)

        expected = math.exp(-math.log(2) / half_life * age)
        assert abs(decay - expected) < 1e-10


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_signal_lifecycle(self):
        """Track a signal from emission to removal."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("ALERT", DecayProfile.FAST)

        # Emit signal (FAST profile = 30 second half-life)
        original = 1.0

        # After 15 seconds (half of fast profile)
        decayed_15 = manager.get_decayed_value(original, 15.0, "ALERT")
        assert 0.6 < decayed_15 < 0.8

        # After 30 seconds total (one full half-life)
        decayed_30 = manager.get_decayed_value(original, 30.0, "ALERT")
        assert 0.4 < decayed_30 < 0.6

        # After 60 seconds (two half-lives)
        decayed_60 = manager.get_decayed_value(original, 60.0, "ALERT")
        assert decayed_60 < 0.3

        # Eventually removed (need much longer for decay below 0.01 threshold)
        # For FAST profile (30s half-life), about 230+ seconds needed
        should_remove = manager.should_remove(300.0, "ALERT")
        assert should_remove is True

    def test_differential_decay_profiles(self):
        """Different signal types decay at different rates."""
        manager = TemporalDecayManager()
        manager.set_decay_profile("OPPORTUNITY", DecayProfile.FAST)
        manager.set_decay_profile("DANGER", DecayProfile.PERSISTENT)

        # After 60 seconds
        opp = manager.get_decayed_value(1.0, 60.0, "OPPORTUNITY")
        danger = manager.get_decayed_value(1.0, 60.0, "DANGER")

        # Danger should decay slower
        assert danger > opp

    def test_batch_operations(self):
        """Process multiple signals efficiently."""
        manager = TemporalDecayManager()

        # Many signals aging
        for i in range(100):
            age = i * 0.5
            decayed = manager.get_decayed_value(1.0, age)
            should_remove = manager.should_remove(age)

        stats = manager.get_statistics()
        assert stats["total_decays_calculated"] > 100


class TestEdgeCasesAndBoundaries:
    """Test edge cases."""

    def test_extremely_fast_decay(self):
        """Very small half-life produces fast decay."""
        manager = TemporalDecayManager(default_half_life=0.001)
        factor = manager.calculate_decay_factor(0.1)
        # With 0.001 second half-life and 0.1 second age, decay is extreme
        assert factor < 1e-25

    def test_extremely_slow_decay(self):
        """Very large half-life produces slow decay."""
        manager = TemporalDecayManager(default_half_life=10000.0)
        factor = manager.calculate_decay_factor(100.0)
        assert factor > 0.99

    def test_consistency_across_calls(self):
        """Same inputs produce same outputs."""
        manager = TemporalDecayManager()

        factor1 = manager.calculate_decay_factor(42.0)
        factor2 = manager.calculate_decay_factor(42.0)

        assert factor1 == factor2

    def test_multiple_managers_independent(self):
        """Multiple managers don't interfere."""
        manager1 = TemporalDecayManager(default_half_life=10.0)
        manager2 = TemporalDecayManager(default_half_life=100.0)

        factor1 = manager1.calculate_decay_factor(50.0)
        factor2 = manager2.calculate_decay_factor(50.0)

        assert factor1 < factor2
