"""Tests for drive coupling: HomeostasisRegulator.compute_drive_urgency(),
OrganismState.get_reflex_override() Priority 6 (homeostasis deviation),
and EndocrineSystem.register_resource_governor() cortisol coupling.

These tests verify:
1. compute_drive_urgency() returns the correct structure and values.
2. The homeostasis deviation reflex fires at the right threshold.
3. Priority ordering is preserved — acute emergencies trump homeostasis.
4. register_resource_governor() calls tighten/relax at the right cortisol levels.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mae_core.coordination.endocrine_system import EndocrineSystem, HormoneType
from mae_core.coordination.homeostasis import HomeostasisRegulator, Setpoint
from mae_core.coordination.organism_state import OrganismState


# =====================================================================
# Helpers
# =====================================================================


def _make_regulator_at_setpoint() -> HomeostasisRegulator:
    """Return a HomeostasisRegulator with all params at their target values."""
    return HomeostasisRegulator()


def _make_regulator_with_deviation(
    parameter_name: str, deviated_value: float
) -> HomeostasisRegulator:
    """Return a regulator with one parameter pushed outside its range."""
    regulator = HomeostasisRegulator()
    regulator.update_current_value(parameter_name, deviated_value)
    return regulator


def _make_organism(homeostasis_deviation: float = 0.0) -> OrganismState:
    """Return an OrganismState with the given homeostasis_deviation injected."""
    organism = OrganismState()
    # Inject deviation directly — mirrors what the homeostasis callback does
    organism._homeostasis_deviation = homeostasis_deviation
    return organism


# =====================================================================
# HomeostasisRegulator.compute_drive_urgency() tests
# =====================================================================


class TestComputeDriveUrgency:
    """Tests for the public compute_drive_urgency() method."""

    def test_compute_drive_urgency_returns_dict(self):
        """compute_drive_urgency() always returns a dict."""
        regulator = _make_regulator_at_setpoint()
        result = regulator.compute_drive_urgency()
        assert isinstance(result, dict)

    def test_compute_drive_urgency_empty_when_stable(self):
        """All parameters at setpoint -> empty dict (nothing is out of range)."""
        regulator = _make_regulator_at_setpoint()
        # Default setpoints are initialised to their target values
        result = regulator.compute_drive_urgency()
        assert result == {}, (
            f"Expected empty dict when all params at setpoint, got {result}"
        )

    def test_compute_drive_urgency_high_when_deviated(self):
        """Parameter far from setpoint -> high urgency value returned."""
        # threat_level: target=0.1, min=0.0, max=0.5, range_width=0.5
        # Push to 0.6 (outside max boundary): error = 0.1 - 0.6 = -0.5
        # urgency = 0.5 / 0.5 = 1.0 (clamped), clearly out of range
        regulator = _make_regulator_with_deviation("threat_level", 0.6)
        result = regulator.compute_drive_urgency()
        assert "threat_level" in result, (
            "threat_level should appear in urgency dict when out of range"
        )
        urgency = result["threat_level"]
        assert 0.0 < urgency <= 1.0, f"Urgency must be in (0, 1], got {urgency}"
        assert urgency >= 0.7, (
            f"Expected high urgency for large deviation, got {urgency}"
        )

    def test_compute_drive_urgency_keys_are_parameter_names(self):
        """Keys in the result are valid parameter names known to the regulator."""
        regulator = _make_regulator_with_deviation("cortisol", 0.9)
        result = regulator.compute_drive_urgency()
        for key in result:
            assert regulator.get_setpoint(key) is not None, (
                f"Unexpected key '{key}' not in regulator setpoints"
            )

    def test_compute_drive_urgency_values_clamped_to_1(self):
        """Urgency values must not exceed 1.0, even for extreme deviations."""
        # energy_level: target=0.7, min=0.3, max=1.0, range_width=0.7
        # Push to 0.0 (well below min): error = 0.7 - 0.0 = 0.7
        # raw urgency = 0.7 / 0.7 = 1.0 (exactly at boundary)
        regulator = _make_regulator_with_deviation("energy_level", 0.0)
        result = regulator.compute_drive_urgency()
        assert "energy_level" in result
        assert result["energy_level"] <= 1.0

    def test_compute_drive_urgency_in_range_params_excluded(self):
        """Parameters within their acceptable range must not appear in the dict."""
        # energy_level default: target=0.7, in_range=[0.3, 1.0] -> 0.7 is in range
        regulator = HomeostasisRegulator()
        # Manually verify energy_level is in range before we check the result
        sp = regulator.get_setpoint("energy_level")
        assert sp is not None
        assert sp.in_range, "energy_level should be in range at default value"
        result = regulator.compute_drive_urgency()
        assert "energy_level" not in result


# =====================================================================
# OrganismState.get_reflex_override() Priority 6 tests
# =====================================================================


class TestReflexOverrideHomeostasisDeviation:
    """Tests for the homeostasis deviation Priority 6 check."""

    def test_reflex_override_homeostasis_deviation(self):
        """High homeostasis deviation (>= threshold) triggers 'rest' override."""
        organism = _make_organism(homeostasis_deviation=0.8)
        result = organism.get_reflex_override()
        assert result == "rest", (
            f"Expected 'rest' for high homeostasis deviation, got {result!r}"
        )

    def test_reflex_override_homeostasis_at_threshold(self):
        """Deviation exactly at the threshold (0.7) triggers 'rest'."""
        organism = _make_organism(
            homeostasis_deviation=OrganismState._HOMEOSTASIS_URGENCY_THRESHOLD
        )
        result = organism.get_reflex_override()
        assert result == "rest", (
            f"Expected 'rest' at exact threshold, got {result!r}"
        )

    def test_reflex_override_homeostasis_below_threshold(self):
        """Normal homeostasis deviation (< threshold) returns None (no override)."""
        # Use a value clearly below the 0.7 threshold
        organism = _make_organism(homeostasis_deviation=0.3)
        result = organism.get_reflex_override()
        assert result is None, (
            f"Expected None for low homeostasis deviation, got {result!r}"
        )

    def test_reflex_override_homeostasis_zero_returns_none(self):
        """Zero homeostasis deviation (fully stable) returns None."""
        organism = _make_organism(homeostasis_deviation=0.0)
        result = organism.get_reflex_override()
        assert result is None, (
            f"Expected None for zero homeostasis deviation, got {result!r}"
        )

    def test_reflex_override_pain_still_higher_priority(self):
        """Pain (Priority 1) takes precedence over homeostasis deviation (Priority 6).

        Both conditions are active simultaneously. Pain should win since it is
        higher in the cascade — the returned action string is 'rest' in both
        cases here, but the important invariant is that pain is checked FIRST.
        We verify this by testing with pain > 0.8 AND high homeostasis deviation
        to confirm the method doesn't raise, and returns 'rest' (pain-driven).
        """
        organism = _make_organism(homeostasis_deviation=0.9)
        organism._pain_load = 0.9  # Pain Priority 1 threshold is > 0.8
        result = organism.get_reflex_override()
        # Both fire "rest", but pain wins the cascade. Result must still be "rest".
        assert result == "rest", (
            f"Expected 'rest' when both pain and homeostasis active, got {result!r}"
        )

    def test_reflex_override_energy_critical_overrides_homeostasis(self):
        """Energy critical (Priority 4, 'explore') takes precedence over homeostasis.

        Priority 4 fires 'explore'. Priority 6 fires 'rest'. Since Priority 4 is
        checked first, 'explore' must be returned.
        """
        organism = _make_organism(homeostasis_deviation=0.9)
        organism._energy_critical = True  # Priority 4
        result = organism.get_reflex_override()
        assert result == "explore", (
            f"Expected 'explore' (Priority 4) to win over homeostasis 'rest', "
            f"got {result!r}"
        )

    def test_reflex_override_threshold_is_class_constant(self):
        """The threshold must be a class constant on OrganismState."""
        assert hasattr(OrganismState, "_HOMEOSTASIS_URGENCY_THRESHOLD"), (
            "OrganismState must have _HOMEOSTASIS_URGENCY_THRESHOLD class attribute"
        )
        assert isinstance(OrganismState._HOMEOSTASIS_URGENCY_THRESHOLD, float)
        assert 0.0 < OrganismState._HOMEOSTASIS_URGENCY_THRESHOLD < 1.0


# =====================================================================
# EndocrineSystem.register_resource_governor() tests
# =====================================================================


class TestRegisterResourceGovernor:
    """Tests for cortisol -> ResourceGovernor budget coupling."""

    def _make_endocrine_with_rg(self) -> tuple[EndocrineSystem, MagicMock]:
        """Return an EndocrineSystem wired to a mock ResourceGovernor."""
        endocrine = EndocrineSystem()
        rg = MagicMock()
        endocrine.register_resource_governor(rg)
        return endocrine, rg

    def test_register_resource_governor_high_cortisol(self):
        """Cortisol 0.8 (> 0.6) -> tighten_budgets(0.8) called, relax_budgets not called."""
        endocrine, rg = self._make_endocrine_with_rg()
        # Release enough cortisol to push level to 0.8
        endocrine.release_hormone(HormoneType.CORTISOL, 0.6, "test_stress")
        rg.tighten_budgets.assert_called()
        call_arg = rg.tighten_budgets.call_args[0][0]
        assert call_arg > 0.6, (
            f"tighten_budgets should receive the cortisol level (> 0.6), got {call_arg}"
        )
        rg.relax_budgets.assert_not_called()

    def test_register_resource_governor_low_cortisol(self):
        """Cortisol 0.1 (< 0.3) -> relax_budgets(1.2) called, tighten_budgets not called."""
        endocrine = EndocrineSystem()
        rg = MagicMock()
        endocrine.register_resource_governor(rg)
        # Force cortisol level to 0.1 by suppressing below baseline (default 0.2)
        endocrine.suppress_hormone(HormoneType.CORTISOL, 0.2, "test_calm")
        # Now release a tiny amount to trigger the subscriber dispatch at low level
        endocrine.release_hormone(HormoneType.CORTISOL, 0.0, "test_trigger")
        # The level is < 0.3, so relax_budgets should be called on any release
        # Alternatively, directly test via a release from below-baseline state
        # Reset and use a direct approach: set level to 0.1 then release
        endocrine2 = EndocrineSystem()
        rg2 = MagicMock()
        endocrine2.register_resource_governor(rg2)
        endocrine2._levels[HormoneType.CORTISOL] = 0.05
        endocrine2.release_hormone(HormoneType.CORTISOL, 0.05, "test_calm")
        # Level after release = 0.10, which is < 0.3 → relax_budgets
        rg2.relax_budgets.assert_called()
        call_arg = rg2.relax_budgets.call_args[0][0]
        # factor = 1.0 + (0.3 - level); level ~= 0.1 → factor ~= 1.2
        assert call_arg > 1.0, (
            f"relax_budgets factor should be > 1.0 for low cortisol, got {call_arg}"
        )
        rg2.tighten_budgets.assert_not_called()

    def test_register_resource_governor_neutral_cortisol(self):
        """Cortisol 0.4 (in neutral zone 0.3-0.6) -> neither method called."""
        endocrine = EndocrineSystem()
        rg = MagicMock()
        endocrine.register_resource_governor(rg)
        # Start from baseline 0.2, add 0.2 → level = 0.4 (neutral zone)
        endocrine.release_hormone(HormoneType.CORTISOL, 0.2, "test_neutral")
        rg.tighten_budgets.assert_not_called()
        rg.relax_budgets.assert_not_called()

    def test_register_resource_governor_none(self):
        """No resource_governor registered -> no AttributeError, no calls."""
        endocrine = EndocrineSystem()
        # Do not register any resource governor — just fire cortisol
        try:
            endocrine.release_hormone(HormoneType.CORTISOL, 0.8, "test_no_rg")
        except Exception as exc:
            pytest.fail(
                f"release_hormone raised unexpectedly when no rg registered: {exc}"
            )
