"""Tests: OctopusColony bootstrap wiring (Round 2).

Verifies that the OctopusColony ecosystem bridge is correctly
instantiated on ctx, registered as a holon, and seeded with the
minimum required octopuses (Law 7: Rule of 3).
"""
from __future__ import annotations

import pytest
from main import create_mae


@pytest.fixture
def mae_organism(tmp_path):
    """Create a minimal Mae organism for testing."""
    model, systems = create_mae(
        num_agents=3,
        cycle_length=20,
        persist_dir=str(tmp_path / "mae_test"),
    )
    yield model, systems
    model.shutdown()


class TestOctopusBootstrap:
    """Verify OctopusColony is correctly wired during Layer 33 bootstrap."""

    def test_colony_on_ctx_after_bootstrap(self, mae_organism):
        """ctx.octopus_colony must be set after market bootstrap.

        If OctopusColony import fails (optional dependency), the attribute
        is set to None — this test will skip gracefully in that case.
        """
        model, systems = mae_organism
        colony = systems.get("octopus_colony")
        # If the network module is unavailable, colony will be None.
        # That is acceptable — what is NOT acceptable is a KeyError or missing key.
        assert "octopus_colony" in systems, (
            "octopus_colony key missing from systems dict — "
            "check main.py _build_systems_dict market intelligence block"
        )

    def test_colony_has_three_octopuses(self, mae_organism):
        """Colony must start with at least 3 octopuses (Law 7: Rule of 3)."""
        _, systems = mae_organism
        colony = systems.get("octopus_colony")
        if colony is None:
            pytest.skip("OctopusColony not available in this environment")
        octopuses = getattr(colony, "octopuses", None)
        assert octopuses is not None, "Colony missing .octopuses attribute"
        assert len(octopuses) >= 3, (
            f"Colony has {len(octopuses)} octopuses — minimum is 3 (Law 7: Rule of 3)"
        )

    def test_colony_registered_as_holon(self, mae_organism):
        """OctopusColony must be registered in the HolonRegistry (Law 3: Holon Protocol)."""
        _, systems = mae_organism
        colony = systems.get("octopus_colony")
        if colony is None:
            pytest.skip("OctopusColony not available in this environment")
        holon_registry = systems.get("holon_registry")
        assert holon_registry is not None, "HolonRegistry not in systems dict"
        all_ids = holon_registry.get_all_ids()
        assert "octopus_colony" in all_ids, (
            "octopus_colony not registered in HolonRegistry — "
            "check market_registration.py _register_market_holons"
        )
