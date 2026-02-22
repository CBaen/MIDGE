"""Law 4 Remediation — Fractal Self-Similarity at Every Scale.

Law 4: Same triadic pattern at every level. 3 processes → subsystem.
3 subsystems → module. 3 modules → organ. Organs → Mae.

Tests verify:
1. Organism level is triadic (3 children: 2 clusters + 1 bridge)
2. All 10 holon capabilities on OrganClusterAction
3. Fractal self-similarity holds at every level
4. ORGAN_GROUPING is enforced in build_fractal_action
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mae_core.backbone.holon_protocol import HolonRegistry
from mae_core.backbone.fractal_generator import (
    FractalGenerator,
    FRACTAL_GROUPING,
    ORGAN_GROUPING,
)
from mae_core.backbone.fractal_act import (
    OrganClusterAction,
    OrganAction,
    OrganismAction,
    SubsystemAction,
    build_fractal_action,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.connection_registry import ConnectionRegistry


class TestOrganismTriadicStructure(unittest.TestCase):
    """Verify organism level has exactly 3 children (K3)."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")

        # Register all systems from FRACTAL_GROUPING
        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            self.registry.register(name, holon_type="system", parent_id="mae")

        # Build fractal hierarchy
        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=self.registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

    def test_organism_has_three_children(self):
        """Organism delegates to exactly 3 children: 2 clusters + 1 bridge."""
        result = build_fractal_action(self.registry)
        children_count = len(result._cluster_actions) + len(result._bridge_organs)
        self.assertEqual(children_count, 3, "Organism must have exactly 3 children (K3)")

    def test_organ_clustering_enabled(self):
        """ORGAN_GROUPING creates 2 clusters."""
        result = build_fractal_action(self.registry)
        self.assertEqual(len(result._cluster_actions), 2, "Should have 2 clusters")
        self.assertIn("organ-cluster-vital", result._cluster_actions)
        self.assertIn("organ-cluster-cognitive", result._cluster_actions)

    def test_nervous_system_is_bridge(self):
        """Nervous system is a bridge organ (not in clusters)."""
        result = build_fractal_action(self.registry)
        self.assertIn("nervous-system", result._bridge_organs)
        self.assertEqual(len(result._bridge_organs), 1)

    def test_cluster_vital_has_two_organs(self):
        """Vital cluster coordinates metabolic + somatic."""
        result = build_fractal_action(self.registry)
        vital = result._cluster_actions["organ-cluster-vital"]
        organ_ids = list(vital._organ_actions.keys())
        self.assertEqual(len(organ_ids), 2)
        self.assertIn("metabolic-system", organ_ids)
        self.assertIn("somatic-system", organ_ids)

    def test_cluster_cognitive_has_two_organs(self):
        """Cognitive cluster coordinates cognitive + sensory."""
        result = build_fractal_action(self.registry)
        cognitive = result._cluster_actions["organ-cluster-cognitive"]
        organ_ids = list(cognitive._organ_actions.keys())
        self.assertEqual(len(organ_ids), 2)
        self.assertIn("cognitive-system", organ_ids)
        self.assertIn("sensory-system", organ_ids)


class TestOrganClusterCapabilities(unittest.TestCase):
    """All 10 holon capabilities on OrganClusterAction (Law 3)."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("organ-a", holon_type="organ", parent_id="mae")
        self.registry.register("organ-b", holon_type="organ", parent_id="mae")

        # Create organ actions with mock subsystems
        organ_a_sub = MagicMock(spec=SubsystemAction)
        organ_a_sub.act.return_value = 0.7
        organ_a_sub.get_statistics.return_value = {"subsystem_id": "sub-a"}
        organ_a = OrganAction("organ-a", {"sub-a": organ_a_sub}, registry=self.registry)

        organ_b_sub = MagicMock(spec=SubsystemAction)
        organ_b_sub.act.return_value = 0.5
        organ_b_sub.get_statistics.return_value = {"subsystem_id": "sub-b"}
        organ_b = OrganAction("organ-b", {"sub-b": organ_b_sub}, registry=self.registry)

        self.cluster = OrganClusterAction(
            cluster_id="test-cluster",
            organ_actions={"organ-a": organ_a, "organ-b": organ_b},
            registry=self.registry,
        )

    def test_act_capability(self):
        """act() — coordinate child organs."""
        result = self.cluster.act()
        self.assertIsInstance(result, float)
        # Expects avg of organ scores: (0.7 + 0.5) / 2 = 0.6
        self.assertAlmostEqual(result, 0.6)

    def test_sense_capability(self):
        """sense() — aggregate child organ states."""
        result = self.cluster.sense()
        self.assertEqual(result["cluster_id"], "test-cluster")
        self.assertIn("organ_states", result)
        self.assertEqual(len(result["organ_states"]), 2)

    def test_remember_capability(self):
        """remember() — persist cluster-level state."""
        self.cluster.remember("key1", "value1")
        self.assertEqual(self.cluster.remember("key1"), "value1")
        self.assertIsNone(self.cluster.remember("missing"))

    def test_decide_capability(self):
        """decide() — rank children by priority (lowest score first)."""
        # Act first to populate organ scores
        self.cluster.act()
        result = self.cluster.decide()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        # Should be sorted by score (organ-b=0.5 before organ-a=0.7)
        self.assertEqual(result[0][0], "organ-b")
        self.assertEqual(result[1][0], "organ-a")

    def test_learn_capability(self):
        """learn() — track performance trends."""
        self.cluster._last_result = 0.8
        self.cluster.learn()
        self.assertEqual(len(self.cluster._score_history), 1)
        self.assertEqual(self.cluster._score_history[0], 0.8)

    def test_heal_capability(self):
        """heal() — detect and restore failing organs."""
        # Set organ-a to failing state (0.2 <= 0.3)
        self.cluster._organ_actions["organ-a"]._last_result = 0.2
        # Mock the heal method
        self.cluster._organ_actions["organ-a"].heal = MagicMock(return_value={"healed": []})
        result = self.cluster.heal()
        # heal() was called but organ-a is actually in the healed list
        self.assertIsInstance(result["healed"], list)

    def test_know_self_capability(self):
        """know_self() — cluster identity and metadata."""
        result = self.cluster.know_self()
        self.assertEqual(result["holon_id"], "test-cluster")
        self.assertEqual(result["holon_type"], "organ_cluster")
        self.assertEqual(result["organ_count"], 2)

    def test_know_up_capability(self):
        """know_up() — parent context (organism)."""
        result = self.cluster.know_up()
        self.assertIsNotNone(result)
        self.assertEqual(result["parent_id"], "mae")
        self.assertEqual(result["parent_type"], "organism")

    def test_know_down_capability(self):
        """know_down() — aware of child organs."""
        result = self.cluster.know_down()
        self.assertEqual(len(result), 2)
        organ_ids = [r["holon_id"] for r in result]
        self.assertIn("organ-a", organ_ids)
        self.assertIn("organ-b", organ_ids)

    def test_know_peers_capability(self):
        """know_peers() — sibling clusters."""
        # Register a peer cluster
        self.registry.register("test-cluster", holon_type="organ_cluster", parent_id="mae")
        self.registry.register(
            "peer-cluster", holon_type="organ_cluster", parent_id="mae"
        )
        result = self.cluster.know_peers()
        self.assertIsInstance(result, list)


class TestOrganismCapabilities10(unittest.TestCase):
    """Verify all 10 capabilities on OrganismAction (Law 3)."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            self.registry.register(name, holon_type="system", parent_id="mae")

        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=self.registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

        self.organism = build_fractal_action(self.registry)

    def test_all_capabilities_present(self):
        """All 10 + get_statistics + get_state on OrganismAction."""
        required = [
            "act", "sense", "remember", "decide", "learn", "heal",
            "know_self", "know_up", "know_down", "know_peers",
            "get_statistics", "get_state",
        ]
        for cap in required:
            self.assertTrue(
                callable(getattr(self.organism, cap, None)),
                f"OrganismAction missing {cap}",
            )

    def test_act_returns_float(self):
        result = self.organism.act()
        self.assertIsInstance(result, float)

    def test_sense_returns_dict(self):
        result = self.organism.sense()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["organism"], "mae")

    def test_remember_persists(self):
        self.organism.remember("test_key", "test_value")
        self.assertEqual(self.organism.remember("test_key"), "test_value")

    def test_decide_returns_list(self):
        result = self.organism.decide()
        self.assertIsInstance(result, list)

    def test_learn_tracks_history(self):
        self.organism._last_result = 0.75
        self.organism.learn()
        self.assertGreater(len(self.organism._score_history), 0)

    def test_heal_returns_dict(self):
        result = self.organism.heal()
        self.assertIsInstance(result, dict)
        self.assertIn("healed", result)

    def test_know_self_identity(self):
        result = self.organism.know_self()
        self.assertEqual(result["holon_id"], "mae")
        self.assertEqual(result["holon_type"], "organism")

    def test_know_up_is_none(self):
        result = self.organism.know_up()
        self.assertIsNone(result)

    def test_know_down_has_three_children(self):
        result = self.organism.know_down()
        self.assertEqual(len(result), 3)

    def test_know_peers_is_empty(self):
        result = self.organism.know_peers()
        self.assertEqual(len(result), 0)


class TestFractalSelfSimilarityLaw4(unittest.TestCase):
    """Law 4: Same pattern at every scale."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            self.registry.register(name, holon_type="system", parent_id="mae")

        conn_reg = ConnectionRegistry(event_bus=EventBus())
        self.gen = FractalGenerator(
            holon_registry=self.registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        self.gen.organize()

    def test_delegation_pattern_consistent(self):
        """Every level delegates down and aggregates up."""
        organism = build_fractal_action(self.registry)

        # All levels should have act() that returns float
        for cluster in organism._cluster_actions.values():
            for organ in cluster._organ_actions.values():
                for sub_action in organ._subsystem_actions.values():
                    self.assertIsInstance(sub_action.act(), float)
                self.assertIsInstance(organ.act(), float)
            self.assertIsInstance(cluster.act(), float)
        self.assertIsInstance(organism.act(), float)

    def test_aggregation_chain(self):
        """Results aggregate from bottom to top."""
        organism = build_fractal_action(self.registry)

        # Inject mock system refs for consistent scoring
        for sub_id in self.registry._holons:
            entry = self.registry.get_entry(sub_id)
            if entry and entry.holon_type == "system":
                proxy = self.registry.get_proxy(sub_id)
                system = MagicMock()
                system.step.return_value = 0.5
                proxy.set_system_ref(system)

        score = organism.act()
        # Should aggregate all scores (with all returning 0.5)
        self.assertAlmostEqual(score, 0.5)

    def test_every_level_has_ten_capabilities(self):
        """Every action class has all 10 + utilities."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        registry.register("cluster-1", holon_type="organ_cluster", parent_id="mae")
        registry.register("organ-1", holon_type="organ", parent_id="cluster-1")
        registry.register("sub-1", holon_type="subsystem", parent_id="organ-1")
        for i in range(3):
            registry.register(f"sys-{i}", holon_type="system", parent_id="sub-1")

        required = [
            "act", "sense", "remember", "decide", "learn", "heal",
            "know_self", "know_up", "know_down", "know_peers",
        ]

        sub_action = SubsystemAction("sub-1", registry)
        for cap in required:
            self.assertTrue(callable(getattr(sub_action, cap, None)))

        organ_action = OrganAction("organ-1", {"sub-1": sub_action}, registry=registry)
        for cap in required:
            self.assertTrue(callable(getattr(organ_action, cap, None)))

        cluster_action = OrganClusterAction(
            "cluster-1", {"organ-1": organ_action}, registry=registry
        )
        for cap in required:
            self.assertTrue(callable(getattr(cluster_action, cap, None)))

        organism_action = OrganismAction(
            cluster_actions={"cluster-1": cluster_action}, registry=registry
        )
        for cap in required:
            self.assertTrue(callable(getattr(organism_action, cap, None)))

    def test_hierarchy_depth(self):
        """Verify the 4-level hierarchy exists: system → subsystem → organ → cluster → organism."""
        # Check registry structure
        max_depth = self.registry._compute_max_depth()
        # nervous-system is bridge (direct child of mae) = depth 2
        # Other organs are in clusters = depth 3
        # Subsystems under organs = depth 4
        # Systems under subsystems = depth 5
        self.assertGreaterEqual(max_depth, 5)


class TestBuildFractalActionUsesOrganGrouping(unittest.TestCase):
    """Verify build_fractal_action respects ORGAN_GROUPING."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            self.registry.register(name, holon_type="system", parent_id="mae")

        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=self.registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

    def test_build_with_default_organ_grouping(self):
        """build_fractal_action uses ORGAN_GROUPING by default."""
        result = build_fractal_action(self.registry)
        # Should have 2 clusters
        self.assertEqual(len(result._cluster_actions), 2)

    def test_build_respects_custom_organ_grouping(self):
        """build_fractal_action respects custom organ_grouping."""
        custom_grouping = {
            "custom-cluster": ["nervous-system", "metabolic-system"],
        }
        result = build_fractal_action(
            self.registry,
            organ_grouping=custom_grouping,
        )
        # Custom grouping creates 1 cluster with 2 organs
        self.assertEqual(len(result._cluster_actions), 1)
        self.assertIn("custom-cluster", result._cluster_actions)

    def test_organs_not_in_clusters_become_bridges(self):
        """Organs not in ORGAN_GROUPING clusters become bridge organs."""
        result = build_fractal_action(self.registry)
        # nervous-system is not in any cluster, should be a bridge
        self.assertIn("nervous-system", result._bridge_organs)
        # Other organs are clustered
        all_clustered = set()
        for cluster in result._cluster_actions.values():
            all_clustered.update(cluster._organ_actions.keys())
        self.assertNotIn("nervous-system", all_clustered)


class TestLaw4RemediationComplete(unittest.TestCase):
    """Integration test: Law 4 requirements all met."""

    def test_organism_is_k3(self):
        """Organism has exactly 3 children (K3 graph)."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            registry.register(name, holon_type="system", parent_id="mae")

        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

        organism = build_fractal_action(registry)
        children = organism.know_down()
        self.assertEqual(len(children), 3)

    def test_every_level_implements_holon_protocol(self):
        """All action classes implement the 10-capability Holon Protocol."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")

        all_systems = set()
        for organ_spec in FRACTAL_GROUPING.values():
            for system_ids in organ_spec.values():
                for sid in system_ids:
                    all_systems.add(sid)
        for name in sorted(all_systems):
            registry.register(name, holon_type="system", parent_id="mae")

        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

        organism = build_fractal_action(registry)

        # Check organism
        caps = [
            "act", "sense", "remember", "decide", "learn", "heal",
            "know_self", "know_up", "know_down", "know_peers",
        ]
        for cap in caps:
            self.assertTrue(
                callable(getattr(organism, cap, None)),
                f"Organism missing {cap}",
            )

        # Check a cluster
        cluster = list(organism._cluster_actions.values())[0]
        for cap in caps:
            self.assertTrue(
                callable(getattr(cluster, cap, None)),
                f"Cluster missing {cap}",
            )

        # Check an organ
        organ = list(cluster._organ_actions.values())[0]
        for cap in caps:
            self.assertTrue(
                callable(getattr(organ, cap, None)),
                f"Organ missing {cap}",
            )


if __name__ == "__main__":
    unittest.main()
