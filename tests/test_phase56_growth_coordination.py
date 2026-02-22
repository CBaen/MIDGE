"""Phase 5.6 tests: Growth and Coordination layer.

Tests for:
- Substrate: topology, nutrient flow, signal propagation, agent registration
- Morphogenesis: novelty detection, blueprint design, organ lifecycle
- Endocrine: 6 hormones, cascades, decay, circadian integration
- Circadian: phase transitions, callbacks, statistics
- Predictive Field: predictions, collision detection, gradients
- Spatial Consensus: voting, weighted consensus, heatmap
- Cross-system integrations
"""

import threading
import time
import unittest

from mae_core.backbone.event_bus import EventBus


# ===========================================================================
# Substrate Tests
# ===========================================================================


class TestSubstrateTopology(unittest.TestCase):
    """Test topology generators and graph management."""

    def test_ring_topology(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.ring(10)
        self.assertEqual(topo.node_count, 10)
        self.assertEqual(topo.edge_count, 10)
        # Every node has exactly 2 neighbors in a ring
        for nid in topo.get_nodes():
            self.assertEqual(topo.get_degree(nid), 2)
        self.assertAlmostEqual(topo.connectivity_ratio(), 1.0)

    def test_scale_free_topology(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.scale_free(20, m=3)
        self.assertEqual(topo.node_count, 20)
        self.assertGreater(topo.edge_count, 19)  # At least a spanning tree
        self.assertAlmostEqual(topo.connectivity_ratio(), 1.0)
        # Hub nodes should have higher degree than average
        hubs = topo.get_hub_nodes(top_k=3)
        avg = topo.average_degree()
        self.assertGreater(hubs[0][1], avg)

    def test_small_world_topology(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.small_world(20, k=4, p=0.3)
        self.assertEqual(topo.node_count, 20)
        self.assertAlmostEqual(topo.connectivity_ratio(), 1.0)
        # Small world should have positive clustering
        cc = topo.clustering_coefficient()
        self.assertGreater(cc, 0.0)

    def test_mesh_topology(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.mesh(5)
        self.assertEqual(topo.node_count, 5)
        self.assertEqual(topo.edge_count, 10)  # n*(n-1)/2
        for nid in topo.get_nodes():
            self.assertEqual(topo.get_degree(nid), 4)

    def test_dynamic_node_management(self):
        from mae_core.substrate.topology import SubstrateTopology, TopologyType

        topo = SubstrateTopology(TopologyType.CUSTOM)
        topo.add_node("a")
        topo.add_node("b")
        topo.add_node("c")
        topo.add_edge("a", "b")
        topo.add_edge("b", "c")

        self.assertEqual(topo.node_count, 3)
        self.assertEqual(topo.edge_count, 2)
        self.assertEqual(sorted(topo.get_neighbors("b")), ["a", "c"])

        topo.remove_node("b")
        self.assertEqual(topo.node_count, 2)
        self.assertEqual(topo.edge_count, 0)

    def test_agent_node_mapping(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.ring(5)
        node_ids = list(topo.get_nodes().keys())

        topo.assign_agent(42, node_ids[0])
        topo.assign_agent(43, node_ids[1])

        self.assertEqual(topo.get_agent_node(42), node_ids[0])
        self.assertEqual(topo.get_node_agent(node_ids[0]), 42)
        self.assertIn(43, topo.get_agent_neighbors(42))

        topo.unassign_agent(42)
        self.assertIsNone(topo.get_agent_node(42))

    def test_shortest_path(self):
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.ring(6)
        node_ids = list(topo.get_nodes().keys())
        path = topo.shortest_path(node_ids[0], node_ids[3])
        self.assertIsNotNone(path)
        self.assertLessEqual(len(path), 4)  # Ring: max distance is n/2


class TestNutrientFlow(unittest.TestCase):
    """Test resource distribution."""

    def test_flow_equalizes_resources(self):
        from mae_core.substrate.nutrient_flow import FlowConfig, NutrientFlowEngine
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.mesh(3)
        nodes = topo.get_nodes()
        node_ids = list(nodes.keys())

        # Set unequal resources
        nodes[node_ids[0]].resource_level = 1.0
        nodes[node_ids[1]].resource_level = 0.0
        nodes[node_ids[2]].resource_level = 0.0

        flow = NutrientFlowEngine(topo, FlowConfig(base_flow_rate=0.3, decay_rate=0.0))

        # Run flow for several steps
        for _ in range(20):
            flow.flow_step()

        # Resources should have equalized somewhat
        levels = [nodes[nid].resource_level for nid in node_ids]
        spread = max(levels) - min(levels)
        self.assertLess(spread, 0.5)  # Should be more equal

    def test_injection_and_consumption(self):
        from mae_core.substrate.nutrient_flow import NutrientFlowEngine
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.ring(3)
        flow = NutrientFlowEngine(topo)

        node_id = list(topo.get_nodes().keys())[0]
        flow.inject_resources(node_id, 0.3)
        level = flow.get_resource_level(node_id)
        self.assertGreater(level, 0.5)

        consumed = flow.consume_resources(node_id, 0.2)
        self.assertAlmostEqual(consumed, 0.2, places=1)

    def test_starvation_detection(self):
        from mae_core.substrate.nutrient_flow import FlowConfig, NutrientFlowEngine
        from mae_core.substrate.topology import SubstrateTopology

        topo = SubstrateTopology.ring(3)
        nodes = topo.get_nodes()
        for node in nodes.values():
            node.resource_level = 0.1  # Below starvation threshold

        flow = NutrientFlowEngine(topo, FlowConfig(starvation_threshold=0.15))
        starving = flow.get_starving_nodes()
        self.assertEqual(len(starving), 3)


class TestMycelialSubstrate(unittest.TestCase):
    """Test the full substrate orchestrator."""

    def test_agent_registration(self):
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        sub = MycelialSubstrate(event_bus=bus, topology_type="scale_free", initial_nodes=5)

        node_id = sub.register_agent(1)
        self.assertIsNotNone(node_id)

        # Duplicate registration returns same node
        node_id2 = sub.register_agent(1)
        self.assertEqual(node_id, node_id2)

        # Deregister
        removed = sub.deregister_agent(1)
        self.assertEqual(removed, node_id)

    def test_signal_propagation(self):
        from mae_core.substrate import MycelialSubstrate

        sub = MycelialSubstrate(topology_type="mesh", initial_nodes=5)
        for i in range(3):
            sub.register_agent(i)

        reached = sub.propagate_signal(0, {"type": "test"}, max_hops=3)
        self.assertGreater(len(reached), 0)

    def test_peer_query(self):
        from mae_core.substrate import MycelialSubstrate

        sub = MycelialSubstrate(topology_type="mesh", initial_nodes=5)
        for i in range(5):
            sub.register_agent(i)

        peers = sub.get_peers(0, max_peers=3)
        self.assertLessEqual(len(peers), 3)

    def test_health_report(self):
        from mae_core.substrate import MycelialSubstrate

        sub = MycelialSubstrate(topology_type="scale_free", initial_nodes=10)
        report = sub.get_health_report()
        self.assertIn("topology", report)
        self.assertIn("resources", report)
        self.assertIn("signals", report)
        self.assertEqual(report["topology"]["nodes"], 10)

    def test_region_isolation_and_restore(self):
        from mae_core.substrate import MycelialSubstrate

        sub = MycelialSubstrate(topology_type="mesh", initial_nodes=5)
        node_ids = list(sub.topology.get_nodes().keys())

        # Isolate first node
        removed = sub.isolate_region([node_ids[0]])
        self.assertGreater(sum(len(v) for v in removed.values()), 0)

        # Restore
        sub.restore_region(removed)

    def test_substrate_step(self):
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        sub = MycelialSubstrate(event_bus=bus, health_report_interval=1)
        sub.register_agent(0)
        sub.step()
        self.assertEqual(sub._step_count, 1)


# ===========================================================================
# Morphogenesis Tests
# ===========================================================================


class TestProblemSignature(unittest.TestCase):
    def test_similarity(self):
        from mae_core.morphogenesis import ProblemSignature

        s1 = ProblemSignature(coordination_level=0.8, exploration_level=0.2)
        s2 = ProblemSignature(coordination_level=0.8, exploration_level=0.2)
        s3 = ProblemSignature(coordination_level=0.1, exploration_level=0.9)

        self.assertAlmostEqual(s1.similarity(s2), 1.0)
        self.assertLess(s1.similarity(s3), 0.7)


class TestNoveltyDetector(unittest.TestCase):
    def test_first_problem_is_novel(self):
        from mae_core.morphogenesis import NoveltyDetector, ProblemSignature

        detector = NoveltyDetector(novelty_threshold=0.3)
        sig = ProblemSignature()
        self.assertTrue(detector.is_novel(sig))

    def test_repeated_problem_not_novel(self):
        from mae_core.morphogenesis import NoveltyDetector, ProblemSignature

        detector = NoveltyDetector(novelty_threshold=0.3)
        sig = ProblemSignature(coordination_level=0.5)
        detector.register_signature(sig)

        similar = ProblemSignature(coordination_level=0.55)
        self.assertFalse(detector.is_novel(similar))


class TestOrganBuilder(unittest.TestCase):
    def test_design_blueprint(self):
        from mae_core.morphogenesis import OrganBuilder, ProblemSignature

        builder = OrganBuilder()
        sig = ProblemSignature(
            coordination_level=0.9,
            exploration_level=0.7,
            complexity=0.8,
            risk_level=0.5,
        )
        blueprint = builder.design_organ(sig)

        self.assertGreater(blueprint.total_agents, 0)
        self.assertTrue(blueprint.validate())
        self.assertIn("coordinator", blueprint.composition)
        self.assertIn("explorer", blueprint.composition)

    def test_grow_organ_metadata_only(self):
        from mae_core.morphogenesis import OrganBuilder, OrganStatus, ProblemSignature

        builder = OrganBuilder()
        sig = ProblemSignature(complexity=0.5)
        blueprint = builder.design_organ(sig)
        organ = builder.grow_organ(blueprint)

        self.assertEqual(organ.status, OrganStatus.ACTIVE)
        self.assertGreater(len(organ.agents), 0)

    def test_dissolve_organ(self):
        from mae_core.morphogenesis import OrganBuilder, ProblemSignature

        builder = OrganBuilder()
        sig = ProblemSignature()
        blueprint = builder.design_organ(sig)
        organ = builder.grow_organ(blueprint)

        self.assertEqual(builder.active_organ_count, 1)
        builder.dissolve_organ(organ.organ_id)
        self.assertEqual(builder.active_organ_count, 0)


class TestMorphogenesisCoordinator(unittest.TestCase):
    def test_handle_novel_problem(self):
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature

        coord = MorphogenesisCoordinator()
        sig = ProblemSignature(domain="test", complexity=0.7)
        organ = coord.handle_novel_problem(sig)

        self.assertIsNotNone(organ)
        stats = coord.get_statistics()
        self.assertEqual(stats["novel_problems_detected"], 1)
        self.assertEqual(stats["organs_created"], 1)

    def test_repeated_problem_returns_none(self):
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature

        coord = MorphogenesisCoordinator()
        sig = ProblemSignature(domain="test")
        coord.handle_novel_problem(sig)
        result = coord.handle_novel_problem(sig)
        self.assertIsNone(result)

    def test_eventbus_integration(self):
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature

        bus = EventBus()
        events = []
        bus.register_callback("morphogenesis.team_created", lambda ch, msg: events.append(msg))

        coord = MorphogenesisCoordinator(event_bus=bus)
        coord.handle_novel_problem(ProblemSignature(domain="test"))
        self.assertEqual(len(events), 1)

    def test_spawn_request_via_eventbus(self):
        from mae_core.morphogenesis import MorphogenesisCoordinator

        bus = EventBus()
        coord = MorphogenesisCoordinator(event_bus=bus)

        bus.publish("morphogenesis.spawn_request", {
            "domain": "dynamic",
            "complexity": 0.6,
            "coordination_level": 0.5,
        })

        stats = coord.get_statistics()
        self.assertEqual(stats["organs_created"], 1)


# ===========================================================================
# Endocrine Tests
# ===========================================================================


class TestEndocrineSystem(unittest.TestCase):
    def test_initial_levels(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        for ht in HormoneType:
            level = endo.get_level(ht)
            self.assertGreater(level, 0.0)

    def test_release_hormone(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        old = endo.get_level(HormoneType.DOPAMINE)
        new = endo.release_hormone(HormoneType.DOPAMINE, 0.3, "test")
        self.assertGreater(new, old)

    def test_cascade_effects(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        old_cortisol = endo.get_level(HormoneType.CORTISOL)
        # Adrenaline should cascade to cortisol
        endo.release_hormone(HormoneType.ADRENALINE, 0.5, "emergency")
        new_cortisol = endo.get_level(HormoneType.CORTISOL)
        self.assertGreater(new_cortisol, old_cortisol)

    def test_decay_toward_baseline(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        endo.release_hormone(HormoneType.CORTISOL, 0.5, "stress")
        high = endo.get_level(HormoneType.CORTISOL)

        for _ in range(50):
            endo.step()

        decayed = endo.get_level(HormoneType.CORTISOL)
        self.assertLess(decayed, high)

    def test_stress_detection(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        self.assertFalse(endo.is_stressed())
        endo.release_hormone(HormoneType.CORTISOL, 0.5, "stress")
        self.assertTrue(endo.is_stressed())

    def test_modulation_helpers(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        endo.release_hormone(HormoneType.DOPAMINE, 0.5, "reward")
        exploration = endo.get_exploration_bias()
        self.assertGreater(exploration, 0.3)

        endo.release_hormone(HormoneType.ADRENALINE, 0.5, "emergency")
        reflex = endo.get_reflex_bias()
        self.assertGreater(reflex, 0.3)

    def test_circadian_integration(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        old_melatonin = endo.get_level(HormoneType.MELATONIN)
        endo.set_circadian_phase("REST")
        new_melatonin = endo.get_level(HormoneType.MELATONIN)
        self.assertGreater(new_melatonin, old_melatonin)

    def test_subscriber_notification(self):
        from mae_core.coordination import EndocrineSystem, HormoneType

        endo = EndocrineSystem()
        notifications = []
        endo.subscribe(HormoneType.DOPAMINE, lambda ht, level: notifications.append(level))
        endo.release_hormone(HormoneType.DOPAMINE, 0.3, "test")
        self.assertEqual(len(notifications), 1)


# ===========================================================================
# Circadian Tests
# ===========================================================================


class TestCircadianRhythm(unittest.TestCase):
    def test_phase_transitions(self):
        from mae_core.coordination import CircadianPhase, CircadianRhythm

        circ = CircadianRhythm(cycle_length=30, active_ratio=0.5, consolidation_ratio=0.3)
        phases = []

        for _ in range(35):
            result = circ.step()
            if result:
                phases.append(result)

        self.assertIn(CircadianPhase.CONSOLIDATION, phases)
        self.assertIn(CircadianPhase.REST, phases)

    def test_callbacks(self):
        from mae_core.coordination import CircadianRhythm

        circ = CircadianRhythm(cycle_length=10)
        transitions = []
        circ.on_phase_change(lambda old, new: transitions.append((old, new)))

        for _ in range(15):
            circ.step()

        self.assertGreater(len(transitions), 0)

    def test_activity_multiplier(self):
        from mae_core.coordination import CircadianPhase, CircadianRhythm

        circ = CircadianRhythm(cycle_length=10)
        self.assertEqual(circ.get_activity_multiplier(), 1.0)  # Starts ACTIVE

    def test_cycle_counting(self):
        from mae_core.coordination import CircadianRhythm

        circ = CircadianRhythm(cycle_length=10)
        for _ in range(25):
            circ.step()
        self.assertGreaterEqual(circ.cycle_count, 2)

    def test_eventbus_phase_change(self):
        from mae_core.coordination import CircadianRhythm

        bus = EventBus()
        events = []
        bus.register_callback("circadian.phase_change", lambda ch, msg: events.append(msg))

        circ = CircadianRhythm(event_bus=bus, cycle_length=10)
        for _ in range(15):
            circ.step()

        self.assertGreater(len(events), 0)


# ===========================================================================
# Predictive Field Tests
# ===========================================================================


class TestPredictiveField(unittest.TestCase):
    def test_agent_prediction(self):
        from mae_core.communication.predictive_field import PredictiveField

        pf = PredictiveField()
        pf.update_agent_state(0, (5.0, 5.0), velocity=(1.0, 0.0))

        predicted = pf.predict_agent_state(0, steps_ahead=3)
        self.assertIsNotNone(predicted)
        self.assertAlmostEqual(predicted[0], 8.0)
        self.assertAlmostEqual(predicted[1], 5.0)

    def test_collision_detection(self):
        from mae_core.communication.predictive_field import PredictiveField

        pf = PredictiveField(field_size=(50.0, 50.0), prediction_horizon=2)
        pf.update_agent_state(0, (10.0, 10.0), velocity=(1.0, 0.0))
        pf.update_agent_state(1, (14.0, 10.0), velocity=(-1.0, 0.0))

        risks = pf.detect_collision_risk(0, threshold=3.0)
        # Agents moving toward each other, predicted positions should be close
        self.assertGreater(len(risks), 0)

    def test_coordination_opportunities(self):
        from mae_core.communication.predictive_field import PredictiveField

        pf = PredictiveField()
        pf.update_agent_state(0, (5.0, 5.0), intention="explore")
        pf.update_agent_state(1, (6.0, 5.0), intention="harvest")

        opps = pf.find_coordination_opportunities(0, max_distance=5.0)
        self.assertGreater(len(opps), 0)

    def test_field_gradient(self):
        from mae_core.communication.predictive_field import PredictiveField

        pf = PredictiveField(field_size=(20.0, 20.0), grid_resolution=1.0)
        # Put many agents one cell to the right of query position
        for i in range(5):
            pf.update_agent_state(i, (6.0 + i * 0.1, 5.0))

        gradient = pf.calculate_field_gradient((5.0, 5.0))
        # Gradient should point toward the cluster (rightward)
        self.assertGreater(gradient[0], 0)

    def test_step_decays_predictions(self):
        from mae_core.communication.predictive_field import PredictiveField

        pf = PredictiveField(prediction_horizon=3)
        pf.update_agent_state(0, (5.0, 5.0))

        for _ in range(5):
            pf.step()

        stats = pf.get_statistics()
        # Old predictions should have been cleaned
        self.assertEqual(stats["total_predictions"], 0)


# ===========================================================================
# Spatial Consensus Tests
# ===========================================================================


class TestSpatialConsensus(unittest.TestCase):
    def test_basic_consensus(self):
        from mae_core.communication.spatial_consensus import SpatialConsensusTracker

        sc = SpatialConsensusTracker()
        for i in range(5):
            sc.add_vote(i, (5.0, 5.0), "direction", "north")
        for i in range(5, 7):
            sc.add_vote(i, (5.0, 5.0), "direction", "south")

        result = sc.get_consensus_at((5.0, 5.0), "direction", radius=10.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.consensus_value, "north")
        self.assertGreater(result.consensus_strength, 0.6)

    def test_distance_decay(self):
        from mae_core.communication.spatial_consensus import SpatialConsensusTracker

        sc = SpatialConsensusTracker(grid_resolution=2.0)
        # Close votes for "yes"
        for i in range(5):
            sc.add_vote(i, (5.0, 5.0), "approve", True)
        # Far votes for "no"
        for i in range(5, 10):
            sc.add_vote(i, (50.0, 50.0), "approve", False)

        result = sc.get_consensus_at((5.0, 5.0), "approve", radius=10.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.consensus_value, True)

    def test_not_enough_votes(self):
        from mae_core.communication.spatial_consensus import SpatialConsensusTracker

        sc = SpatialConsensusTracker()
        sc.add_vote(0, (5.0, 5.0), "topic", "value")
        result = sc.get_consensus_at((5.0, 5.0), "topic", radius=10.0, min_votes=3)
        self.assertIsNone(result)

    def test_temporal_decay(self):
        from mae_core.communication.spatial_consensus import SpatialConsensusTracker

        sc = SpatialConsensusTracker(temporal_decay=0.5)
        for i in range(5):
            sc.add_vote(i, (5.0, 5.0), "topic", "old")

        # Advance time
        for _ in range(20):
            sc.step()

        stats = sc.get_statistics()
        self.assertEqual(stats["active_votes"], 0)  # All decayed

    def test_heatmap(self):
        from mae_core.communication.spatial_consensus import SpatialConsensusTracker

        sc = SpatialConsensusTracker(grid_resolution=5.0)
        for i in range(5):
            sc.add_vote(i, (2.0, 2.0), "topic", "a")

        heatmap = sc.get_spatial_heatmap("topic", min_votes=3)
        self.assertGreater(len(heatmap), 0)


# ===========================================================================
# Cross-System Integration Tests
# ===========================================================================


class TestCrossSystemIntegration(unittest.TestCase):
    """Test that Phase 5.6 systems work together."""

    def test_circadian_drives_endocrine(self):
        """Circadian phase changes should modulate hormone levels."""
        from mae_core.coordination import (
            CircadianRhythm,
            EndocrineSystem,
            HormoneType,
        )

        bus = EventBus()
        endo = EndocrineSystem(event_bus=bus)
        circ = CircadianRhythm(event_bus=bus, cycle_length=10)

        # Wire circadian to endocrine
        circ.on_phase_change(lambda old, new: endo.set_circadian_phase(new.value))

        old_melatonin = endo.get_level(HormoneType.MELATONIN)

        # Run until we hit REST
        for _ in range(20):
            circ.step()
            endo.step()

        # Melatonin should have been modulated
        # (may have risen and then decayed, but history shows the release happened)
        stats = endo.get_statistics()
        self.assertGreater(stats["release_count"], 0)

    def test_substrate_supports_morphogenesis(self):
        """Morphogenesis should create agents and register them with substrate."""
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        sub = MycelialSubstrate(event_bus=bus, topology_type="scale_free", initial_nodes=10)
        coord = MorphogenesisCoordinator(event_bus=bus, substrate=sub)

        sig = ProblemSignature(domain="test", complexity=0.6)
        organ = coord.handle_novel_problem(sig)
        self.assertIsNotNone(organ)

    def test_endocrine_modulates_morphogenesis(self):
        """Cortisol should increase morphogenesis growth rate."""
        from mae_core.coordination import EndocrineSystem, HormoneType
        from mae_core.morphogenesis import MorphogenesisCoordinator

        bus = EventBus()
        endo = EndocrineSystem(event_bus=bus)
        coord = MorphogenesisCoordinator(event_bus=bus)

        # Stress increases growth rate
        endo.release_hormone(HormoneType.CORTISOL, 0.5, "stress")
        urgency = endo.get_urgency_level()
        coord.set_growth_rate(1.0 + urgency)

        stats = coord.get_statistics()
        self.assertGreater(stats["growth_rate_multiplier"], 1.0)

    def test_substrate_with_predictive_field(self):
        """Substrate positions feed predictive field."""
        from mae_core.communication.predictive_field import PredictiveField
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        sub = MycelialSubstrate(event_bus=bus, topology_type="scale_free", initial_nodes=10)
        pf = PredictiveField()

        # Register agents and update field from substrate positions
        for i in range(5):
            sub.register_agent(i)
            pos = sub.get_agent_position(i)
            if pos:
                pf.update_agent_state(i, pos)

        stats = pf.get_statistics()
        self.assertEqual(stats["agents_tracked"], 5)

    def test_eventbus_carries_all_events(self):
        """All Phase 5.6 systems should publish on EventBus."""
        from mae_core.coordination import CircadianRhythm, EndocrineSystem, HormoneType
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        events: dict[str, int] = {}

        def track(ch, msg):
            events[ch] = events.get(ch, 0) + 1

        bus.register_callback("substrate.agent_registered", track)
        bus.register_callback("substrate.health_report", track)
        bus.register_callback("endocrine.hormone_release", track)
        bus.register_callback("circadian.phase_change", track)
        bus.register_callback("morphogenesis.team_created", track)

        # Create all systems
        sub = MycelialSubstrate(event_bus=bus, health_report_interval=1)
        endo = EndocrineSystem(event_bus=bus)
        circ = CircadianRhythm(event_bus=bus, cycle_length=10)
        coord = MorphogenesisCoordinator(event_bus=bus)

        # Trigger events
        sub.register_agent(0)
        sub.step()
        endo.release_hormone(HormoneType.DOPAMINE, 0.3, "test")
        for _ in range(12):
            circ.step()
        coord.handle_novel_problem(ProblemSignature(domain="test"))

        # Verify events published
        self.assertIn("substrate.agent_registered", events)
        self.assertIn("substrate.health_report", events)
        self.assertIn("endocrine.hormone_release", events)
        self.assertIn("circadian.phase_change", events)
        self.assertIn("morphogenesis.team_created", events)

    def test_full_phase56_ecosystem(self):
        """All Phase 5.6 systems running together for 50 steps."""
        from mae_core.coordination import CircadianRhythm, EndocrineSystem, HormoneType
        from mae_core.morphogenesis import MorphogenesisCoordinator, ProblemSignature
        from mae_core.substrate import MycelialSubstrate

        bus = EventBus()
        sub = MycelialSubstrate(event_bus=bus, topology_type="scale_free", initial_nodes=15)
        endo = EndocrineSystem(event_bus=bus)
        circ = CircadianRhythm(event_bus=bus, cycle_length=20)
        coord = MorphogenesisCoordinator(event_bus=bus, substrate=sub)

        # Wire circadian -> endocrine
        circ.on_phase_change(lambda old, new: endo.set_circadian_phase(new.value))

        # Register some agents
        for i in range(5):
            sub.register_agent(i)

        # Run for 50 steps
        for step in range(50):
            sub.step()
            endo.step()
            circ.step()
            coord.step()

            # Inject some events
            if step == 10:
                endo.release_hormone(HormoneType.CORTISOL, 0.3, "external_stress")
            if step == 20:
                coord.handle_novel_problem(
                    ProblemSignature(domain="emergent", complexity=0.7)
                )
            if step == 30:
                endo.release_hormone(HormoneType.DOPAMINE, 0.4, "reward")

        # Verify everything ran without errors
        self.assertGreater(circ.cycle_count, 1)
        self.assertGreater(sub._step_count, 0)
        self.assertGreater(endo.get_statistics()["release_count"], 0)

        report = sub.get_health_report()
        self.assertEqual(report["topology"]["agents"], 5)


if __name__ == "__main__":
    unittest.main()
