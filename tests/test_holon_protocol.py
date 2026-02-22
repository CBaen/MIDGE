"""Tests for Holon Protocol - HolonRegistry, HolonMixin, HolonProxy, AwarenessPulse.

Unit tests for the fractal self-awareness system.
"""

import unittest
from unittest.mock import MagicMock

from mae_core.backbone.holon_mixin import ALL_HOLON_CAPABILITIES, HolonMixin
from mae_core.backbone.holon_protocol import (
    CH_AWARENESS_ANOMALY,
    CH_AWARENESS_PULSE,
    AwarenessPulse,
    HolonEntry,
    HolonProxy,
    HolonRegistry,
)


# =====================================================================
# HolonRegistry Tests
# =====================================================================

class TestHolonRegistryRegistration(unittest.TestCase):
    """Registration and unregistration."""

    def setUp(self):
        self.registry = HolonRegistry()

    def test_register_basic(self):
        entry = self.registry.register("node-1", holon_type="system")
        self.assertIsInstance(entry, HolonEntry)
        self.assertEqual(entry.holon_id, "node-1")
        self.assertEqual(entry.holon_type, "system")
        self.assertIsNone(entry.parent_id)

    def test_register_with_parent(self):
        self.registry.register("root", holon_type="organism")
        entry = self.registry.register("child", holon_type="agent", parent_id="root")
        self.assertEqual(entry.parent_id, "root")

    def test_register_with_capabilities(self):
        caps = {"sense", "decide", "act"}
        entry = self.registry.register("node", capabilities=caps)
        self.assertEqual(entry.capabilities, caps)

    def test_register_with_metadata(self):
        entry = self.registry.register("node", metadata={"layer": 3})
        self.assertEqual(entry.metadata["layer"], 3)

    def test_unregister_existing(self):
        self.registry.register("node")
        result = self.registry.unregister("node")
        self.assertTrue(result)
        self.assertIsNone(self.registry.get_entry("node"))

    def test_unregister_nonexistent(self):
        result = self.registry.unregister("ghost")
        self.assertFalse(result)

    def test_unregister_removes_from_parent_children(self):
        self.registry.register("parent")
        self.registry.register("child", parent_id="parent")
        self.assertIn("child", self.registry.get_children("parent"))
        self.registry.unregister("child")
        self.assertNotIn("child", self.registry.get_children("parent"))

    def test_unregister_orphans_children(self):
        self.registry.register("parent")
        self.registry.register("child", parent_id="parent")
        self.registry.unregister("parent")
        entry = self.registry.get_entry("child")
        self.assertIsNone(entry.parent_id)


class TestHolonRegistryParentChild(unittest.TestCase):
    """Parent/child relationship queries."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("colony", holon_type="colony", parent_id="mae")
        self.registry.register("agent-1", holon_type="agent", parent_id="colony")
        self.registry.register("agent-2", holon_type="agent", parent_id="colony")
        self.registry.register("agent-3", holon_type="agent", parent_id="colony")

    def test_get_parent(self):
        self.assertEqual(self.registry.get_parent("agent-1"), "colony")
        self.assertEqual(self.registry.get_parent("colony"), "mae")
        self.assertIsNone(self.registry.get_parent("mae"))

    def test_get_parent_unregistered(self):
        self.assertIsNone(self.registry.get_parent("ghost"))

    def test_get_children(self):
        children = self.registry.get_children("colony")
        self.assertEqual(children, ["agent-1", "agent-2", "agent-3"])

    def test_get_children_empty(self):
        children = self.registry.get_children("agent-1")
        self.assertEqual(children, [])

    def test_set_parent(self):
        self.registry.register("new-colony", holon_type="colony", parent_id="mae")
        result = self.registry.set_parent("agent-1", "new-colony")
        self.assertTrue(result)
        self.assertEqual(self.registry.get_parent("agent-1"), "new-colony")
        self.assertNotIn("agent-1", self.registry.get_children("colony"))
        self.assertIn("agent-1", self.registry.get_children("new-colony"))

    def test_set_parent_circular_rejected(self):
        result = self.registry.set_parent("mae", "agent-1")
        self.assertFalse(result)
        self.assertIsNone(self.registry.get_parent("mae"))

    def test_set_parent_nonexistent_holon(self):
        result = self.registry.set_parent("ghost", "mae")
        self.assertFalse(result)


class TestHolonRegistryPeers(unittest.TestCase):
    """Peer (sibling) queries."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("colony", holon_type="colony", parent_id="mae")
        self.registry.register("a1", holon_type="agent", parent_id="colony")
        self.registry.register("a2", holon_type="agent", parent_id="colony")
        self.registry.register("a3", holon_type="agent", parent_id="colony")

    def test_get_peers(self):
        peers = self.registry.get_peers("a1")
        self.assertEqual(peers, ["a2", "a3"])

    def test_get_peers_excludes_self(self):
        peers = self.registry.get_peers("a2")
        self.assertNotIn("a2", peers)

    def test_get_peers_only_child(self):
        self.registry.register("lonely-parent")
        self.registry.register("only-child", parent_id="lonely-parent")
        peers = self.registry.get_peers("only-child")
        self.assertEqual(peers, [])

    def test_get_peers_no_parent(self):
        peers = self.registry.get_peers("mae")
        self.assertEqual(peers, [])


class TestHolonRegistryTraversal(unittest.TestCase):
    """Ancestry and subtree traversal."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("colony", holon_type="colony", parent_id="mae")
        self.registry.register("agent-1", holon_type="agent", parent_id="colony")

    def test_get_ancestry(self):
        chain = self.registry.get_ancestry("agent-1")
        self.assertEqual(chain, ["colony", "mae"])

    def test_get_ancestry_root(self):
        chain = self.registry.get_ancestry("mae")
        self.assertEqual(chain, [])

    def test_get_subtree(self):
        subtree = self.registry.get_subtree("mae")
        self.assertIn("colony", subtree)
        self.assertIn("agent-1", subtree)

    def test_get_subtree_leaf(self):
        subtree = self.registry.get_subtree("agent-1")
        self.assertEqual(subtree, [])

    def test_get_all_ids(self):
        ids = self.registry.get_all_ids()
        self.assertEqual(ids, ["agent-1", "colony", "mae"])


class TestHolonRegistryStatistics(unittest.TestCase):
    """Registry statistics."""

    def test_statistics(self):
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        registry.register("colony", holon_type="colony", parent_id="mae")
        registry.register("a1", holon_type="agent", parent_id="colony")
        registry.register("a2", holon_type="agent", parent_id="colony")

        stats = registry.get_statistics()
        self.assertEqual(stats["total_holons"], 4)
        self.assertEqual(stats["type_counts"]["agent"], 2)
        self.assertEqual(stats["type_counts"]["organism"], 1)
        self.assertEqual(stats["roots"], ["mae"])
        self.assertEqual(stats["max_depth"], 2)  # a1 → colony → mae


# =====================================================================
# HolonMixin Tests
# =====================================================================

class MockAgent(HolonMixin):
    """Minimal mock agent for testing HolonMixin in isolation."""

    def __init__(self, unique_id=42, **kwargs):
        self.unique_id = unique_id
        self.step_count = 0
        self.last_reward = 0.0
        self.current_state = {"x": 1.0}
        self.reward_history = []
        self._init_holon(**kwargs)

    def get_state(self):
        return {"unique_id": self.unique_id, "step_count": self.step_count}

    def _decide(self):
        return 0

    def _act(self, action):
        return 1.0

    def _learn(self, action, reward):
        pass

    def get_performance_summary(self):
        return {"mean_reward": 0.5}


class TestHolonMixinInit(unittest.TestCase):
    """HolonMixin initialization."""

    def test_init_defaults(self):
        agent = MockAgent()
        self.assertEqual(agent._holon_id, "42")
        self.assertIsNone(agent._holon_parent_id)
        self.assertIsInstance(agent._holon_capabilities, set)

    def test_init_with_explicit_id(self):
        agent = MockAgent(holon_id="custom-id")
        self.assertEqual(agent._holon_id, "custom-id")

    def test_init_with_registry(self):
        registry = HolonRegistry()
        agent = MockAgent(holon_registry=registry, parent_id="colony")
        self.assertEqual(agent._holon_parent_id, "colony")
        self.assertIs(agent._holon_registry, registry)

    def test_capabilities_detected(self):
        agent = MockAgent()
        # MockAgent has _decide, _act, _learn + always-on capabilities
        self.assertEqual(agent._holon_capabilities, ALL_HOLON_CAPABILITIES)


class TestHolonMixinCapabilities(unittest.TestCase):
    """The 10 holon capability methods."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("colony", holon_type="colony")
        self.registry.register("42", holon_type="agent", parent_id="colony")
        self.registry.register("43", holon_type="agent", parent_id="colony")
        self.agent = MockAgent(holon_registry=self.registry, parent_id="colony")

    def test_holon_sense(self):
        result = self.agent.holon_sense()
        self.assertIn("self", result)
        self.assertIn("step_count", result)

    def test_holon_remember_store_and_retrieve(self):
        self.agent.holon_remember("color", "blue")
        self.assertEqual(self.agent.holon_remember("color"), "blue")

    def test_holon_remember_missing_key(self):
        self.assertIsNone(self.agent.holon_remember("nonexistent"))

    def test_holon_decide(self):
        result = self.agent.holon_decide()
        self.assertEqual(result, 0)

    def test_holon_act(self):
        reward = self.agent.holon_act(action=1)
        self.assertEqual(reward, 1.0)

    def test_holon_learn_no_error(self):
        self.agent.holon_learn(action=0, reward=1.0)

    def test_holon_heal_healthy(self):
        self.agent.step_count = 1  # Agent has stepped at least once
        report = self.agent.holon_heal()
        self.assertTrue(report["healthy"])
        self.assertEqual(report["issues"], [])

    def test_holon_heal_declining_reward(self):
        self.agent.reward_history = [1.0]*5 + [0.1]*5
        report = self.agent.holon_heal()
        self.assertFalse(report["healthy"])
        self.assertIn("reward_declining", report["issues"])

    def test_holon_know_self(self):
        result = self.agent.holon_know_self()
        self.assertEqual(result["holon_id"], "42")
        self.assertEqual(result["holon_type"], "agent")
        self.assertIn("capabilities", result)
        self.assertIn("performance", result)
        self.assertEqual(result["peers_count"], 1)  # agent 43

    def test_holon_know_up(self):
        result = self.agent.holon_know_up()
        self.assertIsNotNone(result)
        self.assertEqual(result["parent_id"], "colony")
        self.assertEqual(result["parent_type"], "colony")

    def test_holon_know_up_no_parent(self):
        agent = MockAgent()
        result = agent.holon_know_up()
        self.assertIsNone(result)

    def test_holon_know_down(self):
        result = self.agent.holon_know_down()
        self.assertEqual(result, [])  # agents are leaf holons

    def test_holon_know_peers(self):
        peers = self.agent.holon_know_peers()
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["holon_id"], "43")

    def test_holon_know_peers_no_registry(self):
        agent = MockAgent()
        peers = agent.holon_know_peers()
        self.assertEqual(peers, [])


class TestHolonMixinSerialization(unittest.TestCase):
    """Persistence roundtrip."""

    def test_serialize(self):
        agent = MockAgent(parent_id="colony")
        agent.holon_remember("key", "val")
        state = agent._serialize_holon()
        self.assertEqual(state["holon_id"], "42")
        self.assertEqual(state["parent_id"], "colony")
        self.assertEqual(state["holon_memory"]["key"], "val")

    def test_restore(self):
        agent = MockAgent()
        agent._restore_holon({
            "holon_id": "restored-id",
            "parent_id": "restored-parent",
            "holon_memory": {"restored_key": 99},
        })
        self.assertEqual(agent._holon_id, "restored-id")
        self.assertEqual(agent._holon_parent_id, "restored-parent")
        self.assertEqual(agent._holon_memory["restored_key"], 99)

    def test_roundtrip(self):
        agent = MockAgent(parent_id="colony")
        agent.holon_remember("test", [1, 2, 3])
        state = agent._serialize_holon()

        agent2 = MockAgent()
        agent2._restore_holon(state)
        self.assertEqual(agent2._holon_id, "42")
        self.assertEqual(agent2._holon_parent_id, "colony")
        self.assertEqual(agent2._holon_memory["test"], [1, 2, 3])


class TestHolonMixinStatistics(unittest.TestCase):
    """get_holon_statistics()."""

    def test_statistics_without_registry(self):
        agent = MockAgent()
        stats = agent.get_holon_statistics()
        self.assertEqual(stats["holon_id"], "42")
        self.assertIn("capabilities", stats)

    def test_statistics_with_registry(self):
        registry = HolonRegistry()
        registry.register("colony")
        registry.register("42", parent_id="colony")
        registry.register("43", parent_id="colony")
        agent = MockAgent(holon_registry=registry, parent_id="colony")
        stats = agent.get_holon_statistics()
        self.assertEqual(stats["peers_count"], 1)
        self.assertEqual(stats["children_count"], 0)


# =====================================================================
# HolonProxy Tests
# =====================================================================

class TestHolonProxy(unittest.TestCase):
    """HolonProxy — lightweight awareness for non-agent systems."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("substrate", holon_type="system", parent_id="mae")
        self.registry.register("endocrine", holon_type="system", parent_id="mae")
        self.registry.register("circadian", holon_type="system", parent_id="mae")

    def test_know_self(self):
        proxy = HolonProxy("substrate", self.registry)
        info = proxy.know_self()
        self.assertEqual(info["holon_id"], "substrate")
        self.assertEqual(info["holon_type"], "system")
        self.assertEqual(info["parent_id"], "mae")
        self.assertEqual(info["peers_count"], 2)  # endocrine + circadian
        self.assertEqual(info["health"], 1.0)

    def test_know_self_unregistered(self):
        proxy = HolonProxy("ghost", self.registry)
        info = proxy.know_self()
        self.assertFalse(info["registered"])

    def test_know_up(self):
        proxy = HolonProxy("substrate", self.registry)
        parent = proxy.know_up()
        self.assertIsNotNone(parent)
        self.assertEqual(parent["parent_id"], "mae")
        self.assertEqual(parent["parent_type"], "organism")
        self.assertEqual(parent["parent_children_count"], 3)

    def test_know_up_root(self):
        proxy = HolonProxy("mae", self.registry)
        self.assertIsNone(proxy.know_up())

    def test_know_down(self):
        proxy = HolonProxy("mae", self.registry)
        children = proxy.know_down()
        self.assertEqual(len(children), 3)
        child_ids = {c["holon_id"] for c in children}
        self.assertEqual(child_ids, {"substrate", "endocrine", "circadian"})

    def test_know_down_leaf(self):
        proxy = HolonProxy("substrate", self.registry)
        self.assertEqual(proxy.know_down(), [])

    def test_know_peers(self):
        proxy = HolonProxy("substrate", self.registry)
        peers = proxy.know_peers()
        self.assertEqual(len(peers), 2)
        peer_ids = {p["holon_id"] for p in peers}
        self.assertEqual(peer_ids, {"endocrine", "circadian"})

    def test_know_peers_only_child(self):
        reg = HolonRegistry()
        reg.register("parent")
        reg.register("only", parent_id="parent")
        proxy = HolonProxy("only", reg)
        self.assertEqual(proxy.know_peers(), [])

    def test_get_health_no_somatic_map(self):
        proxy = HolonProxy("substrate", self.registry)
        self.assertEqual(proxy.get_health(), 1.0)

    def test_get_connections_no_registry(self):
        proxy = HolonProxy("substrate", self.registry)
        self.assertEqual(proxy.get_connections(), [])

    def test_repr(self):
        proxy = HolonProxy("substrate", self.registry)
        self.assertEqual(repr(proxy), "HolonProxy('substrate')")


class TestHolonProxyWithSomaticMap(unittest.TestCase):
    """HolonProxy health queries via SomaticMap."""

    def test_health_from_somatic_map(self):
        registry = HolonRegistry()
        registry.register("test_sys", parent_id=None)

        mock_sm = MagicMock()
        mock_info = MagicMock()
        mock_info.health = 0.75
        mock_sm.get_system_info.return_value = mock_info

        proxy = HolonProxy("test_sys", registry, somatic_map=mock_sm)
        self.assertEqual(proxy.get_health(), 0.75)

    def test_health_somatic_map_missing_system(self):
        registry = HolonRegistry()
        registry.register("test_sys", parent_id=None)

        mock_sm = MagicMock()
        mock_sm.get_system_info.return_value = None

        proxy = HolonProxy("test_sys", registry, somatic_map=mock_sm)
        self.assertEqual(proxy.get_health(), 1.0)


class TestHolonProxyWithConnectionRegistry(unittest.TestCase):
    """HolonProxy connection queries via ConnectionRegistry."""

    def test_connections_returned(self):
        registry = HolonRegistry()
        registry.register("source", parent_id=None)

        mock_conn = MagicMock()
        mock_triad = MagicMock()
        mock_triad.connection_id = "source->target"
        mock_triad.source = "source"
        mock_triad.target = "target"
        mock_triad.witness = "witness"
        mock_triad.connection_type = MagicMock(value="eventbus_pubsub")
        mock_conn.get_connections_for_system.return_value = [mock_triad]

        proxy = HolonProxy("source", registry, connection_registry=mock_conn)
        conns = proxy.get_connections()
        self.assertEqual(len(conns), 1)
        self.assertEqual(conns[0]["source"], "source")
        self.assertEqual(conns[0]["target"], "target")
        self.assertEqual(conns[0]["witness"], "witness")


class TestHolonRegistryProxy(unittest.TestCase):
    """HolonRegistry.get_proxy() and inject methods."""

    def test_get_proxy_creates_proxy(self):
        registry = HolonRegistry()
        registry.register("test")
        proxy = registry.get_proxy("test")
        self.assertIsInstance(proxy, HolonProxy)
        self.assertEqual(proxy.holon_id, "test")

    def test_get_proxy_cached(self):
        registry = HolonRegistry()
        registry.register("test")
        p1 = registry.get_proxy("test")
        p2 = registry.get_proxy("test")
        self.assertIs(p1, p2)

    def test_set_somatic_map_propagates(self):
        registry = HolonRegistry()
        registry.register("test")
        mock_sm = MagicMock()
        registry.set_somatic_map(mock_sm)
        proxy = registry.get_proxy("test")
        self.assertIs(proxy._somatic_map, mock_sm)

    def test_set_connection_registry_propagates(self):
        registry = HolonRegistry()
        registry.register("test")
        mock_cr = MagicMock()
        registry.set_connection_registry(mock_cr)
        proxy = registry.get_proxy("test")
        self.assertIs(proxy._connection_registry, mock_cr)

    def test_proxy_updates_references_on_reget(self):
        registry = HolonRegistry()
        registry.register("test")
        p1 = registry.get_proxy("test")
        self.assertIsNone(p1._somatic_map)

        mock_sm = MagicMock()
        registry.set_somatic_map(mock_sm)
        p2 = registry.get_proxy("test")
        self.assertIs(p1, p2)  # Same object
        self.assertIs(p2._somatic_map, mock_sm)  # Updated reference


# =====================================================================
# HolonProxy — 5 completing capabilities (triadic review 2026-02-12)
# =====================================================================

class TestHolonProxySense(unittest.TestCase):
    """HolonProxy.sense() — perceive operational state."""

    def test_sense_no_system_ref(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        result = proxy.sense()
        self.assertFalse(result["operational"])

    def test_sense_delegates_to_get_state(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock()
        system.get_state.return_value = {"temperature": 37.0, "active": True}
        proxy.set_system_ref(system)
        result = proxy.sense()
        self.assertEqual(result["temperature"], 37.0)
        system.get_state.assert_called_once()

    def test_sense_falls_back_to_get_statistics(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock(spec=[])  # No get_state
        system.get_statistics = MagicMock(return_value={"count": 5})
        proxy.set_system_ref(system)
        result = proxy.sense()
        self.assertEqual(result["count"], 5)

    def test_sense_returns_minimal_when_no_method(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.set_system_ref(object())  # No methods
        result = proxy.sense()
        self.assertTrue(result["operational"])
        self.assertEqual(result["holon_id"], "test")


class TestHolonProxyRemember(unittest.TestCase):
    """HolonProxy.remember() — per-proxy memory store."""

    def test_store_and_retrieve(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.remember("key1", "value1")
        self.assertEqual(proxy.remember("key1"), "value1")

    def test_retrieve_missing_returns_none(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        self.assertIsNone(proxy.remember("nonexistent"))

    def test_store_returns_value(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        result = proxy.remember("key", 42)
        self.assertEqual(result, 42)

    def test_overwrite(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.remember("key", "old")
        proxy.remember("key", "new")
        self.assertEqual(proxy.remember("key"), "new")


class TestHolonProxyDecide(unittest.TestCase):
    """HolonProxy.decide() — delegate to system decision method."""

    def test_decide_no_system_ref(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        self.assertIsNone(proxy.decide())

    def test_decide_delegates_to_decide_method(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock()
        system.decide.return_value = "action_A"
        proxy.set_system_ref(system)
        self.assertEqual(proxy.decide(), "action_A")

    def test_decide_with_stimulus(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock()
        system.decide.return_value = "react"
        proxy.set_system_ref(system)
        result = proxy.decide(stimulus={"threat": 0.8})
        system.decide.assert_called_once_with({"threat": 0.8})
        self.assertEqual(result, "react")

    def test_decide_falls_back_to_evaluate(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock(spec=[])
        system.evaluate = MagicMock(return_value="evaluated")
        proxy.set_system_ref(system)
        self.assertEqual(proxy.decide(), "evaluated")

    def test_decide_no_op_when_no_method(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.set_system_ref(object())
        self.assertIsNone(proxy.decide())


class TestHolonProxyLearn(unittest.TestCase):
    """HolonProxy.learn() — delegate to system learning method."""

    def test_learn_no_system_ref(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.learn()  # Should not raise

    def test_learn_delegates_to_learn_method(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock()
        proxy.set_system_ref(system)
        proxy.learn(feedback={"reward": 0.8})
        system.learn.assert_called_once_with({"reward": 0.8})

    def test_learn_falls_back_to_adapt(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        system = MagicMock(spec=[])
        system.adapt = MagicMock()
        proxy.set_system_ref(system)
        proxy.learn()
        system.adapt.assert_called_once()

    def test_learn_no_op_when_no_method(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)
        proxy.set_system_ref(object())
        proxy.learn()  # Should not raise


class TestHolonProxyHeal(unittest.TestCase):
    """HolonProxy.heal() — self-assessment and recovery."""

    def test_heal_healthy_system(self):
        reg = HolonRegistry()
        reg.register("test")
        proxy = HolonProxy("test", reg)  # No somatic map -> health = 1.0
        report = proxy.heal()
        self.assertTrue(report["healthy"])
        self.assertEqual(report["health"], 1.0)
        self.assertEqual(report["issues"], [])
        self.assertFalse(report["recovery_attempted"])

    def test_heal_degraded_attempts_recovery(self):
        reg = HolonRegistry()
        reg.register("test")
        mock_sm = MagicMock()
        mock_info = MagicMock()
        mock_info.health = 0.3
        mock_sm.get_system_info.return_value = mock_info

        proxy = HolonProxy("test", reg, somatic_map=mock_sm)
        system = MagicMock()
        system.reset = MagicMock()
        proxy.set_system_ref(system)

        report = proxy.heal()
        self.assertFalse(report["healthy"])
        self.assertTrue(report["recovery_attempted"])
        self.assertEqual(report["recovery_method"], "reset")
        system.reset.assert_called_once()

    def test_heal_degraded_tries_recover_fallback(self):
        reg = HolonRegistry()
        reg.register("test")
        mock_sm = MagicMock()
        mock_info = MagicMock()
        mock_info.health = 0.2
        mock_sm.get_system_info.return_value = mock_info

        proxy = HolonProxy("test", reg, somatic_map=mock_sm)
        system = MagicMock(spec=[])
        system.recover = MagicMock()
        proxy.set_system_ref(system)

        report = proxy.heal()
        self.assertTrue(report["recovery_attempted"])
        self.assertEqual(report["recovery_method"], "recover")

    def test_heal_reports_to_somatic_map(self):
        reg = HolonRegistry()
        reg.register("test")
        mock_sm = MagicMock()
        mock_sm.get_system_info.return_value = None  # health = 1.0

        proxy = HolonProxy("test", reg, somatic_map=mock_sm)
        proxy.heal()
        mock_sm.heartbeat.assert_called()

    def test_heal_no_recovery_method_still_reports(self):
        reg = HolonRegistry()
        reg.register("test")
        mock_sm = MagicMock()
        mock_info = MagicMock()
        mock_info.health = 0.4
        mock_sm.get_system_info.return_value = mock_info

        proxy = HolonProxy("test", reg, somatic_map=mock_sm)
        proxy.set_system_ref(object())  # No recovery methods

        report = proxy.heal()
        self.assertFalse(report["healthy"])
        self.assertFalse(report["recovery_attempted"])
        self.assertIn("health_degraded", report["issues"])


# =====================================================================
# AwarenessPulse Tests
# =====================================================================

class MockEventBus:
    """Minimal EventBus for testing AwarenessPulse."""

    def __init__(self):
        self.published: list[tuple[str, dict]] = []

    def publish(self, channel: str, message: dict) -> None:
        self.published.append((channel, message))


class TestAwarenessPulse(unittest.TestCase):
    """AwarenessPulse — periodic hierarchy health check."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("substrate", holon_type="system", parent_id="mae")
        self.registry.register("endocrine", holon_type="system", parent_id="mae")
        self.bus = MockEventBus()

    def test_pulse_fires_at_interval(self):
        pulse = AwarenessPulse(self.registry, self.bus, interval=5)
        for _ in range(4):
            pulse.step()
        self.assertEqual(len(self.bus.published), 0)
        pulse.step()  # Step 5 — fires
        self.assertEqual(len(self.bus.published), 1)
        self.assertEqual(self.bus.published[0][0], CH_AWARENESS_PULSE)

    def test_pulse_summary_contents(self):
        pulse = AwarenessPulse(self.registry, self.bus, interval=1)
        pulse.step()
        _, summary = self.bus.published[0]
        self.assertEqual(summary["pulse_number"], 1)
        self.assertEqual(summary["step"], 1)
        self.assertGreater(summary["total_holons"], 0)

    def test_no_anomalies_on_healthy_hierarchy(self):
        pulse = AwarenessPulse(self.registry, self.bus, interval=1)
        pulse.step()
        # No orphans, no health issues — only pulse, no anomaly
        anomaly_msgs = [m for ch, m in self.bus.published if ch == CH_AWARENESS_ANOMALY]
        self.assertEqual(len(anomaly_msgs), 0)

    def test_orphan_detected(self):
        # Create orphan: parent "ghost" doesn't exist
        self.registry.register("orphan", holon_type="system", parent_id="ghost")
        pulse = AwarenessPulse(self.registry, self.bus, interval=1)
        pulse.step()
        anomaly_msgs = [m for ch, m in self.bus.published if ch == CH_AWARENESS_ANOMALY]
        self.assertEqual(len(anomaly_msgs), 1)
        anomalies = anomaly_msgs[0]["anomalies"]
        orphan_findings = [a for a in anomalies if a["type"] == "orphaned_systems"]
        self.assertEqual(len(orphan_findings), 1)
        self.assertIn("orphan", orphan_findings[0]["holon_ids"])

    def test_statistics(self):
        pulse = AwarenessPulse(self.registry, self.bus, interval=5)
        for _ in range(10):
            pulse.step()
        stats = pulse.get_statistics()
        self.assertEqual(stats["pulse_count"], 2)
        self.assertEqual(stats["step_count"], 10)
        self.assertEqual(stats["interval"], 5)


if __name__ == "__main__":
    unittest.main()
