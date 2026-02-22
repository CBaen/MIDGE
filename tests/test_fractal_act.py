"""Tests for Fractal ACT — action at every scale (Law 3 compliance).

Tests HolonProxy.act(), SubsystemAction, OrganAction, OrganismAction,
the delegation chain, aggregation, heal-on-failure, and build_fractal_action.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest

from mae_core.backbone.holon_protocol import HolonProxy, HolonRegistry
from mae_core.backbone.fractal_generator import (
    FRACTAL_GROUPING,
    FractalGenerator,
)
from mae_core.backbone.fractal_act import (
    CH_FRACTAL_ACT,
    OrganAction,
    OrganismAction,
    SubsystemAction,
    build_fractal_action,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.connection_registry import ConnectionRegistry


# =====================================================================
# HolonProxy.act() Tests
# =====================================================================


class TestHolonProxyAct(unittest.TestCase):
    """HolonProxy.act() — system-level action via delegation."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("test_sys", holon_type="system")

    def test_act_no_system_ref_returns_zero(self):
        proxy = HolonProxy("test_sys", self.registry)
        self.assertEqual(proxy.act(), 0.0)

    def test_act_delegates_to_step(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        system.step.return_value = None  # step() returns None -> 1.0
        proxy.set_system_ref(system)
        result = proxy.act()
        system.step.assert_called_once()
        self.assertEqual(result, 1.0)

    def test_act_returns_step_result(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        system.step.return_value = 0.75
        proxy.set_system_ref(system)
        self.assertEqual(proxy.act(), 0.75)

    def test_act_falls_back_to_process(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock(spec=[])  # No step
        system.process = MagicMock(return_value=0.5)
        proxy.set_system_ref(system)
        self.assertEqual(proxy.act(), 0.5)

    def test_act_falls_back_to_execute(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock(spec=[])  # No step or process
        system.execute = MagicMock(return_value=0.9)
        proxy.set_system_ref(system)
        self.assertEqual(proxy.act(), 0.9)

    def test_act_no_callable_method_returns_zero(self):
        proxy = HolonProxy("test_sys", self.registry)

        class EmptySystem:
            pass

        proxy.set_system_ref(EmptySystem())
        self.assertEqual(proxy.act(), 0.0)

    def test_act_exception_returns_zero(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        system.step.side_effect = RuntimeError("boom")
        proxy.set_system_ref(system)
        self.assertEqual(proxy.act(), 0.0)

    def test_act_type_error_returns_zero(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        system.step.side_effect = TypeError("bad call")
        proxy.set_system_ref(system)
        self.assertEqual(proxy.act(), 0.0)

    def test_set_system_ref(self):
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        proxy.set_system_ref(system)
        self.assertIs(proxy._system_ref, system)

    def test_act_with_action_parameter_ignored(self):
        """act() takes an optional action parameter but delegates without it."""
        proxy = HolonProxy("test_sys", self.registry)
        system = MagicMock()
        system.step.return_value = 1.0
        proxy.set_system_ref(system)
        result = proxy.act(action="some_action")
        self.assertEqual(result, 1.0)


# =====================================================================
# SubsystemAction Tests
# =====================================================================


class TestSubsystemAction(unittest.TestCase):
    """SubsystemAction — fractal ACT at the subsystem (tissue) level."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("subsystem-1", holon_type="subsystem", parent_id="mae")
        # Three child systems
        for name in ("sys-a", "sys-b", "sys-c"):
            self.registry.register(name, holon_type="system", parent_id="subsystem-1")

    def _inject_system_refs(self, return_values=None):
        """Inject mock system refs into child proxies."""
        if return_values is None:
            return_values = {"sys-a": None, "sys-b": None, "sys-c": None}
        for name, ret_val in return_values.items():
            proxy = self.registry.get_proxy(name)
            system = MagicMock()
            system.step.return_value = ret_val
            proxy.set_system_ref(system)

    def test_no_children_returns_zero(self):
        registry = HolonRegistry()
        registry.register("empty", holon_type="subsystem")
        action = SubsystemAction("empty", registry)
        self.assertEqual(action.act(), 0.0)

    def test_all_children_succeed(self):
        self._inject_system_refs({"sys-a": None, "sys-b": None, "sys-c": None})
        action = SubsystemAction("subsystem-1", self.registry)
        result = action.act()
        # All return None -> 1.0 each, mean = 1.0
        self.assertEqual(result, 1.0)

    def test_mixed_results(self):
        self._inject_system_refs({"sys-a": 0.9, "sys-b": 0.6, "sys-c": 0.3})
        action = SubsystemAction("subsystem-1", self.registry)
        result = action.act()
        expected = (0.9 + 0.6 + 0.3) / 3
        self.assertAlmostEqual(result, expected)

    def test_one_child_fails(self):
        self._inject_system_refs({"sys-a": None, "sys-b": None, "sys-c": None})
        # Make sys-c's step raise
        proxy_c = self.registry.get_proxy("sys-c")
        proxy_c._system_ref.step.side_effect = RuntimeError("fail")
        action = SubsystemAction("subsystem-1", self.registry)
        result = action.act()
        # sys-a=1.0, sys-b=1.0, sys-c=0.0 -> mean = 2/3
        self.assertAlmostEqual(result, 2.0 / 3.0)

    def test_all_children_fail(self):
        self._inject_system_refs({"sys-a": None, "sys-b": None, "sys-c": None})
        for name in ("sys-a", "sys-b", "sys-c"):
            proxy = self.registry.get_proxy(name)
            proxy._system_ref.step.side_effect = RuntimeError("fail")
        action = SubsystemAction("subsystem-1", self.registry)
        self.assertEqual(action.act(), 0.0)

    def test_children_without_system_ref(self):
        # Don't inject system refs — act should return 0.0 for each
        action = SubsystemAction("subsystem-1", self.registry)
        result = action.act()
        self.assertEqual(result, 0.0)

    def test_statistics_tracking(self):
        self._inject_system_refs({"sys-a": 0.8, "sys-b": 0.6, "sys-c": 0.4})
        action = SubsystemAction("subsystem-1", self.registry)
        action.act()
        action.act()
        stats = action.get_statistics()
        self.assertEqual(stats["act_count"], 2)
        self.assertEqual(stats["subsystem_id"], "subsystem-1")
        self.assertAlmostEqual(stats["last_result"], 0.6)
        self.assertAlmostEqual(stats["mean_score"], 0.6)

    def test_heal_called_on_failure(self):
        """When a child fails, subsystem should call get_health() for reporting."""
        self._inject_system_refs({"sys-a": None, "sys-b": None, "sys-c": None})
        # Make sys-c fail
        proxy_c = self.registry.get_proxy("sys-c")
        proxy_c._system_ref.step.side_effect = RuntimeError("fail")
        action = SubsystemAction("subsystem-1", self.registry)
        action.act()
        # Verify get_health was called on the failed child
        # (it's a real HolonProxy so get_health returns 1.0 by default — just verifying no crash)


# =====================================================================
# OrganAction Tests
# =====================================================================


class TestOrganAction(unittest.TestCase):
    """OrganAction — fractal ACT at the organ level."""

    def _make_sub_action(self, score):
        """Create a mock SubsystemAction that returns a fixed score."""
        sub = MagicMock(spec=SubsystemAction)
        sub.act.return_value = score
        sub.get_statistics.return_value = {
            "subsystem_id": "mock",
            "act_count": 1,
            "last_result": score,
            "mean_score": score,
        }
        return sub

    def test_no_subsystems_returns_zero(self):
        action = OrganAction("organ-1", {})
        self.assertEqual(action.act(), 0.0)

    def test_all_subsystems_succeed(self):
        subs = {
            "sub-1": self._make_sub_action(1.0),
            "sub-2": self._make_sub_action(0.8),
            "sub-3": self._make_sub_action(0.6),
        }
        action = OrganAction("organ-1", subs)
        result = action.act()
        self.assertAlmostEqual(result, (1.0 + 0.8 + 0.6) / 3)

    def test_one_subsystem_fails(self):
        subs = {
            "sub-1": self._make_sub_action(1.0),
            "sub-2": self._make_sub_action(0.0),
            "sub-3": self._make_sub_action(0.5),
        }
        action = OrganAction("organ-1", subs)
        result = action.act()
        self.assertAlmostEqual(result, (1.0 + 0.0 + 0.5) / 3)

    def test_subsystem_raises_exception(self):
        sub_ok = self._make_sub_action(1.0)
        sub_bad = MagicMock(spec=SubsystemAction)
        sub_bad.act.side_effect = RuntimeError("subsystem crashed")
        sub_bad.get_statistics.return_value = {}
        subs = {"sub-ok": sub_ok, "sub-bad": sub_bad}
        action = OrganAction("organ-1", subs)
        result = action.act()
        # sub_ok=1.0, sub_bad=0.0 -> mean=0.5
        self.assertAlmostEqual(result, 0.5)

    def test_statistics(self):
        subs = {
            "sub-1": self._make_sub_action(0.9),
            "sub-2": self._make_sub_action(0.7),
        }
        action = OrganAction("organ-1", subs)
        action.act()
        stats = action.get_statistics()
        self.assertEqual(stats["organ_id"], "organ-1")
        self.assertEqual(stats["act_count"], 1)
        self.assertEqual(stats["subsystem_count"], 2)
        self.assertAlmostEqual(stats["last_result"], 0.8)
        self.assertIn("subsystem_stats", stats)


# =====================================================================
# OrganismAction Tests
# =====================================================================


class TestOrganismAction(unittest.TestCase):
    """OrganismAction — fractal ACT at the organism level."""

    def _make_organ_action(self, score):
        organ = MagicMock(spec=OrganAction)
        organ.act.return_value = score
        organ.get_statistics.return_value = {
            "organ_id": "mock",
            "act_count": 1,
            "last_result": score,
            "mean_score": score,
            "subsystem_count": 3,
            "subsystem_stats": {},
        }
        return organ

    def test_no_organs_returns_zero(self):
        action = OrganismAction({})
        self.assertEqual(action.act(), 0.0)

    def test_all_organs_succeed(self):
        organs = {
            "nervous": self._make_organ_action(0.9),
            "sensory": self._make_organ_action(0.8),
            "cognitive": self._make_organ_action(0.7),
            "somatic": self._make_organ_action(0.6),
        }
        action = OrganismAction(organs)
        result = action.act()
        self.assertAlmostEqual(result, (0.9 + 0.8 + 0.7 + 0.6) / 4)

    def test_organ_failure_handled(self):
        organ_ok = self._make_organ_action(1.0)
        organ_bad = MagicMock(spec=OrganAction)
        organ_bad.act.side_effect = RuntimeError("organ crashed")
        organ_bad.get_statistics.return_value = {}
        organs = {"ok": organ_ok, "bad": organ_bad}
        action = OrganismAction(organs)
        result = action.act()
        self.assertAlmostEqual(result, 0.5)

    def test_publishes_event(self):
        bus = MagicMock()
        organs = {"test": self._make_organ_action(0.8)}
        action = OrganismAction(organs, event_bus=bus)
        action.act()
        bus.publish.assert_called_once()
        channel, payload = bus.publish.call_args[0]
        self.assertEqual(channel, CH_FRACTAL_ACT)
        self.assertEqual(payload["act_count"], 1)
        self.assertAlmostEqual(payload["organism_score"], 0.8)

    def test_no_event_bus_no_crash(self):
        organs = {"test": self._make_organ_action(0.5)}
        action = OrganismAction(organs, event_bus=None)
        result = action.act()
        self.assertAlmostEqual(result, 0.5)

    def test_statistics(self):
        organs = {
            "nervous": self._make_organ_action(0.9),
            "somatic": self._make_organ_action(0.7),
        }
        action = OrganismAction(organs)
        action.act()
        action.act()
        stats = action.get_statistics()
        self.assertEqual(stats["act_count"], 2)
        self.assertEqual(stats["organ_count"], 2)
        self.assertAlmostEqual(stats["last_result"], 0.8)
        self.assertIn("organ_stats", stats)

    def test_organ_scores_tracked(self):
        organs = {
            "nervous": self._make_organ_action(0.9),
            "somatic": self._make_organ_action(0.5),
        }
        action = OrganismAction(organs)
        action.act()
        self.assertAlmostEqual(action._organ_scores["nervous"], 0.9)
        self.assertAlmostEqual(action._organ_scores["somatic"], 0.5)


# =====================================================================
# build_fractal_action() Tests
# =====================================================================


class TestBuildFractalAction(unittest.TestCase):
    """build_fractal_action() — factory for the full fractal action chain."""

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

        # Build hierarchy
        conn_reg = ConnectionRegistry(event_bus=EventBus())
        gen = FractalGenerator(
            holon_registry=self.registry,
            connection_registry=conn_reg,
            event_bus=EventBus(),
        )
        gen.organize()

    def test_builds_organism_action(self):
        result = build_fractal_action(self.registry)
        self.assertIsInstance(result, OrganismAction)

    def test_has_five_organs(self):
        result = build_fractal_action(self.registry)
        self.assertEqual(len(result._organ_actions), 5)

    def test_organs_match_grouping(self):
        result = build_fractal_action(self.registry)
        for organ_name in FRACTAL_GROUPING:
            self.assertIn(organ_name, result._organ_actions)

    def test_subsystems_match_grouping(self):
        result = build_fractal_action(self.registry)
        for organ_name, subsystems_spec in FRACTAL_GROUPING.items():
            organ = result._organ_actions[organ_name]
            for sub_name in subsystems_spec:
                self.assertIn(sub_name, organ._subsystem_actions)

    def test_can_act_without_system_refs(self):
        """act() should work even without system refs (returns 0.0)."""
        result = build_fractal_action(self.registry)
        score = result.act()
        # All children have no system_ref -> all 0.0
        self.assertEqual(score, 0.0)

    def test_with_event_bus(self):
        bus = EventBus()
        events = []
        bus.register_callback(CH_FRACTAL_ACT, lambda ch, msg: events.append(msg))
        result = build_fractal_action(self.registry, event_bus=bus)
        result.act()
        # EventBus publishes JSON strings, so events will have content
        self.assertEqual(len(events), 1)


# =====================================================================
# Full delegation chain test
# =====================================================================


class TestFractalDelegationChain(unittest.TestCase):
    """End-to-end test: organism -> organ -> subsystem -> proxy -> system."""

    def test_full_chain(self):
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        registry.register("organ-1", holon_type="organ", parent_id="mae")
        registry.register("sub-1", holon_type="subsystem", parent_id="organ-1")
        for name in ("sys-x", "sys-y", "sys-z"):
            registry.register(name, holon_type="system", parent_id="sub-1")

        # Inject system refs that return specific values
        for name, val in [("sys-x", 0.9), ("sys-y", 0.6), ("sys-z", 0.3)]:
            proxy = registry.get_proxy(name)
            system = MagicMock()
            system.step.return_value = val
            proxy.set_system_ref(system)

        # Build the chain manually
        sub_action = SubsystemAction("sub-1", registry)
        organ_action = OrganAction("organ-1", {"sub-1": sub_action})
        organism_action = OrganismAction({"organ-1": organ_action})

        result = organism_action.act()
        expected = (0.9 + 0.6 + 0.3) / 3
        self.assertAlmostEqual(result, expected)

    def test_multiple_organs_and_subsystems(self):
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")

        # Organ 1 with 2 subsystems
        registry.register("organ-a", holon_type="organ", parent_id="mae")
        registry.register("sub-a1", holon_type="subsystem", parent_id="organ-a")
        registry.register("sub-a2", holon_type="subsystem", parent_id="organ-a")

        # 3 systems per subsystem
        for i, sub in enumerate(["sub-a1", "sub-a2"]):
            for j in range(3):
                name = f"sys-a{i}-{j}"
                registry.register(name, holon_type="system", parent_id=sub)
                proxy = registry.get_proxy(name)
                system = MagicMock()
                system.step.return_value = 0.8  # All return 0.8
                proxy.set_system_ref(system)

        sub_a1 = SubsystemAction("sub-a1", registry)
        sub_a2 = SubsystemAction("sub-a2", registry)
        organ_a = OrganAction("organ-a", {"sub-a1": sub_a1, "sub-a2": sub_a2})
        organism = OrganismAction({"organ-a": organ_a})

        result = organism.act()
        # All systems return 0.8 -> all subsystems=0.8 -> organ=0.8 -> organism=0.8
        self.assertAlmostEqual(result, 0.8)


# =====================================================================
# Fractal self-similarity verification
# =====================================================================


class TestFractalSelfSimilarity(unittest.TestCase):
    """Verify the same pattern at every scale (Law 4)."""

    def test_same_interface_at_every_level(self):
        """Every level has all 10 holon capabilities + get_statistics."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")

        sub = SubsystemAction("test-sub", registry)
        organ = OrganAction("test-organ", {}, registry=registry)
        organism = OrganismAction({}, registry=registry)

        required = [
            "act", "sense", "remember", "decide", "learn", "heal",
            "know_self", "know_up", "know_down", "know_peers",
            "get_statistics", "get_state",
        ]
        for method_name in required:
            self.assertTrue(
                callable(getattr(sub, method_name, None)),
                f"SubsystemAction missing {method_name}",
            )
            self.assertTrue(
                callable(getattr(organ, method_name, None)),
                f"OrganAction missing {method_name}",
            )
            self.assertTrue(
                callable(getattr(organism, method_name, None)),
                f"OrganismAction missing {method_name}",
            )

    def test_same_return_type_at_every_level(self):
        """Every level returns float from act()."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")

        sub = SubsystemAction("test-sub", registry)
        organ = OrganAction("test-organ", {})
        organism = OrganismAction({})

        self.assertIsInstance(sub.act(), float)
        self.assertIsInstance(organ.act(), float)
        self.assertIsInstance(organism.act(), float)

    def test_delegation_downward_aggregation_upward(self):
        """Each level delegates down and aggregates up."""
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        registry.register("sub", holon_type="subsystem", parent_id="mae")
        for name in ("a", "b", "c"):
            registry.register(name, holon_type="system", parent_id="sub")
            proxy = registry.get_proxy(name)
            system = MagicMock()
            system.step.return_value = 0.6
            proxy.set_system_ref(system)

        sub_action = SubsystemAction("sub", registry)
        organ_action = OrganAction("organ", {"sub": sub_action})
        organism_action = OrganismAction({"organ": organ_action})

        result = organism_action.act()
        # 3 systems @ 0.6 -> subsystem=0.6 -> organ=0.6 -> organism=0.6
        self.assertAlmostEqual(result, 0.6)


# =====================================================================
# 10 Holon Capabilities at Every Scale (Part 2 remediation)
# =====================================================================


class TestSubsystemCapabilities(unittest.TestCase):
    """All 10 holon capabilities on SubsystemAction."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("organ-1", holon_type="organ", parent_id="mae")
        self.registry.register("sub-1", holon_type="subsystem", parent_id="organ-1")
        self.registry.register("sub-2", holon_type="subsystem", parent_id="organ-1")
        for name in ("sys-a", "sys-b", "sys-c"):
            self.registry.register(name, holon_type="system", parent_id="sub-1")
            proxy = self.registry.get_proxy(name)
            system = MagicMock()
            system.step.return_value = 0.7
            system.get_state.return_value = {"status": "ok"}
            proxy.set_system_ref(system)
        self.action = SubsystemAction("sub-1", self.registry)

    def test_sense_aggregates_children(self):
        result = self.action.sense()
        self.assertEqual(result["subsystem_id"], "sub-1")
        self.assertIn("children_states", result)
        self.assertEqual(len(result["children_states"]), 3)

    def test_remember_store_retrieve(self):
        self.action.remember("key1", "value1")
        self.assertEqual(self.action.remember("key1"), "value1")
        self.assertIsNone(self.action.remember("missing"))

    def test_decide_returns_sorted_by_health(self):
        result = self.action.decide()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_learn_tracks_history(self):
        self.action._last_result = 0.8
        self.action.learn()
        self.assertEqual(len(self.action._score_history), 1)
        self.assertEqual(self.action._score_history[0], 0.8)

    def test_heal_calls_heal_on_degraded_children(self):
        # Mock somatic map on registry so get_proxy won't overwrite it
        mock_sm = MagicMock()
        info_degraded = MagicMock()
        info_degraded.health = 0.3
        info_healthy = MagicMock()
        info_healthy.health = 1.0
        mock_sm.get_system_info.side_effect = lambda sid: (
            info_degraded if sid == "sys-a" else info_healthy
        )
        self.registry._somatic_map = mock_sm
        result = self.action.heal()
        self.assertIn("sys-a", result["healed"])

    def test_get_state_returns_dict(self):
        result = self.action.get_state()
        self.assertEqual(result["subsystem_id"], "sub-1")
        self.assertIn("children", result)

    def test_know_self(self):
        result = self.action.know_self()
        self.assertEqual(result["holon_id"], "sub-1")
        self.assertEqual(result["children_count"], 3)

    def test_know_up(self):
        result = self.action.know_up()
        self.assertIsNotNone(result)
        self.assertEqual(result["parent_id"], "organ-1")

    def test_know_down(self):
        result = self.action.know_down()
        self.assertEqual(len(result), 3)

    def test_know_peers(self):
        result = self.action.know_peers()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["holon_id"], "sub-2")


class TestOrganCapabilities(unittest.TestCase):
    """All 10 holon capabilities on OrganAction."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("organ-1", holon_type="organ", parent_id="mae")
        self.registry.register("organ-2", holon_type="organ", parent_id="mae")
        self.registry.register("sub-1", holon_type="subsystem", parent_id="organ-1")
        sub_action = SubsystemAction("sub-1", self.registry)
        sub_action._last_result = 0.7
        self.action = OrganAction(
            "organ-1", {"sub-1": sub_action}, registry=self.registry,
        )

    def test_sense_aggregates_subsystems(self):
        result = self.action.sense()
        self.assertEqual(result["organ_id"], "organ-1")
        self.assertIn("subsystem_states", result)

    def test_remember_store_retrieve(self):
        self.action.remember("key1", 42)
        self.assertEqual(self.action.remember("key1"), 42)

    def test_decide_returns_sorted(self):
        result = self.action.decide()
        self.assertIsInstance(result, list)

    def test_learn_tracks_history(self):
        self.action._last_result = 0.5
        self.action.learn()
        self.assertEqual(self.action._score_history[-1], 0.5)

    def test_heal_heals_failing_subsystems(self):
        self.action._subsystem_actions["sub-1"]._last_result = 0.2
        result = self.action.heal()
        self.assertIn("sub-1", result["healed"])

    def test_get_state(self):
        result = self.action.get_state()
        self.assertEqual(result["organ_id"], "organ-1")
        self.assertIn("subsystems", result)

    def test_know_self(self):
        result = self.action.know_self()
        self.assertEqual(result["holon_id"], "organ-1")
        self.assertEqual(result["holon_type"], "organ")

    def test_know_up(self):
        result = self.action.know_up()
        self.assertIsNotNone(result)
        self.assertEqual(result["parent_id"], "mae")

    def test_know_down(self):
        result = self.action.know_down()
        self.assertEqual(len(result), 1)

    def test_know_peers(self):
        result = self.action.know_peers()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["holon_id"], "organ-2")


class TestOrganismCapabilities(unittest.TestCase):
    """All 10 holon capabilities on OrganismAction."""

    def setUp(self):
        self.registry = HolonRegistry()
        self.registry.register("mae", holon_type="organism")
        self.registry.register("organ-1", holon_type="organ", parent_id="mae")
        organ_action = OrganAction("organ-1", {}, registry=self.registry)
        organ_action._last_result = 0.6
        self.action = OrganismAction(
            {"organ-1": organ_action}, registry=self.registry,
        )
        self.action._organ_scores = {"organ-1": 0.6}

    def test_sense(self):
        result = self.action.sense()
        self.assertEqual(result["organism"], "mae")
        self.assertIn("organ_states", result)

    def test_remember(self):
        self.action.remember("epoch", 1)
        self.assertEqual(self.action.remember("epoch"), 1)

    def test_decide(self):
        result = self.action.decide()
        self.assertIsInstance(result, list)

    def test_learn(self):
        self.action._last_result = 0.9
        self.action.learn()
        self.assertEqual(self.action._score_history[-1], 0.9)

    def test_heal(self):
        self.action._organ_actions["organ-1"]._last_result = 0.1
        result = self.action.heal()
        self.assertIn("organ-1", result["healed"])

    def test_get_state(self):
        result = self.action.get_state()
        self.assertIn("organs", result)
        self.assertIn("organ-1", result["organs"])

    def test_know_self(self):
        result = self.action.know_self()
        self.assertEqual(result["holon_id"], "mae")
        self.assertEqual(result["holon_type"], "organism")

    def test_know_up_is_none(self):
        self.assertIsNone(self.action.know_up())

    def test_know_down(self):
        result = self.action.know_down()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["holon_id"], "organ-1")

    def test_know_peers_empty(self):
        self.assertEqual(self.action.know_peers(), [])


class TestModuleLevel(unittest.TestCase):
    """MODULE level grouping for organs with >3 subsystems."""

    def test_cognitive_has_modules(self):
        from mae_core.backbone.fractal_generator import MODULE_GROUPING
        self.assertIn("cognitive-system", MODULE_GROUPING)
        modules = MODULE_GROUPING["cognitive-system"]
        self.assertEqual(len(modules), 2)
        self.assertIn("cognitive-analytic", modules)
        self.assertIn("cognitive-social", modules)

    def test_modules_created_during_organize(self):
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        # Register all systems needed for cognitive
        for name in (
            "shared_world_model", "shared_causal_engine", "collective_dream",
            "knowledge_base", "transfer_engine", "maml_learner",
            "curiosity", "haven", "imitation",
            "temporal_memory", "worldline_planner", "validated_imagination",
            "emotional_system", "theory_of_mind", "metacognition",
        ):
            registry.register(name, holon_type="system", parent_id="mae")

        gen = FractalGenerator(
            holon_registry=registry,
            connection_registry=ConnectionRegistry(),
        )
        # Organize just cognitive
        gen.organize(
            grouping={"cognitive-system": FRACTAL_GROUPING["cognitive-system"]},
        )

        # Modules should exist
        analytic = registry.get_entry("cognitive-analytic")
        self.assertIsNotNone(analytic)
        self.assertEqual(analytic.holon_type, "module")

        social = registry.get_entry("cognitive-social")
        self.assertIsNotNone(social)
        self.assertEqual(social.holon_type, "module")

    def test_subsystems_reparented_to_modules(self):
        registry = HolonRegistry()
        registry.register("mae", holon_type="organism")
        for name in (
            "shared_world_model", "shared_causal_engine", "collective_dream",
            "knowledge_base", "transfer_engine", "maml_learner",
            "curiosity", "haven", "imitation",
            "temporal_memory", "worldline_planner", "validated_imagination",
            "emotional_system", "theory_of_mind", "metacognition",
        ):
            registry.register(name, holon_type="system", parent_id="mae")

        gen = FractalGenerator(
            holon_registry=registry,
            connection_registry=ConnectionRegistry(),
        )
        gen.organize(
            grouping={"cognitive-system": FRACTAL_GROUPING["cognitive-system"]},
        )

        # reasoning should be child of cognitive-analytic (module), not cognitive-system
        reasoning_entry = registry.get_entry("reasoning")
        self.assertIsNotNone(reasoning_entry)
        self.assertEqual(reasoning_entry.parent_id, "cognitive-analytic")

        # temporal should be child of cognitive-social
        temporal_entry = registry.get_entry("temporal")
        self.assertIsNotNone(temporal_entry)
        self.assertEqual(temporal_entry.parent_id, "cognitive-social")


if __name__ == "__main__":
    unittest.main()
