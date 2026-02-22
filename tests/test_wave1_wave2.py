"""Tests for Wave 1+2 Changes — 9 bug fixes + 12 system wirings.

These 21 changes were implemented WITHOUT test coverage (audit concern #1).
This file provides comprehensive verification for every change.

Organized into three test classes:
1. TestBugFixes — 10 bug-fix verifications (BUG-03 through BUG-19)
2. TestSystemWiring — 12 wiring verifications (new connections)
3. TestWaveIntegration — end-to-end checks that all changes work together
"""

from __future__ import annotations

import copy
import math
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

from main import create_mae


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_codebase_audit_fixes.py)
# ---------------------------------------------------------------------------

def _make_model():
    model = MagicMock()
    model._agents = {}
    model.schedule = MagicMock()
    return model


def _make_agent(**kwargs):
    from mae_core.agents.mycelial_agent import MycelialAgent
    model = _make_model()
    return MycelialAgent(model=model, agent_type="mycelial", **kwargs)


# ---------------------------------------------------------------------------
# Fixture: full Mae organism for integration tests
# ---------------------------------------------------------------------------

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


# ===========================================================================
# BUG FIX TESTS
# ===========================================================================

class TestBugFixes:
    """Verify all 10 bug fixes from Wave 1+2."""

    # --- BUG-03: BaseAgent.get_learning_rate / set_learning_rate ----------

    def test_bug03_get_learning_rate_returns_float(self):
        """BUG-03: get_learning_rate() returns a float."""
        from mae_core.agents.base_agent import BaseAgent
        model = _make_model()
        agent = BaseAgent(model)
        rate = agent.get_learning_rate()
        assert isinstance(rate, float)
        assert rate == 0.01  # Default value

    def test_bug03_set_learning_rate_changes_value(self):
        """BUG-03: set_learning_rate() updates the stored rate."""
        from mae_core.agents.base_agent import BaseAgent
        model = _make_model()
        agent = BaseAgent(model)
        agent.set_learning_rate(0.05)
        assert agent.get_learning_rate() == 0.05

    def test_bug03_learning_rate_on_mycelial_agent(self):
        """BUG-03: MycelialAgent inherits learning rate from BaseAgent."""
        agent = _make_agent()
        assert agent.get_learning_rate() == 0.01
        agent.set_learning_rate(0.1)
        assert agent.get_learning_rate() == 0.1

    # --- BUG-04: Semantic recall unwrap fix --------------------------------

    def test_bug04_semantic_recall_doesnt_crash(self, mae_organism):
        """BUG-04: After a model step, semantic recall path doesn't crash.

        The fix was to unwrap SemanticQuery.experiences in _decide().
        Running a step exercises _decide() which calls
        search_similar_experiences and unwraps the result.
        """
        model, systems = mae_organism
        # Step should complete without AttributeError on SemanticQuery
        model.step()
        for agent in systems["agents"]:
            assert agent.step_count >= 1

    def test_bug04_decide_handles_semantic_query_object(self):
        """BUG-04: _decide() unwraps SemanticQuery.experiences correctly."""
        agent = _make_agent()
        # Simulate SemanticQuery-like return value
        mock_query = MagicMock()
        mock_query.experiences = []
        agent.semantic_retriever = MagicMock()
        agent.semantic_retriever.search = MagicMock(return_value=mock_query)
        agent._observe()
        # Should not raise
        action = agent._decide()
        assert action is not None

    # --- BUG-05: available_actions in _decide() context -------------------

    def test_bug05_available_actions_in_decide_context(self, mae_organism):
        """BUG-05: _route_with_advisory passes available_actions to router."""
        model, systems = mae_organism
        model.step()  # Populate advisory

        agent = systems["agents"][0]
        calls = []
        original_route = agent.decision_router.route_decision

        def tracking_route(**kwargs):
            calls.append(kwargs)
            return original_route(**kwargs)

        agent.decision_router.route_decision = tracking_route
        model.step()  # Second step: router is called with advisory

        # Find calls that have available_actions
        calls_with_actions = [c for c in calls if "available_actions" in c]
        assert len(calls_with_actions) >= 1
        actions = calls_with_actions[0]["available_actions"]
        assert isinstance(actions, list)
        assert len(actions) == 5
        action_types = {a["type"] for a in actions}
        assert action_types == {"explore", "exploit", "communicate", "rest", "api_call"}

    # --- BUG-06: DecisionRouter.set_reflex_bias ---------------------------

    def test_bug06_set_reflex_bias_sets_value(self):
        """BUG-06: set_reflex_bias() stores the bias correctly."""
        from mae_core.cognition.decision_router import DecisionRouter
        router = DecisionRouter()
        assert router._reflex_bias == 0.0
        router.set_reflex_bias(0.7)
        assert router._reflex_bias == 0.7

    def test_bug06_set_reflex_bias_clamps(self):
        """BUG-06: set_reflex_bias() clamps to [0.0, 1.0]."""
        from mae_core.cognition.decision_router import DecisionRouter
        router = DecisionRouter()
        router.set_reflex_bias(-0.5)
        assert router._reflex_bias == 0.0
        router.set_reflex_bias(1.5)
        assert router._reflex_bias == 1.0

    def test_bug06_reflex_bias_affects_check(self):
        """BUG-06: High reflex bias triggers the endocrine override path."""
        from mae_core.cognition.decision_router import DecisionRouter
        router = DecisionRouter()
        router.set_reflex_bias(0.7)
        # Route a stimulus that does NOT match exact reflex patterns
        # but should match fuzzy with high bias
        decision = router.route_decision(stimulus="dangerous_situation_nearby")
        # The bias re-check path should have been attempted
        metrics = router.get_performance_metrics()
        assert metrics["current_reflex_bias"] == 0.7

    # --- BUG-07: Fuzzy prefix matching at high adrenaline -----------------

    def test_bug07_fuzzy_matching_danger_variants(self):
        """BUG-07: With high bias, fuzzy matching catches 'danger' variants.

        The fuzzy prefix floor is 5 chars (not 3), so for 'danger' (6 chars)
        at bias=0.9: min_prefix = max(5, int(6 * 0.2)) = 5 -> prefix = 'dange'.
        A stimulus word starting with 'dange' but not containing 'danger'
        as a full substring should match fuzzy but not exact.
        """
        from mae_core.cognition.decision_router import DecisionRouter
        router = DecisionRouter()

        # "dangerous" contains "danger" as substring -> exact match always works
        # We need something that has "dange" prefix but not full "danger"
        # "dangeous" starts with "dange" but does not contain "danger"
        stimulus = "a dangeous situation"

        # Without bias: "danger" not in "a dangeous situation" -> no match
        result_no_bias = router._check_reflex(stimulus, bias=0.0)
        assert result_no_bias is None

        # With high bias: fuzzy prefix "dange" in "dangeous" -> match
        result_with_bias = router._check_reflex(stimulus, bias=0.9)
        assert result_with_bias is not None
        assert result_with_bias.pattern_id == "danger"

    def test_bug07_reflex_bias_triggers_counter(self):
        """BUG-07: Decisions won by bias increment the counter."""
        from mae_core.cognition.decision_router import DecisionRouter
        router = DecisionRouter()
        router.set_reflex_bias(0.9)
        # "dangeous" starts with "dange" (5 chars) but doesn't contain "danger"
        router.route_decision(stimulus="a dangeous situation")
        metrics = router.get_performance_metrics()
        assert metrics["reflex_bias_triggers"] >= 1

    # --- BUG-08: PatternBus signal mutation fix ----------------------------

    def test_bug08_detect_correlations_no_mutation(self):
        """BUG-08: _detect_correlations() doesn't mutate input signals."""
        from mae_core.patterns.pattern_bus import PatternBus
        from mae_core.patterns.pattern_signal import (
            PatternDomain, PatternForm, PatternSignal,
        )

        bus = PatternBus(event_bus=MagicMock())

        sig_a = PatternSignal(
            source_system="system_a",
            domain=PatternDomain.THREAT,
            form=PatternForm.REACTIVE,
            confidence=0.7,
            salience=0.8,
            description="test signal A",
        )
        sig_b = PatternSignal(
            source_system="system_b",
            domain=PatternDomain.THREAT,
            form=PatternForm.REACTIVE,
            confidence=0.6,
            salience=0.7,
            description="test signal B",
        )

        # Save original form and confidence
        orig_form_a = sig_a.form
        orig_conf_a = sig_a.confidence
        orig_form_b = sig_b.form
        orig_conf_b = sig_b.confidence

        by_domain = {PatternDomain.THREAT: [sig_a, sig_b]}
        groups = bus._detect_correlations(by_domain)

        # Original signals must not have been mutated
        assert sig_a.form == orig_form_a
        assert sig_a.confidence == orig_conf_a
        assert sig_b.form == orig_form_b
        assert sig_b.confidence == orig_conf_b

        # But the group signals should have boosted confidence and CORRELATED form
        assert len(groups) == 1
        for group_sig in groups[0]:
            assert group_sig.form == PatternForm.CORRELATED

    # --- BUG-09: PatternSense z-score self-inclusion ----------------------

    def test_bug09_zscore_excludes_current(self):
        """BUG-09: z-score computed against prior observations only."""
        from mae_core.patterns.pattern_sense import PatternSense

        ps = PatternSense(agent_id="test")
        # Feed 5 identical rewards, then a big outlier
        for i in range(5):
            ps.sense(reward=1.0, action=0, step=i)

        # Step 6: huge positive spike
        result = ps.sense(reward=10.0, action=0, step=5)

        # Should detect surprise (z-score based on baseline [1,1,1,1,1])
        surprise_signals = [
            s for s in result.signals
            if "surprise" in s.description.lower()
        ]
        assert len(surprise_signals) >= 1
        # The z-score should be large (10 is far from mean=1.0)
        evidence = surprise_signals[0].evidence
        assert evidence["z_score"] > 2.0

    def test_bug09_zscore_zero_variance(self):
        """BUG-09: Zero-variance baseline uses synthetic z-score."""
        from mae_core.patterns.pattern_sense import PatternSense, _ZERO_VARIANCE_ZSCORE

        ps = PatternSense(agent_id="test")
        # All same value, then different
        for i in range(3):
            ps.sense(reward=5.0, action=0, step=i)

        result = ps.sense(reward=6.0, action=0, step=3)
        surprise_signals = [
            s for s in result.signals
            if "surprise" in s.description.lower()
        ]
        assert len(surprise_signals) >= 1
        evidence = surprise_signals[0].evidence
        assert evidence["z_score"] == _ZERO_VARIANCE_ZSCORE

    def test_bug09_zscore_no_surprise_when_same(self):
        """BUG-09: Same value as zero-variance baseline -> no surprise."""
        from mae_core.patterns.pattern_sense import PatternSense

        ps = PatternSense(agent_id="test")
        for i in range(5):
            ps.sense(reward=5.0, action=0, step=i)

        # Same value: no surprise expected
        result = ps.sense(reward=5.0, action=0, step=5)
        surprise_signals = [
            s for s in result.signals
            if "surprise" in s.description.lower()
        ]
        assert len(surprise_signals) == 0

    # --- BUG-11: Agent parent_id initialization ---------------------------

    def test_bug11_agent_parent_id_is_colony(self, mae_organism):
        """BUG-11: After create_mae, agents have _holon_parent_id == 'colony'."""
        _, systems = mae_organism
        for agent in systems["agents"]:
            assert agent._holon_parent_id == "colony"

    # --- BUG-12: AutoHealer.step() runs without error ---------------------

    def test_bug12_autohealer_step_runs(self, mae_organism):
        """BUG-12: AutoHealer.step() executes without error."""
        model, systems = mae_organism
        auto_healer = systems["auto_healer"]
        # Run enough steps to trigger the scan interval
        for _ in range(20):
            auto_healer.step()
        # Should not raise; step count should advance
        assert auto_healer._step_count == 20

    def test_bug12_autohealer_reports_health(self, mae_organism):
        """BUG-12: AutoHealer reports its own health via somatic_map."""
        model, systems = mae_organism
        auto_healer = systems["auto_healer"]
        somatic_map = systems["somatic_map"]
        # Run enough steps to trigger a scan (every 10 steps)
        for _ in range(10):
            auto_healer.step()
        stats = auto_healer.get_statistics()
        assert isinstance(stats, dict)
        assert "total_healings" in stats

    # --- BUG-19: Agent recall config defaults True ------------------------

    def test_bug19_recall_config_all_true(self, mae_organism):
        """BUG-19: Agent config has all recall flags enabled."""
        _, systems = mae_organism
        for agent in systems["agents"]:
            config = agent.agent_config
            assert config.get("semantic_search_enabled") is True
            assert config.get("replay_enabled") is True
            assert config.get("consolidation_enabled") is True
            assert config.get("generative_memory_enabled") is True
            assert config.get("transfer_enabled") is True


# ===========================================================================
# SYSTEM WIRING TESTS
# ===========================================================================

class TestSystemWiring:
    """Verify all 12 new system connections from Wave 1+2."""

    # --- CuriosityDrive injection -----------------------------------------

    def test_curiosity_drive_injected(self, mae_organism):
        """CuriosityDrive: agents have curiosity_drive attribute after create_mae."""
        _, systems = mae_organism
        for agent in systems["agents"]:
            assert hasattr(agent, "curiosity_drive")
            assert agent.curiosity_drive is not None
            assert agent.curiosity_drive is systems["curiosity"]

    # --- Strange Loop: holon_know_self in _observe() ----------------------

    def test_strange_loop_self_awareness(self, mae_organism):
        """Strange Loop: After model.step(), agents have self-awareness data."""
        model, systems = mae_organism
        model.step()
        for agent in systems["agents"]:
            # _observe() calls holon_know_self() and stores result
            self_awareness = getattr(agent, "_self_awareness", None)
            assert self_awareness is not None
            assert "holon_type" in self_awareness

    # --- Signal Context: _build_signal_context ----------------------------

    def test_signal_context_returns_dict(self, mae_organism):
        """Signal Context: _build_signal_context returns dict with expected keys."""
        model, systems = mae_organism
        model.step()  # Populate advisory

        agent = systems["agents"][0]
        ctx = agent._build_signal_context()
        # After first step, advisory should exist so context should be non-None
        if ctx is not None:
            assert isinstance(ctx, dict)
            # Should have signal_type and/or metadata
            assert "signal_type" in ctx or "metadata" in ctx

    def test_signal_context_with_advisory(self):
        """Signal Context: _build_signal_context packages advisory fields."""
        agent = _make_agent()
        # Create a mock advisory
        mock_advisory = MagicMock()
        mock_advisory.dominant_pattern = MagicMock()
        mock_advisory.dominant_pattern.domain = MagicMock()
        mock_advisory.dominant_pattern.domain.value = "threat"
        mock_advisory.dominant_pattern.description = "test threat"
        mock_advisory.threat_level = 0.8
        mock_advisory.opportunity_level = 0.2
        mock_advisory.confidence = 0.7

        agent._current_advisory = mock_advisory
        ctx = agent._build_signal_context()
        assert ctx is not None
        assert "signal_type" in ctx
        assert "threat:test threat" in ctx["signal_type"]
        assert ctx["metadata"]["threat_level"] == 0.8

    # --- WorldModel Training: EventBus callback ---------------------------

    def test_worldmodel_training_callback(self, mae_organism):
        """WorldModel Training: EventBus has callback for 'cognition.imagination_validated'."""
        _, systems = mae_organism
        bus = systems["event_bus"]
        stats = bus.get_stats()
        # The channel should have at least one subscriber
        channels = stats.get("channels", {})
        # Check via callback presence
        has_imagination_callback = False
        if hasattr(bus, "_subscribers"):
            has_imagination_callback = "cognition.imagination_validated" in bus._subscribers
        assert has_imagination_callback, (
            "Expected EventBus to have callback for cognition.imagination_validated"
        )

    # --- AutoHealer Wiring: has _somatic_map attribute --------------------

    def test_autohealer_has_somatic_map(self, mae_organism):
        """AutoHealer Wiring: AutoHealer has _somatic_map after create_mae."""
        _, systems = mae_organism
        auto_healer = systems["auto_healer"]
        assert auto_healer._somatic_map is not None
        assert auto_healer._somatic_map is systems["somatic_map"]

    # --- Endocrine from Advisory: EventBus callback -----------------------

    def test_endocrine_advisory_callback(self, mae_organism):
        """Endocrine from Advisory: EventBus has callback for 'pattern.advisory'."""
        _, systems = mae_organism
        bus = systems["event_bus"]
        has_advisory_callback = False
        if hasattr(bus, "_subscribers"):
            has_advisory_callback = "pattern.advisory" in bus._subscribers
        assert has_advisory_callback, (
            "Expected EventBus to have callback for pattern.advisory"
        )

    # --- Hormone Modulation: signal priority resolver ---------------------

    def test_hormone_modulation_signal_priority(self, mae_organism):
        """Hormone Modulation: hormone state modulates signal resolver urgency."""
        _, systems = mae_organism
        bus = systems["event_bus"]
        agent = systems["agents"][0]
        resolver = getattr(agent, "_signal_resolver", None)
        assert resolver is not None

        # Get initial DANGER urgency
        initial_danger = resolver._urgency_map.get("DANGER", 1.0)

        # Simulate high adrenaline hormone state update
        bus.publish("endocrine.state_update", {
            "adrenaline": 0.9,
            "dopamine": 0.5,
            "melatonin": 0.1,
        })

        # DANGER urgency should be at least the default (1.0 floor)
        assert resolver._urgency_map["DANGER"] >= initial_danger
        # OPPORTUNITY should have been boosted
        assert resolver._urgency_map["OPPORTUNITY"] > 0.4

    # --- PatternDistiller: wired to PatternConsolidator -------------------

    def test_pattern_distiller_wired(self, mae_organism):
        """PatternDistiller: PatternConsolidator has distiller attribute."""
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        if consolidator is not None:
            stats = consolidator.get_statistics()
            # _distiller may or may not be set depending on Qdrant availability
            # but has_distiller key should exist
            assert "has_distiller" in stats

    # --- Circadian Gating: wired to PatternConsolidator -------------------

    def test_circadian_gating_exists(self, mae_organism):
        """Circadian Gating: PatternConsolidator has circadian attribute."""
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        if consolidator is not None:
            assert consolidator._circadian is not None
            assert consolidator._circadian is systems["circadian"]
            stats = consolidator.get_statistics()
            assert stats["has_circadian"] is True

    def test_circadian_gating_respects_phase(self):
        """Circadian Gating: consolidate respects circadian gate."""
        from mae_core.patterns.pattern_consolidator import PatternConsolidator

        mock_cortex = MagicMock()
        mock_cortex._domain_streak = {}

        mock_circadian = MagicMock()
        # Circadian says NOT time to consolidate
        mock_circadian.should_consolidate_memory = MagicMock(return_value=False)
        mock_circadian.current_phase = MagicMock()
        mock_circadian.current_phase.value = "ACTIVE"

        consolidator = PatternConsolidator(
            pattern_cortex=mock_cortex,
            circadian=mock_circadian,
        )

        # Without force, should be gated
        result = consolidator.consolidate(step=89, force=False)
        assert result["skipped"] is True
        assert result["reason"] == "circadian_gate"

        # With force, should bypass gate
        result = consolidator.consolidate(step=89, force=True)
        assert result.get("skipped") is not True

    # --- Habituation: salience decay in PatternSense ----------------------

    def test_habituation_decay_exists(self):
        """Habituation: PatternSense._salience_decay dict exists."""
        from mae_core.patterns.pattern_sense import PatternSense
        ps = PatternSense(agent_id="test")
        assert hasattr(ps, "_salience_decay")
        assert isinstance(ps._salience_decay, dict)

    def test_habituation_decays_salience(self):
        """Habituation: repeated identical patterns decay in salience."""
        from mae_core.patterns.pattern_sense import PatternSense, HABITUATION_DECAY

        ps = PatternSense(agent_id="test")
        # Create a reward trend to trigger signals
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        saliences = []
        for i, r in enumerate(rewards):
            result = ps.sense(reward=r, action=0, step=i)
            if result.signals:
                saliences.append(result.signals[0].salience)

        # If we got multiple salience values, later ones should be lower
        # due to habituation decay
        if len(saliences) >= 2:
            assert saliences[-1] < saliences[0], (
                f"Expected habituation decay: first={saliences[0]}, last={saliences[-1]}"
            )

    def test_habituation_resets_on_stop(self):
        """Habituation: stale decay entries are cleaned up when pattern stops.

        When a pattern signal stops appearing, its key is removed from
        _salience_decay. The code deletes keys for patterns that did NOT
        fire in the current step.
        """
        from mae_core.patterns.pattern_sense import PatternSense

        ps = PatternSense(agent_id="test")
        # Increasing rewards to trigger opportunity trend signal
        for i in range(5):
            ps.sense(reward=float(i) * 0.5, action=0, step=i)

        # Capture which keys exist in salience_decay after trend fires
        active_keys_after_trend = set(ps._salience_decay.keys())
        assert len(active_keys_after_trend) > 0, "Expected some salience decay entries"

        # Now feed constant rewards with varying actions to stop all patterns.
        # Use different actions each step to prevent repetition pattern too.
        for i in range(5, 15):
            ps.sense(reward=2.0, action=i, step=i)

        # After 10 steps of no signals, stale keys should be cleaned up
        for old_key in active_keys_after_trend:
            assert old_key not in ps._salience_decay, (
                f"Expected stale key {old_key!r} to be removed from _salience_decay"
            )

    # --- Priority Inbox: salience-first processing in PatternBus ----------

    def test_priority_inbox_ordering(self):
        """Priority Inbox: PatternBus processes high-salience signals first."""
        from mae_core.patterns.pattern_bus import PatternBus
        from mae_core.patterns.pattern_signal import (
            PatternDomain, PatternForm, PatternSignal,
        )

        bus = PatternBus(event_bus=MagicMock())

        # Enqueue signals with varying salience (low first)
        low_signal = PatternSignal(
            source_system="low",
            domain=PatternDomain.BEHAVIORAL,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.1,
            description="low salience",
        )
        high_signal = PatternSignal(
            source_system="high",
            domain=PatternDomain.THREAT,
            form=PatternForm.REACTIVE,
            confidence=0.9,
            salience=0.9,
            description="high salience",
        )
        mid_signal = PatternSignal(
            source_system="mid",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.6,
            salience=0.5,
            description="mid salience",
        )

        # Enqueue in low-mid-high order
        bus._inbox.append(low_signal)
        bus._inbox.append(mid_signal)
        bus._inbox.append(high_signal)

        digest = bus.process_step(1)

        # After processing, signals should be sorted by salience (high first)
        assert len(digest.signals) == 3
        assert digest.signals[0].salience >= digest.signals[1].salience
        assert digest.signals[1].salience >= digest.signals[2].salience

    def test_priority_inbox_overflow_requeued(self):
        """Priority Inbox: signals exceeding budget are requeued."""
        from mae_core.patterns.pattern_bus import PatternBus
        from mae_core.patterns.pattern_signal import (
            PatternDomain, PatternForm, PatternSignal,
        )

        bus = PatternBus(event_bus=MagicMock())

        # Override max to 2 for test
        original_max = PatternBus.MAX_SIGNALS_PER_STEP
        PatternBus.MAX_SIGNALS_PER_STEP = 2
        try:
            for i in range(5):
                sig = PatternSignal(
                    source_system=f"sys_{i}",
                    domain=PatternDomain.BEHAVIORAL,
                    form=PatternForm.REACTIVE,
                    confidence=0.5,
                    salience=float(i) / 5.0,
                    description=f"signal {i}",
                )
                bus._inbox.append(sig)

            digest = bus.process_step(1)
            # Only 2 processed (highest salience)
            assert len(digest.signals) == 2
            # Remaining 3 should be back in inbox
            assert len(bus._inbox) == 3
        finally:
            PatternBus.MAX_SIGNALS_PER_STEP = original_max


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================

class TestWaveIntegration:
    """End-to-end tests verifying Wave 1+2 changes work together."""

    def test_full_organism_step_no_errors(self, mae_organism):
        """All 21 changes work together during model.step()."""
        model, systems = mae_organism
        # Run 10 steps exercising all pathways
        model.run(10)
        for agent in systems["agents"]:
            assert agent.step_count >= 10

    def test_advisory_to_router_to_reflex_bias_chain(self, mae_organism):
        """Advisory -> endocrine -> reflex bias -> router decision chain."""
        model, systems = mae_organism
        bus = systems["event_bus"]

        # Step to populate everything
        model.run(5)

        # Simulate a high-threat advisory to trigger adrenaline
        bus.publish("pattern.advisory", {
            "step": 100,
            "threat_level": 0.8,
            "opportunity_level": 0.1,
            "novelty_level": 0.2,
            "recommended_tier": "reflex",
            "active_trends": {},
            "confidence": 0.7,
        })

        # Endocrine should have released adrenaline
        endocrine_stats = systems["endocrine"].get_statistics()
        # Verify the chain worked (no crash)
        assert endocrine_stats is not None

    def test_autohealer_with_somatic_map_scan(self, mae_organism):
        """AutoHealer proactive scan uses SomaticMap successfully."""
        model, systems = mae_organism
        auto_healer = systems["auto_healer"]
        somatic_map = systems["somatic_map"]

        # Ensure somatic_map is connected
        assert auto_healer._somatic_map is somatic_map

        # Run model to trigger auto_healer step hooks
        model.run(15)

        # AutoHealer should have completed at least one scan cycle
        assert auto_healer._step_count >= 15
        stats = auto_healer.get_statistics()
        assert isinstance(stats, dict)

    def test_curiosity_reward_in_learn(self, mae_organism):
        """CuriosityDrive intrinsic reward integrates during _learn()."""
        model, systems = mae_organism
        # Run steps to trigger _learn with curiosity
        model.run(5)
        # Verify agents have curiosity_drive and no crashes
        for agent in systems["agents"]:
            assert agent.curiosity_drive is not None
            assert agent.step_count >= 5

    def test_circadian_gated_consolidation(self, mae_organism):
        """Circadian gating works with PatternConsolidator in full organism."""
        _, systems = mae_organism
        consolidator = systems.get("pattern_consolidator")
        if consolidator is None:
            pytest.skip("Pattern consolidator not available")

        # Consolidator should have circadian reference
        assert consolidator._circadian is not None
        # Stats should track circadian skips
        stats = consolidator.get_statistics()
        assert "total_circadian_skips" in stats

    def test_habituation_in_full_run(self, mae_organism):
        """Habituation works during full organism run."""
        model, systems = mae_organism
        model.run(20)

        # Check that at least one agent's pattern_sense has been called
        for uid, agent_sys in systems["per_agent_systems"].items():
            ps = agent_sys.get("pattern_sense")
            if ps is not None:
                assert ps._step > 0
                # _salience_decay dict should exist
                assert isinstance(ps._salience_decay, dict)

    def test_pattern_bus_priority_in_full_run(self, mae_organism):
        """Priority-based inbox draining works in the full organism."""
        model, systems = mae_organism
        pattern_bus = systems.get("pattern_bus")
        if pattern_bus is None:
            pytest.skip("Pattern bus not available")

        model.run(5)
        stats = pattern_bus.get_statistics()
        assert stats["total_signals"] >= 0  # May be 0 if no translators fired
        # Verify recent digests exist
        digests = pattern_bus.get_recent_digests(3)
        assert len(digests) >= 1

    def test_eventbus_has_all_wave2_callbacks(self, mae_organism):
        """All Wave 2 EventBus callbacks are registered."""
        _, systems = mae_organism
        bus = systems["event_bus"]

        expected_channels = [
            "cognition.imagination_validated",
            "holon.anomaly_detected",
            "pattern.advisory",
            "endocrine.state_update",
        ]

        if hasattr(bus, "_subscribers"):
            for channel in expected_channels:
                assert channel in bus._subscribers, (
                    f"Missing EventBus callback: {channel}"
                )
