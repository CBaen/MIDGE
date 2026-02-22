"""Integration Tests - Pattern Ecosystem (Phase 3: Expression).

Tests the complete autopoietic loop:
1. PatternBus -> PatternCortex -> PatternAdvisory
2. Advisory flows into agent _decide() via DecisionRouter
3. PatternConsolidator fires at interval boundaries
4. Full loop runs without error
"""

from __future__ import annotations

from unittest.mock import MagicMock

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


# ── Advisory Flow ────────────────────────────────────────────────────

class TestAdvisoryFlow:
    """Verify advisory generation and agent access."""

    def test_pattern_bus_exists(self, mae_organism):
        _, systems = mae_organism
        assert systems.get("pattern_bus") is not None

    def test_pattern_cortex_exists(self, mae_organism):
        _, systems = mae_organism
        assert systems.get("pattern_cortex") is not None

    def test_advisory_generated_after_step(self, mae_organism):
        model, systems = mae_organism
        model.step()
        cortex = systems["pattern_cortex"]
        stats = cortex.get_statistics()
        assert stats["total_advisories"] >= 1

    def test_advisory_reaches_agents(self, mae_organism):
        model, systems = mae_organism
        model.step()
        for agent in systems["agents"]:
            advisory = getattr(agent, "_current_advisory", None)
            assert advisory is not None


# ── Decision Router Wiring ───────────────────────────────────────────

class TestDecisionRouterWiring:
    """Verify DecisionRouter is called in _decide() when advisory present."""

    def test_decision_router_exists_on_agents(self, mae_organism):
        _, systems = mae_organism
        for agent in systems["agents"]:
            assert agent.decision_router is not None

    def test_router_called_during_step(self, mae_organism):
        model, systems = mae_organism
        # First step populates advisory
        model.step()

        agent = systems["agents"][0]
        decisions_before = agent.decision_router._total_decisions

        # Second step: router should be consulted (advisory now available)
        model.step()

        decisions_after = agent.decision_router._total_decisions
        assert decisions_after > decisions_before

    def test_graceful_without_advisory(self, mae_organism):
        _, systems = mae_organism
        agent = systems["agents"][0]
        agent._current_advisory = None
        # _decide() should still work via existing fallback logic
        action = agent._decide()
        assert action is not None

    def test_graceful_without_router(self, mae_organism):
        _, systems = mae_organism
        agent = systems["agents"][0]
        agent.decision_router = None
        # _decide() should still work via existing fallback logic
        action = agent._decide()
        assert action is not None

    def test_router_none_tier_falls_through(self, mae_organism):
        model, systems = mae_organism
        model.step()  # Populate advisory

        agent = systems["agents"][0]
        from mae_core.cognition.decision_router import DecisionTier, RouterDecision

        mock_router = MagicMock()
        mock_decision = RouterDecision(
            decision_id="test",
            tier_used=DecisionTier.NONE,
            stimulus="test",
        )
        mock_router.route_decision.return_value = mock_decision
        agent.decision_router = mock_router

        action = agent._decide()
        assert action is not None
        mock_router.route_decision.assert_called_once()

    def test_high_confidence_forces_tier(self, mae_organism):
        model, systems = mae_organism
        model.step()  # Populate advisory

        agent = systems["agents"][0]
        from mae_core.cognition.decision_router import DecisionTier
        from mae_core.patterns.pattern_cortex import PatternAdvisory
        from mae_core.patterns.pattern_signal import (
            PatternDomain,
            PatternForm,
            PatternSignal,
        )

        # High-confidence advisory recommending reflex
        agent._current_advisory = PatternAdvisory(
            step=1,
            dominant_pattern=PatternSignal(
                source_system="test",
                domain=PatternDomain.THREAT,
                form=PatternForm.REACTIVE,
                confidence=0.9,
                salience=0.9,
                description="test threat",
            ),
            threat_level=0.8,
            recommended_tier="reflex",
            confidence=0.8,
        )

        calls = []
        original_route = agent.decision_router.route_decision

        def tracking_route(**kwargs):
            calls.append(kwargs)
            return original_route(**kwargs)

        agent.decision_router.route_decision = tracking_route
        agent._decide()

        assert len(calls) >= 1
        assert calls[0].get("force_tier") == DecisionTier.REFLEX

    def test_low_confidence_no_force(self, mae_organism):
        model, systems = mae_organism
        model.step()

        agent = systems["agents"][0]
        from mae_core.patterns.pattern_cortex import PatternAdvisory
        from mae_core.patterns.pattern_signal import (
            PatternDomain,
            PatternForm,
            PatternSignal,
        )

        agent._current_advisory = PatternAdvisory(
            step=1,
            dominant_pattern=PatternSignal(
                source_system="test",
                domain=PatternDomain.THREAT,
                form=PatternForm.REACTIVE,
                confidence=0.3,
                salience=0.3,
                description="weak threat",
            ),
            recommended_tier="reflex",
            confidence=0.4,  # Below 0.6 threshold
        )

        calls = []
        original_route = agent.decision_router.route_decision

        def tracking_route(**kwargs):
            calls.append(kwargs)
            return original_route(**kwargs)

        agent.decision_router.route_decision = tracking_route
        agent._decide()

        assert len(calls) >= 1
        assert calls[0].get("force_tier") is None


# ── Pattern Consolidator ─────────────────────────────────────────────

class TestPatternConsolidator:
    """Verify PatternConsolidator extraction and storage."""

    def test_consolidator_exists(self, mae_organism):
        _, systems = mae_organism
        assert systems.get("pattern_consolidator") is not None

    def test_consolidator_statistics(self, mae_organism):
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        stats = consolidator.get_statistics()
        assert "total_consolidations" in stats
        assert "total_trends_stored" in stats
        assert stats["has_cortex"] is True

    def test_consolidator_runs_at_interval(self, mae_organism):
        model, systems = mae_organism
        consolidator = systems["pattern_consolidator"]

        model.run(89)
        stats = consolidator.get_statistics()
        assert stats["total_consolidations"] >= 1

    def test_consolidator_extracts_trends(self, mae_organism):
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        cortex = systems["pattern_cortex"]

        from mae_core.patterns.pattern_signal import PatternDomain

        # Manually set a domain streak above threshold
        cortex._domain_streak[PatternDomain.THREAT] = 5

        result = consolidator.consolidate(step=89, force=True)
        # Should extract at least one trend (storage may fail without Qdrant)
        assert result.get("skipped") is not True

    def test_consolidator_graceful_without_bridge(self, mae_organism):
        _, systems = mae_organism
        cortex = systems["pattern_cortex"]

        from mae_core.patterns.pattern_consolidator import PatternConsolidator

        consolidator = PatternConsolidator(
            pattern_cortex=cortex,
            memory_bridge=None,
        )
        result = consolidator.consolidate(step=89, force=True)
        assert result["trends_stored"] == 0
        assert result["meta_stored"] == 0

    def test_consolidator_graceful_without_cortex(self):
        from mae_core.patterns.pattern_consolidator import PatternConsolidator

        consolidator = PatternConsolidator(pattern_cortex=None)
        result = consolidator.consolidate(step=89, force=True)
        assert result["skipped"] is True

    def test_consolidator_publishes_event(self, mae_organism):
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        bus = systems["event_bus"]

        received = []
        bus.register_callback(
            "pattern.consolidation",
            lambda ch, msg: received.append(msg),
        )

        consolidator.consolidate(step=89, force=True)
        assert len(received) >= 1


# ── Full Autopoietic Loop ───────────────────────────────────────────

class TestAutopoieticLoop:
    """Verify the full loop: detect -> advise -> decide -> consolidate."""

    def test_full_loop_100_steps(self, mae_organism):
        model, systems = mae_organism
        model.run(100)

        cortex = systems["pattern_cortex"]
        stats = cortex.get_statistics()
        assert stats["total_advisories"] >= 100

        for agent in systems["agents"]:
            assert agent.step_count >= 100

    def test_cortex_window_fills(self, mae_organism):
        model, systems = mae_organism
        model.run(20)

        cortex = systems["pattern_cortex"]
        assert cortex.window_size == 13  # Window maxlen

    def test_agents_use_router_during_run(self, mae_organism):
        model, systems = mae_organism

        total_before = sum(
            a.decision_router._total_decisions
            for a in systems["agents"]
        )

        model.run(10)

        total_after = sum(
            a.decision_router._total_decisions
            for a in systems["agents"]
        )

        # Routers should have been called (after step 1 when advisories exist)
        assert total_after > total_before

    def test_consolidator_fires_in_extended_run(self, mae_organism):
        model, systems = mae_organism
        model.run(90)

        consolidator = systems["pattern_consolidator"]
        assert consolidator.get_statistics()["total_consolidations"] >= 1

    def test_repr_strings(self, mae_organism):
        _, systems = mae_organism
        consolidator = systems["pattern_consolidator"]
        assert "PatternConsolidator" in repr(consolidator)
