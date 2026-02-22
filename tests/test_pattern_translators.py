"""Tests for all pattern translators."""

from __future__ import annotations

import pytest

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)
from mae_core.patterns.translators.cognition import (
    CausalEngineTranslator,
    DecisionRouterTranslator,
    WorldModelTranslator,
)
from mae_core.patterns.translators.defense import (
    AutoHealerTranslator,
    HAVENTranslator,
    ThreatTranslator,
)
from mae_core.patterns.translators.emergent import CapabilityTranslator
from mae_core.patterns.translators.learning import CuriosityTranslator
from mae_core.patterns.translators.memory import PatternDistillerTranslator
from mae_core.patterns.translators.opportunity import OpportunityTranslator


# ── WorldModelTranslator ─────────────────────────────────────────────

class TestWorldModelTranslator:
    def setup_method(self):
        self.t = WorldModelTranslator()

    def test_source_name(self):
        assert self.t.source_name == "world_model"

    def test_channels(self):
        assert "cognition.prediction_made" in self.t.channels

    def test_translates_high_uncertainty(self):
        sig = self.t.translate("cognition.prediction_made", {
            "uncertainty": 0.8,
            "ensemble_disagreement": 0.6,
            "reward": 0.5,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.PREDICTION
        assert sig.form == PatternForm.REACTIVE
        assert sig.salience >= 0.6

    def test_skips_low_uncertainty(self):
        sig = self.t.translate("cognition.prediction_made", {
            "uncertainty": 0.05,
            "ensemble_disagreement": 0.02,
            "reward": 0.5,
        })
        assert sig is None

    def test_skips_non_dict(self):
        assert self.t.translate("cognition.prediction_made", "not a dict") is None


# ── CausalEngineTranslator ───────────────────────────────────────────

class TestCausalEngineTranslator:
    def setup_method(self):
        self.t = CausalEngineTranslator()

    def test_source_name(self):
        assert self.t.source_name == "causal_engine"

    def test_channels(self):
        assert "cognition.causal_query_result" in self.t.channels
        assert "temporal.causal_link_discovered" in self.t.channels

    def test_translates_causal_query_positive(self):
        sig = self.t.translate("cognition.causal_query_result", {
            "query_id": "q1",
            "cause": "stress",
            "effect": "poor_decisions",
            "is_causal": True,
            "causal_strength": 0.7,
            "confidence": 0.8,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.CAUSATION
        assert "stress" in sig.description

    def test_skips_non_causal_query(self):
        sig = self.t.translate("cognition.causal_query_result", {
            "query_id": "q1",
            "cause": "A",
            "effect": "B",
            "is_causal": False,
        })
        assert sig is None

    def test_translates_temporal_link(self):
        sig = self.t.translate("temporal.causal_link_discovered", {
            "cause": "exploration",
            "effect": "reward",
            "strength": 0.6,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.CAUSATION
        assert sig.confidence == 0.5  # Correlations start at 0.5

    def test_skips_non_dict(self):
        assert self.t.translate("cognition.causal_query_result", 42) is None


# ── DecisionRouterTranslator ─────────────────────────────────────────

class TestDecisionRouterTranslator:
    def setup_method(self):
        self.t = DecisionRouterTranslator()

    def test_source_name(self):
        assert self.t.source_name == "decision_router"

    def test_translates_reflex_decision(self):
        sig = self.t.translate("cognition.decision_routed", {
            "decision_id": "d1",
            "tier": "reflex",
            "stimulus": "danger",
            "confidence": 0.9,
            "response_time_ms": 0.5,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.BEHAVIORAL
        assert "reflex" in sig.description

    def test_translates_low_confidence(self):
        sig = self.t.translate("cognition.decision_routed", {
            "decision_id": "d2",
            "tier": "prefrontal",
            "stimulus": "novel",
            "confidence": 0.2,
            "response_time_ms": 500,
        })
        assert sig is not None

    def test_skips_normal_habit_decision(self):
        sig = self.t.translate("cognition.decision_routed", {
            "decision_id": "d3",
            "tier": "habit",
            "stimulus": "routine",
            "confidence": 0.8,
            "response_time_ms": 50,
        })
        assert sig is None  # Normal habit decisions are not interesting


# ── CuriosityTranslator ──────────────────────────────────────────────

class TestCuriosityTranslator:
    def setup_method(self):
        self.t = CuriosityTranslator()

    def test_source_name(self):
        assert self.t.source_name == "curiosity"

    def test_channels(self):
        assert "memory.novel_experience" in self.t.channels

    def test_translates_novel_experience(self):
        sig = self.t.translate("memory.novel_experience", {
            "agent_id": "agent-0",
            "novelty_score": 0.85,
            "reward": 0.3,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.NOVELTY
        assert sig.confidence == 0.85
        assert "agent-0" in sig.description

    def test_skips_non_dict(self):
        assert self.t.translate("memory.novel_experience", "string") is None


# ── AutoHealerTranslator ─────────────────────────────────────────────

class TestAutoHealerTranslator:
    def setup_method(self):
        self.t = AutoHealerTranslator()

    def test_source_name(self):
        assert self.t.source_name == "auto_healer"

    def test_channels(self):
        assert "healing.failure_detected" in self.t.channels

    def test_translates_failure(self):
        sig = self.t.translate("healing.failure_detected", {
            "failure_id": "f1",
            "failure_type": "performance_degradation",
            "severity": 0.7,
            "affected_agents": [1, 2],
        })
        assert sig is not None
        assert sig.domain == PatternDomain.FAILURE
        assert sig.salience == 0.7
        assert "2 agents" in sig.description


# ── HAVENTranslator ───────────────────────────────────────────────────

class TestHAVENTranslator:
    def setup_method(self):
        self.t = HAVENTranslator()

    def test_source_name(self):
        assert self.t.source_name == "haven"

    def test_translates_risk_alert(self):
        sig = self.t.translate("haven.risk_alert", {
            "agent_id": "agent-3",
            "risk_score": 0.85,
            "risk_level": "critical",
        })
        assert sig is not None
        assert sig.domain == PatternDomain.THREAT
        assert sig.salience == 0.85
        assert "critical" in sig.description


# ── ThreatTranslator ─────────────────────────────────────────────────

class TestThreatTranslator:
    def setup_method(self):
        self.t = ThreatTranslator()

    def test_source_name(self):
        assert self.t.source_name == "threat_detector"

    def test_translates_defense_activated(self):
        sig = self.t.translate("defense.activated", {
            "strategy": "turtle",
            "integrity": 0.8,
            "action": "shell_up",
        })
        assert sig is not None
        assert sig.domain == PatternDomain.THREAT
        assert "turtle" in sig.description


# ── CapabilityTranslator ─────────────────────────────────────────────

class TestCapabilityTranslator:
    def setup_method(self):
        self.t = CapabilityTranslator()

    def test_source_name(self):
        assert self.t.source_name == "capability_discovery"

    def test_channels(self):
        assert "improvement.capability_found" in self.t.channels
        assert "improvement.capability_validated" in self.t.channels

    def test_translates_capability_found(self):
        sig = self.t.translate("improvement.capability_found", {
            "capability_id": "cap1",
            "agent_id": "agent-2",
            "context": "foraging",
            "performance_delta": 0.45,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.CAPABILITY
        assert sig.confidence == 0.5  # Not yet validated

    def test_translates_capability_validated(self):
        sig = self.t.translate("improvement.capability_validated", {
            "capability_id": "cap1",
            "validation_score": 0.8,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.CAPABILITY
        assert sig.confidence == 0.8


# ── PatternDistillerTranslator ────────────────────────────────────────

class TestPatternDistillerTranslator:
    def setup_method(self):
        self.t = PatternDistillerTranslator()

    def test_source_name(self):
        assert self.t.source_name == "pattern_distiller"

    def test_channels(self):
        assert "memory.consolidation_complete" in self.t.channels

    def test_translates_consolidation_event(self):
        sig = self.t.translate("memory.consolidation_complete", {
            "agent_id": "agent-0",
            "episodes_consolidated": 15,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.BEHAVIORAL

    def test_skips_zero_episodes(self):
        sig = self.t.translate("memory.consolidation_complete", {
            "agent_id": "agent-0",
            "episodes_consolidated": 0,
        })
        assert sig is None

    def test_translates_distilled_pattern(self):
        sig = self.t.translate_distilled_pattern({
            "pattern_type": "behavioral",
            "domain": "action-0",
            "occurrence_count": 5,
            "confidence": 0.8,
            "description": "Action 0 yields positive reward",
        })
        assert sig is not None
        assert sig.domain == PatternDomain.BEHAVIORAL
        assert sig.form == PatternForm.ANCESTRAL
        assert sig.occurrence_count == 5
        assert sig.ttl_steps == 21  # Longer TTL for ancestral

    def test_translates_state_pattern(self):
        sig = self.t.translate_distilled_pattern({
            "pattern_type": "state",
            "domain": "dim-3-dominant",
            "occurrence_count": 7,
            "confidence": 0.6,
            "description": "Dimension 3 dominant correlates with negative outcome",
        })
        assert sig is not None
        assert sig.domain == PatternDomain.STATE
        assert sig.form == PatternForm.ANCESTRAL


# ── OpportunityTranslator ────────────────────────────────────────────

class TestOpportunityTranslator:
    def setup_method(self):
        self.t = OpportunityTranslator()

    def test_source_name(self):
        assert self.t.source_name == "opportunity"

    def test_channels(self):
        assert "improvement.capability_validated" in self.t.channels
        assert "memory.novel_experience" in self.t.channels

    def test_translates_validated_capability(self):
        sig = self.t.translate("improvement.capability_validated", {
            "capability_id": "cap1",
            "validation_score": 0.8,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.OPPORTUNITY
        assert sig.form == PatternForm.REACTIVE
        assert sig.confidence == 0.8
        assert "cap1" in sig.description

    def test_skips_weak_validation(self):
        sig = self.t.translate("improvement.capability_validated", {
            "capability_id": "cap1",
            "validation_score": 0.2,
        })
        assert sig is None

    def test_translates_novel_positive_reward(self):
        sig = self.t.translate("memory.novel_experience", {
            "agent_id": "agent-0",
            "novelty_score": 0.9,
            "reward": 0.5,
        })
        assert sig is not None
        assert sig.domain == PatternDomain.OPPORTUNITY
        assert "agent-0" in sig.description

    def test_skips_negative_reward(self):
        sig = self.t.translate("memory.novel_experience", {
            "agent_id": "agent-0",
            "novelty_score": 0.9,
            "reward": -0.1,
        })
        assert sig is None

    def test_skips_zero_reward(self):
        sig = self.t.translate("memory.novel_experience", {
            "agent_id": "agent-0",
            "novelty_score": 0.9,
            "reward": 0.0,
        })
        assert sig is None

    def test_skips_non_dict(self):
        assert self.t.translate("improvement.capability_validated", "string") is None

    def test_unknown_channel_returns_none(self):
        assert self.t.translate("unknown.channel", {"data": 1}) is None


# ── Protocol Compliance ───────────────────────────────────────────────

class TestProtocolCompliance:
    """Verify all translators implement the PatternTranslator protocol."""

    @pytest.mark.parametrize("translator_cls", [
        WorldModelTranslator,
        CausalEngineTranslator,
        DecisionRouterTranslator,
        CuriosityTranslator,
        AutoHealerTranslator,
        HAVENTranslator,
        ThreatTranslator,
        CapabilityTranslator,
        PatternDistillerTranslator,
        OpportunityTranslator,
    ])
    def test_has_required_properties(self, translator_cls):
        t = translator_cls()
        assert isinstance(t.source_name, str)
        assert isinstance(t.channels, list)
        assert len(t.channels) > 0
        assert callable(t.translate)

    @pytest.mark.parametrize("translator_cls", [
        WorldModelTranslator,
        CausalEngineTranslator,
        DecisionRouterTranslator,
        CuriosityTranslator,
        AutoHealerTranslator,
        HAVENTranslator,
        ThreatTranslator,
        CapabilityTranslator,
        PatternDistillerTranslator,
        OpportunityTranslator,
    ])
    def test_returns_none_for_non_dict(self, translator_cls):
        t = translator_cls()
        for ch in t.channels:
            result = t.translate(ch, "not a dict")
            assert result is None
