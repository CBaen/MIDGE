"""Tests for AttentionalGate -- Mae's TRN analog (top-down attentional filtering).

Covers:
- Neutral attention baseline
- Domain boosting from advisory
- Decay toward neutral for non-dominant domains
- Survival override (threat forcing)
- Surprise bypass (pop-out effect)
- Prediction error widening
- Gating math (salience modulation formula)
- Integration with PatternBus
- Reset behavior
- Statistics tracking
- Edge cases (None advisory, empty signals, etc.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from mae_core.patterns.attentional_gate import (
    BOOST_PER_ADVISORY,
    DECAY_RATE,
    NEUTRAL_ATTENTION,
    PREDICTION_ERROR_THRESHOLD,
    SURPRISE_BYPASS_THRESHOLD,
    SURVIVAL_OVERRIDE_THRESHOLD,
    WIDENED_ATTENTION_TARGET,
    AttentionalGate,
)
from mae_core.patterns.pattern_bus import PatternBus
from mae_core.patterns.pattern_cortex import PatternAdvisory
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_signal(
    domain: PatternDomain = PatternDomain.THREAT,
    salience: float = 0.5,
    confidence: float = 0.7,
    source: str = "test",
) -> PatternSignal:
    """Create a test signal with sensible defaults."""
    return PatternSignal(
        source_system=source,
        domain=domain,
        form=PatternForm.REACTIVE,
        confidence=confidence,
        salience=salience,
        description=f"Test {domain.value} signal",
    )


def _make_advisory(
    dominant_domain: PatternDomain | None = None,
    threat_level: float = 0.0,
    opportunity_level: float = 0.0,
    novelty_level: float = 0.0,
) -> PatternAdvisory:
    """Create a test advisory with a dominant pattern."""
    dominant_pattern = None
    if dominant_domain is not None:
        dominant_pattern = _make_signal(domain=dominant_domain)
    return PatternAdvisory(
        step=1,
        dominant_pattern=dominant_pattern,
        threat_level=threat_level,
        opportunity_level=opportunity_level,
        novelty_level=novelty_level,
    )


# ── Neutral Attention Tests ──────────────────────────────────────────

class TestNeutralAttention:
    """All domains start at neutral (0.5) -- slight suppression."""

    def test_initial_attention_is_neutral(self):
        gate = AttentionalGate()
        for domain in PatternDomain:
            assert gate.attention_vector[domain] == NEUTRAL_ATTENTION

    def test_neutral_gating_math(self):
        """At neutral: gated = raw * (0.5 + 0.5 * 0.5) = raw * 0.75."""
        gate = AttentionalGate()
        sig = _make_signal(salience=0.8)
        gate.gate_signals([sig])
        expected = 0.8 * (0.5 + 0.5 * NEUTRAL_ATTENTION)  # 0.8 * 0.75 = 0.6
        assert abs(sig.salience - expected) < 1e-6

    def test_neutral_gate_does_not_block(self):
        """Even at neutral, signals get through (just suppressed)."""
        gate = AttentionalGate()
        sig = _make_signal(salience=0.4)
        gate.gate_signals([sig])
        assert sig.salience > 0.0


# ── Gating Formula Tests ────────────────────────────────────────────

class TestGatingFormula:
    """Verify the gating formula: gated = raw * (0.5 + 0.5 * attention)."""

    def test_full_attention_no_suppression(self):
        """At attention=1.0: gated = raw * 1.0 (no change)."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 1.0
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.6)
        gate.gate_signals([sig])
        assert abs(sig.salience - 0.6) < 1e-6

    def test_zero_attention_half_suppression(self):
        """At attention=0.0: gated = raw * 0.5."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.NOVELTY] = 0.0
        sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.8)
        gate.gate_signals([sig])
        expected = 0.8 * 0.5  # 0.4
        assert abs(sig.salience - expected) < 1e-6

    def test_salience_clamped_at_zero(self):
        """Gated salience never goes negative."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.BEHAVIORAL] = 0.0
        sig = _make_signal(domain=PatternDomain.BEHAVIORAL, salience=0.0)
        gate.gate_signals([sig])
        assert sig.salience >= 0.0

    def test_salience_clamped_at_one(self):
        """Gated salience never exceeds 1.0."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 1.0
        sig = _make_signal(domain=PatternDomain.THREAT, salience=1.0)
        gate.gate_signals([sig])
        assert sig.salience <= 1.0

    def test_multiple_signals_gated_independently(self):
        """Each signal is gated based on its own domain."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 1.0
        gate._attention[PatternDomain.NOVELTY] = 0.0

        threat_sig = _make_signal(domain=PatternDomain.THREAT, salience=0.5)
        novelty_sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.5)
        gate.gate_signals([threat_sig, novelty_sig])

        # Threat: 0.5 * 1.0 = 0.5 (no suppression)
        assert abs(threat_sig.salience - 0.5) < 1e-6
        # Novelty: 0.5 * 0.5 = 0.25 (max suppression)
        assert abs(novelty_sig.salience - 0.25) < 1e-6


# ── Surprise Bypass Tests ───────────────────────────────────────────

class TestSurpriseBypass:
    """Very salient signals bypass the gate entirely (pop-out effect)."""

    def test_high_salience_bypasses_gate(self):
        """Signals above SURPRISE_BYPASS_THRESHOLD are not modified."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.0  # Zero attention
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.95)
        gate.gate_signals([sig])
        # Should be unchanged (bypassed)
        assert sig.salience == 0.95

    def test_exact_threshold_is_not_bypassed(self):
        """Signal at exactly the threshold IS gated (> not >=)."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.0
        sig = _make_signal(
            domain=PatternDomain.THREAT,
            salience=SURPRISE_BYPASS_THRESHOLD,
        )
        gate.gate_signals([sig])
        # 0.9 * 0.5 = 0.45 (gated)
        assert sig.salience < SURPRISE_BYPASS_THRESHOLD

    def test_bypass_counter_incremented(self):
        """Statistics track bypassed signals."""
        gate = AttentionalGate()
        sig = _make_signal(salience=0.95)
        gate.gate_signals([sig])
        stats = gate.get_statistics()
        assert stats["signals_bypassed"] == 1
        assert stats["signals_gated"] == 0

    def test_mixed_bypass_and_gate(self):
        """Some signals bypass, others are gated -- both counted."""
        gate = AttentionalGate()
        high = _make_signal(salience=0.95)
        low = _make_signal(salience=0.3)
        gate.gate_signals([high, low])
        stats = gate.get_statistics()
        assert stats["signals_bypassed"] == 1
        assert stats["signals_gated"] == 1
        assert stats["total_signals_processed"] == 2


# ── Advisory Update Tests ───────────────────────────────────────────

class TestAdvisoryUpdate:
    """Top-down attention updates from PatternAdvisory."""

    def test_dominant_domain_boosted(self):
        """Dominant domain gets +BOOST_PER_ADVISORY."""
        gate = AttentionalGate()
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)
        expected = NEUTRAL_ATTENTION + BOOST_PER_ADVISORY
        assert abs(gate._attention[PatternDomain.THREAT] - expected) < 1e-6

    def test_non_dominant_domains_decay(self):
        """Non-dominant domains with above-neutral attention decay."""
        gate = AttentionalGate()
        # First boost NOVELTY
        gate._attention[PatternDomain.NOVELTY] = 0.8
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)
        # NOVELTY should decay toward neutral
        expected = 0.8 - DECAY_RATE  # 0.7
        assert abs(gate._attention[PatternDomain.NOVELTY] - expected) < 1e-6

    def test_below_neutral_domains_increase_toward_neutral(self):
        """Domains below neutral inch upward during decay."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.BEHAVIORAL] = 0.2
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)
        # Should move toward neutral (0.5)
        expected = 0.2 + DECAY_RATE  # 0.3
        assert abs(gate._attention[PatternDomain.BEHAVIORAL] - expected) < 1e-6

    def test_boost_capped_at_one(self):
        """Repeated boosting cannot exceed 1.0."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.95
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)
        assert gate._attention[PatternDomain.THREAT] == 1.0

    def test_decay_stops_at_neutral(self):
        """Decay cannot go below neutral."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.NOVELTY] = NEUTRAL_ATTENTION + 0.05
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)
        # 0.55 - 0.1 = 0.45, but floor is 0.5
        assert gate._attention[PatternDomain.NOVELTY] == NEUTRAL_ATTENTION

    def test_none_advisory_is_noop(self):
        """Passing None does nothing."""
        gate = AttentionalGate()
        original = dict(gate._attention)
        gate.update_attention(None)
        assert gate._attention == original

    def test_advisory_without_dominant_pattern(self):
        """Advisory with no dominant_pattern only decays, does not boost."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.8
        advisory = _make_advisory(dominant_domain=None)
        gate.update_attention(advisory)
        # THREAT should decay toward neutral
        assert gate._attention[PatternDomain.THREAT] < 0.8

    def test_successive_advisories_accumulate(self):
        """Multiple advisories for the same domain accumulate boost."""
        gate = AttentionalGate()
        for _ in range(3):
            advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
            gate.update_attention(advisory)
        expected = min(1.0, NEUTRAL_ATTENTION + 3 * BOOST_PER_ADVISORY)
        assert abs(gate._attention[PatternDomain.THREAT] - expected) < 1e-6

    def test_last_dominant_domain_tracked(self):
        """Statistics show the last dominant domain."""
        gate = AttentionalGate()
        advisory = _make_advisory(dominant_domain=PatternDomain.NOVELTY)
        gate.update_attention(advisory)
        stats = gate.get_statistics()
        assert stats["last_dominant_domain"] == "novelty"


# ── Survival Override Tests ─────────────────────────────────────────

class TestSurvivalOverride:
    """High threat forces THREAT domain to max attention (amygdala hijack)."""

    def test_high_threat_forces_max_attention(self):
        """When threat_level > threshold, THREAT domain goes to 1.0."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.3  # Below neutral
        advisory = _make_advisory(
            threat_level=SURVIVAL_OVERRIDE_THRESHOLD + 0.1,
        )
        gate.update_attention(advisory)
        assert gate._attention[PatternDomain.THREAT] == 1.0

    def test_low_threat_no_override(self):
        """When threat_level <= threshold, no forced override."""
        gate = AttentionalGate()
        advisory = _make_advisory(
            threat_level=SURVIVAL_OVERRIDE_THRESHOLD - 0.1,
        )
        gate.update_attention(advisory)
        # Should still be at neutral (no dominant pattern to boost)
        assert gate._attention[PatternDomain.THREAT] <= NEUTRAL_ATTENTION + 0.01

    def test_exact_threshold_no_override(self):
        """Exactly at threshold: no override (> not >=)."""
        gate = AttentionalGate()
        advisory = _make_advisory(
            threat_level=SURVIVAL_OVERRIDE_THRESHOLD,
        )
        gate.update_attention(advisory)
        assert gate._attention[PatternDomain.THREAT] < 1.0

    def test_survival_override_with_different_dominant(self):
        """Threat override works even when a different domain is dominant."""
        gate = AttentionalGate()
        advisory = _make_advisory(
            dominant_domain=PatternDomain.NOVELTY,
            threat_level=0.9,
        )
        gate.update_attention(advisory)
        # NOVELTY gets boosted, but THREAT is forced to 1.0
        assert gate._attention[PatternDomain.THREAT] == 1.0
        assert gate._attention[PatternDomain.NOVELTY] > NEUTRAL_ATTENTION


# ── Prediction Error Modulation Tests ───────────────────────────────

class TestPredictionErrorModulation:
    """High prediction error widens attention globally."""

    def test_set_prediction_error(self):
        """Can set prediction error."""
        gate = AttentionalGate()
        gate.set_prediction_error(0.8)
        assert gate.prediction_error == 0.8

    def test_prediction_error_clamped(self):
        """Prediction error clamped to [0, 1]."""
        gate = AttentionalGate()
        gate.set_prediction_error(5.0)
        assert gate.prediction_error == 1.0
        gate.set_prediction_error(-2.0)
        assert gate.prediction_error == 0.0

    def test_high_error_widens_attention(self):
        """When error > threshold, low-attention domains increase."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.BEHAVIORAL] = 0.2
        gate.set_prediction_error(0.9)
        # Trigger via update_attention
        advisory = _make_advisory()
        gate.update_attention(advisory)
        # BEHAVIORAL should have moved toward WIDENED_ATTENTION_TARGET
        assert gate._attention[PatternDomain.BEHAVIORAL] > 0.2

    def test_low_error_no_widening(self):
        """When error <= threshold, no widening occurs."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.BEHAVIORAL] = 0.2
        gate.set_prediction_error(0.3)
        advisory = _make_advisory()
        gate.update_attention(advisory)
        # Might have normal decay toward neutral but no extra widening
        # The decay from below neutral would move it up by DECAY_RATE
        # but not the extra widening amount
        assert gate._attention[PatternDomain.BEHAVIORAL] <= NEUTRAL_ATTENTION

    def test_widening_does_not_exceed_target(self):
        """Widening moves toward target, not above it."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.META] = 0.6
        gate.set_prediction_error(1.0)
        advisory = _make_advisory()
        gate.update_attention(advisory)
        # 0.6 is already below WIDENED_ATTENTION_TARGET (0.7), so it should
        # increase but not exceed 0.7
        assert gate._attention[PatternDomain.META] <= WIDENED_ATTENTION_TARGET

    def test_already_above_target_no_increase(self):
        """Domains already above target are not modified by widening."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.9
        gate.set_prediction_error(1.0)
        advisory = _make_advisory()
        gate.update_attention(advisory)
        # Should not increase above 0.9 due to widening (may decay from
        # non-dominant decay though)
        # Actually, decay moves it down by DECAY_RATE to 0.8
        assert gate._attention[PatternDomain.THREAT] >= WIDENED_ATTENTION_TARGET

    def test_is_widened_statistic(self):
        """Statistics report whether attention is currently widened."""
        gate = AttentionalGate()
        assert not gate.get_statistics()["is_widened"]
        gate.set_prediction_error(0.8)
        assert gate.get_statistics()["is_widened"]


# ── PatternBus Integration Tests ────────────────────────────────────

class TestPatternBusIntegration:
    """Attentional gate integrated with PatternBus.process_step()."""

    def _make_bus_with_gate(self, gate: AttentionalGate | None = None):
        """Create a PatternBus with an event bus mock and optional gate."""
        event_bus = MagicMock()
        event_bus.register_callback = MagicMock()
        if gate is None:
            gate = AttentionalGate()
        bus = PatternBus(event_bus=event_bus, attentional_gate=gate)
        return bus, gate

    def test_bus_with_gate_gates_signals(self):
        """PatternBus applies gate during process_step."""
        bus, gate = self._make_bus_with_gate()

        # Set THREAT to zero attention (max suppression)
        gate._attention[PatternDomain.THREAT] = 0.0

        # Inject a signal into the inbox
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.6)
        bus._inbox.append(sig)

        digest = bus.process_step(1)

        # Signal should be gated: 0.6 * 0.5 = 0.3
        assert len(digest.signals) == 1
        assert abs(digest.signals[0].salience - 0.3) < 1e-6

    def test_bus_without_gate_passes_through(self):
        """PatternBus without gate passes all signals unchanged."""
        event_bus = MagicMock()
        event_bus.register_callback = MagicMock()
        bus = PatternBus(event_bus=event_bus)

        sig = _make_signal(salience=0.6)
        bus._inbox.append(sig)

        digest = bus.process_step(1)
        assert abs(digest.signals[0].salience - 0.6) < 1e-6

    def test_bus_gate_statistics_in_bus_stats(self):
        """Gate statistics are included in bus statistics."""
        bus, gate = self._make_bus_with_gate()
        sig = _make_signal(salience=0.5)
        bus._inbox.append(sig)
        bus.process_step(1)

        stats = bus.get_statistics()
        assert "attentional_gate" in stats
        assert stats["attentional_gate"]["signals_gated"] == 1

    def test_bus_gate_bypass_high_salience(self):
        """High-salience signals bypass gate in PatternBus."""
        bus, gate = self._make_bus_with_gate()
        gate._attention[PatternDomain.THREAT] = 0.0

        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.95)
        bus._inbox.append(sig)

        digest = bus.process_step(1)
        # Bypassed: salience unchanged
        assert digest.signals[0].salience == 0.95


# ── Reset Tests ─────────────────────────────────────────────────────

class TestReset:
    """Gate can be reset to initial state."""

    def test_reset_restores_neutral(self):
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 1.0
        gate.set_prediction_error(0.9)
        gate.gate_signals([_make_signal()])
        gate.reset()

        for domain in PatternDomain:
            assert gate._attention[domain] == NEUTRAL_ATTENTION
        assert gate.prediction_error == 0.0
        stats = gate.get_statistics()
        assert stats["signals_gated"] == 0
        assert stats["signals_bypassed"] == 0

    def test_reset_clears_last_dominant(self):
        gate = AttentionalGate()
        gate.update_attention(_make_advisory(dominant_domain=PatternDomain.THREAT))
        gate.reset()
        assert gate.get_statistics()["last_dominant_domain"] is None


# ── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_signal_list(self):
        """Gating an empty list is a no-op."""
        gate = AttentionalGate()
        result = gate.gate_signals([])
        assert result == []
        assert gate.get_statistics()["total_signals_processed"] == 0

    def test_unknown_domain_uses_neutral(self):
        """If a signal's domain is not in the attention dict, neutral is used."""
        gate = AttentionalGate()
        # PatternDomain has all domains in init, but test the getattr default
        sig = _make_signal(domain=PatternDomain.STATE, salience=0.6)
        gate.gate_signals([sig])
        expected = 0.6 * (0.5 + 0.5 * NEUTRAL_ATTENTION)
        assert abs(sig.salience - expected) < 1e-6

    def test_repr_shows_attended_domains(self):
        """repr shows domains with above-neutral attention."""
        gate = AttentionalGate()
        gate._attention[PatternDomain.THREAT] = 0.9
        r = repr(gate)
        assert "threat" in r

    def test_gate_returns_same_list(self):
        """gate_signals returns the same list object (in-place modification)."""
        gate = AttentionalGate()
        signals = [_make_signal(), _make_signal()]
        result = gate.gate_signals(signals)
        assert result is signals

    def test_advisory_with_non_pattern_domain_dominant(self):
        """Advisory whose dominant_pattern has no domain attribute is handled."""
        gate = AttentionalGate()

        @dataclass
        class FakeAdvisory:
            dominant_pattern: Any = None
            threat_level: float = 0.0

        advisory = FakeAdvisory()
        advisory.dominant_pattern = "not a pattern signal"
        # Should not crash
        gate.update_attention(advisory)


# ── Full Feedback Loop Test ─────────────────────────────────────────

class TestFeedbackLoop:
    """End-to-end test of the sense -> gate -> cortex -> advisory -> gate loop."""

    def test_feedback_loop_amplifies_attended_domain(self):
        """Repeatedly attending to THREAT amplifies threat signals over time."""
        gate = AttentionalGate()

        # Step 1: Advisory says THREAT is dominant
        advisory1 = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory1)

        # Step 2: Gate a threat signal -- should be amplified
        sig1 = _make_signal(domain=PatternDomain.THREAT, salience=0.5)
        gate.gate_signals([sig1])
        after_one = sig1.salience

        # Step 3: Another advisory boosts THREAT further
        advisory2 = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory2)

        sig2 = _make_signal(domain=PatternDomain.THREAT, salience=0.5)
        gate.gate_signals([sig2])
        after_two = sig2.salience

        # Threat signals should get progressively more amplified
        assert after_two >= after_one

    def test_feedback_loop_suppresses_unattended_domain(self):
        """Attending THREAT suppresses NOVELTY signals."""
        gate = AttentionalGate()

        # Neutral baseline
        baseline_sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.5)
        gate.gate_signals([baseline_sig])
        neutral_salience = baseline_sig.salience

        # After attending to THREAT, NOVELTY stays at or below neutral
        advisory = _make_advisory(dominant_domain=PatternDomain.THREAT)
        gate.update_attention(advisory)

        test_sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.5)
        gate.gate_signals([test_sig])

        # NOVELTY should not increase (it decays or stays at neutral)
        assert test_sig.salience <= neutral_salience + 1e-6

    def test_prediction_error_restores_suppressed_signals(self):
        """High prediction error widens attention, restoring suppressed domains."""
        gate = AttentionalGate()

        # Suppress BEHAVIORAL to zero
        gate._attention[PatternDomain.BEHAVIORAL] = 0.0

        # Set high prediction error
        gate.set_prediction_error(1.0)

        # Update triggers widening
        advisory = _make_advisory()
        gate.update_attention(advisory)

        # BEHAVIORAL should have moved toward target (from 0.0 + decay + widening)
        assert gate._attention[PatternDomain.BEHAVIORAL] > 0.0
