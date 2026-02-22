"""Tests for PatternSignal, PatternForm, PatternDomain."""

from __future__ import annotations

import time

import pytest

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


# ── Construction ──────────────────────────────────────────────────────

class TestPatternSignalConstruction:
    def test_minimal_construction(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.8,
            salience=0.5,
            description="Test signal",
        )
        assert sig.source_system == "test"
        assert sig.domain == PatternDomain.NOVELTY
        assert sig.form == PatternForm.REACTIVE
        assert sig.confidence == 0.8
        assert sig.salience == 0.5
        assert sig.description == "Test signal"

    def test_auto_generated_id(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        assert len(sig.signal_id) == 12  # hex[:12]

    def test_unique_ids(self):
        ids = set()
        for _ in range(100):
            sig = PatternSignal(
                source_system="test",
                domain=PatternDomain.NOVELTY,
                form=PatternForm.REACTIVE,
                confidence=0.5,
                salience=0.5,
                description="Test",
            )
            ids.add(sig.signal_id)
        assert len(ids) == 100

    def test_auto_timestamp(self):
        before = time.time()
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        after = time.time()
        assert before <= sig.timestamp <= after

    def test_default_evidence_empty(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        assert sig.evidence == {}

    def test_default_ttl(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        assert sig.ttl_steps == 5

    def test_default_occurrence_count(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        assert sig.occurrence_count == 1


# ── Clamping ──────────────────────────────────────────────────────────

class TestPatternSignalClamping:
    def test_confidence_clamped_above(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=1.5,
            salience=0.5,
            description="Test",
        )
        assert sig.confidence == 1.0

    def test_confidence_clamped_below(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=-0.3,
            salience=0.5,
            description="Test",
        )
        assert sig.confidence == 0.0

    def test_salience_clamped_above(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=2.0,
            description="Test",
        )
        assert sig.salience == 1.0

    def test_salience_clamped_below(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=-0.1,
            description="Test",
        )
        assert sig.salience == 0.0


# ── TTL and Expiry ────────────────────────────────────────────────────

class TestPatternSignalTTL:
    def test_tick_decrements_ttl(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
            ttl_steps=3,
        )
        sig.tick()
        assert sig.ttl_steps == 2

    def test_not_expired_initially(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
        )
        assert not sig.is_expired

    def test_expired_after_ticks(self):
        sig = PatternSignal(
            source_system="test",
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=0.5,
            salience=0.5,
            description="Test",
            ttl_steps=2,
        )
        sig.tick()
        sig.tick()
        assert sig.is_expired


# ── Enums ─────────────────────────────────────────────────────────────

class TestEnums:
    def test_pattern_form_values(self):
        assert PatternForm.REACTIVE.value == "reactive"
        assert PatternForm.CORRELATED.value == "correlated"
        assert PatternForm.ANCESTRAL.value == "ancestral"
        assert len(PatternForm) == 3  # Triadic

    def test_pattern_domain_count(self):
        assert len(PatternDomain) == 10

    def test_pattern_domain_values(self):
        expected = {
            "novelty", "prediction", "causation", "threat",
            "opportunity", "capability", "failure", "behavioral",
            "state", "meta",
        }
        actual = {d.value for d in PatternDomain}
        assert actual == expected


# ── Repr ──────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_key_info(self):
        sig = PatternSignal(
            source_system="world_model",
            domain=PatternDomain.PREDICTION,
            form=PatternForm.REACTIVE,
            confidence=0.75,
            salience=0.6,
            description="Test",
        )
        r = repr(sig)
        assert "prediction" in r
        assert "reactive" in r
        assert "world_model" in r
        assert "0.75" in r
