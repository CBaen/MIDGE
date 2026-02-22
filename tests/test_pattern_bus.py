"""Tests for PatternBus and PatternDigest."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from unittest.mock import MagicMock

import pytest

from mae_core.patterns.pattern_bus import PatternBus, PatternDigest
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


# ── Helpers ───────────────────────────────────────────────────────────

class FakeEventBus:
    """Minimal EventBus stub that stores callbacks."""

    def __init__(self):
        self._subscribers: dict[str, list] = defaultdict(list)

    def register_callback(self, channel: str, callback):
        self._subscribers[channel].append(callback)

    def publish(self, channel: str, message: Any):
        import json
        serialized = json.dumps(message) if isinstance(message, dict) else str(message)
        for cb in self._subscribers.get(channel, []):
            cb(channel, serialized)


class SimpleTranslator:
    """Test translator that translates any dict into a PatternSignal."""

    def __init__(self, name: str, channels: list[str], domain: PatternDomain):
        self._name = name
        self._channels = channels
        self._domain = domain

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def channels(self) -> list[str]:
        return self._channels

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None
        return PatternSignal(
            source_system=self._name,
            domain=self._domain,
            form=PatternForm.REACTIVE,
            confidence=0.7,
            salience=0.5,
            description=f"Signal from {self._name}",
            evidence=message,
        )


# ── Construction ──────────────────────────────────────────────────────

class TestPatternBusConstruction:
    def test_creates_empty(self):
        bus = PatternBus(FakeEventBus())
        assert bus.translator_count == 0

    def test_register_translator(self):
        bus = PatternBus(FakeEventBus())
        t = SimpleTranslator("test", ["ch.a"], PatternDomain.NOVELTY)
        bus.register_translator(t)
        assert bus.translator_count == 1

    def test_register_multiple(self):
        bus = PatternBus(FakeEventBus())
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))
        bus.register_translator(SimpleTranslator("b", ["ch.b"], PatternDomain.THREAT))
        assert bus.translator_count == 2


# ── Signal Collection ─────────────────────────────────────────────────

class TestSignalCollection:
    def test_collects_signal_from_event(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("test", ["ch.a"], PatternDomain.NOVELTY))

        eb.publish("ch.a", {"data": "hello"})
        digest = bus.process_step(1)

        assert digest.signal_count == 1
        assert digest.signals[0].source_system == "test"

    def test_no_signal_on_unknown_channel(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("test", ["ch.a"], PatternDomain.NOVELTY))

        eb.publish("ch.UNKNOWN", {"data": "hello"})
        digest = bus.process_step(1)

        assert digest.signal_count == 0

    def test_collects_from_multiple_translators(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))
        bus.register_translator(SimpleTranslator("b", ["ch.b"], PatternDomain.THREAT))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        assert digest.signal_count == 2

    def test_translator_returning_none_skips(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)

        class SkipTranslator:
            @property
            def source_name(self):
                return "skip"
            @property
            def channels(self):
                return ["ch.a"]
            def translate(self, channel, message):
                return None  # Always skip

        bus.register_translator(SkipTranslator())
        eb.publish("ch.a", {"data": 1})
        digest = bus.process_step(1)

        assert digest.signal_count == 0

    def test_handles_non_dict_message(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("test", ["ch.a"], PatternDomain.NOVELTY))

        # Non-dict message -- translator returns None
        eb.publish("ch.a", "just a string")
        digest = bus.process_step(1)

        assert digest.signal_count == 0


# ── Digest Fields ─────────────────────────────────────────────────────

class TestPatternDigest:
    def test_empty_digest(self):
        bus = PatternBus(FakeEventBus())
        digest = bus.process_step(42)

        assert digest.step == 42
        assert digest.signal_count == 0
        assert digest.dominant_domain is None
        assert digest.aggregate_salience == 0.0
        assert digest.correlated_groups == []

    def test_by_domain_grouping(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))
        bus.register_translator(SimpleTranslator("b", ["ch.b"], PatternDomain.THREAT))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        assert PatternDomain.NOVELTY in digest.by_domain
        assert PatternDomain.THREAT in digest.by_domain
        assert len(digest.by_domain[PatternDomain.NOVELTY]) == 1
        assert len(digest.by_domain[PatternDomain.THREAT]) == 1

    def test_by_form_grouping(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))

        eb.publish("ch.a", {"data": 1})
        digest = bus.process_step(1)

        assert PatternForm.REACTIVE in digest.by_form
        assert len(digest.by_form[PatternForm.REACTIVE]) == 1

    def test_dominant_domain_highest_salience(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)

        class HighSalienceTranslator:
            @property
            def source_name(self):
                return "high"
            @property
            def channels(self):
                return ["ch.h"]
            def translate(self, channel, message):
                return PatternSignal(
                    source_system="high",
                    domain=PatternDomain.THREAT,
                    form=PatternForm.REACTIVE,
                    confidence=0.9,
                    salience=0.95,
                    description="High threat",
                )

        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))
        bus.register_translator(HighSalienceTranslator())

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.h", {"data": 2})
        digest = bus.process_step(1)

        assert digest.dominant_domain == PatternDomain.THREAT

    def test_aggregate_salience(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.a", {"data": 2})
        digest = bus.process_step(1)

        # Each signal has salience=0.5
        assert abs(digest.aggregate_salience - 1.0) < 0.01


# ── Correlation Detection ─────────────────────────────────────────────

class TestCorrelation:
    def test_no_correlation_single_source(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.THREAT))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.a", {"data": 2})
        digest = bus.process_step(1)

        # Same source, same domain -> no correlation
        assert len(digest.correlated_groups) == 0

    def test_correlation_different_sources_same_domain(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("haven", ["ch.h"], PatternDomain.THREAT))
        bus.register_translator(SimpleTranslator("threat_det", ["ch.t"], PatternDomain.THREAT))

        eb.publish("ch.h", {"data": 1})
        eb.publish("ch.t", {"data": 2})
        digest = bus.process_step(1)

        assert len(digest.correlated_groups) == 1
        group = digest.correlated_groups[0]
        assert len(group) == 2
        sources = {sig.source_system for sig in group}
        assert sources == {"haven", "threat_det"}

    def test_correlated_signals_elevated_to_correlated_form(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("haven", ["ch.h"], PatternDomain.THREAT))
        bus.register_translator(SimpleTranslator("threat_det", ["ch.t"], PatternDomain.THREAT))

        eb.publish("ch.h", {"data": 1})
        eb.publish("ch.t", {"data": 2})
        digest = bus.process_step(1)

        for sig in digest.correlated_groups[0]:
            assert sig.form == PatternForm.CORRELATED

    def test_correlated_confidence_boosted(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.THREAT))
        bus.register_translator(SimpleTranslator("b", ["ch.b"], PatternDomain.THREAT))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        # Original confidence was 0.7
        for sig in digest.correlated_groups[0]:
            assert sig.confidence > 0.7

    def test_no_correlation_different_domains(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))
        bus.register_translator(SimpleTranslator("b", ["ch.b"], PatternDomain.THREAT))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        # Different domains -> no correlation
        assert len(digest.correlated_groups) == 0


# ── Cross-Domain Correlation ─────────────────────────────────────────

class HighSalienceTranslator:
    """Translator with configurable salience for cross-domain tests."""

    def __init__(self, name: str, channels: list[str], domain: PatternDomain, salience: float = 0.5):
        self._name = name
        self._channels = channels
        self._domain = domain
        self._salience = salience

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def channels(self) -> list[str]:
        return self._channels

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        if not isinstance(message, dict):
            return None
        return PatternSignal(
            source_system=self._name,
            domain=self._domain,
            form=PatternForm.REACTIVE,
            confidence=0.7,
            salience=self._salience,
            description=f"Signal from {self._name}",
            evidence=message,
        )


class TestCrossDomainCorrelation:
    def test_threat_novelty_produces_cross_domain(self):
        """THREAT + NOVELTY in same step → 'novel threat' cross-domain group."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("threat_src", ["ch.t"], PatternDomain.THREAT, 0.6))
        bus.register_translator(HighSalienceTranslator("novelty_src", ["ch.n"], PatternDomain.NOVELTY, 0.5))

        eb.publish("ch.t", {"data": 1})
        eb.publish("ch.n", {"data": 2})
        digest = bus.process_step(1)

        assert len(digest.cross_domain_groups) >= 1
        # Group contains best_a, best_b, and synthetic
        group = digest.cross_domain_groups[0]
        assert len(group) == 3
        synthetic = [s for s in group if s.source_system == "cross_domain"]
        assert len(synthetic) == 1
        assert synthetic[0].form == PatternForm.CORRELATED
        assert "Cross-domain" in synthetic[0].description

    def test_cross_domain_requires_min_salience(self):
        """Both domains must have salience >= 0.3."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("threat_src", ["ch.t"], PatternDomain.THREAT, 0.5))
        bus.register_translator(HighSalienceTranslator("novelty_src", ["ch.n"], PatternDomain.NOVELTY, 0.1))

        eb.publish("ch.t", {"data": 1})
        eb.publish("ch.n", {"data": 2})
        digest = bus.process_step(1)

        # Novelty too low → no cross-domain
        assert len(digest.cross_domain_groups) == 0

    def test_cross_domain_synthetic_salience_boosted(self):
        """Synthetic signal gets max(saliences) + 0.1 bonus."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("a", ["ch.a"], PatternDomain.THREAT, 0.6))
        bus.register_translator(HighSalienceTranslator("b", ["ch.b"], PatternDomain.NOVELTY, 0.5))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        synthetic = [s for g in digest.cross_domain_groups for s in g if s.source_system == "cross_domain"]
        assert len(synthetic) == 1
        assert abs(synthetic[0].salience - 0.7) < 0.01  # max(0.6, 0.5) + 0.1

    def test_cross_domain_confidence_geometric_mean(self):
        """Synthetic confidence = sqrt(conf_a * conf_b)."""
        import math
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("a", ["ch.a"], PatternDomain.THREAT, 0.5))
        bus.register_translator(HighSalienceTranslator("b", ["ch.b"], PatternDomain.NOVELTY, 0.5))

        eb.publish("ch.a", {"data": 1})
        eb.publish("ch.b", {"data": 2})
        digest = bus.process_step(1)

        synthetic = [s for g in digest.cross_domain_groups for s in g if s.source_system == "cross_domain"]
        expected_conf = math.sqrt(0.7 * 0.7)  # Both translators use confidence=0.7
        assert abs(synthetic[0].confidence - expected_conf) < 0.01

    def test_no_cross_domain_when_single_domain(self):
        """Single domain present → no cross-domain groups."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("a", ["ch.a"], PatternDomain.THREAT, 0.8))

        eb.publish("ch.a", {"data": 1})
        digest = bus.process_step(1)

        assert len(digest.cross_domain_groups) == 0

    def test_empty_digest_has_no_cross_domain(self):
        """Empty step → no cross-domain groups."""
        bus = PatternBus(FakeEventBus())
        digest = bus.process_step(1)
        assert digest.cross_domain_groups == []

    def test_multiple_cross_domain_pairs(self):
        """Multiple qualifying pairs detected in same step."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        # THREAT + NOVELTY and NOVELTY + CAUSATION are both high-value pairs
        bus.register_translator(HighSalienceTranslator("t", ["ch.t"], PatternDomain.THREAT, 0.5))
        bus.register_translator(HighSalienceTranslator("n", ["ch.n"], PatternDomain.NOVELTY, 0.5))
        bus.register_translator(HighSalienceTranslator("c", ["ch.c"], PatternDomain.CAUSATION, 0.5))

        eb.publish("ch.t", {"data": 1})
        eb.publish("ch.n", {"data": 2})
        eb.publish("ch.c", {"data": 3})
        digest = bus.process_step(1)

        # Should detect at least THREAT+NOVELTY and NOVELTY+CAUSATION
        assert len(digest.cross_domain_groups) >= 2

    def test_cross_domain_increments_total_correlations(self):
        """Cross-domain detection increments the bus correlation counter."""
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(HighSalienceTranslator("t", ["ch.t"], PatternDomain.THREAT, 0.5))
        bus.register_translator(HighSalienceTranslator("n", ["ch.n"], PatternDomain.NOVELTY, 0.5))

        eb.publish("ch.t", {"data": 1})
        eb.publish("ch.n", {"data": 2})
        bus.process_step(1)

        assert bus.get_statistics()["total_correlations"] >= 1


# ── Recent Digests ────────────────────────────────────────────────────

class TestRecentDigests:
    def test_stores_recent(self):
        bus = PatternBus(FakeEventBus())
        bus.process_step(1)
        bus.process_step(2)
        bus.process_step(3)

        recent = bus.get_recent_digests(2)
        assert len(recent) == 2
        assert recent[-1].step == 3

    def test_returns_all_if_fewer_than_n(self):
        bus = PatternBus(FakeEventBus())
        bus.process_step(1)

        recent = bus.get_recent_digests(10)
        assert len(recent) == 1


# ── Statistics ────────────────────────────────────────────────────────

class TestStatistics:
    def test_statistics_keys(self):
        bus = PatternBus(FakeEventBus())
        stats = bus.get_statistics()
        assert "translators" in stats
        assert "total_signals" in stats
        assert "total_correlations" in stats

    def test_signal_count_accumulates(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))

        eb.publish("ch.a", {"data": 1})
        bus.process_step(1)
        eb.publish("ch.a", {"data": 2})
        bus.process_step(2)

        assert bus.get_statistics()["total_signals"] == 2


# ── Budget Limit ──────────────────────────────────────────────────────

class TestBudget:
    def test_max_signals_per_step(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))

        # Flood with more than MAX_SIGNALS_PER_STEP
        for i in range(100):
            eb.publish("ch.a", {"data": i})

        digest = bus.process_step(1)
        assert digest.signal_count <= PatternBus.MAX_SIGNALS_PER_STEP

    def test_overflow_preserved_for_next_step(self):
        eb = FakeEventBus()
        bus = PatternBus(eb)
        bus.register_translator(SimpleTranslator("a", ["ch.a"], PatternDomain.NOVELTY))

        for i in range(60):
            eb.publish("ch.a", {"data": i})

        d1 = bus.process_step(1)
        d2 = bus.process_step(2)

        assert d1.signal_count + d2.signal_count == 60
