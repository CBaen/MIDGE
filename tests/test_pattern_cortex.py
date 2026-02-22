"""Tests for PatternCortex -- temporal integration and meta-pattern detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from mae_core.patterns.pattern_bus import PatternDigest
from mae_core.patterns.pattern_cortex import (
    TREND_THRESHOLD,
    WINDOW_SIZE,
    PatternAdvisory,
    PatternCortex,
)
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_signal(
    domain: PatternDomain = PatternDomain.NOVELTY,
    salience: float = 0.5,
    confidence: float = 0.7,
    source: str = "test",
    form: PatternForm = PatternForm.REACTIVE,
    description: str = "test signal",
) -> PatternSignal:
    return PatternSignal(
        source_system=source,
        domain=domain,
        form=form,
        confidence=confidence,
        salience=salience,
        description=description,
    )


def _make_digest(
    step: int,
    signals: list[PatternSignal] | None = None,
    correlated_groups: list[list[PatternSignal]] | None = None,
) -> PatternDigest:
    sigs = signals or []
    by_domain: dict[PatternDomain, list[PatternSignal]] = {}
    by_form: dict[PatternForm, list[PatternSignal]] = {}
    for s in sigs:
        by_domain.setdefault(s.domain, []).append(s)
        by_form.setdefault(s.form, []).append(s)

    dominant = None
    max_sal = 0.0
    for domain, dsigs in by_domain.items():
        sal = sum(s.salience for s in dsigs)
        if sal > max_sal:
            max_sal = sal
            dominant = domain

    return PatternDigest(
        step=step,
        signals=sigs,
        by_domain=by_domain,
        by_form=by_form,
        correlated_groups=correlated_groups or [],
        dominant_domain=dominant,
        aggregate_salience=sum(s.salience for s in sigs),
        signal_count=len(sigs),
    )


@dataclass
class FakeSearchResult:
    """Mimics a Qdrant SearchResult for ancestral recall tests."""
    payload: dict = field(default_factory=dict)
    score: float = 0.5


class FakeMemoryBridge:
    """Fake MemoryBridge that returns canned ancestral patterns."""

    def __init__(self, results: list | None = None):
        self._results = results or []
        self.queries: list[str] = []

    def recall_ancestral_patterns(
        self, query_text: str, limit: int = 5,
    ) -> list:
        self.queries.append(query_text)
        return self._results[:limit]


class BrokenMemoryBridge:
    """MemoryBridge that raises on recall."""

    def recall_ancestral_patterns(self, query_text: str, limit: int = 5):
        raise RuntimeError("Qdrant down")


# ── Construction ──────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_empty(self):
        cortex = PatternCortex()
        assert cortex.window_size == 0
        assert cortex.get_statistics()["total_advisories"] == 0

    def test_creates_with_memory_bridge(self):
        bridge = FakeMemoryBridge()
        cortex = PatternCortex(memory_bridge=bridge)
        assert cortex.get_statistics()["has_memory_bridge"] is True

    def test_creates_without_memory_bridge(self):
        cortex = PatternCortex()
        assert cortex.get_statistics()["has_memory_bridge"] is False

    def test_repr(self):
        cortex = PatternCortex()
        r = repr(cortex)
        assert "PatternCortex" in r
        assert "0/13" in r


# ── Basic Advisory Production ─────────────────────────────────────────

class TestAdvisoryProduction:
    def test_produces_advisory_from_empty_digest(self):
        cortex = PatternCortex()
        digest = _make_digest(step=1)
        adv = cortex.process_digest(digest)

        assert isinstance(adv, PatternAdvisory)
        assert adv.step == 1
        assert adv.dominant_pattern is None
        assert adv.confidence > 0

    def test_produces_advisory_with_signals(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.8)
        digest = _make_digest(step=1, signals=[sig])
        adv = cortex.process_digest(digest)

        assert adv.dominant_pattern is not None
        assert adv.dominant_pattern.domain == PatternDomain.THREAT

    def test_window_fills(self):
        cortex = PatternCortex()
        for i in range(WINDOW_SIZE + 3):
            cortex.process_digest(_make_digest(step=i))

        assert cortex.window_size == WINDOW_SIZE

    def test_advisories_accumulate(self):
        cortex = PatternCortex()
        for i in range(5):
            cortex.process_digest(_make_digest(step=i))

        stats = cortex.get_statistics()
        assert stats["total_advisories"] == 5

    def test_get_recent_advisories(self):
        cortex = PatternCortex()
        for i in range(10):
            cortex.process_digest(_make_digest(step=i))

        recent = cortex.get_recent_advisories(3)
        assert len(recent) == 3
        assert recent[-1].step == 9


# ── Trend Detection ──────────────────────────────────────────────────

class TestTrendDetection:
    def test_no_trend_before_threshold(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.THREAT)

        for i in range(TREND_THRESHOLD - 1):
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert len(adv.active_trends) == 0

    def test_trend_at_threshold(self):
        cortex = PatternCortex()

        for i in range(TREND_THRESHOLD):
            sig = _make_signal(domain=PatternDomain.THREAT)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert "threat" in adv.active_trends
        assert adv.active_trends["threat"] >= TREND_THRESHOLD

    def test_trend_breaks_on_absence(self):
        cortex = PatternCortex()

        # Build a threat streak
        for i in range(TREND_THRESHOLD):
            sig = _make_signal(domain=PatternDomain.THREAT)
            cortex.process_digest(_make_digest(step=i, signals=[sig]))

        # Step without threat breaks the streak
        adv = cortex.process_digest(_make_digest(step=TREND_THRESHOLD))
        assert "threat" not in adv.active_trends

    def test_multiple_simultaneous_trends(self):
        cortex = PatternCortex()

        for i in range(TREND_THRESHOLD):
            sigs = [
                _make_signal(domain=PatternDomain.THREAT),
                _make_signal(domain=PatternDomain.NOVELTY, source="curiosity"),
            ]
            adv = cortex.process_digest(_make_digest(step=i, signals=sigs))

        assert "threat" in adv.active_trends
        assert "novelty" in adv.active_trends

    def test_trend_generates_insight(self):
        cortex = PatternCortex()

        for i in range(TREND_THRESHOLD):
            sig = _make_signal(domain=PatternDomain.THREAT)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert any("Trend" in ins and "threat" in ins for ins in adv.correlated_insights)

    def test_trend_count_in_statistics(self):
        cortex = PatternCortex()

        for i in range(TREND_THRESHOLD + 2):
            sig = _make_signal(domain=PatternDomain.THREAT)
            cortex.process_digest(_make_digest(step=i, signals=[sig]))

        stats = cortex.get_statistics()
        assert stats["total_trends_detected"] > 0


# ── Meta-Pattern Detection ───────────────────────────────────────────

class TestMetaPatterns:
    def test_no_meta_with_few_advisories(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.9)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert len(adv.meta_patterns) == 0

    def test_meta_pattern_on_recurring_dominant(self):
        cortex = PatternCortex()

        # Same domain dominant for >= TREND_THRESHOLD advisories
        for i in range(TREND_THRESHOLD + 2):
            sig = _make_signal(domain=PatternDomain.THREAT, salience=0.9)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert len(adv.meta_patterns) > 0
        meta = adv.meta_patterns[0]
        assert meta.domain == PatternDomain.META
        assert meta.form == PatternForm.CORRELATED
        assert "threat" in meta.description.lower()

    def test_meta_pattern_confidence_scales_with_count(self):
        cortex = PatternCortex()

        for i in range(6):
            sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.8)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        if adv.meta_patterns:
            assert adv.meta_patterns[0].confidence > 0.5

    def test_meta_pattern_counted_in_statistics(self):
        cortex = PatternCortex()

        for i in range(5):
            sig = _make_signal(domain=PatternDomain.THREAT, salience=0.9)
            cortex.process_digest(_make_digest(step=i, signals=[sig]))

        stats = cortex.get_statistics()
        assert stats["total_meta_patterns"] > 0

    def test_no_meta_with_varying_dominants(self):
        cortex = PatternCortex()
        domains = [
            PatternDomain.THREAT,
            PatternDomain.NOVELTY,
            PatternDomain.CAUSATION,
            PatternDomain.CAPABILITY,
            PatternDomain.FAILURE,
        ]

        for i, domain in enumerate(domains):
            sig = _make_signal(domain=domain, salience=0.9)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert len(adv.meta_patterns) == 0


# ── Ancestral Recall ─────────────────────────────────────────────────

class TestAncestralRecall:
    def test_no_recall_without_bridge(self):
        cortex = PatternCortex()
        sig = _make_signal(salience=0.8)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.ancestral_matches == []

    def test_recall_with_bridge(self):
        results = [FakeSearchResult(payload={"domain": "threat"}, score=0.9)]
        bridge = FakeMemoryBridge(results=results)
        cortex = PatternCortex(memory_bridge=bridge)

        sig = _make_signal(salience=0.8)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert len(adv.ancestral_matches) == 1
        assert adv.ancestral_matches[0]["score"] == 0.9

    def test_no_recall_when_low_salience(self):
        bridge = FakeMemoryBridge(results=[FakeSearchResult()])
        cortex = PatternCortex(memory_bridge=bridge)

        sig = _make_signal(salience=0.1)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.ancestral_matches == []
        assert len(bridge.queries) == 0

    def test_no_recall_when_no_signals(self):
        bridge = FakeMemoryBridge(results=[FakeSearchResult()])
        cortex = PatternCortex(memory_bridge=bridge)

        adv = cortex.process_digest(_make_digest(step=1))
        assert adv.ancestral_matches == []

    def test_recall_query_includes_dominant_domain(self):
        bridge = FakeMemoryBridge(results=[])
        cortex = PatternCortex(memory_bridge=bridge)

        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.8)
        cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert len(bridge.queries) == 1
        assert "threat" in bridge.queries[0]

    def test_graceful_degradation_on_bridge_error(self):
        cortex = PatternCortex(memory_bridge=BrokenMemoryBridge())

        sig = _make_signal(salience=0.8)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.ancestral_matches == []

    def test_ancestral_query_count_in_statistics(self):
        bridge = FakeMemoryBridge(results=[])
        cortex = PatternCortex(memory_bridge=bridge)

        sig = _make_signal(salience=0.8)
        cortex.process_digest(_make_digest(step=1, signals=[sig]))

        stats = cortex.get_statistics()
        assert stats["total_ancestral_queries"] >= 1


# ── Domain Level Computation ─────────────────────────────────────────

class TestDomainLevels:
    def test_threat_level_zero_when_no_threats(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.NOVELTY)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.threat_level == 0.0

    def test_threat_level_positive_when_threats(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.THREAT, salience=0.8)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.threat_level > 0.0

    def test_recent_signals_weight_more(self):
        cortex = PatternCortex()

        # Old threat
        sig_old = _make_signal(domain=PatternDomain.THREAT, salience=0.9)
        cortex.process_digest(_make_digest(step=1, signals=[sig_old]))

        # Several steps without threat
        for i in range(5):
            cortex.process_digest(_make_digest(step=i + 2))

        adv = cortex.get_recent_advisories(1)[0]
        # Threat level should have decayed from the earlier signal
        assert adv.threat_level < 0.5

    def test_novelty_level_tracked(self):
        cortex = PatternCortex()
        sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.7)
        adv = cortex.process_digest(_make_digest(step=1, signals=[sig]))

        assert adv.novelty_level > 0.0


# ── Decision Tier Recommendation ─────────────────────────────────────

class TestTierRecommendation:
    def test_reflex_on_high_threat(self):
        cortex = PatternCortex()

        # Multiple high-threat steps to build level above 0.6
        for i in range(3):
            sig = _make_signal(domain=PatternDomain.THREAT, salience=0.95)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert adv.recommended_tier == "reflex"

    def test_prefrontal_on_high_novelty(self):
        cortex = PatternCortex()

        for i in range(3):
            sig = _make_signal(domain=PatternDomain.NOVELTY, salience=0.9)
            adv = cortex.process_digest(_make_digest(step=i, signals=[sig]))

        assert adv.recommended_tier == "prefrontal"

    def test_habit_on_normal(self):
        cortex = PatternCortex()
        adv = cortex.process_digest(_make_digest(step=1))

        assert adv.recommended_tier == "habit"

    def test_prefrontal_on_high_aggregate_salience(self):
        cortex = PatternCortex()
        sigs = [
            _make_signal(salience=0.8, source="a"),
            _make_signal(salience=0.7, source="b"),
            _make_signal(salience=0.6, source="c"),
        ]
        adv = cortex.process_digest(_make_digest(step=1, signals=sigs))

        assert adv.recommended_tier == "prefrontal"


# ── Confidence Computation ───────────────────────────────────────────

class TestConfidence:
    def test_base_confidence_on_empty(self):
        cortex = PatternCortex()
        adv = cortex.process_digest(_make_digest(step=1))

        assert adv.confidence == 0.3  # Base only

    def test_confidence_increases_with_signals(self):
        cortex = PatternCortex()
        sigs = [_make_signal() for _ in range(3)]
        adv = cortex.process_digest(_make_digest(step=1, signals=sigs))

        assert adv.confidence > 0.3

    def test_confidence_increases_with_correlations(self):
        cortex = PatternCortex()
        sig1 = _make_signal(source="a", domain=PatternDomain.THREAT)
        sig2 = _make_signal(source="b", domain=PatternDomain.THREAT)
        group = [sig1, sig2]
        adv = cortex.process_digest(
            _make_digest(step=1, signals=[sig1, sig2], correlated_groups=[group])
        )

        assert adv.confidence > 0.4

    def test_confidence_increases_with_ancestral(self):
        results = [FakeSearchResult(payload={"p": "test"}, score=0.8)]
        bridge = FakeMemoryBridge(results=results)
        cortex_with = PatternCortex(memory_bridge=bridge)
        cortex_without = PatternCortex()

        sig = _make_signal(salience=0.8)
        adv_with = cortex_with.process_digest(_make_digest(step=1, signals=[sig]))
        adv_without = cortex_without.process_digest(_make_digest(step=1, signals=[sig]))

        # Ancestral matches boost pre-penalty confidence by 0.15.
        # Both advisories get the same GWT uncorroborated penalty,
        # so the one WITH ancestral support should be strictly higher.
        assert adv_with.confidence > adv_without.confidence

    def test_confidence_capped_at_1(self):
        cortex = PatternCortex()
        # Many signals + correlations
        sigs = [_make_signal(source=f"s{i}") for i in range(20)]
        groups = [sigs[:5], sigs[5:10]]
        adv = cortex.process_digest(
            _make_digest(step=1, signals=sigs, correlated_groups=groups)
        )

        assert adv.confidence <= 1.0


# ── Correlated Insights ──────────────────────────────────────────────

class TestInsights:
    def test_correlation_generates_insight(self):
        cortex = PatternCortex()
        sig1 = _make_signal(source="haven", domain=PatternDomain.THREAT)
        sig2 = _make_signal(source="threat_det", domain=PatternDomain.THREAT)
        group = [sig1, sig2]
        adv = cortex.process_digest(
            _make_digest(step=1, signals=[sig1, sig2], correlated_groups=[group])
        )

        assert any("haven" in ins and "threat_det" in ins for ins in adv.correlated_insights)

    def test_no_insights_on_empty(self):
        cortex = PatternCortex()
        adv = cortex.process_digest(_make_digest(step=1))

        assert adv.correlated_insights == []
