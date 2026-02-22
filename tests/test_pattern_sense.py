"""Tests for PatternSense -- per-agent pattern membrane."""

from __future__ import annotations

import time

import pytest

from mae_core.patterns.pattern_sense import PatternSense, SenseResult
from mae_core.patterns.pattern_signal import PatternDomain, PatternForm


# ── Construction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_creates_with_agent_id(self):
        ps = PatternSense("agent-0")
        assert ps.agent_id == "agent-0"

    def test_repr(self):
        ps = PatternSense("agent-1")
        assert "agent-1" in repr(ps)

    def test_statistics_empty(self):
        ps = PatternSense("agent-0")
        stats = ps.get_statistics()
        assert stats["agent_id"] == "agent-0"
        assert stats["rewards_buffered"] == 0


# ── Reward Trend Detection ───────────────────────────────────────────

class TestRewardTrend:
    def test_no_trend_below_threshold(self):
        ps = PatternSense("a")
        # Only 2 steps -- below Rule of Three
        ps.sense(0.1, "act", 1)
        result = ps.sense(0.2, "act", 2)
        assert all(s.domain != PatternDomain.OPPORTUNITY or "trend" not in s.description.lower()
                    for s in result.signals)

    def test_increasing_trend_detected(self):
        ps = PatternSense("a")
        ps.sense(0.1, "act", 1)
        ps.sense(0.2, "act", 2)
        ps.sense(0.3, "act", 3)
        result = ps.sense(0.4, "act", 4)

        trend_signals = [s for s in result.signals if "trend UP" in s.description]
        assert len(trend_signals) == 1
        assert trend_signals[0].domain == PatternDomain.OPPORTUNITY

    def test_decreasing_trend_detected(self):
        ps = PatternSense("a")
        ps.sense(0.4, "act", 1)
        ps.sense(0.3, "act", 2)
        ps.sense(0.2, "act", 3)
        result = ps.sense(0.1, "act", 4)

        trend_signals = [s for s in result.signals if "trend DOWN" in s.description]
        assert len(trend_signals) == 1
        assert trend_signals[0].domain == PatternDomain.THREAT

    def test_trend_includes_agent_source(self):
        ps = PatternSense("agent-5")
        for i in range(5):
            result = ps.sense(0.1 * (i + 1), "act", i + 1)

        trend = [s for s in result.signals if "trend" in s.description.lower()]
        assert len(trend) >= 1
        assert trend[0].source_system == "agent:agent-5"

    def test_no_trend_when_flat(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.5, "act", i + 1)

        trend = [s for s in result.signals if "trend" in s.description.lower()]
        assert len(trend) == 0

    def test_trend_confidence_increases_with_length(self):
        ps = PatternSense("a")
        # 4-step trend
        for i in range(5):
            result = ps.sense(0.1 * (i + 1), "act", i + 1)
        conf_4 = [s for s in result.signals if "trend UP" in s.description][0].confidence

        ps2 = PatternSense("b")
        # 6-step trend
        for i in range(7):
            result2 = ps2.sense(0.1 * (i + 1), "act", i + 1)
        conf_6 = [s for s in result2.signals if "trend UP" in s.description][0].confidence

        assert conf_6 > conf_4


# ── Action Repetition Detection ──────────────────────────────────────

class TestActionRepetition:
    def test_no_repetition_diverse_actions(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.5, f"act-{i}", i + 1)

        rep = [s for s in result.signals if "repetition" in s.description.lower()]
        assert len(rep) == 0

    def test_repetition_detected(self):
        ps = PatternSense("a")
        for i in range(4):
            result = ps.sense(0.5, "same_action", i + 1)

        rep = [s for s in result.signals if "repetition" in s.description.lower()]
        assert len(rep) == 1
        assert rep[0].domain == PatternDomain.BEHAVIORAL
        assert "same_action" in rep[0].description

    def test_repetition_needs_threshold(self):
        ps = PatternSense("a")
        # Only 2 of same action (below threshold of 3)
        ps.sense(0.5, "other", 1)
        ps.sense(0.5, "other", 2)
        ps.sense(0.5, "act-X", 3)
        result = ps.sense(0.5, "act-X", 4)

        rep = [s for s in result.signals if "repetition" in s.description.lower()]
        assert len(rep) == 0

    def test_repetition_source_system(self):
        ps = PatternSense("agent-3")
        for i in range(4):
            result = ps.sense(0.5, "loop", i + 1)

        rep = [s for s in result.signals if "repetition" in s.description.lower()]
        assert rep[0].source_system == "agent:agent-3"


# ── Reward Surprise Detection ────────────────────────────────────────

class TestRewardSurprise:
    def test_no_surprise_when_stable(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.5, "act", i + 1)

        surprise = [s for s in result.signals if "surprise" in s.description.lower()]
        assert len(surprise) == 0

    def test_positive_surprise_detected(self):
        ps = PatternSense("a")
        # Build a stable baseline, then spike
        for i in range(6):
            ps.sense(0.5, "act", i + 1)
        result = ps.sense(2.0, "act", 7)  # Big positive spike

        surprise = [s for s in result.signals if "surprise" in s.description.lower()]
        assert len(surprise) == 1
        assert surprise[0].domain == PatternDomain.OPPORTUNITY
        assert "positive" in surprise[0].description

    def test_negative_surprise_detected(self):
        ps = PatternSense("a")
        for i in range(6):
            ps.sense(0.5, "act", i + 1)
        result = ps.sense(-1.0, "act", 7)  # Big negative spike

        surprise = [s for s in result.signals if "surprise" in s.description.lower()]
        assert len(surprise) == 1
        assert surprise[0].domain == PatternDomain.THREAT
        assert "negative" in surprise[0].description

    def test_no_surprise_with_zero_variance(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.5, "act", i + 1)

        # All identical → no variance → no surprise
        surprise = [s for s in result.signals if "surprise" in s.description.lower()]
        assert len(surprise) == 0

    def test_surprise_includes_z_score(self):
        ps = PatternSense("a")
        for i in range(6):
            ps.sense(0.5, "act", i + 1)
        result = ps.sense(2.0, "act", 7)

        surprise = [s for s in result.signals if "surprise" in s.description.lower()]
        assert "z=" in surprise[0].description


# ── SenseResult ──────────────────────────────────────────────────────

class TestSenseResult:
    def test_result_has_step(self):
        ps = PatternSense("a")
        result = ps.sense(0.5, "act", 42)
        assert result.step == 42

    def test_result_signals_are_pattern_signals(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.1 * (i + 1), "act", i + 1)

        for sig in result.signals:
            assert isinstance(sig.source_system, str)
            assert isinstance(sig.domain, PatternDomain)
            assert isinstance(sig.form, PatternForm)

    def test_all_signals_reactive_form(self):
        ps = PatternSense("a")
        for i in range(6):
            ps.sense(0.5, "same", i + 1)
        result = ps.sense(2.0, "same", 7)

        for sig in result.signals:
            assert sig.form == PatternForm.REACTIVE


# ── Performance ──────────────────────────────────────────────────────

class TestPerformance:
    def test_sense_under_500_microseconds(self):
        ps = PatternSense("a")
        # Warm up
        for i in range(8):
            ps.sense(0.1 * i, "act", i)

        # Measure
        start = time.perf_counter()
        iterations = 1000
        for i in range(iterations):
            ps.sense(0.5, "act", 100 + i)
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1_000_000
        assert avg_us < 500, f"Average {avg_us:.0f}us exceeds 500us target"


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_window_returns_empty(self):
        ps = PatternSense("a")
        result = ps.sense(0.5, "act", 1)
        # First call can't detect trends (need 3+)
        assert isinstance(result, SenseResult)

    def test_none_action_handled(self):
        ps = PatternSense("a")
        for i in range(5):
            result = ps.sense(0.5, None, i + 1)
        # None action repeated → should still detect
        rep = [s for s in result.signals if "repetition" in s.description.lower()]
        assert len(rep) >= 1

    def test_custom_window_size(self):
        ps = PatternSense("a", window_size=4)
        for i in range(10):
            ps.sense(0.5, "act", i + 1)
        assert ps.get_statistics()["rewards_buffered"] == 4
