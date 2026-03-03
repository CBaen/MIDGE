"""Tests for DeceptionDetector (Gift 5 — Deception Immune Response).

Verifies detection of pump-and-dump, retail trap, and wash trading
patterns in signal flow. No network calls, pure unit tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mae_core.market.edge.deception_detector import (
    DeceptionAssessment,
    DeceptionDetector,
)


class TestDeceptionDetectorConstruction:
    """Basic construction and interface."""

    def test_constructs_without_args(self):
        dd = DeceptionDetector()
        assert dd is not None

    def test_constructs_with_event_bus(self):
        dd = DeceptionDetector(event_bus=object())
        assert dd._event_bus is not None

    def test_statistics_on_fresh_detector(self):
        dd = DeceptionDetector()
        stats = dd.get_statistics()
        assert stats["tracked_tickers"] == 0
        assert stats["total_records"] == 0


class TestSignalRecording:
    """Test signal recording and history management."""

    def test_record_signal_creates_history(self):
        dd = DeceptionDetector()
        dd.record_signal("AAPL", "sentiment", "bullish", 0.7)
        assert "AAPL" in dd._signal_history
        assert len(dd._signal_history["AAPL"]) == 1

    def test_record_multiple_signals(self):
        dd = DeceptionDetector()
        for i in range(5):
            dd.record_signal("AAPL", "sentiment", "bullish", 0.5 + i * 0.05)
        assert len(dd._signal_history["AAPL"]) == 5

    def test_record_signal_with_custom_timestamp(self):
        dd = DeceptionDetector()
        ts = datetime(2026, 1, 1, 12, 0)
        dd.record_signal("AAPL", "sentiment", "bullish", 0.7, timestamp=ts)
        assert dd._signal_history["AAPL"][0].timestamp == ts

    def test_record_signal_ignores_empty_ticker(self):
        dd = DeceptionDetector()
        dd.record_signal("", "sentiment", "bullish", 0.7)
        assert len(dd._signal_history) == 0

    def test_statistics_after_recording(self):
        dd = DeceptionDetector()
        dd.record_signal("AAPL", "sentiment", "bullish", 0.7)
        dd.record_signal("TSLA", "insider", "bearish", 0.6)
        stats = dd.get_statistics()
        assert stats["tracked_tickers"] == 2
        assert stats["total_records"] == 2


class TestNoDeception:
    """Cases where no deception should be detected."""

    def test_unknown_ticker_returns_none(self):
        dd = DeceptionDetector()
        result = dd.assess_signal_authenticity("UNKNOWN")
        assert result.deception_probability == 0.0
        assert result.pattern_type == "none"

    def test_single_signal_returns_none(self):
        dd = DeceptionDetector()
        dd.record_signal("AAPL", "sentiment", "bullish", 0.7)
        result = dd.assess_signal_authenticity("AAPL")
        assert result.deception_probability == 0.0

    def test_balanced_social_and_insider(self):
        """Social + insider signals together = organic, not deception."""
        dd = DeceptionDetector()
        for _ in range(4):
            dd.record_signal("AAPL", "sentiment", "bullish", 0.7)
        dd.record_signal("AAPL", "insider", "bullish", 0.8)
        result = dd.assess_signal_authenticity("AAPL")
        # Insider present → pump_dump check fails
        assert result.pattern_type != "pump_dump" or result.deception_probability < 0.5

    def test_old_signals_ignored(self):
        """Signals outside the 48h window should not trigger detection."""
        dd = DeceptionDetector()
        old_ts = datetime.now() - timedelta(hours=72)
        for _ in range(5):
            dd.record_signal("AAPL", "sentiment", "bullish", 0.7, timestamp=old_ts)
        result = dd.assess_signal_authenticity("AAPL")
        assert result.deception_probability == 0.0


class TestPumpDump:
    """Pump-and-dump pattern detection."""

    def test_social_surge_no_insider_triggers(self):
        """3+ social signals + zero insider = pump_dump pattern."""
        dd = DeceptionDetector()
        for _ in range(4):
            dd.record_signal("MEME", "sentiment", "bullish", 0.8)
        result = dd.assess_signal_authenticity("MEME")
        assert result.pattern_type == "pump_dump"
        assert result.deception_probability >= 0.5

    def test_higher_social_count_increases_probability(self):
        dd = DeceptionDetector()
        for _ in range(6):
            dd.record_signal("MEME", "sentiment", "bullish", 0.8)
        result = dd.assess_signal_authenticity("MEME")
        assert result.deception_probability >= 0.6

    def test_uni_directional_signals_boost_probability(self):
        """All bullish social signals = more suspicious."""
        dd = DeceptionDetector()
        for _ in range(5):
            dd.record_signal("MEME", "sentiment", "bullish", 0.8)
        result = dd.assess_signal_authenticity("MEME")
        assert result.deception_probability >= 0.6

    def test_mixed_direction_social_less_suspicious(self):
        """Mixed bullish/bearish social is less suspicious than uni-directional."""
        dd = DeceptionDetector()
        dd.record_signal("MEME", "sentiment", "bullish", 0.8)
        dd.record_signal("MEME", "sentiment", "bearish", 0.6)
        dd.record_signal("MEME", "sentiment", "bullish", 0.7)
        dd.record_signal("MEME", "social", "bearish", 0.5)
        result = dd.assess_signal_authenticity("MEME")
        # With mixed directions, still flags but with lower uni-directional boost
        assert result.deception_probability < 0.8

    def test_social_domain_alias(self):
        """Both 'sentiment' and 'social' domains count as social."""
        dd = DeceptionDetector()
        dd.record_signal("MEME", "social", "bullish", 0.8)
        dd.record_signal("MEME", "social", "bullish", 0.7)
        dd.record_signal("MEME", "sentiment", "bullish", 0.6)
        result = dd.assess_signal_authenticity("MEME")
        assert result.pattern_type == "pump_dump"

    def test_evidence_populated(self):
        dd = DeceptionDetector()
        for _ in range(4):
            dd.record_signal("MEME", "sentiment", "bullish", 0.8)
        result = dd.assess_signal_authenticity("MEME")
        assert len(result.evidence) > 0
        assert any("social" in e.lower() or "Social" in e for e in result.evidence)


class TestRetailTrap:
    """Retail trap pattern detection."""

    def test_social_bullish_insider_bearish_triggers(self):
        """Social bullish + insider selling = retail trap."""
        dd = DeceptionDetector()
        dd.record_signal("TRAP", "sentiment", "bullish", 0.8)
        dd.record_signal("TRAP", "sentiment", "bullish", 0.7)
        dd.record_signal("TRAP", "insider", "bearish", 0.9)
        result = dd.assess_signal_authenticity("TRAP")
        assert result.pattern_type == "retail_trap"
        assert result.deception_probability >= 0.5

    def test_technical_breakout_boosts_trap(self):
        """Adding a technical breakout signal amplifies retail trap."""
        dd = DeceptionDetector()
        dd.record_signal("TRAP", "sentiment", "bullish", 0.8)
        dd.record_signal("TRAP", "sentiment", "bullish", 0.7)
        dd.record_signal("TRAP", "insider", "bearish", 0.9)
        dd.record_signal("TRAP", "technical", "bullish", 0.6)
        result = dd.assess_signal_authenticity("TRAP")
        assert result.deception_probability >= 0.6

    def test_aligned_social_and_insider_not_trap(self):
        """Social bullish + insider bullish = organic, not a trap."""
        dd = DeceptionDetector()
        dd.record_signal("REAL", "sentiment", "bullish", 0.8)
        dd.record_signal("REAL", "sentiment", "bullish", 0.7)
        dd.record_signal("REAL", "insider", "bullish", 0.9)
        result = dd.assess_signal_authenticity("REAL")
        assert result.pattern_type != "retail_trap"


class TestWashTrading:
    """Wash trading proxy detection."""

    def test_high_strength_technical_no_catalysts(self):
        """Strong technical signals with no catalysts → wash trading."""
        dd = DeceptionDetector()
        dd.record_signal("WASH", "technical", "bullish", 0.85)
        dd.record_signal("WASH", "technical", "bearish", 0.80)
        result = dd.assess_signal_authenticity("WASH")
        assert result.pattern_type == "wash_trading"
        assert result.deception_probability >= 0.4

    def test_technical_with_insider_catalyst_not_wash(self):
        """Technical + insider present = has catalyst, not suspicious."""
        dd = DeceptionDetector()
        dd.record_signal("REAL", "technical", "bullish", 0.85)
        dd.record_signal("REAL", "technical", "bullish", 0.80)
        dd.record_signal("REAL", "insider", "bullish", 0.7)
        result = dd.assess_signal_authenticity("REAL")
        assert result.pattern_type != "wash_trading"

    def test_weak_technical_signals_not_wash(self):
        """Low-strength technical signals are not suspicious."""
        dd = DeceptionDetector()
        dd.record_signal("WEAK", "technical", "bullish", 0.3)
        dd.record_signal("WEAK", "technical", "bearish", 0.4)
        result = dd.assess_signal_authenticity("WEAK")
        assert result.pattern_type != "wash_trading"


class TestDeceptionAssessmentDataclass:
    """Test the DeceptionAssessment dataclass."""

    def test_default_evidence_is_empty_list(self):
        a = DeceptionAssessment(
            ticker="X", deception_probability=0.0, pattern_type="none"
        )
        assert a.evidence == []

    def test_fields_accessible(self):
        a = DeceptionAssessment(
            ticker="AAPL",
            deception_probability=0.75,
            pattern_type="pump_dump",
            evidence=["test evidence"],
        )
        assert a.ticker == "AAPL"
        assert a.deception_probability == 0.75
        assert a.pattern_type == "pump_dump"
        assert len(a.evidence) == 1
