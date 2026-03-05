"""Tests for plain-language alert formatting — zero jargon output."""

import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from mae_core.market.plain_language import (
    DIRECTION_PLAIN,
    DOMAIN_PLAIN,
    SOURCE_PLAIN,
    COMPANY_DESCRIPTIONS,
    _describe_company,
    _translate_direction,
    _translate_domains,
    _section_what,
    _section_history,
    _section_timing,
    _section_action,
    _section_tracking,
    format_pattern_stack_alert,
    format_convergence_alert,
    write_plain_alert,
)
from mae_core.market.archaeology.fingerprint import PatternTemplate


# ── Jargon Blacklist ─────────────────────────────────────────────

JARGON_TERMS = [
    "bullish", "bearish", "insider", "macro", "technical",
    "positioning", "RSI", "MACD", "Bollinger", "sec_form",
    "fred_macro", "cot_", "ta_", "finviz", "convergence_combo",
]


def _contains_jargon(text: str) -> list[str]:
    """Return any jargon terms found in the text."""
    found = []
    lower = text.lower()
    for term in JARGON_TERMS:
        if term.lower() in lower:
            found.append(term)
    return found


# ── Translation Tests ────────────────────────────────────────────


class TestTranslationMaps:

    def test_all_directions_have_translations(self):
        for key in ("bullish", "bearish", "up", "down"):
            assert key in DIRECTION_PLAIN
            assert "financial" not in DIRECTION_PLAIN[key].lower()

    def test_all_domains_have_translations(self):
        for key in ("insider", "macro", "technical", "events", "positioning",
                     "government", "contracts", "sentiment"):
            assert key in DOMAIN_PLAIN

    def test_company_descriptions_are_plain(self):
        for ticker, desc in COMPANY_DESCRIPTIONS.items():
            assert len(desc) > 5
            assert "Inc." not in desc  # No corporate suffixes

    def test_describe_known_company(self):
        result = _describe_company("NVDA")
        assert "NVIDIA" in result
        assert "semiconductor" in result

    def test_describe_unknown_company(self):
        result = _describe_company("XYZQ")
        assert "XYZQ" in result
        assert "publicly traded" in result

    def test_translate_direction(self):
        assert "increase" in _translate_direction("bullish")
        assert "decrease" in _translate_direction("bearish")

    def test_translate_domains(self):
        result = _translate_domains(["insider", "technical"])
        assert len(result) == 2
        assert all(isinstance(r, str) for r in result)
        # No raw domain codes in output
        assert "insider" not in result[0]


# ── Section Tests ────────────────────────────────────────────────


class TestSections:

    def test_what_section_no_jargon(self):
        text = _section_what("NVDA", "bullish", ["insider", "technical"], 3)
        jargon = _contains_jargon(text)
        assert jargon == [], f"Jargon found in WHAT section: {jargon}"
        assert "NVIDIA" in text
        assert "WHAT:" in text

    def test_what_section_single_pattern(self):
        text = _section_what("AAPL", "bearish", ["events"], 1)
        assert "detected a pattern" in text
        assert "decrease" in text

    def test_history_section_with_data(self):
        text = _section_history(50, 12, 0.35, 20)
        assert "50 times" in text
        assert "12 different" in text
        assert "35%" in text

    def test_history_section_new_pattern(self):
        text = _section_history(0, 0, None, 0)
        assert "new pattern" in text

    def test_history_section_small_sample(self):
        text = _section_history(10, 3, 0.5, 3)
        assert "not enough" in text

    def test_timing_dynamic(self):
        text = _section_timing(7, "dynamic")
        assert "7 days" in text
        assert "historical" in text

    def test_timing_fallback(self):
        text = _section_timing(14, "fallback")
        assert "14 days" in text
        assert "enough timing data" in text

    def test_action_section(self):
        text = _section_action("high", 0.82)
        assert "strong" in text
        assert "82%" in text
        assert "information" in text  # disclaimer
        jargon = _contains_jargon(text)
        assert jargon == [], f"Jargon in ACTION: {jargon}"

    def test_tracking_section(self):
        text = _section_tracking("MSFT")
        assert "MSFT" in text
        assert "monitor" in text


# ── Full Alert Formatting ────────────────────────────────────────


class TestFormatPatternStackAlert:

    def _make_stack(self):
        tmpl = PatternTemplate(
            template_id="t1", direction="bullish",
            domain_signature="insider+technical",
            domains=["insider", "technical"],
            lag_profile_normalized={"short": 0.6, "medium": 0.4},
            n_instances=25,
            symbols_seen=["NVDA", "AAPL", "MSFT", "AMD"],
            wins=8, losses=12,
        )
        act = SimpleNamespace(
            template=tmpl, match_score=0.8,
            matched_domains=["insider", "technical"],
            missing_domains=[],
            matched_sources=["sec_form4", "ta_rsi"],
            missing_sources=[],
        )
        stack = SimpleNamespace(
            symbol="NVDA",
            direction="bullish",
            activations=[act],
            stack_confidence=0.72,
            tier="medium",
            independent_pairs=1,
            created_at="2024-01-15T12:00:00",
        )
        return stack

    def test_full_alert_no_jargon(self):
        stack = self._make_stack()
        text = format_pattern_stack_alert(stack, window_days=7, window_source="dynamic")
        jargon = _contains_jargon(text)
        assert jargon == [], f"Jargon in full alert: {jargon}"

    def test_full_alert_has_all_sections(self):
        stack = self._make_stack()
        text = format_pattern_stack_alert(stack, window_days=7, window_source="dynamic")
        assert "WHAT:" in text
        assert "HISTORY:" in text
        assert "TIMING:" in text
        assert "ACTION:" in text
        assert "TRACKING:" in text

    def test_full_alert_mentions_ticker(self):
        stack = self._make_stack()
        text = format_pattern_stack_alert(stack, window_days=7, window_source="dynamic")
        assert "NVDA" in text
        assert "NVIDIA" in text

    def test_full_alert_has_win_rate(self):
        stack = self._make_stack()
        text = format_pattern_stack_alert(stack, window_days=7, window_source="dynamic")
        assert "40%" in text  # 8/(8+12)

    def test_full_alert_has_timing(self):
        stack = self._make_stack()
        text = format_pattern_stack_alert(stack, window_days=7, window_source="dynamic")
        assert "7 days" in text


class TestFormatConvergenceAlert:

    def test_convergence_alert_no_jargon(self):
        alert = SimpleNamespace(
            direction="bearish",
            domains_converging=["events", "macro", "technical"],
            confidence=0.65,
            strength=0.7,
        )
        text = format_convergence_alert(alert, "TSLA", window_days=14)
        jargon = _contains_jargon(text)
        assert jargon == [], f"Jargon in convergence alert: {jargon}"

    def test_convergence_alert_has_sections(self):
        alert = SimpleNamespace(
            direction="bullish",
            domains_converging=["insider", "events"],
            confidence=0.55,
            strength=0.6,
        )
        text = format_convergence_alert(alert, "AAPL", window_days=10)
        assert "WHAT:" in text
        assert "TRACKING:" in text
        assert "Apple" in text


# ── JSONL Writer Test ────────────────────────────────────────────


class TestWritePlainAlert:

    def test_write_creates_file(self, tmp_path, monkeypatch):
        import mae_core.market.plain_language as pl
        monkeypatch.setattr(pl, "ALERTS_PATH", tmp_path / "alerts.jsonl")

        write_plain_alert(
            "Test message", "NVDA", "bullish",
            metadata={"test": True},
        )
        lines = (tmp_path / "alerts.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ticker"] == "NVDA"
        assert record["message"] == "Test message"
        assert record["metadata"]["test"] is True

    def test_write_appends(self, tmp_path, monkeypatch):
        import mae_core.market.plain_language as pl
        monkeypatch.setattr(pl, "ALERTS_PATH", tmp_path / "alerts.jsonl")

        write_plain_alert("First", "NVDA", "bullish")
        write_plain_alert("Second", "AAPL", "bearish")
        lines = (tmp_path / "alerts.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
