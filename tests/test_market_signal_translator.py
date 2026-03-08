"""Tests for market signal translators.

Covers MarketConvergenceTranslator and MarketPartialTranslator.
"""

from __future__ import annotations

import json

import pytest

from mae_core.market.channels import CH_CONVERGENCE, CH_PARTIAL_CONVERGENCE
from mae_core.market.translators.market_signal_translator import (
    MarketConvergenceTranslator,
    MarketPartialTranslator,
)
from mae_core.patterns.pattern_signal import PatternDomain, PatternForm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conv_translator():
    return MarketConvergenceTranslator()


@pytest.fixture
def partial_translator():
    return MarketPartialTranslator()


def _convergence_msg(direction: str, confidence: float, domains: list[str]) -> dict:
    return {
        "direction": direction,
        "ticker": "AAPL",
        "confidence": confidence,
        "domains": domains,
        "domain_count": len(domains),
    }


def _partial_msg(domains_seen: list[str], confidence: float = 0.3) -> dict:
    return {
        "ticker": "TSLA",
        "domains_seen": domains_seen,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# MarketConvergenceTranslator tests
# ---------------------------------------------------------------------------

class TestMarketConvergenceTranslator:

    def test_bullish_convergence_becomes_opportunity(self, conv_translator):
        msg = _convergence_msg("bullish", 0.8, ["insider", "macro", "technical"])
        signal = conv_translator.translate(CH_CONVERGENCE, msg)

        assert signal is not None
        assert signal.domain == PatternDomain.OPPORTUNITY
        assert signal.form == PatternForm.CORRELATED
        assert signal.confidence == pytest.approx(0.8)
        assert signal.ttl_steps == 10

    def test_bearish_convergence_becomes_threat(self, conv_translator):
        msg = _convergence_msg("bearish", 0.7, ["insider", "events"])
        signal = conv_translator.translate(CH_CONVERGENCE, msg)

        assert signal is not None
        assert signal.domain == PatternDomain.THREAT
        assert signal.form == PatternForm.CORRELATED
        assert signal.confidence == pytest.approx(0.7)

    def test_neutral_direction_returns_none(self, conv_translator):
        msg = _convergence_msg("neutral", 0.5, ["macro"])
        signal = conv_translator.translate(CH_CONVERGENCE, msg)
        assert signal is None

    def test_missing_direction_returns_none(self, conv_translator):
        msg = {"ticker": "AAPL", "confidence": 0.6, "domains": ["macro"]}
        signal = conv_translator.translate(CH_CONVERGENCE, msg)
        assert signal is None

    def test_salience_scales_with_domain_count(self, conv_translator):
        """More domains AND higher confidence both raise salience."""
        few_domains = _convergence_msg("bullish", 0.5, ["insider", "macro"])
        many_domains = _convergence_msg(
            "bullish", 0.5, ["insider", "macro", "technical", "events",
                             "government", "positioning"]
        )
        sig_few = conv_translator.translate(CH_CONVERGENCE, few_domains)
        sig_many = conv_translator.translate(CH_CONVERGENCE, many_domains)

        assert sig_few is not None
        assert sig_many is not None
        assert sig_many.salience > sig_few.salience

    def test_salience_formula_capped_at_one(self, conv_translator):
        """Perfect confidence + maximum domain count must not exceed 1.0."""
        msg = _convergence_msg("bullish", 1.0, ["a", "b", "c", "d", "e",
                                                  "f", "g", "h", "i", "j",
                                                  "k", "l"])
        signal = conv_translator.translate(CH_CONVERGENCE, msg)
        assert signal is not None
        assert signal.salience <= 1.0

    def test_accepts_json_string_message(self, conv_translator):
        msg = json.dumps(_convergence_msg("bullish", 0.75, ["macro", "insider", "events"]))
        signal = conv_translator.translate(CH_CONVERGENCE, msg)
        assert signal is not None
        assert signal.domain == PatternDomain.OPPORTUNITY

    def test_non_dict_non_string_returns_none(self, conv_translator):
        assert conv_translator.translate(CH_CONVERGENCE, 42) is None
        assert conv_translator.translate(CH_CONVERGENCE, None) is None

    def test_source_name_and_channels(self, conv_translator):
        assert conv_translator.source_name == "market_convergence"
        assert CH_CONVERGENCE in conv_translator.channels


# ---------------------------------------------------------------------------
# MarketPartialTranslator tests
# ---------------------------------------------------------------------------

class TestMarketPartialTranslator:

    def test_partial_becomes_novelty(self, partial_translator):
        msg = _partial_msg(["insider", "macro"])
        signal = partial_translator.translate(CH_PARTIAL_CONVERGENCE, msg)

        assert signal is not None
        assert signal.domain == PatternDomain.NOVELTY
        assert signal.form == PatternForm.REACTIVE
        assert signal.ttl_steps == 15

    def test_salience_scales_with_domain_count(self, partial_translator):
        one = partial_translator.translate(CH_PARTIAL_CONVERGENCE, _partial_msg(["insider"]))
        two = partial_translator.translate(CH_PARTIAL_CONVERGENCE, _partial_msg(["insider", "macro"]))
        three = partial_translator.translate(CH_PARTIAL_CONVERGENCE, _partial_msg(["insider", "macro", "technical"]))

        assert one.salience == pytest.approx(0.2)
        assert two.salience == pytest.approx(0.4)
        assert three.salience == pytest.approx(0.6)

    def test_salience_capped_at_0_6(self, partial_translator):
        """Even many domains must not exceed the partial cap of 0.6."""
        msg = _partial_msg(["a", "b", "c", "d", "e", "f"])
        signal = partial_translator.translate(CH_PARTIAL_CONVERGENCE, msg)
        assert signal is not None
        assert signal.salience == pytest.approx(0.6)

    def test_empty_domains_gives_zero_salience(self, partial_translator):
        msg = _partial_msg([])
        signal = partial_translator.translate(CH_PARTIAL_CONVERGENCE, msg)
        assert signal is not None
        assert signal.salience == pytest.approx(0.0)

    def test_accepts_json_string_message(self, partial_translator):
        msg = json.dumps(_partial_msg(["macro", "events"]))
        signal = partial_translator.translate(CH_PARTIAL_CONVERGENCE, msg)
        assert signal is not None
        assert signal.domain == PatternDomain.NOVELTY

    def test_source_name_and_channels(self, partial_translator):
        assert partial_translator.source_name == "market_partial_convergence"
        assert CH_PARTIAL_CONVERGENCE in partial_translator.channels
