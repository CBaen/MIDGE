"""Integration/wiring tests for MIDGE's 5 new Layer 6 data sources.

Verifies that the routing tables, Thompson key maps, and learning config
are all consistent with each other and with the signal converters that
populate MarketSignal.source. These are structural contract tests — no
network calls, no market data, no mocks of real APIs.

Sources under test (Layer 6):
  - cot_positioning     (COTClient)
  - stocktwits_sentiment (StockTwitsClient)
  - vix_term_structure  (VIXClient)
  - google_trends       (TrendsClient)
  - finnhub_economic    (FinnhubClient.get_economic_calendar)
  - finnhub_analyst     (FinnhubClient.get_analyst_recommendations)
  - finnhub_earnings_calendar  (FinnhubClient.get_earnings_calendar — TIER/THOMPSON entry)
"""

from __future__ import annotations

from datetime import datetime, date
from types import SimpleNamespace
from typing import Any

import pytest

from mae_core.market.sensing_hook import SOURCE_ROTATION, TIER_ROUTING
from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
from mae_core.market.signal import (
    from_cot_positioning,
    from_stocktwits_sentiment,
    from_vix_structure,
    from_trends_signal,
    from_economic_event,
    from_analyst_recommendation,
)

# ---------------------------------------------------------------------------
# Constants — the 7 new TIER_ROUTING signal-source keys
# ---------------------------------------------------------------------------

NEW_TIER_KEYS = [
    "cot_positioning",
    "stocktwits_sentiment",
    "vix_term_structure",
    "google_trends",
    "finnhub_economic",
    "finnhub_analyst",
    "finnhub_earnings_calendar",
]

# The 5 new SOURCE_ROTATION names (sensing-hook level, not signal-source level)
NEW_ROTATION_SOURCES = [
    "cot_positioning",
    "stocktwits",
    "vix_structure",
    "google_trends",
    "finnhub_extras",
]

# Full expected SOURCE_ROTATION (all 27)
EXPECTED_ROTATION = [
    "sec_form4",
    "sec_form8k",
    "congressional",
    "senate",
    "hiring",
    "usa_spending",
    "sam_gov_and_prices",
    "social_sentiment",
    "finra_short",
    "sec_efts",
    "finnhub",
    "fred_macro",
    "session_sweep",
    "ta_indicators",
    # Layer 6
    "cot_positioning",
    "stocktwits",
    "vix_structure",
    "google_trends",
    "finnhub_extras",
    # Ten Gifts: Wave 1
    "order_flow",
    # Ten Gifts: Wave 2
    "fractal_resonance",
    # Always-On Waves 2+3
    "crypto_prices",
    "crypto_exchange",
    "openinsider",
    "institutional_13f",
    "finviz",
    "economic_calendar",
]

VALID_TIERS = {"tactical", "strategic", "thematic"}


# ---------------------------------------------------------------------------
# Helper factories — build minimal domain objects that converters accept
# ---------------------------------------------------------------------------

def _make_cot(
    ticker="ES=F",
    commercial_net=50000,
    pct_commercial_long=0.60,
    noncommercial_net=-30000,
    small_trader_net=-20000,
    open_interest=300000,
    report_date="2026-02-21",
    contract_name="E-Mini S&P 500",
    confidence=0.55,
    decay_rate=0.03,
) -> Any:
    return SimpleNamespace(
        ticker=ticker,
        commercial_net=commercial_net,
        pct_commercial_long=pct_commercial_long,
        noncommercial_net=noncommercial_net,
        small_trader_net=small_trader_net,
        open_interest=open_interest,
        report_date=report_date,
        contract_name=contract_name,
        confidence=confidence,
        decay_rate=decay_rate,
    )


def _make_stocktwits(
    ticker="AAPL",
    bull_count=700,
    bear_count=300,
    bull_ratio=0.70,
    total_messages=1000,
    trending=True,
    detected_at=None,
    confidence=0.50,
    decay_rate=0.3,
) -> Any:
    if detected_at is None:
        detected_at = datetime.now()
    return SimpleNamespace(
        ticker=ticker,
        bull_count=bull_count,
        bear_count=bear_count,
        bull_ratio=bull_ratio,
        total_messages=total_messages,
        trending=trending,
        detected_at=detected_at,
        confidence=confidence,
        decay_rate=decay_rate,
    )


def _make_vix(
    structure_type="contango",
    vix_spot=18.0,
    vix_1m=19.0,
    vix_3m=20.5,
    term_spread=2.5,
    date_val=None,
    confidence=0.60,
    decay_rate=0.30,
) -> Any:
    if date_val is None:
        date_val = date.today().isoformat()
    return SimpleNamespace(
        structure_type=structure_type,
        vix_spot=vix_spot,
        vix_1m=vix_1m,
        vix_3m=vix_3m,
        term_spread=term_spread,
        date=date_val,
        confidence=confidence,
        decay_rate=decay_rate,
    )


def _make_trend(
    keyword="NVDA",
    interest_score=75,
    interest_delta_7d=25,
    is_breakout=True,
    related_queries=None,
    detected_at=None,
    confidence=0.45,
    decay_rate=0.3,
) -> Any:
    if related_queries is None:
        related_queries = ["nvidia gpu", "nvidia stock"]
    if detected_at is None:
        detected_at = datetime.now()
    return SimpleNamespace(
        keyword=keyword,
        interest_score=interest_score,
        interest_delta_7d=interest_delta_7d,
        is_breakout=is_breakout,
        related_queries=related_queries,
        detected_at=detected_at,
        confidence=confidence,
        decay_rate=decay_rate,
    )


def _make_economic_event(
    event="CPI",
    country="US",
    impact="high",
    actual=3.2,
    estimate=3.0,
    previous=3.1,
    unit="%",
    date_val=None,
    confidence=0.55,
    decay_rate=0.5,
) -> Any:
    if date_val is None:
        date_val = date.today().isoformat()

    # surprise_pct is called by the converter — include it
    def _surprise_pct():
        if actual is not None and estimate is not None and estimate != 0:
            return (actual - estimate) / abs(estimate)
        return None

    return SimpleNamespace(
        event=event,
        country=country,
        impact=impact,
        actual=actual,
        estimate=estimate,
        previous=previous,
        unit=unit,
        date=date_val,
        confidence=confidence,
        decay_rate=decay_rate,
        surprise_pct=_surprise_pct,
    )


def _make_analyst_rec(
    symbol="MSFT",
    strong_buy=8,
    buy=12,
    hold=4,
    sell=1,
    strong_sell=0,
    buy_ratio=0.80,
    total=25,
    period=None,
    confidence=0.50,
    decay_rate=0.05,
) -> Any:
    if period is None:
        period = date.today().isoformat()
    return SimpleNamespace(
        symbol=symbol,
        strong_buy=strong_buy,
        buy=buy,
        hold=hold,
        sell=sell,
        strong_sell=strong_sell,
        buy_ratio=buy_ratio,
        total=total,
        period=period,
        confidence=confidence,
        decay_rate=decay_rate,
    )


# ===========================================================================
# Test classes
# ===========================================================================


class TestSourceRotation:
    """Verify SOURCE_ROTATION contains exactly the expected 21 entries."""

    def test_rotation_length_is_21(self):
        assert len(SOURCE_ROTATION) == 21, (
            f"Expected 21 sources in SOURCE_ROTATION, got {len(SOURCE_ROTATION)}"
        )

    def test_rotation_contains_all_expected_sources(self):
        missing = [s for s in EXPECTED_ROTATION if s not in SOURCE_ROTATION]
        assert not missing, f"Sources missing from SOURCE_ROTATION: {missing}"

    def test_rotation_has_no_unexpected_sources(self):
        extra = [s for s in SOURCE_ROTATION if s not in EXPECTED_ROTATION]
        assert not extra, f"Unexpected sources in SOURCE_ROTATION: {extra}"

    def test_all_new_rotation_sources_present(self):
        for src in NEW_ROTATION_SOURCES:
            assert src in SOURCE_ROTATION, (
                f"New Layer 6 source '{src}' missing from SOURCE_ROTATION"
            )

    def test_no_duplicate_rotation_entries(self):
        assert len(SOURCE_ROTATION) == len(set(SOURCE_ROTATION)), (
            "SOURCE_ROTATION contains duplicate entries"
        )


class TestTierRouting:
    """Verify TIER_ROUTING has valid entries for all 7 new signal-source keys."""

    def test_all_new_tier_keys_present(self):
        missing = [k for k in NEW_TIER_KEYS if k not in TIER_ROUTING]
        assert not missing, (
            f"New signal-source keys missing from TIER_ROUTING: {missing}"
        )

    def test_cot_positioning_is_strategic(self):
        assert TIER_ROUTING["cot_positioning"] == "strategic"

    def test_stocktwits_sentiment_is_thematic(self):
        assert TIER_ROUTING["stocktwits_sentiment"] == "thematic"

    def test_vix_term_structure_is_strategic(self):
        assert TIER_ROUTING["vix_term_structure"] == "strategic"

    def test_google_trends_is_thematic(self):
        assert TIER_ROUTING["google_trends"] == "thematic"

    def test_finnhub_economic_is_tactical(self):
        assert TIER_ROUTING["finnhub_economic"] == "tactical"

    def test_finnhub_analyst_is_strategic(self):
        assert TIER_ROUTING["finnhub_analyst"] == "strategic"

    def test_finnhub_earnings_calendar_is_tactical(self):
        assert TIER_ROUTING["finnhub_earnings_calendar"] == "tactical"

    def test_all_new_tier_values_are_valid(self):
        for key in NEW_TIER_KEYS:
            val = TIER_ROUTING.get(key)
            assert val in VALID_TIERS, (
                f"TIER_ROUTING['{key}'] = '{val}' — not a valid tier. "
                f"Must be one of {VALID_TIERS}"
            )


class TestThompsonKeyMap:
    """Verify _SOURCE_TO_THOMPSON_KEY has entries for all 7 new signal-source keys."""

    def test_all_new_sources_in_thompson_map(self):
        key_map = ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY
        missing = [k for k in NEW_TIER_KEYS if k not in key_map]
        assert not missing, (
            f"New signal-source keys missing from _SOURCE_TO_THOMPSON_KEY: {missing}"
        )

    def test_new_thompson_keys_are_self_mapping(self):
        """Each new source maps to its own name as the Thompson key."""
        key_map = ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY
        for src in NEW_TIER_KEYS:
            if src in key_map:
                assert key_map[src] == src, (
                    f"Expected _SOURCE_TO_THOMPSON_KEY['{src}'] == '{src}', "
                    f"got '{key_map[src]}'"
                )

    def test_no_orphan_thompson_keys(self):
        """Every Thompson key from new sources must exist in source_reliability."""
        key_map = ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY
        reliability = LEARNING_CONFIG["source_reliability"]
        for src in NEW_TIER_KEYS:
            thompson_key = key_map.get(src)
            if thompson_key is not None:
                assert thompson_key in reliability, (
                    f"Thompson key '{thompson_key}' (from source '{src}') "
                    f"has no entry in learning_config source_reliability"
                )


class TestLearningConfig:
    """Verify learning_config has correct entries for all new sources and domains."""

    def test_all_new_sources_in_source_reliability(self):
        reliability = LEARNING_CONFIG["source_reliability"]
        missing = [k for k in NEW_TIER_KEYS if k not in reliability]
        assert not missing, (
            f"New signal-source keys missing from source_reliability: {missing}"
        )

    def test_source_reliability_values_in_valid_range(self):
        reliability = LEARNING_CONFIG["source_reliability"]
        for key in NEW_TIER_KEYS:
            if key in reliability:
                val = reliability[key]
                assert 0.0 <= val <= 1.0, (
                    f"source_reliability['{key}'] = {val} — must be in [0.0, 1.0]"
                )

    def test_decay_rates_has_positioning(self):
        """COT data is weekly — positioning domain needs its own decay rate."""
        assert "positioning" in LEARNING_CONFIG["decay_rates"], (
            "decay_rates missing 'positioning' domain (needed for cot_positioning source)"
        )

    def test_decay_rates_has_volatility(self):
        """VIX data changes daily — volatility domain needs its own decay rate."""
        assert "volatility" in LEARNING_CONFIG["decay_rates"], (
            "decay_rates missing 'volatility' domain (needed for vix_term_structure source)"
        )

    def test_positioning_decay_rate_value(self):
        """COT is weekly data — decay should be slow (longer half-life than daily data)."""
        rate = LEARNING_CONFIG["decay_rates"]["positioning"]
        assert rate <= 0.10, (
            f"positioning decay_rate is {rate} — expected slow decay (<=0.10) "
            f"for weekly COT data"
        )

    def test_volatility_decay_rate_value(self):
        """VIX changes daily — decay should be faster than insider/contract signals."""
        rate = LEARNING_CONFIG["decay_rates"]["volatility"]
        assert 0.10 <= rate <= 0.50, (
            f"volatility decay_rate is {rate} — expected moderate decay "
            f"(0.10-0.50) for daily VIX data"
        )


class TestConverterSourceFields:
    """Verify that each new converter produces a MarketSignal with the correct
    .source field, so TIER_ROUTING and Thompson lookups will find the right entry.
    """

    def test_from_cot_positioning_sets_source(self):
        cot = _make_cot()
        sig = from_cot_positioning(cot)
        assert sig.source == "cot_positioning", (
            f"from_cot_positioning produced source='{sig.source}', "
            f"expected 'cot_positioning'"
        )

    def test_from_cot_positioning_source_has_tier(self):
        cot = _make_cot()
        sig = from_cot_positioning(cot)
        assert sig.source in TIER_ROUTING, (
            f"from_cot_positioning source='{sig.source}' not found in TIER_ROUTING"
        )

    def test_from_stocktwits_sentiment_sets_source(self):
        st = _make_stocktwits()
        sig = from_stocktwits_sentiment(st)
        assert sig.source == "stocktwits_sentiment", (
            f"from_stocktwits_sentiment produced source='{sig.source}', "
            f"expected 'stocktwits_sentiment'"
        )

    def test_from_stocktwits_sentiment_source_has_tier(self):
        st = _make_stocktwits()
        sig = from_stocktwits_sentiment(st)
        assert sig.source in TIER_ROUTING, (
            f"from_stocktwits_sentiment source='{sig.source}' not found in TIER_ROUTING"
        )

    def test_from_vix_structure_sets_source(self):
        vix = _make_vix()
        sig = from_vix_structure(vix)
        assert sig.source == "vix_term_structure", (
            f"from_vix_structure produced source='{sig.source}', "
            f"expected 'vix_term_structure'"
        )

    def test_from_vix_structure_source_has_tier(self):
        vix = _make_vix()
        sig = from_vix_structure(vix)
        assert sig.source in TIER_ROUTING, (
            f"from_vix_structure source='{sig.source}' not found in TIER_ROUTING"
        )

    def test_from_trends_signal_sets_source(self):
        trend = _make_trend()
        sig = from_trends_signal(trend)
        assert sig.source == "google_trends", (
            f"from_trends_signal produced source='{sig.source}', "
            f"expected 'google_trends'"
        )

    def test_from_trends_signal_source_has_tier(self):
        trend = _make_trend()
        sig = from_trends_signal(trend)
        assert sig.source in TIER_ROUTING, (
            f"from_trends_signal source='{sig.source}' not found in TIER_ROUTING"
        )

    def test_from_economic_event_sets_source(self):
        event = _make_economic_event()
        sig = from_economic_event(event)
        assert sig.source == "finnhub_economic", (
            f"from_economic_event produced source='{sig.source}', "
            f"expected 'finnhub_economic'"
        )

    def test_from_economic_event_source_has_tier(self):
        event = _make_economic_event()
        sig = from_economic_event(event)
        assert sig.source in TIER_ROUTING, (
            f"from_economic_event source='{sig.source}' not found in TIER_ROUTING"
        )

    def test_from_analyst_recommendation_sets_source(self):
        rec = _make_analyst_rec()
        sig = from_analyst_recommendation(rec)
        assert sig.source == "finnhub_analyst", (
            f"from_analyst_recommendation produced source='{sig.source}', "
            f"expected 'finnhub_analyst'"
        )

    def test_from_analyst_recommendation_source_has_tier(self):
        rec = _make_analyst_rec()
        sig = from_analyst_recommendation(rec)
        assert sig.source in TIER_ROUTING, (
            f"from_analyst_recommendation source='{sig.source}' not found in TIER_ROUTING"
        )


class TestConverterOutputShape:
    """Spot-check that converters produce signals with required field shapes.
    These catch obvious bugs in the converter logic without network I/O.
    """

    def test_cot_bullish_when_commercial_long(self):
        """Heavy commercial longs = bullish (hedgers protecting their longs)."""
        cot = _make_cot(commercial_net=100000, pct_commercial_long=0.65)
        sig = from_cot_positioning(cot)
        assert sig.direction == "bullish"

    def test_cot_bearish_when_commercial_short(self):
        cot = _make_cot(commercial_net=-100000, pct_commercial_long=0.35)
        sig = from_cot_positioning(cot)
        assert sig.direction == "bearish"

    def test_stocktwits_bullish_above_70_pct(self):
        st = _make_stocktwits(bull_ratio=0.75)
        sig = from_stocktwits_sentiment(st)
        assert sig.direction == "bullish"

    def test_stocktwits_bearish_below_30_pct(self):
        st = _make_stocktwits(bull_ratio=0.25)
        sig = from_stocktwits_sentiment(st)
        assert sig.direction == "bearish"

    def test_vix_bearish_in_backwardation(self):
        vix = _make_vix(structure_type="backwardation", vix_spot=22.0)
        sig = from_vix_structure(vix)
        assert sig.direction == "bearish"

    def test_vix_bearish_when_spot_above_30(self):
        vix = _make_vix(structure_type="contango", vix_spot=35.0)
        sig = from_vix_structure(vix)
        assert sig.direction == "bearish"

    def test_vix_bullish_in_contango_low_vix(self):
        vix = _make_vix(structure_type="contango", vix_spot=15.0)
        sig = from_vix_structure(vix)
        assert sig.direction == "bullish"

    def test_trends_fear_keyword_bearish(self):
        trend = _make_trend(keyword="recession", interest_score=75, interest_delta_7d=5)
        sig = from_trends_signal(trend)
        assert sig.direction == "bearish"

    def test_trends_ticker_rising_bullish(self):
        trend = _make_trend(keyword="NVDA", interest_score=80, interest_delta_7d=30)
        sig = from_trends_signal(trend)
        assert sig.direction == "bullish"

    def test_economic_event_bullish_positive_surprise(self):
        event = _make_economic_event(actual=3.5, estimate=3.0)
        sig = from_economic_event(event)
        assert sig.direction == "bullish"

    def test_economic_event_bearish_negative_surprise(self):
        event = _make_economic_event(actual=2.5, estimate=3.0)
        sig = from_economic_event(event)
        assert sig.direction == "bearish"

    def test_analyst_bullish_strong_consensus(self):
        rec = _make_analyst_rec(buy_ratio=0.85)
        sig = from_analyst_recommendation(rec)
        assert sig.direction == "bullish"

    def test_analyst_bearish_sell_consensus(self):
        rec = _make_analyst_rec(buy_ratio=0.20)
        sig = from_analyst_recommendation(rec)
        assert sig.direction == "bearish"


class TestConverterThompsonKeyCoverage:
    """Verify that every signal.source value produced by the new converters
    has a corresponding entry in _SOURCE_TO_THOMPSON_KEY. This is the wiring
    contract that ensures Thompson learning is applied when signals flow through
    the convergence alerter.
    """

    def _converter_sources(self) -> list[str]:
        """Return the .source values produced by each new Layer 6 converter."""
        return [
            from_cot_positioning(_make_cot()).source,
            from_stocktwits_sentiment(_make_stocktwits()).source,
            from_vix_structure(_make_vix()).source,
            from_trends_signal(_make_trend()).source,
            from_economic_event(_make_economic_event()).source,
            from_analyst_recommendation(_make_analyst_rec()).source,
        ]

    def test_all_converter_sources_in_thompson_map(self):
        key_map = ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY
        sources = self._converter_sources()
        missing = [s for s in sources if s not in key_map]
        assert not missing, (
            f"Converter output sources not in _SOURCE_TO_THOMPSON_KEY: {missing}. "
            f"These signals will silently get no Thompson weighting."
        )

    def test_all_converter_sources_in_source_reliability(self):
        reliability = LEARNING_CONFIG["source_reliability"]
        sources = self._converter_sources()
        missing = [s for s in sources if s not in reliability]
        assert not missing, (
            f"Converter output sources not in source_reliability: {missing}. "
            f"Thompson will have no prior for these sources."
        )

    def test_all_converter_sources_in_tier_routing(self):
        sources = self._converter_sources()
        missing = [s for s in sources if s not in TIER_ROUTING]
        assert not missing, (
            f"Converter output sources not in TIER_ROUTING: {missing}. "
            f"These signals will be silently routed to the default tier."
        )
