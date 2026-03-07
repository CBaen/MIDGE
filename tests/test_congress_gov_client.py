"""Tests for Congress.gov legislative client and signal adapter integration."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from mae_core.market.apis.congress_gov_client import (
    CongressGovClient,
    LegislativeIndicator,
    LEGISLATIVE_TICKER_MAP,
    ADVANCING_KEYWORDS,
    _determine_direction,
    _compute_strength,
    _classify_signal_type,
)
from mae_core.market.signal_adapters.market_data import from_legislative_indicator
from mae_core.market.sensing_fetchers import fetch_congress_legislation


# --- Direction logic ---

class TestDirectionLogic:
    def test_bullish_by_default(self):
        assert _determine_direction("Defense Appropriations Act", "Passed House") == "bullish"

    def test_bearish_on_restrict(self):
        assert _determine_direction("Restrict Trading Act", "Passed Senate") == "bearish"

    def test_bearish_on_ban(self):
        assert _determine_direction("Social Media Ban Act", "Signed") == "bearish"

    def test_bearish_on_repeal(self):
        assert _determine_direction("Repeal Clean Energy Credits", "") == "bearish"

    def test_bullish_normal_title(self):
        assert _determine_direction("Infrastructure Investment Act", "Enacted") == "bullish"


# --- Strength computation ---

class TestStrengthComputation:
    def test_enacted_full_strength(self):
        assert _compute_strength("Became Public Law 119-42") == 1.0

    def test_passed_high_strength(self):
        assert _compute_strength("Passed House by 250-180") == 0.8

    def test_reported_medium_strength(self):
        assert _compute_strength("Ordered to be reported by committee") == 0.5

    def test_cloture_strength(self):
        assert _compute_strength("Cloture invoked in Senate") == 0.6

    def test_default_strength_for_unknown(self):
        assert _compute_strength("Some other advancing action") == 0.3


# --- Signal type classification ---

class TestSignalTypeClassification:
    def test_enacted(self):
        assert _classify_signal_type("Became Public Law") == "bill_enacted"

    def test_signed(self):
        assert _classify_signal_type("Signed by President") == "bill_enacted"

    def test_passed(self):
        assert _classify_signal_type("Passed House") == "bill_passed"

    def test_agreed_to(self):
        assert _classify_signal_type("Agreed to in Senate") == "bill_passed"

    def test_advancing(self):
        assert _classify_signal_type("Reported by committee") == "bill_advancing"


# --- Ticker mapping ---

class TestTickerMapping:
    def test_defense_mapping(self):
        tickers = LEGISLATIVE_TICKER_MAP.get("Armed Forces and National Security")
        assert tickers is not None
        assert "ITA" in tickers
        assert "LMT" in tickers

    def test_health_mapping(self):
        tickers = LEGISLATIVE_TICKER_MAP.get("Health")
        assert tickers is not None
        assert "XLV" in tickers

    def test_energy_mapping(self):
        tickers = LEGISLATIVE_TICKER_MAP.get("Energy")
        assert tickers is not None
        assert "XLE" in tickers

    def test_unmapped_area_returns_empty(self):
        assert LEGISLATIVE_TICKER_MAP.get("Underwater Basket Weaving") is None


# --- LegislativeIndicator dataclass ---

class TestLegislativeIndicator:
    def _make_indicator(self, **overrides):
        defaults = dict(
            bill_id="hr-1234-119",
            bill_number="HR 1234",
            title="Defense Spending Act",
            congress=119,
            policy_area="Armed Forces and National Security",
            action_text="Passed House",
            action_date="2026-03-01",
            signal_type="bill_passed",
            direction="bullish",
            strength=0.8,
            affected_tickers=["ITA", "LMT", "RTX"],
        )
        defaults.update(overrides)
        return LegislativeIndicator(**defaults)

    def test_default_decay_rate(self):
        ind = self._make_indicator()
        assert ind.decay_rate == 0.03

    def test_default_confidence(self):
        ind = self._make_indicator()
        assert ind.confidence == 0.65

    def test_custom_strength(self):
        ind = self._make_indicator(strength=1.0)
        assert ind.strength == 1.0

    def test_affected_tickers(self):
        ind = self._make_indicator()
        assert len(ind.affected_tickers) == 3
        assert "ITA" in ind.affected_tickers


# --- Signal adapter ---

class TestSignalAdapter:
    def _make_indicator(self, **overrides):
        defaults = dict(
            bill_id="hr-5678-119",
            bill_number="HR 5678",
            title="Tech Innovation Act",
            congress=119,
            policy_area="Science, Technology, Communications",
            action_text="Signed by President",
            action_date="2026-03-05",
            signal_type="bill_enacted",
            direction="bullish",
            strength=1.0,
            affected_tickers=["XLK", "QQQ", "AAPL"],
        )
        defaults.update(overrides)
        return LegislativeIndicator(**defaults)

    def test_adapter_returns_market_signal(self):
        ind = self._make_indicator()
        sig = from_legislative_indicator(ind)
        assert sig.source == "congress_legislation"
        assert sig.domain == "government"

    def test_adapter_signal_id_format(self):
        ind = self._make_indicator()
        sig = from_legislative_indicator(ind)
        assert sig.signal_id.startswith("congress_legislation:hr-5678-119:")

    def test_adapter_carries_direction(self):
        ind = self._make_indicator(direction="bearish")
        sig = from_legislative_indicator(ind)
        assert sig.direction == "bearish"

    def test_adapter_uses_first_ticker(self):
        ind = self._make_indicator(affected_tickers=["XLK", "QQQ"])
        sig = from_legislative_indicator(ind)
        assert sig.symbol == "XLK"
        assert sig.outcome_symbol == "XLK"

    def test_adapter_empty_tickers(self):
        ind = self._make_indicator(affected_tickers=[])
        sig = from_legislative_indicator(ind)
        assert sig.symbol == ""

    def test_adapter_metadata(self):
        ind = self._make_indicator()
        sig = from_legislative_indicator(ind)
        assert sig.metadata["bill_id"] == "hr-5678-119"
        assert sig.metadata["signal_type"] == "bill_enacted"
        assert sig.metadata["policy_area"] == "Science, Technology, Communications"

    def test_adapter_asset_class(self):
        ind = self._make_indicator()
        sig = from_legislative_indicator(ind)
        assert sig.asset_class == "equity"

    def test_adapter_raw_type(self):
        ind = self._make_indicator()
        sig = from_legislative_indicator(ind)
        assert sig.raw_type == "LegislativeIndicator"


# --- Sensing fetcher ---

class TestSensingFetcher:
    def test_fetch_returns_empty_if_no_client(self):
        assert fetch_congress_legislation(None, lambda x: x) == []

    def test_fetch_converts_indicators(self):
        mock_client = MagicMock()
        ind = LegislativeIndicator(
            bill_id="s-100-119", bill_number="S 100",
            title="Energy Act", congress=119,
            policy_area="Energy", action_text="Passed Senate",
            action_date="2026-03-01", signal_type="bill_passed",
            direction="bullish", strength=0.8,
            affected_tickers=["XLE"],
        )
        mock_client.get_legislative_snapshot.return_value = [ind]
        results = fetch_congress_legislation(mock_client, from_legislative_indicator)
        assert len(results) == 1
        assert results[0].source == "congress_legislation"

    def test_fetch_handles_exception(self):
        mock_client = MagicMock()
        mock_client.get_legislative_snapshot.side_effect = Exception("API error")
        results = fetch_congress_legislation(mock_client, from_legislative_indicator)
        assert results == []


# --- Intelligence layer wiring ---

class TestIntelligenceWiring:
    def test_congress_legislation_in_source_to_thompson_key(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        assert ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY.get("congress_legislation") == "congress_legislation"

    def test_congress_legislation_in_domain_sources(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        assert "congress_legislation" in ConvergenceAlerter._DOMAIN_SOURCES["government"]

    def test_congress_legislation_in_source_reliability(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "congress_legislation" in LEARNING_CONFIG["source_reliability"]
        val = LEARNING_CONFIG["source_reliability"]["congress_legislation"]
        assert 0.0 < val <= 1.0

    def test_government_decay_rate_exists(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "government" in LEARNING_CONFIG["decay_rates"]

    def test_congress_legislation_in_source_domain_map(self):
        from mae_core.market.archaeology.pattern_library import PatternLibrary
        assert PatternLibrary._SOURCE_DOMAIN_MAP.get("congress_legislation") == "government"

    def test_congress_legislation_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "congress_legislation" in SOURCE_ROTATION

    def test_congress_legislation_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert _ROTATION_TO_THOMPSON.get("congress_legislation") == "congress_legislation"

    def test_congress_legislation_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "congress_legislation" in TIER_ROUTING
        assert TIER_ROUTING["congress_legislation"] == "strategic"

    def test_congress_legislation_in_absence_domains(self):
        from mae_core.market.sensing_hook import _ABSENCE_SOURCE_DOMAINS
        assert _ABSENCE_SOURCE_DOMAINS.get("congress_legislation") == "government"

    def test_congress_legislation_in_plain_language(self):
        from mae_core.market.plain_language import SOURCE_PLAIN
        assert "congress_legislation" in SOURCE_PLAIN


# --- Client unit tests ---

class TestCongressGovClient:
    def test_client_initializes_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            client = CongressGovClient(api_key=None)
            assert client.api_key is None

    def test_client_initializes_with_key(self):
        client = CongressGovClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"

    def test_bill_id_from_raw(self):
        raw = {"type": "HR", "number": 1234, "congress": 119}
        assert CongressGovClient._bill_id_from_raw(raw) == "hr-1234-119"

    def test_bill_id_missing_fields(self):
        raw = {"type": None, "number": None, "congress": None}
        result = CongressGovClient._bill_id_from_raw(raw)
        assert "unknown" in result

    def test_request_fails_without_key(self):
        client = CongressGovClient(api_key=None)
        result = client._request("/bill", {})
        assert result is None

    def test_get_recent_bills_uses_cache(self):
        client = CongressGovClient(api_key="test")
        import time
        client._cache["bill_list_7_50"] = ([{"mock": True}], time.time())
        result = client.get_recent_bills()
        assert result == [{"mock": True}]

    def test_advancing_keyword_filter(self):
        for kw in ADVANCING_KEYWORDS:
            assert kw.islower(), f"Advancing keyword '{kw}' should be lowercase"
