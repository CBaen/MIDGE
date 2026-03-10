"""Tests for USDA agricultural client and signal adapter integration."""

import pytest
from unittest.mock import MagicMock

from mae_core.market.apis.usda_client import (
    USDAClient,
    AgriculturalIndicator,
    _compute_direction_strength,
)
from mae_core.market.signal_adapters.market_data import from_usda_indicator
from mae_core.market.sensing_fetchers import fetch_usda


# --- Direction and strength logic ---

class TestDirectionStrength:
    def test_very_tight_supply_is_bullish_strong(self):
        direction, strength = _compute_direction_strength(0.10)
        assert direction == "bullish"
        assert strength >= 0.5

    def test_mild_tightness_is_bullish_mild(self):
        direction, strength = _compute_direction_strength(0.20)
        assert direction == "bullish"
        assert 0.2 <= strength <= 0.5

    def test_neutral_zone(self):
        direction, strength = _compute_direction_strength(0.27)
        assert direction == "neutral"

    def test_mild_surplus_is_bearish_mild(self):
        direction, strength = _compute_direction_strength(0.35)
        assert direction == "bearish"
        assert 0.2 <= strength <= 0.4

    def test_loose_supply_is_bearish_strong(self):
        direction, strength = _compute_direction_strength(0.50)
        assert direction == "bearish"
        assert strength >= 0.3

    def test_zero_stocks_to_use_is_bullish(self):
        direction, strength = _compute_direction_strength(0.0)
        assert direction == "bullish"
        assert strength >= 0.5

    def test_very_high_stocks_to_use_clamps_strength(self):
        direction, strength = _compute_direction_strength(1.0)
        assert direction == "bearish"
        assert strength <= 1.0

    def test_exact_lower_bullish_boundary(self):
        # 0.15 is the boundary between "very tight" and "mild tightness"
        direction, _ = _compute_direction_strength(0.15)
        assert direction == "bullish"

    def test_exact_upper_bearish_boundary(self):
        # 0.40 is the boundary between neutral and mild surplus
        direction, _ = _compute_direction_strength(0.40)
        assert direction == "neutral"

    def test_just_above_upper_bearish_boundary(self):
        direction, _ = _compute_direction_strength(0.401)
        assert direction == "bearish"

    def test_strength_is_float_in_range(self):
        for stu in [0.05, 0.18, 0.28, 0.33, 0.55]:
            _, strength = _compute_direction_strength(stu)
            assert 0.0 <= strength <= 1.0, f"Strength out of range for stu={stu}"


# --- AgriculturalIndicator dataclass ---

class TestAgriculturalIndicatorDataclass:
    def test_default_signal_source(self):
        ind = AgriculturalIndicator(
            commodity_key="wheat",
            commodity_name="Wheat",
            market_year="2025/2026",
            production=780.0,
            consumption=800.0,
            ending_stocks=200.0,
            supply_surplus_ratio=0.25,
            direction="neutral",
            strength=0.2,
            affected_tickers=["WEAT", "DBA"],
            futures_ticker="ZW",
        )
        assert ind.signal_source == "usda_agriculture"

    def test_default_decay_rate(self):
        ind = AgriculturalIndicator(
            commodity_key="corn",
            commodity_name="Corn",
            market_year="2025/2026",
            production=1200.0,
            consumption=1150.0,
            ending_stocks=280.0,
            supply_surplus_ratio=0.243,
            direction="bullish",
            strength=0.3,
            affected_tickers=["CORN", "DBA"],
            futures_ticker="ZC",
        )
        assert ind.decay_rate == 0.03

    def test_default_confidence(self):
        ind = AgriculturalIndicator(
            commodity_key="soybeans",
            commodity_name="Soybeans",
            market_year="2025/2026",
            production=395.0,
            consumption=375.0,
            ending_stocks=100.0,
            supply_surplus_ratio=0.267,
            direction="neutral",
            strength=0.2,
            affected_tickers=["SOYB"],
            futures_ticker="ZS",
        )
        assert ind.confidence == 0.65


# --- Signal adapter ---

class TestFromUsdaIndicator:
    def _make_indicator(self, **kwargs):
        defaults = dict(
            commodity_key="wheat",
            commodity_name="Wheat",
            market_year="2025/2026",
            production=780.0,
            consumption=800.0,
            ending_stocks=160.0,
            supply_surplus_ratio=0.20,
            direction="bullish",
            strength=0.45,
            affected_tickers=["WEAT", "DBA", "ADM", "BG"],
            futures_ticker="ZW",
        )
        defaults.update(kwargs)
        return AgriculturalIndicator(**defaults)

    def test_converts_to_market_signal(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.source == "usda_agriculture"
        assert sig.domain == "macro"
        assert sig.asset_class == "commodity"
        assert sig.direction == "bullish"

    def test_symbol_is_futures_ticker(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.symbol == "ZW"

    def test_symbol_falls_back_to_first_affected_ticker(self):
        ind = self._make_indicator(futures_ticker="")
        sig = from_usda_indicator(ind)
        assert sig.symbol == "WEAT"

    def test_symbol_is_empty_when_no_tickers(self):
        ind = self._make_indicator(futures_ticker="", affected_tickers=[])
        sig = from_usda_indicator(ind)
        assert sig.symbol == ""

    def test_signal_id_format(self):
        ind = self._make_indicator(commodity_key="wheat", market_year="2025/2026")
        sig = from_usda_indicator(ind)
        assert sig.signal_id == "usda_agriculture:wheat:2025/2026"

    def test_signal_id_uses_commodity_key_and_year(self):
        ind = self._make_indicator(commodity_key="corn", market_year="2024/2025")
        sig = from_usda_indicator(ind)
        assert "corn" in sig.signal_id
        assert "2024/2025" in sig.signal_id

    def test_confidence_and_decay_rate_propagated(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.confidence == 0.65
        assert sig.decay_rate == 0.03

    def test_metadata_commodity_key(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["commodity_key"] == "wheat"

    def test_metadata_commodity_name(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["commodity_name"] == "Wheat"

    def test_metadata_market_year(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["market_year"] == "2025/2026"

    def test_metadata_supply_surplus_ratio(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["supply_surplus_ratio"] == 0.20

    def test_metadata_futures_ticker(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["futures_ticker"] == "ZW"

    def test_metadata_affected_tickers(self):
        ind = self._make_indicator()
        sig = from_usda_indicator(ind)
        assert sig.metadata["affected_tickers"] == ["WEAT", "DBA", "ADM", "BG"]

    def test_metadata_production_and_consumption(self):
        ind = self._make_indicator(production=780.0, consumption=800.0)
        sig = from_usda_indicator(ind)
        assert sig.metadata["production"] == 780.0
        assert sig.metadata["consumption"] == 800.0

    def test_metadata_ending_stocks(self):
        ind = self._make_indicator(ending_stocks=160.0)
        sig = from_usda_indicator(ind)
        assert sig.metadata["ending_stocks"] == 160.0

    def test_bearish_signal(self):
        ind = self._make_indicator(direction="bearish", supply_surplus_ratio=0.50)
        sig = from_usda_indicator(ind)
        assert sig.direction == "bearish"

    def test_corn_uses_zc_futures(self):
        ind = self._make_indicator(
            commodity_key="corn",
            commodity_name="Corn",
            futures_ticker="ZC",
            affected_tickers=["CORN", "DBA"],
        )
        sig = from_usda_indicator(ind)
        assert sig.symbol == "ZC"
        assert sig.metadata["commodity_key"] == "corn"


# --- Fetch function ---

class TestFetchUsda:
    def test_returns_empty_when_no_client(self):
        assert fetch_usda(None, lambda x: x) == []

    def test_fetches_and_converts(self):
        mock_client = MagicMock()
        ind = AgriculturalIndicator(
            commodity_key="wheat",
            commodity_name="Wheat",
            market_year="2025/2026",
            production=780.0,
            consumption=800.0,
            ending_stocks=160.0,
            supply_surplus_ratio=0.20,
            direction="bullish",
            strength=0.45,
            affected_tickers=["WEAT", "DBA"],
            futures_ticker="ZW",
        )
        mock_client.get_agricultural_snapshot.return_value = [ind]

        converter = MagicMock(return_value="converted")
        result = fetch_usda(mock_client, converter)

        assert result == ["converted"]
        converter.assert_called_once_with(ind)

    def test_fetches_multiple_indicators(self):
        mock_client = MagicMock()

        def make_ind(key, ticker):
            return AgriculturalIndicator(
                commodity_key=key, commodity_name=key.title(),
                market_year="2025/2026", production=500.0,
                consumption=480.0, ending_stocks=120.0,
                supply_surplus_ratio=0.25, direction="neutral",
                strength=0.2, affected_tickers=["DBA"],
                futures_ticker=ticker,
            )

        mock_client.get_agricultural_snapshot.return_value = [
            make_ind("wheat", "ZW"),
            make_ind("corn", "ZC"),
            make_ind("soybeans", "ZS"),
        ]

        converter = MagicMock(side_effect=lambda x: f"sig:{x.commodity_key}")
        result = fetch_usda(mock_client, converter)
        assert len(result) == 3
        assert "sig:wheat" in result
        assert "sig:corn" in result
        assert "sig:soybeans" in result

    def test_handles_fetch_exception(self):
        mock_client = MagicMock()
        mock_client.get_agricultural_snapshot.side_effect = Exception("network error")
        result = fetch_usda(mock_client, lambda x: x)
        assert result == []

    def test_handles_converter_exception(self):
        mock_client = MagicMock()
        ind = AgriculturalIndicator(
            commodity_key="wheat", commodity_name="Wheat",
            market_year="2025/2026", production=780.0,
            consumption=800.0, ending_stocks=160.0,
            supply_surplus_ratio=0.20, direction="bullish",
            strength=0.45, affected_tickers=["WEAT"],
            futures_ticker="ZW",
        )
        mock_client.get_agricultural_snapshot.return_value = [ind]

        def bad_converter(x):
            raise ValueError("conversion failed")

        result = fetch_usda(mock_client, bad_converter)
        assert result == []

    def test_skips_failed_conversions_continues_rest(self):
        """One failing converter should not suppress successful conversions."""
        mock_client = MagicMock()

        def make_ind(key):
            return AgriculturalIndicator(
                commodity_key=key, commodity_name=key.title(),
                market_year="2025/2026", production=500.0,
                consumption=480.0, ending_stocks=120.0,
                supply_surplus_ratio=0.25, direction="neutral",
                strength=0.2, affected_tickers=["DBA"],
                futures_ticker="ZW",
            )

        mock_client.get_agricultural_snapshot.return_value = [
            make_ind("wheat"),
            make_ind("corn"),
        ]

        call_count = [0]

        def sometimes_bad(x):
            call_count[0] += 1
            if x.commodity_key == "wheat":
                raise ValueError("wheat conversion failed")
            return f"sig:{x.commodity_key}"

        result = fetch_usda(mock_client, sometimes_bad)
        assert result == ["sig:corn"]
