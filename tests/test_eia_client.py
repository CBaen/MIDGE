"""Tests for EIA energy client and signal adapter integration."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from mae_core.market.apis.eia_client import (
    EIAClient,
    EnergyIndicator,
    EIA_SERIES,
    ENERGY_TICKER_MAP,
    _determine_direction,
    _compute_strength,
)
from mae_core.market.signal_adapters.market_data import from_energy_indicator
from mae_core.market.sensing_fetchers import fetch_eia


# --- Direction logic ---

class TestDirectionLogic:
    def test_crude_build_is_bearish(self):
        assert _determine_direction("crude_inventory", 3000) == "bearish"

    def test_crude_draw_is_bullish(self):
        assert _determine_direction("crude_inventory", -4000) == "bullish"

    def test_gasoline_build_is_bearish(self):
        assert _determine_direction("gasoline_inventory", 1500) == "bearish"

    def test_natgas_injection_is_bearish(self):
        assert _determine_direction("natgas_storage", 50) == "bearish"

    def test_natgas_withdrawal_is_bullish(self):
        assert _determine_direction("natgas_storage", -70) == "bullish"

    def test_production_increase_is_bearish(self):
        assert _determine_direction("crude_production", 50) == "bearish"

    def test_production_decrease_is_bullish(self):
        assert _determine_direction("crude_production", -30) == "bullish"

    def test_zero_change_is_neutral(self):
        assert _determine_direction("crude_inventory", 0) == "neutral"


# --- Strength calibration ---

class TestStrengthCalibration:
    def test_typical_change_gives_moderate_strength(self):
        # 5000 barrel change is typical for crude → ~1.0
        s = _compute_strength("crude_inventory", 5000)
        assert 0.8 <= s <= 1.0

    def test_small_change_gives_low_strength(self):
        s = _compute_strength("crude_inventory", 500)
        assert s >= 0.1
        assert s < 0.3

    def test_large_change_clamps_to_1(self):
        s = _compute_strength("crude_inventory", 20000)
        assert s == 1.0

    def test_natgas_typical_change(self):
        # 80 Bcf is typical for natgas → ~1.0
        s = _compute_strength("natgas_storage", 80)
        assert 0.8 <= s <= 1.0


# --- EnergyIndicator dataclass ---

class TestEnergyIndicator:
    def test_fields(self):
        ind = EnergyIndicator(
            series_key="crude_stocks",
            series_name="US Crude Oil Stocks",
            value=440000,
            prior_value=443000,
            change=-3000,
            change_pct=-0.68,
            date="2026-03-01",
            signal_type="crude_inventory",
            direction="bullish",
            strength=0.6,
            affected_tickers=["XLE", "XOP"],
        )
        assert ind.direction == "bullish"
        assert ind.signal_source == "eia_energy"
        assert ind.confidence == 0.75
        assert ind.decay_rate == 0.05


# --- Signal adapter ---

class TestFromEnergyIndicator:
    def _make_indicator(self, **kwargs):
        defaults = dict(
            series_key="crude_stocks",
            series_name="US Crude Oil Stocks (Weekly)",
            value=440000,
            prior_value=443000,
            change=-3000,
            change_pct=-0.68,
            date="2026-03-01",
            signal_type="crude_inventory",
            direction="bullish",
            strength=0.6,
            affected_tickers=["XLE", "XOP", "OIH"],
            unit="thousand_barrels",
        )
        defaults.update(kwargs)
        return EnergyIndicator(**defaults)

    def test_converts_to_market_signal(self):
        ind = self._make_indicator()
        sig = from_energy_indicator(ind)
        assert sig.source == "eia_energy"
        assert sig.domain == "energy"
        assert sig.direction == "bullish"
        assert sig.symbol == "XLE"  # First affected ticker
        assert sig.asset_class == "commodity"
        assert sig.confidence == 0.75

    def test_signal_id_format(self):
        ind = self._make_indicator()
        sig = from_energy_indicator(ind)
        assert sig.signal_id == "eia_energy:crude_inventory:2026-03-01"

    def test_metadata_carries_energy_data(self):
        ind = self._make_indicator()
        sig = from_energy_indicator(ind)
        assert sig.metadata["series_key"] == "crude_stocks"
        assert sig.metadata["change"] == -3000
        assert sig.metadata["affected_tickers"] == ["XLE", "XOP", "OIH"]

    def test_empty_tickers_uses_empty_symbol(self):
        ind = self._make_indicator(affected_tickers=[])
        sig = from_energy_indicator(ind)
        assert sig.symbol == ""

    def test_natgas_signal(self):
        ind = self._make_indicator(
            series_key="natgas_storage",
            signal_type="natgas_storage",
            direction="bearish",
            affected_tickers=["UNG", "EQT"],
        )
        sig = from_energy_indicator(ind)
        assert sig.symbol == "UNG"
        assert sig.direction == "bearish"


# --- Fetch function ---

class TestFetchEia:
    def test_returns_empty_when_no_client(self):
        assert fetch_eia(None, lambda x: x) == []

    def test_fetches_and_converts(self):
        mock_client = MagicMock()
        ind = EnergyIndicator(
            series_key="crude_stocks",
            series_name="Test",
            value=100, prior_value=105, change=-5, change_pct=-4.8,
            date="2026-03-01", signal_type="crude_inventory",
            direction="bullish", strength=0.5,
            affected_tickers=["XLE"],
        )
        mock_client.get_energy_snapshot.return_value = [ind]

        converter = MagicMock(return_value="converted")
        result = fetch_eia(mock_client, converter)

        assert result == ["converted"]
        converter.assert_called_once_with(ind)

    def test_handles_fetch_exception(self):
        mock_client = MagicMock()
        mock_client.get_energy_snapshot.side_effect = Exception("timeout")
        result = fetch_eia(mock_client, lambda x: x)
        assert result == []

    def test_handles_converter_exception(self):
        mock_client = MagicMock()
        ind = EnergyIndicator(
            series_key="crude_stocks", series_name="Test",
            value=100, prior_value=105, change=-5, change_pct=-4.8,
            date="2026-03-01", signal_type="crude_inventory",
            direction="bullish", strength=0.5, affected_tickers=["XLE"],
        )
        mock_client.get_energy_snapshot.return_value = [ind]

        def bad_converter(x):
            raise ValueError("conversion failed")

        result = fetch_eia(mock_client, bad_converter)
        assert result == []


# --- Series definitions ---

class TestSeriesDefinitions:
    def test_all_series_have_required_fields(self):
        for key, series in EIA_SERIES.items():
            assert "route" in series, f"{key} missing route"
            assert "name" in series, f"{key} missing name"
            assert "signal_type" in series, f"{key} missing signal_type"
            assert "frequency" in series, f"{key} missing frequency"

    def test_all_signal_types_have_ticker_map(self):
        for key, series in EIA_SERIES.items():
            assert series["signal_type"] in ENERGY_TICKER_MAP, (
                f"{key} signal_type {series['signal_type']} not in ENERGY_TICKER_MAP"
            )


# --- Client (unit tests with mocked HTTP) ---

class TestEIAClientUnit:
    def test_no_key_returns_none(self):
        with patch.dict("os.environ", {}, clear=True):
            client = EIAClient(api_key=None)
            result = client.get_series("crude_stocks")
            assert result is None

    def test_caching(self):
        client = EIAClient(api_key="test_key")
        ind = EnergyIndicator(
            series_key="crude_stocks", series_name="Test",
            value=100, prior_value=105, change=-5, change_pct=-4.8,
            date="2026-03-01", signal_type="crude_inventory",
            direction="bullish", strength=0.5, affected_tickers=["XLE"],
        )
        import time
        client._cache["crude_stocks"] = (ind, time.time())

        result = client.get_series("crude_stocks")
        assert result is ind  # Returns cached, no HTTP

    def test_unknown_series_returns_none(self):
        client = EIAClient(api_key="test_key")
        result = client.get_series("nonexistent_series")
        assert result is None


# --- Integration: sensing hook wiring ---

class TestSensingHookWiring:
    def test_eia_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "eia_energy" in SOURCE_ROTATION

    def test_eia_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "eia_energy" in TIER_ROUTING
        assert TIER_ROUTING["eia_energy"] == "strategic"

    def test_eia_in_thompson_map(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "eia_energy" in _ROTATION_TO_THOMPSON

    def test_eia_in_absence_domains(self):
        from mae_core.market.sensing_hook import _ABSENCE_SOURCE_DOMAINS
        assert "eia_energy" in _ABSENCE_SOURCE_DOMAINS
        assert _ABSENCE_SOURCE_DOMAINS["eia_energy"] == "energy"

    def test_eia_in_source_domain_map(self):
        from mae_core.market.archaeology.pattern_library import PatternLibrary
        assert PatternLibrary._SOURCE_DOMAIN_MAP.get("eia_energy") == "energy"

    def test_energy_domain_in_convergence_alerter(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        alerter = ConvergenceAlerter(min_domains=2)
        assert "energy" in alerter.domain_categories
        assert "energy" in alerter._domain_windows

    def test_energy_in_plain_language(self):
        from mae_core.market.plain_language import DOMAIN_PLAIN, SOURCE_PLAIN
        assert "energy" in DOMAIN_PLAIN
        assert "eia_energy" in SOURCE_PLAIN

    def test_eia_in_source_to_thompson_key(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        assert ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY.get("eia_energy") == "eia_energy"

    def test_eia_in_domain_sources(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        assert "energy" in ConvergenceAlerter._DOMAIN_SOURCES
        assert "eia_energy" in ConvergenceAlerter._DOMAIN_SOURCES["energy"]

    def test_eia_in_source_reliability(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "eia_energy" in LEARNING_CONFIG["source_reliability"]
        val = LEARNING_CONFIG["source_reliability"]["eia_energy"]
        assert 0.0 < val <= 1.0

    def test_energy_decay_rate(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "energy" in LEARNING_CONFIG["decay_rates"]
