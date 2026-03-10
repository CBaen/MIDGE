"""Tests for FRED client, yield additions, and FRED signal adapters."""

import pytest
from unittest.mock import MagicMock

from mae_core.market.apis.fred_client import FREDClient, MacroIndicator, _determine_direction
from mae_core.market.signal_adapters.market_data import from_macro_indicator, from_fred_yield
from mae_core.market.sensing_fetchers import fetch_fred, fetch_fred_yields


# --- Direction logic ---

class TestDirectionLogic:
    # T10Y2Y (yield curve)
    def test_t10y2y_negative_is_bearish(self):
        assert _determine_direction("T10Y2Y", -0.5) == "bearish"

    def test_t10y2y_positive_above_threshold_is_bullish(self):
        assert _determine_direction("T10Y2Y", 0.8) == "bullish"

    def test_t10y2y_between_zero_and_threshold_is_neutral(self):
        assert _determine_direction("T10Y2Y", 0.3) == "neutral"

    # BAMLH0A0HYM2 (credit spread)
    def test_credit_spread_high_is_bearish(self):
        assert _determine_direction("BAMLH0A0HYM2", 6.0) == "bearish"

    def test_credit_spread_low_is_bullish(self):
        assert _determine_direction("BAMLH0A0HYM2", 2.5) == "bullish"

    def test_credit_spread_mid_is_neutral(self):
        assert _determine_direction("BAMLH0A0HYM2", 4.0) == "neutral"

    # VIXCLS (volatility)
    def test_vix_high_is_bearish(self):
        assert _determine_direction("VIXCLS", 35) == "bearish"

    def test_vix_low_is_bullish(self):
        assert _determine_direction("VIXCLS", 12) == "bullish"

    def test_vix_mid_is_neutral(self):
        assert _determine_direction("VIXCLS", 20) == "neutral"

    # DFF (federal funds rate) — always neutral per regime-context design
    def test_dff_always_neutral(self):
        assert _determine_direction("DFF", 0.0) == "neutral"
        assert _determine_direction("DFF", 5.5) == "neutral"

    # UNRATE (unemployment)
    def test_unrate_high_is_bearish(self):
        assert _determine_direction("UNRATE", 6.0) == "bearish"

    def test_unrate_low_is_bullish(self):
        assert _determine_direction("UNRATE", 3.5) == "bullish"

    def test_unrate_mid_is_neutral(self):
        assert _determine_direction("UNRATE", 4.5) == "neutral"

    # CPIAUCSL (inflation) — raw level is neutral per design
    def test_cpiaucsl_always_neutral(self):
        assert _determine_direction("CPIAUCSL", 300.0) == "neutral"
        assert _determine_direction("CPIAUCSL", 100.0) == "neutral"

    # Unknown series
    def test_unknown_series_returns_neutral(self):
        assert _determine_direction("UNKNOWN_SERIES_XYZ", 99.9) == "neutral"
        assert _determine_direction("", 0.0) == "neutral"


# --- MacroIndicator dataclass defaults ---

class TestMacroIndicatorDataclass:
    def test_signal_source_default(self):
        ind = MacroIndicator(
            series_id="T10Y2Y",
            series_name="10Y-2Y Spread",
            value=1.5,
            date="2026-03-09",
            signal_type="yield_curve",
            direction="bullish",
        )
        assert ind.signal_source == "fred_macro"

    def test_decay_rate_default(self):
        ind = MacroIndicator(
            series_id="VIXCLS",
            series_name="VIX",
            value=20.0,
            date="2026-03-09",
            signal_type="volatility",
            direction="neutral",
        )
        assert ind.decay_rate == 0.01

    def test_confidence_default(self):
        ind = MacroIndicator(
            series_id="DFF",
            series_name="Fed Funds Rate",
            value=5.25,
            date="2026-03-09",
            signal_type="rates",
            direction="neutral",
        )
        assert ind.confidence == 0.70


# --- from_macro_indicator signal adapter ---

class TestFromMacroIndicator:
    def _make_indicator(self, **kwargs):
        defaults = dict(
            series_id="T10Y2Y",
            series_name="10Y-2Y Treasury Spread (Yield Curve)",
            value=1.5,
            date="2026-03-09",
            signal_type="yield_curve",
            direction="bullish",
        )
        defaults.update(kwargs)
        return MacroIndicator(**defaults)

    def test_source_is_fred_macro(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.source == "fred_macro"

    def test_domain_is_macro(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.domain == "macro"

    def test_direction_carried_through(self):
        sig = from_macro_indicator(self._make_indicator(direction="bearish"))
        assert sig.direction == "bearish"

    def test_signal_id_format(self):
        sig = from_macro_indicator(self._make_indicator(series_id="T10Y2Y", date="2026-03-09"))
        assert sig.signal_id == "fred:T10Y2Y:2026-03-09"

    def test_symbol_is_empty_for_macro(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.symbol == ""

    def test_asset_class_is_macro(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.asset_class == "macro"

    def test_confidence_carried_from_indicator(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.confidence == 0.70

    def test_decay_rate_carried_from_indicator(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.decay_rate == 0.01

    def test_metadata_has_series_id(self):
        sig = from_macro_indicator(self._make_indicator(series_id="VIXCLS"))
        assert sig.metadata["series_id"] == "VIXCLS"

    def test_metadata_has_series_name(self):
        ind = self._make_indicator(series_name="VIX Volatility Index")
        sig = from_macro_indicator(ind)
        assert sig.metadata["series_name"] == "VIX Volatility Index"

    def test_metadata_has_value(self):
        sig = from_macro_indicator(self._make_indicator(value=1.23))
        assert sig.metadata["value"] == 1.23

    def test_metadata_has_signal_type(self):
        sig = from_macro_indicator(self._make_indicator(signal_type="yield_curve"))
        assert sig.metadata["signal_type"] == "yield_curve"

    def test_raw_id_is_series_id(self):
        sig = from_macro_indicator(self._make_indicator(series_id="UNRATE"))
        assert sig.raw_id == "UNRATE"

    def test_raw_type_is_macro_indicator(self):
        sig = from_macro_indicator(self._make_indicator())
        assert sig.raw_type == "MacroIndicator"


# --- from_fred_yield produces identical output to from_macro_indicator ---

class TestFromFredYield:
    def _make_indicator(self, **kwargs):
        defaults = dict(
            series_id="DGS2",
            series_name="2-Year Treasury Constant Maturity Rate",
            value=4.85,
            date="2026-03-09",
            signal_type="treasury_2y",
            direction="neutral",
        )
        defaults.update(kwargs)
        return MacroIndicator(**defaults)

    def test_identical_to_from_macro_indicator(self):
        ind = self._make_indicator()
        sig_macro = from_macro_indicator(ind)
        sig_yield = from_fred_yield(ind)
        # Core fields must match
        assert sig_yield.signal_id == sig_macro.signal_id
        assert sig_yield.source == sig_macro.source
        assert sig_yield.domain == sig_macro.domain
        assert sig_yield.direction == sig_macro.direction
        assert sig_yield.confidence == sig_macro.confidence
        assert sig_yield.decay_rate == sig_macro.decay_rate
        assert sig_yield.metadata == sig_macro.metadata
        assert sig_yield.raw_id == sig_macro.raw_id

    def test_signal_id_format_for_dgs10(self):
        ind = self._make_indicator(series_id="DGS10", date="2026-03-09")
        sig = from_fred_yield(ind)
        assert sig.signal_id == "fred:DGS10:2026-03-09"

    def test_t10y3m_bearish_when_inverted(self):
        ind = self._make_indicator(
            series_id="T10Y3M",
            value=-0.3,
            signal_type="yield_curve_3m",
            direction="bearish",
        )
        sig = from_fred_yield(ind)
        assert sig.direction == "bearish"


# --- FREDClient.get_yield_curves_and_dollar ---

class TestGetYieldCurvesAndDollar:
    def _make_indicator(self, series_id):
        return MacroIndicator(
            series_id=series_id,
            series_name=f"Series {series_id}",
            value=4.0,
            date="2026-03-09",
            signal_type="treasury_2y",
            direction="neutral",
        )

    def test_calls_get_series_for_all_four_series(self):
        client = FREDClient(api_key="test_key")
        indicators = {
            "DGS2": self._make_indicator("DGS2"),
            "DGS10": self._make_indicator("DGS10"),
            "T10Y3M": self._make_indicator("T10Y3M"),
            "DTWEXBGS": self._make_indicator("DTWEXBGS"),
        }
        client.get_series = MagicMock(side_effect=lambda sid: indicators.get(sid))

        result = client.get_yield_curves_and_dollar()

        assert client.get_series.call_count == 4
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert called_ids == {"DGS2", "DGS10", "T10Y3M", "DTWEXBGS"}

    def test_returns_list_of_macro_indicators(self):
        client = FREDClient(api_key="test_key")
        ind = self._make_indicator("DGS2")
        client.get_series = MagicMock(return_value=ind)

        result = client.get_yield_curves_and_dollar()

        assert len(result) == 4
        for item in result:
            assert isinstance(item, MacroIndicator)

    def test_skips_series_that_return_none(self):
        client = FREDClient(api_key="test_key")
        # Only DGS2 returns a value; others return None
        client.get_series = MagicMock(
            side_effect=lambda sid: self._make_indicator(sid) if sid == "DGS2" else None
        )

        result = client.get_yield_curves_and_dollar()

        assert len(result) == 1
        assert result[0].series_id == "DGS2"

    def test_returns_empty_when_all_fail(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(return_value=None)

        result = client.get_yield_curves_and_dollar()

        assert result == []


# --- fetch_fred sensing_fetchers function ---

class TestFetchFred:
    def _make_indicator(self):
        return MacroIndicator(
            series_id="T10Y2Y",
            series_name="10Y-2Y Spread",
            value=0.75,
            date="2026-03-09",
            signal_type="yield_curve",
            direction="bullish",
        )

    def test_returns_empty_when_no_client(self):
        assert fetch_fred(None, lambda x: x) == []

    def test_fetches_and_converts(self):
        mock_client = MagicMock()
        ind = self._make_indicator()
        mock_client.get_macro_snapshot.return_value = [ind]

        converter = MagicMock(return_value="converted_signal")
        result = fetch_fred(mock_client, converter)

        assert result == ["converted_signal"]
        converter.assert_called_once_with(ind)

    def test_fetches_multiple_indicators(self):
        mock_client = MagicMock()
        indicators = [self._make_indicator(), self._make_indicator()]
        mock_client.get_macro_snapshot.return_value = indicators

        converter = MagicMock(side_effect=lambda x: f"sig_{id(x)}")
        result = fetch_fred(mock_client, converter)

        assert len(result) == 2
        assert converter.call_count == 2

    def test_handles_fetch_exception(self):
        mock_client = MagicMock()
        mock_client.get_macro_snapshot.side_effect = Exception("network timeout")

        result = fetch_fred(mock_client, lambda x: x)

        assert result == []

    def test_handles_converter_exception(self):
        mock_client = MagicMock()
        mock_client.get_macro_snapshot.return_value = [self._make_indicator()]

        def bad_converter(x):
            raise ValueError("conversion failed")

        result = fetch_fred(mock_client, bad_converter)

        # Converter failure is swallowed — returns empty, not an exception
        assert result == []

    def test_partial_converter_failure_skips_bad_items(self):
        mock_client = MagicMock()
        ind_good = self._make_indicator()
        ind_bad = self._make_indicator()
        mock_client.get_macro_snapshot.return_value = [ind_good, ind_bad]

        call_count = {"n": 0}

        def flaky_converter(x):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise ValueError("second conversion failed")
            return "ok"

        result = fetch_fred(mock_client, flaky_converter)

        # First succeeds, second is dropped
        assert result == ["ok"]


# --- fetch_fred_yields sensing_fetchers function ---

class TestFetchFredYields:
    def _make_yield_indicator(self, series_id="DGS2"):
        return MacroIndicator(
            series_id=series_id,
            series_name=f"Series {series_id}",
            value=4.75,
            date="2026-03-09",
            signal_type="treasury_2y",
            direction="neutral",
        )

    def test_returns_empty_when_no_client(self):
        assert fetch_fred_yields(None, lambda x: x) == []

    def test_fetches_and_converts(self):
        mock_client = MagicMock()
        indicators = [
            self._make_yield_indicator("DGS2"),
            self._make_yield_indicator("DGS10"),
            self._make_yield_indicator("T10Y3M"),
            self._make_yield_indicator("DTWEXBGS"),
        ]
        mock_client.get_yield_curves_and_dollar.return_value = indicators

        converter = MagicMock(return_value="yield_signal")
        result = fetch_fred_yields(mock_client, converter)

        assert mock_client.get_yield_curves_and_dollar.called
        assert len(result) == 4
        assert all(r == "yield_signal" for r in result)

    def test_calls_get_yield_curves_not_get_macro_snapshot(self):
        mock_client = MagicMock()
        mock_client.get_yield_curves_and_dollar.return_value = []

        fetch_fred_yields(mock_client, lambda x: x)

        mock_client.get_yield_curves_and_dollar.assert_called_once()
        mock_client.get_macro_snapshot.assert_not_called()

    def test_handles_fetch_exception(self):
        mock_client = MagicMock()
        mock_client.get_yield_curves_and_dollar.side_effect = RuntimeError("API down")

        result = fetch_fred_yields(mock_client, lambda x: x)

        assert result == []

    def test_handles_converter_exception(self):
        mock_client = MagicMock()
        mock_client.get_yield_curves_and_dollar.return_value = [
            self._make_yield_indicator("DGS2")
        ]

        def bad_converter(x):
            raise TypeError("bad type")

        result = fetch_fred_yields(mock_client, bad_converter)

        assert result == []


# --- Sensing hook wiring ---

class TestSensingHookWiring:
    def test_fred_macro_in_source_rotation(self):
        from mae_core.market.sensing_hook import SOURCE_ROTATION
        assert "fred_macro" in SOURCE_ROTATION

    def test_fred_yields_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "fred_yields" in TIER_ROUTING
        assert TIER_ROUTING["fred_yields"] == "thematic"

    def test_fred_macro_in_tier_routing(self):
        from mae_core.market.sensing_hook import TIER_ROUTING
        assert "fred_macro" in TIER_ROUTING
        assert TIER_ROUTING["fred_macro"] == "thematic"

    def test_fred_macro_in_rotation_to_thompson(self):
        from mae_core.market.sensing_hook import _ROTATION_TO_THOMPSON
        assert "fred_macro" in _ROTATION_TO_THOMPSON

    def test_fred_macro_in_absence_source_domains(self):
        from mae_core.market.sensing_hook import _ABSENCE_SOURCE_DOMAINS
        assert "fred_macro" in _ABSENCE_SOURCE_DOMAINS
        assert _ABSENCE_SOURCE_DOMAINS["fred_macro"] == "macro"

    def test_fred_macro_in_source_to_thompson_key(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        assert ConvergenceAlerter._SOURCE_TO_THOMPSON_KEY.get("fred_macro") == "fred_macro"

    def test_macro_domain_in_convergence_alerter(self):
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        alerter = ConvergenceAlerter(min_domains=2)
        assert "macro" in alerter.domain_categories
        # macro uses the default 72h window so it is NOT in _domain_windows
        # (only domains with custom windows appear there)
        assert "macro" not in alerter._domain_windows

    def test_fred_macro_in_source_reliability(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "fred_macro" in LEARNING_CONFIG["source_reliability"]
        val = LEARNING_CONFIG["source_reliability"]["fred_macro"]
        assert 0.0 < val <= 1.0

    def test_macro_decay_rate_configured(self):
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        assert "macro" in LEARNING_CONFIG["decay_rates"]
