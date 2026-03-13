"""Tests for FRED client, yield additions, and FRED signal adapters."""

import pytest
from unittest.mock import MagicMock, patch

from mae_core.market.apis.fred_client import FREDClient, MacroIndicator, _determine_direction
from mae_core.market.apis.fred_models import DELTA_SERIES, _determine_direction_from_delta
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

    # DFF (federal funds rate) — delta-based: rising = bearish (tighter conditions)
    def test_dff_without_delta_is_neutral(self):
        # No prior value available → neutral (cannot determine direction from level alone)
        assert _determine_direction("DFF", 5.5) == "neutral"
        assert _determine_direction("DFF", 0.0) == "neutral"

    def test_dff_rising_is_bearish(self):
        # Fed hikes by 25 bps = tighter conditions = bearish
        assert _determine_direction("DFF", 5.5, change_pct=4.76) == "bearish"

    def test_dff_falling_is_bullish(self):
        # Fed cuts = loosening conditions = bullish
        assert _determine_direction("DFF", 4.75, change_pct=-13.64) == "bullish"

    def test_dff_flat_is_neutral(self):
        # Change below neutral band (0.10%) = neutral
        assert _determine_direction("DFF", 5.33, change_pct=0.05) == "neutral"

    # UNRATE (unemployment)
    def test_unrate_high_is_bearish(self):
        assert _determine_direction("UNRATE", 6.0) == "bearish"

    def test_unrate_low_is_bullish(self):
        assert _determine_direction("UNRATE", 3.5) == "bullish"

    def test_unrate_mid_is_neutral(self):
        assert _determine_direction("UNRATE", 4.5) == "neutral"

    # CPIAUCSL (inflation) — delta-based: rising = bearish (price pressure)
    def test_cpiaucsl_without_delta_is_neutral(self):
        assert _determine_direction("CPIAUCSL", 300.0) == "neutral"
        assert _determine_direction("CPIAUCSL", 100.0) == "neutral"

    def test_cpiaucsl_rising_is_bearish(self):
        assert _determine_direction("CPIAUCSL", 310.0, change_pct=0.30) == "bearish"

    def test_cpiaucsl_falling_is_bullish(self):
        assert _determine_direction("CPIAUCSL", 299.0, change_pct=-0.33) == "bullish"

    def test_cpiaucsl_flat_is_neutral(self):
        assert _determine_direction("CPIAUCSL", 300.0, change_pct=0.02) == "neutral"

    # TSIFRGHT (Freight Transportation Services Index — logistics demand)
    def test_freight_tsi_high_is_bullish(self):
        assert _determine_direction("TSIFRGHT", 118) == "bullish"

    def test_freight_tsi_low_is_bearish(self):
        assert _determine_direction("TSIFRGHT", 95) == "bearish"

    def test_freight_tsi_mid_is_neutral(self):
        assert _determine_direction("TSIFRGHT", 108) == "neutral"

    # Unknown series
    def test_unknown_series_returns_neutral(self):
        assert _determine_direction("UNKNOWN_SERIES_XYZ", 99.9) == "neutral"
        assert _determine_direction("", 0.0) == "neutral"


# --- New series direction logic (Wave 2 additions) ---

class TestNewSeriesDirectionLogic:
    # PCEPI — Fed's preferred inflation measure: delta-based, rising = bearish
    def test_pcepi_without_delta_is_neutral(self):
        assert _determine_direction("PCEPI", 115.0) == "neutral"
        assert _determine_direction("PCEPI", 80.0) == "neutral"

    def test_pcepi_rising_is_bearish(self):
        assert _determine_direction("PCEPI", 115.0, change_pct=0.25) == "bearish"

    def test_pcepi_falling_is_bullish(self):
        assert _determine_direction("PCEPI", 114.0, change_pct=-0.20) == "bullish"

    def test_pcepi_below_neutral_band_is_neutral(self):
        # 0.03% change < 0.05% band → neutral
        assert _determine_direction("PCEPI", 115.0, change_pct=0.03) == "neutral"

    # PCEPILFE — Core PCE: delta-based, rising = bearish
    def test_pcepilfe_without_delta_is_neutral(self):
        assert _determine_direction("PCEPILFE", 120.0) == "neutral"
        assert _determine_direction("PCEPILFE", 95.0) == "neutral"

    def test_pcepilfe_rising_is_bearish(self):
        assert _determine_direction("PCEPILFE", 120.0, change_pct=0.18) == "bearish"

    def test_pcepilfe_falling_is_bullish(self):
        assert _determine_direction("PCEPILFE", 119.0, change_pct=-0.12) == "bullish"

    def test_pcepilfe_below_neutral_band_is_neutral(self):
        assert _determine_direction("PCEPILFE", 120.0, change_pct=0.02) == "neutral"

    # RSXFS — Retail Sales: delta-based, rising = bullish
    def test_rsxfs_without_delta_is_neutral(self):
        assert _determine_direction("RSXFS", 500000.0) == "neutral"
        assert _determine_direction("RSXFS", 300000.0) == "neutral"

    def test_rsxfs_rising_is_bullish(self):
        assert _determine_direction("RSXFS", 510000.0, change_pct=2.0) == "bullish"

    def test_rsxfs_falling_is_bearish(self):
        assert _determine_direction("RSXFS", 490000.0, change_pct=-2.0) == "bearish"

    def test_rsxfs_below_neutral_band_is_neutral(self):
        # 0.15% < 0.20% band → neutral
        assert _determine_direction("RSXFS", 500300.0, change_pct=0.15) == "neutral"

    # UMCSENT — University of Michigan Consumer Sentiment
    def test_umcsent_high_is_bullish(self):
        assert _determine_direction("UMCSENT", 90.0) == "bullish"

    def test_umcsent_low_is_bearish(self):
        assert _determine_direction("UMCSENT", 60.0) == "bearish"

    def test_umcsent_mid_is_neutral(self):
        assert _determine_direction("UMCSENT", 75.0) == "neutral"

    def test_umcsent_exactly_at_bullish_threshold(self):
        assert _determine_direction("UMCSENT", 85.0) == "bullish"

    def test_umcsent_exactly_at_bearish_threshold(self):
        assert _determine_direction("UMCSENT", 65.0) == "bearish"

    # M2SL — M2 Money Supply: delta-based, rising = bullish (liquidity expanding)
    def test_m2sl_without_delta_is_neutral(self):
        assert _determine_direction("M2SL", 21000.0) == "neutral"
        assert _determine_direction("M2SL", 15000.0) == "neutral"

    def test_m2sl_rising_is_bullish(self):
        assert _determine_direction("M2SL", 21200.0, change_pct=0.95) == "bullish"

    def test_m2sl_falling_is_bearish(self):
        # M2 contraction = liquidity withdrawal = bearish
        assert _determine_direction("M2SL", 20800.0, change_pct=-0.95) == "bearish"

    def test_m2sl_below_neutral_band_is_neutral(self):
        # 0.05% < 0.10% band → neutral
        assert _determine_direction("M2SL", 21010.0, change_pct=0.05) == "neutral"

    # T5YIE — 5-Year Breakeven Inflation Rate
    def test_t5yie_high_is_bearish(self):
        assert _determine_direction("T5YIE", 3.5) == "bearish"

    def test_t5yie_low_is_bullish(self):
        assert _determine_direction("T5YIE", 1.8) == "bullish"

    def test_t5yie_anchored_is_neutral(self):
        assert _determine_direction("T5YIE", 2.4) == "neutral"

    def test_t5yie_exactly_at_bearish_threshold_is_bearish(self):
        # 3.0% exactly is dangerous — classified bearish (threshold is inclusive)
        assert _determine_direction("T5YIE", 3.1) == "bearish"

    def test_t5yie_exactly_at_bullish_threshold_is_bullish(self):
        # Below 2.0% = anchored inflation = bullish
        assert _determine_direction("T5YIE", 1.9) == "bullish"

    def test_t5yie_boundary_at_3_is_bearish(self):
        # The bearish threshold is >= 3.0 (inclusive), so exactly 3.0 is bearish
        assert _determine_direction("T5YIE", 3.0) == "bearish"

    def test_t5yie_boundary_at_2_is_neutral(self):
        # The bullish threshold is < 2.0, so 2.0 itself is neutral
        assert _determine_direction("T5YIE", 2.0) == "neutral"

    # HOUST — Housing Starts
    def test_houst_strong_is_bullish(self):
        assert _determine_direction("HOUST", 1500.0) == "bullish"

    def test_houst_weak_is_bearish(self):
        assert _determine_direction("HOUST", 800.0) == "bearish"

    def test_houst_mid_is_neutral(self):
        assert _determine_direction("HOUST", 1200.0) == "neutral"

    # PERMIT — Building Permits
    def test_permit_strong_is_bullish(self):
        assert _determine_direction("PERMIT", 1450.0) == "bullish"

    def test_permit_weak_is_bearish(self):
        assert _determine_direction("PERMIT", 850.0) == "bearish"

    def test_permit_mid_is_neutral(self):
        assert _determine_direction("PERMIT", 1150.0) == "neutral"

    # DCOILWTICO — WTI Crude Oil Price
    def test_dcoilwtico_very_high_is_bearish(self):
        assert _determine_direction("DCOILWTICO", 95.0) == "bearish"

    def test_dcoilwtico_low_is_bullish(self):
        assert _determine_direction("DCOILWTICO", 45.0) == "bullish"

    def test_dcoilwtico_mid_is_neutral(self):
        assert _determine_direction("DCOILWTICO", 70.0) == "neutral"

    # BAMLC0A0CM — Investment Grade Corporate Bond Spread
    def test_ig_spread_wide_is_bearish(self):
        assert _determine_direction("BAMLC0A0CM", 2.5) == "bearish"

    def test_ig_spread_tight_is_bullish(self):
        assert _determine_direction("BAMLC0A0CM", 0.8) == "bullish"

    def test_ig_spread_mid_is_neutral(self):
        assert _determine_direction("BAMLC0A0CM", 1.5) == "neutral"

    # TEDRATE — TED Spread (bank stress indicator)
    def test_tedrate_high_is_bearish(self):
        assert _determine_direction("TEDRATE", 1.5) == "bearish"

    def test_tedrate_low_is_bullish(self):
        assert _determine_direction("TEDRATE", 0.2) == "bullish"

    def test_tedrate_mid_is_neutral(self):
        assert _determine_direction("TEDRATE", 0.6) == "neutral"

    # ICSA — Initial Jobless Claims
    def test_icsa_very_low_is_bullish(self):
        assert _determine_direction("ICSA", 200.0) == "bullish"

    def test_icsa_high_is_bearish(self):
        assert _determine_direction("ICSA", 400.0) == "bearish"

    def test_icsa_mid_is_neutral(self):
        assert _determine_direction("ICSA", 270.0) == "neutral"

    # DGS2 / DGS10 — treasury yields: delta-based, rising = bearish (tighter)
    def test_dgs2_without_delta_is_neutral(self):
        assert _determine_direction("DGS2", 4.85) == "neutral"

    def test_dgs2_rising_is_bearish(self):
        assert _determine_direction("DGS2", 5.00, change_pct=3.09) == "bearish"

    def test_dgs2_falling_is_bullish(self):
        assert _determine_direction("DGS2", 4.50, change_pct=-7.22) == "bullish"

    def test_dgs2_below_neutral_band_is_neutral(self):
        # 0.03% < 0.05% band → neutral
        assert _determine_direction("DGS2", 4.86, change_pct=0.02) == "neutral"

    def test_dgs10_without_delta_is_neutral(self):
        assert _determine_direction("DGS10", 4.50) == "neutral"

    def test_dgs10_rising_is_bearish(self):
        assert _determine_direction("DGS10", 4.60, change_pct=2.22) == "bearish"

    def test_dgs10_falling_is_bullish(self):
        assert _determine_direction("DGS10", 4.30, change_pct=-4.44) == "bullish"


# --- DELTA_SERIES set and _determine_direction_from_delta ---

class TestDeltaSeriesSet:
    def test_delta_series_contains_expected_series(self):
        expected = {"PCEPI", "PCEPILFE", "CPIAUCSL", "RSXFS", "M2SL", "DFF", "DGS2", "DGS10"}
        assert expected.issubset(DELTA_SERIES), f"Missing: {expected - DELTA_SERIES}"

    def test_delta_series_does_not_contain_level_threshold_series(self):
        # Series with meaningful level thresholds should NOT be in DELTA_SERIES
        for sid in ("T10Y2Y", "VIXCLS", "UMCSENT", "BAMLH0A0HYM2", "T5YIE",
                    "HOUST", "PERMIT", "DCOILWTICO"):
            assert sid not in DELTA_SERIES, f"{sid} should not be in DELTA_SERIES"


class TestDetermineDirectionFromDelta:
    def test_inflation_positive_is_bearish(self):
        # Rising inflation = bearish
        assert _determine_direction_from_delta("PCEPI", 0.25) == "bearish"

    def test_inflation_negative_is_bullish(self):
        assert _determine_direction_from_delta("PCEPI", -0.25) == "bullish"

    def test_inflation_near_zero_is_neutral(self):
        assert _determine_direction_from_delta("PCEPI", 0.02) == "neutral"

    def test_growth_positive_is_bullish(self):
        # Rising retail sales = bullish
        assert _determine_direction_from_delta("RSXFS", 2.0) == "bullish"

    def test_growth_negative_is_bearish(self):
        assert _determine_direction_from_delta("RSXFS", -2.0) == "bearish"

    def test_growth_near_zero_is_neutral(self):
        assert _determine_direction_from_delta("RSXFS", 0.10) == "neutral"

    def test_m2sl_expanding_is_bullish(self):
        assert _determine_direction_from_delta("M2SL", 0.50) == "bullish"

    def test_m2sl_contracting_is_bearish(self):
        assert _determine_direction_from_delta("M2SL", -0.50) == "bearish"

    def test_dff_rising_is_bearish(self):
        assert _determine_direction_from_delta("DFF", 4.76) == "bearish"

    def test_dff_falling_is_bullish(self):
        assert _determine_direction_from_delta("DFF", -13.64) == "bullish"

    def test_dgs2_rising_is_bearish(self):
        assert _determine_direction_from_delta("DGS2", 3.09) == "bearish"

    def test_dgs10_falling_is_bullish(self):
        assert _determine_direction_from_delta("DGS10", -4.44) == "bullish"

    def test_cpiaucsl_rising_is_bearish(self):
        assert _determine_direction_from_delta("CPIAUCSL", 0.30) == "bearish"

    def test_cpiaucsl_falling_is_bullish(self):
        assert _determine_direction_from_delta("CPIAUCSL", -0.10) == "bullish"


# --- get_series fetches two observations for delta series ---

class TestGetSeriesDeltaFetch:
    """Verify get_series() fetches limit=2 for DELTA_SERIES and populates delta fields."""

    def _two_obs_response(self, current_val, prior_val):
        """Build a mock FRED API response with two observations."""
        return {
            "observations": [
                {"date": "2026-03-01", "value": str(current_val)},
                {"date": "2026-02-01", "value": str(prior_val)},
            ]
        }

    def _one_obs_response(self, val):
        return {"observations": [{"date": "2026-03-01", "value": str(val)}]}

    def _make_client(self):
        return FREDClient(api_key="test_key")

    def test_pcepi_fetches_limit_2(self):
        client = self._make_client()
        captured = {}

        def fake_request(endpoint, params):
            captured["limit"] = params.get("limit")
            return self._two_obs_response(115.0, 114.5)

        client._request = fake_request
        client.get_series("PCEPI")
        assert captured["limit"] == 2

    def test_m2sl_fetches_limit_2(self):
        client = self._make_client()
        captured = {}

        def fake_request(endpoint, params):
            captured["limit"] = params.get("limit")
            return self._two_obs_response(21000.0, 20900.0)

        client._request = fake_request
        client.get_series("M2SL")
        assert captured["limit"] == 2

    def test_dff_fetches_limit_2(self):
        client = self._make_client()
        captured = {}

        def fake_request(endpoint, params):
            captured["limit"] = params.get("limit")
            return self._two_obs_response(5.5, 5.25)

        client._request = fake_request
        client.get_series("DFF")
        assert captured["limit"] == 2

    def test_vixcls_still_fetches_limit_1(self):
        # Non-delta series uses the caller's default limit
        client = self._make_client()
        captured = {}

        def fake_request(endpoint, params):
            captured["limit"] = params.get("limit")
            return self._one_obs_response(20.0)

        client._request = fake_request
        client.get_series("VIXCLS")
        assert captured["limit"] == 1

    def test_pcepi_indicator_has_prior_value(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(115.0, 114.5)
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.prior_value == 114.5

    def test_pcepi_indicator_has_change_pct(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(115.0, 114.5)
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.change_pct is not None
        expected = (115.0 - 114.5) / 114.5 * 100.0
        assert abs(ind.change_pct - expected) < 1e-9

    def test_pcepi_rising_produces_bearish_direction(self):
        # 115 up from 114.5 = +0.436% = above 0.05% band → bearish (inflation rising)
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(115.0, 114.5)
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_pcepi_falling_produces_bullish_direction(self):
        # 114.0 down from 115.0 = -0.87% → bullish (inflation cooling)
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(114.0, 115.0)
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.direction == "bullish"

    def test_rsxfs_rising_produces_bullish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(510000.0, 500000.0)
        ind = client.get_series("RSXFS")
        assert ind is not None
        assert ind.direction == "bullish"

    def test_m2sl_falling_produces_bearish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(20800.0, 21000.0)
        ind = client.get_series("M2SL")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_dff_rising_produces_bearish_direction(self):
        # 5.5 up from 5.25 = +4.76% → bearish
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(5.5, 5.25)
        ind = client.get_series("DFF")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_dgs2_rising_produces_bearish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(5.00, 4.85)
        ind = client.get_series("DGS2")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_dgs10_falling_produces_bullish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(4.20, 4.50)
        ind = client.get_series("DGS10")
        assert ind is not None
        assert ind.direction == "bullish"

    def test_cpiaucsl_rising_produces_bearish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(310.0, 309.0)
        ind = client.get_series("CPIAUCSL")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_pcepilfe_rising_produces_bearish_direction(self):
        client = self._make_client()
        client._request = lambda ep, params: self._two_obs_response(120.0, 119.5)
        ind = client.get_series("PCEPILFE")
        assert ind is not None
        assert ind.direction == "bearish"

    def test_only_one_obs_available_falls_back_to_neutral(self):
        # If API only returns 1 observation (rare), delta falls back to neutral
        client = self._make_client()
        client._request = lambda ep, params: self._one_obs_response(115.0)
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.prior_value is None
        assert ind.change_pct is None
        assert ind.direction == "neutral"

    def test_prior_value_missing_falls_back_to_neutral(self):
        # Prior observation has FRED missing-value marker "."
        client = self._make_client()
        client._request = lambda ep, params: {
            "observations": [
                {"date": "2026-03-01", "value": "115.0"},
                {"date": "2026-02-01", "value": "."},
            ]
        }
        ind = client.get_series("PCEPI")
        assert ind is not None
        assert ind.prior_value is None
        assert ind.direction == "neutral"

    def test_non_delta_series_has_no_prior_value(self):
        client = self._make_client()
        client._request = lambda ep, params: self._one_obs_response(20.0)
        ind = client.get_series("VIXCLS")
        assert ind is not None
        assert ind.prior_value is None
        assert ind.change_pct is None


# --- New series are present in FRED_SERIES with correct metadata ---

class TestNewSeriesInFredSeries:
    from mae_core.market.apis.fred_models import FRED_SERIES

    def test_pcepi_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "PCEPI" in FRED_SERIES
        name, signal_type = FRED_SERIES["PCEPI"]
        assert "PCE" in name
        assert signal_type == "inflation_pce"

    def test_pcepilfe_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "PCEPILFE" in FRED_SERIES
        name, signal_type = FRED_SERIES["PCEPILFE"]
        assert "Core PCE" in name
        assert signal_type == "inflation_core_pce"

    def test_rsxfs_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "RSXFS" in FRED_SERIES
        name, signal_type = FRED_SERIES["RSXFS"]
        assert "Retail" in name
        assert signal_type == "consumer_spending"

    def test_umcsent_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "UMCSENT" in FRED_SERIES
        name, signal_type = FRED_SERIES["UMCSENT"]
        assert "Michigan" in name or "Sentiment" in name
        assert signal_type == "consumer_sentiment"

    def test_m2sl_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "M2SL" in FRED_SERIES
        name, signal_type = FRED_SERIES["M2SL"]
        assert "M2" in name
        assert signal_type == "money_supply"

    def test_t5yie_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "T5YIE" in FRED_SERIES
        name, signal_type = FRED_SERIES["T5YIE"]
        assert "Breakeven" in name or "Inflation" in name
        assert signal_type == "inflation_expectations"

    def test_houst_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "HOUST" in FRED_SERIES
        name, signal_type = FRED_SERIES["HOUST"]
        assert "Housing Starts" in name
        assert signal_type == "housing"

    def test_permit_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "PERMIT" in FRED_SERIES
        name, signal_type = FRED_SERIES["PERMIT"]
        assert "Permit" in name or "permit" in name.lower()
        assert signal_type == "housing_permits"

    def test_dcoilwtico_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "DCOILWTICO" in FRED_SERIES
        name, signal_type = FRED_SERIES["DCOILWTICO"]
        assert "WTI" in name or "Crude" in name
        assert signal_type == "energy_price"

    def test_gold_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "GOLDAMGBD228NLBM" in FRED_SERIES
        name, signal_type = FRED_SERIES["GOLDAMGBD228NLBM"]
        assert "Gold" in name
        assert signal_type == "gold_price"

    def test_bamlc0a0cm_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "BAMLC0A0CM" in FRED_SERIES
        name, signal_type = FRED_SERIES["BAMLC0A0CM"]
        assert "Investment Grade" in name or "Corporate" in name
        assert signal_type == "credit_spread_ig"

    def test_tedrate_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "TEDRATE" in FRED_SERIES
        name, signal_type = FRED_SERIES["TEDRATE"]
        assert "TED" in name
        assert signal_type == "bank_stress"

    def test_icsa_in_fred_series(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        assert "ICSA" in FRED_SERIES
        name, signal_type = FRED_SERIES["ICSA"]
        assert "Jobless" in name or "Claims" in name
        assert signal_type == "jobless_claims"

    def test_all_new_series_have_non_empty_name(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        new_series = [
            "PCEPI", "PCEPILFE", "RSXFS", "UMCSENT", "M2SL",
            "T5YIE", "HOUST", "PERMIT", "DCOILWTICO",
            "GOLDAMGBD228NLBM", "BAMLC0A0CM", "TEDRATE", "ICSA",
        ]
        for sid in new_series:
            name, signal_type = FRED_SERIES[sid]
            assert name, f"Empty name for {sid}"
            assert signal_type, f"Empty signal_type for {sid}"

    def test_total_series_count_is_at_least_24(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        # Original 11 + 13 new = 24 minimum
        # (ISM Manufacturing has no free FRED proxy without subscription — omitted per task note)
        assert len(FRED_SERIES) >= 24

    def test_all_original_series_still_present(self):
        from mae_core.market.apis.fred_models import FRED_SERIES
        originals = [
            "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DFF", "UNRATE",
            "CPIAUCSL", "DGS2", "DGS10", "T10Y3M", "DTWEXBGS", "TSIFRGHT",
        ]
        for sid in originals:
            assert sid in FRED_SERIES, f"Original series {sid} was removed"


# --- get_macro_snapshot includes new series ---

class TestMacroSnapshotIncludesNewSeries:
    def _make_indicator(self, series_id):
        from mae_core.market.apis.fred_models import FRED_SERIES
        name, signal_type = FRED_SERIES.get(series_id, (series_id, "macro"))
        return MacroIndicator(
            series_id=series_id,
            series_name=name,
            value=1.0,
            date="2026-03-11",
            signal_type=signal_type,
            direction="neutral",
        )

    def test_snapshot_requests_pcepi(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "PCEPI" in called_ids

    def test_snapshot_requests_pcepilfe(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "PCEPILFE" in called_ids

    def test_snapshot_requests_t5yie(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "T5YIE" in called_ids

    def test_snapshot_requests_icsa(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "ICSA" in called_ids

    def test_snapshot_requests_umcsent(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "UMCSENT" in called_ids

    def test_snapshot_requests_m2sl(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "M2SL" in called_ids

    def test_snapshot_requests_houst(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "HOUST" in called_ids

    def test_snapshot_requests_permit(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "PERMIT" in called_ids

    def test_snapshot_requests_dcoilwtico(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "DCOILWTICO" in called_ids

    def test_snapshot_requests_gold(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "GOLDAMGBD228NLBM" in called_ids

    def test_snapshot_requests_bamlc0a0cm(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "BAMLC0A0CM" in called_ids

    def test_snapshot_requests_tedrate(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "TEDRATE" in called_ids

    def test_snapshot_requests_rsxfs(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert "RSXFS" in called_ids

    def test_snapshot_still_includes_all_original_series(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        originals = {"T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DFF", "UNRATE", "DGS2", "DGS10", "T10Y3M", "DTWEXBGS"}
        assert originals.issubset(called_ids), f"Missing originals: {originals - called_ids}"

    def test_snapshot_requests_at_least_22_series(self):
        client = FREDClient(api_key="test_key")
        client.get_series = MagicMock(side_effect=self._make_indicator)
        client.get_macro_snapshot()
        # Original 9 (in snapshot) + 13 new = 22 minimum
        assert client.get_series.call_count >= 22


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

    def test_calls_get_series_for_all_series(self):
        client = FREDClient(api_key="test_key")
        indicators = {
            "DGS2": self._make_indicator("DGS2"),
            "DGS10": self._make_indicator("DGS10"),
            "T10Y3M": self._make_indicator("T10Y3M"),
            "DTWEXBGS": self._make_indicator("DTWEXBGS"),
            "TSIFRGHT": self._make_indicator("TSIFRGHT"),
        }
        client.get_series = MagicMock(side_effect=lambda sid: indicators.get(sid))

        result = client.get_yield_curves_and_dollar()

        assert client.get_series.call_count == 5
        called_ids = {call.args[0] for call in client.get_series.call_args_list}
        assert called_ids == {"DGS2", "DGS10", "T10Y3M", "DTWEXBGS", "TSIFRGHT"}

    def test_returns_list_of_macro_indicators(self):
        client = FREDClient(api_key="test_key")
        ind = self._make_indicator("DGS2")
        client.get_series = MagicMock(return_value=ind)

        result = client.get_yield_curves_and_dollar()

        assert len(result) == 5
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
