"""Tests for FinViz client.

Covers: dataclass creation, get_insider_trades, get_unusual_volume,
get_high_short_float, rate limiter, graceful ImportError degradation,
exception handling, and numeric parsing with commas/dollar signs/percentages.
All finvizfinance calls are mocked — no real network traffic.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mae_core.market.apis.finviz_client import (
    FinVizClient,
    FinVizInsiderTrade,
    UnusualVolume,
    ShortSqueezeCandidate,
    _RATE_LIMIT_SECONDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insider_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a mock insider trades DataFrame."""
    default = [
        {
            "Ticker": "AAPL",
            "Owner": "Tim Cook",
            "Relationship": "CEO",
            "Transaction": "Sale",
            "Date": "Mar 01 '26",
            "Shares Traded": "10,000",
            "Value ($)": "$1,500,000",
            "Shares Owned": "3,280,000",
        }
    ]
    return pd.DataFrame(rows if rows is not None else default)


def _volume_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a mock unusual volume DataFrame."""
    default = [
        {
            "Ticker": "GME",
            "Company": "GameStop Corp",
            "Sector": "Consumer Cyclical",
            "Price": "25.50",
            "Change": "12.30%",
            "Volume": "45,000,000",
            "Avg Volume": "8,000,000",
        }
    ]
    return pd.DataFrame(rows if rows is not None else default)


def _short_df(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a mock high short float DataFrame."""
    default = [
        {
            "Ticker": "BBBY",
            "Company": "Bed Bath & Beyond",
            "Sector": "Consumer Cyclical",
            "Price": "3.25",
            "Short Float": "87.50%",
            "Short Ratio": "4.2",
            "Float": "115.5M",
            "Short Interest": "101.1M",
        }
    ]
    return pd.DataFrame(rows if rows is not None else default)


# ---------------------------------------------------------------------------
# Dataclass creation tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_finviz_insider_trade_creation(self):
        trade = FinVizInsiderTrade(
            ticker="AAPL",
            owner_name="Tim Cook",
            relationship="CEO",
            transaction_type="Sale",
            date="Mar 01 '26",
            shares_traded=10_000,
            value=1_500_000.0,
            shares_owned=3_280_000,
        )
        assert trade.ticker == "AAPL"
        assert trade.owner_name == "Tim Cook"
        assert trade.relationship == "CEO"
        assert trade.transaction_type == "Sale"
        assert trade.shares_traded == 10_000
        assert trade.value == pytest.approx(1_500_000.0)
        assert trade.shares_owned == 3_280_000

    def test_unusual_volume_creation(self):
        uv = UnusualVolume(
            ticker="GME",
            company="GameStop Corp",
            sector="Consumer Cyclical",
            price=25.50,
            change_pct=12.30,
            volume=45_000_000,
            avg_volume=8_000_000,
            volume_ratio=5.625,
        )
        assert uv.ticker == "GME"
        assert uv.volume_ratio == pytest.approx(5.625)

    def test_short_squeeze_candidate_creation(self):
        ssc = ShortSqueezeCandidate(
            ticker="BBBY",
            company="Bed Bath & Beyond",
            sector="Consumer Cyclical",
            price=3.25,
            short_float_pct=87.5,
            short_ratio=4.2,
            float_shares=115_500_000,
            short_interest=101_100_000,
        )
        assert ssc.ticker == "BBBY"
        assert ssc.short_float_pct == pytest.approx(87.5)
        assert ssc.float_shares == 115_500_000


# ---------------------------------------------------------------------------
# get_insider_trades tests
# ---------------------------------------------------------------------------

class TestGetInsiderTrades:
    def test_returns_parsed_trades(self):
        client = FinVizClient()
        mock_insider = MagicMock()
        mock_insider.getInsider.return_value = _insider_df()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Insider", return_value=mock_insider), \
             patch.object(client, "_rate_limit"):
            trades = client.get_insider_trades()

        assert len(trades) == 1
        t = trades[0]
        assert t.ticker == "AAPL"
        assert t.owner_name == "Tim Cook"
        assert t.relationship == "CEO"
        assert t.transaction_type == "Sale"
        assert t.shares_traded == 10_000
        assert t.value == pytest.approx(1_500_000.0)
        assert t.shares_owned == 3_280_000

    def test_empty_dataframe_returns_empty_list(self):
        client = FinVizClient()
        mock_insider = MagicMock()
        mock_insider.getInsider.return_value = pd.DataFrame()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Insider", return_value=mock_insider), \
             patch.object(client, "_rate_limit"):
            trades = client.get_insider_trades()

        assert trades == []

    def test_returns_empty_when_finviz_not_installed(self):
        client = FinVizClient()
        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", False):
            trades = client.get_insider_trades()
        assert trades == []

    def test_exception_during_fetch_returns_empty_list(self):
        client = FinVizClient()
        mock_insider = MagicMock()
        mock_insider.getInsider.side_effect = RuntimeError("network failure")

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Insider", return_value=mock_insider), \
             patch.object(client, "_rate_limit"):
            trades = client.get_insider_trades()

        assert trades == []

    def test_numeric_parsing_strips_commas_and_dollar_signs(self):
        rows = [
            {
                "Ticker": "TSLA",
                "Owner": "Elon Musk",
                "Relationship": "10% Owner",
                "Transaction": "Buy",
                "Date": "Mar 02 '26",
                "Shares Traded": "2,500,000",
                "Value ($)": "$500,000,000",
                "Shares Owned": "411,062,076",
            }
        ]
        client = FinVizClient()
        mock_insider = MagicMock()
        mock_insider.getInsider.return_value = _insider_df(rows)

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Insider", return_value=mock_insider), \
             patch.object(client, "_rate_limit"):
            trades = client.get_insider_trades()

        assert len(trades) == 1
        assert trades[0].shares_traded == 2_500_000
        assert trades[0].value == pytest.approx(500_000_000.0)
        assert trades[0].shares_owned == 411_062_076


# ---------------------------------------------------------------------------
# get_unusual_volume tests
# ---------------------------------------------------------------------------

class TestGetUnusualVolume:
    def test_returns_parsed_volume(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = _volume_df()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_unusual_volume()

        assert len(results) == 1
        r = results[0]
        assert r.ticker == "GME"
        assert r.company == "GameStop Corp"
        assert r.volume == 45_000_000
        assert r.avg_volume == 8_000_000
        assert r.volume_ratio == pytest.approx(45_000_000 / 8_000_000)
        assert r.change_pct == pytest.approx(12.30)

    def test_empty_dataframe_returns_empty_list(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = pd.DataFrame()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_unusual_volume()

        assert results == []

    def test_returns_empty_when_finviz_not_installed(self):
        client = FinVizClient()
        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", False):
            results = client.get_unusual_volume()
        assert results == []

    def test_exception_during_fetch_returns_empty_list(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.side_effect = ConnectionError("timeout")

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_unusual_volume()

        assert results == []

    def test_volume_ratio_computed_correctly(self):
        rows = [
            {
                "Ticker": "AMC",
                "Company": "AMC Entertainment",
                "Sector": "Communication Services",
                "Price": "5.10",
                "Change": "8.00%",
                "Volume": "30,000,000",
                "Avg Volume": "5,000,000",
            }
        ]
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = _volume_df(rows)

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_unusual_volume()

        assert results[0].volume_ratio == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# get_high_short_float tests
# ---------------------------------------------------------------------------

class TestGetHighShortFloat:
    def test_returns_parsed_short_candidates(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = _short_df()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_high_short_float()

        assert len(results) == 1
        r = results[0]
        assert r.ticker == "BBBY"
        assert r.short_float_pct == pytest.approx(87.5)
        assert r.short_ratio == pytest.approx(4.2)
        # "115.5M" → 115_500_000
        assert r.float_shares == pytest.approx(115_500_000, rel=1e-3)
        assert r.short_interest == pytest.approx(101_100_000, rel=1e-3)

    def test_empty_dataframe_returns_empty_list(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = pd.DataFrame()

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_high_short_float()

        assert results == []

    def test_returns_empty_when_finviz_not_installed(self):
        client = FinVizClient()
        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", False):
            results = client.get_high_short_float()
        assert results == []

    def test_exception_during_fetch_returns_empty_list(self):
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.side_effect = RuntimeError("scrape blocked")

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_high_short_float()

        assert results == []

    def test_suffix_parsing_M_B_K(self):
        """Float / Short Interest suffixes M/B/K are parsed to integers correctly."""
        rows = [
            {
                "Ticker": "XYZ",
                "Company": "XYZ Corp",
                "Sector": "Technology",
                "Price": "10.00",
                "Short Float": "25.00%",
                "Short Ratio": "2.5",
                "Float": "1.5B",
                "Short Interest": "375.0M",
            }
        ]
        client = FinVizClient()
        mock_overview = MagicMock()
        mock_overview.screener_view.return_value = _short_df(rows)

        with patch("mae_core.market.apis.finviz_client._HAS_FINVIZ", True), \
             patch("mae_core.market.apis.finviz_client.Overview", return_value=mock_overview), \
             patch.object(client, "_rate_limit"):
            results = client.get_high_short_float()

        assert results[0].float_shares == pytest.approx(1_500_000_000, rel=1e-3)
        assert results[0].short_interest == pytest.approx(375_000_000, rel=1e-3)


# ---------------------------------------------------------------------------
# Rate limiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_rate_limit_constant_is_5_seconds(self):
        assert _RATE_LIMIT_SECONDS == 5.0

    def test_initial_last_request_time_is_zero(self):
        client = FinVizClient()
        assert client._last_request_time == 0.0

    def test_rate_limit_updates_timestamp(self):
        client = FinVizClient()
        before = time.time()
        client._rate_limit()
        after = time.time()
        assert before <= client._last_request_time <= after

    def test_rapid_successive_call_is_throttled(self):
        """Second call within 5s window should invoke sleep."""
        client = FinVizClient()
        client._last_request_time = time.time()  # Simulate a just-completed request

        with patch("mae_core.market.apis.finviz_client.time.sleep") as mock_sleep:
            client._rate_limit()
            mock_sleep.assert_called_once()
            sleep_duration = mock_sleep.call_args[0][0]
            # Should sleep for close to 5 seconds (minus the tiny elapsed time)
            assert 0 < sleep_duration <= _RATE_LIMIT_SECONDS
