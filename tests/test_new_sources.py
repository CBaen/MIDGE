"""Tests for MIDGE's 5 new data sources and their signal converters.

Covers:
  - COTClient / COTSignal / from_cot_positioning
  - StockTwitsClient / StockTwitsSentiment / from_stocktwits_sentiment
  - VIXClient / VIXSignal / from_vix_structure
  - TrendsClient / TrendsSignal / from_trends_signal
  - FinnhubClient new methods (get_economic_calendar, get_analyst_recommendations,
    get_earnings_calendar) / EconomicEvent / AnalystRec / from_economic_event
    / from_analyst_recommendation

All HTTP and library calls are mocked — no real API traffic.
"""

import time
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from mae_core.market.apis.cot_client import COTClient, COTSignal
from mae_core.market.apis.stocktwits_client import StockTwitsClient, StockTwitsSentiment
from mae_core.market.apis.vix_client import VIXClient, VIXSignal
from mae_core.market.apis.trends_client import TrendsClient, TrendsSignal
from mae_core.market.apis.finnhub_client import (
    FinnhubClient, EconomicEvent, AnalystRec, EarningsEvent, NewsSentiment,
)
from mae_core.market.signal import (
    MarketSignal,
    from_cot_positioning,
    from_stocktwits_sentiment,
    from_vix_structure,
    from_trends_signal,
    from_economic_event,
    from_analyst_recommendation,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_cot(
    ticker="ES=F",
    comm_long=300_000,
    comm_short=100_000,
    noncomm_long=200_000,
    noncomm_short=50_000,
    oi=500_000,
    report_date="2026-02-21",
) -> COTSignal:
    comm_net = comm_long - comm_short
    noncomm_net = noncomm_long - noncomm_short
    small_net = oi - (comm_long + comm_short + noncomm_long + noncomm_short)
    pct_comm = comm_long / max(1, oi)
    return COTSignal(
        ticker=ticker,
        contract_name="E-MINI S&P 500",
        commercial_long=comm_long,
        commercial_short=comm_short,
        commercial_net=comm_net,
        noncommercial_long=noncomm_long,
        noncommercial_short=noncomm_short,
        noncommercial_net=noncomm_net,
        small_trader_net=small_net,
        open_interest=oi,
        pct_commercial_long=round(pct_comm, 4),
        report_date=report_date,
    )


def _make_stocktwits(ticker="AAPL", bull=70, bear=30, total=200, trending=False) -> StockTwitsSentiment:
    return StockTwitsSentiment(
        ticker=ticker,
        bull_count=bull,
        bear_count=bear,
        bull_ratio=round(bull / (bull + bear), 4),
        total_messages=total,
        trending=trending,
        detected_at=datetime.utcnow().isoformat(),
    )


def _make_vix(
    spot=18.0,
    vix_1m=20.0,
    vix_3m=21.0,
    term_spread=2.0,
    structure_type="contango",
    date="2026-02-27",
) -> VIXSignal:
    return VIXSignal(
        vix_spot=spot,
        vix_1m=vix_1m,
        vix_3m=vix_3m,
        term_spread=term_spread,
        structure_type=structure_type,
        date=date,
    )


def _make_trends(keyword="NVDA", score=80, delta=25.0, is_breakout=True) -> TrendsSignal:
    return TrendsSignal(
        keyword=keyword,
        interest_score=score,
        interest_delta_7d=delta,
        is_breakout=is_breakout,
        detected_at=datetime.utcnow().isoformat(),
    )


def _make_economic_event(
    event="CPI",
    country="US",
    date="2026-03-01",
    impact="high",
    actual=None,
    estimate=3.0,
    previous=3.2,
    unit="%",
) -> EconomicEvent:
    return EconomicEvent(
        event=event,
        country=country,
        date=date,
        time="08:30",
        impact=impact,
        actual=actual,
        estimate=estimate,
        previous=previous,
        unit=unit,
    )


def _make_analyst_rec(
    symbol="AAPL",
    period="2026-02-01",
    strong_buy=20,
    buy=15,
    hold=5,
    sell=2,
    strong_sell=0,
) -> AnalystRec:
    return AnalystRec(
        symbol=symbol,
        period=period,
        strong_buy=strong_buy,
        buy=buy,
        hold=hold,
        sell=sell,
        strong_sell=strong_sell,
    )


# ===========================================================================
# COTClient
# ===========================================================================

class TestCOTSignalDataclass:
    def test_fields_are_set(self):
        cot = _make_cot()
        assert cot.ticker == "ES=F"
        assert cot.contract_name == "E-MINI S&P 500"
        assert cot.commercial_net == 200_000
        assert cot.open_interest == 500_000
        assert cot.pct_commercial_long == 0.6
        assert cot.signal_source == "cot_positioning"

    def test_to_plain_language_long(self):
        cot = _make_cot()
        text = cot.to_plain_language()
        assert "ES=F" in text
        assert "long" in text
        assert "200,000" in text

    def test_to_plain_language_short(self):
        cot = _make_cot(comm_long=100_000, comm_short=300_000)
        text = cot.to_plain_language()
        assert "short" in text

    def test_defaults_applied(self):
        cot = _make_cot()
        assert cot.decay_rate == 0.03
        assert cot.confidence == 0.55


class TestCOTClient:
    def test_missing_library_returns_empty(self):
        """When cot-reports is not installed, get_latest_positions returns []."""
        client = COTClient()
        with patch.dict("sys.modules", {"cot_reports": None}):
            # Force ImportError path by removing from sys.modules
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "cot_reports":
                    raise ImportError("not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                result = client.get_latest_positions()
        assert result == []

    def test_cache_is_used_on_second_call(self):
        """A populated cache is returned without another fetch."""
        client = COTClient()
        cot = _make_cot()
        client._cache = [cot]
        client._cache_time = time.time()  # freshly populated

        # _rate_limit should NOT be called if cache is valid
        with patch.object(client, "_rate_limit") as mock_rl:
            result = client.get_latest_positions()

        mock_rl.assert_not_called()
        assert result == [cot]

    def test_cache_filters_by_symbol(self):
        """Symbols filter works when cache is hot."""
        client = COTClient()
        client._cache = [_make_cot("ES=F"), _make_cot("GC=F")]
        client._cache_time = time.time()

        result = client.get_latest_positions(["GC=F"])
        assert len(result) == 1
        assert result[0].ticker == "GC=F"

    def test_exception_during_fetch_returns_empty(self):
        """Any exception during cot.cot_year() returns []."""
        client = COTClient()
        fake_cot = MagicMock()
        fake_cot.cot_year.side_effect = RuntimeError("network gone")

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cot_reports":
                return fake_cot
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch.object(client, "_rate_limit"):
                result = client.get_latest_positions()

        assert result == []

    def test_parse_dataframe_extracts_signal(self):
        """_parse_dataframe correctly builds COTSignal from a minimal DataFrame."""
        import pandas as pd

        rows = [{
            "Market and Exchange Names": "E-MINI S&P 500 - CME",
            "As of Date in Form YYYY-MM-DD": "2026-02-21",
            "Comm_Positions_Long_All": 300000,
            "Comm_Positions_Short_All": 100000,
            "NonComm_Positions_Long_All": 200000,
            "NonComm_Positions_Short_All": 50000,
            "Open_Interest_All": 500000,
        }]
        df = pd.DataFrame(rows)

        client = COTClient()
        signals = client._parse_dataframe(df, ["ES=F"])

        assert len(signals) == 1
        sig = signals[0]
        assert sig.ticker == "ES=F"
        assert sig.commercial_net == 200_000
        assert sig.open_interest == 500_000

    def test_rate_limit_updates_last_request_time(self):
        """_rate_limit sets _last_request_time after call."""
        client = COTClient()
        client._last_request_time = 0.0
        with patch("time.sleep"):  # skip actual sleeping
            client._rate_limit()
        assert client._last_request_time > 0.0


# ===========================================================================
# StockTwitsClient
# ===========================================================================

class TestStockTwitsSentimentDataclass:
    def test_fields_are_set(self):
        st = _make_stocktwits(bull=70, bear=30)
        assert st.ticker == "AAPL"
        assert st.bull_count == 70
        assert st.bear_count == 30
        assert abs(st.bull_ratio - 0.70) < 0.01
        assert st.signal_source == "stocktwits_sentiment"

    def test_to_plain_language_bullish(self):
        st = _make_stocktwits(bull=70, bear=30)
        text = st.to_plain_language()
        assert "bullish" in text
        assert "70%" in text

    def test_to_plain_language_bearish(self):
        st = _make_stocktwits(bull=25, bear=75)
        text = st.to_plain_language()
        assert "bearish" in text

    def test_to_plain_language_trending_tag(self):
        st = _make_stocktwits(trending=True)
        assert "[TRENDING]" in st.to_plain_language()

    def test_defaults_applied(self):
        st = _make_stocktwits()
        assert st.decay_rate == 0.50
        assert st.confidence == 0.50


class TestStockTwitsClient:
    def _client_with_mock_response(self, ticker: str, data: dict) -> StockTwitsClient:
        client = StockTwitsClient()
        client._last_request_time = 0.0

        def fake_request(url, params=None):
            if ticker.upper() in url:
                return data
            return None

        client._request = fake_request
        return client

    def test_bullish_majority_parsed(self):
        messages = [
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bullish"}}},
            {"entities": {"sentiment": {"basic": "Bearish"}}},
        ]
        client = self._client_with_mock_response("AAPL", {"messages": messages, "symbol": {}})
        # Bypass rate limit
        with patch.object(client, "_rate_limit"):
            result = client.get_sentiment(["AAPL"])
        assert len(result) == 1
        st = result[0]
        assert st.bull_count == 2
        assert st.bear_count == 1
        assert abs(st.bull_ratio - 2 / 3) < 0.01

    def test_no_labeled_messages_returns_empty(self):
        messages = [{"entities": {}}, {"entities": {}}]
        client = self._client_with_mock_response("TSLA", {"messages": messages, "symbol": {}})
        with patch.object(client, "_rate_limit"):
            result = client.get_sentiment(["TSLA"])
        assert result == []

    def test_api_down_returns_empty(self):
        client = StockTwitsClient()
        client._request = lambda url, params=None: None
        with patch.object(client, "_rate_limit"):
            result = client.get_sentiment(["AAPL"])
        assert result == []

    def test_cache_is_used_on_second_call(self):
        cached_sentiment = _make_stocktwits("NVDA", bull=60, bear=40)
        client = StockTwitsClient()
        client._cache["NVDA"] = (cached_sentiment, time.time())

        with patch.object(client, "_rate_limit") as mock_rl:
            result = client.get_sentiment(["NVDA"])

        mock_rl.assert_not_called()
        assert len(result) == 1
        assert result[0].ticker == "NVDA"

    def test_trending_detected_by_message_count(self):
        messages = [{"entities": {"sentiment": {"basic": "Bullish"}}} for _ in range(30)]
        client = self._client_with_mock_response("SPY", {"messages": messages, "symbol": {}})
        with patch.object(client, "_rate_limit"):
            result = client.get_sentiment(["SPY"])
        assert len(result) == 1
        assert result[0].trending is True  # >= 25 messages triggers trending

    def test_graceful_degradation_on_parse_error(self):
        # Malformed message structure — should not crash
        client = self._client_with_mock_response("AAPL", {"messages": [{"bad": "data"}], "symbol": {}})
        with patch.object(client, "_rate_limit"):
            result = client.get_sentiment(["AAPL"])
        # bull=0 bear=0 → returns None → empty list
        assert result == []


# ===========================================================================
# VIXClient
# ===========================================================================

class TestVIXSignalDataclass:
    def test_fields_are_set(self):
        v = _make_vix()
        assert v.vix_spot == 18.0
        assert v.structure_type == "contango"
        assert v.term_spread == 2.0
        assert v.signal_source == "vix_term_structure"

    def test_to_plain_language(self):
        v = _make_vix(spot=22.5, structure_type="backwardation", term_spread=-3.5)
        text = v.to_plain_language()
        assert "22.5" in text
        assert "backwardation" in text
        assert "-3.50" in text

    def test_defaults_applied(self):
        v = _make_vix()
        assert v.decay_rate == 0.30
        assert v.confidence == 0.60


def _build_vix_csv(rows: list) -> str:
    """Build a minimal CBOE-style CSV string from a list of (date, close) tuples."""
    lines = ["DATE,OPEN,HIGH,LOW,CLOSE"]
    for date, close in rows:
        lines.append(f"{date},{close},{close+0.5},{close-0.5},{close}")
    return "\n".join(lines)


class TestVIXClient:
    def test_parse_contango_csv(self):
        """When recent VIX is lower than 20-day avg, term spread is positive → contango."""
        # 20 rows: first 19 have close=22, last (most recent) has close=18
        rows = [(f"2026-01-{i+1:02d}", 22.0) for i in range(19)]
        rows.append(("2026-01-20", 18.0))
        csv_text = _build_vix_csv(rows)

        client = VIXClient()
        result = client._parse_vix_csv(csv_text)

        assert result is not None
        assert result.vix_spot == 18.0
        # 20-day avg ~ 21.9, spread > 1 → contango
        assert result.structure_type == "contango"
        assert result.term_spread > 1.0

    def test_parse_backwardation_csv(self):
        """When recent VIX spikes above 20-day avg by >1, structure is backwardation."""
        rows = [(f"2026-01-{i+1:02d}", 15.0) for i in range(19)]
        rows.append(("2026-01-20", 35.0))  # spike
        csv_text = _build_vix_csv(rows)

        client = VIXClient()
        result = client._parse_vix_csv(csv_text)

        assert result is not None
        assert result.vix_spot == 35.0
        # 20-day avg ~15.9, spread < -1 → backwardation
        assert result.structure_type == "backwardation"

    def test_parse_flat_structure(self):
        """When spread is within ±1, structure is flat."""
        rows = [(f"2026-01-{i+1:02d}", 20.0) for i in range(20)]
        csv_text = _build_vix_csv(rows)

        client = VIXClient()
        result = client._parse_vix_csv(csv_text)

        assert result is not None
        assert result.structure_type == "flat"
        assert abs(result.term_spread) <= 1.0

    def test_empty_csv_returns_none(self):
        client = VIXClient()
        result = client._parse_vix_csv("DATE,OPEN,HIGH,LOW,CLOSE\n")
        assert result is None

    def test_cache_returned_on_second_call(self):
        cached = _make_vix()
        client = VIXClient()
        client._cache = cached
        client._cache_time = time.time()

        with patch.object(client, "_rate_limit") as mock_rl:
            result = client.get_vix_structure()

        mock_rl.assert_not_called()
        assert result is cached

    def test_api_down_returns_none(self):
        client = VIXClient()
        client._fetch_text = lambda url: None
        with patch.object(client, "_rate_limit"):
            result = client.get_vix_structure()
        assert result is None

    def test_result_is_cached_after_fetch(self):
        rows = [(f"2026-01-{i+1:02d}", 18.0) for i in range(25)]
        csv_text = _build_vix_csv(rows)

        client = VIXClient()
        client._fetch_text = lambda url: csv_text
        with patch.object(client, "_rate_limit"):
            result = client.get_vix_structure()

        assert client._cache is result
        assert client._cache_time > 0


# ===========================================================================
# TrendsClient
# ===========================================================================

class TestTrendsSignalDataclass:
    def test_fields_are_set(self):
        t = _make_trends("NVDA", score=80, delta=25.0, is_breakout=True)
        assert t.keyword == "NVDA"
        assert t.interest_score == 80
        assert t.interest_delta_7d == 25.0
        assert t.is_breakout is True
        assert t.signal_source == "google_trends"

    def test_to_plain_language_rising(self):
        t = _make_trends(delta=30.0)
        text = t.to_plain_language()
        assert "rising" in text

    def test_to_plain_language_falling(self):
        t = _make_trends(delta=-25.0)
        text = t.to_plain_language()
        assert "falling" in text

    def test_to_plain_language_stable(self):
        t = _make_trends(delta=5.0)
        text = t.to_plain_language()
        assert "stable" in text

    def test_breakout_tag_shown(self):
        t = _make_trends(is_breakout=True)
        assert "[BREAKOUT]" in t.to_plain_language()

    def test_defaults_applied(self):
        t = _make_trends()
        assert t.decay_rate == 0.50
        assert t.confidence == 0.45


class TestTrendsClient:
    def test_missing_pytrends_returns_empty(self):
        """When pytrends is not installed, get_interest returns []."""
        client = TrendsClient()
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pytrends.request":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = client.get_interest(["SPY"])
        assert result == []

    def test_cache_returned_on_second_call(self):
        cached = [_make_trends("SPY"), _make_trends("recession")]
        client = TrendsClient()
        client._cache = cached
        client._cache_time = time.time()

        result = client.get_interest()
        assert result is cached

    def test_cache_filter_by_keywords(self):
        client = TrendsClient()
        client._cache = [_make_trends("SPY"), _make_trends("recession")]
        client._cache_time = time.time()

        result = client.get_interest(["SPY"])
        assert len(result) == 1
        assert result[0].keyword == "SPY"

    def test_batch_exception_returns_empty(self):
        """If pytrends raises during fetch, batch returns []."""
        client = TrendsClient()
        client._cache_time = 0.0  # bypass cache

        mock_pytrends_cls = MagicMock()
        mock_pytrends_cls.return_value.interest_over_time.side_effect = Exception("rate limited")

        # TrendReq is imported inside _fetch_batch via 'from pytrends.request import TrendReq'
        # so we must patch the symbol at its source location in the pytrends module.
        with patch("pytrends.request.TrendReq", mock_pytrends_cls):
            with patch.object(client, "_rate_limit"):
                result = client._fetch_batch(["SPY"])
        assert result == []

    def test_batch_processes_dataframe(self):
        """_fetch_batch returns TrendsSignal objects from a valid DataFrame."""
        import pandas as pd

        client = TrendsClient()

        idx = pd.date_range("2026-02-20", periods=7, freq="D")
        df = pd.DataFrame({"SPY": [50, 55, 60, 65, 70, 75, 80]}, index=idx)
        df["isPartial"] = False

        mock_pt = MagicMock()
        mock_pt.interest_over_time.return_value = df
        mock_pt.related_queries.return_value = {}

        mock_cls = MagicMock(return_value=mock_pt)

        # TrendReq is imported inside _fetch_batch via 'from pytrends.request import TrendReq'
        # so we must patch it at the source module, not at the trends_client module level.
        with patch("pytrends.request.TrendReq", mock_cls):
            with patch.object(client, "_rate_limit"):
                result = client._fetch_batch(["SPY"])

        assert len(result) == 1
        sig = result[0]
        assert sig.keyword == "SPY"
        assert sig.interest_score == 80        # last value in [50,55,60,65,70,75,80]
        assert sig.interest_delta_7d == 30.0   # 80 - 50
        assert sig.is_breakout is True         # 80 > 75


# ===========================================================================
# FinnhubClient — EconomicEvent and AnalystRec dataclasses
# ===========================================================================

class TestEconomicEventDataclass:
    def test_fields_are_set(self):
        evt = _make_economic_event(actual=3.5, estimate=3.0)
        assert evt.event == "CPI"
        assert evt.country == "US"
        assert evt.actual == 3.5
        assert evt.estimate == 3.0
        assert evt.impact == "high"

    def test_surprise_pct_positive(self):
        evt = _make_economic_event(actual=3.6, estimate=3.0)
        surprise = evt.surprise_pct()
        assert surprise is not None
        assert abs(surprise - 0.20) < 0.01

    def test_surprise_pct_negative(self):
        evt = _make_economic_event(actual=2.5, estimate=3.0)
        surprise = evt.surprise_pct()
        assert surprise is not None
        assert abs(surprise - (-1/6)) < 0.01

    def test_surprise_pct_none_when_missing_actual(self):
        evt = _make_economic_event(actual=None, estimate=3.0)
        assert evt.surprise_pct() is None

    def test_surprise_pct_none_when_zero_estimate(self):
        evt = _make_economic_event(actual=1.0, estimate=0.0)
        assert evt.surprise_pct() is None

    def test_to_plain_language_upcoming(self):
        evt = _make_economic_event(actual=None)
        text = evt.to_plain_language()
        assert "upcoming" in text
        assert "3.0" in text

    def test_to_plain_language_reported(self):
        evt = _make_economic_event(actual=3.5, estimate=3.0)
        text = evt.to_plain_language()
        assert "actual=3.5" in text
        assert "surprise" in text

    def test_defaults_applied(self):
        evt = _make_economic_event()
        assert evt.signal_source == "finnhub_economic"
        assert evt.confidence == 0.55


class TestAnalystRecDataclass:
    def test_total_computed(self):
        rec = _make_analyst_rec(strong_buy=10, buy=10, hold=5, sell=3, strong_sell=2)
        assert rec.total == 30

    def test_buy_ratio_bullish(self):
        rec = _make_analyst_rec(strong_buy=20, buy=15, hold=5, sell=2, strong_sell=0)
        assert rec.buy_ratio > 0.80

    def test_buy_ratio_bearish(self):
        rec = _make_analyst_rec(strong_buy=1, buy=1, hold=3, sell=10, strong_sell=5)
        assert rec.buy_ratio < 0.30

    def test_to_plain_language(self):
        rec = _make_analyst_rec()
        text = rec.to_plain_language()
        assert "AAPL" in text
        assert "strong buy" in text

    def test_zero_analysts_doesnt_divide_by_zero(self):
        rec = _make_analyst_rec(strong_buy=0, buy=0, hold=0, sell=0, strong_sell=0)
        assert rec.total == 0
        assert rec.buy_ratio == 0.0  # 0 / max(1, 0) = 0

    def test_defaults_applied(self):
        rec = _make_analyst_rec()
        assert rec.signal_source == "finnhub_analyst"
        assert rec.decay_rate == 0.05


# ===========================================================================
# FinnhubClient — new methods
# ===========================================================================

class TestFinnhubClientEconomicCalendar:
    def test_returns_major_economy_events(self):
        """US all impacts + major economies high-impact only."""
        payload = {
            "economicCalendar": [
                {"country": "US", "event": "CPI", "date": "2026-03-01",
                 "time": "08:30", "impact": "high", "actual": None,
                 "estimate": 3.0, "prev": 3.2, "unit": "%"},
                {"country": "EU", "event": "ECB Rate Decision", "date": "2026-03-02",
                 "time": "12:00", "impact": "high", "actual": None,
                 "estimate": None, "prev": None, "unit": ""},
                {"country": "JP", "event": "BoJ Rate Decision", "date": "2026-03-03",
                 "time": "03:00", "impact": "high", "actual": None,
                 "estimate": None, "prev": None, "unit": "%"},
                {"country": "EU", "event": "EU Consumer Confidence", "date": "2026-03-03",
                 "time": "10:00", "impact": "medium", "actual": None,
                 "estimate": -14.0, "prev": -14.5, "unit": ""},
                {"country": "BR", "event": "Brazil CPI", "date": "2026-03-04",
                 "time": "09:00", "impact": "high", "actual": None,
                 "estimate": 5.0, "prev": 4.8, "unit": "%"},
            ]
        }
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            events = client.get_economic_calendar(days=14)

        # US CPI (any impact) + ECB high + BoJ high = 3
        # EU medium filtered, Brazil not in MAJOR_ECONOMIES
        assert len(events) == 3
        countries = {e.country for e in events}
        assert countries == {"US", "EU", "JP"}

    def test_empty_response_returns_empty_list(self):
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=None):
            events = client.get_economic_calendar()
        assert events == []

    def test_events_sorted_by_date(self):
        payload = {
            "economicCalendar": [
                {"country": "US", "event": "NFP", "date": "2026-03-05",
                 "time": "08:30", "impact": "high", "actual": None,
                 "estimate": 200.0, "prev": 185.0, "unit": "K"},
                {"country": "US", "event": "CPI", "date": "2026-03-01",
                 "time": "08:30", "impact": "high", "actual": None,
                 "estimate": 3.0, "prev": 3.2, "unit": "%"},
            ]
        }
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            events = client.get_economic_calendar(days=14)

        assert events[0].event == "CPI"
        assert events[1].event == "NFP"

    def test_actual_values_parsed_as_float(self):
        payload = {
            "economicCalendar": [
                {"country": "US", "event": "CPI", "date": "2026-02-27",
                 "time": "08:30", "impact": "high", "actual": "3.5",
                 "estimate": "3.0", "prev": "3.2", "unit": "%"},
            ]
        }
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            events = client.get_economic_calendar()

        assert events[0].actual == 3.5
        assert isinstance(events[0].actual, float)


class TestFinnhubClientAnalystRecommendations:
    def test_returns_analyst_recs(self):
        payload = [
            {"symbol": "AAPL", "period": "2026-02-01",
             "strongBuy": 20, "buy": 15, "hold": 5, "sell": 2, "strongSell": 0},
            {"symbol": "AAPL", "period": "2026-01-01",
             "strongBuy": 18, "buy": 12, "hold": 6, "sell": 3, "strongSell": 1},
        ]
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            recs = client.get_analyst_recommendations("AAPL")

        assert len(recs) == 2
        assert recs[0].period == "2026-02-01"  # Most recent first
        assert recs[0].strong_buy == 20

    def test_empty_response_returns_empty_list(self):
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=None):
            recs = client.get_analyst_recommendations("AAPL")
        assert recs == []

    def test_non_list_response_returns_empty_list(self):
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value={"error": "no data"}):
            recs = client.get_analyst_recommendations("AAPL")
        assert recs == []

    def test_sorted_most_recent_first(self):
        payload = [
            {"symbol": "MSFT", "period": "2025-11-01",
             "strongBuy": 10, "buy": 8, "hold": 4, "sell": 1, "strongSell": 0},
            {"symbol": "MSFT", "period": "2026-02-01",
             "strongBuy": 22, "buy": 10, "hold": 3, "sell": 0, "strongSell": 0},
        ]
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            recs = client.get_analyst_recommendations("MSFT")

        assert recs[0].period == "2026-02-01"


class TestFinnhubClientEarningsCalendar:
    def test_returns_both_upcoming_and_reported(self):
        payload = {
            "earningsCalendar": [
                {"symbol": "AAPL", "date": "2026-02-20",
                 "epsEstimate": 2.10, "epsActual": 2.25,
                 "revenueEstimate": 120e9, "revenueActual": 124e9, "hour": "amc"},
                {"symbol": "MSFT", "date": "2026-03-05",
                 "epsEstimate": 3.00, "epsActual": None,
                 "revenueEstimate": 60e9, "revenueActual": None, "hour": "amc"},
            ]
        }
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            events = client.get_earnings_calendar(days=14)

        assert len(events) == 2

    def test_sorted_by_date_ascending(self):
        payload = {
            "earningsCalendar": [
                {"symbol": "MSFT", "date": "2026-03-05",
                 "epsEstimate": 3.00, "epsActual": None,
                 "revenueEstimate": None, "revenueActual": None, "hour": "amc"},
                {"symbol": "AAPL", "date": "2026-02-20",
                 "epsEstimate": 2.10, "epsActual": 2.25,
                 "revenueEstimate": None, "revenueActual": None, "hour": "amc"},
            ]
        }
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=payload):
            events = client.get_earnings_calendar(days=14)

        assert events[0].symbol == "AAPL"
        assert events[1].symbol == "MSFT"

    def test_empty_response_returns_empty(self):
        client = FinnhubClient(api_key="test_key")
        with patch.object(client, "_get", return_value=None):
            events = client.get_earnings_calendar()
        assert events == []

    def test_403_blocks_endpoint(self):
        """A 403 from the API blocks that endpoint to prevent spam."""
        client = FinnhubClient(api_key="test_key")
        mock_resp = MagicMock()
        mock_resp.status_code = 403

        with patch.object(client._session, "get", return_value=mock_resp):
            with patch.object(client, "_rate_limit"):
                result = client._get("/calendar/economic")

        assert result is None
        assert "/calendar/economic" in client._blocked_endpoints

    def test_no_api_key_returns_none(self):
        client = FinnhubClient(api_key=None)
        # ensure env var is absent too
        with patch.dict("os.environ", {}, clear=True):
            result = client._get("/news-sentiment", {"symbol": "AAPL"})
        assert result is None


# ===========================================================================
# Signal converters
# ===========================================================================

class TestFromCOTPositioning:
    def test_bullish_when_commercials_heavily_long(self):
        cot = _make_cot(comm_long=300_000, comm_short=100_000, oi=500_000)
        # net=200k, pct_comm=0.60 > 0.55 → bullish
        sig = from_cot_positioning(cot)
        assert isinstance(sig, MarketSignal)
        assert sig.direction == "bullish"
        assert sig.source == "cot_positioning"

    def test_bearish_when_commercials_heavily_short(self):
        # pct_comm = 100k/500k = 0.20 < 0.45 and net < 0
        cot = _make_cot(comm_long=100_000, comm_short=300_000, oi=500_000)
        sig = from_cot_positioning(cot)
        assert sig.direction == "bearish"

    def test_neutral_when_balanced(self):
        # net > 0 but pct barely above 0.50 — not past 0.55 threshold
        cot = _make_cot(comm_long=260_000, comm_short=240_000, oi=500_000)
        # pct_comm = 0.52, net=20k > 0 but pct < 0.55 → neutral
        sig = from_cot_positioning(cot)
        assert sig.direction == "neutral"

    def test_strength_scales_with_net_ratio(self):
        # Net 200k / OI 500k = 0.40 → capped at 1.0 (0.40*5=2.0 → min=1.0)
        cot = _make_cot(comm_long=300_000, comm_short=100_000, oi=500_000)
        sig = from_cot_positioning(cot)
        assert sig.strength == 1.0

    def test_small_net_gives_low_strength(self):
        # Net 5k / OI 500k = 0.01 → strength = 0.05
        cot = _make_cot(comm_long=252_500, comm_short=247_500, oi=500_000)
        sig = from_cot_positioning(cot)
        assert sig.strength < 0.10

    def test_signal_id_format(self):
        cot = _make_cot(ticker="GC=F", report_date="2026-02-21")
        sig = from_cot_positioning(cot)
        assert sig.signal_id == "cot:GC=F:2026-02-21"

    def test_metadata_propagated(self):
        cot = _make_cot()
        sig = from_cot_positioning(cot)
        assert "commercial_net" in sig.metadata
        assert "open_interest" in sig.metadata
        assert sig.metadata["open_interest"] == 500_000

    def test_asset_class_and_domain(self):
        cot = _make_cot()
        sig = from_cot_positioning(cot)
        assert sig.asset_class == "futures"
        assert sig.domain == "positioning"

    def test_outcome_window_21_days(self):
        cot = _make_cot()
        sig = from_cot_positioning(cot)
        assert sig.outcome_window_days == 21


class TestFromStockTwitsSentiment:
    def test_bullish_when_over_70_pct(self):
        st = _make_stocktwits(bull=75, bear=25)  # 0.75 → bullish
        sig = from_stocktwits_sentiment(st)
        assert sig.direction == "bullish"
        assert sig.source == "stocktwits_sentiment"

    def test_bearish_when_under_30_pct(self):
        st = _make_stocktwits(bull=20, bear=80)  # 0.20 → bearish
        sig = from_stocktwits_sentiment(st)
        assert sig.direction == "bearish"

    def test_neutral_in_middle(self):
        st = _make_stocktwits(bull=50, bear=50)  # 0.50 → neutral
        sig = from_stocktwits_sentiment(st)
        assert sig.direction == "neutral"

    def test_strength_zero_at_50_50(self):
        st = _make_stocktwits(bull=50, bear=50)
        sig = from_stocktwits_sentiment(st)
        assert sig.strength == 0.0

    def test_strength_max_at_extreme(self):
        st = _make_stocktwits(bull=100, bear=0)  # ratio=1.0
        sig = from_stocktwits_sentiment(st)
        assert sig.strength == 1.0  # min(1.0, 0.50*3) = 1.0 (capped)

    def test_metadata_propagated(self):
        st = _make_stocktwits(bull=60, bear=40, total=150, trending=True)
        sig = from_stocktwits_sentiment(st)
        assert sig.metadata["trending"] is True
        assert sig.metadata["total_messages"] == 150

    def test_signal_id_format(self):
        st = _make_stocktwits("TSLA")
        sig = from_stocktwits_sentiment(st)
        assert sig.signal_id.startswith("stocktwits:TSLA:")

    def test_asset_class_is_stock(self):
        st = _make_stocktwits()
        sig = from_stocktwits_sentiment(st)
        assert sig.asset_class == "stock"
        assert sig.domain == "sentiment"


class TestFromVIXStructure:
    def test_bearish_in_backwardation(self):
        v = _make_vix(structure_type="backwardation", spot=22.0, term_spread=-3.0)
        sig = from_vix_structure(v)
        assert sig.direction == "bearish"
        assert sig.source == "vix_term_structure"

    def test_bearish_when_vix_over_30(self):
        v = _make_vix(spot=35.0, structure_type="contango", term_spread=2.0)
        sig = from_vix_structure(v)
        assert sig.direction == "bearish"

    def test_bullish_in_low_vix_contango(self):
        v = _make_vix(spot=15.0, structure_type="contango", term_spread=2.0)
        sig = from_vix_structure(v)
        assert sig.direction == "bullish"

    def test_neutral_in_high_contango(self):
        # contango but VIX >= 20 → neutral
        v = _make_vix(spot=22.0, structure_type="contango", term_spread=2.0)
        sig = from_vix_structure(v)
        assert sig.direction == "neutral"

    def test_strength_scales_with_vix_level(self):
        v = _make_vix(spot=40.0)
        sig = from_vix_structure(v)
        assert sig.strength == 1.0  # min(1.0, 40/40)

    def test_strength_low_when_vix_low(self):
        v = _make_vix(spot=16.0)
        sig = from_vix_structure(v)
        assert abs(sig.strength - 0.40) < 0.01  # 16/40

    def test_metadata_propagated(self):
        v = _make_vix()
        sig = from_vix_structure(v)
        assert sig.metadata["vix_spot"] == 18.0
        assert sig.metadata["structure_type"] == "contango"

    def test_symbol_is_empty_macro(self):
        v = _make_vix()
        sig = from_vix_structure(v)
        assert sig.symbol == ""
        assert sig.asset_class == "macro"
        assert sig.outcome_symbol == "SPY"


class TestFromTrendsSignal:
    def test_bullish_on_rising_ticker(self):
        t = _make_trends(keyword="NVDA", delta=30.0, score=70)
        sig = from_trends_signal(t)
        assert sig.direction == "bullish"

    def test_bearish_on_falling_ticker(self):
        t = _make_trends(keyword="SPY", delta=-25.0, score=40)
        sig = from_trends_signal(t)
        assert sig.direction == "bearish"

    def test_bearish_on_fear_keyword_high_score(self):
        t = _make_trends(keyword="recession", score=70, delta=5.0)
        sig = from_trends_signal(t)
        assert sig.direction == "bearish"

    def test_neutral_on_fear_keyword_low_score(self):
        t = _make_trends(keyword="recession", score=40, delta=5.0)
        sig = from_trends_signal(t)
        assert sig.direction == "neutral"

    def test_neutral_in_small_delta_range(self):
        t = _make_trends(keyword="AAPL", delta=5.0, score=50)
        sig = from_trends_signal(t)
        assert sig.direction == "neutral"

    def test_ticker_keyword_becomes_symbol(self):
        t = _make_trends(keyword="NVDA")
        sig = from_trends_signal(t)
        assert sig.symbol == "NVDA"
        assert sig.asset_class == "stock"

    def test_macro_keyword_has_empty_symbol(self):
        t = _make_trends(keyword="recession")
        sig = from_trends_signal(t)
        assert sig.symbol == ""
        assert sig.asset_class == "macro"

    def test_strength_from_interest_score(self):
        t = _make_trends(score=80)
        sig = from_trends_signal(t)
        assert abs(sig.strength - 0.80) < 0.01

    def test_metadata_propagated(self):
        t = _make_trends(keyword="TSLA", score=90, delta=40.0, is_breakout=True)
        sig = from_trends_signal(t)
        assert sig.metadata["is_breakout"] is True
        assert sig.metadata["interest_score"] == 90


class TestFromEconomicEvent:
    def test_bullish_on_positive_surprise(self):
        # actual 10% above estimate → bullish
        evt = _make_economic_event(actual=3.3, estimate=3.0)
        sig = from_economic_event(evt)
        assert sig.direction == "bullish"
        assert sig.source == "finnhub_economic"

    def test_bearish_on_negative_surprise(self):
        # actual significantly below estimate → bearish
        evt = _make_economic_event(actual=2.5, estimate=3.0)
        sig = from_economic_event(evt)
        assert sig.direction == "bearish"

    def test_neutral_when_small_surprise(self):
        # 1% surprise < 5% threshold → neutral
        evt = _make_economic_event(actual=3.03, estimate=3.0)
        sig = from_economic_event(evt)
        assert sig.direction == "neutral"

    def test_neutral_upcoming_high_impact(self):
        evt = _make_economic_event(actual=None, impact="high")
        sig = from_economic_event(evt)
        assert sig.direction == "neutral"
        assert sig.strength == 0.4

    def test_neutral_upcoming_low_impact(self):
        evt = _make_economic_event(actual=None, impact="low")
        sig = from_economic_event(evt)
        assert sig.strength == 0.2

    def test_strength_scales_with_surprise(self):
        # 20% surprise → strength = min(1.0, 0.20*5) = 1.0
        evt = _make_economic_event(actual=3.6, estimate=3.0)
        sig = from_economic_event(evt)
        assert sig.strength == 1.0

    def test_metadata_propagated(self):
        evt = _make_economic_event(actual=3.5, estimate=3.0)
        sig = from_economic_event(evt)
        assert sig.metadata["event"] == "CPI"
        assert sig.metadata["impact"] == "high"
        assert sig.metadata["actual"] == 3.5

    def test_macro_classification(self):
        evt = _make_economic_event()
        sig = from_economic_event(evt)
        assert sig.symbol == ""
        assert sig.asset_class == "macro"
        assert sig.domain == "events"
        assert sig.outcome_symbol == "SPY"


class TestFromAnalystRecommendation:
    def test_bullish_on_strong_buy_consensus(self):
        rec = _make_analyst_rec(strong_buy=25, buy=15, hold=2, sell=0, strong_sell=0)
        sig = from_analyst_recommendation(rec)
        assert sig.direction == "bullish"
        assert sig.source == "finnhub_analyst"

    def test_bearish_on_strong_sell_consensus(self):
        rec = _make_analyst_rec(strong_buy=1, buy=2, hold=3, sell=8, strong_sell=6)
        # buy_ratio = 3/20 = 0.15 < 0.30 → bearish
        sig = from_analyst_recommendation(rec)
        assert sig.direction == "bearish"

    def test_neutral_when_mixed(self):
        rec = _make_analyst_rec(strong_buy=5, buy=5, hold=8, sell=4, strong_sell=3)
        # buy_ratio = 10/25 = 0.40, between 0.30 and 0.80 → neutral
        sig = from_analyst_recommendation(rec)
        assert sig.direction == "neutral"

    def test_strength_zero_at_50_50(self):
        # buy_ratio = 0.50 → deviation = 0 → strength = 0
        rec = _make_analyst_rec(strong_buy=5, buy=5, hold=0, sell=5, strong_sell=5)
        sig = from_analyst_recommendation(rec)
        assert sig.strength == 0.0

    def test_metadata_propagated(self):
        rec = _make_analyst_rec(strong_buy=20, buy=15, hold=5, sell=2, strong_sell=0)
        sig = from_analyst_recommendation(rec)
        assert sig.metadata["strong_buy"] == 20
        assert sig.metadata["buy_ratio"] > 0.80

    def test_signal_id_format(self):
        rec = _make_analyst_rec(symbol="AAPL", period="2026-02-01")
        sig = from_analyst_recommendation(rec)
        assert sig.signal_id == "analyst:AAPL:2026-02-01"

    def test_asset_class_is_stock(self):
        rec = _make_analyst_rec()
        sig = from_analyst_recommendation(rec)
        assert sig.asset_class == "stock"
        assert sig.domain == "fundamentals"

    def test_outcome_window_30_days(self):
        rec = _make_analyst_rec()
        sig = from_analyst_recommendation(rec)
        assert sig.outcome_window_days == 30

    def test_all_signal_fields_valid(self):
        rec = _make_analyst_rec()
        sig = from_analyst_recommendation(rec)
        assert 0.0 <= sig.strength <= 1.0
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.decay_rate > 0
        assert sig.direction in ("bullish", "bearish", "neutral")
        assert isinstance(sig.timestamp, datetime)
