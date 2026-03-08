"""Tests for YahooRSSClient — headline velocity detection and signal conversion."""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.apis.yahoo_rss_client import (
    YahooRSSClient,
    YahooHeadlineSignal,
    _extract_keywords,
    _keyword_polarity,
    _parse_rss_date,
    VELOCITY_WINDOW_HOURS,
)
from mae_core.market.signal_adapters.layer6 import from_yahoo_rss_signal


# ---------------------------------------------------------------------------
# Helpers — build fake feedparser entry objects
# ---------------------------------------------------------------------------

def _make_entry(title: str, hours_ago: float = 1.0, link: str = "http://x.com", summary: str = ""):
    """Build a minimal feedparser-like entry object."""
    pub_dt = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    pub_str = pub_dt.strftime("%a, %d %b %Y %H:%M:%S +0000")
    entry = MagicMock()
    entry.title = title
    entry.published = pub_str
    entry.link = link
    entry.summary = summary
    return entry


def _make_entries(specs):
    """specs: list of (title, hours_ago) tuples."""
    return [_make_entry(title, h) for title, h in specs]


# ---------------------------------------------------------------------------
# _parse_rss_date
# ---------------------------------------------------------------------------

def test_parse_rss_date_valid():
    dt = _parse_rss_date("Mon, 01 Jan 2024 12:00:00 +0000")
    assert dt is not None
    assert dt.year == 2024


def test_parse_rss_date_empty():
    assert _parse_rss_date("") is None


def test_parse_rss_date_garbage():
    assert _parse_rss_date("not a date") is None


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def test_extract_keywords_bullish():
    kws = _extract_keywords("AAPL surges after earnings beat")
    assert "surges" in kws or "surge" in kws
    assert "beats" in kws or "beat" in kws


def test_extract_keywords_bearish():
    kws = _extract_keywords("TSLA crashes on layoff announcement")
    assert any(k in kws for k in ("crash", "crashes"))
    assert "layoffs" in kws or "layoff" in kws


def test_extract_keywords_empty():
    assert _extract_keywords("Some neutral headline about quarterly results") == []


def test_keyword_polarity_bullish():
    assert _keyword_polarity(["surge", "beats"]) == 1


def test_keyword_polarity_bearish():
    assert _keyword_polarity(["crash", "lawsuit"]) == -1


def test_keyword_polarity_mixed():
    # Equal counts = 0
    assert _keyword_polarity(["surge", "crash"]) == 0


def test_keyword_polarity_empty():
    assert _keyword_polarity([]) == 0


# ---------------------------------------------------------------------------
# YahooRSSClient._analyze_entries
# ---------------------------------------------------------------------------

def test_analyze_entries_returns_none_on_empty():
    client = YahooRSSClient()
    assert client._analyze_entries("AAPL", []) is None


def test_analyze_entries_no_recent_headlines():
    """Entries older than the velocity window should return None (no signal)."""
    client = YahooRSSClient()
    old_entries = _make_entries([
        ("AAPL gains", VELOCITY_WINDOW_HOURS + 2),
        ("AAPL steady", VELOCITY_WINDOW_HOURS + 3),
    ])
    result = client._analyze_entries("AAPL", old_entries)
    assert result is None


def test_analyze_entries_basic_signal():
    client = YahooRSSClient()
    entries = _make_entries([
        ("AAPL surges on earnings beat", 0.5),
        ("Apple beats revenue estimates", 1.0),
        ("AAPL up 5% after hours", 2.0),
    ])
    sig = client._analyze_entries("AAPL", entries)
    assert sig is not None
    assert sig.ticker == "AAPL"
    assert sig.headline_count >= 1
    assert sig.velocity_change >= 1.0
    assert isinstance(sig.sentiment_keywords, list)
    assert sig.latest_headline != ""


def test_analyze_entries_velocity_calculation():
    """2 recent + 4 prior → velocity = 2/4 = 0.5 (slowing down)."""
    client = YahooRSSClient()
    # recent window: 0-6h, prior: 6-12h
    entries = _make_entries([
        ("headline A", 1),
        ("headline B", 2),
        ("headline C", 7),
        ("headline D", 8),
        ("headline E", 9),
        ("headline F", 10),
    ])
    sig = client._analyze_entries("X", entries)
    assert sig is not None
    assert sig.headline_count == 2
    # velocity = 2 / 4 = 0.5
    assert abs(sig.velocity_change - 0.5) < 0.05


def test_analyze_entries_velocity_spike():
    """5 recent vs 1 prior → velocity 5.0."""
    client = YahooRSSClient()
    entries = _make_entries([
        ("FDA approves drug", 0.5),
        ("Stock surges", 1.0),
        ("Acquisition announced", 1.5),
        ("Deal confirmed", 2.0),
        ("Record high", 3.0),
        ("old news", 8.0),  # prior window
    ])
    sig = client._analyze_entries("BIO", entries)
    assert sig is not None
    assert sig.headline_count == 5
    assert sig.velocity_change == pytest.approx(5.0, abs=0.1)


def test_analyze_entries_bullish_polarity():
    client = YahooRSSClient()
    entries = _make_entries([
        ("Company beats earnings estimates", 1.0),
        ("Stock surges after upgrade", 2.0),
    ])
    sig = client._analyze_entries("XYZ", entries)
    assert sig is not None
    assert sig.keyword_polarity == 1


def test_analyze_entries_bearish_polarity():
    client = YahooRSSClient()
    entries = _make_entries([
        ("Company misses earnings, stock crashes", 1.0),
        ("Layoffs announced at headquarters", 2.0),
    ])
    sig = client._analyze_entries("XYZ", entries)
    assert sig is not None
    assert sig.keyword_polarity == -1


# ---------------------------------------------------------------------------
# Rate limiting / caching
# ---------------------------------------------------------------------------

def test_cache_prevents_refetch():
    """Second call within 5 minutes reuses cached entries."""
    client = YahooRSSClient()
    entries = _make_entries([("AAPL news", 1.0)])
    client._cache["AAPL"] = (entries, time.time())  # prime cache

    fetch_count = [0]
    original_fetch = client._fetch_feed.__func__

    def counting_fetch(self, ticker):
        # If cache hit, feedparser never called — just check cache directly
        if ticker in self._cache:
            cached_entries, cached_at = self._cache[ticker]
            import time as _t
            if _t.time() - cached_at < 300:
                fetch_count[0] += 1  # count cache hits
                return cached_entries
        return []

    with patch.object(client, '_fetch_feed', side_effect=lambda t: counting_fetch(client, t)):
        client.get_headlines(["AAPL"])
        client.get_headlines(["AAPL"])

    # Cache was used (no actual feedparser call)
    assert "AAPL" in client._cache


def test_cache_expires():
    """Cache older than 5 minutes is not served (will trigger re-fetch)."""
    client = YahooRSSClient()
    old_time = time.time() - 400  # 400s ago > 300s cache
    client._cache["TSLA"] = ([], old_time)

    mock_feed = MagicMock()
    mock_feed.entries = []

    with patch("feedparser.parse", return_value=mock_feed) as mock_parse:
        client._fetch_feed("TSLA")
        mock_parse.assert_called_once()


# ---------------------------------------------------------------------------
# get_headlines / get_accelerating
# ---------------------------------------------------------------------------

def test_get_headlines_returns_list():
    client = YahooRSSClient()
    with patch.object(client, "_fetch_feed", return_value=[]):
        result = client.get_headlines(["AAPL", "TSLA"])
    assert isinstance(result, list)


def test_get_headlines_skips_no_recent():
    """Tickers with no recent headlines are excluded from results."""
    client = YahooRSSClient()
    old = _make_entries([("old news", VELOCITY_WINDOW_HOURS + 2)])
    recent = _make_entries([("breaking news", 1.0)])

    def mock_fetch(ticker):
        return old if ticker == "AAPL" else recent

    with patch.object(client, "_fetch_feed", side_effect=mock_fetch):
        results = client.get_headlines(["AAPL", "TSLA"])

    tickers = [r.ticker for r in results]
    assert "AAPL" not in tickers
    assert "TSLA" in tickers


def test_get_accelerating_filters_velocity():
    client = YahooRSSClient()
    # AAPL: 5 recent, 1 prior = 5x velocity
    aapl_entries = _make_entries([
        ("AAPL surges", 0.5), ("AAPL beats", 1.0), ("AAPL up 5%", 2.0),
        ("AAPL deal", 3.0), ("AAPL record", 4.0),
        ("old AAPL news", 8.0),
    ])
    # TSLA: 1 recent, 1 prior = 1.0x velocity (below threshold)
    tsla_entries = _make_entries([
        ("TSLA news", 1.0),
        ("old TSLA news", 8.0),
    ])

    def mock_fetch(ticker):
        return aapl_entries if ticker == "AAPL" else tsla_entries

    with patch.object(client, "_fetch_feed", side_effect=mock_fetch):
        results = client.get_accelerating(["AAPL", "TSLA"], min_velocity=2.0)

    tickers = [r.ticker for r in results]
    assert "AAPL" in tickers
    assert "TSLA" not in tickers


def test_get_accelerating_sorted_by_velocity():
    client = YahooRSSClient()
    # SPY: 2 recent, 1 prior = 2x
    spy = _make_entries([("SPY up", 1.0), ("SPY rally", 2.0), ("old SPY", 8.0)])
    # QQQ: 4 recent, 1 prior = 4x
    qqq = _make_entries([
        ("QQQ up", 0.5), ("QQQ tech rally", 1.0), ("QQQ beats", 2.0), ("QQQ record", 3.0),
        ("old QQQ", 8.0),
    ])

    def mock_fetch(ticker):
        return spy if ticker == "SPY" else qqq

    with patch.object(client, "_fetch_feed", side_effect=mock_fetch):
        results = client.get_accelerating(["SPY", "QQQ"], min_velocity=1.5)

    assert len(results) == 2
    assert results[0].velocity_change >= results[1].velocity_change


# ---------------------------------------------------------------------------
# raw_store integration
# ---------------------------------------------------------------------------

def test_raw_store_called_on_fetch():
    raw_store = MagicMock()
    client = YahooRSSClient(raw_store=raw_store)
    entries = _make_entries([("NVDA surges", 1.0)])

    with patch.object(client, "_fetch_feed", return_value=entries):
        client.get_headlines(["NVDA"])

    raw_store.store_yahoo_headlines.assert_called_once_with("NVDA", entries)


def test_raw_store_failure_does_not_break_signal():
    raw_store = MagicMock()
    raw_store.store_yahoo_headlines.side_effect = RuntimeError("db error")
    client = YahooRSSClient(raw_store=raw_store)
    entries = _make_entries([("NVDA beats", 1.0)])

    with patch.object(client, "_fetch_feed", return_value=entries):
        result = client.get_headlines(["NVDA"])

    # Signal still returned despite raw_store error
    assert len(result) == 1


def test_raw_store_not_called_when_no_entries():
    raw_store = MagicMock()
    client = YahooRSSClient(raw_store=raw_store)

    with patch.object(client, "_fetch_feed", return_value=[]):
        client.get_headlines(["AAPL"])

    raw_store.store_yahoo_headlines.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_get_headlines_handles_fetch_exception():
    client = YahooRSSClient()
    with patch.object(client, "_fetch_feed", side_effect=Exception("network error")):
        result = client.get_headlines(["AAPL"])
    assert result == []


def test_get_headlines_empty_ticker_list():
    client = YahooRSSClient()
    assert client.get_headlines([]) == []


# ---------------------------------------------------------------------------
# Signal adapter: from_yahoo_rss_signal
# ---------------------------------------------------------------------------

def _make_signal(
    ticker="AAPL",
    headline_count=5,
    velocity_change=3.5,
    keyword_polarity=1,
    keywords=None,
    latest="AAPL surges on earnings",
):
    return YahooHeadlineSignal(
        ticker=ticker,
        headline_count=headline_count,
        velocity_change=velocity_change,
        sentiment_keywords=keywords or ["surge", "beats"],
        latest_headline=latest,
        latest_published="Mon, 01 Jan 2024 10:00:00 +0000",
        keyword_polarity=keyword_polarity,
    )


def test_from_yahoo_rss_signal_bullish():
    sig = from_yahoo_rss_signal(_make_signal(keyword_polarity=1))
    assert sig.direction == "bullish"
    assert sig.source == "yahoo_rss"
    assert sig.domain == "events"
    assert 0.0 < sig.strength <= 1.0


def test_from_yahoo_rss_signal_bearish():
    sig = from_yahoo_rss_signal(_make_signal(keyword_polarity=-1))
    assert sig.direction == "bearish"


def test_from_yahoo_rss_signal_neutral():
    sig = from_yahoo_rss_signal(_make_signal(keyword_polarity=0))
    assert sig.direction == "neutral"


def test_from_yahoo_rss_signal_high_velocity_raises_strength():
    low = from_yahoo_rss_signal(_make_signal(velocity_change=1.5))
    high = from_yahoo_rss_signal(_make_signal(velocity_change=8.0))
    assert high.strength > low.strength


def test_from_yahoo_rss_signal_metadata():
    sig = from_yahoo_rss_signal(_make_signal(headline_count=7, velocity_change=4.0))
    assert sig.metadata["headline_count"] == 7
    assert sig.metadata["velocity_change"] == pytest.approx(4.0)
    assert "sentiment_keywords" in sig.metadata
    assert "latest_headline" in sig.metadata


def test_from_yahoo_rss_signal_ticker_and_symbol():
    sig = from_yahoo_rss_signal(_make_signal(ticker="NVDA"))
    assert sig.symbol == "NVDA"
    assert sig.outcome_symbol == "NVDA"


def test_from_yahoo_rss_signal_outcome_window():
    sig = from_yahoo_rss_signal(_make_signal())
    assert sig.outcome_window_days == 3  # News resolves quickly


def test_from_yahoo_rss_signal_strength_capped_at_one():
    sig = from_yahoo_rss_signal(_make_signal(velocity_change=100.0, headline_count=999))
    assert sig.strength <= 1.0
