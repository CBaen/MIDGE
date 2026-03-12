"""Tests for RawDataAnalyst — cross-domain insight engine.

Tests cover:
  - Each of the four analysis routines with mock RawStore data
  - Enriched signals are valid MarketSignal objects
  - Cross-domain query logic returns expected results
  - Cadence gate (only runs every 100 steps)
  - Graceful degradation when data is sparse/absent
  - Statistics reporting
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.raw_data_analyst import RawDataAnalyst
from mae_core.market.signal import MarketSignal


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_store(
    insider_trades=None,
    price_snapshots=None,
    fred_obs=None,
    trends=None,
    headlines=None,
    funding=None,
    eia_obs=None,
    vix=None,
    cot=None,
    coingecko=None,
    congressional=None,
    finnhub_earnings=None,
) -> MagicMock:
    """Create a fully-mocked RawStore with configurable return values."""
    store = MagicMock()
    store.get_insider_trades.return_value = insider_trades or []
    store.get_price_snapshots.return_value = price_snapshots or []
    store.get_fred_observations.return_value = fred_obs or []
    store.get_trends_history.return_value = trends or []
    store.get_yahoo_headlines.return_value = headlines or []
    store.get_binance_funding_history.return_value = funding or []
    store.get_eia_observations.return_value = eia_obs or []
    store.get_vix_history.return_value = vix or []
    store.get_cot_history.return_value = cot or []
    store.get_coingecko_history.return_value = coingecko or []
    store.get_congressional_trades.return_value = congressional or []
    store.get_finnhub_earnings.return_value = finnhub_earnings or []
    return store


def _make_insider(ticker: str, n: int = 2, price: float = 50.0) -> list:
    """Build N synthetic insider trade dicts for a ticker."""
    return [
        {
            "ticker": ticker,
            "insider_name": f"Exec{i}",
            "title": "CEO",
            "transaction_date": "2026-03-01",
            "transaction_type": "P",
            "shares": 1000,
            "price_per_share": price,
            "total_value": price * 1000,
            "source": "sec_form4",
        }
        for i in range(n)
    ]


def _make_snapshot(ticker: str, price: float, low: float, high: float) -> dict:
    return {
        "symbol": ticker,
        "timestamp": "2026-03-10T12",
        "price": price,
        "market_cap": 1e9,
        "short_ratio": 2.0,
        "fifty_two_week_low": low,
        "fifty_two_week_high": high,
        "info_json": "{}",
        "ingested_at": "2026-03-10T12:00:00",
    }


def _make_fred_obs(series_id: str, values: list) -> list:
    """Build list of FRED observation dicts with evenly-spaced dates."""
    base = datetime(2025, 12, 1)
    return [
        {"series_id": series_id, "date": (base + timedelta(days=i * 30)).strftime("%Y-%m-%d"), "value": v}
        for i, v in enumerate(values)
    ]


def _make_funding(symbol: str, rates: list) -> list:
    """Build Binance funding rate dicts (oldest-first)."""
    base = datetime(2026, 3, 9)
    return [
        {
            "symbol": symbol,
            "funding_time": (base + timedelta(hours=i * 8)).isoformat(),
            "funding_rate": r,
            "mark_price": 50000.0,
        }
        for i, r in enumerate(rates)
    ]


# ── Cadence gate ─────────────────────────────────────────────────────────────

class TestCadence:
    def test_returns_empty_on_off_step(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        result = analyst.analyze(step=50)
        assert result == []

    def test_runs_on_cadence_step(self):
        """analyze(100) should attempt all routines (returns [] when data empty)."""
        analyst = RawDataAnalyst(raw_store=_make_store())
        result = analyst.analyze(step=100)
        assert isinstance(result, list)

    def test_runs_on_multiple_cadence_steps(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        analyst.analyze(step=100)
        analyst.analyze(step=200)
        assert analyst._run_count == 2

    def test_does_not_run_on_step_99_or_101(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        for step in (1, 99, 101, 199, 201):
            assert analyst.analyze(step) == []

    def test_returns_empty_when_store_is_none(self):
        analyst = RawDataAnalyst(raw_store=None)
        # Bypasses cadence check — store is None guard fires instead
        # But cadence check fires first on off-steps
        result = analyst.analyze(step=100)
        assert result == []


# ── Signal validity ───────────────────────────────────────────────────────────

class TestSignalValidity:
    """All emitted signals must satisfy MarketSignal contract."""

    def _check_signal(self, sig):
        assert isinstance(sig, MarketSignal)
        assert isinstance(sig.signal_id, str) and sig.signal_id
        assert sig.direction in ("bullish", "bearish", "neutral")
        assert 0.0 <= sig.strength <= 1.0
        assert 0.0 <= sig.confidence <= 1.0
        assert isinstance(sig.timestamp, datetime)
        assert isinstance(sig.received_at, datetime)
        assert sig.source != ""
        assert sig.domain != ""

    def test_insider_context_signal_valid(self):
        store = _make_store(
            insider_trades=_make_insider("AAPL", n=2, price=140.0),
            price_snapshots=[_make_snapshot("AAPL", price=142.0, low=135.0, high=200.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert len(sigs) >= 1
        for s in sigs:
            self._check_signal(s)

    def test_fred_macro_signal_valid(self):
        # T10Y2Y inverted + CPI accelerating
        obs = (
            _make_fred_obs("T10Y2Y", [-1.2, -1.3, -1.4, -1.5])
            + _make_fred_obs("CPIAUCSL", [280.0, 281.0, 282.5, 284.5, 287.0, 290.0])
        )
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        for s in sigs:
            self._check_signal(s)

    def test_funding_squeeze_signal_valid(self):
        funding = _make_funding("BTCUSDT", [-0.01, -0.015, -0.02, -0.025])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        for s in sigs:
            self._check_signal(s)

    def test_preconvergence_signal_valid(self):
        insiders = _make_insider("NVDA", n=2)
        trends = [
            {"keyword": "NVDA", "timestamp": "2026-03-08T10:00:00", "interest": 30},
            {"keyword": "NVDA", "timestamp": "2026-03-09T10:00:00", "interest": 55},
            {"keyword": "NVDA", "timestamp": "2026-03-10T10:00:00", "interest": 80},
        ]
        store = _make_store(insider_trades=insiders, trends=trends)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        for s in sigs:
            self._check_signal(s)


# ── Routine 1: Insider Price Context ─────────────────────────────────────────

class TestInsiderPriceContext:
    def test_near_52wk_low_amplifies_signal(self):
        """Insider buying within 10% of 52-wk low should produce strength > base."""
        # Price at $105 — just above low of $100. High is $200. Position = 5%.
        store = _make_store(
            insider_trades=_make_insider("MSFT", n=3, price=105.0),
            price_snapshots=[_make_snapshot("MSFT", price=105.0, low=100.0, high=200.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert len(sigs) == 1
        sig = sigs[0]
        assert sig.symbol == "MSFT"
        assert sig.direction == "bullish"
        assert sig.metadata["price_context"] == "near_52wk_low"
        # Amplified — should be > base (0.4 + 0.1*3 = 0.7, * 1.35 = 0.945)
        assert sig.strength > 0.7

    def test_near_52wk_high_dampens_signal(self):
        """Insider buying near 52-wk high should have dampened strength."""
        # Price at $198 — near high of $200. Position = 96%.
        store = _make_store(
            insider_trades=_make_insider("TSLA", n=2, price=198.0),
            price_snapshots=[_make_snapshot("TSLA", price=198.0, low=100.0, high=200.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert len(sigs) == 1
        assert sigs[0].metadata["price_context"] == "near_52wk_high"
        # Dampened by 0.75 multiplier vs "near low"
        assert sigs[0].strength < 0.8

    def test_mid_range_no_amplification(self):
        """Mid-range buying produces standard strength without amplification."""
        store = _make_store(
            insider_trades=_make_insider("AMZN", n=2, price=150.0),
            price_snapshots=[_make_snapshot("AMZN", price=150.0, low=100.0, high=200.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert len(sigs) == 1
        assert sigs[0].metadata["price_context"] == "mid_range"

    def test_skips_ticker_without_snapshot(self):
        """If no price snapshot, ticker should be skipped silently."""
        store = _make_store(
            insider_trades=_make_insider("XYZ", n=3),
            price_snapshots=[],  # No snapshot
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert sigs == []

    def test_skips_ticker_below_min_trades(self):
        """Tickers with fewer trades than min_insider are skipped."""
        store = _make_store(
            insider_trades=_make_insider("LOW", n=1),  # Only 1 trade
            price_snapshots=[_make_snapshot("LOW", price=200.0, low=100.0, high=210.0)],
        )
        analyst = RawDataAnalyst(raw_store=store, min_insider_trades=2)
        sigs = analyst._analyze_insider_price_context()
        assert sigs == []

    def test_skips_invalid_price_range(self):
        """52-wk high <= low → skip (can't compute position)."""
        store = _make_store(
            insider_trades=_make_insider("BAD", n=3, price=50.0),
            price_snapshots=[_make_snapshot("BAD", price=50.0, low=60.0, high=60.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert sigs == []

    def test_handles_multiple_tickers(self):
        """Multiple tickers each get their own signal."""
        insider_trades = _make_insider("AAA", n=2, price=10.0) + _make_insider("BBB", n=3, price=50.0)
        store = _make_store(
            insider_trades=insider_trades,
            price_snapshots=[
                _make_snapshot("AAA", price=10.0, low=9.0, high=20.0),
                _make_snapshot("BBB", price=50.0, low=40.0, high=60.0),
            ],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        tickers = {s.symbol for s in sigs}
        assert "AAA" in tickers
        assert "BBB" in tickers

    def test_metadata_contains_expected_keys(self):
        store = _make_store(
            insider_trades=_make_insider("GOOG", n=2, price=150.0),
            price_snapshots=[_make_snapshot("GOOG", price=150.0, low=100.0, high=200.0)],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        assert len(sigs) == 1
        meta = sigs[0].metadata
        assert "insider_count" in meta
        assert "price_context" in meta
        assert "wk52_low" in meta
        assert "wk52_high" in meta
        assert "price_position_pct" in meta


# ── Routine 2: FRED Macro Regime ─────────────────────────────────────────────

class TestFredMacroRegime:
    def test_inversion_inflation_pattern_bearish(self):
        """Inverted yield curve + accelerating CPI → bearish macro_warning."""
        obs = (
            _make_fred_obs("T10Y2Y", [-1.0, -1.1, -1.2])  # Inverted
            + _make_fred_obs("CPIAUCSL", [280.0, 282.0, 284.5, 287.5, 291.5, 296.0])  # Accelerating
        )
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        patterns = {s.metadata.get("pattern") for s in sigs}
        assert "inversion_inflation" in patterns
        inv_sigs = [s for s in sigs if s.metadata.get("pattern") == "inversion_inflation"]
        assert all(s.direction == "bearish" for s in inv_sigs)

    def test_no_signal_when_curve_positive(self):
        """Positive yield spread should not trigger inversion_inflation."""
        obs = (
            _make_fred_obs("T10Y2Y", [0.5, 0.6, 0.7])   # Not inverted
            + _make_fred_obs("CPIAUCSL", [280.0, 282.0, 284.5, 287.5, 291.5, 296.0])
        )
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        patterns = {s.metadata.get("pattern") for s in sigs}
        assert "inversion_inflation" not in patterns

    def test_recession_risk_pattern(self):
        """High fed funds rate + rising unemployment → recession_risk."""
        obs = (
            _make_fred_obs("DFF", [5.0, 5.0, 5.25, 5.25, 5.5, 5.5])
            + _make_fred_obs("UNRATE", [3.5, 3.6, 3.8, 4.0, 4.3, 4.7])  # Rising fast
        )
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        patterns = {s.metadata.get("pattern") for s in sigs}
        assert "recession_risk" in patterns
        rec_sigs = [s for s in sigs if s.metadata.get("pattern") == "recession_risk"]
        assert all(s.direction == "bearish" for s in rec_sigs)

    def test_no_recession_risk_when_low_rates(self):
        """Low fed funds rate should not trigger recession_risk."""
        obs = (
            _make_fred_obs("DFF", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            + _make_fred_obs("UNRATE", [3.5, 3.6, 3.8, 4.0, 4.3, 4.7])
        )
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        patterns = {s.metadata.get("pattern") for s in sigs}
        assert "recession_risk" not in patterns

    def test_yield_steepening_pattern_bullish(self):
        """Curve steepening from deep inversion → bullish yield_steepening signal."""
        # Was deep inversion 10+ periods ago, now recovering
        vals = [-1.5, -1.4, -1.2, -1.0, -0.8, -0.5, -0.3, -0.2, -0.1, 0.0, -0.1]
        obs = _make_fred_obs("T10Y3M", vals)
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        patterns = {s.metadata.get("pattern") for s in sigs}
        assert "yield_steepening" in patterns
        steep_sigs = [s for s in sigs if s.metadata.get("pattern") == "yield_steepening"]
        assert all(s.direction == "bullish" for s in steep_sigs)

    def test_returns_empty_when_no_fred_data(self):
        store = _make_store(fred_obs=[])
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        assert sigs == []

    def test_fred_signal_has_macro_domain(self):
        """All FRED signals should use the 'macro' domain."""
        obs = _make_fred_obs("DFF", [5.5, 5.5, 5.5, 5.5, 5.5, 5.5]) + \
              _make_fred_obs("UNRATE", [3.5, 3.8, 4.0, 4.3, 4.6, 5.0])
        store = _make_store(fred_obs=obs)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_fred_macro_regime()
        for s in sigs:
            assert s.domain == "macro"
            assert s.symbol == ""   # Macro signals have no specific ticker
            assert s.asset_class == "macro"


# ── Routine 3: Cross-Domain Pre-Convergence ───────────────────────────────────

class TestCrossDomainPreconvergence:
    def _make_trends_rising(self, keyword: str) -> list:
        """Build rising trends data for a keyword."""
        base = datetime(2026, 3, 5)
        return [
            {"keyword": keyword, "timestamp": (base + timedelta(hours=i)).isoformat(), "interest": 20 + i * 5}
            for i in range(10)
        ]

    def test_two_domains_triggers_signal(self):
        """Insider + rising Trends (2 domains) should trigger pre-convergence."""
        store = _make_store(
            insider_trades=_make_insider("META", n=2),
            trends=self._make_trends_rising("META"),
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert len(sigs) >= 1
        assert sigs[0].symbol == "META"
        assert sigs[0].metadata["has_insider"] is True
        assert sigs[0].metadata["has_trends"] is True

    def test_three_domains_higher_strength(self):
        """Insider + Trends + Headlines (3 domains) → higher strength than 2 domains."""
        insider_trades = _make_insider("NVDA", n=2)
        trends = self._make_trends_rising("NVDA")
        headlines = [
            {"ticker": "NVDA", "title": "NVDA beats earnings record", "published_at": "", "summary": ""},
        ]
        store = _make_store(insider_trades=insider_trades, trends=trends, headlines=headlines)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert len(sigs) >= 1
        sig = sigs[0]
        assert sig.metadata["domains_hit"] == 3

    def test_insider_only_no_signal(self):
        """Only insider trades (1 domain) should NOT trigger pre-convergence."""
        store = _make_store(insider_trades=_make_insider("XOM", n=3))
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert sigs == []

    def test_no_insiders_no_signal(self):
        """Without insider data, no pre-convergence possible."""
        store = _make_store(
            insider_trades=[],
            trends=self._make_trends_rising("SPY"),
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert sigs == []

    def test_flat_trends_not_counted(self):
        """Flat Google Trends (not rising) should not count as a domain."""
        flat_trends = [
            {"keyword": "AAPL", "timestamp": f"2026-03-0{i}T10:00", "interest": 50}
            for i in range(1, 8)
        ]
        store = _make_store(
            insider_trades=_make_insider("AAPL", n=2),
            trends=flat_trends,
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        # Only 1 domain (insider) → no pre-convergence
        assert all(s.metadata["has_trends"] is False for s in sigs) or sigs == []

    def test_positive_headline_keywords_detected(self):
        """Headlines containing bullish keywords count as the headline domain."""
        positive_headlines = [
            {"ticker": "PLTR", "title": "PLTR surges on record contract win", "published_at": "", "summary": ""},
        ]
        trends = self._make_trends_rising("PLTR")
        store = _make_store(
            insider_trades=_make_insider("PLTR", n=2),
            trends=trends,
            headlines=positive_headlines,
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert any(s.metadata.get("has_headlines") for s in sigs)

    def test_negative_headline_keywords_not_counted(self):
        """Headlines without bullish keywords should not count as a domain."""
        neutral_headlines = [
            {"ticker": "F", "title": "Ford announces quarterly report", "published_at": "", "summary": ""},
        ]
        store = _make_store(
            insider_trades=_make_insider("F", n=2),
            headlines=neutral_headlines,
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        # Only 1 domain (insider alone) → no signal
        assert sigs == []

    def test_preconvergence_source_name(self):
        """Pre-convergence signals should have the correct source name."""
        store = _make_store(
            insider_trades=_make_insider("AMD", n=2),
            trends=self._make_trends_rising("AMD"),
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_cross_domain_preconvergence()
        assert all(s.source == "raw_preconvergence" for s in sigs)


# ── Routine 4: Funding Rate Squeeze ──────────────────────────────────────────

class TestFundingRateSqueeze:
    def test_three_consecutive_negative_triggers_signal(self):
        """3+ consecutive negative funding periods → short squeeze signal."""
        funding = _make_funding("BTCUSDT", [-0.01, -0.015, -0.02])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        assert len(sigs) >= 1
        assert sigs[0].direction == "bullish"
        assert sigs[0].metadata["pattern"] == "short_squeeze_precursor"

    def test_mixed_funding_no_signal(self):
        """Mixed positive/negative funding should not trigger squeeze."""
        funding = _make_funding("ETHUSDT", [0.01, -0.005, 0.008, -0.01])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        # ETHUSDT has mixed — no consecutive negatives at end
        eth_sigs = [s for s in sigs if "ETH" in s.symbol]
        assert eth_sigs == []

    def test_all_positive_no_signal(self):
        """All positive funding = longs dominant = no squeeze signal."""
        funding = _make_funding("SOLUSDT", [0.005, 0.010, 0.008, 0.012])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        sol_sigs = [s for s in sigs if "SOL" in s.symbol]
        assert sol_sigs == []

    def test_squeeze_signal_asset_class_crypto(self):
        """Squeeze signals should have asset_class='crypto'."""
        funding = _make_funding("BTCUSDT", [-0.01, -0.02, -0.03])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        btc_sigs = [s for s in sigs if "BTC" in s.symbol]
        assert all(s.asset_class == "crypto" for s in btc_sigs)

    def test_squeeze_signal_domain_crypto(self):
        """Squeeze signals should use domain='crypto'."""
        funding = _make_funding("BTCUSDT", [-0.01, -0.02, -0.03, -0.04])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        assert all(s.domain == "crypto" for s in sigs)

    def test_consecutive_count_in_metadata(self):
        """Consecutive negative period count should be in signal metadata."""
        funding = _make_funding("BTCUSDT", [-0.01, -0.02, -0.03, -0.04, -0.05])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        btc_sigs = [s for s in sigs if "BTC" in s.symbol]
        assert len(btc_sigs) >= 1
        assert btc_sigs[0].metadata["consecutive_negative_periods"] == 5

    def test_insufficient_data_no_signal(self):
        """Less than _NEGATIVE_FUNDING_MIN records → no signal."""
        funding = _make_funding("BNBUSDT", [-0.01, -0.02])  # Only 2 records
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        bnb_sigs = [s for s in sigs if "BNB" in s.symbol]
        assert bnb_sigs == []

    def test_empty_funding_data_no_error(self):
        """Empty funding history should silently produce no signals."""
        store = _make_store(funding=[])
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_funding_rate_squeeze()
        assert sigs == []


# ── Asset class inference ─────────────────────────────────────────────────────

class TestAssetClassInference:
    def test_empty_symbol_is_macro(self):
        assert RawDataAnalyst._infer_asset_class("") == "macro"

    def test_crypto_usd_suffix(self):
        assert RawDataAnalyst._infer_asset_class("BTC-USD") == "crypto"

    def test_known_crypto_symbols(self):
        for sym in ("BTC", "ETH", "SOL", "BNB", "XRP"):
            assert RawDataAnalyst._infer_asset_class(sym) == "crypto"

    def test_futures_suffix(self):
        assert RawDataAnalyst._infer_asset_class("CL=F") == "futures"
        assert RawDataAnalyst._infer_asset_class("NQ=F") == "futures"

    def test_known_futures(self):
        for sym in ("GC", "CL", "NQ", "ES"):
            assert RawDataAnalyst._infer_asset_class(sym) == "futures"

    def test_forex(self):
        assert RawDataAnalyst._infer_asset_class("EURUSD=X") == "forex"

    def test_stock_default(self):
        assert RawDataAnalyst._infer_asset_class("AAPL") == "stock"
        assert RawDataAnalyst._infer_asset_class("MSFT") == "stock"


# ── Statistics ────────────────────────────────────────────────────────────────

class TestStatistics:
    def test_initial_stats(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        stats = analyst.get_statistics()
        assert stats["run_count"] == 0
        assert stats["signals_emitted"] == 0
        assert stats["last_run_at"] is None

    def test_stats_after_run(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        analyst.analyze(step=100)
        stats = analyst.get_statistics()
        assert stats["run_count"] == 1
        assert stats["last_run_at"] is not None

    def test_signals_emitted_count(self):
        funding = _make_funding("BTCUSDT", [-0.01, -0.02, -0.03])
        store = _make_store(funding=funding)
        analyst = RawDataAnalyst(raw_store=store)
        analyst.analyze(step=100)
        stats = analyst.get_statistics()
        # Should have emitted at least 1 funding signal
        assert stats["signals_emitted"] >= 1

    def test_run_count_increments_each_cadence(self):
        analyst = RawDataAnalyst(raw_store=_make_store())
        analyst.analyze(100)
        analyst.analyze(200)
        analyst.analyze(300)
        assert analyst._run_count == 3


# ── Error resilience ─────────────────────────────────────────────────────────

class TestErrorResilience:
    def test_store_exception_in_insider_handled_gracefully(self):
        """If get_insider_trades raises, analyze() should return empty list (not crash)."""
        store = _make_store()
        store.get_insider_trades.side_effect = RuntimeError("DB locked")
        store.get_fred_observations.side_effect = RuntimeError("DB locked")
        store.get_trends_history.side_effect = RuntimeError("DB locked")
        store.get_binance_funding_history.side_effect = RuntimeError("DB locked")
        analyst = RawDataAnalyst(raw_store=store)
        result = analyst.analyze(step=100)
        # Should not raise — all errors caught internally
        assert isinstance(result, list)

    def test_store_exception_partial_failure_continues(self):
        """If one routine fails, others should still run."""
        funding = _make_funding("BTCUSDT", [-0.01, -0.02, -0.03])
        store = _make_store(funding=funding)
        store.get_insider_trades.side_effect = RuntimeError("Table missing")
        # FRED and trends also fail, but funding should work
        store.get_fred_observations.side_effect = RuntimeError("Table missing")
        store.get_trends_history.side_effect = RuntimeError("Table missing")
        analyst = RawDataAnalyst(raw_store=store)
        result = analyst.analyze(step=100)
        # Funding routine should still have run
        assert isinstance(result, list)

    def test_none_values_in_price_snapshot_handled(self):
        """Snapshot with None fields should be skipped without error."""
        bad_snap = {
            "symbol": "XYZ", "timestamp": "2026-03-10T12",
            "price": None, "market_cap": None, "short_ratio": None,
            "fifty_two_week_low": None, "fifty_two_week_high": None,
            "info_json": "{}", "ingested_at": "2026-03-10T12:00:00",
        }
        store = _make_store(
            insider_trades=_make_insider("XYZ", n=2),
            price_snapshots=[bad_snap],
        )
        analyst = RawDataAnalyst(raw_store=store)
        sigs = analyst._analyze_insider_price_context()
        # Should skip gracefully
        assert sigs == []
