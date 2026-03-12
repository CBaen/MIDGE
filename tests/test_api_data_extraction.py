"""Tests for extended field extraction from API clients.

Covers all 5 top data-waste offenders that previously discarded high-value
fields. Each class verifies that the dataclass carries the new fields AND
that the raw_store persists them correctly.

1. PriceData — yfinance extended fundamentals
2. StockTwitsSentiment — engagement/influence metrics
3. CryptoPrice — 24h range, ATH distance, supply
4. InsiderTrade / DerivativeTransaction — Form 4 Table II derivative parsing
5. LegislativeIndicator — sponsors, committees, subjects enrichment
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import fields
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ===========================================================================
# 1. PriceData — yfinance extended fundamentals
# ===========================================================================


class TestPriceDataExtendedFields:
    """PriceData dataclass must carry all 12 extended fundamental fields."""

    EXPECTED_OPTIONAL_FIELDS = [
        "short_ratio",
        "held_pct_insiders",
        "held_pct_institutions",
        "beta",
        "forward_pe",
        "sector",
        "industry",
        "fifty_two_week_high",
        "fifty_two_week_low",
        "shares_short",
        "target_mean_price",
        "recommendation_mean",
    ]

    def test_price_data_has_extended_fields(self):
        """All 12 extended fields exist on PriceData with None defaults."""
        from mae_core.market.apis.price_fetcher import PriceData

        field_names = {f.name for f in fields(PriceData)}
        for name in self.EXPECTED_OPTIONAL_FIELDS:
            assert name in field_names, f"PriceData missing field: {name}"

    def test_price_data_extended_fields_default_to_none(self):
        """Extended fields default to None so Alpha Vantage results aren't broken."""
        from mae_core.market.apis.price_fetcher import PriceData

        pd = PriceData(symbol="AAPL", price=150.0, timestamp="2026-03-11T12:00:00", source="test")
        for name in self.EXPECTED_OPTIONAL_FIELDS:
            assert getattr(pd, name) is None, f"Expected {name} to default to None"

    def test_price_data_extended_fields_accept_values(self):
        """Extended fields accept real values when provided."""
        from mae_core.market.apis.price_fetcher import PriceData

        pd = PriceData(
            symbol="AAPL",
            price=150.0,
            timestamp="2026-03-11T12:00:00",
            source="yfinance",
            short_ratio=1.5,
            held_pct_insiders=0.02,
            held_pct_institutions=0.75,
            beta=1.1,
            forward_pe=28.0,
            sector="Technology",
            industry="Consumer Electronics",
            fifty_two_week_high=195.0,
            fifty_two_week_low=124.0,
            shares_short=80_000_000,
            target_mean_price=185.0,
            recommendation_mean=2.1,
        )
        assert pd.short_ratio == 1.5
        assert pd.held_pct_insiders == 0.02
        assert pd.held_pct_institutions == 0.75
        assert pd.beta == 1.1
        assert pd.forward_pe == 28.0
        assert pd.sector == "Technology"
        assert pd.industry == "Consumer Electronics"
        assert pd.fifty_two_week_high == 195.0
        assert pd.fifty_two_week_low == 124.0
        assert pd.shares_short == 80_000_000
        assert pd.target_mean_price == 185.0
        assert pd.recommendation_mean == 2.1

    def test_fetch_yfinance_populates_extended_fields(self):
        """_fetch_yfinance() extracts extended fields from ticker.info dict."""
        from mae_core.market.apis.price_fetcher import PriceFetcher, YFINANCE_AVAILABLE

        if not YFINANCE_AVAILABLE:
            pytest.skip("yfinance not installed")

        fake_info = {
            "currentPrice": 150.0,
            "open": 149.0,
            "dayHigh": 151.0,
            "dayLow": 148.0,
            "volume": 50_000_000,
            "regularMarketChangePercent": 0.5,
            "shortRatio": 1.8,
            "heldPercentInsiders": 0.03,
            "heldPercentInstitutions": 0.72,
            "beta": 1.2,
            "forwardPE": 29.5,
            "sector": "Technology",
            "industry": "Semiconductors",
            "fiftyTwoWeekHigh": 200.0,
            "fiftyTwoWeekLow": 120.0,
            "sharesShort": 75_000_000,
            "targetMeanPrice": 190.0,
            "recommendationMean": 1.9,
        }

        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.info = fake_info
            mock_ticker_cls.return_value = mock_ticker

            fetcher = PriceFetcher()
            result = fetcher._fetch_yfinance("NVDA")

        assert result is not None
        assert result.short_ratio == 1.8
        assert result.held_pct_insiders == 0.03
        assert result.held_pct_institutions == 0.72
        assert result.beta == 1.2
        assert result.forward_pe == 29.5
        assert result.sector == "Technology"
        assert result.industry == "Semiconductors"
        assert result.fifty_two_week_high == 200.0
        assert result.fifty_two_week_low == 120.0
        assert result.shares_short == 75_000_000
        assert result.target_mean_price == 190.0
        assert result.recommendation_mean == 1.9

    def test_get_multiple_prices_populates_extended_fields(self):
        """get_multiple_prices() populates extended fields for each symbol."""
        from mae_core.market.apis.price_fetcher import PriceFetcher, YFINANCE_AVAILABLE

        if not YFINANCE_AVAILABLE:
            pytest.skip("yfinance not installed")

        fake_info = {
            "currentPrice": 200.0,
            "open": 198.0,
            "dayHigh": 202.0,
            "dayLow": 197.0,
            "volume": 10_000_000,
            "regularMarketChangePercent": 1.0,
            "shortRatio": 2.5,
            "sector": "Healthcare",
            "beta": 0.8,
            "fiftyTwoWeekHigh": 250.0,
            "fiftyTwoWeekLow": 150.0,
        }

        mock_ticker = MagicMock()
        mock_ticker.info = fake_info

        with patch("yfinance.Tickers") as mock_tickers_cls:
            mock_tickers_inst = MagicMock()
            mock_tickers_inst.tickers = {"JNJ": mock_ticker}
            mock_tickers_cls.return_value = mock_tickers_inst

            fetcher = PriceFetcher()
            results = fetcher.get_multiple_prices(["JNJ"])

        assert "JNJ" in results
        pd = results["JNJ"]
        assert pd is not None
        assert pd.short_ratio == 2.5
        assert pd.sector == "Healthcare"
        assert pd.beta == 0.8
        assert pd.fifty_two_week_high == 250.0
        assert pd.fifty_two_week_low == 150.0

    def test_alpha_vantage_result_has_none_for_extended_fields(self):
        """Alpha Vantage fallback results have None for all extended fields."""
        from mae_core.market.apis.price_fetcher import PriceFetcher

        fake_av_response = {
            "Global Quote": {
                "05. price": "150.0",
                "02. open": "149.0",
                "03. high": "151.0",
                "04. low": "148.0",
                "06. volume": "5000000",
                "10. change percent": "0.5%",
            }
        }

        with patch("requests.Session.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = fake_av_response
            mock_get.return_value = mock_resp

            fetcher = PriceFetcher(alpha_vantage_key="TESTKEY")
            result = fetcher._fetch_alpha_vantage("AAPL")

        assert result is not None
        assert result.sector is None
        assert result.beta is None
        assert result.short_ratio is None


# ===========================================================================
# 2. StockTwitsSentiment — engagement / influence metrics
# ===========================================================================


class TestStockTwitsSentimentEngagementFields:
    """StockTwitsSentiment must carry avg_likes, max_followers, message_count."""

    def test_dataclass_has_engagement_fields(self):
        """All three engagement fields exist on StockTwitsSentiment."""
        from mae_core.market.apis.stocktwits_client import StockTwitsSentiment

        field_names = {f.name for f in fields(StockTwitsSentiment)}
        assert "avg_likes" in field_names
        assert "max_followers" in field_names
        assert "message_count" in field_names

    def test_engagement_fields_default_values(self):
        """Engagement fields default to zero so existing code isn't broken."""
        from mae_core.market.apis.stocktwits_client import StockTwitsSentiment

        s = StockTwitsSentiment(
            ticker="AAPL", bull_count=3, bear_count=2, bull_ratio=0.6,
            total_messages=5, trending=False,
        )
        assert s.avg_likes == 0.0
        assert s.max_followers == 0
        assert s.message_count == 0

    def test_client_extracts_engagement_from_messages(self):
        """_get_ticker_sentiment() computes avg_likes, max_followers, message_count."""
        from mae_core.market.apis.stocktwits_client import StockTwitsClient

        fake_response = {
            "symbol": {"is_following": False},
            "messages": [
                {
                    "id": "1",
                    "body": "Bullish!",
                    "created_at": "2026-03-11T10:00:00Z",
                    "entities": {"sentiment": {"basic": "Bullish"}},
                    "likes": {"total": 10},
                    "user": {"username": "trader1", "followers_count": 500},
                },
                {
                    "id": "2",
                    "body": "Bearish.",
                    "created_at": "2026-03-11T10:01:00Z",
                    "entities": {"sentiment": {"basic": "Bearish"}},
                    "likes": {"total": 4},
                    "user": {"username": "trader2", "followers_count": 1200},
                },
                {
                    "id": "3",
                    "body": "Hmm.",
                    "created_at": "2026-03-11T10:02:00Z",
                    "entities": {},
                    "likes": 2,
                    "user": {"username": "trader3", "followers_count": 300},
                },
            ],
        }

        with patch.object(StockTwitsClient, "_request", return_value=fake_response):
            with patch.object(StockTwitsClient, "_rate_limit"):
                client = StockTwitsClient()
                result = client._get_ticker_sentiment("AAPL")

        assert result is not None
        assert result.message_count == 3
        # avg_likes = (10 + 4 + 2) / 3 = 5.33
        assert abs(result.avg_likes - 16 / 3) < 0.01
        # max_followers = 1200
        assert result.max_followers == 1200

    def test_raw_store_persists_likes_and_followers(self, tmp_path):
        """store_stocktwits_messages() stores likes and user_followers columns."""
        from mae_core.market.raw_store import RawStore

        store = RawStore(base_dir=tmp_path)
        messages = [
            {
                "id": "msg-001",
                "body": "To the moon!",
                "created_at": "2026-03-11T10:00:00Z",
                "entities": {"sentiment": {"basic": "Bullish"}},
                "likes": {"total": 25},
                "user": {"username": "guru", "followers_count": 9800},
            },
            {
                "id": "msg-002",
                "body": "Selling.",
                "created_at": "2026-03-11T10:01:00Z",
                "entities": {"sentiment": {"basic": "Bearish"}},
                "likes": 3,
                "user": {"username": "bear99", "followers_count": 150},
            },
        ]
        count = store.store_stocktwits_messages("TSLA", messages)
        assert count == 2

        conn = store._get_conn("stocktwits")
        cursor = conn.execute(
            "SELECT likes, user_followers FROM stocktwits_messages ORDER BY message_id"
        )
        rows = cursor.fetchall()
        assert len(rows) == 2
        assert rows[0][0] == 25     # msg-001 likes
        assert rows[0][1] == 9800   # msg-001 followers
        assert rows[1][0] == 3      # msg-002 likes (plain int)
        assert rows[1][1] == 150    # msg-002 followers
        store.close()

    def test_influencer_detection_with_high_follower_count(self):
        """max_followers correctly identifies high-follower accounts."""
        from mae_core.market.apis.stocktwits_client import StockTwitsClient

        messages = []
        for i in range(5):
            messages.append({
                "id": str(i),
                "body": f"msg {i}",
                "created_at": "2026-03-11T10:00:00Z",
                "entities": {"sentiment": {"basic": "Bullish"}},
                "likes": {"total": 1},
                "user": {"username": f"user{i}", "followers_count": i * 1000},
            })

        fake_response = {"symbol": {}, "messages": messages}

        with patch.object(StockTwitsClient, "_request", return_value=fake_response):
            with patch.object(StockTwitsClient, "_rate_limit"):
                client = StockTwitsClient()
                result = client._get_ticker_sentiment("SPY")

        assert result is not None
        assert result.max_followers == 4000  # 4 * 1000 is highest


# ===========================================================================
# 3. CryptoPrice — 24h range, ATH distance, supply
# ===========================================================================


class TestCryptoPriceExtendedFields:
    """CryptoPrice must carry high_24h, low_24h, ath_change_percentage,
    circulating_supply, total_supply — and raw_store must persist them."""

    EXTRA_FIELDS = [
        "high_24h", "low_24h", "ath_change_percentage",
        "circulating_supply", "total_supply",
    ]

    def test_dataclass_has_extended_fields(self):
        """All 5 extended fields exist on CryptoPrice."""
        from mae_core.market.apis.coingecko_client import CryptoPrice

        field_names = {f.name for f in fields(CryptoPrice)}
        for name in self.EXTRA_FIELDS:
            assert name in field_names, f"CryptoPrice missing field: {name}"

    def test_extended_fields_default_to_none(self):
        """Extended fields default to None so missing API data is handled safely."""
        from mae_core.market.apis.coingecko_client import CryptoPrice

        cp = CryptoPrice(
            coin_id="bitcoin", symbol="BTC", price_usd=60000.0,
            volume_24h=30e9, change_24h_pct=2.0, change_7d_pct=5.0,
            market_cap=1.2e12, last_updated="2026-03-11T12:00:00Z",
        )
        for name in self.EXTRA_FIELDS:
            assert getattr(cp, name) is None, f"Expected {name} to default to None"

    def test_get_prices_populates_extended_fields(self):
        """get_prices() extracts all 5 extended fields from the API response."""
        from mae_core.market.apis.coingecko_client import CoinGeckoClient

        fake_response = [
            {
                "id": "bitcoin",
                "symbol": "btc",
                "current_price": 60000.0,
                "total_volume": 30e9,
                "price_change_percentage_24h": 2.0,
                "price_change_percentage_7d_in_currency": 5.0,
                "market_cap": 1.2e12,
                "last_updated": "2026-03-11T12:00:00Z",
                "high_24h": 62000.0,
                "low_24h": 58000.0,
                "ath_change_percentage": -12.5,
                "circulating_supply": 19_600_000.0,
                "total_supply": 21_000_000.0,
            }
        ]

        with patch.object(CoinGeckoClient, "_get", return_value=fake_response):
            client = CoinGeckoClient()
            results = client.get_prices(["bitcoin"])

        assert len(results) == 1
        cp = results[0]
        assert cp.high_24h == 62000.0
        assert cp.low_24h == 58000.0
        assert cp.ath_change_percentage == -12.5
        assert cp.circulating_supply == 19_600_000.0
        assert cp.total_supply == 21_000_000.0

    def test_get_prices_handles_none_extended_fields_gracefully(self):
        """Coins without ATH/supply data produce None extended fields (not crash)."""
        from mae_core.market.apis.coingecko_client import CoinGeckoClient

        fake_response = [
            {
                "id": "newcoin",
                "symbol": "new",
                "current_price": 0.001,
                "total_volume": 1000.0,
                "price_change_percentage_24h": 0.0,
                "price_change_percentage_7d_in_currency": 0.0,
                "market_cap": 100_000.0,
                "last_updated": "2026-03-11T12:00:00Z",
                # high_24h, low_24h, ath_change_percentage, circulating_supply,
                # total_supply all absent
            }
        ]

        with patch.object(CoinGeckoClient, "_get", return_value=fake_response):
            client = CoinGeckoClient()
            results = client.get_prices(["newcoin"])

        assert len(results) == 1
        cp = results[0]
        assert cp.high_24h is None
        assert cp.low_24h is None
        assert cp.ath_change_percentage is None
        assert cp.circulating_supply is None
        assert cp.total_supply is None

    def test_raw_store_persists_extended_fields(self, tmp_path):
        """store_coingecko_prices() persists all 5 extended fields to SQLite."""
        from mae_core.market.apis.coingecko_client import CryptoPrice
        from mae_core.market.raw_store import RawStore

        store = RawStore(base_dir=tmp_path)
        coins = [
            CryptoPrice(
                coin_id="ethereum",
                symbol="ETH",
                price_usd=3500.0,
                volume_24h=15e9,
                change_24h_pct=1.5,
                change_7d_pct=3.0,
                market_cap=420e9,
                last_updated="2026-03-11T12:00:00Z",
                high_24h=3600.0,
                low_24h=3400.0,
                ath_change_percentage=-28.0,
                circulating_supply=120_000_000.0,
                total_supply=None,  # ETH has no hard cap
            )
        ]

        count = store.store_coingecko_prices(coins)
        assert count == 1

        conn = store._get_conn("crypto")
        cursor = conn.execute(
            "SELECT high_24h, low_24h, ath_change_percentage, "
            "circulating_supply, total_supply FROM coingecko_prices"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 3600.0          # high_24h
        assert row[1] == 3400.0          # low_24h
        assert row[2] == -28.0           # ath_change_percentage
        assert row[3] == 120_000_000.0   # circulating_supply
        assert row[4] is None            # total_supply (no hard cap)
        store.close()

    def test_raw_store_migration_adds_missing_columns(self, tmp_path):
        """Idempotent column migration runs on pre-existing tables without crashing."""
        import sqlite3
        from mae_core.market.raw_store import RawStore

        # Create an old-format table missing the new columns
        db_path = tmp_path / "crypto.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE coingecko_prices (
                coin_id TEXT, timestamp TEXT, symbol TEXT, price_usd REAL,
                volume_24h REAL, change_24h_pct REAL, change_7d_pct REAL,
                market_cap REAL, last_updated TEXT, ingested_at TEXT,
                PRIMARY KEY (coin_id, timestamp)
            )
        """)
        conn.commit()
        conn.close()

        from mae_core.market.apis.coingecko_client import CryptoPrice
        store = RawStore(base_dir=tmp_path)
        coins = [
            CryptoPrice(
                coin_id="solana", symbol="SOL", price_usd=150.0,
                volume_24h=3e9, change_24h_pct=2.0, change_7d_pct=4.0,
                market_cap=65e9, last_updated="2026-03-11T12:00:00Z",
                high_24h=155.0, low_24h=145.0,
            )
        ]
        # Should not raise — migration is idempotent
        count = store.store_coingecko_prices(coins)
        assert count == 1
        store.close()


# ===========================================================================
# 4. SEC EDGAR Form 4 — derivative transaction parsing
# ===========================================================================


class TestSECEdgarDerivativeTransactionParsing:
    """DerivativeTransaction must be parsed from Form 4 Table II XML."""

    def _make_derivative_xml(
        self,
        trans_code: str = "M",
        shares: float = 1000.0,
        price: float = 0.0,
        acq_disp: str = "A",
        exercise_price: float = 45.0,
        expiration_date: str = "2027-01-15",
        security_title: str = "Employee Stock Option (right to buy)",
        underlying_shares: float = 1000.0,
    ) -> ET.Element:
        """Build a minimal derivativeTransaction XML element."""
        xml_str = f"""
        <derivativeTransaction>
            <securityTitle><value>{security_title}</value></securityTitle>
            <conversionOrExercisePrice><value>{exercise_price}</value></conversionOrExercisePrice>
            <transactionDate><value>2026-03-11</value></transactionDate>
            <transactionCoding>
                <transactionCode>{trans_code}</transactionCode>
            </transactionCoding>
            <transactionAmounts>
                <transactionShares><value>{shares}</value></transactionShares>
                <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>{acq_disp}</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
            <expirationDate><value>{expiration_date}</value></expirationDate>
            <underlyingSecurityShares><value>{underlying_shares}</value></underlyingSecurityShares>
            <postTransactionAmounts>
                <sharesOwnedFollowingTransaction><value>5000</value></sharesOwnedFollowingTransaction>
            </postTransactionAmounts>
        </derivativeTransaction>
        """
        return ET.fromstring(xml_str.strip())

    def test_derivative_transaction_dataclass_exists(self):
        """DerivativeTransaction dataclass is importable from models."""
        from mae_core.market.apis.sec_edgar.models import DerivativeTransaction
        assert DerivativeTransaction is not None

    def test_derivative_transaction_has_required_fields(self):
        """DerivativeTransaction dataclass has all 9 required fields."""
        from mae_core.market.apis.sec_edgar.models import DerivativeTransaction

        expected = {
            "security_title", "transaction_date", "transaction_code",
            "transaction_type", "shares", "price_per_share", "exercise_price",
            "expiration_date", "underlying_shares", "shares_owned_after",
            "is_plan_sale",
        }
        actual = {f.name for f in fields(DerivativeTransaction)}
        assert expected <= actual

    def test_parse_derivative_transaction_option_exercise(self):
        """_parse_derivative_transaction parses an option exercise (code M)."""
        from mae_core.market.apis.sec_edgar.sec_edgar_parsers import (
            _parse_derivative_transaction,
        )

        elem = self._make_derivative_xml(
            trans_code="M",
            shares=500.0,
            exercise_price=42.50,
            expiration_date="2027-06-30",
            security_title="Stock Option (right to buy)",
        )
        result = _parse_derivative_transaction(elem)

        assert result is not None
        assert result.transaction_code == "M"
        assert result.shares == 500.0
        assert result.exercise_price == 42.50
        assert result.expiration_date == "2027-06-30"
        assert result.security_title == "Stock Option (right to buy)"
        assert result.transaction_type == "A"
        assert result.shares_owned_after == 5000.0

    def test_parse_derivative_transaction_is_plan_sale_flag(self):
        """is_plan_sale flag is forwarded to the DerivativeTransaction."""
        from mae_core.market.apis.sec_edgar.sec_edgar_parsers import (
            _parse_derivative_transaction,
        )

        elem = self._make_derivative_xml()
        result = _parse_derivative_transaction(elem, is_plan_sale=True)

        assert result is not None
        assert result.is_plan_sale is True

    def test_parse_derivative_transaction_empty_code_and_zero_shares_returns_none(self):
        """Returns None when there's no transaction code AND zero shares."""
        from mae_core.market.apis.sec_edgar.sec_edgar_parsers import (
            _parse_derivative_transaction,
        )

        xml_str = """
        <derivativeTransaction>
            <securityTitle><value>RSU</value></securityTitle>
            <transactionDate><value>2026-03-11</value></transactionDate>
            <transactionCoding></transactionCoding>
            <transactionAmounts>
                <transactionShares><value>0</value></transactionShares>
                <transactionPricePerShare><value>0</value></transactionPricePerShare>
                <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
            </transactionAmounts>
        </derivativeTransaction>
        """
        elem = ET.fromstring(xml_str.strip())
        result = _parse_derivative_transaction(elem)
        assert result is None

    def test_insider_trade_has_derivative_transactions_field(self):
        """InsiderTrade.derivative_transactions field exists and defaults to empty list."""
        from mae_core.market.apis.sec_edgar.models import InsiderTrade

        trade = InsiderTrade(
            filer_name="John Doe",
            filer_title="CEO",
            filer_relationship="Officer",
            company_name="Acme Corp",
            company_cik="0001234567",
            ticker_symbol="ACME",
            transaction_date="2026-03-11",
            transaction_type="A",
        )
        assert hasattr(trade, "derivative_transactions")
        assert trade.derivative_transactions == []

    def test_parse_form4_xml_attaches_derivatives_to_insider_trade(self):
        """parse_form4() attaches derivative transactions to non-derivative trades."""
        from mae_core.market.apis.sec_edgar.client import SECEdgarClient

        # Minimal valid Form 4 XML with one non-derivative and one derivative
        xml_content = """<?xml version="1.0"?>
        <ownershipDocument>
            <issuer>
                <issuerCik>0001234567</issuerCik>
                <issuerName>Test Corp</issuerName>
                <issuerTradingSymbol>TEST</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerName>Jane Smith</rptOwnerName>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <isOfficer>1</isOfficer>
                    <officerTitle>CFO</officerTitle>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTable>
                <nonDerivativeTransaction>
                    <securityTitle><value>Common Stock</value></securityTitle>
                    <transactionDate><value>2026-03-11</value></transactionDate>
                    <transactionCoding>
                        <transactionCode>P</transactionCode>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>500</value></transactionShares>
                        <transactionPricePerShare><value>100.00</value></transactionPricePerShare>
                        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                    </transactionAmounts>
                    <postTransactionAmounts>
                        <sharesOwnedFollowingTransaction><value>1500</value></sharesOwnedFollowingTransaction>
                    </postTransactionAmounts>
                </nonDerivativeTransaction>
            </nonDerivativeTable>
            <derivativeTable>
                <derivativeTransaction>
                    <securityTitle><value>Employee Stock Option (right to buy)</value></securityTitle>
                    <conversionOrExercisePrice><value>50.00</value></conversionOrExercisePrice>
                    <transactionDate><value>2026-03-11</value></transactionDate>
                    <transactionCoding>
                        <transactionCode>M</transactionCode>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>1000</value></transactionShares>
                        <transactionPricePerShare><value>0</value></transactionPricePerShare>
                        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                    </transactionAmounts>
                    <expirationDate><value>2027-12-31</value></expirationDate>
                    <underlyingSecurityShares><value>1000</value></underlyingSecurityShares>
                    <postTransactionAmounts>
                        <sharesOwnedFollowingTransaction><value>3000</value></sharesOwnedFollowingTransaction>
                    </postTransactionAmounts>
                </derivativeTransaction>
            </derivativeTable>
        </ownershipDocument>
        """

        mock_response = MagicMock()
        mock_response.text = xml_content
        mock_response.content = xml_content.encode("utf-8")

        client = SECEdgarClient()
        with patch.object(client, "_get", return_value=mock_response):
            trades = client.parse_form4("0001234567", "0001234567-26-000001")

        assert len(trades) >= 1
        # The non-derivative trade should carry the derivative transaction
        trade = trades[0]
        assert len(trade.derivative_transactions) == 1
        dtx = trade.derivative_transactions[0]
        assert dtx.transaction_code == "M"
        assert dtx.exercise_price == 50.0
        assert dtx.shares == 1000.0
        assert dtx.expiration_date == "2027-12-31"

    def test_parse_form4_xml_options_only_creates_synthetic_carrier(self):
        """Options-only filings create a synthetic InsiderTrade carrier."""
        from mae_core.market.apis.sec_edgar.client import SECEdgarClient

        xml_content = """<?xml version="1.0"?>
        <ownershipDocument>
            <issuer>
                <issuerCik>0009876543</issuerCik>
                <issuerName>Options Corp</issuerName>
                <issuerTradingSymbol>OPTS</issuerTradingSymbol>
            </issuer>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerName>Bob Builder</rptOwnerName>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <isDirector>1</isDirector>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTable></nonDerivativeTable>
            <derivativeTable>
                <derivativeTransaction>
                    <securityTitle><value>Stock Option</value></securityTitle>
                    <conversionOrExercisePrice><value>75.00</value></conversionOrExercisePrice>
                    <transactionDate><value>2026-03-11</value></transactionDate>
                    <transactionCoding>
                        <transactionCode>M</transactionCode>
                    </transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>2000</value></transactionShares>
                        <transactionPricePerShare><value>0</value></transactionPricePerShare>
                        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
                    </transactionAmounts>
                    <expirationDate><value>2028-01-01</value></expirationDate>
                    <underlyingSecurityShares><value>2000</value></underlyingSecurityShares>
                    <postTransactionAmounts>
                        <sharesOwnedFollowingTransaction><value>2000</value></sharesOwnedFollowingTransaction>
                    </postTransactionAmounts>
                </derivativeTransaction>
            </derivativeTable>
        </ownershipDocument>
        """

        mock_response = MagicMock()
        mock_response.text = xml_content
        mock_response.content = xml_content.encode("utf-8")

        client = SECEdgarClient()
        with patch.object(client, "_get", return_value=mock_response):
            trades = client.parse_form4("0009876543", "0009876543-26-000001")

        # Should create a synthetic carrier for the options-only filing
        assert len(trades) == 1
        assert trades[0].ticker_symbol == "OPTS"
        assert len(trades[0].derivative_transactions) == 1
        assert trades[0].derivative_transactions[0].exercise_price == 75.0


# ===========================================================================
# 5. LegislativeIndicator — sponsors, committees, subjects
# ===========================================================================


class TestLegislativeIndicatorEnrichment:
    """LegislativeIndicator must carry sponsors, committees, and subjects."""

    def test_dataclass_has_enrichment_fields(self):
        """All three enrichment fields exist on LegislativeIndicator."""
        from mae_core.market.apis.congress_gov_client import LegislativeIndicator

        field_names = {f.name for f in fields(LegislativeIndicator)}
        assert "sponsors" in field_names
        assert "committees" in field_names
        assert "subjects" in field_names

    def test_enrichment_fields_default_to_empty_lists(self):
        """Enrichment fields default to empty lists so old call sites aren't broken."""
        from mae_core.market.apis.congress_gov_client import LegislativeIndicator

        ind = LegislativeIndicator(
            bill_id="hr-1234-119",
            bill_number="HR 1234",
            title="Defense Spending Act",
            congress=119,
            policy_area="Armed Forces and National Security",
            action_text="Passed House",
            action_date="2026-03-10",
            signal_type="bill_passed",
            direction="bullish",
            strength=0.8,
        )
        assert ind.sponsors == []
        assert ind.committees == []
        assert ind.subjects == []

    def test_build_indicator_extracts_sponsors(self):
        """_build_indicator() extracts sponsor list from bill detail."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        raw_bill = {
            "congress": 119,
            "type": "HR",
            "number": "7539",
            "title": "National Defense Authorization Act",
            "latestAction": {
                "text": "Passed House",
                "actionDate": "2026-03-10",
            },
        }

        detail_response = {
            "policyArea": {"name": "Armed Forces and National Security"},
            "sponsors": [
                {
                    "bioguideId": "A000001",
                    "fullName": "Rep. Alice Veteran (R-TX)",
                    "party": "R",
                    "state": "TX",
                },
                {
                    "bioguideId": "B000002",
                    "firstName": "Bob",
                    "lastName": "Smith",
                    "party": "D",
                    "state": "CA",
                },
            ],
        }

        committees_response = {"committees": [
            {"name": "House Armed Services", "systemCode": "hsas00", "chamber": "House"}
        ]}

        subjects_response = {"subjects": {
            "legislativeSubjects": [
                {"name": "Defense Procurement"},
                {"name": "Military Personnel"},
            ]
        }}

        def mock_request(route, params):
            if "/committees" in route:
                return committees_response
            if "/subjects" in route:
                return subjects_response
            # bill detail
            return {"bill": detail_response}

        with patch.object(client, "_request", side_effect=mock_request):
            indicator = client._build_indicator(raw_bill)

        assert indicator is not None

        # Sponsors
        assert len(indicator.sponsors) == 2
        assert indicator.sponsors[0]["name"] == "Rep. Alice Veteran (R-TX)"
        assert indicator.sponsors[0]["party"] == "R"
        assert indicator.sponsors[0]["state"] == "TX"
        assert indicator.sponsors[1]["name"] == "Bob Smith"

        # Committees
        assert len(indicator.committees) == 1
        assert indicator.committees[0]["name"] == "House Armed Services"
        assert indicator.committees[0]["chamber"] == "House"

        # Subjects
        assert "Defense Procurement" in indicator.subjects
        assert "Military Personnel" in indicator.subjects

    def test_get_bill_subjects_parses_legislative_subjects(self):
        """get_bill_subjects() extracts subject names from nested API structure."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        subjects_response = {"subjects": {
            "legislativeSubjects": [
                {"name": "Energy Policy"},
                {"name": "Oil and Gas"},
                {"name": "Climate Change"},
            ],
            "policyArea": {"name": "Energy"},
        }}

        with patch.object(client, "_request", return_value=subjects_response):
            subjects = client.get_bill_subjects(119, "hr", "5000")

        assert subjects == ["Energy Policy", "Oil and Gas", "Climate Change"]

    def test_get_bill_subjects_returns_empty_list_on_failure(self):
        """get_bill_subjects() returns [] gracefully when API is unavailable."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        with patch.object(client, "_request", return_value=None):
            subjects = client.get_bill_subjects(119, "hr", "9999")

        assert subjects == []

    def test_get_bill_committees_extracts_committee_details(self):
        """get_bill_committees() returns committee name, systemCode, chamber."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        committees_response = {"committees": [
            {"name": "Senate Finance", "systemCode": "ssfi00", "chamber": "Senate"},
            {"name": "House Ways and Means", "systemCode": "hswm00", "chamber": "House"},
        ]}

        with patch.object(client, "_request", return_value=committees_response):
            result = client.get_bill_committees(119, "s", "1234")

        assert len(result) == 2
        assert result[0]["name"] == "Senate Finance"
        assert result[0]["chamber"] == "Senate"
        assert result[1]["name"] == "House Ways and Means"

    def test_get_bill_committees_returns_empty_list_on_failure(self):
        """get_bill_committees() returns [] gracefully when API is unavailable."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        with patch.object(client, "_request", return_value=None):
            result = client.get_bill_committees(119, "hr", "0001")

        assert result == []

    def test_sponsors_extracted_when_fullname_missing(self):
        """Sponsor name falls back to firstName+lastName when fullName absent."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        raw_bill = {
            "congress": 119,
            "type": "S",
            "number": "100",
            "title": "Healthcare Reform Act",
            "latestAction": {"text": "Passed Senate", "actionDate": "2026-03-10"},
        }

        detail_response = {
            "policyArea": {"name": "Health"},
            "sponsors": [
                {
                    "bioguideId": "C000003",
                    "firstName": "Carol",
                    "lastName": "Jones",
                    "party": "D",
                    "state": "NY",
                    # fullName intentionally absent
                },
            ],
        }

        def mock_request(route, params):
            if "/committees" in route:
                return {"committees": []}
            if "/subjects" in route:
                return {"subjects": {"legislativeSubjects": []}}
            return {"bill": detail_response}

        with patch.object(client, "_request", side_effect=mock_request):
            indicator = client._build_indicator(raw_bill)

        assert indicator is not None
        assert len(indicator.sponsors) == 1
        assert indicator.sponsors[0]["name"] == "Carol Jones"

    def test_enrichment_fields_survive_committee_api_failure(self):
        """Bill indicator is still created even if committee fetch fails."""
        from mae_core.market.apis.congress_gov_client import CongressGovClient

        client = CongressGovClient(api_key="TESTKEY")

        raw_bill = {
            "congress": 119,
            "type": "HR",
            "number": "5000",
            "title": "Energy Infrastructure Act",
            "latestAction": {"text": "Signed", "actionDate": "2026-03-10"},
        }

        detail_response = {
            "policyArea": {"name": "Energy"},
            "sponsors": [{"bioguideId": "D000004", "fullName": "Rep. David Green", "party": "R", "state": "TX"}],
        }

        call_count = [0]

        def mock_request(route, params):
            call_count[0] += 1
            if "/committees" in route:
                raise RuntimeError("API timeout")
            if "/subjects" in route:
                return {"subjects": {"legislativeSubjects": [{"name": "Energy Policy"}]}}
            return {"bill": detail_response}

        with patch.object(client, "_request", side_effect=mock_request):
            indicator = client._build_indicator(raw_bill)

        assert indicator is not None
        assert indicator.committees == []  # Failed gracefully
        assert "Energy Policy" in indicator.subjects
        assert len(indicator.sponsors) == 1
