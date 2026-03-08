"""Tests for the raw data storage layer."""

import sqlite3
from pathlib import Path

import pytest

from mae_core.market.raw_store import RawStore


@pytest.fixture
def store(tmp_path):
    """Create a RawStore backed by a temporary directory."""
    s = RawStore(base_dir=tmp_path)
    yield s
    s.close()


class TestVIXStorage:
    def test_store_vix_daily(self, store):
        rows = [
            {"date": "2026-03-01", "open": 15.0, "high": 16.0, "low": 14.5, "close": 15.5},
            {"date": "2026-03-02", "open": 15.5, "high": 17.0, "low": 15.0, "close": 16.0},
            {"date": "2026-03-03", "close": 14.0},
        ]
        count = store.store_vix_daily(rows)
        assert count == 3

        conn = store._get_conn("vix")
        cursor = conn.execute("SELECT COUNT(*) FROM vix_daily")
        assert cursor.fetchone()[0] == 3

    def test_vix_upsert_dedup(self, store):
        rows1 = [{"date": "2026-03-01", "close": 15.0}]
        rows2 = [{"date": "2026-03-01", "close": 16.0}]  # Same date, new value
        store.store_vix_daily(rows1)
        store.store_vix_daily(rows2)

        conn = store._get_conn("vix")
        cursor = conn.execute("SELECT close FROM vix_daily WHERE date='2026-03-01'")
        assert cursor.fetchone()[0] == 16.0  # Updated, not duplicated

        cursor = conn.execute("SELECT COUNT(*) FROM vix_daily")
        assert cursor.fetchone()[0] == 1

    def test_vix_empty_input(self, store):
        assert store.store_vix_daily([]) == 0


class TestCOTStorage:
    def test_store_cot_report(self, store):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        df = pd.DataFrame({
            "Market and Exchange Names": ["E-MINI S&P 500", "GOLD", "SILVER"],
            "As of Date in Form YYYY-MM-DD": ["2026-03-01", "2026-03-01", "2026-03-01"],
            "Comm_Positions_Long_All": [100000, 50000, 30000],
            "Comm_Positions_Short_All": [80000, 60000, 25000],
            "NonComm_Positions_Long_All": [200000, 100000, 50000],
            "NonComm_Positions_Short_All": [150000, 90000, 45000],
            "Open_Interest_All": [500000, 300000, 200000],
        })

        count = store.store_cot_report(df)
        assert count == 3

        conn = store._get_conn("cot")
        cursor = conn.execute("SELECT COUNT(*) FROM cot_weekly")
        assert cursor.fetchone()[0] == 3

    def test_cot_composite_key_dedup(self, store):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        df1 = pd.DataFrame({
            "Market and Exchange Names": ["GOLD"],
            "As of Date in Form YYYY-MM-DD": ["2026-03-01"],
            "Comm_Positions_Long_All": [50000],
            "Comm_Positions_Short_All": [60000],
            "NonComm_Positions_Long_All": [100000],
            "NonComm_Positions_Short_All": [90000],
            "Open_Interest_All": [300000],
        })
        df2 = pd.DataFrame({
            "Market and Exchange Names": ["GOLD"],
            "As of Date in Form YYYY-MM-DD": ["2026-03-01"],
            "Comm_Positions_Long_All": [55000],  # Updated value
            "Comm_Positions_Short_All": [60000],
            "NonComm_Positions_Long_All": [100000],
            "NonComm_Positions_Short_All": [90000],
            "Open_Interest_All": [300000],
        })

        store.store_cot_report(df1)
        store.store_cot_report(df2)

        conn = store._get_conn("cot")
        cursor = conn.execute("SELECT commercial_long FROM cot_weekly WHERE contract_name='GOLD'")
        assert cursor.fetchone()[0] == 55000  # Updated

        cursor = conn.execute("SELECT COUNT(*) FROM cot_weekly")
        assert cursor.fetchone()[0] == 1


class TestEIAStorage:
    def test_store_eia_series(self, store):
        observations = [
            {"period": "2026-02-28", "value": 420000},
            {"period": "2026-02-21", "value": 418000},
            {"period": "2026-02-14", "value": 415000},
            {"period": "2026-02-07", "value": 412000},
            {"period": "2026-01-31", "value": 410000},
        ]
        count = store.store_eia_series("crude_stocks", observations)
        assert count == 5

    def test_eia_separate_series(self, store):
        store.store_eia_series("crude_stocks", [{"period": "2026-03-01", "value": 420000}])
        store.store_eia_series("natgas_storage", [{"period": "2026-03-01", "value": 1800}])

        conn = store._get_conn("eia")
        cursor = conn.execute("SELECT COUNT(*) FROM eia_observations")
        assert cursor.fetchone()[0] == 2

        cursor = conn.execute(
            "SELECT value FROM eia_observations WHERE series_key='natgas_storage'"
        )
        assert cursor.fetchone()[0] == 1800

    def test_eia_empty_input(self, store):
        assert store.store_eia_series("crude_stocks", []) == 0


class TestTrendsStorage:
    def test_store_trends(self, store):
        rows = [{"timestamp": f"2026-03-01 {h:02d}:00", "interest": 50 + h} for h in range(24)]
        count = store.store_trends("SPY", rows)
        assert count == 24

    def test_trends_multiple_keywords(self, store):
        store.store_trends("SPY", [{"timestamp": "2026-03-01 00:00", "interest": 50}])
        store.store_trends("recession", [{"timestamp": "2026-03-01 00:00", "interest": 30}])

        conn = store._get_conn("trends")
        cursor = conn.execute("SELECT COUNT(*) FROM trends_hourly")
        assert cursor.fetchone()[0] == 2

    def test_trends_empty_input(self, store):
        assert store.store_trends("SPY", []) == 0


class TestInfrastructure:
    def test_wal_mode(self, store):
        conn = store._get_conn("test_domain")
        cursor = conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0] == "wal"

    def test_close_and_reopen(self, tmp_path):
        store1 = RawStore(base_dir=tmp_path)
        store1.store_vix_daily([{"date": "2026-03-01", "close": 15.0}])
        store1.close()

        store2 = RawStore(base_dir=tmp_path)
        conn = store2._get_conn("vix")
        cursor = conn.execute("SELECT close FROM vix_daily WHERE date='2026-03-01'")
        assert cursor.fetchone()[0] == 15.0
        store2.close()

    def test_missing_dir_created(self, tmp_path):
        new_dir = tmp_path / "nested" / "raw"
        store = RawStore(base_dir=new_dir)
        assert new_dir.exists()
        store.close()


class TestPriceSnapshotStorage:
    def test_store_price_snapshot(self, store):
        info = {
            "currentPrice": 150.25,
            "marketCap": 2400000000000,
            "shortRatio": 1.5,
            "fiftyTwoWeekHigh": 180.0,
            "fiftyTwoWeekLow": 120.0,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "trailingPE": 28.5,
            "forwardPE": 25.0,
            "floatShares": 15000000000,
            "sharesShort": 100000000,
            "targetMeanPrice": 175.0,
            "averageVolume": 60000000,
        }
        count = store.store_price_snapshot("AAPL", info)
        assert count == 1

        conn = store._get_conn("prices")
        row = conn.execute("SELECT * FROM price_snapshots WHERE symbol='AAPL'").fetchone()
        assert row is not None
        assert row[2] == 150.25  # price
        assert row[3] == 2400000000000  # market_cap

    def test_price_snapshot_full_json_preserved(self, store):
        import json
        info = {"currentPrice": 100, "sector": "Tech", "unusual_field": "preserved"}
        store.store_price_snapshot("TSLA", info)

        conn = store._get_conn("prices")
        row = conn.execute("SELECT info_json FROM price_snapshots WHERE symbol='TSLA'").fetchone()
        parsed = json.loads(row[0])
        assert parsed["unusual_field"] == "preserved"
        assert parsed["sector"] == "Tech"

    def test_price_snapshot_empty(self, store):
        assert store.store_price_snapshot("AAPL", {}) == 0


class TestFREDStorage:
    def test_store_fred_observations(self, store):
        observations = [
            {"date": "2026-03-01", "value": "-0.45", "realtime_start": "2026-03-02", "realtime_end": "2026-03-02"},
            {"date": "2026-02-28", "value": "-0.40", "realtime_start": "2026-03-01", "realtime_end": "2026-03-01"},
            {"date": "2026-02-27", "value": ".", "realtime_start": "2026-02-28", "realtime_end": "2026-02-28"},
        ]
        count = store.store_fred_observations("T10Y2Y", observations)
        assert count == 2  # One skipped due to "." value

        conn = store._get_conn("fred")
        cursor = conn.execute("SELECT COUNT(*) FROM fred_observations WHERE series_id='T10Y2Y'")
        assert cursor.fetchone()[0] == 2

    def test_fred_vintage_metadata_preserved(self, store):
        observations = [
            {"date": "2026-03-01", "value": "5.25", "realtime_start": "2026-03-02", "realtime_end": "2026-03-08"},
        ]
        store.store_fred_observations("DFF", observations)

        conn = store._get_conn("fred")
        row = conn.execute("SELECT realtime_start, realtime_end FROM fred_observations").fetchone()
        assert row[0] == "2026-03-02"
        assert row[1] == "2026-03-08"

    def test_fred_empty(self, store):
        assert store.store_fred_observations("T10Y2Y", []) == 0


class TestStockTwitsStorage:
    def test_store_stocktwits_messages(self, store):
        messages = [
            {"id": "msg1", "body": "AAPL to the moon!", "entities": {"sentiment": {"basic": "Bullish"}},
             "user": {"username": "trader1"}, "created_at": "2026-03-01T10:00:00Z", "likes": {"total": 5}},
            {"id": "msg2", "body": "bearish on AAPL", "entities": {"sentiment": {"basic": "Bearish"}},
             "user": {"username": "trader2"}, "created_at": "2026-03-01T10:05:00Z", "likes": {"total": 2}},
            {"id": "msg3", "body": "no sentiment label", "entities": {},
             "user": {"username": "trader3"}, "created_at": "2026-03-01T10:10:00Z"},
        ]
        count = store.store_stocktwits_messages("AAPL", messages)
        assert count == 3

        conn = store._get_conn("stocktwits")
        rows = conn.execute("SELECT sentiment, body FROM stocktwits_messages ORDER BY message_id").fetchall()
        assert rows[0][0] == "Bullish"
        assert "moon" in rows[0][1]
        assert rows[2][0] == ""  # No sentiment label

    def test_stocktwits_dedup(self, store):
        msg = [{"id": "msg1", "body": "first", "entities": {}, "user": {"username": "u"}, "created_at": ""}]
        store.store_stocktwits_messages("AAPL", msg)
        msg[0]["body"] = "updated"
        store.store_stocktwits_messages("AAPL", msg)

        conn = store._get_conn("stocktwits")
        row = conn.execute("SELECT body FROM stocktwits_messages WHERE message_id='msg1'").fetchone()
        assert row[0] == "updated"

    def test_stocktwits_empty(self, store):
        assert store.store_stocktwits_messages("AAPL", []) == 0


class TestFinnhubStorage:
    def test_store_finnhub_sentiment(self, store):
        raw_data = {
            "buzz": {"articlesInLastWeek": 42, "weeklyAverage": 30, "buzz": 1.4},
            "companyNewsScore": 0.85,
            "sentiment": {"bullishPercent": 0.65, "bearishPercent": 0.35},
        }
        count = store.store_finnhub_sentiment("AAPL", raw_data)
        assert count == 1

        conn = store._get_conn("finnhub")
        row = conn.execute("SELECT bullish_pct, buzz_articles FROM finnhub_sentiment").fetchone()
        assert row[0] == 0.65
        assert row[1] == 42

    def test_store_finnhub_earnings(self, store):
        events = [
            {"symbol": "AAPL", "date": "2026-04-25", "quarter": 2, "year": 2026,
             "epsEstimate": 1.62, "epsActual": None, "revenueEstimate": 94000000000,
             "revenueActual": None, "hour": "amc"},
            {"symbol": "MSFT", "date": "2026-04-22", "quarter": 3, "year": 2026,
             "epsEstimate": 3.22, "epsActual": 3.45, "revenueEstimate": 68000000000,
             "revenueActual": 69500000000, "hour": "bmo"},
        ]
        count = store.store_finnhub_earnings(events)
        assert count == 2

        conn = store._get_conn("finnhub")
        row = conn.execute("SELECT quarter, year FROM finnhub_earnings WHERE symbol='AAPL'").fetchone()
        assert row[0] == 2
        assert row[1] == 2026

    def test_store_finnhub_economic_all_countries(self, store):
        events = [
            {"event": "CPI", "country": "US", "date": "2026-03-12", "time": "08:30",
             "impact": "high", "actual": 3.2, "estimate": 3.0, "prev": 2.9, "unit": "%"},
            {"event": "ECB Rate Decision", "country": "EU", "date": "2026-03-13",
             "time": "13:15", "impact": "high", "actual": None, "estimate": 3.5,
             "prev": 3.5, "unit": "%"},
            {"event": "CPI", "country": "JP", "date": "2026-03-14", "time": "23:30",
             "impact": "medium", "actual": 2.1, "estimate": 2.0, "prev": 1.9, "unit": "%"},
        ]
        count = store.store_finnhub_economic(events)
        assert count == 3  # ALL countries stored, not just US

        conn = store._get_conn("finnhub")
        cursor = conn.execute("SELECT COUNT(*) FROM finnhub_economic")
        assert cursor.fetchone()[0] == 3

    def test_finnhub_empty(self, store):
        assert store.store_finnhub_sentiment("AAPL", {}) == 0
        assert store.store_finnhub_earnings([]) == 0
        assert store.store_finnhub_economic([]) == 0


class TestCongressionalTradesStorage:
    def test_store_congressional_trades(self, store):
        trades = [
            {"representative": "Nancy Pelosi", "ticker": "NVDA",
             "transaction_date": "2026-02-15", "transaction_type": "purchase",
             "amount_range": "$1,001 - $15,000", "amount_low": 1001, "amount_high": 15000,
             "asset_description": "NVIDIA Corp", "district": "CA-11",
             "party": "D", "owner": "Spouse", "disclosure_date": "2026-03-01",
             "chamber": "house"},
        ]
        count = store.store_congressional_trades(trades)
        assert count == 1

        conn = store._get_conn("congressional")
        row = conn.execute("SELECT * FROM congressional_trades").fetchone()
        assert row[0] == "Nancy Pelosi"
        assert row[9] == "D"  # party

    def test_congressional_trades_dedup(self, store):
        trade = [{"representative": "Test Rep", "ticker": "AAPL",
                  "transaction_date": "2026-01-01", "transaction_type": "purchase",
                  "amount_low": 1000, "amount_high": 5000, "chamber": "house"}]
        store.store_congressional_trades(trade)
        store.store_congressional_trades(trade)

        conn = store._get_conn("congressional")
        cursor = conn.execute("SELECT COUNT(*) FROM congressional_trades")
        assert cursor.fetchone()[0] == 1

    def test_congressional_empty(self, store):
        assert store.store_congressional_trades([]) == 0


class TestCongressBillsStorage:
    def test_store_congress_bills(self, store):
        bills = [
            {"congress": 119, "type": "hr", "number": "7539",
             "title": "National Defense Authorization Act",
             "latestAction": {"text": "Passed House", "actionDate": "2026-03-01"},
             "policyArea": {"name": "Armed Forces and National Security"},
             "sponsors": [{"bioguideId": "A001", "fullName": "Rep. Adams"}],
             "url": "https://congress.gov/...", "introducedDate": "2026-01-15",
             "updateDate": "2026-03-01", "originChamber": "House"},
        ]
        count = store.store_congress_bills(bills)
        assert count == 1

        conn = store._get_conn("legislation")
        row = conn.execute("SELECT title, policy_area, sponsors_json FROM congress_bills").fetchone()
        assert "Defense" in row[0]
        assert "Armed Forces" in row[1]
        assert "Adams" in row[2]

    def test_congress_bills_empty(self, store):
        assert store.store_congress_bills([]) == 0


class TestFINRAShortVolumeStorage:
    def test_store_finra_short_volume_dicts(self, store):
        records = [
            {"symbol": "AAPL", "date": "2026-03-01", "short_volume": 5000000,
             "short_exempt_volume": 200000, "total_volume": 10000000,
             "short_ratio": 0.5, "speculative_short_ratio": 0.48},
            {"symbol": "NVDA", "date": "2026-03-01", "short_volume": 8000000,
             "short_exempt_volume": 500000, "total_volume": 12000000,
             "short_ratio": 0.667, "speculative_short_ratio": 0.625},
        ]
        count = store.store_finra_short_volume(records)
        assert count == 2

        conn = store._get_conn("finra")
        cursor = conn.execute("SELECT COUNT(*) FROM finra_short_volume")
        assert cursor.fetchone()[0] == 2

    def test_store_finra_short_volume_objects(self, store):
        from mae_core.market.apis.finra_short_interest import ShortInterestData
        record = ShortInterestData(
            symbol="TSLA", date="2026-03-01", short_volume=3000000,
            short_exempt_volume=100000, total_volume=6000000,
            short_ratio=0.5, speculative_short_ratio=0.483,
        )
        count = store.store_finra_short_volume([record])
        assert count == 1

    def test_finra_empty(self, store):
        assert store.store_finra_short_volume([]) == 0


# ---------------------------------------------------------------------------
# TASK 7: Tests for 12 newly wired client store methods
# ---------------------------------------------------------------------------


class TestSECForm4Storage:
    def test_store_form4_dicts(self, store):
        trade = {
            "filing_id": "0001234567-26-000001",
            "ticker": "AAPL",
            "company_cik": "0000320193",
            "insider_name": "Tim Cook",
            "insider_title": "CEO",
            "transaction_date": "2026-03-01",
            "transaction_type": "P",
            "shares": 5000,
            "price_per_share": 175.0,
            "total_value": 875000.0,
            "shares_after": 1000000,
            "is_derivative": False,
            "filing_date": "2026-03-03",
            "form_type": "4",
        }
        count = store.store_sec_form4([trade])
        assert count == 1
        conn = store._get_conn("sec_edgar")
        cursor = conn.execute("SELECT COUNT(*) FROM form4_trades")
        assert cursor.fetchone()[0] == 1

    def test_store_form4_empty(self, store):
        assert store.store_sec_form4([]) == 0


class TestMassiveBarsStorage:
    def test_store_massive_bars_objects(self, store):
        from mae_core.market.apis.massive_client import TickerBar
        bars = [
            TickerBar(ticker="SPY", open=495.0, high=498.0, low=494.0,
                      close=497.0, volume=80_000_000.0, vwap=496.5,
                      transactions=1_200_000, date="2026-03-07"),
            TickerBar(ticker="QQQ", open=430.0, high=433.0, low=429.0,
                      close=432.0, volume=50_000_000.0, vwap=431.0,
                      transactions=800_000, date="2026-03-07"),
        ]
        count = store.store_massive_bars(bars)
        assert count == 2
        conn = store._get_conn("massive")
        cursor = conn.execute("SELECT COUNT(*) FROM daily_bars")
        assert cursor.fetchone()[0] == 2

    def test_store_massive_bars_empty(self, store):
        assert store.store_massive_bars([]) == 0


class TestCoinGeckoStorage:
    def test_store_coingecko_prices(self, store):
        coins = [
            {"coin_id": "bitcoin", "symbol": "BTC",
             "price_usd": 75000.0, "market_cap": 1_500_000_000_000.0,
             "volume_24h": 30_000_000_000.0, "change_24h_pct": 2.5,
             "change_7d_pct": 5.0, "last_updated": "2026-03-07T00:00:00Z"},
        ]
        count = store.store_coingecko_prices(coins)
        assert count == 1
        conn = store._get_conn("crypto")
        cursor = conn.execute("SELECT COUNT(*) FROM coingecko_prices")
        assert cursor.fetchone()[0] == 1

    def test_store_coingecko_empty(self, store):
        assert store.store_coingecko_prices([]) == 0


class TestCoinCapStorage:
    def test_store_coincap_assets_objects(self, store):
        from mae_core.market.apis.coincap_client import CryptoAsset
        asset = CryptoAsset(
            asset_id="bitcoin", symbol="BTC", name="Bitcoin",
            rank=1, price_usd=75000.0, volume_24h_usd=30_000_000_000.0,
            change_24h_pct=2.5, market_cap_usd=1_500_000_000_000.0,
            supply=19_700_000.0, max_supply=21_000_000.0, vwap_24h=74800.0,
        )
        count = store.store_coincap_assets([asset])
        assert count == 1
        conn = store._get_conn("crypto")
        cursor = conn.execute("SELECT COUNT(*) FROM coincap_assets")
        assert cursor.fetchone()[0] == 1

    def test_store_coincap_empty(self, store):
        assert store.store_coincap_assets([]) == 0


class TestOpenInsiderStorage:
    def test_store_openinsider_purchases_objects(self, store):
        from mae_core.market.apis.openinsider_client import InsiderPurchase
        purchase = InsiderPurchase(
            filing_date="2026-03-03", trade_date="2026-03-01",
            ticker="NVDA", company_name="NVIDIA Corporation",
            insider_name="Jensen Huang", title="CEO",
            trade_type="P - Purchase", price=875.0,
            quantity=10000, owned=3_000_000,
            delta_owned_pct=0.33, value=8_750_000.0,
        )
        count = store.store_openinsider_purchases([purchase])
        assert count == 1
        conn = store._get_conn("openinsider")
        cursor = conn.execute("SELECT COUNT(*) FROM insider_purchases")
        assert cursor.fetchone()[0] == 1

    def test_store_openinsider_empty(self, store):
        assert store.store_openinsider_purchases([]) == 0


class TestFinVizStorage:
    def test_store_finviz_insider_trades_objects(self, store):
        from mae_core.market.apis.finviz_client import FinVizInsiderTrade
        trade = FinVizInsiderTrade(
            ticker="AAPL", owner_name="Tim Cook", relationship="CEO",
            transaction_type="Buy", date="2026-03-01",
            shares_traded=5000, value=875000.0, shares_owned=1_000_000,
        )
        count = store.store_finviz_insider_trades([trade])
        assert count == 1
        conn = store._get_conn("finviz")
        cursor = conn.execute("SELECT COUNT(*) FROM finviz_insider_trades")
        assert cursor.fetchone()[0] == 1

    def test_store_finviz_unusual_volume_objects(self, store):
        from mae_core.market.apis.finviz_client import UnusualVolume
        item = UnusualVolume(
            ticker="GME", company="GameStop Corp", sector="Consumer Cyclical",
            price=25.0, change_pct=15.0, volume=50_000_000,
            avg_volume=10_000_000, volume_ratio=5.0,
        )
        count = store.store_finviz_unusual_volume([item])
        assert count == 1
        conn = store._get_conn("finviz")
        cursor = conn.execute("SELECT COUNT(*) FROM finviz_unusual_volume")
        assert cursor.fetchone()[0] == 1

    def test_store_finviz_empty(self, store):
        assert store.store_finviz_insider_trades([]) == 0
        assert store.store_finviz_unusual_volume([]) == 0


class TestEdgarFilingsStorage:
    def test_store_edgar_filings_objects(self, store):
        from mae_core.market.apis.edgar_enhanced_client import ActivistFiling
        filing = ActivistFiling(
            filer_name="ValueAct Capital", filer_cik="0001234567",
            subject_company="Salesforce Inc", subject_ticker="CRM",
            filing_date="2026-03-01", form_type="SC 13D",
            percent_owned=5.2, purpose="Activist position (>5% ownership)",
        )
        count = store.store_edgar_filings([filing])
        assert count == 1
        conn = store._get_conn("sec_edgar")
        cursor = conn.execute("SELECT COUNT(*) FROM institutional_filings")
        assert cursor.fetchone()[0] == 1

    def test_store_edgar_filings_empty(self, store):
        assert store.store_edgar_filings([]) == 0


class TestFinnhubTicksStorage:
    def test_store_finnhub_ticks_dicts(self, store):
        ticks = [
            {"symbol": "AAPL", "price": 175.5, "volume": 100.0,
             "timestamp_ms": 1741200000000, "conditions": []},
            {"symbol": "MSFT", "price": 420.0, "volume": 50.0,
             "timestamp_ms": 1741200001000, "conditions": ["R"]},
        ]
        count = store.store_finnhub_ticks(ticks)
        assert count == 2
        conn = store._get_conn("finnhub")
        cursor = conn.execute("SELECT COUNT(*) FROM finnhub_ticks")
        assert cursor.fetchone()[0] == 2

    def test_store_finnhub_ticks_empty(self, store):
        assert store.store_finnhub_ticks([]) == 0


class TestApeWisdomStorage:
    def test_store_apewisdom_sentiment_objects(self, store):
        from mae_core.market.apis.apewisdom import SocialSentiment
        record = SocialSentiment(
            ticker="GME", mentions_24h=5000, mentions_prior_24h=1000,
            upvotes=2500, rank=1, mention_change=5.0,
        )
        count = store.store_apewisdom_sentiment([record])
        assert count == 1
        conn = store._get_conn("social")
        cursor = conn.execute("SELECT COUNT(*) FROM apewisdom_sentiment")
        assert cursor.fetchone()[0] == 1

    def test_store_apewisdom_empty(self, store):
        assert store.store_apewisdom_sentiment([]) == 0


class TestJobPostingsStorage:
    def test_store_job_postings_objects(self, store):
        from mae_core.market.apis.job_tracker import HiringSignal
        signal = HiringSignal(
            company_name="Lockheed Martin", ticker="LMT",
            jobs_24h=45, jobs_7d=200, jobs_30d=600,
            is_spike=True, spike_ratio=3.15,
            engineering_jobs=30, cleared_jobs=15,
            contract_related_jobs=40, confidence=0.80,
        )
        count = store.store_job_postings([signal])
        assert count == 1
        conn = store._get_conn("jobs")
        cursor = conn.execute("SELECT COUNT(*) FROM job_postings")
        assert cursor.fetchone()[0] == 1

    def test_store_job_postings_empty(self, store):
        assert store.store_job_postings([]) == 0


class TestSAMOpportunitiesStorage:
    def test_store_sam_opportunities_objects(self, store):
        from mae_core.market.apis.sam_gov import ContractOpportunity
        opp = ContractOpportunity(
            notice_id="W912CN-26-R-0042",
            title="Missile Defense System Integration",
            solicitation_number="W912CN-26-R-0042",
            department="Department of Defense",
            agency="Army", office="Army Contracting Command",
            naics_code="541330", set_aside="",
            type_of_contract="Firm Fixed Price",
            place_of_performance="AL",
            estimated_value=250_000_000.0,
            award_date="", posted_date="2026-03-01",
            response_deadline="2026-04-01",
            active=True, contract_type="Solicitation",
            url="https://sam.gov/opp/W912CN-26-R-0042",
        )
        count = store.store_sam_opportunities([opp])
        assert count == 1
        conn = store._get_conn("contracts")
        cursor = conn.execute("SELECT COUNT(*) FROM sam_opportunities")
        assert cursor.fetchone()[0] == 1

    def test_store_sam_opportunities_empty(self, store):
        assert store.store_sam_opportunities([]) == 0
