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
