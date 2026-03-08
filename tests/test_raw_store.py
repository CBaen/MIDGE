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
