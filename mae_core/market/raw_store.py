"""
raw_store.py - Raw Data Storage Layer

Persists ALL data from API calls before processing. Each data domain
gets its own SQLite database in data/market/raw/. WAL mode enables
concurrent read/write for daemon operation.

This is MIDGE's long-term memory for raw market data. The signal
pipeline extracts 1-2 processed values per API call; this layer
preserves the other 95% for future pattern discovery.
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/market/raw")


class RawStore:
    """
    Persists raw API data to SQLite databases (one per domain).

    WAL mode for concurrent daemon read/write. Upsert semantics
    (INSERT OR REPLACE) handle dedup automatically.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = Path(base_dir) if base_dir else RAW_DATA_DIR
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._connections: Dict[str, sqlite3.Connection] = {}

    def _get_conn(self, domain: str) -> sqlite3.Connection:
        if domain not in self._connections:
            db_path = self._base_dir / f"{domain}.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._connections[domain] = conn
        return self._connections[domain]

    def close(self):
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    # --- VIX ---

    def store_vix_daily(self, rows: List[Dict[str, Any]]) -> int:
        """Store VIX daily OHLC data.

        Args:
            rows: List of dicts with keys: date, open, high, low, close.
                  At minimum 'date' and 'close' are required.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0

        conn = self._get_conn("vix")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vix_daily (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = [
            (
                r.get("date", ""),
                r.get("open"),
                r.get("high"),
                r.get("low"),
                r.get("close"),
                now,
            )
            for r in rows
            if r.get("date")
        ]

        conn.executemany(
            "INSERT OR REPLACE INTO vix_daily (date, open, high, low, close, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            data,
        )
        conn.commit()
        logger.debug("RawStore: stored %d VIX daily rows", len(data))
        return len(data)

    # --- COT ---

    def store_cot_report(self, df) -> int:
        """Store COT report data from a cot-reports DataFrame.

        Stores ALL contracts (not just the 10 MIDGE currently maps),
        preserving the full positioning picture for future pattern discovery.

        Args:
            df: pandas DataFrame from cot_reports.cot_year().

        Returns:
            Number of rows upserted.
        """
        if df is None or (hasattr(df, "empty") and df.empty):
            return 0

        conn = self._get_conn("cot")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_weekly (
                report_date TEXT,
                contract_name TEXT,
                commercial_long INTEGER,
                commercial_short INTEGER,
                noncommercial_long INTEGER,
                noncommercial_short INTEGER,
                open_interest INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (report_date, contract_name)
            )
        """)

        # Find column names (cot-reports uses varying naming conventions)
        name_col = None
        for col in ["Market and Exchange Names", "Market_and_Exchange_Names"]:
            if col in df.columns:
                name_col = col
                break
        if name_col is None:
            logger.debug("RawStore: COT DataFrame missing name column")
            return 0

        date_col = None
        for col in ["As of Date in Form YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD",
                     "Report_Date_as_YYYY-MM-DD"]:
            if col in df.columns:
                date_col = col
                break

        def _safe_int(row, col_variants, default=0):
            for v in col_variants:
                if v in row.index:
                    try:
                        return int(float(row[v]))
                    except (ValueError, TypeError):
                        continue
            return default

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for _, row in df.iterrows():
            try:
                contract = str(row.get(name_col, ""))
                report_date = str(row.get(date_col, "")) if date_col else ""
                if hasattr(report_date, "strftime"):
                    report_date = report_date.strftime("%Y-%m-%d")
                report_date = report_date[:10] if report_date else ""

                if not contract or not report_date:
                    continue

                comm_long = _safe_int(row, ["Comm_Positions_Long_All", "Commercial Long",
                                            "Comm_Positions_Long_Old"])
                comm_short = _safe_int(row, ["Comm_Positions_Short_All", "Commercial Short",
                                             "Comm_Positions_Short_Old"])
                noncomm_long = _safe_int(row, ["NonComm_Positions_Long_All", "Noncommercial Long",
                                               "NonComm_Positions-Long_All"])
                noncomm_short = _safe_int(row, ["NonComm_Positions_Short_All", "Noncommercial Short",
                                                "NonComm_Positions-Short_All"])
                oi = _safe_int(row, ["Open_Interest_All", "Open Interest",
                                     "Open_Interest_Old"], default=0)

                data.append((
                    report_date, contract,
                    comm_long, comm_short,
                    noncomm_long, noncomm_short,
                    oi, now,
                ))
            except Exception:
                continue

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO cot_weekly "
                "(report_date, contract_name, commercial_long, commercial_short, "
                "noncommercial_long, noncommercial_short, open_interest, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d COT report rows", len(data))
        return len(data)

    # --- EIA ---

    def store_eia_series(self, series_key: str, observations: List[Dict[str, Any]]) -> int:
        """Store EIA API observations for a series.

        Args:
            series_key: e.g. "crude_stocks", "natgas_storage"
            observations: Raw response.data list from EIA API v2.
                         Each dict has at least 'period' and 'value'.

        Returns:
            Number of rows upserted.
        """
        if not observations:
            return 0

        conn = self._get_conn("eia")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eia_observations (
                series_key TEXT,
                period TEXT,
                value REAL,
                ingested_at TEXT,
                PRIMARY KEY (series_key, period)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for obs in observations:
            try:
                period = str(obs.get("period", ""))
                value = float(obs.get("value", 0))
                if period:
                    data.append((series_key, period, value, now))
            except (ValueError, TypeError):
                continue

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO eia_observations "
                "(series_key, period, value, ingested_at) VALUES (?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d EIA observations for %s", len(data), series_key)
        return len(data)

    # --- Google Trends ---

    def store_trends(self, keyword: str, rows: List[Dict[str, Any]]) -> int:
        """Store Google Trends hourly interest data.

        Args:
            keyword: Search term (e.g. "SPY", "recession")
            rows: List of dicts with 'timestamp' and 'interest' keys.

        Returns:
            Number of rows upserted.
        """
        if not rows:
            return 0

        conn = self._get_conn("trends")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trends_hourly (
                keyword TEXT,
                timestamp TEXT,
                interest INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (keyword, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = [
            (keyword, str(r.get("timestamp", "")), int(r.get("interest", 0)), now)
            for r in rows
            if r.get("timestamp")
        ]

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO trends_hourly "
                "(keyword, timestamp, interest, ingested_at) VALUES (?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Trends rows for '%s'", len(data), keyword)
        return len(data)
