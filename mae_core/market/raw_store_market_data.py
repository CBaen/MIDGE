"""
raw_store_market_data.py - Market data storage mixin.

Handles VIX, COT, EIA, Google Trends, price snapshots, FRED observations,
FRED yield curves, and USDA agricultural data.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RawStoreMarketDataMixin:
    """Mixin for storing market data: VIX, COT, EIA, Trends, prices, FRED, USDA."""

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

    def get_cot_history(
        self, contract_name: str, weeks: int = 52
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent N weeks of COT data for a contract.

        Used to compute week-over-week change and COT Index (percentile rank).
        Returns rows sorted oldest-first so callers can compute diffs directly.

        Args:
            contract_name: CFTC contract name (e.g. "GOLD", "E-MINI S&P 500").
            weeks: How many weekly rows to return (default 52 for COT Index).

        Returns:
            List of dicts with keys: report_date, commercial_long, commercial_short,
            noncommercial_long, noncommercial_short, open_interest.
            Empty list if the cot_weekly table doesn't exist yet.
        """
        try:
            conn = self._get_conn("cot")
            # Use LIKE for partial match — contract names in the store may contain
            # exchange suffix (e.g. "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE")
            cursor = conn.execute(
                """
                SELECT report_date, commercial_long, commercial_short,
                       noncommercial_long, noncommercial_short, open_interest
                FROM cot_weekly
                WHERE contract_name LIKE ?
                ORDER BY report_date DESC
                LIMIT ?
                """,
                (f"%{contract_name}%", weeks),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_cot_history failed for %s: %s", contract_name, exc)
            return []

        return [
            {
                "report_date": r[0],
                "commercial_long": r[1],
                "commercial_short": r[2],
                "noncommercial_long": r[3],
                "noncommercial_short": r[4],
                "open_interest": r[5],
            }
            for r in reversed(rows)  # oldest-first
        ]

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

    # --- Price Snapshots (yfinance ticker.info — 80+ fields) ---

    def store_price_snapshot(self, symbol: str, info: Dict[str, Any]) -> int:
        """Store full yfinance ticker.info dict as JSON.

        This captures fundamentals, float/short data, analyst targets, 52-week
        context, sector info — everything the price_fetcher currently discards.

        Args:
            symbol: Ticker symbol.
            info: Full ticker.info dict from yfinance.

        Returns:
            1 on success, 0 on failure.
        """
        if not info:
            return 0

        conn = self._get_conn("prices")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_snapshots (
                symbol TEXT,
                timestamp TEXT,
                price REAL,
                market_cap REAL,
                short_ratio REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL,
                info_json TEXT,
                ingested_at TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        try:
            info_json = json.dumps(info, default=str)
        except (TypeError, ValueError):
            info_json = "{}"

        conn.execute(
            "INSERT OR REPLACE INTO price_snapshots "
            "(symbol, timestamp, price, market_cap, short_ratio, "
            "fifty_two_week_high, fifty_two_week_low, info_json, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol, now[:13],  # Truncate to hour for dedup
                price,
                info.get("marketCap"),
                info.get("shortRatio"),
                info.get("fiftyTwoWeekHigh"),
                info.get("fiftyTwoWeekLow"),
                info_json, now,
            ),
        )
        conn.commit()
        logger.debug("RawStore: stored price snapshot for %s (%d info fields)", symbol, len(info))
        return 1

    # --- FRED Observations ---

    def store_fred_observations(self, series_id: str, observations: List[Dict[str, Any]]) -> int:
        """Store FRED API observations with vintage metadata.

        Args:
            series_id: FRED series ID (e.g. "T10Y2Y").
            observations: Raw observations list from FRED API response.

        Returns:
            Number of rows upserted.
        """
        if not observations:
            return 0

        conn = self._get_conn("fred")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fred_observations (
                series_id TEXT,
                date TEXT,
                value REAL,
                realtime_start TEXT,
                realtime_end TEXT,
                ingested_at TEXT,
                PRIMARY KEY (series_id, date)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for obs in observations:
            raw_val = obs.get("value", "")
            if raw_val in (".", ""):
                continue
            try:
                value = float(raw_val)
            except (ValueError, TypeError):
                continue
            data.append((
                series_id, obs.get("date", ""), value,
                obs.get("realtime_start", ""), obs.get("realtime_end", ""), now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO fred_observations "
                "(series_id, date, value, realtime_start, realtime_end, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d FRED observations for %s", len(data), series_id)
        return len(data)

    # --- FRED Yield Curve / Dollar Index ---

    def store_fred_yields(self, series_id: str, observations: List[Dict[str, Any]]) -> int:
        """Store FRED treasury yield / dollar index observations.

        Separate from store_fred_observations to allow distinct querying of
        forex-critical rate series (DGS2, DGS10, T10Y3M, DTWEXBGS) vs general
        macro observations.

        Internally reuses the same fred_observations table with series_id as
        discriminator — no schema duplication.

        Args:
            series_id: FRED series ID (e.g. "DGS2", "DTWEXBGS").
            observations: Raw observations list from FRED API response.

        Returns:
            Number of rows upserted.
        """
        # Delegate to the existing store_fred_observations — same table, same schema
        return self.store_fred_observations(series_id, observations)

    # --- USDA Agricultural Data ---

    def store_usda_data(self, commodity_key: str, records: List[Dict[str, Any]]) -> int:
        """Store USDA PSD commodity supply/demand records.

        Args:
            commodity_key: e.g. "wheat", "corn", "soybeans", "cotton"
            records: Raw record list from USDA PSD API response.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        conn = self._get_conn("usda")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usda_psd (
                commodity_key TEXT,
                market_year TEXT,
                attribute_id INTEGER,
                value REAL,
                unit_description TEXT,
                country_code TEXT,
                ingested_at TEXT,
                PRIMARY KEY (commodity_key, market_year, attribute_id, country_code)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for rec in records:
            market_year = str(rec.get("marketYear", ""))
            attr_id = rec.get("attributeId")
            raw_val = rec.get("value")
            if not market_year or attr_id is None or raw_val is None:
                continue
            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                continue
            data.append((
                commodity_key,
                market_year,
                int(attr_id),
                value,
                rec.get("unitDescription", ""),
                str(rec.get("countryCode", "0000")),
                now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO usda_psd "
                "(commodity_key, market_year, attribute_id, value, unit_description, country_code, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d USDA PSD records for %s", len(data), commodity_key)
        return len(data)
