"""
raw_store.py - Raw Data Storage Layer

Persists ALL data from API calls before processing. Each data domain
gets its own SQLite database in data/market/raw/. WAL mode enables
concurrent read/write for daemon operation.

This is MIDGE's long-term memory for raw market data. The signal
pipeline extracts 1-2 processed values per API call; this layer
preserves the other 95% for future pattern discovery.
"""

import json
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

    # --- StockTwits Messages ---

    def store_stocktwits_messages(self, ticker: str, messages: List[Dict[str, Any]]) -> int:
        """Store raw StockTwits messages with full metadata.

        Args:
            ticker: Stock symbol.
            messages: Raw message list from StockTwits API.

        Returns:
            Number of rows upserted.
        """
        if not messages:
            return 0

        conn = self._get_conn("stocktwits")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stocktwits_messages (
                message_id TEXT PRIMARY KEY,
                ticker TEXT,
                sentiment TEXT,
                body TEXT,
                user_name TEXT,
                created_at TEXT,
                likes INTEGER,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for msg in messages:
            msg_id = str(msg.get("id", ""))
            if not msg_id:
                continue
            entities = msg.get("entities", {})
            sentiment_obj = entities.get("sentiment")
            sentiment = sentiment_obj.get("basic", "") if sentiment_obj else ""
            data.append((
                msg_id, ticker, sentiment,
                msg.get("body", "")[:2000],
                msg.get("user", {}).get("username", ""),
                msg.get("created_at", ""),
                msg.get("likes", {}).get("total", 0) if isinstance(msg.get("likes"), dict) else 0,
                now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO stocktwits_messages "
                "(message_id, ticker, sentiment, body, user_name, created_at, likes, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d StockTwits messages for %s", len(data), ticker)
        return len(data)

    # --- Finnhub (sentiment, earnings, economic — all in one DB) ---

    def store_finnhub_sentiment(self, symbol: str, raw_data: Dict[str, Any]) -> int:
        """Store full Finnhub news-sentiment response blob.

        Args:
            symbol: Ticker symbol.
            raw_data: Full API response dict (buzz, sentiment, companyNewsScore).

        Returns:
            1 on success, 0 on failure.
        """
        if not raw_data:
            return 0

        conn = self._get_conn("finnhub")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finnhub_sentiment (
                symbol TEXT,
                timestamp TEXT,
                bullish_pct REAL,
                bearish_pct REAL,
                news_score REAL,
                buzz_articles INTEGER,
                data_json TEXT,
                ingested_at TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        sentiment = raw_data.get("sentiment", {})
        buzz = raw_data.get("buzz", {})
        try:
            data_json = json.dumps(raw_data, default=str)
        except (TypeError, ValueError):
            data_json = "{}"

        conn.execute(
            "INSERT OR REPLACE INTO finnhub_sentiment "
            "(symbol, timestamp, bullish_pct, bearish_pct, news_score, "
            "buzz_articles, data_json, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                symbol, now[:13],
                sentiment.get("bullishPercent", 0),
                sentiment.get("bearishPercent", 0),
                raw_data.get("companyNewsScore", 0),
                buzz.get("articlesInLastWeek", 0),
                data_json, now,
            ),
        )
        conn.commit()
        return 1

    def store_finnhub_earnings(self, events: List[Dict[str, Any]]) -> int:
        """Store Finnhub earnings calendar events (including quarter/year).

        Args:
            events: Raw earningsCalendar list from Finnhub API.

        Returns:
            Number of rows upserted.
        """
        if not events:
            return 0

        conn = self._get_conn("finnhub")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finnhub_earnings (
                symbol TEXT,
                date TEXT,
                quarter INTEGER,
                year INTEGER,
                eps_estimate REAL,
                eps_actual REAL,
                revenue_estimate REAL,
                revenue_actual REAL,
                hour TEXT,
                ingested_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for ev in events:
            symbol = ev.get("symbol", "")
            date = ev.get("date", "")
            if not symbol or not date:
                continue
            data.append((
                symbol, date,
                ev.get("quarter"), ev.get("year"),
                ev.get("epsEstimate"), ev.get("epsActual"),
                ev.get("revenueEstimate"), ev.get("revenueActual"),
                ev.get("hour", ""), now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finnhub_earnings "
                "(symbol, date, quarter, year, eps_estimate, eps_actual, "
                "revenue_estimate, revenue_actual, hour, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Finnhub earnings events", len(data))
        return len(data)

    def store_finnhub_economic(self, events: List[Dict[str, Any]]) -> int:
        """Store ALL Finnhub economic calendar events (not just US).

        Args:
            events: Raw economicCalendar list from Finnhub API.

        Returns:
            Number of rows upserted.
        """
        if not events:
            return 0

        conn = self._get_conn("finnhub")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finnhub_economic (
                event TEXT,
                country TEXT,
                date TEXT,
                time TEXT,
                impact TEXT,
                actual REAL,
                estimate REAL,
                previous REAL,
                unit TEXT,
                ingested_at TEXT,
                PRIMARY KEY (event, country, date)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for ev in events:
            event_name = ev.get("event", "")
            country = ev.get("country", "")
            date = ev.get("date", "")
            if not event_name or not date:
                continue
            data.append((
                event_name, country, date,
                ev.get("time", ""), ev.get("impact", ""),
                ev.get("actual"), ev.get("estimate"), ev.get("prev"),
                ev.get("unit", ""), now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finnhub_economic "
                "(event, country, date, time, impact, actual, estimate, "
                "previous, unit, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Finnhub economic events (all countries)", len(data))
        return len(data)

    # --- Congressional Trades (House + Senate) ---

    def store_congressional_trades(self, trades: List[Dict[str, Any]]) -> int:
        """Store full congressional trade records.

        Args:
            trades: List of trade dicts with representative, ticker, amounts, etc.

        Returns:
            Number of rows upserted.
        """
        if not trades:
            return 0

        conn = self._get_conn("congressional")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS congressional_trades (
                representative TEXT,
                ticker TEXT,
                transaction_date TEXT,
                transaction_type TEXT,
                amount_range TEXT,
                amount_low REAL,
                amount_high REAL,
                asset_description TEXT,
                district TEXT,
                party TEXT,
                owner TEXT,
                disclosure_date TEXT,
                chamber TEXT,
                ingested_at TEXT,
                PRIMARY KEY (representative, ticker, transaction_date, transaction_type)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for t in trades:
            rep = t.get("representative", "") or t.get("senator", "")
            ticker = t.get("ticker", "")
            tx_date = t.get("transaction_date", "")
            if not rep or not tx_date:
                continue
            data.append((
                rep, ticker, tx_date,
                t.get("transaction_type", ""),
                t.get("amount", "") or t.get("amount_range", ""),
                t.get("amount_low", 0), t.get("amount_high", 0),
                t.get("asset_description", "")[:500],
                t.get("district", "") or t.get("office", ""),
                t.get("party", ""),
                t.get("owner", ""),
                t.get("disclosure_date", ""),
                t.get("chamber", "house"),
                now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO congressional_trades "
                "(representative, ticker, transaction_date, transaction_type, "
                "amount_range, amount_low, amount_high, asset_description, "
                "district, party, owner, disclosure_date, chamber, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d congressional trades", len(data))
        return len(data)

    # --- Congress.gov Bills ---

    def store_congress_bills(self, bills: List[Dict[str, Any]]) -> int:
        """Store full bill metadata from Congress.gov API.

        Stores sponsors, committees, cosponsors count — data currently discarded.

        Args:
            bills: Raw bill dicts from Congress.gov API response.

        Returns:
            Number of rows upserted.
        """
        if not bills:
            return 0

        conn = self._get_conn("legislation")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS congress_bills (
                congress INTEGER,
                bill_type TEXT,
                bill_number TEXT,
                title TEXT,
                latest_action TEXT,
                action_date TEXT,
                policy_area TEXT,
                sponsors_json TEXT,
                url TEXT,
                introduced_date TEXT,
                update_date TEXT,
                origin_chamber TEXT,
                ingested_at TEXT,
                PRIMARY KEY (congress, bill_type, bill_number)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for bill in bills:
            congress = bill.get("congress", 0)
            bill_type = bill.get("type", "")
            bill_number = str(bill.get("number", ""))
            if not bill_number:
                continue

            latest = bill.get("latestAction", {})
            sponsors = bill.get("sponsors", [])
            try:
                sponsors_json = json.dumps(sponsors, default=str) if sponsors else "[]"
            except (TypeError, ValueError):
                sponsors_json = "[]"

            data.append((
                congress, bill_type, bill_number,
                bill.get("title", "")[:500],
                latest.get("text", "")[:500],
                latest.get("actionDate", ""),
                bill.get("policyArea", {}).get("name", "") if isinstance(bill.get("policyArea"), dict) else "",
                sponsors_json,
                bill.get("url", ""),
                bill.get("introducedDate", ""),
                bill.get("updateDate", ""),
                bill.get("originChamber", ""),
                now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO congress_bills "
                "(congress, bill_type, bill_number, title, latest_action, "
                "action_date, policy_area, sponsors_json, url, introduced_date, "
                "update_date, origin_chamber, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Congress bills", len(data))
        return len(data)

    # --- FINRA Short Volume ---

    def store_finra_short_volume(self, records: List[Any]) -> int:
        """Store FINRA daily short volume records.

        Args:
            records: List of ShortInterestData objects or dicts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        conn = self._get_conn("finra")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finra_short_volume (
                symbol TEXT,
                date TEXT,
                short_volume INTEGER,
                short_exempt_volume INTEGER,
                total_volume INTEGER,
                short_ratio REAL,
                speculative_short_ratio REAL,
                ingested_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for r in records:
            if hasattr(r, "symbol"):
                data.append((
                    r.symbol, r.date, r.short_volume,
                    getattr(r, "short_exempt_volume", 0),
                    r.total_volume, r.short_ratio,
                    getattr(r, "speculative_short_ratio", 0.0), now,
                ))
            elif isinstance(r, dict):
                data.append((
                    r.get("symbol", ""), r.get("date", ""),
                    r.get("short_volume", 0), r.get("short_exempt_volume", 0),
                    r.get("total_volume", 0), r.get("short_ratio", 0),
                    r.get("speculative_short_ratio", 0.0), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finra_short_volume "
                "(symbol, date, short_volume, short_exempt_volume, total_volume, "
                "short_ratio, speculative_short_ratio, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d FINRA short volume records", len(data))
        return len(data)

    # --- SEC EDGAR Form 4 + Form 8-K ---

    def store_sec_form4(self, trades: List[Any]) -> int:
        """Store full SEC Form 4 insider trade records.

        Args:
            trades: List of InsiderTrade dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not trades:
            return 0

        conn = self._get_conn("sec_edgar")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS form4_trades (
                filing_id TEXT PRIMARY KEY,
                ticker TEXT,
                company_cik TEXT,
                insider_name TEXT,
                insider_title TEXT,
                transaction_date TEXT,
                transaction_type TEXT,
                shares INTEGER,
                price_per_share REAL,
                total_value REAL,
                shares_after REAL,
                is_derivative INTEGER,
                filing_date TEXT,
                form_type TEXT,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for t in trades:
            if hasattr(t, "filing_id"):
                # InsiderTrade dataclass
                filing_id = getattr(t, "filing_id", "") or getattr(t, "accession_number", "")
                data.append((
                    filing_id or f"{t.ticker}:{t.transaction_date}:{t.insider_name}",
                    getattr(t, "ticker", ""),
                    getattr(t, "company_cik", ""),
                    getattr(t, "insider_name", ""),
                    getattr(t, "insider_title", ""),
                    getattr(t, "transaction_date", ""),
                    getattr(t, "transaction_type", ""),
                    getattr(t, "shares", 0),
                    getattr(t, "price_per_share", 0.0),
                    getattr(t, "total_value", 0.0),
                    getattr(t, "shares_after", 0.0),
                    int(getattr(t, "is_derivative", False)),
                    getattr(t, "filing_date", ""),
                    getattr(t, "form_type", "4"),
                    now,
                ))
            elif isinstance(t, dict):
                fid = t.get("filing_id") or t.get("accession_number") or f"{t.get('ticker','')}:{t.get('transaction_date','')}:{t.get('insider_name','')}"
                data.append((
                    fid,
                    t.get("ticker", ""), t.get("company_cik", ""),
                    t.get("insider_name", ""), t.get("insider_title", ""),
                    t.get("transaction_date", ""), t.get("transaction_type", ""),
                    t.get("shares", 0), t.get("price_per_share", 0.0),
                    t.get("total_value", 0.0), t.get("shares_after", 0.0),
                    int(t.get("is_derivative", False)),
                    t.get("filing_date", ""), t.get("form_type", "4"), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO form4_trades "
                "(filing_id, ticker, company_cik, insider_name, insider_title, "
                "transaction_date, transaction_type, shares, price_per_share, "
                "total_value, shares_after, is_derivative, filing_date, form_type, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d SEC Form 4 trades", len(data))
        return len(data)

    # --- Massive/Polygon.io grouped daily bars ---

    def store_massive_bars(self, bars: List[Any]) -> int:
        """Store Polygon.io grouped daily OHLCV bars.

        Args:
            bars: List of TickerBar dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not bars:
            return 0

        conn = self._get_conn("massive")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_bars (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                vwap REAL,
                transactions INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (ticker, date)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for b in bars:
            if hasattr(b, "ticker"):
                data.append((
                    b.ticker, b.date, b.open, b.high, b.low, b.close,
                    b.volume, b.vwap, b.transactions, now,
                ))
            elif isinstance(b, dict):
                data.append((
                    b.get("ticker", ""), b.get("date", ""),
                    b.get("open", 0), b.get("high", 0), b.get("low", 0), b.get("close", 0),
                    b.get("volume", 0), b.get("vwap", 0), b.get("transactions", 0), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO daily_bars "
                "(ticker, date, open, high, low, close, volume, vwap, transactions, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Massive daily bars", len(data))
        return len(data)

    # --- CoinGecko full market data ---

    def store_coingecko_prices(self, coins: List[Any]) -> int:
        """Store full CoinGecko market data including ATH/ATL/supply.

        Args:
            coins: List of CryptoPrice dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not coins:
            return 0

        conn = self._get_conn("crypto")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coingecko_prices (
                coin_id TEXT,
                timestamp TEXT,
                symbol TEXT,
                price_usd REAL,
                volume_24h REAL,
                change_24h_pct REAL,
                change_7d_pct REAL,
                market_cap REAL,
                last_updated TEXT,
                ingested_at TEXT,
                PRIMARY KEY (coin_id, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for c in coins:
            if hasattr(c, "coin_id"):
                data.append((
                    c.coin_id, now[:13],
                    getattr(c, "symbol", ""),
                    getattr(c, "price_usd", 0.0),
                    getattr(c, "volume_24h", 0.0),
                    getattr(c, "change_24h_pct", 0.0),
                    getattr(c, "change_7d_pct", 0.0),
                    getattr(c, "market_cap", 0.0),
                    getattr(c, "last_updated", ""),
                    now,
                ))
            elif isinstance(c, dict):
                data.append((
                    c.get("coin_id", ""), now[:13],
                    c.get("symbol", ""), c.get("price_usd", 0.0),
                    c.get("volume_24h", 0.0), c.get("change_24h_pct", 0.0),
                    c.get("change_7d_pct", 0.0), c.get("market_cap", 0.0),
                    c.get("last_updated", ""), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO coingecko_prices "
                "(coin_id, timestamp, symbol, price_usd, volume_24h, "
                "change_24h_pct, change_7d_pct, market_cap, last_updated, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d CoinGecko price rows", len(data))
        return len(data)

    # --- CoinCap asset data ---

    def store_coincap_assets(self, assets: List[Any]) -> int:
        """Store CoinCap asset data including supply and VWAP.

        Args:
            assets: List of CryptoAsset dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not assets:
            return 0

        conn = self._get_conn("crypto")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS coincap_assets (
                asset_id TEXT,
                timestamp TEXT,
                symbol TEXT,
                name TEXT,
                rank INTEGER,
                price_usd REAL,
                volume_24h_usd REAL,
                change_24h_pct REAL,
                market_cap_usd REAL,
                supply REAL,
                max_supply REAL,
                vwap_24h REAL,
                ingested_at TEXT,
                PRIMARY KEY (asset_id, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for a in assets:
            if hasattr(a, "asset_id"):
                data.append((
                    a.asset_id, now[:13],
                    getattr(a, "symbol", ""), getattr(a, "name", ""),
                    getattr(a, "rank", 0), getattr(a, "price_usd", 0.0),
                    getattr(a, "volume_24h_usd", 0.0), getattr(a, "change_24h_pct", 0.0),
                    getattr(a, "market_cap_usd", 0.0), getattr(a, "supply", 0.0),
                    getattr(a, "max_supply", None), getattr(a, "vwap_24h", None),
                    now,
                ))
            elif isinstance(a, dict):
                data.append((
                    a.get("asset_id", ""), now[:13],
                    a.get("symbol", ""), a.get("name", ""),
                    a.get("rank", 0), a.get("price_usd", 0.0),
                    a.get("volume_24h_usd", 0.0), a.get("change_24h_pct", 0.0),
                    a.get("market_cap_usd", 0.0), a.get("supply", 0.0),
                    a.get("max_supply"), a.get("vwap_24h"), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO coincap_assets "
                "(asset_id, timestamp, symbol, name, rank, price_usd, volume_24h_usd, "
                "change_24h_pct, market_cap_usd, supply, max_supply, vwap_24h, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d CoinCap asset rows", len(data))
        return len(data)

    # --- OpenInsider purchases ---

    def store_openinsider_purchases(self, purchases: List[Any]) -> int:
        """Store OpenInsider pre-filtered insider purchase records.

        Args:
            purchases: List of InsiderPurchase dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not purchases:
            return 0

        conn = self._get_conn("openinsider")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_purchases (
                filing_date TEXT,
                trade_date TEXT,
                ticker TEXT,
                company_name TEXT,
                insider_name TEXT,
                title TEXT,
                trade_type TEXT,
                price REAL,
                quantity INTEGER,
                owned INTEGER,
                delta_owned_pct REAL,
                value REAL,
                ingested_at TEXT,
                PRIMARY KEY (filing_date, ticker, insider_name, trade_type)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for p in purchases:
            if hasattr(p, "ticker"):
                data.append((
                    getattr(p, "filing_date", ""), getattr(p, "trade_date", ""),
                    getattr(p, "ticker", ""), getattr(p, "company_name", ""),
                    getattr(p, "insider_name", ""), getattr(p, "title", ""),
                    getattr(p, "trade_type", ""),
                    getattr(p, "price", 0.0), getattr(p, "quantity", 0),
                    getattr(p, "owned", 0), getattr(p, "delta_owned_pct", 0.0),
                    getattr(p, "value", 0.0), now,
                ))
            elif isinstance(p, dict):
                data.append((
                    p.get("filing_date", ""), p.get("trade_date", ""),
                    p.get("ticker", ""), p.get("company_name", ""),
                    p.get("insider_name", ""), p.get("title", ""),
                    p.get("trade_type", ""),
                    p.get("price", 0.0), p.get("quantity", 0),
                    p.get("owned", 0), p.get("delta_owned_pct", 0.0),
                    p.get("value", 0.0), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO insider_purchases "
                "(filing_date, trade_date, ticker, company_name, insider_name, "
                "title, trade_type, price, quantity, owned, delta_owned_pct, value, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d OpenInsider purchases", len(data))
        return len(data)

    # --- FinViz ---

    def store_finviz_insider_trades(self, trades: List[Any]) -> int:
        """Store FinViz insider trade records.

        Args:
            trades: List of FinVizInsiderTrade dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not trades:
            return 0

        conn = self._get_conn("finviz")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finviz_insider_trades (
                ticker TEXT,
                date TEXT,
                owner_name TEXT,
                relationship TEXT,
                transaction_type TEXT,
                shares_traded INTEGER,
                value REAL,
                shares_owned INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (ticker, date, owner_name, transaction_type)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for t in trades:
            if hasattr(t, "ticker"):
                data.append((
                    getattr(t, "ticker", ""), getattr(t, "date", ""),
                    getattr(t, "owner_name", ""), getattr(t, "relationship", ""),
                    getattr(t, "transaction_type", ""),
                    getattr(t, "shares_traded", 0), getattr(t, "value", 0.0),
                    getattr(t, "shares_owned", 0), now,
                ))
            elif isinstance(t, dict):
                data.append((
                    t.get("ticker", ""), t.get("date", ""),
                    t.get("owner_name", ""), t.get("relationship", ""),
                    t.get("transaction_type", ""),
                    t.get("shares_traded", 0), t.get("value", 0.0),
                    t.get("shares_owned", 0), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finviz_insider_trades "
                "(ticker, date, owner_name, relationship, transaction_type, "
                "shares_traded, value, shares_owned, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d FinViz insider trades", len(data))
        return len(data)

    def store_finviz_unusual_volume(self, items: List[Any]) -> int:
        """Store FinViz unusual volume screening results.

        Args:
            items: List of UnusualVolume dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not items:
            return 0

        conn = self._get_conn("finviz")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finviz_unusual_volume (
                ticker TEXT,
                timestamp TEXT,
                company TEXT,
                sector TEXT,
                price REAL,
                change_pct REAL,
                volume INTEGER,
                avg_volume INTEGER,
                volume_ratio REAL,
                ingested_at TEXT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for item in items:
            if hasattr(item, "ticker"):
                data.append((
                    getattr(item, "ticker", ""), now[:13],
                    getattr(item, "company", ""), getattr(item, "sector", ""),
                    getattr(item, "price", 0.0), getattr(item, "change_pct", 0.0),
                    getattr(item, "volume", 0), getattr(item, "avg_volume", 0),
                    getattr(item, "volume_ratio", 0.0), now,
                ))
            elif isinstance(item, dict):
                data.append((
                    item.get("ticker", ""), now[:13],
                    item.get("company", ""), item.get("sector", ""),
                    item.get("price", 0.0), item.get("change_pct", 0.0),
                    item.get("volume", 0), item.get("avg_volume", 0),
                    item.get("volume_ratio", 0.0), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finviz_unusual_volume "
                "(ticker, timestamp, company, sector, price, change_pct, "
                "volume, avg_volume, volume_ratio, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d FinViz unusual volume rows", len(data))
        return len(data)

    # --- EDGAR Enhanced (13F/13D) ---

    def store_edgar_filings(self, filings: List[Any]) -> int:
        """Store SEC 13D/13F filing metadata.

        Args:
            filings: List of InstitutionalHolding or ActivistFiling dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not filings:
            return 0

        conn = self._get_conn("sec_edgar")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS institutional_filings (
                filer_name TEXT,
                filer_cik TEXT,
                form_type TEXT,
                filing_date TEXT,
                ticker TEXT,
                subject_company TEXT,
                shares INTEGER,
                value_usd REAL,
                period_of_report TEXT,
                percent_owned REAL,
                purpose TEXT,
                ingested_at TEXT,
                PRIMARY KEY (filer_cik, form_type, filing_date, ticker)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for f in filings:
            if hasattr(f, "filer_cik"):
                form_type = getattr(f, "form_type", "13F")
                ticker = getattr(f, "ticker", "") or getattr(f, "subject_ticker", "")
                data.append((
                    getattr(f, "filer_name", ""), getattr(f, "filer_cik", ""),
                    form_type, getattr(f, "filing_date", ""),
                    ticker,
                    getattr(f, "company_name", "") or getattr(f, "subject_company", ""),
                    getattr(f, "shares", 0),
                    getattr(f, "value_usd", 0.0),
                    getattr(f, "period_of_report", ""),
                    getattr(f, "percent_owned", 0.0),
                    getattr(f, "purpose", "")[:500],
                    now,
                ))
            elif isinstance(f, dict):
                data.append((
                    f.get("filer_name", ""), f.get("filer_cik", ""),
                    f.get("form_type", "13F"), f.get("filing_date", ""),
                    f.get("ticker", "") or f.get("subject_ticker", ""),
                    f.get("company_name", "") or f.get("subject_company", ""),
                    f.get("shares", 0), f.get("value_usd", 0.0),
                    f.get("period_of_report", ""),
                    f.get("percent_owned", 0.0),
                    str(f.get("purpose", ""))[:500],
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO institutional_filings "
                "(filer_name, filer_cik, form_type, filing_date, ticker, "
                "subject_company, shares, value_usd, period_of_report, "
                "percent_owned, purpose, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d SEC institutional filings", len(data))
        return len(data)

    # --- FinnhubWebSocket trade ticks ---

    def store_finnhub_ticks(self, ticks: List[Dict[str, Any]]) -> int:
        """Store Finnhub WebSocket real-time trade tick data.

        Args:
            ticks: List of trade tick dicts with keys: symbol, price, volume, timestamp.

        Returns:
            Number of rows upserted.
        """
        if not ticks:
            return 0

        conn = self._get_conn("finnhub")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finnhub_ticks (
                symbol TEXT,
                timestamp_ms INTEGER,
                price REAL,
                volume REAL,
                ingested_at TEXT,
                PRIMARY KEY (symbol, timestamp_ms)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for t in ticks:
            symbol = t.get("s", "") or t.get("symbol", "")
            ts_ms = t.get("t", 0) or t.get("timestamp_ms", 0)
            price = float(t.get("p", 0) or t.get("price", 0))
            volume = float(t.get("v", 0) or t.get("volume", 0))
            if symbol and ts_ms:
                data.append((symbol, ts_ms, price, volume, now))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finnhub_ticks "
                "(symbol, timestamp_ms, price, volume, ingested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Finnhub tick records", len(data))
        return len(data)

    # --- ApeWisdom social sentiment ---

    def store_apewisdom_sentiment(self, records: List[Any]) -> int:
        """Store ApeWisdom Reddit social sentiment records.

        Args:
            records: List of SocialSentiment dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        conn = self._get_conn("social")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS apewisdom_sentiment (
                ticker TEXT,
                timestamp TEXT,
                mentions_24h INTEGER,
                mentions_prior_24h INTEGER,
                upvotes INTEGER,
                rank INTEGER,
                mention_change REAL,
                source_subreddit TEXT,
                ingested_at TEXT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for r in records:
            if hasattr(r, "ticker"):
                data.append((
                    getattr(r, "ticker", ""), now[:13],
                    getattr(r, "mentions_24h", 0),
                    getattr(r, "mentions_prior_24h", 0),
                    getattr(r, "upvotes", 0),
                    getattr(r, "rank", 0),
                    getattr(r, "mention_change", 0.0),
                    getattr(r, "source_subreddit", "wallstreetbets"),
                    now,
                ))
            elif isinstance(r, dict):
                data.append((
                    r.get("ticker", ""), now[:13],
                    r.get("mentions_24h", 0), r.get("mentions_prior_24h", 0),
                    r.get("upvotes", 0), r.get("rank", 0),
                    r.get("mention_change", 0.0),
                    r.get("source_subreddit", "wallstreetbets"),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO apewisdom_sentiment "
                "(ticker, timestamp, mentions_24h, mentions_prior_24h, upvotes, "
                "rank, mention_change, source_subreddit, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d ApeWisdom sentiment rows", len(data))
        return len(data)

    # --- Job Tracker (hiring blitz signals) ---

    def store_job_postings(self, signals: List[Any]) -> int:
        """Store full job posting records from hiring blitz detection.

        Args:
            signals: List of HiringSignal dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not signals:
            return 0

        conn = self._get_conn("jobs")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_postings (
                company_name TEXT,
                timestamp TEXT,
                ticker TEXT,
                jobs_24h INTEGER,
                jobs_7d INTEGER,
                jobs_30d INTEGER,
                is_spike INTEGER,
                spike_ratio REAL,
                top_roles_json TEXT,
                ingested_at TEXT,
                PRIMARY KEY (company_name, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for s in signals:
            if hasattr(s, "company_name"):
                top_roles = getattr(s, "top_roles", [])
                try:
                    roles_json = json.dumps(top_roles, default=str) if top_roles else "[]"
                except (TypeError, ValueError):
                    roles_json = "[]"
                data.append((
                    getattr(s, "company_name", ""), now[:13],
                    getattr(s, "ticker", "") or "",
                    getattr(s, "jobs_24h", 0),
                    getattr(s, "jobs_7d", 0),
                    getattr(s, "jobs_30d", 0),
                    int(getattr(s, "is_spike", False)),
                    getattr(s, "spike_ratio", 0.0),
                    roles_json, now,
                ))
            elif isinstance(s, dict):
                roles_json = json.dumps(s.get("top_roles", []), default=str)
                data.append((
                    s.get("company_name", ""), now[:13],
                    s.get("ticker", "") or "",
                    s.get("jobs_24h", 0), s.get("jobs_7d", 0), s.get("jobs_30d", 0),
                    int(s.get("is_spike", False)), s.get("spike_ratio", 0.0),
                    roles_json, now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO job_postings "
                "(company_name, timestamp, ticker, jobs_24h, jobs_7d, jobs_30d, "
                "is_spike, spike_ratio, top_roles_json, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d job posting signals", len(data))
        return len(data)

    # --- SAM.gov opportunities ---

    def store_sam_opportunities(self, opportunities: List[Any]) -> int:
        """Store SAM.gov federal contract opportunity records.

        Args:
            opportunities: List of ContractOpportunity dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not opportunities:
            return 0

        conn = self._get_conn("contracts")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sam_opportunities (
                notice_id TEXT PRIMARY KEY,
                title TEXT,
                solicitation_number TEXT,
                department TEXT,
                agency TEXT,
                office TEXT,
                naics_code TEXT,
                set_aside TEXT,
                type_of_contract TEXT,
                place_of_performance TEXT,
                estimated_value REAL,
                award_date TEXT,
                posted_date TEXT,
                response_deadline TEXT,
                active INTEGER,
                contract_type TEXT,
                url TEXT,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for opp in opportunities:
            if hasattr(opp, "notice_id"):
                data.append((
                    getattr(opp, "notice_id", ""),
                    getattr(opp, "title", "")[:500],
                    getattr(opp, "solicitation_number", ""),
                    getattr(opp, "department", ""),
                    getattr(opp, "agency", ""),
                    getattr(opp, "office", ""),
                    getattr(opp, "naics_code", ""),
                    getattr(opp, "set_aside", ""),
                    getattr(opp, "type_of_contract", ""),
                    getattr(opp, "place_of_performance", ""),
                    getattr(opp, "estimated_value", 0.0),
                    getattr(opp, "award_date", ""),
                    getattr(opp, "posted_date", ""),
                    getattr(opp, "response_deadline", ""),
                    int(getattr(opp, "active", True)),
                    getattr(opp, "contract_type", ""),
                    getattr(opp, "url", ""),
                    now,
                ))
            elif isinstance(opp, dict):
                data.append((
                    opp.get("notice_id", ""),
                    str(opp.get("title", ""))[:500],
                    opp.get("solicitation_number", ""),
                    opp.get("department", ""), opp.get("agency", ""),
                    opp.get("office", ""), opp.get("naics_code", ""),
                    opp.get("set_aside", ""), opp.get("type_of_contract", ""),
                    opp.get("place_of_performance", ""),
                    opp.get("estimated_value", 0.0),
                    opp.get("award_date", ""), opp.get("posted_date", ""),
                    opp.get("response_deadline", ""),
                    int(opp.get("active", True)),
                    opp.get("contract_type", ""), opp.get("url", ""),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO sam_opportunities "
                "(notice_id, title, solicitation_number, department, agency, office, "
                "naics_code, set_aside, type_of_contract, place_of_performance, "
                "estimated_value, award_date, posted_date, response_deadline, "
                "active, contract_type, url, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d SAM.gov opportunities", len(data))
        return len(data)

    # --- Yahoo Finance RSS ---

    def store_yahoo_headlines(self, ticker: str, entries: list) -> int:
        """Store Yahoo Finance RSS headline entries for a ticker.

        Deduplicates by (ticker, title, published_at) so re-fetching the
        same feed window is safe. Stores the full title + summary for
        future NLP/sentiment analysis beyond keyword matching.

        Args:
            ticker: Stock symbol (uppercase).
            entries: List of feedparser entry objects or dicts.

        Returns:
            Number of rows upserted.
        """
        if not entries:
            return 0

        conn = self._get_conn("yahoo_rss")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS yahoo_headlines (
                ticker TEXT,
                title TEXT,
                published_at TEXT,
                link TEXT,
                summary TEXT,
                ingested_at TEXT,
                PRIMARY KEY (ticker, title, published_at)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for entry in entries:
            # Support both feedparser Entry objects and plain dicts
            if hasattr(entry, "get"):
                title = (entry.get("title") or "")[:500]
                published = (entry.get("published") or "")[:100]
                link = (entry.get("link") or "")[:500]
                summary = (entry.get("summary") or "")[:1000]
            else:
                title = (getattr(entry, "title", "") or "")[:500]
                published = (getattr(entry, "published", "") or "")[:100]
                link = (getattr(entry, "link", "") or "")[:500]
                summary = (getattr(entry, "summary", "") or "")[:1000]

            if not title:
                continue

            data.append((ticker.upper(), title, published, link, summary, now))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO yahoo_headlines "
                "(ticker, title, published_at, link, summary, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Yahoo RSS headlines for %s", len(data), ticker)
        return len(data)
