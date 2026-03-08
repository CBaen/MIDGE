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
