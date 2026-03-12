"""
raw_store_operational.py - Operational data storage mixin.

Handles Polygon.io/Massive daily bars, CoinGecko prices, CoinCap assets,
job postings (hiring blitz), and SAM.gov contract opportunities.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Any

logger = logging.getLogger(__name__)


class RawStoreOperationalMixin:
    """Mixin for storing operational market data: bars, crypto, jobs, contracts."""

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

    # --- Polygon.io ticker reference data ---

    def store_polygon_ticker_details(self, details: List[Any]) -> int:
        """Store Polygon.io /v3/reference/tickers/{ticker} reference metadata.

        Args:
            details: List of TickerDetails dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not details:
            return 0

        conn = self._get_conn("massive")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS polygon_ticker_details (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                market TEXT,
                locale TEXT,
                primary_exchange TEXT,
                type TEXT,
                active INTEGER,
                currency_name TEXT,
                market_cap REAL,
                sic_code TEXT,
                sic_description TEXT,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for d in details:
            if hasattr(d, "ticker"):
                data.append((
                    getattr(d, "ticker", ""),
                    getattr(d, "name", ""),
                    getattr(d, "market", ""),
                    getattr(d, "locale", ""),
                    getattr(d, "primary_exchange", ""),
                    getattr(d, "type", ""),
                    int(getattr(d, "active", True)),
                    getattr(d, "currency_name", ""),
                    getattr(d, "market_cap", 0.0),
                    getattr(d, "sic_code", ""),
                    getattr(d, "sic_description", ""),
                    now,
                ))
            elif isinstance(d, dict):
                data.append((
                    d.get("ticker", ""),
                    d.get("name", ""),
                    d.get("market", ""),
                    d.get("locale", ""),
                    d.get("primary_exchange", ""),
                    d.get("type", ""),
                    int(d.get("active", True)),
                    d.get("currency_name", ""),
                    d.get("market_cap", 0.0),
                    d.get("sic_code", ""),
                    d.get("sic_description", ""),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO polygon_ticker_details "
                "(ticker, name, market, locale, primary_exchange, type, active, "
                "currency_name, market_cap, sic_code, sic_description, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Polygon ticker detail rows", len(data))
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
                high_24h REAL,
                low_24h REAL,
                ath_change_percentage REAL,
                circulating_supply REAL,
                total_supply REAL,
                ingested_at TEXT,
                PRIMARY KEY (coin_id, timestamp)
            )
        """)
        # Idempotent migration for pre-existing tables missing the new columns
        for col, col_type in [
            ("high_24h", "REAL"), ("low_24h", "REAL"),
            ("ath_change_percentage", "REAL"), ("circulating_supply", "REAL"),
            ("total_supply", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE coingecko_prices ADD COLUMN {col} {col_type}")
                conn.commit()
            except Exception:
                pass  # Column already exists

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
                    getattr(c, "high_24h", None),
                    getattr(c, "low_24h", None),
                    getattr(c, "ath_change_percentage", None),
                    getattr(c, "circulating_supply", None),
                    getattr(c, "total_supply", None),
                    now,
                ))
            elif isinstance(c, dict):
                data.append((
                    c.get("coin_id", ""), now[:13],
                    c.get("symbol", ""), c.get("price_usd", 0.0),
                    c.get("volume_24h", 0.0), c.get("change_24h_pct", 0.0),
                    c.get("change_7d_pct", 0.0), c.get("market_cap", 0.0),
                    c.get("last_updated", ""),
                    c.get("high_24h"), c.get("low_24h"),
                    c.get("ath_change_percentage"), c.get("circulating_supply"),
                    c.get("total_supply"), now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO coingecko_prices "
                "(coin_id, timestamp, symbol, price_usd, volume_24h, "
                "change_24h_pct, change_7d_pct, market_cap, last_updated, "
                "high_24h, low_24h, ath_change_percentage, circulating_supply, "
                "total_supply, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d CoinGecko price rows", len(data))
        return len(data)

    def get_coingecko_history(
        self, symbol: str = None, lookback_days: int = 7
    ) -> List[dict]:
        """Retrieve CoinGecko price history.

        Args:
            symbol: Filter by coin symbol e.g. "BTC" (None = all coins).
            lookback_days: How many days of history to return.

        Returns:
            List of dicts with keys: coin_id, symbol, price_usd, volume_24h,
            change_24h_pct, change_7d_pct, market_cap, timestamp.
            Sorted oldest-first. Empty list on error.
        """
        try:
            conn = self._get_conn("crypto")
            params: list = [f"-{lookback_days} days"]
            where = "WHERE ingested_at >= datetime('now', ?)"
            if symbol:
                where += " AND LOWER(symbol) = ?"
                params.append(symbol.lower())
            cursor = conn.execute(
                f"""
                SELECT coin_id, symbol, price_usd, volume_24h, change_24h_pct,
                       change_7d_pct, market_cap, timestamp
                FROM coingecko_prices
                {where}
                ORDER BY timestamp ASC
                """,
                params,
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_coingecko_history failed: %s", exc)
            return []

        return [
            {
                "coin_id": r[0], "symbol": r[1], "price_usd": r[2],
                "volume_24h": r[3], "change_24h_pct": r[4],
                "change_7d_pct": r[5], "market_cap": r[6], "timestamp": r[7],
            }
            for r in rows
        ]

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
                description TEXT,
                ingested_at TEXT
            )
        """)
        # Migrate existing databases that predate the description column.
        try:
            conn.execute("ALTER TABLE sam_opportunities ADD COLUMN description TEXT DEFAULT ''")
            conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore

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
                    getattr(opp, "description", "")[:1000],
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
                    str(opp.get("description", ""))[:1000],
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO sam_opportunities "
                "(notice_id, title, solicitation_number, department, agency, office, "
                "naics_code, set_aside, type_of_contract, place_of_performance, "
                "estimated_value, award_date, posted_date, response_deadline, "
                "active, contract_type, url, description, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d SAM.gov opportunities", len(data))
        return len(data)

    def get_sam_opportunities(self, lookback_days: int = 30) -> List[dict]:
        """Retrieve SAM.gov opportunities posted within the last N days.

        Args:
            lookback_days: How many days of history to return (default 30).

        Returns:
            List of dicts with all opportunity fields, sorted newest-first by
            posted_date. Empty list if table doesn't exist or no data.
        """
        try:
            conn = self._get_conn("contracts")
            cursor = conn.execute(
                """
                SELECT notice_id, title, solicitation_number, department, agency,
                       office, naics_code, set_aside, type_of_contract,
                       place_of_performance, estimated_value, award_date,
                       posted_date, response_deadline, active, contract_type,
                       url, description, ingested_at
                FROM sam_opportunities
                WHERE posted_date >= date('now', ?)
                ORDER BY posted_date DESC
                """,
                (f"-{lookback_days} days",),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_sam_opportunities failed: %s", exc)
            return []

        keys = [
            "notice_id", "title", "solicitation_number", "department", "agency",
            "office", "naics_code", "set_aside", "type_of_contract",
            "place_of_performance", "estimated_value", "award_date",
            "posted_date", "response_deadline", "active", "contract_type",
            "url", "description", "ingested_at",
        ]
        return [dict(zip(keys, r)) for r in rows]

    # --- USASpending.gov contract awards ---

    def store_usaspending_contracts(self, contracts: List[Any]) -> int:
        """Store USASpending.gov government contract award records.

        Args:
            contracts: List of GovernmentContract dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not contracts:
            return 0

        conn = self._get_conn("contracts")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usaspending_contracts (
                award_id TEXT PRIMARY KEY,
                recipient_name TEXT,
                recipient_duns TEXT,
                recipient_location TEXT,
                award_amount REAL,
                award_date TEXT,
                award_type TEXT,
                description TEXT,
                naics_code TEXT,
                naics_description TEXT,
                awarding_agency TEXT,
                awarding_sub_agency TEXT,
                funding_agency TEXT,
                start_date TEXT,
                end_date TEXT,
                ingested_at TEXT
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for c in contracts:
            if hasattr(c, "award_id"):
                data.append((
                    getattr(c, "award_id", "") or "",
                    getattr(c, "recipient_name", "")[:500],
                    getattr(c, "recipient_duns", ""),
                    getattr(c, "recipient_location", ""),
                    float(getattr(c, "award_amount", 0.0)),
                    getattr(c, "award_date", ""),
                    getattr(c, "award_type", ""),
                    getattr(c, "description", "")[:1000],
                    getattr(c, "naics_code", "") or "",
                    getattr(c, "naics_description", "") or "",
                    getattr(c, "awarding_agency", "") or "",
                    getattr(c, "awarding_sub_agency", "") or "",
                    getattr(c, "funding_agency", "") or "",
                    getattr(c, "start_date", ""),
                    getattr(c, "end_date", ""),
                    now,
                ))
            elif isinstance(c, dict):
                data.append((
                    c.get("award_id", "") or "",
                    str(c.get("recipient_name", ""))[:500],
                    c.get("recipient_duns", ""),
                    c.get("recipient_location", ""),
                    float(c.get("award_amount", 0.0)),
                    c.get("award_date", ""),
                    c.get("award_type", ""),
                    str(c.get("description", ""))[:1000],
                    c.get("naics_code", "") or "",
                    c.get("naics_description", "") or "",
                    c.get("awarding_agency", "") or "",
                    c.get("awarding_sub_agency", "") or "",
                    c.get("funding_agency", "") or "",
                    c.get("start_date", ""),
                    c.get("end_date", ""),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO usaspending_contracts "
                "(award_id, recipient_name, recipient_duns, recipient_location, "
                "award_amount, award_date, award_type, description, naics_code, "
                "naics_description, awarding_agency, awarding_sub_agency, "
                "funding_agency, start_date, end_date, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d USASpending contracts", len(data))
        return len(data)

    def get_usaspending_contracts(self, lookback_days: int = 30) -> List[dict]:
        """Retrieve USASpending contracts awarded within the last N days.

        Args:
            lookback_days: How many days of history to return (default 30).

        Returns:
            List of dicts with all contract fields, sorted newest-first by
            award_date. Empty list if table doesn't exist or no data.
        """
        try:
            conn = self._get_conn("contracts")
            cursor = conn.execute(
                """
                SELECT award_id, recipient_name, recipient_duns, recipient_location,
                       award_amount, award_date, award_type, description, naics_code,
                       naics_description, awarding_agency, awarding_sub_agency,
                       funding_agency, start_date, end_date, ingested_at
                FROM usaspending_contracts
                WHERE award_date >= date('now', ?)
                ORDER BY award_date DESC
                """,
                (f"-{lookback_days} days",),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_usaspending_contracts failed: %s", exc)
            return []

        keys = [
            "award_id", "recipient_name", "recipient_duns", "recipient_location",
            "award_amount", "award_date", "award_type", "description", "naics_code",
            "naics_description", "awarding_agency", "awarding_sub_agency",
            "funding_agency", "start_date", "end_date", "ingested_at",
        ]
        return [dict(zip(keys, r)) for r in rows]

    # --- Binance Futures Funding Rates ---

    def store_binance_funding(self, symbol: str, raw_response: List[Any]) -> int:
        """Store Binance perpetual futures funding rate records.

        Raw response is the JSON list returned by /fapi/v1/fundingRate.
        Each entry contains fundingRate, fundingTime, and optionally markPrice.

        Args:
            symbol: Perpetual futures symbol, e.g. "BTCUSDT".
            raw_response: List of dicts from Binance API (limit=1 typical,
                          but may contain multiple historical records).

        Returns:
            Number of rows upserted.
        """
        if not raw_response:
            return 0

        conn = self._get_conn("crypto")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS binance_funding_rates (
                symbol TEXT,
                funding_time TEXT,
                funding_rate REAL,
                mark_price REAL,
                ingested_at TEXT,
                PRIMARY KEY (symbol, funding_time)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for entry in raw_response:
            if not isinstance(entry, dict):
                continue
            try:
                funding_time_ms = int(entry.get("fundingTime", 0))
                funding_time = datetime.fromtimestamp(
                    funding_time_ms / 1000, tz=timezone.utc
                ).isoformat() if funding_time_ms else ""
                if not funding_time:
                    continue
                funding_rate = float(entry.get("fundingRate", 0.0))
                mark_price_raw = entry.get("markPrice", None)
                mark_price = float(mark_price_raw) if mark_price_raw else None
                data.append((symbol, funding_time, funding_rate, mark_price, now))
            except (ValueError, TypeError):
                continue

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO binance_funding_rates "
                "(symbol, funding_time, funding_rate, mark_price, ingested_at) "
                "VALUES (?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Binance funding rate rows for %s", len(data), symbol)
        return len(data)

    def get_binance_funding_history(
        self, symbol: str, lookback_days: int = 30
    ) -> List[dict]:
        """Retrieve recent Binance funding rate history for a symbol.

        Useful for computing funding rate trend and identifying sustained
        long/short positioning pressure.

        Args:
            symbol: Perpetual futures symbol, e.g. "BTCUSDT".
            lookback_days: How many days of history to return (default 30).
                           Funding settles every 8h so 30 days = ~90 records.

        Returns:
            List of dicts with keys: symbol, funding_time, funding_rate, mark_price.
            Sorted oldest-first so callers can compute trends directly.
            Empty list if the table doesn't exist or no data found.
        """
        try:
            conn = self._get_conn("crypto")
            # Each day has 3 funding events (every 8h); add a safety margin
            limit = lookback_days * 3 + 10
            cursor = conn.execute(
                """
                SELECT symbol, funding_time, funding_rate, mark_price
                FROM binance_funding_rates
                WHERE symbol = ?
                ORDER BY funding_time DESC
                LIMIT ?
                """,
                (symbol, limit),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_binance_funding_history failed for %s: %s", symbol, exc)
            return []

        return [
            {
                "symbol": r[0],
                "funding_time": r[1],
                "funding_rate": r[2],
                "mark_price": r[3],
            }
            for r in reversed(rows)  # oldest-first
        ]

    # --- Kalshi Prediction Markets ---

    def store_kalshi_markets(self, markets: List[Any]) -> int:
        """Store Kalshi prediction market snapshots.

        Each record is a point-in-time snapshot of a market's probability.
        The timestamp is truncated to the hour so repeated polling within
        the same hour produces a single upserted row per ticker.

        Args:
            markets: List of dicts (or KalshiMarket-like objects) with keys:
                     ticker, title, yes_price, volume_24h, category.
                     Additional fields (open_interest, status) stored if present.

        Returns:
            Number of rows upserted.
        """
        if not markets:
            return 0

        conn = self._get_conn("kalshi")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kalshi_market_snapshots (
                ticker TEXT,
                snapshot_hour TEXT,
                title TEXT,
                yes_price REAL,
                volume_24h INTEGER,
                open_interest INTEGER,
                status TEXT,
                category TEXT,
                ingested_at TEXT,
                PRIMARY KEY (ticker, snapshot_hour)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        snapshot_hour = now[:13]  # "YYYY-MM-DDTHH" — one row per ticker per hour
        data = []
        for m in markets:
            if hasattr(m, "ticker"):
                data.append((
                    getattr(m, "ticker", ""),
                    snapshot_hour,
                    getattr(m, "title", ""),
                    float(getattr(m, "yes_price", 0.0)),
                    int(getattr(m, "volume_24h", 0)),
                    int(getattr(m, "open_interest", 0)),
                    getattr(m, "status", ""),
                    getattr(m, "category", ""),
                    now,
                ))
            elif isinstance(m, dict):
                data.append((
                    m.get("ticker", ""),
                    snapshot_hour,
                    m.get("title", ""),
                    float(m.get("yes_price", 0.0)),
                    int(m.get("volume_24h", 0)),
                    int(m.get("open_interest", 0)),
                    m.get("status", ""),
                    m.get("category", ""),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO kalshi_market_snapshots "
                "(ticker, snapshot_hour, title, yes_price, volume_24h, "
                "open_interest, status, category, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d Kalshi market snapshots", len(data))
        return len(data)

    def get_kalshi_market_history(
        self, ticker: str, lookback_days: int = 30
    ) -> List[dict]:
        """Retrieve hourly Kalshi market probability history for a ticker.

        Useful for detecting probability trend, acceleration, and regime shifts
        (e.g. a market moving from 40% → 70% over 3 days signals strong crowd
        consensus shift and is a powerful convergence signal).

        Args:
            ticker: Kalshi market ticker, e.g. "FED-RATE-25MAR12".
            lookback_days: How many days of history to return (default 30).
                           Each day has up to 24 hourly snapshots.

        Returns:
            List of dicts with keys: ticker, snapshot_hour, yes_price,
            volume_24h, open_interest, status, category.
            Sorted oldest-first for trend computation.
            Empty list if the table doesn't exist or no data found.
        """
        try:
            conn = self._get_conn("kalshi")
            limit = lookback_days * 24 + 24  # hourly rows + buffer
            cursor = conn.execute(
                """
                SELECT ticker, snapshot_hour, yes_price, volume_24h,
                       open_interest, status, category
                FROM kalshi_market_snapshots
                WHERE ticker = ?
                ORDER BY snapshot_hour DESC
                LIMIT ?
                """,
                (ticker, limit),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_kalshi_market_history failed for %s: %s", ticker, exc)
            return []

        return [
            {
                "ticker": r[0],
                "snapshot_hour": r[1],
                "yes_price": r[2],
                "volume_24h": r[3],
                "open_interest": r[4],
                "status": r[5],
                "category": r[6],
            }
            for r in reversed(rows)  # oldest-first
        ]
