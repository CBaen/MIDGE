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
