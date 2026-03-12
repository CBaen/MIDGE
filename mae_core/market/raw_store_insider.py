"""
raw_store_insider.py - Insider trading data storage mixin.

Handles SEC Form 4, OpenInsider purchases, FinViz insider trades,
FinViz unusual volume, and EDGAR 13F/13D institutional filings.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Any

logger = logging.getLogger(__name__)


class RawStoreInsiderMixin:
    """Mixin for storing insider trading and institutional filing data."""

    # --- SEC EDGAR Form 4 ---

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

    def store_finviz_short_squeeze(self, candidates: List[Any]) -> int:
        """Store FinViz high short float / squeeze candidate records.

        Args:
            candidates: List of ShortSqueezeCandidate dataclass instances or dicts.

        Returns:
            Number of rows upserted.
        """
        if not candidates:
            return 0

        conn = self._get_conn("finviz")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS finviz_short_squeeze (
                ticker TEXT,
                timestamp TEXT,
                company TEXT,
                sector TEXT,
                price REAL,
                short_float_pct REAL,
                short_ratio REAL,
                float_shares INTEGER,
                short_interest INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (ticker, timestamp)
            )
        """)

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for c in candidates:
            if hasattr(c, "ticker"):
                data.append((
                    getattr(c, "ticker", ""), now[:13],
                    getattr(c, "company", ""), getattr(c, "sector", ""),
                    getattr(c, "price", 0.0),
                    getattr(c, "short_float_pct", 0.0),
                    getattr(c, "short_ratio", 0.0),
                    getattr(c, "float_shares", 0),
                    getattr(c, "short_interest", 0),
                    now,
                ))
            elif isinstance(c, dict):
                data.append((
                    c.get("ticker", ""), now[:13],
                    c.get("company", ""), c.get("sector", ""),
                    c.get("price", 0.0),
                    c.get("short_float_pct", 0.0),
                    c.get("short_ratio", 0.0),
                    c.get("float_shares", 0),
                    c.get("short_interest", 0),
                    now,
                ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO finviz_short_squeeze "
                "(ticker, timestamp, company, sector, price, short_float_pct, "
                "short_ratio, float_shares, short_interest, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data,
            )
            conn.commit()

        logger.debug("RawStore: stored %d FinViz short squeeze candidates", len(data))
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
