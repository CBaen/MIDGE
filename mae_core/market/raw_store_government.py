"""
raw_store_government.py - Government data storage mixin.

Handles congressional trades, Congress.gov bills, and FINRA short volume data.
"""

import sqlite3
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RawStoreGovernmentMixin:
    """Mixin for storing government and regulatory data."""

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

    def get_congressional_trades(
        self, ticker: str = None, lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieve congressional trade records.

        Args:
            ticker: Filter by symbol (None = all tickers).
            lookback_days: How many days of history to return.

        Returns:
            List of dicts with keys: representative, ticker, transaction_date,
            transaction_type, amount_range, amount_low, amount_high, party, chamber.
            Sorted newest-first. Empty list on error.
        """
        try:
            conn = self._get_conn("congressional")
            params = [f"-{lookback_days} days"]
            where = "WHERE transaction_date >= date('now', ?)"
            if ticker:
                where += " AND ticker = ?"
                params.append(ticker)
            cursor = conn.execute(
                f"""
                SELECT representative, ticker, transaction_date, transaction_type,
                       amount_range, amount_low, amount_high, party, chamber
                FROM congressional_trades
                {where}
                ORDER BY transaction_date DESC
                """,
                params,
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_congressional_trades failed: %s", exc)
            return []

        return [
            {
                "representative": r[0], "ticker": r[1], "transaction_date": r[2],
                "transaction_type": r[3], "amount_range": r[4],
                "amount_low": r[5], "amount_high": r[6],
                "party": r[7], "chamber": r[8],
            }
            for r in rows
        ]

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
