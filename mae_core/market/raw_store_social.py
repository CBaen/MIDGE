"""
raw_store_social.py - Social and financial sentiment data storage mixin.

Handles StockTwits messages, Finnhub sentiment/earnings/economic/ticks,
Yahoo RSS headlines, ApeWisdom Reddit sentiment.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class RawStoreSocialMixin:
    """Mixin for storing social and financial news/sentiment data."""

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
                user_followers INTEGER,
                ingested_at TEXT
            )
        """)
        # Add user_followers column to pre-existing tables (idempotent migration)
        try:
            conn.execute(
                "ALTER TABLE stocktwits_messages ADD COLUMN user_followers INTEGER DEFAULT 0"
            )
            conn.commit()
        except Exception:
            pass  # Column already exists

        now = datetime.now(timezone.utc).isoformat()
        data = []
        for msg in messages:
            msg_id = str(msg.get("id", ""))
            if not msg_id:
                continue
            entities = msg.get("entities", {})
            sentiment_obj = entities.get("sentiment")
            sentiment = sentiment_obj.get("basic", "") if sentiment_obj else ""
            likes_val = msg.get("likes", 0)
            if isinstance(likes_val, dict):
                likes = likes_val.get("total", 0) or 0
            elif isinstance(likes_val, (int, float)):
                likes = int(likes_val)
            else:
                likes = 0
            user = msg.get("user", {}) or {}
            user_followers = user.get("followers_count", 0) or 0
            data.append((
                msg_id, ticker, sentiment,
                msg.get("body", "")[:2000],
                user.get("username", ""),
                msg.get("created_at", ""),
                likes,
                user_followers,
                now,
            ))

        if data:
            conn.executemany(
                "INSERT OR REPLACE INTO stocktwits_messages "
                "(message_id, ticker, sentiment, body, user_name, created_at, likes, "
                "user_followers, ingested_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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

    def get_finnhub_earnings(
        self, ticker: str = None, lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Retrieve Finnhub earnings calendar events.

        Args:
            ticker: Filter by symbol (None = all symbols).
            lookback_days: How many days of history to return.

        Returns:
            List of dicts with keys: symbol, date, quarter, year, eps_estimate,
            eps_actual, revenue_estimate, revenue_actual, hour.
            Sorted newest-first. Empty list on error.
        """
        try:
            conn = self._get_conn("finnhub")
            params = [f"-{lookback_days} days"]
            where = "WHERE date >= date('now', ?)"
            if ticker:
                where += " AND symbol = ?"
                params.append(ticker)
            cursor = conn.execute(
                f"""
                SELECT symbol, date, quarter, year, eps_estimate, eps_actual,
                       revenue_estimate, revenue_actual, hour
                FROM finnhub_earnings
                {where}
                ORDER BY date DESC
                """,
                params,
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_finnhub_earnings failed: %s", exc)
            return []

        return [
            {
                "symbol": r[0], "date": r[1], "quarter": r[2], "year": r[3],
                "eps_estimate": r[4], "eps_actual": r[5],
                "revenue_estimate": r[6], "revenue_actual": r[7], "hour": r[8],
            }
            for r in rows
        ]

    def get_yahoo_headlines(
        self, ticker: str = None, lookback_days: int = 3
    ) -> List[Dict[str, Any]]:
        """Retrieve Yahoo Finance RSS headlines.

        Args:
            ticker: Filter by symbol (None = all tickers).
            lookback_days: How many days of history to return.

        Returns:
            List of dicts with keys: ticker, title, published_at, summary.
            Sorted newest-first. Empty list on error.
        """
        try:
            conn = self._get_conn("yahoo_rss")
            params = [f"-{lookback_days} days"]
            where = "WHERE ingested_at >= datetime('now', ?)"
            if ticker:
                where += " AND ticker = ?"
                params.append(ticker.upper())
            cursor = conn.execute(
                f"""
                SELECT ticker, title, published_at, summary
                FROM yahoo_headlines
                {where}
                ORDER BY ingested_at DESC
                """,
                params,
            )
            rows = cursor.fetchall()
        except Exception as exc:
            logger.debug("RawStore: get_yahoo_headlines failed: %s", exc)
            return []

        return [
            {"ticker": r[0], "title": r[1], "published_at": r[2], "summary": r[3]}
            for r in rows
        ]

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
