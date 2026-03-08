"""OpenInsider Client — Pre-filtered insider purchase data.

Scrapes openinsider.com for purchase-only insider transactions.
OpenInsider pre-filters RSU grants, gifts, and 10b5-1 plan exercises,
providing higher signal quality than raw SEC EDGAR Form 4 data.

Rate limit: 1 request per 3 seconds.
Part of WP-F1: Always-On MIDGE Wave 3.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("midge.market.openinsider")

_BASE_URL = "http://openinsider.com"
_RATE_LIMIT_SECONDS = 3.0


@dataclass
class InsiderPurchase:
    filing_date: str
    trade_date: str
    ticker: str
    company_name: str
    insider_name: str
    title: str  # CEO, CFO, Director, etc.
    trade_type: str  # P - Purchase
    price: float
    quantity: int
    owned: int  # shares owned after transaction
    delta_owned_pct: float  # % change in ownership
    value: float  # total dollar value of trade


@dataclass
class ClusterBuy:
    ticker: str
    company_name: str
    insider_count: int
    total_value: float
    insiders: List[InsiderPurchase]
    date_range: str  # "2026-02-15 to 2026-03-02"


class OpenInsiderClient:
    def __init__(self, raw_store=None):
        self._raw_store = raw_store
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "MIDGE/1.0 (Market Intelligence; +https://github.com/CBaen/MIDGE)"
        )
        self._last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_LIMIT_SECONDS:
            time.sleep(_RATE_LIMIT_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        self._rate_limit()
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException:
            logger.debug("OpenInsider request failed: %s", url, exc_info=True)
            return None

    def get_recent_purchases(self, days: int = 7) -> List[InsiderPurchase]:
        """Get recent insider PURCHASES (buy-only, pre-filtered by OpenInsider).

        URL params: fd=N means last N days, lt=1 means purchase only, cnt=100 max results.
        Stores to raw_store if available — captures the SEC filing URL in cell[0].
        """
        url = (
            f"{_BASE_URL}/screener"
            f"?s=&o=&pl=&ph=&ll=&lh=&fd={days}&fdr=&td=0&tdr="
            f"&feession=&cession=&lt=1&lk=&cnt=100&page=1"
        )
        soup = self._fetch_page(url)
        if soup is None:
            return []
        results = self._parse_table(soup)
        if self._raw_store is not None and results:
            try:
                self._raw_store.store_openinsider_purchases(results)
            except Exception:
                pass
        return results

    def get_cluster_buys(self, min_insiders: int = 3) -> List[ClusterBuy]:
        """Get cluster buys — multiple insiders buying the same stock within days.

        Fetches recent purchases and groups by ticker, filtering for min_insiders.
        """
        purchases = self.get_recent_purchases(days=30)
        if not purchases:
            return []

        # Group by ticker
        by_ticker: Dict[str, List[InsiderPurchase]] = {}
        for p in purchases:
            if p.ticker not in by_ticker:
                by_ticker[p.ticker] = []
            by_ticker[p.ticker].append(p)

        clusters = []
        for ticker, insiders in by_ticker.items():
            # Deduplicate by insider name
            unique_insiders: Dict[str, InsiderPurchase] = {}
            for ins in insiders:
                if ins.insider_name not in unique_insiders:
                    unique_insiders[ins.insider_name] = ins

            if len(unique_insiders) >= min_insiders:
                total_value = sum(i.value for i in unique_insiders.values())
                dates = [i.trade_date for i in unique_insiders.values() if i.trade_date]
                date_range = f"{min(dates)} to {max(dates)}" if dates else ""
                clusters.append(
                    ClusterBuy(
                        ticker=ticker,
                        company_name=insiders[0].company_name,
                        insider_count=len(unique_insiders),
                        total_value=total_value,
                        insiders=list(unique_insiders.values()),
                        date_range=date_range,
                    )
                )
        return sorted(clusters, key=lambda c: c.total_value, reverse=True)

    def _parse_table(self, soup: BeautifulSoup) -> List[InsiderPurchase]:
        """Parse the OpenInsider HTML table into InsiderPurchase objects."""
        results = []
        table = soup.find("table", class_="tinytable")
        if table is None:
            # Try alternative table selector — look for a table whose first <th> mentions "Filing"
            for t in soup.find_all("table"):
                if t.find("th") and "Filing" in str(t.find("th")):
                    table = t
                    break
        if table is None:
            return []

        rows = table.find_all("tr")[1:]  # Skip header row
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 12:
                continue
            try:
                filing_date = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                trade_date = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                ticker_elem = cells[3].find("a") if len(cells) > 3 else None
                ticker = (
                    ticker_elem.get_text(strip=True)
                    if ticker_elem
                    else (cells[3].get_text(strip=True) if len(cells) > 3 else "")
                )
                company_name = cells[4].get_text(strip=True) if len(cells) > 4 else ""
                insider_name = cells[5].get_text(strip=True) if len(cells) > 5 else ""
                title = cells[6].get_text(strip=True) if len(cells) > 6 else ""
                trade_type = cells[7].get_text(strip=True) if len(cells) > 7 else ""

                # Parse numeric fields (may have $, commas, +/- signs)
                price_text = (
                    cells[8].get_text(strip=True).replace("$", "").replace(",", "")
                    if len(cells) > 8
                    else "0"
                )
                qty_text = (
                    cells[9].get_text(strip=True).replace(",", "").replace("+", "")
                    if len(cells) > 9
                    else "0"
                )
                owned_text = (
                    cells[10].get_text(strip=True).replace(",", "")
                    if len(cells) > 10
                    else "0"
                )
                delta_raw = (
                    cells[11].get_text(strip=True) if len(cells) > 11 else "0"
                )
                delta_text = (
                    delta_raw.replace("%", "")
                    .replace("+", "")
                    .replace("New", "100")
                )
                value_text = (
                    cells[12].get_text(strip=True).replace("$", "").replace(",", "")
                    if len(cells) > 12
                    else "0"
                )

                results.append(
                    InsiderPurchase(
                        filing_date=filing_date,
                        trade_date=trade_date,
                        ticker=ticker.upper(),
                        company_name=company_name,
                        insider_name=insider_name,
                        title=title,
                        trade_type=trade_type,
                        price=float(price_text) if price_text else 0.0,
                        quantity=int(float(qty_text)) if qty_text else 0,
                        owned=int(float(owned_text)) if owned_text else 0,
                        delta_owned_pct=float(delta_text) if delta_text else 0.0,
                        value=float(value_text) if value_text else 0.0,
                    )
                )
            except (ValueError, TypeError, IndexError):
                continue
        return results
