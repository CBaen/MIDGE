"""
edgar_xbrl_client.py - SEC EDGAR XBRL Fundamental Data Client

Fetches machine-readable financial statement data for every public US company
via the SEC's free XBRL API. No API key required.

XBRL (eXtensible Business Reporting Language) is the structured data layer
behind every 10-Q and 10-K filed since 2009. The SEC compiles all tagged
values into a single JSON endpoint per company, making this the most complete
free fundamental data source available.

API endpoint: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json

SEC compliance:
  - User-Agent header REQUIRED: must include name and contact
  - Rate limit: 10 requests/second max
  - Cache aggressively: quarterly data changes ~4x per year

Signal logic:
  - Revenue declining 2+ consecutive quarters → bearish
  - Debt/cash > 3.0 AND ratio rising → bearish (liquidity stress)
  - Net income >> operating CF (earnings quality gap) → bearish (accrual bloat)
  - Revenue accelerating + strong operating CF → bullish
  - Strong cash build + debt falling → bullish (balance sheet healing)
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# SEC EDGAR XBRL API
XBRL_BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# SEC requires a descriptive User-Agent with contact info.
# Blocks requests that look like scrapers or have empty/generic agents.
SEC_USER_AGENT = "MIDGE Research research@midge.dev"

# Rate limiting — SEC allows 10 req/sec, we stay conservative
REQUEST_DELAY = 0.12   # 120ms → ~8 req/sec, safe headroom

# Cache — quarterly data changes ~4x/year, 24h cache is appropriate
CACHE_DURATION = 86400  # 24 hours

# Minimum quarters of data needed to compute a trend signal
MIN_QUARTERS_FOR_TREND = 3

# Earnings quality: if net income > operating CF by this multiple, flag it
EARNINGS_QUALITY_DIVERGENCE_THRESHOLD = 1.5

# Debt-to-cash ratio above which we consider the balance sheet stressed
DEBT_CASH_RATIO_BEARISH = 3.0


# ---------------------------------------------------------------------------
# CIK lookup table — top ~50 S&P 500 constituents (hardcoded for zero-latency)
# CIK source: SEC EDGAR company_tickers.json (public)
# ---------------------------------------------------------------------------

TICKER_CIK_MAP: Dict[str, str] = {
    # Mega-cap tech
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "GOOG":  "0001652044",
    "AMZN":  "0001018724",
    "NVDA":  "0001045810",
    "META":  "0001326801",
    "TSLA":  "0001318605",
    "AVGO":  "0001730168",
    "ADBE":  "0000796343",
    "CRM":   "0001108524",
    "CSCO":  "0000858877",
    "INTC":  "0000050863",
    "AMD":   "0000002488",
    "QCOM":  "0000804328",
    "INTU":  "0000896878",
    "TXN":   "0000097476",
    # Financials
    "BRK-B": "0001067983",
    "BRK-A": "0001067983",
    "JPM":   "0000019617",
    "V":     "0001403161",
    "MA":    "0001141391",
    # Healthcare / pharma
    "UNH":   "0000731766",
    "JNJ":   "0000200406",
    "PG":    "0000080424",
    "MRK":   "0000310158",
    "ABBV":  "0001551152",
    "LLY":   "0000059478",
    "ABT":   "0000001800",
    "TMO":   "0000097745",
    "DHR":   "0000313616",
    "AMGN":  "0000820081",
    # Energy
    "XOM":   "0000034088",
    "CVX":   "0000093410",
    # Consumer
    "WMT":   "0000104169",
    "HD":    "0000354950",
    "COST":  "0000909832",
    "PEP":   "0000077476",
    "KO":    "0000021344",
    "MCD":   "0000063908",
    "NKE":   "0000320187",
    "LOW":   "0000060667",
    "PM":    "0001413329",
    # Industrials / diversified
    "ACN":   "0001467373",
    "LIN":   "0001707092",
    "UNP":   "0000100885",
    "RTX":   "0000101829",
    "BA":    "0000012927",
    "GE":    "0000040533",
    "CAT":   "0000018230",
    "DE":    "0000315189",
    "NEE":   "0000753308",
}


# ---------------------------------------------------------------------------
# XBRL concept names — each list is searched in order; first hit is used
# ---------------------------------------------------------------------------

_REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "RevenuesNetOfInterestExpense",
]

_NET_INCOME_CONCEPTS = [
    "NetIncomeLoss",
    "NetIncomeLossAvailableToCommonStockholdersBasic",
    "ProfitLoss",
]

_LONG_TERM_DEBT_CONCEPTS = [
    "LongTermDebt",
    "LongTermDebtNoncurrent",
    "LongTermDebtAndCapitalLeaseObligations",
    "DebtLongtermAndShorttermCombinedAmount",
]

_CASH_CONCEPTS = [
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsAndShortTermInvestments",
    "CashAndCashEquivalentsPeriodIncreaseDecrease",
]

_OPERATING_CF_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByOperatingActivities",
]

_SHARES_CONCEPTS = [
    "EntityCommonStockSharesOutstanding",
    "CommonStockSharesOutstanding",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QuarterlyMetric:
    """A single quarterly observation for a financial metric."""
    period_end: str           # YYYY-MM-DD (10-Q end date)
    form: str                 # "10-Q" or "10-K"
    value: float              # Reported value (native units)
    accn: str = ""            # Accession number for provenance


@dataclass
class FundamentalSnapshot:
    """Latest fundamental metrics for a company, with trend history."""
    ticker: str
    cik: str
    fetched_at: str           # ISO timestamp

    # Latest values
    revenue_ttm: Optional[float] = None       # Trailing 12-month revenue
    net_income_ttm: Optional[float] = None    # TTM net income
    total_debt: Optional[float] = None        # Most recent quarter
    cash: Optional[float] = None              # Most recent quarter
    operating_cf_ttm: Optional[float] = None  # TTM operating cash flow
    shares_outstanding: Optional[float] = None

    # Quarterly series (newest first, up to 8 quarters)
    revenue_quarters: List[QuarterlyMetric] = field(default_factory=list)
    net_income_quarters: List[QuarterlyMetric] = field(default_factory=list)
    operating_cf_quarters: List[QuarterlyMetric] = field(default_factory=list)

    # Derived ratios
    debt_to_cash: Optional[float] = None
    cash_flow_margin: Optional[float] = None  # Operating CF / Revenue
    revenue_growth_qoq: Optional[float] = None   # Latest vs prior quarter
    revenue_growth_yoy: Optional[float] = None   # Latest vs 4 quarters ago
    earnings_quality_ratio: Optional[float] = None  # Net income / Operating CF


@dataclass
class FundamentalSignal:
    """
    Tradeable fundamental signal derived from XBRL financial data.

    Follows the same pattern as VIXSignal / EIASignal for drop-in
    compatibility with the convergence engine.
    """
    ticker: str
    direction: str              # "bullish", "bearish", or "neutral"
    strength: float             # 0.0 – 1.0
    confidence: float           # 0.0 – 1.0
    reasoning: List[str]        # Plain-language factors driving the signal
    snapshot: FundamentalSnapshot

    signal_source: str = "edgar_xbrl"
    domain: str = "fundamental"
    decay_rate: float = 0.05    # Slow decay — quarterly data is durable

    def to_plain_language(self) -> str:
        factors = "; ".join(self.reasoning) if self.reasoning else "No factors"
        return (
            f"EDGAR XBRL [{self.ticker}] {self.direction.upper()} "
            f"strength={self.strength:.2f} | {factors}"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EdgarXBRLClient:
    """
    SEC EDGAR XBRL fundamental data client.

    Downloads all XBRL-tagged financial data for a company in a single
    request, extracts key metrics, computes trends, and returns a
    FundamentalSignal suitable for the convergence engine.

    No API key required. SEC-compliant User-Agent header is set automatically.

    Rate limiting: 120ms between requests (≈8 req/sec, under 10 req/sec limit).
    Cache: 24 hours (quarterly data does not change intra-day).

    Usage:
        client = EdgarXBRLClient()
        signal = client.get_fundamentals("AAPL")
        if signal:
            print(signal.to_plain_language())
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        })
        self._last_request_time: float = 0.0
        # Per-ticker cache: ticker → (FundamentalSnapshot, cache_timestamp)
        self._cache: Dict[str, Tuple[FundamentalSnapshot, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_fundamentals(self, ticker: str) -> Optional[FundamentalSignal]:
        """
        Fetch fundamental data for a single ticker and return a signal.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")

        Returns:
            FundamentalSignal or None if data is unavailable.
        """
        ticker = ticker.upper()
        cik = self._resolve_cik(ticker)
        if not cik:
            logger.debug("XBRL: no CIK for %s", ticker)
            return None

        snapshot = self._get_snapshot(ticker, cik)
        if snapshot is None:
            return None

        return self._build_signal(snapshot)

    def scan_watchlist(self, tickers: List[str]) -> List[FundamentalSignal]:
        """
        Fetch fundamental signals for a list of tickers.

        Unknown tickers (not in CIK map) are silently skipped.

        Args:
            tickers: List of stock ticker symbols.

        Returns:
            List of FundamentalSignal objects (one per resolved ticker).
        """
        results: List[FundamentalSignal] = []
        for ticker in tickers:
            try:
                sig = self.get_fundamentals(ticker)
                if sig is not None:
                    results.append(sig)
            except Exception as exc:
                logger.warning("XBRL scan_watchlist error for %s: %s", ticker, exc)
        return results

    # ------------------------------------------------------------------
    # CIK resolution
    # ------------------------------------------------------------------

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        """
        Resolve a ticker to a zero-padded 10-digit CIK.

        Uses the hardcoded lookup table first. Falls back to SEC's live
        company_tickers.json for tickers not in the table.
        """
        if ticker in TICKER_CIK_MAP:
            return TICKER_CIK_MAP[ticker]

        # Live fallback — fetches SEC's full ticker→CIK mapping (~700KB JSON)
        logger.debug("XBRL: %s not in CIK map, querying SEC live", ticker)
        try:
            self._rate_limit()
            url = "https://www.sec.gov/files/company_tickers.json"
            data = self._fetch_json(url, source_name="sec_company_tickers")
            if data:
                for entry in data.values():
                    if entry.get("ticker", "").upper() == ticker:
                        cik = str(entry.get("cik_str", "")).zfill(10)
                        TICKER_CIK_MAP[ticker] = cik  # warm the cache
                        return cik
        except Exception as exc:
            logger.warning("XBRL CIK live lookup failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Data fetching and caching
    # ------------------------------------------------------------------

    def _get_snapshot(self, ticker: str, cik: str) -> Optional[FundamentalSnapshot]:
        """Return a cached snapshot or fetch a fresh one from the SEC."""
        cached, ts = self._cache.get(ticker, (None, 0.0))
        if cached is not None and time.time() - ts < CACHE_DURATION:
            return cached

        self._rate_limit()
        url = f"{XBRL_BASE_URL}/CIK{cik}.json"
        data = self._fetch_json(url, source_name="edgar_xbrl")
        if not data:
            return None

        try:
            snapshot = self._parse_xbrl(ticker, cik, data)
        except Exception as exc:
            logger.warning("XBRL parse error for %s: %s", ticker, exc)
            return None

        if self._raw_store:
            try:
                self._raw_store.store_generic(
                    domain="fundamental",
                    ticker=ticker,
                    payload={"cik": cik, "fetched_at": snapshot.fetched_at},
                )
            except Exception as exc:
                logger.debug("XBRL raw_store write failed: %s", exc)

        self._cache[ticker] = (snapshot, time.time())
        return snapshot

    def _fetch_json(self, url: str, source_name: str = "edgar_xbrl") -> Optional[dict]:
        """Fetch a JSON endpoint, routing through MarketDataProvider if available."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"},
                source_name=source_name, timeout_ms=45000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                payload = resp.payload
                if isinstance(payload, dict):
                    return payload
            return None

        try:
            r = self.session.get(url, timeout=45)
            if r.status_code == 200:
                return r.json()
            logger.warning("XBRL HTTP %d for %s", r.status_code, url)
            return None
        except Exception as exc:
            logger.warning("XBRL fetch error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # XBRL parsing
    # ------------------------------------------------------------------

    def _parse_xbrl(self, ticker: str, cik: str, data: dict) -> FundamentalSnapshot:
        """
        Parse the raw XBRL companyfacts JSON into a FundamentalSnapshot.

        The SEC JSON structure is:
          {
            "cik": 320193,
            "entityName": "Apple Inc.",
            "facts": {
              "us-gaap": { "<ConceptName>": { "units": { "USD": [...] } } },
              "dei":     { "<ConceptName>": { "units": { "shares": [...] } } }
            }
          }

        Each unit list is an array of filings:
          [{"end": "2023-09-30", "val": 383285000000, "form": "10-K", "accn": "..."}]
        """
        facts = data.get("facts", {})
        usgaap = facts.get("us-gaap", {})
        dei = facts.get("dei", {})

        snap = FundamentalSnapshot(
            ticker=ticker,
            cik=cik,
            fetched_at=datetime.utcnow().isoformat(),
        )

        # --- Revenue ---
        rev_quarters = self._extract_quarterly(usgaap, _REVENUE_CONCEPTS, unit="USD")
        snap.revenue_quarters = rev_quarters[:8]
        snap.revenue_ttm = self._sum_ttm(rev_quarters)

        # --- Net Income ---
        ni_quarters = self._extract_quarterly(usgaap, _NET_INCOME_CONCEPTS, unit="USD")
        snap.net_income_quarters = ni_quarters[:8]
        snap.net_income_ttm = self._sum_ttm(ni_quarters)

        # --- Total Debt (balance sheet — point-in-time, not TTM) ---
        debt_quarters = self._extract_quarterly(usgaap, _LONG_TERM_DEBT_CONCEPTS, unit="USD")
        if debt_quarters:
            snap.total_debt = debt_quarters[0].value

        # --- Cash ---
        cash_quarters = self._extract_quarterly(usgaap, _CASH_CONCEPTS, unit="USD")
        if cash_quarters:
            snap.cash = cash_quarters[0].value

        # --- Operating Cash Flow ---
        ocf_quarters = self._extract_quarterly(usgaap, _OPERATING_CF_CONCEPTS, unit="USD")
        snap.operating_cf_quarters = ocf_quarters[:8]
        snap.operating_cf_ttm = self._sum_ttm(ocf_quarters)

        # --- Shares Outstanding (dei namespace, shares unit) ---
        shares_quarters = self._extract_quarterly(dei, _SHARES_CONCEPTS, unit="shares")
        if shares_quarters:
            snap.shares_outstanding = shares_quarters[0].value

        # --- Derived ratios ---
        if snap.total_debt is not None and snap.cash and snap.cash > 0:
            snap.debt_to_cash = round(snap.total_debt / snap.cash, 3)

        if snap.revenue_ttm and snap.revenue_ttm > 0 and snap.operating_cf_ttm is not None:
            snap.cash_flow_margin = round(snap.operating_cf_ttm / snap.revenue_ttm, 4)

        if snap.net_income_ttm is not None and snap.operating_cf_ttm and snap.operating_cf_ttm != 0:
            snap.earnings_quality_ratio = round(snap.net_income_ttm / snap.operating_cf_ttm, 4)

        # --- Revenue growth rates ---
        if len(rev_quarters) >= 2:
            latest = rev_quarters[0].value
            prior = rev_quarters[1].value
            if prior != 0:
                snap.revenue_growth_qoq = round((latest - prior) / abs(prior), 4)

        if len(rev_quarters) >= 5:
            latest = rev_quarters[0].value
            year_ago = rev_quarters[4].value
            if year_ago != 0:
                snap.revenue_growth_yoy = round((latest - year_ago) / abs(year_ago), 4)

        return snap

    def _extract_quarterly(
        self,
        namespace: dict,
        concepts: List[str],
        unit: str = "USD",
    ) -> List[QuarterlyMetric]:
        """
        Extract quarterly observations for the first matching concept.

        Searches concepts in order and returns the first non-empty result.
        Filters to quarterly (10-Q) and annual (10-K) filings only.
        Deduplicates on period_end (annual 10-K takes priority over amended 10-K/A).
        Returns newest-first, limited to the last 12 quarters.
        """
        for concept in concepts:
            if concept not in namespace:
                continue

            units_block = namespace[concept].get("units", {})
            rows = units_block.get(unit, [])

            if not rows:
                continue

            # Keep only 10-Q and 10-K; skip 10-K/A, 10-Q/A, NT filings
            # Also skip rows without an "end" date (point-in-time balance sheet items
            # sometimes only have "instant" dates — we use "end" as the period key)
            seen_ends: Dict[str, QuarterlyMetric] = {}
            for row in rows:
                form = row.get("form", "")
                if form not in ("10-Q", "10-K"):
                    continue
                end = row.get("end", "")
                if not end:
                    end = row.get("instant", "")
                if not end:
                    continue
                val = row.get("val")
                if val is None:
                    continue

                # For duplicate period_ends, prefer 10-K over 10-Q (more audited)
                existing = seen_ends.get(end)
                if existing is None or (form == "10-K" and existing.form == "10-Q"):
                    seen_ends[end] = QuarterlyMetric(
                        period_end=end,
                        form=form,
                        value=float(val),
                        accn=row.get("accn", ""),
                    )

            if not seen_ends:
                continue

            sorted_quarters = sorted(seen_ends.values(), key=lambda q: q.period_end, reverse=True)
            return sorted_quarters[:12]

        return []

    def _sum_ttm(self, quarters: List[QuarterlyMetric]) -> Optional[float]:
        """
        Compute trailing twelve months (TTM) by summing the last 4 quarterly values.

        Returns None if fewer than 4 quarters are available.
        """
        q_only = [q for q in quarters if q.form == "10-Q"]
        if len(q_only) >= 4:
            return sum(q.value for q in q_only[:4])

        # Fall back: if annual 10-K is most recent, use it directly
        if quarters and quarters[0].form == "10-K":
            return quarters[0].value

        return None

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def _build_signal(self, snap: FundamentalSnapshot) -> FundamentalSignal:
        """
        Translate a FundamentalSnapshot into a tradeable FundamentalSignal.

        Each flag adds to a bullish or bearish score. The dominant direction
        wins; strength is the normalized margin between the two sides.
        """
        bullish_score = 0.0
        bearish_score = 0.0
        reasoning: List[str] = []

        # --- Revenue trend ---
        rev_decline_qtrs = self._count_consecutive_revenue_declines(snap.revenue_quarters)
        if rev_decline_qtrs >= 2:
            bearish_score += 0.35
            reasoning.append(
                f"Revenue declining for {rev_decline_qtrs} consecutive quarters"
            )
        elif snap.revenue_growth_qoq is not None and snap.revenue_growth_qoq > 0.05:
            bullish_score += 0.20
            reasoning.append(
                f"Revenue accelerating QoQ +{snap.revenue_growth_qoq:.1%}"
            )

        if snap.revenue_growth_yoy is not None:
            if snap.revenue_growth_yoy < -0.05:
                bearish_score += 0.20
                reasoning.append(
                    f"Revenue down YoY {snap.revenue_growth_yoy:.1%}"
                )
            elif snap.revenue_growth_yoy > 0.10:
                bullish_score += 0.25
                reasoning.append(
                    f"Revenue up YoY +{snap.revenue_growth_yoy:.1%}"
                )

        # --- Balance sheet health (debt/cash) ---
        if snap.debt_to_cash is not None:
            prev_ratio = self._prev_debt_to_cash(snap)
            ratio_rising = prev_ratio is not None and snap.debt_to_cash > prev_ratio * 1.10

            if snap.debt_to_cash > DEBT_CASH_RATIO_BEARISH and ratio_rising:
                bearish_score += 0.30
                reasoning.append(
                    f"Debt/cash ratio {snap.debt_to_cash:.1f}x and rising "
                    f"(was {prev_ratio:.1f}x) — liquidity stress"
                )
            elif snap.debt_to_cash < 1.0:
                bullish_score += 0.15
                reasoning.append(
                    f"Clean balance sheet: debt/cash {snap.debt_to_cash:.1f}x"
                )

        # --- Cash flow margin ---
        if snap.cash_flow_margin is not None:
            if snap.cash_flow_margin > 0.15:
                bullish_score += 0.20
                reasoning.append(
                    f"Strong operating CF margin {snap.cash_flow_margin:.1%}"
                )
            elif snap.cash_flow_margin < 0.0:
                bearish_score += 0.20
                reasoning.append(
                    f"Negative operating CF margin {snap.cash_flow_margin:.1%}"
                )

        # --- Earnings quality (accrual divergence) ---
        if snap.earnings_quality_ratio is not None:
            if snap.earnings_quality_ratio > EARNINGS_QUALITY_DIVERGENCE_THRESHOLD:
                # Net income materially exceeds operating CF — accrual bloat
                bearish_score += 0.25
                reasoning.append(
                    f"Earnings quality gap: net income {snap.earnings_quality_ratio:.1f}x "
                    f"operating CF — possible accrual inflation"
                )
            elif 0.5 < snap.earnings_quality_ratio < 1.2:
                # Net income close to operating CF — clean earnings
                bullish_score += 0.10
                reasoning.append(
                    f"High earnings quality: NI/OCF {snap.earnings_quality_ratio:.2f}"
                )

        # --- Cash flow trend (bullish: growing OCF) ---
        if len(snap.operating_cf_quarters) >= 4:
            recent_ocf = snap.operating_cf_quarters[0].value
            older_ocf = snap.operating_cf_quarters[3].value
            if older_ocf > 0 and recent_ocf > older_ocf * 1.15:
                bullish_score += 0.15
                reasoning.append(
                    "Operating cash flow growing strongly (last 4 quarters)"
                )

        # --- Synthesize direction and strength ---
        total_score = bullish_score + bearish_score
        if total_score == 0.0:
            direction = "neutral"
            strength = 0.0
            confidence = 0.30
        elif bullish_score > bearish_score:
            direction = "bullish"
            strength = round(min((bullish_score - bearish_score) / max(total_score, 0.01), 1.0), 3)
            confidence = round(min(0.45 + strength * 0.35, 0.80), 3)
        elif bearish_score > bullish_score:
            direction = "bearish"
            strength = round(min((bearish_score - bullish_score) / max(total_score, 0.01), 1.0), 3)
            confidence = round(min(0.45 + strength * 0.35, 0.80), 3)
        else:
            direction = "neutral"
            strength = 0.0
            confidence = 0.35

        if not reasoning:
            reasoning.append("Insufficient data for directional signal")

        return FundamentalSignal(
            ticker=snap.ticker,
            direction=direction,
            strength=strength,
            confidence=confidence,
            reasoning=reasoning,
            snapshot=snap,
        )

    # ------------------------------------------------------------------
    # Trend helpers
    # ------------------------------------------------------------------

    def _count_consecutive_revenue_declines(
        self, quarters: List[QuarterlyMetric]
    ) -> int:
        """
        Count how many consecutive quarters revenue has declined, starting
        from the most recent quarter. Requires at least 2 data points.
        """
        if len(quarters) < 2:
            return 0
        count = 0
        for i in range(len(quarters) - 1):
            if quarters[i].value < quarters[i + 1].value:
                count += 1
            else:
                break
        return count

    def _prev_debt_to_cash(self, snap: FundamentalSnapshot) -> Optional[float]:
        """
        Approximate the prior-quarter debt/cash ratio.

        Not directly available from the snapshot; we use the ratio of
        the second-most-recent quarters for each metric if present.
        Returns None if insufficient data.
        """
        # This is a rough proxy — the full XBRL data has per-quarter balance
        # sheet values but we only retain the most recent in the snapshot.
        # Future enhancement: retain full balance sheet series.
        return None

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> Optional[FundamentalSignal]:
    """Convenience function for one-shot fundamental data fetch."""
    return EdgarXBRLClient().get_fundamentals(ticker)


def scan_watchlist(tickers: List[str]) -> List[FundamentalSignal]:
    """Convenience function for watchlist scanning."""
    return EdgarXBRLClient().scan_watchlist(tickers)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    test_tickers = sys.argv[1:] or ["AAPL", "MSFT", "NVDA"]
    print(f"Testing EDGAR XBRL fundamentals for: {', '.join(test_tickers)}\n")

    client = EdgarXBRLClient()
    for ticker in test_tickers:
        print(f"--- {ticker} ---")
        sig = client.get_fundamentals(ticker)
        if sig:
            snap = sig.snapshot
            print(f"  Signal: {sig.to_plain_language()}")
            if snap.revenue_ttm:
                print(f"  Revenue TTM: ${snap.revenue_ttm/1e9:.1f}B")
            if snap.net_income_ttm:
                print(f"  Net Income TTM: ${snap.net_income_ttm/1e9:.1f}B")
            if snap.operating_cf_ttm:
                print(f"  Operating CF TTM: ${snap.operating_cf_ttm/1e9:.1f}B")
            if snap.total_debt:
                print(f"  Total Debt: ${snap.total_debt/1e9:.1f}B")
            if snap.cash:
                print(f"  Cash: ${snap.cash/1e9:.1f}B")
            if snap.debt_to_cash is not None:
                print(f"  Debt/Cash: {snap.debt_to_cash:.1f}x")
            if snap.cash_flow_margin is not None:
                print(f"  CF Margin: {snap.cash_flow_margin:.1%}")
            if snap.earnings_quality_ratio is not None:
                print(f"  EQ Ratio: {snap.earnings_quality_ratio:.2f}")
            if snap.revenue_growth_qoq is not None:
                print(f"  Rev QoQ: {snap.revenue_growth_qoq:+.1%}")
            if snap.revenue_growth_yoy is not None:
                print(f"  Rev YoY: {snap.revenue_growth_yoy:+.1%}")
        else:
            print("  No data returned")
        print()

    print("Done.")
