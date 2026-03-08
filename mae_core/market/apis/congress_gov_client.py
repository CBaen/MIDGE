"""
congress_gov_client.py - Congress.gov Legislative Signal Client

Fetches recent bill activity from the Congress.gov API v3.
Free key registration: https://api.data.gov/signup/
Rate limit: 1,000 requests/hour with registered key.

Legislative signals are slow-moving but high-conviction. A defense bill
enacted while defense insiders are buying is far stronger than either alone.

API base: https://api.congress.gov/v3
Auth: ?api_key={KEY}&format=json (query parameters)
Response: {"bills": [...]} or {"bill": {...}}
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import requests

logger = logging.getLogger(__name__)

CONGRESS_BASE_URL = "https://api.congress.gov/v3"
REQUEST_DELAY = 1.2       # 1,000/hour → stay well under with 1.2s gap
CACHE_DURATION = 4 * 3600  # 4 hours

# latestAction.text keywords indicating meaningful legislative progress
ADVANCING_KEYWORDS = [
    "passed", "signed", "enacted", "became public law",
    "reported", "ordered to be reported", "agreed to", "cloture",
]

# Title/action keywords that flip direction to bearish
BEARISH_KEYWORDS = ["restrict", "ban", "prohibit", "repeal", "reduce", "cut"]

# Signal strength by action stage (highest match wins)
_ACTION_STRENGTHS: Dict[str, float] = {
    "enacted": 1.0, "became public law": 1.0, "signed": 1.0,
    "passed house": 0.8, "passed senate": 0.8, "passed": 0.8,
    "agreed to": 0.7, "cloture": 0.6,
    "reported": 0.5, "ordered to be reported": 0.5,
}

# Policy area → sector tickers
LEGISLATIVE_TICKER_MAP: Dict[str, List[str]] = {
    "Armed Forces and National Security": ["ITA", "LMT", "RTX", "NOC", "GD", "BA"],
    "Health": ["XLV", "UNH", "JNJ", "PFE", "ABBV", "IHF"],
    "Energy": ["XLE", "XOP", "XOM", "CVX", "USO"],
    "Science, Technology, Communications": ["XLK", "QQQ", "AAPL", "MSFT", "GOOGL"],
    "Finance and Financial Sector": ["XLF", "KBE", "JPM", "GS", "BAC"],
    "Transportation and Public Works": ["XLI", "IYT", "UNP", "CAT"],
    "Agriculture and Food": ["MOO", "DBA", "ADM", "BG"],
    "Economics and Public Finance": ["SPY", "TLT", "BND"],
    "Taxation": ["SPY", "XLF"],
    "Environmental Protection": ["ICLN", "TAN", "ENPH"],
    "International Trade and Finance": ["SPY", "EFA", "EEM"],
}


def _determine_direction(title: str, action_text: str) -> str:
    """Bill advancing = bullish. Restriction/repeal language = bearish."""
    combined = (title + " " + action_text).lower()
    if any(kw in combined for kw in BEARISH_KEYWORDS):
        return "bearish"
    return "bullish"


def _compute_strength(action_text: str) -> float:
    """Signal strength from legislative stage. Higher stage = higher strength."""
    action_lower = action_text.lower()
    for keyword, strength in sorted(_ACTION_STRENGTHS.items(), key=lambda x: -x[1]):
        if keyword in action_lower:
            return strength
    return 0.3  # Default for other advancing keywords


def _classify_signal_type(action_text: str) -> str:
    """Derive signal_type label from action text."""
    action_lower = action_text.lower()
    if any(kw in action_lower for kw in ("enacted", "became public law", "signed")):
        return "bill_enacted"
    if "passed" in action_lower or "agreed to" in action_lower:
        return "bill_passed"
    return "bill_advancing"


@dataclass
class LegislativeIndicator:
    """A single legislative signal from Congress.gov.

    Carries the same signal metadata as other MIDGE signals so the convergence
    alerter can weight legislative data against other domains.
    Decay rate 0.03 ≈ 23-day half-life (bill law impact is structural).
    """
    bill_id: str              # e.g. "hr-7539-119"
    bill_number: str          # e.g. "HR 7539"
    title: str
    congress: int
    policy_area: str          # e.g. "Armed Forces and National Security"
    action_text: str          # latestAction text
    action_date: str          # YYYY-MM-DD
    signal_type: str          # "bill_enacted", "bill_passed", "bill_advancing"
    direction: str            # bullish/bearish/neutral
    strength: float           # 0-1
    affected_tickers: list = field(default_factory=list)
    signal_source: str = "congress_legislation"
    decay_rate: float = 0.03  # ~23-day half-life
    confidence: float = 0.65


class CongressGovClient:
    """Client for the Congress.gov API v3.

    Fetches advancing bills and translates each into a LegislativeIndicator
    that the convergence engine can cross-reference with other domains.

    Usage:
        client = CongressGovClient()                    # key from env
        snapshot = client.get_legislative_snapshot()    # all advancing bills

    Rate limits: 1,000/hour. Client enforces 1.2s delay.
    Cache: 4 hours.
    """

    def __init__(self, api_key: Optional[str] = None, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.api_key = api_key or os.environ.get("CONGRESS_GOV_API_KEY")

        if self.api_key:
            logger.info("CongressGovClient initialized with API key")
        else:
            logger.warning(
                "No CONGRESS_GOV_API_KEY found. Set CONGRESS_GOV_API_KEY env var "
                "or pass api_key=. Register free at https://api.data.gov/signup/"
            )

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request(self, route: str, params: Dict) -> Optional[dict]:
        """Make a GET request to the Congress.gov API v3."""
        if not self.api_key:
            logger.error("Cannot make Congress.gov request: no API key configured")
            return None

        full_params = dict(params)
        full_params["api_key"] = self.api_key
        full_params["format"] = "json"
        url = f"{CONGRESS_BASE_URL}{route}"

        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider, url,
                headers={"User-Agent": "MIDGE Trading Research"},
                params=full_params,
                source_name="congress_legislation",
                timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            logger.warning(
                "Congress.gov request failed via provider: %s", resp.error_message
            )
            return None

        try:
            self._rate_limit()
            response = self.session.get(url, params=full_params, timeout=30)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "Congress.gov HTTP %s for %s: %s",
                response.status_code, url, response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("Congress.gov request exception for %s: %s", url, exc)
            return None

    def get_recent_bills(self, days: int = 7, limit: int = 50) -> List[dict]:
        """Fetch bills updated in the past `days` days that show advancing action.

        Calls /v3/bill sorted by updateDate descending, filters to entries
        whose latestAction.text matches an advancing keyword.
        Returns raw bill dicts (no detail fetch).
        """
        cache_key = f"bill_list_{days}_{limit}"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if time.time() - cached_at < CACHE_DURATION:
                return data

        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params: Dict[str, Any] = {
            "fromDateTime": from_date,
            "limit": limit,
            "sort": "updateDate desc",
        }

        data = self._request("/bill", params)
        if data is None:
            return []

        bills = data.get("bills", [])
        if not bills:
            logger.debug("Congress.gov: no bills returned for past %d days", days)
            return []

        # Store ALL bills before filtering to advancing-only
        if self._raw_store:
            try:
                self._raw_store.store_congress_bills(bills)
            except Exception:
                pass

        advancing = [
            bill for bill in bills
            if any(
                kw in (bill.get("latestAction", {}).get("text", "") or "").lower()
                for kw in ADVANCING_KEYWORDS
            )
        ]

        logger.debug(
            "Congress.gov: %d bills returned, %d advancing", len(bills), len(advancing)
        )
        self._cache[cache_key] = (advancing, time.time())
        return advancing

    def get_bill_detail(
        self, congress: int, bill_type: str, number: str
    ) -> Optional[dict]:
        """Fetch detail for a single bill to retrieve policyArea.

        Bill type should be lowercase (e.g. "hr", "s", "hjres").
        Returns raw bill detail dict or None on error.
        """
        cache_key = f"bill_detail_{congress}_{bill_type}_{number}"
        if cache_key in self._cache:
            data, cached_at = self._cache[cache_key]
            if time.time() - cached_at < CACHE_DURATION:
                return data

        data = self._request(f"/bill/{congress}/{bill_type}/{number}", {})
        if data is None:
            return None

        detail = data.get("bill")
        if detail is None:
            logger.debug(
                "Congress.gov: no 'bill' key in detail for %s/%s/%s",
                congress, bill_type, number,
            )
            return None

        self._cache[cache_key] = (detail, time.time())
        return detail

    def get_legislative_snapshot(self) -> List[LegislativeIndicator]:
        """Fetch advancing bills (last 7 days) and build LegislativeIndicators.

        Fetches bill detail for each to obtain policyArea. Skips bills where
        detail fetch fails or policyArea is unmapped. Resilient per-bill.
        """
        raw_bills = self.get_recent_bills()
        if not raw_bills:
            return []

        indicators: List[LegislativeIndicator] = []
        for bill in raw_bills:
            try:
                indicator = self._build_indicator(bill)
                if indicator is not None:
                    indicators.append(indicator)
            except Exception as exc:
                logger.warning(
                    "Congress.gov: skipping bill %s — error: %s",
                    self._bill_id_from_raw(bill), exc,
                )

        logger.info(
            "Congress.gov: %d indicators from %d advancing bills",
            len(indicators), len(raw_bills),
        )
        return indicators

    def _build_indicator(self, bill: dict) -> Optional[LegislativeIndicator]:
        """Build a LegislativeIndicator from a raw bill dict + its detail."""
        congress = bill.get("congress")
        bill_type = (bill.get("type") or "").lower()
        number = str(bill.get("number") or "")

        if not (congress and bill_type and number):
            return None

        detail = self.get_bill_detail(congress, bill_type, number)
        if detail is None:
            return None

        policy_area = (detail.get("policyArea") or {}).get("name", "") or "General"
        affected_tickers = LEGISLATIVE_TICKER_MAP.get(policy_area, [])
        if not affected_tickers:
            logger.debug(
                "Congress.gov: no ticker map for policyArea '%s', skipping", policy_area
            )
            return None

        action_text = (bill.get("latestAction", {}).get("text", "") or "")
        action_date = (bill.get("latestAction", {}).get("actionDate", "") or "")
        title = bill.get("title", "") or ""

        direction = _determine_direction(title, action_text)
        strength = _compute_strength(action_text)
        signal_type = _classify_signal_type(action_text)
        bill_id = self._bill_id_from_raw(bill)

        logger.debug(
            "Congress.gov: %s [%s] %s strength=%.2f tickers=%s",
            bill_id, signal_type, direction, strength,
            ", ".join(affected_tickers[:3]),
        )
        return LegislativeIndicator(
            bill_id=bill_id,
            bill_number=f"{bill_type.upper()} {number}",
            title=title,
            congress=int(congress),
            policy_area=policy_area,
            action_text=action_text,
            action_date=action_date,
            signal_type=signal_type,
            direction=direction,
            strength=round(strength, 3),
            affected_tickers=affected_tickers,
        )

    @staticmethod
    def _bill_id_from_raw(bill: dict) -> str:
        """Build a stable bill ID string from a raw bill dict."""
        bill_type = (bill.get("type") or "unknown").lower()
        return f"{bill_type}-{bill.get('number', '0')}-{bill.get('congress', '0')}"


def get_legislative_snapshot() -> List[LegislativeIndicator]:
    """Convenience function: get all recent advancing bill indicators."""
    client = CongressGovClient()
    return client.get_legislative_snapshot()


if __name__ == "__main__":
    import sys

    print("Congress.gov Legislative Signal Test")
    print("=" * 60)

    if not os.environ.get("CONGRESS_GOV_API_KEY"):
        print("WARNING: CONGRESS_GOV_API_KEY not set. Register free at https://api.data.gov/signup/")

    client = CongressGovClient()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    print(f"\nFetching advancing bills from the past {days} days...")
    raw = client.get_recent_bills(days=days)
    print(f"  {len(raw)} advancing bills found")

    print("\nLegislative Snapshot (all mapped bills):")
    snapshot = client.get_legislative_snapshot()
    if snapshot:
        for ind in snapshot:
            arrow = {"bullish": "+", "bearish": "-", "neutral": "~"}[ind.direction]
            print(f"  [{arrow}] {ind.bill_id:<28} strength={ind.strength:.2f}  "
                  f"{ind.policy_area:<38}  {','.join(ind.affected_tickers[:3])}")
            print(f"       {ind.action_text[:80]}")
    else:
        print("  No indicators available (check API key or no advancing bills this week)")

    print("\nDone.")
