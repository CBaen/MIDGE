"""
polymarket_client.py - Polymarket Prediction Market Client

Polymarket is the world's largest prediction market by volume. Contract prices
are crowd-sourced probability estimates ($0.01-$1.00 = 1-100%). When 50,000
traders price "Fed cuts rates in March" at $0.72, that IS 72% probability —
the crowd is putting money behind their forecast.

MIDGE uses Polymarket as a leading indicator: when crowd consensus shifts fast
(>10% probability swing in 24h), something real is being priced in. This
signal is orthogonal to technical, insider, and macro domains, making it high
value for convergence.

Finance-relevant categories:
  - Federal Reserve rate decisions (FOMC, rate cut/hike)
  - Economic data releases (CPI, NFP, GDP)
  - Company earnings (beat/miss)
  - Crypto price milestones (BTC above/below X)
  - Market index levels (S&P 500 above/below X)

API: https://gamma-api.polymarket.com — free, no auth, no key.
Rate limit: conservative 2s between requests.
Cache: 30 minutes (markets are liquid but not tick-by-tick).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
MARKETS_URL = f"{GAMMA_API_BASE}/markets"

REQUEST_DELAY = 2.0          # seconds between API requests
CACHE_DURATION = 1800        # 30 minutes — liquid markets but not tick-level
PAGE_SIZE = 100              # records per page
MAX_PAGES = 5                # cap at 500 markets to avoid long fetches
MIN_SHIFT_FOR_MOVER = 0.10   # 10% probability shift = significant

# Keywords that identify finance-relevant markets
_FINANCE_KEYWORDS = {
    "fed", "federal reserve", "fomc", "rate cut", "rate hike", "interest rate",
    "cpi", "inflation", "nfp", "nonfarm", "jobs report", "gdp",
    "earnings", "revenue", "eps",
    "btc", "bitcoin", "ethereum", "eth", "crypto",
    "s&p", "spx", "nasdaq", "dow", "market",
    "recession", "treasury", "yield", "bond", "dollar", "dxy",
    "oil", "crude", "gold", "silver",
    "unemployment", "payrolls",
}

# Map market question keywords → affected tickers + direction logic
# Each entry: (keywords_any, bullish_if_yes, bearish_if_yes, tickers)
_OUTCOME_MAP: List[Tuple[List[str], List[str], List[str]]] = [
    # Fed rate cuts → bullish for equities/bonds
    (["fed cut", "rate cut", "fomc cut", "lower rate"],
     ["SPY", "QQQ", "TLT", "GLD", "IWM"],
     ["UUP"]),
    # Fed rate hikes → bearish for equities
    (["fed hike", "rate hike", "raise rate", "increase rate"],
     ["UUP", "TBT"],
     ["SPY", "QQQ", "TLT", "GLD"]),
    # High CPI / inflation
    (["cpi above", "inflation above", "cpi higher", "cpi over"],
     ["GLD", "TIP", "XLE"],
     ["TLT", "IEF"]),
    # BTC price milestones
    (["bitcoin above", "btc above", "btc over", "bitcoin over", "btc reach"],
     ["BTC-USD", "ETH-USD", "COIN", "MSTR"],
     []),
    # BTC price drops
    (["bitcoin below", "btc below", "btc under", "bitcoin under"],
     [],
     ["BTC-USD", "ETH-USD", "COIN", "MSTR"]),
    # S&P 500 levels
    (["s&p above", "spx above", "s&p 500 above", "spy above", "s&p over"],
     ["SPY", "QQQ", "IWM"],
     []),
    (["s&p below", "spx below", "s&p 500 below", "spy below", "s&p under"],
     [],
     ["SPY", "QQQ", "IWM"]),
    # Earnings beats
    (["beat earnings", "earnings beat", "miss earnings", "earnings miss"],
     [],  # direction depends on beat/miss — handled in logic
     []),
    # Recession
    (["recession", "economic contraction"],
     ["TLT", "GLD", "VIX"],
     ["SPY", "QQQ", "XLF"]),
    # Oil / energy
    (["oil above", "crude above", "oil over", "crude over"],
     ["XLE", "CL=F", "XOM", "CVX"],
     ["airlines", "XLY"]),
]


@dataclass
class PredictionMarketSignal:
    """A Polymarket prediction market with financial signal data."""

    market_id: str
    question: str                  # "Will the Fed cut rates in March 2026?"
    probability: float             # Current crowd probability (0.0-1.0)
    previous_probability: float    # Probability 24h ago (or at last fetch)
    shift: float                   # Change in probability (signed)
    volume: float                  # Trading volume (USD)
    direction: str                 # "bullish" or "bearish" for affected markets
    affected_tickers: List[str]    # Tickers this market outcome affects
    end_date: Optional[str] = None # Market resolution date (YYYY-MM-DD)
    signal_source: str = "polymarket"
    domain: str = "prediction_market"
    confidence: float = 0.0        # Derived from probability extremity + volume

    def to_plain_language(self) -> str:
        shift_pct = self.shift * 100
        prob_pct = self.probability * 100
        tickers = ", ".join(self.affected_tickers[:4]) if self.affected_tickers else "multiple markets"
        return (
            f"Polymarket: '{self.question}' | "
            f"Probability: {prob_pct:.0f}% ({shift_pct:+.0f}% shift) | "
            f"Affects: {tickers} | Direction: {self.direction}"
        )


class PolymarketClient:
    """
    Client for Polymarket prediction market data.

    Fetches active finance/economics markets and identifies movers — markets
    where crowd probability shifted significantly. No auth required.

    Rate limited to 1 req/2s. 30-minute cache.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MIDGE Trading Research",
            "Accept": "application/json",
        })
        self._last_request_time: float = 0.0
        self._cache: Optional[List[PredictionMarketSignal]] = None
        self._cache_time: float = 0.0
        # Price memory for mover detection: market_id → last seen probability
        self._price_memory: Dict[str, float] = {}
        self._call_count = 0
        self._error_count = 0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_json(self, url: str, params: Optional[dict] = None) -> Optional[dict]:
        """Fetch JSON from Polymarket Gamma API."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                source_name="polymarket", timeout_ms=20000.0,
                params=params or {},
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                return resp.payload
            return None

        try:
            r = self.session.get(url, params=params, timeout=20)
            self._call_count += 1
            if r.status_code == 200:
                return r.json()
            logger.warning("Polymarket HTTP %d for %s", r.status_code, url)
            return None
        except Exception as exc:
            self._error_count += 1
            logger.warning("Polymarket fetch error: %s", exc)
            return None

    def get_finance_markets(self) -> List[PredictionMarketSignal]:
        """Fetch active finance/economics prediction markets.

        Filters by finance keywords in question text. Maps each market to
        affected tickers and derives a directional signal.

        Returns:
            List of PredictionMarketSignal, sorted by volume descending.
        """
        if self._cache is not None and time.time() - self._cache_time < CACHE_DURATION:
            return self._cache

        signals = []
        offset = 0

        for _ in range(MAX_PAGES):
            self._rate_limit()
            data = self._fetch_json(
                MARKETS_URL,
                params={
                    "limit": PAGE_SIZE,
                    "offset": offset,
                    "active": "true",
                    "closed": "false",
                },
            )
            if not data:
                break

            # Gamma API returns a list directly or {"markets": [...]}
            if isinstance(data, list):
                markets_raw = data
            elif isinstance(data, dict):
                markets_raw = data.get("markets", data.get("results", []))
            else:
                break

            if not markets_raw:
                break

            for m in markets_raw:
                signal = self._parse_market(m)
                if signal is not None:
                    signals.append(signal)

            if len(markets_raw) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

        # Store raw data if raw_store is wired
        if self._raw_store is not None and signals:
            try:
                self._raw_store.store_polymarket_markets([
                    {
                        "market_id": s.market_id,
                        "question": s.question,
                        "probability": s.probability,
                        "volume": s.volume,
                        "direction": s.direction,
                        "shift": s.shift,
                    }
                    for s in signals[:100]
                ])
            except Exception as exc:
                logger.debug("RawStore Polymarket write failed: %s", exc)

        # Sort by volume — highest volume = most credible crowd signal
        signals.sort(key=lambda s: s.volume, reverse=True)

        self._cache = signals
        self._cache_time = time.time()

        if signals:
            logger.info(
                "Polymarket: %d finance markets fetched (top: %s, prob=%.0f%%)",
                len(signals),
                signals[0].question[:60],
                signals[0].probability * 100,
            )

        return signals

    def get_market_movers(self, hours: int = 24) -> List[PredictionMarketSignal]:
        """Find markets where probability shifted >10% since last fetch.

        A large shift means crowd intelligence is updating fast — something
        new is being priced in. These are the highest-value signals.

        Args:
            hours: Unused (shift is relative to last fetch, not absolute hours).
                   Kept for API consistency.

        Returns:
            List of movers (shift > MIN_SHIFT_FOR_MOVER), sorted by |shift|.
        """
        all_markets = self.get_finance_markets()
        if not all_markets:
            return []

        movers = [
            s for s in all_markets
            if abs(s.shift) >= MIN_SHIFT_FOR_MOVER
        ]
        movers.sort(key=lambda s: abs(s.shift), reverse=True)

        if movers:
            logger.info(
                "Polymarket movers: %d markets shifted >%.0f%% (top: %s %+.0f%%)",
                len(movers),
                MIN_SHIFT_FOR_MOVER * 100,
                movers[0].question[:50],
                movers[0].shift * 100,
            )

        return movers

    def _parse_market(self, m: dict) -> Optional[PredictionMarketSignal]:
        """Parse a raw Gamma API market dict into a PredictionMarketSignal.

        Returns None if the market is not finance-relevant or lacks data.
        """
        question = (
            m.get("question") or
            m.get("title") or
            m.get("description") or ""
        ).strip()

        if not question:
            return None

        # Finance relevance filter
        question_lower = question.lower()
        if not any(kw in question_lower for kw in _FINANCE_KEYWORDS):
            return None

        # Extract probability — Gamma API uses various field names
        probability = _parse_probability(m)
        if probability is None:
            return None

        market_id = str(m.get("id") or m.get("conditionId") or m.get("condition_id") or "")
        if not market_id:
            return None

        # Volume — Gamma API uses outcomePrices or volume fields
        volume = float(m.get("volume") or m.get("volumeNum") or m.get("usdLiquidity") or 0.0)

        # Compute shift vs. last seen probability
        previous_probability = self._price_memory.get(market_id, probability)
        shift = probability - previous_probability
        self._price_memory[market_id] = probability

        # Resolve end date
        end_date = None
        for key in ("endDate", "end_date", "closeTime", "close_time", "expirationDate"):
            val = m.get(key)
            if val:
                end_date = str(val)[:10]
                break

        # Map to tickers and direction
        affected_tickers, direction = self._resolve_tickers_and_direction(question_lower)

        # Confidence: combination of probability extremity and volume
        # Extremity: markets near 0 or 1 are more resolved (less uncertainty)
        extremity = abs(probability - 0.5) * 2.0  # 0 at 50%, 1 at 0% or 100%
        vol_score = min(1.0, volume / 100_000.0)   # saturates at $100k volume
        confidence = round(0.4 * extremity + 0.6 * vol_score, 3)

        return PredictionMarketSignal(
            market_id=market_id,
            question=question,
            probability=round(probability, 4),
            previous_probability=round(previous_probability, 4),
            shift=round(shift, 4),
            volume=round(volume, 2),
            direction=direction,
            affected_tickers=affected_tickers,
            end_date=end_date,
            confidence=confidence,
        )

    def _resolve_tickers_and_direction(
        self, question_lower: str
    ) -> Tuple[List[str], str]:
        """Map a market question to affected tickers and direction.

        Returns (tickers, direction) where direction is "bullish" or "bearish"
        relative to the affected tickers IF the outcome resolves YES.
        """
        for keywords, bullish_tickers, bearish_tickers in _OUTCOME_MAP:
            for kw in keywords:
                if kw in question_lower:
                    # If both sides populated, return the dominant side
                    if bullish_tickers and not bearish_tickers:
                        return bullish_tickers[:], "bullish"
                    if bearish_tickers and not bullish_tickers:
                        return bearish_tickers[:], "bearish"
                    if bullish_tickers:
                        return bullish_tickers[:], "bullish"
                    break

        # Fallback: derive from question sentiment
        if any(w in question_lower for w in ["above", "exceed", "beat", "bull", "rise", "increase"]):
            return ["SPY"], "bullish"
        if any(w in question_lower for w in ["below", "miss", "bear", "fall", "decrease", "cut"]):
            return ["SPY"], "bearish"

        return ["SPY"], "neutral"

    def get_statistics(self) -> dict:
        """Return client statistics."""
        return {
            "calls_made": self._call_count,
            "errors": self._error_count,
            "cached_markets": len(self._cache) if self._cache else 0,
            "price_memory_size": len(self._price_memory),
            "cache_age_seconds": round(time.time() - self._cache_time, 1) if self._cache_time else None,
        }


def _parse_probability(m: dict) -> Optional[float]:
    """Extract probability (0.0-1.0) from a Polymarket Gamma API market dict.

    Gamma API encodes probabilities in several ways:
      - outcomePrices: list like ["0.72", "0.28"] (YES token, NO token)
      - bestBid / bestAsk on YES side
      - price field (direct 0-1 float)
    """
    # outcomePrices: first element is YES token price
    outcome_prices = m.get("outcomePrices")
    if outcome_prices and isinstance(outcome_prices, list) and len(outcome_prices) >= 1:
        try:
            val = float(outcome_prices[0])
            if 0.0 <= val <= 1.0:
                return val
            # Some encode as cents (0-100) — normalize
            if 0.0 < val <= 100.0:
                return val / 100.0
        except (ValueError, TypeError):
            pass

    # Direct price field
    for key in ("price", "lastTradedPrice", "lastTradePrice", "midPrice"):
        val = m.get(key)
        if val is not None:
            try:
                fval = float(val)
                if 0.0 <= fval <= 1.0:
                    return fval
                if 0.0 < fval <= 100.0:
                    return fval / 100.0
            except (ValueError, TypeError):
                pass

    # tokens list (some endpoints)
    tokens = m.get("tokens")
    if tokens and isinstance(tokens, list):
        for token in tokens:
            if isinstance(token, dict) and token.get("outcome", "").lower() == "yes":
                val = token.get("price")
                if val is not None:
                    try:
                        fval = float(val)
                        if 0.0 <= fval <= 1.0:
                            return fval
                    except (ValueError, TypeError):
                        pass

    return None


def get_finance_markets() -> List[PredictionMarketSignal]:
    """Convenience function for one-shot finance market fetch."""
    return PolymarketClient().get_finance_markets()


def get_market_movers() -> List[PredictionMarketSignal]:
    """Convenience function for one-shot market mover detection."""
    return PolymarketClient().get_market_movers()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing Polymarket client...")

    client = PolymarketClient()
    markets = client.get_finance_markets()
    print(f"\nFinance markets: {len(markets)}")
    for s in markets[:5]:
        print(f"  {s.to_plain_language()}")

    print("\nMarket movers (>10% shift):")
    movers = client.get_market_movers()
    if movers:
        for s in movers[:5]:
            print(f"  {s.to_plain_language()}")
    else:
        print("  None (first run — no prior prices to compare against)")

    print(f"\nStats: {client.get_statistics()}")
    print("\nDone.")
