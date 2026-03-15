"""
crypto_fear_greed_client.py - Crypto Fear & Greed Index Client

Fetches the Crypto Fear and Greed Index from alternative.me's free API.
No API key required.

The index is a composite sentiment gauge (0-100) for crypto markets:
- Extreme Fear (0-20):  Markets oversold, contrarian bullish signal
- Fear (21-35):         Negative sentiment, mild contrarian bullish
- Neutral (36-64):      No strong directional signal
- Greed (65-79):        Complacency building, mild contrarian bearish
- Extreme Greed (80+):  Markets overbought, contrarian bearish signal

Contrarian framing: crowd extremes predict reversals. Extreme Fear is
historically when smart money buys; Extreme Greed is when bubbles form.

No API key required. Data source: https://alternative.me/crypto/fear-and-greed-index/
"""

import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=10&format=json"

REQUEST_DELAY = 2.0
CACHE_DURATION = 14400  # 4 hours — index updates once daily


@dataclass
class CryptoFearGreedSignal:
    """Crypto Fear & Greed Index reading with contrarian signal interpretation."""

    value: int               # 0-100 composite score
    classification: str      # "Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"
    direction: str           # "bullish" (contrarian on fear) or "bearish" (contrarian on greed)
    strength: float          # Signal strength 0-1, higher = more extreme reading
    confidence: float        # Confidence based on reading extremity
    trend: str               # "falling", "rising", or "stable" (7-day direction)
    history: list            # Last 10 daily readings [{"value": int, "classification": str, "date": str}]
    signal_source: str = "crypto_fear_greed"
    decay_rate: float = 0.20  # Slow decay — sentiment shifts persist
    domain: str = "sentiment"

    def to_plain_language(self) -> str:
        return (
            f"Crypto Fear & Greed: {self.value} ({self.classification}) | "
            f"Signal: {self.direction.upper()} strength={self.strength:.2f} | "
            f"Trend: {self.trend}"
        )


def _classify_signal(value: int) -> tuple:
    """
    Derive (direction, strength, confidence) from a Fear & Greed value.

    Contrarian logic:
    - Extreme Fear  → market oversold  → bullish opportunity
    - Fear          → negative bias    → mild bullish lean
    - Neutral       → no clear edge    → weak signal
    - Greed         → complacency      → mild bearish lean
    - Extreme Greed → market overbought → bearish opportunity
    """
    if value <= 20:
        # Extreme Fear — strongest contrarian bullish
        strength = 0.80 + (20 - value) / 20 * 0.20  # 0.80 → 1.00 as value → 0
        return "bullish", round(min(strength, 1.0), 3), 0.65
    elif value <= 35:
        # Fear — moderate contrarian bullish
        # Strength scales from 0.60 (at 35) to 0.79 (approaching 21)
        strength = 0.60 + (35 - value) / 15 * 0.19
        return "bullish", round(strength, 3), 0.55
    elif value >= 80:
        # Extreme Greed — strongest contrarian bearish
        strength = 0.80 + (value - 80) / 20 * 0.20  # 0.80 → 1.00 as value → 100
        return "bearish", round(min(strength, 1.0), 3), 0.65
    elif value >= 65:
        # Greed — moderate contrarian bearish
        # Strength scales from 0.60 (at 65) to 0.79 (approaching 79)
        strength = 0.60 + (value - 65) / 15 * 0.19
        return "bearish", round(strength, 3), 0.55
    else:
        # Neutral (36-64) — weak signal, no directional lean
        return "neutral", 0.30, 0.40


def _compute_trend(history: List[dict]) -> str:
    """
    Determine whether sentiment is falling, rising, or stable over 7 days.

    A 15-point swing in either direction over 7 days qualifies as a trend.
    Uses oldest vs newest reading in the 7-day window.
    """
    if len(history) < 2:
        return "stable"

    # history[0] is newest, history[-1] is oldest
    recent = history[0]["value"]
    week_ago = history[min(6, len(history) - 1)]["value"]
    delta = recent - week_ago

    if delta <= -15:
        return "falling"
    elif delta >= 15:
        return "rising"
    return "stable"


class CryptoFearGreedClient:
    """
    Client for the Crypto Fear & Greed Index (alternative.me).

    Returns a contrarian sentiment signal — Extreme Fear is bullish,
    Extreme Greed is bearish. Includes 10-day history for trend detection.

    Rate limited to 1 req/2s. 4-hour cache.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0
        self._cache: Optional[CryptoFearGreedSignal] = None
        self._cache_time: float = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _fetch_json(self, url: str) -> Optional[dict]:
        """Fetch JSON from a URL, respecting the provider abstraction."""
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus
            resp = market_request(
                self._provider, url,
                source_name="crypto_fear_greed", timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                payload = resp.payload
                if isinstance(payload, dict):
                    return payload
            return None

        try:
            r = self.session.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
            logger.warning("Fear & Greed fetch HTTP %d", r.status_code)
            return None
        except Exception as exc:
            logger.warning("Fear & Greed fetch error: %s", exc)
            return None

    def get_fear_greed(self) -> Optional[CryptoFearGreedSignal]:
        """
        Fetch the current Crypto Fear & Greed Index with contrarian signal.

        Returns:
            CryptoFearGreedSignal with direction, strength, confidence, and
            7-day trend, or None if data is unavailable.
        """
        if self._cache is not None and time.time() - self._cache_time < CACHE_DURATION:
            return self._cache

        self._rate_limit()
        raw = self._fetch_json(FEAR_GREED_URL)
        if not raw:
            return None

        try:
            return self._parse_response(raw)
        except Exception as exc:
            logger.warning("Fear & Greed parse error: %s", exc)
            return None

    def _parse_response(self, raw: dict) -> Optional[CryptoFearGreedSignal]:
        """Parse API response into a CryptoFearGreedSignal."""
        data = raw.get("data", [])
        if not data:
            logger.warning("Fear & Greed: empty data array")
            return None

        # Build normalized history (newest first, matching API order)
        history = []
        for entry in data:
            try:
                v = int(entry["value"])
                ts = int(entry.get("timestamp", 0))
                date_str = (
                    datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
                    if ts else ""
                )
                history.append({
                    "value": v,
                    "classification": entry.get("value_classification", ""),
                    "date": date_str,
                })
            except (KeyError, ValueError, TypeError):
                continue

        if not history:
            return None

        # Persist raw payload before processing
        if self._raw_store is not None:
            try:
                self._raw_store.store_crypto_fear_greed(history)
            except Exception as exc:
                logger.debug("RawStore crypto_fear_greed write failed: %s", exc)

        current = history[0]
        value = current["value"]
        classification = current["classification"]

        direction, strength, confidence = _classify_signal(value)
        trend = _compute_trend(history)

        result = CryptoFearGreedSignal(
            value=value,
            classification=classification,
            direction=direction,
            strength=strength,
            confidence=confidence,
            trend=trend,
            history=history,
        )

        self._cache = result
        self._cache_time = time.time()
        logger.debug(
            "Fear & Greed: value=%d (%s) direction=%s strength=%.2f trend=%s",
            value, classification, direction, strength, trend,
        )
        return result


def get_fear_greed() -> Optional[CryptoFearGreedSignal]:
    """Convenience function for one-shot Fear & Greed fetch."""
    return CryptoFearGreedClient().get_fear_greed()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing Crypto Fear & Greed Index...")

    client = CryptoFearGreedClient()
    signal = client.get_fear_greed()

    if signal:
        print(f"  {signal.to_plain_language()}")
        print(f"  Last 3 readings: {signal.history[:3]}")
    else:
        print("  No data returned")

    print("\nDone.")
