"""
cboe_options_client.py - CBOE Options Market Data Client

Fetches free options market data from CBOE's public CSV feeds. No API key required.

Expands MIDGE's volatility picture beyond VIX itself into four complementary
fear gauges and the Put/Call ratio — forming the 13th convergence domain: options.

Data sources:
  VVIX  — VIX of VIX. When VVIX spikes before VIX, options traders are buying
           protection on the protection instrument itself. Early warning signal.
  VIX9D — 9-day VIX. Ultra-short-term implied volatility. VIX9D > VIX means
           fear is accelerating (short-dated options more expensive than medium).
  OVX   — CBOE Crude Oil ETF Volatility Index. Energy fear gauge. Cross-
           references with EIA inventory data to strengthen or contradict
           energy domain signals.
  GVZ   — CBOE Gold ETF Volatility. Safe-haven fear gauge. High GVZ means
           even gold holders are hedging — extreme uncertainty environment.
  P/C   — CBOE Equity Put/Call Ratio. Contrarian signal: extreme hedging demand
           (P/C > 1.2) historically precedes rallies; extreme complacency
           (P/C < 0.5) historically precedes corrections.

No API key required. Data source: CBOE public data feeds (cdn.cboe.com).
Rate limited to 1 req/2s. 4-hour cache (daily data — no need to poll more often).
"""

import csv
import io
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CBOE public data URLs
# ---------------------------------------------------------------------------
_VIX_FAMILY_URLS: Dict[str, str] = {
    "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "OVX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",
    "GVZ": "https://cdn.cboe.com/api/global/us_indices/daily_prices/GVZ_History.csv",
}

# CBOE daily market statistics (equity put/call ratio lives here)
_PUT_CALL_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily.csv"

REQUEST_DELAY = 2.0          # Seconds between requests — same as VIX client
CACHE_DURATION = 14400       # 4 hours — daily data updates once per trading day

# ---------------------------------------------------------------------------
# Business logic thresholds
# ---------------------------------------------------------------------------

# VVIX: VIX of VIX — fear-of-fear.
# >= 120: elevated hedging demand on VIX options → bearish
# >= 100: moderate uncertainty → neutral
# < 100: calm → bullish
_VVIX_HIGH = 120.0
_VVIX_MODERATE = 100.0

# VIX9D vs VIX: when 9-day > spot VIX, fear is accelerating near-term
# We compare against a VIX spot reference stored at signal creation time.
# The direction logic lives in CBOEOptionsClient._classify_vix9d().

# OVX: CBOE Crude Oil Volatility
# >= 50: extreme energy uncertainty → bearish for energy stocks
# >= 35: elevated → neutral signal (worth monitoring)
_OVX_EXTREME = 50.0
_OVX_ELEVATED = 35.0

# GVZ: CBOE Gold Volatility
# >= 25: elevated safe-haven fear → bearish risk-off signal
# >= 20: moderate
_GVZ_HIGH = 25.0
_GVZ_MODERATE = 20.0

# Put/Call ratio (equity):
# > 1.2: extreme hedging demand → contrarian BULLISH (sellers exhausted)
# < 0.5: extreme complacency → contrarian BEARISH (no hedges = crowded longs)
# 0.5 – 1.2: neutral
_PCR_EXTREME_HEDGING = 1.2
_PCR_EXTREME_COMPLACENCY = 0.5

# Confidence by source reliability:
# CBOE indices (VVIX, VIX9D, OVX, GVZ): well-constructed, highly liquid → 0.70
# Put/Call ratio: noisier, single-day reads frequently misleading → 0.55
_INDEX_CONFIDENCE = 0.70
_PCR_CONFIDENCE = 0.55

# Decay rates: daily data, superseded by next day's print → moderate decay
_INDEX_DECAY = 0.30
_PCR_DECAY = 0.35


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class VIXFamilySignal:
    """A reading from one CBOE volatility index (VVIX, VIX9D, OVX, or GVZ).

    Carries the same signal metadata fields as other MIDGE signals so the
    convergence alerter can weight options domain signals against other domains.
    """

    index_name: str         # "VVIX", "VIX9D", "OVX", "GVZ"
    value: float            # Most recent closing value
    prior_value: float      # Previous day closing value
    change: float           # value - prior_value
    change_pct: float       # Percentage change
    date: str               # YYYY-MM-DD of the observation
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0 – 1.0 signal strength
    note: str               # Human-readable interpretation
    signal_source: str = "cboe_options"
    decay_rate: float = _INDEX_DECAY
    confidence: float = _INDEX_CONFIDENCE

    # For VIX9D only: the VIX spot value used in the comparison (populated
    # when a VIX9D signal is generated alongside live VIX context)
    vix_spot_reference: Optional[float] = None

    def to_plain_language(self) -> str:
        arrow = "+" if self.change >= 0 else ""
        return (
            f"{self.index_name} {self.value:.2f} ({arrow}{self.change_pct:.1f}%) "
            f"| {self.direction.upper()} | {self.note} | Date: {self.date}"
        )


@dataclass
class PutCallSignal:
    """CBOE daily equity Put/Call ratio signal.

    The put/call ratio is a contrarian sentiment indicator. Extremes in either
    direction tend to precede reversals, not continuation.
    """

    equity_pcr: float         # Equity-only P/C ratio (most reliable)
    index_pcr: Optional[float]   # Index P/C ratio (if available)
    total_pcr: Optional[float]   # Total P/C ratio (if available)
    date: str                 # YYYY-MM-DD
    direction: str            # "bullish" (extreme hedging), "bearish" (complacency), "neutral"
    strength: float           # 0.0 – 1.0
    note: str                 # Human-readable interpretation
    signal_source: str = "cboe_options"
    decay_rate: float = _PCR_DECAY
    confidence: float = _PCR_CONFIDENCE

    def to_plain_language(self) -> str:
        return (
            f"Equity P/C {self.equity_pcr:.2f} "
            f"| {self.direction.upper()} (contrarian) | {self.note} | Date: {self.date}"
        )


# ---------------------------------------------------------------------------
# Direction / strength helpers
# ---------------------------------------------------------------------------

def _classify_vvix(value: float, change: float) -> Tuple[str, float, str]:
    """Return (direction, strength, note) for a VVIX reading."""
    if value >= _VVIX_HIGH:
        strength = min(1.0, (value - _VVIX_HIGH) / 40.0 + 0.7)
        return "bearish", round(strength, 3), f"VVIX {value:.1f} — extreme fear-of-fear"
    if value >= _VVIX_MODERATE:
        strength = min(0.69, (value - _VVIX_MODERATE) / (_VVIX_HIGH - _VVIX_MODERATE) * 0.5 + 0.2)
        return "neutral", round(strength, 3), f"VVIX {value:.1f} — elevated uncertainty"
    # Below 100: calm options market, mild bullish read
    strength = min(0.5, max(0.1, (1.0 - value / _VVIX_MODERATE) * 0.5 + 0.1))
    return "bullish", round(strength, 3), f"VVIX {value:.1f} — low fear-of-fear"


def _classify_vix9d(
    value: float, prior: float, vix_spot: Optional[float] = None
) -> Tuple[str, float, str]:
    """Return (direction, strength, note) for a VIX9D reading.

    Primary signal: VIX9D > VIX spot → fear is front-loaded → bearish.
    Fallback when VIX spot not available: compare against prior VIX9D reading.
    """
    if vix_spot is not None:
        spread = value - vix_spot
        if spread > 2.0:
            strength = min(1.0, spread / 8.0 + 0.5)
            return "bearish", round(strength, 3), (
                f"VIX9D {value:.1f} > VIX {vix_spot:.1f} by {spread:.1f} "
                f"— near-term fear accelerating"
            )
        if spread > 0.5:
            return "neutral", 0.4, (
                f"VIX9D {value:.1f} slightly above VIX {vix_spot:.1f} "
                f"— watch for escalation"
            )
        return "bullish", 0.3, (
            f"VIX9D {value:.1f} below VIX {vix_spot:.1f} "
            f"— near-term fear contained"
        )
    # No VIX spot: use day-over-day change as fallback
    pct_change = (value - prior) / prior * 100 if prior else 0.0
    if pct_change > 10.0:
        return "bearish", min(1.0, pct_change / 30.0 + 0.4), (
            f"VIX9D jumped {pct_change:.1f}% to {value:.1f}"
        )
    if pct_change < -10.0:
        return "bullish", min(0.7, abs(pct_change) / 30.0 + 0.3), (
            f"VIX9D fell {pct_change:.1f}% to {value:.1f}"
        )
    return "neutral", 0.2, f"VIX9D {value:.1f} — stable"


def _classify_ovx(value: float, change: float) -> Tuple[str, float, str]:
    """Return (direction, strength, note) for an OVX reading."""
    if value >= _OVX_EXTREME:
        strength = min(1.0, (value - _OVX_EXTREME) / 30.0 + 0.65)
        return "bearish", round(strength, 3), (
            f"OVX {value:.1f} — extreme crude oil uncertainty"
        )
    if value >= _OVX_ELEVATED:
        strength = min(0.64, (value - _OVX_ELEVATED) / (_OVX_EXTREME - _OVX_ELEVATED) * 0.4 + 0.25)
        return "neutral", round(strength, 3), (
            f"OVX {value:.1f} — elevated oil vol, watch energy names"
        )
    return "bullish", 0.25, f"OVX {value:.1f} — calm crude vol"


def _classify_gvz(value: float, change: float) -> Tuple[str, float, str]:
    """Return (direction, strength, note) for a GVZ reading."""
    if value >= _GVZ_HIGH:
        strength = min(1.0, (value - _GVZ_HIGH) / 20.0 + 0.6)
        return "bearish", round(strength, 3), (
            f"GVZ {value:.1f} — elevated safe-haven demand with high uncertainty"
        )
    if value >= _GVZ_MODERATE:
        strength = min(0.59, (value - _GVZ_MODERATE) / (_GVZ_HIGH - _GVZ_MODERATE) * 0.4 + 0.2)
        return "neutral", round(strength, 3), f"GVZ {value:.1f} — moderate gold vol"
    return "bullish", 0.2, f"GVZ {value:.1f} — low gold vol, risk-on environment"


def _classify_pcr(equity_pcr: float) -> Tuple[str, float, str]:
    """Return (direction, strength, note) for an equity put/call ratio.

    Contrarian: extreme hedging (high PCR) → contrarian bullish.
    Extreme complacency (low PCR) → contrarian bearish.
    """
    if equity_pcr > _PCR_EXTREME_HEDGING:
        strength = min(1.0, (equity_pcr - _PCR_EXTREME_HEDGING) / 0.5 + 0.55)
        return "bullish", round(strength, 3), (
            f"P/C {equity_pcr:.2f} — extreme hedging demand, contrarian bullish setup"
        )
    if equity_pcr < _PCR_EXTREME_COMPLACENCY:
        strength = min(1.0, (_PCR_EXTREME_COMPLACENCY - equity_pcr) / 0.3 + 0.5)
        return "bearish", round(strength, 3), (
            f"P/C {equity_pcr:.2f} — extreme complacency, contrarian bearish setup"
        )
    # Neutral zone
    mid = (_PCR_EXTREME_HEDGING + _PCR_EXTREME_COMPLACENCY) / 2
    distance = abs(equity_pcr - mid) / mid
    strength = round(0.1 + distance * 0.3, 3)
    return "neutral", strength, f"P/C {equity_pcr:.2f} — normal hedging activity"


# Dispatcher for per-index classification
_CLASSIFIERS = {
    "VVIX": _classify_vvix,
    "OVX": _classify_ovx,
    "GVZ": _classify_gvz,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CBOEOptionsClient:
    """
    Client for CBOE free options market data.

    Fetches VVIX, VIX9D, OVX, GVZ, and the equity Put/Call ratio from CBOE's
    public CSV feeds. No API key required.

    All four VIX-family indices are fetched together via get_vix_family().
    The put/call ratio is fetched separately via get_put_call_ratio().

    Signals belong to the 'options' domain in MIDGE's convergence engine.

    Rate limited to 1 req/2s. 4-hour cache per index.
    """

    def __init__(self, provider=None, raw_store=None):
        self._provider = provider
        self._raw_store = raw_store
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MIDGE Trading Research"})
        self._last_request_time: float = 0.0

        # Per-index cache: index_name -> (VIXFamilySignal, timestamp)
        self._vix_cache: Dict[str, Tuple[VIXFamilySignal, float]] = {}

        # Put/call cache: (PutCallSignal, timestamp) or None
        self._pcr_cache: Optional[Tuple[PutCallSignal, float]] = None

    # -----------------------------------------------------------------------
    # Rate limiting
    # -----------------------------------------------------------------------

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    # -----------------------------------------------------------------------
    # HTTP
    # -----------------------------------------------------------------------

    def _fetch_text(self, url: str, source_name: str) -> Optional[str]:
        """Fetch raw CSV text from a URL.

        Routes through MarketDataProvider when one is available (honours
        the same BoundaryMembrane + immune system as all other MIDGE clients),
        otherwise falls back to a direct requests.Session call.
        """
        if self._provider is not None:
            from mae_core.market.apis.market_data_provider import market_request
            from mae_core.external.api_client import ApiResponseStatus

            resp = market_request(
                self._provider, url,
                source_name=source_name, timeout_ms=30000.0,
            )
            if resp.status == ApiResponseStatus.SUCCESS:
                payload = resp.payload
                if isinstance(payload, dict) and "text" in payload:
                    return payload["text"]
                if isinstance(payload, str):
                    return payload
            return None

        try:
            r = self.session.get(url, timeout=30)
            if r.status_code == 200:
                return r.text
            logger.warning("CBOE fetch HTTP %d for %s", r.status_code, url)
            return None
        except Exception as exc:
            logger.warning("CBOE fetch error for %s: %s", url, exc)
            return None

    # -----------------------------------------------------------------------
    # CSV parsing helpers
    # -----------------------------------------------------------------------

    def _parse_index_csv(self, text: str, index_name: str) -> Optional[VIXFamilySignal]:
        """Parse a CBOE daily-prices CSV for any VIX-family index.

        CSV format (same across all CBOE indices):
            DATE,OPEN,HIGH,LOW,CLOSE
            01/03/2006,17.65,17.97,16.72,16.97
            ...
        Returns a VIXFamilySignal for the most recent trading day.
        """
        reader = csv.DictReader(io.StringIO(text))
        rows: List[Dict] = []
        raw_rows: List[Dict] = []

        for row in reader:
            try:
                close = float(row.get("CLOSE", row.get("Close", 0) or 0))
                date_str = (row.get("DATE", row.get("Date", "")) or "").strip()
                if close > 0 and date_str:
                    rows.append({"date": date_str[:10], "close": close})
                    raw_rows.append({
                        "date": date_str[:10],
                        "open": _safe_float(row.get("OPEN", row.get("Open"))),
                        "high": _safe_float(row.get("HIGH", row.get("High"))),
                        "low": _safe_float(row.get("LOW", row.get("Low"))),
                        "close": close,
                    })
            except (ValueError, TypeError):
                continue

        if self._raw_store and raw_rows:
            self._store_index_raw(index_name, raw_rows)

        if not rows:
            return None

        rows.sort(key=lambda r: r["date"], reverse=True)
        current = rows[0]
        prior = rows[1] if len(rows) >= 2 else rows[0]

        value = current["close"]
        prior_value = prior["close"]
        change = value - prior_value
        change_pct = (change / prior_value * 100) if prior_value else 0.0

        classifier = _CLASSIFIERS.get(index_name)
        if classifier:
            direction, strength, note = classifier(value, change)
        else:
            direction, strength, note = "neutral", 0.2, f"{index_name} {value:.2f}"

        return VIXFamilySignal(
            index_name=index_name,
            value=round(value, 2),
            prior_value=round(prior_value, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            date=current["date"],
            direction=direction,
            strength=strength,
            note=note,
        )

    def _parse_pcr_csv(self, text: str) -> Optional[PutCallSignal]:
        """Parse CBOE daily market statistics CSV for the put/call ratio.

        The CSV has varying column names across CBOE releases. We search for
        columns containing "equity" and "put" to be resilient to layout changes.

        Typical columns (as of 2025):
            DATE, TOTAL PUT/CALL RATIO, INDEX PUT/CALL RATIO,
            EQUITY PUT/CALL RATIO, CBOE VOLATILITY INDEX (VIX)
        """
        reader = csv.DictReader(io.StringIO(text))
        rows: List[Dict] = []
        raw_rows: List[Dict] = []

        for row in reader:
            try:
                date_str = (row.get("DATE", row.get("Date", "")) or "").strip()
                if not date_str:
                    continue

                equity_pcr = _extract_pcr_field(row, ("EQUITY PUT/CALL RATIO",
                                                       "Equity Put/Call Ratio",
                                                       "EQUITY P/C",
                                                       "Equity P/C"))
                if equity_pcr is None or equity_pcr <= 0:
                    continue

                total_pcr = _extract_pcr_field(row, ("TOTAL PUT/CALL RATIO",
                                                      "Total Put/Call Ratio",
                                                      "TOTAL P/C",
                                                      "Total P/C"))
                index_pcr = _extract_pcr_field(row, ("INDEX PUT/CALL RATIO",
                                                      "Index Put/Call Ratio",
                                                      "INDEX P/C",
                                                      "Index P/C"))

                rows.append({
                    "date": date_str[:10],
                    "equity_pcr": equity_pcr,
                    "total_pcr": total_pcr,
                    "index_pcr": index_pcr,
                })
                raw_rows.append({
                    "date": date_str[:10],
                    "equity_pcr": equity_pcr,
                    "total_pcr": total_pcr,
                    "index_pcr": index_pcr,
                    "raw": dict(row),
                })
            except (ValueError, TypeError):
                continue

        if self._raw_store and raw_rows:
            self._store_pcr_raw(raw_rows)

        if not rows:
            logger.warning("CBOE P/C CSV parsed 0 usable rows — column layout may have changed")
            return None

        rows.sort(key=lambda r: r["date"], reverse=True)
        latest = rows[0]

        equity_pcr = latest["equity_pcr"]
        direction, strength, note = _classify_pcr(equity_pcr)

        return PutCallSignal(
            equity_pcr=round(equity_pcr, 3),
            index_pcr=round(latest["index_pcr"], 3) if latest["index_pcr"] else None,
            total_pcr=round(latest["total_pcr"], 3) if latest["total_pcr"] else None,
            date=latest["date"],
            direction=direction,
            strength=strength,
            note=note,
        )

    # -----------------------------------------------------------------------
    # Raw store helpers
    # -----------------------------------------------------------------------

    def _store_index_raw(self, index_name: str, rows: List[Dict]) -> None:
        """Persist raw CBOE index data via raw_store if available."""
        try:
            # Use store_vix_daily for VIX-family indices — same schema.
            # Table names are isolated per-database, so VVIX/OVX/GVZ rows
            # go into their own SQLite files via a generic volatility method.
            if hasattr(self._raw_store, "store_cboe_index_daily"):
                self._raw_store.store_cboe_index_daily(index_name.lower(), rows)
            elif hasattr(self._raw_store, "store_vix_daily") and index_name == "VIX9D":
                # VIX9D is close enough to reuse the VIX table
                self._raw_store.store_vix_daily(rows)
            else:
                # Fallback: store via generic method if available
                if hasattr(self._raw_store, "store_generic"):
                    self._raw_store.store_generic(
                        f"cboe_{index_name.lower()}_daily", rows
                    )
        except Exception as exc:
            logger.debug("RawStore CBOE %s write failed: %s", index_name, exc)

    def _store_pcr_raw(self, rows: List[Dict]) -> None:
        """Persist raw put/call ratio data via raw_store if available."""
        try:
            if hasattr(self._raw_store, "store_cboe_pcr_daily"):
                self._raw_store.store_cboe_pcr_daily(rows)
            elif hasattr(self._raw_store, "store_generic"):
                self._raw_store.store_generic("cboe_pcr_daily", rows)
        except Exception as exc:
            logger.debug("RawStore CBOE P/C write failed: %s", exc)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_vix_family(
        self, vix_spot: Optional[float] = None
    ) -> List[VIXFamilySignal]:
        """
        Fetch VVIX, VIX9D, OVX, and GVZ from CBOE's public CSV feeds.

        Each index is fetched independently with rate limiting and a 4-hour
        cache. Results are returned in a consistent order (VVIX, VIX9D, OVX, GVZ).

        Args:
            vix_spot: Current VIX spot level, used to compute the VIX9D spread
                      signal (VIX9D > VIX means near-term fear is accelerating).
                      Pass from VIXClient.get_vix_structure().vix_spot if available.

        Returns:
            List of VIXFamilySignal objects. Partial results if some fetches fail.
        """
        results: List[VIXFamilySignal] = []

        for index_name, url in _VIX_FAMILY_URLS.items():
            # Check per-index cache
            cached = self._vix_cache.get(index_name)
            if cached is not None:
                signal, cached_at = cached
                if time.time() - cached_at < CACHE_DURATION:
                    results.append(signal)
                    continue

            self._rate_limit()
            text = self._fetch_text(url, source_name=f"cboe_{index_name.lower()}")
            if not text:
                logger.warning("CBOE: failed to fetch %s", index_name)
                continue

            try:
                if index_name == "VIX9D":
                    signal = self._parse_index_csv(text, index_name)
                    if signal and vix_spot is not None:
                        # Re-classify with VIX spot context
                        dir_, str_, note = _classify_vix9d(
                            signal.value, signal.prior_value, vix_spot
                        )
                        signal.direction = dir_
                        signal.strength = str_
                        signal.note = note
                        signal.vix_spot_reference = vix_spot
                    elif signal:
                        # Fallback classification without VIX spot
                        dir_, str_, note = _classify_vix9d(
                            signal.value, signal.prior_value, None
                        )
                        signal.direction = dir_
                        signal.strength = str_
                        signal.note = note
                else:
                    signal = self._parse_index_csv(text, index_name)
            except Exception as exc:
                logger.warning("CBOE: parse error for %s: %s", index_name, exc)
                continue

            if signal is None:
                logger.warning("CBOE: no usable data in %s CSV", index_name)
                continue

            self._vix_cache[index_name] = (signal, time.time())
            results.append(signal)
            logger.debug(
                "CBOE %s: %.2f (%+.1f%%) → %s [%s]",
                index_name, signal.value, signal.change_pct,
                signal.direction, signal.date,
            )

        return results

    def get_put_call_ratio(self) -> Optional[PutCallSignal]:
        """
        Fetch the most recent CBOE equity Put/Call ratio.

        The equity P/C ratio is a contrarian sentiment signal. Extreme readings
        (> 1.2 or < 0.5) historically precede reversals, not continuation.

        The underlying CSV may change layout — column extraction is resilient
        to naming variations. Returns None if the URL is unavailable or the
        column layout is unrecognisable.

        Returns:
            PutCallSignal with equity P/C, direction, strength, and note,
            or None if data unavailable.
        """
        if self._pcr_cache is not None:
            signal, cached_at = self._pcr_cache
            if time.time() - cached_at < CACHE_DURATION:
                return signal

        self._rate_limit()
        text = self._fetch_text(_PUT_CALL_URL, source_name="cboe_pcr")
        if not text:
            logger.warning(
                "CBOE P/C fetch failed. URL may have changed: %s", _PUT_CALL_URL
            )
            return None

        try:
            signal = self._parse_pcr_csv(text)
        except Exception as exc:
            logger.warning("CBOE P/C parse error: %s", exc)
            return None

        if signal is None:
            return None

        self._pcr_cache = (signal, time.time())
        logger.debug(
            "CBOE P/C: equity=%.3f → %s [%s]",
            signal.equity_pcr, signal.direction, signal.date,
        )
        return signal

    def get_options_snapshot(
        self, vix_spot: Optional[float] = None
    ) -> Dict[str, object]:
        """
        Fetch all CBOE options signals in one call.

        Convenience method that runs get_vix_family() and get_put_call_ratio()
        sequentially. Suitable for the sensing hook step cadence.

        Args:
            vix_spot: Optional VIX spot for VIX9D spread calculation.

        Returns:
            Dict with keys:
                'vix_family': List[VIXFamilySignal]
                'put_call': Optional[PutCallSignal]
        """
        return {
            "vix_family": self.get_vix_family(vix_spot=vix_spot),
            "put_call": self.get_put_call_ratio(),
        }


# ---------------------------------------------------------------------------
# Parsing utilities
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    """Convert a raw CSV value to float, returning None on failure."""
    try:
        v = float(value)
        return v if v != 0 else None
    except (ValueError, TypeError):
        return None


def _extract_pcr_field(row: Dict, candidates: tuple) -> Optional[float]:
    """Try each candidate key in order, returning the first parseable float."""
    for key in candidates:
        raw = row.get(key)
        if raw is not None:
            try:
                v = float(raw)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def get_vix_family(vix_spot: Optional[float] = None) -> List[VIXFamilySignal]:
    """One-shot fetch of VVIX, VIX9D, OVX, GVZ."""
    return CBOEOptionsClient().get_vix_family(vix_spot=vix_spot)


def get_put_call_ratio() -> Optional[PutCallSignal]:
    """One-shot fetch of the CBOE equity put/call ratio."""
    return CBOEOptionsClient().get_put_call_ratio()


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Testing CBOE Options Client...")
    print("=" * 60)

    client = CBOEOptionsClient()

    print("\nVIX Family (VVIX, VIX9D, OVX, GVZ):")
    family = client.get_vix_family()
    if family:
        for sig in family:
            arrow = {"bullish": "[+]", "bearish": "[-]", "neutral": "[~]"}.get(
                sig.direction, "[?]"
            )
            print(f"  {arrow} {sig.to_plain_language()}")
    else:
        print("  No data returned")

    print("\nPut/Call Ratio:")
    pcr = client.get_put_call_ratio()
    if pcr:
        arrow = {"bullish": "[+]", "bearish": "[-]", "neutral": "[~]"}.get(
            pcr.direction, "[?]"
        )
        print(f"  {arrow} {pcr.to_plain_language()}")
        if pcr.index_pcr:
            print(f"  Index P/C: {pcr.index_pcr:.3f}")
        if pcr.total_pcr:
            print(f"  Total P/C: {pcr.total_pcr:.3f}")
    else:
        print("  No data returned — P/C URL may need updating")

    print("\nDone.")
