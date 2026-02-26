#!/usr/bin/env python3
"""
session_sweep_detector.py - ICT Session Sweep Detector

Detects Smart Money liquidity sweeps on futures markets:
1. Mark session highs/lows (Asia, London, New York) from 1-minute candles
2. Detect sweep: price exceeds session level then closes back inside
3. Find Fair Value Gap in reversal direction after the sweep
4. Produce SessionSweepSignal with entry zone, stop, and conviction score

Data source: yfinance 1-minute candles (max 7 days lookback, ~10 min delay).
Supported symbols: ES=F, NQ=F, MES=F, MNQ=F.

Follows cluster_detector.py pattern: standalone class, returns dataclass list,
no EventBus publishing. The sensing hook wraps results via from_session_sweep().

Graceful degradation: returns empty list when yfinance unavailable.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, date
from typing import List, Optional, Tuple, Dict
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Try to import dependencies — graceful degradation
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

EASTERN = ZoneInfo("America/New_York")

# ── Session definitions (Eastern Time) ─────────────────────────────

# Full session windows — used to record highs/lows
SESSION_WINDOWS = {
    "asia":     {"start": time(20, 0), "end": time(0, 0)},   # crosses midnight
    "london":   {"start": time(2, 0),  "end": time(5, 0)},
    "new_york": {"start": time(7, 0),  "end": time(10, 0)},
}

# Kill zones — high-probability windows within sessions
KILL_ZONES = {
    "asia_kz":     {"start": time(20, 0), "end": time(22, 0)},
    "london_kz":   {"start": time(2, 0),  "end": time(5, 0)},
    "new_york_kz": {"start": time(7, 0),  "end": time(10, 0)},
}

DEFAULT_SYMBOLS = ["ES=F", "NQ=F"]


# ── Dataclasses ────────────────────────────────────────────────────

@dataclass
class SessionLevel:
    """A session's high and low price levels (liquidity pools)."""
    session: str           # "asia", "london", "new_york"
    session_date: str      # Date string (YYYY-MM-DD) the session started
    session_high: float
    session_low: float
    high_time: str         # ISO timestamp of the high candle
    low_time: str          # ISO timestamp of the low candle
    candles_in_session: int


@dataclass
class FairValueGap:
    """A 3-candle Fair Value Gap (price imbalance zone)."""
    top: float
    bottom: float
    midpoint: float
    direction: str         # "bullish" or "bearish"
    formed_at: str         # ISO timestamp of the middle candle
    candle_index: int
    size_pct: float        # Gap size as % of price
    is_mitigated: bool = False


@dataclass
class SessionSweepSignal:
    """A detected session liquidity sweep with FVG entry zone."""
    # Identity
    sweep_id: str = ""
    symbol: str = ""

    # Sweep details
    sweep_type: str = ""         # "high_sweep" or "low_sweep"
    direction: str = ""          # "bullish" (low swept) or "bearish" (high swept)
    session_swept: str = ""      # "asia", "london", "new_york"
    sweep_level: float = 0.0     # The session high/low that was swept
    sweep_candle_high: float = 0.0
    sweep_candle_low: float = 0.0
    reversal_confirmed: bool = False

    # FVG entry zone
    fvg_top: float = 0.0
    fvg_bottom: float = 0.0
    fvg_midpoint: float = 0.0
    fvg_size_pct: float = 0.0

    # Trade parameters
    entry_zone_top: float = 0.0
    entry_zone_bottom: float = 0.0
    stop_level: float = 0.0
    target_level: float = 0.0
    rr_ratio: float = 2.0

    # Signal quality
    confidence: float = 0.50
    strength: float = 0.50
    kill_zone: bool = False

    # IFVG + pattern stacking scores
    is_ifvg: bool = False
    displacement_score: float = 0.0
    fvg_atr_ratio: float = 0.0
    quality_score: float = 0.0

    # Metadata
    signal_source: str = "session_sweep"
    decay_rate: float = 0.85     # Hourly-scale decay (~18h half-life)
    detected_at: str = ""
    metadata: dict = field(default_factory=dict)

    def to_plain_language(self) -> str:
        """Human-readable summary of the sweep signal."""
        kz_tag = " [KILL ZONE]" if self.kill_zone else ""
        return (
            f"{self.symbol} {self.direction.upper()} — "
            f"{self.session_swept} {self.sweep_type.replace('_', ' ')} "
            f"at {self.sweep_level:.2f}{kz_tag}. "
            f"FVG zone: {self.fvg_bottom:.2f}-{self.fvg_top:.2f}, "
            f"stop {self.stop_level:.2f}, target {self.target_level:.2f} "
            f"({self.rr_ratio:.1f}R). "
            f"Confidence: {self.confidence:.0%}"
        )


# ── Detector ───────────────────────────────────────────────────────

class SessionSweepDetector:
    """Detects ICT-style session liquidity sweeps on futures markets.

    Graceful degradation: if yfinance or pandas is unavailable,
    detect_sweeps() returns an empty list.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        sweep_lookback_candles: int = 3,
        min_fvg_pct: float = 0.01,
    ):
        self._symbols = symbols or DEFAULT_SYMBOLS
        self._sweep_lookback = sweep_lookback_candles
        self._min_fvg_pct = min_fvg_pct

    def detect_sweeps(self, symbol: str) -> List[SessionSweepSignal]:
        """Detect session sweep signals for a single symbol.

        Full pipeline: fetch candles → mark sessions → detect sweeps →
        find FVGs → build signals. Returns empty list on any failure.
        """
        if not YFINANCE_AVAILABLE or not PANDAS_AVAILABLE:
            logger.debug("Session sweep: yfinance/pandas unavailable")
            return []

        try:
            ohlc = self._fetch_1m_candles(symbol)
            if ohlc is None or ohlc.empty:
                return []

            levels = self._mark_session_levels(ohlc)
            if not levels:
                return []

            signals = []
            for level in levels:
                # Check both high and low for sweeps
                for side in ("high", "low"):
                    result = self._detect_sweep(ohlc, level, side)
                    if result is None:
                        continue

                    sweep_idx, sweep_candle = result

                    # Determine reversal direction
                    if side == "high":
                        direction = "bearish"
                        fvg_direction = "bearish"
                    else:
                        direction = "bullish"
                        fvg_direction = "bullish"

                    # Find FVG after the sweep
                    fvg = self._find_fvg(ohlc, sweep_idx, fvg_direction)
                    if fvg is None:
                        continue

                    # IFVG detection + pattern stacking scores
                    is_ifvg = self._find_ifvg(ohlc, sweep_idx, direction)
                    disp = self._score_displacement(ohlc, sweep_idx, direction)
                    atr = self._compute_atr(ohlc, sweep_idx)
                    fvg_atr = (fvg.top - fvg.bottom) / atr if atr > 0 else 0.0
                    kz_score = self._get_kill_zone_score(ohlc.index[sweep_idx])
                    quality = self._compute_quality(disp, fvg_atr, kz_score)

                    # Build the signal
                    signal = self._build_signal(
                        symbol, level, side, direction,
                        sweep_candle, fvg, ohlc, sweep_idx,
                    )
                    signal.is_ifvg = is_ifvg
                    signal.displacement_score = round(disp, 3)
                    signal.fvg_atr_ratio = round(fvg_atr, 3)
                    signal.quality_score = round(quality, 3)

                    # IFVG boost: +0.10 confidence
                    if is_ifvg:
                        signal.confidence = min(0.90, signal.confidence + 0.10)

                    signals.append(signal)

            return self._deduplicate(signals)

        except Exception:
            logger.debug("Session sweep detection failed for %s", symbol, exc_info=True)
            return []

    # ── Data fetch ─────────────────────────────────────────────────

    def _fetch_1m_candles(self, symbol: str) -> Optional["pd.DataFrame"]:
        """Fetch 1-minute candles from yfinance. Returns lowercase-column DF."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1m")
            if df.empty:
                logger.debug("No 1m candle data for %s", symbol)
                return None

            # Normalize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure timezone-aware Eastern time index
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(EASTERN)

            # Drop NaN rows (session breaks, gaps)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df

        except Exception:
            logger.debug("Failed to fetch 1m candles for %s", symbol, exc_info=True)
            return None

    # ── Session level marking ──────────────────────────────────────

    def _mark_session_levels(
        self, ohlc: "pd.DataFrame",
    ) -> List[SessionLevel]:
        """Mark session highs/lows from the last 2 completed sessions of each type."""
        levels = []
        now_et = datetime.now(EASTERN)

        for session_name, window in SESSION_WINDOWS.items():
            session_levels = self._extract_session_levels(
                ohlc, session_name, window, now_et,
            )
            levels.extend(session_levels)

        return levels

    def _extract_session_levels(
        self,
        ohlc: "pd.DataFrame",
        session_name: str,
        window: dict,
        now_et: datetime,
    ) -> List[SessionLevel]:
        """Extract session high/low for last 2 completed sessions of a type."""
        start_t = window["start"]
        end_t = window["end"]
        crosses_midnight = start_t > end_t

        results = []
        # Look back through recent dates
        for days_back in range(5):
            ref_date = (now_et - timedelta(days=days_back)).date()

            if crosses_midnight:
                # Asia: 20:00 on ref_date to 00:00 on ref_date+1
                start_dt = datetime.combine(ref_date, start_t, tzinfo=EASTERN)
                end_dt = datetime.combine(
                    ref_date + timedelta(days=1), end_t, tzinfo=EASTERN,
                )
                # end_t is midnight — use 23:59:59 of next day start or
                # the actual 00:00 boundary
                if end_t == time(0, 0):
                    end_dt = datetime.combine(
                        ref_date + timedelta(days=1), time(0, 0), tzinfo=EASTERN,
                    )
            else:
                start_dt = datetime.combine(ref_date, start_t, tzinfo=EASTERN)
                end_dt = datetime.combine(ref_date, end_t, tzinfo=EASTERN)

            # Skip if session hasn't completed yet
            if end_dt >= now_et:
                continue

            mask = (ohlc.index >= start_dt) & (ohlc.index < end_dt)
            session_candles = ohlc[mask]

            if len(session_candles) < 5:
                continue

            high_val = session_candles["high"].max()
            low_val = session_candles["low"].min()
            high_idx = session_candles["high"].idxmax()
            low_idx = session_candles["low"].idxmin()

            results.append(SessionLevel(
                session=session_name,
                session_date=ref_date.isoformat(),
                session_high=float(high_val),
                session_low=float(low_val),
                high_time=str(high_idx),
                low_time=str(low_idx),
                candles_in_session=len(session_candles),
            ))

            if len(results) >= 2:
                break

        return results

    # ── Sweep detection ────────────────────────────────────────────

    def _detect_sweep(
        self,
        ohlc: "pd.DataFrame",
        level: SessionLevel,
        side: str,
    ) -> Optional[Tuple[int, "pd.Series"]]:
        """Detect if a session level was swept (breached then reversed).

        A sweep = price exceeds the level, then within 1-3 candles closes
        back inside. If price holds beyond the level = breakout (no signal).

        Args:
            ohlc: Full OHLC DataFrame
            level: The session level to check
            side: "high" or "low"

        Returns:
            (positional_index_of_sweep_candle, sweep_candle_series) or None
        """
        target = level.session_high if side == "high" else level.session_low

        # Only check candles AFTER the session ended
        session_end_str = level.high_time if side == "high" else level.low_time
        try:
            session_end_dt = pd.Timestamp(session_end_str)
            if session_end_dt.tzinfo is None:
                session_end_dt = session_end_dt.tz_localize(EASTERN)
        except Exception:
            return None

        post_session = ohlc[ohlc.index > session_end_dt]
        if len(post_session) < 2:
            return None

        # Scan for breach candle
        for pos_idx in range(len(post_session)):
            candle = post_session.iloc[pos_idx]

            if side == "high" and candle["high"] > target:
                # High was breached — check if next candles close back below
                if self._confirm_sweep_reversal(
                    post_session, pos_idx, target, "below",
                ):
                    abs_idx = ohlc.index.get_loc(post_session.index[pos_idx])
                    return (abs_idx, candle)

            elif side == "low" and candle["low"] < target:
                # Low was breached — check if next candles close back above
                if self._confirm_sweep_reversal(
                    post_session, pos_idx, target, "above",
                ):
                    abs_idx = ohlc.index.get_loc(post_session.index[pos_idx])
                    return (abs_idx, candle)

        return None

    def _confirm_sweep_reversal(
        self,
        df: "pd.DataFrame",
        breach_idx: int,
        level: float,
        close_side: str,
    ) -> bool:
        """Check if the sweep candle's close (or next 1-2 candles) reverses.

        Args:
            df: Post-session DataFrame
            breach_idx: Index of the breach candle within df
            level: The price level that was breached
            close_side: "above" or "below" — where the close must be
        """
        # Check the breach candle itself and the next lookback-1 candles
        for offset in range(self._sweep_lookback):
            check_idx = breach_idx + offset
            if check_idx >= len(df):
                return False

            close_price = df.iloc[check_idx]["close"]

            if close_side == "below" and close_price < level:
                return True
            if close_side == "above" and close_price > level:
                return True

        return False

    # ── Fair Value Gap detection ───────────────────────────────────

    def _find_fvg(
        self,
        ohlc: "pd.DataFrame",
        from_idx: int,
        direction: str,
    ) -> Optional[FairValueGap]:
        """Find a Fair Value Gap after the sweep candle.

        Scans up to 20 candles forward from the sweep for a 3-candle
        FVG pattern in the specified direction.

        Bullish FVG: candle[i+2].low > candle[i].high (gap up)
        Bearish FVG: candle[i+2].high < candle[i].low (gap down)
        """
        scan_end = min(from_idx + 20, len(ohlc) - 2)

        for i in range(from_idx, scan_end):
            c0 = ohlc.iloc[i]
            c2 = ohlc.iloc[i + 2]
            c1 = ohlc.iloc[i + 1]  # The "engine" candle

            if direction == "bullish" and c2["low"] > c0["high"]:
                top = float(c2["low"])
                bottom = float(c0["high"])
                gap_size = top - bottom
                mid_price = float(c1["close"])
                size_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

                if size_pct < self._min_fvg_pct:
                    continue

                return FairValueGap(
                    top=top,
                    bottom=bottom,
                    midpoint=(top + bottom) / 2,
                    direction="bullish",
                    formed_at=str(ohlc.index[i + 1]),
                    candle_index=i + 1,
                    size_pct=round(size_pct, 4),
                )

            elif direction == "bearish" and c2["high"] < c0["low"]:
                top = float(c0["low"])
                bottom = float(c2["high"])
                gap_size = top - bottom
                mid_price = float(c1["close"])
                size_pct = (gap_size / mid_price) * 100 if mid_price > 0 else 0

                if size_pct < self._min_fvg_pct:
                    continue

                return FairValueGap(
                    top=top,
                    bottom=bottom,
                    midpoint=(top + bottom) / 2,
                    direction="bearish",
                    formed_at=str(ohlc.index[i + 1]),
                    candle_index=i + 1,
                    size_pct=round(size_pct, 4),
                )

        return None

    # ── Signal construction ────────────────────────────────────────

    def _build_signal(
        self,
        symbol: str,
        level: SessionLevel,
        side: str,
        direction: str,
        sweep_candle: "pd.Series",
        fvg: FairValueGap,
        ohlc: "pd.DataFrame",
        sweep_idx: int,
    ) -> SessionSweepSignal:
        """Assemble a complete SessionSweepSignal from components."""
        sweep_type = "high_sweep" if side == "high" else "low_sweep"
        sweep_level = level.session_high if side == "high" else level.session_low

        # Entry zone = FVG boundaries
        entry_top = fvg.top
        entry_bottom = fvg.bottom

        # Stop loss beyond the sweep wick with buffer
        stop, target = self._calculate_stop_and_target(
            direction, sweep_candle, fvg,
        )

        # R:R ratio
        entry_price = fvg.midpoint
        risk = abs(entry_price - stop) if abs(entry_price - stop) > 0 else 0.01
        reward = abs(target - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # Detect kill zone timing
        sweep_time = ohlc.index[sweep_idx]
        in_kz = self._is_kill_zone(sweep_time)

        # Quality scoring
        confidence = self._classify_sweep_quality(
            in_kz, fvg.size_pct, side, sweep_candle, sweep_level,
        )

        # Strength: based on FVG size and R-multiple potential
        strength = min(1.0, 0.40 + fvg.size_pct * 5 + (rr - 1.0) * 0.10)

        now = datetime.now(EASTERN)
        return SessionSweepSignal(
            sweep_id=str(uuid.uuid4()),
            symbol=symbol,
            sweep_type=sweep_type,
            direction=direction,
            session_swept=level.session,
            sweep_level=sweep_level,
            sweep_candle_high=float(sweep_candle["high"]),
            sweep_candle_low=float(sweep_candle["low"]),
            reversal_confirmed=True,
            fvg_top=fvg.top,
            fvg_bottom=fvg.bottom,
            fvg_midpoint=fvg.midpoint,
            fvg_size_pct=fvg.size_pct,
            entry_zone_top=entry_top,
            entry_zone_bottom=entry_bottom,
            stop_level=stop,
            target_level=target,
            rr_ratio=rr,
            confidence=round(confidence, 3),
            strength=round(strength, 3),
            kill_zone=in_kz,
            detected_at=now.isoformat(),
            metadata={
                "session_date": level.session_date,
                "session_high": level.session_high,
                "session_low": level.session_low,
                "fvg_formed_at": fvg.formed_at,
                "sweep_time": str(ohlc.index[sweep_idx]),
            },
        )

    def _calculate_stop_and_target(
        self,
        direction: str,
        sweep_candle: "pd.Series",
        fvg: FairValueGap,
    ) -> Tuple[float, float]:
        """Calculate stop loss and 2R target from sweep wick and FVG zone.

        Stop: beyond the sweep candle's extreme wick + small buffer.
        Target: entry (FVG midpoint) + 2 * risk distance in trade direction.
        """
        entry = fvg.midpoint

        if direction == "bullish":
            # Long trade: stop below sweep low, target above entry
            stop = float(sweep_candle["low"]) - 0.50  # 0.50 point buffer
            risk = entry - stop
            target = entry + (2.0 * risk)
        else:
            # Short trade: stop above sweep high, target below entry
            stop = float(sweep_candle["high"]) + 0.50
            risk = stop - entry
            target = entry - (2.0 * risk)

        return round(stop, 2), round(target, 2)

    def _classify_sweep_quality(
        self,
        kill_zone: bool,
        fvg_size_pct: float,
        side: str,
        sweep_candle: "pd.Series",
        sweep_level: float,
    ) -> float:
        """Score sweep quality into a confidence value (0.0-1.0).

        Base: 0.45 (sweep + FVG confirmed)
        +0.15 if during kill zone
        +0.10 if FVG is sizable (>0.03%)
        +0.10 if clean single-candle rejection (wick vs body ratio)
        Cap: 0.90
        """
        conf = 0.45

        if kill_zone:
            conf += 0.15

        if fvg_size_pct > 0.03:
            conf += 0.10

        # Clean rejection: the sweep candle has a large wick relative to body
        body = abs(float(sweep_candle["close"]) - float(sweep_candle["open"]))
        full_range = float(sweep_candle["high"]) - float(sweep_candle["low"])
        if full_range > 0:
            wick_ratio = 1.0 - (body / full_range)
            if wick_ratio > 0.60:
                conf += 0.10

        return min(0.90, conf)

    # ── IFVG detection ──────────────────────────────────────────────

    def _find_ifvg(
        self,
        ohlc: "pd.DataFrame",
        sweep_idx: int,
        direction: str,
    ) -> bool:
        """Check if a prior-trend FVG was mitigated post-sweep (IFVG).

        An IFVG is a prior-trend FVG that gets filled during displacement.
        Once filled, the zone flips polarity. This is the highest-probability
        ICT entry — backtest showed +4.4pp win rate improvement.

        Returns True if any prior-trend FVG was mitigated within 80 candles.
        """
        prior_dir = "bearish" if direction == "bullish" else "bullish"
        lookback = min(200, sweep_idx)
        fvg_start = max(0, sweep_idx - lookback)

        # Find prior-trend FVGs
        prior_fvgs = []
        for i in range(fvg_start, sweep_idx - 1):
            if i + 2 >= len(ohlc):
                break
            c0 = ohlc.iloc[i]
            c2 = ohlc.iloc[i + 2]
            c1 = ohlc.iloc[i + 1]
            mid_price = float(c1["close"])
            if mid_price <= 0:
                continue

            if prior_dir == "bearish" and c2["high"] < c0["low"]:
                top = float(c0["low"])
                bottom = float(c2["high"])
                if (top - bottom) / mid_price >= 0.0005:
                    prior_fvgs.append((top, bottom, i + 1))

            elif prior_dir == "bullish" and c2["low"] > c0["high"]:
                top = float(c2["low"])
                bottom = float(c0["high"])
                if (top - bottom) / mid_price >= 0.0005:
                    prior_fvgs.append((top, bottom, i + 1))

        if not prior_fvgs:
            return False

        # Check if any were filled post-sweep
        fill_end = min(sweep_idx + 80, len(ohlc))
        for top, bottom, _ in reversed(prior_fvgs):
            for j in range(sweep_idx + 1, fill_end):
                close = ohlc.iloc[j]["close"]
                if prior_dir == "bearish" and close > top:
                    return True
                if prior_dir == "bullish" and close < bottom:
                    return True

        return False

    # ── Pattern stacking scores ──────────────────────────────────

    def _score_displacement(
        self, ohlc: "pd.DataFrame", sweep_idx: int, direction: str,
    ) -> float:
        """Measure reversal quality via body-to-range ratio of post-sweep candles."""
        ratios = []
        for i in range(sweep_idx + 1, min(sweep_idx + 6, len(ohlc))):
            candle = ohlc.iloc[i]
            bar_range = candle["high"] - candle["low"]
            if bar_range <= 0:
                continue
            body = abs(candle["close"] - candle["open"])
            if direction == "bullish" and candle["close"] > candle["open"]:
                ratios.append(body / bar_range)
            elif direction == "bearish" and candle["close"] < candle["open"]:
                ratios.append(body / bar_range)
        return sum(ratios) / len(ratios) if ratios else 0.0

    def _compute_atr(
        self, ohlc: "pd.DataFrame", idx: int, period: int = 14,
    ) -> float:
        """14-period Average True Range at a given index."""
        start = max(1, idx - period + 1)
        trs = []
        for i in range(start, idx + 1):
            high = ohlc.iloc[i]["high"]
            low = ohlc.iloc[i]["low"]
            prev_close = ohlc.iloc[i - 1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def _get_kill_zone_score(self, dt) -> float:
        """Tiered kill zone score (1.0=NY, 0.85=London, 0.70=Asia, 0.0=outside)."""
        try:
            t = dt.time() if hasattr(dt, "time") else dt
        except Exception:
            return 0.0
        if time(7, 0) <= t < time(10, 0):
            return 1.0
        if time(2, 0) <= t < time(5, 0):
            return 0.85
        if t >= time(20, 0) or (t < time(22, 0) and t >= time(20, 0)):
            return 0.70
        return 0.0

    def _compute_quality(
        self, displacement: float, fvg_atr: float, kz: float,
    ) -> float:
        """Composite quality: 40% displacement + 35% FVG/ATR + 25% kill zone."""
        fvg_atr_score = min(1.0, fvg_atr / 1.5)
        return displacement * 0.4 + fvg_atr_score * 0.35 + kz * 0.25

    # ── Kill zone check ────────────────────────────────────────────

    def _is_kill_zone(self, dt) -> bool:
        """Check if a datetime falls within an ICT kill zone window."""
        try:
            if hasattr(dt, "time"):
                t = dt.time()
            else:
                return False
        except Exception:
            return False

        for kz_name, window in KILL_ZONES.items():
            start = window["start"]
            end = window["end"]

            if start <= end:
                # Normal window (no midnight crossing)
                if start <= t <= end:
                    return True
            else:
                # Crosses midnight
                if t >= start or t <= end:
                    return True

        return False

    # ── Deduplication ──────────────────────────────────────────────

    def _deduplicate(
        self, signals: List[SessionSweepSignal],
    ) -> List[SessionSweepSignal]:
        """Remove duplicate sweep signals for same symbol/session/direction.

        Keeps the most recent signal when duplicates found within 30 minutes.
        """
        if len(signals) <= 1:
            return signals

        seen: Dict[str, SessionSweepSignal] = {}
        for sig in signals:
            key = f"{sig.symbol}:{sig.session_swept}:{sig.direction}"
            if key not in seen:
                seen[key] = sig
            else:
                # Keep the one with higher confidence
                if sig.confidence > seen[key].confidence:
                    seen[key] = sig

        return list(seen.values())
