#!/usr/bin/env python3
"""
session_sweep_detector.py - ICT Session Sweep Detector

Detects Smart Money liquidity sweeps on futures markets:
1. Mark session highs/lows (Asia, London, New York) from 1-minute candles
2. Detect sweep: price exceeds session level then closes back inside
3. Find Fair Value Gap in reversal direction after the sweep
4. Produce SessionSweepSignal with entry zone, stop, and conviction score

Sub-modules:
  session_sweep_models.py   — SessionLevel, FairValueGap, SessionSweepSignal
  session_sweep_geometry.py — mark_session_levels, extract_session_levels,
                               find_fvg, find_ifvg, score_displacement, compute_atr
  session_sweep_scoring.py  — get_kill_zone_score, compute_quality, is_kill_zone,
                               classify_sweep_quality

Data source: yfinance 1-minute candles (max 7 days lookback, ~10 min delay).
Supported symbols: ES=F, NQ=F, MES=F, MNQ=F.

Graceful degradation: returns empty list when yfinance unavailable.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from zoneinfo import ZoneInfo

# Re-export models for backward compatibility
from mae_core.market.edge.session_sweep_models import (  # noqa: F401
    SessionLevel, FairValueGap, SessionSweepSignal,
)
from mae_core.market.edge.session_sweep_geometry import (
    mark_session_levels, extract_session_levels, find_fvg, find_ifvg,
    score_displacement, compute_atr, SESSION_WINDOWS,
)
from mae_core.market.edge.session_sweep_scoring import (
    get_kill_zone_score, compute_quality, is_kill_zone, classify_sweep_quality,
    KILL_ZONES,
)

logger = logging.getLogger(__name__)

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

DEFAULT_SYMBOLS = ["ES=F", "NQ=F"]


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

        Full pipeline: fetch candles -> mark sessions -> detect sweeps ->
        find FVGs -> build signals. Returns empty list on any failure.
        """
        if not YFINANCE_AVAILABLE or not PANDAS_AVAILABLE:
            logger.debug("Session sweep: yfinance/pandas unavailable")
            return []

        try:
            ohlc = self._fetch_1m_candles(symbol)
            if ohlc is None or ohlc.empty:
                return []

            levels = mark_session_levels(ohlc, self._min_fvg_pct)
            if not levels:
                return []

            signals = []
            for level in levels:
                for side in ("high", "low"):
                    result = self._detect_sweep(ohlc, level, side)
                    if result is None:
                        continue

                    sweep_idx, sweep_candle = result

                    if side == "high":
                        direction = "bearish"
                        fvg_direction = "bearish"
                    else:
                        direction = "bullish"
                        fvg_direction = "bullish"

                    fvg = find_fvg(ohlc, sweep_idx, fvg_direction, self._min_fvg_pct)
                    if fvg is None:
                        continue

                    is_ifvg = find_ifvg(ohlc, sweep_idx, direction)
                    disp = score_displacement(ohlc, sweep_idx, direction)
                    atr = compute_atr(ohlc, sweep_idx)
                    fvg_atr = (fvg.top - fvg.bottom) / atr if atr > 0 else 0.0
                    kz_score = get_kill_zone_score(ohlc.index[sweep_idx])
                    quality = compute_quality(disp, fvg_atr, kz_score)

                    signal = self._build_signal(
                        symbol, level, side, direction,
                        sweep_candle, fvg, ohlc, sweep_idx,
                    )
                    signal.is_ifvg = is_ifvg
                    signal.displacement_score = round(disp, 3)
                    signal.fvg_atr_ratio = round(fvg_atr, 3)
                    signal.quality_score = round(quality, 3)

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

            df.columns = [c.lower() for c in df.columns]

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert(EASTERN)
            df = df.dropna(subset=["open", "high", "low", "close"])
            return df

        except Exception:
            logger.debug("Failed to fetch 1m candles for %s", symbol, exc_info=True)
            return None

    # ── Session level marking (delegates to geometry module) ───────

    def _mark_session_levels(self, ohlc) -> List[SessionLevel]:
        return mark_session_levels(ohlc, self._min_fvg_pct)

    def _extract_session_levels(self, ohlc, session_name, window, now_et) -> List[SessionLevel]:
        return extract_session_levels(ohlc, session_name, window, now_et)

    # ── Sweep detection ────────────────────────────────────────────

    def _detect_sweep(
        self,
        ohlc: "pd.DataFrame",
        level: SessionLevel,
        side: str,
    ) -> Optional[Tuple[int, "pd.Series"]]:
        """Detect if a session level was swept (breached then reversed)."""
        target = level.session_high if side == "high" else level.session_low

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

        for pos_idx in range(len(post_session)):
            candle = post_session.iloc[pos_idx]

            if side == "high" and candle["high"] > target:
                if self._confirm_sweep_reversal(post_session, pos_idx, target, "below"):
                    abs_idx = ohlc.index.get_loc(post_session.index[pos_idx])
                    return (abs_idx, candle)

            elif side == "low" and candle["low"] < target:
                if self._confirm_sweep_reversal(post_session, pos_idx, target, "above"):
                    abs_idx = ohlc.index.get_loc(post_session.index[pos_idx])
                    return (abs_idx, candle)

        return None

    def _confirm_sweep_reversal(
        self, df, breach_idx: int, level: float, close_side: str,
    ) -> bool:
        """Check if the sweep candle's close (or next 1-2 candles) reverses."""
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

    # ── FVG + scoring delegates ────────────────────────────────────

    def _find_fvg(self, ohlc, from_idx: int, direction: str) -> Optional[FairValueGap]:
        return find_fvg(ohlc, from_idx, direction, self._min_fvg_pct)

    def _find_ifvg(self, ohlc, sweep_idx: int, direction: str) -> bool:
        return find_ifvg(ohlc, sweep_idx, direction)

    def _score_displacement(self, ohlc, sweep_idx: int, direction: str) -> float:
        return score_displacement(ohlc, sweep_idx, direction)

    def _compute_atr(self, ohlc, idx: int, period: int = 14) -> float:
        return compute_atr(ohlc, idx, period)

    def _get_kill_zone_score(self, dt) -> float:
        return get_kill_zone_score(dt)

    def _compute_quality(self, displacement: float, fvg_atr: float, kz: float) -> float:
        return compute_quality(displacement, fvg_atr, kz)

    def _is_kill_zone(self, dt) -> bool:
        return is_kill_zone(dt)

    # ── Signal construction ────────────────────────────────────────

    def _build_signal(
        self,
        symbol: str,
        level: SessionLevel,
        side: str,
        direction: str,
        sweep_candle,
        fvg: FairValueGap,
        ohlc,
        sweep_idx: int,
    ) -> SessionSweepSignal:
        """Assemble a complete SessionSweepSignal from components."""
        sweep_type = "high_sweep" if side == "high" else "low_sweep"
        sweep_level = level.session_high if side == "high" else level.session_low

        entry_top = fvg.top
        entry_bottom = fvg.bottom

        stop, target = self._calculate_stop_and_target(direction, sweep_candle, fvg)

        entry_price = fvg.midpoint
        risk = abs(entry_price - stop) if abs(entry_price - stop) > 0 else 0.01
        reward = abs(target - entry_price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        sweep_time = ohlc.index[sweep_idx]
        in_kz = is_kill_zone(sweep_time)

        confidence = classify_sweep_quality(in_kz, fvg.size_pct, side, sweep_candle, sweep_level)
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

    def _calculate_stop_and_target(self, direction: str, sweep_candle, fvg: FairValueGap):
        """Calculate stop loss and 2R target from sweep wick and FVG zone."""
        entry = fvg.midpoint

        if direction == "bullish":
            stop = float(sweep_candle["low"]) - 0.50
            risk = entry - stop
            target = entry + (2.0 * risk)
        else:
            stop = float(sweep_candle["high"]) + 0.50
            risk = stop - entry
            target = entry - (2.0 * risk)

        return round(stop, 2), round(target, 2)

    def _classify_sweep_quality(self, kill_zone: bool, fvg_size_pct: float,
                                 side: str, sweep_candle, sweep_level: float) -> float:
        return classify_sweep_quality(kill_zone, fvg_size_pct, side, sweep_candle, sweep_level)

    # ── Deduplication ──────────────────────────────────────────────

    def _deduplicate(self, signals: List[SessionSweepSignal]) -> List[SessionSweepSignal]:
        """Remove duplicate sweep signals. Keeps highest confidence."""
        if len(signals) <= 1:
            return signals

        seen: Dict[str, SessionSweepSignal] = {}
        for sig in signals:
            key = f"{sig.symbol}:{sig.session_swept}:{sig.direction}"
            if key not in seen:
                seen[key] = sig
            elif sig.confidence > seen[key].confidence:
                seen[key] = sig

        return list(seen.values())
