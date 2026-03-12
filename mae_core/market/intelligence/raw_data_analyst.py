"""Raw Data Analyst — cross-domain insight engine reading from SQLite raw stores.

The raw stores are MIDGE's long-term memory: every API call persists before
processing, but almost nothing reads that stored data back. This system is the
first reader — it scans across domains, finds correlations invisible to the
single-pass pipeline, and injects enriched signals into the convergence engine.

Biological analogy: hippocampal replay during slow-wave sleep — the organism
re-processes stored experiences to extract patterns that weren't obvious in the
moment of recording.

Runs every 100 steps via _run_slow_cadence_ops. Returns a list of MarketSignal
objects that the caller feeds directly into convergence_alerter.record_signal().

Analysis routines:
  1. insider_price_context   — insider buying near 52-wk low = amplified signal
  2. fred_macro_regime        — inflation acceleration + yield inversion = warning
  3. cross_domain_preconv     — insider + trends + headlines on same ticker
  4. funding_rate_squeeze     — 3+ consecutive negative Binance funding → squeeze

Split points: orchestrator (this file, <500 lines) stays lean.
Heavy computation lives here since four routines fit easily within the limit.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from mae_core.market.raw_store import RawStore

from mae_core.market.signal import MarketSignal

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_CADENCE = 100  # Run every N steps

# FRED series used for macro regime detection
_FRED_YIELD_SPREAD = "T10Y2Y"          # 10yr - 2yr spread (inversion proxy)
_FRED_T10Y3M = "T10Y3M"               # 10yr - 3mo spread (recession predictor)
_FRED_UNEMPLOYMENT = "UNRATE"          # Unemployment rate
_FRED_FED_FUNDS = "DFF"               # Effective federal funds rate
_FRED_CPI = "CPIAUCSL"               # CPI all items (inflation proxy)

# Price position thresholds for insider context
_LOW_PCT_THRESHOLD = 0.10   # Within 10% of 52-wk low = "near low"
_HIGH_PCT_THRESHOLD = 0.10  # Within 10% of 52-wk high = "near high"

# Minimum trades to signal cluster (same ticker, same lookback)
_INSIDER_CLUSTER_MIN = 2

# Strength amplification for insider buying at 52-wk low
_LOW_AMPLIFIER = 1.35   # 35% strength boost
_HIGH_AMPLIFIER = 0.75  # 25% strength penalty (selling near high is less meaningful)

# Funding rate: N consecutive negative periods = squeeze signal
_NEGATIVE_FUNDING_MIN = 3


class RawDataAnalyst:
    """Cross-domain insight engine — reads raw SQLite stores, produces enriched signals.

    Call analyze(step) from the step hook every 100 steps. Returns a list of
    MarketSignal objects ready for convergence_alerter.record_signal().

    Constructor:
        raw_store: RawStore instance (required). Analyst is useless without it.
        min_insider_trades: Minimum insider trades in window to trigger analysis.
    """

    def __init__(
        self,
        raw_store: "RawStore",
        min_insider_trades: int = 1,
    ):
        self._store = raw_store
        self._min_insider = min_insider_trades
        self._run_count = 0
        self._last_run_at: Optional[datetime] = None
        self._signals_emitted = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def analyze(self, step: int) -> List[MarketSignal]:
        """Run all cross-domain analysis routines and return enriched signals.

        Only executes every _CADENCE steps. Returns empty list on off-steps
        or when raw_store is unavailable.

        Args:
            step: Current simulation step number.

        Returns:
            List of MarketSignal objects to inject into convergence engine.
        """
        if step % _CADENCE != 0:
            return []
        if self._store is None:
            return []

        self._run_count += 1
        self._last_run_at = datetime.now()
        signals: List[MarketSignal] = []

        try:
            signals.extend(self._analyze_insider_price_context())
        except Exception:
            logger.debug("RawDataAnalyst: insider_price_context failed", exc_info=True)

        try:
            signals.extend(self._analyze_fred_macro_regime())
        except Exception:
            logger.debug("RawDataAnalyst: fred_macro_regime failed", exc_info=True)

        try:
            signals.extend(self._analyze_cross_domain_preconvergence())
        except Exception:
            logger.debug("RawDataAnalyst: cross_domain_preconv failed", exc_info=True)

        try:
            signals.extend(self._analyze_funding_rate_squeeze())
        except Exception:
            logger.debug("RawDataAnalyst: funding_rate_squeeze failed", exc_info=True)

        self._signals_emitted += len(signals)
        if signals:
            logger.info(
                "RawDataAnalyst: step %d — emitted %d enriched signals (total %d)",
                step, len(signals), self._signals_emitted,
            )
        return signals

    def get_statistics(self) -> dict:
        """Return system health stats (HolonProxy delegation)."""
        return {
            "run_count": self._run_count,
            "signals_emitted": self._signals_emitted,
            "last_run_at": self._last_run_at.isoformat() if self._last_run_at else None,
        }

    # ── Routine 1: Insider + Price Context ─────────────────────────────────

    def _analyze_insider_price_context(self) -> List[MarketSignal]:
        """Amplify or dampen insider signals based on 52-week price position.

        Logic:
          - Fetch all insider buy trades in past 30 days.
          - For each ticker with >= _INSIDER_CLUSTER_MIN trades, look up the
            price snapshot for 52-week high/low.
          - Insider buying within 10% of 52-wk low → strong bullish conviction
            (management rarely buys at lows unless they expect reversal).
          - Insider buying within 10% of 52-wk high → weaker signal (comfort buying).
          - Emit a MarketSignal with source="insider_context", modified strength.
        """
        signals: List[MarketSignal] = []
        trades = self._store.get_insider_trades(lookback_days=30)
        if not trades:
            return signals

        # Group by ticker
        by_ticker: dict = defaultdict(list)
        for t in trades:
            ticker = (t.get("ticker") or "").upper()
            if ticker:
                by_ticker[ticker].append(t)

        for ticker, ticker_trades in by_ticker.items():
            if len(ticker_trades) < self._min_insider:
                continue

            # Get most recent price snapshot
            snapshots = self._store.get_price_snapshots(ticker=ticker, lookback_days=7)
            if not snapshots:
                continue

            snap = snapshots[0]  # newest-first — most recent
            price = snap.get("price") or 0.0
            wk52_low = snap.get("fifty_two_week_low") or 0.0
            wk52_high = snap.get("fifty_two_week_high") or 0.0

            if not price or not wk52_low or not wk52_high or wk52_high <= wk52_low:
                continue

            # Compute price position within 52-week range [0=low, 1=high]
            price_range = wk52_high - wk52_low
            position = (price - wk52_low) / price_range  # 0.0..1.0

            # Base strength: proportional to trade count (capped at 1.0)
            base_strength = min(1.0, 0.4 + 0.1 * len(ticker_trades))
            total_value = sum(abs(t.get("total_value") or 0.0) for t in ticker_trades)

            # Classify and amplify
            if position <= _LOW_PCT_THRESHOLD:
                # Near 52-wk low — strongest signal
                strength = min(1.0, base_strength * _LOW_AMPLIFIER)
                direction = "bullish"
                context = "near_52wk_low"
            elif position >= (1.0 - _HIGH_PCT_THRESHOLD):
                # Near 52-wk high — buying at peak, weaker signal
                strength = base_strength * _HIGH_AMPLIFIER
                direction = "bullish"
                context = "near_52wk_high"
            else:
                # Mid-range — standard signal, no amplification needed
                strength = base_strength
                direction = "bullish"
                context = "mid_range"

            signals.append(self._build_signal(
                source="insider_context",
                symbol=ticker,
                domain="insider",
                direction=direction,
                strength=strength,
                confidence=0.72,
                metadata={
                    "insider_count": len(ticker_trades),
                    "total_value": total_value,
                    "price_position_pct": round(position * 100, 1),
                    "price_context": context,
                    "wk52_low": wk52_low,
                    "wk52_high": wk52_high,
                    "current_price": price,
                },
            ))

        return signals

    # ── Routine 2: FRED Macro Regime Detection ──────────────────────────────

    def _analyze_fred_macro_regime(self) -> List[MarketSignal]:
        """Detect macro warning patterns from FRED data series.

        Patterns detected:
          A. Yield curve inversion + accelerating inflation → macro_warning (bearish)
          B. Rising fed funds rate + rising unemployment → recession_risk (bearish)
          C. Yield curve steepening from inverted state → recovery signal (bullish)

        Returns macro-level signals with symbol="" (no specific ticker).
        """
        signals: List[MarketSignal] = []
        obs = self._store.get_fred_observations(lookback_days=90)
        if not obs:
            return signals

        # Group observations by series into {series_id: [(date, value), ...]}
        series: dict = defaultdict(list)
        for o in obs:
            sid = o.get("series_id", "")
            date = o.get("date", "")
            val = o.get("value")
            if sid and date and val is not None:
                series[sid].append((date, float(val)))

        # Sort each series oldest-first (already returned that way, but be safe)
        for sid in series:
            series[sid].sort(key=lambda x: x[0])

        # Helper: compute month-over-month acceleration (last 3 vs prior 3)
        def _mom_acceleration(vals: list) -> float:
            """Positive = accelerating, negative = decelerating."""
            if len(vals) < 6:
                return 0.0
            recent_change = vals[-1][1] - vals[-3][1]
            prior_change = vals[-3][1] - vals[-6][1] if len(vals) >= 6 else 0.0
            return recent_change - prior_change

        # Helper: latest value from series
        def _latest(sid: str) -> Optional[float]:
            if sid in series and series[sid]:
                return series[sid][-1][1]
            return None

        # --- Pattern A: Yield inversion + inflation acceleration ---
        spread = _latest(_FRED_YIELD_SPREAD)   # T10Y2Y
        cpi_accel = _mom_acceleration(series.get(_FRED_CPI, []))

        if spread is not None and spread < 0 and cpi_accel > 0.05:
            signals.append(self._build_signal(
                source="fred_macro",
                symbol="",
                domain="macro",
                direction="bearish",
                strength=min(1.0, 0.5 + abs(spread) * 0.15 + cpi_accel * 2),
                confidence=0.85,
                metadata={
                    "pattern": "inversion_inflation",
                    "yield_spread_t10y2y": round(spread, 3),
                    "cpi_acceleration": round(cpi_accel, 4),
                    "description": "Yield curve inverted + CPI accelerating",
                },
            ))

        # --- Pattern B: High fed funds + rising unemployment ---
        fed_rate = _latest(_FRED_FED_FUNDS)
        unemp_accel = _mom_acceleration(series.get(_FRED_UNEMPLOYMENT, []))

        if fed_rate is not None and fed_rate > 4.0 and unemp_accel > 0.1:
            signals.append(self._build_signal(
                source="fred_macro",
                symbol="",
                domain="macro",
                direction="bearish",
                strength=min(1.0, 0.45 + unemp_accel * 1.5),
                confidence=0.80,
                metadata={
                    "pattern": "recession_risk",
                    "fed_funds_rate": round(fed_rate, 2),
                    "unemployment_acceleration": round(unemp_accel, 3),
                    "description": "High rates + rising unemployment = recession risk",
                },
            ))

        # --- Pattern C: Yield curve steepening from deep inversion ---
        t10y3m = _latest(_FRED_T10Y3M)
        t10y3m_series = series.get(_FRED_T10Y3M, [])
        if (t10y3m is not None and len(t10y3m_series) >= 10
                and t10y3m > -0.25                     # Currently less inverted
                and t10y3m_series[-10][1] < -0.5):     # Was deeply inverted 10 periods ago
            signals.append(self._build_signal(
                source="fred_macro",
                symbol="",
                domain="macro",
                direction="bullish",
                strength=0.55,
                confidence=0.70,
                metadata={
                    "pattern": "yield_steepening",
                    "t10y3m_now": round(t10y3m, 3),
                    "t10y3m_prior": round(t10y3m_series[-10][1], 3),
                    "description": "Yield curve steepening from inverted — historically bullish",
                },
            ))

        return signals

    # ── Routine 3: Cross-Domain Pre-Convergence ─────────────────────────────

    def _analyze_cross_domain_preconvergence(self) -> List[MarketSignal]:
        """Detect tickers with multi-domain evidence not yet triggering convergence.

        Checks three independent domains for the same ticker:
          - Insider buying (past 14 days)
          - Rising Google Trends interest (past 7 days)
          - Positive Yahoo RSS headline count (past 3 days)

        When 2+ domains align → emit a "pre_convergence" signal as head-start
        data for the convergence engine to complete the triad from live sensing.
        """
        signals: List[MarketSignal] = []

        # Gather insider tickers (past 14 days)
        insider_tickers: set = set()
        for t in self._store.get_insider_trades(lookback_days=14):
            tkr = (t.get("ticker") or "").upper()
            if tkr:
                insider_tickers.add(tkr)

        if not insider_tickers:
            return signals

        # Gather Trends data — group by keyword to detect rising interest
        trends_rows = self._store.get_trends_history(lookback_days=7)
        trending_keywords: set = set()
        kw_interest: dict = defaultdict(list)
        for row in trends_rows:
            kw_interest[row["keyword"]].append(row["interest"])
        for kw, vals in kw_interest.items():
            if len(vals) >= 3:
                # Rising if last 3 >= first 3 average + 10 points
                recent = sum(vals[-3:]) / 3
                early = sum(vals[:3]) / 3
                if recent > early + 10:
                    trending_keywords.add(kw.upper())

        # Gather Yahoo headlines — count positive per ticker
        headline_tickers: set = set()
        for row in self._store.get_yahoo_headlines(lookback_days=3):
            tkr = (row.get("ticker") or "").upper()
            title = (row.get("title") or "").lower()
            # Simple positive keyword filter
            if tkr and any(w in title for w in ("beat", "surge", "record", "win",
                                                  "rise", "jump", "soar", "growth")):
                headline_tickers.add(tkr)

        # Find tickers with insider evidence PLUS at least one other domain
        for ticker in insider_tickers:
            has_trends = ticker in trending_keywords
            has_headlines = ticker in headline_tickers
            domains_hit = 1 + int(has_trends) + int(has_headlines)

            if domains_hit < 2:
                continue

            strength = 0.35 + 0.15 * domains_hit  # 0.65 if all 3
            signals.append(self._build_signal(
                source="raw_preconvergence",
                symbol=ticker,
                domain="insider",  # lead domain
                direction="bullish",
                strength=min(1.0, strength),
                confidence=0.55,
                metadata={
                    "domains_hit": domains_hit,
                    "has_insider": True,
                    "has_trends": has_trends,
                    "has_headlines": has_headlines,
                    "note": "Pre-convergence: multi-domain alignment detected before full alert",
                },
            ))

        return signals

    # ── Routine 4: Funding Rate Squeeze Detection ───────────────────────────

    def _analyze_funding_rate_squeeze(self) -> List[MarketSignal]:
        """Detect impending short squeezes from Binance perpetual funding rates.

        Negative funding = shorts pay longs = crowd is net short.
        3+ consecutive negative 8-hour periods → shorts are overcrowded.
        Historical pattern: overcrowded shorts precede sudden violent squeezes.

        Only emits signals if Binance funding data exists in raw_store.
        """
        signals: List[MarketSignal] = []

        # Common perpetual pairs to check
        pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

        for symbol in pairs:
            history = self._store.get_binance_funding_history(symbol, lookback_days=7)
            if len(history) < _NEGATIVE_FUNDING_MIN:
                continue

            # Check last N consecutive periods
            recent = history[-_NEGATIVE_FUNDING_MIN:]
            rates = [r["funding_rate"] for r in recent if r.get("funding_rate") is not None]
            if len(rates) < _NEGATIVE_FUNDING_MIN:
                continue

            all_negative = all(r < 0 for r in rates)
            if not all_negative:
                continue

            # Strength proportional to magnitude of negative funding
            avg_neg = abs(sum(rates) / len(rates))
            strength = min(1.0, 0.5 + avg_neg * 100)  # 0.01 rate = +1.0 strength

            # Consecutive negative count (might be more than minimum)
            consecutive = 0
            for r in reversed(history):
                if (r.get("funding_rate") or 0) < 0:
                    consecutive += 1
                else:
                    break

            # Convert crypto symbol to approximate ticker for convergence
            crypto_symbol = symbol.replace("USDT", "")

            signals.append(self._build_signal(
                source="binance_funding",
                symbol=crypto_symbol + "-USD",
                domain="crypto",
                direction="bullish",  # squeeze = forced buy = price up
                strength=strength,
                confidence=0.65,
                metadata={
                    "binance_symbol": symbol,
                    "consecutive_negative_periods": consecutive,
                    "avg_funding_rate": round(-avg_neg, 6),
                    "pattern": "short_squeeze_precursor",
                    "note": f"{consecutive} consecutive negative funding periods",
                },
            ))

        return signals

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_signal(
        self,
        source: str,
        symbol: str,
        domain: str,
        direction: str,
        strength: float,
        confidence: float,
        metadata: dict = None,
    ) -> MarketSignal:
        """Construct a MarketSignal from enriched analysis output."""
        now = datetime.now()
        return MarketSignal(
            signal_id=f"{source}:{symbol}:{uuid.uuid4().hex[:8]}",
            source=source,
            symbol=symbol,
            asset_class=self._infer_asset_class(symbol),
            domain=domain,
            direction=direction,
            strength=max(0.0, min(1.0, strength)),
            confidence=max(0.0, min(1.0, confidence)),
            decay_rate=0.15,
            timestamp=now,
            received_at=now,
            outcome_symbol=symbol,
            outcome_window_days=14,
            raw_id="raw_analyst",
            raw_type="cross_domain_analysis",
            metadata=metadata or {},
        )

    @staticmethod
    def _infer_asset_class(symbol: str) -> str:
        """Infer asset class from ticker format."""
        if not symbol:
            return "macro"
        sym_upper = symbol.upper()
        if sym_upper.endswith("-USD") or sym_upper in ("BTC", "ETH", "SOL", "BNB", "XRP"):
            return "crypto"
        if sym_upper.endswith("=F") or sym_upper in ("GC", "CL", "NQ", "ES"):
            return "futures"
        if "USD" in sym_upper and "=" in sym_upper:
            return "forex"
        return "stock"
