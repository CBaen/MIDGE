"""Historical Data Fetcher — synthesize retroactive signals for pattern excavation.

For each dig site (a known historical price move), this module reaches back
through every available data source to reconstruct what signals existed BEFORE
the move happened. The output is a list of signal dicts in archive format —
identical to what live monitoring produces — so the Excavator treats
historical and live signals uniformly.

Data source tiers:
  Tier 1 (fully retroactive — compute from raw data):
    - TA indicators: RSI, MACD, Bollinger, market structure, candles from price data
    - Price/volume: gap analysis, unusual volume from OHLCV history

  Tier 2 (API-fetchable history):
    - SEC EDGAR Form 4: insider trades (years of history per company)
    - SEC EDGAR Form 8-K: material events
    - FRED: macro indicators (decades of history)
    - COT: CFTC positioning data (years by year)
    - Congressional: STOCK Act disclosures (full history via free bulk download)

  Tier 3 (archive-only — whatever was captured during live monitoring):
    - Social sentiment, Google Trends, StockTwits, Reddit, job tracker, etc.
    - The signal archive in data/midge/signals/ covers these
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SIGNAL_ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"

# Domains that apply to all symbols (no symbol filter needed)
SYMBOL_AGNOSTIC_DOMAINS = {"macro", "events", "positioning", "volatility"}


class HistoricalDataFetcher:
    """Synthesize retroactive signals for a symbol + date window.

    Combines three data layers:
    1. TA indicators computed from price history (zero API calls)
    2. Active API fetches from historical-capable sources
    3. Signal archive for anything captured during live monitoring
    """

    def __init__(
        self,
        signal_dir: Optional[Path] = None,
        sec_client: Any = None,
        fred_client: Any = None,
        cot_client: Any = None,
        congress_client: Any = None,
        senate_client: Any = None,
    ):
        self._signal_dir = signal_dir or SIGNAL_ARCHIVE_DIR
        self._sec_client = sec_client
        self._fred_client = fred_client
        self._cot_client = cot_client
        self._congress_client = congress_client
        self._senate_client = senate_client
        self._signal_cache: dict[str, list[dict]] = {}
        self._archive_preloaded = False
        # Per-symbol TA cache: symbol -> list of all TA signals (full history)
        self._ta_cache: dict[str, list[dict]] = {}
        # Bulk data caches (loaded once, reused across dig sites)
        self._fred_cache: dict[str, list] = {}
        self._cot_cache: dict[int, list] = {}  # year -> positions
        self._congress_cache: Optional[list] = None

    def fetch_all(
        self,
        symbol: str,
        price_history: list,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch all available historical signals for a symbol + date window.

        Args:
            symbol: Ticker symbol.
            price_history: Full PriceData list (used for TA computation).
            start_date: Beginning of lookback window.
            end_date: End of lookback window (the move date, exclusive).

        Returns:
            List of signal dicts in archive format, suitable for Excavator.
        """
        signals: list[dict] = []

        # Tier 1: TA indicators from price data (always available, zero API calls)
        signals.extend(self._compute_ta_signals(symbol, price_history, start_date, end_date))

        # Tier 2: API-fetchable history
        signals.extend(self._fetch_sec_signals(symbol, start_date, end_date))
        signals.extend(self._fetch_fred_signals(start_date, end_date))
        signals.extend(self._fetch_cot_signals(symbol, start_date, end_date))
        signals.extend(self._fetch_congressional_signals(symbol, start_date, end_date))

        # Tier 3: Signal archive (whatever was captured live)
        signals.extend(self._load_archive_signals(symbol, start_date, end_date))

        # Deduplicate by signal_id
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for sig in signals:
            sid = sig.get("signal_id", "")
            if sid and sid in seen_ids:
                continue
            if sid:
                seen_ids.add(sid)
            deduped.append(sig)

        return deduped

    # ── Tier 1: TA from price data ────────────────────────────────────────

    def _compute_ta_signals(
        self,
        symbol: str,
        price_history: list,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Compute TA indicators retroactively from price history.

        For each day in [start_date, end_date), check if significant
        TA conditions existed. Generate signal records for:
        - RSI extreme (oversold < 30, overbought > 70)
        - MACD crossover
        - Bollinger squeeze / breakout
        - Volume anomaly (> 2x 20-day average)
        """
        if len(price_history) < 35:
            return []

        signals: list[dict] = []

        # Build date-indexed price map
        prices_by_date: dict[str, int] = {}
        for idx, p in enumerate(price_history):
            ts = getattr(p, "timestamp", "")[:10]
            if ts:
                prices_by_date[ts] = idx

        # RSI series
        rsi_signals = self._compute_rsi_series(symbol, price_history, prices_by_date, start_date, end_date)
        signals.extend(rsi_signals)

        # MACD crossover detection
        macd_signals = self._compute_macd_series(symbol, price_history, prices_by_date, start_date, end_date)
        signals.extend(macd_signals)

        # Bollinger squeeze/breakout
        bb_signals = self._compute_bollinger_series(symbol, price_history, prices_by_date, start_date, end_date)
        signals.extend(bb_signals)

        # Volume anomaly
        vol_signals = self._compute_volume_signals(symbol, price_history, prices_by_date, start_date, end_date)
        signals.extend(vol_signals)

        return signals

    def _compute_rsi_series(
        self, symbol: str, history: list, date_idx: dict, start: date, end: date,
    ) -> list[dict]:
        """Compute RSI for all dates and flag extremes in the window."""
        closes = [p.price for p in history]
        if len(closes) < 15:
            return []

        period = 14
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [max(d, 0) for d in deltas]
        losses_arr = [abs(min(d, 0)) for d in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses_arr[:period]) / period

        rsi_values: list[Optional[float]] = [None] * (period + 1)
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(round(100 - (100 / (1 + rs)), 2))

        for i in range(period + 1, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses_arr[i]) / period
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(round(100 - (100 / (1 + rs)), 2))

        signals: list[dict] = []
        for idx in range(len(rsi_values)):
            rsi = rsi_values[idx]
            if rsi is None:
                continue
            if idx >= len(history):
                continue
            ts = getattr(history[idx], "timestamp", "")[:10]
            if not ts:
                continue
            try:
                sig_date = date.fromisoformat(ts)
            except ValueError:
                continue
            if sig_date < start or sig_date >= end:
                continue

            if rsi < 30:
                signals.append(self._make_signal(
                    f"hist:ta_rsi:oversold:{symbol}:{ts}",
                    "ta_rsi", symbol, "technical", "bullish",
                    round(min(1.0, (30 - rsi) / 30), 2), ts,
                    {"rsi_value": rsi, "condition": "oversold"},
                ))
            elif rsi > 70:
                signals.append(self._make_signal(
                    f"hist:ta_rsi:overbought:{symbol}:{ts}",
                    "ta_rsi", symbol, "technical", "bearish",
                    round(min(1.0, (rsi - 70) / 30), 2), ts,
                    {"rsi_value": rsi, "condition": "overbought"},
                ))

        return signals

    def _compute_macd_series(
        self, symbol: str, history: list, date_idx: dict, start: date, end: date,
    ) -> list[dict]:
        """Detect MACD crossovers in the window."""
        closes = [p.price for p in history]
        if len(closes) < 35:
            return []

        fast, slow, sig_period = 12, 26, 9

        def ema(data: list[float], period: int) -> list[float]:
            result = [data[0]]
            mult = 2 / (period + 1)
            for val in data[1:]:
                result.append(val * mult + result[-1] * (1 - mult))
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = ema(macd_line[slow - 1:], sig_period)

        # Align signal_line with macd_line
        offset = slow - 1
        signals: list[dict] = []

        for i in range(1, len(signal_line)):
            abs_idx = offset + i
            if abs_idx >= len(history):
                continue

            ts = getattr(history[abs_idx], "timestamp", "")[:10]
            if not ts:
                continue
            try:
                sig_date = date.fromisoformat(ts)
            except ValueError:
                continue
            if sig_date < start or sig_date >= end:
                continue

            prev_macd = macd_line[offset + i - 1]
            curr_macd = macd_line[offset + i]
            prev_sig = signal_line[i - 1]
            curr_sig = signal_line[i]

            # Bullish crossover: MACD crosses above signal
            if prev_macd <= prev_sig and curr_macd > curr_sig:
                signals.append(self._make_signal(
                    f"hist:ta_macd:crossover_bull:{symbol}:{ts}",
                    "ta_macd", symbol, "technical", "bullish",
                    round(min(1.0, abs(curr_macd - curr_sig) * 10), 2), ts,
                    {"crossover": "bullish", "macd": round(curr_macd, 4), "signal": round(curr_sig, 4)},
                ))
            # Bearish crossover: MACD crosses below signal
            elif prev_macd >= prev_sig and curr_macd < curr_sig:
                signals.append(self._make_signal(
                    f"hist:ta_macd:crossover_bear:{symbol}:{ts}",
                    "ta_macd", symbol, "technical", "bearish",
                    round(min(1.0, abs(curr_macd - curr_sig) * 10), 2), ts,
                    {"crossover": "bearish", "macd": round(curr_macd, 4), "signal": round(curr_sig, 4)},
                ))

        return signals

    def _compute_bollinger_series(
        self, symbol: str, history: list, date_idx: dict, start: date, end: date,
    ) -> list[dict]:
        """Detect Bollinger Band squeezes and breakouts in the window."""
        closes = [p.price for p in history]
        period = 20
        if len(closes) < period + 1:
            return []

        signals: list[dict] = []

        for i in range(period, len(closes)):
            if i >= len(history):
                break
            ts = getattr(history[i], "timestamp", "")[:10]
            if not ts:
                continue
            try:
                sig_date = date.fromisoformat(ts)
            except ValueError:
                continue
            if sig_date < start or sig_date >= end:
                continue

            window = closes[i - period:i]
            sma = sum(window) / period
            variance = sum((x - sma) ** 2 for x in window) / period
            std = math.sqrt(variance) if variance > 0 else 0.001
            upper = sma + 2 * std
            lower = sma - 2 * std
            bandwidth = (upper - lower) / sma if sma > 0 else 0

            price = closes[i]

            # Breakout above upper band
            if price > upper:
                signals.append(self._make_signal(
                    f"hist:ta_bollinger:upper_break:{symbol}:{ts}",
                    "ta_bollinger", symbol, "technical", "bullish",
                    round(min(1.0, (price - upper) / (upper - sma) if upper != sma else 0.5), 2), ts,
                    {"condition": "upper_breakout", "bandwidth": round(bandwidth, 4)},
                ))
            # Breakout below lower band
            elif price < lower:
                signals.append(self._make_signal(
                    f"hist:ta_bollinger:lower_break:{symbol}:{ts}",
                    "ta_bollinger", symbol, "technical", "bearish",
                    round(min(1.0, (lower - price) / (sma - lower) if sma != lower else 0.5), 2), ts,
                    {"condition": "lower_breakout", "bandwidth": round(bandwidth, 4)},
                ))
            # Squeeze (bandwidth < 0.04 = tight range)
            elif bandwidth < 0.04 and i > period + 10:
                # Check if this is a new squeeze (wasn't squeezing before)
                prev_window = closes[i - period - 1:i - 1]
                prev_sma = sum(prev_window) / period
                prev_var = sum((x - prev_sma) ** 2 for x in prev_window) / period
                prev_std = math.sqrt(prev_var) if prev_var > 0 else 0.001
                prev_bw = (2 * prev_std * 2) / prev_sma if prev_sma > 0 else 0
                if prev_bw >= 0.04:
                    signals.append(self._make_signal(
                        f"hist:ta_bollinger:squeeze:{symbol}:{ts}",
                        "ta_bollinger", symbol, "technical", "neutral",
                        0.5, ts,
                        {"condition": "squeeze_entry", "bandwidth": round(bandwidth, 4)},
                    ))

        return signals

    def _compute_volume_signals(
        self, symbol: str, history: list, date_idx: dict, start: date, end: date,
    ) -> list[dict]:
        """Detect unusual volume (> 2x 20-day average) in the window."""
        if len(history) < 21:
            return []

        signals: list[dict] = []
        for i in range(20, len(history)):
            ts = getattr(history[i], "timestamp", "")[:10]
            if not ts:
                continue
            try:
                sig_date = date.fromisoformat(ts)
            except ValueError:
                continue
            if sig_date < start or sig_date >= end:
                continue

            vol = getattr(history[i], "volume", 0) or 0
            avg_vol = sum(getattr(history[j], "volume", 0) or 0 for j in range(i - 20, i)) / 20
            if avg_vol <= 0 or vol <= 0:
                continue

            ratio = vol / avg_vol
            if ratio >= 2.0:
                # Direction from price change
                prev_price = history[i - 1].price
                curr_price = history[i].price
                direction = "bullish" if curr_price > prev_price else "bearish"
                signals.append(self._make_signal(
                    f"hist:volume_anomaly:{symbol}:{ts}",
                    "finviz_unusual_volume", symbol, "technical", direction,
                    round(min(1.0, (ratio - 2) / 3), 2), ts,
                    {"volume_ratio": round(ratio, 2), "volume": vol, "avg_volume": round(avg_vol, 0)},
                ))

        return signals

    # ── Tier 2: API-fetchable history ─────────────────────────────────────

    def _fetch_sec_signals(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        """Fetch SEC Form 4 insider trades for the lookback window."""
        if self._sec_client is None:
            return []

        signals: list[dict] = []
        try:
            # Fetch with large enough window from today to cover the dig site
            days_from_now = (date.today() - start_date).days + 30
            trades = self._sec_client.get_recent_form4s(symbol, days=days_from_now)

            for trade in trades:
                trade_date_str = getattr(trade, "transaction_date", "") or getattr(trade, "filing_date", "")
                if not trade_date_str:
                    continue
                try:
                    trade_date = date.fromisoformat(trade_date_str[:10])
                except ValueError:
                    continue
                if trade_date < start_date or trade_date >= end_date:
                    continue

                is_purchase = getattr(trade, "transaction_type", "").upper() in ("P", "PURCHASE", "P-Purchase")
                direction = "bullish" if is_purchase else "bearish"
                total_value = abs(getattr(trade, "total_value", 0) or 0)
                strength = min(1.0, total_value / 1_000_000) if total_value > 0 else 0.3

                signals.append(self._make_signal(
                    f"hist:sec_form4:{symbol}:{trade_date_str}:{getattr(trade, 'accession_number', '')}",
                    "sec_form4", symbol, "insider", direction,
                    round(strength, 2), trade_date_str[:10],
                    {"transaction_type": getattr(trade, "transaction_type", ""),
                     "total_value": total_value,
                     "owner_name": getattr(trade, "owner_name", "")},
                ))
        except Exception:
            logger.debug("SEC Form 4 historical fetch failed for %s", symbol, exc_info=True)

        return signals

    def _fetch_fred_signals(self, start_date: date, end_date: date) -> list[dict]:
        """Fetch FRED macro indicators for the lookback window."""
        if self._fred_client is None:
            return []

        signals: list[dict] = []
        series_configs = [
            ("T10Y2Y", "Yield curve (10Y-2Y)"),
            ("DFF", "Fed funds rate"),
            ("VIXCLS", "VIX"),
        ]

        for series_id, name in series_configs:
            try:
                # Check cache first
                if series_id not in self._fred_cache:
                    days_from_now = (date.today() - start_date).days + 30
                    data = self._fred_client.get_historical_series(series_id, days=days_from_now)
                    self._fred_cache[series_id] = data

                for indicator in self._fred_cache[series_id]:
                    ind_date_str = getattr(indicator, "date", "")
                    if not ind_date_str:
                        continue
                    try:
                        ind_date = date.fromisoformat(ind_date_str[:10])
                    except ValueError:
                        continue
                    if ind_date < start_date or ind_date >= end_date:
                        continue

                    direction = getattr(indicator, "direction", "neutral")
                    if direction == "neutral":
                        continue

                    signals.append(self._make_signal(
                        f"hist:fred_macro:{series_id}:{ind_date_str}",
                        "fred_macro", "", "macro", direction,
                        0.5, ind_date_str[:10],
                        {"series_id": series_id, "series_name": name,
                         "value": getattr(indicator, "value", 0)},
                    ))
            except Exception:
                logger.debug("FRED historical fetch failed for %s", series_id, exc_info=True)

        return signals

    def _fetch_cot_signals(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        """Fetch CFTC Commitment of Traders positioning data."""
        if self._cot_client is None:
            return []

        signals: list[dict] = []
        years_needed = list(set([start_date.year, end_date.year]))

        try:
            for year in years_needed:
                if year not in self._cot_cache:
                    positions = self._cot_client.get_all_positions(years=[year])
                    self._cot_cache[year] = positions

            for year in years_needed:
                for pos in self._cot_cache.get(year, []):
                    # COT tickers might not match stock tickers directly
                    # COT is futures-focused, applies as macro context
                    report_date_str = getattr(pos, "report_date", "")
                    if not report_date_str:
                        continue
                    try:
                        report_date = date.fromisoformat(report_date_str[:10])
                    except ValueError:
                        continue
                    if report_date < start_date or report_date >= end_date:
                        continue

                    # COT positioning direction from commercial net position trend
                    commercial_net = getattr(pos, "commercial_net", 0) or 0
                    direction = "bullish" if commercial_net > 0 else "bearish" if commercial_net < 0 else "neutral"
                    if direction == "neutral":
                        continue

                    ticker = getattr(pos, "ticker", "")
                    signals.append(self._make_signal(
                        f"hist:cot_positioning:{ticker}:{report_date_str}",
                        "cot_positioning", "", "positioning", direction,
                        0.5, report_date_str[:10],
                        {"ticker": ticker, "commercial_net": commercial_net,
                         "open_interest": getattr(pos, "open_interest", 0)},
                    ))
        except Exception:
            logger.debug("COT historical fetch failed", exc_info=True)

        return signals

    def _fetch_congressional_signals(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        """Fetch congressional stock trades for the lookback window."""
        signals: list[dict] = []

        for client_name, client in [
            ("house", self._congress_client),
            ("senate", self._senate_client),
        ]:
            if client is None:
                continue
            try:
                if self._congress_cache is None:
                    # Bulk download all trades (the free path gets everything)
                    days_from_now = (date.today() - start_date).days + 60
                    self._congress_cache = client.search_by_ticker(symbol, days=days_from_now)

                for trade in self._congress_cache:
                    trade_date_str = getattr(trade, "transaction_date", "")
                    if not trade_date_str:
                        continue
                    try:
                        trade_date = date.fromisoformat(trade_date_str[:10])
                    except ValueError:
                        continue
                    if trade_date < start_date or trade_date >= end_date:
                        continue

                    tx_type = getattr(trade, "transaction_type", "").lower()
                    direction = "bullish" if "purchase" in tx_type else "bearish" if "sale" in tx_type else "neutral"
                    if direction == "neutral":
                        continue

                    signals.append(self._make_signal(
                        f"hist:congressional:{client_name}:{symbol}:{trade_date_str}:{getattr(trade, 'representative', '')}",
                        "congressional", symbol, "government", direction,
                        0.5, trade_date_str[:10],
                        {"representative": getattr(trade, "representative", ""),
                         "party": getattr(trade, "party", ""),
                         "chamber": client_name},
                    ))
            except Exception:
                logger.debug("Congressional historical fetch failed for %s", client_name, exc_info=True)

        return signals

    # ── Tier 3: Signal archive ────────────────────────────────────────────

    def _load_archive_signals(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        """Load signals from the archive for the date window."""
        signals: list[dict] = []
        current = start_date
        while current < end_date:
            date_str = current.isoformat()
            for sig in self._load_signals_for_date(date_str):
                sig_symbol = sig.get("symbol", "")
                sig_domain = sig.get("domain", "")
                # Match symbol OR symbol-agnostic domains
                if sig_symbol == symbol or sig_domain in SYMBOL_AGNOSTIC_DOMAINS:
                    # Look-ahead bias guard
                    received = sig.get("received_at") or sig.get("timestamp", "")
                    if received:
                        try:
                            received_date = date.fromisoformat(received[:10])
                            if received_date >= end_date:
                                continue
                        except ValueError:
                            pass
                    signals.append(sig)
            current += timedelta(days=1)

        return signals

    def _load_signals_for_date(self, date_str: str) -> list[dict]:
        """Load all signals from archive for a given date (cached)."""
        if date_str in self._signal_cache:
            return self._signal_cache[date_str]

        file_path = self._signal_dir / f"{date_str}.jsonl"
        signals: list[dict] = []
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                signals.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                pass

        self._signal_cache[date_str] = signals
        return signals

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_signal(
        signal_id: str, source: str, symbol: str, domain: str,
        direction: str, strength: float, date_str: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create a signal dict in archive format."""
        return {
            "signal_id": signal_id,
            "source": source,
            "symbol": symbol,
            "domain": domain,
            "direction": direction,
            "strength": strength,
            "confidence": 0.5,
            "velocity": 0.0,
            "timestamp": f"{date_str}T00:00:00",
            "received_at": f"{date_str}T00:00:00",
            "metadata": metadata or {},
        }

    def clear_cache(self) -> None:
        """Clear all caches to free memory between symbol batches."""
        self._signal_cache.clear()

    def clear_all_caches(self) -> None:
        """Clear all caches including bulk data (between full runs)."""
        self._signal_cache.clear()
        self._fred_cache.clear()
        self._cot_cache.clear()
        self._congress_cache = None
