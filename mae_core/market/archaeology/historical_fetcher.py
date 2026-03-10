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
    Sub-module: historical_ta.py

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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mae_core.market.archaeology.historical_ta import (
    compute_ta_signals,
    compute_rsi_series,
    compute_macd_series,
    compute_bollinger_series,
    compute_volume_signals,
    _make_signal,
)

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

    def preload_archive(self) -> int:
        """Load all signal archive files into memory at startup.

        Eliminates per-dig-site file I/O. Call once before bulk excavation.

        Returns:
            Number of archive files loaded.
        """
        if self._archive_preloaded:
            return len(self._signal_cache)

        count = 0
        signal_dir = self._signal_dir
        if not signal_dir.exists():
            self._archive_preloaded = True
            return 0

        for f in sorted(signal_dir.glob("*.jsonl")):
            date_str = f.stem
            if date_str in self._signal_cache:
                continue
            signals: list[dict] = []
            try:
                with open(f, "r") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                signals.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                continue
            self._signal_cache[date_str] = signals
            count += 1

        self._archive_preloaded = True
        logger.info("Pre-loaded %d archive files into memory", count)
        return count

    def _get_ta_cached(self, symbol: str, price_history: list) -> list[dict]:
        """Get TA signals for full price history, computing once per symbol.

        Instead of recomputing RSI/MACD/Bollinger/Volume for every dig site
        (50+ times per symbol with identical results), compute once and cache.
        Each dig site filters the cached result by its date window.
        """
        if symbol in self._ta_cache:
            return self._ta_cache[symbol]

        if not price_history or len(price_history) < 35:
            self._ta_cache[symbol] = []
            return []

        first_ts = getattr(price_history[0], "timestamp", "")[:10]
        last_ts = getattr(price_history[-1], "timestamp", "")[:10]
        if not first_ts or not last_ts:
            self._ta_cache[symbol] = []
            return []

        try:
            full_start = date.fromisoformat(first_ts)
            full_end = date.fromisoformat(last_ts) + timedelta(days=1)
        except ValueError:
            self._ta_cache[symbol] = []
            return []

        signals = self._compute_ta_signals(symbol, price_history, full_start, full_end)
        self._ta_cache[symbol] = signals
        return signals

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

        # Tier 1: TA indicators — use cached full-history computation, filter by window
        all_ta = self._get_ta_cached(symbol, price_history)
        for sig in all_ta:
            ts = sig.get("timestamp", "")[:10]
            try:
                sig_date = date.fromisoformat(ts)
            except ValueError:
                continue
            if start_date <= sig_date < end_date:
                signals.append(sig)

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

    # ── Tier 1: TA from price data (delegates to historical_ta module) ────────

    def _compute_ta_signals(
        self,
        symbol: str,
        price_history: list,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        return compute_ta_signals(symbol, price_history, start_date, end_date)

    def _compute_rsi_series(self, symbol, history, date_idx, start, end):
        return compute_rsi_series(symbol, history, date_idx, start, end)

    def _compute_macd_series(self, symbol, history, date_idx, start, end):
        return compute_macd_series(symbol, history, date_idx, start, end)

    def _compute_bollinger_series(self, symbol, history, date_idx, start, end):
        return compute_bollinger_series(symbol, history, date_idx, start, end)

    def _compute_volume_signals(self, symbol, history, date_idx, start, end):
        return compute_volume_signals(symbol, history, date_idx, start, end)

    # ── Tier 2: API-fetchable history ─────────────────────────────────────

    def _fetch_sec_signals(
        self, symbol: str, start_date: date, end_date: date,
    ) -> list[dict]:
        """Fetch SEC Form 4 insider trades for the lookback window."""
        if self._sec_client is None:
            return []

        signals: list[dict] = []
        try:
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

                signals.append(_make_signal(
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

                    signals.append(_make_signal(
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
                    report_date_str = getattr(pos, "report_date", "")
                    if not report_date_str:
                        continue
                    try:
                        report_date = date.fromisoformat(report_date_str[:10])
                    except ValueError:
                        continue
                    if report_date < start_date or report_date >= end_date:
                        continue

                    commercial_net = getattr(pos, "commercial_net", 0) or 0
                    direction = "bullish" if commercial_net > 0 else "bearish" if commercial_net < 0 else "neutral"
                    if direction == "neutral":
                        continue

                    ticker = getattr(pos, "ticker", "")
                    signals.append(_make_signal(
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

                    signals.append(_make_signal(
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
                if sig_symbol == symbol or sig_domain in SYMBOL_AGNOSTIC_DOMAINS:
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
        return _make_signal(signal_id, source, symbol, domain, direction, strength, date_str, metadata)

    def clear_cache(self) -> None:
        """Clear per-symbol caches between symbols.

        Preserves the signal archive (pre-loaded or lazily loaded) and
        bulk API caches. Only clears TA cache (per-symbol computation).
        """
        self._ta_cache.clear()

    def clear_all_caches(self) -> None:
        """Clear all caches including archive and bulk data (between full runs)."""
        self._signal_cache.clear()
        self._ta_cache.clear()
        self._archive_preloaded = False
        self._fred_cache.clear()
        self._cot_cache.clear()
        self._congress_cache = None
