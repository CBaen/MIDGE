#!/usr/bin/env python3
"""crypto_education_sprint.py — Massive parallel crypto pattern archaeology.

MIDGE's fastest learning process. Downloads years of crypto history from
Binance (free, no auth), finds every major move, extracts precursor signals,
and builds pattern templates at 10x the rate of the normal research team.

Architecture:
  - 8 parallel workers via ProcessPoolExecutor
  - Each worker gets a batch of 12-15 coins
  - For each coin: fetch candles → find dig sites → extract precursors →
    build fingerprints → accumulate templates
  - Runs three passes: daily → 4H → 1H candles (each timeframe reveals
    different patterns)
  - Continues indefinitely until killed

Output:
  data/market/pattern_library.jsonl       — shared fingerprint store (appended)
  data/market/pattern_templates.jsonl     — shared template store (merged)
  data/midge/crypto_templates.jsonl       — crypto-only templates (appended)
  data/midge/crypto_sprint_state.json     — progress state (restartable)
  data/midge/crypto_sprint.log            — run log

Process registry: midge-crypto-sprint  tier=research  critical=True

Usage:
    python -m mae_core.market.parallel.crypto_education_sprint
    python -m mae_core.market.parallel.crypto_education_sprint --workers 4
    python -m mae_core.market.parallel.crypto_education_sprint --dry-run
    python -m mae_core.market.parallel.crypto_education_sprint --timeframe 4h
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]  # mae_core/market/parallel/… -> MIDGE root

DATA_MIDGE = _PROJECT_ROOT / "data" / "midge"
DATA_MARKET = _PROJECT_ROOT / "data" / "market"

PATTERN_LIBRARY_PATH = DATA_MARKET / "pattern_library.jsonl"
TEMPLATES_PATH = DATA_MARKET / "pattern_templates.jsonl"
CRYPTO_TEMPLATES_PATH = DATA_MIDGE / "crypto_templates.jsonl"
SIGNALS_DIR = DATA_MIDGE / "signals"

STATE_PATH = DATA_MIDGE / "crypto_sprint_state.json"
LOG_PATH = DATA_MIDGE / "crypto_sprint.log"
PID_PATH = DATA_MIDGE / "crypto_sprint.pid"

# ---------------------------------------------------------------------------
# Crypto universe — top 100 by market cap + volatility (Binance USDT pairs)
# ---------------------------------------------------------------------------

CRYPTO_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "UNIUSDT",
    "SHIBUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "FILUSDT", "ICPUSDT", "HBARUSDT", "VETUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "AAVEUSDT", "GRTUSDT", "MKRUSDT",
    "SNXUSDT", "CRVUSDT", "LDOUSDT", "RNDRUSDT", "INJUSDT", "SUIUSDT",
    "SEIUSDT", "TIAUSDT", "JUPUSDT", "WLDUSDT", "STXUSDT", "IMXUSDT",
    "PEPEUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT", "ORDIUSDT",
    "FETUSDT", "AGIXUSDT", "OCEANUSDT", "RUNEUSDT", "ENAUSDT",
    "PENDLEUSDT", "EIGENUSDT", "ETHFIUSDT",
    "TRXUSDT", "TONUSDT", "ALGOUSDT", "XLMUSDT", "EOSUSDT",
    "FTMUSDT", "THETAUSDT", "EGLDUSDT", "FLOWUSDT", "MINAUSDT",
    "ZILUSDT", "ENJUSDT", "GALAUSDT", "CHZUSDT", "APEUSDT",
    "DYDXUSDT", "GMXUSDT", "COMPUSDT", "YFIUSDT", "SUSHIUSDT",
    "1INCHUSDT", "CELOUSDT", "KSMUSDT", "ZECUSDT", "DASHUSDT",
    "KAVAUSDT", "ANKRUSDT", "SKLUSDT", "IOTAUSDT", "ONTUSDT",
    "QTUMUSDT", "WAVESUSDT", "ZENUSDT", "BATUSDT", "COTIUSDT",
    "RVNUSDT", "LRCUSDT", "STORJUSDT", "CELRUSDT", "BAKEUSDT",
    "TLMUSDT", "SFPUSDT", "REEFUSDT", "XVSUSDT", "ALPHAUSDT",
]

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Dig site thresholds — crypto moves more than equities
MIN_SINGLE_DAY_PCT: float = 5.0    # 5%+ daily move
MIN_5DAY_PCT: float = 15.0         # 15%+ over 5 days
MIN_20DAY_PCT: float = 30.0        # 30%+ over 20 days
VOLUME_SPIKE_MULTIPLIER: float = 3.0  # 3x average volume

LOOKBACK_DAYS: int = 30            # How far back to look for precursors
MIN_PRECURSOR_SIGNALS: int = 2     # Crypto-specific: lower threshold (fewer sources)

# Timeframe configs: (interval_str, candles_to_fetch, days_of_history_label)
TIMEFRAME_CONFIGS = {
    "1d": ("1d", 730, "2yr daily"),
    "4h": ("4h", 1000, "166d 4H"),
    "1h": ("1h", 1000, "41d 1H"),
}

# Binance rate limit: 1200 req/min. With 8 workers each doing 1 req/2s = 4 req/s = 240/min.
# Well under limit. Per-worker delay is enforced inside BinanceCryptoHistoryClient.
WORKERS: int = 8
BATCH_SIZE: int = 13  # coins per worker (100 / 8 ≈ 12-13)

# How long to rest between full-universe passes (all timeframes done)
CYCLE_REST_SECONDS: float = 300.0  # 5 minutes between complete cycles

# Source→domain map for TA signals derived from Binance candles
_CRYPTO_TA_DOMAIN = "technical"
_CRYPTO_PRICE_DOMAIN = "technical"
_CRYPTO_VOLUME_DOMAIN = "technical"
_CRYPTO_MOMENTUM_DOMAIN = "technical"

# Source names for crypto-derived TA signals
_SRC_RSI = "crypto_ta_rsi"
_SRC_MACD = "crypto_ta_macd"
_SRC_BOLLINGER = "crypto_ta_bollinger"
_SRC_VOLUME = "crypto_ta_volume"
_SRC_PRICE_STRUCTURE = "crypto_ta_structure"

# ---------------------------------------------------------------------------
# Logging helper — each worker configures its own logger
# ---------------------------------------------------------------------------


def _configure_logging(log_path: Path, worker_id: Optional[int] = None) -> logging.Logger:
    name = f"crypto_sprint.worker{worker_id}" if worker_id is not None else "crypto_sprint.main"
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        f"%(asctime)s [SPRINT{'' if worker_id is None else f'-W{worker_id}'}] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Pure TA computations from raw candles — no external dependencies
# ---------------------------------------------------------------------------


def _compute_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Compute RSI. Returns list aligned with closes (NaN as -1 for missing)."""
    if len(closes) < period + 1:
        return [-1.0] * len(closes)
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_values = [-1.0] * (period + 1)

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
        rsi_values.append(100.0 - 100.0 / (1.0 + rs))

    return rsi_values


def _compute_ema(values: list[float], period: int) -> list[float]:
    """Compute EMA. Returns list aligned with values."""
    if not values:
        return []
    k = 2.0 / (period + 1)
    ema = values[0]
    result = [ema]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
        result.append(ema)
    return result


def _compute_macd(closes: list[float]) -> list[dict]:
    """Compute MACD line, signal, histogram. Returns list of dicts."""
    if len(closes) < 26:
        return [{"macd": 0.0, "signal": 0.0, "hist": 0.0}] * len(closes)
    ema12 = _compute_ema(closes, 12)
    ema26 = _compute_ema(closes, 26)
    macd_line = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal_line = _compute_ema(macd_line, 9)
    result = []
    for i in range(len(closes)):
        result.append({
            "macd": macd_line[i],
            "signal": signal_line[i],
            "hist": macd_line[i] - signal_line[i],
        })
    return result


def _compute_bollinger(closes: list[float], period: int = 20) -> list[dict]:
    """Compute Bollinger Bands. Returns list of dicts."""
    result = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append({"upper": 0.0, "middle": 0.0, "lower": 0.0, "width": 0.0, "pct_b": 0.5})
            continue
        window = closes[i - period + 1: i + 1]
        mid = sum(window) / period
        std = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
        upper = mid + 2 * std
        lower = mid - 2 * std
        width = (upper - lower) / mid if mid != 0 else 0.0
        pct_b = (closes[i] - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        result.append({"upper": upper, "middle": mid, "lower": lower,
                        "width": width, "pct_b": pct_b})
    return result


def _avg_volume(volumes: list[float], window: int = 20) -> list[float]:
    """Rolling average volume. Returns list aligned with volumes."""
    result = []
    for i in range(len(volumes)):
        w = volumes[max(0, i - window + 1): i + 1]
        result.append(sum(w) / len(w))
    return result


# ---------------------------------------------------------------------------
# Dig site finder
# ---------------------------------------------------------------------------


def _find_dig_sites(candles: list[dict]) -> list[dict]:
    """Find significant price moves in candle data.

    Returns list of dicts: {idx, date, direction, pct, trigger_type}
    trigger_type: "single_day" | "5day" | "20day" | "volume_spike"
    """
    if len(candles) < 25:
        return []

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    avg_vols = _avg_volume(volumes, window=20)

    sites = {}  # idx → site (deduplicate: one site per index)

    for i in range(1, len(candles)):
        date_str = candles[i]["date"][:10]

        # Single-day move
        if candles[i - 1]["close"] != 0:
            daily_pct = (closes[i] - closes[i - 1]) / closes[i - 1] * 100.0
            if abs(daily_pct) >= MIN_SINGLE_DAY_PCT:
                direction = "bullish" if daily_pct > 0 else "bearish"
                if i not in sites or abs(daily_pct) > abs(sites[i]["pct"]):
                    sites[i] = {
                        "idx": i, "date": date_str, "direction": direction,
                        "pct": daily_pct, "trigger_type": "single_day",
                    }

        # 5-day move
        if i >= 5 and closes[i - 5] != 0:
            pct5 = (closes[i] - closes[i - 5]) / closes[i - 5] * 100.0
            if abs(pct5) >= MIN_5DAY_PCT:
                direction = "bullish" if pct5 > 0 else "bearish"
                if i not in sites or abs(pct5) > abs(sites[i]["pct"]):
                    sites[i] = {
                        "idx": i, "date": date_str, "direction": direction,
                        "pct": pct5, "trigger_type": "5day",
                    }

        # 20-day move
        if i >= 20 and closes[i - 20] != 0:
            pct20 = (closes[i] - closes[i - 20]) / closes[i - 20] * 100.0
            if abs(pct20) >= MIN_20DAY_PCT:
                direction = "bullish" if pct20 > 0 else "bearish"
                if i not in sites or abs(pct20) > abs(sites[i]["pct"]):
                    sites[i] = {
                        "idx": i, "date": date_str, "direction": direction,
                        "pct": pct20, "trigger_type": "20day",
                    }

        # Volume spike (independent — can overlay another dig site)
        if avg_vols[i] > 0 and volumes[i] >= avg_vols[i] * VOLUME_SPIKE_MULTIPLIER:
            if i not in sites:
                # Volume spike alone: direction from price
                daily_pct = (closes[i] - closes[i - 1]) / closes[i - 1] * 100.0 if closes[i - 1] != 0 else 0.0
                direction = "bullish" if daily_pct >= 0 else "bearish"
                sites[i] = {
                    "idx": i, "date": date_str, "direction": direction,
                    "pct": daily_pct, "trigger_type": "volume_spike",
                }
            else:
                # Annotate existing site
                sites[i]["volume_spike"] = True
                sites[i]["volume_ratio"] = volumes[i] / avg_vols[i]

    return sorted(sites.values(), key=lambda x: x["idx"])


# ---------------------------------------------------------------------------
# Signal archive loader (read-only, no writes)
# ---------------------------------------------------------------------------


def _load_archive_signals(signals_dir: Path, symbol_base: str, move_date_str: str) -> list[dict]:
    """Load signals from archive for the 30 days before move_date.

    symbol_base: e.g. "BTC" (from "BTCUSDT")
    Returns list of signal dicts with keys: source, domain, direction, strength, date
    """
    try:
        move_dt = datetime.strptime(move_date_str, "%Y-%m-%d").date()
    except ValueError:
        return []

    results = []
    # Check each day in the lookback window
    for lag in range(1, LOOKBACK_DAYS + 1):
        check_dt = move_dt - timedelta(days=lag)
        archive_file = signals_dir / f"{check_dt.isoformat()}.jsonl"
        if not archive_file.exists():
            continue
        try:
            with open(archive_file, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sig = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Match by ticker or symbol
                    sig_ticker = (sig.get("ticker") or sig.get("symbol") or "").upper()
                    if symbol_base.upper() not in sig_ticker and sig_ticker not in symbol_base.upper():
                        continue
                    sig["_lag_days"] = lag
                    results.append(sig)
        except OSError:
            continue

    return results


# ---------------------------------------------------------------------------
# Fingerprint builder
# ---------------------------------------------------------------------------


def _build_precursors_from_ta(
    candles: list[dict],
    dig_site_idx: int,
    direction: str,
) -> list[dict]:
    """Compute retroactive TA signals from candle data as precursor signals.

    Returns list of precursor dicts compatible with PrecursorSignal.to_dict().
    """
    if dig_site_idx < 5:
        return []

    # Use up to 200 candles ending at the day BEFORE the move (no look-ahead)
    end_idx = dig_site_idx  # exclusive — move candle not included
    start_idx = max(0, end_idx - 200)
    window = candles[start_idx:end_idx]
    if len(window) < 20:
        return []

    closes = [c["close"] for c in window]
    volumes = [c["volume"] for c in window]

    rsi = _compute_rsi(closes)
    macd_data = _compute_macd(closes)
    bb_data = _compute_bollinger(closes)
    avg_vols = _avg_volume(volumes, window=20)

    precursors = []
    now_idx = len(window) - 1  # last candle = day before move

    # -- RSI signal --
    rsi_val = rsi[now_idx]
    if rsi_val > 0:
        # Oversold (<30) before bullish, overbought (>70) before bearish
        if direction == "bullish" and rsi_val < 35:
            strength = (35.0 - rsi_val) / 35.0
            precursors.append({
                "source": _SRC_RSI, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bullish", "strength": min(strength, 1.0),
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"rsi": rsi_val},
            })
        elif direction == "bearish" and rsi_val > 65:
            strength = (rsi_val - 65.0) / 35.0
            precursors.append({
                "source": _SRC_RSI, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bearish", "strength": min(strength, 1.0),
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"rsi": rsi_val},
            })

    # -- MACD signal --
    if now_idx < len(macd_data):
        m = macd_data[now_idx]
        hist = m.get("hist", 0.0)
        # Histogram crossing zero (momentum shift)
        prev_hist = macd_data[now_idx - 1].get("hist", 0.0) if now_idx > 0 else 0.0
        if direction == "bullish" and hist > 0 and prev_hist <= 0:
            precursors.append({
                "source": _SRC_MACD, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bullish", "strength": min(abs(hist) / (abs(closes[-1]) * 0.01 + 1e-9), 1.0),
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"macd_hist": hist},
            })
        elif direction == "bearish" and hist < 0 and prev_hist >= 0:
            precursors.append({
                "source": _SRC_MACD, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bearish", "strength": min(abs(hist) / (abs(closes[-1]) * 0.01 + 1e-9), 1.0),
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"macd_hist": hist},
            })

    # -- Bollinger signal --
    if now_idx < len(bb_data):
        bb = bb_data[now_idx]
        pct_b = bb.get("pct_b", 0.5)
        width = bb.get("width", 0.0)
        if direction == "bullish" and pct_b < 0.1:
            precursors.append({
                "source": _SRC_BOLLINGER, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bullish", "strength": (0.1 - pct_b) / 0.1,
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"pct_b": pct_b, "bb_width": width},
            })
        elif direction == "bearish" and pct_b > 0.9:
            precursors.append({
                "source": _SRC_BOLLINGER, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bearish", "strength": (pct_b - 0.9) / 0.1,
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"pct_b": pct_b, "bb_width": width},
            })
        # Bollinger squeeze (low width → expansion imminent)
        if width < 0.03 and now_idx >= 10:
            prev_widths = [bb_data[i].get("width", 0.0) for i in range(max(0, now_idx - 10), now_idx)]
            if prev_widths and width < min(prev_widths) * 1.2:
                precursors.append({
                    "source": _SRC_BOLLINGER, "domain": _CRYPTO_TA_DOMAIN,
                    "direction": direction, "strength": 0.6,
                    "lag_days": 2, "lag_bucket": "immediate", "signal_id": "",
                    "entity_metadata": {"type": "squeeze", "bb_width": width},
                })

    # -- Volume signal --
    if now_idx < len(avg_vols) and avg_vols[now_idx] > 0:
        vol_ratio = volumes[now_idx] / avg_vols[now_idx]
        if vol_ratio >= VOLUME_SPIKE_MULTIPLIER:
            precursors.append({
                "source": _SRC_VOLUME, "domain": _CRYPTO_TA_DOMAIN,
                "direction": direction, "strength": min(vol_ratio / 10.0, 1.0),
                "lag_days": 1, "lag_bucket": "immediate", "signal_id": "",
                "entity_metadata": {"volume_ratio": vol_ratio},
            })

    # -- Multi-day RSI divergence (look at recent 5 days) --
    rsi_window = rsi[max(0, now_idx - 5): now_idx + 1]
    valid_rsi = [r for r in rsi_window if r > 0]
    if len(valid_rsi) >= 3:
        price_trend = closes[-1] - closes[max(0, now_idx - 5)]
        rsi_trend = valid_rsi[-1] - valid_rsi[0]
        # Divergence: price up but RSI down (bearish) or price down but RSI up (bullish)
        if direction == "bullish" and price_trend < 0 and rsi_trend > 3:
            precursors.append({
                "source": _SRC_RSI, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bullish", "strength": 0.7,
                "lag_days": 3, "lag_bucket": "short", "signal_id": "",
                "entity_metadata": {"type": "bullish_divergence"},
            })
        elif direction == "bearish" and price_trend > 0 and rsi_trend < -3:
            precursors.append({
                "source": _SRC_RSI, "domain": _CRYPTO_TA_DOMAIN,
                "direction": "bearish", "strength": 0.7,
                "lag_days": 3, "lag_bucket": "short", "signal_id": "",
                "entity_metadata": {"type": "bearish_divergence"},
            })

    return precursors


def _build_precursors_from_archive(archive_signals: list[dict], direction: str) -> list[dict]:
    """Convert raw archive signal dicts into precursor dicts."""
    # Source→domain map (from historical_research_team.py)
    domain_map = {
        "sec_form4": "insider", "sec_form4_cluster": "insider",
        "openinsider": "insider", "finviz_insider": "insider",
        "coingecko": "crypto", "coincap": "crypto",
        "stocktwits": "sentiment", "google_trends": "sentiment",
        "finnhub_news": "events", "yahoo_rss": "events",
        "fred_macro": "macro", "fred_yield_curve": "macro",
        "economic_calendar": "macro",
        "ta_rsi": "technical", "ta_macd": "technical",
        "ta_bollinger": "technical", "ta_volume": "technical",
    }
    precursors = []
    for sig in archive_signals:
        source = sig.get("source", "unknown")
        domain = sig.get("domain") or domain_map.get(source, "unknown")
        if domain == "unknown":
            continue
        sig_dir = sig.get("direction", direction)
        strength = float(sig.get("strength", 0.5))
        lag = int(sig.get("_lag_days", 1))
        bucket = _lag_bucket(lag)
        precursors.append({
            "source": source, "domain": domain,
            "direction": sig_dir, "strength": min(max(strength, 0.0), 1.0),
            "lag_days": lag, "lag_bucket": bucket,
            "signal_id": sig.get("signal_id", ""),
            "entity_metadata": sig.get("entity_metadata", {}),
        })
    return precursors


def _lag_bucket(days: int) -> str:
    if days <= 2:
        return "immediate"
    if days <= 5:
        return "short"
    if days <= 10:
        return "medium"
    if days <= 20:
        return "long"
    return "extended"


def _fingerprint_id(symbol: str, direction: str, move_date: str) -> str:
    raw = f"{symbol}:{direction}:{move_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _domain_signature(precursors: list[dict]) -> str:
    domains = sorted(set(p["domain"] for p in precursors))
    return "+".join(domains)


def _template_id(direction: str, domain_sig: str) -> str:
    raw = f"{direction}:{domain_sig}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Template accumulator (in-process, local to each worker batch)
# ---------------------------------------------------------------------------


class _TemplateAccumulator:
    """Accumulates MoveFingerprints into PatternTemplates within one worker.

    At end of batch, merged into shared files via _merge_outputs().
    """

    def __init__(self) -> None:
        # template_key → template dict
        self._templates: dict[str, dict] = {}
        # fingerprint_id set (dedup)
        self._seen_fp: set[str] = set()
        # list of new fingerprint dicts to append
        self._new_fingerprints: list[dict] = []

    def add_fingerprint(self, fp: dict) -> None:
        fp_id = fp["fingerprint_id"]
        if fp_id in self._seen_fp:
            return
        self._seen_fp.add(fp_id)
        self._new_fingerprints.append(fp)

        # Accumulate into template
        direction = fp["direction"]
        domain_sig = fp["domain_signature"]
        tkey = f"{direction}:{domain_sig}"
        tid = _template_id(direction, domain_sig)

        if tkey not in self._templates:
            domains = domain_sig.split("+") if domain_sig else []
            self._templates[tkey] = {
                "template_id": tid,
                "direction": direction,
                "domain_signature": domain_sig,
                "domains": domains,
                "lag_profile_raw": {},
                "lag_profile_normalized": {},
                "instances": [],
                "symbol_set": [],
                "wins": 0,
                "losses": 0,
                "total_activations": 0,
                "cross_validated": False,
                "crypto_specific": True,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }

        tmpl = self._templates[tkey]

        # Accumulate lag profile
        for precursor in fp.get("precursor_signals", []):
            bucket = precursor.get("lag_bucket", "immediate")
            tmpl["lag_profile_raw"][bucket] = tmpl["lag_profile_raw"].get(bucket, 0) + 1

        # Add instance
        instance = {
            "symbol": fp["symbol"],
            "move_date": fp["move_date"],
            "move_pct": fp["move_pct"],
            "fingerprint_id": fp_id,
            "regime": fp.get("regime", "default"),
        }
        tmpl["instances"].append(instance)

        # Update symbol set
        sym = fp["symbol"]
        if sym not in tmpl["symbol_set"]:
            tmpl["symbol_set"].append(sym)

        # Cross-validation flag
        if len(tmpl["symbol_set"]) >= 3:
            tmpl["cross_validated"] = True

        tmpl["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    @property
    def fingerprint_count(self) -> int:
        return len(self._new_fingerprints)

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def templates_list(self) -> list[dict]:
        return list(self._templates.values())

    def fingerprints_list(self) -> list[dict]:
        return self._new_fingerprints


# ---------------------------------------------------------------------------
# Worker function — runs in subprocess
# ---------------------------------------------------------------------------


def _worker_process_batch(
    worker_id: int,
    symbols: list[str],
    timeframe: str,
    signals_dir_str: str,
    dry_run: bool,
) -> dict:
    """Worker function executed in a subprocess.

    Downloads candle data for each symbol, finds dig sites, builds
    fingerprints, and returns accumulated data back to the coordinator.

    Returns:
        {
            "worker_id": int,
            "symbols_done": list[str],
            "fingerprints": list[dict],
            "templates": list[dict],
            "error": str or None,
        }
    """
    log_path = Path(LOG_PATH)
    logger = _configure_logging(log_path, worker_id=worker_id)

    # Use yfinance instead of Binance API (Binance geo-blocks US IPs)
    import yfinance as yf
    signals_dir = Path(signals_dir_str)
    accumulator = _TemplateAccumulator()

    tf_config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["1d"])
    interval_str, candle_count, tf_label = tf_config

    symbols_done = []

    for symbol in symbols:
        logger.info("Processing %s [%s]", symbol, tf_label)
        # Symbol base for archive lookup (e.g. "BTCUSDT" → "BTC")
        symbol_base = symbol.replace("USDT", "")

        # Fetch candle data via yfinance (Binance API is geo-blocked in US)
        try:
            yf_symbol = symbol.replace("USDT", "-USD")  # BTCUSDT → BTC-USD
            period_map = {"1d": "2y", "4h": "60d", "1h": "7d"}
            interval_map = {"1d": "1d", "4h": "1h", "1h": "1h"}  # yfinance doesn't have 4h
            df = yf.download(
                yf_symbol,
                period=period_map.get(interval_str, "2y"),
                interval=interval_map.get(interval_str, "1d"),
                progress=False,
            )
            if df.empty:
                logger.debug("No yfinance data for %s", yf_symbol)
                continue
            candles = []
            for idx_dt, row in df.iterrows():
                candles.append({
                    "date": str(idx_dt.date()) if hasattr(idx_dt, "date") else str(idx_dt)[:10],
                    "open": float(row.get("Open", row.iloc[0])),
                    "high": float(row.get("High", row.iloc[1])),
                    "low": float(row.get("Low", row.iloc[2])),
                    "close": float(row.get("Close", row.iloc[3])),
                    "volume": float(row.get("Volume", row.iloc[4]) if len(row) > 4 else 0),
                })
        except Exception as exc:
            logger.warning("Candle fetch failed for %s: %s", symbol, exc)
            continue

        if len(candles) < 30:
            logger.debug("Insufficient candles for %s (%d)", symbol, len(candles))
            continue

        # Find dig sites
        dig_sites = _find_dig_sites(candles)
        logger.debug("%s: %d dig sites found", symbol, len(dig_sites))

        for site in dig_sites:
            idx = site["idx"]
            move_date = site["date"]
            direction = site["direction"]
            move_pct = site["pct"]

            # Build TA-derived precursors (always available)
            ta_precursors = _build_precursors_from_ta(candles, idx, direction)

            # Build archive-derived precursors (when archive covers this date)
            archive_sigs = _load_archive_signals(signals_dir, symbol_base, move_date)
            archive_precursors = _build_precursors_from_archive(archive_sigs, direction)

            all_precursors = ta_precursors + archive_precursors

            # Dedup by source
            seen_sources: set[str] = set()
            deduped: list[dict] = []
            for p in all_precursors:
                key = f"{p['source']}:{p['direction']}"
                if key not in seen_sources:
                    seen_sources.add(key)
                    deduped.append(p)

            if len(deduped) < MIN_PRECURSOR_SIGNALS:
                continue  # Not enough evidence

            fp_id = _fingerprint_id(symbol, direction, move_date)
            domain_sig = _domain_signature(deduped)
            lag_profile: dict[str, int] = {}
            for p in deduped:
                b = p["lag_bucket"]
                lag_profile[b] = lag_profile.get(b, 0) + 1

            fp = {
                "fingerprint_id": fp_id,
                "symbol": symbol,
                "direction": direction,
                "move_date": move_date,
                "move_pct": move_pct,
                "regime": "default",
                "lookback_days": LOOKBACK_DAYS,
                "precursor_signals": deduped,
                "domain_signature": domain_sig,
                "lag_profile": lag_profile,
                "created_at": datetime.now(tz=timezone.utc).isoformat(),
                "wins": 0,
                "losses": 0,
                "total_activations": 0,
                "last_activation": "",
                "timeframe": timeframe,
                "source": "crypto_sprint",
            }

            if not dry_run:
                accumulator.add_fingerprint(fp)

        symbols_done.append(symbol)
        logger.info(
            "Worker %d: %s done — %d fps / %d templates so far",
            worker_id, symbol, accumulator.fingerprint_count, accumulator.template_count,
        )

    return {
        "worker_id": worker_id,
        "symbols_done": symbols_done,
        "fingerprints": accumulator.fingerprints_list(),
        "templates": accumulator.templates_list(),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Output merger — runs in main process, writes to shared files
# ---------------------------------------------------------------------------


def _append_fingerprints(fps: list[dict], lib_path: Path, crypto_lib_path: Path) -> int:
    """Append new fingerprints to pattern_library.jsonl and crypto_templates.jsonl.

    Loads existing IDs to avoid duplicates. Returns count of new entries written.
    """
    # Load existing IDs
    existing_ids: set[str] = set()
    if lib_path.exists():
        try:
            with open(lib_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            existing_ids.add(obj.get("fingerprint_id", ""))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    new_fps = [fp for fp in fps if fp.get("fingerprint_id") not in existing_ids]
    if not new_fps:
        return 0

    lib_path.parent.mkdir(parents=True, exist_ok=True)
    crypto_lib_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lib_path, "a", encoding="utf-8") as fh:
        for fp in new_fps:
            fh.write(json.dumps(fp) + "\n")

    with open(crypto_lib_path, "a", encoding="utf-8") as fh:
        for fp in new_fps:
            fh.write(json.dumps(fp) + "\n")

    return len(new_fps)


def _merge_templates(new_templates: list[dict], templates_path: Path) -> int:
    """Merge new templates into pattern_templates.jsonl.

    Existing templates are updated (instances accumulated), new ones appended.
    Returns count of templates upserted.
    """
    # Load existing templates
    existing: dict[str, dict] = {}
    if templates_path.exists():
        try:
            with open(templates_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            tid = obj.get("template_id", "")
                            if tid:
                                existing[tid] = obj
                        except json.JSONDecodeError:
                            pass
        except OSError:
            pass

    changed = 0
    for tmpl in new_templates:
        tid = tmpl.get("template_id", "")
        if not tid:
            continue
        if tid in existing:
            # Merge instances and symbol_set
            ex = existing[tid]
            ex_fp_ids = {inst["fingerprint_id"] for inst in ex.get("instances", [])}
            for inst in tmpl.get("instances", []):
                if inst["fingerprint_id"] not in ex_fp_ids:
                    ex.setdefault("instances", []).append(inst)
            for sym in tmpl.get("symbol_set", []):
                if sym not in ex.get("symbol_set", []):
                    ex.setdefault("symbol_set", []).append(sym)
            # Merge lag profile
            for bucket, count in tmpl.get("lag_profile_raw", {}).items():
                ex.setdefault("lag_profile_raw", {})[bucket] = (
                    ex["lag_profile_raw"].get(bucket, 0) + count
                )
            # Update cross-validation
            if len(ex.get("symbol_set", [])) >= 3:
                ex["cross_validated"] = True
            ex["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        else:
            existing[tid] = tmpl
        changed += 1

    # Rewrite file
    templates_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = templates_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for tmpl in existing.values():
            fh.write(json.dumps(tmpl) + "\n")
    tmp_path.replace(templates_path)

    return changed


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "cycle": 0,
        "timeframe_pass": 0,
        "timeframes_completed": [],
        "symbols_completed_this_pass": [],
        "total_fingerprints": 0,
        "total_templates": 0,
        "last_updated": "",
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def _save_state(state: dict) -> None:
    state["last_updated"] = datetime.now(tz=timezone.utc).isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp.replace(STATE_PATH)


# ---------------------------------------------------------------------------
# Batch splitter
# ---------------------------------------------------------------------------


def _batches(items: list, batch_size: int) -> list[list]:
    """Split list into batches of given size."""
    result = []
    for i in range(0, len(items), batch_size):
        result.append(items[i: i + batch_size])
    return result


# ---------------------------------------------------------------------------
# Main coordinator loop
# ---------------------------------------------------------------------------


def run_sprint(
    workers: int = WORKERS,
    dry_run: bool = False,
    single_timeframe: Optional[str] = None,
) -> None:
    """Main entry point. Runs continuously: daily → 4H → 1H → repeat."""
    DATA_MIDGE.mkdir(parents=True, exist_ok=True)
    DATA_MARKET.mkdir(parents=True, exist_ok=True)

    logger = _configure_logging(LOG_PATH)

    # Write PID
    with open(PID_PATH, "w") as fh:
        fh.write(str(os.getpid()))

    logger.info(
        "Crypto education sprint started. PID=%d workers=%d dry_run=%s",
        os.getpid(), workers, dry_run,
    )

    state = _load_state()

    # Timeframe sequence
    if single_timeframe:
        timeframes = [single_timeframe]
    else:
        timeframes = ["1d", "4h", "1h"]

    try:
        while True:
            state["cycle"] += 1
            logger.info("=== CYCLE %d BEGIN ===", state["cycle"])

            for tf in timeframes:
                if tf not in TIMEFRAME_CONFIGS:
                    logger.warning("Unknown timeframe %s, skipping", tf)
                    continue

                _, _, tf_label = TIMEFRAME_CONFIGS[tf]
                logger.info("--- Timeframe pass: %s ---", tf_label)

                # Determine which symbols still need processing this pass
                completed = set(state.get("symbols_completed_this_pass", []))
                remaining = [s for s in CRYPTO_UNIVERSE if s not in completed]

                if not remaining:
                    logger.info("All symbols done for %s, resetting pass", tf_label)
                    state["symbols_completed_this_pass"] = []
                    remaining = list(CRYPTO_UNIVERSE)
                    _save_state(state)

                batches = _batches(remaining, BATCH_SIZE)
                logger.info(
                    "Timeframe %s: %d symbols in %d batches (%d workers)",
                    tf, len(remaining), len(batches), workers,
                )

                batch_idx = 0
                for batch_group in _batches(batches, workers):
                    # Submit up to `workers` batches in parallel
                    futures = {}
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        for i, batch in enumerate(batch_group):
                            worker_id = batch_idx * workers + i
                            fut = pool.submit(
                                _worker_process_batch,
                                worker_id,
                                batch,
                                tf,
                                str(SIGNALS_DIR),
                                dry_run,
                            )
                            futures[fut] = (worker_id, batch)

                        for fut in as_completed(futures):
                            worker_id, batch = futures[fut]
                            try:
                                result = fut.result(timeout=600)
                            except Exception as exc:
                                logger.error("Worker %d crashed: %s", worker_id, exc)
                                continue

                            if result.get("error"):
                                logger.error("Worker %d error: %s", worker_id, result["error"])
                                continue

                            fps = result.get("fingerprints", [])
                            tmpls = result.get("templates", [])
                            syms_done = result.get("symbols_done", [])

                            # Merge into shared files
                            new_fp_count = 0
                            new_tmpl_count = 0
                            if not dry_run:
                                new_fp_count = _append_fingerprints(
                                    fps, PATTERN_LIBRARY_PATH, CRYPTO_TEMPLATES_PATH
                                )
                                new_tmpl_count = _merge_templates(tmpls, TEMPLATES_PATH)

                            state["total_fingerprints"] += new_fp_count
                            state["total_templates"] = new_tmpl_count or state["total_templates"]
                            state["symbols_completed_this_pass"].extend(syms_done)
                            _save_state(state)

                            done_count = len(state["symbols_completed_this_pass"])
                            logger.info(
                                "Sprint: %d/%d coins done, %d fingerprints, %d templates [%s]",
                                done_count, len(CRYPTO_UNIVERSE),
                                state["total_fingerprints"],
                                state["total_templates"],
                                tf_label,
                            )

                    batch_idx += 1

                # Print milestone every 10 coins
                done_count = len(state["symbols_completed_this_pass"])
                if done_count % 10 < BATCH_SIZE:
                    logger.info(
                        "Sprint: %d/%d coins done, %d fingerprints, %d templates",
                        done_count, len(CRYPTO_UNIVERSE),
                        state["total_fingerprints"],
                        state["total_templates"],
                    )

                # Mark timeframe complete
                if tf not in state.get("timeframes_completed", []):
                    state.setdefault("timeframes_completed", []).append(tf)
                # Reset for next timeframe pass
                state["symbols_completed_this_pass"] = []
                _save_state(state)

            logger.info(
                "=== CYCLE %d COMPLETE === fps=%d templates=%d — resting %ds",
                state["cycle"],
                state["total_fingerprints"],
                state["total_templates"],
                int(CYCLE_REST_SECONDS),
            )
            time.sleep(CYCLE_REST_SECONDS)

    except KeyboardInterrupt:
        logger.info("Sprint interrupted by user. State saved.")
        _save_state(state)
    finally:
        if PID_PATH.exists():
            PID_PATH.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Process registry entry
# ---------------------------------------------------------------------------

PROCESS_REGISTRY_ENTRY = {
    "name": "midge-crypto-sprint",
    "module": "mae_core.market.parallel.crypto_education_sprint",
    "entry": "run_sprint",
    "tier": "research",
    "critical": True,
    "description": (
        "Massive parallel crypto education sprint. Downloads years of Binance "
        "OHLCV for 100 coins, extracts pattern fingerprints, builds templates. "
        "Runs daily → 4H → 1H passes continuously."
    ),
    "restart_policy": "always",
    "workers": WORKERS,
    "outputs": [
        "data/market/pattern_library.jsonl",
        "data/market/pattern_templates.jsonl",
        "data/midge/crypto_templates.jsonl",
        "data/midge/crypto_sprint_state.json",
        "data/midge/crypto_sprint.log",
    ],
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIDGE Crypto Education Sprint — massive parallel crypto archaeology"
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS,
        help=f"Number of parallel worker processes (default: {WORKERS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and analyze but do not write output files",
    )
    parser.add_argument(
        "--timeframe", choices=["1d", "4h", "1h"], default=None,
        help="Run a single timeframe pass only (default: cycle through all three)",
    )
    args = parser.parse_args()

    run_sprint(
        workers=args.workers,
        dry_run=args.dry_run,
        single_timeframe=args.timeframe,
    )


if __name__ == "__main__":
    main()
