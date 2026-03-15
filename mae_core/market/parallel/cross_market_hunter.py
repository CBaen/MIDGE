"""cross_market_hunter.py — Cross-market anomaly detection process.

Runs as an INDEPENDENT OS process alongside the daemon. Discovers patterns
the convergence engine misses: the convergence engine stacks domains per
ticker; this process looks ACROSS tickers and markets for structural
anomalies that span entire asset classes.

What it finds:
  1. Correlation breakdown — two normally-correlated domains decouple
     (r > 0.5 over 30 days → r < 0.2 over last 7 days). Regime change signal.
  2. Volume clusters — multiple unrelated tickers show volume spikes on the
     same day. Coordinated institutional activity or news arbitrage.
  3. Cross-asset divergence — simultaneous fear + greed signals (VIX up +
     stocks up, or crypto up + gold up). Contradictory regime = inflection.
  4. Domain silence — a domain that normally generates 50+ signals/day
     suddenly goes quiet (< 10). Either a broken data source or a genuinely
     quiet market (itself a signal).

Design:
  - Reads 60 days of signal archive from data/midge/signals/
  - No daemon imports, no bootstrap, no EventBus
  - Writes discoveries to data/market/cross_market_discoveries.jsonl
  - Writes bridge signals to data/market/cross_market_signals.jsonl
    (daemon picks these up via sensing_hook._ingest_bridge_file)
  - Logs to data/market/cross_market_hunter.log
  - State checkpoint: data/market/cross_market_state.json
  - 120-second sleep between cycles

Usage:
    python -m mae_core.market.parallel.cross_market_hunter
    python -m mae_core.market.parallel.cross_market_hunter --once
    python -m mae_core.market.parallel.cross_market_hunter --interval 300
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SIGNALS_DIR   = _PROJECT_ROOT / "data" / "midge" / "signals"
_DATA_MARKET   = _PROJECT_ROOT / "data" / "market"

DISCOVERIES_FILE  = _DATA_MARKET / "cross_market_discoveries.jsonl"
BRIDGE_FILE       = _DATA_MARKET / "cross_market_signals.jsonl"
STATE_FILE        = _DATA_MARKET / "cross_market_state.json"
LOG_FILE          = _DATA_MARKET / "cross_market_hunter.log"

# ── Config ─────────────────────────────────────────────────────────────────────
LOOKBACK_DAYS          = 60    # Days of archive to load each cycle
BASELINE_DAYS          = 30    # Window for "normal" correlation baseline
RECENT_DAYS            = 7     # Window for "recent" correlation check
CORRELATION_HIGH       = 0.50  # r above this = "normally correlated"
CORRELATION_LOW        = 0.20  # r below this (when normally high) = breakdown
MIN_BASELINE_POINTS    = 20    # Minimum data points to compute reliable baseline
MIN_RECENT_POINTS      = 5     # Minimum points in the recent window
VOLUME_SPIKE_FACTOR    = 2.5   # Strength must be N× its rolling mean to count
VOLUME_CLUSTER_MIN     = 3     # Minimum tickers needed to call it a cluster
SILENCE_NORMAL_FLOOR   = 50    # Signals/day below this = silence threshold
SILENCE_THRESHOLD      = 10    # Signals/day below this = domain silence alert
MAX_BRIDGE_SIGNALS     = 500   # Safety cap per cycle

# Domain pairs to correlate (both directions checked)
DOMAIN_CORRELATION_PAIRS: List[Tuple[str, str]] = [
    ("crypto",     "sentiment"),     # Crypto vs equity sentiment
    ("technical",  "sentiment"),     # Price action vs social mood
    ("insider",    "institutional"), # Insider vs institutional — who leads?
    ("energy",     "macro"),         # Energy shocks vs macro expectations
    ("macro",      "technical"),     # FRED rates vs price action
    ("positioning","technical"),     # COT positioning vs technicals
    ("government", "contracts"),     # Legislative + contracting correlation
    ("crypto",     "technical"),     # Crypto risk vs equity technicals
]

# Which domains are "fear" vs "risk-on" for cross-asset divergence detection
_FEAR_DOMAINS      = {"macro"}          # VIX-adjacent, FRED rates, fear signals
_RISK_ON_DOMAINS   = {"crypto", "sentiment", "technical"}
_SAFE_HAVEN_SOURCES = {"vix_client", "fred_yield_curve", "finnhub_economic"}
_CRYPTO_SOURCES     = {"coingecko", "coincap", "finnhub_crypto"}
_EQUITY_SOURCES     = {"yfinance_price", "ta_indicators", "session_sweep"}


# ── Data structures ─────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    source:     str
    symbol:     str
    domain:     str
    direction:  str
    strength:   float
    confidence: float
    timestamp:  datetime
    asset_class: str = ""
    metadata:   dict = field(default_factory=dict)


@dataclass
class Discovery:
    discovery_id:      str
    discovery_type:    str          # correlation_breakdown | volume_cluster | cross_asset_divergence | domain_silence
    affected_domains:  List[str]
    affected_tickers:  List[str]
    strength:          float        # 0–1: how anomalous
    confidence:        float        # 0–1: statistical confidence
    timestamp:         str          # ISO
    description:       str          # Plain English
    detail:            dict = field(default_factory=dict)


def _discovery_to_bridge_signal(d: Discovery) -> dict:
    """Convert a Discovery to a MarketSignal-compatible bridge dict."""
    # Infer direction from description keywords
    direction = "neutral"
    if "bullish" in d.description.lower() or "risk-on" in d.description.lower():
        direction = "bullish"
    elif "bearish" in d.description.lower() or "fear" in d.description.lower():
        direction = "bearish"

    primary_ticker = d.affected_tickers[0] if d.affected_tickers else ""
    ts = d.timestamp

    return {
        "signal_id":          f"cross_hunter:{d.discovery_type}:{d.discovery_id[:8]}",
        "source":             f"cross_hunter_{d.discovery_type}",
        "symbol":             primary_ticker,
        "asset_class":        "multi",
        "domain":             "cross_market",
        "direction":          direction,
        "strength":           round(d.strength, 4),
        "confidence":         round(d.confidence, 4),
        "decay_rate":         0.15,
        "timestamp":          ts,
        "received_at":        datetime.now(timezone.utc).isoformat(),
        "velocity":           0.0,
        "outcome_symbol":     primary_ticker,
        "outcome_window_days": 7,
        "raw_id":             "",
        "raw_type":           "CrossMarketHunter",
        "metadata": {
            "discovery_type":   d.discovery_type,
            "affected_domains": d.affected_domains,
            "affected_tickers": d.affected_tickers,
            "description":      d.description,
        },
    }


# ── Archive loading ─────────────────────────────────────────────────────────────

def _parse_ts(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp to timezone-aware datetime, None on failure."""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def load_signal_archive(lookback_days: int, logger: logging.Logger) -> List[SignalRecord]:
    """Load all JSONL signal files from the last N days.

    Returns list of SignalRecord objects, sorted by timestamp ascending.
    Skips malformed lines silently (logs count at end).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    records: List[SignalRecord] = []
    skipped = 0
    files_read = 0

    if not _SIGNALS_DIR.exists():
        logger.warning("Signal archive directory not found: %s", _SIGNALS_DIR)
        return records

    for jsonl_path in sorted(_SIGNALS_DIR.glob("*.jsonl")):
        # Fast-path: skip files older than lookback by filename date (YYYY-MM-DD)
        stem = jsonl_path.stem
        try:
            file_date = datetime.strptime(stem, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff - timedelta(days=1):
                continue
        except ValueError:
            pass  # Non-date filenames are read anyway

        files_read += 1
        try:
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ts  = _parse_ts(obj.get("timestamp", ""))
                        if ts is None or ts < cutoff:
                            continue
                        records.append(SignalRecord(
                            source      = obj.get("source", ""),
                            symbol      = obj.get("symbol", ""),
                            domain      = obj.get("domain", ""),
                            direction   = obj.get("direction", ""),
                            strength    = float(obj.get("strength", 0.0)),
                            confidence  = float(obj.get("confidence", 0.0)),
                            timestamp   = ts,
                            asset_class = obj.get("asset_class", ""),
                            metadata    = obj.get("metadata", {}),
                        ))
                    except (KeyError, ValueError, json.JSONDecodeError):
                        skipped += 1
        except OSError as exc:
            logger.warning("Cannot read %s: %s", jsonl_path.name, exc)

    records.sort(key=lambda r: r.timestamp)
    logger.info(
        "Loaded %d signals from %d files (skipped %d malformed) | cutoff=%s",
        len(records), files_read, skipped, cutoff.date(),
    )
    return records


# ── Feature time series ─────────────────────────────────────────────────────────

def _build_domain_daily_series(
    records: List[SignalRecord],
) -> Dict[str, pd.Series]:
    """Aggregate signals by domain into daily mean-strength time series.

    Returns dict: domain → pd.Series indexed by date string (YYYY-MM-DD).
    """
    domain_day: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for r in records:
        if not r.domain:
            continue
        day = r.timestamp.strftime("%Y-%m-%d")
        domain_day[r.domain][day].append(r.strength)

    series: Dict[str, pd.Series] = {}
    for domain, day_map in domain_day.items():
        idx = sorted(day_map.keys())
        vals = [float(np.mean(day_map[d])) for d in idx]
        series[domain] = pd.Series(vals, index=pd.to_datetime(idx), name=domain)

    return series


def _build_source_daily_counts(
    records: List[SignalRecord],
) -> Dict[str, pd.Series]:
    """Count signals per source per day (for volume cluster and silence detection)."""
    source_day: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        if not r.source:
            continue
        day = r.timestamp.strftime("%Y-%m-%d")
        source_day[r.source][day] += 1

    series: Dict[str, pd.Series] = {}
    for source, day_map in source_day.items():
        idx = sorted(day_map.keys())
        vals = [day_map[d] for d in idx]
        series[source] = pd.Series(vals, index=pd.to_datetime(idx), name=source, dtype=float)

    return series


def _build_ticker_daily_strength(
    records: List[SignalRecord],
) -> Dict[str, Dict[str, float]]:
    """Map ticker → day → mean_strength (for volume cluster detection)."""
    ticker_day: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        sym = r.symbol
        if not sym:
            continue
        day = r.timestamp.strftime("%Y-%m-%d")
        ticker_day[sym][day].append(r.strength)

    result: Dict[str, Dict[str, float]] = {}
    for sym, day_map in ticker_day.items():
        result[sym] = {d: float(np.mean(v)) for d, v in day_map.items()}
    return result


# ── Anomaly detectors ───────────────────────────────────────────────────────────

def detect_correlation_breakdowns(
    domain_series: Dict[str, pd.Series],
    now: datetime,
) -> List[Discovery]:
    """Find domain pairs that were correlated but recently decoupled."""
    discoveries: List[Discovery] = []
    baseline_cutoff = (now - timedelta(days=BASELINE_DAYS)).strftime("%Y-%m-%d")
    recent_cutoff   = (now - timedelta(days=RECENT_DAYS)).strftime("%Y-%m-%d")
    today_str       = now.strftime("%Y-%m-%d")

    for domain_a, domain_b in DOMAIN_CORRELATION_PAIRS:
        sa = domain_series.get(domain_a)
        sb = domain_series.get(domain_b)
        if sa is None or sb is None:
            continue

        # Align on shared dates
        combined = pd.concat([sa, sb], axis=1, join="inner").dropna()
        if len(combined) < MIN_BASELINE_POINTS:
            continue

        # Baseline window
        baseline = combined[combined.index <= baseline_cutoff]
        if len(baseline) < MIN_BASELINE_POINTS:
            # Use all available if not enough baseline-only days
            baseline = combined

        # Recent window
        recent = combined[
            (combined.index >= recent_cutoff) & (combined.index <= today_str)
        ]
        if len(recent) < MIN_RECENT_POINTS:
            continue

        # Correlations
        baseline_r = float(baseline.iloc[:, 0].corr(baseline.iloc[:, 1]))
        recent_r   = float(recent.iloc[:, 0].corr(recent.iloc[:, 1]))

        if np.isnan(baseline_r) or np.isnan(recent_r):
            continue

        # Breakdown: was correlated, now decoupled
        if baseline_r >= CORRELATION_HIGH and recent_r <= CORRELATION_LOW:
            drop = baseline_r - recent_r
            strength   = min(1.0, drop / (1.0 - CORRELATION_LOW))
            confidence = min(0.95, (len(baseline) / 60) * 0.8 + (len(recent) / RECENT_DAYS) * 0.2)

            desc = (
                f"{domain_a.title()} and {domain_b.title()} domains decoupled: "
                f"30-day baseline correlation {baseline_r:.2f} dropped to {recent_r:.2f} "
                f"over the last {RECENT_DAYS} days. Correlation breakdowns signal "
                f"regime transitions — one domain is diverging from historical behaviour."
            )
            discoveries.append(Discovery(
                discovery_id     = uuid.uuid4().hex,
                discovery_type   = "correlation_breakdown",
                affected_domains = [domain_a, domain_b],
                affected_tickers = [],
                strength         = round(strength, 4),
                confidence       = round(confidence, 4),
                timestamp        = now.isoformat(),
                description      = desc,
                detail           = {
                    "domain_a":      domain_a,
                    "domain_b":      domain_b,
                    "baseline_r":    round(baseline_r, 4),
                    "recent_r":      round(recent_r, 4),
                    "r_drop":        round(drop, 4),
                    "baseline_n":    len(baseline),
                    "recent_n":      len(recent),
                },
            ))

    return discoveries


def detect_volume_clusters(
    ticker_strength: Dict[str, Dict[str, float]],
    now: datetime,
) -> List[Discovery]:
    """Detect days where many unrelated tickers show coordinated strength spikes."""
    discoveries: List[Discovery] = []

    # Build per-ticker rolling mean strength (last 30 days excluding today)
    today_str   = now.strftime("%Y-%m-%d")
    cutoff_str  = (now - timedelta(days=30)).strftime("%Y-%m-%d")

    # day → list of (ticker, strength, spike_factor)
    day_spikes: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)

    for ticker, day_map in ticker_strength.items():
        # Baseline: days before today within lookback
        baseline_vals = [
            v for d, v in day_map.items()
            if d >= cutoff_str and d < today_str
        ]
        if len(baseline_vals) < 5:
            continue
        mean_strength = float(np.mean(baseline_vals))
        std_strength  = float(np.std(baseline_vals)) + 1e-6

        # Check recent days for spikes
        recent_days = sorted(d for d in day_map if d >= cutoff_str)
        for day in recent_days:
            val    = day_map[day]
            factor = val / (mean_strength + 1e-6)
            z_score = (val - mean_strength) / std_strength
            if factor >= VOLUME_SPIKE_FACTOR and z_score >= 2.0:
                day_spikes[day].append((ticker, val, round(factor, 2)))

    # Any day with VOLUME_CLUSTER_MIN+ spiking tickers is a cluster
    for day, spikes in day_spikes.items():
        if len(spikes) < VOLUME_CLUSTER_MIN:
            continue

        spikes.sort(key=lambda x: x[2], reverse=True)  # Strongest first
        tickers      = [s[0] for s in spikes]
        mean_factor  = float(np.mean([s[2] for s in spikes]))
        strength     = min(1.0, (len(spikes) / 20) * 0.6 + (mean_factor / 10) * 0.4)
        confidence   = min(0.9, 0.5 + (len(spikes) - VOLUME_CLUSTER_MIN) * 0.05)

        desc = (
            f"{len(spikes)} unrelated tickers showed coordinated strength spikes on {day} "
            f"(average {mean_factor:.1f}× above their 30-day baseline). "
            f"Top tickers: {', '.join(tickers[:5])}. "
            f"Coordinated spikes across unrelated instruments suggest institutional "
            f"activity, sector rotation, or broad risk-on/risk-off positioning."
        )
        discoveries.append(Discovery(
            discovery_id     = uuid.uuid4().hex,
            discovery_type   = "volume_cluster",
            affected_domains = ["multi"],
            affected_tickers = tickers[:10],
            strength         = round(strength, 4),
            confidence       = round(confidence, 4),
            timestamp        = now.isoformat(),
            description      = desc,
            detail           = {
                "cluster_date":   day,
                "ticker_count":   len(spikes),
                "mean_factor":    round(mean_factor, 2),
                "top_spikes":     spikes[:10],
            },
        ))

    return discoveries


def detect_cross_asset_divergence(
    records: List[SignalRecord],
    now: datetime,
) -> List[Discovery]:
    """Find simultaneous contradictory regime signals across asset classes.

    Two patterns:
      A) Fear rally: VIX up (fear) + equities up (risk-on) simultaneously
      B) Double safety: crypto up + gold/bonds up (both risk-on AND safe-haven)
    """
    discoveries: List[Discovery] = []
    recent_cutoff = now - timedelta(days=RECENT_DAYS)

    # Partition recent signals by source family
    fear_signals:      List[SignalRecord] = []
    crypto_signals:    List[SignalRecord] = []
    equity_signals:    List[SignalRecord] = []

    for r in records:
        if r.timestamp < recent_cutoff:
            continue
        src = r.source.lower()
        if any(s in src for s in _SAFE_HAVEN_SOURCES):
            fear_signals.append(r)
        if any(s in src for s in _CRYPTO_SOURCES):
            crypto_signals.append(r)
        if any(s in src for s in _EQUITY_SOURCES):
            equity_signals.append(r)

    def _net_direction(sigs: List[SignalRecord]) -> Tuple[float, int]:
        """Returns (net_score, count) where positive = net bullish."""
        if not sigs:
            return 0.0, 0
        scores = []
        for s in sigs:
            w = s.strength * s.confidence
            scores.append(w if s.direction == "bullish" else -w if s.direction == "bearish" else 0.0)
        return float(np.sum(scores)), len(sigs)

    fear_net,   fear_n   = _net_direction(fear_signals)
    crypto_net, crypto_n = _net_direction(crypto_signals)
    equity_net, equity_n = _net_direction(equity_signals)

    MIN_SIGNALS = 3

    # Pattern A: Fear rally — fear signals bullish (VIX rising = fear rising) +
    # equity signals also bullish. VIX "bullish" in archive = VIX going up = fear rising.
    if fear_n >= MIN_SIGNALS and equity_n >= MIN_SIGNALS:
        if fear_net > 0.3 and equity_net > 0.3:
            combined_mag = (fear_net + equity_net) / (fear_n + equity_n)
            strength     = min(1.0, combined_mag * 2)
            confidence   = min(0.85, (fear_n + equity_n) / 30)
            desc = (
                f"Fear rally detected over the last {RECENT_DAYS} days: fear indicators "
                f"(VIX, macro) are rising simultaneously with equity bullish signals. "
                f"Fear net score {fear_net:.2f} (n={fear_n}), equity net score "
                f"{equity_net:.2f} (n={equity_n}). "
                f"This contradiction often precedes sharp reversals — the market is "
                f"advancing on diminishing conviction."
            )
            discoveries.append(Discovery(
                discovery_id     = uuid.uuid4().hex,
                discovery_type   = "cross_asset_divergence",
                affected_domains = ["macro", "technical"],
                affected_tickers = [],
                strength         = round(strength, 4),
                confidence       = round(confidence, 4),
                timestamp        = now.isoformat(),
                description      = desc,
                detail           = {
                    "pattern":      "fear_rally",
                    "fear_net":     round(fear_net, 4),
                    "equity_net":   round(equity_net, 4),
                    "fear_n":       fear_n,
                    "equity_n":     equity_n,
                },
            ))

    # Pattern B: Simultaneous risk-on + safe-haven demand (crypto up + fear up)
    if crypto_n >= MIN_SIGNALS and fear_n >= MIN_SIGNALS:
        if crypto_net > 0.3 and fear_net > 0.3:
            combined_mag = (crypto_net + fear_net) / (crypto_n + fear_n)
            strength     = min(1.0, combined_mag * 2)
            confidence   = min(0.80, (crypto_n + fear_n) / 20)
            desc = (
                f"Simultaneous risk-on and safe-haven demand detected over the last "
                f"{RECENT_DAYS} days: crypto (risk-on) net {crypto_net:.2f} (n={crypto_n}) "
                f"and fear indicators net {fear_net:.2f} (n={fear_n}) are both elevated. "
                f"Money is flowing into both speculative and defensive assets — rare "
                f"condition that often precedes a decisive regime shift."
            )
            discoveries.append(Discovery(
                discovery_id     = uuid.uuid4().hex,
                discovery_type   = "cross_asset_divergence",
                affected_domains = ["crypto", "macro"],
                affected_tickers = [],
                strength         = round(strength, 4),
                confidence       = round(confidence, 4),
                timestamp        = now.isoformat(),
                description      = desc,
                detail           = {
                    "pattern":      "risk_on_and_safe_haven",
                    "crypto_net":   round(crypto_net, 4),
                    "fear_net":     round(fear_net, 4),
                    "crypto_n":     crypto_n,
                    "fear_n":       fear_n,
                },
            ))

    return discoveries


def detect_domain_silence(
    records: List[SignalRecord],
    now: datetime,
) -> List[Discovery]:
    """Detect domains that have gone unusually quiet compared to their baseline."""
    discoveries: List[Discovery] = []

    baseline_cutoff = now - timedelta(days=BASELINE_DAYS)
    recent_cutoff   = now - timedelta(days=RECENT_DAYS)

    # Count signals per domain per day
    baseline_counts: Dict[str, List[int]] = defaultdict(list)
    recent_counts:   Dict[str, List[int]] = defaultdict(list)

    # Day-level aggregation
    domain_day_baseline: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    domain_day_recent:   Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in records:
        if not r.domain:
            continue
        day = r.timestamp.strftime("%Y-%m-%d")
        if baseline_cutoff <= r.timestamp < recent_cutoff:
            domain_day_baseline[r.domain][day] += 1
        elif r.timestamp >= recent_cutoff:
            domain_day_recent[r.domain][day] += 1

    all_domains = set(domain_day_baseline.keys()) | set(domain_day_recent.keys())

    for domain in all_domains:
        baseline_daily = list(domain_day_baseline[domain].values())
        recent_daily   = list(domain_day_recent[domain].values())

        if len(baseline_daily) < 5:
            continue

        baseline_mean = float(np.mean(baseline_daily))
        if baseline_mean < SILENCE_NORMAL_FLOOR:
            continue  # Domain was never busy enough to matter

        recent_mean = float(np.mean(recent_daily)) if recent_daily else 0.0

        if recent_mean <= SILENCE_THRESHOLD:
            drop_pct   = (baseline_mean - recent_mean) / (baseline_mean + 1e-6)
            strength   = min(1.0, drop_pct)
            confidence = min(0.9, len(baseline_daily) / 20)

            desc = (
                f"Domain silence detected in '{domain}': normally generates "
                f"{baseline_mean:.0f} signals/day (baseline {BASELINE_DAYS}d) but "
                f"only {recent_mean:.1f}/day in the last {RECENT_DAYS} days. "
                f"Either a data source has failed or the market in this domain is "
                f"genuinely inactive — unusual quiet is itself a signal."
            )
            discoveries.append(Discovery(
                discovery_id     = uuid.uuid4().hex,
                discovery_type   = "domain_silence",
                affected_domains = [domain],
                affected_tickers = [],
                strength         = round(strength, 4),
                confidence       = round(confidence, 4),
                timestamp        = now.isoformat(),
                description      = desc,
                detail           = {
                    "domain":         domain,
                    "baseline_mean":  round(baseline_mean, 1),
                    "recent_mean":    round(recent_mean, 1),
                    "drop_pct":       round(drop_pct * 100, 1),
                    "baseline_days":  len(baseline_daily),
                    "recent_days":    len(recent_daily),
                },
            ))

    return discoveries


# ── I/O helpers ─────────────────────────────────────────────────────────────────

def _load_existing_discovery_ids() -> Set[str]:
    """Load discovery_ids already written to avoid duplicating on each cycle."""
    ids: Set[str] = set()
    if not DISCOVERIES_FILE.exists():
        return ids
    try:
        with open(DISCOVERIES_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line.strip())
                    did = obj.get("discovery_id", "")
                    if did:
                        ids.add(did)
                except (json.JSONDecodeError, ValueError):
                    pass
    except OSError:
        pass
    return ids


def _write_discoveries(
    discoveries: List[Discovery],
    existing_ids: Set[str],
    logger: logging.Logger,
) -> int:
    """Append new discoveries to JSONL file, skipping known IDs."""
    _DATA_MARKET.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with open(DISCOVERIES_FILE, "a", encoding="utf-8") as fh:
            for d in discoveries:
                if d.discovery_id in existing_ids:
                    continue
                fh.write(json.dumps(asdict(d), default=str) + "\n")
                existing_ids.add(d.discovery_id)
                written += 1
    except OSError as exc:
        logger.error("Failed to write discoveries: %s", exc)
    return written


def _write_bridge_signals(
    discoveries: List[Discovery],
    logger: logging.Logger,
) -> int:
    """Convert discoveries to bridge signals and append to bridge file."""
    _DATA_MARKET.mkdir(parents=True, exist_ok=True)
    signals = [_discovery_to_bridge_signal(d) for d in discoveries]

    # Cap per cycle
    if len(signals) > MAX_BRIDGE_SIGNALS:
        signals.sort(key=lambda s: s["strength"], reverse=True)
        signals = signals[:MAX_BRIDGE_SIGNALS]
        logger.warning("Bridge cap hit — kept top %d signals", MAX_BRIDGE_SIGNALS)

    written = 0
    try:
        with open(BRIDGE_FILE, "a", encoding="utf-8") as fh:
            for sig in signals:
                fh.write(json.dumps(sig, default=str) + "\n")
                written += 1
    except OSError as exc:
        logger.error("Failed to write bridge signals: %s", exc)
    return written


def _write_state(state: dict) -> None:
    """Overwrite state checkpoint file atomically."""
    _DATA_MARKET.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, default=str)
        tmp.replace(STATE_FILE)
    except OSError:
        pass  # State is informational; non-fatal


# ── Logging setup ───────────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    _DATA_MARKET.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cross_market_hunter")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ── Main cycle ──────────────────────────────────────────────────────────────────

def run_cycle(logger: logging.Logger) -> dict:
    """Run one full detection cycle. Returns summary dict."""
    now     = datetime.now(timezone.utc)
    summary = {
        "cycle_time":          now.isoformat(),
        "signals_loaded":      0,
        "discoveries_found":   0,
        "discoveries_written": 0,
        "bridge_signals":      0,
        "by_type":             {},
    }

    logger.info("-- Cycle start ---------------------------------------------")

    # 1. Load archive
    records = load_signal_archive(LOOKBACK_DAYS, logger)
    summary["signals_loaded"] = len(records)
    if not records:
        logger.warning("No signals loaded — skipping detection")
        return summary

    # 2. Build feature structures
    domain_series    = _build_domain_daily_series(records)
    ticker_strength  = _build_ticker_daily_strength(records)
    logger.info(
        "Feature structures built | domains=%d | tickers=%d",
        len(domain_series), len(ticker_strength),
    )

    # 3. Run detectors
    all_discoveries: List[Discovery] = []

    corr_breaks = detect_correlation_breakdowns(domain_series, now)
    vol_clusters = detect_volume_clusters(ticker_strength, now)
    cross_divs   = detect_cross_asset_divergence(records, now)
    dom_silence  = detect_domain_silence(records, now)

    all_discoveries.extend(corr_breaks)
    all_discoveries.extend(vol_clusters)
    all_discoveries.extend(cross_divs)
    all_discoveries.extend(dom_silence)

    summary["discoveries_found"] = len(all_discoveries)
    summary["by_type"] = {
        "correlation_breakdown":  len(corr_breaks),
        "volume_cluster":         len(vol_clusters),
        "cross_asset_divergence": len(cross_divs),
        "domain_silence":         len(dom_silence),
    }

    logger.info(
        "Detectors complete | corr_breaks=%d | vol_clusters=%d | "
        "cross_divs=%d | dom_silence=%d",
        len(corr_breaks), len(vol_clusters), len(cross_divs), len(dom_silence),
    )

    if not all_discoveries:
        logger.info("No anomalies found this cycle")
        return summary

    # 4. Persist
    existing_ids = _load_existing_discovery_ids()
    written      = _write_discoveries(all_discoveries, existing_ids, logger)
    bridge       = _write_bridge_signals(all_discoveries, logger)

    summary["discoveries_written"] = written
    summary["bridge_signals"]      = bridge

    logger.info(
        "Cycle complete | new_discoveries=%d | bridge_signals=%d",
        written, bridge,
    )

    # Log notable findings
    for d in sorted(all_discoveries, key=lambda x: x.strength, reverse=True)[:3]:
        logger.info("  [%s] strength=%.2f  %s", d.discovery_type, d.strength, d.description[:100])

    return summary


def main(interval: int = 120, once: bool = False) -> None:
    """Entry point. Runs in infinite loop unless --once is set."""
    logger = _setup_logger()
    logger.info(
        "Cross-market hunter starting | interval=%ds | lookback=%dd | "
        "bridge=%s | discoveries=%s",
        interval, LOOKBACK_DAYS, BRIDGE_FILE, DISCOVERIES_FILE,
    )

    cycle_count  = 0
    total_found  = 0

    while True:
        try:
            summary = run_cycle(logger)
            cycle_count += 1
            total_found += summary["discoveries_found"]

            _write_state({
                "last_cycle":        summary["cycle_time"],
                "cycles_run":        cycle_count,
                "total_discoveries": total_found,
                "last_summary":      summary,
                "status":            "running",
            })

        except KeyboardInterrupt:
            logger.info("Interrupted by user — shutting down")
            _write_state({"status": "stopped", "cycles_run": cycle_count})
            break
        except Exception as exc:
            logger.exception("Unexpected error in cycle: %s", exc)
            # Continue — don't let a transient error kill the process

        if once:
            logger.info("--once flag set — exiting after single cycle")
            break

        logger.info("Sleeping %ds before next cycle …", interval)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-market anomaly hunter")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit (useful for testing)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Seconds between cycles (default: 120)",
    )
    args = parser.parse_args()
    main(interval=args.interval, once=args.once)
