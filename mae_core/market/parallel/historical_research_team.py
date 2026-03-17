#!/usr/bin/env python3
"""historical_research_team.py — Massive parallel historical pattern research.

This is the most important process in MIDGE's ecosystem. It runs four
independent research workers in parallel, each mining the full historical
record in a different way. The live pipeline is just the thin recognition
layer on top of what this process discovers.

Four workers:

  Worker 1: Move Reverse-Engineer
    Scans price history for major moves (>5% single day, >10% over 5 days,
    >20% over 20 days). For each move, looks BACKWARD up to 30 days to find
    precursor signals in the archive. Converts those into MoveFingerprints
    and accumulates PatternTemplates. This is the core excavation loop —
    but run aggressively across the full watchlist, not 5 symbols per cycle.

  Worker 2: Cross-Instrument Pattern Matcher
    Takes all existing templates and tests each one against every symbol
    in the watchlist. Validates: does a pattern discovered on NVDA also
    appear on AMD? On AAPL? On GC=F? Cross-validated templates (3+ symbols)
    are structurally more reliable. Writes confidence boosts to a shared
    validation file that the PatternLibrary can consume.

  Worker 3: Temporal Sequence Miner
    Reads every graded outcome and reconstructs the temporal ORDER in which
    signals arrived before the move. Builds "temporal templates" — not just
    "these domains agree" but "domain A fires, then 3 days later domain B
    fires, then 2 days later the move happens." These are more predictive
    than domain-set patterns because sequence encodes causality.

  Worker 4: Template Stacker
    Finds instances in the archive where MULTIPLE templates were simultaneously
    active on the same symbol. Records outcomes for each stack combination.
    Builds stack profiles: "when template X AND template Y are both active,
    what is the joint win rate?" This is the inevitability detector.

Architecture:
  - ProcessPoolExecutor with 4 workers (one per research type)
  - Each worker runs its own infinite loop, cycling the full dataset
  - Workers write to dedicated append-only JSONL files
  - Progress and status tracked in data/midge/research_team_state.json
  - Logs to data/midge/research_team.log
  - Workers are restartable: crash recovery via state file

Usage:
    python -m mae_core.market.parallel.historical_research_team
    python -m mae_core.market.parallel.historical_research_team --workers 2
    python -m mae_core.market.parallel.historical_research_team --dry-run
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]   # mae_core/market/parallel/… -> MIDGE root

DATA_MIDGE = _PROJECT_ROOT / "data" / "midge"
DATA_MARKET = _PROJECT_ROOT / "data" / "market"

SIGNALS_DIR           = DATA_MIDGE / "signals"
WATCHLIST_PATH        = DATA_MIDGE / "watchlist.json"
OUTCOMES_PATH         = DATA_MARKET / "outcomes.jsonl"
PATTERN_LIBRARY_PATH  = DATA_MARKET / "pattern_library.jsonl"
TEMPLATES_PATH        = DATA_MARKET / "pattern_templates.jsonl"

# Worker-specific output files (append-only JSONL)
CROSS_INSTRUMENT_PATH = DATA_MIDGE / "cross_instrument_validation.jsonl"
TEMPORAL_TEMPLATES_PATH = DATA_MIDGE / "temporal_templates.jsonl"
STACK_PROFILES_PATH     = DATA_MIDGE / "stack_profiles.jsonl"

STATE_PATH = DATA_MIDGE / "research_team_state.json"
LOG_PATH   = DATA_MIDGE / "research_team.log"
PID_PATH   = DATA_MIDGE / "research_team.pid"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

CYCLE_REST_SECONDS: float = 30.0          # Sleep between full cycles

# Worker 1 — Move Reverse-Engineer
MIN_SINGLE_DAY_PCT: float = 5.0           # 5%+ single-day move qualifies
MIN_5DAY_PCT: float = 10.0                # 10%+ over 5 days
MIN_20DAY_PCT: float = 20.0               # 20%+ over 20 days
LOOKBACK_DAYS: int = 30                   # How far back to search for precursors
PRICE_HISTORY_DAYS: int = 730             # 2 years of price history
MIN_PRECURSOR_SIGNALS: int = 3            # Discard fingerprints with fewer precursors

# Worker 2 — Cross-Instrument Matcher
CROSS_VALIDATION_MIN_SYMBOLS: int = 3     # Templates need 3+ symbols to be cross-validated
SIGNAL_WINDOW_DAYS: int = 30              # Signal window for checking template presence

# Worker 3 — Temporal Sequence Miner
MIN_SEQUENCE_LENGTH: int = 2              # Need at least 2 domain stages in a sequence
MAX_SEQUENCE_DOMAINS: int = 6             # Cap sequence complexity

# Worker 4 — Template Stacker
MIN_STACK_INSTANCES: int = 3              # Need 3+ co-occurrences to form a stack profile
STACK_WINDOW_DAYS: int = 7               # Days within which templates must both be active

# Source → domain mapping (mirrors ConvergenceAlerter / pattern_watcher)
_SOURCE_DOMAIN_MAP: dict[str, str] = {
    "sec_form4": "insider", "sec_form4_cluster": "insider",
    "openinsider": "insider", "finviz_insider": "insider",
    "house_stock": "government", "senate_stock": "government",
    "usa_spending": "contracts", "sam_gov": "contracts",
    "contract_predictor": "contracts",
    "fred_macro": "macro", "fred_yield_curve": "macro",
    "cot_positioning": "positioning",
    "ta_rsi": "technical", "ta_macd": "technical", "ta_bollinger": "technical",
    "ta_volume": "technical", "ta_candles": "technical", "ta_structure": "technical",
    "ta_market_structure": "technical",
    "sec_8k": "events", "sec_13f": "institutional", "sec_13d": "institutional",
    "edgar_13f": "institutional", "edgar_13d": "institutional",
    "finnhub_news": "events", "finnhub_earnings": "events",
    "yahoo_rss": "events",
    "stocktwits": "sentiment", "google_trends": "sentiment",
    "social_text": "sentiment",
    "finviz_unusual_volume": "technical", "finviz_short_squeeze": "technical",
    "economic_calendar": "macro", "finnhub_economic": "macro",
    "coingecko": "crypto", "coincap": "crypto",
    "massive_ohlcv": "technical", "polygon_ohlcv": "technical",
    "eia_energy": "energy", "eia_crude": "energy",
    "job_tracker": "fundamental", "price_fetcher": "technical",
    "price_action": "technical",
    "congress_legislative": "government", "usda_wasde": "fundamental",
    "vix": "volatility", "finra_short": "fundamental",
    "alpha_vantage": "technical", "yfinance": "technical",
}


def _source_to_domain(source: str) -> str:
    """Map a signal source name to its domain."""
    return _SOURCE_DOMAIN_MAP.get(source, source.split("_")[0])


# ---------------------------------------------------------------------------
# Logging helpers — set up per-process
# ---------------------------------------------------------------------------

def _setup_logging(worker_name: str) -> logging.Logger:
    """Configure a logger for a worker process."""
    log_path = LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = f"%(asctime)s %(levelname)-7s [{worker_name}] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    name = f"research_team.{worker_name}"
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(sh)

    return log


# ---------------------------------------------------------------------------
# Shared helpers — used across multiple workers
# ---------------------------------------------------------------------------

def _load_watchlist() -> list[str]:
    """Load the ticker watchlist. Returns empty list on failure."""
    if not WATCHLIST_PATH.exists():
        return []
    try:
        with open(WATCHLIST_PATH) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            tickers = data.get("tickers", [])
            if isinstance(tickers, list):
                return tickers
    except Exception:
        pass
    return []


def _load_signal_archive_for_window(
    symbol: str,
    window_start: date,
    window_end: date,
) -> list[dict]:
    """Load all archive signals for a symbol in a date window.

    Scans JSONL day files that fall within the window. Returns signals
    matching the symbol (or symbol-agnostic domains like macro/events).
    """
    signals: list[dict] = []
    if not SIGNALS_DIR.exists():
        return signals

    # Build set of YYYY-MM-DD filenames within the window
    candidate_dates: list[date] = []
    cur = window_start
    while cur <= window_end:
        candidate_dates.append(cur)
        cur += timedelta(days=1)

    symbol_agnostic_domains = {"macro", "events", "positioning", "volatility"}

    for d in candidate_dates:
        day_file = SIGNALS_DIR / f"{d.isoformat()}.jsonl"
        if not day_file.exists():
            continue
        try:
            with open(day_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rec_symbol = rec.get("symbol", "")
                    rec_domain = _source_to_domain(rec.get("source", ""))
                    if rec_symbol == symbol or rec_domain in symbol_agnostic_domains:
                        signals.append(rec)
        except OSError:
            continue

    return signals


def _load_existing_fingerprint_ids() -> set[str]:
    """Load the set of existing fingerprint IDs for deduplication."""
    ids: set[str] = set()
    if not PATTERN_LIBRARY_PATH.exists():
        return ids
    try:
        with open(PATTERN_LIBRARY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    fid = d.get("fingerprint_id", "")
                    if fid:
                        ids.add(fid)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return ids


def _load_all_templates() -> dict[str, dict]:
    """Load all templates from disk. Returns template_id -> template dict."""
    templates: dict[str, dict] = {}
    if not TEMPLATES_PATH.exists():
        return templates
    try:
        with open(TEMPLATES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    tid = t.get("template_id", "")
                    if tid:
                        templates[tid] = t
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return templates


def _append_jsonl(path: Path, record: dict) -> None:
    """Atomically append a JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _fingerprint_id(symbol: str, direction: str, move_date: str) -> str:
    raw = f"{symbol}:{direction}:{move_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _template_id(direction: str, domain_signature: str) -> str:
    raw = f"{direction}:{domain_signature}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Worker 1: Move Reverse-Engineer
# ---------------------------------------------------------------------------

def _worker1_move_reverse_engineer(dry_run: bool = False) -> None:
    """Continuously excavate historical price moves and reverse-engineer precursors.

    For each symbol in the watchlist:
    1. Fetch 2 years of daily price history
    2. Detect significant moves (single-day 5%+, 5-day 10%+, 20-day 20%+)
    3. For each move, scan the signal archive 30 days BEFORE the move
    4. Build a MoveFingerprint from those precursor signals
    5. Store fingerprint + update template

    This is the foundational excavation loop — the source of all templates.
    """
    log = _setup_logging("w1-reverse-engineer")
    log.info("Worker 1 starting — Move Reverse-Engineer")

    # Delayed import: process pool requires imports inside the worker
    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher
    except ImportError as e:
        log.error("Cannot import PriceFetcher: %s", e)
        return

    price_fetcher = PriceFetcher()

    # Load state for progress tracking
    state_file = DATA_MIDGE / "research_team_w1_state.json"
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    completed_symbols: set[str] = set(state.get("completed_symbols", []))
    cycle = int(state.get("cycle", 0))

    while True:
        watchlist = _load_watchlist()
        if not watchlist:
            log.warning("Empty watchlist — sleeping 60s")
            time.sleep(60)
            continue

        cycle += 1
        log.info("Cycle %d — %d symbols to excavate", cycle, len(watchlist))

        existing_ids = _load_existing_fingerprint_ids()
        new_fps = 0
        new_templates = 0
        symbols_processed = 0

        for symbol in watchlist:
            try:
                fps_this_symbol, tmpl_this_symbol = _excavate_symbol(
                    symbol=symbol,
                    price_fetcher=price_fetcher,
                    existing_ids=existing_ids,
                    dry_run=dry_run,
                    log=log,
                )
                new_fps += fps_this_symbol
                new_templates += tmpl_this_symbol
                existing_ids  # dedup set is mutable; already updated in place
                symbols_processed += 1

                if symbols_processed % 50 == 0:
                    log.info(
                        "Progress: %d/%d symbols — %d new fingerprints, %d new templates",
                        symbols_processed, len(watchlist), new_fps, new_templates,
                    )

            except Exception as exc:
                log.warning("Error excavating %s: %s", symbol, exc, exc_info=False)
                continue

        log.info(
            "Cycle %d complete — %d symbols, %d new fingerprints, %d new templates",
            cycle, symbols_processed, new_fps, new_templates,
        )

        # Persist state
        state = {
            "cycle": cycle,
            "last_cycle_at": datetime.now().isoformat(),
            "last_cycle_symbols": symbols_processed,
            "last_cycle_fingerprints": new_fps,
            "last_cycle_templates": new_templates,
        }
        try:
            state_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

        time.sleep(CYCLE_REST_SECONDS)


def _excavate_symbol(
    symbol: str,
    price_fetcher: Any,
    existing_ids: set[str],
    dry_run: bool,
    log: logging.Logger,
) -> tuple[int, int]:
    """Excavate a single symbol. Returns (new_fingerprints, new_templates)."""
    history = price_fetcher.get_daily_history(symbol, days=PRICE_HISTORY_DAYS)
    if len(history) < 30:
        return 0, 0

    # Find significant moves in price history
    dig_sites = _find_moves(history, symbol)
    if not dig_sites:
        return 0, 0

    new_fps = 0
    new_templates = 0

    for site in dig_sites:
        fid = _fingerprint_id(symbol, site["direction"], site["move_date"])
        if fid in existing_ids:
            continue

        # Look backward: collect signals in 30-day window before the move
        try:
            move_date = date.fromisoformat(site["move_date"])
        except ValueError:
            continue

        window_end = move_date - timedelta(days=1)
        window_start = move_date - timedelta(days=LOOKBACK_DAYS)

        signals = _load_signal_archive_for_window(symbol, window_start, window_end)

        # Build precursor signal list with lag days
        precursors: list[dict] = []
        for sig in signals:
            sig_dir = sig.get("direction", "")
            if sig_dir and sig_dir != site["direction"]:
                continue  # Skip signals in opposite direction

            received_at_str = sig.get("received_at", sig.get("timestamp", ""))
            if not received_at_str:
                continue
            try:
                received_at = date.fromisoformat(received_at_str[:10])
            except ValueError:
                continue

            lag = (move_date - received_at).days
            if lag < 0 or lag > LOOKBACK_DAYS:
                continue

            domain = _source_to_domain(sig.get("source", ""))
            precursors.append({
                "source": sig.get("source", "unknown"),
                "domain": domain,
                "direction": sig.get("direction", site["direction"]),
                "strength": float(sig.get("strength", 0.5)),
                "lag_days": lag,
                "lag_bucket": _lag_bucket(lag),
                "signal_id": sig.get("signal_id", ""),
            })

        if len(precursors) < MIN_PRECURSOR_SIGNALS:
            continue

        # Compute domain signature
        domains = sorted(set(p["domain"] for p in precursors))
        domain_sig = "+".join(domains)

        # Build lag profile
        lag_profile: dict[str, int] = {}
        for p in precursors:
            bucket = p["lag_bucket"]
            lag_profile[bucket] = lag_profile.get(bucket, 0) + 1

        fingerprint = {
            "fingerprint_id": fid,
            "symbol": symbol,
            "direction": site["direction"],
            "move_date": site["move_date"],
            "move_pct": site["move_pct"],
            "regime": "default",
            "lookback_days": LOOKBACK_DAYS,
            "precursor_signals": precursors,
            "domain_signature": domain_sig,
            "lag_profile": lag_profile,
            "created_at": datetime.now().isoformat(),
            "wins": 0,
            "losses": 0,
            "total_activations": 0,
            "last_activation": "",
        }

        if not dry_run:
            _append_jsonl(PATTERN_LIBRARY_PATH, fingerprint)
            existing_ids.add(fid)

            # Update or create template
            template_key = f"{site['direction']}:{domain_sig}"
            tid = _template_id(site["direction"], domain_sig)
            is_new = _upsert_template(tid, site["direction"], domain_sig, domains, fingerprint)
            if is_new:
                new_templates += 1

        new_fps += 1

    return new_fps, new_templates


def _find_moves(history: list, symbol: str) -> list[dict]:
    """Detect significant price moves in a price history list.

    Returns list of {symbol, direction, move_date, move_pct} dicts.
    Detects three tiers: single-day 5%+, 5-day compound 10%+, 20-day 20%+.
    """
    sites: list[dict] = []
    used_dates: set[str] = set()

    prices = [(h.timestamp[:10], h.price) for h in history if h.price > 0]
    if len(prices) < 2:
        return sites

    # Build daily return series
    returns: list[tuple[str, float, float, float]] = []  # (date, pct, open, close)
    for i in range(1, len(prices)):
        prev_date, prev_price = prices[i - 1]
        curr_date, curr_price = prices[i]
        if prev_price <= 0:
            continue
        pct = (curr_price - prev_price) / prev_price * 100
        returns.append((curr_date, pct, prev_price, curr_price))

    MIN_GAP = 5  # trading days between sites to prevent overlap

    last_site_i = -MIN_GAP

    for i, (d, pct, prev_p, curr_p) in enumerate(returns):
        if i - last_site_i < MIN_GAP:
            continue
        if d in used_dates:
            continue

        # Tier 1: single-day 5%+
        if abs(pct) >= MIN_SINGLE_DAY_PCT:
            direction = "bullish" if pct > 0 else "bearish"
            sites.append({"symbol": symbol, "direction": direction, "move_date": d, "move_pct": round(pct, 2)})
            used_dates.add(d)
            last_site_i = i
            continue

        # Tier 2: 5-day compound 10%+
        if i + 4 < len(returns):
            compound_5 = sum(returns[i + j][1] for j in range(5))
            if abs(compound_5) >= MIN_5DAY_PCT:
                direction = "bullish" if compound_5 > 0 else "bearish"
                end_date = returns[i + 4][0]
                if end_date not in used_dates:
                    sites.append({"symbol": symbol, "direction": direction, "move_date": end_date, "move_pct": round(compound_5, 2)})
                    used_dates.add(end_date)
                    last_site_i = i + 4
                continue

        # Tier 3: 20-day compound 20%+
        if i + 19 < len(returns):
            compound_20 = sum(returns[i + j][1] for j in range(20))
            if abs(compound_20) >= MIN_20DAY_PCT:
                direction = "bullish" if compound_20 > 0 else "bearish"
                end_date = returns[i + 19][0]
                if end_date not in used_dates:
                    sites.append({"symbol": symbol, "direction": direction, "move_date": end_date, "move_pct": round(compound_20, 2)})
                    used_dates.add(end_date)
                    last_site_i = i + 19

    return sites


def _lag_bucket(days: int) -> str:
    """Convert lag days to bucket name."""
    if days <= 2:
        return "immediate"
    if days <= 5:
        return "short"
    if days <= 10:
        return "medium"
    if days <= 20:
        return "long"
    return "extended"


# Template upsert — single writer per process (Worker 1 owns template writes)
_template_cache: dict[str, dict] = {}
_template_cache_loaded = False


def _upsert_template(
    tid: str,
    direction: str,
    domain_sig: str,
    domains: list[str],
    fingerprint: dict,
) -> bool:
    """Update or create a template entry in the JSONL file. Returns True if new."""
    global _template_cache, _template_cache_loaded

    if not _template_cache_loaded:
        _template_cache = _load_all_templates()
        _template_cache_loaded = True

    is_new = tid not in _template_cache

    if is_new:
        lag_raw = dict(fingerprint.get("lag_profile", {}))
        total = sum(lag_raw.values()) or 1
        lag_norm = {b: c / total for b, c in lag_raw.items()}
        template = {
            "template_id": tid,
            "direction": direction,
            "domain_signature": domain_sig,
            "domains": domains,
            "lag_profile_raw": lag_raw,
            "lag_profile_normalized": lag_norm,
            "source_examples": {},
            "instances": [{
                "symbol": fingerprint["symbol"],
                "move_date": fingerprint["move_date"],
                "move_pct": fingerprint["move_pct"],
                "fingerprint_id": fingerprint["fingerprint_id"],
                "regime": "default",
            }],
            "symbols_seen": [fingerprint["symbol"]],
            "n_instances": 1,
            "avg_move_pct": abs(fingerprint.get("move_pct", 0.0)),
            "_move_pct_sum": abs(fingerprint.get("move_pct", 0.0)),
            "wins": 0,
            "losses": 0,
            "created_at": datetime.now().isoformat(),
        }
        _template_cache[tid] = template
    else:
        t = _template_cache[tid]
        # Accumulate lag profile
        for bucket, count in fingerprint.get("lag_profile", {}).items():
            t["lag_profile_raw"][bucket] = t["lag_profile_raw"].get(bucket, 0) + count
        total = sum(t["lag_profile_raw"].values()) or 1
        t["lag_profile_normalized"] = {b: c / total for b, c in t["lag_profile_raw"].items()}
        # Add instance (capped at 200)
        inst = {
            "symbol": fingerprint["symbol"],
            "move_date": fingerprint["move_date"],
            "move_pct": fingerprint["move_pct"],
            "fingerprint_id": fingerprint["fingerprint_id"],
            "regime": "default",
        }
        instances = t.get("instances", [])
        instances.append(inst)
        if len(instances) > 200:
            instances = instances[-200:]
        t["instances"] = instances
        # Update symbols seen
        seen = t.get("symbols_seen", [])
        if fingerprint["symbol"] not in seen:
            seen.append(fingerprint["symbol"])
        t["symbols_seen"] = seen
        # Update n_instances and avg
        n = t.get("n_instances", 0) + 1
        t["n_instances"] = n
        pct_sum = t.get("_move_pct_sum", 0.0) + abs(fingerprint.get("move_pct", 0.0))
        t["_move_pct_sum"] = pct_sum
        t["avg_move_pct"] = pct_sum / n

    # Persist the full templates file (rewrite — templates are small, 44 entries)
    try:
        TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TEMPLATES_PATH, "w", encoding="utf-8") as f:
            for t in _template_cache.values():
                f.write(json.dumps(t) + "\n")
    except OSError:
        pass

    return is_new


# ---------------------------------------------------------------------------
# Worker 2: Cross-Instrument Pattern Matcher
# ---------------------------------------------------------------------------

def _worker2_cross_instrument_matcher(dry_run: bool = False) -> None:
    """Validate existing templates across instruments.

    For each template, scans signals for EVERY other symbol to see if
    the domain pattern appears there too. Writes cross-validation records
    that boost confidence when the pattern appears on 3+ symbols.

    This answers: "Is this an NVDA-specific quirk, or a real structural pattern?"
    """
    log = _setup_logging("w2-cross-instrument")
    log.info("Worker 2 starting — Cross-Instrument Pattern Matcher")

    state_file = DATA_MIDGE / "research_team_w2_state.json"
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    cycle = int(state.get("cycle", 0))

    while True:
        templates = _load_all_templates()
        watchlist = _load_watchlist()

        if not templates or not watchlist:
            log.warning("No templates or watchlist — sleeping 60s")
            time.sleep(60)
            continue

        cycle += 1
        log.info("Cycle %d — %d templates × %d symbols", cycle, len(templates), len(watchlist))

        validations_written = 0

        # Load what we've already validated to avoid redundant writes
        validated_pairs: set[str] = _load_validated_pairs()

        for tid, template in templates.items():
            direction = template.get("direction", "")
            domain_sig = template.get("domain_signature", "")
            template_domains = set(template.get("domains", []))
            symbols_already_in_template = set(template.get("symbols_seen", []))

            for symbol in watchlist:
                if symbol in symbols_already_in_template:
                    continue  # Already known — skip

                pair_key = f"{tid}:{symbol}"
                if pair_key in validated_pairs:
                    continue

                # Check if this symbol has signals matching the template domains
                # Use a 30-day rolling window from the latest available archive date
                today = date.today()
                window_end = today
                window_start = today - timedelta(days=SIGNAL_WINDOW_DAYS * 12)  # 1 year scan

                signals = _load_signal_archive_for_window(symbol, window_start, window_end)
                if not signals:
                    # Mark as checked (nothing found) to avoid re-scanning
                    validated_pairs.add(pair_key)
                    continue

                # Check if template's domains are present in this symbol's signals
                signal_domains_by_day: dict[str, set[str]] = collections.defaultdict(set)
                for sig in signals:
                    if sig.get("direction", "") != direction:
                        continue
                    d_str = sig.get("received_at", sig.get("timestamp", ""))[:10]
                    domain = _source_to_domain(sig.get("source", ""))
                    signal_domains_by_day[d_str].add(domain)

                # Slide a 30-day window to find dates where template_domains are present
                matched_dates: list[str] = []
                all_days = sorted(signal_domains_by_day.keys())
                for i, start_d in enumerate(all_days):
                    try:
                        start_dt = date.fromisoformat(start_d)
                    except ValueError:
                        continue
                    window_domains: set[str] = set()
                    for d_str in all_days[i:]:
                        try:
                            d_dt = date.fromisoformat(d_str)
                        except ValueError:
                            continue
                        if (d_dt - start_dt).days > SIGNAL_WINDOW_DAYS:
                            break
                        window_domains |= signal_domains_by_day[d_str]

                    # Match: need >= 70% of template domains present
                    if len(template_domains) == 0:
                        continue
                    overlap = len(window_domains & template_domains)
                    coverage = overlap / len(template_domains)
                    if coverage >= 0.70:
                        matched_dates.append(start_d)

                if matched_dates:
                    validation_record = {
                        "template_id": tid,
                        "direction": direction,
                        "domain_signature": domain_sig,
                        "symbol": symbol,
                        "match_count": len(matched_dates),
                        "first_match_date": matched_dates[0],
                        "last_match_date": matched_dates[-1],
                        "coverage_pct": round(
                            len(template_domains & set().union(*[signal_domains_by_day[d] for d in matched_dates[:5]])) /
                            max(len(template_domains), 1) * 100, 1
                        ),
                        "validated_at": datetime.now().isoformat(),
                    }
                    if not dry_run:
                        _append_jsonl(CROSS_INSTRUMENT_PATH, validation_record)
                    validations_written += 1
                    validated_pairs.add(pair_key)
                    log.debug("Template %s validated on %s (%d matches)", tid[:8], symbol, len(matched_dates))
                else:
                    validated_pairs.add(pair_key)

        log.info("Cycle %d complete — %d new cross-validations", cycle, validations_written)

        state = {
            "cycle": cycle,
            "last_cycle_at": datetime.now().isoformat(),
            "last_cycle_validations": validations_written,
        }
        try:
            state_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

        time.sleep(CYCLE_REST_SECONDS)


def _load_validated_pairs() -> set[str]:
    """Load already-validated template:symbol pairs to avoid redundant re-checking."""
    pairs: set[str] = set()
    if not CROSS_INSTRUMENT_PATH.exists():
        return pairs
    try:
        with open(CROSS_INSTRUMENT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    tid = rec.get("template_id", "")
                    sym = rec.get("symbol", "")
                    if tid and sym:
                        pairs.add(f"{tid}:{sym}")
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return pairs


# ---------------------------------------------------------------------------
# Worker 3: Temporal Sequence Miner
# ---------------------------------------------------------------------------

def _worker3_temporal_sequence_miner(dry_run: bool = False) -> None:
    """Mine temporal ordering from graded outcomes to build sequence templates.

    For every graded outcome, reconstructs the chronological sequence in which
    signals arrived before the prediction was made. Extracts the domain
    firing ORDER, not just the domain SET. Groups sequences with the same
    ordering into "temporal templates" — the when matters, not just the what.

    Example output:
      macro fires → day 0
      insider fires → day 3
      technical fires → day 5
      → move happened 7 days later
      Historical win rate for this sequence: 78% (n=18)
    """
    log = _setup_logging("w3-temporal-miner")
    log.info("Worker 3 starting — Temporal Sequence Miner")

    state_file = DATA_MIDGE / "research_team_w3_state.json"
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    cycle = int(state.get("cycle", 0))
    processed_ids: set[str] = set(state.get("processed_prediction_ids", []))

    # In-memory accumulator: sequence_key -> {wins, losses, examples, timing_data}
    sequence_stats: dict[str, dict] = _load_temporal_template_stats()

    while True:
        outcomes = _load_outcomes()
        if not outcomes:
            log.warning("No outcomes — sleeping 60s")
            time.sleep(60)
            continue

        new_outcomes = [o for o in outcomes if o.get("prediction_id", "") not in processed_ids]
        if not new_outcomes:
            log.info("Cycle %d — no new outcomes, sleeping", cycle + 1)
            time.sleep(CYCLE_REST_SECONDS * 2)
            continue

        cycle += 1
        log.info("Cycle %d — processing %d new outcomes", cycle, len(new_outcomes))

        new_sequences = 0
        updated_sequences = 0

        for outcome in new_outcomes:
            pid = outcome.get("prediction_id", "")
            if not pid:
                continue

            symbol = outcome.get("symbol", "")
            direction = outcome.get("direction", "")
            predicted_at_str = outcome.get("predicted_at", "")
            was_correct = bool(outcome.get("was_correct", False))

            if not (symbol and direction and predicted_at_str):
                processed_ids.add(pid)
                continue

            try:
                predicted_at = datetime.fromisoformat(predicted_at_str)
            except ValueError:
                processed_ids.add(pid)
                continue

            # Collect signals in the 30 days before prediction
            pred_date = predicted_at.date()
            window_start = pred_date - timedelta(days=LOOKBACK_DAYS)
            window_end = pred_date - timedelta(days=1)

            signals = _load_signal_archive_for_window(symbol, window_start, window_end)
            dir_signals = [
                s for s in signals
                if s.get("direction", direction) == direction
            ]

            if len(dir_signals) < 2:
                processed_ids.add(pid)
                continue

            # Extract domain firing order — first time each domain appeared
            domain_first_seen: dict[str, date] = {}
            for sig in dir_signals:
                domain = _source_to_domain(sig.get("source", ""))
                ts_str = sig.get("received_at", sig.get("timestamp", ""))[:10]
                try:
                    ts = date.fromisoformat(ts_str)
                except ValueError:
                    continue
                if domain not in domain_first_seen or ts < domain_first_seen[domain]:
                    domain_first_seen[domain] = ts

            if len(domain_first_seen) < MIN_SEQUENCE_LENGTH:
                processed_ids.add(pid)
                continue

            # Build ordered sequence: [(domain, lag_days_before_prediction)]
            ordered = sorted(domain_first_seen.items(), key=lambda x: x[1])
            # Cap at MAX_SEQUENCE_DOMAINS to prevent excessive specificity
            ordered = ordered[:MAX_SEQUENCE_DOMAINS]

            # Compute inter-domain lags
            sequence_stages: list[dict] = []
            for i, (domain, first_date) in enumerate(ordered):
                lag_to_pred = (pred_date - first_date).days
                lag_to_next = None
                if i + 1 < len(ordered):
                    next_date = ordered[i + 1][1]
                    lag_to_next = (next_date - first_date).days
                sequence_stages.append({
                    "domain": domain,
                    "lag_to_prediction": lag_to_pred,
                    "lag_to_next_domain": lag_to_next,
                    "stage": i + 1,
                })

            # Sequence key: direction + ordered domain names
            domain_order = tuple(s["domain"] for s in sequence_stages)
            seq_key = f"{direction}:{':'.join(domain_order)}"

            # Update or create temporal template
            if seq_key not in sequence_stats:
                sequence_stats[seq_key] = {
                    "sequence_key": seq_key,
                    "direction": direction,
                    "domain_order": list(domain_order),
                    "sequence_length": len(domain_order),
                    "wins": 0,
                    "losses": 0,
                    "n_instances": 0,
                    "avg_lag_days": {},
                    "lag_sum": {},
                    "examples": [],
                    "created_at": datetime.now().isoformat(),
                }
                new_sequences += 1
            else:
                updated_sequences += 1

            ss = sequence_stats[seq_key]
            if was_correct:
                ss["wins"] += 1
            else:
                ss["losses"] += 1
            ss["n_instances"] += 1

            # Accumulate lag averages
            for stage in sequence_stages:
                dom = stage["domain"]
                ss["lag_sum"][dom] = ss["lag_sum"].get(dom, 0) + stage["lag_to_prediction"]
                n = ss["n_instances"]
                ss["avg_lag_days"][dom] = ss["lag_sum"][dom] / n

            # Keep up to 10 examples
            example = {
                "symbol": symbol,
                "predicted_at": predicted_at_str[:10],
                "was_correct": was_correct,
                "stages": sequence_stages,
            }
            if len(ss.get("examples", [])) < 10:
                ss.setdefault("examples", []).append(example)

            processed_ids.add(pid)

        # Write all updated temporal templates
        if not dry_run and (new_sequences > 0 or updated_sequences > 0):
            _flush_temporal_templates(sequence_stats)

        log.info(
            "Cycle %d complete — %d new sequences, %d updated, %d total",
            cycle, new_sequences, updated_sequences, len(sequence_stats),
        )

        # Persist state (keep processed IDs bounded to last 50K)
        if len(processed_ids) > 50_000:
            # Keep the most recently added (hard to do with set — just trim)
            processed_ids = set(list(processed_ids)[-40_000:])

        state = {
            "cycle": cycle,
            "last_cycle_at": datetime.now().isoformat(),
            "processed_prediction_ids": list(processed_ids),
            "total_sequences": len(sequence_stats),
        }
        try:
            state_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

        time.sleep(CYCLE_REST_SECONDS)


def _load_outcomes() -> list[dict]:
    """Load all graded outcomes from disk."""
    outcomes: list[dict] = []
    if not OUTCOMES_PATH.exists():
        return outcomes
    try:
        with open(OUTCOMES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    outcomes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return outcomes


def _load_temporal_template_stats() -> dict[str, dict]:
    """Load existing temporal templates into the accumulator dict."""
    stats: dict[str, dict] = {}
    if not TEMPORAL_TEMPLATES_PATH.exists():
        return stats
    try:
        with open(TEMPORAL_TEMPLATES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("sequence_key", "")
                    if key:
                        stats[key] = rec
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return stats


def _flush_temporal_templates(sequence_stats: dict[str, dict]) -> None:
    """Rewrite the temporal templates file with current stats."""
    TEMPORAL_TEMPLATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(TEMPORAL_TEMPLATES_PATH, "w", encoding="utf-8") as f:
            for rec in sequence_stats.values():
                # Add computed win rate before writing
                n = rec.get("n_instances", 0)
                w = rec.get("wins", 0)
                rec["win_rate"] = round(w / n, 4) if n > 0 else 0.0
                f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Worker 4: Template Stacker
# ---------------------------------------------------------------------------

def _worker4_template_stacker(dry_run: bool = False) -> None:
    """Build stack profiles from co-occurring templates.

    Scans the signal archive to find dates and symbols where MULTIPLE
    templates were simultaneously active. Records the joint outcome.

    The power: "template X alone → 55% win rate. Template Y alone → 60%
    win rate. Templates X+Y simultaneously → 84% win rate." This is
    inevitability: the stacking of multiple independent proven patterns.

    A template is "active" for a symbol if the live signal window contains
    enough of the template's domains (>= 60% coverage).
    """
    log = _setup_logging("w4-template-stacker")
    log.info("Worker 4 starting — Template Stacker")

    state_file = DATA_MIDGE / "research_team_w4_state.json"
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}

    cycle = int(state.get("cycle", 0))

    # In-memory accumulator: stack_key -> {wins, losses, examples}
    stack_stats: dict[str, dict] = _load_stack_profile_stats()

    while True:
        templates = _load_all_templates()
        outcomes = _load_outcomes()
        watchlist = _load_watchlist()

        if not templates or not outcomes:
            log.warning("No templates or outcomes — sleeping 60s")
            time.sleep(60)
            continue

        cycle += 1
        log.info("Cycle %d — %d templates, %d outcomes, %d symbols", cycle, len(templates), len(outcomes), len(watchlist))

        # Build outcome lookup: symbol+direction -> list of (predicted_at_date, was_correct, return_pct)
        outcome_lookup: dict[str, list[tuple[date, bool, float]]] = collections.defaultdict(list)
        for out in outcomes:
            sym = out.get("symbol", "")
            dr = out.get("direction", "")
            pred_at = out.get("predicted_at", "")[:10]
            was_correct = bool(out.get("was_correct", False))
            return_pct = float(out.get("return_pct", 0.0))
            if sym and dr and pred_at:
                try:
                    outcome_lookup[f"{sym}:{dr}"].append(
                        (date.fromisoformat(pred_at), was_correct, return_pct)
                    )
                except ValueError:
                    continue

        new_stacks = 0
        updated_stacks = 0
        stack_instances_found = 0

        # For each outcome, check how many templates were simultaneously active
        for out in outcomes:
            sym = out.get("symbol", "")
            dr = out.get("direction", "")
            pred_at_str = out.get("predicted_at", "")[:10]
            was_correct = bool(out.get("was_correct", False))
            return_pct = float(out.get("return_pct", 0.0))

            if not (sym and dr and pred_at_str):
                continue
            try:
                pred_date = date.fromisoformat(pred_at_str)
            except ValueError:
                continue

            # Get signals active in the 30-day window before this prediction
            window_start = pred_date - timedelta(days=LOOKBACK_DAYS)
            window_end = pred_date - timedelta(days=1)
            signals = _load_signal_archive_for_window(sym, window_start, window_end)
            active_domains: set[str] = set()
            for sig in signals:
                if sig.get("direction", dr) == dr:
                    active_domains.add(_source_to_domain(sig.get("source", "")))

            # Check which templates are active (>= 60% domain coverage)
            active_template_ids: list[str] = []
            for tid, tmpl in templates.items():
                if tmpl.get("direction", "") != dr:
                    continue
                tmpl_domains = set(tmpl.get("domains", []))
                if not tmpl_domains:
                    continue
                coverage = len(active_domains & tmpl_domains) / len(tmpl_domains)
                if coverage >= 0.60:
                    active_template_ids.append(tid)

            if len(active_template_ids) < 2:
                continue  # No stacking — skip

            stack_instances_found += 1

            # Generate all pairs and triples from active templates
            combos = _generate_combos(active_template_ids)

            for combo in combos:
                # Stack key: sorted template IDs joined
                stack_key = f"{dr}:" + ":".join(sorted(combo))

                if stack_key not in stack_stats:
                    stack_stats[stack_key] = {
                        "stack_key": stack_key,
                        "direction": dr,
                        "template_ids": sorted(combo),
                        "stack_size": len(combo),
                        "wins": 0,
                        "losses": 0,
                        "n_instances": 0,
                        "total_return_pct": 0.0,
                        "avg_return_pct": 0.0,
                        "examples": [],
                        "created_at": datetime.now().isoformat(),
                    }
                    new_stacks += 1
                else:
                    updated_stacks += 1

                ss = stack_stats[stack_key]
                if was_correct:
                    ss["wins"] += 1
                else:
                    ss["losses"] += 1
                ss["n_instances"] += 1
                ss["total_return_pct"] = ss.get("total_return_pct", 0.0) + return_pct
                n = ss["n_instances"]
                ss["avg_return_pct"] = ss["total_return_pct"] / n

                if len(ss.get("examples", [])) < 10:
                    ss.setdefault("examples", []).append({
                        "symbol": sym,
                        "predicted_at": pred_at_str,
                        "was_correct": was_correct,
                        "return_pct": return_pct,
                    })

        # Filter: only keep stacks with MIN_STACK_INSTANCES observations
        qualifying_stacks = {k: v for k, v in stack_stats.items() if v.get("n_instances", 0) >= MIN_STACK_INSTANCES}

        if not dry_run and (new_stacks > 0 or updated_stacks > 0):
            _flush_stack_profiles(qualifying_stacks)

        log.info(
            "Cycle %d complete — %d instances found, %d new stacks, %d updated, %d qualifying stacks",
            cycle, stack_instances_found, new_stacks, updated_stacks, len(qualifying_stacks),
        )

        state = {
            "cycle": cycle,
            "last_cycle_at": datetime.now().isoformat(),
            "stack_instances_found": stack_instances_found,
            "qualifying_stacks": len(qualifying_stacks),
        }
        try:
            state_file.write_text(json.dumps(state, indent=2))
        except Exception:
            pass

        time.sleep(CYCLE_REST_SECONDS)


def _generate_combos(template_ids: list[str]) -> list[tuple[str, ...]]:
    """Generate pairs and triples from a list of template IDs.

    Returns all 2-combinations and 3-combinations (capped at reasonable size).
    """
    from itertools import combinations
    combos: list[tuple[str, ...]] = []
    ids = template_ids[:10]  # Cap at 10 to prevent combinatorial explosion
    for r in (2, 3):
        for combo in combinations(ids, r):
            combos.append(combo)
    return combos


def _load_stack_profile_stats() -> dict[str, dict]:
    """Load existing stack profiles into the accumulator dict."""
    stats: dict[str, dict] = {}
    if not STACK_PROFILES_PATH.exists():
        return stats
    try:
        with open(STACK_PROFILES_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = rec.get("stack_key", "")
                    if key:
                        stats[key] = rec
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return stats


def _flush_stack_profiles(stack_stats: dict[str, dict]) -> None:
    """Rewrite the stack profiles file with current stats."""
    STACK_PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(STACK_PROFILES_PATH, "w", encoding="utf-8") as f:
            for rec in stack_stats.values():
                n = rec.get("n_instances", 0)
                w = rec.get("wins", 0)
                rec["win_rate"] = round(w / n, 4) if n > 0 else 0.0
                f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Status reporting and state management
# ---------------------------------------------------------------------------

def _load_team_state() -> dict[str, Any]:
    """Load the master team state file."""
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {}


def _update_team_state(worker_name: str, status: str, extra: dict | None = None) -> None:
    """Update the team state file for a specific worker."""
    state = _load_team_state()
    if "workers" not in state:
        state["workers"] = {}
    state["workers"][worker_name] = {
        "status": status,
        "updated_at": datetime.now().isoformat(),
        **(extra or {}),
    }
    state["last_updated"] = datetime.now().isoformat()
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker dispatch — runs inside the worker process
# ---------------------------------------------------------------------------

def _run_worker(worker_id: int, dry_run: bool = False) -> None:
    """Entry point for each worker process.

    Each worker runs in its own OS process via ProcessPoolExecutor.
    Imports happen INSIDE the process to avoid multiprocessing pickling issues.
    """
    # Write PID
    pid_file = DATA_MIDGE / f"research_team_w{worker_id}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    if worker_id == 1:
        _worker1_move_reverse_engineer(dry_run=dry_run)
    elif worker_id == 2:
        _worker2_cross_instrument_matcher(dry_run=dry_run)
    elif worker_id == 3:
        _worker3_temporal_sequence_miner(dry_run=dry_run)
    elif worker_id == 4:
        _worker4_template_stacker(dry_run=dry_run)
    else:
        raise ValueError(f"Unknown worker_id: {worker_id}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_forever(
    workers: int = 4,
    dry_run: bool = False,
) -> None:
    """Start all research workers in parallel and supervise them.

    If a worker crashes, it is restarted after a brief pause.
    This function blocks forever.
    """
    DATA_MIDGE.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))

    log = _setup_logging("supervisor")
    log.info(
        "Historical Research Team starting — %d workers, dry_run=%s",
        workers, dry_run,
    )

    # Initial state
    _update_team_state("supervisor", "running", {"workers": workers, "dry_run": dry_run})

    active_worker_ids = list(range(1, workers + 1))

    # Use spawn context — safer for Windows multiprocessing
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")

    def _start_worker(wid: int) -> multiprocessing.Process:
        p = ctx.Process(
            target=_run_worker,
            args=(wid, dry_run),
            name=f"research-w{wid}",
            daemon=True,
        )
        p.start()
        log.info("Started worker %d (PID %d)", wid, p.pid)
        return p

    processes: dict[int, multiprocessing.Process] = {}
    for wid in active_worker_ids:
        processes[wid] = _start_worker(wid)

    # Supervisor loop — restart crashed workers
    try:
        while True:
            time.sleep(30)

            for wid, proc in list(processes.items()):
                if not proc.is_alive():
                    exit_code = proc.exitcode
                    log.warning(
                        "Worker %d exited with code %s — restarting in 10s",
                        wid, exit_code,
                    )
                    time.sleep(10)
                    processes[wid] = _start_worker(wid)

            # Log heartbeat
            alive = sum(1 for p in processes.values() if p.is_alive())
            log.info("Heartbeat — %d/%d workers alive", alive, len(processes))
            _update_team_state("supervisor", "running", {
                "workers_alive": alive,
                "workers_total": len(processes),
            })

    except KeyboardInterrupt:
        log.info("Supervisor interrupted — shutting down workers")
        for wid, proc in processes.items():
            proc.terminate()
        for wid, proc in processes.items():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
        log.info("All workers stopped")

    finally:
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MIDGE Historical Research Team — parallel pattern mining",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of research workers (1-4, default: 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing output files (for testing)",
    )
    parser.add_argument(
        "--worker", type=int, default=None,
        choices=[1, 2, 3, 4],
        help="Run a single specific worker (for debugging)",
    )
    args = parser.parse_args()

    if args.worker:
        # Single-worker debug mode — runs in this process
        print(f"Running single worker {args.worker} in foreground...", file=sys.stderr)
        _run_worker(args.worker, dry_run=args.dry_run)
    else:
        workers = max(1, min(4, args.workers))
        run_forever(workers=workers, dry_run=args.dry_run)


if __name__ == "__main__":
    # Required on Windows for spawn-based multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
