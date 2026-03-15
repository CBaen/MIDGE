#!/usr/bin/env python3
"""continuous_replay.py — Continuous historical replay process.

Runs as an INDEPENDENT OS process (not inside the daemon's event loop).
Continuously cycles through the signal archive oldest-first, replaying
each day through a lightweight convergence check with current Thompson
weights, grading outcomes, and writing results to its own output file.

The daemon keeps sensing the present. This process mines the past.

Design principles:
  - Read-only access to the signal archive
  - Loads fresh Thompson distributions at the start of each cycle
  - Writes only to data/midge/continuous_replay_results.jsonl (append)
  - Writes progress state to data/midge/continuous_replay_state.json
  - Logs to data/midge/continuous_replay.log
  - Sleeps 0.05s between days (50ms — yields CPU without stalling throughput)
  - Sleeps 60s between full archive cycles (Thompson weights may have updated)
  - Splits archive into 4 chunks processed in parallel

Usage:
    python -m mae_core.market.parallel.continuous_replay
    python -m mae_core.market.parallel.continuous_replay --min-domains 3
    python -m mae_core.market.parallel.continuous_replay --start 2026-01-01
    python -m mae_core.market.parallel.continuous_replay --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths — all relative to project root, resolved at module load time
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]          # mae_core/market/parallel/… → MIDGE root

SIGNALS_DIR        = _PROJECT_ROOT / "data" / "midge" / "signals"
THOMPSON_PATH      = _PROJECT_ROOT / "data" / "market" / "thompson_distributions.json"
RESULTS_JSONL      = _PROJECT_ROOT / "data" / "midge" / "continuous_replay_results.jsonl"
STATE_PATH         = _PROJECT_ROOT / "data" / "midge" / "continuous_replay_state.json"
LOG_PATH           = _PROJECT_ROOT / "data" / "midge" / "continuous_replay.log"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Seconds to sleep between processing each day's signals
# 50ms — enough to yield CPU without stalling throughput
DAY_SLEEP_SECONDS: float = 0.05

# Seconds to rest after completing a full archive cycle before restarting
# Short sleep: Thompson weights may have updated, so replay again quickly
CYCLE_REST_SECONDS: float = 60.0

# Number of parallel workers — 4 chunks processed simultaneously
NUM_WORKERS: int = 4

# Price-lookup rate limit (yfinance)
PRICE_LOOKUP_DELAY: float = 0.25

# Log progress every N days within each worker
PROGRESS_LOG_INTERVAL: int = 50

# Outcome window: days after alert to check price
DEFAULT_OUTCOME_WINDOW_DAYS: int = 14

# A trade is a "success" if price moved >= this % in the predicted direction
SUCCESS_THRESHOLD_PCT: float = 5.0

# ---------------------------------------------------------------------------
# Logging — file + stderr
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("midge.continuous_replay")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler — keeps a permanent record
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    # Stderr so the parent process (or systemd) can see it too
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(sh)

    return logger


logger = _setup_logging(LOG_PATH)


# ---------------------------------------------------------------------------
# SimulatedDateTime — same pattern as replay_history.py
# ---------------------------------------------------------------------------

class SimulatedDateTime:
    """Drop-in replacement for the datetime class with controllable now().

    Replaces the ``datetime`` name in convergence_alerter's module namespace
    so that all ``datetime.now()`` calls return the simulated time instead
    of wall-clock time.  Everything else delegates to the real datetime.
    """

    def __init__(self, frozen_time: datetime):
        self._frozen = frozen_time
        self._real_datetime = datetime

    def now(self, tz=None):
        if tz is not None:
            return self._frozen.replace(tzinfo=tz)
        return self._frozen

    def __call__(self, *args, **kwargs):
        return self._real_datetime(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_datetime, name)


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------

def _discover_archive_dates(signals_dir: Path = SIGNALS_DIR) -> List[date]:
    """Return all dates that have archive files, sorted oldest-first."""
    if not signals_dir.exists():
        return []
    dates = []
    for f in signals_dir.glob("*.jsonl"):
        try:
            dates.append(date.fromisoformat(f.stem))
        except ValueError:
            pass
    return sorted(dates)


def _load_day_signals(day: date, signals_dir: Path = SIGNALS_DIR) -> List[dict]:
    """Load all signals from a single day's JSONL archive file."""
    path = signals_dir / f"{day.isoformat()}.jsonl"
    if not path.exists():
        return []
    signals = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return signals


def _parse_timestamp(ts_str: str) -> datetime:
    if not ts_str:
        return datetime.min
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.min


def _group_into_windows(signals: List[dict], window_hours: float = 4.0) -> List[List[dict]]:
    """Group pre-sorted signals into time windows of *window_hours* hours."""
    if not signals:
        return []
    windows: List[List[dict]] = []
    current: List[dict] = []
    window_start = _parse_timestamp(signals[0].get("timestamp", ""))
    delta = timedelta(hours=window_hours)
    for sig in signals:
        ts = _parse_timestamp(sig.get("timestamp", ""))
        if ts - window_start >= delta and current:
            windows.append(current)
            current = [sig]
            window_start = ts
        else:
            current.append(sig)
    if current:
        windows.append(current)
    return windows


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

def _extract_symbols(alert) -> List[str]:
    symbols = []
    for sig in alert.signals:
        meta = sig.metadata or {}
        symbol = meta.get("symbol", meta.get("ticker", ""))
        if symbol and symbol not in ("", "N/A", "UNKNOWN"):
            symbols.append(symbol)
    return symbols


def _primary_symbol(alert) -> Optional[str]:
    symbols = _extract_symbols(alert)
    if not symbols:
        return None
    return Counter(symbols).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# State persistence — tracks which day the process last completed
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception as exc:
        logger.warning("Could not save state: %s", exc)


# ---------------------------------------------------------------------------
# Results writer — append-only JSONL
# ---------------------------------------------------------------------------

def _append_result(record: dict) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(RESULTS_JSONL, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("Could not write result record: %s", exc)


def _append_results_bulk(records: List[dict]) -> None:
    """Write a batch of records to the shared results file in one open."""
    if not records:
        return
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(RESULTS_JSONL, "a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("Could not write bulk result records: %s", exc)


# ---------------------------------------------------------------------------
# Price grading
# ---------------------------------------------------------------------------

_price_cache: Dict[tuple, Optional[float]] = {}


def _get_price(symbol: str, dt: date, price_fetcher) -> Optional[float]:
    key = (symbol, dt.isoformat())
    if key not in _price_cache:
        try:
            pd_obj = price_fetcher.get_historical_price(symbol, dt.isoformat())
            _price_cache[key] = pd_obj.price if pd_obj else None
        except Exception:
            _price_cache[key] = None
        time.sleep(PRICE_LOOKUP_DELAY)
    return _price_cache[key]


def _grade_alert(
    alert_dict: dict,
    price_fetcher,
    outcome_window_days: int = DEFAULT_OUTCOME_WINDOW_DAYS,
) -> dict:
    """Add grade/price fields to an alert dict. Returns the dict (mutated in place)."""
    symbol = alert_dict.get("primary_symbol")
    if not symbol:
        alert_dict["grade"] = "ungraded"
        alert_dict["grade_reason"] = "no_symbol"
        return alert_dict

    try:
        alert_dt = datetime.fromisoformat(alert_dict["timestamp"]).date()
    except (ValueError, KeyError):
        alert_dict["grade"] = "ungraded"
        alert_dict["grade_reason"] = "bad_timestamp"
        return alert_dict

    # Cannot grade if the outcome window hasn't closed yet
    outcome_dt = alert_dt + timedelta(days=outcome_window_days)
    if outcome_dt > date.today():
        alert_dict["grade"] = "pending"
        alert_dict["grade_reason"] = "outcome_window_not_closed"
        return alert_dict

    entry_price = _get_price(symbol, alert_dt, price_fetcher)
    exit_price  = _get_price(symbol, outcome_dt, price_fetcher)

    if entry_price is None or exit_price is None:
        alert_dict["grade"] = "ungraded"
        alert_dict["grade_reason"] = "no_price_data"
        return alert_dict

    pct_change = ((exit_price - entry_price) / entry_price) * 100
    alert_dict["entry_price"]          = round(entry_price, 4)
    alert_dict["exit_price"]           = round(exit_price, 4)
    alert_dict["pct_change"]           = round(pct_change, 2)
    alert_dict["outcome_window_days"]  = outcome_window_days

    direction = alert_dict.get("direction", "neutral")
    if direction == "bullish" and pct_change >= SUCCESS_THRESHOLD_PCT:
        alert_dict["grade"] = "success"
    elif direction == "bearish" and pct_change <= -SUCCESS_THRESHOLD_PCT:
        alert_dict["grade"] = "success"
    else:
        alert_dict["grade"] = "failure"

    return alert_dict


# ---------------------------------------------------------------------------
# Core: replay a single day
# ---------------------------------------------------------------------------

def _replay_day(
    day: date,
    alerter,
    ca_module,
    original_datetime,
    use_thompson: bool,
) -> List[dict]:
    """Feed one day's archive through the alerter. Returns discovered alert dicts."""
    signals = _load_day_signals(day)
    if not signals:
        return []

    signals.sort(key=lambda s: s.get("timestamp", ""))
    windows = _group_into_windows(signals, window_hours=4.0)
    discovered: List[dict] = []

    for window in windows:
        if not window:
            continue

        timestamps = [_parse_timestamp(s.get("timestamp", "")) for s in window]
        valid_ts   = [t for t in timestamps if t != datetime.min]
        if not valid_ts:
            continue
        window_time = max(valid_ts)

        ca_module.datetime = SimulatedDateTime(window_time)

        for sig in window:
            ts = _parse_timestamp(sig.get("timestamp", ""))
            if ts == datetime.min:
                continue
            metadata = dict(sig.get("metadata", {}) or {})
            if sig.get("symbol") and "symbol" not in metadata:
                metadata["symbol"] = sig["symbol"]

            alerter.record_signal(
                signal_id  = sig.get("signal_id", ""),
                strength   = float(sig.get("strength", 0.0)),
                domain     = sig.get("domain", ""),
                direction  = sig.get("direction", "neutral"),
                confidence = float(sig.get("confidence", 0.5)),
                velocity   = float(sig.get("velocity", 0.0)),
                timestamp  = ts,
                metadata   = metadata,
                source     = sig.get("source", ""),
            )

        alerts = alerter.check_convergence()
        for alert in alerts:
            ad = alert.to_dict()
            ad["replay_date"]   = day.isoformat()
            ad["primary_symbol"] = _primary_symbol(alert)
            ad["all_symbols"]   = list(set(_extract_symbols(alert)))
            ad["source"]        = "continuous_replay"
            ad["replay_ts"]     = datetime.utcnow().isoformat()
            discovered.append(ad)

    # Clear dedup so each day is independent for alert suppression
    alerter._last_alert_times.clear()

    return discovered


# ---------------------------------------------------------------------------
# Build a fresh alerter + import ca_module (called once per cycle)
# ---------------------------------------------------------------------------

def _build_alerter(min_domains: int, use_thompson: bool):
    """Import convergence_alerter and build a fresh instance."""
    import mae_core.market.intelligence.convergence_alerter as ca_module
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    thompson = None
    if use_thompson:
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        thompson = ThompsonSampler()

    alerter = ConvergenceAlerter(
        min_domains=min_domains,
        thompson_sampler=thompson,
    )
    return alerter, ca_module


# ---------------------------------------------------------------------------
# Chunk helper
# ---------------------------------------------------------------------------

def _split_into_chunks(items: list, n: int) -> List[list]:
    """Split *items* into *n* roughly equal chunks."""
    if not items:
        return [[] for _ in range(n)]
    chunk_size = math.ceil(len(items) / n)
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


# ---------------------------------------------------------------------------
# Worker function — runs in a subprocess, processes one chunk of days
# ---------------------------------------------------------------------------

def _worker_process_chunk(
    worker_id: int,
    days_iso: List[str],
    min_domains: int,
    use_thompson: bool,
    outcome_window_days: int,
    dry_run: bool,
    tmp_output_path: str,
    signals_dir_str: str,
    project_root_str: str,
) -> dict:
    """
    Top-level function executed in a subprocess.

    Must be a module-level function (not a lambda or nested def) so that
    Python's multiprocessing can handle it on Windows (spawn start method).

    Each worker gets its own ThompsonSampler + ConvergenceAlerter instance.
    Results are written to a per-worker temp file; the parent merges them.
    Returns a stats dict summarising what the worker found.
    """
    signals_dir = Path(signals_dir_str)

    prefix = f"Replay worker {worker_id}"

    def _wlog(msg: str) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{ts} INFO    [{prefix}] {msg}", file=sys.stderr, flush=True)

    _wlog(f"starting -- {len(days_iso)} days to process")

    try:
        alerter, ca_module = _build_alerter(min_domains, use_thompson)
    except Exception as exc:
        _wlog(f"FATAL: could not build alerter: {exc}")
        return {"worker_id": worker_id, "error": str(exc), "alerts_found": 0,
                "days_processed": 0, "total_signals": 0, "alerts_graded": 0,
                "alerts_pending": 0, "alerts_ungraded": 0, "successes": 0, "failures": 0}

    original_datetime = ca_module.datetime

    price_fetcher = None
    if not dry_run:
        try:
            from mae_core.market.apis.price_fetcher import PriceFetcher
            price_fetcher = PriceFetcher()
        except Exception as exc:
            _wlog(f"PriceFetcher unavailable -- alerts will be ungraded: {exc}")

    stats: dict = {
        "worker_id":      worker_id,
        "days_processed": 0,
        "total_signals":  0,
        "alerts_found":   0,
        "alerts_graded":  0,
        "alerts_pending": 0,
        "alerts_ungraded": 0,
        "successes":      0,
        "failures":       0,
    }

    tmp_records: List[dict] = []

    try:
        for i, day_iso in enumerate(days_iso):
            day = date.fromisoformat(day_iso)
            day_signals = _load_day_signals(day, signals_dir)
            stats["total_signals"] += len(day_signals)

            alerts = _replay_day(day, alerter, ca_module, original_datetime, use_thompson)

            for ad in alerts:
                if price_fetcher and not dry_run:
                    _grade_alert(ad, price_fetcher, outcome_window_days)
                else:
                    ad["grade"] = "dry_run" if dry_run else "ungraded"

                grade = ad.get("grade", "ungraded")
                if grade == "success":
                    stats["successes"] += 1
                    stats["alerts_graded"] += 1
                elif grade == "failure":
                    stats["failures"] += 1
                    stats["alerts_graded"] += 1
                elif grade == "pending":
                    stats["alerts_pending"] += 1
                else:
                    stats["alerts_ungraded"] += 1

                stats["alerts_found"] += 1
                tmp_records.append(ad)

            stats["days_processed"] += 1

            # Progress report every PROGRESS_LOG_INTERVAL days
            if (i + 1) % PROGRESS_LOG_INTERVAL == 0:
                graded = stats["alerts_graded"]
                _wlog(
                    f"{i + 1}/{len(days_iso)} days complete, "
                    f"{stats['alerts_found']} alerts found, "
                    f"{graded} graded"
                )

            time.sleep(DAY_SLEEP_SECONDS)

    finally:
        ca_module.datetime = original_datetime

    # Write all records to the worker's own temp file (no cross-process locking needed)
    if not dry_run and tmp_records:
        try:
            Path(tmp_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_output_path, "w", encoding="utf-8") as fh:
                for rec in tmp_records:
                    fh.write(json.dumps(rec, default=str) + "\n")
        except Exception as exc:
            _wlog(f"WARNING: could not write temp output: {exc}")

    _wlog(
        f"done -- {stats['days_processed']} days, "
        f"{stats['alerts_found']} alerts, "
        f"{stats['alerts_graded']} graded"
    )
    return stats


# ---------------------------------------------------------------------------
# One full archive cycle — parallel edition
# ---------------------------------------------------------------------------

def _run_cycle(
    cycle_num: int,
    min_domains: int,
    use_thompson: bool,
    start_date: Optional[date],
    outcome_window_days: int,
    dry_run: bool,
) -> dict:
    """Run a complete pass over the archive. Returns summary stats."""
    logger.info("=" * 60)
    logger.info("Cycle %d starting", cycle_num)

    all_dates = _discover_archive_dates()
    if not all_dates:
        logger.error("No signal archives found at %s", SIGNALS_DIR)
        return {"error": "no_archives"}

    # Filter by optional start_date
    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]

    logger.info(
        "Archive: %d days (%s → %s)",
        len(all_dates), all_dates[0], all_dates[-1]
    )

    # Load state to resume from last processed day
    state = _load_state()
    last_completed = state.get("last_completed_day")

    if last_completed:
        try:
            resume_after = date.fromisoformat(last_completed)
            remaining = [d for d in all_dates if d > resume_after]
            if not remaining:
                logger.info("All days already processed this cycle — resetting to oldest")
                remaining = all_dates
            else:
                logger.info("Resuming from %s (%d days remaining)", resume_after, len(remaining))
            dates_to_process = remaining
        except ValueError:
            dates_to_process = all_dates
    else:
        dates_to_process = all_dates

    # Build alerter (loads current Thompson weights from disk)
    alerter, ca_module = _build_alerter(min_domains, use_thompson)
    original_datetime  = ca_module.datetime

    # Load price fetcher once per cycle
    price_fetcher = None
    if not dry_run:
        try:
            from mae_core.market.apis.price_fetcher import PriceFetcher
            price_fetcher = PriceFetcher()
        except Exception as exc:
            logger.warning("PriceFetcher unavailable — alerts will be ungraded: %s", exc)

    stats: dict = {
        "cycle": cycle_num,
        "started_at": datetime.utcnow().isoformat(),
        "days_processed": 0,
        "total_signals":  0,
        "alerts_found":   0,
        "alerts_graded":  0,
        "alerts_pending": 0,
        "alerts_ungraded": 0,
        "successes":      0,
        "failures":       0,
    }

    try:
        for i, day in enumerate(dates_to_process):
            day_signals = _load_day_signals(day)
            n_signals   = len(day_signals)
            stats["total_signals"] += n_signals

            alerts = _replay_day(day, alerter, ca_module, original_datetime, use_thompson)

            for ad in alerts:
                # Attempt grading
                if price_fetcher and not dry_run:
                    _grade_alert(ad, price_fetcher, outcome_window_days)
                else:
                    ad["grade"] = "dry_run" if dry_run else "ungraded"

                grade = ad.get("grade", "ungraded")
                if grade == "success":
                    stats["successes"] += 1
                    stats["alerts_graded"] += 1
                elif grade == "failure":
                    stats["failures"] += 1
                    stats["alerts_graded"] += 1
                elif grade == "pending":
                    stats["alerts_pending"] += 1
                else:
                    stats["alerts_ungraded"] += 1

                stats["alerts_found"] += 1

                if not dry_run:
                    _append_result(ad)

                logger.info(
                    "  ALERT %s %s | conf=%.2f str=%.2f | %s | grade=%s | sym=%s",
                    day.isoformat(),
                    ad.get("direction", "?").upper(),
                    ad.get("confidence", 0),
                    ad.get("strength", 0),
                    "+".join(ad.get("domains", [])),
                    ad.get("grade", "?"),
                    ad.get("primary_symbol", "?"),
                )

            stats["days_processed"] += 1

            # Save progress after each day
            _save_state({
                "last_completed_day": day.isoformat(),
                "cycle": cycle_num,
                "updated_at": datetime.utcnow().isoformat(),
            })

            if (i + 1) % 30 == 0:
                graded_total = stats["alerts_graded"]
                win_rate = (
                    stats["successes"] / graded_total * 100
                    if graded_total > 0 else 0.0
                )
                logger.info(
                    "  Progress: %d/%d days | %d alerts | WR=%.1f%% (%d/%d graded)",
                    i + 1, len(dates_to_process),
                    stats["alerts_found"], win_rate,
                    stats["successes"], graded_total,
                )

            time.sleep(DAY_SLEEP_SECONDS)

    finally:
        # ALWAYS restore the real datetime
        ca_module.datetime = original_datetime

    stats["finished_at"] = datetime.utcnow().isoformat()

    graded_total = stats["alerts_graded"]
    win_rate = stats["successes"] / graded_total * 100 if graded_total > 0 else 0.0

    logger.info("=" * 60)
    logger.info(
        "Cycle %d complete: %d days, %d signals, %d alerts",
        cycle_num, stats["days_processed"],
        stats["total_signals"], stats["alerts_found"],
    )
    logger.info(
        "  Graded: %d | Wins: %d | Losses: %d | WR: %.1f%%",
        graded_total, stats["successes"], stats["failures"], win_rate,
    )
    logger.info(
        "  Pending: %d | Ungraded: %d",
        stats["alerts_pending"], stats["alerts_ungraded"],
    )
    logger.info("=" * 60)

    # Append cycle summary to results as a meta-record
    if not dry_run:
        _append_result({"type": "cycle_summary", **stats})

    return stats


# ---------------------------------------------------------------------------
# Entry point — infinite loop
# ---------------------------------------------------------------------------

def run_forever(
    min_domains: int = 3,
    use_thompson: bool = True,
    start_date: Optional[date] = None,
    outcome_window_days: int = DEFAULT_OUTCOME_WINDOW_DAYS,
    dry_run: bool = False,
) -> None:
    """Main loop — runs forever, cycling through the archive."""
    logger.info("=" * 60)
    logger.info("MIDGE Continuous Replay — starting")
    logger.info("  min_domains=%d | thompson=%s | outcome_window=%dd",
                min_domains, use_thompson, outcome_window_days)
    logger.info("  results → %s", RESULTS_JSONL)
    logger.info("  state   → %s", STATE_PATH)
    logger.info("  log     → %s", LOG_PATH)
    if dry_run:
        logger.info("  DRY RUN — results will not be written")
    logger.info("=" * 60)

    cycle = 0
    while True:
        cycle += 1
        try:
            _run_cycle(
                cycle_num=cycle,
                min_domains=min_domains,
                use_thompson=use_thompson,
                start_date=start_date,
                outcome_window_days=outcome_window_days,
                dry_run=dry_run,
            )
        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down after cycle %d", cycle)
            break
        except Exception as exc:
            logger.exception("Cycle %d failed with unhandled error: %s — sleeping 60s", cycle, exc)
            time.sleep(60.0)
            continue

        logger.info(
            "Cycle %d done. Resting %.0f seconds before next cycle.",
            cycle, CYCLE_REST_SECONDS,
        )
        time.sleep(CYCLE_REST_SECONDS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIDGE continuous historical replay — runs forever",
    )
    parser.add_argument(
        "--min-domains", type=int, default=3,
        help="Minimum domains required for convergence alert (default: 3)",
    )
    parser.add_argument(
        "--no-thompson", action="store_true",
        help="Disable Thompson weighting (use raw convergence confidence)",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Only replay days on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--outcome-window", type=int, default=DEFAULT_OUTCOME_WINDOW_DAYS,
        help="Days after alert to check price for grading (default: %d)" % DEFAULT_OUTCOME_WINDOW_DAYS,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing results — useful for testing",
    )
    args = parser.parse_args()

    start_date = None
    if args.start:
        try:
            start_date = date.fromisoformat(args.start)
        except ValueError:
            logger.error("Invalid --start date: %s (expected YYYY-MM-DD)", args.start)
            sys.exit(1)

    run_forever(
        min_domains=args.min_domains,
        use_thompson=not args.no_thompson,
        start_date=start_date,
        outcome_window_days=args.outcome_window,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
