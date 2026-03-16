#!/usr/bin/env python3
"""continuous_grader.py — Continuous outcome grading process.

Runs as an INDEPENDENT OS process (not inside the daemon's event loop).
Reads predictions.jsonl, finds mature ungraded predictions, batch-fetches
prices via yfinance, grades each one, then appends to outcomes.jsonl and
writes a bridge file the daemon can read.

Why this exists:
  MIDGE had 53,860 predictions and only 22 graded outcomes (0.04%).
  The Thompson Sampler cannot learn from outcomes it doesn't know about.
  This process exists purely to drain that backlog and keep it drained.

Grading logic (mirrors OutcomeTracker for consistency):
  - Entry price: closing price on signal_date  (yfinance historical)
  - Exit  price: closing price on outcome_date (yfinance historical)
  - WIN  if abs(pct_change) >= 5% AND direction matches (or direction == "")
  - LOSS otherwise

Priority ordering:
  1. High-confidence predictions first (Thompson learns most from these)
  2. Oldest predictions first within same confidence tier

Deduplication:
  - Loads all existing signal_ids from outcomes.jsonl at startup.
  - Skips any prediction whose signal_id is already there.
  - Also skips if outcome_window_days has NOT yet elapsed (window still open).

Bridge file (data/market/grader_bridge.json):
  Written atomically after every cycle. The daemon reads this to surface
  grading activity in its heartbeat logs. Format:
    {
      "last_run_at": "ISO",
      "cycle": N,
      "graded_this_cycle": N,
      "graded_total": N,
      "pending_count": N,
      "win_rate_lifetime": 0.0–1.0,
      "wins_total": N,
      "losses_total": N,
      "price_fetch_failures": N
    }

Usage:
    python -m mae_core.market.parallel.continuous_grader
    python -m mae_core.market.parallel.continuous_grader --once
    python -m mae_core.market.parallel.continuous_grader --interval 120
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[4]  # .../mae_core/market/parallel/ → project root

PREDICTIONS_PATH  = _PROJECT_ROOT / "data" / "market" / "predictions.jsonl"
OUTCOMES_PATH     = _PROJECT_ROOT / "data" / "market" / "outcomes.jsonl"
BRIDGE_PATH       = _PROJECT_ROOT / "data" / "market" / "grader_bridge.json"
LOG_PATH          = _PROJECT_ROOT / "data" / "midge" / "continuous_grader.log"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_SECONDS: int = 300    # 5 minutes; price data doesn't refresh faster
BATCH_SIZE: int = 20                   # yfinance.download handles up to ~20 tickers well
MIN_PRICE_MOVE_PCT: float = 5.0        # 5% = win (looser than OutcomeTracker's 2% because
                                       # convergence alerts are higher-conviction signals)
MAX_PREDICTIONS_PER_CYCLE: int = 500   # Throttle per cycle to avoid rate limits

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("continuous_grader")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(sh)
    return lg


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_graded_ids(outcomes_path: Path) -> Set[str]:
    """Return set of signal_ids already in outcomes.jsonl."""
    graded: Set[str] = set()
    if not outcomes_path.exists():
        return graded
    with open(outcomes_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            sid = rec.get("signal_id") or rec.get("prediction_id", "")
            if sid:
                graded.add(sid)
    return graded


def _load_pending_predictions(
    predictions_path: Path,
    graded_ids: Set[str],
    logger: logging.Logger,
) -> List[dict]:
    """
    Return mature, ungraded predictions sorted by priority:
      1. Highest confidence first (confidence field in metadata or root).
      2. Oldest timestamp first within equal confidence.
    """
    if not predictions_path.exists():
        return []

    now = datetime.now()
    pending: List[dict] = []

    with open(predictions_path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Normalise signal_id (old records may use "prediction_id")
            sid = rec.get("signal_id") or rec.get("prediction_id", "")
            if not sid:
                continue

            # Skip already graded
            if sid in graded_ids:
                continue

            # Parse timestamp
            ts_str = rec.get("timestamp") or rec.get("predicted_at", "")
            if not ts_str:
                continue
            try:
                ts = datetime.fromisoformat(ts_str)
            except ValueError:
                logger.debug("Unparseable timestamp: %s", ts_str)
                continue

            # Skip if window has not yet elapsed
            window_days = int(rec.get("outcome_window_days", 14))
            outcome_date = ts + timedelta(days=window_days)
            if outcome_date > now:
                continue

            # Attach derived fields for sorting
            confidence = _extract_confidence(rec)
            rec["_ts"] = ts
            rec["_outcome_date"] = outcome_date
            rec["_confidence"] = confidence
            rec["_sid"] = sid
            pending.append(rec)

    # Priority: highest confidence first, then oldest first
    pending.sort(key=lambda r: (-r["_confidence"], r["_ts"]))
    return pending


def _extract_confidence(rec: dict) -> float:
    """Extract confidence from wherever it might live in the record."""
    # Direct field
    if "confidence" in rec:
        try:
            return float(rec["confidence"])
        except (TypeError, ValueError):
            pass
    # Inside metadata
    metadata = rec.get("metadata") or {}
    if isinstance(metadata, dict):
        if "confidence" in metadata:
            try:
                return float(metadata["confidence"])
            except (TypeError, ValueError):
                pass
    return 0.0


# ---------------------------------------------------------------------------
# Price fetching — yfinance batch API
# ---------------------------------------------------------------------------


def _fetch_price_batch(
    ticker_date_pairs: List[Tuple[str, datetime, datetime]],
    logger: logging.Logger,
) -> Dict[Tuple[str, str], float]:
    """
    Batch-fetch closing prices for (ticker, date) pairs.

    Args:
        ticker_date_pairs: list of (ticker, entry_date, exit_date)

    Returns:
        Dict keyed by (ticker, "YYYY-MM-DD") → closing price float.
        Missing entries mean price was unavailable.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — cannot grade. Run: pip install yfinance")
        return {}

    # Collect unique (ticker, date_str) pairs so we don't double-fetch
    needed: Dict[str, Set[str]] = {}  # ticker → set of date strings
    for ticker, entry_dt, exit_dt in ticker_date_pairs:
        t = ticker.upper()
        if t not in needed:
            needed[t] = set()
        # We need the day AFTER the date so yfinance includes that day in the range
        needed[t].add(entry_dt.strftime("%Y-%m-%d"))
        needed[t].add(exit_dt.strftime("%Y-%m-%d"))

    results: Dict[Tuple[str, str], float] = {}
    tickers_list = sorted(needed.keys())

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, len(tickers_list), BATCH_SIZE):
        batch = tickers_list[batch_start : batch_start + BATCH_SIZE]
        # Find the widest date range we need across this batch
        all_dates: List[str] = []
        for t in batch:
            all_dates.extend(needed[t])
        if not all_dates:
            continue
        all_dates_dt = [datetime.strptime(d, "%Y-%m-%d") for d in all_dates]
        start_str = (min(all_dates_dt) - timedelta(days=5)).strftime("%Y-%m-%d")
        end_str   = (max(all_dates_dt) + timedelta(days=5)).strftime("%Y-%m-%d")

        try:
            raw = yf.download(
                tickers=" ".join(batch),
                start=start_str,
                end=end_str,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as e:
            logger.warning("yfinance batch download failed for %s: %s", batch, e)
            continue

        if raw is None or raw.empty:
            continue

        # yfinance returns multi-level columns for multi-ticker downloads:
        # ("Close", "AAPL"), ("Close", "MSFT") etc.
        # For single ticker it returns flat columns: "Close", "Open" etc.
        import pandas as pd
        for ticker in batch:
            try:
                if len(batch) == 1:
                    close_series = raw.get("Close")
                else:
                    # Multi-level columns
                    close_series = raw.get(("Close", ticker))
                if close_series is None or close_series.empty:
                    continue

                for date_str in needed[ticker]:
                    target = datetime.strptime(date_str, "%Y-%m-%d")
                    # Find nearest available trading day (forward then backward)
                    price = _nearest_close(close_series, target)
                    if price is not None:
                        results[(ticker, date_str)] = price
            except Exception as e:
                logger.debug("Price extraction failed for %s: %s", ticker, e)

    return results


def _nearest_close(series, target: datetime, max_offset_days: int = 5) -> Optional[float]:
    """
    Find the closing price nearest to `target` within `max_offset_days`.
    Tries forward first (next trading day), then backward.
    """
    import pandas as pd
    for offset in range(0, max_offset_days + 1):
        for sign in (1, -1):
            if offset == 0 and sign == -1:
                continue
            candidate = target + timedelta(days=offset * sign)
            candidate_ts = pd.Timestamp(candidate.date())
            if candidate_ts in series.index:
                val = series.loc[candidate_ts]
                if val is not None and not (hasattr(val, '__float__') and __import__('math').isnan(float(val))):
                    return float(val)
    return None


# ---------------------------------------------------------------------------
# Grading logic
# ---------------------------------------------------------------------------


def _grade_prediction(
    pred: dict,
    price_map: Dict[Tuple[str, str], float],
    logger: logging.Logger,
) -> Optional[dict]:
    """
    Grade a single prediction using pre-fetched prices.

    Returns an outcome dict ready to append to outcomes.jsonl,
    or None if prices were unavailable.
    """
    ticker = (pred.get("outcome_symbol") or pred.get("symbol", "")).upper()
    if not ticker:
        return None

    ts: datetime = pred["_ts"]
    outcome_date: datetime = pred["_outcome_date"]
    entry_date_str = ts.strftime("%Y-%m-%d")
    exit_date_str  = outcome_date.strftime("%Y-%m-%d")

    entry_price = price_map.get((ticker, entry_date_str))
    exit_price  = price_map.get((ticker, exit_date_str))

    if entry_price is None or exit_price is None or entry_price == 0:
        logger.debug("Missing price for %s  entry=%s exit=%s", ticker, entry_date_str, exit_date_str)
        return None

    pct_change = ((exit_price - entry_price) / entry_price) * 100.0
    direction  = pred.get("direction", "")
    magnitude_ok = abs(pct_change) >= MIN_PRICE_MOVE_PCT

    if direction == "up":
        direction_ok = pct_change > 0
    elif direction == "down":
        direction_ok = pct_change < 0
    else:
        direction_ok = True  # direction-agnostic prediction

    success = magnitude_ok and direction_ok

    return {
        "signal_id":          pred["_sid"],
        "source":             pred.get("source", "unknown"),
        "symbol":             ticker,
        "direction":          direction,
        "predicted_at":       pred.get("timestamp") or pred.get("predicted_at", ""),
        "evaluated_at":       datetime.now().isoformat(),
        "window_days":        int(pred.get("outcome_window_days", 14)),
        "entry_price":        round(entry_price, 4),
        "exit_price":         round(exit_price, 4),
        "price_change_pct":   round(pct_change, 4),
        "success":            success,
        "confidence":         pred.get("_confidence", 0.0),
        "min_move_threshold": MIN_PRICE_MOVE_PCT,
        "grader":             "continuous_grader",  # distinguish from OutcomeTracker grades
    }


# ---------------------------------------------------------------------------
# Atomic file writes
# ---------------------------------------------------------------------------


def _append_outcomes(outcomes: List[dict], outcomes_path: Path, logger: logging.Logger) -> None:
    """Append graded outcomes to outcomes.jsonl."""
    if not outcomes:
        return
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(outcomes_path, "a", encoding="utf-8") as f:
            for outcome in outcomes:
                f.write(json.dumps(outcome) + "\n")
    except Exception as e:
        logger.error("Failed to write outcomes: %s", e)


def _write_bridge(stats: dict, bridge_path: Path, logger: logging.Logger) -> None:
    """Write bridge file atomically via temp file + rename."""
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(stats, indent=2)
    try:
        tmp = bridge_path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(bridge_path)
    except Exception as e:
        logger.warning("Bridge write failed: %s", e)


# ---------------------------------------------------------------------------
# Single grading cycle
# ---------------------------------------------------------------------------


def run_cycle(
    graded_ids: Set[str],
    cycle_stats: dict,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """
    Run one grading cycle.

    Returns:
        (graded_this_cycle, price_fetch_failures)
    """
    pending = _load_pending_predictions(PREDICTIONS_PATH, graded_ids, logger)
    total_pending = len(pending)

    if total_pending == 0:
        logger.info("No mature ungraded predictions found.")
        return 0, 0

    # Throttle: only process top MAX_PREDICTIONS_PER_CYCLE per cycle
    batch = pending[:MAX_PREDICTIONS_PER_CYCLE]
    logger.info(
        "Found %d mature ungraded predictions — processing %d this cycle.",
        total_pending, len(batch),
    )

    # Collect (ticker, entry_date, exit_date) for batch price fetch
    ticker_date_pairs: List[Tuple[str, datetime, datetime]] = []
    for pred in batch:
        ticker = (pred.get("outcome_symbol") or pred.get("symbol", "")).upper()
        if ticker:
            ticker_date_pairs.append((ticker, pred["_ts"], pred["_outcome_date"]))

    price_map = _fetch_price_batch(ticker_date_pairs, logger)
    logger.info("Price map populated: %d entries.", len(price_map))

    graded_outcomes: List[dict] = []
    failures = 0

    for pred in batch:
        outcome = _grade_prediction(pred, price_map, logger)
        if outcome is None:
            failures += 1
            continue
        graded_outcomes.append(outcome)
        graded_ids.add(pred["_sid"])

    _append_outcomes(graded_outcomes, OUTCOMES_PATH, logger)

    wins   = sum(1 for o in graded_outcomes if o["success"])
    losses = sum(1 for o in graded_outcomes if not o["success"])
    cycle_stats["wins_total"]   += wins
    cycle_stats["losses_total"] += losses

    logger.info(
        "Cycle complete: graded=%d  wins=%d  losses=%d  price_failures=%d",
        len(graded_outcomes), wins, losses, failures,
    )
    return len(graded_outcomes), failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous outcome grader — grades mature predictions every N seconds."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle then exit (useful for testing / manual backfill)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"Seconds between grading cycles (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    args = parser.parse_args()

    logger = _setup_logging(LOG_PATH)
    logger.info(
        "Continuous grader starting — interval=%ds, once=%s, min_move=%.1f%%",
        args.interval, args.once, MIN_PRICE_MOVE_PCT,
    )

    # Load already-graded IDs once at startup; updated in-memory each cycle
    graded_ids: Set[str] = _load_graded_ids(OUTCOMES_PATH)
    logger.info("Loaded %d already-graded outcome IDs.", len(graded_ids))

    cycle_stats: dict = {
        "cycle":              0,
        "graded_total":       0,
        "wins_total":         0,
        "losses_total":       0,
    }

    cycle = 0
    while True:
        cycle += 1
        cycle_stats["cycle"] = cycle
        logger.info("=== Grading cycle %d ===", cycle)

        graded_this_cycle = 0
        failures_this_cycle = 0
        try:
            graded_this_cycle, failures_this_cycle = run_cycle(graded_ids, cycle_stats, logger)
            cycle_stats["graded_total"] += graded_this_cycle
        except Exception:
            logger.exception("Cycle %d raised an unhandled exception", cycle)

        # Count remaining pending to report in bridge
        try:
            pending_count = len(
                _load_pending_predictions(PREDICTIONS_PATH, graded_ids, logger)
            )
        except Exception:
            pending_count = -1

        total_graded = cycle_stats["wins_total"] + cycle_stats["losses_total"]
        win_rate = (
            cycle_stats["wins_total"] / total_graded if total_graded > 0 else 0.0
        )

        bridge = {
            "last_run_at":          datetime.now().isoformat(),
            "cycle":                cycle,
            "graded_this_cycle":    graded_this_cycle,
            "graded_total":         cycle_stats["graded_total"],
            "pending_count":        pending_count,
            "win_rate_lifetime":    round(win_rate, 4),
            "wins_total":           cycle_stats["wins_total"],
            "losses_total":         cycle_stats["losses_total"],
            "price_fetch_failures": failures_this_cycle,
        }
        _write_bridge(bridge, BRIDGE_PATH, logger)

        if args.once:
            logger.info("--once flag set — exiting after first cycle.")
            break

        logger.info("Sleeping %d seconds until next cycle...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
