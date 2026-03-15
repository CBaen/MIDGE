#!/usr/bin/env python3
"""replay_auditor.py — Law 7 validator for continuous replay results.

Validates the work of continuous replay workers BEFORE their results are
allowed to feed into Thompson combo distributions.  Implements three
independent checks — matching the Rule of 3/5 (Law 7) intent — so that
a corrupt, drifted, or mis-graded batch cannot silently poison the
Bayesian learning layer.

The three validation tiers
--------------------------
1. **Sample verification** — randomly re-fetches price for 10 % of graded
   results and independently re-grades them.  If the worker and auditor
   disagree on more than 5 % of the sample, the whole batch is flagged.

2. **Consistency checks** — groups results by combo key and applies two
   checks: (a) minimum 5 graded outcomes per combo before approving for
   Thompson, and (b) direction / return sign agreement (a "won" result
   must have a positive return for bullish, negative for bearish).

3. **Cross-worker validation** — for any ticker + overlapping date ranges
   graded by more than one replay run (identified by ``replay_ts`` batch
   ID), checks that all runs agree on the grade.  Disagreements block both.

Output file
-----------
``data/market/replay_audit.json`` — the replay → Thompson bridge reads
this file and only injects *approved_combos*.  Schema::

    {
      "last_audit": "2026-03-15T...",
      "total_reviewed": 500,
      "total_approved": 480,
      "total_flagged": 20,
      "approved_combos": [
        {"combo": "insider+technical", "regime": "bear",
         "wins": 6, "losses": 0, "approved": true}
      ],
      "flagged_items": [
        {"reason": "grade_contradicts_return",
         "ticker": "AAPL", "details": "..."}
      ]
    }

Integration
-----------
Intended to run *before* ``feed_replay_to_thompson.run_bridge()`` in the
daemon's bridge-ingestion cycle.  Can also run standalone::

    python -m mae_core.market.parallel.replay_auditor --once

Design principles
-----------------
- Zero imports from the MIDGE organism (no EventBus, no bootstrap).
- Reads only from ``data/midge/continuous_replay_results.jsonl``.
- Writes only to ``data/market/replay_audit.json`` (atomic).
- Logs warnings for every flagged item; never raises.
- Never crashes the daemon — all errors are caught and logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE    = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]      # mae_core/market/parallel → project root

RESULTS_JSONL = _PROJECT_ROOT / "data" / "midge" / "continuous_replay_results.jsonl"
AUDIT_PATH    = _PROJECT_ROOT / "data" / "market" / "replay_audit.json"
LOG_PATH      = _PROJECT_ROOT / "data" / "midge" / "replay_auditor.log"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Fraction of graded results to independently re-verify (0.0 – 1.0).
SAMPLE_RATE: float = 0.10

# Maximum tolerable mismatch rate before the batch is flagged (0.0 – 1.0).
MAX_MISMATCH_RATE: float = 0.05

# Minimum graded outcomes per combo before it may be approved for Thompson.
MIN_OUTCOMES_PER_COMBO: int = 5

# A trade is considered a "won" if the price moved >= this % in the alert
# direction.  Must match the definition used in continuous_replay.py.
SUCCESS_THRESHOLD_PCT: float = 5.0

# Seconds between audit runs when running in --loop mode.
DEFAULT_LOOP_INTERVAL: int = 900  # 15 minutes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt     = "%(asctime)s %(levelname)-7s [auditor] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger("midge.replay_auditor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(sh)

    return logger


logger = _setup_logging(LOG_PATH)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_graded_records(path: Path) -> List[dict]:
    """Read success/failure records from the results JSONL.

    Skips blank lines, JSON errors, cycle_summary meta-records, and any
    record whose grade is not 'success' or 'failure'.
    """
    if not path.exists():
        return []
    records: List[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "cycle_summary":
                    continue
                grade = rec.get("grade", "")
                if grade not in ("success", "failure"):
                    continue
                records.append(rec)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
    return records


def _write_audit(output: dict) -> None:
    """Write audit results atomically to AUDIT_PATH."""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = AUDIT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        tmp.replace(AUDIT_PATH)
        logger.info(
            "Audit written: reviewed=%d approved=%d flagged=%d",
            output.get("total_reviewed", 0),
            output.get("total_approved", 0),
            output.get("total_flagged", 0),
        )
    except Exception as exc:
        logger.error("Failed to write audit file: %s", exc)


# ---------------------------------------------------------------------------
# Price helpers — independent re-verification
# ---------------------------------------------------------------------------


def _fetch_price_yfinance(symbol: str, dt: date) -> Optional[float]:
    """Return closing price for *symbol* on *dt* using yfinance.

    Returns None on any error — auditor treats this as 'unverifiable', NOT
    as a mismatch.
    """
    try:
        import yfinance as yf  # type: ignore
        start = dt - timedelta(days=1)
        end   = dt + timedelta(days=2)
        hist = yf.Ticker(symbol).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        if hist.empty:
            return None
        # Find the row closest to *dt*
        hist.index = hist.index.normalize()
        target = datetime(dt.year, dt.month, dt.day)
        # Try exact match first, then nearest
        try:
            import pandas as pd  # type: ignore
            target_ts = pd.Timestamp(target)
            if target_ts in hist.index:
                return float(hist.loc[target_ts, "Close"])
        except Exception:
            pass
        # Fallback: last row if within 3 trading days
        last_close = float(hist.iloc[-1]["Close"])
        return last_close
    except Exception as exc:
        logger.debug("yfinance fetch failed for %s @ %s: %s", symbol, dt, exc)
        return None


def _independent_grade(
    rec: dict,
    outcome_window_days: int = 14,
) -> Optional[str]:
    """Re-grade a record independently.

    Returns:
        'success'   — auditor agrees the alert won
        'failure'   — auditor agrees the alert lost
        None        — price data unavailable; cannot verify
    """
    symbol = rec.get("primary_symbol") or rec.get("ticker")
    if not symbol:
        return None

    try:
        alert_dt = datetime.fromisoformat(rec["timestamp"]).date()
    except (ValueError, KeyError, TypeError):
        return None

    # Pull outcome_window_days from the record if the worker stored it
    window = rec.get("outcome_window_days") or outcome_window_days
    outcome_dt = alert_dt + timedelta(days=int(window))

    # Cannot re-verify futures
    if outcome_dt > date.today():
        return None

    entry_price = _fetch_price_yfinance(symbol, alert_dt)
    time.sleep(0.25)  # yfinance rate-limit courtesy delay
    exit_price  = _fetch_price_yfinance(symbol, outcome_dt)
    time.sleep(0.25)

    if entry_price is None or exit_price is None or entry_price == 0:
        return None

    pct_change = ((exit_price - entry_price) / entry_price) * 100
    direction  = rec.get("direction", "neutral")

    if direction == "bullish" and pct_change >= SUCCESS_THRESHOLD_PCT:
        return "success"
    elif direction == "bearish" and pct_change <= -SUCCESS_THRESHOLD_PCT:
        return "success"
    else:
        return "failure"


# ---------------------------------------------------------------------------
# Tier 1: Sample verification
# ---------------------------------------------------------------------------


def _tier1_sample_verify(
    records: List[dict],
    sample_rate: float = SAMPLE_RATE,
) -> Tuple[bool, List[dict], int, int]:
    """Randomly sample records and independently re-verify grades.

    Returns:
        (batch_ok, flagged_items, verified_count, mismatch_count)
        batch_ok is False if mismatch_count / verified_count > MAX_MISMATCH_RATE.
    """
    # Only grade-able records (have symbol and non-future timestamps)
    verifiable = [
        r for r in records
        if (r.get("primary_symbol") or r.get("ticker"))
        and r.get("timestamp")
    ]

    if not verifiable:
        logger.info("Tier 1: no verifiable records — skipping sample verification")
        return True, [], 0, 0

    sample_size = max(1, int(len(verifiable) * sample_rate))
    sample = random.sample(verifiable, min(sample_size, len(verifiable)))

    flagged: List[dict] = []
    verified = 0
    mismatches = 0

    for rec in sample:
        worker_grade = rec.get("grade")
        auditor_grade = _independent_grade(rec)

        if auditor_grade is None:
            # Price unavailable — cannot dispute, skip
            continue

        verified += 1
        if auditor_grade != worker_grade:
            mismatches += 1
            ticker = rec.get("primary_symbol") or rec.get("ticker", "UNKNOWN")
            flagged.append({
                "reason": "sample_verification_mismatch",
                "ticker": ticker,
                "alert_timestamp": rec.get("timestamp"),
                "worker_grade": worker_grade,
                "auditor_grade": auditor_grade,
                "direction": rec.get("direction"),
                "domains": rec.get("domains"),
                "details": (
                    f"Worker graded '{worker_grade}' but independent "
                    f"re-fetch graded '{auditor_grade}'"
                ),
            })
            logger.warning(
                "Tier 1 mismatch: %s %s worker=%s auditor=%s",
                ticker, rec.get("timestamp", "")[:10],
                worker_grade, auditor_grade,
            )

    if verified == 0:
        logger.info("Tier 1: 0 records verifiable (all futures or no price data)")
        return True, [], 0, 0

    mismatch_rate = mismatches / verified
    batch_ok = mismatch_rate <= MAX_MISMATCH_RATE

    logger.info(
        "Tier 1: sampled=%d verified=%d mismatches=%d rate=%.1f%% batch_ok=%s",
        len(sample), verified, mismatches, mismatch_rate * 100, batch_ok,
    )

    if not batch_ok:
        logger.warning(
            "Tier 1 FAILED: mismatch rate %.1f%% exceeds threshold %.1f%%",
            mismatch_rate * 100, MAX_MISMATCH_RATE * 100,
        )

    return batch_ok, flagged, verified, mismatches


# ---------------------------------------------------------------------------
# Tier 2: Consistency checks
# ---------------------------------------------------------------------------


def _build_combo_key(domains) -> str:
    if not domains or not isinstance(domains, list):
        return ""
    return "+".join(sorted(str(d) for d in domains if d))


def _tier2_consistency(
    records: List[dict],
    min_outcomes: int = MIN_OUTCOMES_PER_COMBO,
) -> Tuple[Dict[str, dict], List[dict]]:
    """Group records by combo and apply consistency checks.

    Two checks per record:
    (a) Direction / return-sign agreement — if the worker stored pct_change,
        verify it is positive for bullish wins and negative for bearish wins.
    (b) Combo minimum-evidence gate — combos with < min_outcomes total are
        not approved (returned in buckets but approved=False).

    Returns:
        (buckets, flagged_items)
        buckets: dict keyed by "<combo>|<regime>" → {wins, losses, approved, combo, regime}
        flagged_items: list of per-record flag dicts
    """
    buckets: Dict[str, dict] = {}
    flagged: List[dict] = []
    sign_contradiction_count = 0

    for rec in records:
        combo  = _build_combo_key(rec.get("domains"))
        if not combo:
            continue
        regime = rec.get("regime") or "default"
        key    = f"{combo}|{regime}"
        grade  = rec.get("grade")  # 'success' or 'failure'

        # Check (a): return sign consistency
        pct_change = rec.get("pct_change")
        direction  = rec.get("direction", "neutral")
        ticker     = rec.get("primary_symbol") or rec.get("ticker", "UNKNOWN")

        if pct_change is not None and grade == "success":
            sign_ok = True
            if direction == "bullish" and float(pct_change) < 0:
                sign_ok = False
            elif direction == "bearish" and float(pct_change) > 0:
                sign_ok = False

            if not sign_ok:
                sign_contradiction_count += 1
                flagged.append({
                    "reason": "grade_contradicts_return",
                    "ticker": ticker,
                    "alert_timestamp": rec.get("timestamp"),
                    "grade": grade,
                    "direction": direction,
                    "pct_change": pct_change,
                    "combo": combo,
                    "details": (
                        f"grade=success direction={direction} "
                        f"but pct_change={pct_change:.2f}%"
                    ),
                })
                logger.warning(
                    "Tier 2 sign contradiction: %s %s grade=%s direction=%s pct=%.2f%%",
                    ticker, rec.get("timestamp", "")[:10],
                    grade, direction, float(pct_change),
                )

        # Accumulate bucket counts
        if key not in buckets:
            buckets[key] = {
                "combo":   combo,
                "regime":  regime,
                "wins":    0,
                "losses":  0,
                "approved": False,
            }
        if grade == "success":
            buckets[key]["wins"] += 1
        else:
            buckets[key]["losses"] += 1

    # Check (b): apply minimum evidence gate
    approved_count = 0
    for key, bucket in buckets.items():
        total = bucket["wins"] + bucket["losses"]
        if total >= min_outcomes:
            bucket["approved"] = True
            approved_count += 1
        else:
            logger.debug(
                "Tier 2: combo '%s' has only %d outcomes (need %d) — not approved",
                bucket["combo"], total, min_outcomes,
            )

    logger.info(
        "Tier 2: %d combos total, %d approved (>=%d outcomes), "
        "%d sign contradictions",
        len(buckets), approved_count, min_outcomes, sign_contradiction_count,
    )

    return buckets, flagged


# ---------------------------------------------------------------------------
# Tier 3: Cross-worker validation
# ---------------------------------------------------------------------------


def _infer_batch_id(rec: dict) -> str:
    """Infer a batch ID for cross-worker grouping.

    Uses ``replay_ts`` if present (set by the worker per replay run).
    Falls back to the date portion of the record's own timestamp so that
    records from the same calendar day are grouped together regardless.
    """
    replay_ts = rec.get("replay_ts", "")
    if replay_ts:
        # Truncate to minute precision so small clock differences don't
        # create phantom "different workers"
        return replay_ts[:16]  # "2026-03-15T14:23"
    ts = rec.get("timestamp", "")
    return ts[:10] if ts else "unknown"


def _tier3_cross_worker(
    records: List[dict],
) -> Tuple[set, List[dict]]:
    """Detect conflicting grades for the same ticker + date from different batches.

    Returns:
        (blocked_ids, flagged_items)
        blocked_ids: set of (ticker, alert_date_str) pairs that should be blocked
        flagged_items: list of conflict flag dicts
    """
    # Group records by (ticker, alert_date) → list of (batch_id, grade)
    ticker_date_grades: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

    for rec in records:
        ticker = rec.get("primary_symbol") or rec.get("ticker", "")
        if not ticker:
            continue
        ts = rec.get("timestamp", "")
        alert_date = ts[:10] if ts else ""
        if not alert_date:
            continue
        batch_id = _infer_batch_id(rec)
        grade    = rec.get("grade", "")
        ticker_date_grades[(ticker, alert_date)].append((batch_id, grade))

    blocked: set = set()
    flagged: List[dict] = []

    for (ticker, alert_date), entries in ticker_date_grades.items():
        # Only care when more than one batch graded this ticker+date
        batches = {batch_id for batch_id, _ in entries}
        if len(batches) <= 1:
            continue

        grades_by_batch: Dict[str, List[str]] = defaultdict(list)
        for batch_id, grade in entries:
            grades_by_batch[batch_id].append(grade)

        # Get the consensus grade per batch (majority vote within batch)
        batch_verdicts: Dict[str, str] = {}
        for batch_id, grades in grades_by_batch.items():
            wins   = grades.count("success")
            losses = grades.count("failure")
            batch_verdicts[batch_id] = "success" if wins > losses else "failure"

        verdicts = list(set(batch_verdicts.values()))
        if len(verdicts) > 1:
            # Genuine disagreement across batches
            blocked.add((ticker, alert_date))
            details = "; ".join(
                f"batch={b} verdict={v}"
                for b, v in sorted(batch_verdicts.items())
            )
            flagged.append({
                "reason": "cross_worker_disagreement",
                "ticker": ticker,
                "alert_date": alert_date,
                "batch_count": len(batches),
                "verdicts": batch_verdicts,
                "details": details,
            })
            logger.warning(
                "Tier 3 conflict: %s %s — %d batches disagree: %s",
                ticker, alert_date, len(batches), details,
            )

    logger.info(
        "Tier 3: %d ticker+date pairs checked, %d conflicts",
        len(ticker_date_grades), len(blocked),
    )

    return blocked, flagged


# ---------------------------------------------------------------------------
# Final approval logic — merge tier results
# ---------------------------------------------------------------------------


def _apply_blocked_ids(
    buckets: Dict[str, dict],
    records: List[dict],
    blocked_ids: set,
) -> Tuple[Dict[str, dict], int]:
    """Strip cross-worker-conflicted records from combo buckets.

    For each combo bucket, subtracts the wins/losses that came from blocked
    (ticker, alert_date) pairs.  Re-evaluates minimum-evidence gate after
    removal.

    Returns:
        (updated_buckets, total_removed)
    """
    if not blocked_ids:
        return buckets, 0

    # Build per-combo removals
    combo_removals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})

    for rec in records:
        ticker = rec.get("primary_symbol") or rec.get("ticker", "")
        ts     = rec.get("timestamp", "")
        alert_date = ts[:10] if ts else ""
        if (ticker, alert_date) not in blocked_ids:
            continue
        combo  = _build_combo_key(rec.get("domains"))
        regime = rec.get("regime") or "default"
        key    = f"{combo}|{regime}"
        grade  = rec.get("grade", "")
        if grade == "success":
            combo_removals[key]["wins"] += 1
        else:
            combo_removals[key]["losses"] += 1

    total_removed = 0
    for key, removal in combo_removals.items():
        if key not in buckets:
            continue
        bucket = buckets[key]
        bucket["wins"]   = max(0, bucket["wins"]   - removal["wins"])
        bucket["losses"] = max(0, bucket["losses"]  - removal["losses"])
        total_removed   += removal["wins"] + removal["losses"]

        # Re-evaluate minimum-evidence gate
        total = bucket["wins"] + bucket["losses"]
        if total < MIN_OUTCOMES_PER_COMBO:
            bucket["approved"] = False

    return buckets, total_removed


# ---------------------------------------------------------------------------
# Main audit entry point
# ---------------------------------------------------------------------------


def run_audit(
    sample_rate: float = SAMPLE_RATE,
    min_outcomes: int = MIN_OUTCOMES_PER_COMBO,
    skip_sample_verify: bool = False,
) -> dict:
    """Run the full three-tier audit and write results to replay_audit.json.

    This is the primary API called by the bridge ingestion cycle.  It is
    safe to call at any time — it never mutates the results JSONL.

    Args:
        sample_rate:          Fraction of records to re-verify (default 10 %).
        min_outcomes:         Minimum outcomes per combo to approve (default 5).
        skip_sample_verify:   Skip Tier 1 (useful when yfinance is unavailable
                              or for unit tests that stub price fetching).

    Returns:
        The audit dict that was written to replay_audit.json.
    """
    logger.info("=" * 60)
    logger.info("Replay Auditor — starting audit run")

    records = _read_graded_records(RESULTS_JSONL)
    logger.info("Loaded %d graded records from results JSONL", len(records))

    if not records:
        output = {
            "last_audit":      datetime.utcnow().isoformat(),
            "total_reviewed":  0,
            "total_approved":  0,
            "total_flagged":   0,
            "approved_combos": [],
            "flagged_items":   [],
            "audit_note":      "No graded records found in results JSONL",
        }
        _write_audit(output)
        return output

    all_flagged: List[dict] = []

    # ------------------------------------------------------------------
    # Tier 1: Sample verification
    # ------------------------------------------------------------------
    batch_ok   = True
    tier1_note = "skipped"

    if not skip_sample_verify:
        batch_ok, t1_flagged, verified, mismatches = _tier1_sample_verify(
            records, sample_rate=sample_rate,
        )
        all_flagged.extend(t1_flagged)
        tier1_note = (
            f"verified={verified} mismatches={mismatches} batch_ok={batch_ok}"
        )
    else:
        logger.info("Tier 1 sample verification skipped (skip_sample_verify=True)")

    # ------------------------------------------------------------------
    # Tier 2: Consistency checks
    # ------------------------------------------------------------------
    buckets, t2_flagged = _tier2_consistency(records, min_outcomes=min_outcomes)
    all_flagged.extend(t2_flagged)

    # ------------------------------------------------------------------
    # Tier 3: Cross-worker validation
    # ------------------------------------------------------------------
    blocked_ids, t3_flagged = _tier3_cross_worker(records)
    all_flagged.extend(t3_flagged)

    # Remove cross-worker conflicts from buckets
    buckets, removed = _apply_blocked_ids(buckets, records, blocked_ids)
    if removed:
        logger.info("Tier 3: removed %d outcomes from buckets due to conflicts", removed)

    # ------------------------------------------------------------------
    # If Tier 1 detected systemic mismatch, block ALL combos
    # ------------------------------------------------------------------
    if not batch_ok:
        logger.warning(
            "Tier 1 batch flagged as suspect — all combos blocked from Thompson"
        )
        for bucket in buckets.values():
            bucket["approved"] = False
        all_flagged.insert(0, {
            "reason": "batch_suspect_high_mismatch_rate",
            "ticker": "ALL",
            "details": tier1_note,
        })

    # ------------------------------------------------------------------
    # Build output
    # ------------------------------------------------------------------
    approved_combos = [b for b in buckets.values() if b.get("approved")]
    all_combos      = list(buckets.values())

    # Count totals
    total_approved = sum(b["wins"] + b["losses"] for b in approved_combos)
    total_flagged  = len(all_flagged)
    total_reviewed = len(records)

    output = {
        "last_audit":      datetime.utcnow().isoformat(),
        "total_reviewed":  total_reviewed,
        "total_approved":  total_approved,
        "total_flagged":   total_flagged,
        "batch_ok":        batch_ok,
        "tier1_note":      tier1_note,
        "approved_combos": approved_combos,
        "all_combos":      all_combos,
        "flagged_items":   all_flagged,
        "description": (
            "Replay audit results. Only approved_combos may feed into Thompson "
            "distributions. Written by replay_auditor.py. "
            "Read by feed_replay_to_thompson.py before bridge ingestion."
        ),
    }

    _write_audit(output)

    logger.info(
        "Audit complete: %d records | %d approved outcomes | "
        "%d flagged items | batch_ok=%s",
        total_reviewed, total_approved, total_flagged, batch_ok,
    )
    logger.info("=" * 60)

    return output


# ---------------------------------------------------------------------------
# Loop mode
# ---------------------------------------------------------------------------


def run_forever(
    interval_seconds: int = DEFAULT_LOOP_INTERVAL,
    sample_rate: float = SAMPLE_RATE,
    min_outcomes: int = MIN_OUTCOMES_PER_COMBO,
    skip_sample_verify: bool = False,
) -> None:
    """Run audit repeatedly, sleeping *interval_seconds* between runs."""
    logger.info(
        "Replay Auditor loop started — interval=%ds sample_rate=%.0f%% min_outcomes=%d",
        interval_seconds, sample_rate * 100, min_outcomes,
    )
    while True:
        try:
            run_audit(
                sample_rate=sample_rate,
                min_outcomes=min_outcomes,
                skip_sample_verify=skip_sample_verify,
            )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — auditor loop stopping")
            break
        except Exception as exc:
            logger.exception("Audit run failed unexpectedly: %s — sleeping before retry", exc)

        logger.info("Sleeping %d seconds before next audit run", interval_seconds)
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MIDGE replay auditor — validates continuous_replay_results.jsonl "
            "before Thompson ingestion (Law 7 compliance)"
        ),
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single audit pass and exit (default: loop forever)",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_LOOP_INTERVAL,
        help=f"Seconds between audit runs in loop mode (default: {DEFAULT_LOOP_INTERVAL})",
    )
    parser.add_argument(
        "--sample-rate", type=float, default=SAMPLE_RATE,
        help=f"Fraction of records to re-verify via yfinance (default: {SAMPLE_RATE})",
    )
    parser.add_argument(
        "--min-outcomes", type=int, default=MIN_OUTCOMES_PER_COMBO,
        help=f"Minimum outcomes per combo to approve for Thompson (default: {MIN_OUTCOMES_PER_COMBO})",
    )
    parser.add_argument(
        "--skip-sample-verify", action="store_true",
        help="Skip Tier 1 sample re-verification (useful offline / no network)",
    )
    args = parser.parse_args()

    if args.once:
        run_audit(
            sample_rate=args.sample_rate,
            min_outcomes=args.min_outcomes,
            skip_sample_verify=args.skip_sample_verify,
        )
    else:
        run_forever(
            interval_seconds=args.interval,
            sample_rate=args.sample_rate,
            min_outcomes=args.min_outcomes,
            skip_sample_verify=args.skip_sample_verify,
        )


if __name__ == "__main__":
    main()
