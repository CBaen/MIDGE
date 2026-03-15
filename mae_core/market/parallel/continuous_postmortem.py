#!/usr/bin/env python3
"""continuous_postmortem.py — Continuous post-mortem analysis process.

Runs as an INDEPENDENT OS process (not inside the daemon's event loop).
Reads ALL predictions and ALL outcomes, analyzes every graded result,
and writes a human-readable summary to data/midge/postmortem_continuous.json.
Then sleeps 300 seconds and reruns with any new outcomes.

What it answers:
  - Which domain combos win most often, and in which regime?
  - Which individual signal sources have the highest win rates?
  - How often does the move happen inside the predicted window?
  - How many predictions were directionally correct but outside the window?
  - What are the top 5 best and worst domain combos?

Design principles:
  - Read-only access to predictions.jsonl and outcomes.jsonl
  - Joins predictions to outcomes by prediction_id
  - Writes only to data/midge/postmortem_continuous.json (atomic overwrite)
  - Logs to data/midge/continuous_postmortem.log
  - Human-readable JSON output — Guiding Light can open it directly
  - No daemon imports, no bootstrap, no EventBus

Usage:
    python -m mae_core.market.parallel.continuous_postmortem
    python -m mae_core.market.parallel.continuous_postmortem --once
    python -m mae_core.market.parallel.continuous_postmortem --interval 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths — all resolved relative to project root
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[4]   # .../mae_core/market/parallel/ → project root

PREDICTIONS_PATH = _PROJECT_ROOT / "data" / "market" / "predictions.jsonl"
OUTCOMES_PATH    = _PROJECT_ROOT / "data" / "market" / "outcomes.jsonl"
OUTPUT_PATH      = _PROJECT_ROOT / "data" / "midge" / "postmortem_continuous.json"
LOG_PATH         = _PROJECT_ROOT / "data" / "midge" / "continuous_postmortem.log"

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_SECONDS: int = 300        # 5 minutes between analysis runs
MIN_SAMPLES_FOR_STATS:    int = 3          # Minimum outcomes to report a combo
TOP_N_COMBOS:             int = 5          # How many best/worst to surface
LOOKBACK_DAYS:            int = 365        # How far back to load outcomes (all-time = 0)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("continuous_postmortem")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path, lookback_days: int = 0) -> List[dict]:
    """Load all valid JSON lines from a JSONL file.

    If lookback_days > 0, filters to records where predicted_at or
    outcome_at falls within the lookback window.
    """
    if not path.exists():
        return []

    cutoff: Optional[datetime] = None
    if lookback_days > 0:
        cutoff = datetime.now() - timedelta(days=lookback_days)

    records: List[dict] = []
    seen_ids: set = set()

    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            # Dedup by prediction_id + outcome_at (avoids replay duplicates)
            pid = rec.get("prediction_id") or rec.get("signal_id") or ""
            oat = rec.get("outcome_at", "")
            dedup_key = f"{pid}:{oat}"
            if dedup_key in seen_ids:
                continue
            seen_ids.add(dedup_key)

            if cutoff is not None:
                ts_str = oat or rec.get("predicted_at", "")
                if ts_str:
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts < cutoff:
                            continue
                    except ValueError:
                        pass

            records.append(rec)

    return records


def _build_prediction_index(predictions: List[dict]) -> Dict[str, dict]:
    """Build a lookup dict from prediction_id (or signal_id) to the prediction record."""
    index: Dict[str, dict] = {}
    for pred in predictions:
        pid = pred.get("prediction_id") or pred.get("signal_id")
        if pid:
            # Later records overwrite earlier ones — keep the most recent version
            index[pid] = pred
    return index


def _merge_with_predictions(
    outcomes: List[dict],
    pred_index: Dict[str, dict],
) -> List[dict]:
    """Merge each outcome with its source prediction to recover missing fields.

    Predictions often carry fields that outcomes don't: contributing_signals,
    metadata.domain_sequence, source, outcome_window_days, confidence.
    We merge prediction fields into outcome but let outcome values win.
    """
    merged: List[dict] = []
    for outcome in outcomes:
        pid = outcome.get("prediction_id") or outcome.get("signal_id")
        if pid and pid in pred_index:
            base = dict(pred_index[pid])
            base.update(outcome)   # outcome fields win
            merged.append(base)
        else:
            merged.append(outcome)
    return merged


# ---------------------------------------------------------------------------
# Graded record filtering
# ---------------------------------------------------------------------------

def _is_graded(rec: dict) -> bool:
    """A record is graded if was_correct is a boolean (not None)."""
    return isinstance(rec.get("was_correct"), bool)


# ---------------------------------------------------------------------------
# Key extraction helpers
# ---------------------------------------------------------------------------

_SOURCE_TO_DOMAIN: Dict[str, str] = {
    # Insider
    "sec_form4": "insider",
    "insider_form4": "insider",
    "openinsider": "insider",
    "insider_cluster": "insider",
    "finviz_insider": "insider",
    # Congressional / government
    "congressional": "government",
    "congress_trade": "government",
    "politician_buy": "government",
    "politician_sell": "government",
    # Macro / economic
    "fred_yield": "macro",
    "economic_calendar": "macro",
    "finnhub_economic": "macro",
    "cot_positioning": "macro",
    "eia_energy": "energy",
    # Technical
    "technical_macd": "technical",
    "technical_rsi": "technical",
    "technical_bb": "technical",
    "rsi": "technical",
    "macd": "technical",
    "bollinger": "technical",
    "stochastic": "technical",
    "williams_r": "technical",
    "cci": "technical",
    "ta_structure": "technical",
    "ta_candle": "technical",
    "session_sweep": "technical",
    "session_sweep_ifvg": "technical",
    "ifvg": "technical",
    # Fundamental / events
    "sec_form8k": "events",
    "finnhub_earnings": "events",
    "contract_award": "contracts",
    "usa_spending": "contracts",
    # Sentiment / social
    "stocktwits": "sentiment",
    "finnhub_news": "sentiment",
    "google_trends": "sentiment",
    "yahoo_rss": "sentiment",
    # Institutional
    "edgar_13f": "institutional",
    "edgar_13d": "institutional",
    "finviz_short": "institutional",
    "finra_short": "institutional",
    # Price
    "yfinance_price": "price",
    "price_fetcher": "price",
    "polygon": "price",
    # Positioning
    "finnhub_analyst": "fundamental",
    "analyst": "fundamental",
    # Crypto
    "coingecko": "crypto",
    "coincap": "crypto",
    # Pattern
    "pattern_stack": "pattern",
    "cascade": "causal",
    # Misc
    "finra_short": "institutional",
}


def _signals_to_domains(signals: list) -> List[str]:
    """Map a list of signal source names to their domain names, deduped, sorted."""
    domains: set = set()
    for sig in signals:
        sig_str = str(sig).lower()
        # Handle pattern_stack:hash+hash style
        if sig_str.startswith("pattern_stack:"):
            domains.add("pattern")
            continue
        domain = _SOURCE_TO_DOMAIN.get(sig_str)
        if domain:
            domains.add(domain)
        else:
            # Use the signal name itself as a domain if unmapped
            domains.add(sig_str)
    return sorted(domains)


def _extract_combo_key(rec: dict) -> str:
    """Extract a human-readable domain combo key from an outcome/prediction record.

    Priority order:
    1. source field starting with 'combo:' (convergence alert combos)
    2. contributing_signals list mapped through _SOURCE_TO_DOMAIN
    3. source field as a single domain
    """
    source = rec.get("source", "")

    if source.startswith("combo:"):
        # Re-sort domains to normalize key
        raw_domains = source[6:].split("+")
        return "+".join(sorted(d.strip() for d in raw_domains if d.strip()))

    signals = rec.get("contributing_signals") or []
    if signals:
        domains = _signals_to_domains(signals)
        if domains:
            return "+".join(domains)

    if source.startswith("pattern_stack:"):
        return "pattern"

    if source:
        return _SOURCE_TO_DOMAIN.get(source, source)

    return "unknown"


def _extract_regime(rec: dict) -> str:
    """Extract regime from metadata or direct field."""
    metadata = rec.get("metadata") or {}
    if isinstance(metadata, dict):
        regime = metadata.get("regime_at_prediction") or metadata.get("regime", "")
        if regime:
            return regime
    return rec.get("regime_at_prediction") or rec.get("regime", "unknown")


def _extract_source_names(rec: dict) -> List[str]:
    """Extract individual signal source names from a record."""
    signals = rec.get("contributing_signals") or []
    if signals:
        return [str(s) for s in signals if s]
    source = rec.get("source", "")
    if source:
        return [source]
    return []


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _win_rate_label(wr: float) -> str:
    if wr >= 0.70:
        return "STRONG"
    if wr >= 0.55:
        return "GOOD"
    if wr >= 0.40:
        return "MEDIOCRE"
    return "WEAK"


def _payoff_ratio(recs: List[dict]) -> Optional[float]:
    winners = [abs(r.get("return_pct", 0.0)) for r in recs if r.get("was_correct")]
    losers  = [abs(r.get("return_pct", 0.0)) for r in recs if not r.get("was_correct")]
    if not winners or not losers:
        return None
    avg_win  = sum(winners) / len(winners)
    avg_loss = sum(losers) / len(losers)
    return round(avg_win / avg_loss, 3) if avg_loss else None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def analyze_combo_stats(graded: List[dict]) -> dict:
    """Win rate, payoff ratio, and MFE/MAE by domain combination.

    Returns a dict keyed by combo string, sorted best→worst by win rate.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in graded:
        key = _extract_combo_key(rec)
        groups[key].append(rec)

    stats: Dict[str, dict] = {}
    for combo, recs in groups.items():
        if len(recs) < MIN_SAMPLES_FOR_STATS:
            continue
        wins = [r for r in recs if r.get("was_correct")]
        wr   = len(wins) / len(recs)
        mfe_vals = [r["mfe_pct"] for r in recs if r.get("mfe_pct") is not None]
        mae_vals = [r["mae_pct"] for r in recs if r.get("mae_pct") is not None]
        ret_vals = [abs(r.get("return_pct", 0.0)) for r in recs]

        stats[combo] = {
            "n":            len(recs),
            "wins":         len(wins),
            "win_rate":     round(wr, 4),
            "win_rate_pct": f"{wr * 100:.1f}%",
            "grade":        _win_rate_label(wr),
            "avg_return_pct":  _safe_mean(ret_vals),
            "payoff_ratio": _payoff_ratio(recs),
            "avg_mfe_pct":  _safe_mean(mfe_vals),
            "avg_mae_pct":  _safe_mean(mae_vals),
        }

    return dict(sorted(stats.items(), key=lambda x: x[1]["win_rate"], reverse=True))


def analyze_combo_by_regime(graded: List[dict]) -> dict:
    """Win rate per combo broken down by regime.

    Structure: {combo: {regime: {n, wins, win_rate, win_rate_pct}}}
    Only includes combos with at least MIN_SAMPLES_FOR_STATS total graded.
    """
    # First collect: combo -> regime -> [records]
    groups: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in graded:
        combo  = _extract_combo_key(rec)
        regime = _extract_regime(rec)
        groups[combo][regime].append(rec)

    result: Dict[str, dict] = {}
    for combo, regime_map in groups.items():
        total_n = sum(len(v) for v in regime_map.values())
        if total_n < MIN_SAMPLES_FOR_STATS:
            continue
        regime_stats: Dict[str, dict] = {}
        for regime, recs in regime_map.items():
            if len(recs) < MIN_SAMPLES_FOR_STATS:
                continue
            wins = sum(1 for r in recs if r.get("was_correct"))
            wr   = wins / len(recs)
            regime_stats[regime] = {
                "n":            len(recs),
                "wins":         wins,
                "win_rate":     round(wr, 4),
                "win_rate_pct": f"{wr * 100:.1f}%",
            }
        if regime_stats:
            result[combo] = regime_stats

    return result


def analyze_source_stats(graded: List[dict]) -> dict:
    """Win rate per individual signal source.

    Counts a source once per prediction regardless of how many times it
    appears. Sorted best→worst.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in graded:
        for source in _extract_source_names(rec):
            groups[source].append(rec)

    stats: Dict[str, dict] = {}
    for source, recs in groups.items():
        if len(recs) < MIN_SAMPLES_FOR_STATS:
            continue
        wins = [r for r in recs if r.get("was_correct")]
        wr   = len(wins) / len(recs)
        stats[source] = {
            "n":            len(recs),
            "wins":         len(wins),
            "win_rate":     round(wr, 4),
            "win_rate_pct": f"{wr * 100:.1f}%",
            "grade":        _win_rate_label(wr),
            "domain":       _SOURCE_TO_DOMAIN.get(source, "unknown"),
        }

    return dict(sorted(stats.items(), key=lambda x: x[1]["win_rate"], reverse=True))


def analyze_timing(graded: List[dict]) -> dict:
    """Timing accuracy: did the move happen inside the expected window?

    Buckets:
      early   — outcome graded before 50% of the expected window elapsed
      on_time — outcome graded within the expected window
      late    — move happened but after the window
      missed  — prediction was wrong (no move or wrong direction)
    """
    buckets = {"early": 0, "on_time": 0, "late": 0, "missed": 0}
    right_thesis_wrong_timing: List[dict] = []
    total = 0

    for rec in graded:
        total += 1

        if not rec.get("was_correct"):
            # Check right-thesis-wrong-timing: large MFE but wrong final outcome
            mfe = rec.get("mfe_pct")
            ret = abs(rec.get("return_pct", 0.0))
            if mfe is not None and mfe >= 5.0 and ret < 2.0:
                right_thesis_wrong_timing.append({
                    "symbol":      rec.get("symbol", ""),
                    "direction":   rec.get("direction", ""),
                    "combo":       _extract_combo_key(rec),
                    "mfe_pct":     mfe,
                    "return_pct":  rec.get("return_pct"),
                })
            buckets["missed"] += 1
            continue

        pred_at_str    = rec.get("predicted_at", "")
        outcome_at_str = rec.get("outcome_at", "")
        window_days    = float(
            rec.get("outcome_window_days")
            or rec.get("window_days")
            or rec.get("timeframe_days")
            or 14
        )

        if not pred_at_str or not outcome_at_str:
            buckets["on_time"] += 1
            continue

        try:
            pred_at    = datetime.fromisoformat(pred_at_str)
            outcome_at = datetime.fromisoformat(outcome_at_str)
            elapsed    = (outcome_at - pred_at).total_seconds() / 86400.0

            if elapsed < window_days * 0.5:
                buckets["early"] += 1
            elif elapsed <= window_days:
                buckets["on_time"] += 1
            else:
                buckets["late"] += 1
        except (ValueError, TypeError):
            buckets["on_time"] += 1

    on_time_count = buckets["on_time"] + buckets["early"]
    on_time_rate  = on_time_count / total if total > 0 else None
    late_rate     = buckets["late"] / total if total > 0 else None
    rtwt_rate     = len(right_thesis_wrong_timing) / total if total > 0 else None

    insight_parts: List[str] = []
    if total > 0:
        if buckets["missed"] / total > 0.5:
            insight_parts.append(
                f"Majority ({buckets['missed']}/{total}) missed entirely"
            )
        if buckets["late"] / total > 0.2:
            insight_parts.append(
                f"{buckets['late']} moves happened after the window — "
                "consider extending outcome_window_days"
            )
        if buckets["early"] / total > 0.2:
            insight_parts.append(
                f"{buckets['early']} moves happened in the first half of the window — "
                "signals may fire early"
            )
        if right_thesis_wrong_timing:
            insight_parts.append(
                f"{len(right_thesis_wrong_timing)} had MFE ≥5% but final return <2% "
                "(right direction, wrong exit timing)"
            )

    return {
        "total_graded": total,
        "buckets":      buckets,
        "on_time_rate": round(on_time_rate, 4) if on_time_rate is not None else None,
        "on_time_rate_pct": (
            f"{on_time_rate * 100:.1f}%" if on_time_rate is not None else "N/A"
        ),
        "late_rate":    round(late_rate, 4) if late_rate is not None else None,
        "right_thesis_wrong_timing": {
            "count": len(right_thesis_wrong_timing),
            "rate":  round(rtwt_rate, 4) if rtwt_rate is not None else None,
            "examples": right_thesis_wrong_timing[:5],
        },
        "insight": "; ".join(insight_parts) if insight_parts else "Timing looks normal",
    }


def _build_top_bottom(
    combo_stats: dict,
    n: int = TOP_N_COMBOS,
) -> Tuple[List[dict], List[dict]]:
    """Extract top N and bottom N combos from combo_stats.

    Returns (top_list, bottom_list) where each item is a readable dict.
    """
    ranked = [
        {"combo": k, **v}
        for k, v in combo_stats.items()
    ]
    if not ranked:
        return [], []

    top    = ranked[:n]
    bottom = list(reversed(ranked[-n:]))
    return top, bottom


def analyze_overall(graded: List[dict]) -> dict:
    """Overall win rate, direction breakdown, confidence buckets."""
    if not graded:
        return {"total_graded": 0}

    total  = len(graded)
    wins   = [r for r in graded if r.get("was_correct")]
    losses = [r for r in graded if not r.get("was_correct")]
    wr     = len(wins) / total

    # Direction breakdown
    direction_groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in graded:
        direction_groups[rec.get("direction", "unknown")].append(rec)

    direction_stats: dict = {}
    for direction, recs in direction_groups.items():
        w = sum(1 for r in recs if r.get("was_correct"))
        direction_stats[direction] = {
            "n":            len(recs),
            "wins":         w,
            "win_rate_pct": f"{w / len(recs) * 100:.1f}%",
        }

    # Confidence buckets
    conf_buckets: Dict[str, List[dict]] = defaultdict(list)
    for rec in graded:
        conf = rec.get("predicted_confidence") or rec.get("confidence") or 0.0
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.0
        if conf >= 0.80:
            bucket = "0.80+"
        elif conf >= 0.65:
            bucket = "0.65-0.80"
        elif conf >= 0.50:
            bucket = "0.50-0.65"
        else:
            bucket = "below 0.50"
        conf_buckets[bucket].append(rec)

    conf_stats: dict = {}
    for bucket, recs in conf_buckets.items():
        w = sum(1 for r in recs if r.get("was_correct"))
        conf_stats[bucket] = {
            "n":            len(recs),
            "wins":         w,
            "win_rate_pct": f"{w / len(recs) * 100:.1f}%",
        }

    return {
        "total_graded":    total,
        "wins":            len(wins),
        "losses":          len(losses),
        "overall_win_rate":     round(wr, 4),
        "overall_win_rate_pct": f"{wr * 100:.1f}%",
        "grade":           _win_rate_label(wr),
        "by_direction":    direction_stats,
        "by_confidence":   conf_stats,
        "avg_win_return_pct":  _safe_mean(
            [abs(r.get("return_pct", 0.0)) for r in wins]
        ),
        "avg_loss_return_pct": _safe_mean(
            [abs(r.get("return_pct", 0.0)) for r in losses]
        ),
        "payoff_ratio": _payoff_ratio(graded),
    }


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------

def run_analysis(logger: logging.Logger, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """Load data, run all analyses, return the full report dict."""
    logger.info("Loading predictions from %s", PREDICTIONS_PATH)
    predictions = _load_jsonl(PREDICTIONS_PATH, lookback_days=lookback_days)
    logger.info("Loaded %d prediction records", len(predictions))

    logger.info("Loading outcomes from %s", OUTCOMES_PATH)
    outcomes = _load_jsonl(OUTCOMES_PATH, lookback_days=lookback_days)
    logger.info("Loaded %d outcome records", len(outcomes))

    # Merge predictions into outcomes to recover extra fields
    pred_index = _build_prediction_index(predictions)
    merged     = _merge_with_predictions(outcomes, pred_index)

    # Filter to graded only
    graded = [r for r in merged if _is_graded(r)]
    ungraded_count = len(merged) - len(graded)
    logger.info(
        "Graded: %d | Ungraded (pending): %d | Total outcomes: %d",
        len(graded), ungraded_count, len(merged),
    )

    if not graded:
        logger.warning("No graded outcomes found — writing minimal report")
        return {
            "generated_at":   datetime.now().isoformat(),
            "data_summary":   {
                "total_predictions":    len(predictions),
                "total_outcomes":       len(outcomes),
                "total_graded":         0,
                "total_ungraded":       ungraded_count,
                "lookback_days":        lookback_days,
            },
            "message": "No graded outcomes yet. Outcomes become graded when the "
                       "OutcomeCollector evaluates predictions against actual price moves.",
        }

    logger.info("Running combo analysis...")
    combo_stats  = analyze_combo_stats(graded)

    logger.info("Running regime breakdown...")
    regime_stats = analyze_combo_by_regime(graded)

    logger.info("Running source analysis...")
    source_stats = analyze_source_stats(graded)

    logger.info("Running timing analysis...")
    timing       = analyze_timing(graded)

    logger.info("Running overall stats...")
    overall      = analyze_overall(graded)

    top_combos, worst_combos = _build_top_bottom(combo_stats, n=TOP_N_COMBOS)

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_summary": {
            "total_predictions":    len(predictions),
            "total_outcomes":       len(outcomes),
            "total_graded":         len(graded),
            "total_ungraded":       ungraded_count,
            "lookback_days":        lookback_days,
            "graded_rate_pct":      f"{len(graded) / len(outcomes) * 100:.1f}%"
                                    if outcomes else "N/A",
        },
        "overall": overall,
        "top_5_best_combos":  top_combos,
        "top_5_worst_combos": worst_combos,
        "combo_stats": combo_stats,
        "combo_stats_by_regime": regime_stats,
        "source_stats": source_stats,
        "timing": timing,
        "_notes": {
            "combo_key_format": (
                "Domain names joined by '+', e.g. 'insider+macro+technical'. "
                "Sorted alphabetically so order does not affect grouping."
            ),
            "win_rate_grades": {
                "STRONG":   ">= 70%",
                "GOOD":     "55-70%",
                "MEDIOCRE": "40-55%",
                "WEAK":     "< 40%",
            },
            "timing_buckets": {
                "early":   "Move confirmed in first 50% of expected window",
                "on_time": "Move confirmed within expected window",
                "late":    "Move confirmed after expected window expired",
                "missed":  "Prediction was wrong (no move or wrong direction)",
            },
        },
    }

    logger.info(
        "Analysis complete: %d graded, overall WR=%.1f%%, %d combos analyzed",
        len(graded),
        overall["overall_win_rate"] * 100,
        len(combo_stats),
    )
    return report


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_report(report: dict, logger: logging.Logger) -> None:
    """Atomic write of the report JSON to OUTPUT_PATH."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(OUTPUT_PATH)
        logger.info("Report written to %s", OUTPUT_PATH)
    except Exception:
        logger.exception("Failed to write report")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous post-mortem analysis — "
                    "analyzes graded predictions every N seconds."
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single analysis pass then exit (useful for testing)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
        metavar="SECONDS",
        help=f"Seconds between analysis runs (default: {DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_DAYS,
        metavar="DAYS",
        help=f"Lookback window in days, 0 = all time (default: {LOOKBACK_DAYS})",
    )
    args = parser.parse_args()

    global LOOKBACK_DAYS  # noqa: PLW0603
    LOOKBACK_DAYS = args.lookback

    logger = _setup_logging(LOG_PATH)
    logger.info(
        "Continuous post-mortem starting — interval=%ds, lookback=%dd, once=%s",
        args.interval, args.lookback, args.once,
    )

    cycle = 0
    while True:
        cycle += 1
        logger.info("=== Analysis cycle %d ===", cycle)
        try:
            report = run_analysis(logger)
            write_report(report, logger)
        except Exception:
            logger.exception("Cycle %d failed", cycle)

        if args.once:
            logger.info("--once flag set — exiting after first cycle")
            break

        logger.info("Sleeping %d seconds until next cycle...", args.interval)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
