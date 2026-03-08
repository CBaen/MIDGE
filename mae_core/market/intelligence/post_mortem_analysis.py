"""Post-Mortem Analysis Functions — statistical computation helpers for PostMortemReviewer.

Extracted from post_mortem.py to stay under the 500-line monolith cap.
These functions are pure computation: they receive raw outcome records and
return structured statistics dicts. No I/O, no Thompson, no external state.

Imported and called by PostMortemReviewer in post_mortem.py.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from mae_core.market.intelligence.post_mortem_utils import (
    _safe_mean,
    _payoff_ratio,
    _timing_insight,
)

MIN_OUTCOMES_FOR_STATS = 3
_SEQ_SEP = ">>"


def compute_combo_stats(
    outcomes: List[dict],
    extract_combo_key,
) -> dict:
    """Win rate, avg MFE/MAE, and count by domain combination.

    Domain combo is extracted from contributing_signals or source field.
    Sorted by win_rate descending.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in outcomes:
        combo = extract_combo_key(rec)
        groups[combo].append(rec)

    stats = {}
    for combo, recs in groups.items():
        if len(recs) < MIN_OUTCOMES_FOR_STATS:
            continue
        wins = [r for r in recs if r.get("was_correct", False)]
        mfe_vals = [r.get("mfe_pct", 0.0) for r in recs if r.get("mfe_pct") is not None]
        mae_vals = [r.get("mae_pct", 0.0) for r in recs if r.get("mae_pct") is not None]
        ret_vals = [abs(r.get("return_pct", 0.0)) for r in recs]

        stats[combo] = {
            "n": len(recs),
            "win_rate": len(wins) / len(recs),
            "wins": len(wins),
            "avg_mfe": _safe_mean(mfe_vals),
            "avg_mae": _safe_mean(mae_vals),
            "avg_abs_return": _safe_mean(ret_vals),
            "payoff_ratio": _payoff_ratio(recs),
        }

    return dict(sorted(stats.items(), key=lambda x: x[1]["win_rate"], reverse=True))


def compute_sequence_stats(
    outcomes: List[dict],
    extract_sequence_key,
) -> dict:
    """Win rate by domain ORDERING (not just set membership).

    Two predictions with the same domains but different firing order
    (insider→macro vs. macro→insider) get separate entries.
    Sorted by win_rate descending.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in outcomes:
        seq = extract_sequence_key(rec)
        if seq:
            groups[seq].append(rec)

    stats = {}
    for seq, recs in groups.items():
        if len(recs) < MIN_OUTCOMES_FOR_STATS:
            continue
        wins = [r for r in recs if r.get("was_correct", False)]
        stats[seq] = {
            "n": len(recs),
            "win_rate": len(wins) / len(recs),
            "wins": len(wins),
            "avg_return": _safe_mean([abs(r.get("return_pct", 0.0)) for r in recs]),
        }

    return dict(sorted(stats.items(), key=lambda x: x[1]["win_rate"], reverse=True))


def compute_timing_accuracy(outcomes: List[dict]) -> dict:
    """Did the move happen within the expected window, early, or late?

    Timing buckets:
      - early: outcome graded before half the expected window elapsed
      - on_time: outcome graded within the expected window
      - late: move happened but after expected window
      - missed: no move materialized (was_correct=False)
    """
    buckets: Dict[str, int] = {"early": 0, "on_time": 0, "late": 0, "missed": 0}
    total = 0

    for rec in outcomes:
        if not rec.get("was_correct"):
            buckets["missed"] += 1
            total += 1
            continue

        pred_at_str = rec.get("predicted_at", "")
        outcome_at_str = rec.get("outcome_at", "")
        window_days = rec.get("outcome_window_days", rec.get("timeframe_days", 14))

        if not pred_at_str or not outcome_at_str:
            buckets["on_time"] += 1  # Can't classify → assume ok
            total += 1
            continue

        try:
            pred_at = datetime.fromisoformat(pred_at_str)
            outcome_at = datetime.fromisoformat(outcome_at_str)
            elapsed_days = (outcome_at - pred_at).total_seconds() / 86400

            if elapsed_days < window_days * 0.5:
                buckets["early"] += 1
            elif elapsed_days <= window_days:
                buckets["on_time"] += 1
            else:
                buckets["late"] += 1
        except (ValueError, TypeError):
            buckets["on_time"] += 1

        total += 1

    on_time_rate = (
        (buckets["on_time"] + buckets["early"]) / total if total > 0 else None
    )
    return {
        "total": total,
        "buckets": buckets,
        "on_time_rate": round(on_time_rate, 3) if on_time_rate is not None else None,
        "insight": _timing_insight(buckets, total),
    }


def compute_regime_failures(outcomes: List[dict]) -> dict:
    """Predictions made in one regime that played out in a different regime.

    A 'regime transition failure' is a prediction that was wrong AND the
    regime tag at prediction time differs from regime at outcome time.
    These are structural failures, not signal failures — the environment changed.
    """
    transition_fails = []
    total_wrong = 0

    for rec in outcomes:
        if rec.get("was_correct"):
            continue
        total_wrong += 1
        pred_regime = rec.get("regime_at_prediction", "")
        outcome_regime = rec.get("regime_at_outcome", "")
        if pred_regime and outcome_regime and pred_regime != outcome_regime:
            transition_fails.append({
                "symbol": rec.get("symbol", ""),
                "pred_regime": pred_regime,
                "outcome_regime": outcome_regime,
                "return_pct": rec.get("return_pct", 0.0),
            })

    failure_rate = (
        len(transition_fails) / total_wrong if total_wrong > 0 else 0.0
    )
    return {
        "total_wrong": total_wrong,
        "regime_transitions": len(transition_fails),
        "failure_rate": round(failure_rate, 3),
        "examples": transition_fails[:5],
        "insight": (
            f"{len(transition_fails)}/{total_wrong} losses involved regime change"
            if total_wrong > 0 else "No losses to analyze"
        ),
    }


def compute_mfe_mae_patterns(outcomes: List[dict], extract_combo_key) -> dict:
    """Identify predictions where MFE was large but final outcome was small/negative.

    'Right thesis, wrong timing' = MFE >= 5% but final return < 2%.
    These are cases where the prediction was directionally correct but
    the exit timing was off — the move happened but reversed.
    """
    right_thesis_wrong_timing = []
    total_with_mfe = 0

    for rec in outcomes:
        mfe = rec.get("mfe_pct")
        if mfe is None:
            continue
        total_with_mfe += 1
        final_ret = rec.get("return_pct", 0.0)
        abs_final = abs(final_ret)

        if mfe >= 5.0 and abs_final < 2.0:
            right_thesis_wrong_timing.append({
                "symbol": rec.get("symbol", ""),
                "direction": rec.get("direction", ""),
                "mfe_pct": mfe,
                "final_return_pct": final_ret,
                "source": rec.get("source", ""),
                "combo": extract_combo_key(rec),
            })

    rtwt_rate = (
        len(right_thesis_wrong_timing) / total_with_mfe
        if total_with_mfe > 0 else 0.0
    )
    return {
        "total_with_mfe_data": total_with_mfe,
        "right_thesis_wrong_timing_count": len(right_thesis_wrong_timing),
        "rtwt_rate": round(rtwt_rate, 3),
        "examples": right_thesis_wrong_timing[:5],
        "insight": (
            f"{len(right_thesis_wrong_timing)} predictions had MFE ≥5% but "
            f"final return <2% — timing capture issue, not signal quality"
            if right_thesis_wrong_timing else "No right-thesis-wrong-timing patterns found"
        ),
    }


def identify_flagged_orderings(sequence_stats: dict) -> list:
    """Flag domain orderings that consistently fail (win_rate < 30% with n >= 5).

    These orderings should be penalized in future sequence scoring.
    """
    flagged = []
    for seq, stats in sequence_stats.items():
        if stats["n"] >= 5 and stats["win_rate"] < 0.30:
            flagged.append({
                "sequence": seq,
                "win_rate": stats["win_rate"],
                "n": stats["n"],
                "flag": "consistently_fails",
            })
    flagged.sort(key=lambda x: x["win_rate"])
    return flagged
