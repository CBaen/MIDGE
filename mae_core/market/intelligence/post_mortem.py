"""Post-Mortem Prediction Reviewer — periodic retrospective analysis of graded outcomes.

Answers the questions nobody was asking:
  - WHY did this prediction succeed or fail?
  - Which domain combos win when fired in a specific ORDER vs. any order?
  - Was MFE much larger than the final outcome? (Right thesis, wrong timing.)
  - Did regime change mid-prediction window? (Bull-to-bear regime transition failures.)
  - Which domain orderings consistently fail? (The inverse lag problem.)

Runs every 500 steps (same cadence as Granger/Lag). Reads outcomes.jsonl, computes
aggregate statistics, writes post_mortem_insights.json, and feeds actionable findings
back into Thompson distributions with sequence-aware weights.

Biological analogy: Hippocampal consolidation during sleep — the organism reviews
what happened, why, and updates its priors so tomorrow it navigates better.
"""

import json
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_OUTCOMES_PATH = _DATA_DIR / "outcomes.jsonl"
_INSIGHTS_PATH = _DATA_DIR / "post_mortem_insights.json"

# Minimum outcomes per category to report meaningful statistics
MIN_OUTCOMES_FOR_STATS = 3

# Sequence key separator (used in Thompson combo keys for ordered sequences)
_SEQ_SEP = ">>"


class PostMortemReviewer:
    """Periodic retrospective analysis of graded predictions.

    Wired into the step hook chain (every 500 steps). Reads outcomes.jsonl,
    groups by domain combo and domain ordering, computes win rates and
    MFE/MAE patterns, then feeds sequence-aware updates back to Thompson.

    Dependencies:
        thompson_sampler: Optional — used to update sequence-aware distributions.
        regime_classifier: Optional — used to tag regime at review time.
    """

    def __init__(
        self,
        thompson_sampler=None,
        regime_classifier=None,
        data_dir: Optional[Path] = None,
        lookback_days: int = 90,
    ):
        self._thompson = thompson_sampler
        self._regime_classifier = regime_classifier
        self._data_dir = data_dir or _DATA_DIR
        self._outcomes_path = self._data_dir / "outcomes.jsonl"
        self._insights_path = self._data_dir / "post_mortem_insights.json"
        self._lookback_days = lookback_days
        self._total_reviews = 0
        self._last_review_at: Optional[datetime] = None

    # ── Public API ─────────────────────────────────────────────────────

    def review(self) -> dict:
        """Run a full post-mortem review cycle.

        1. Load graded outcomes from outcomes.jsonl (last N days).
        2. Compute aggregate statistics by combo, ordering, MFE/MAE, timing, regime.
        3. Write post_mortem_insights.json (atomic write).
        4. Feed sequence-aware Thompson updates.

        Returns:
            Summary dict with counts and key findings.
        """
        self._total_reviews += 1
        self._last_review_at = datetime.now()

        outcomes = self._load_outcomes()
        if not outcomes:
            logger.debug("PostMortem: no graded outcomes found — skipping")
            return {"outcomes_reviewed": 0, "total_reviews": self._total_reviews}

        logger.info("PostMortem: reviewing %d graded outcomes", len(outcomes))

        try:
            insights = self._compute_insights(outcomes)
            self._write_insights(insights)
            self._feed_thompson_updates(insights)

            summary = {
                "outcomes_reviewed": len(outcomes),
                "total_reviews": self._total_reviews,
                "combos_analyzed": len(insights.get("combo_stats", {})),
                "sequences_analyzed": len(insights.get("sequence_stats", {})),
                "timing_accuracy": insights.get("timing_summary", {}).get("on_time_rate"),
                "regime_failures": insights.get("regime_summary", {}).get("failure_rate"),
            }
            logger.info(
                "PostMortem: %d outcomes, %d combos, %d sequences",
                len(outcomes),
                summary["combos_analyzed"],
                summary["sequences_analyzed"],
            )
            return summary

        except Exception:
            logger.warning("PostMortem: review cycle failed", exc_info=True)
            return {"outcomes_reviewed": len(outcomes), "error": True,
                    "total_reviews": self._total_reviews}

    def get_statistics(self) -> dict:
        """HolonProxy.sense() delegation."""
        return {
            "total_reviews": self._total_reviews,
            "last_review_at": (
                self._last_review_at.isoformat() if self._last_review_at else None
            ),
            "insights_path": str(self._insights_path),
            "insights_exist": self._insights_path.exists(),
        }

    # ── Data Loading ───────────────────────────────────────────────────

    def _load_outcomes(self) -> List[dict]:
        """Load graded outcomes from outcomes.jsonl (last N days).

        Deduplicates by prediction_id + outcome_at so the same outcome
        graded multiple times (replay artifacts) only counts once.
        """
        if not self._outcomes_path.exists():
            return []

        cutoff = datetime.now() - timedelta(days=self._lookback_days)
        seen: set = set()
        outcomes = []

        try:
            with open(self._outcomes_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Dedup key: prediction_id + outcome timestamp
                    pid = rec.get("prediction_id", "")
                    oat = rec.get("outcome_at", "")
                    dedup = f"{pid}:{oat}"
                    if dedup in seen:
                        continue
                    seen.add(dedup)

                    # Apply lookback filter (use outcome_at if available)
                    ts_str = oat or rec.get("predicted_at", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts < cutoff:
                                continue
                        except ValueError:
                            pass

                    outcomes.append(rec)
        except Exception:
            logger.warning("PostMortem: failed to read outcomes.jsonl", exc_info=True)

        return outcomes

    # ── Core Analysis ──────────────────────────────────────────────────

    def _compute_insights(self, outcomes: List[dict]) -> dict:
        """Compute all post-mortem statistics from raw outcome records.

        Returns a dict with sections:
          combo_stats, sequence_stats, timing_summary, regime_summary,
          mfe_mae_patterns, flagged_orderings, generated_at
        """
        combo_stats = self._compute_combo_stats(outcomes)
        sequence_stats = self._compute_sequence_stats(outcomes)
        timing_summary = self._compute_timing_accuracy(outcomes)
        regime_summary = self._compute_regime_failures(outcomes)
        mfe_mae_patterns = self._compute_mfe_mae_patterns(outcomes)
        flagged_orderings = self._identify_flagged_orderings(sequence_stats)

        return {
            "generated_at": datetime.now().isoformat(),
            "outcomes_reviewed": len(outcomes),
            "lookback_days": self._lookback_days,
            "combo_stats": combo_stats,
            "sequence_stats": sequence_stats,
            "timing_summary": timing_summary,
            "regime_summary": regime_summary,
            "mfe_mae_patterns": mfe_mae_patterns,
            "flagged_orderings": flagged_orderings,
        }

    def _compute_combo_stats(self, outcomes: List[dict]) -> dict:
        """Win rate, avg MFE/MAE, and count by domain combination.

        Domain combo is extracted from contributing_signals or source field.
        """
        # Group by sorted combo key
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in outcomes:
            combo = self._extract_combo_key(rec)
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

        # Sort by win_rate descending
        return dict(sorted(stats.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    def _compute_sequence_stats(self, outcomes: List[dict]) -> dict:
        """Win rate by domain ORDERING (not just set membership).

        Two predictions with the same domains but different firing order
        (insider→macro vs. macro→insider) get separate entries.
        """
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in outcomes:
            seq = self._extract_sequence_key(rec)
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

    def _compute_timing_accuracy(self, outcomes: List[dict]) -> dict:
        """Did the move happen within the expected window, early, or late?

        Timing buckets:
          - early: outcome graded before half the expected window elapsed
          - on_time: outcome graded within the expected window
          - late: move happened but after expected window (if outcome_at > window end)
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

    def _compute_regime_failures(self, outcomes: List[dict]) -> dict:
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

    def _compute_mfe_mae_patterns(self, outcomes: List[dict]) -> dict:
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
            # For bearish predictions, return_pct is negative on success
            # Treat abs value as the magnitude
            abs_final = abs(final_ret)

            if mfe >= 5.0 and abs_final < 2.0:
                right_thesis_wrong_timing.append({
                    "symbol": rec.get("symbol", ""),
                    "direction": rec.get("direction", ""),
                    "mfe_pct": mfe,
                    "final_return_pct": final_ret,
                    "source": rec.get("source", ""),
                    "combo": self._extract_combo_key(rec),
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

    def _identify_flagged_orderings(self, sequence_stats: dict) -> List[dict]:
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

    # ── Thompson Feedback ──────────────────────────────────────────────

    def _feed_thompson_updates(self, insights: dict) -> None:
        """Update Thompson distributions with sequence-aware findings.

        For each combo+sequence with enough data:
          - Register success/failure counts as Thompson updates.
          - Key format: "seq:{domain_a}>>{domain_b}>>..." for ordered combos.

        This is additive — we don't reset distributions, just push new evidence.
        """
        if self._thompson is None:
            return

        sequence_stats = insights.get("sequence_stats", {})
        updates_made = 0

        for seq_key, stats in sequence_stats.items():
            n = stats.get("n", 0)
            wins = stats.get("wins", 0)
            if n < MIN_OUTCOMES_FOR_STATS:
                continue

            thompson_key = f"seq:{seq_key}"
            try:
                # Push one update per outcome (successes first, then failures)
                # This is a batch approximation — full replay would be ideal
                # but we only have aggregated stats here.
                regime = self._get_regime()
                for _ in range(wins):
                    self._thompson.update(thompson_key, success=True, regime=regime)
                for _ in range(n - wins):
                    self._thompson.update(thompson_key, success=False, regime=regime)
                updates_made += 1
            except Exception:
                logger.debug("Thompson update failed for seq key %s", thompson_key, exc_info=True)

        if updates_made > 0:
            logger.info("PostMortem: pushed %d sequence-aware Thompson updates", updates_made)

    def _get_regime(self) -> str:
        """Get current market regime string."""
        if self._regime_classifier is None:
            return "default"
        try:
            return self._regime_classifier.classify()
        except Exception:
            return "default"

    # ── Persistence ────────────────────────────────────────────────────

    def _write_insights(self, insights: dict) -> None:
        """Atomic write of post_mortem_insights.json."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._insights_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(insights, indent=2, default=_json_default),
                encoding="utf-8",
            )
            tmp.replace(self._insights_path)
            logger.debug("PostMortem: insights written to %s", self._insights_path)
        except Exception:
            logger.warning("PostMortem: failed to write insights", exc_info=True)

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_combo_key(self, rec: dict) -> str:
        """Extract a sorted domain combo key from an outcome record.

        Tries, in order:
        1. source field if it starts with 'combo:' (convergence alerts)
        2. contributing_signals list (set of signal source names)
        3. source field as single-domain key
        """
        source = rec.get("source", "")
        if source.startswith("combo:"):
            # Already a combo key — normalize by re-sorting domains
            domains = source[6:].split("+")
            return "combo:" + "+".join(sorted(domains))

        signals = rec.get("contributing_signals", [])
        if signals:
            # Use contributing signal names as a rough proxy for domains
            unique_sigs = sorted(set(str(s) for s in signals if s))
            if unique_sigs:
                return "signals:" + "+".join(unique_sigs)

        # Fall back to source as single-domain
        return f"source:{source}" if source else "unknown"

    def _extract_sequence_key(self, rec: dict) -> str:
        """Extract an ordered sequence key from an outcome record.

        Sequence data is stored in metadata.domain_sequence (set by convergence
        alerter's domain_sequence field). Falls back to None if unavailable.
        """
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            seq = metadata.get("domain_sequence", [])
            if seq and isinstance(seq, list):
                return _SEQ_SEP.join(str(d) for d in seq)
        return ""


# ── Module-level utilities ─────────────────────────────────────────────


def _safe_mean(values: list) -> Optional[float]:
    """Mean of a list, None if empty."""
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _payoff_ratio(recs: List[dict]) -> Optional[float]:
    """Avg winning return / avg losing return magnitude."""
    winners = [abs(r.get("return_pct", 0.0)) for r in recs if r.get("was_correct")]
    losers = [abs(r.get("return_pct", 0.0)) for r in recs if not r.get("was_correct")]
    if not winners or not losers:
        return None
    avg_win = sum(winners) / len(winners)
    avg_loss = sum(losers) / len(losers)
    if avg_loss == 0:
        return None
    return round(avg_win / avg_loss, 3)


def _timing_insight(buckets: dict, total: int) -> str:
    """Plain-language summary of timing distribution."""
    if total == 0:
        return "No outcomes to analyze"
    missed = buckets.get("missed", 0)
    early = buckets.get("early", 0)
    late = buckets.get("late", 0)
    parts = []
    if missed / total > 0.5:
        parts.append(f"majority ({missed}/{total}) missed entirely")
    if early / total > 0.2:
        parts.append(f"{early} moved early (watch for faster-moving opportunities)")
    if late / total > 0.2:
        parts.append(f"{late} moved late (window may be too short)")
    return "; ".join(parts) if parts else "Timing distribution looks normal"


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-standard types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
