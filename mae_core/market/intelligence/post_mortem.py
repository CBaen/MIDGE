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

Statistical computation is delegated to post_mortem_analysis.py.
Utility helpers live in post_mortem_utils.py.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.post_mortem_utils import _json_default
from mae_core.market.intelligence.post_mortem_analysis import (
    MIN_OUTCOMES_FOR_STATS,
    compute_combo_stats,
    compute_sequence_stats,
    compute_timing_accuracy,
    compute_regime_failures,
    compute_mfe_mae_patterns,
    identify_flagged_orderings,
)

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_OUTCOMES_PATH = _DATA_DIR / "outcomes.jsonl"
_INSIGHTS_PATH = _DATA_DIR / "post_mortem_insights.json"

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

        Delegates to post_mortem_analysis functions. Returns a dict with:
          combo_stats, sequence_stats, timing_summary, regime_summary,
          mfe_mae_patterns, flagged_orderings, generated_at
        """
        combo_stats = compute_combo_stats(outcomes, self._extract_combo_key)
        sequence_stats = compute_sequence_stats(outcomes, self._extract_sequence_key)
        timing_summary = compute_timing_accuracy(outcomes)
        regime_summary = compute_regime_failures(outcomes)
        mfe_mae_patterns = compute_mfe_mae_patterns(outcomes, self._extract_combo_key)
        flagged_orderings = identify_flagged_orderings(sequence_stats)

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
                # Push one update per outcome (successes first, then failures).
                # Batch approximation — full replay would be ideal but we only
                # have aggregated stats here.
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
            unique_sigs = sorted(set(str(s) for s in signals if s))
            if unique_sigs:
                return "signals:" + "+".join(unique_sigs)

        return f"source:{source}" if source else "unknown"

    def _extract_sequence_key(self, rec: dict) -> str:
        """Extract an ordered sequence key from an outcome record.

        Sequence data is stored in metadata.domain_sequence (set by convergence
        alerter's domain_sequence field). Falls back to empty string if unavailable.
        """
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            seq = metadata.get("domain_sequence", [])
            if seq and isinstance(seq, list):
                return _SEQ_SEP.join(str(d) for d in seq)
        return ""
