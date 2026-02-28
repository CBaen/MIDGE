"""
thompson_calibrator.py - Thompson Sampling Diagnostic and Seed-Fix Module

Two responsibilities:

1. SEED FIX (runs in __init__):
   The 15 signal-source-level keys in learning_config.py (sec_form4, congressional,
   insider_cluster, etc.) were never seeded into thompson_distributions.json. Every
   lookup falls through to uninformative Beta(1,1). This module seeds them from the
   source_reliability values in LEARNING_CONFIG using the same formula ThompsonSampler
   uses: alpha = r * prior_scale, beta = (1-r) * prior_scale. Idempotent — existing
   keys with alpha+beta > 2.1 are not overwritten.

2. CALIBRATION DIAGNOSTIC (runs on cadence):
   Joins predictions.jsonl to outcomes.jsonl by prediction_id, groups by source,
   and for sources with >=5 matched pairs computes Brier score, 5 calibration
   buckets, over/under-confidence flags, and a human-readable recommendation. Persists
   the report to data/market/calibration_report.json.
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
CALIBRATION_PATH = DATA_DIR / "calibration_report.json"

# Minimum alpha+beta sum to consider a distribution "seeded" (above uninformative 2.0)
_SEEDED_THRESHOLD = 2.1

# Minimum matched prediction/outcome pairs required before computing calibration
_MIN_PAIRS_FOR_CALIBRATION = 5

# Confidence buckets: (min_inclusive, max_exclusive)
_BUCKET_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.001]  # Last edge > 1.0 to catch exactly 1.0


@dataclass
class CalibrationBucket:
    confidence_min: float
    confidence_max: float
    predicted_confidence: float   # Midpoint of the bucket
    observed_hit_rate: float
    sample_count: int
    brier_contribution: float


@dataclass
class SourceCalibration:
    source: str
    brier_score: float            # Lower is better (0=perfect, 0.25=random)
    mean_predicted_confidence: float
    mean_observed_hit_rate: float
    sample_count: int
    is_overconfident: bool        # predicted > observed by >10%
    is_underconfident: bool       # predicted < observed by >10%
    recommendation: str
    buckets: List[CalibrationBucket]


class ThompsonCalibrator:
    """
    Diagnostic and seed-fix layer for ThompsonSampler.

    On construction, seeds any missing signal-source keys from LEARNING_CONFIG
    into the sampler's distributions so that lookups return informative priors
    rather than falling back to Beta(1,1).

    The calibrate() method produces a SourceCalibration report per source and
    persists the result to calibration_report.json. Designed to run on a periodic
    cadence (e.g., daily) after outcomes have been collected.
    """

    def __init__(self, thompson_sampler, data_dir: Path = None):
        """
        Args:
            thompson_sampler: Live ThompsonSampler instance.
            data_dir: Override for data directory (default: DATA_DIR).
        """
        self._sampler = thompson_sampler
        self._data_dir = data_dir or DATA_DIR
        self._last_report: Optional[List[SourceCalibration]] = None

        seeds_written = self._seed_missing_keys()
        if seeds_written:
            logger.info(
                "ThompsonCalibrator: seeded %d missing distribution keys from LEARNING_CONFIG",
                seeds_written,
            )

    # ------------------------------------------------------------------
    # Seed fix
    # ------------------------------------------------------------------

    def _seed_missing_keys(self) -> int:
        """
        Idempotently seed signal-source-level keys from LEARNING_CONFIG.

        A key is considered already seeded when its alpha+beta sum exceeds
        _SEEDED_THRESHOLD (2.1), indicating it was set from a real reliability
        value rather than the uninformative Beta(1,1) prior.

        Returns:
            Number of keys newly seeded.
        """
        try:
            from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        except ImportError:
            logger.warning("ThompsonCalibrator: could not import LEARNING_CONFIG, skipping seed")
            return 0

        source_reliability: Dict[str, float] = LEARNING_CONFIG.get("source_reliability", {})
        prior_scale: float = self._sampler.prior_scale
        seeds_written = 0

        for key, reliability in source_reliability.items():
            dist_entry = self._sampler.distributions.get(key, {})
            default_params = dist_entry.get("default", {})
            current_alpha = default_params.get("alpha", 0.0)
            current_beta = default_params.get("beta", 0.0)
            current_sum = current_alpha + current_beta

            # A key is considered seeded if it exists AND its sum exceeds the
            # uninformative Beta(1,1) threshold of 2.0, OR if it was previously
            # written by us (alpha != beta at 1.0/1.0, i.e. non-uniform prior).
            # We skip re-seeding if alpha+beta > 2.1 (updated by Bayesian observations)
            # OR if the key already holds our seeded values (alpha == r*scale exactly).
            expected_alpha = reliability * self._sampler.prior_scale
            expected_beta = (1.0 - reliability) * self._sampler.prior_scale
            already_seeded = (
                current_sum > _SEEDED_THRESHOLD
                or (
                    abs(current_alpha - expected_alpha) < 1e-9
                    and abs(current_beta - expected_beta) < 1e-9
                )
            )
            if already_seeded:
                continue

            alpha = reliability * prior_scale
            beta = (1.0 - reliability) * prior_scale

            if key not in self._sampler.distributions:
                self._sampler.distributions[key] = {}

            self._sampler.distributions[key]["default"] = {
                "alpha": alpha,
                "beta": beta,
            }
            seeds_written += 1

        if seeds_written:
            try:
                self._sampler._save_distributions()
            except Exception:
                logger.warning("ThompsonCalibrator: failed to persist seeded distributions", exc_info=True)

        return seeds_written

    # ------------------------------------------------------------------
    # Calibration pipeline
    # ------------------------------------------------------------------

    def calibrate(self) -> List[SourceCalibration]:
        """
        Run the full calibration pipeline.

        Loads predictions and outcomes, joins them by prediction_id, groups
        by source, and computes per-source calibration metrics for sources
        with at least _MIN_PAIRS_FOR_CALIBRATION matched pairs.

        Returns:
            List of SourceCalibration dataclasses (one per qualifying source).
        """
        predictions = self._load_predictions()
        outcomes = self._load_outcomes()

        if not predictions or not outcomes:
            logger.info("ThompsonCalibrator.calibrate: no data to calibrate (predictions=%d, outcomes=%d)",
                        len(predictions), len(outcomes))
            return []

        # Index outcomes by prediction_id
        outcome_index: Dict[str, dict] = {}
        for o in outcomes:
            pid = o.get("prediction_id")
            if pid:
                outcome_index[pid] = o

        # Join and group by source
        # Predictions use "prediction_id", outcomes use "prediction_id"
        # Confidence field: predictions.jsonl uses "confidence" OR "predicted_confidence"
        # Outcome field: outcomes.jsonl uses "was_correct" (bool)
        by_source: Dict[str, List[Tuple[float, float]]] = {}  # source -> [(confidence, outcome)]

        for pred in predictions:
            pid = pred.get("prediction_id")
            if not pid or pid not in outcome_index:
                continue

            outcome = outcome_index[pid]
            confidence = pred.get("confidence") or pred.get("predicted_confidence")
            was_correct = outcome.get("was_correct")

            if confidence is None or was_correct is None:
                continue

            try:
                confidence = float(confidence)
                outcome_val = 1.0 if was_correct else 0.0
            except (TypeError, ValueError):
                continue

            # Source: prefer outcome.source, fall back to prediction source fields
            source = (
                outcome.get("source")
                or pred.get("source")
                or pred.get("prediction_source")
                or "unknown"
            )

            by_source.setdefault(source, []).append((confidence, outcome_val))

        calibrations: List[SourceCalibration] = []
        for source, pairs in by_source.items():
            if len(pairs) < _MIN_PAIRS_FOR_CALIBRATION:
                continue
            try:
                cal = self._compute_source_calibration(source, pairs)
                calibrations.append(cal)
            except Exception:
                logger.warning("ThompsonCalibrator: failed to calibrate source '%s'", source, exc_info=True)

        self._last_report = calibrations
        self._save_report()
        return calibrations

    def _compute_source_calibration(
        self, source: str, pairs: List[Tuple[float, float]]
    ) -> SourceCalibration:
        """
        Compute calibration metrics for a single source.

        Args:
            source: Source identifier string.
            pairs: List of (predicted_confidence, actual_outcome) tuples.

        Returns:
            SourceCalibration dataclass.
        """
        n = len(pairs)
        confidences = [p[0] for p in pairs]
        outcomes = [p[1] for p in pairs]

        brier_score = sum((c - o) ** 2 for c, o in pairs) / n
        mean_confidence = sum(confidences) / n
        mean_hit_rate = sum(outcomes) / n

        gap = mean_confidence - mean_hit_rate
        is_overconfident = gap > 0.10
        is_underconfident = gap < -0.10

        if is_overconfident:
            recommendation = (
                f"Source '{source}' is OVERCONFIDENT by {gap:.1%}: "
                f"predicted {mean_confidence:.2f} but hit rate is {mean_hit_rate:.2f}. "
                f"Consider reducing signal weight or tightening confidence thresholds."
            )
        elif is_underconfident:
            recommendation = (
                f"Source '{source}' is UNDERCONFIDENT by {abs(gap):.1%}: "
                f"predicted {mean_confidence:.2f} but hit rate is {mean_hit_rate:.2f}. "
                f"Model is leaving edge on the table — consider raising confidence scaling."
            )
        else:
            recommendation = (
                f"Source '{source}' is WELL CALIBRATED: "
                f"predicted {mean_confidence:.2f}, observed {mean_hit_rate:.2f} "
                f"(gap {gap:+.2f}, Brier {brier_score:.4f})."
            )

        # Build calibration buckets
        buckets: List[CalibrationBucket] = []
        for i in range(len(_BUCKET_EDGES) - 1):
            lo = _BUCKET_EDGES[i]
            hi = _BUCKET_EDGES[i + 1]
            bucket_pairs = [(c, o) for c, o in pairs if lo <= c < hi]
            count = len(bucket_pairs)
            if count == 0:
                predicted_mid = (lo + min(hi, 1.0)) / 2.0
                buckets.append(CalibrationBucket(
                    confidence_min=lo,
                    confidence_max=min(hi, 1.0),
                    predicted_confidence=predicted_mid,
                    observed_hit_rate=0.0,
                    sample_count=0,
                    brier_contribution=0.0,
                ))
                continue

            bucket_confidences = [p[0] for p in bucket_pairs]
            bucket_outcomes = [p[1] for p in bucket_pairs]
            avg_confidence = sum(bucket_confidences) / count
            observed_rate = sum(bucket_outcomes) / count
            brier_contrib = sum((c - o) ** 2 for c, o in bucket_pairs) / n  # Contribution to total

            buckets.append(CalibrationBucket(
                confidence_min=lo,
                confidence_max=min(hi, 1.0),
                predicted_confidence=avg_confidence,
                observed_hit_rate=observed_rate,
                sample_count=count,
                brier_contribution=brier_contrib,
            ))

        return SourceCalibration(
            source=source,
            brier_score=brier_score,
            mean_predicted_confidence=mean_confidence,
            mean_observed_hit_rate=mean_hit_rate,
            sample_count=n,
            is_overconfident=is_overconfident,
            is_underconfident=is_underconfident,
            recommendation=recommendation,
            buckets=buckets,
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_predictions(self) -> List[dict]:
        """Read predictions.jsonl from data_dir."""
        path = self._data_dir / "predictions.jsonl"
        records = []
        try:
            if not path.exists():
                return records
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            logger.warning("ThompsonCalibrator: could not read predictions.jsonl", exc_info=True)
        return records

    def _load_outcomes(self) -> List[dict]:
        """Read outcomes.jsonl from data_dir."""
        path = self._data_dir / "outcomes.jsonl"
        records = []
        try:
            if not path.exists():
                return records
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            logger.warning("ThompsonCalibrator: could not read outcomes.jsonl", exc_info=True)
        return records

    def _save_report(self) -> None:
        """Persist calibration report to calibration_report.json."""
        if not self._last_report:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            report = {
                "generated_at": datetime.now().isoformat(),
                "source_count": len(self._last_report),
                "sources": [asdict(cal) for cal in self._last_report],
            }
            CALIBRATION_PATH.write_text(json.dumps(report, indent=2))
            logger.info(
                "ThompsonCalibrator: wrote calibration report for %d sources to %s",
                len(self._last_report),
                CALIBRATION_PATH,
            )
        except OSError:
            logger.warning("ThompsonCalibrator: could not write calibration_report.json", exc_info=True)

    def save(self) -> None:
        """Public alias for _save_report — satisfies HolonProxy.act() delegation."""
        self._save_report()

    # ------------------------------------------------------------------
    # HolonProxy delegation
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """
        Return diagnostic statistics for HolonProxy.sense() delegation.

        Returns summary of last calibration run and current seed state.
        """
        seeded_keys = [
            k for k, v in self._sampler.distributions.items()
            if v.get("default", {}).get("alpha", 0.0) + v.get("default", {}).get("beta", 0.0) > _SEEDED_THRESHOLD
        ]

        report: dict = {
            "seeded_key_count": len(seeded_keys),
            "seeded_keys": seeded_keys,
            "calibration_report_exists": CALIBRATION_PATH.exists(),
        }

        if self._last_report:
            overconfident = [c.source for c in self._last_report if c.is_overconfident]
            underconfident = [c.source for c in self._last_report if c.is_underconfident]
            report.update({
                "calibrated_source_count": len(self._last_report),
                "overconfident_sources": overconfident,
                "underconfident_sources": underconfident,
                "mean_brier_score": (
                    sum(c.brier_score for c in self._last_report) / len(self._last_report)
                    if self._last_report else None
                ),
            })
        else:
            report["calibrated_source_count"] = 0

        return report

    def get_calibration_feedback(self) -> dict:
        """Bridge 5: Return structured feedback for meta-learning adjustments.

        Reads the last calibration report and returns recommended deltas
        for source_reliability in learning_config. Only includes sources
        that exist in source_reliability.

        Returns:
            {"overconfident": [...], "underconfident": [...]}
            Each item: {"key": str, "gap": float, "delta": float}
        """
        result: dict = {"overconfident": [], "underconfident": []}
        if not self._last_report:
            return result

        try:
            from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
            reliability = LEARNING_CONFIG.get("source_reliability", {})
        except ImportError:
            return result

        for cal in self._last_report:
            if cal.source not in reliability:
                continue

            gap = cal.mean_predicted_confidence - cal.mean_observed_hit_rate

            if cal.is_overconfident and gap > 0.10:
                delta = max(-0.05, gap * -0.3)  # Reduce, capped at -0.05
                result["overconfident"].append({
                    "key": cal.source, "gap": round(gap, 4),
                    "delta": round(delta, 4),
                })
            elif cal.is_underconfident and gap < -0.10:
                delta = min(0.05, gap * -0.3)  # Increase, capped at +0.05
                result["underconfident"].append({
                    "key": cal.source, "gap": round(gap, 4),
                    "delta": round(delta, 4),
                })

        return result
