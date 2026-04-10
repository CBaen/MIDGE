#!/usr/bin/env python3
"""Pattern Auditor — trust scoring for pattern templates.

Every pattern MIDGE discovers needs to earn its place.  High win rate
on a small sample of similar trades by the same insider is overfit noise,
not edge.  This auditor scores each template on six independent dimensions
and produces a trust verdict.

Trust scoring (max 100 points):
  +30  Statistical significance (binomial test, p < 0.05)
  +20  Cross-instrument validation (3+ instruments)
  +15  Regime robustness (works in 2+ regimes)
  +15  Recency (instances in last 90 days)
  +10  Entity diversity (3+ different actors)
  +10  Instance count (10+)
  -20  Overfitting risk (win_rate > 90% with < 5 instances)

Verdicts:
  80-100: "trust"    — deploy capital weight
  60-79:  "watch"   — wait for more observations
  40-59:  "suspect" — needs more evidence
  0-39:   "retire"  — likely noise, pull from live pipeline

Biological analogy: this is MIDGE's immune system for bad patterns —
distinguishing self (real edge) from non-self (overfitted noise).

Data sources:
  - data/market/pattern_templates.jsonl  — template statistics
  - data/midge/cross_instrument_validation.jsonl  — cross-symbol evidence
  - data/midge/entity_profiles.jsonl  — entity diversity (via EntityTracker)
  - data/market/outcomes.jsonl  — for regime lookup if available

Output: data/midge/pattern_audit_reports.jsonl
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATES_PATH = _PROJECT_ROOT / "data" / "market" / "pattern_templates.jsonl"
_CROSS_INSTRUMENT_PATH = _PROJECT_ROOT / "data" / "midge" / "cross_instrument_validation.jsonl"
_ENTITY_PROFILES_PATH = _PROJECT_ROOT / "data" / "midge" / "entity_profiles.jsonl"
_OUTCOMES_PATH = _PROJECT_ROOT / "data" / "market" / "outcomes.jsonl"
_AUDIT_OUTPUT_PATH = _PROJECT_ROOT / "data" / "midge" / "pattern_audit_reports.jsonl"

# Trust thresholds
TRUST_THRESHOLD = 80
WATCH_THRESHOLD = 60
SUSPECT_THRESHOLD = 40

# Overfitting: win rate > this with fewer than this many instances
OVERFIT_WIN_RATE_THRESHOLD = 0.90
OVERFIT_MIN_INSTANCES = 5

# Recency window
RECENCY_DAYS = 90


@dataclass
class PatternAuditReport:
    """Trust audit report for a single pattern template.

    Scores each template on 6 independent dimensions to produce a
    composite trust_score (0-100) and a verdict.
    """
    template_id: str
    direction: str
    domain_signature: str
    trust_score: int                    # 0-100
    statistical_significance: float    # p-value from binomial test (lower = more significant)
    regime_robust: bool                 # Works in bull AND bear?
    cross_instrument: bool              # Validated on 3+ instruments?
    recency_score: float                # 0-1: fraction of instances in last 90 days
    entity_diversity: float             # 0-1: unique actors / total appearances
    overfitting_risk: str               # "low", "medium", "high"
    recommendation: str                 # "trust", "watch", "suspect", "retire"
    # Breakdown of points awarded
    score_breakdown: dict = field(default_factory=dict)
    # Raw template stats for reference
    n_instances: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    cross_validated_symbols: int = 0
    unique_entities: int = 0
    audited_at: str = ""

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "direction": self.direction,
            "domain_signature": self.domain_signature,
            "trust_score": self.trust_score,
            "statistical_significance": self.statistical_significance,
            "regime_robust": self.regime_robust,
            "cross_instrument": self.cross_instrument,
            "recency_score": self.recency_score,
            "entity_diversity": self.entity_diversity,
            "overfitting_risk": self.overfitting_risk,
            "recommendation": self.recommendation,
            "score_breakdown": self.score_breakdown,
            "n_instances": self.n_instances,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "cross_validated_symbols": self.cross_validated_symbols,
            "unique_entities": self.unique_entities,
            "audited_at": self.audited_at,
        }


class PatternAuditor:
    """Audits pattern templates for trustworthiness.

    Usage:
        auditor = PatternAuditor()
        report = auditor.audit_template("tmpl_abc123")
        all_reports = auditor.audit_all()
        trusted = auditor.get_trusted_templates(min_score=60)
    """

    def __init__(self):
        self._templates: Dict[str, dict] = {}
        self._cross_instrument_data: Dict[str, dict] = {}  # template_id -> {symbols: [...]}
        self._entity_appearances: Dict[str, List[str]] = {}  # template_id -> [entity_names]
        self._outcome_regimes: Dict[str, List[str]] = {}  # template_id -> [regime_names]
        self._outcome_dates: Dict[str, List[str]] = {}    # template_id -> [date strings]
        self._load_all_data()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def audit_template(self, template_id: str) -> Optional[PatternAuditReport]:
        """Full trust audit of a single pattern template.

        Args:
            template_id: The template to audit.

        Returns:
            PatternAuditReport, or None if template not found.
        """
        template = self._templates.get(template_id)
        if not template:
            return None

        return self._score_template(template_id, template)

    def audit_all(self) -> List[PatternAuditReport]:
        """Audit every template, return sorted by trust score descending.

        Returns:
            List of PatternAuditReport.
        """
        reports = []
        for tid, tmpl in self._templates.items():
            try:
                report = self._score_template(tid, tmpl)
                reports.append(report)
            except Exception:
                logger.debug("Audit failed for template %s", tid, exc_info=True)
        reports.sort(key=lambda r: r.trust_score, reverse=True)
        return reports

    def get_trusted_templates(self, min_score: int = WATCH_THRESHOLD) -> List[PatternAuditReport]:
        """Return only templates that pass the trust threshold.

        Args:
            min_score: Minimum trust_score to include (default: 60 = "watch" tier).

        Returns:
            List of PatternAuditReport with trust_score >= min_score.
        """
        return [r for r in self.audit_all() if r.trust_score >= min_score]

    def save_reports(self, reports: List[PatternAuditReport] = None) -> None:
        """Write all audit results to data/midge/pattern_audit_reports.jsonl.

        Args:
            reports: If None, runs audit_all() first.
        """
        if reports is None:
            reports = self.audit_all()

        _AUDIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _AUDIT_OUTPUT_PATH.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for r in reports:
                    f.write(json.dumps(r.to_dict(), default=str) + "\n")
            tmp.replace(_AUDIT_OUTPUT_PATH)
            logger.info(
                "PatternAuditor saved %d audit reports to %s",
                len(reports), _AUDIT_OUTPUT_PATH,
            )
        except Exception:
            logger.warning("PatternAuditor save failed", exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def get_statistics(self) -> dict:
        """Summary of audit results."""
        reports = self.audit_all()
        if not reports:
            return {"total_templates": 0}

        by_recommendation: dict = {}
        for r in reports:
            by_recommendation[r.recommendation] = by_recommendation.get(r.recommendation, 0) + 1

        return {
            "total_templates": len(reports),
            "avg_trust_score": round(sum(r.trust_score for r in reports) / len(reports), 1),
            "by_recommendation": by_recommendation,
            "trusted": sum(1 for r in reports if r.trust_score >= TRUST_THRESHOLD),
            "watchable": sum(1 for r in reports if WATCH_THRESHOLD <= r.trust_score < TRUST_THRESHOLD),
            "suspect": sum(1 for r in reports if SUSPECT_THRESHOLD <= r.trust_score < WATCH_THRESHOLD),
            "retire": sum(1 for r in reports if r.trust_score < SUSPECT_THRESHOLD),
        }

    # ------------------------------------------------------------------
    # Scoring engine
    # ------------------------------------------------------------------

    def _score_template(self, template_id: str, template: dict) -> PatternAuditReport:
        """Compute the trust score for one template."""
        wins = int(template.get("wins", 0))
        losses = int(template.get("losses", 0))
        n_total = wins + losses
        win_rate = (wins / n_total) if n_total > 0 else 0.0
        direction = template.get("direction", "")
        domain_sig = template.get("domain_signature", "")
        unique_symbols = template.get("unique_symbols", [])
        if isinstance(unique_symbols, int):
            n_symbols = unique_symbols
        else:
            n_symbols = len(unique_symbols) if unique_symbols else 0

        score = 0
        breakdown: dict = {}

        # ----------------------------------------------------------------
        # +30: Statistical significance (binomial test, p < 0.05)
        # Null hypothesis: win_rate = 0.50 (random)
        # Using exact binomial CDF
        # ----------------------------------------------------------------
        p_value = self._binomial_p_value(wins, n_total, null_p=0.50)
        sig_points = 0
        if n_total >= 5 and p_value < 0.05:
            sig_points = 30
        elif n_total >= 5 and p_value < 0.10:
            sig_points = 15
        score += sig_points
        breakdown["statistical_significance"] = sig_points

        # ----------------------------------------------------------------
        # +20: Cross-instrument validation (3+ unique symbols)
        # ----------------------------------------------------------------
        cross_data = self._cross_instrument_data.get(template_id, {})
        cross_symbols = cross_data.get("symbols_validated", n_symbols)
        if isinstance(cross_symbols, list):
            cross_symbols = len(cross_symbols)
        cross_points = 20 if cross_symbols >= 3 else (10 if cross_symbols >= 2 else 0)
        score += cross_points
        breakdown["cross_instrument"] = cross_points

        # ----------------------------------------------------------------
        # +15: Regime robustness (outcomes in 2+ different regimes)
        # ----------------------------------------------------------------
        regimes = set(self._outcome_regimes.get(template_id, []))
        regime_points = 15 if len(regimes) >= 2 else (7 if len(regimes) == 1 else 0)
        score += regime_points
        breakdown["regime_robust"] = regime_points

        # ----------------------------------------------------------------
        # +15: Recency (instances in last 90 days)
        # ----------------------------------------------------------------
        recent_dates = self._outcome_dates.get(template_id, [])
        cutoff = (date.today() - timedelta(days=RECENCY_DAYS)).isoformat()
        recent_count = sum(1 for d in recent_dates if d >= cutoff)
        recency_score = (recent_count / n_total) if n_total > 0 else 0.0
        recency_points = 0
        if recency_score >= 0.5:
            recency_points = 15
        elif recency_score >= 0.2:
            recency_points = 8
        score += recency_points
        breakdown["recency"] = recency_points

        # ----------------------------------------------------------------
        # +10: Entity diversity (3+ different actors appearing in template)
        # ----------------------------------------------------------------
        entity_names = self._entity_appearances.get(template_id, [])
        unique_actors = len(set(entity_names))
        entity_points = 10 if unique_actors >= 3 else (5 if unique_actors >= 2 else 0)
        entity_diversity = (unique_actors / len(entity_names)) if entity_names else 0.0
        score += entity_points
        breakdown["entity_diversity"] = entity_points

        # ----------------------------------------------------------------
        # +10: Instance count (10+)
        # ----------------------------------------------------------------
        instance_points = 10 if n_total >= 10 else (5 if n_total >= 5 else 0)
        score += instance_points
        breakdown["instance_count"] = instance_points

        # ----------------------------------------------------------------
        # -20: Overfitting detected (win_rate > 90% with < 5 instances)
        # ----------------------------------------------------------------
        overfit_risk = "low"
        if win_rate > OVERFIT_WIN_RATE_THRESHOLD and n_total < OVERFIT_MIN_INSTANCES:
            score -= 20
            overfit_risk = "high"
            breakdown["overfitting_penalty"] = -20
        elif win_rate > 0.80 and n_total < 8:
            overfit_risk = "medium"

        score = max(0, min(100, score))

        # Verdict
        if score >= TRUST_THRESHOLD:
            recommendation = "trust"
        elif score >= WATCH_THRESHOLD:
            recommendation = "watch"
        elif score >= SUSPECT_THRESHOLD:
            recommendation = "suspect"
        else:
            recommendation = "retire"

        return PatternAuditReport(
            template_id=template_id,
            direction=direction,
            domain_signature=domain_sig,
            trust_score=score,
            statistical_significance=p_value,
            regime_robust=len(regimes) >= 2,
            cross_instrument=cross_symbols >= 3,
            recency_score=round(recency_score, 3),
            entity_diversity=round(entity_diversity, 3),
            overfitting_risk=overfit_risk,
            recommendation=recommendation,
            score_breakdown=breakdown,
            n_instances=n_total,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            cross_validated_symbols=cross_symbols if isinstance(cross_symbols, int) else len(cross_symbols),
            unique_entities=unique_actors,
            audited_at=datetime.now().isoformat(),
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _binomial_p_value(successes: int, n: int, null_p: float = 0.5) -> float:
        """One-sided binomial p-value for win_rate > null_p.

        Uses normal approximation for large n, exact calculation for small n.
        Returns 1.0 if insufficient data.
        """
        if n < 2:
            return 1.0
        if successes == 0:
            return 1.0

        try:
            # Normal approximation (works well for n >= 10)
            if n >= 10:
                mean = n * null_p
                std = math.sqrt(n * null_p * (1 - null_p))
                if std == 0:
                    return 1.0
                z = (successes - 0.5 - mean) / std  # Continuity correction
                # Standard normal CDF (1 - Φ(z))
                p_value = 0.5 * math.erfc(z / math.sqrt(2))
                return max(0.001, min(1.0, p_value))
            else:
                # Exact binomial for small n
                # P(X >= successes | n, null_p) using CDF
                p_value = 0.0
                for k in range(successes, n + 1):
                    binom_coeff = math.comb(n, k)
                    p_value += binom_coeff * (null_p ** k) * ((1 - null_p) ** (n - k))
                return max(0.001, min(1.0, p_value))
        except (OverflowError, ValueError):
            return 1.0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_all_data(self) -> None:
        """Load templates, cross-instrument validation, entity profiles, outcomes."""
        self._load_templates()
        self._load_cross_instrument()
        self._load_entity_profiles()
        self._load_outcomes()

    def _load_templates(self) -> None:
        """Load pattern templates from JSONL."""
        if not _TEMPLATES_PATH.exists():
            return
        try:
            with open(_TEMPLATES_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        t = json.loads(line)
                        tid = t.get("template_id", "")
                        if tid:
                            self._templates[tid] = t
                    except json.JSONDecodeError:
                        continue
            logger.info("PatternAuditor loaded %d templates", len(self._templates))
        except Exception:
            logger.debug("Template load failed", exc_info=True)

    def _load_cross_instrument(self) -> None:
        """Load cross-instrument validation data."""
        if not _CROSS_INSTRUMENT_PATH.exists():
            return
        try:
            with open(_CROSS_INSTRUMENT_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        tid = rec.get("template_id", "")
                        if tid:
                            if tid not in self._cross_instrument_data:
                                self._cross_instrument_data[tid] = {
                                    "symbols_validated": set(),
                                }
                            symbols = rec.get("symbols_validated", [])
                            if isinstance(symbols, list):
                                self._cross_instrument_data[tid]["symbols_validated"].update(symbols)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("Cross-instrument load failed", exc_info=True)

        # Convert sets to counts
        for tid, data in self._cross_instrument_data.items():
            svs = data.get("symbols_validated", set())
            if isinstance(svs, set):
                data["symbols_validated"] = len(svs)

    def _load_entity_profiles(self) -> None:
        """Load entity profiles to extract per-template entity diversity."""
        if not _ENTITY_PROFILES_PATH.exists():
            return
        try:
            with open(_ENTITY_PROFILES_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        name = rec.get("name", "")
                        patterns = rec.get("patterns_involved", [])
                        if name and patterns:
                            for tid in patterns:
                                if tid not in self._entity_appearances:
                                    self._entity_appearances[tid] = []
                                self._entity_appearances[tid].append(name)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("Entity profile load failed", exc_info=True)

    def _load_outcomes(self) -> None:
        """Load outcomes to extract regime and date info per template."""
        if not _OUTCOMES_PATH.exists():
            return
        try:
            with open(_OUTCOMES_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        # Outcomes carry template_ids in convergence alert metadata
                        template_ids = rec.get("template_ids", [])
                        if not template_ids:
                            # Some outcomes reference template via signal_ids
                            signal_meta = rec.get("signal_metadata", {})
                            template_ids = signal_meta.get("template_ids", [])

                        regime = rec.get("regime", "")
                        pred_at = rec.get("predicted_at", "")[:10] if rec.get("predicted_at") else ""

                        for tid in template_ids:
                            if not tid:
                                continue
                            if regime:
                                if tid not in self._outcome_regimes:
                                    self._outcome_regimes[tid] = []
                                self._outcome_regimes[tid].append(regime)
                            if pred_at:
                                if tid not in self._outcome_dates:
                                    self._outcome_dates[tid] = []
                                self._outcome_dates[tid].append(pred_at)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("Outcome load failed", exc_info=True)
