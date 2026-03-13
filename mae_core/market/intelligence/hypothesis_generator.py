"""Hypothesis Generator — Converts lag-correlation findings into formal hypotheses.

Reads lag findings from data/market/lag_correlations.json, filters by
statistical significance, populates causal stories from templates,
deduplicates against the registry, and registers qualifying candidates.

Causal story templates are the anti-overfitting gate: if the pair has
no known causal mechanism, the hypothesis is flagged as requiring manual
review before promotion. This prevents the system from promoting
spurious correlations that happen to backtest well.

Generation cadence: every 500 steps (driven by HypothesisEngine).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

_PAIR_OUTCOMES_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pair_outcomes.json"

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_causal import (  # noqa: F401
    _get_gen_threshold,
    _auto_generate_causal_story,
    _get_causal_story,
    CAUSAL_STORY_TEMPLATES,
    _GEN_FALLBACKS,
    _DOMAIN_ROLES,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# Legacy module-level aliases for backward compatibility
MIN_CORRELATION = _GEN_FALLBACKS["min_correlation"]
MIN_PAIRS = _GEN_FALLBACKS["min_pairs"]


class HypothesisGenerator:
    """Generates hypotheses from lag-correlation findings.

    Reads lag_correlations.json, filters by quality thresholds,
    deduplicates against the registry, and registers new candidates.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        lag_data_path: Path = None,
        granger_data_path: Path = None,
    ):
        self._registry = registry
        self._lag_data_path = lag_data_path or (DATA_DIR / "lag_correlations.json")
        if granger_data_path is not None:
            self._granger_data_path = granger_data_path
        elif lag_data_path is not None:
            # When a custom lag path is provided (e.g. in tests), look for
            # granger file alongside it so tests stay isolated.
            self._granger_data_path = lag_data_path.parent / "granger_causality.json"
        else:
            self._granger_data_path = DATA_DIR / "granger_causality.json"

        # Derive pair outcomes path from registry's data_dir so tests using
        # tmp_path registries get isolated persistence (no cross-test contamination).
        registry_data_dir = getattr(registry, "_data_dir", None)
        if registry_data_dir is not None:
            self._pair_outcomes_path = Path(registry_data_dir) / "pair_outcomes.json"
        else:
            self._pair_outcomes_path = _PAIR_OUTCOMES_PATH

        # Bridge 5: pair quality memory — tracks which (source_a, source_b) produce
        # promoted vs retired hypotheses. Persisted across restarts.
        self._pair_outcomes: dict[tuple, dict] = {}
        self._load_pair_outcomes()

    def record_outcome(self, source_a: str, source_b: str, outcome: str) -> None:
        """Record whether a hypothesis from this pair was promoted or retired.

        Called by HypothesisEngine._promote() and _retire(). Builds a soft
        priority signal so better-understood pairs are investigated first.
        """
        pair = (source_a, source_b)
        if pair not in self._pair_outcomes:
            self._pair_outcomes[pair] = {"promoted": 0, "retired": 0}
        if outcome in ("promoted", "retired"):
            self._pair_outcomes[pair][outcome] += 1
        self.save_pair_outcomes()

    def save_pair_outcomes(self) -> None:
        """Persist pair quality memory to disk.

        Tuple keys are serialized as "source_a|source_b" strings (JSON only
        supports string keys). Non-critical — failures are logged but not raised.
        """
        tmp = self._pair_outcomes_path.with_suffix(".tmp")
        try:
            self._pair_outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            serialized = {
                f"{a}|{b}": counts
                for (a, b), counts in self._pair_outcomes.items()
            }
            tmp.write_text(json.dumps(serialized, indent=2))
            os.replace(tmp, self._pair_outcomes_path)
        except Exception as e:
            logger.debug("Failed to save pair outcomes: %s", e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _load_pair_outcomes(self) -> None:
        """Restore pair quality memory from disk.

        Pipe-separated string keys are split back to (source_a, source_b) tuples.
        Safe to call when the file is absent (first boot).
        """
        if not self._pair_outcomes_path.exists():
            return
        try:
            data = json.loads(self._pair_outcomes_path.read_text())
            self._pair_outcomes = {
                tuple(key.split("|", 1)): counts
                for key, counts in data.items()
                if "|" in key
            }
            logger.info(
                "Loaded pair outcomes: %d pairs tracked", len(self._pair_outcomes)
            )
        except Exception as e:
            logger.warning("Failed to load pair outcomes: %s", e)

    def generate(self) -> List[Hypothesis]:
        """Read lag findings, generate qualifying hypotheses.

        Findings are sorted by correlation strength + pair quality priority,
        so better-understood pairs are investigated first when many compete.
        Returns list of newly created hypotheses (bivariate + composite).
        """
        findings = self._load_lag_findings()
        # Arc 3: Also load Granger causal findings (directional evidence)
        granger_findings = self._load_granger_findings()
        findings.extend(granger_findings)
        if not findings:
            return []

        # Sort by correlation + pair quality priority (Bridge 5)
        findings.sort(key=self._finding_priority, reverse=True)

        # Collect qualifying findings before generating (needed for composites)
        qualifying: list[dict] = []
        new_hypotheses = []
        for finding in findings:
            hyp = self._finding_to_hypothesis(finding)
            if hyp is not None:
                self._registry.register(hyp)
                new_hypotheses.append(hyp)
                qualifying.append(finding)

        # Generate composite (multi-factor) hypotheses from qualifying findings
        composite_hypotheses = self._generate_composites(qualifying)
        new_hypotheses.extend(composite_hypotheses)

        if new_hypotheses:
            logger.info(
                "HypothesisGenerator: %d new hypotheses (%d bivariate, %d composite) "
                "from %d lag findings",
                len(new_hypotheses),
                len(new_hypotheses) - len(composite_hypotheses),
                len(composite_hypotheses),
                len(findings),
            )
        return new_hypotheses

    def _finding_to_hypothesis(self, finding: dict) -> Optional[Hypothesis]:
        """Convert a single lag finding to a hypothesis, or None if filtered."""
        source_a = finding.get("source_a", "")
        source_b = finding.get("source_b", "")
        lag_days = finding.get("lag_days", 0)
        correlation = finding.get("correlation", 0.0)
        n_pairs = finding.get("n_pairs", 0)
        direction = finding.get("direction", "")

        # Quality filters (read live from config, fall back to hardcoded)
        if abs(correlation) < _get_gen_threshold("min_correlation"):
            return None
        if n_pairs < _get_gen_threshold("min_pairs"):
            return None

        # Dedup: check if similar hypothesis already exists
        existing = self._registry.find_by_trigger(source_a, source_b, lag_days)
        if existing is not None:
            return None

        # Build hypothesis
        name = f"{source_a}→{source_b} (lag {lag_days}d, r={correlation:.3f})"
        causal_story = _get_causal_story(source_a, source_b)

        return Hypothesis(
            name=name,
            trigger=TriggerPattern(
                source_a=source_a,
                source_b=source_b,
                lag_days=lag_days,
                direction=direction,
            ),
            causal_story=causal_story,
            source_type=SourceType.LAG_CORRELATION,
            parent_lag_finding=finding,
        )

    def _generate_composites(self, qualifying: list[dict]) -> List[Hypothesis]:
        """Generate composite (multi-factor) hypotheses from qualifying findings.

        Groups findings by source_b. When two or more independently qualifying
        findings share the same source_b, generates a composite hypothesis
        requiring BOTH leading sources to co-fire. This captures "A AND C
        together predict B" patterns missed by bivariate hypotheses.

        The composite lag is the shorter of the two constituent lags (both
        must fire within that window for the pattern to trigger).
        """
        # Group qualifying findings by source_b
        by_target: dict[str, list[dict]] = {}
        for f in qualifying:
            target = f.get("source_b", "")
            if not target:
                continue
            by_target.setdefault(target, []).append(f)

        new_composites: List[Hypothesis] = []
        for source_b, group in by_target.items():
            if len(group) < 2:
                continue

            # Generate all pairs within the group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    f_a = group[i]
                    f_c = group[j]
                    hyp = self._findings_to_composite(f_a, f_c)
                    if hyp is not None:
                        self._registry.register(hyp)
                        new_composites.append(hyp)

        return new_composites

    def _findings_to_composite(
        self,
        finding_a: dict,
        finding_c: dict,
    ) -> Optional[Hypothesis]:
        """Convert two findings sharing source_b into a composite hypothesis.

        Returns None if the composite is a duplicate of an existing hypothesis.
        """
        source_a = finding_a.get("source_a", "")
        source_c = finding_c.get("source_a", "")
        source_b = finding_a.get("source_b", "")

        if not source_a or not source_c or not source_b:
            return None

        # Dedup: check composite signature (order of a/c doesn't matter)
        if self._is_duplicate_composite(source_a, source_c, source_b):
            return None

        lag_a = finding_a.get("lag_days", 0)
        lag_c = finding_c.get("lag_days", 0)
        composite_lag = min(lag_a, lag_c)  # Both must fire within the shorter window

        corr_a = finding_a.get("correlation", 0.0)
        corr_c = finding_c.get("correlation", 0.0)

        # Causal story for the composite
        story_a = _get_causal_story(source_a, source_b)
        story_c = _get_causal_story(source_c, source_b)
        composite_story = (
            f"[COMPOSITE] Multi-factor pattern: {source_a} AND {source_c} together "
            f"predict {source_b}. "
            f"{source_a} story: {story_a[:120].rstrip()}... "
            f"{source_c} story: {story_c[:120].rstrip()}..."
        )

        name = (
            f"[{source_a}+{source_c}]→{source_b} "
            f"(lag {composite_lag}d, r_a={corr_a:.3f}, r_c={corr_c:.3f})"
        )

        return Hypothesis(
            name=name,
            trigger=TriggerPattern(
                source_a=source_a,
                source_b=source_b,
                lag_days=composite_lag,
                direction=finding_a.get("direction", ""),
                conjunct_source=source_c,
            ),
            causal_story=composite_story,
            source_type=SourceType.LAG_CORRELATION,
            parent_lag_finding={
                "composite": True,
                "finding_a": finding_a,
                "finding_c": finding_c,
            },
        )

    def _is_duplicate_composite(
        self, source_a: str, conjunct_source: str, source_b: str
    ) -> bool:
        """Check if a composite hypothesis already exists with this signature.

        Order of source_a and conjunct_source doesn't matter — (A+C→B) is
        the same as (C+A→B).
        """
        for hyp in self._registry.get_all():
            if hyp.status.value == "retired":
                continue
            t = hyp.trigger
            if t.source_b != source_b:
                continue
            if not t.conjunct_source:
                continue  # Skip bivariate hypotheses
            # Canonical pair: sorted tuple so order doesn't matter
            existing_pair = frozenset({t.source_a, t.conjunct_source})
            candidate_pair = frozenset({source_a, conjunct_source})
            if existing_pair == candidate_pair:
                return True
        return False

    def _load_granger_findings(self) -> list:
        """Load Granger causality findings and convert to lag correlation format.

        Maps Granger fields to the same dict shape as lag_correlations.json so
        _finding_to_hypothesis() can process them without branching.
        """
        granger_path = self._granger_data_path
        if not granger_path.exists():
            return []
        try:
            granger_data = json.loads(granger_path.read_text())
            if not isinstance(granger_data, list):
                return []
            results = []
            for entry in granger_data:
                p_value = entry.get("p_value", 1.0)
                converted = {
                    "source_a": entry.get("cause_source", ""),
                    "source_b": entry.get("effect_source", ""),
                    "lag_days": entry.get("best_lag", 0),
                    "correlation": max(0.5, 1.0 - p_value * 5),
                    "n_pairs": 10,
                    "direction": entry.get("direction", ""),
                    "evidence_type": "granger",
                }
                if converted["source_a"] and converted["source_b"]:
                    results.append(converted)
            return results
        except Exception as e:
            logger.warning("Failed to load granger_causality.json: %s", e)
            return []

    def _load_lag_findings(self) -> list:
        """Load lag-correlation findings from disk."""
        findings = []

        if not self._lag_data_path.exists():
            logger.debug("No lag correlations file at %s", self._lag_data_path)
        else:
            try:
                data = json.loads(self._lag_data_path.read_text())
                if isinstance(data, list):
                    findings.extend(data)
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Failed to load lag correlations: %s", e)

        return findings

    def _finding_priority(self, finding: dict) -> float:
        """Compute sort priority for a lag finding. Higher = processed first.

        Base is |correlation|. Known-good pairs (more promotions than retirements)
        get +0.1 bonus. This is ordering only, not filtering.
        """
        corr = abs(finding.get("correlation", 0.0))
        pair = (finding.get("source_a", ""), finding.get("source_b", ""))
        outcomes = self._pair_outcomes.get(pair, {})
        promoted = outcomes.get("promoted", 0)
        retired = outcomes.get("retired", 0)
        bonus = 0.1 if promoted > retired and promoted > 0 else 0.0
        return corr + bonus

    def get_statistics(self) -> dict:
        """Summary stats for monitoring."""
        min_corr = _get_gen_threshold("min_correlation")
        min_pairs = _get_gen_threshold("min_pairs")
        findings = self._load_lag_findings()
        qualifying = [
            f for f in findings
            if abs(f.get("correlation", 0)) >= min_corr
            and f.get("n_pairs", 0) >= min_pairs
        ]
        return {
            "total_lag_findings": len(findings),
            "qualifying_findings": len(qualifying),
            "causal_templates": len(CAUSAL_STORY_TEMPLATES),
            "live_min_correlation": min_corr,
            "live_min_pairs": min_pairs,
            "pair_outcomes_tracked": len(self._pair_outcomes),
        }
