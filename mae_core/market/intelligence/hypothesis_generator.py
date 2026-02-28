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
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# ── Minimum thresholds for hypothesis generation ────────────────────
# Fallback values if learning_config is unavailable
_GEN_FALLBACKS = {"min_correlation": 0.6, "min_pairs": 10}

# Legacy module-level aliases for backward compatibility
MIN_CORRELATION = _GEN_FALLBACKS["min_correlation"]
MIN_PAIRS = _GEN_FALLBACKS["min_pairs"]


def _get_gen_threshold(key: str) -> float:
    """Read a generator threshold from learning_config, with fallback.

    Graceful degradation: if config import fails or key is missing,
    returns the hardcoded fallback value.
    """
    try:
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        cfg = LEARNING_CONFIG.get("generator_thresholds", {})
        return float(cfg.get(key, _GEN_FALLBACKS.get(key, 0.0)))
    except ImportError:
        return float(_GEN_FALLBACKS.get(key, 0.0))

# ── Causal story templates ──────────────────────────────────────────
# Maps (source_a, source_b) to explanatory text. Order-independent:
# the generator checks both (a,b) and (b,a).
#
# Pairs WITHOUT a template get a default "requires manual review" story,
# which blocks promotion until a human or future analysis adds one.

CAUSAL_STORY_TEMPLATES = {
    ("sec_form4", "finnhub_earnings"): (
        "Insider trading activity (Form 4) precedes earnings announcements. "
        "Insiders have material non-public information about upcoming results. "
        "Lakonishok & Lee (2001) demonstrated insider alpha persists 3-12 months."
    ),
    ("congressional", "finra_short"): (
        "Congressional trades precede short interest changes. Members with "
        "committee access may trade ahead of regulatory or legislative actions "
        "that affect short sellers. STOCK Act disclosure lag creates "
        "information asymmetry."
    ),
    ("finra_short", "finnhub_earnings"): (
        "Short interest changes precede earnings surprises. Short sellers "
        "conduct deep fundamental analysis; rising short interest before "
        "earnings correlates with negative surprises (Desai et al. 2002)."
    ),
    ("sec_form4", "sec_efts"): (
        "Insider trades precede SEC filing activity. Insiders may trade "
        "before major filings are submitted. SEC EFTS keyword matches "
        "catch the filing; Form 4 catches the trade."
    ),
    ("sec_efts", "congressional"): (
        "SEC filing text mentions precede congressional trades. Material "
        "events disclosed in SEC filings (contracts, M&A, regulatory) may "
        "inform committee members' trading decisions."
    ),
    ("finra_short", "congressional"): (
        "Short interest changes precede congressional trading. Short sellers' "
        "public positions signal market sentiment that members observe "
        "before trading."
    ),
    ("insider_cluster", "finnhub_earnings"): (
        "Insider buying clusters (3+ officers) precede earnings outcomes. "
        "Alldredge (2019) showed cluster alpha peaks 40-80 days. Coordinated "
        "insider buying signals management conviction about upcoming results."
    ),
    ("sec_form8k", "sec_form4"): (
        "Material events (8-K) trigger subsequent insider trades. After a "
        "material event disclosure, insiders trade on their assessment of "
        "the market's reaction efficiency."
    ),
    ("hiring_tracker", "contract_award"): (
        "Hiring surges precede government contract awards. Companies hire "
        "before contract execution begins. Hiring lead time is 60-120 days "
        "before contract announcement."
    ),
    ("sam_gov", "contract_award"): (
        "SAM.gov opportunity postings lead to contract awards. The "
        "solicitation → proposal → evaluation → award pipeline has "
        "predictable timing by agency and contract type."
    ),
    ("fred_macro", "finra_short"): (
        "Macroeconomic indicator changes precede short interest shifts. "
        "FRED data releases (GDP, CPI, unemployment) signal regime changes "
        "that drive short positioning."
    ),
}


def _get_causal_story(source_a: str, source_b: str) -> str:
    """Look up causal story for a source pair (order-independent)."""
    story = CAUSAL_STORY_TEMPLATES.get((source_a, source_b))
    if story:
        return story
    story = CAUSAL_STORY_TEMPLATES.get((source_b, source_a))
    if story:
        return story
    return (
        f"REQUIRES MANUAL REVIEW: Statistical correlation found between "
        f"{source_a} and {source_b} but no known causal mechanism. "
        f"Do not promote until a causal story is established."
    )


class HypothesisGenerator:
    """Generates hypotheses from lag-correlation findings.

    Reads lag_correlations.json, filters by quality thresholds,
    deduplicates against the registry, and registers new candidates.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        lag_data_path: Path = None,
    ):
        self._registry = registry
        self._lag_data_path = lag_data_path or (DATA_DIR / "lag_correlations.json")

    def generate(self) -> List[Hypothesis]:
        """Read lag findings, generate qualifying hypotheses.

        Returns list of newly created hypotheses.
        """
        findings = self._load_lag_findings()
        if not findings:
            return []

        new_hypotheses = []
        for finding in findings:
            hyp = self._finding_to_hypothesis(finding)
            if hyp is not None:
                self._registry.register(hyp)
                new_hypotheses.append(hyp)

        if new_hypotheses:
            logger.info(
                "HypothesisGenerator: %d new hypotheses from %d lag findings",
                len(new_hypotheses), len(findings),
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

        # Quality filters
        if abs(correlation) < MIN_CORRELATION:
            return None
        if n_pairs < MIN_PAIRS:
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

    def _load_lag_findings(self) -> list:
        """Load lag-correlation findings from disk."""
        if not self._lag_data_path.exists():
            logger.debug("No lag correlations file at %s", self._lag_data_path)
            return []

        try:
            data = json.loads(self._lag_data_path.read_text())
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to load lag correlations: %s", e)
            return []

    def get_statistics(self) -> dict:
        """Summary stats for monitoring."""
        findings = self._load_lag_findings()
        qualifying = [
            f for f in findings
            if abs(f.get("correlation", 0)) >= MIN_CORRELATION
            and f.get("n_pairs", 0) >= MIN_PAIRS
        ]
        return {
            "total_lag_findings": len(findings),
            "qualifying_findings": len(qualifying),
            "causal_templates": len(CAUSAL_STORY_TEMPLATES),
        }
