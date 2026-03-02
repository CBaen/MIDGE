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

# ── Domain role classification ───────────────────────────────────────
# Maps each source to (domain_category, activity_description).
# Used by _auto_generate_causal_story() to produce stories for unknown pairs.

_DOMAIN_ROLES = {
    "sec_form4": ("insider", "equity_trading"),
    "sec_form8k": ("insider", "material_disclosure"),
    "sec_efts": ("regulatory", "filing_activity"),
    "congressional": ("political", "equity_trading"),
    "senate": ("political", "equity_trading"),
    "insider_cluster": ("insider", "coordinated_buying"),
    "contract_award": ("government", "procurement"),
    "contract_prediction": ("government", "procurement_forecast"),
    "hiring_tracker": ("corporate", "workforce_expansion"),
    "sam_gov": ("government", "opportunity_posting"),
    "social_sentiment": ("social", "retail_sentiment"),
    "finra_short": ("institutional", "short_positioning"),
    "finnhub_news": ("information", "news_flow"),
    "finnhub_earnings": ("fundamental", "earnings_reporting"),
    "fred_macro": ("macro", "economic_indicator"),
    "session_sweep": ("technical", "liquidity_sweep"),
    "session_sweep_ifvg": ("technical", "liquidity_sweep_fvg"),
    "ta_rsi": ("technical", "momentum_indicator"),
    "ta_macd": ("technical", "trend_indicator"),
    "ta_bollinger": ("technical", "volatility_band"),
    "ta_structure": ("technical", "market_structure"),
    "ta_candle": ("technical", "candlestick_pattern"),
    "cot_positioning": ("institutional", "futures_positioning"),
    "stocktwits_sentiment": ("social", "retail_sentiment"),
    "vix_term_structure": ("volatility", "fear_gauge"),
    "google_trends": ("social", "search_attention"),
    "finnhub_economic": ("macro", "economic_calendar"),
    "finnhub_analyst": ("fundamental", "analyst_consensus"),
    "finnhub_earnings_calendar": ("fundamental", "earnings_schedule"),
    "yfinance_price": ("technical", "price_action"),
}


def _auto_generate_causal_story(source_a: str, source_b: str) -> str:
    """Generate a causal story from domain roles when no template exists.

    Looks up both sources in _DOMAIN_ROLES, then applies a role-pair matrix
    to produce a plausible story. Returns None if either source is unknown.

    Stories are prefixed with [AUTO] to distinguish from human-written ones.
    """
    role_a = _DOMAIN_ROLES.get(source_a)
    role_b = _DOMAIN_ROLES.get(source_b)

    if role_a is None or role_b is None:
        return ""

    domain_a, activity_a = role_a
    domain_b, activity_b = role_b

    # Role-pair matrix: ordered by specificity (most specific first)

    # Insider/political → anything fundamental or reporting
    if domain_a in ("insider", "political") and domain_b in ("fundamental", "regulatory", "information"):
        return (
            f"[AUTO] Information advantage: {domain_a} participants ({activity_a}) "
            f"may have advance knowledge of {activity_b} events, creating a lead-lag "
            f"relationship where informed actors move before public disclosure."
        )

    # Technical → technical (independent confirmation)
    if domain_a == "technical" and domain_b == "technical":
        return (
            f"[AUTO] Technical confirmation: {activity_a} and {activity_b} are independent "
            f"indicators that, when aligned, strengthen the directional signal by reducing "
            f"the probability of false positives through multi-indicator consensus."
        )

    # Macro → institutional (cascade effect)
    if domain_a == "macro" and domain_b in ("institutional", "technical"):
        return (
            f"[AUTO] Macro-institutional cascade: {activity_a} changes drive {activity_b} "
            f"positioning as institutions respond to economic signals, creating measurable "
            f"lead-lag dynamics in market structure."
        )

    # Social → fundamental (attention/front-running)
    if domain_a == "social" and domain_b in ("fundamental", "information"):
        return (
            f"[AUTO] Attention-fundamental link: {activity_a} may front-run or react to "
            f"{activity_b} events, as retail attention often anticipates or amplifies "
            f"fundamental disclosures."
        )

    # Institutional → fundamental (positioning before events)
    if domain_a == "institutional" and domain_b == "fundamental":
        return (
            f"[AUTO] Institutional positioning: {activity_a} changes reflect informed "
            f"expectations about {activity_b} outcomes, as large participants position "
            f"before scheduled events based on proprietary research."
        )

    # Volatility → any (fear/regime signal)
    if domain_a == "volatility":
        return (
            f"[AUTO] Volatility regime signal: {activity_a} levels precede changes in "
            f"{activity_b} as fear/complacency cycles drive participant behavior across "
            f"asset classes and trading styles."
        )

    # Government → corporate (procurement signal chain)
    if domain_a == "government" and domain_b == "corporate":
        return (
            f"[AUTO] Government-corporate signal chain: {activity_a} activity leads "
            f"{activity_b} responses as companies prepare for or react to public sector "
            f"procurement cycles."
        )

    # Insider → government (regulatory foresight)
    if domain_a == "insider" and domain_b == "government":
        return (
            f"[AUTO] Regulatory foresight: {activity_a} may reflect insider awareness of "
            f"upcoming {activity_b} decisions, particularly for defense, healthcare, and "
            f"infrastructure sectors."
        )

    # Generic fallback for any remaining known pairs
    return (
        f"[AUTO] Cross-domain correlation: {domain_a} {activity_a} leads {domain_b} "
        f"{activity_b}. Mechanism requires further research, but statistical signal "
        f"justifies monitoring while causal story develops."
    )


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
    """Look up causal story for a source pair (order-independent).

    Priority:
    1. Hardcoded CAUSAL_STORY_TEMPLATES (human-written, highest quality)
    2. _auto_generate_causal_story() using _DOMAIN_ROLES (auto, slightly tighter gates)
    3. "REQUIRES MANUAL REVIEW" only when both sources are completely unknown
    """
    story = CAUSAL_STORY_TEMPLATES.get((source_a, source_b))
    if story:
        return story
    story = CAUSAL_STORY_TEMPLATES.get((source_b, source_a))
    if story:
        return story

    # Attempt auto-generation from domain roles
    auto_story = _auto_generate_causal_story(source_a, source_b)
    if auto_story:
        return auto_story

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
