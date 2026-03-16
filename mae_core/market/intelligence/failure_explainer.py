"""
failure_explainer.py — WHY did MIDGE get it wrong?

Post-mortem knows combo win rates. This layer explains the mechanism.
Each failed outcome is assigned a category from the failure taxonomy,
with a confidence score and a plain-English explanation.

Taxonomy:
  regime_shift       — market regime changed between signal and outcome
  stale_signal       — signal was old data (e.g. congressional trade from months ago)
  correlated_domains — domains that seemed independent were actually correlated
  timing_error       — right direction, wrong window
  deception          — pump-and-dump or manipulation detected after the fact
  overcrowded        — too many signals in same direction (contrarian reversal risk)
  external_shock     — unpredictable event (black swan, policy change)
  insufficient_evidence — convergence fired at minimum domains, weak Thompson scores
  unknown            — cannot determine the cause

Persistence:
  data/midge/failure_explanations.jsonl  — append-only per-prediction log
  data/midge/failure_summary.json        — rolling category counts (read by daily_narrative)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("midge.failure_explainer")

# ── Constants ────────────────────────────────────────────────────────────────

# Domains known to be highly correlated — appearing together isn't true independence
_CORRELATED_PAIRS: list[frozenset] = [
    frozenset({"macro", "technical"}),       # r ≈ 0.73 (both react to rate moves)
    frozenset({"sentiment", "technical"}),   # retail sentiment drives TA signals
    frozenset({"institutional", "insider"}), # institutions and insiders see same info
]

# Threshold for "barely above convergence threshold" — insufficient evidence
_WEAK_CONFIDENCE_CEILING = 0.50   # confidence <= this → insufficient evidence
_MIN_DOMAIN_COUNT = 3             # at or near minimum = weak
_OVERCROWDED_DOMAIN_COUNT = 5     # 5+ agreeing domains = contrarian risk

# Signal age threshold for staleness (days)
_STALE_SIGNAL_DAYS = 7

# Deception file — written by deception_detector in bio-market layer
_DECEPTION_STATE_PATH = Path("data/market/deception_state.json")
_CONVERGENCE_STATE_PATH = Path("data/midge/convergence_state.json")

# Output paths
_EXPLANATIONS_PATH = Path("data/midge/failure_explanations.jsonl")
_SUMMARY_PATH = Path("data/midge/failure_summary.json")


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class FailureExplanation:
    prediction_id: str
    ticker: str
    category: str
    confidence_score: float      # How confident the explainer is about this explanation
    explanation: str             # "This failed because..."
    evidence: list[str]          # Specific data points
    recommendation: str          # "In future, when X, also check Y"
    analyzed_at: str = field(default_factory=lambda: datetime.now().isoformat())
    direction: str = ""
    predicted_confidence: float = 0.0
    contributing_signals: list[str] = field(default_factory=list)


# ── Core class ────────────────────────────────────────────────────────────────

class FailureExplainer:
    """Analyzes WHY failed predictions failed.

    Usage:
        explainer = FailureExplainer()
        explanation = explainer.explain_failure(prediction, outcome)
        summary = explainer.get_failure_rate_by_category()
    """

    def __init__(self, data_dir: str = "data/market"):
        self._data_dir = Path(data_dir)
        self._midge_dir = Path("data/midge")
        self._midge_dir.mkdir(parents=True, exist_ok=True)
        self._explanations_path = self._midge_dir / "failure_explanations.jsonl"
        self._summary_path = self._midge_dir / "failure_summary.json"
        self._regime_history: list[dict] = []   # cached for batch runs
        self._deception_flags: dict[str, Any] = {}  # ticker → deception data
        self._load_caches()

    # ── Public API ──────────────────────────────────────────────────────────

    def explain_failure(self, prediction: dict, outcome: dict) -> FailureExplanation:
        """Given a prediction dict and its graded outcome, explain the failure.

        Both dicts use the schema from outcomes.jsonl / predictions.jsonl.
        Only called for predictions where was_correct == False.
        """
        ticker = prediction.get("symbol", outcome.get("symbol", "UNKNOWN"))
        pred_id = prediction.get("prediction_id", outcome.get("prediction_id", "unknown"))
        signals = (
            prediction.get("contributing_signals")
            or outcome.get("contributing_signals", [])
        )
        confidence = (
            prediction.get("confidence")
            or prediction.get("predicted_confidence")
            or outcome.get("predicted_confidence", 0.0)
        )
        predicted_at_raw = prediction.get("predicted_at") or outcome.get("predicted_at", "")
        outcome_at_raw = outcome.get("outcome_at", "")
        direction = prediction.get("direction") or outcome.get("direction", "")
        return_pct = outcome.get("return_pct", 0.0)
        mfe = outcome.get("mfe", None)

        predicted_at = _parse_dt(predicted_at_raw)
        outcome_at = _parse_dt(outcome_at_raw)

        # Run checks in priority order — first strong match wins
        checks = [
            self._check_regime_shift(ticker, predicted_at, outcome_at),
            self._check_stale_signal(signals, predicted_at),
            self._check_correlated_domains(signals),
            self._check_timing_error(return_pct, direction, mfe, predicted_at, outcome_at),
            self._check_deception(ticker, predicted_at),
            self._check_overcrowded(signals),
            self._check_insufficient_evidence(signals, confidence),
        ]

        # Pick highest-confidence explanation; fall through to unknown
        best = max(checks, key=lambda x: x[1])
        category, conf, explanation, evidence, recommendation = best

        if conf < 0.3:
            category = "unknown"
            conf = 0.5
            explanation = (
                f"This failed but I cannot determine the mechanism. "
                f"{ticker} moved {abs(return_pct):.1f}% against my {direction} call."
            )
            evidence = [f"return_pct={return_pct:.2f}%", f"confidence={confidence:.2f}"]
            recommendation = "Review raw signal timestamps and domain overlap before next call."

        expl = FailureExplanation(
            prediction_id=pred_id,
            ticker=ticker,
            category=category,
            confidence_score=round(conf, 3),
            explanation=explanation,
            evidence=evidence,
            recommendation=recommendation,
            direction=direction,
            predicted_confidence=confidence,
            contributing_signals=signals,
        )

        self._persist(expl)
        return expl

    def batch_explain(
        self, predictions_with_outcomes: list[tuple[dict, dict]]
    ) -> list[FailureExplanation]:
        """Explain multiple failures. Updates summary once at the end."""
        self._load_caches()  # refresh before batch
        results = []
        for pred, outcome in predictions_with_outcomes:
            try:
                if not outcome.get("was_correct", True):
                    results.append(self.explain_failure(pred, outcome))
            except Exception:
                logger.debug("explain_failure failed for %s", pred.get("prediction_id"), exc_info=True)
        self._update_summary()
        return results

    def get_common_failure_modes(self, n: int = 5) -> list[tuple[str, int, str]]:
        """Return [(category, count, example_explanation), ...] sorted by count."""
        summary = self._load_summary()
        counts = summary.get("category_counts", {})
        examples = summary.get("category_examples", {})
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(cat, cnt, examples.get(cat, "")) for cat, cnt in ranked]

    def get_failure_rate_by_category(self) -> dict[str, float]:
        """Return {category: percentage_of_all_failures}."""
        summary = self._load_summary()
        counts = summary.get("category_counts", {})
        total = sum(counts.values())
        if total == 0:
            return {}
        return {cat: round(cnt / total * 100, 1) for cat, cnt in counts.items()}

    # ── Check implementations ────────────────────────────────────────────────

    def _check_regime_shift(
        self,
        ticker: str,
        predicted_at: Optional[datetime],
        outcome_at: Optional[datetime],
    ) -> tuple[str, float, str, list[str], str]:
        """Detect if market regime changed between prediction and outcome."""
        try:
            state = _load_json(_CONVERGENCE_STATE_PATH)
            regime_at_grading = state.get("regime", "")
            if not regime_at_grading or not predicted_at:
                return _null_check("regime_shift")
            # We don't have full regime history, but can use deception_state timestamp
            # as a proxy: if regime is bear/volatile and prediction was bullish, flag it
            if regime_at_grading in ("bear", "volatile"):
                return (
                    "regime_shift",
                    0.65,
                    f"This failed because the market regime was '{regime_at_grading}' "
                    f"when the outcome was evaluated. Bullish signals underperform in bear/volatile regimes.",
                    [f"regime_at_outcome={regime_at_grading}", f"ticker={ticker}"],
                    "In future, gate bullish calls when regime is bear or volatile.",
                )
        except Exception:
            pass
        return _null_check("regime_shift")

    def _check_stale_signal(
        self, signals: list[str], predicted_at: Optional[datetime]
    ) -> tuple[str, float, str, list[str], str]:
        """Detect signals that reference old data."""
        stale_sources = [s for s in signals if any(
            kw in s.lower() for kw in ("congressional", "politician", "congress", "senate", "house")
        )]
        if not stale_sources:
            return _null_check("stale_signal")

        # Congressional trades: disclosed within 45 days, but trade itself can be months old
        conf = 0.80
        return (
            "stale_signal",
            conf,
            f"This failed because the signal included congressional trade data "
            f"({', '.join(stale_sources)}). Congressional trades are disclosed up to 45 days "
            f"after they occur — the market has already priced this information.",
            [f"stale_sources={stale_sources}"],
            "Weight congressional signals lower when disclosure lag > 30 days. "
            "Use as confirmation only, never as primary driver.",
        )

    def _check_correlated_domains(
        self, signals: list[str]
    ) -> tuple[str, float, str, list[str], str]:
        """Detect when domains appeared independent but were actually correlated."""
        domains = _signals_to_domains(signals)
        found_pairs = []
        for pair in _CORRELATED_PAIRS:
            if pair.issubset(domains):
                found_pairs.append(" + ".join(sorted(pair)))
        if not found_pairs:
            return _null_check("correlated_domains")

        return (
            "correlated_domains",
            0.70,
            f"This failed because the converging domains were correlated, not independent. "
            f"Correlated pair(s) detected: {', '.join(found_pairs)}. "
            f"Convergence requires truly independent confirmation.",
            [f"correlated_pairs={found_pairs}", f"domains_active={sorted(domains)}"],
            "When correlated domain pairs fire together, require a 3rd uncorrelated domain "
            "(e.g. insider, government, events) before treating it as convergence.",
        )

    def _check_timing_error(
        self,
        return_pct: float,
        direction: str,
        mfe: Optional[float],
        predicted_at: Optional[datetime],
        outcome_at: Optional[datetime],
    ) -> tuple[str, float, str, list[str], str]:
        """Detect right direction, wrong window."""
        if mfe is None:
            return _null_check("timing_error")
        # MFE > SUCCESS_THRESHOLD but outcome_pct didn't reach it within window
        if mfe > 5.0 and abs(return_pct) < 5.0:
            days_held = (
                (outcome_at - predicted_at).days
                if predicted_at and outcome_at
                else "unknown"
            )
            return (
                "timing_error",
                0.85,
                f"This failed on timing, not direction. The position reached a max favorable "
                f"excursion of {mfe:.1f}% but closed the window at only {return_pct:.1f}%. "
                f"The move was real — the window was too short.",
                [f"mfe={mfe:.2f}%", f"final_return={return_pct:.2f}%", f"days_held={days_held}"],
                "Consider extending the outcome window for signals with strong MFE. "
                "Or use a trailing stop approach rather than a fixed window.",
            )
        return _null_check("timing_error")

    def _check_deception(
        self, ticker: str, predicted_at: Optional[datetime]
    ) -> tuple[str, float, str, list[str], str]:
        """Detect if deception was flagged for this ticker."""
        deception = self._deception_flags.get(ticker, {})
        if not deception:
            return _null_check("deception")
        score = deception.get("suspicion_score", 0.0)
        if score > 0.6:
            return (
                "deception",
                min(0.5 + score * 0.4, 0.90),
                f"This failed because deception indicators were active for {ticker} "
                f"(suspicion score: {score:.2f}). Artificial price/volume moves "
                f"can generate false convergence signals.",
                [f"suspicion_score={score:.2f}", f"ticker={ticker}"],
                "Suppress convergence alerts for tickers with suspicion_score > 0.6 "
                "until the flag clears.",
            )
        return _null_check("deception")

    def _check_overcrowded(
        self, signals: list[str]
    ) -> tuple[str, float, str, list[str], str]:
        """Detect when too many signals agree — contrarian reversal risk."""
        domains = _signals_to_domains(signals)
        if len(domains) >= _OVERCROWDED_DOMAIN_COUNT:
            return (
                "overcrowded",
                0.65,
                f"This failed because {len(domains)} independent domains all agreed "
                f"on the same direction. When the trade is this obvious, "
                f"smart money has already positioned — reversal risk is elevated.",
                [f"domain_count={len(domains)}", f"domains={sorted(domains)}"],
                "When 5+ domains agree, require a contrarian check: "
                "COT positioning, options skew, or short interest.",
            )
        return _null_check("overcrowded")

    def _check_insufficient_evidence(
        self, signals: list[str], confidence: float
    ) -> tuple[str, float, str, list[str], str]:
        """Detect weak convergence — fired at minimum threshold."""
        domains = _signals_to_domains(signals)
        is_weak_confidence = confidence <= _WEAK_CONFIDENCE_CEILING
        is_min_domains = len(domains) <= _MIN_DOMAIN_COUNT

        if is_weak_confidence and is_min_domains:
            conf = 0.75
        elif is_weak_confidence or is_min_domains:
            conf = 0.55
        else:
            return _null_check("insufficient_evidence")

        return (
            "insufficient_evidence",
            conf,
            f"This failed because the convergence was at minimum strength: "
            f"{len(domains)} domain(s) with confidence {confidence:.2f}. "
            f"Low-conviction signals have significantly lower win rates.",
            [f"domains={len(domains)}", f"confidence={confidence:.2f}"],
            "Raise the paper-trade gate from current threshold. "
            "Minimum 4 domains OR confidence > 0.60 for any action.",
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, expl: FailureExplanation) -> None:
        """Append explanation to JSONL log and refresh summary."""
        try:
            with open(self._explanations_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(expl)) + "\n")
        except Exception:
            logger.debug("Failed to persist explanation", exc_info=True)
        self._update_summary()

    def _update_summary(self) -> None:
        """Rebuild failure_summary.json from the explanations log."""
        summary = self._load_summary()
        counts: dict[str, int] = summary.get("category_counts", {})
        examples: dict[str, str] = summary.get("category_examples", {})
        total: int = summary.get("total_explained", 0)

        try:
            if not self._explanations_path.exists():
                return
            # Only read new lines since last update
            last_line = summary.get("last_line_processed", 0)
            with open(self._explanations_path, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
            new_lines = all_lines[last_line:]
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    cat = e.get("category", "unknown")
                    counts[cat] = counts.get(cat, 0) + 1
                    total += 1
                    if cat not in examples:
                        examples[cat] = e.get("explanation", "")[:120]
                except json.JSONDecodeError:
                    continue

            updated_summary = {
                "total_explained": total,
                "category_counts": counts,
                "category_examples": examples,
                "last_line_processed": len(all_lines),
                "updated_at": datetime.now().isoformat(),
                "top_category": max(counts, key=counts.get) if counts else "unknown",
            }
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(updated_summary, f, indent=2)
        except Exception:
            logger.debug("Failed to update failure summary", exc_info=True)

    def _load_summary(self) -> dict:
        return _load_json(self._summary_path) or {}

    def _load_caches(self) -> None:
        """Load deception flags and any other state needed for checks."""
        try:
            raw = _load_json(_DECEPTION_STATE_PATH) or {}
            # deception_state.json schema: {ticker: {suspicion_score: float, ...}}
            self._deception_flags = {
                k: v for k, v in raw.items()
                if isinstance(v, dict) and "suspicion_score" in v
            }
        except Exception:
            self._deception_flags = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

# Map signal names → canonical domain names (mirrors ConvergenceAlerter._SOURCE_DOMAIN_MAP)
_SIGNAL_TO_DOMAIN: dict[str, str] = {
    "insider": "insider", "sec_form4": "insider", "insider_cluster": "insider",
    "insider_form4": "insider", "openinsider": "insider", "finviz_insider": "insider",
    "macro": "macro", "fred": "macro", "vix": "macro", "economic_calendar": "macro",
    "cot": "positioning", "cot_positioning": "positioning",
    "technical": "technical", "rsi": "technical", "macd": "technical",
    "technical_rsi": "technical", "technical_macd": "technical",
    "bollinger": "technical", "stochastic": "technical", "williams_r": "technical",
    "cci": "technical", "atr": "technical", "session_sweep": "technical",
    "ifvg": "technical",
    "institutional": "institutional", "sec_13f": "institutional", "sec_13d": "institutional",
    "congressional": "government", "politician_sell": "government",
    "politician_buy": "government", "congress_trade": "government",
    "contract": "contracts", "contract_prediction": "contracts",
    "sam_gov": "contracts", "usa_spending": "contracts",
    "sentiment": "sentiment", "stocktwits": "sentiment", "trends": "sentiment",
    "yahoo_rss": "sentiment", "finviz_sentiment": "sentiment",
    "events": "events", "sec_form8k": "events", "hiring": "events",
    "energy": "energy", "eia": "energy",
    "crypto": "crypto", "coingecko": "crypto", "coincap": "crypto",
}


def _signals_to_domains(signals: list[str]) -> set[str]:
    """Convert raw signal names to canonical domain names."""
    domains: set[str] = set()
    for s in signals:
        s_lower = s.lower()
        mapped = _SIGNAL_TO_DOMAIN.get(s_lower)
        if mapped:
            domains.add(mapped)
        else:
            # Best-effort: check if signal contains a known domain keyword
            for kw, dom in _SIGNAL_TO_DOMAIN.items():
                if kw in s_lower:
                    domains.add(dom)
                    break
            else:
                domains.add(s_lower)  # keep as-is if no match
    return domains


def _null_check(category: str) -> tuple[str, float, str, list[str], str]:
    """Return a zero-confidence tuple for a check that found nothing."""
    return (category, 0.0, "", [], "")


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None
