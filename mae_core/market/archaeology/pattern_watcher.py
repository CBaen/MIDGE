"""Pattern Watcher — live stacking detection engine.

Runs alongside the existing ConvergenceAlerter during each sensing cycle.
Compares current signal state against PatternTemplates (symbol-AGNOSTIC)
and detects when multiple independent patterns activate on the same
ticker — that's where the overwhelming confidence comes from.

Key change from v1: matches against TEMPLATES, not fingerprints.
A pattern discovered on NVDA can now fire on ANY symbol. Cross-validated
templates (seen on 3+ symbols) get a confidence boost.

Stacking tiers:
  N=1: Log only (single pattern = 30-50% WR)
  N=2: Low-confidence advisory
  N=3: Medium-confidence alert
  N=4+: High-confidence alert ("4+ independent patterns converging")

Independence check: Two patterns are "independent" if their domain
sets overlap < 30%. Prevents double-counting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PatternTemplate
from mae_core.market.archaeology.pattern_library import PatternLibrary, PatternMatch

logger = logging.getLogger(__name__)

INDEPENDENCE_THRESHOLD = 0.30


@dataclass
class PatternActivation:
    """A single pattern template that has activated in the live market."""
    template: PatternTemplate
    match_score: float
    matched_domains: list[str]
    missing_domains: list[str]
    matched_sources: list[str]
    missing_sources: list[str]
    description: str = ""

    # Backward compatibility: also accept fingerprint kwarg
    fingerprint: Optional[MoveFingerprint] = None

    @property
    def win_rate(self) -> float:
        return self.template.win_rate

    @property
    def sample_size(self) -> int:
        return self.template.wins + self.template.losses

    @property
    def cross_validated(self) -> bool:
        return self.template.cross_validated

    @property
    def symbols_seen(self) -> int:
        return len(self.template.unique_symbols)


@dataclass
class PatternStack:
    """Multiple independent patterns activating on the same symbol+direction.

    This is what MIDGE surfaces to Guiding Light: "4 independent pattern
    templates are converging on NVDA bullish right now. Here's what each
    pattern is, where it was validated across the market, and what signals
    are driving the activation."
    """
    symbol: str
    direction: str
    activations: list[PatternActivation]
    stack_confidence: float = 0.0
    independent_pairs: int = 0
    total_pairs: int = 0
    tier: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.tier:
            self.tier = self._compute_tier()
        if self.stack_confidence == 0.0:
            self.stack_confidence = self._compute_confidence()

    def _compute_tier(self) -> str:
        n = len(self.activations)
        if n >= 4:
            return "high"
        elif n == 3:
            return "medium"
        elif n == 2:
            return "low"
        return "trace"

    def _compute_confidence(self) -> float:
        """Compute stack confidence from individual pattern win rates.

        Cross-validated templates get boosted. Independence-weighted
        combination: if patterns are truly independent, the probability
        they're ALL wrong is the product of individual failure rates.
        """
        if not self.activations:
            return 0.0

        wrs = []
        for a in self.activations:
            base_wr = a.win_rate if a.sample_size >= 5 else 0.5
            # Cross-validated templates get a confidence boost
            if a.cross_validated:
                base_wr = min(0.95, base_wr * a.template.confidence_multiplier)
            wrs.append(base_wr)

        if len(wrs) == 1:
            return wrs[0]

        independence_ratio = self.independent_pairs / max(self.total_pairs, 1)
        discount = 0.5 + 0.5 * independence_ratio

        prob_all_wrong = 1.0
        for wr in wrs:
            prob_all_wrong *= (1.0 - wr * discount)

        confidence = 1.0 - prob_all_wrong
        return round(min(0.99, max(0.0, confidence)), 3)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "tier": self.tier,
            "stack_confidence": self.stack_confidence,
            "n_patterns": len(self.activations),
            "independent_pairs": self.independent_pairs,
            "total_pairs": self.total_pairs,
            "activations": [
                {
                    "template_id": a.template.template_id,
                    "match_score": a.match_score,
                    "win_rate": a.win_rate,
                    "sample_size": a.sample_size,
                    "cross_validated": a.cross_validated,
                    "symbols_seen": a.symbols_seen,
                    "matched_domains": a.matched_domains,
                    "missing_domains": a.missing_domains,
                    "matched_sources": a.matched_sources,
                    "domain_signature": a.template.domain_signature,
                    "description": a.description,
                }
                for a in self.activations
            ],
            "created_at": self.created_at,
        }

    def format_alert(self) -> str:
        """Format the stack as a human-readable alert.

        This is what MIDGE surfaces: what patterns are converging, what kind
        of patterns, where they come from, how many symbols they've been
        validated on, and what the thresholds are.
        """
        lines = [
            f"PATTERN STACK: {self.symbol} {self.direction.upper()} | "
            f"{len(self.activations)} patterns | stack_confidence: {self.stack_confidence:.2f}",
        ]
        for i, a in enumerate(self.activations, 1):
            wr_str = f"WR: {a.win_rate:.0%}" if a.sample_size >= 5 else "WR: pending"
            cross_str = f", validated on {a.symbols_seen} symbols" if a.cross_validated else ""
            lines.append(
                f"  Pattern {i}: \"{a.template.domain_signature}\" "
                f"({wr_str}, n={a.sample_size}, "
                f"instances={a.template.n_instances}{cross_str})"
            )
            lines.append(f"    - Activated by: {', '.join(a.matched_sources)}")
            if a.missing_domains:
                lines.append(f"    - Missing domains: {', '.join(a.missing_domains)}")

        if self.total_pairs > 0:
            lines.append(
                f"  Independence: {self.independent_pairs}/{self.total_pairs} "
                f"pairs < {INDEPENDENCE_THRESHOLD:.0%} overlap"
            )
        return "\n".join(lines)


class PatternWatcher:
    """Live stacking detection engine.

    Matches current signal state against PatternTemplates (symbol-agnostic)
    and detects when multiple independent templates activate on the same ticker.
    """

    def __init__(
        self,
        library: PatternLibrary,
        bus: Any = None,
        min_stack: int = 2,
        independence_threshold: float = INDEPENDENCE_THRESHOLD,
    ):
        self._library = library
        self._bus = bus
        self._min_stack = min_stack
        self._independence_threshold = independence_threshold
        self._recent_stacks: dict[str, str] = {}
        self._stacks_detected = 0
        self._checks_performed = 0

    def check(
        self,
        active_signals: dict[str, dict[str, set[str]]],
    ) -> list[PatternStack]:
        """Check for pattern stacking across all symbols with active signals.

        Args:
            active_signals: {symbol: {direction: {source1, source2, ...}}}

        Returns:
            List of PatternStack alerts meeting min_stack threshold.
        """
        self._checks_performed += 1
        stacks: list[PatternStack] = []

        for symbol, directions in active_signals.items():
            for direction, sources in directions.items():
                if not sources:
                    continue

                # Query templates (symbol-agnostic matching)
                matches = self._library.query_similar(
                    live_sources=sources,
                    symbol=symbol,
                    direction=direction,
                )

                if len(matches) < self._min_stack:
                    if matches:
                        logger.debug(
                            "Single template match for %s %s: %s (score=%.2f)",
                            symbol, direction,
                            matches[0].template.domain_signature,
                            matches[0].match_score,
                        )
                    continue

                activations = [
                    PatternActivation(
                        template=m.template,
                        match_score=m.match_score,
                        matched_domains=m.matched_domains,
                        missing_domains=m.missing_domains,
                        matched_sources=m.matched_sources,
                        missing_sources=m.missing_sources,
                        description=m.template.domain_signature,
                    )
                    for m in matches
                ]

                independent_pairs, total_pairs = self._compute_independence(activations)

                # Dedup: don't re-alert the same stack within 24 hours
                dedup_key = f"{symbol}:{direction}:{len(activations)}"
                now = datetime.now()
                if dedup_key in self._recent_stacks:
                    last_alert = datetime.fromisoformat(self._recent_stacks[dedup_key])
                    if (now - last_alert).total_seconds() < 86400:
                        continue

                stack = PatternStack(
                    symbol=symbol,
                    direction=direction,
                    activations=activations,
                    independent_pairs=independent_pairs,
                    total_pairs=total_pairs,
                )

                stacks.append(stack)
                self._recent_stacks[dedup_key] = now.isoformat()
                self._stacks_detected += 1

                logger.info(stack.format_alert())

                if self._bus is not None:
                    try:
                        from mae_core.market.channels import CH_PATTERN_STACK_DETECTED
                        self._bus.publish(CH_PATTERN_STACK_DETECTED, stack.to_dict())
                    except Exception:
                        logger.debug("Could not publish pattern stack", exc_info=True)

        return stacks

    def _compute_independence(
        self, activations: list[PatternActivation],
    ) -> tuple[int, int]:
        """Compute independence between activation pairs.

        Two templates are independent if their domain sets overlap < threshold.
        Domain-level independence (not source-level) is the right granularity
        because templates ARE domain-level abstractions.
        """
        n = len(activations)
        if n < 2:
            return (0, 0)

        independent = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                domains_i = set(activations[i].template.domains)
                domains_j = set(activations[j].template.domains)
                union = domains_i | domains_j
                if not union:
                    independent += 1
                    continue
                overlap = len(domains_i & domains_j) / len(union)
                if overlap < self._independence_threshold:
                    independent += 1

        return (independent, total)

    def build_active_signals(self, recent_signals: list[dict]) -> dict[str, dict[str, set[str]]]:
        """Convert signal dicts into the format check() expects."""
        result: dict[str, dict[str, set[str]]] = {}
        for sig in recent_signals:
            symbol = sig.get("symbol", "")
            direction = sig.get("direction", "")
            source = sig.get("source", "")
            if not symbol or not direction or not source:
                continue
            if symbol not in result:
                result[symbol] = {}
            if direction not in result[symbol]:
                result[symbol][direction] = set()
            result[symbol][direction].add(source)
        return result

    def get_statistics(self) -> dict:
        return {
            "checks_performed": self._checks_performed,
            "stacks_detected": self._stacks_detected,
            "library_size": self._library.size,
            "template_count": self._library.template_count,
            "active_dedup_keys": len(self._recent_stacks),
        }
