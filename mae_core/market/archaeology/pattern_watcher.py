"""Pattern Watcher — live stacking detection engine.

Runs alongside the existing ConvergenceAlerter during each sensing cycle.
Compares current signal state against the Pattern Library and detects
when multiple independent historical patterns activate on the same
ticker — that's where the 95%+ confidence comes from.

Stacking tiers:
  N=1: Log only (single pattern = 30-50% WR)
  N=2: Low-confidence advisory
  N=3: Medium-confidence alert
  N=4+: High-confidence alert

Independence check: Two patterns are "independent" if their precursor
signal sets overlap < 30%. Prevents double-counting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mae_core.market.archaeology.fingerprint import MoveFingerprint
from mae_core.market.archaeology.pattern_library import PatternLibrary, PatternMatch

logger = logging.getLogger(__name__)

INDEPENDENCE_THRESHOLD = 0.30  # Max source overlap for "independent" patterns


@dataclass
class PatternActivation:
    """A single pattern that has activated in the live market."""
    fingerprint: MoveFingerprint
    match_score: float
    matched_sources: list[str]
    missing_sources: list[str]
    description: str = ""

    @property
    def win_rate(self) -> float:
        return self.fingerprint.win_rate

    @property
    def sample_size(self) -> int:
        return self.fingerprint.wins + self.fingerprint.losses


@dataclass
class PatternStack:
    """Multiple independent patterns activating on the same symbol+direction."""
    symbol: str
    direction: str
    activations: list[PatternActivation]
    stack_confidence: float = 0.0
    independent_pairs: int = 0
    total_pairs: int = 0
    tier: str = ""  # "low", "medium", "high"
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

        Uses independence-weighted combination. If all patterns are truly
        independent, the probability they're ALL wrong simultaneously is
        the product of their individual failure rates. But we discount
        for non-independent pairs.
        """
        if not self.activations:
            return 0.0

        # Get win rates (use 0.5 if no data yet)
        wrs = []
        for a in self.activations:
            wr = a.win_rate if a.sample_size >= 5 else 0.5
            wrs.append(wr)

        if len(wrs) == 1:
            return wrs[0]

        # P(at least one is right) = 1 - product(1 - wr_i)
        # But discount for non-independence
        independence_ratio = self.independent_pairs / max(self.total_pairs, 1)
        discount = 0.5 + 0.5 * independence_ratio  # range [0.5, 1.0]

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
                    "fingerprint_id": a.fingerprint.fingerprint_id,
                    "match_score": a.match_score,
                    "win_rate": a.win_rate,
                    "sample_size": a.sample_size,
                    "matched_sources": a.matched_sources,
                    "missing_sources": a.missing_sources,
                    "domain_signature": a.fingerprint.domain_signature,
                    "description": a.description,
                }
                for a in self.activations
            ],
            "created_at": self.created_at,
        }

    def format_alert(self) -> str:
        """Format the stack as a human-readable alert — what Guiding Light sees."""
        lines = [
            f"PATTERN STACK: {self.symbol} {self.direction.upper()} | "
            f"{len(self.activations)} patterns | stack_confidence: {self.stack_confidence:.2f}",
        ]
        for i, a in enumerate(self.activations, 1):
            wr_str = f"WR: {a.win_rate:.0%}" if a.sample_size >= 5 else "WR: pending"
            lines.append(
                f"  Pattern {i}: \"{a.fingerprint.domain_signature}\" "
                f"({wr_str}, n={a.sample_size}, move={a.fingerprint.move_pct:+.1f}%)"
            )
            lines.append(f"    - Activated by: {', '.join(a.matched_sources)}")
            if a.missing_sources:
                lines.append(f"    - Missing: {', '.join(a.missing_sources)}")

        if self.total_pairs > 0:
            lines.append(
                f"  Independence: {self.independent_pairs}/{self.total_pairs} "
                f"pairs < {INDEPENDENCE_THRESHOLD:.0%} overlap"
            )
        return "\n".join(lines)


class PatternWatcher:
    """Live stacking detection engine.

    On each check cycle, compares current signal state against the
    Pattern Library and detects pattern stacking.
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
        self._recent_stacks: dict[str, str] = {}  # dedup_key -> ISO timestamp
        self._stacks_detected = 0
        self._checks_performed = 0

    def check(
        self,
        active_signals: dict[str, dict[str, set[str]]],
    ) -> list[PatternStack]:
        """Check for pattern stacking across all symbols with active signals.

        Args:
            active_signals: Nested dict of {symbol: {direction: {source1, source2, ...}}}
                           Built from recent signals in the sensing hook.

        Returns:
            List of PatternStack alerts (only those meeting min_stack threshold).
        """
        self._checks_performed += 1
        stacks: list[PatternStack] = []

        for symbol, directions in active_signals.items():
            for direction, sources in directions.items():
                if not sources:
                    continue

                # Query library for matching fingerprints
                matches = self._library.query_similar(
                    live_sources=sources,
                    symbol=symbol,
                    direction=direction,
                )

                if len(matches) < self._min_stack:
                    if matches:
                        logger.debug(
                            "Single pattern match for %s %s: %s (score=%.2f)",
                            symbol, direction,
                            matches[0].fingerprint.domain_signature,
                            matches[0].match_score,
                        )
                    continue

                # Build activations
                activations = [
                    PatternActivation(
                        fingerprint=m.fingerprint,
                        match_score=m.match_score,
                        matched_sources=m.matched_sources,
                        missing_sources=m.missing_sources,
                        description=m.fingerprint.domain_signature,
                    )
                    for m in matches
                ]

                # Compute independence
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

                # Log the alert
                logger.info(stack.format_alert())

                # Publish to EventBus
                if self._bus is not None:
                    try:
                        from mae_core.market.channels import CH_PATTERN_STACK_DETECTED
                        self._bus.publish(CH_PATTERN_STACK_DETECTED, stack.to_dict())
                    except Exception:
                        logger.debug("Could not publish pattern stack", exc_info=True)

        return stacks

    def _compute_independence(
        self, activations: list[PatternActivation]
    ) -> tuple[int, int]:
        """Compute how many pairs of activations are truly independent.

        Two patterns are independent if their precursor source sets overlap
        less than the threshold (default 30%).

        Returns: (independent_pairs, total_pairs)
        """
        n = len(activations)
        if n < 2:
            return (0, 0)

        independent = 0
        total = 0

        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                sources_i = activations[i].fingerprint.source_set
                sources_j = activations[j].fingerprint.source_set
                union = sources_i | sources_j
                if not union:
                    independent += 1
                    continue
                overlap = len(sources_i & sources_j) / len(union)
                if overlap < self._independence_threshold:
                    independent += 1

        return (independent, total)

    def build_active_signals(self, recent_signals: list[dict]) -> dict[str, dict[str, set[str]]]:
        """Convert a list of signal dicts into the format check() expects.

        Helper for integration — takes raw signal records and groups them
        into {symbol: {direction: {sources}}}.
        """
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
        """Return watcher stats for monitoring."""
        return {
            "checks_performed": self._checks_performed,
            "stacks_detected": self._stacks_detected,
            "library_size": self._library.size,
            "active_dedup_keys": len(self._recent_stacks),
        }
