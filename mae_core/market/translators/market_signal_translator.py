"""Market signal translators -- ConvergenceAlerter and partial convergences.

Bridges market intelligence signals into Mae's core attention pipeline
(PatternBus) so that convergence events drive organism-level attention
and partial convergences trigger investigation.

Two translators:
- MarketConvergenceTranslator: full convergences (bullish=OPPORTUNITY,
  bearish=THREAT), high confidence, CORRELATED form.
- MarketPartialTranslator: partial convergences (any=NOVELTY),
  capped salience, REACTIVE form -- investigation triggers only.
"""

from __future__ import annotations

import json
from typing import Any

from mae_core.market.channels import CH_CONVERGENCE, CH_PARTIAL_CONVERGENCE
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


class MarketConvergenceTranslator:
    """Translates full market convergence alerts into PatternSignals.

    Listens to: CH_CONVERGENCE (market.intel.convergence)

    Bullish convergence → OPPORTUNITY / CORRELATED
    Bearish convergence → THREAT / CORRELATED
    Neutral / missing direction → skipped (returns None)

    Salience formula:
        min(1.0, confidence * 0.6 + (domain_count / 12) * 0.4)

    The formula blends the Thompson-weighted confidence score (how reliable
    are the underlying signals) with domain breadth (how many independent
    domains agree). A highly confident 3-domain alert scores lower than a
    moderately confident 8-domain alert, which is the intended behavior.
    """

    @property
    def source_name(self) -> str:
        return "market_convergence"

    @property
    def channels(self) -> list[str]:
        return [CH_CONVERGENCE]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        data = _parse(message)
        if data is None:
            return None

        direction = data.get("direction", "").lower()
        if direction == "bullish":
            domain = PatternDomain.OPPORTUNITY
        elif direction == "bearish":
            domain = PatternDomain.THREAT
        else:
            return None

        confidence = float(data.get("confidence", 0.0))
        domain_count = int(data.get("domain_count", len(data.get("domains", []))))
        salience = min(1.0, confidence * 0.6 + (domain_count / 12) * 0.4)

        ticker = data.get("ticker", "UNKNOWN")
        description = (
            f"Market convergence: {direction} on {ticker} "
            f"(conf={confidence:.2f}, domains={domain_count})"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=domain,
            form=PatternForm.CORRELATED,
            confidence=confidence,
            salience=salience,
            description=description,
            evidence=data,
            ttl_steps=10,
        )


class MarketPartialTranslator:
    """Translates partial market convergences into investigation triggers.

    Listens to: CH_PARTIAL_CONVERGENCE (market.intel.partial_convergence)

    Any partial convergence → NOVELTY / REACTIVE
    Salience is intentionally capped at 0.6 -- these are hints, not calls.

    Salience formula:
        min(0.6, len(domains_seen) * 0.2)

    A single-domain partial gets 0.2, two domains 0.4, three+ are capped at
    0.6. This keeps partials below any full convergence in organism priority
    while still surfacing them as investigation-worthy novelty signals.
    """

    @property
    def source_name(self) -> str:
        return "market_partial_convergence"

    @property
    def channels(self) -> list[str]:
        return [CH_PARTIAL_CONVERGENCE]

    def translate(self, channel: str, message: Any) -> PatternSignal | None:
        data = _parse(message)
        if data is None:
            return None

        domains_seen: list[str] = data.get("domains_seen", [])
        salience = min(0.6, len(domains_seen) * 0.2)

        ticker = data.get("ticker", "UNKNOWN")
        description = (
            f"Partial convergence forming on {ticker}: "
            f"domains={domains_seen}"
        )

        return PatternSignal(
            source_system=self.source_name,
            domain=PatternDomain.NOVELTY,
            form=PatternForm.REACTIVE,
            confidence=float(data.get("confidence", 0.3)),
            salience=salience,
            description=description,
            evidence=data,
            ttl_steps=15,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse(message: Any) -> dict | None:
    """Normalise EventBus message to a dict. Returns None if unparseable."""
    if isinstance(message, dict):
        return message
    if isinstance(message, str):
        try:
            parsed = json.loads(message)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            return None
    return None
