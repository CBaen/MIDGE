"""Market awareness functions for agent perception.

Biological analogy: The visual cortex. Agents already have market organs
(sensing hook, convergence engine) connected to their nervous system
(endocrine/EventBus). These functions give agents the cortical regions
to interpret market signals — encoding raw market state into dimensions
their brains can process.

Four standalone functions, not a mixin. Law 5 says specialization via
configuration, not different code. These functions read agent refs
(attached by bootstrap/market.py during differentiation) and return data.

Used by:
- lifecycle_sensing.py: _build_state_vector() calls get_market_state_dims()
- lifecycle_decision.py: _route_with_advisory() calls get_market_context_for_router()
- lifecycle_decision.py: _act_api_call() calls build_market_oracle_prompt()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Market roles from stem_cell.py ROLE_PROFILES
_MARKET_ROLES = frozenset({"SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST"})

# Regime → float encoding (normalized 0-1 range for state vector)
_REGIME_ENCODING = {
    "bull": 0.8,
    "bear": 0.2,
    "volatile": 0.5,
    "sideways": 0.4,
    "default": 0.5,
    "unknown": 0.5,
}


def is_market_agent(agent: Any) -> bool:
    """Check if agent has a market role."""
    role = getattr(agent, "role", "STEM")
    return role in _MARKET_ROLES


def get_market_state_dims(agent: Any) -> list[float]:
    """Return 4 market dimensions for state vector extension.

    Dims:
        [0] regime_encoded: bull=0.8, bear=0.2, volatile=0.5, sideways=0.4
        [1] convergence_strength: max alert strength from advisory (0.0-1.0)
        [2] convergence_direction: bullish=1.0, bearish=-1.0, neutral=0.0
        [3] signal_density: total signals across domains / 100 (capped at 1.0)

    Non-market agents get [0.5, 0.0, 0.0, 0.0] — the regime default
    with no convergence. This preserves backward compatibility: the world
    model sees a neutral market environment and learns nothing extra.
    """
    if not is_market_agent(agent):
        return [0.5, 0.0, 0.0, 0.0]

    # Regime
    regime_ref = getattr(agent, "_regime_classifier_ref", None)
    if regime_ref is not None:
        try:
            regime = regime_ref.classify()
            regime_val = _REGIME_ENCODING.get(regime, 0.5)
        except Exception:
            regime_val = 0.5
    else:
        regime_val = 0.5

    # Convergence from market advisory (set by sensing hook step)
    advisory = getattr(agent, "_market_advisory_ref", None)
    if advisory is not None and isinstance(advisory, dict):
        alert = advisory.get("alert")
        if alert is not None and isinstance(alert, dict):
            conv_strength = float(alert.get("strength", 0.0))
            direction_str = alert.get("direction", "neutral")
            conv_direction = {"bullish": 1.0, "bearish": -1.0}.get(direction_str, 0.0)
        else:
            conv_strength = 0.0
            conv_direction = 0.0
    else:
        conv_strength = 0.0
        conv_direction = 0.0

    # Signal density from convergence alerter buffer
    alerter = getattr(agent, "_convergence_alerter_ref", None)
    if alerter is not None:
        try:
            total_signals = sum(
                len(sigs) for sigs in alerter.signals.values()
            )
            signal_density = min(1.0, total_signals / 100.0)
        except Exception:
            signal_density = 0.0
    else:
        signal_density = 0.0

    return [regime_val, conv_strength, conv_direction, signal_density]


def get_market_context_for_router(agent: Any) -> dict[str, Any]:
    """Build market-enriched context dict for DecisionRouter.

    Adds market-specific fields alongside the existing threat_level,
    opportunity_level, and body_state context in _route_with_advisory().
    """
    context: dict[str, Any] = {}

    role = getattr(agent, "role", "STEM")
    context["agent_market_role"] = role

    # Regime
    regime_ref = getattr(agent, "_regime_classifier_ref", None)
    if regime_ref is not None:
        try:
            context["market_regime"] = regime_ref.classify()
        except Exception:
            context["market_regime"] = "unknown"
    else:
        context["market_regime"] = "unknown"

    # Convergence state
    advisory = getattr(agent, "_market_advisory_ref", None)
    if advisory is not None and isinstance(advisory, dict):
        alert = advisory.get("alert")
        if alert is not None and isinstance(alert, dict):
            context["convergence_strength"] = alert.get("strength", 0.0)
            context["convergence_direction"] = alert.get("direction", "neutral")
            context["convergence_domains"] = alert.get("domains", [])
        else:
            context["convergence_strength"] = 0.0

    # Domain status snapshot for richer routing
    alerter = getattr(agent, "_convergence_alerter_ref", None)
    if alerter is not None:
        try:
            status = alerter.get_domain_status()
            context["active_domains"] = list(status.keys())
            context["signal_density"] = sum(
                v.get("signal_count", 0) for v in status.values()
            )
            # Opportunity level from convergence strength
            bullish_domains = sum(
                1 for v in status.values() if v.get("direction") == "bullish"
            )
            context["market_opportunity_level"] = min(
                1.0, bullish_domains / max(1, len(status))
            )
        except Exception:
            pass

    return context


def build_market_oracle_prompt(agent: Any, base_question: str) -> dict:
    """Build role-specific oracle payload with market context.

    Returns a dict with 'question' and 'context' keys, suitable for
    inject_external_task payload. The question is role-specific and
    includes live market data. The context dict carries structured
    data for the LLM.
    """
    role = getattr(agent, "role", "STEM")
    pred_error = getattr(agent, "_prediction_error", 0.0)
    risk = getattr(agent, "risk_score", 0.0)

    # Gather market state
    regime = "unknown"
    regime_ref = getattr(agent, "_regime_classifier_ref", None)
    if regime_ref is not None:
        try:
            regime = regime_ref.classify()
        except Exception:
            pass

    # Convergence summary
    conv_summary = "No convergence detected"
    advisory = getattr(agent, "_market_advisory_ref", None)
    if advisory is not None and isinstance(advisory, dict):
        alert = advisory.get("alert")
        if alert is not None and isinstance(alert, dict):
            direction = alert.get("direction", "neutral")
            strength = alert.get("strength", 0)
            domains = alert.get("domains", [])
            conv_summary = (
                f"{direction.upper()} convergence (strength={strength:.2f}) "
                f"across {len(domains)} domains: {', '.join(domains)}"
            )

    # Domain status for context
    domain_status = {}
    alerter = getattr(agent, "_convergence_alerter_ref", None)
    if alerter is not None:
        try:
            domain_status = alerter.get_domain_status()
        except Exception:
            pass

    # Role-specific question
    if role == "SEC_WATCHER":
        question = (
            f"You are a SEC filings analyst (agent {agent.unique_id}). "
            f"Market regime: {regime}. {conv_summary}. "
            f"Prediction error: {pred_error:.2f}, risk: {risk:.2f}. "
            f"Active domains: {list(domain_status.keys())}. "
            f"Which insider trading patterns are most actionable right now? "
            f"Focus on Form 4 cluster buys, unusual filing times, and "
            f"insider-congressional trade correlations."
        )
    elif role == "CONTRACT_TRACKER":
        question = (
            f"You are a government contracts analyst (agent {agent.unique_id}). "
            f"Market regime: {regime}. {conv_summary}. "
            f"Prediction error: {pred_error:.2f}, risk: {risk:.2f}. "
            f"Active domains: {list(domain_status.keys())}. "
            f"What procurement patterns suggest upcoming contract awards? "
            f"Focus on hiring spikes near contract deadlines, congressional "
            f"trades correlated with committee assignments, and SAM.gov "
            f"opportunity-to-award patterns."
        )
    elif role == "MARKET_ANALYST":
        question = (
            f"You are a cross-domain market analyst (agent {agent.unique_id}). "
            f"Market regime: {regime}. {conv_summary}. "
            f"Prediction error: {pred_error:.2f}, risk: {risk:.2f}. "
            f"Domain status: {_format_domain_status(domain_status)}. "
            f"What convergence patterns across domains suggest the highest-"
            f"confidence opportunities? Synthesize insider behavior, "
            f"government activity, and institutional signals."
        )
    else:
        question = base_question

    return {
        "question": question,
        "context": {
            "agent_id": str(agent.unique_id),
            "role": role,
            "market_regime": regime,
            "prediction_error": float(pred_error),
            "risk": float(risk),
            "convergence": conv_summary,
            "domain_status": {
                k: {"direction": v.get("direction"), "strength": v.get("strength")}
                for k, v in domain_status.items()
            },
        },
    }


def _format_domain_status(status: dict) -> str:
    """Format domain status dict into readable string for LLM prompt."""
    if not status:
        return "no active domains"
    parts = []
    for domain, info in sorted(status.items()):
        direction = info.get("direction", "?")
        strength = info.get("strength", 0)
        count = info.get("signal_count", 0)
        parts.append(f"{domain}={direction}(s={strength:.2f},n={count})")
    return ", ".join(parts)
