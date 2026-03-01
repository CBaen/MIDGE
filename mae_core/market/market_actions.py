"""Market Actions — Role-specific action dispatch for market agents.

When a SEC_WATCHER "explores", it scans insider signals instead of claiming
a generic TaskPool task. When a HYPOTHESIS_VALIDATOR "exploits", it validates
a hypothesis. Same agent class (Law 5), different behavior via role config.

Biological analogy: Differentiated cells performing tissue-specific functions.
A hepatocyte metabolizes toxins; a neuron fires signals. Same genome,
different gene expression → different behavior.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Roles that route through market actions instead of generic TaskPool
MARKET_ROLES = frozenset({
    "SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST",
    "HYPOTHESIS_EXPLORER", "HYPOTHESIS_VALIDATOR",
})

# Output directory for activity logs (gitignored)
_OUTPUT_DIR = Path("data/midge")


def act_market(agent: Any, action_type: str) -> Optional[float]:
    """Dispatch a market-role agent to its role-specific action.

    Returns a reward float if a market action was executed, or None to
    fall through to generic TaskPool behavior.
    """
    role = getattr(agent, "role", None)
    if role not in MARKET_ROLES:
        return None

    key = (role, action_type)
    handler = _DISPATCH.get(key)
    if handler is None:
        return None  # Fall through (e.g. rest, api_call)

    try:
        reward = handler(agent)
        return max(0.0, min(0.5, reward))  # Clamp to [0.0, 0.5]
    except Exception:
        logger.debug("Market action failed: %s/%s", role, action_type, exc_info=True)
        return 0.0


# ---------------------------------------------------------------------------
# Advisory helpers
# ---------------------------------------------------------------------------


def _get_advisory(agent: Any) -> dict:
    """Read the shared market advisory dict. Returns empty dict if missing."""
    ref = getattr(agent, "_market_advisory_ref", None)
    return ref if isinstance(ref, dict) else {}


def _get_alerter(agent: Any):
    """Get the convergence alerter ref or None."""
    return getattr(agent, "_convergence_alerter_ref", None)


def _get_alert(agent: Any) -> Optional[dict]:
    """Get the current strongest convergence alert dict, or None if stale."""
    adv = _get_advisory(agent)
    alert = adv.get("alert")
    if alert is None:
        return None
    # Check freshness: advisory updated_step vs agent step
    updated = adv.get("updated_step", 0)
    agent_step = getattr(agent, "step_count", 0)
    if agent_step - updated > 50:
        return None  # Stale
    return alert


def _get_agent_id(agent: Any) -> str:
    return str(getattr(agent, "unique_id", "unknown"))


def _get_step(agent: Any) -> int:
    """Get the current model step (not the agent's internal cadence counter).

    agent.step_count resets to 0 and counts up to the agent's own cadence
    (typically 100), so it never reflects the true model time. The model's
    Mesa schedule exposes the wall-clock step via model.time (Mesa 3.x) or
    model.schedule.time (Mesa 2.x). We prefer that. Fallback to step_count
    if the model reference is unavailable (tests, isolated instantiation).
    """
    model = getattr(agent, "model", None)
    if model is not None:
        t = getattr(model, "time", None)
        if t is not None:
            return int(t)
    return int(getattr(agent, "step_count", 0))


# ---------------------------------------------------------------------------
# SEC_WATCHER actions
# ---------------------------------------------------------------------------


def _sec_scan(agent: Any) -> float:
    """SEC_WATCHER explore: scan insider signal buffer for fresh patterns."""
    alerter = _get_alerter(agent)
    if alerter is None:
        return 0.0

    # Read insider + government domain signals from the alerter's buffer
    insider_signals = alerter.signals.get("insider", [])
    sec_signals = alerter.signals.get("regulatory", [])
    total = len(insider_signals) + len(sec_signals)

    if total == 0:
        return 0.05  # Scanned but nothing found — small effort reward

    # Deposit exploration marker encoding signal density
    try:
        agent.deposit_marker(
            "EXPLORATION",
            intensity=min(1.0, total / 10.0),
            metadata={"source": "sec_scan", "signal_count": total},
        )
    except Exception:
        pass

    return min(0.3, total / 10.0)


def _sec_deepen(agent: Any) -> float:
    """SEC_WATCHER exploit: deepen convergence on insider-domain alerts."""
    alert = _get_alert(agent)
    if alert is None:
        return 0.05

    strength = alert.get("strength", 0.0)
    domains = alert.get("domains", [])

    # Only deepen if insider-related domains are contributing
    insider_domains = {"insider", "regulatory", "sec_filings"}
    if not (set(domains) & insider_domains) and strength < 0.5:
        return 0.05

    # Deposit discovery marker with alert details
    try:
        agent.deposit_marker(
            "DISCOVERY",
            intensity=strength,
            metadata={
                "source": "sec_deepen",
                "direction": alert.get("direction", "neutral"),
                "strength": strength,
            },
        )
    except Exception:
        pass

    return strength * 0.4


# ---------------------------------------------------------------------------
# CONTRACT_TRACKER actions
# ---------------------------------------------------------------------------


def _contract_scan(agent: Any) -> float:
    """CONTRACT_TRACKER explore: scan government/contract signal buffer."""
    alerter = _get_alerter(agent)
    if alerter is None:
        return 0.0

    govt_signals = alerter.signals.get("government", [])
    contract_signals = alerter.signals.get("contracts", [])
    total = len(govt_signals) + len(contract_signals)

    if total == 0:
        return 0.05

    try:
        agent.deposit_marker(
            "EXPLORATION",
            intensity=min(1.0, total / 8.0),
            metadata={"source": "contract_scan", "signal_count": total},
        )
    except Exception:
        pass

    return min(0.3, total / 8.0)


def _contract_deepen(agent: Any) -> float:
    """CONTRACT_TRACKER exploit: deepen per-ticker convergence w/ contract data."""
    adv = _get_advisory(agent)
    ticker_alerts = adv.get("ticker_alerts", [])
    if not ticker_alerts:
        alert = _get_alert(agent)
        if alert is None:
            return 0.05
        return alert.get("strength", 0.0) * 0.3

    # Find strongest per-ticker alert
    strongest = max(ticker_alerts, key=lambda a: a.get("strength", 0) if isinstance(a, dict) else 0)
    strength = strongest.get("strength", 0.0) if isinstance(strongest, dict) else 0.0

    try:
        agent.deposit_marker(
            "DISCOVERY",
            intensity=strength,
            metadata={"source": "contract_deepen", "ticker_alert": strongest},
        )
    except Exception:
        pass

    return strength * 0.4


# ---------------------------------------------------------------------------
# MARKET_ANALYST actions
# ---------------------------------------------------------------------------


def _convergence_scan(agent: Any) -> float:
    """MARKET_ANALYST explore: survey all active domains + tiered alerters."""
    alerter = _get_alerter(agent)
    if alerter is None:
        return 0.0

    try:
        domain_status = alerter.get_domain_status()
    except Exception:
        return 0.0

    active_domains = len(domain_status)
    if active_domains == 0:
        return 0.05

    # Also check tiered alerters for broader context
    adv = _get_advisory(agent)
    tiers_active = sum(1 for t in ("tactical", "strategic", "thematic") if adv.get(t) is not None)

    reward = min(0.3, active_domains / 14.0 + tiers_active * 0.05)

    try:
        agent.deposit_marker(
            "EXPLORATION",
            intensity=min(1.0, active_domains / 7.0),
            metadata={
                "source": "convergence_scan",
                "active_domains": active_domains,
                "tiers_active": tiers_active,
            },
        )
    except Exception:
        pass

    return reward


def _convergence_deepen(agent: Any) -> float:
    """MARKET_ANALYST exploit: deepen strongest convergence + read Kelly sizing."""
    alert = _get_alert(agent)
    if alert is None:
        return 0.05

    strength = alert.get("strength", 0.0)
    if strength < 0.3:
        return 0.05

    # Read Kelly sizing if available
    ctx = getattr(agent, "_model_ctx_ref", None)
    kelly = getattr(ctx, "_latest_kelly", {}) if ctx else {}

    # Package discovery with full context
    discovery = {
        "direction": alert.get("direction", "neutral"),
        "strength": strength,
        "domains": alert.get("domains", []),
        "kelly": kelly.get("kelly_capped", 0.0) if kelly else 0.0,
    }

    try:
        agent.deposit_marker(
            "DISCOVERY",
            intensity=strength,
            metadata={"source": "convergence_deepen", **discovery},
        )
    except Exception:
        pass

    return strength * 0.5


# ---------------------------------------------------------------------------
# HYPOTHESIS_EXPLORER actions
# ---------------------------------------------------------------------------


def _hypothesis_generate(agent: Any) -> float:
    """HYPOTHESIS_EXPLORER explore: trigger hypothesis generation."""
    engine = getattr(agent, "_hypothesis_engine_ref", None)
    if engine is None:
        return 0.0

    try:
        count = engine.request_generation()
    except AttributeError:
        # request_generation not yet added (Phase 3a)
        return 0.0
    except Exception:
        logger.debug("Hypothesis generation request failed", exc_info=True)
        return 0.0

    if count > 0:
        _log_hypothesis_activity(agent, "generated", count=count)
        return min(0.3, count * 0.1)

    return 0.05  # Attempted but nothing new to generate


def _hypothesis_deepen(agent: Any) -> float:
    """HYPOTHESIS_EXPLORER exploit: find hypothesis needing more observations."""
    registry = getattr(agent, "_hypothesis_registry_ref", None)
    if registry is None:
        return 0.0

    try:
        active = registry.get_active()
    except Exception:
        return 0.0

    if not active:
        return 0.05

    # Find hypothesis with fewest observations (needs more data)
    least_observed = min(active, key=lambda h: h.stats.total_observations if hasattr(h, "stats") else 0)
    obs = least_observed.stats.total_observations if hasattr(least_observed, "stats") else 0

    try:
        agent.deposit_marker(
            "EXPLORATION",
            intensity=0.6,
            metadata={
                "source": "hypothesis_deepen",
                "hypothesis": least_observed.name,
                "observations": obs,
            },
        )
    except Exception:
        pass

    return 0.1


# ---------------------------------------------------------------------------
# HYPOTHESIS_VALIDATOR actions
# ---------------------------------------------------------------------------


def _hypothesis_sample(agent: Any) -> float:
    """HYPOTHESIS_VALIDATOR explore: sample probation hypotheses for readiness."""
    registry = getattr(agent, "_hypothesis_registry_ref", None)
    if registry is None:
        return 0.0

    try:
        probation = registry.get_probation()
    except Exception:
        return 0.0

    if not probation:
        return 0.05

    # Count how many are validatable (>= 20 observations)
    validatable = sum(
        1 for h in probation
        if hasattr(h, "stats") and h.stats.total_observations >= 20
    )

    return min(0.2, 0.05 * max(1, validatable))


def _hypothesis_validate(agent: Any) -> float:
    """HYPOTHESIS_VALIDATOR exploit: validate a ready hypothesis."""
    engine = getattr(agent, "_hypothesis_engine_ref", None)
    if engine is None:
        return 0.0

    try:
        result = engine.request_validation()
    except AttributeError:
        # request_validation not yet added (Phase 3a)
        return 0.0
    except Exception:
        logger.debug("Hypothesis validation request failed", exc_info=True)
        return 0.0

    if result == "promoted":
        _log_hypothesis_activity(agent, "promoted")
        return 0.4
    elif result == "retired":
        _log_hypothesis_activity(agent, "retired")
        return 0.2  # Catching a bad hypothesis is valuable
    elif result == "busy":
        return 0.05
    return 0.05  # "none" — no validatable hypotheses


# ---------------------------------------------------------------------------
# Shared communicate action
# ---------------------------------------------------------------------------


def _market_broadcast(agent: Any) -> float:
    """All market agents communicate: broadcast discovery to the colony."""
    alert = _get_alert(agent)
    if alert is None:
        return 0.05  # Nothing to broadcast

    role = getattr(agent, "role", "UNKNOWN")
    step = _get_step(agent)
    agent_id = _get_agent_id(agent)

    record = {
        "step": step,
        "agent": agent_id,
        "role": role,
        "action": "communicate",
        "direction": alert.get("direction", "neutral"),
        "strength": alert.get("strength", 0.0),
        "domains": len(alert.get("domains", [])),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }

    _append_activity(record)
    return 0.1


# ---------------------------------------------------------------------------
# Activity logging
# ---------------------------------------------------------------------------


def _append_activity(record: dict) -> None:
    """Append a JSON record to data/midge/agent_activity.jsonl."""
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / "agent_activity.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write agent activity", exc_info=True)


def _log_hypothesis_activity(agent: Any, event: str, **kwargs) -> None:
    """Append a hypothesis event to data/midge/hypothesis_activity.jsonl."""
    record = {
        "step": _get_step(agent),
        "agent": _get_agent_id(agent),
        "role": getattr(agent, "role", "UNKNOWN"),
        "event": event,
        "ts": datetime.now().isoformat(timespec="seconds"),
        **kwargs,
    }
    try:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / "hypothesis_activity.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        logger.debug("Failed to write hypothesis activity", exc_info=True)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


_DISPATCH: dict[tuple[str, str], Any] = {
    # SEC_WATCHER
    ("SEC_WATCHER", "explore"): _sec_scan,
    ("SEC_WATCHER", "exploit"): _sec_deepen,
    ("SEC_WATCHER", "communicate"): _market_broadcast,
    # CONTRACT_TRACKER
    ("CONTRACT_TRACKER", "explore"): _contract_scan,
    ("CONTRACT_TRACKER", "exploit"): _contract_deepen,
    ("CONTRACT_TRACKER", "communicate"): _market_broadcast,
    # MARKET_ANALYST
    ("MARKET_ANALYST", "explore"): _convergence_scan,
    ("MARKET_ANALYST", "exploit"): _convergence_deepen,
    ("MARKET_ANALYST", "communicate"): _market_broadcast,
    # HYPOTHESIS_EXPLORER
    ("HYPOTHESIS_EXPLORER", "explore"): _hypothesis_generate,
    ("HYPOTHESIS_EXPLORER", "exploit"): _hypothesis_deepen,
    ("HYPOTHESIS_EXPLORER", "communicate"): _market_broadcast,
    # HYPOTHESIS_VALIDATOR
    ("HYPOTHESIS_VALIDATOR", "explore"): _hypothesis_sample,
    ("HYPOTHESIS_VALIDATOR", "exploit"): _hypothesis_validate,
    ("HYPOTHESIS_VALIDATOR", "communicate"): _market_broadcast,
}
