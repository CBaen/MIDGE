"""Bootstrap Layer 33i: Agent differentiation and market reflex registration.

One job: differentiate agents into market roles and wire per-agent reflexes.
Two sub-tasks:
  - _differentiate_market_agents: assign roles by index (K3 market, K3 intelligence,
    K3 hypothesis), disable oracle API for all agents, wire RedifferentiationMonitor.
  - _register_market_reflexes: install per-agent DecisionRouter reflex patterns
    so market agents react instinctively to market stimuli.

Law 2: K3 is the atom of all structure — groups of 3, always.
Law 5: Same genome, specialized epigenome — redifferentiate() applies role config.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _differentiate_market_agents(ctx: SimpleNamespace) -> None:
    """Differentiate agents into triadic groups per Law 2.

    Progressive differentiation by agent count:
      6 agents: K3 general (1-3 STEM) + K3 market (4-6)
      9 agents: K3 general + K3 market + K3 intelligence (7-9)

    Law 5: same genome, specialized epigenome.
    Law 2: K3 is the atom of all structure — groups of 3, always.
    """
    from mae_core.agents.stem_cell import redifferentiate

    agents = getattr(ctx, "agents", [])
    if len(agents) < 6:
        logger.info(
            "Layer 33i - Need at least 6 agents for market differentiation "
            "(K3 general + K3 market per Law 2), have %d — skipping",
            len(agents),
        )
        return

    registry = getattr(ctx, "stem_cell_registry", None)
    market_advisory = getattr(ctx, "_market_advisory", None)

    # --- K3 Market: agents[-6], [-5], [-4] (or [-3], [-2], [-1] when only 6) ---
    # With 6 agents: indices 3, 4, 5. With 9: indices 3, 4, 5.
    market_roles = [
        ("SEC_WATCHER", 3),
        ("CONTRACT_TRACKER", 4),
        ("MARKET_ANALYST", 5),
    ]

    differentiated = 0
    for role, idx in market_roles:
        if idx >= len(agents):
            break
        try:
            agent = agents[idx]
            redifferentiate(agent, role, registry=registry, step=0)
            if market_advisory is not None:
                agent._market_advisory_ref = market_advisory
            # Attach live market system refs for market_awareness.py functions
            agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
            agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
            agent._model_ctx_ref = ctx
            differentiated += 1
            logger.info(
                "Market differentiation: agent %s → %s",
                getattr(agent, "unique_id", id(agent)), role,
            )
        except Exception:
            logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    # --- K3 Intelligence: agents 6, 7, 8 (only when 9+ agents) ---
    if len(agents) >= 9:
        intel_roles = [
            ("EXPLORER", 6),
            ("LEARNER", 7),
            ("LLM_SPECIALIST", 8),
        ]
        for role, idx in intel_roles:
            try:
                agent = agents[idx]
                redifferentiate(agent, role, registry=registry, step=0)
                if market_advisory is not None:
                    agent._market_advisory_ref = market_advisory
                agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
                agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
                agent._model_ctx_ref = ctx
                differentiated += 1
                logger.info(
                    "Intelligence differentiation: agent %s → %s",
                    getattr(agent, "unique_id", id(agent)), role,
                )
            except Exception:
                logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    # --- K3 Hypothesis: agents 9, 10, 11 (only when 12+ agents) ---
    if len(agents) >= 12:
        hyp_roles = [
            ("HYPOTHESIS_EXPLORER", 9),
            ("HYPOTHESIS_VALIDATOR", 10),
            ("MARKET_ANALYST", 11),  # Third member of the hypothesis triad
        ]
        for role, idx in hyp_roles:
            try:
                agent = agents[idx]
                redifferentiate(agent, role, registry=registry, step=0)
                if market_advisory is not None:
                    agent._market_advisory_ref = market_advisory
                agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
                agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
                agent._model_ctx_ref = ctx
                # Attach hypothesis engine/registry refs for direct interaction
                agent._hypothesis_engine_ref = getattr(ctx, "hypothesis_engine", None)
                agent._hypothesis_registry_ref = getattr(ctx, "hypothesis_registry", None)
                differentiated += 1
                logger.info(
                    "Hypothesis differentiation: agent %s → %s",
                    getattr(agent, "unique_id", id(agent)), role,
                )
            except Exception:
                logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    # --- MIDGE-specific: disable oracle API calls for ALL agents ---
    # In MIDGE, the external LLM oracle pathway adds nothing to market learning.
    # Agents were auto-redifferentiating into API_CALLER and burning 1.8s per call
    # for generic "prioritize exploration" filler. Market agents already have it off;
    # this ensures STEM agents can't re-enable it via auto-redifferentiation.
    for agent in agents:
        config = getattr(agent, "agent_config", None)
        if config is not None and isinstance(config, dict):
            config["api_call_enabled"] = False

    # Wire market refs into RedifferentiationMonitor so auto-rediff
    # re-attaches live system connections (bug fix: refs were lost on role switch)
    rediff_monitor = getattr(ctx, "rediff_monitor", None)
    if rediff_monitor is not None and hasattr(rediff_monitor, "set_market_refs"):
        rediff_monitor.set_market_refs({
            "advisory": getattr(ctx, "_market_advisory", None),
            "alerter": getattr(ctx, "convergence_alerter", None),
            "regime": getattr(ctx, "regime_classifier", None),
            "engine": getattr(ctx, "hypothesis_engine", None),
            "registry": getattr(ctx, "hypothesis_registry", None),
            "ctx": ctx,
        })

    # Register market-aware reflexes on differentiated agents' routers.
    # These give market agents instinctive reactions to market stimuli
    # (convergence:strong → exploit, hypothesis:empty → explore, etc.)
    _register_market_reflexes(agents, differentiated)

    triads = (
        1
        + (1 if len(agents) >= 6 else 0)
        + (1 if len(agents) >= 9 else 0)
        + (1 if len(agents) >= 12 else 0)
    )
    logger.info(
        "Layer 33i - Agent differentiation: %d agents across %d K3 triads "
        "(general + %s)",
        differentiated, triads,
        "market + intelligence + hypothesis" if len(agents) >= 12
        else ("market + intelligence" if len(agents) >= 9 else "market"),
    )


def _register_market_reflexes(agents: list, differentiated_count: int) -> None:
    """Register market-aware reflex patterns on differentiated agents.

    Each market agent's per-agent DecisionRouter gets reflexes that
    map market stimulus strings to actions. This is the instinct layer —
    when convergence is strong, exploit immediately; when hypotheses are
    empty, explore to generate new ones.

    Stimulus patterns use substring matching (decision_router._check_reflex
    line 412: `pattern.stimulus_pattern in stimulus_lower`).
    """
    if differentiated_count == 0:
        return

    try:
        from mae_core.cognition.decision_router import ReflexPattern
    except ImportError:
        logger.debug("Could not import ReflexPattern for market reflexes")
        return

    # Role → list of (stimulus_pattern, action, priority)
    _MARKET_REFLEXES = {
        "SEC_WATCHER": [
            ("convergence:strong", {"type": "exploit"}, 20),
            ("market:ambient", {"type": "explore"}, 5),
        ],
        "CONTRACT_TRACKER": [
            ("convergence:strong", {"type": "exploit"}, 20),
        ],
        "MARKET_ANALYST": [
            ("convergence:strong", {"type": "exploit"}, 25),
            ("convergence:moderate", {"type": "exploit"}, 15),
        ],
        "HYPOTHESIS_EXPLORER": [
            ("hypothesis:empty", {"type": "explore"}, 20),
        ],
        "HYPOTHESIS_VALIDATOR": [
            ("convergence:strong", {"type": "exploit"}, 15),
        ],
    }

    registered = 0
    for agent in agents:
        role = getattr(agent, "role", "STEM")
        reflexes = _MARKET_REFLEXES.get(role)
        if not reflexes:
            continue

        router = getattr(agent, "decision_router", None)
        if router is None or not hasattr(router, "register_reflex"):
            continue

        agent_id = str(getattr(agent, "unique_id", id(agent)))
        for stimulus, action, priority in reflexes:
            pattern_id = f"market_{role.lower()}_{stimulus.replace(':', '_')}_{agent_id}"
            try:
                router.register_reflex(ReflexPattern(
                    pattern_id=pattern_id,
                    stimulus_pattern=stimulus,
                    action=action,
                    confidence=0.90,
                    priority=priority,
                ))
                registered += 1
            except Exception:
                logger.debug(
                    "Failed to register reflex %s on agent %s",
                    stimulus, agent_id, exc_info=True,
                )

    if registered:
        logger.info(
            "Layer 33i - Registered %d market reflexes across differentiated agents",
            registered,
        )
