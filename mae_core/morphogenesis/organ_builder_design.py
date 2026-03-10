"""
organ_builder_design.py - Blueprint design and senescence handling for OrganBuilder.

Extracted from organ_builder.py. Contains:
  - design_organ: problem signature -> OrganBlueprint translation
  - on_system_senescent: EventBus callback for end-of-life events
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from mae_core.morphogenesis.organ_builder_models import (
    CoordinationProtocol,
    OrganBlueprint,
    OrganTopology,
    ProblemSignature,
)

logger = logging.getLogger(__name__)

CH_REBUILD_REQUESTED = "morphogenesis.rebuild_requested"


def design_organ(
    signature: ProblemSignature,
    name: Optional[str] = None,
) -> OrganBlueprint:
    """Design an organ blueprint based on problem characteristics.

    Composition rules (inspired by v5-pivot OrganBuilder):
    - High coordination -> add coordinators
    - High exploration -> add explorers
    - Sparse rewards or high complexity -> add specialists
    - High risk -> add risk managers
    - Always: generalists fill remaining slots
    """
    composition: dict[str, int] = {}

    # Coordinators: needed when tight coordination required
    if signature.coordination_level > 0.6:
        coord_count = max(1, int(signature.coordination_level * 5))
        composition["coordinator"] = coord_count

    # Explorers: needed when exploration is high
    if signature.exploration_level > 0.5:
        explore_count = max(1, int(signature.exploration_level * 8))
        composition["explorer"] = explore_count

    # Specialists: for complex or sparse-reward problems
    if signature.sparse_rewards or signature.complexity > 0.6:
        spec_count = max(2, int(signature.complexity * 6))
        composition["specialist"] = spec_count

    # Risk managers: for dangerous environments
    if signature.risk_level > 0.6:
        risk_count = max(1, int(signature.risk_level * 3))
        composition["risk_manager"] = risk_count

    # Generalists: fill to minimum team size
    current_total = sum(composition.values())
    min_team = max(3, current_total // 4)  # At least 3, or 1/4 of specialized
    if current_total < min_team:
        composition["generalist"] = min_team - current_total
    elif "generalist" not in composition:
        composition["generalist"] = max(1, current_total // 4)

    # Select topology based on coordination level
    if signature.coordination_level > 0.8:
        topology = OrganTopology.HIERARCHICAL
    elif signature.coordination_level > 0.5:
        topology = OrganTopology.STAR
    else:
        topology = OrganTopology.MESH

    # Select protocol
    if signature.coordination_level > 0.7:
        protocol = CoordinationProtocol.HIERARCHICAL
    elif signature.exploration_level > 0.6:
        protocol = CoordinationProtocol.AUCTION
    else:
        protocol = CoordinationProtocol.CONSENSUS

    # Lifecycle
    transient = signature.temporal_pattern == "episodic"
    max_lifetime = 3600.0 if transient else None

    organ_name = name or f"organ_{signature.domain}_{uuid.uuid4().hex[:4]}"

    blueprint = OrganBlueprint(
        name=organ_name,
        purpose=f"Solve {signature.domain} problems (complexity={signature.complexity:.1f})",
        composition=composition,
        topology=topology,
        protocol=protocol,
        transient=transient,
        max_lifetime=max_lifetime,
        problem_signature=signature,
    )

    logger.info(
        "Designed organ '%s': %d agents, %s topology, %s protocol",
        blueprint.name,
        blueprint.total_agents,
        blueprint.topology.value,
        blueprint.protocol.value,
    )
    return blueprint


def on_system_senescent(
    organ_builder: Any,
    channel: str,
    message: Any,
) -> None:
    """Handle a senescent system event from SenescenceManager.

    Triggered when a system's wear reaches 1.0 (end-of-life). Prunes
    any organs that should dissolve, then publishes a rebuild request
    so downstream systems know replacement is needed.

    Args:
        organ_builder: The OrganBuilder instance.
        channel: The EventBus channel (always "emergent.system_senescent").
        message: JSON-serialized payload containing at minimum
            ``system_name``, ``wear_level``, ``total_steps_active``,
            and ``step``.
    """
    # Parse the message payload (EventBus delivers JSON strings)
    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, ValueError):
            logger.warning("OrganBuilder: could not parse senescent message: %s", message)
            return
    elif isinstance(message, dict):
        payload = message
    else:
        logger.warning("OrganBuilder: unexpected message type %s", type(message))
        return

    system_name = payload.get("system_name", "unknown")

    # Prune organs that have reached end-of-life
    organ_builder.prune_organs()

    # Publish rebuild notification so downstream systems can act
    if organ_builder._event_bus is not None:
        organ_builder._event_bus.publish(
            CH_REBUILD_REQUESTED,
            {"system_name": system_name, "reason": "senescence"},
        )

    logger.info(
        "OrganBuilder: senescent event for %s — pruned and rebuild requested",
        system_name,
    )
