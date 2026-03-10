"""Integration Meter — Markov blanket computation.

Extracted from integration_meter.py for single-responsibility.
"""

from __future__ import annotations

from typing import Any, Optional

from mae_core.backbone.integration_meter_models import MarkovBlanketResult


def compute_markov_blanket(
    holon_id: str,
    holon_registry: Any,
    connection_registry: Any,
    grouping: dict,
) -> Optional[MarkovBlanketResult]:
    """Compute Markov blanket for a subsystem or organ.

    Internal states: children of holon_id
    Blanket states: parent + siblings + cross-connected systems
    External states: everything not internal or blanket
    Effectiveness: proportion of connections that stay within blanket

    Args:
        holon_id: The holon to analyze.
        holon_registry: HolonRegistry instance for hierarchy queries.
        connection_registry: ConnectionRegistry instance for connection queries.
        grouping: FRACTAL_GROUPING dict for enumerating all known systems.

    Returns:
        MarkovBlanketResult or None if holon not registered.
    """
    entry = holon_registry.get_entry(holon_id)
    if entry is None:
        return None

    # Internal: children
    internal = list(holon_registry.get_children(holon_id))

    # Parent and siblings
    parent_id = entry.parent_id
    blanket = set()
    if parent_id:
        blanket.add(parent_id)
        siblings = holon_registry.get_children(parent_id)
        for sib in siblings:
            if sib != holon_id:
                blanket.add(sib)

    # Cross-connections: systems from other subsystems connected to this one
    cross_count = 0
    if connection_registry is not None:
        for child_id in internal:
            try:
                connections = connection_registry.get_connections_for_system(child_id)
                for conn in connections:
                    source = getattr(conn, "source", None)
                    target = getattr(conn, "target", None)
                    other = target if source == child_id else source
                    if other and other not in internal and other != holon_id:
                        blanket.add(other)
                        cross_count += 1
            except Exception:
                pass

    blanket_list = sorted(blanket)

    # External: all registered holons not in internal or blanket
    all_holons = set()
    try:
        stats = holon_registry.get_statistics()
        all_holons = set(stats.get("type_counts", {}).keys())
    except Exception:
        pass

    # Simpler: iterate known systems from grouping
    all_systems = set()
    for organ_name, subs in grouping.items():
        all_systems.add(organ_name)
        for sub_name, sys_ids in subs.items():
            all_systems.add(sub_name)
            all_systems.update(sys_ids)
    all_systems.add("mae")

    internal_set = set(internal)
    blanket_set = set(blanket_list)
    external = sorted(all_systems - internal_set - blanket_set - {holon_id})

    # Effectiveness: what fraction of this subsystem's connections
    # stay within the blanket (vs. reaching external)
    total_conn = cross_count + len(internal) * 2  # internal K3 connections
    blanket_conn = cross_count  # connections that cross the boundary
    if total_conn > 0:
        # Higher is better: most connections are internal
        effectiveness = 1.0 - (blanket_conn / total_conn)
    else:
        effectiveness = 1.0  # No connections = trivially isolated

    return MarkovBlanketResult(
        holon_id=holon_id,
        internal_states=internal,
        blanket_states=blanket_list,
        external_states=external,
        blanket_effectiveness=effectiveness,
        cross_connections=cross_count,
    )
