"""Fractal Generator - Makes Mae's recursive structure explicit.

Step 4 of the fractal architecture roadmap. Groups existing systems into
triadic subsystems and organs, giving Mae a proper fractal holarchy.

The generator is a triad with the holon protocol: 3 nodes, fully connected (K3),
where each node implements the 10 holon capabilities. Recursive application
builds the hierarchy: 3 processes → subsystem, 3 subsystems → organ, organs → Mae.

Advisory mode only. Virtual parent holons are registry metadata, not Python objects.

Action delegation classes (SubsystemAction, OrganAction, OrganismAction,
build_fractal_action) live in fractal_act.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mae_core.backbone.connection_registry import (
    ConnectionRegistry,
    ConnectionType,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.holon_protocol import HolonRegistry

logger = logging.getLogger(__name__)

# =====================================================================
# EventBus channels
# =====================================================================
CH_FRACTAL_TRIAD_CREATED = "fractal.triad_created"
CH_FRACTAL_ORGANIZED = "fractal.organized"

# =====================================================================
# Fractal levels
# =====================================================================


class FractalLevel(Enum):
    """Tiers in the fractal holarchy, from finest to coarsest."""

    PROCESS = "process"        # Individual system (leaf)
    SUBSYSTEM = "subsystem"    # Triad of processes
    MODULE = "module"          # Triad of subsystems (intermediate)
    ORGAN = "organ"            # Triad of subsystems/modules
    ORGAN_CLUSTER = "organ_cluster"  # Triad of organs (intermediate)
    ORGANISM = "organism"      # Mae (root)


# =====================================================================
# Result dataclasses
# =====================================================================


@dataclass
class TriadResult:
    """Result of creating a single triadic group."""

    parent_id: str
    children_ids: list[str]
    connections_created: int
    holon_type: str


@dataclass
class FractalReport:
    """Report from a full organize() operation."""

    organs_created: int = 0
    organ_clusters_created: int = 0
    subsystems_created: int = 0
    connections_created: int = 0
    max_depth: int = 0
    non_triadic_groups: list[str] = field(default_factory=list)
    grouping_map: dict[str, dict[str, list[str]]] = field(default_factory=dict)


# =====================================================================
# Mae's fractal blueprint
# =====================================================================

FRACTAL_GROUPING: dict[str, dict[str, list[str]]] = {
    "nervous-system": {
        "core-backbone": ["event_bus", "holon_registry", "connection_registry"],
        "enforcement": ["enforcer", "watchdog", "auditor"],
        "rhythm": ["circadian", "endocrine", "awareness_pulse"],
    },
    "sensory-system": {
        "fabric": ["substrate", "physarum", "signal_bus"],
        "routing": ["gnn_communicator", "stigmergy", "predictive_field"],
        "consensus": ["quorum_space", "nociception", "proprioception"],
    },
    "cognitive-system": {
        "reasoning": ["shared_world_model", "shared_causal_engine", "collective_dream"],
        "meta-learning": ["knowledge_base", "transfer_engine", "maml_learner"],
        "adaptation": ["curiosity", "haven", "imitation"],
        "temporal": ["temporal_memory", "worldline_planner", "validated_imagination"],
        "social-cognition": ["emotional_system", "theory_of_mind", "metacognition"],
    },
    "somatic-system": {
        "defense": ["threat_detector", "input_validator", "pearl_defense"],
        "emergence": ["auto_healer", "capability_discovery", "somatic_map"],
        "growth": ["morph_coordinator", "organ_builder", "reproductive_system"],
        "maintenance": ["lymphatic_system", "senescence", "boundary_membrane"],
    },
    "metabolic-system": {
        "digestion": ["digestive_system", "renal_filter", "microbiome"],
        "circulation": ["circulatory_system", "respiratory_system", "energy_reserve"],
        "regulation": ["homeostasis", "thermoregulation", "vestibular_system"],
    },
}

# Module-level grouping for organs with >3 subsystems.
# Groups subsystems into triadic modules where possible.
# Organs not listed here skip the module level (already triadic).
MODULE_GROUPING: dict[str, dict[str, list[str]]] = {
    "cognitive-system": {
        "cognitive-analytic": ["reasoning", "meta-learning", "adaptation"],
        "cognitive-social": ["temporal", "social-cognition"],
    },
    "somatic-system": {
        # Only [defense, emergence] grouped — leaves [growth, maintenance]
        # ungrouped, giving somatic-system 3 children (K3).
        "somatic-active": ["defense", "emergence"],
    },
}

# Organism-level grouping: groups 5 organs into triadic structure under mae.
# nervous-system is the bridge organ — stays as direct child of mae.
# Biologically: the nervous system spans internal regulation (vital) and
# external processing (cognitive), making it the natural integrator.
# Mae's children: [organ-cluster-vital, organ-cluster-cognitive, nervous-system] = K3.
# Note: organ clusters are holon-registry groupings only (structural awareness).
# The action hierarchy (build_fractal_action) bypasses clusters intentionally —
# same pattern as MODULE_GROUPING where modules exist in holon registry but not
# in the action chain.
ORGAN_GROUPING: dict[str, list[str]] = {
    "organ-cluster-vital": ["metabolic-system", "somatic-system"],
    "organ-cluster-cognitive": ["cognitive-system", "sensory-system"],
}


# =====================================================================
# FractalGenerator
# =====================================================================


class FractalGenerator:
    """Formalizes Mae's recursive triadic structure.

    Uses existing HolonRegistry and ConnectionRegistry APIs to reorganize
    flat system holons into a nested fractal hierarchy. Virtual parent
    holons (subsystems, organs) are registry metadata only.
    """

    def __init__(
        self,
        holon_registry: HolonRegistry,
        connection_registry: Optional[ConnectionRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self._registry = holon_registry
        self._connections = connection_registry
        self._bus = event_bus
        self._created_holons: list[str] = []

    # -----------------------------------------------------------------
    # Core atomic operation: create one triadic group
    # -----------------------------------------------------------------

    def generate_triad(
        self,
        name: str,
        holon_type: str,
        children_ids: list[str],
        parent_id: str,
    ) -> TriadResult:
        """Create a parent holon that groups children into a triad.

        1. Registers the parent holon in HolonRegistry
        2. Reparents each child to the new parent
        3. Wires K3 connections (each pair witnessed by the third)

        Advisory: warns if len(children_ids) != 3 but still proceeds.
        Idempotent: skips if parent already exists.
        """
        # Idempotent check
        if self._registry.get_entry(name) is not None:
            existing_children = self._registry.get_children(name)
            return TriadResult(
                parent_id=name,
                children_ids=existing_children,
                connections_created=0,
                holon_type=holon_type,
            )

        # Advisory: triadic principle prefers exactly 3
        if len(children_ids) != 3:
            logger.warning(
                "Fractal: %s has %d children (ideal: 3) — advisory",
                name, len(children_ids),
            )

        # Validate children exist
        valid_children = []
        for cid in children_ids:
            if self._registry.get_entry(cid) is not None:
                valid_children.append(cid)
            else:
                logger.warning("Fractal: child %s not found, skipping", cid)

        if not valid_children:
            logger.error("Fractal: no valid children for %s, aborting", name)
            return TriadResult(
                parent_id=name,
                children_ids=[],
                connections_created=0,
                holon_type=holon_type,
            )

        # 1. Register parent holon
        self._registry.register(
            holon_id=name,
            holon_type=holon_type,
            parent_id=parent_id,
        )
        self._created_holons.append(name)

        # 2. Reparent children
        for cid in valid_children:
            self._registry.set_parent(cid, name)

        # 3. Wire K3 connections (every pair, witnessed by the third)
        conns = 0
        if self._connections is not None and len(valid_children) >= 2:
            for i in range(len(valid_children)):
                for j in range(i + 1, len(valid_children)):
                    # Find a witness: prefer a third member of this triad
                    witness = None
                    for k in range(len(valid_children)):
                        if k != i and k != j:
                            witness = valid_children[k]
                            break
                    self._connections.register_connection(
                        source=valid_children[i],
                        target=valid_children[j],
                        connection_type=ConnectionType.DIRECT_REFERENCE,
                        description=f"Fractal K3: {name}",
                        witness=witness,
                    )
                    conns += 1

        # Publish event
        if self._bus:
            self._bus.publish(CH_FRACTAL_TRIAD_CREATED, {
                "parent_id": name,
                "holon_type": holon_type,
                "children": valid_children,
                "connections": conns,
                "timestamp": time.time(),
            })

        return TriadResult(
            parent_id=name,
            children_ids=valid_children,
            connections_created=conns,
            holon_type=holon_type,
        )

    # -----------------------------------------------------------------
    # Batch operation: create a level of triads
    # -----------------------------------------------------------------

    def generate_level(
        self,
        triads_spec: dict[str, list[str]],
        holon_type: str,
        parent_id: str,
    ) -> list[TriadResult]:
        """Create multiple triads at the same level under a common parent.

        Args:
            triads_spec: {triad_name: [child_ids]}
            holon_type: type for all created parent holons
            parent_id: parent for all triads at this level
        """
        results = []
        for triad_name, children in triads_spec.items():
            result = self.generate_triad(
                name=triad_name,
                holon_type=holon_type,
                children_ids=children,
                parent_id=parent_id,
            )
            results.append(result)
        return results

    # -----------------------------------------------------------------
    # Full reorganization
    # -----------------------------------------------------------------

    def organize(
        self,
        grouping: Optional[dict[str, dict[str, list[str]]]] = None,
        module_grouping: Optional[dict[str, dict[str, list[str]]]] = None,
        organ_grouping: Optional[dict[str, list[str]]] = None,
    ) -> FractalReport:
        """Reorganize flat system holons into a fractal hierarchy.

        Uses FRACTAL_GROUPING by default. For each organ:
        1. Creates subsystem triads (groups of ~3 systems)
        2. If MODULE_GROUPING specifies modules for this organ, groups
           subsystems into modules (intermediate level)
        3. Creates the organ triad of subsystems (or modules if present)
        4. If ORGAN_GROUPING is provided, groups organs into clusters
           under the organism (intermediate level between organ and mae)

        Idempotent: safe to call multiple times.
        """
        grouping = grouping or FRACTAL_GROUPING
        module_grouping = module_grouping or MODULE_GROUPING
        organ_grouping = organ_grouping or ORGAN_GROUPING
        report = FractalReport(grouping_map=dict(grouping))

        for organ_name, subsystems_spec in grouping.items():
            # First pass: create subsystem triads
            subsystem_ids = []
            for sub_name, system_ids in subsystems_spec.items():
                result = self.generate_triad(
                    name=sub_name,
                    holon_type=FractalLevel.SUBSYSTEM.value,
                    children_ids=system_ids,
                    parent_id=organ_name,  # Will be reparented when organ/module is created
                )
                report.connections_created += result.connections_created
                report.subsystems_created += 1
                subsystem_ids.append(sub_name)

                if len(system_ids) != 3:
                    report.non_triadic_groups.append(
                        f"{sub_name} ({len(system_ids)} members)"
                    )

            # Module pass: if this organ has module grouping, create modules
            organ_children = subsystem_ids  # Default: subsystems are organ children
            if organ_name in module_grouping:
                modules_spec = module_grouping[organ_name]
                module_ids = []
                for mod_name, mod_subsystems in modules_spec.items():
                    mod_result = self.generate_triad(
                        name=mod_name,
                        holon_type=FractalLevel.MODULE.value,
                        children_ids=mod_subsystems,
                        parent_id=organ_name,
                    )
                    report.connections_created += mod_result.connections_created
                    module_ids.append(mod_name)

                    if len(mod_subsystems) != 3:
                        report.non_triadic_groups.append(
                            f"{mod_name} ({len(mod_subsystems)} subsystems)"
                        )

                # Any subsystems NOT in a module stay as direct organ children
                grouped = set()
                for mod_subs in modules_spec.values():
                    grouped.update(mod_subs)
                ungrouped = [s for s in subsystem_ids if s not in grouped]
                organ_children = module_ids + ungrouped

            # Final pass: create organ triad
            organ_result = self.generate_triad(
                name=organ_name,
                holon_type=FractalLevel.ORGAN.value,
                children_ids=organ_children,
                parent_id="mae",
            )
            report.connections_created += organ_result.connections_created
            report.organs_created += 1

            if len(organ_children) != 3:
                report.non_triadic_groups.append(
                    f"{organ_name} ({len(organ_children)} children)"
                )

        # Organism-level grouping: cluster organs under mae.
        # Organs listed in clusters get reparented; unlisted organs
        # (bridge organs like nervous-system) stay as direct mae children.
        # Only applied when all referenced organs exist (skip for partial organize).
        if organ_grouping:
            for cluster_name, cluster_organs in organ_grouping.items():
                # Only create cluster if all its organ children exist
                valid = [o for o in cluster_organs
                         if self._registry.get_entry(o) is not None]
                if len(valid) < 2:
                    continue  # Skip: not enough organs for a meaningful cluster

                cluster_result = self.generate_triad(
                    name=cluster_name,
                    holon_type=FractalLevel.ORGAN_CLUSTER.value,
                    children_ids=cluster_organs,
                    parent_id="mae",
                )
                report.connections_created += cluster_result.connections_created
                report.organ_clusters_created += 1

                if len(cluster_organs) != 3:
                    report.non_triadic_groups.append(
                        f"{cluster_name} ({len(cluster_organs)} organs)"
                    )

        report.max_depth = self._registry._compute_max_depth()

        # Publish summary
        if self._bus:
            self._bus.publish(CH_FRACTAL_ORGANIZED, {
                "organs": report.organs_created,
                "organ_clusters": report.organ_clusters_created,
                "subsystems": report.subsystems_created,
                "connections": report.connections_created,
                "max_depth": report.max_depth,
                "non_triadic": report.non_triadic_groups,
                "timestamp": time.time(),
            })

        logger.info(
            "Fractal organized: %d organs, %d clusters, %d subsystems, %d K3 connections, depth=%d",
            report.organs_created,
            report.organ_clusters_created,
            report.subsystems_created,
            report.connections_created,
            report.max_depth,
        )
        if report.non_triadic_groups:
            logger.warning(
                "Non-triadic groups (advisory): %s",
                ", ".join(report.non_triadic_groups),
            )

        return report

    # -----------------------------------------------------------------
    # Verification
    # -----------------------------------------------------------------

    def verify_triadic_integrity(self) -> dict[str, Any]:
        """Check that every non-leaf holon has exactly 3 children.

        Returns advisory report — does not block anything.
        """
        violations: list[dict[str, Any]] = []
        clean = 0

        for holon_id in self._created_holons:
            entry = self._registry.get_entry(holon_id)
            if entry is None:
                continue
            children = self._registry.get_children(holon_id)
            if len(children) == 3:
                clean += 1
            elif len(children) > 0:
                violations.append({
                    "holon_id": holon_id,
                    "holon_type": entry.holon_type,
                    "children_count": len(children),
                    "expected": 3,
                })

        return {
            "clean_triads": clean,
            "violations": violations,
            "total_checked": len(self._created_holons),
        }

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Current fractal structure statistics."""
        stats = self._registry.get_statistics()
        return {
            "created_holons": len(self._created_holons),
            "created_holon_ids": list(self._created_holons),
            "registry_max_depth": stats.get("max_depth", 0),
            "registry_total": stats.get("total_holons", 0),
            "type_counts": stats.get("type_counts", {}),
        }
