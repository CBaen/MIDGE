"""Stem Cell Protocol - Universal agent genome, epigenome, and role profiles.

Every MycelialAgent carries the same genome (capability catalog). Differentiation
happens through the epigenome (per-agent configuration). Role profiles are
predefined epigenome configurations. Agents can re-differentiate at runtime.

Like biological stem cells: same DNA, different gene expression.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channel constants
CH_STEM_CELL_REGISTERED = "stem_cell.registered"
CH_STEM_CELL_REDIFFERENTIATED = "stem_cell.redifferentiated"


# ---------------------------------------------------------------------------
# Gene Descriptor — one configurable knob
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneDescriptor:
    """A single configurable parameter in the agent genome."""

    name: str
    mixin: str
    gene_type: type
    default: Any
    min_val: Any = None
    max_val: Any = None

    def validate(self, value: Any) -> bool:
        """Check if a value is valid for this gene."""
        if not isinstance(value, self.gene_type) and self.gene_type is not set:
            return False
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False
        return True


# ---------------------------------------------------------------------------
# Agent Genome — the universal capability catalog (shared by all agents)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentGenome:
    """The universal agent blueprint. Every agent shares the same genome.

    This is the DNA — it catalogues what's possible. The 20 knobs across
    10 mixins that define the full space of agent behavior.
    """

    genes: tuple[GeneDescriptor, ...]

    def get_gene(self, name: str) -> GeneDescriptor | None:
        """Look up a gene by name."""
        for gene in self.genes:
            if gene.name == name:
                return gene
        return None

    def get_defaults(self) -> dict[str, Any]:
        """Return dict of all gene names to their default values."""
        return {g.name: g.default for g in self.genes}

    def get_gene_names(self) -> set[str]:
        """Return set of all gene names."""
        return {g.name for g in self.genes}


# The 20 knobs across 10 mixins — discovered from actual mixin code
DEFAULT_GENOME = AgentGenome(genes=(
    # Convergence (safety)
    GeneDescriptor("max_learning_iterations", "convergence", int, 100, 10, 1000),
    GeneDescriptor("convergence_threshold", "convergence", float, 0.01, 0.001, 0.1),
    GeneDescriptor("satisfaction_threshold", "convergence", float, 0.85, 0.5, 0.99),
    # Gamification (motivation)
    GeneDescriptor("exploration_bonus", "gamification", float, 0.1, 0.0, 0.5),
    GeneDescriptor("novelty_threshold", "gamification", float, 0.8, 0.3, 1.0),
    # Stigmergy (pheromones)
    GeneDescriptor("sensing_radius", "stigmergy", float, 5.0, 0.5, 20.0),
    GeneDescriptor("trail_following", "stigmergy", bool, True),
    # GNN Communication (routing)
    GeneDescriptor("capabilities", "gnn_communication", set, frozenset()),
    # Transfer Learning (knowledge sharing)
    GeneDescriptor("transfer_enabled", "transfer", bool, False),
    GeneDescriptor("maml_enabled", "transfer", bool, False),
    # Episodic Memory (experience)
    GeneDescriptor("replay_enabled", "episodic", bool, False),
    GeneDescriptor("consolidation_enabled", "episodic", bool, False),
    GeneDescriptor("semantic_search_enabled", "episodic", bool, False),
    GeneDescriptor("generative_memory_enabled", "episodic", bool, False),
    GeneDescriptor("replay_frequency", "episodic", int, 4, 1, 20),
    GeneDescriptor("replay_batch_size", "episodic", int, 32, 8, 128),
    # Collective Consensus (swarm coordination)
    GeneDescriptor("quorum_sensing_enabled", "collective", bool, False),
    GeneDescriptor("quorum_check_interval", "collective", float, 1.0, 0.1, 5.0),
    # Advanced Features (planning)
    GeneDescriptor("world_model_enabled", "advanced", bool, False),
    GeneDescriptor("planning_horizon", "advanced", int, 5, 1, 20),
    # External API (oracle pathway)
    GeneDescriptor("api_call_enabled", "external", bool, False),
    GeneDescriptor("llm_prompt_quality", "external", float, 0.5, 0.0, 1.0),
))


# ---------------------------------------------------------------------------
# Agent Epigenome — per-agent expression of the genome
# ---------------------------------------------------------------------------

@dataclass
class AgentEpigenome:
    """Per-agent gene expression. Same DNA, different activation.

    The epigenome records which genes are "turned on" and at what level.
    It also tracks the agent's role history (lineage) for memory across
    re-differentiation events.
    """

    role: str = "STEM"
    config: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    locked: bool = False
    applied_step: int = 0


# ---------------------------------------------------------------------------
# Role Profiles — predefined epigenome configurations
# ---------------------------------------------------------------------------

ROLE_PROFILES: dict[str, dict[str, Any]] = {
    "STEM": {
        # Balanced, undifferentiated — all genome defaults
    },
    "EXPLORER": {
        "exploration_bonus": 0.4,
        "novelty_threshold": 0.5,
        "sensing_radius": 15.0,
        "trail_following": True,
    },
    "LEARNER": {
        "replay_enabled": True,
        "consolidation_enabled": True,
        "transfer_enabled": True,
        "maml_enabled": True,
        "replay_frequency": 2,
    },
    "COMMUNICATOR": {
        "quorum_sensing_enabled": True,
        "capabilities": frozenset({"broadcast", "relay", "coordinate"}),
    },
    "HEALER": {
        "satisfaction_threshold": 0.95,
        "convergence_threshold": 0.001,
        "max_learning_iterations": 300,
    },
    "COORDINATOR": {
        "quorum_sensing_enabled": True,
        "world_model_enabled": True,
        "planning_horizon": 10,
    },
    "SPECIALIST": {
        "max_learning_iterations": 500,
        "replay_enabled": True,
        "replay_batch_size": 64,
        "semantic_search_enabled": True,
        "consolidation_enabled": True,
    },
    "API_CALLER": {
        "api_call_enabled": True,
        "llm_prompt_quality": 0.7,
        "world_model_enabled": True,
        "planning_horizon": 5,
    },
    "LLM_SPECIALIST": {
        "api_call_enabled": True,
        "llm_prompt_quality": 0.9,
        "world_model_enabled": True,
        "planning_horizon": 10,
        "semantic_search_enabled": True,
        "consolidation_enabled": True,
        "replay_enabled": True,
    },
    # --- MIDGE market roles (Law 5: same genome, market-specialized epigenome) ---
    "SEC_WATCHER": {
        "api_call_enabled": True,
        "llm_prompt_quality": 0.6,
        "replay_enabled": True,
        "consolidation_enabled": True,
        "semantic_search_enabled": True,
        "sensing_radius": 10.0,
        "exploration_bonus": 0.15,
        "world_model_enabled": True,
        "planning_horizon": 7,
        "capabilities": frozenset({"market_sense", "insider_track", "sec_watch"}),
    },
    "CONTRACT_TRACKER": {
        "api_call_enabled": True,
        "llm_prompt_quality": 0.6,
        "quorum_sensing_enabled": True,
        "world_model_enabled": True,
        "planning_horizon": 14,
        "replay_enabled": True,
        "transfer_enabled": True,
        "capabilities": frozenset({"market_sense", "contract_track", "govt_monitor"}),
    },
    "MARKET_ANALYST": {
        "api_call_enabled": True,
        "llm_prompt_quality": 0.85,
        "world_model_enabled": True,
        "planning_horizon": 10,
        "replay_enabled": True,
        "consolidation_enabled": True,
        "semantic_search_enabled": True,
        "generative_memory_enabled": True,
        "transfer_enabled": True,
        "maml_enabled": True,
        "quorum_sensing_enabled": True,
        "capabilities": frozenset({"market_analyze", "convergence_detect", "signal_synthesize"}),
    },
}


# ---------------------------------------------------------------------------
# Redifferentiate — change an agent's role at runtime
# ---------------------------------------------------------------------------

def redifferentiate(
    agent: Any,
    new_role: str,
    registry: "StemCellRegistry | None" = None,
    step: int = 0,
) -> AgentEpigenome:
    """Change an agent's role by applying a new epigenome configuration.

    Like a stem cell differentiating into a new cell type. The agent's
    mixin attributes are updated to match the role profile. Previous
    role is recorded in lineage.

    Args:
        agent: A MycelialAgent (or any agent with agent_config dict).
        new_role: Role name from ROLE_PROFILES (or custom).
        registry: Optional StemCellRegistry to update.
        step: Current model step for timestamp.

    Returns:
        The new AgentEpigenome after re-differentiation.
    """
    agent_id = str(getattr(agent, "unique_id", id(agent)))

    # Get current epigenome
    if registry and agent_id in registry._epigenomes:
        epigenome = registry._epigenomes[agent_id]
    else:
        epigenome = AgentEpigenome()

    # Respect lock
    if epigenome.locked:
        logger.warning(
            "Agent %s is locked — cannot redifferentiate to %s",
            agent_id, new_role,
        )
        return epigenome

    # Record lineage
    if epigenome.role:
        epigenome.lineage.append(epigenome.role)

    # Get role config (empty dict for unknown roles — advisory, not blocking)
    role_config = ROLE_PROFILES.get(new_role, {})

    # Build full config: genome defaults + role overrides
    full_config = DEFAULT_GENOME.get_defaults()
    full_config.update(role_config)

    # Apply to agent's config dict
    if hasattr(agent, "agent_config") and isinstance(agent.agent_config, dict):
        agent.agent_config.update(full_config)

    # Apply directly to agent attributes where they exist
    for knob_name, value in full_config.items():
        if hasattr(agent, knob_name):
            setattr(agent, knob_name, value)

    # Update epigenome
    epigenome.role = new_role
    epigenome.config = dict(role_config)
    epigenome.applied_step = step

    # Update registry if available
    if registry:
        registry._epigenomes[agent_id] = epigenome
        registry._rediff_count += 1

        # Emit event
        if registry._event_bus:
            registry._event_bus.publish(CH_STEM_CELL_REDIFFERENTIATED, {
                "agent_id": agent_id,
                "new_role": new_role,
                "previous_role": epigenome.lineage[-1] if epigenome.lineage else "STEM",
                "step": step,
            })

    logger.info("Agent %s redifferentiated to %s", agent_id, new_role)
    return epigenome


# ---------------------------------------------------------------------------
# Stem Cell Registry — tracks all agents and their epigenomes
# ---------------------------------------------------------------------------

class StemCellRegistry:
    """Registry tracking agent genomes, epigenomes, and role distribution.

    Like the bone marrow — keeps track of all stem cells and their
    current differentiation state.
    """

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus
        self._epigenomes: dict[str, AgentEpigenome] = {}
        self._agents: dict[str, Any] = {}
        self._rediff_count: int = 0

    def register(self, agent: Any) -> AgentEpigenome:
        """Register an agent with default STEM epigenome."""
        agent_id = str(getattr(agent, "unique_id", id(agent)))
        epigenome = AgentEpigenome(role="STEM", config={})
        self._epigenomes[agent_id] = epigenome
        self._agents[agent_id] = agent

        if self._event_bus:
            self._event_bus.publish(CH_STEM_CELL_REGISTERED, {
                "agent_id": agent_id,
            })

        return epigenome

    def get_epigenome(self, agent_id: str) -> AgentEpigenome | None:
        """Get an agent's current epigenome."""
        return self._epigenomes.get(str(agent_id))

    def get_role_distribution(self) -> dict[str, int]:
        """Count agents by current role."""
        dist: dict[str, int] = {}
        for epi in self._epigenomes.values():
            dist[epi.role] = dist.get(epi.role, 0) + 1
        return dist

    def get_agents_by_role(self, role: str) -> list[str]:
        """Get agent IDs with a specific role."""
        return [
            aid for aid, epi in self._epigenomes.items()
            if epi.role == role
        ]

    def get_lineage(self, agent_id: str) -> list[str]:
        """Get an agent's role history."""
        epi = self._epigenomes.get(str(agent_id))
        return list(epi.lineage) if epi else []

    def redifferentiate(self, agent_id: str, new_role: str, step: int = 0) -> AgentEpigenome | None:
        """Redifferentiate an agent by ID."""
        agent = self._agents.get(str(agent_id))
        if agent is None:
            logger.warning("Agent %s not registered — cannot redifferentiate", agent_id)
            return None
        return redifferentiate(agent, new_role, registry=self, step=step)

    def get_statistics(self) -> dict[str, Any]:
        """Summary statistics for monitoring."""
        return {
            "total_registered": len(self._epigenomes),
            "role_distribution": self.get_role_distribution(),
            "total_redifferentiations": self._rediff_count,
            "genome_size": len(DEFAULT_GENOME.genes),
            "available_roles": list(ROLE_PROFILES.keys()),
        }
