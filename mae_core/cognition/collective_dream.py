"""Collective Dream Planning - Swarm-validated imagination before execution.

Multiple expert agents imagine trajectories using the WorldModel.
All agents vote on plans. Expertise-weighted consensus determines action.
Low consensus triggers morphogenesis (creates specialist agents).

Biological analogy: Bee waggle dance consensus.
Based on: Seeley (2010) "Honeybee Democracy", swarm intelligence.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .world_model import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class Dream:
    """An imagined trajectory from a single expert dreamer."""

    dreamer_id: str
    trajectory: list[tuple[Any, Any, float]]  # [(state, action, reward), ...]
    total_reward: float = 0.0
    dream_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.total_reward and self.trajectory:
            self.total_reward = sum(r for _, _, r in self.trajectory)
        if not self.dream_hash:
            content = f"{self.dreamer_id}:{len(self.trajectory)}:{self.total_reward:.4f}"
            self.dream_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ConsensusResult:
    """Outcome of collective dream voting."""

    approved: bool
    consensus_strength: float  # [0, 1]
    best_dream: Dream | None
    vote_counts: dict[str, int] = field(default_factory=dict)
    total_voters: int = 0
    expert_votes: int = 0


@dataclass
class DreamAgent:
    """Minimal agent interface for dream planning."""

    agent_id: str
    expertise: float = 0.5  # [0, 1]
    domain: str = "general"

    def select_action(self, state: np.ndarray) -> int:
        return int(np.random.randint(0, 5))


class CollectiveDreamPlanner:
    """Swarm-validated imagination: experts dream, everyone votes.

    Process:
    1. Select expert dreamers (by expertise score)
    2. Generate dreams using WorldModel.rollout()
    3. Broadcast dreams to all agents
    4. Collect expertise-weighted votes
    5. If consensus >= threshold: execute best plan
    6. If consensus < threshold: trigger morphogenesis for specialist creation

    Expert weighting: expertise >= 0.9 gets 5x vote weight.
    """

    def __init__(
        self,
        world_model: WorldModel,
        consensus_threshold: float = 0.7,
        expert_weight: float = 5.0,
        specialist_weight: float = 2.5,
        expert_threshold: float = 0.9,
        specialist_threshold: float = 0.7,
        morphogenesis_callback: Any | None = None,
        event_bus: Any = None,
        world_models: list[WorldModel] | None = None,
    ) -> None:
        self._world_model = world_model
        self._consensus_threshold = consensus_threshold
        self._expert_weight = expert_weight
        self._specialist_weight = specialist_weight
        self._expert_thresh = expert_threshold
        self._specialist_thresh = specialist_threshold
        self._morphogenesis_cb = morphogenesis_callback
        self._bus = event_bus
        self._world_models = world_models or [world_model]

        self._agents: list[DreamAgent] = []
        self._dreamer_world_models: dict[str, Any] = {}
        self._stats = {
            "total_plans": 0,
            "plans_approved": 0,
            "plans_rejected": 0,
            "low_consensus_events": 0,
            "specialists_created": 0,
            "avg_consensus": 0.0,
            "consensus_history": [],
        }

    def register_agent(self, agent: DreamAgent) -> None:
        self._agents.append(agent)

    def remove_agent(self, agent_id: str) -> bool:
        before = len(self._agents)
        self._agents = [a for a in self._agents if a.agent_id != agent_id]
        return len(self._agents) < before

    def collective_plan(
        self,
        initial_state: np.ndarray,
        horizon: int = 10,
        num_dreamers: int = 5,
    ) -> tuple[list[tuple[Any, Any, float]], dict[str, Any]]:
        """Execute full collective dream planning cycle."""
        self._stats["total_plans"] += 1

        # 1. Select expert dreamers (highest expertise)
        dreamers = self.select_expert_dreamers(num_dreamers)

        # 2. Generate dreams
        dreams = []
        for dreamer in dreamers:
            dream = self._generate_dream(initial_state, dreamer, horizon)
            if dream.trajectory:
                dreams.append(dream)

        if not dreams:
            return [], {"status": "no_dreams", "consensus": 0.0}

        # 3. Collect consensus votes
        result = self.collect_consensus_votes(dreams)

        # 4. Update statistics
        self._stats["consensus_history"].append(result.consensus_strength)
        if len(self._stats["consensus_history"]) > 100:
            self._stats["consensus_history"] = self._stats["consensus_history"][-100:]
        self._stats["avg_consensus"] = float(
            np.mean(self._stats["consensus_history"])
        )

        # 5. Handle result
        if result.approved and result.best_dream:
            self._stats["plans_approved"] += 1
            info = {
                "status": "approved",
                "consensus": result.consensus_strength,
                "dreamer": result.best_dream.dreamer_id,
                "total_reward": result.best_dream.total_reward,
                "total_voters": result.total_voters,
            }
            self._publish_dream_complete(info)
            return result.best_dream.trajectory, info

        # 6. Low consensus → morphogenesis trigger
        self._stats["plans_rejected"] += 1
        self._stats["low_consensus_events"] += 1
        self._handle_low_consensus(initial_state, dreams, result.consensus_strength)

        # Return best dream anyway (it's the best we have)
        best = max(dreams, key=lambda d: d.total_reward)
        info = {
            "status": "low_consensus",
            "consensus": result.consensus_strength,
            "threshold": self._consensus_threshold,
            "specialist_requested": True,
        }
        self._publish_dream_complete(info)
        return best.trajectory, info

    def select_expert_dreamers(self, num_dreamers: int) -> list[DreamAgent]:
        """Select top expertise agents as dreamers."""
        if not self._agents:
            return [DreamAgent(agent_id=f"default-{i}") for i in range(num_dreamers)]

        sorted_agents = sorted(self._agents, key=lambda a: a.expertise, reverse=True)
        return sorted_agents[:num_dreamers]

    def collect_consensus_votes(self, dreams: list[Dream]) -> ConsensusResult:
        """Collect expertise-weighted votes from all agents."""
        voters = self._agents or [DreamAgent(agent_id=f"voter-{i}") for i in range(10)]
        vote_counts: dict[str, float] = {}  # dream_hash -> weighted votes
        raw_counts: dict[str, int] = {}

        total_weight = 0.0

        for voter in voters:
            # Determine vote weight based on expertise
            if voter.expertise >= self._expert_thresh:
                weight = self._expert_weight
            elif voter.expertise >= self._specialist_thresh:
                weight = self._specialist_weight
            else:
                weight = 1.0

            # Agent votes for dream with highest reward (simple heuristic)
            voted_dream = max(dreams, key=lambda d: d.total_reward + np.random.randn() * 0.1)
            vote_counts[voted_dream.dream_hash] = (
                vote_counts.get(voted_dream.dream_hash, 0.0) + weight
            )
            raw_counts[voted_dream.dream_hash] = (
                raw_counts.get(voted_dream.dream_hash, 0) + 1
            )
            total_weight += weight

        # Find winning dream
        best_hash = max(vote_counts, key=vote_counts.get) if vote_counts else ""
        best_dream = next(
            (d for d in dreams if d.dream_hash == best_hash), None
        )

        consensus_strength = (
            vote_counts.get(best_hash, 0.0) / total_weight
            if total_weight > 0
            else 0.0
        )

        expert_votes = sum(1 for v in voters if v.expertise >= self._expert_thresh)

        return ConsensusResult(
            approved=consensus_strength >= self._consensus_threshold,
            consensus_strength=consensus_strength,
            best_dream=best_dream,
            vote_counts={k: v for k, v in raw_counts.items()},
            total_voters=len(voters),
            expert_votes=expert_votes,
        )

    def register_dreamer_world_model(self, agent_id: str, world_model: Any) -> None:
        """Collect per-agent world models for ensemble dreaming."""
        self._dreamer_world_models[agent_id] = world_model

    def get_planning_statistics(self) -> dict[str, Any]:
        total = max(self._stats["total_plans"], 1)
        return {
            **self._stats,
            "approval_rate": self._stats["plans_approved"] / total,
            "rejection_rate": self._stats["plans_rejected"] / total,
            "agents_registered": len(self._agents),
        }

    # --- Internal ---

    def _publish_dream_complete(self, info: dict[str, Any]) -> None:
        """Publish collective dream completion event on EventBus."""
        if self._bus is not None:
            try:
                self._bus.publish("cognition.collective_dream_complete", {
                    "status": info.get("status", "unknown"),
                    "consensus": info.get("consensus", 0.0),
                    "total_plans": self._stats["total_plans"],
                    "agents_count": len(self._agents),
                    "world_models_count": len(self._world_models),
                })
            except Exception:
                logger.debug("EventBus publish failed for collective_dream_complete")

    def _generate_dream(
        self, initial_state: np.ndarray, dreamer: DreamAgent, horizon: int
    ) -> Dream:
        """Generate a dream trajectory using WorldModel."""
        try:
            rollout = self._world_model.rollout(
                initial_state, dreamer, horizon, deterministic=False
            )
            trajectory = []
            for i in range(len(rollout["actions"])):
                trajectory.append((
                    rollout["states"][i],
                    rollout["actions"][i],
                    rollout["rewards"][i],
                ))
            return Dream(
                dreamer_id=dreamer.agent_id,
                trajectory=trajectory,
                total_reward=rollout["total_reward"],
            )
        except Exception:
            logger.exception("Dream generation failed for %s", dreamer.agent_id)
            return Dream(dreamer_id=dreamer.agent_id, trajectory=[])

    def _handle_low_consensus(
        self,
        initial_state: np.ndarray,
        dreams: list[Dream],
        consensus: float,
    ) -> None:
        """Trigger morphogenesis when consensus is low."""
        if self._morphogenesis_cb is not None:
            problem_sig = {
                "problem_type": "low_planning_consensus",
                "state_shape": initial_state.shape,
                "num_dreams": len(dreams),
                "consensus_level": consensus,
                "dream_rewards": [d.total_reward for d in dreams],
                "timestamp": time.time(),
            }
            try:
                self._morphogenesis_cb(problem_sig)
                self._stats["specialists_created"] += 1
            except Exception:
                logger.exception("Morphogenesis callback failed")

    def __repr__(self) -> str:
        return (
            f"CollectiveDreamPlanner(agents={len(self._agents)}, "
            f"plans={self._stats['total_plans']}, "
            f"approved={self._stats['plans_approved']})"
        )
