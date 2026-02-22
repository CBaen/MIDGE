"""Mycelial Agent - Full-featured agent composing all mixins.

This is Mae's standard agent type. It composes all 9 capability mixins
onto the thin BaseAgent, creating the complete biological agent.

The mixin composition order follows the biological metaphor:
1. Convergence (safety) - knows when to stop
2. Gamification (motivation) - knows why to continue
3. Signal Processing (nerves) - fast reflexes
4. Stigmergy (pheromones) - environmental memory
5. GNN Communication (intelligent routing) - targeted messages
6. Transfer Learning (knowledge sharing) - cross-task wisdom
7. Episodic Memory (experience) - learns from the past
8. Collective Consensus (swarm) - population coordination
9. Advanced Features (cognition) - world model, morphogenesis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from mae_core.agents.base_agent import BaseAgent
from mae_core.communication.signal_priority import PriorityConfig, SignalPriorityResolver
from mae_core.agents.mixins.advanced_features import AdvancedFeaturesMixin
from mae_core.agents.mixins.collective_consensus import CollectiveConsensusMixin
from mae_core.agents.mixins.convergence import ConvergenceMixin
from mae_core.agents.mixins.episodic_memory import EpisodicMemoryMixin
from mae_core.agents.mixins.gamification import GamificationMixin
from mae_core.agents.mixins.gnn_communication import GNNCommunicationMixin
from mae_core.agents.mixins.signal_processing import SignalProcessingMixin
from mae_core.agents.mixins.stigmergy import StigmergyMixin
from mae_core.agents.mixins.transfer_learning import TransferLearningMixin
from mae_core.backbone.holon_mixin import HolonMixin
from mae_core.agents.lifecycle_sensing import SensingLifecycleMixin
from mae_core.agents.lifecycle_decision import DecisionActionLifecycleMixin
from mae_core.agents.lifecycle_learning import LearningLifecycleMixin
from mae_core.agents.lifecycle_communication import CommunicationLifecycleMixin

logger = logging.getLogger(__name__)


class MycelialAgent(
    HolonMixin,
    SensingLifecycleMixin,
    DecisionActionLifecycleMixin,
    LearningLifecycleMixin,
    CommunicationLifecycleMixin,
    ConvergenceMixin,
    GamificationMixin,
    SignalProcessingMixin,
    StigmergyMixin,
    GNNCommunicationMixin,
    TransferLearningMixin,
    EpisodicMemoryMixin,
    CollectiveConsensusMixin,
    AdvancedFeaturesMixin,
    BaseAgent,
):
    """Complete biological agent with all subsystems.

    Inherits from BaseAgent (Mesa 3.4) and all 10 mixins.
    Each mixin's _init_*() is called explicitly to avoid MRO issues.
    HolonMixin is initialized LAST so all other mixin state is available.
    """

    def __init__(
        self,
        model: Any,
        agent_type: str = "mycelial",
        agent_config: Optional[dict[str, Any]] = None,
        # Injected subsystems (all optional)
        signal_bus: Any = None,
        stigmergy_env: Any = None,
        gnn_communicator: Any = None,
        knowledge_base: Any = None,
        transfer_engine: Any = None,
        maml_learner: Any = None,
        episodic_memory: Any = None,
        memory_consolidator: Any = None,
        semantic_retriever: Any = None,
        generative_memory: Any = None,
        quorum_sensor: Any = None,
        world_model: Any = None,
        decision_router: Any = None,
        causal_engine: Any = None,
        morphogenesis_enabled: bool = False,
        # Pattern recognition (per-agent membrane + triadic sharing)
        pattern_sense: Any = None,
        pattern_sharer: Any = None,
        # Holon protocol (fractal self-awareness)
        holon_registry: Any = None,
        somatic_map: Any = None,
    ) -> None:
        config = agent_config or {}

        # Mesa base
        super().__init__(model, agent_type=agent_type, agent_config=config)

        # Initialize all mixins explicitly (avoids MRO __init__ chains)
        self._init_convergence(config)
        self._init_gamification(config)
        self._init_signal_processing(signal_bus=signal_bus)

        # Signal Priority Resolver (thalamus — triages signals before cortex)
        self._signal_resolver: SignalPriorityResolver | None = None
        if signal_bus is not None:
            self._signal_resolver = SignalPriorityResolver(
                agent_id=str(self.unique_id),
                signal_bus=signal_bus,
            )
            self._setup_prioritized_signal_handlers()

        self._init_stigmergy(stigmergy_env=stigmergy_env, agent_config=config)
        self._init_gnn_communication(gnn_communicator=gnn_communicator, agent_config=config)
        self._init_transfer_learning(
            knowledge_base=knowledge_base,
            transfer_engine=transfer_engine,
            maml_learner=maml_learner,
            agent_config=config,
        )
        self._init_episodic_memory(
            episodic_memory=episodic_memory,
            memory_consolidator=memory_consolidator,
            semantic_retriever=semantic_retriever,
            generative_memory=generative_memory,
            agent_config=config,
        )
        self._init_collective_consensus(quorum_sensor=quorum_sensor, agent_config=config)
        self._init_advanced_features(
            world_model=world_model,
            morphogenesis_enabled=morphogenesis_enabled,
            decision_router=decision_router,
            causal_engine=causal_engine,
            agent_config=config,
        )

        # Per-agent pattern membrane + triadic sharing (optional)
        self._pattern_sense = pattern_sense
        self._pattern_sharer = pattern_sharer
        self._last_sense_result = None

        # Holon protocol (LAST - all other mixin state must exist first)
        self._init_holon(
            holon_registry=holon_registry,
            somatic_map=somatic_map,
            holon_id=str(self.unique_id),
            agent_config=config,
        )

        # FEP (Free Energy Principle) prediction state
        # (biological: predictive processing — the brain as a prediction machine)
        self._last_prediction: np.ndarray | None = None
        self._prediction_error: float = 0.0

        # WorldModel training counter (Law 6: Autopoietic Closure tracking)
        self._wm_train_steps: int = 0

        # TaskPool action state (biological: motor cortex + musculoskeletal system)
        # Initialized here so agents work with or without a TaskPool injected.
        self._current_task_id: str | None = None
        self._resting: bool = False

    def step(self) -> None:
        """Execute one agent lifecycle step with FEP predictive processing.

        Extended lifecycle (Free Energy Principle + Cognitive Architecture):
        0. Triage (signal priority resolver)
        1. Predict (generate expectation)
        2. Attend (precision-weighted gating)
        3. Observe (sense environment)
        4. Compare (prediction error)
        5. Inhibit (Go/No-Go gate)
        6. Decide (policy selection)
        7. Act (execute action)
        8. Learn (update models)
        9. Manage goals (track progress, detect impasses)
        10. Communicate (signals, stigmergy, GNN)
        11. Broadcast (GWT competitive ignition, cadenced every 3 steps)
        12. Regulate (arousal homeostasis, cadenced every 21 steps)
        """
        self.step_count += 1

        # Triage queued signals before processing (thalamus)
        resolver = getattr(self, "_signal_resolver", None)
        if resolver is not None:
            resolver.process()

        try:
            self._predict()
            self._attend()
            self._observe()
            self._compare()

            # Go/No-Go gate: inhibition can suppress action
            if self._inhibit():
                # Inhibited — skip decide/act, still learn from non-action
                # Biological: rest has value. Hardcoded 0.0 collapses the world
                # model into predicting zero for all states (death spiral).
                action = "inhibited"
                reward = 0.05
            else:
                action = self._decide()
                reward = self._act(action)

            self._learn(action, reward)
            self._manage_goals(reward)
            self._communicate()

            # Cadenced steps (Fibonacci timing)
            if self.step_count % 3 == 0:
                self._broadcast()
            if self.step_count % 21 == 0:
                self._regulate()

        except Exception:
            logger.exception(
                "%s: Error in step %d", self.unique_id, self.step_count
            )

    def _setup_prioritized_signal_handlers(self) -> None:
        """Register standard signal handlers through the priority resolver."""
        if self._signal_resolver is None:
            return
        handlers = {
            "DANGER": self._handle_danger_signal,
            "OPPORTUNITY": self._handle_opportunity_signal,
            "CONVERGENCE": self._handle_convergence_signal,
            "COLLABORATION_REQUEST": self._handle_collaboration_signal,
            "KNOWLEDGE_SHARE": self._handle_knowledge_share_signal,
        }
        for signal_type, handler in handlers.items():
            self._signal_resolver.register_handler(signal_type, handler)

    def get_all_statistics(self) -> dict[str, Any]:
        """Get statistics from all subsystems."""
        stats: dict[str, Any] = {
            "base": self.get_state(),
            "performance": self.get_performance_summary(),
            "convergence": {
                "has_converged": self.has_reached_convergence,
                "satisfaction": self.satisfaction_score,
                "is_satisfied": self.is_satisfied_state,
            },
            "gamification": self.get_gamification_status(),
        }

        # Optional subsystem stats
        signal_stats = self.get_signal_statistics()
        if signal_stats:
            stats["signals"] = signal_stats

        stigmergy_stats = self.get_stigmergy_statistics()
        if stigmergy_stats:
            stats["stigmergy"] = stigmergy_stats

        gnn_stats = self.get_gnn_communication_stats()
        if gnn_stats:
            stats["gnn_communication"] = gnn_stats

        transfer_stats = self.get_transfer_statistics()
        if transfer_stats:
            stats["transfer"] = transfer_stats

        maml_stats = self.get_maml_statistics()
        if maml_stats:
            stats["maml"] = maml_stats

        memory_stats = self.get_episodic_memory_statistics()
        if memory_stats:
            stats["episodic_memory"] = memory_stats

        collective_stats = self.get_collective_statistics()
        if collective_stats:
            stats["collective"] = collective_stats

        stats["advanced"] = self.get_advanced_statistics()
        stats["holon"] = self.get_holon_statistics()

        # Prediction-training loop metrics (Law 6: Autopoietic Closure)
        stats["prediction_training_loop"] = {
            "wm_train_steps": self._wm_train_steps,
            "prediction_error": self._prediction_error,
            "loop_closed": self._wm_train_steps > 0,
        }

        if self._signal_resolver is not None:
            stats["signal_priority"] = self._signal_resolver.get_statistics()

        return stats

    def serialize_state(self) -> dict[str, Any]:
        """Serialize full agent state including all mixin states."""
        state = super().serialize_state()
        state["convergence"] = self._serialize_convergence()
        state["gamification"] = self._serialize_gamification()
        state["signal_processing"] = self._serialize_signal_processing()
        state["stigmergy"] = self._serialize_stigmergy()
        state["gnn_communication"] = self._serialize_gnn_communication()
        state["transfer_learning"] = self._serialize_transfer_learning()
        state["episodic_memory"] = self._serialize_episodic_memory()
        state["collective_consensus"] = self._serialize_collective_consensus()
        state["advanced_features"] = self._serialize_advanced_features()
        state["holon"] = self._serialize_holon()
        if self._signal_resolver is not None:
            state["signal_priority"] = self._signal_resolver.serialize(Path(""))
        return state

    def restore_state(self, data: dict[str, Any]) -> None:
        """Restore full agent state including all mixin states."""
        super().restore_state(data)
        if "convergence" in data:
            self._restore_convergence(data["convergence"])
        if "gamification" in data:
            self._restore_gamification(data["gamification"])
        if "signal_processing" in data:
            self._restore_signal_processing(data["signal_processing"])
        if "stigmergy" in data:
            self._restore_stigmergy(data["stigmergy"])
        if "gnn_communication" in data:
            self._restore_gnn_communication(data["gnn_communication"])
        if "transfer_learning" in data:
            self._restore_transfer_learning(data["transfer_learning"])
        if "episodic_memory" in data:
            self._restore_episodic_memory(data["episodic_memory"])
        if "collective_consensus" in data:
            self._restore_collective_consensus(data["collective_consensus"])
        if "advanced_features" in data:
            self._restore_advanced_features(data["advanced_features"])
        if "holon" in data:
            self._restore_holon(data["holon"])
        if "signal_priority" in data and self._signal_resolver is not None:
            self._signal_resolver.restore(Path(""), data["signal_priority"])
        logger.info(
            "Agent %s: restored state (step %d, reward %.2f)",
            self.unique_id, self.step_count, self.cumulative_reward,
        )

