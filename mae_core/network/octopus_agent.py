"""Octopus Agent - Individual octopus with specialization and health.

Wraps OctopusDistributedCognition (8 arms) with:
- Specialization (sensory, memory, decision, etc.)
- Health tracking (arm health * 0.7 + success rate * 0.3)
- Substrate registration
- DecisionRouter integration (three-tier arm cognition)
- WorldModel integration (mini-world model per arm concept)
- SignalBus integration (electrical signal participation)

Biological analogy: A single octopus organism with 8 thinking arms.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus
from .octopus_cognition import OctopusDistributedCognition
from .octopus_signals import (
    ArmCapability,
    CognitionMode,
    OctopusSpecialization,
    SPECIALIZATION_CAPABILITIES,
    CH_OCTOPUS_HEALTH,
)

logger = logging.getLogger(__name__)


class OctopusAgent:
    """A single octopus in the colony - 8 arms + specialization.

    Cross-system connections:
    - DecisionRouter: Arms use three-tier cognition (reflex/habit/prefrontal)
    - WorldModel: Each arm has mini-world model for domain prediction
    - SignalBus: Arms participate in electrical signaling
    - Morphogenesis: Colony spawns new octopuses for novel problems
    - Memory: Task outcomes stored in episodic memory
    - Learning: Arms share learning updates via coordination signals
    """

    def __init__(
        self,
        octopus_id: str,
        event_bus: EventBus | None = None,
        specialization: OctopusSpecialization = OctopusSpecialization.GENERAL,
        num_arms: int = 8,
        decision_router: Any | None = None,
        world_model: Any | None = None,
        signal_bus: Any | None = None,
        memory_coordinator: Any | None = None,
    ) -> None:
        self.octopus_id = octopus_id
        self._bus = event_bus
        self.specialization = specialization
        self.num_arms = num_arms

        # Core cognition system (8 arms)
        self.cognition = OctopusDistributedCognition(
            event_bus=event_bus,
            num_arms=num_arms,
        )

        # Cross-system integrations
        self._decision_router = decision_router
        self._world_model = world_model
        self._signal_bus = signal_bus
        self._memory = memory_coordinator

        # Health and performance
        self.health: float = 1.0
        self.workload: float = 0.0
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self.success_rate: float = 1.0
        self.avg_task_time: float = 0.0
        self.last_task_time: float = 0.0
        self.spawn_time: float = time.time()

        # Arm-level mini-world models (confidence-based escalation)
        self._arm_confidence_threshold = 0.6
        self._arm_prediction_counts: dict[str, int] = {}
        self._arm_prediction_accuracy: dict[str, float] = {}

        # Configure specialization capabilities
        self._configure_specialization()

    def start(self) -> None:
        """Start the distributed cognition system."""
        self.cognition.start_system()

        # Subscribe to signal bus if available
        if self._signal_bus is not None and hasattr(self._signal_bus, "subscribe"):
            self._signal_bus.subscribe([f"octopus.{self.octopus_id}"])

    def stop(self) -> None:
        """Stop the distributed cognition system."""
        self.cognition.stop_system()

    def submit_task(
        self,
        task_data: dict[str, Any],
        task_type: str,
        priority: int = 5,
    ) -> str:
        """Submit a task to this octopus's arms.

        If memory is available, retrieves similar past experiences
        and enriches task_data with relevant context before submission.
        """
        # Memory retrieval: consult past experiences before acting
        if self._memory is not None:
            similar = self._recall_similar(task_type)
            if similar:
                task_data["memory_context"] = {
                    "similar_count": len(similar),
                    "avg_reward": sum(
                        getattr(e, "reward", 0.0) for e in similar
                    ) / max(len(similar), 1),
                }

        task_id = self.cognition.submit_task(task_data, task_type, priority)
        self._update_workload()
        return task_id

    def route_decision(
        self,
        stimulus: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Route a decision through three-tier arm cognition.

        Three-tier cascade:
        1. Reflex: Arm-local pattern matching (<1ms)
        2. Habit: Learned automatic routines (10-100ms)
        3. Prefrontal: Escalate to DecisionRouter/WorldModel (1s+)

        If arm confidence < threshold, escalates to central WorldModel.
        """
        if self._decision_router is not None:
            decision = self._decision_router.route_decision(stimulus, context)
            return {
                "tier": decision.tier_used.value,
                "action": decision.action_taken,
                "confidence": decision.confidence,
                "response_time_ms": decision.response_time,
                "octopus_id": self.octopus_id,
            }

        # Fallback: simple local decision
        return {
            "tier": "local",
            "action": {"type": "default_response", "stimulus": stimulus},
            "confidence": 0.5,
            "octopus_id": self.octopus_id,
        }

    def predict_with_confidence(
        self,
        arm_id: str,
        state: Any,
        action: Any,
    ) -> dict[str, Any]:
        """Arm-level prediction with confidence-based escalation.

        If arm's prediction confidence < threshold:
          → Escalate to central WorldModel
        Else:
          → Use arm-local prediction

        This prevents the central brain from becoming a bottleneck.
        """
        arm_accuracy = self._arm_prediction_accuracy.get(arm_id, 0.5)

        if arm_accuracy >= self._arm_confidence_threshold and self._world_model is not None:
            # Arm is confident - use local prediction
            prediction = self._local_predict(state, action)
            return {
                "source": "arm_local",
                "arm_id": arm_id,
                "prediction": prediction,
                "confidence": arm_accuracy,
                "escalated": False,
            }

        # Low confidence - escalate to central WorldModel
        if self._world_model is not None and hasattr(self._world_model, "predict"):
            prediction = self._world_model.predict(state, action)
            return {
                "source": "central_world_model",
                "arm_id": arm_id,
                "prediction": prediction,
                "confidence": 0.7,  # Central model is more reliable
                "escalated": True,
            }

        return {
            "source": "fallback",
            "arm_id": arm_id,
            "prediction": state,  # No change predicted
            "confidence": 0.3,
            "escalated": False,
        }

    def update_arm_prediction_accuracy(
        self, arm_id: str, was_accurate: bool
    ) -> None:
        """Update an arm's prediction accuracy (EMA)."""
        current = self._arm_prediction_accuracy.get(arm_id, 0.5)
        self._arm_prediction_accuracy[arm_id] = (
            0.9 * current + 0.1 * (1.0 if was_accurate else 0.0)
        )
        self._arm_prediction_counts[arm_id] = (
            self._arm_prediction_counts.get(arm_id, 0) + 1
        )

    def emit_signal(self, signal_type: str, data: dict[str, Any]) -> None:
        """Emit a signal on the electrical signal bus."""
        if self._signal_bus is not None and hasattr(self._signal_bus, "emit_signal"):
            self._signal_bus.emit_signal(
                signal_type=signal_type,
                source=self.octopus_id,
                data=data,
                priority=5,
            )

    def record_task_outcome(
        self,
        task_type: str,
        success: bool,
        reward: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Store task outcome in episodic memory.

        Called after a task completes. The outcome becomes a retrievable
        experience for future similar tasks -- like an octopus learning
        which hunting strategies work for which prey.
        """
        if self._memory is None:
            return

        state = np.zeros(74)  # Default state vector
        # Action encoded as numeric (task_type hash + success flag)
        action = hash(task_type) % 100 + (1.0 if success else 0.0)
        info = {
            "octopus_id": self.octopus_id,
            "task_type": task_type,
            "success": success,
            "specialization": self.specialization.value,
            **(context or {}),
        }

        self._memory.store(
            state=state,
            action=action,
            reward=reward if success else -abs(reward),
            next_state=state,
            done=True,
            info=info,
            priority=0.9 if not success else 0.5,  # Failures are more memorable
        )

    def _recall_similar(self, task_type: str, k: int = 3) -> list:
        """Retrieve similar past experiences from memory.

        Searches episodic memory for experiences matching this task type.
        Returns up to k similar experiences.
        """
        if self._memory is None:
            return []

        state = np.zeros(74)
        return self._memory.search(state, k=k)

    def update_metrics(self) -> None:
        """Refresh health, workload, and task counters."""
        self._update_workload()
        self._update_health()

        status = self.cognition.get_system_status()
        self.tasks_completed = status.get("tasks_completed", 0)
        self.tasks_failed = status.get("tasks_failed", 0)

        total = self.tasks_completed + self.tasks_failed
        self.success_rate = self.tasks_completed / max(total, 1)

    def get_status(self) -> dict[str, Any]:
        """Full octopus status report."""
        self.update_metrics()
        return {
            "octopus_id": self.octopus_id,
            "specialization": self.specialization.value,
            "health": self.health,
            "workload": self.workload,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "success_rate": self.success_rate,
            "uptime": time.time() - self.spawn_time,
            "num_arms": self.num_arms,
            "cognition_mode": self.cognition.coordination_mode.value,
            "has_decision_router": self._decision_router is not None,
            "has_world_model": self._world_model is not None,
            "has_signal_bus": self._signal_bus is not None,
            "has_memory": self._memory is not None,
            "arm_prediction_accuracy": dict(self._arm_prediction_accuracy),
            "capabilities": SPECIALIZATION_CAPABILITIES.get(
                self.specialization, []
            ),
        }

    def get_capabilities(self) -> dict[str, Any]:
        """Return base + specialization capabilities."""
        base = ["parallel_processing", "distributed_cognition", "fault_tolerant", "adaptive_learning"]
        spec = SPECIALIZATION_CAPABILITIES.get(self.specialization, [])
        return {
            "base": base,
            "specialization": spec,
            "specialization_type": self.specialization.value,
        }

    # --- Internal ---

    def _configure_specialization(self) -> None:
        """Configure arm capabilities based on specialization.

        Specialized octopuses get extra capabilities in their domain.
        E.g., SENSORY octopus gets more arms with SENSORY_PROCESSING.
        """
        if self.specialization == OctopusSpecialization.GENERAL:
            return  # Default diverse configuration

        # Map specialization to arm capability to boost
        spec_to_cap: dict[OctopusSpecialization, ArmCapability] = {
            OctopusSpecialization.SENSORY: ArmCapability.SENSORY_PROCESSING,
            OctopusSpecialization.MEMORY: ArmCapability.MEMORY_ACCESS,
            OctopusSpecialization.DECISION: ArmCapability.DECISION_MAKING,
            OctopusSpecialization.COMMUNICATION: ArmCapability.COMMUNICATION,
            OctopusSpecialization.LEARNING: ArmCapability.LEARNING,
            OctopusSpecialization.ADAPTATION: ArmCapability.ADAPTATION,
            OctopusSpecialization.EMERGENCY: ArmCapability.ADAPTATION,
        }

        boost_cap = spec_to_cap.get(self.specialization)
        if boost_cap is None:
            return

        # Add the specialized capability to half the arms
        arms = list(self.cognition.arms.values())
        for arm in arms[: len(arms) // 2]:
            arm.state.capabilities.add(boost_cap)

    def _update_workload(self) -> None:
        """Calculate average workload across arms."""
        arms = list(self.cognition.arms.values())
        if arms:
            self.workload = float(np.mean([a.state.workload for a in arms]))

    def _update_health(self) -> None:
        """Calculate health: arm health * 0.7 + success rate * 0.3."""
        arms = list(self.cognition.arms.values())
        arm_health = float(np.mean([a.state.health for a in arms])) if arms else 1.0
        self.health = arm_health * 0.7 + self.success_rate * 0.3

    def _local_predict(self, state: Any, action: Any) -> Any:
        """Simple arm-local prediction (fallback)."""
        if isinstance(state, np.ndarray):
            return state + np.random.randn(*state.shape) * 0.01
        return state

    def __repr__(self) -> str:
        return (
            f"OctopusAgent({self.octopus_id}, "
            f"spec={self.specialization.value}, "
            f"health={self.health:.2f})"
        )
