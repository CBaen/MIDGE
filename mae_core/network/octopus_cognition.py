"""Octopus Distributed Cognition - Central brain coordinating 8 arms.

Orchestrates arms with adaptive coordination modes. NOT hierarchical
control - arms retain autonomy. Central brain adjusts coordination
level based on system health and workload. Supports emergency mode
(maximum decentralization) and learning propagation.

Biological analogy: Octopus central brain + interbrachial commissures.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus
from .octopus_arm import OctopusArm
from .octopus_signals import (
    ArmCapability,
    CognitionMode,
    CoordinationSignal,
    DEFAULT_ARM_CAPABILITIES,
    TASK_CAPABILITY_MAP,
    Task,
    CH_OCTOPUS_TASK,
    CH_OCTOPUS_COMPLETED,
    CH_OCTOPUS_EMERGENCY,
    CH_OCTOPUS_LEARNING,
    CH_OCTOPUS_HEALTH,
)

logger = logging.getLogger(__name__)


class OctopusDistributedCognition:
    """Coordinator for 8 semi-autonomous arms.

    Two-level coordination:
    - Inner: Arms process tasks independently using local intelligence
    - Outer: Central brain adjusts coordination mode, balances workload,
             shares learning, and responds to emergencies

    Coordination modes:
    - CENTRALIZED: Central brain controls tightly (low workload, high health)
    - HYBRID: Balanced autonomy (default)
    - DISTRIBUTED: Arms mostly independent (high workload or low health)
    - EMERGENCY: Maximum decentralization (crisis response)
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        num_arms: int = 8,
    ) -> None:
        self._bus = event_bus
        self.num_arms = num_arms

        # Arms
        self.arms: dict[str, OctopusArm] = {}
        self._initialize_arms()

        # Coordination state
        self.coordination_mode = CognitionMode.HYBRID
        self.global_coordination_level: float = 0.5
        self.emergency_mode: bool = False
        self.last_coordination_update: float = 0.0

        # Task registry
        self._task_registry: dict[str, Task] = {}

        # Performance tracking
        self._performance_history: deque[dict[str, Any]] = deque(maxlen=1000)
        self._learning_updates: deque[dict[str, Any]] = deque(maxlen=50)

        self._lock = threading.RLock()

    def start_system(self) -> None:
        """Start all arm processing threads."""
        for arm in self.arms.values():
            arm.start_processing()
        logger.info("Octopus cognition started: %d arms", len(self.arms))

    def stop_system(self) -> None:
        """Stop all arm processing gracefully."""
        for arm in self.arms.values():
            arm.stop_processing()
        logger.info("Octopus cognition stopped")

    def submit_task(
        self,
        task_data: dict[str, Any],
        task_type: str,
        priority: int = 5,
    ) -> str:
        """Distribute a task to the best available arm."""
        required = TASK_CAPABILITY_MAP.get(task_type, set())

        task = Task(
            task_type=task_type,
            priority=priority,
            data=task_data,
            required_capabilities=required,
        )

        with self._lock:
            self._task_registry[task.task_id] = task

        success = self._distribute_task(task)

        if self._bus:
            self._bus.publish(CH_OCTOPUS_TASK, {
                "task_id": task.task_id,
                "task_type": task_type,
                "priority": priority,
                "distributed": success,
            })

        return task.task_id

    def run_coordination_cycle(self) -> None:
        """Periodic coordination: mode update, balance, learning, reporting."""
        with self._lock:
            self._update_coordination_mode()
            self._balance_workload()
            self._share_learning_updates()
            self._update_performance_history()
            self.last_coordination_update = time.time()

    def trigger_emergency_mode(
        self, emergency_type: str = "system_overload"
    ) -> None:
        """Activate emergency protocol - maximum arm autonomy."""
        with self._lock:
            self.emergency_mode = True
            self.coordination_mode = CognitionMode.EMERGENCY
            self.global_coordination_level = 0.1

        # Broadcast to all arms
        signal = CoordinationSignal(
            source_arm="central",
            signal_type="emergency",
            data={"emergency_type": emergency_type},
            priority=10,
        )
        for arm in self.arms.values():
            arm.receive_coordination_signal(signal)

        if self._bus:
            self._bus.publish(CH_OCTOPUS_EMERGENCY, {
                "emergency_type": emergency_type,
                "arms_notified": len(self.arms),
            })

    def exit_emergency_mode(self) -> None:
        """Return to normal coordination."""
        with self._lock:
            self.emergency_mode = False
            self._update_coordination_mode()

    def get_system_status(self) -> dict[str, Any]:
        """Comprehensive system health report."""
        with self._lock:
            arm_statuses = {
                arm_id: arm.get_arm_status()
                for arm_id, arm in self.arms.items()
            }
            avg_workload = (
                float(np.mean([a.state.workload for a in self.arms.values()]))
                if self.arms else 0.0
            )
            avg_health = (
                float(np.mean([a.state.health for a in self.arms.values()]))
                if self.arms else 0.0
            )

            return {
                "num_arms": len(self.arms),
                "coordination_mode": self.coordination_mode.value,
                "global_coordination_level": self.global_coordination_level,
                "emergency_mode": self.emergency_mode,
                "average_workload": avg_workload,
                "average_health": avg_health,
                "tasks_registered": len(self._task_registry),
                "tasks_completed": sum(
                    1 for t in self._task_registry.values()
                    if t.status == "completed"
                ),
                "tasks_failed": sum(
                    1 for t in self._task_registry.values()
                    if t.status == "failed"
                ),
                "arm_statuses": arm_statuses,
            }

    # --- Internal ---

    def _initialize_arms(self) -> None:
        """Create arms with diverse capabilities and ring topology."""
        caps = DEFAULT_ARM_CAPABILITIES
        for i in range(self.num_arms):
            arm_id = f"arm-{i}"
            arm_caps = caps[i % len(caps)]
            arm = OctopusArm(arm_id=arm_id, capabilities=arm_caps)
            self.arms[arm_id] = arm

        # Establish ring topology
        arm_ids = list(self.arms.keys())
        n = len(arm_ids)
        for i in range(n):
            arm = self.arms[arm_ids[i]]
            arm.connected_arms.add(arm_ids[(i + 1) % n])
            arm.connected_arms.add(arm_ids[(i - 1) % n])

    def _distribute_task(self, task: Task) -> bool:
        """Find best arm for task: capable + least loaded."""
        suitable = []
        for arm in self.arms.values():
            if task.required_capabilities.issubset(arm.state.capabilities):
                suitable.append(arm)

        if not suitable:
            # Fallback: any arm with lowest workload
            suitable = list(self.arms.values())

        best = min(suitable, key=lambda a: a.state.workload)
        return best.submit_task(task)

    def _update_coordination_mode(self) -> None:
        """Adapt coordination based on system state."""
        if self.emergency_mode:
            self.coordination_mode = CognitionMode.EMERGENCY
            self.global_coordination_level = 0.1
            return

        avg_health = float(
            np.mean([a.state.health for a in self.arms.values()])
        ) if self.arms else 1.0
        avg_workload = float(
            np.mean([a.state.workload for a in self.arms.values()])
        ) if self.arms else 0.0

        if avg_health < 0.5 or avg_workload > 0.8:
            self.coordination_mode = CognitionMode.DISTRIBUTED
            self.global_coordination_level = 0.3
        elif avg_workload < 0.3:
            self.coordination_mode = CognitionMode.CENTRALIZED
            self.global_coordination_level = 0.8
        else:
            self.coordination_mode = CognitionMode.HYBRID
            self.global_coordination_level = 0.5

        # Propagate to arms
        for arm in self.arms.values():
            arm.state.coordination_level = self.global_coordination_level

    def _balance_workload(self) -> None:
        """Transfer tasks from overloaded to underloaded arms."""
        if len(self.arms) < 2:
            return

        sorted_arms = sorted(self.arms.values(), key=lambda a: a.state.workload)
        min_arm = sorted_arms[0]
        max_arm = sorted_arms[-1]

        if max_arm.state.workload - min_arm.state.workload <= 0.3:
            return

        # Find a transferable task
        with max_arm._lock:
            for i, task in enumerate(max_arm.task_queue):
                if task.required_capabilities.issubset(min_arm.state.capabilities):
                    transferred = max_arm.task_queue.pop(i)
                    min_arm.submit_task(transferred)
                    break

    def _share_learning_updates(self) -> None:
        """Broadcast recent learning to all arms."""
        updates = list(self._learning_updates)[-5:]
        if not updates:
            return

        for update in updates:
            signal = CoordinationSignal(
                source_arm="central",
                signal_type="learning_update",
                data={"metrics": update},
            )
            for arm in self.arms.values():
                arm.receive_coordination_signal(signal)

        if self._bus:
            self._bus.publish(CH_OCTOPUS_LEARNING, {
                "updates_shared": len(updates),
                "arms_notified": len(self.arms),
            })

    def _update_performance_history(self) -> None:
        """Snapshot current system state for history."""
        snapshot = {
            "timestamp": time.time(),
            "coordination_mode": self.coordination_mode.value,
            "coordination_level": self.global_coordination_level,
            "emergency": self.emergency_mode,
            "arm_count": len(self.arms),
            "avg_workload": float(
                np.mean([a.state.workload for a in self.arms.values()])
            ) if self.arms else 0.0,
            "avg_health": float(
                np.mean([a.state.health for a in self.arms.values()])
            ) if self.arms else 0.0,
        }
        self._performance_history.append(snapshot)

    def add_learning_update(self, metrics: dict[str, float]) -> None:
        """Add a learning update to be shared with arms."""
        self._learning_updates.append(metrics)

    def __repr__(self) -> str:
        return (
            f"OctopusCognition(arms={len(self.arms)}, "
            f"mode={self.coordination_mode.value}, "
            f"emergency={self.emergency_mode})"
        )
