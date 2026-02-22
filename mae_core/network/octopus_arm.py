"""Octopus Arm - Semi-autonomous processing unit.

Each arm has local capabilities, processes tasks independently,
and communicates with peers via coordination signals. Arms can
operate without central brain coordination in emergency mode.

Biological analogy: Octopus arm with local neural ganglia.
2/3 of octopus neurons reside in the arms, enabling independent action.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from .octopus_signals import (
    ArmCapability,
    ArmState,
    CoordinationSignal,
    Task,
)

logger = logging.getLogger(__name__)


class OctopusArm:
    """An autonomous processing arm with local intelligence.

    Each arm:
    - Has specialized capabilities (sensory, memory, decision, etc.)
    - Processes tasks matching its capabilities
    - Communicates with peers via coordination signals
    - Maintains health and workload metrics
    - Can operate autonomously or under central coordination

    Signal types handled:
    - task_completion: Peer finished a task
    - resource_request: Peer needs help
    - emergency: System crisis (drops low-priority tasks, becomes autonomous)
    - learning_update: Shared learning from peers (EMA integration)
    """

    def __init__(
        self,
        arm_id: str,
        capabilities: set[ArmCapability],
    ) -> None:
        self.arm_id = arm_id
        self.state = ArmState(arm_id=arm_id, capabilities=capabilities)

        # Task management
        self.task_queue: list[Task] = []
        self.current_task: Task | None = None
        self.task_history: deque[Task] = deque(maxlen=1000)

        # Coordination
        self.coordination_signals: list[CoordinationSignal] = []
        self.connected_arms: set[str] = set()
        self.performance_metrics: dict[str, float] = {}

        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None

        # Arm-local learning (EMA from peer signals)
        self._adaptation_level = 0.5

    def start_processing(self) -> None:
        """Start background processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop, daemon=True,
            name=f"arm-{self.arm_id}",
        )
        self._thread.start()

    def stop_processing(self) -> None:
        """Stop background processing gracefully."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit_task(self, task: Task) -> bool:
        """Accept a task if capabilities match."""
        if not task.required_capabilities.issubset(self.state.capabilities):
            return False
        with self._lock:
            self.task_queue.append(task)
            self.state.workload = min(1.0, self.state.workload + 0.1)
        return True

    def receive_coordination_signal(self, signal: CoordinationSignal) -> None:
        """Queue an incoming signal from a peer or central brain."""
        with self._lock:
            self.coordination_signals.append(signal)

    def get_arm_status(self) -> dict[str, Any]:
        """Comprehensive arm status report."""
        with self._lock:
            return {
                "arm_id": self.arm_id,
                "capabilities": [c.value for c in self.state.capabilities],
                "current_task": self.current_task.task_id if self.current_task else None,
                "queue_size": len(self.task_queue),
                "workload": self.state.workload,
                "health": self.state.health,
                "coordination_level": self.state.coordination_level,
                "connected_arms": list(self.connected_arms),
                "performance": dict(self.performance_metrics),
                "tasks_completed": sum(
                    1 for t in self.task_history if t.status == "completed"
                ),
                "pending_signals": len(self.coordination_signals),
            }

    # --- Processing Loop ---

    def _processing_loop(self) -> None:
        """Main arm processing cycle (runs in background thread)."""
        while self._running:
            try:
                self._process_coordination_signals()
                self._assign_next_task()
                self._execute_current_task()
                self._update_arm_state()
                time.sleep(0.1)
            except Exception:
                logger.exception("Arm %s processing error", self.arm_id)

    def _process_coordination_signals(self) -> None:
        """Handle incoming signals, priority-ordered, max 5 per cycle."""
        with self._lock:
            if not self.coordination_signals:
                return
            signals = sorted(
                self.coordination_signals, key=lambda s: s.priority, reverse=True
            )
            to_process = signals[:5]
            self.coordination_signals = signals[5:]

        for signal in to_process:
            self._handle_signal(signal)

    def _handle_signal(self, signal: CoordinationSignal) -> None:
        """Route signal to appropriate handler."""
        handlers = {
            "task_completion": self._on_task_completion,
            "resource_request": self._on_resource_request,
            "emergency": self._on_emergency,
            "learning_update": self._on_learning_update,
        }
        handler = handlers.get(signal.signal_type)
        if handler:
            handler(signal)

    def _on_task_completion(self, signal: CoordinationSignal) -> None:
        """Peer completed a task."""
        logger.debug("Arm %s: peer %s completed task", self.arm_id, signal.source_arm)

    def _on_resource_request(self, signal: CoordinationSignal) -> None:
        """Peer requesting help - offer if we have capacity."""
        requested = signal.data.get("capabilities", set())
        if isinstance(requested, list):
            requested = {ArmCapability(c) for c in requested if c in [e.value for e in ArmCapability]}

        if requested.issubset(self.state.capabilities) and self.state.workload < 0.8:
            logger.debug(
                "Arm %s: can help peer %s (workload=%.2f)",
                self.arm_id, signal.source_arm, self.state.workload,
            )

    def _on_emergency(self, signal: CoordinationSignal) -> None:
        """System emergency - shed load, become autonomous."""
        emergency_type = signal.data.get("emergency_type", "unknown")

        if emergency_type == "system_overload":
            self._drop_low_priority_tasks()
        elif emergency_type == "coordination_failure":
            with self._lock:
                self.state.coordination_level = 0.1  # Maximum autonomy

    def _on_learning_update(self, signal: CoordinationSignal) -> None:
        """Integrate learning from peers via exponential moving average."""
        learning_data = signal.data.get("metrics", {})
        with self._lock:
            for key, value in learning_data.items():
                if key in self.performance_metrics:
                    self.performance_metrics[key] = (
                        0.9 * self.performance_metrics[key] + 0.1 * value
                    )
                else:
                    self.performance_metrics[key] = value

    def _assign_next_task(self) -> None:
        """Pick a task from queue that matches capabilities."""
        if self.current_task is not None:
            return
        with self._lock:
            for i, task in enumerate(self.task_queue):
                if task.required_capabilities.issubset(self.state.capabilities):
                    self.current_task = task
                    task.status = "assigned"
                    task.assigned_arm = self.arm_id
                    self.task_queue.pop(i)
                    break

    def _execute_current_task(self) -> None:
        """Execute the currently assigned task."""
        if self.current_task is None:
            return
        with self._lock:
            self.current_task.status = "completed"
            self.task_history.append(self.current_task)
            self.state.workload = max(0.0, self.state.workload - 0.1)
            self.current_task = None

    def _drop_low_priority_tasks(self) -> None:
        """Emergency: shed low-priority tasks to free capacity."""
        with self._lock:
            self.task_queue = [t for t in self.task_queue if t.priority >= 5]

    def _update_arm_state(self) -> None:
        """Update health and workload metrics."""
        with self._lock:
            self.state.workload = min(
                1.0,
                len(self.task_queue) * 0.1 + (0.3 if self.current_task else 0.0),
            )
            # Health degrades with recent failures
            recent = list(self.task_history)[-10:]
            failures = sum(1 for t in recent if t.status == "failed")
            self.state.health = max(0.0, 1.0 - failures * 0.1)
            self.state.last_activity = time.time()

    def __repr__(self) -> str:
        caps = ",".join(c.value for c in self.state.capabilities)
        return f"OctopusArm({self.arm_id}, caps=[{caps}], wl={self.state.workload:.2f})"
