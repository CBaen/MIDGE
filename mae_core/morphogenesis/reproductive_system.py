"""ReproductiveSystem - population management via mitosis and apoptosis.

Biological basis: Hayflick limit / telomere biology (1961). Cells divide
(mitosis) to grow the population and undergo programmed cell death
(apoptosis) to shrink it. Population is maintained within bounds by
negative feedback: high load triggers division, low load triggers
retirement. The Hayflick limit prevents runaway growth.

Mae's reproductive system manages agent population:
- Monitors per-agent load to compute population health metrics
- Triggers spawn requests when average load exceeds threshold
- Triggers retire requests when idle agents accumulate
- Enforces Rule of 3 minimum (3 agents always alive)
- Enforces Fibonacci cap maximum (21 agents)
- Uses Fibonacci cooldown (13 steps) between spawns to prevent oscillation

Law compliance:
- Law 6: autopoietic closure — the system produces the agents that
  produce the system. Agents do work, work generates load signals,
  load signals trigger spawning of more agents.
- Law 7: Rule of 3/5 — minimum 3 agents always, 5 for critical processes.
  Population never drops below the triadic minimum.

EventBus channels:
- morphogenesis.spawn_request: request to create a new agent
- morphogenesis.retire_request: request to retire an idle agent
- morphogenesis.population_status: per-step population health report
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_SPAWN = "morphogenesis.spawn_request"
CH_RETIRE = "morphogenesis.retire_request"
CH_POPULATION = "morphogenesis.population_status"

# Population bounds
_MIN_POPULATION = 3  # Rule of 3 — triadic minimum
_MAX_POPULATION = 21  # Fibonacci cap (F(8) = 21)
_FIBONACCI_COOLDOWN = 13  # Fibonacci cooldown between spawns (F(7) = 13)

# Load thresholds
_SPAWN_THRESHOLD = 0.8  # Spawn when avg load > this
_RETIRE_THRESHOLD = 0.1  # Retire when avg load < this AND idle > 1
_TARGET_LOAD_PER_AGENT = 0.5  # For optimal population calculation
_OVERLOAD_THRESHOLD = 0.9  # Agent is "overloaded" above this


@dataclass
class PopulationMetrics:
    """Snapshot of the current agent population's health.

    Biological analogy: a complete blood count (CBC) — measuring the
    number and health of cells to assess overall system vitality.
    """

    current_agents: int = 0
    optimal_agents: int = 3
    load_per_agent: float = 0.0  # Average load across agents
    idle_agents: int = 0  # Agents with load < retire_threshold
    overloaded_agents: int = 0  # Agents with load > overload_threshold


class ReproductiveSystem:
    """Mae's reproductive system — manages agent population through
    mitosis (spawning) and apoptosis (retirement).

    Monitors per-agent load, computes population health, and triggers
    spawn/retire requests through the EventBus. The MorphogenesisCoordinator
    handles the actual agent creation/removal; this system only decides
    WHEN to grow or shrink.
    """

    CH_SPAWN = CH_SPAWN
    CH_RETIRE = CH_RETIRE
    CH_POPULATION = CH_POPULATION

    def __init__(
        self,
        event_bus: Any = None,
        morph_coordinator: Any = None,
        spawn_threshold: float = _SPAWN_THRESHOLD,
        retire_threshold: float = _RETIRE_THRESHOLD,
        min_population: int = _MIN_POPULATION,
        max_population: int = _MAX_POPULATION,
        spawn_cooldown: int = _FIBONACCI_COOLDOWN,
    ) -> None:
        self.event_bus = event_bus
        self._morph_coordinator = morph_coordinator

        # Thresholds
        self._spawn_threshold = spawn_threshold
        self._retire_threshold = retire_threshold
        self._min_population = max(3, min_population)  # Enforce Rule of 3 floor
        self._max_population = max_population

        # Cooldown
        self._spawn_cooldown = spawn_cooldown
        self._last_spawn_step: int = -spawn_cooldown  # Allow immediate first spawn
        self._last_retire_step: int = 0

        # Population state
        self._population_metrics = PopulationMetrics()
        self._agent_loads: dict[int, float] = {}

        # Counters
        self._division_count: int = 0
        self._retirement_count: int = 0
        self._step_count: int = 0

        # History for trend analysis
        self._metrics_history: list[dict[str, Any]] = []
        self._max_history = 100

        # Subscribe to relevant channels
        if self.event_bus is not None:
            self.event_bus.register_callback(
                "substrate.circulation_update", self._on_circulation_update
            )

        logger.info(
            "ReproductiveSystem initialized: min=%d, max=%d, "
            "spawn_threshold=%.2f, retire_threshold=%.2f, cooldown=%d",
            self._min_population,
            self._max_population,
            self._spawn_threshold,
            self._retire_threshold,
            self._spawn_cooldown,
        )

    # =========================================================================
    # Metrics Update
    # =========================================================================

    def update_metrics(self, agent_loads: dict[int, float]) -> None:
        """Update population metrics from per-agent load data.

        Args:
            agent_loads: Dict of agent_id -> load (0.0 to 1.0).
                         Load represents how busy each agent is.
        """
        self._agent_loads = dict(agent_loads)

        if not agent_loads:
            self._population_metrics = PopulationMetrics()
            return

        loads = list(agent_loads.values())
        current = len(loads)
        avg_load = sum(loads) / current if current > 0 else 0.0
        idle = sum(1 for l in loads if l < self._retire_threshold)
        overloaded = sum(1 for l in loads if l > _OVERLOAD_THRESHOLD)
        optimal = self.get_optimal_population()

        self._population_metrics = PopulationMetrics(
            current_agents=current,
            optimal_agents=optimal,
            load_per_agent=avg_load,
            idle_agents=idle,
            overloaded_agents=overloaded,
        )

    # =========================================================================
    # Decision Logic
    # =========================================================================

    def _should_spawn(self) -> bool:
        """Decide whether to trigger mitosis (agent creation).

        Conditions (ALL must be true):
        1. Average load exceeds spawn threshold
        2. Population is below maximum
        3. Cooldown period has elapsed since last spawn
        """
        m = self._population_metrics
        if m.current_agents == 0:
            return True  # Always spawn if no agents exist

        if m.load_per_agent <= self._spawn_threshold:
            return False

        if m.current_agents >= self._max_population:
            return False

        steps_since_spawn = self._step_count - self._last_spawn_step
        if steps_since_spawn < self._spawn_cooldown:
            return False

        return True

    def _should_retire(self) -> bool:
        """Decide whether to trigger apoptosis (agent retirement).

        Conditions (ALL must be true):
        1. More than 1 idle agent (keep at least 1 reserve)
        2. Population is above minimum (Rule of 3)
        3. Average load is below retire threshold
        """
        m = self._population_metrics
        if m.current_agents <= self._min_population:
            return False

        if m.idle_agents <= 1:
            return False

        if m.load_per_agent >= self._retire_threshold:
            return False

        return True

    def _select_retire_candidate(self) -> Optional[int]:
        """Select the best agent to retire.

        Picks the agent with the lowest load (most idle).
        """
        if not self._agent_loads:
            return None

        # Find agent with lowest load
        candidate_id = min(self._agent_loads, key=self._agent_loads.get)
        return candidate_id

    def _select_spawn_role(self) -> str:
        """Select the role for a newly spawned agent.

        If there are overloaded agents, clone their role. Otherwise
        spawn a general-purpose worker.
        """
        # Default: general worker
        return "worker"

    # =========================================================================
    # Step (called by model each tick)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Evaluate population health and trigger spawn/retire as needed.

        1. Check spawn conditions -> publish spawn_request
        2. Check retire conditions -> publish retire_request
        3. Publish population_status
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        spawn_triggered = False
        retire_triggered = False

        # Check spawn
        if self._should_spawn():
            spawn_triggered = True
            self._division_count += 1
            self._last_spawn_step = self._step_count
            target_role = self._select_spawn_role()

            if self.event_bus is not None:
                self.event_bus.publish(
                    CH_SPAWN,
                    {
                        "reason": "high_load",
                        "target_role": target_role,
                        "avg_load": round(self._population_metrics.load_per_agent, 4),
                        "current_population": self._population_metrics.current_agents,
                        "step": self._step_count,
                    },
                )

            # If morph_coordinator is available, trigger directly
            if self._morph_coordinator is not None:
                if hasattr(self._morph_coordinator, "set_growth_rate"):
                    self._morph_coordinator.set_growth_rate(1.5)

            logger.info(
                "Spawn triggered: avg_load=%.2f, population=%d/%d",
                self._population_metrics.load_per_agent,
                self._population_metrics.current_agents,
                self._max_population,
            )

        # Check retire
        if self._should_retire():
            retire_triggered = True
            self._retirement_count += 1
            candidate = self._select_retire_candidate()

            if self.event_bus is not None:
                self.event_bus.publish(
                    CH_RETIRE,
                    {
                        "reason": "low_load",
                        "candidate_agent": candidate,
                        "avg_load": round(self._population_metrics.load_per_agent, 4),
                        "idle_agents": self._population_metrics.idle_agents,
                        "current_population": self._population_metrics.current_agents,
                        "step": self._step_count,
                    },
                )

            logger.info(
                "Retire triggered: avg_load=%.2f, idle=%d, candidate=%s",
                self._population_metrics.load_per_agent,
                self._population_metrics.idle_agents,
                candidate,
            )

        # Determine recommendation
        m = self._population_metrics
        if m.current_agents < m.optimal_agents:
            recommendation = "grow"
        elif m.current_agents > m.optimal_agents + 2:
            recommendation = "shrink"
        else:
            recommendation = "stable"

        # Publish population status
        if self.event_bus is not None:
            self.event_bus.publish(
                CH_POPULATION,
                {
                    "current": m.current_agents,
                    "optimal": m.optimal_agents,
                    "recommendation": recommendation,
                    "load_per_agent": round(m.load_per_agent, 4),
                    "idle_agents": m.idle_agents,
                    "overloaded_agents": m.overloaded_agents,
                    "spawn_triggered": spawn_triggered,
                    "retire_triggered": retire_triggered,
                    "step": self._step_count,
                },
            )

        # Record history
        self._metrics_history.append({
            "step": self._step_count,
            "current": m.current_agents,
            "optimal": m.optimal_agents,
            "load": round(m.load_per_agent, 4),
            "idle": m.idle_agents,
            "overloaded": m.overloaded_agents,
            "spawned": spawn_triggered,
            "retired": retire_triggered,
        })
        if len(self._metrics_history) > self._max_history:
            self._metrics_history = self._metrics_history[-self._max_history:]

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_circulation_update(self, channel: str, message: Any) -> None:
        """React to circulation updates — high unfulfilled demand suggests
        more agents are needed (they generate demand that circulation can't meet).
        """
        data = self._parse_message(message)
        if data is None:
            return
        # If circulation is struggling, nudge toward spawning
        unfulfilled = data.get("unfulfilled", 0.0)
        if isinstance(unfulfilled, (int, float)) and unfulfilled > 10.0:
            # Artificially increase perceived load
            if self._population_metrics.current_agents > 0:
                boost = min(0.1, unfulfilled / 1000.0)
                self._population_metrics = PopulationMetrics(
                    current_agents=self._population_metrics.current_agents,
                    optimal_agents=self._population_metrics.optimal_agents,
                    load_per_agent=min(
                        1.0, self._population_metrics.load_per_agent + boost
                    ),
                    idle_agents=self._population_metrics.idle_agents,
                    overloaded_agents=self._population_metrics.overloaded_agents,
                )

    # =========================================================================
    # Public Query API
    # =========================================================================

    def get_optimal_population(self) -> int:
        """Compute optimal population based on total load.

        optimal = ceil(total_load / target_load_per_agent)
        Clamped to [min_population, max_population].
        """
        if not self._agent_loads:
            return self._min_population

        total_load = sum(self._agent_loads.values())
        raw_optimal = math.ceil(total_load / _TARGET_LOAD_PER_AGENT)
        return max(self._min_population, min(self._max_population, raw_optimal))

    def is_population_healthy(self) -> bool:
        """Whether the population is within healthy bounds.

        Healthy means:
        - Population is within [min, max]
        - No extreme overload (>50% of agents overloaded)
        """
        m = self._population_metrics
        if m.current_agents < self._min_population:
            return False
        if m.current_agents > self._max_population:
            return False
        if m.current_agents > 0:
            overload_fraction = m.overloaded_agents / m.current_agents
            if overload_fraction > 0.5:
                return False
        return True

    @property
    def population_metrics(self) -> PopulationMetrics:
        """Current population metrics snapshot."""
        return self._population_metrics

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return reproductive system statistics for monitoring."""
        m = self._population_metrics
        return {
            "current_agents": m.current_agents,
            "optimal_agents": m.optimal_agents,
            "load_per_agent": round(m.load_per_agent, 4),
            "idle_agents": m.idle_agents,
            "overloaded_agents": m.overloaded_agents,
            "is_healthy": self.is_population_healthy(),
            "division_count": self._division_count,
            "retirement_count": self._retirement_count,
            "spawn_threshold": self._spawn_threshold,
            "retire_threshold": self._retire_threshold,
            "min_population": self._min_population,
            "max_population": self._max_population,
            "spawn_cooldown": self._spawn_cooldown,
            "last_spawn_step": self._last_spawn_step,
            "step_count": self._step_count,
            "history_length": len(self._metrics_history),
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict:
        """Serialize reproductive system state for persistence."""
        m = self._population_metrics
        return {
            "population_metrics": {
                "current_agents": m.current_agents,
                "optimal_agents": m.optimal_agents,
                "load_per_agent": m.load_per_agent,
                "idle_agents": m.idle_agents,
                "overloaded_agents": m.overloaded_agents,
            },
            "agent_loads": {str(k): v for k, v in self._agent_loads.items()},
            "division_count": self._division_count,
            "retirement_count": self._retirement_count,
            "step_count": self._step_count,
            "last_spawn_step": self._last_spawn_step,
            "heart_rate_of_spawning": self._spawn_threshold,
        }

    def restore(self, data: dict) -> None:
        """Restore reproductive system state from serialized data."""
        if not isinstance(data, dict):
            return

        pm = data.get("population_metrics")
        if isinstance(pm, dict):
            self._population_metrics = PopulationMetrics(
                current_agents=int(pm.get("current_agents", 0)),
                optimal_agents=int(pm.get("optimal_agents", 3)),
                load_per_agent=float(pm.get("load_per_agent", 0.0)),
                idle_agents=int(pm.get("idle_agents", 0)),
                overloaded_agents=int(pm.get("overloaded_agents", 0)),
            )

        agent_loads = data.get("agent_loads", {})
        if isinstance(agent_loads, dict):
            self._agent_loads = {int(k): float(v) for k, v in agent_loads.items()}

        self._division_count = int(data.get("division_count", 0))
        self._retirement_count = int(data.get("retirement_count", 0))
        self._step_count = int(data.get("step_count", 0))
        self._last_spawn_step = int(data.get("last_spawn_step", 0))

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_message(message: Any) -> Optional[dict]:
        """Parse an EventBus message (may be JSON string or dict)."""
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None
