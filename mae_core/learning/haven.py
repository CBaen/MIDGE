"""HAVEN - Hierarchical Adversarial Value Estimation Network.

Immune-system-inspired defense: monitors agent risk, detects policy
contagion, and intervenes when agents diverge dangerously.

Biological analogy: Immune system (T-cells, quarantine, recovery).
Based on: Gleave et al. (2020) "Adversarial Policies".

This file keeps risk models, data classes, and the coordinator core.
Validator methods live in haven_validators.py (HavenValidatorsMixin).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

from ..backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

CH_RISK_ALERT = "haven.risk_alert"
CH_INTERVENTION = "haven.intervention"


class RiskLevel(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        if score < 0.2:
            return cls.MINIMAL
        if score < 0.4:
            return cls.LOW
        if score < 0.6:
            return cls.MODERATE
        if score < 0.8:
            return cls.HIGH
        return cls.CRITICAL


class InterventionType(Enum):
    NONE = "none"
    MONITORING = "monitoring"
    POLICY_FREEZE = "policy_freeze"
    ISOLATION = "isolation"
    ROLLBACK = "rollback"
    EMERGENCY_STOP = "emergency_stop"


class ContagionStatus(Enum):
    HEALTHY = "healthy"
    EARLY_WARNING = "early_warning"
    SPREADING = "spreading"
    CONTAINED = "contained"
    SYSTEM_WIDE = "system_wide"


@dataclass
class RiskAssessment:
    """Risk evaluation for a single agent."""

    agent_id: str
    risk_score: float  # [0, 1]
    risk_level: RiskLevel
    factors: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ContagionReport:
    """Report on policy contagion across agents."""

    status: ContagionStatus
    affected_agents: list[str] = field(default_factory=list)
    source_agents: list[str] = field(default_factory=list)
    spread_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class _HavenCoordinatorCore:
    """Immune-system-inspired risk management core."""

    def __init__(
        self,
        event_bus: EventBus,
        coordinator_id: str = "haven",
        risk_threshold: float = 0.7,
        contagion_threshold: float = 0.5,
        performance_window: int = 50,
        intervention_cooldown: int = 10,
    ) -> None:
        self._bus = event_bus
        self._id = coordinator_id
        self._risk_threshold = risk_threshold
        self._contagion_threshold = contagion_threshold
        self._perf_window = performance_window
        self._cooldown = intervention_cooldown

        self._agents: set[str] = set()
        self._risk_assessments: dict[str, RiskAssessment] = {}
        self._performance_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=performance_window)
        )
        self._contagion_graph: dict[str, set[str]] = defaultdict(set)
        self._active_interventions: dict[str, InterventionType] = {}
        self._intervention_log: list[dict[str, Any]] = []
        self._isolated_agents: set[str] = set()
        self._lock = threading.RLock()

    def register_agent(self, agent_id: str) -> None:
        with self._lock:
            self._agents.add(agent_id)

    def assess_agent_risk(
        self,
        agent_id: str,
        recent_performance: list[float] | None = None,
        behavioral_metrics: dict[str, float] | None = None,
    ) -> RiskAssessment:
        """Evaluate risk level of an individual agent."""
        with self._lock:
            factors: dict[str, float] = {}

            if recent_performance:
                for p in recent_performance:
                    self._performance_history[agent_id].append(p)

            perf_history = list(self._performance_history.get(agent_id, []))
            if len(perf_history) >= 10:
                first_half = np.mean(perf_history[: len(perf_history) // 2])
                second_half = np.mean(perf_history[len(perf_history) // 2:])
                degradation = max(0.0, (first_half - second_half) / max(abs(first_half), 0.01))
                factors["performance_degradation"] = min(1.0, degradation)
            else:
                factors["performance_degradation"] = 0.0

            if behavioral_metrics:
                anomaly = sum(
                    1 for v in behavioral_metrics.values() if abs(v) > 2.0
                ) / max(len(behavioral_metrics), 1)
                factors["behavioral_anomaly"] = anomaly
            else:
                factors["behavioral_anomaly"] = 0.0

            factors["is_isolated"] = 1.0 if agent_id in self._isolated_agents else 0.0

            risk_score = (
                0.5 * factors["performance_degradation"]
                + 0.3 * factors["behavioral_anomaly"]
                + 0.2 * factors["is_isolated"]
            )
            risk_score = max(0.0, min(1.0, risk_score))

            assessment = RiskAssessment(
                agent_id=agent_id,
                risk_score=risk_score,
                risk_level=RiskLevel.from_score(risk_score),
                factors=factors,
            )
            self._risk_assessments[agent_id] = assessment

            if risk_score >= self._risk_threshold:
                self._bus.publish(CH_RISK_ALERT, {
                    "agent_id": agent_id,
                    "risk_score": risk_score,
                    "risk_level": assessment.risk_level.value,
                })

            return assessment

    def assess_system_risk(self) -> float:
        """Overall system risk score [0, 1]."""
        with self._lock:
            if not self._risk_assessments:
                return 0.0
            scores = [a.risk_score for a in self._risk_assessments.values()]
            return 0.6 * max(scores) + 0.4 * float(np.mean(scores))

    def detect_policy_contagion(self, time_window: float = 60.0) -> ContagionReport:
        """Detect if bad policies are spreading through the network."""
        with self._lock:
            high_risk = [
                aid for aid, a in self._risk_assessments.items()
                if a.risk_score >= self._contagion_threshold
                and (time.time() - a.timestamp) < time_window
            ]

            if len(high_risk) == 0:
                return ContagionReport(status=ContagionStatus.HEALTHY)
            if len(high_risk) == 1:
                return ContagionReport(
                    status=ContagionStatus.EARLY_WARNING,
                    affected_agents=high_risk,
                )

            spread_rate = len(high_risk) / max(len(self._agents), 1)
            if spread_rate > 0.5:
                status = ContagionStatus.SYSTEM_WIDE
            elif spread_rate > 0.2:
                status = ContagionStatus.SPREADING
            else:
                status = ContagionStatus.EARLY_WARNING

            return ContagionReport(
                status=status,
                affected_agents=high_risk,
                spread_rate=spread_rate,
            )

    def recommend_intervention(self, assessment: RiskAssessment) -> InterventionType:
        """Recommend intervention based on risk level."""
        if assessment.risk_level == RiskLevel.CRITICAL:
            return InterventionType.ROLLBACK
        if assessment.risk_level == RiskLevel.HIGH:
            return InterventionType.POLICY_FREEZE
        if assessment.risk_level == RiskLevel.MODERATE:
            return InterventionType.MONITORING
        return InterventionType.NONE

    def execute_intervention(
        self, agent_id: str, intervention: InterventionType, reason: str = ""
    ) -> bool:
        """Apply an intervention to an agent."""
        with self._lock:
            self._active_interventions[agent_id] = intervention
            self._intervention_log.append({
                "agent_id": agent_id,
                "intervention": intervention.value,
                "reason": reason,
                "timestamp": time.time(),
            })

            if intervention == InterventionType.ISOLATION:
                self._isolated_agents.add(agent_id)

            self._bus.publish(CH_INTERVENTION, {
                "agent_id": agent_id,
                "intervention": intervention.value,
                "reason": reason,
            })

            return True

    def isolate_agent(self, agent_id: str, reason: str = "") -> None:
        self.execute_intervention(agent_id, InterventionType.ISOLATION, reason)

    def restore_agent(self, agent_id: str) -> bool:
        with self._lock:
            self._isolated_agents.discard(agent_id)
            self._active_interventions.pop(agent_id, None)
            return True

    def is_agent_isolated(self, agent_id: str) -> bool:
        return agent_id in self._isolated_agents

    def get_system_health_report(self) -> dict[str, Any]:
        with self._lock:
            contagion = self.detect_policy_contagion()
            return {
                "system_risk": self.assess_system_risk(),
                "total_agents": len(self._agents),
                "monitored_agents": len(self._risk_assessments),
                "isolated_agents": len(self._isolated_agents),
                "active_interventions": {
                    aid: it.value for aid, it in self._active_interventions.items()
                },
                "contagion_status": contagion.status.value,
                "contagion_affected": len(contagion.affected_agents),
                "intervention_count": len(self._intervention_log),
            }

    def __repr__(self) -> str:
        return (
            f"HAVEN(agents={len(self._agents)}, "
            f"risk={self.assess_system_risk():.2f}, "
            f"isolated={len(self._isolated_agents)})"
        )


# Compose the final class after all definitions are complete.
from mae_core.learning.haven_validators import HavenValidatorsMixin  # noqa: E402


class HavenRiskCoordinator(_HavenCoordinatorCore, HavenValidatorsMixin):
    """Immune-system-inspired risk management for the agent swarm.

    Monitors agent behavior, detects policy contagion (when bad policies
    spread through FRL sharing), and intervenes to protect the system.

    Composed from:
    - _HavenCoordinatorCore: risk assessment, intervention, contagion detection
    - HavenValidatorsMixin: validate_decision, validate_modification,
                            validate_healing, validate_policy, validate_threat
    """


__all__ = [
    "HavenRiskCoordinator",
    "RiskLevel",
    "InterventionType",
    "ContagionStatus",
    "RiskAssessment",
    "ContagionReport",
    "CH_RISK_ALERT",
    "CH_INTERVENTION",
    "HavenValidatorsMixin",
]
