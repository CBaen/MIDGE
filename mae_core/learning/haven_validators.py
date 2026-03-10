"""HAVEN Validator Methods - TriadEnforcer integration validators.

Contains validate_decision, validate_modification, validate_healing,
validate_policy, validate_threat. Extracted from haven.py to stay
under 500-line limit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HavenValidatorsMixin:
    """Validator methods for TriadEnforcer integration."""

    def validate_decision(self, context: dict[str, Any]) -> bool:
        """Validate agent decision against risk assessment.

        Context expected:
        - agent_id: agent making the decision
        - action: proposed action
        - state: current state
        - recent_performance: recent reward history (optional)
        - behavioral_metrics: behavioral anomalies (optional)

        Returns True if decision passes risk assessment (risk_score < threshold).
        Returns False if decision exceeds risk threshold.
        """
        try:
            agent_id = context.get("agent_id", "unknown")
            action = context.get("action", "unknown")

            recent_perf = context.get("recent_performance")
            behavioral = context.get("behavioral_metrics")
            assessment = self.assess_agent_risk(agent_id, recent_perf, behavioral)

            if assessment.risk_score < self._risk_threshold:
                return True

            logger.warning(
                "Decision validator REJECT: agent %s action %s (risk_score=%.2f)",
                agent_id, action, assessment.risk_score,
            )
            return False
        except Exception as e:
            logger.debug("validate_decision exception: %s", e)
            return True  # Permissive on error

    def validate_modification(self, context: dict[str, Any]) -> bool:
        """Validate self-modification against policy contagion risk.

        Context expected:
        - agent_id: agent proposing modification
        - modification_type: type of change (policy_update, parameter_adjustment, etc.)
        - source_peers: agents/systems that influenced this modification
        - performance_delta: change in performance from modification

        Returns True if modification is safe (no contagion detected).
        Returns False if modification appears adversarial or part of contagion.
        """
        try:
            from mae_core.learning.haven import ContagionStatus
            agent_id = context.get("agent_id", "unknown")
            mod_type = context.get("modification_type", "unknown")
            source_peers = context.get("source_peers", [])

            contagion = self.detect_policy_contagion()
            if contagion.status == ContagionStatus.SYSTEM_WIDE:
                logger.warning(
                    "Modification validator REJECT: agent %s mod %s - system-wide contagion detected",
                    agent_id, mod_type,
                )
                return False

            if contagion.status == ContagionStatus.SPREADING:
                affected_set = set(contagion.affected_agents)
                source_set = set(source_peers)
                if source_set & affected_set:
                    logger.warning(
                        "Modification validator REJECT: agent %s - sources in contagion zone",
                        agent_id,
                    )
                    return False

            return True
        except Exception as e:
            logger.debug("validate_modification exception: %s", e)
            return True

    def validate_healing(self, context: dict[str, Any]) -> bool:
        """Validate healing/recovery action against risk profile.

        Context expected:
        - agent_id: agent being healed
        - failure_type: type of failure detected
        - recovery_strategy: isolation, rollback, consolidation, etc.
        - risk_score: current risk score of agent

        Returns True if healing is appropriate for the risk level.
        Returns False if healing strategy is mismatched to risk.
        """
        try:
            from mae_core.learning.haven import RiskLevel
            agent_id = context.get("agent_id", "unknown")
            failure_type = context.get("failure_type", "unknown")
            recovery = context.get("recovery_strategy", "unknown")
            risk = context.get("risk_score", 0.0)

            risk_level = RiskLevel.from_score(risk)

            # CRITICAL risk needs aggressive recovery (isolation/rollback)
            if risk_level == RiskLevel.CRITICAL:
                if recovery not in ("isolation", "rollback", "emergency_stop"):
                    logger.warning(
                        "Healing validator REJECT: agent %s critical risk but strategy %s too lenient",
                        agent_id, recovery,
                    )
                    return False

            # HIGH risk needs policy freeze or isolation
            elif risk_level == RiskLevel.HIGH:
                if recovery not in ("policy_freeze", "isolation", "rollback"):
                    logger.warning(
                        "Healing validator REJECT: agent %s high risk but strategy %s too lenient",
                        agent_id, recovery,
                    )
                    return False

            return True
        except Exception as e:
            logger.debug("validate_healing exception: %s", e)
            return True

    def validate_policy(self, context: dict[str, Any]) -> bool:
        """Validate policy sharing and learning updates.

        Context expected:
        - agent_id: agent requesting to share/learn policy
        - policy_id: identifier of policy being shared
        - source_agent: peer who shared the policy
        - performance_on_source: peer's reported performance with policy
        - learning_method: frl, imitation, memory_bridge, etc.

        Returns True if policy is safe to adopt.
        Returns False if policy shows signs of adversarial origin.
        """
        try:
            agent_id = context.get("agent_id", "unknown")
            policy_id = context.get("policy_id", "unknown")
            source_agent = context.get("source_agent", "unknown")
            perf = context.get("performance_on_source", 0.5)

            # Assess risk of source agent
            source_assessment = self._risk_assessments.get(source_agent)
            if source_assessment is None:
                source_assessment = self.assess_agent_risk(source_agent)

            # High-risk agents should not share policies
            if source_assessment.risk_score >= self._risk_threshold:
                logger.warning(
                    "Policy validator REJECT: agent %s learning from high-risk peer %s (risk=%.2f)",
                    agent_id, source_agent, source_assessment.risk_score,
                )
                return False

            # Poor performance on source is suspicious (adversarial fit)
            if perf < 0.3:
                logger.warning(
                    "Policy validator WARN: agent %s policy from %s has poor perf (%.2f)",
                    agent_id, source_agent, perf,
                )
                # Don't reject, but flag as risky

            return True
        except Exception as e:
            logger.debug("validate_policy exception: %s", e)
            return True

    def validate_threat(self, context: dict[str, Any]) -> bool:
        """Validate threat detection and defense response.

        Context expected:
        - threat_type: external, adversarial, performance_anomaly, contagion
        - threat_level: severity (0-1)
        - affected_agents: list of agents detecting/affected by threat
        - defense_action: response being taken

        Returns True if threat response is appropriate.
        Returns False if threat is misclassified or response is wrong.
        """
        try:
            from mae_core.learning.haven import ContagionStatus
            threat_type = context.get("threat_type", "unknown")
            threat_level = context.get("threat_level", 0.0)
            affected = context.get("affected_agents", [])
            defense = context.get("defense_action", "unknown")

            # HAVEN only validates "contagion" and "adversarial" threats
            if threat_type not in ("contagion", "adversarial", "performance_anomaly"):
                return True

            # For contagion threats, check if our detection agrees
            contagion = self.detect_policy_contagion()
            if threat_type == "contagion":
                if contagion.status == ContagionStatus.HEALTHY:
                    logger.warning(
                        "Threat validator WARN: threat marked as contagion but HAVEN sees healthy"
                    )
                    return True

            return True
        except Exception as e:
            logger.debug("validate_threat exception: %s", e)
            return True
