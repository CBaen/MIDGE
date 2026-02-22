"""Capability Discovery - Finding what Mae can do that she doesn't know yet.

Biological analogy: Immune system's adaptive immunity. When T-cells
encounter a novel pathogen, they don't have pre-programmed responses.
Instead, through VDJ recombination and clonal selection, they DISCOVER
new capabilities that didn't exist before. The capability persists -
this is how vaccination works.

Similarly, Mae's agents may develop novel behaviors through interaction
that weren't explicitly designed. Capability discovery detects these
emergent behaviors, validates them, and registers them as new skills.

Pipeline:
1. OBSERVE - Monitor agent interaction patterns for anomalies
2. CHARACTERIZE - Describe what the new behavior does
3. VALIDATE - Test if it's genuinely useful (not noise)
4. REGISTER - Make it available to other agents

Connection points:
- EventBus receives performance/behavior metrics
- Morphogenesis can spawn specialists with new capabilities
- Memory stores capability signatures for recall
- HAVEN validates capabilities aren't adversarial
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_CAPABILITY_FOUND = "improvement.capability_found"
CH_CAPABILITY_VALIDATED = "improvement.capability_validated"
CH_CAPABILITY_RETIRED = "improvement.capability_retired"
CH_IMPROVEMENT_METRIC = "improvement.metric"


class CapabilityStatus(Enum):
    OBSERVED = "observed"  # Seen but not validated
    TESTING = "testing"  # Under validation
    VALIDATED = "validated"  # Confirmed useful
    DEPLOYED = "deployed"  # Available to all agents
    RETIRED = "retired"  # No longer useful


@dataclass
class CapabilitySignature:
    """Description of an emergent capability."""

    capability_id: str
    description: str
    discovered_by: str  # Agent ID
    context: str  # What conditions trigger it
    performance_delta: float  # How much it improves over baseline
    status: CapabilityStatus = CapabilityStatus.OBSERVED
    validation_score: float = 0.0  # [0, 1] after testing
    usage_count: int = 0
    discovered_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementMetric:
    """Tracked metric for self-improvement."""

    name: str
    current_value: float
    baseline_value: float
    trend: float = 0.0  # Positive = improving
    samples: int = 0
    last_updated: float = field(default_factory=time.time)


class CapabilityDiscovery:
    """Detects, validates, and registers emergent capabilities.

    Watches agent behavior for performance anomalies that might
    indicate novel capabilities emerging from interaction patterns.
    """

    def __init__(
        self,
        event_bus: Any = None,
        novelty_threshold: float = 0.3,
        validation_rounds: int = 5,
        min_performance_delta: float = 0.1,
        max_capabilities: int = 50,
        morphogenesis_coordinator: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
    ) -> None:
        self._bus = event_bus
        self._novelty_threshold = novelty_threshold
        self._validation_rounds = validation_rounds
        self._min_delta = min_performance_delta
        self._max_capabilities = max_capabilities
        self._morphogenesis = morphogenesis_coordinator
        self._knowledge_base = knowledge_base

        # Known capabilities
        self._capabilities: dict[str, CapabilitySignature] = {}

        # Performance baselines per agent
        self._baselines: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._recent_performance: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=20)
        )

        # Improvement metrics
        self._metrics: dict[str, ImprovementMetric] = {}

        # Validation tracking
        self._validation_scores: dict[str, list[float]] = defaultdict(list)

        self._lock = threading.RLock()
        self._total_discoveries = 0

        # Subscribe to EventBus channels
        if self._bus:
            self._bus.register_callback(
                "morphogenesis.novelty_detected", self.on_novelty_detected
            )

        logger.info("CapabilityDiscovery initialized (novelty_threshold=%.2f)", novelty_threshold)

    # =========================================================================
    # Convenience Wrappers
    # =========================================================================

    def observe_agent_performance(
        self, agent_id: str, metric_name: str, value: float
    ) -> None:
        """Convenience wrapper: observe performance and track metric together.

        Combines observe_performance() and track_metric() in a single call
        for external systems that report agent-level metrics.
        """
        self.observe_performance(
            agent_id=agent_id,
            performance=value,
            context=metric_name,
        )
        self.track_metric(f"{agent_id}.{metric_name}", value)

    def on_novelty_detected(self, channel: str, message: Any) -> None:
        """Handle morphogenesis.novelty_detected events.

        Cross-references novelty with the capability pipeline to see
        if the novel problem domain matches any emerging capabilities.
        """
        if isinstance(message, str):
            try:
                import json
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return
        if not isinstance(message, dict):
            return

        domain = message.get("domain", "")
        complexity = message.get("complexity", 0.5)
        # Track as an improvement metric so we can see novelty trends
        self.track_metric(f"novelty.{domain}", complexity)

    # =========================================================================
    # Observation
    # =========================================================================

    def observe_performance(
        self,
        agent_id: str,
        performance: float,
        context: str = "",
        behavior_signature: str = "",
    ) -> Optional[CapabilitySignature]:
        """Observe an agent's performance and check for novel capability.

        Returns a CapabilitySignature if a novel capability is detected.
        """
        with self._lock:
            self._baselines[agent_id].append(performance)
            self._recent_performance[agent_id].append(performance)

            # Need enough baseline data
            baseline = list(self._baselines[agent_id])
            if len(baseline) < 20:
                return None

            # Compare recent to baseline
            baseline_mean = sum(baseline) / len(baseline)
            recent = list(self._recent_performance[agent_id])
            recent_mean = sum(recent) / len(recent)

            delta = recent_mean - baseline_mean

            # Is this a significant positive anomaly?
            if delta > self._novelty_threshold and delta > self._min_delta:
                # Check if we already know about this
                sig_key = f"{agent_id}:{behavior_signature or context}"
                if sig_key in self._capabilities:
                    cap = self._capabilities[sig_key]
                    cap.usage_count += 1
                    cap.performance_delta = max(cap.performance_delta, delta)
                    return None  # Already known

                # New capability detected!
                capability = CapabilitySignature(
                    capability_id=sig_key,
                    description=f"Emergent behavior in {context or 'unknown context'}",
                    discovered_by=agent_id,
                    context=context,
                    performance_delta=delta,
                    metadata={"behavior_signature": behavior_signature},
                )

                if len(self._capabilities) < self._max_capabilities:
                    self._capabilities[sig_key] = capability
                    self._total_discoveries += 1

                    if self._bus:
                        self._bus.publish(CH_CAPABILITY_FOUND, {
                            "capability_id": sig_key,
                            "agent_id": agent_id,
                            "context": context,
                            "performance_delta": delta,
                        })

                    # Notify morphogenesis coordinator if available
                    if self._morphogenesis is not None:
                        try:
                            if hasattr(self._morphogenesis, "handle_novel_problem"):
                                from mae_core.morphogenesis.organ_builder import ProblemSignature
                                self._morphogenesis.handle_novel_problem(
                                    ProblemSignature(
                                        domain=context or "emergent",
                                        complexity=min(1.0, delta),
                                        exploration_level=0.7,
                                    ),
                                    name=f"capability_{sig_key[:20]}",
                                )
                        except Exception:
                            logger.debug("Could not notify morphogenesis of new capability")

                    logger.info(
                        "Novel capability detected: %s (delta=%.3f)",
                        sig_key, delta,
                    )
                    return capability

            return None

    # =========================================================================
    # Validation
    # =========================================================================

    def submit_validation(
        self, capability_id: str, score: float
    ) -> Optional[CapabilitySignature]:
        """Submit a validation score for a capability.

        Returns updated capability if validation is complete.
        """
        with self._lock:
            cap = self._capabilities.get(capability_id)
            if cap is None:
                return None

            self._validation_scores[capability_id].append(score)
            scores = self._validation_scores[capability_id]

            if len(scores) >= self._validation_rounds:
                avg_score = sum(scores) / len(scores)
                cap.validation_score = avg_score

                if avg_score >= 0.6:
                    cap.status = CapabilityStatus.VALIDATED
                    if self._bus:
                        self._bus.publish(CH_CAPABILITY_VALIDATED, {
                            "capability_id": capability_id,
                            "validation_score": avg_score,
                        })
                    # Store validated capability in knowledge base
                    if self._knowledge_base is not None:
                        try:
                            if hasattr(self._knowledge_base, "store"):
                                self._knowledge_base.store(
                                    f"capability:{capability_id}",
                                    {
                                        "capability_id": capability_id,
                                        "validation_score": avg_score,
                                        "description": cap.description,
                                        "context": cap.context,
                                        "performance_delta": cap.performance_delta,
                                    },
                                )
                        except Exception:
                            logger.debug("Could not store capability in knowledge base")
                    logger.info(
                        "Capability validated: %s (score=%.2f)",
                        capability_id, avg_score,
                    )
                else:
                    cap.status = CapabilityStatus.RETIRED
                    if self._bus:
                        self._bus.publish(CH_CAPABILITY_RETIRED, {
                            "capability_id": capability_id,
                            "reason": "failed_validation",
                        })

            return cap

    def deploy_capability(self, capability_id: str) -> bool:
        """Mark a validated capability as deployed (available to all)."""
        with self._lock:
            cap = self._capabilities.get(capability_id)
            if cap and cap.status == CapabilityStatus.VALIDATED:
                cap.status = CapabilityStatus.DEPLOYED
                return True
            return False

    def retire_capability(self, capability_id: str, reason: str = "") -> bool:
        """Retire a capability that is no longer useful."""
        with self._lock:
            cap = self._capabilities.get(capability_id)
            if cap:
                cap.status = CapabilityStatus.RETIRED
                if self._bus:
                    self._bus.publish(CH_CAPABILITY_RETIRED, {
                        "capability_id": capability_id,
                        "reason": reason,
                    })
                return True
            return False

    # =========================================================================
    # Improvement Tracking
    # =========================================================================

    def track_metric(
        self, name: str, value: float, baseline: Optional[float] = None
    ) -> ImprovementMetric:
        """Track a self-improvement metric over time."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = ImprovementMetric(
                    name=name,
                    current_value=value,
                    baseline_value=baseline if baseline is not None else value,
                )
            metric = self._metrics[name]

            # Update trend (exponential moving average)
            old_value = metric.current_value
            metric.current_value = value
            metric.samples += 1
            delta = value - old_value
            metric.trend = 0.9 * metric.trend + 0.1 * delta
            metric.last_updated = time.time()

            if self._bus:
                self._bus.publish(CH_IMPROVEMENT_METRIC, {
                    "metric": name,
                    "value": value,
                    "trend": metric.trend,
                    "improvement": value - metric.baseline_value,
                })

            return metric

    def get_improvement_summary(self) -> dict[str, dict[str, float]]:
        """Get summary of all improvement metrics."""
        with self._lock:
            return {
                name: {
                    "current": m.current_value,
                    "baseline": m.baseline_value,
                    "improvement": m.current_value - m.baseline_value,
                    "trend": m.trend,
                    "samples": m.samples,
                }
                for name, m in self._metrics.items()
            }

    # =========================================================================
    # Queries
    # =========================================================================

    def get_capabilities(
        self, status: Optional[CapabilityStatus] = None
    ) -> list[CapabilitySignature]:
        """Get all known capabilities, optionally filtered by status."""
        with self._lock:
            caps = list(self._capabilities.values())
            if status:
                caps = [c for c in caps if c.status == status]
            return caps

    def get_deployed_capabilities(self) -> list[CapabilitySignature]:
        return self.get_capabilities(CapabilityStatus.DEPLOYED)

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            status_counts: dict[str, int] = defaultdict(int)
            for cap in self._capabilities.values():
                status_counts[cap.status.value] += 1

            return {
                "total_discoveries": self._total_discoveries,
                "active_capabilities": len(self._capabilities),
                "status_breakdown": dict(status_counts),
                "tracked_metrics": len(self._metrics),
                "agents_monitored": len(self._baselines),
            }
