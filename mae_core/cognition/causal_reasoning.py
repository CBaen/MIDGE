"""Causal Reasoning Engine - Understanding WHY, not just WHAT.

Builds and queries a causal graph to distinguish correlation from causation.
Supports do-calculus (interventions), counterfactual reasoning, and
confounder identification.

Biological analogy: Prefrontal cortex causal inference.
Based on: Pearl (2009) "Causality", Scholkopf (2022) "Causal Representation Learning".
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

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    DIRECT = "direct"  # A directly causes B
    INDIRECT = "indirect"  # A → X → B
    CONFOUNDED = "confounded"  # C causes both A and B
    COLLIDER = "collider"  # A and B both cause C
    MEDIATOR = "mediator"  # A → M → B


@dataclass
class CausalLink:
    """A causal relationship between two variables."""

    cause: str
    effect: str
    relation_type: CausalRelationType = CausalRelationType.DIRECT
    strength: float = 0.5  # [0, 1]
    confidence: float = 0.5  # [0, 1]
    evidence_count: int = 0
    discovered_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.evidence_count = max(0, self.evidence_count)


@dataclass
class CausalQueryResult:
    """Result of a causal query."""

    query_id: str
    cause: str
    effect: str
    is_causal: bool
    causal_path: list[str] = field(default_factory=list)
    causal_strength: float = 0.0
    counterfactual: str = ""
    explanation: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


class CausalReasoningEngine:
    """Builds causal graphs from observations and interventions.

    Key distinction: Correlation is NOT causation.
    - observe_correlation(): Weak evidence, needs confirmation
    - observe_intervention(): Strong evidence (do-calculus)
    - query_causation(): Answer "does X cause Y?"
    - generate_counterfactual(): "What if X hadn't happened?"
    """

    def __init__(
        self,
        min_evidence_threshold: int = 3,
        confidence_threshold: float = 0.6,
        max_path_length: int = 5,
        event_bus: Any | None = None,
    ) -> None:
        self._min_evidence = min_evidence_threshold
        self._conf_threshold = confidence_threshold
        self._max_path = max_path_length
        self._bus = event_bus

        # Causal graph
        self._links: dict[tuple[str, str], CausalLink] = {}
        self._causes: dict[str, set[str]] = defaultdict(set)  # effect -> {causes}
        self._effects: dict[str, set[str]] = defaultdict(set)  # cause -> {effects}

        # Evidence storage
        self._observations: deque[dict[str, Any]] = deque(maxlen=100000)
        self._interventions: deque[dict[str, Any]] = deque(maxlen=100000)

        # Statistics
        self._queries_answered = 0
        self._links_discovered = 0
        self._total_evidence = 0
        self._lock = threading.RLock()

        # Subscribe to temporal events if EventBus available
        if self._bus is not None:
            self._bus.register_callback(
                "temporal.causal_link_discovered",
                self._on_causal_link_discovered,
            )
            self._bus.register_callback(
                "temporal.event_recorded",
                self._on_event_recorded,
            )

    def observe_correlation(
        self,
        variable_a: str,
        variable_b: str,
        correlation_strength: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record a correlation (does NOT establish causation)."""
        self._observations.append({
            "type": "correlation",
            "a": variable_a,
            "b": variable_b,
            "strength": correlation_strength,
            "context": context or {},
            "timestamp": time.time(),
        })
        self._total_evidence += 1

        # Only consider establishing causal link if strong and enough evidence
        if abs(correlation_strength) > 0.5:
            key = (variable_a, variable_b)
            with self._lock:
                if key in self._links:
                    link = self._links[key]
                    link.evidence_count += 1
                    # EMA update of strength
                    link.strength = 0.9 * link.strength + 0.1 * abs(correlation_strength)
                    link.confidence = min(
                        1.0,
                        link.confidence + 0.05 * (link.evidence_count / self._min_evidence),
                    )
                elif self._count_evidence(variable_a, variable_b) >= self._min_evidence:
                    self.add_causal_link(
                        variable_a, variable_b,
                        strength=abs(correlation_strength) * 0.5,
                        confidence=0.3,
                    )

    def observe_intervention(
        self,
        intervention_variable: str,
        intervention_value: Any,
        observed_effects: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an intervention (strong causal evidence via do-calculus)."""
        self._interventions.append({
            "type": "intervention",
            "variable": intervention_variable,
            "value": intervention_value,
            "effects": observed_effects,
            "context": context or {},
            "timestamp": time.time(),
        })
        self._total_evidence += 1

        # Interventions create/strengthen direct causal links
        with self._lock:
            for effect_var, effect_val in observed_effects.items():
                key = (intervention_variable, effect_var)
                if key in self._links:
                    link = self._links[key]
                    link.evidence_count += 1
                    link.strength = min(1.0, link.strength + 0.1)
                    link.confidence = min(1.0, link.confidence + 0.1)
                else:
                    self.add_causal_link(
                        intervention_variable, effect_var,
                        strength=0.8,
                        confidence=0.8,
                    )

    def add_causal_link(
        self,
        cause: str,
        effect: str,
        relation_type: CausalRelationType = CausalRelationType.DIRECT,
        strength: float = 0.5,
        confidence: float = 0.5,
    ) -> CausalLink:
        """Add or update a causal link in the graph."""
        with self._lock:
            key = (cause, effect)
            if key in self._links:
                link = self._links[key]
                link.evidence_count += 1
                link.strength = max(link.strength, strength)
                link.confidence = max(link.confidence, confidence)
            else:
                link = CausalLink(
                    cause=cause, effect=effect,
                    relation_type=relation_type,
                    strength=strength, confidence=confidence,
                    evidence_count=1,
                )
                self._links[key] = link
                self._causes[effect].add(cause)
                self._effects[cause].add(effect)
                self._links_discovered += 1

            return link

    def query_causation(
        self, potential_cause: str, potential_effect: str
    ) -> CausalQueryResult:
        """Answer: Does X cause Y?"""
        self._queries_answered += 1
        query_id = f"cq-{self._queries_answered}"

        with self._lock:
            path = self.find_causal_path(potential_cause, potential_effect)

            if path is None:
                return CausalQueryResult(
                    query_id=query_id,
                    cause=potential_cause,
                    effect=potential_effect,
                    is_causal=False,
                    explanation=f"No causal path from '{potential_cause}' to '{potential_effect}'",
                )

            strength = self.compute_causal_strength(path)
            counterfactual = self.generate_counterfactual(potential_cause, potential_effect)
            explanation = self.explain_causation(potential_cause, potential_effect, path)
            confidence = min(
                link.confidence
                for i in range(len(path) - 1)
                if (link := self._links.get((path[i], path[i + 1]))) is not None
            ) if len(path) > 1 else 0.0

            result = CausalQueryResult(
                query_id=query_id,
                cause=potential_cause,
                effect=potential_effect,
                is_causal=strength > 0 and confidence >= self._conf_threshold,
                causal_path=path,
                causal_strength=strength,
                counterfactual=counterfactual,
                explanation=explanation,
                confidence=confidence,
            )

            if self._bus is not None:
                try:
                    self._bus.publish("cognition.causal_query_result", {
                        "query_id": query_id,
                        "cause": potential_cause,
                        "effect": potential_effect,
                        "is_causal": result.is_causal,
                        "causal_strength": strength,
                        "confidence": confidence,
                    })
                except Exception:
                    logger.debug("EventBus publish failed for causal_query_result")

            return result

    def find_causal_path(
        self, start: str, end: str
    ) -> list[str] | None:
        """BFS search for shortest causal path."""
        if start == end:
            return [start]

        visited = {start}
        queue: deque[list[str]] = deque([[start]])

        while queue:
            path = queue.popleft()
            if len(path) > self._max_path:
                continue

            current = path[-1]
            for next_var in self._effects.get(current, set()):
                if next_var == end:
                    return path + [end]
                if next_var not in visited:
                    visited.add(next_var)
                    queue.append(path + [next_var])

        return None

    def compute_causal_strength(self, path: list[str]) -> float:
        """Strength of a causal chain (weakest link principle)."""
        if len(path) < 2:
            return 0.0

        link_strengths = []
        for i in range(len(path) - 1):
            link = self._links.get((path[i], path[i + 1]))
            if link is None:
                return 0.0
            link_strengths.append(link.strength)

        min_strength = min(link_strengths)
        chain_penalty = 0.9 ** (len(path) - 1)
        return min_strength * chain_penalty

    def generate_counterfactual(
        self, cause: str, effect: str
    ) -> str:
        """Generate counterfactual reasoning statement."""
        path = self.find_causal_path(cause, effect)
        if path and len(path) > 1:
            if len(path) == 2:
                return (
                    f"If '{cause}' had not occurred, "
                    f"'{effect}' would likely not have happened."
                )
            intermediates = " → ".join(path[1:-1])
            return (
                f"If '{cause}' had not occurred, the chain through "
                f"{intermediates} would not have propagated to '{effect}'."
            )
        return f"No established causal link between '{cause}' and '{effect}'."

    def explain_causation(
        self, cause: str, effect: str, path: list[str]
    ) -> str:
        """Human-readable causal explanation."""
        if len(path) == 2:
            link = self._links.get((cause, effect))
            if link:
                return (
                    f"'{cause}' directly causes '{effect}' "
                    f"(strength={link.strength:.2f}, evidence={link.evidence_count})"
                )
        chain = " → ".join(path)
        strength = self.compute_causal_strength(path)
        return f"Causal chain: {chain} (combined strength={strength:.2f})"

    def identify_confounders(
        self, variable_a: str, variable_b: str
    ) -> list[str]:
        """Find common causes of both A and B (confounders)."""
        causes_a = self._causes.get(variable_a, set())
        causes_b = self._causes.get(variable_b, set())
        confounders = causes_a & causes_b

        # Exclude variables that are on the causal path A→B or B→A
        path_ab = self.find_causal_path(variable_a, variable_b) or []
        path_ba = self.find_causal_path(variable_b, variable_a) or []
        path_vars = set(path_ab + path_ba)

        return [c for c in confounders if c not in path_vars]

    def infer_causes(
        self, effect: str, observations: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Infer root causes of an observed effect."""
        direct_causes = list(self._causes.get(effect, set()))
        if not direct_causes:
            return {
                "causes": [],
                "confidence": 0.0,
                "reasoning": f"No known causes for '{effect}'",
            }

        scored = []
        for cause in direct_causes:
            link = self._links.get((cause, effect))
            if link:
                scored.append((cause, link.strength * link.confidence))

        scored.sort(key=lambda x: x[1], reverse=True)

        return {
            "causes": [c for c, _ in scored],
            "confidence": scored[0][1] if scored else 0.0,
            "reasoning": f"Top cause: '{scored[0][0]}' (score={scored[0][1]:.2f})"
            if scored
            else "Unknown",
        }

    def get_all_causes(self, variable: str) -> set[str]:
        return set(self._causes.get(variable, set()))

    def get_all_effects(self, variable: str) -> set[str]:
        return set(self._effects.get(variable, set()))

    def get_causal_metrics(self) -> dict[str, Any]:
        with self._lock:
            avg_confidence = (
                float(np.mean([l.confidence for l in self._links.values()]))
                if self._links
                else 0.0
            )
            return {
                "total_links": len(self._links),
                "total_variables": len(set(self._causes.keys()) | set(self._effects.keys())),
                "queries_answered": self._queries_answered,
                "links_discovered": self._links_discovered,
                "total_evidence": self._total_evidence,
                "observations_count": len(self._observations),
                "interventions_count": len(self._interventions),
                "average_confidence": avg_confidence,
            }

    def _on_causal_link_discovered(self, channel: str, message: Any) -> None:
        """Handle temporal.causal_link_discovered events."""
        try:
            import json
            data = json.loads(message) if isinstance(message, str) else message
            cause = data.get("cause", "")
            effect = data.get("effect", "")
            strength = data.get("strength", 0.5)
            if cause and effect:
                self.observe_correlation(cause, effect, strength)
        except Exception:
            logger.debug("Failed to process causal_link_discovered event")

    def _on_event_recorded(self, channel: str, message: Any) -> None:
        """Handle temporal.event_recorded events as observation data."""
        try:
            import json
            data = json.loads(message) if isinstance(message, str) else message
            event_type = data.get("event_type", "")
            entity_id = data.get("entity_id", "")
            if event_type and entity_id:
                self._observations.append({
                    "type": "temporal_event",
                    "a": entity_id,
                    "b": event_type,
                    "strength": data.get("importance", 0.5),
                    "context": data,
                    "timestamp": data.get("timestamp", time.time()),
                })
                self._total_evidence += 1
        except Exception:
            logger.debug("Failed to process event_recorded event")

    def _count_evidence(self, var_a: str, var_b: str) -> int:
        """Count observations mentioning both variables."""
        count = 0
        for obs in self._observations:
            if obs.get("a") == var_a and obs.get("b") == var_b:
                count += 1
        return count

    def __repr__(self) -> str:
        return (
            f"CausalReasoning(links={len(self._links)}, "
            f"queries={self._queries_answered})"
        )
