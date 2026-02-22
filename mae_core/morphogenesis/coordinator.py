"""Morphogenesis coordinator - growth engine for Mae.

Biological analogy: In multicellular organisms, morphogenesis is the
process by which an organism develops its shape. Cells differentiate
based on signals from their environment. When a wound occurs or a
new capability is needed, stem cells are activated to grow new tissue.

Mae's morphogenesis works the same way:
1. Problems arrive that existing agents can't solve well
2. Novelty detection identifies the gap
3. Blueprint design creates a plan for the new team
4. Agent spawning creates the team via Mesa 3.4
5. The team works, and may dissolve when done

Integration points:
- CollectiveDream: Low consensus triggers specialist creation
- AutoHealing: Failure detection triggers replacement
- OctopusColony: Capacity overflow triggers scaling
- Memory: Pattern recognition identifies needed capabilities
- Endocrine: Growth hormones modulate spawning rate

EventBus channels:
- morphogenesis.spawn_request: External system requests team creation
- morphogenesis.team_created: New organ created
- morphogenesis.team_dissolved: Organ dissolved
- morphogenesis.novelty_detected: Novel problem found
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.morphogenesis.organ_builder import (
    Organ,
    OrganBlueprint,
    OrganBuilder,
    ProblemSignature,
)

logger = logging.getLogger(__name__)

# EventBus channels
CH_SPAWN_REQUEST = "morphogenesis.spawn_request"
CH_TEAM_CREATED = "morphogenesis.team_created"
CH_TEAM_DISSOLVED = "morphogenesis.team_dissolved"
CH_NOVELTY_DETECTED = "morphogenesis.novelty_detected"


class NoveltyDetector:
    """Detects when a problem is genuinely novel (not seen before).

    Maintains a registry of known problem signatures and compares
    new signatures against them. If the best similarity is below
    the novelty threshold, the problem is declared novel.
    """

    def __init__(self, novelty_threshold: float = 0.3) -> None:
        self.novelty_threshold = novelty_threshold
        self._known_signatures: list[ProblemSignature] = []

    def is_novel(self, signature: ProblemSignature) -> bool:
        """Check if a problem signature is novel."""
        if not self._known_signatures:
            return True

        best_similarity = max(
            sig.similarity(signature) for sig in self._known_signatures
        )
        return best_similarity < (1.0 - self.novelty_threshold)

    def register_signature(self, signature: ProblemSignature) -> None:
        """Register a signature as known (no longer novel)."""
        self._known_signatures.append(signature)

    @property
    def known_count(self) -> int:
        return len(self._known_signatures)


class MorphogenesisCoordinator:
    """Orchestrates Mae's growth - detecting needs and spawning teams.

    The coordinator monitors for novel problems (via episode buffer or
    EventBus signals), designs appropriate agent teams, and manages
    their lifecycle.

    Can operate in two modes:
    1. Reactive: External systems call handle_novel_problem() directly
    2. Proactive: Background monitoring via start_monitoring()

    Thread-safe: All state access is locked.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        organ_builder: Optional[OrganBuilder] = None,
        novelty_threshold: float = 0.3,
        pruning_interval: int = 100,  # Steps between pruning checks
        model: Optional[Any] = None,
        substrate: Optional[Any] = None,
        somatic_map: Optional[Any] = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.organ_builder = organ_builder or OrganBuilder()
        self.novelty_detector = NoveltyDetector(novelty_threshold)
        self._model = model
        self._substrate = substrate
        self._somatic_map = somatic_map
        self._pruning_interval = pruning_interval
        self._step_count = 0
        self._lock = threading.Lock()

        # Statistics
        self._novel_problems_detected = 0
        self._organs_created = 0
        self._organs_dissolved = 0

        # Episode buffer for proactive monitoring
        self._episode_buffer: deque[dict[str, Any]] = deque(maxlen=1000)

        # Callbacks registered by external systems
        self._on_team_created: list[Callable[[Organ], None]] = []
        self._on_team_dissolved: list[Callable[[str], None]] = []

        # Hormonal modulation
        self._growth_rate_multiplier = 1.0  # Modulated by endocrine system

        # Subscribe to spawn requests on EventBus
        self.event_bus.register_callback(CH_SPAWN_REQUEST, self._handle_spawn_request)

        # Subscribe to capability discovery - new capabilities may need new organs
        self.event_bus.register_callback(
            "improvement.capability_found", self._on_capability_found
        )

        logger.info("MorphogenesisCoordinator initialized")

    # =========================================================================
    # Reactive Mode: Direct Problem Handling
    # =========================================================================

    def handle_novel_problem(
        self,
        signature: ProblemSignature,
        name: Optional[str] = None,
    ) -> Optional[Organ]:
        """Handle a novel problem by creating a specialized organ.

        Full morphogenesis pipeline:
        1. Check novelty
        2. Design blueprint
        3. Grow organ (spawn agents)
        4. Register as known
        5. Publish events

        Returns the created Organ, or None if not novel.
        """
        with self._lock:
            # Check novelty
            if not self.novelty_detector.is_novel(signature):
                logger.debug("Problem not novel: %s", signature.signature_id)
                return None

            self._novel_problems_detected += 1

            # Publish novelty detection
            self.event_bus.publish(
                CH_NOVELTY_DETECTED,
                {
                    "signature_id": signature.signature_id,
                    "domain": signature.domain,
                    "complexity": signature.complexity,
                },
            )

            # Design blueprint
            blueprint = self.organ_builder.design_organ(signature, name=name)

            # Grow organ
            organ = self.organ_builder.grow_organ(
                blueprint,
                model=self._model,
                substrate=self._substrate,
            )

            # Register as known
            self.novelty_detector.register_signature(signature)
            self._organs_created += 1

            # Publish creation event
            self.event_bus.publish(
                CH_TEAM_CREATED,
                {
                    "organ_id": organ.organ_id,
                    "name": blueprint.name,
                    "agent_count": len(organ.agents),
                    "topology": blueprint.topology.value,
                },
            )

            # Notify callbacks
            for callback in self._on_team_created:
                try:
                    callback(organ)
                except Exception:
                    logger.exception("Error in team_created callback")

            # Update somatic map with new organ
            if self._somatic_map is not None:
                try:
                    self._somatic_map.register_system(
                        f"organ:{organ.organ_id}",
                        f"Morphogenesis organ ({blueprint.name})",
                        depends_on=["morphogenesis"],
                    )
                except Exception:
                    logger.debug("Could not register organ in somatic map")

            logger.info(
                "Novel problem -> organ '%s' (%d agents)",
                organ.organ_id,
                len(organ.agents),
            )
            return organ

    def force_create_organ(
        self,
        signature: ProblemSignature,
        name: Optional[str] = None,
    ) -> Organ:
        """Force-create an organ without novelty check (for testing)."""
        with self._lock:
            blueprint = self.organ_builder.design_organ(signature, name=name)
            organ = self.organ_builder.grow_organ(
                blueprint, model=self._model, substrate=self._substrate
            )
            self._organs_created += 1

            self.event_bus.publish(
                CH_TEAM_CREATED,
                {
                    "organ_id": organ.organ_id,
                    "name": blueprint.name,
                    "agent_count": len(organ.agents),
                },
            )
            return organ

    def dissolve_organ(self, organ_id: str) -> bool:
        """Dissolve a specific organ."""
        with self._lock:
            result = self.organ_builder.dissolve_organ(
                organ_id, model=self._model, substrate=self._substrate
            )
            if result:
                self._organs_dissolved += 1
                self.event_bus.publish(
                    CH_TEAM_DISSOLVED,
                    {"organ_id": organ_id},
                )
                for callback in self._on_team_dissolved:
                    try:
                        callback(organ_id)
                    except Exception:
                        logger.exception("Error in team_dissolved callback")
                # Update somatic map: organ dissolved
                if self._somatic_map is not None:
                    try:
                        self._somatic_map.heartbeat(
                            f"organ:{organ_id}", health=0.0
                        )
                    except Exception:
                        logger.debug("Could not notify somatic map of organ dissolution")
            return result

    # =========================================================================
    # Step-Based Operation (called by model)
    # =========================================================================

    def step(self) -> None:
        """Periodic maintenance step.

        - Process any buffered episodes
        - Prune underperforming organs
        """
        self._step_count += 1

        # Periodic pruning
        if self._step_count % self._pruning_interval == 0:
            self._prune()

    def _prune(self) -> None:
        """Prune organs that should be dissolved."""
        dissolved = self.organ_builder.prune_organs(
            model=self._model, substrate=self._substrate
        )
        for organ_id in dissolved:
            self._organs_dissolved += 1
            self.event_bus.publish(CH_TEAM_DISSOLVED, {"organ_id": organ_id})
            for callback in self._on_team_dissolved:
                try:
                    callback(organ_id)
                except Exception:
                    logger.exception("Error in team_dissolved callback")

    # =========================================================================
    # EventBus Integration
    # =========================================================================

    def _handle_spawn_request(self, channel: str, message: Any) -> None:
        """Handle spawn requests from other systems via EventBus."""
        if isinstance(message, str):
            import json

            try:
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if not isinstance(message, dict):
            return

        sig = ProblemSignature(
            coordination_level=message.get("coordination_level", 0.5),
            exploration_level=message.get("exploration_level", 0.5),
            complexity=message.get("complexity", 0.5),
            risk_level=message.get("risk_level", 0.3),
            domain=message.get("domain", "general"),
            temporal_pattern=message.get("temporal_pattern", "persistent"),
        )
        name = message.get("name")
        self.handle_novel_problem(sig, name=name)

    def _on_capability_found(self, channel: str, message: Any) -> None:
        """Handle improvement.capability_found events.

        New capabilities may require new organs to exploit them.
        If the capability's context suggests a domain mismatch,
        we consider creating a specialized organ.
        """
        if isinstance(message, str):
            import json

            try:
                message = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                return

        if not isinstance(message, dict):
            return

        context = message.get("context", "general")
        performance_delta = message.get("performance_delta", 0.0)

        # Only consider organ creation for significant capabilities
        if performance_delta >= 0.5:
            sig = ProblemSignature(
                domain=context,
                complexity=min(1.0, performance_delta),
                exploration_level=0.7,
                coordination_level=0.5,
            )
            self.handle_novel_problem(sig, name=f"capability_{context}")

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_team_created(self, callback: Callable[[Organ], None]) -> None:
        """Register callback for when a new organ is created."""
        self._on_team_created.append(callback)

    def on_team_dissolved(self, callback: Callable[[str], None]) -> None:
        """Register callback for when an organ is dissolved."""
        self._on_team_dissolved.append(callback)

    # =========================================================================
    # Hormonal Modulation
    # =========================================================================

    def set_growth_rate(self, multiplier: float) -> None:
        """Modulate growth rate (endocrine integration).

        Cortisol stress -> faster spawning (multiplier > 1.0)
        Serotonin calm -> normal spawning (multiplier = 1.0)
        """
        self._growth_rate_multiplier = max(0.1, min(3.0, multiplier))

    # =========================================================================
    # Queries
    # =========================================================================

    def get_organ(self, organ_id: str) -> Optional[Organ]:
        return self.organ_builder.get_organ(organ_id)

    def get_all_organs(self) -> list[Organ]:
        return self.organ_builder.get_all_organs()

    def get_statistics(self) -> dict[str, Any]:
        return {
            "novel_problems_detected": self._novel_problems_detected,
            "organs_created": self._organs_created,
            "organs_dissolved": self._organs_dissolved,
            "active_organs": self.organ_builder.active_organ_count,
            "known_signatures": self.novelty_detector.known_count,
            "growth_rate_multiplier": self._growth_rate_multiplier,
            "step_count": self._step_count,
            "builder": self.organ_builder.get_statistics(),
        }
