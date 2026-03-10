"""Organism state aggregator - Mae's hypothalamus.

Biological basis: The hypothalamus integrates signals from every major body
system (temperature, hunger, thirst, fatigue, pain, hormones) into a unified
internal state representation. This allows the organism to answer "how am I
doing right now?" with a single coherent snapshot, rather than consulting
dozens of independent sensors.

Mae's OrganismState serves the same purpose:
- Subscribes to EventBus channels from all 18 biological systems
- Aggregates them into a single body_state dict
- Provides reflex overrides for emergency conditions (pain, starvation, etc.)
- Enriches the decision context for the advisory routing cascade
- Tracks action outcomes to build an overall vitality metric

Connection points:
- All 18 biological systems publish to EventBus; OrganismState listens
- Agents call get_body_state() for holistic awareness
- DecisionRouter calls get_reflex_override() before the advisory cascade
- DecisionRouter calls get_decision_context() to enrich routing context
- Agents call report_action_outcome() to close the feedback loop

Law compliance:
- Law 3 (Holon Protocol): know_self - the organism knows its own state
- Law 6 (Autopoietic Closure): signals produce the state that drives the
  actions that produce the signals
- Law 8 (Consciousness): Property 1 - integration (combining all signals
  into a unified representation); Property 3 - self-reference (the organism
  monitoring itself)

This file is a thin composition hub. Implementation lives in:
  - organism_state_subscriptions.py  (OrganismStateSubscriptionsMixin)
  - organism_state_outputs.py        (OrganismStateOutputsMixin)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from mae_core.backbone.event_bus import EventBus
from mae_core.coordination.organism_state_outputs import (
    CH_ORGANISM_ACTION_OUTCOME,
    OrganismStateOutputsMixin,
    _OUTCOME_WINDOW,
    _VITALITY_ALPHA,
)
from mae_core.coordination.organism_state_subscriptions import OrganismStateSubscriptionsMixin

logger = logging.getLogger(__name__)


class OrganismState(OrganismStateSubscriptionsMixin, OrganismStateOutputsMixin):
    """Mae's hypothalamus - unified body state aggregator.

    Subscribes to EventBus channels from all 18 biological systems and
    aggregates their signals into a single coherent body state. Provides
    reflex overrides for emergencies and enriched decision context.

    Args:
        event_bus: The EventBus instance for pub/sub integration.
    """

    CH_ORGANISM_ACTION_OUTCOME = CH_ORGANISM_ACTION_OUTCOME

    # Tunable threshold for homeostasis-driven rest reflex (Priority 6).
    _HOMEOSTASIS_URGENCY_THRESHOLD: float = 0.7

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.event_bus = event_bus
        self._step_count: int = 0

        # ===================================================================
        # Body state — all metrics with healthy baselines
        # ===================================================================

        # Metabolic
        self._energy_level: float = 1.0
        self._energy_critical: bool = False
        self._oxygen_level: float = 1.0
        self._toxin_load: float = 0.0
        self._circulation_adequate: bool = True
        self._digestion_active: bool = False

        # Emotional / Social
        self._emotional_valence: float = 0.3       # CALM baseline
        self._emotional_arousal: float = 0.2        # CALM baseline
        self._dominant_emotion: str = "CALM"
        self._metacognition_score: float = 0.5
        self._social_confidence: float = 0.3

        # Pain / Stability
        self._pain_load: float = 0.0
        self._stability: float = 1.0
        self._temperature_zone: str = "optimal"

        # Maintenance
        self._organism_age: float = 0.0
        self._waste_capacity_used: float = 0.0
        self._membrane_permeability: float = 0.5
        self._microbiome_diversity: float = 1.0

        # Regulation
        self._homeostasis_deviation: float = 0.0
        self._population_healthy: bool = True

        # ===================================================================
        # Action outcome tracking
        # ===================================================================
        self._recent_outcomes: deque[float] = deque(maxlen=_OUTCOME_WINDOW)
        self._vitality: float = 0.5     # EMA of rewards

        # ===================================================================
        # Subscribe to all 18 biological system channels
        # ===================================================================
        if self.event_bus is not None:
            self._subscribe_all()

        logger.info("OrganismState initialized (hypothalamus aggregator)")

    # =====================================================================
    # Core Step
    # =====================================================================

    def step(self, current_step: int = 0) -> None:
        """Advance one organism state cycle.

        Resets per-step accumulators (homeostasis deviation is re-populated
        by incoming corrections each step). The actual state updates happen
        reactively via EventBus callbacks.
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # Reset homeostasis deviation so it is freshly computed from
        # this step's correction messages
        self._homeostasis_deviation = 0.0


__all__ = [
    "OrganismState",
    "CH_ORGANISM_ACTION_OUTCOME",
    "OrganismStateSubscriptionsMixin",
    "OrganismStateOutputsMixin",
]
