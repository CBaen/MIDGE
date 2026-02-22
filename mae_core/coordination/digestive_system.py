"""Digestive system - input triage, energy budgeting, decomposition.

Biological analogy: The gastrointestinal tract receives food, triages
it by nutritional value vs. energy cost, breaks complex inputs into
simpler absorbable components, and rejects what cannot be processed.

Mae's digestive system:
- Ingests input packets (from agents, external sources, or other systems)
- Triages by nutritional_value / energy_cost ratio (best value first)
- Decomposes complex inputs into simpler NutrientPackets
- Enforces an energy budget — processing costs energy, which regenerates
- Publishes digestion results on EventBus for downstream consumers

Law compliance:
- Law 6 (Autopoietic Closure): Energy regeneration sustains processing capacity
- Law 8 (Consciousness): (2) differentiation via triage, (7) competition/selection

EventBus channel: coordination.digestion_complete
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from mae_core.backbone.event_bus import EventBus

logger = logging.getLogger(__name__)

# EventBus channel
CH_DIGESTION_COMPLETE = "coordination.digestion_complete"


@dataclass
class NutrientPacket:
    """A single unit of input to be digested.

    Attributes:
        source: Where the input came from (agent ID, system name, etc.)
        content: The actual data payload.
        energy_cost: How much energy is required to process this packet.
        nutritional_value: How much value this packet provides once processed.
        decomposed: Whether this packet has already been broken down.
        timestamp: The step at which this packet was ingested.
    """

    source: str
    content: dict
    energy_cost: float = 1.0
    nutritional_value: float = 1.0
    decomposed: bool = False
    timestamp: int = 0


@dataclass
class EnergyBudget:
    """Tracks available processing energy.

    Attributes:
        total_capacity: Maximum energy the system can hold.
        current_energy: Energy available right now.
        regen_rate: Energy regenerated per step.
    """

    total_capacity: float = 100.0
    current_energy: float = 100.0
    regen_rate: float = 5.0


class DigestiveSystem:
    """Mae's input triage and energy budgeting system.

    Receives raw inputs, triages them by value-to-cost ratio,
    decomposes complex inputs, and enforces energy constraints.
    Items that cannot be processed (insufficient energy) remain
    in the queue for the next step.
    """

    CH_DIGESTION_COMPLETE = CH_DIGESTION_COMPLETE

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        energy_budget: Optional[EnergyBudget] = None,
    ) -> None:
        self.event_bus = event_bus
        self._budget = energy_budget or EnergyBudget()
        self._intake_queue: list[NutrientPacket] = []
        self._processed_count: int = 0
        self._rejected_count: int = 0
        self._total_energy_spent: float = 0.0
        self._step_count: int = 0

        logger.info(
            "DigestiveSystem initialized: capacity=%.1f, regen=%.1f/step",
            self._budget.total_capacity,
            self._budget.regen_rate,
        )

    # =========================================================================
    # Ingestion
    # =========================================================================

    def ingest(
        self,
        source: str,
        content: dict,
        energy_cost: float = 1.0,
        nutritional_value: float = 1.0,
    ) -> NutrientPacket:
        """Add an input to the intake queue for processing.

        Args:
            source: Origin identifier.
            content: Data payload.
            energy_cost: Energy required to process.
            nutritional_value: Value provided once processed.

        Returns:
            The created NutrientPacket.
        """
        packet = NutrientPacket(
            source=source,
            content=content,
            energy_cost=energy_cost,
            nutritional_value=nutritional_value,
            timestamp=self._step_count,
        )
        self._intake_queue.append(packet)
        return packet

    # =========================================================================
    # Triage (Law 8: competition/selection)
    # =========================================================================

    def _triage(self) -> None:
        """Sort the intake queue by nutritional_value / energy_cost ratio.

        Best-value items (highest ratio) are processed first. This implements
        biological triage: the gut prioritises easily-digestible, high-value
        nutrients over hard-to-process, low-value ones.
        """
        self._intake_queue.sort(
            key=lambda p: (
                p.nutritional_value / p.energy_cost if p.energy_cost > 0 else float("inf")
            ),
            reverse=True,
        )

    # =========================================================================
    # Decomposition (Law 8: differentiation)
    # =========================================================================

    def _digest(self, packet: NutrientPacket) -> list[NutrientPacket]:
        """Break a complex input into simpler NutrientPackets.

        Complex is defined as: content has more than 5 keys, or any value
        is itself a dict (nested structure). Simple packets pass through
        unchanged.

        Args:
            packet: The packet to decompose.

        Returns:
            List of NutrientPackets (one element if already simple).
        """
        content = packet.content

        # Determine complexity
        has_nested = any(isinstance(v, dict) for v in content.values())
        is_complex = len(content) > 5 or has_nested

        if not is_complex or packet.decomposed:
            # Already simple — mark decomposed and return as-is
            packet.decomposed = True
            return [packet]

        # Decompose: split each key-value pair into its own packet
        sub_packets: list[NutrientPacket] = []
        num_keys = max(len(content), 1)
        per_key_cost = packet.energy_cost / num_keys
        per_key_value = packet.nutritional_value / num_keys

        for key, value in content.items():
            sub_content: dict
            if isinstance(value, dict):
                sub_content = value
            else:
                sub_content = {key: value}

            sub_packets.append(
                NutrientPacket(
                    source=packet.source,
                    content=sub_content,
                    energy_cost=per_key_cost,
                    nutritional_value=per_key_value,
                    decomposed=True,
                    timestamp=packet.timestamp,
                )
            )

        return sub_packets

    # =========================================================================
    # Energy Management
    # =========================================================================

    def get_energy_remaining(self) -> float:
        """Return current energy level."""
        return self._budget.current_energy

    def can_afford(self, cost: float) -> bool:
        """Check whether the budget can cover the given cost."""
        return self._budget.current_energy >= cost

    def spend_energy(self, amount: float) -> bool:
        """Deduct energy from the budget.

        Args:
            amount: Energy to spend.

        Returns:
            True if the spend was successful, False if insufficient energy.
        """
        if amount < 0:
            return False
        if self._budget.current_energy < amount:
            return False
        self._budget.current_energy -= amount
        self._total_energy_spent += amount
        return True

    # =========================================================================
    # Step (Law 6: autopoietic closure via energy regeneration)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Run one processing cycle.

        1. Regenerate energy (capped at total_capacity).
        2. Triage the intake queue.
        3. Process items until energy is exhausted.
        4. Publish digestion summary.

        Items that could not be processed remain in the queue.
        """
        self._step_count = current_step if current_step else self._step_count + 1

        # 1. Regenerate energy
        self._budget.current_energy = min(
            self._budget.total_capacity,
            self._budget.current_energy + self._budget.regen_rate,
        )

        # 2. Triage
        self._triage()

        # 3. Process
        processed_this_step = 0
        rejected_this_step = 0
        remaining: list[NutrientPacket] = []

        for packet in self._intake_queue:
            # Decompose first
            sub_packets = self._digest(packet)

            all_processed = True
            for sub in sub_packets:
                if self.can_afford(sub.energy_cost):
                    self.spend_energy(sub.energy_cost)
                    processed_this_step += 1
                    self._processed_count += 1
                else:
                    # Cannot afford — keep for next step
                    remaining.append(sub)
                    all_processed = False
                    rejected_this_step += 1
                    self._rejected_count += 1

        self._intake_queue = remaining

        # 4. Publish summary
        if self.event_bus is not None:
            self.event_bus.publish(
                CH_DIGESTION_COMPLETE,
                {
                    "processed_count": processed_this_step,
                    "rejected_count": rejected_this_step,
                    "energy_remaining": self._budget.current_energy,
                    "queue_size": len(self._intake_queue),
                },
            )

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return digestive system statistics."""
        return {
            "energy_remaining": self._budget.current_energy,
            "energy_capacity": self._budget.total_capacity,
            "regen_rate": self._budget.regen_rate,
            "queue_size": len(self._intake_queue),
            "processed_count": self._processed_count,
            "rejected_count": self._rejected_count,
            "total_energy_spent": self._total_energy_spent,
            "step_count": self._step_count,
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize state for Tier 2 persistence."""
        return {
            "budget": {
                "total_capacity": self._budget.total_capacity,
                "current_energy": self._budget.current_energy,
                "regen_rate": self._budget.regen_rate,
            },
            "intake_queue": [
                {
                    "source": p.source,
                    "content": p.content,
                    "energy_cost": p.energy_cost,
                    "nutritional_value": p.nutritional_value,
                    "decomposed": p.decomposed,
                    "timestamp": p.timestamp,
                }
                for p in self._intake_queue
            ],
            "processed_count": self._processed_count,
            "rejected_count": self._rejected_count,
            "total_energy_spent": self._total_energy_spent,
            "step_count": self._step_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from serialized data."""
        if "budget" in data:
            b = data["budget"]
            self._budget.total_capacity = b.get("total_capacity", self._budget.total_capacity)
            self._budget.current_energy = b.get("current_energy", self._budget.current_energy)
            self._budget.regen_rate = b.get("regen_rate", self._budget.regen_rate)

        if "intake_queue" in data:
            self._intake_queue = [
                NutrientPacket(
                    source=p["source"],
                    content=p["content"],
                    energy_cost=p.get("energy_cost", 1.0),
                    nutritional_value=p.get("nutritional_value", 1.0),
                    decomposed=p.get("decomposed", False),
                    timestamp=p.get("timestamp", 0),
                )
                for p in data["intake_queue"]
            ]

        self._processed_count = data.get("processed_count", self._processed_count)
        self._rejected_count = data.get("rejected_count", self._rejected_count)
        self._total_energy_spent = data.get("total_energy_spent", self._total_energy_spent)
        self._step_count = data.get("step_count", self._step_count)
