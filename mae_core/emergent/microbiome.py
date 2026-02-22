"""Microbiome - Mae's symbiotic processing ecosystem.

Biological basis: The gut microbiome contains trillions of symbiotic
organisms. They are not the body, but the body depends on them. They
decompose food, produce vitamins (B12, K), train the immune system,
modulate mood via the gut-brain axis, and compete for resources in a
dynamic ecosystem. Diversity is health; monoculture is disease (dysbiosis).

Mae's Microbiome is the equivalent:
- PATTERN DECOMPOSERS break complex patterns into digestible parts
- ANOMALY DETECTORS flag unusual inputs (immune training)
- SIGNAL AMPLIFIERS boost weak but important signals
- NOISE FILTERS remove noise before it reaches higher processing
- NUTRIENT SYNTHESIZERS produce derived insights from raw data

The strains compete for resources (fitness). Active strains grow; idle
strains shrink. Diversity (Shannon index) is health. When one strain
dominates, dysbiosis occurs and the system becomes fragile.

Connection points:
- Inputs from any system that processes data (learning, memory, signals)
- EventBus publishes microbiome status and dysbiosis alerts
- LymphaticSystem may listen for waste from dead strains
- SenescenceManager may track microbiome fitness as organism health

Law compliance:
- Law 8, Property 2 (Differentiation): diverse specialized strains
  with distinct functions. Not a monolith.
- Law 8, Property 7 (Competition/Selection): strains compete for
  resources, fittest grow, idle shrink. Natural selection in silico.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels
CH_MICROBIOME_STATUS = "emergent.microbiome_status"
CH_DYSBIOSIS = "emergent.microbiome_imbalanced"

# Specialization types
SPECIALIZATIONS = frozenset({
    "pattern_decomposer",
    "anomaly_detector",
    "signal_amplifier",
    "noise_filter",
    "nutrient_synthesizer",
})


@dataclass
class MicrobialStrain:
    """A single strain in the microbiome ecosystem."""

    name: str
    specialization: str  # one of SPECIALIZATIONS
    population: float = 1.0
    fitness: float = 0.5

    @property
    def is_alive(self) -> bool:
        return self.population > 0.01


def _default_strains() -> list[MicrobialStrain]:
    """Create the 5 default strains — one of each specialization."""
    return [
        MicrobialStrain(name="decomposer_alpha", specialization="pattern_decomposer"),
        MicrobialStrain(name="detector_beta", specialization="anomaly_detector"),
        MicrobialStrain(name="amplifier_gamma", specialization="signal_amplifier"),
        MicrobialStrain(name="filter_delta", specialization="noise_filter"),
        MicrobialStrain(name="synthesizer_epsilon", specialization="nutrient_synthesizer"),
    ]


class Microbiome:
    """Mae's symbiotic input processing ecosystem.

    Routes inputs to specialist strains for processing. Strains compete
    for resources — active strains grow, idle strains shrink. Diversity
    (Shannon index) indicates ecosystem health. Low diversity triggers
    a dysbiosis alert.

    Like the gut microbiome: you want many different species, each doing
    their job. A monoculture of E. coli is food poisoning, not health.
    """

    CH_MICROBIOME_STATUS = CH_MICROBIOME_STATUS
    CH_DYSBIOSIS = CH_DYSBIOSIS

    def __init__(self, event_bus: Any = None, **kwargs: Any) -> None:
        self._bus = event_bus

        # The ecosystem
        self._strains: list[MicrobialStrain] = _default_strains()

        # Ecosystem metrics
        self._diversity_score: float = self._compute_diversity()
        self._symbiosis_health: float = 1.0
        self._max_population: float = 20.0  # Carrying capacity

        # Processing counters per specialization
        self._process_counts: dict[str, int] = {s: 0 for s in SPECIALIZATIONS}
        self._total_processed: int = 0

        logger.info("Microbiome initialized (%d strains, diversity=%.2f)",
                     len(self._strains), self._diversity_score)

    # =========================================================================
    # Input Processing
    # =========================================================================

    def process_input(self, input_type: str, input_data: dict) -> dict:
        """Route an input to the appropriate specialist strain.

        Each specialization processes differently:
        - pattern_decomposer: splits complex patterns into parts
        - anomaly_detector: flags if the input is unusual
        - signal_amplifier: boosts weak signals
        - noise_filter: strips noise
        - nutrient_synthesizer: derives secondary insights

        Returns a dict with the processed result and strain info.
        """
        # Find the specialist strain
        specialist = self._find_specialist(input_type)
        if specialist is None:
            return {"processed": False, "reason": "no_specialist", "input_type": input_type}

        # Process based on specialization
        result = self._apply_specialization(specialist, input_data)

        # Reward the active strain
        specialist.fitness = min(1.0, specialist.fitness + 0.01)
        self._process_counts[specialist.specialization] = (
            self._process_counts.get(specialist.specialization, 0) + 1
        )
        self._total_processed += 1

        return {
            "processed": True,
            "strain": specialist.name,
            "specialization": specialist.specialization,
            "result": result,
        }

    def _find_specialist(self, input_type: str) -> MicrobialStrain | None:
        """Find the best strain for a given input type.

        Maps input types to specializations:
        - "pattern"/"complex" -> pattern_decomposer
        - "anomaly"/"unusual" -> anomaly_detector
        - "weak_signal"/"faint" -> signal_amplifier
        - "noisy"/"raw" -> noise_filter
        - "data"/"raw_data" -> nutrient_synthesizer
        - Default: strain with highest fitness
        """
        type_map = {
            "pattern": "pattern_decomposer",
            "complex": "pattern_decomposer",
            "anomaly": "anomaly_detector",
            "unusual": "anomaly_detector",
            "weak_signal": "signal_amplifier",
            "faint": "signal_amplifier",
            "noisy": "noise_filter",
            "raw": "noise_filter",
            "data": "nutrient_synthesizer",
            "raw_data": "nutrient_synthesizer",
        }

        target_spec = type_map.get(input_type)

        if target_spec:
            candidates = [s for s in self._strains
                          if s.specialization == target_spec and s.is_alive]
            if candidates:
                return max(candidates, key=lambda s: s.fitness)

        # Fallback: highest fitness alive strain
        alive = [s for s in self._strains if s.is_alive]
        return max(alive, key=lambda s: s.fitness) if alive else None

    def _apply_specialization(
        self, strain: MicrobialStrain, input_data: dict,
    ) -> dict:
        """Apply the strain's specialization to the input data."""
        spec = strain.specialization

        if spec == "pattern_decomposer":
            # Split input into parts
            parts = list(input_data.keys()) if input_data else ["empty"]
            return {"decomposed_parts": parts, "part_count": len(parts)}

        elif spec == "anomaly_detector":
            # Flag unusual inputs (heuristic: more than 5 keys or values > 100)
            num_keys = len(input_data)
            has_large_values = any(
                isinstance(v, (int, float)) and v > 100
                for v in input_data.values()
            )
            is_anomalous = num_keys > 5 or has_large_values
            return {"is_anomalous": is_anomalous, "key_count": num_keys}

        elif spec == "signal_amplifier":
            # Boost numeric values by strain fitness
            amplified = {}
            for k, v in input_data.items():
                if isinstance(v, (int, float)):
                    amplified[k] = v * (1.0 + strain.fitness)
                else:
                    amplified[k] = v
            return {"amplified": amplified, "boost_factor": 1.0 + strain.fitness}

        elif spec == "noise_filter":
            # Keep only non-None, non-empty values
            filtered = {
                k: v for k, v in input_data.items()
                if v is not None and v != "" and v != 0
            }
            return {"filtered": filtered, "removed_count": len(input_data) - len(filtered)}

        elif spec == "nutrient_synthesizer":
            # Derive summary statistics
            numeric_vals = [v for v in input_data.values() if isinstance(v, (int, float))]
            return {
                "total_fields": len(input_data),
                "numeric_count": len(numeric_vals),
                "numeric_mean": (
                    sum(numeric_vals) / len(numeric_vals) if numeric_vals else 0.0
                ),
                "insight": "synthesized",
            }

        return {"raw": input_data}

    # =========================================================================
    # Population Dynamics
    # =========================================================================

    def _evolve_populations(self) -> None:
        """Evolve strain populations based on fitness.

        Active strains (recently used) grow. Idle strains shrink.
        Total population is capped at carrying capacity.
        """
        total_fitness = sum(s.fitness for s in self._strains if s.is_alive)
        if total_fitness <= 0:
            return

        # Idle strains lose fitness
        for strain in self._strains:
            if self._process_counts.get(strain.specialization, 0) == 0:
                strain.fitness = max(0.0, strain.fitness - 0.02)
            # Population proportional to fitness share
            strain.population = (strain.fitness / total_fitness) * self._max_population

        # Update symbiosis health based on diversity
        self._symbiosis_health = min(1.0, self._diversity_score / 1.5)

    def _compute_diversity(self) -> float:
        """Shannon diversity index: -sum(p_i * log(p_i)).

        Higher is healthier. Maximum for 5 equal strains is log(5) ~ 1.609.
        Returns 0.0 if only one strain or no living strains.
        """
        alive = [s for s in self._strains if s.is_alive]
        if len(alive) <= 1:
            return 0.0

        total_pop = sum(s.population for s in alive)
        if total_pop <= 0:
            return 0.0

        entropy = 0.0
        for strain in alive:
            p = strain.population / total_pop
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    # =========================================================================
    # Strain Management
    # =========================================================================

    def introduce_strain(self, name: str, specialization: str) -> bool:
        """Introduce a new strain (like taking probiotics).

        Returns True if the strain was added successfully.
        """
        if specialization not in SPECIALIZATIONS:
            logger.warning("Microbiome: unknown specialization %r", specialization)
            return False

        # Check for duplicate names
        if any(s.name == name for s in self._strains):
            logger.warning("Microbiome: strain %r already exists", name)
            return False

        new_strain = MicrobialStrain(
            name=name,
            specialization=specialization,
            population=1.0,
            fitness=0.5,
        )
        self._strains.append(new_strain)
        logger.info("Microbiome: introduced strain %s (%s)", name, specialization)
        return True

    def get_dominant_strain(self) -> MicrobialStrain | None:
        """Get the strain with the highest population."""
        alive = [s for s in self._strains if s.is_alive]
        if not alive:
            return None
        return max(alive, key=lambda s: s.population)

    # =========================================================================
    # Step
    # =========================================================================

    def step(self, current_step: int) -> None:
        """Per-step update: evolve populations, compute diversity, check health.

        If diversity drops below 0.5, publish dysbiosis alert (one strain
        is dominating the ecosystem).
        """
        self._evolve_populations()
        self._diversity_score = self._compute_diversity()

        # Dysbiosis detection
        if self._diversity_score < 0.5 and self._bus:
            dominant = self.get_dominant_strain()
            self._bus.publish(CH_DYSBIOSIS, {
                "diversity": self._diversity_score,
                "dominant_strain": dominant.name if dominant else "none",
                "dominant_specialization": dominant.specialization if dominant else "none",
                "alive_strains": sum(1 for s in self._strains if s.is_alive),
                "step": current_step,
            })
            logger.warning(
                "Microbiome DYSBIOSIS: diversity=%.2f, dominant=%s",
                self._diversity_score,
                dominant.name if dominant else "none",
            )

        # Publish status
        if self._bus:
            dominant = self.get_dominant_strain()
            self._bus.publish(CH_MICROBIOME_STATUS, {
                "diversity": self._diversity_score,
                "total_population": sum(s.population for s in self._strains),
                "dominant_strain": dominant.name if dominant else "none",
                "symbiosis_health": self._symbiosis_health,
                "alive_strains": sum(1 for s in self._strains if s.is_alive),
                "total_processed": self._total_processed,
                "step": current_step,
            })

        # Reset per-step process counts (so idle detection works next step)
        self._process_counts = {s: 0 for s in SPECIALIZATIONS}

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        dominant = self.get_dominant_strain()
        return {
            "total_strains": len(self._strains),
            "alive_strains": sum(1 for s in self._strains if s.is_alive),
            "diversity_score": self._diversity_score,
            "symbiosis_health": self._symbiosis_health,
            "total_population": sum(s.population for s in self._strains),
            "max_population": self._max_population,
            "dominant_strain": dominant.name if dominant else "none",
            "total_processed": self._total_processed,
            "strain_details": [
                {
                    "name": s.name,
                    "specialization": s.specialization,
                    "population": s.population,
                    "fitness": s.fitness,
                    "alive": s.is_alive,
                }
                for s in self._strains
            ],
        }

    # =========================================================================
    # Tier 2 Persistence
    # =========================================================================

    def serialize(self) -> dict:
        return {
            "diversity_score": self._diversity_score,
            "symbiosis_health": self._symbiosis_health,
            "max_population": self._max_population,
            "total_processed": self._total_processed,
            "strains": [
                {
                    "name": s.name,
                    "specialization": s.specialization,
                    "population": s.population,
                    "fitness": s.fitness,
                }
                for s in self._strains
            ],
        }

    def restore(self, data: dict) -> None:
        self._diversity_score = data.get("diversity_score", 0.0)
        self._symbiosis_health = data.get("symbiosis_health", 1.0)
        self._max_population = data.get("max_population", 20.0)
        self._total_processed = data.get("total_processed", 0)
        self._strains = [
            MicrobialStrain(
                name=s["name"],
                specialization=s["specialization"],
                population=s.get("population", 1.0),
                fitness=s.get("fitness", 0.5),
            )
            for s in data.get("strains", [])
        ]
        if not self._strains:
            self._strains = _default_strains()
