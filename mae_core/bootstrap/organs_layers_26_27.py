"""Bootstrap Layers 26-27: Metabolic systems and social cognition.

Layer 26: DigestiveSystem, RespiratorySystem, VestibularSystem,
          HomeostasisRegulator, ThermoregulationSystem, EnergyReserve,
          CirculatorySystem, RenalFilter, Microbiome + step hooks.

Layer 27: EmotionalSystem, TheoryOfMind, MetacognitionMonitor,
          NociceptionSystem, ProprioceptionSystem + step hooks +
          persistence restore + Tier 2 wiring.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness."""
    for name, system in systems.items():
        try:
            if hasattr(somatic_map, "register_system"):
                somatic_map.register_system(
                    system_id=name,
                    description=type(system).__name__,
                    depends_on=[],
                )
            elif hasattr(somatic_map, "heartbeat"):
                somatic_map.heartbeat(name)
        except Exception:
            logger.debug("Could not register %s with SomaticMap", name)


def _bootstrap_layers_26_27(ctx: SimpleNamespace) -> None:
    """Create metabolic systems (Layer 26) and social cognition (Layer 27)."""
    from mae_core.coordination.emotional_system import EmotionalSystem
    from mae_core.coordination.homeostasis import HomeostasisRegulator
    from mae_core.coordination.thermoregulation import ThermoregulationSystem
    from mae_core.coordination.digestive_system import DigestiveSystem
    from mae_core.coordination.respiratory_system import RespiratorySystem
    from mae_core.coordination.vestibular_system import VestibularSystem
    from mae_core.cognition.theory_of_mind import TheoryOfMind
    from mae_core.cognition.metacognition import MetacognitionMonitor
    from mae_core.communication.nociception import NociceptionSystem
    from mae_core.defense.renal_filter import RenalFilter
    from mae_core.memory.energy_reserve import EnergyReserve
    from mae_core.substrate.circulatory_system import CirculatorySystem
    from mae_core.emergent.proprioception import ProprioceptionSystem
    from mae_core.emergent.microbiome import Microbiome

    # =================================================================
    # Layer 26: Metabolic Systems (digestion, circulation, regulation)
    # =================================================================
    ctx.digestive_system = DigestiveSystem(event_bus=ctx.bus)
    ctx.respiratory_system = RespiratorySystem(event_bus=ctx.bus)
    ctx.vestibular_system = VestibularSystem(event_bus=ctx.bus)
    ctx.homeostasis = HomeostasisRegulator(event_bus=ctx.bus)
    ctx.thermoregulation = ThermoregulationSystem(event_bus=ctx.bus)
    ctx.energy_reserve = EnergyReserve(event_bus=ctx.bus)
    ctx.circulatory_system = CirculatorySystem(event_bus=ctx.bus, substrate=ctx.substrate)
    ctx.renal_filter = RenalFilter(event_bus=ctx.bus)
    ctx.microbiome = Microbiome(event_bus=ctx.bus)

    # Step hooks for metabolic systems
    ctx.model.add_step_hook(lambda s=ctx.digestive_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.respiratory_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.vestibular_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.homeostasis: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.thermoregulation: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.energy_reserve: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.circulatory_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.renal_filter: s.step(current_step=int(ctx.model.time)))

    # Wire microbiome: feed BEFORE step() so _process_counts are populated
    # when _evolve_populations checks idle status (step resets counts at end)
    _micro = ctx.microbiome
    _micro_types = ["pattern", "anomaly", "weak_signal", "noisy", "data"]

    def _feed_microbiome(channel, data, input_type="pattern"):
        try:
            if isinstance(data, dict):
                _micro.process_input(input_type, data)
            else:
                _micro.process_input(input_type, {"raw": data})
        except Exception:
            pass

    # Event-driven feeding from high-frequency channels
    ctx.bus.register_callback("signal.PREDICTION_ERROR", lambda ch, d: _feed_microbiome(ch, d, "anomaly"))
    ctx.bus.register_callback("external.response_received", lambda ch, d: _feed_microbiome(ch, d, "data"))

    # Step-driven: feed all specializations BEFORE microbiome.step() each step.
    # Must run before step() because step() resets _process_counts at the end.
    def _microbiome_step_feed(step):
        try:
            for input_type in _micro_types:
                _micro.process_input(input_type, {"step": step, "source": "organism_rhythm"})
        except Exception:
            pass
    ctx.model.add_step_hook(lambda s=None: _microbiome_step_feed(int(ctx.model.time)))
    # Microbiome step AFTER feed hook — checks _process_counts, then resets
    ctx.model.add_step_hook(lambda s=ctx.microbiome: s.step(current_step=int(ctx.model.time)))

    logger.info(
        "Layer 26 - Metabolic Systems: 9 systems created (digestion, respiration, vestibular, "
        "homeostasis, thermoregulation, energy, circulation, renal, microbiome)"
    )

    # =================================================================
    # Layer 27: Social Cognition + Sensory Extension
    # =================================================================
    ctx.emotional_system = EmotionalSystem(event_bus=ctx.bus)
    ctx.theory_of_mind = TheoryOfMind(event_bus=ctx.bus)
    ctx.metacognition = MetacognitionMonitor(event_bus=ctx.bus)
    ctx.nociception = NociceptionSystem(event_bus=ctx.bus, somatic_map=ctx.somatic_map)
    ctx.proprioception = ProprioceptionSystem(
        event_bus=ctx.bus, somatic_map=ctx.somatic_map, fractal_generator=ctx.fractal_generator,
    )

    # Step hooks for social cognition + sensory
    ctx.model.add_step_hook(lambda s=ctx.emotional_system: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.theory_of_mind: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.metacognition: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.nociception: s.step(current_step=int(ctx.model.time)))
    ctx.model.add_step_hook(lambda s=ctx.proprioception: s.step(current_step=int(ctx.model.time)))

    # Wire metacognition into Tier 2 persistence (created here at Layer 27,
    # after _tier2_refs was built at Layer 12 in agents.py)
    if hasattr(ctx.model, "_tier2_refs") and "shared_systems" in ctx.model._tier2_refs:
        ctx.model._tier2_refs["shared_systems"]["metacognition"] = ctx.metacognition

    # Restore metacognition from prior run (must happen here, after creation)
    metacog_meta = ctx.model.load_subsystem_metadata("subsystem:shared:metacognition")
    if metacog_meta:
        ctx.metacognition.restore(metacog_meta)
        logger.info("Layer 27 - Metacognition: restored %d decisions from prior run",
                     len(ctx.metacognition._decision_history))

    # Wire theory_of_mind into Tier 2 persistence
    if hasattr(ctx.model, "_tier2_refs") and "shared_systems" in ctx.model._tier2_refs:
        ctx.model._tier2_refs["shared_systems"]["theory_of_mind"] = ctx.theory_of_mind

    # Restore theory_of_mind from prior run
    tom_meta = ctx.model.load_subsystem_metadata("subsystem:shared:theory_of_mind")
    if tom_meta:
        ctx.theory_of_mind.restore(tom_meta)
        logger.info("Layer 27 - TheoryOfMind: restored %d agent models from prior run",
                     len(ctx.theory_of_mind._agent_models))

    logger.info(
        "Layer 27 - Social Cognition + Sensory: 5 systems created "
        "(emotions, theory-of-mind, metacognition, nociception, proprioception)"
    )
