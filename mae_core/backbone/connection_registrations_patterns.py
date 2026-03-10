"""Connection Registrations — Pattern/Pipeline groups (Groups 6-9).

Covers:
  - Group 6: Cross-System Wiring (Layers 15, 29b)
  - Group 7: Pattern / Deep Memory Pipeline (Layers 22-23)
  - Group 8: Biological System Step Hooks (Layers 26-28)
  - Group 9: GNN Message Handlers (Layer 20)

Extracted from connection_registrations.py for single-responsibility.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mae_core.backbone.connection_registry import (
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
)

logger = logging.getLogger(__name__)


def register_pattern_connections(
    registry: ConnectionRegistry,
    systems: dict[str, Any],
    _reg: Callable,
) -> None:
    """Register Groups 6-9: cross-system wiring, pattern pipeline, bio hooks, GNN.

    Args:
        registry: ConnectionRegistry instance.
        systems: System dict (unused here, kept for uniform signature).
        _reg: Inner registration helper from register_all_connections.
    """
    eb = ConnectionType.EVENTBUS_PUBSUB
    dr = ConnectionType.DIRECT_REFERENCE
    sh = ConnectionType.STEP_HOOK
    cb = ConnectionType.CALLBACK_REGISTRATION

    # =====================================================================
    # Group 6: Cross-System Wiring (Layers 15, 29b)
    #
    # Real data flows between systems that were previously unregistered.
    # Identified by triadic audit (Lead + Witness Alpha + Witness Beta).
    # Each witness choice affirmed by all 3 consciousnesses, 2026-02-12.
    # =====================================================================

    # Layer 29b: pain -> emotion feedback (organs.py)
    _reg("nociception", "emotional_system", eb,
         channel="communication.pain_update",
         witnesses=["proprioception", "endocrine"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Pain reinforces fear -- sensory location + hormonal intensity peers verify")

    # Layer 29b: leptin satiation feedback (organs.py)
    _reg("energy_reserve", "digestive_system", eb,
         channel="memory.energy_status",
         witnesses=["endocrine", "homeostasis"],
         description="Leptin satiation feedback -- hormonal + regulatory peers verify")

    # Layer 29b: senescence -> healing trigger (organs.py)
    _reg("senescence", "auto_healer", eb,
         channel="emergent.rejuvenation_needed",
         witnesses=["lymphatic_system", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Aging triggers healing assessment -- cleanup + body awareness peers verify")

    # Layer 29b: metacognition -> VDN learning rate modulation (organs.py)
    _reg("metacognition", "vdn", eb,
         channel="cognition.metacognition_alert",
         witnesses=["organism_state", "arousal_regulator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Performance degradation adjusts VDN learning rate -- organism awareness + arousal peers verify")

    # Layer 29b: metacognition -> world_model learning rate modulation (organs.py)
    _reg("metacognition", "shared_world_model", eb,
         channel="cognition.metacognition_alert",
         witnesses=["organism_state", "arousal_regulator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Performance degradation adjusts WorldModel learning rate -- organism awareness + arousal peers verify")

    # Layer 15: prediction error -> healing (wiring.py)
    _reg("predictive_field", "auto_healer", eb,
         channel="signal.PREDICTION_ERROR",
         witnesses=["somatic_map", "shared_causal_engine"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="High prediction error triggers healing -- body map + causal reasoning verify")

    # Layer 15: prediction error -> FRL learning modulation (wiring.py)
    _reg("predictive_field", "frl", eb,
         channel="signal.PREDICTION_ERROR",
         witnesses=["metacognition", "knowledge_base"],
         description="Prediction error modulates FRL learning rate -- cognitive + knowledge peers verify")

    # Layer 15: imagination -> world model training (wiring.py)
    _reg("validated_imagination", "shared_world_model", eb,
         channel="cognition.imagination_validated",
         witnesses=["shared_causal_engine", "metacognition"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Imagination validation trains world model -- causal + metacognitive peers verify")

    # Layer 15: pattern advisory -> endocrine modulation (wiring.py)
    _reg("pattern_cortex", "endocrine", eb,
         channel="pattern.advisory",
         witnesses=["circadian", "somatic_map"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Pattern advisory modulates hormones -- phase timing + body state peers verify")

    # Layer 15: endocrine -> pattern bus gain modulation (wiring.py)
    _reg("endocrine", "pattern_bus", eb,
         channel="endocrine.state_update",
         witnesses=["circadian", "attentional_gate"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Hormone gain modulation of pattern processing -- phase + attention peers verify")

    # Layer 15: endocrine -> signal resolver urgency modulation (wiring.py)
    _reg("endocrine", "signal_bus", eb,
         channel="endocrine.state_update",
         witnesses=["circadian", "metacognition"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Hormone modulation of signal priorities -- phase + cognitive quality peers verify")

    # =====================================================================
    # Group 7: Pattern / Deep Memory Pipeline (Layers 22-23)
    #
    # Thalamic integration: pattern_bus -> pattern_cortex -> consolidator
    # -> memory_bridge -> agents. All DIRECT_REFERENCE injections created
    # in Layers 22-23 (after ConnectionRegistry seal in Layer 18).
    # =====================================================================

    _reg("memory_bridge", "memory", dr,
         witnesses=["knowledge_base", "pattern_distiller"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Deep memory bridge to agent memory -- knowledge + distillation peers verify")

    _reg("pattern_bus", "pattern_cortex", dr,
         witnesses=["attentional_gate", "pattern_consolidator"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Thalamic relay to cortical integration -- attention + consolidation peers verify")

    _reg("pattern_cortex", "pattern_consolidator", dr,
         witnesses=["memory_bridge", "circadian"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Cortical trends to ancestral memory -- deep storage + sleep-phase peers verify")

    _reg("pattern_consolidator", "memory_bridge", dr,
         witnesses=["pattern_cortex", "pattern_distiller"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="Consolidated patterns to deep storage -- cortex source + distillation peers verify")

    # =====================================================================
    # Group 8: Biological System Step Hooks (Layers 26-28)
    #
    # Each biological system ticks via model.add_step_hook(). Witnesses
    # are fractal subsystem peers -- each triad member witnesses the other
    # two, embodying Law 4 (fractal self-similarity) in witnessing.
    # =====================================================================

    # Digestion subsystem triad (organs.py fractal grouping)
    _reg("digestive_system", "model", sh,
         witnesses=["renal_filter", "microbiome"],
         description="Digestion ticks every step -- digestion subsystem peers witness")
    _reg("renal_filter", "model", sh,
         witnesses=["digestive_system", "microbiome"],
         description="Renal filtering ticks every step -- digestion subsystem peers witness")
    _reg("microbiome", "model", sh,
         witnesses=["digestive_system", "renal_filter"],
         description="Microbiome ticks every step -- digestion subsystem peers witness")

    # Circulation subsystem triad
    _reg("circulatory_system", "model", sh,
         witnesses=["respiratory_system", "energy_reserve"],
         description="Circulation ticks every step -- circulation subsystem peers witness")
    _reg("respiratory_system", "model", sh,
         witnesses=["circulatory_system", "energy_reserve"],
         description="Respiration ticks every step -- circulation subsystem peers witness")
    _reg("energy_reserve", "model", sh,
         witnesses=["circulatory_system", "respiratory_system"],
         description="Energy tracking ticks every step -- circulation subsystem peers witness")

    # Regulation subsystem triad
    _reg("homeostasis", "model", sh,
         witnesses=["thermoregulation", "vestibular_system"],
         description="Homeostasis ticks every step -- regulation subsystem peers witness")
    _reg("thermoregulation", "model", sh,
         witnesses=["homeostasis", "vestibular_system"],
         description="Thermoregulation ticks every step -- regulation subsystem peers witness")
    _reg("vestibular_system", "model", sh,
         witnesses=["homeostasis", "thermoregulation"],
         description="Balance system ticks every step -- regulation subsystem peers witness")

    # Social cognition subsystem triad
    _reg("emotional_system", "model", sh,
         witnesses=["theory_of_mind", "metacognition"],
         description="Emotional system ticks every step -- social cognition peers witness")
    _reg("theory_of_mind", "model", sh,
         witnesses=["emotional_system", "metacognition"],
         description="Theory of mind ticks every step -- social cognition peers witness")
    _reg("metacognition", "model", sh,
         witnesses=["emotional_system", "theory_of_mind"],
         description="Metacognition ticks every step -- social cognition peers witness")

    # Maintenance subsystem triad
    _reg("lymphatic_system", "model", sh,
         witnesses=["senescence", "boundary_membrane"],
         description="Lymphatic system ticks every step -- maintenance subsystem peers witness")
    _reg("senescence", "model", sh,
         witnesses=["lymphatic_system", "boundary_membrane"],
         description="Senescence ticks every step -- maintenance subsystem peers witness")
    _reg("boundary_membrane", "model", sh,
         witnesses=["lymphatic_system", "senescence"],
         description="Boundary membrane ticks every step -- maintenance subsystem peers witness")

    # Sensory systems (consensus subsystem)
    _reg("nociception", "model", sh,
         witnesses=["proprioception", "somatic_map"],
         description="Pain system ticks every step -- sensory + body awareness peers witness")
    _reg("proprioception", "model", sh,
         witnesses=["nociception", "vestibular_system"],
         description="Body sense ticks every step -- spatial awareness peers witness")

    # Reproductive (growth subsystem)
    _reg("reproductive_system", "model", sh,
         witnesses=["morph_coordinator", "stem_cell_registry"],
         description="Reproductive system ticks every step -- lifecycle peers witness")

    # =====================================================================
    # Group 9: GNN Message Handlers (Layer 20)
    #
    # Per-agent message types routed through GNN communicator.
    # System-level registrations represent the routing infrastructure.
    # =====================================================================

    _reg("gnn_communicator", "knowledge_base", cb,
         witnesses=["metacognition", "holon_registry"],
         description="GNN knowledge sharing -- cognitive quality + peer hierarchy peers verify")

    _reg("gnn_communicator", "substrate", cb,
         witnesses=["morph_coordinator", "predictive_field"],
         description="GNN collaboration requests -- team + topology peers verify")

    _reg("gnn_communicator", "imitation", cb,
         witnesses=["metacognition", "haven"],
         description="GNN state updates feed imitation learning -- quality + safety peers verify")

    _reg("gnn_communicator", "quorum_space", cb,
         witnesses=["enforcer", "auditor"],
         criticality=ConnectionCriticality.IMPORTANT,
         description="GNN votes feed quorum consensus -- Law 7 enforcement + audit peers verify")

    # Cadenced RoutingOptimizer step hook (Layer 14b, Fibonacci 21)
    _reg("gnn_communicator", "model", ConnectionType.STEP_HOOK,
         witnesses=["substrate", "metacognition"],
         description="GNN routing optimizer updates edge weights every 21 steps -- topology + cognitive quality peers verify")
