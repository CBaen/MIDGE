"""Triad Registry - Wires all Mae processes with their validator triads.

This is the startup wiring that connects existing systems as formal
validators for every major process. Called once at system initialization
to bring Mae to full Rule of 3 compliance.

Each process gets complementary validators using DIFFERENT approaches:
- STRUCTURAL: Architecture/dependency analysis (SomaticMap)
- BEHAVIORAL: Trust/anomaly detection (HAVEN)
- OPERATIONAL: Runtime health/recovery (AutoHealer)
- PREDICTIVE: Prediction accuracy (WorldModel/ValidatedImagination)
- CONSENSUS: Collective agreement (Quorum/CollectiveDream)
- CAUSAL: Cause-effect reasoning (CausalEngine)
- TEMPORAL: Time-based patterns (TemporalMemory)
- RESOURCE: Resource availability (NutrientFlow/Endocrine)
- FORMAL: Contract/invariant checking (TriadEnforcer)

Odd numbers only. No even counts. No deadlock.
"""

from __future__ import annotations

import logging
from typing import Any

from .triad_auditor import TriadAuditor
from .triad_enforcer import ProcessCriticality, TriadEnforcer, ValidatorType
from .triad_watchdog import TriadWatchdog

logger = logging.getLogger(__name__)


def register_all_triads(
    enforcer: TriadEnforcer,
    watchdog: TriadWatchdog,
    auditor: TriadAuditor,
    systems: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register all Mae processes with their validator triads.

    Args:
        enforcer: The TriadEnforcer instance.
        watchdog: The TriadWatchdog instance.
        auditor: The TriadAuditor instance.
        systems: Optional dict of live system instances for real validation.
                 If None, uses permissive defaults (for testing/startup).

    Returns:
        Registration report with compliance status.
    """
    systems = systems or {}

    # Helper: make a validator function from a system method
    def _make_validator(system_key: str, method: str, default: bool = True):
        """Create a validator function that calls a system method if available."""
        sys = systems.get(system_key)
        if sys and hasattr(sys, method):
            fn = getattr(sys, method)
            return lambda ctx: fn(ctx)
        return lambda ctx: default  # Permissive default until system is wired

    registered = 0
    validators_added = 0

    # =================================================================
    # CRITICAL PROCESSES (5 validators each)
    # =================================================================

    # 1. Self-Modification / Self-Improvement
    enforcer.register_process(
        "self_modification",
        "Self-improvement, capability changes, code evolution",
        ProcessCriticality.CRITICAL,
    )
    for vid, vtype, desc in [
        ("somatic_map", ValidatorType.STRUCTURAL,
         "Blast radius analysis before any modification"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Policy contagion check - is the improvement adversarial?"),
        ("capability_discovery", ValidatorType.OPERATIONAL,
         "Validation pipeline - has the capability been tested?"),
        ("causal_engine", ValidatorType.CAUSAL,
         "Predict downstream consequences of modification"),
        ("triad_enforcer", ValidatorType.FORMAL,
         "Meta-check: does the modification break any triads?"),
    ]:
        enforcer.add_validator(
            "self_modification", vid, vtype,
            _make_validator(vid, "validate_modification"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("improvement.capability_found", "self_modification")
    registered += 1

    # 2. Decision Making
    enforcer.register_process(
        "decision_making",
        "Agent action selection and deliberation",
        ProcessCriticality.CRITICAL,
    )
    for vid, vtype, desc in [
        ("decision_router", ValidatorType.OPERATIONAL,
         "3-tier reflex/habit/prefrontal evaluation"),
        ("collective_dream", ValidatorType.CONSENSUS,
         "Swarm validation of proposed actions"),
        ("world_model", ValidatorType.PREDICTIVE,
         "Predict consequences before acting"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Risk assessment of proposed action"),
        ("causal_engine", ValidatorType.CAUSAL,
         "Causal analysis of action implications"),
    ]:
        enforcer.add_validator(
            "decision_making", vid, vtype,
            _make_validator(vid, "validate_decision"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 3. Healing / Recovery
    enforcer.register_process(
        "healing",
        "Failure detection, isolation, and recovery",
        ProcessCriticality.CRITICAL,
    )
    for vid, vtype, desc in [
        ("auto_healer", ValidatorType.OPERATIONAL,
         "3-phase biological recovery execution"),
        ("somatic_map", ValidatorType.STRUCTURAL,
         "Blast radius of healing actions"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Was the failure adversarial or organic?"),
        ("nutrient_flow", ValidatorType.RESOURCE,
         "Resource availability for recovery"),
        ("causal_engine", ValidatorType.CAUSAL,
         "Root cause analysis before healing"),
    ]:
        enforcer.add_validator(
            "healing", vid, vtype,
            _make_validator(vid, "validate_healing"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("healing.started", "healing")
    watchdog.monitor_channel("healing.complete", "healing")
    registered += 1

    # =================================================================
    # PROTECTED PROCESSES (5 validators each)
    # =================================================================

    # 4. Agent Lifecycle (spawn/hibernate/kill)
    enforcer.register_process(
        "agent_lifecycle",
        "Agent creation, hibernation, and termination",
        ProcessCriticality.PROTECTED,
    )
    for vid, vtype, desc in [
        ("morphogenesis", ValidatorType.OPERATIONAL,
         "Novelty detection and blueprint validation"),
        ("octopus_colony", ValidatorType.CONSENSUS,
         "Colony workload assessment for scaling"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Risk check before spawn/kill decisions"),
        ("substrate", ValidatorType.STRUCTURAL,
         "Network capacity and topology impact"),
        ("endocrine", ValidatorType.RESOURCE,
         "Hormonal state permits growth/pruning"),
    ]:
        enforcer.add_validator(
            "agent_lifecycle", vid, vtype,
            _make_validator(vid, "validate_lifecycle"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("morphogenesis.organ_created", "agent_lifecycle")
    registered += 1

    # 5. Resource Allocation
    enforcer.register_process(
        "resource_allocation",
        "Nutrient distribution and energy management",
        ProcessCriticality.PROTECTED,
    )
    for vid, vtype, desc in [
        ("nutrient_flow", ValidatorType.RESOURCE,
         "Diffusion-based resource distribution"),
        ("endocrine", ValidatorType.BEHAVIORAL,
         "Hormonal modulation of resource priority"),
        ("substrate", ValidatorType.STRUCTURAL,
         "Network topology for distribution paths"),
        ("circadian", ValidatorType.TEMPORAL,
         "Phase-appropriate resource allocation"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Detect resource hoarding/starvation attacks"),
    ]:
        enforcer.add_validator(
            "resource_allocation", vid, vtype,
            _make_validator(vid, "validate_resource"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("substrate.starvation_alert", "resource_allocation")
    registered += 1

    # 6. Learning / Policy Updates
    enforcer.register_process(
        "learning_policy",
        "Policy sharing, updates, and knowledge transfer",
        ProcessCriticality.PROTECTED,
    )
    for vid, vtype, desc in [
        ("frl", ValidatorType.OPERATIONAL,
         "Federated peer-to-peer policy validation"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Policy contagion detection"),
        ("validated_imagination", ValidatorType.PREDICTIVE,
         "Prediction accuracy of policy source"),
        ("causal_engine", ValidatorType.CAUSAL,
         "Causal analysis of policy effects"),
        ("memory_coordinator", ValidatorType.TEMPORAL,
         "Historical performance of similar policies"),
    ]:
        enforcer.add_validator(
            "learning_policy", vid, vtype,
            _make_validator(vid, "validate_policy"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("frl.policy_shared", "learning_policy")
    registered += 1

    # =================================================================
    # STANDARD PROCESSES (3 validators each)
    # =================================================================

    # 7. Temporal Reasoning
    enforcer.register_process(
        "temporal_reasoning",
        "Temporal event chains and pattern prediction",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("temporal_memory", ValidatorType.TEMPORAL,
         "4D event timeline and pattern detection"),
        ("causal_engine", ValidatorType.CAUSAL,
         "Validate temporal causal links"),
        ("world_model", ValidatorType.PREDICTIVE,
         "Prediction accuracy for temporal patterns"),
    ]:
        enforcer.add_validator(
            "temporal_reasoning", vid, vtype,
            _make_validator(vid, "validate_temporal"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 8. Causal Reasoning
    enforcer.register_process(
        "causal_reasoning",
        "Cause-effect inference and counterfactual analysis",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("causal_engine", ValidatorType.CAUSAL,
         "Primary causal graph and inference"),
        ("temporal_memory", ValidatorType.TEMPORAL,
         "Temporal event chains validate causal order"),
        ("world_model", ValidatorType.PREDICTIVE,
         "Interventional predictions test causal claims"),
    ]:
        enforcer.add_validator(
            "causal_reasoning", vid, vtype,
            _make_validator(vid, "validate_causal"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 9. World Model Predictions
    enforcer.register_process(
        "world_model_predictions",
        "Imagination accuracy and prediction validation",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("world_model", ValidatorType.PREDICTIVE,
         "Ensemble prediction with uncertainty"),
        ("validated_imagination", ValidatorType.FORMAL,
         "Historical accuracy tracking per domain"),
        ("collective_dream", ValidatorType.CONSENSUS,
         "Expert consensus on prediction quality"),
    ]:
        enforcer.add_validator(
            "world_model_predictions", vid, vtype,
            _make_validator(vid, "validate_prediction"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 10. Morphogenesis
    enforcer.register_process(
        "morphogenesis",
        "Organ creation, team building, agent specialization",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("morphogenesis_coord", ValidatorType.OPERATIONAL,
         "Novelty detection and blueprint validation"),
        ("collective_dream", ValidatorType.CONSENSUS,
         "Low consensus triggers specialist spawning"),
        ("auto_healer", ValidatorType.OPERATIONAL,
         "Failure triggers replacement spawning"),
    ]:
        enforcer.add_validator(
            "morphogenesis", vid, vtype,
            _make_validator(vid, "validate_morphogenesis"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 11. Endocrine Regulation
    enforcer.register_process(
        "endocrine_regulation",
        "Hormone levels and cascade effects",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("endocrine", ValidatorType.RESOURCE,
         "Hormone release and cascade management"),
        ("circadian", ValidatorType.TEMPORAL,
         "Phase-appropriate hormone modulation"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Stress response monitoring (cortisol/adrenaline)"),
    ]:
        enforcer.add_validator(
            "endocrine_regulation", vid, vtype,
            _make_validator(vid, "validate_endocrine"),
            desc,
        )
        validators_added += 1
    registered += 1

    # 12. Circadian Rhythm
    enforcer.register_process(
        "circadian_rhythm",
        "Sleep/wake cycles and phase transitions",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("circadian", ValidatorType.TEMPORAL,
         "Phase transition timing and activity multipliers"),
        ("endocrine", ValidatorType.RESOURCE,
         "Melatonin/cortisol alignment with phase"),
        ("memory_consolidator", ValidatorType.OPERATIONAL,
         "Consolidation timing during rest phase"),
    ]:
        enforcer.add_validator(
            "circadian_rhythm", vid, vtype,
            _make_validator(vid, "validate_circadian"),
            desc,
        )
        validators_added += 1
    registered += 1

    # =================================================================
    # ALREADY COMPLIANT (register for tracking)
    # =================================================================

    # Threat Detection (already 3+)
    enforcer.register_process(
        "threat_detection",
        "Threat detection and defense response",
        ProcessCriticality.CRITICAL,
    )
    for vid, vtype, desc in [
        ("threat_detector", ValidatorType.BEHAVIORAL,
         "4 bio strategies: porcupine/turtle/lizard/kangaroo"),
        ("input_validator", ValidatorType.FORMAL,
         "Zero-trust input validation and anomaly detection"),
        ("haven", ValidatorType.BEHAVIORAL,
         "Immune system policy contagion detection"),
        ("somatic_map", ValidatorType.STRUCTURAL,
         "Blast radius analysis of threat response"),
        ("auto_healer", ValidatorType.OPERATIONAL,
         "Recovery readiness assessment"),
    ]:
        enforcer.add_validator(
            "threat_detection", vid, vtype,
            _make_validator(vid, "validate_threat"),
            desc,
        )
        validators_added += 1
    watchdog.monitor_channel("defense.threat_detected", "threat_detection")
    registered += 1

    # Communication Routing (already 5)
    enforcer.register_process(
        "communication",
        "Multi-channel agent communication",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("signal_bus", ValidatorType.OPERATIONAL,
         "Electrical signal propagation"),
        ("stigmergy", ValidatorType.RESOURCE,
         "Pheromone trail communication"),
        ("quorum", ValidatorType.CONSENSUS,
         "Chemical concentration consensus"),
    ]:
        enforcer.add_validator(
            "communication", vid, vtype,
            _make_validator(vid, "validate_communication"),
            desc,
        )
        validators_added += 1
    registered += 1

    # Memory Storage (already 5)
    enforcer.register_process(
        "memory_storage",
        "Experience storage and retrieval",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("episodic_memory", ValidatorType.TEMPORAL,
         "Priority-weighted experience storage"),
        ("semantic_retriever", ValidatorType.PREDICTIVE,
         "Embedding-based similarity validation"),
        ("working_memory", ValidatorType.OPERATIONAL,
         "Activation-gated short-term storage"),
    ]:
        enforcer.add_validator(
            "memory_storage", vid, vtype,
            _make_validator(vid, "validate_memory"),
            desc,
        )
        validators_added += 1
    registered += 1

    # Consensus (already 3)
    enforcer.register_process(
        "consensus",
        "Collective decision-making and voting",
        ProcessCriticality.STANDARD,
    )
    for vid, vtype, desc in [
        ("quorum_sensor", ValidatorType.CONSENSUS,
         "Chemical concentration consensus"),
        ("consensus_calc", ValidatorType.FORMAL,
         "Statistical confidence calculation"),
        ("spatial_consensus", ValidatorType.STRUCTURAL,
         "Location-weighted spatial voting"),
    ]:
        enforcer.add_validator(
            "consensus", vid, vtype,
            _make_validator(vid, "validate_consensus"),
            desc,
        )
        validators_added += 1
    registered += 1

    # =================================================================
    # Compliance Report
    # =================================================================

    report = enforcer.check_compliance()
    report["total_registered"] = registered
    report["total_validators_added"] = validators_added

    logger.info(
        "Triad registry complete: %d processes, %d validators, %.0f%% compliant",
        registered,
        validators_added,
        report["compliance_rate"] * 100,
    )

    return report


# =====================================================================
# Deferred Wiring — Replace permissive defaults with real systems
# =====================================================================

# Map: (validator_id, process_id) → (system_key, method_name)
# Only entries where the validator_id and method name differ from
# the convention need explicit mapping. Convention: validator_id is
# the system key, method is "validate_{process_type}".
_VALIDATOR_METHOD_MAP: dict[str, dict[str, str]] = {
    # validator_id → { process_id → method_name }
    "haven": {
        "self_modification": "validate_modification",
        "decision_making": "validate_decision",
        "healing": "validate_healing",
        "agent_lifecycle": "validate_lifecycle",
        "resource_allocation": "validate_threat",
        "learning_policy": "validate_policy",
        "endocrine_regulation": "validate_endocrine",
        "threat_detection": "validate_threat",
    },
}


def wire_triad_systems(enforcer: TriadEnforcer, systems: dict[str, Any]) -> int:
    """Replace permissive default lambdas with real system methods.

    Called after all systems are created (post-Layer 9) to wire
    actual validation logic into the TriadEnforcer. Validators
    registered with permissive defaults in register_all_triads()
    get upgraded to real system method calls.

    Args:
        enforcer: The TriadEnforcer with registered processes.
        systems: Dict mapping system keys to live system instances.

    Returns:
        Number of validators successfully wired.
    """
    wired = 0

    for process_id, triad in enforcer._processes.items():
        for validator in triad.validators:
            vid = validator.validator_id
            sys = systems.get(vid)
            if sys is None:
                continue

            # Look up method name: explicit map first, then convention
            method_map = _VALIDATOR_METHOD_MAP.get(vid, {})
            method_name = method_map.get(process_id)
            if method_name is None:
                # Convention: process "decision_making" → "validate_decision"
                # Take first word of process_id
                method_name = f"validate_{process_id.split('_')[0]}"

            if hasattr(sys, method_name):
                fn = getattr(sys, method_name)
                validator.validate_fn = lambda ctx, _fn=fn: _fn(ctx)
                wired += 1
                logger.debug(
                    "Wired %s.%s → process %s", vid, method_name, process_id,
                )

    logger.info("wire_triad_systems: %d validators wired to real systems", wired)
    return wired
