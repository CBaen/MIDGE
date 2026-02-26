"""Mae Bootstrap - Wakes the organism.

python main.py              # Run with defaults (5 agents, 100 steps)
python main.py --agents 10  # More agents
python main.py --steps 500  # Longer run

33-layer organism bootstrap orchestrator. Each layer group is in
mae_core/bootstrap/. This file is the conductor; the musicians
are in their own modules.
"""

from __future__ import annotations

import argparse
import logging

from mae_core.bootstrap.context import create_context
from mae_core.bootstrap.foundation import bootstrap_foundation
from mae_core.bootstrap.agents import bootstrap_agents
from mae_core.bootstrap.wiring import bootstrap_wiring
from mae_core.bootstrap.patterns import bootstrap_patterns
from mae_core.bootstrap.organs import bootstrap_organs
from mae_core.bootstrap.external import bootstrap_external
from mae_core.bootstrap.market import bootstrap_market
from mae_core.bootstrap.audit import bootstrap_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mae.bootstrap")


def create_mae(
    num_agents: int = 5,
    cycle_length: int = 100,
    persist_dir: str = "data/mae",
) -> tuple:
    """Create and wire a complete Mae organism (33-layer bootstrap).

    Instantiates ALL systems in biological order, injects subsystems
    into agents, registers step hooks, and wires EventBus channels.

    Returns (model, systems_dict) where systems_dict holds references
    to all subsystems for inspection/testing.
    """
    # Load .env file (API keys, provider config)
    from dotenv import load_dotenv
    load_dotenv()

    ctx = create_context(num_agents=num_agents, cycle_length=cycle_length, persist_dir=persist_dir)

    bootstrap_foundation(ctx)    # Layers 1-11: model, bus, shared systems
    bootstrap_agents(ctx)        # Layers 12-13: agents, learning engines
    bootstrap_wiring(ctx)        # Layers 14-21: connections, holons, fractal
    bootstrap_patterns(ctx)      # Layers 22-25: patterns, deep memory, action
    bootstrap_organs(ctx)        # Layers 26-30: organs, organism, lifecycle
    bootstrap_external(ctx)      # Layer 31: external API gateway
    bootstrap_market(ctx)        # Layer 33: market intelligence organ
    bootstrap_audit(ctx)         # Layer 32: triadic bootstrap audit (self-check)

    systems = _build_systems_dict(ctx)

    logger.info("=" * 60)
    logger.info("Mae is fully wired: %d shared systems, %d agents, %d per-agent systems, %d holons, %d connections",
                41, ctx.num_agents, len(ctx.per_agent_systems) * 5, len(ctx.holon_registry.get_all_ids()),
                ctx.connection_registry.get_statistics()["total_connections"])
    logger.info("=" * 60)

    return ctx.model, systems


def _build_systems_dict(ctx) -> dict:
    """Collect all systems from context into the return dict."""
    return {
        # Foundation
        "model": ctx.model,
        "event_bus": ctx.bus,
        # Coordination
        "circadian": ctx.circadian,
        "endocrine": ctx.endocrine,
        # Enforcement
        "enforcer": ctx.enforcer,
        "watchdog": ctx.watchdog,
        "auditor": ctx.auditor,
        "triad_report": ctx.triad_report,
        # Substrate
        "substrate": ctx.substrate,
        "physarum": ctx.physarum,
        # Communication
        "signal_bus": ctx.signal_bus,
        "gnn_communicator": ctx.gnn_comm,
        "stigmergy": ctx.stigmergy,
        "quorum_space": ctx.quorum_space,
        "predictive_field": ctx.predictive_field,
        # Learning
        "knowledge_base": ctx.knowledge_base,
        "transfer_engine": ctx.transfer_engine,
        "maml_learner": ctx.maml_learner,
        "curiosity": ctx.curiosity,
        "haven": ctx.haven,
        "imitation": ctx.imitation,
        # Defense
        "threat_detector": ctx.threat_detector,
        "input_validator": ctx.input_validator,
        "pearl_defense": ctx.pearl_defense,
        # Cognition (shared)
        "shared_world_model": ctx.shared_world_model,
        "collective_dream": ctx.collective_dream,
        "validated_imagination": ctx.validated_imagination,
        "shared_causal_engine": ctx.shared_causal_engine,
        # Emergent
        "auto_healer": ctx.auto_healer,
        "capability_discovery": ctx.capability_discovery,
        "somatic_map": ctx.somatic_map,
        # Morphogenesis
        "morph_coordinator": ctx.morph_coordinator,
        "organ_builder": ctx.organ_builder,
        # Planning
        "temporal_memory": ctx.temporal_memory,
        "worldline_planner": ctx.worldline_planner,
        # Per-agent learning
        "frl_engines": ctx.frl_engines,
        "vdn_engines": ctx.vdn_engines,
        # Agents
        "agents": ctx.agents,
        "per_agent_systems": ctx.per_agent_systems,
        # Holon Protocol
        "holon_registry": ctx.holon_registry,
        # Connection Registry
        "connection_registry": ctx.connection_registry,
        # Witness Notifier (Layer 18b — operational witnessing)
        "witness_notifier": ctx.witness_notifier,
        # Bidirectional Awareness
        "awareness_pulse": ctx.awareness_pulse,
        # Fractal Generator
        "fractal_generator": ctx.fractal_generator,
        # Stem Cell Registry
        "stem_cell_registry": ctx.stem_cell_registry,
        # Deep Memory (Layer 22) — may be None if Qdrant unavailable
        "deep_store": ctx.deep_store,
        "memory_bridge": ctx.memory_bridge,
        "pattern_distiller": ctx.pattern_distiller,
        "experience_narrator": ctx.narrator,
        # Pattern Ecosystem (Layer 23) — may be None if failed
        "pattern_bus": ctx.pattern_bus,
        "pattern_cortex": ctx.pattern_cortex,
        "pattern_consolidator": ctx.pattern_consolidator,
        "attentional_gate": ctx.attentional_gate,
        # Action Environment (Layer 24)
        "task_pool": ctx.task_pool,
        # Fractal ACT (Layer 25)
        "organism_action": ctx.organism_action,
        # Integration Meter (Layer 25b)
        "integration_meter": ctx.integration_meter,
        # Topology Analyzer (Layer 25c)
        "topology_analyzer": ctx.topology_analyzer,
        # Triadic Verifier (Layer 25d)
        "triadic_verifier": ctx.triadic_verifier,
        # Metabolic Systems (Layer 26)
        "digestive_system": ctx.digestive_system,
        "respiratory_system": ctx.respiratory_system,
        "vestibular_system": ctx.vestibular_system,
        "homeostasis": ctx.homeostasis,
        "thermoregulation": ctx.thermoregulation,
        "energy_reserve": ctx.energy_reserve,
        "circulatory_system": ctx.circulatory_system,
        "renal_filter": ctx.renal_filter,
        "microbiome": ctx.microbiome,
        # Social Cognition + Sensory (Layer 27)
        "emotional_system": ctx.emotional_system,
        "theory_of_mind": ctx.theory_of_mind,
        "metacognition": ctx.metacognition,
        "nociception": ctx.nociception,
        "proprioception": ctx.proprioception,
        # Maintenance + Growth (Layer 28)
        "lymphatic_system": ctx.lymphatic_system,
        "senescence": ctx.senescence,
        "boundary_membrane": ctx.boundary_membrane,
        "reproductive_system": ctx.reproductive_system,
        # Organism State + Deep Integration (Layer 29)
        "organism_state": ctx.organism_state,
        # Triage Classifier (Layer 29e — biological urgency triage)
        "triage_classifier": ctx.triage_classifier,
        # Mitosis Monitor (Layer 29a3 — autopoietic production loop)
        "mitosis_monitor": ctx.mitosis_monitor,
        # Lifecycle Steps (Layer 30)
        "inhibition_system": ctx.inhibition_system,
        "goal_manager": ctx.goal_manager,
        "arousal_regulator": ctx.arousal_regulator,
        # External API Gateway (Layer 31)
        "api_gateway": ctx.api_gateway,
        # Market Intelligence (Layer 33) — all tolerate None for graceful degradation
        "sec_edgar_client":     getattr(ctx, "sec_edgar_client", None),
        "price_fetcher":        getattr(ctx, "price_fetcher", None),
        "house_stock_watcher":  getattr(ctx, "house_stock_watcher", None),
        "job_tracker":          getattr(ctx, "job_tracker", None),
        "usa_spending_client":  getattr(ctx, "usa_spending_client", None),
        "sam_gov_client":       getattr(ctx, "sam_gov_client", None),
        "cluster_detector":     getattr(ctx, "cluster_detector", None),
        "politician_tracker":   getattr(ctx, "politician_tracker", None),
        "filing_time_analyzer": getattr(ctx, "filing_time_analyzer", None),
        "contract_predictor":   getattr(ctx, "contract_predictor", None),
        "thompson_sampler":     getattr(ctx, "thompson_sampler", None),
        "convergence_alerter":  getattr(ctx, "convergence_alerter", None),
        "velocity_detector":        getattr(ctx, "velocity_detector", None),
        "correlation_tracker":      getattr(ctx, "correlation_tracker", None),
        "outcome_tracker":          getattr(ctx, "outcome_tracker", None),
        "session_sweep_detector":   getattr(ctx, "session_sweep_detector", None),
        "signal_archive_reader":    getattr(ctx, "signal_archive_reader", None),
        "lag_correlation_analyzer":  getattr(ctx, "lag_correlation_analyzer", None),
        "thompson_calibrator":      getattr(ctx, "thompson_calibrator", None),
        "kelly_position_sizer":     getattr(ctx, "kelly_position_sizer", None),
        # Hypothesis loop (RSI Layer 2)
        "hypothesis_registry":      getattr(ctx, "hypothesis_registry", None),
        "hypothesis_generator":     getattr(ctx, "hypothesis_generator", None),
        "hypothesis_validator":     getattr(ctx, "hypothesis_validator", None),
        "hypothesis_engine":        getattr(ctx, "hypothesis_engine", None),
        "backtest_analyzer":        getattr(ctx, "backtest_analyzer", None),
        # Technical analysis (Trades by Sci indicators)
        "ta_indicators":            getattr(ctx, "ta_indicators", None),
    }


def _register_somatic_systems(somatic_map, systems: dict) -> None:
    """Register all systems with SomaticMap for body awareness.

    Uses register_system() if available, falls back to heartbeat().
    """
    for name, system in systems.items():
        try:
            if hasattr(somatic_map, "register_system"):
                somatic_map.register_system(
                    system_id=name,
                    system_type=type(system).__name__,
                    dependencies=[],
                )
            elif hasattr(somatic_map, "heartbeat"):
                somatic_map.heartbeat(name)
        except Exception:
            logger.debug("Could not register %s with SomaticMap", name)


class RunReport:
    """Collects events during a run and writes a readable markdown report."""

    ACTION_LABELS = {
        "explore": "Explored (searched for new information)",
        "exploit": "Exploited (used what it already knows)",
        "communicate": "Communicated (shared knowledge with others)",
        "rest": "Rested (consolidated memories)",
        "api_call": "Asked the Oracle (external API call)",
        "idle": "Idle (waiting)",
    }

    def __init__(self):
        self.steps: list[list[dict]] = []  # steps[i] = list of agent actions
        self.oracle_calls: list[dict] = []  # {step, agent, provider, question, response, latency}

    def record_step(self, step: int, agents) -> None:
        """Record what every agent did this step."""
        actions = []
        for agent in agents:
            uid = agent.unique_id
            action = getattr(agent, "last_action", None)
            reward = getattr(agent, "last_reward", 0.0)
            pred_err = getattr(agent, "_prediction_error", 0.0)
            role = getattr(agent, "role", "STEM")
            config = getattr(agent, "agent_config", {})

            if isinstance(action, dict):
                action_name = action.get("type", "?")
            elif isinstance(action, str):
                action_name = action
            else:
                action_name = str(action) if action is not None else "idle"

            actions.append({
                "uid": uid, "role": role, "action": action_name,
                "reward": reward, "surprise": pred_err,
                "provider": config.get("preferred_provider", ""),
            })
        self.steps.append(actions)

    def record_oracle(self, step, agent_id, provider, question, response_text, latency_ms, tokens_in, tokens_out):
        """Record an oracle call result."""
        self.oracle_calls.append({
            "step": step, "agent": agent_id, "provider": provider,
            "question": question, "response": response_text,
            "latency_ms": latency_ms, "tokens_in": tokens_in, "tokens_out": tokens_out,
        })

    def write(self, path: str, agents, systems) -> None:
        """Write the full run report as markdown."""
        import datetime
        lines = []
        w = lines.append

        w(f"---")
        w(f"")
        w(f"# Run — {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        w(f"")

        # -- Vital signs --
        w(f"## Vital Signs")
        w(f"")
        circadian = systems.get("circadian")
        endocrine = systems.get("endocrine")
        if circadian and endocrine:
            cs = circadian.get_statistics()
            es = endocrine.get_statistics()
            w(f"| Metric | Value |")
            w(f"|--------|-------|")
            w(f"| Steps run | {len(self.steps)} |")
            w(f"| Agents | {len(agents)} |")
            w(f"| Circadian phase | {cs.get('current_phase', '?')} |")
            w(f"| Stressed | {'Yes' if es.get('is_stressed') else 'No'} |")
            w(f"| Exploration bias | {es.get('exploration_bias', 0):.2f} |")
            w(f"| Trust level | {es.get('trust_level', 0):.2f} |")
            w(f"")

        # -- Voices --
        gw = systems.get("api_gateway")
        if gw:
            gw_stats = gw.get_statistics()
            providers = gw_stats.get("providers", [])
            w(f"## Voices (External API Providers)")
            w(f"")
            if providers:
                w(f"| Provider | Status |")
                w(f"|----------|--------|")
                for p in providers:
                    w(f"| {p} | Connected |")
            else:
                w(f"*No providers connected.*")
            w(f"")
            w(f"Oracle calls: **{gw_stats.get('total_processed', 0)}** processed, "
              f"**{gw_stats.get('total_rejected', 0)}** rejected")
            client = gw_stats.get("client", {})
            if client.get("total_calls", 0) > 0:
                w(f", success rate: **{client.get('success_rate', 0):.0%}**")
            w(f"")

        # -- Agent profiles --
        w(f"## Agent Profiles")
        w(f"")
        for agent in agents:
            uid = agent.unique_id
            role = getattr(agent, "role", "STEM")
            config = getattr(agent, "agent_config", {})
            has_voice = config.get("api_call_enabled", False)
            provider = config.get("preferred_provider", "")
            voice_tag = f" | Voice: **{provider}**" if has_voice else ""

            w(f"### Agent {uid} ({role}){voice_tag}")
            w(f"")

            # Count actions this run
            action_counts: dict[str, int] = {}
            total_reward = 0.0
            for step_actions in self.steps:
                for a in step_actions:
                    if a["uid"] == uid:
                        action_counts[a["action"]] = action_counts.get(a["action"], 0) + 1
                        total_reward += a["reward"]

            if action_counts:
                w(f"| Action | Count | What it means |")
                w(f"|--------|-------|---------------|")
                for action_name, count in sorted(action_counts.items(), key=lambda x: -x[1]):
                    label = self.ACTION_LABELS.get(action_name, action_name)
                    w(f"| {action_name} | {count} | {label} |")
                w(f"")
                w(f"Total reward this run: **{total_reward:.2f}**")

            # Memory
            mem = getattr(agent, "episodic_memory", None)
            if mem is not None:
                try:
                    count = len(mem) if hasattr(mem, "__len__") else getattr(mem, "count", 0)
                    if count > 0:
                        w(f" | Episodic memories: **{count}**")
                except Exception:
                    pass
            w(f"")

        # -- Step-by-step timeline --
        w(f"## Step-by-Step Timeline")
        w(f"")
        w(f"| Step | Agent 1 | Agent 2 | Agent 3 |")
        w(f"|------|---------|---------|---------|")
        for i, step_actions in enumerate(self.steps):
            step_num = i + 1
            cols = []
            for a in step_actions[:3]:  # first 3 agents
                name = a["action"]
                extras = []
                if a["reward"] > 0.1:
                    extras.append(f"+{a['reward']:.1f}")
                if a["surprise"] > 0.1:
                    extras.append(f"surprise")
                if name == "api_call":
                    extras.append("ORACLE")
                detail = f" ({', '.join(extras)})" if extras else ""
                cols.append(f"{name}{detail}")
            # Pad if fewer than 3 agents
            while len(cols) < 3:
                cols.append("")
            w(f"| {step_num} | {cols[0]} | {cols[1]} | {cols[2]} |")
        w(f"")

        # -- Oracle conversations --
        if self.oracle_calls:
            w(f"## What the Oracles Said")
            w(f"")
            for call in self.oracle_calls:
                w(f"### Agent {call['agent']} → {call['provider']} ({call['latency_ms']:.0f}ms)")
                w(f"")
                q = call.get("question", "")
                if q:
                    w(f"**Question:** {q}")
                    w(f"")
                r = call.get("response", "")
                if r:
                    w(f"**Response:** {r}")
                    w(f"")
                w(f"*{call['tokens_in']} tokens in, {call['tokens_out']} tokens out*")
                w(f"")
        else:
            w(f"## Oracle Calls")
            w(f"")
            w(f"*No oracle calls this run.*")
            w(f"")

        # -- Organism stats --
        w(f"## Organism Stats")
        w(f"")
        bus = systems.get("event_bus")
        if bus:
            try:
                bs = bus.get_statistics() if hasattr(bus, "get_statistics") else {}
                w(f"- EventBus: **{bs.get('total_messages', 0)}** messages across "
                  f"**{bs.get('active_channels', 0)}** channels")
            except Exception:
                pass
        pool = systems.get("task_pool")
        if pool:
            try:
                ps = pool.get_statistics() if hasattr(pool, "get_statistics") else {}
                w(f"- Tasks: **{ps.get('total_generated', 0)}** generated, "
                  f"**{ps.get('total_completed', 0)}** completed")
            except Exception:
                pass
        meter = systems.get("integration_meter")
        if meter:
            try:
                ms = meter.get_statistics() if hasattr(meter, "get_statistics") else {}
                phi = ms.get("current_phi", 0.0)
                if phi > 0:
                    w(f"- Integration (phi): **{phi:.3f}**")
            except Exception:
                pass
        w(f"")

        # Append to log file
        import pathlib
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
        logger.info("Run appended to %s", path)


def run(
    num_agents: int = 5,
    num_steps: int = 100,
    num_rounds: int = 1,
    cycle_length: int = 100,
) -> None:
    """Create Mae once and run multiple rounds of simulation.

    State persists between rounds — memories, Q-tables, policies all accumulate.
    Each round runs num_steps steps on the same organism.
    """
    import time as _time

    logger.info("=" * 60)
    logger.info("Mae is waking up...")
    logger.info("=" * 60)

    model, systems = create_mae(
        num_agents=num_agents,
        cycle_length=cycle_length,
    )

    agents = systems["agents"]

    # Deep memory store (may be None if Qdrant unavailable)
    deep_store = systems.get("deep_store")

    # Narrative journal — plain-English story of each step
    from mae_core.backbone.journal_writer import JournalWriter
    from mae_core.backbone.growth_tracker import GrowthTracker
    journal = JournalWriter(output_dir="data/mae", deep_store=deep_store)
    growth = GrowthTracker(output_dir="data/mae", deep_store=deep_store)

    # Report collector (reset per round)
    report = RunReport()

    # Collect data every step (quiet — no terminal spam)
    def narrator_hook():
        step = int(model.time)
        if step > 0:
            report.record_step(step, agents)

    def journal_hook():
        step = int(model.time)
        if step > 0:
            journal.record_step(step, agents, systems)
        # Reset per-step inhibition flags so they don't bleed into next step
        for agent in agents:
            agent._inhibited_this_step = False
            agent._last_inhibit_reason = ""
            agent._inhibit_veto_sources = []

    model.add_step_hook(narrator_hook)
    model.add_step_hook(journal_hook)

    # Hook into gateway to capture oracle responses
    gw = systems.get("api_gateway")
    if gw is not None:
        _orig_process = gw._process_request

        def _capturing_process(request):
            _orig_process(request)
            # Check if the task was completed (response available)
            pool = systems.get("task_pool")
            if pool is not None:
                task = pool._tasks.get(request.request_id)
                if task is not None:
                    spec = getattr(task, "external_spec", None)
                    if spec is not None and spec.response is not None:
                        report.record_oracle(
                            step=int(model.time),
                            agent_id=request.agent_id,
                            provider=request.provider,
                            question=spec.payload.get("question", ""),
                            response_text=spec.response.get("text", ""),
                            latency_ms=spec.response.get("latency_ms", 0),
                            tokens_in=spec.response.get("tokens_in", 0),
                            tokens_out=spec.response.get("tokens_out", 0),
                        )

        gw._process_request = _capturing_process

    # -- Run rounds --
    round_times: list[float] = []
    interrupted = False

    try:
        for r in range(1, num_rounds + 1):
            if num_rounds > 1:
                logger.info("=" * 60)
                logger.info("ROUND %d / %d", r, num_rounds)
                if round_times:
                    avg = sum(round_times) / len(round_times)
                    remaining = (num_rounds - r + 1) * avg
                    logger.info("Avg round: %.1fs | Est remaining: %.0fs (%.1fh)",
                                avg, remaining, remaining / 3600)
                logger.info("=" * 60)

            report = RunReport()
            journal.begin_run(
                step_count=num_steps, agent_count=num_agents, round_num=r,
            )

            round_start = _time.time()
            try:
                model.run(num_steps)
            except KeyboardInterrupt:
                logger.info("Interrupted by user at round %d", r)
                interrupted = True
            except Exception:
                logger.exception("Round %d failed — continuing to next round", r)

            round_elapsed = _time.time() - round_start
            round_times.append(round_elapsed)

            stats = model.get_system_stats()
            steps_done = stats["current_step"]

            # Per-round reports
            report_path = "data/mae/run-log.md"
            report.write(report_path, agents, systems)

            journal.end_run(steps_completed=steps_done)
            growth.record_run(agents, systems, report, steps_done, round_num=r)

            logger.info(
                "Round %d done: %d steps in %.1fs",
                r, steps_done, round_elapsed,
            )

            if interrupted:
                break

    finally:
        total_time = sum(round_times)
        logger.info("=" * 60)
        logger.info(
            "Mae completed %d rounds in %.1fs (%.1fh). Avg %.1fs/round.",
            len(round_times), total_time, total_time / 3600,
            total_time / max(1, len(round_times)),
        )
        logger.info("=" * 60)

        model.shutdown()
        logger.info("Mae is resting.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mae - Mycelial Agent Engine")
    parser.add_argument("--agents", type=int, default=5, help="Number of agents (min 3)")
    parser.add_argument("--steps", type=int, default=100, help="Simulation steps")
    parser.add_argument("--cycle", type=int, default=100, help="Circadian cycle length")
    parser.add_argument("--rounds", type=int, default=1, help="Number of rounds (state persists between rounds)")
    args = parser.parse_args()

    run(
        num_agents=args.agents,
        num_steps=args.steps,
        num_rounds=args.rounds,
        cycle_length=args.cycle,
    )


if __name__ == "__main__":
    main()
