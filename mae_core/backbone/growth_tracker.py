"""MIDGE Growth Tracker

Appends one compact metrics row per run to data/midge/growth-tracker.md.
Lets you see learning trajectory at a glance without digging through
multiple log files.

Usage in main.py:
    tracker = GrowthTracker(output_dir="data/midge")
    # ... after run completes ...
    tracker.record_run(agents, systems, run_report)
"""
from __future__ import annotations

import datetime
import logging
import pathlib
from typing import Any

logger = logging.getLogger("midge.growth")


class GrowthTracker:
    """Appends one metrics row per run to a persistent markdown table."""

    def __init__(self, output_dir: str = "data/midge", deep_store: Any = None) -> None:
        self._path = pathlib.Path(output_dir) / "growth-tracker.md"
        self._deep_store = deep_store

    def record_run(
        self,
        agents,
        systems: dict,
        run_report: Any,
        steps_completed: int,
        round_num: int = 0,
    ) -> None:
        """Collect metrics from the completed run and append a row."""
        is_new = not self._path.exists() or self._path.stat().st_size == 0

        # -- Collect metrics --
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        agent_count = len(list(agents))

        # Rewards this run (from run_report steps data)
        run_reward = 0.0
        action_counts: dict[str, int] = {}
        if hasattr(run_report, "steps"):
            for step_actions in run_report.steps:
                for a in step_actions:
                    run_reward += a.get("reward", 0.0)
                    act = a.get("action", "idle")
                    action_counts[act] = action_counts.get(act, 0) + 1

        # Cumulative reward (all time, from agent state)
        cumulative_reward = 0.0
        total_memories = 0
        total_policies = 0
        for agent in agents:
            cumulative_reward += float(getattr(agent, "cumulative_reward", 0.0) or 0.0)

            # Episodic memory count
            mem = getattr(agent, "episodic_memory", None)
            if mem is not None:
                try:
                    total_memories += len(mem) if hasattr(mem, "__len__") else getattr(mem, "count", 0)
                except Exception:
                    pass

            # Semantic policy count
            sem = getattr(agent, "semantic_retriever", None)
            if sem is not None:
                try:
                    total_policies += getattr(sem, "n_indexed", 0)
                except Exception:
                    pass

        # Oracle calls
        oracle_calls = len(run_report.oracle_calls) if hasattr(run_report, "oracle_calls") else 0

        # Phi (integration)
        phi = self._read_phi(systems)

        # Action distribution string (compact)
        total_actions = sum(action_counts.values()) or 1
        abbrev = {
            "explore": "Xpl", "exploit": "Xpt", "communicate": "Com",
            "rest": "Rst", "api_call": "Orc", "idle": "Idl",
        }
        action_parts = []
        for act in ["explore", "exploit", "communicate", "rest", "api_call", "idle"]:
            count = action_counts.get(act, 0)
            if count > 0:
                pct = int(count / total_actions * 100)
                action_parts.append(f"{abbrev[act]}:{pct}%")
        action_dist = " ".join(action_parts) if action_parts else "-"

        # Self-sufficiency: % of agent-steps that didn't need oracle
        agent_steps = steps_completed * agent_count
        oracle_pct = int(oracle_calls / max(agent_steps, 1) * 100)
        self_sufficiency = f"{100 - oracle_pct}%"

        # -- Write --
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("# MIDGE Growth Tracker\n\n")
                f.write("One row per run. Watch the numbers change over time.\n\n")
                f.write("| When | Steps | Agents | Run Reward | Lifetime Reward "
                        "| Memories | Policies | Oracle Calls | Self-Sufficient "
                        "| Phi | Actions |\n")
                f.write("|------|-------|--------|------------|----------------"
                        "|----------|----------|--------------|----------------"
                        "|-----|----------|\n")

            f.write(
                f"| {now} "
                f"| {steps_completed} "
                f"| {agent_count} "
                f"| {run_reward:+.2f} "
                f"| {cumulative_reward:.1f} "
                f"| {total_memories} "
                f"| {total_policies} "
                f"| {oracle_calls} "
                f"| {self_sufficiency} "
                f"| {phi:.3f} "
                f"| {action_dist} "
                f"|\n"
            )

        # Store run metrics in Qdrant midge_meta collection
        if self._deep_store is not None:
            try:
                summary_text = (
                    f"MIDGE run round {round_num}: {steps_completed} steps, "
                    f"{agent_count} agents. "
                    f"Run reward {run_reward:+.2f}, lifetime {cumulative_reward:.1f}. "
                    f"{total_memories} memories, {total_policies} policies. "
                    f"{oracle_calls} oracle calls ({self_sufficiency} self-sufficient). "
                    f"Phi {phi:.3f}. Actions: {action_dist}."
                )
                self._deep_store.store_point(
                    "midge_meta",
                    text=summary_text,
                    payload={
                        "type": "run_metrics",
                        "describes_collection": "midge_narrative",
                        "round": round_num,
                        "steps": steps_completed,
                        "agents": agent_count,
                        "run_reward": round(run_reward, 4),
                        "cumulative_reward": round(cumulative_reward, 2),
                        "memories": total_memories,
                        "policies": total_policies,
                        "oracle_calls": oracle_calls,
                        "phi": round(phi, 4),
                        "action_dist": action_dist,
                        "timestamp": now,
                    },
                )
            except Exception:
                logger.debug("Growth: Qdrant storage failed", exc_info=True)

        logger.info("Growth tracked: %s", self._path)

    def _read_phi(self, systems: dict) -> float:
        meter = systems.get("integration_meter")
        if meter:
            try:
                # If no measurement exists yet (cadence not reached), force one
                if getattr(meter, "_last_report", None) is None:
                    if hasattr(meter, "_compute_and_publish"):
                        meter._compute_and_publish()
                stats = meter.get_statistics()
                return float(stats.get("current_phi", 0.0) or 0.0)
            except Exception:
                pass
        return 0.0
