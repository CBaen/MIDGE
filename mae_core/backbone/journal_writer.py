"""Mae Narrative Journal

Writes a human-readable story of what Mae does each step.
No technical jargon. Every action in plain English.

Output files:
  data/mae/mae-journal.md  — latest run only (overwritten each run)
  data/mae/journal-log.md  — all runs accumulated (append-only)

Usage in main.py:
    journal = JournalWriter()
    journal.begin_run(step_count=num_steps, agent_count=num_agents)
    model.add_step_hook(lambda: journal.record_step(int(model.time), agents, systems))
    ...
    journal.end_run(steps_completed=N)
"""
from __future__ import annotations

import datetime
import logging
import pathlib
from typing import Any

logger = logging.getLogger("mae.journal")


class JournalWriter:
    """Narrates Mae's internal experience step by step in plain English."""

    ACTION_DESC: dict[str, str] = {
        "explore":    "Explored (searched for new information)",
        "exploit":    "Exploited existing knowledge",
        "communicate": "Communicated (shared knowledge with peers)",
        "rest":       "Rested (consolidated memories)",
        "api_call":   "Asked the oracle (external AI guidance)",
        "idle":       "Idle (no signal this step)",
        "inhibited":  "Paused (body said wait)",
    }

    # (threshold, label) — first match wins
    SURPRISE_LABELS: list[tuple[float, str]] = [
        (0.20, "Low surprise — familiar territory"),
        (0.40, "Mild surprise — slight deviation"),
        (0.60, "Moderate surprise — something new"),
        (0.80, "High surprise — unexpected situation"),
        (1.00, "Extreme surprise — completely unknown"),
    ]

    def __init__(
        self,
        output_dir: str = "data/mae",
        compact: bool = True,
        deep_store: Any = None,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._journal_path = self._output_dir / "mae-journal.md"
        self._log_path = self._output_dir / "journal-log.md"
        self._run_lines: list[str] = []
        self._run_start = ""
        self._compact = compact
        self._prev_body: dict[str, str] = {}  # agent_id -> last body description
        self._deep_store = deep_store
        self._qdrant_buffer: list[dict[str, Any]] = []
        self._qdrant_flush_interval = 50
        self._round_num = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def begin_run(
        self, step_count: int, agent_count: int, round_num: int = 0,
    ) -> None:
        """Open a new run — clears the current journal and writes the header."""
        self._run_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self._run_lines = []
        self._qdrant_buffer = []
        self._round_num = round_num
        w = self._run_lines.append

        round_tag = f" (Round {round_num})" if round_num > 0 else ""
        w(f"# Mae Run Journal — {self._run_start}{round_tag}")
        w(f"")
        w(f"*{agent_count} agents, up to {step_count} steps.*")
        w(f"")
        w(f"---")
        w(f"")

        self._write_file(self._journal_path, overwrite=True)
        logger.info("Journal started: %s", self._journal_path)

    def record_step(self, step: int, agents, systems: dict) -> None:
        """Read agent + organism state and append one step of narrative."""
        # --- Organism context ---
        phase = self._read_phase(systems)
        endocrine = self._read_endocrine(systems)
        phi = self._read_phi(systems)
        bus_stats = self._read_bus(systems)
        pool_stats = self._read_pool(systems)

        lines: list[str] = []
        w = lines.append

        # Step header
        explore_pct = int(endocrine.get("exploration_bias", 0.5) * 100)
        trust = self._describe_trust(endocrine.get("trust_level", 0.5))
        agent_list = list(agents)
        w(f"## Step {step} | {phase} | Exploration: {explore_pct}% | Trust: {trust}")
        w(f"")
        w(f"*{self._describe_organism(endocrine, phi, len(agent_list))}*")
        w(f"")

        # Gather per-agent data
        action_counts: dict[str, int] = {}
        total_reward = 0.0
        agent_records: list[dict] = []

        for agent in agent_list:
            uid = str(getattr(agent, "unique_id", "?"))
            role = getattr(agent, "role", "STEM")
            config = getattr(agent, "agent_config", {})
            provider = config.get("preferred_provider", "") if isinstance(config, dict) else ""

            action = self._normalize_action(getattr(agent, "last_action", None))
            reward = float(getattr(agent, "last_reward", 0.0) or 0.0)
            pred_error = float(getattr(agent, "_prediction_error", 0.0) or 0.0)
            body_state = getattr(agent, "_body_state", {})
            sensed_markers = getattr(agent, "_sensed_markers", {})
            inhibited = bool(getattr(agent, "_inhibited_this_step", False))
            inhibit_reason = getattr(agent, "_last_inhibit_reason", "")
            veto_sources = getattr(agent, "_inhibit_veto_sources", [])
            body_desc = self._describe_body(body_state)

            # Notable = something genuinely interesting happened
            # Routine high surprise (1-5) is normal once markers accumulate;
            # the body system decides what matters by pausing.
            notable = (
                inhibited
                or action == "api_call"
                or abs(reward) >= 0.5
            )

            agent_records.append({
                "uid": uid, "role": role, "provider": provider,
                "action": action, "reward": reward, "pred_error": pred_error,
                "body_desc": body_desc, "sensed_markers": sensed_markers,
                "inhibited": inhibited, "inhibit_reason": inhibit_reason,
                "veto_sources": veto_sources, "notable": notable,
            })

            action_counts[action] = action_counts.get(action, 0) + 1
            total_reward += reward

        # Write agent narratives
        if self._compact:
            self._write_compact_agents(w, agent_records)
        else:
            self._write_verbose_agents(w, agent_records)

        # Step summary
        parts = [f"{count} {act}" for act, count in sorted(action_counts.items())]
        summary = ", ".join(parts) if parts else "no actions"
        total_str = f"+{total_reward:.2f}" if total_reward >= 0 else f"{total_reward:.2f}"
        msgs = bus_stats.get("deliver_count", bus_stats.get("publish_count", 0))
        dropped = bus_stats.get("dropped_count", 0)
        tasks_done = pool_stats.get("total_completed", 0)

        w(f"*Step {step} summary: {summary}. "
          f"Net reward: {total_str}. "
          f"Messages: {msgs} sent, {dropped} dropped. "
          f"Tasks completed: {tasks_done}.*")
        w(f"")
        w(f"---")
        w(f"")

        self._run_lines.extend(lines)
        # Write the live journal after every step so it's always readable mid-run
        self._write_file(self._journal_path, overwrite=True)

        # Buffer step for Qdrant batch storage
        if self._deep_store is not None:
            narrative_text = "\n".join(lines)
            top_action = max(action_counts, key=action_counts.get) if action_counts else "idle"
            self._qdrant_buffer.append({
                "text": narrative_text,
                "payload": {
                    "type": "step_narrative",
                    "step": step,
                    "round": self._round_num,
                    "circadian_phase": phase,
                    "total_reward": round(total_reward, 4),
                    "reward_sign": "positive" if total_reward > 0 else "negative" if total_reward < 0 else "zero",
                    "top_action": top_action,
                    "agent_count": len(agent_list),
                    "notable_count": sum(1 for r in agent_records if r["notable"]),
                },
            })
            if len(self._qdrant_buffer) >= self._qdrant_flush_interval:
                self._flush_qdrant_buffer()

    def end_run(self, steps_completed: int) -> None:
        """Write footer, flush Qdrant buffer, and optionally append to log file."""
        end_time = datetime.datetime.now().strftime("%H:%M:%S")
        self._run_lines.append(
            f"*Run ended {end_time} — {steps_completed} steps completed.*"
        )
        self._run_lines.append(f"")
        # Flush final journal state (latest run, always written for human reading)
        self._write_file(self._journal_path, overwrite=True)

        # Flush remaining Qdrant buffer
        qdrant_ok = self._flush_qdrant_buffer()

        # Only append to journal-log.md when Qdrant is unavailable (fallback)
        if not qdrant_ok:
            self._write_file(self._log_path, overwrite=False)
            logger.info("Journal log (file fallback): %s", self._log_path)

        logger.info("Journal complete: %s (%d steps, qdrant=%s)",
                     self._journal_path, steps_completed, qdrant_ok)

    def _flush_qdrant_buffer(self) -> bool:
        """Batch-flush buffered step narratives to Qdrant mae_narrative collection.

        Returns True if Qdrant stored successfully, False otherwise.
        """
        if not self._qdrant_buffer or self._deep_store is None:
            return self._deep_store is not None

        try:
            result = self._deep_store.store_points_batch(
                "mae_narrative", self._qdrant_buffer,
            )
            stored = result.get("stored", 0)
            failed = result.get("failed", 0)
            if stored > 0:
                logger.debug(
                    "Journal: flushed %d step narratives to Qdrant (%d failed)",
                    stored, failed,
                )
                self._qdrant_buffer = []
                return True
            # All failed — Qdrant might be down
            return False
        except Exception:
            logger.debug("Journal: Qdrant flush failed", exc_info=True)
            return False

    # ── Agent rendering modes ─────────────────────────────────────────────────

    def _write_compact_agents(self, w, records: list[dict]) -> None:
        """Group quiet agents by action; expand only notable ones."""
        notable = [r for r in records if r["notable"]]
        quiet = [r for r in records if not r["notable"]]

        # Notable agents get full detail
        for r in notable:
            voice = f" | Voice: {r['provider'].capitalize()}" if r["provider"] else ""
            w(f"### Agent {r['uid']} ({r['role']}{voice})")
            sense_desc = self._describe_markers(r["sensed_markers"])
            if sense_desc:
                w(f"- **Sensed:** {sense_desc}")
            w(f"- **Feeling:** {self._describe_surprise(r['pred_error'])}")
            if r["inhibited"]:
                w(f"- **PAUSED:** {self._describe_inhibition(r['inhibit_reason'], r['veto_sources'])}")
            if r["action"] == "api_call":
                w(f"- **ORACLE:** Surprise level too high — called for external guidance")
            action_label = self.ACTION_DESC.get(r["action"], r["action"])
            reward_str = f"+{r['reward']:.2f}" if r["reward"] >= 0 else f"{r['reward']:.2f}"
            w(f"- **Chose:** {action_label}")
            w(f"- **Earned:** {reward_str} reward")
            # Body: only show if changed
            prev = self._prev_body.get(r["uid"])
            if r["body_desc"] != prev:
                w(f"- **Body:** {r['body_desc']}")
                self._prev_body[r["uid"]] = r["body_desc"]
            w("")

        # Quiet agents grouped by action
        if quiet:
            groups: dict[str, list[str]] = {}
            for r in quiet:
                groups.setdefault(r["action"], []).append(r["uid"])
                self._prev_body[r["uid"]] = r["body_desc"]  # track silently

            group_parts = []
            for action, uids in sorted(groups.items()):
                label = self.ACTION_DESC.get(action, action)
                ids = ", ".join(uids)
                group_parts.append(f"Agents {ids} — {label}")
            reward_sample = quiet[0]["reward"]
            reward_str = f"+{reward_sample:.2f}" if reward_sample >= 0 else f"{reward_sample:.2f}"
            w(f"*Quiet:* {'; '.join(group_parts)} ({reward_str} each)")
            w("")

    def _write_verbose_agents(self, w, records: list[dict]) -> None:
        """Original full-detail output for every agent."""
        for r in records:
            voice = f" | Voice: {r['provider'].capitalize()}" if r["provider"] else ""
            w(f"### Agent {r['uid']} ({r['role']}{voice})")
            sense_desc = self._describe_markers(r["sensed_markers"])
            if sense_desc:
                w(f"- **Sensed:** {sense_desc}")
            w(f"- **Feeling:** {self._describe_surprise(r['pred_error'])}")
            if r["inhibited"]:
                w(f"- **PAUSED:** {self._describe_inhibition(r['inhibit_reason'], r['veto_sources'])}")
            if r["action"] == "api_call":
                w(f"- **ORACLE:** Surprise level too high — called for external guidance")
            action_label = self.ACTION_DESC.get(r["action"], r["action"])
            reward_str = f"+{r['reward']:.2f}" if r["reward"] >= 0 else f"{r['reward']:.2f}"
            w(f"- **Chose:** {action_label}")
            w(f"- **Earned:** {reward_str} reward")
            w(f"- **Body:** {r['body_desc']}")
            w("")

    # ── Private formatters ────────────────────────────────────────────────────

    def _normalize_action(self, action: Any) -> str:
        if isinstance(action, dict):
            return action.get("type", "idle")
        if isinstance(action, str):
            return action
        return "idle" if action is None else str(action)

    def _describe_surprise(self, pred_error: float) -> str:
        for threshold, label in self.SURPRISE_LABELS:
            if pred_error <= threshold:
                return f"{label} ({pred_error:.2f})"
        return f"Extreme surprise ({pred_error:.2f})"

    def _describe_body(self, body_state: Any) -> str:
        if not isinstance(body_state, dict):
            return "State unknown"
        energy = float(body_state.get("energy_level", 1.0) or 1.0)
        valence = float(body_state.get("emotional_valence", 0.0) or 0.0)

        if energy >= 0.8:
            energy_desc = "Energized"
        elif energy >= 0.5:
            energy_desc = "Moderate energy"
        elif energy >= 0.2:
            energy_desc = "Low energy"
        else:
            energy_desc = "Exhausted"

        if valence >= 0.3:
            mood_desc = f"calm (+{valence:.2f})"
        elif valence >= -0.1:
            mood_desc = f"neutral ({valence:.2f})"
        elif valence >= -0.4:
            mood_desc = f"uneasy ({valence:.2f})"
        else:
            mood_desc = f"stressed ({valence:.2f})"

        return f"{energy_desc} ({energy:.2f}), {mood_desc}"

    def _describe_inhibition(self, reason: str, veto_sources: list) -> str:
        if veto_sources:
            causes = ", ".join(str(v) for v in veto_sources[:3])
            return f"Body said wait — causes: {causes}"
        if reason:
            # Strip technical boilerplate (e.g., "INHIBIT: NoGo(0.6) > Go(0.3)")
            if ":" in reason:
                reason = reason.split(":", 1)[1].strip()
            return f"Body said wait — {reason}"
        return "Body said wait"

    def _describe_markers(self, sensed_markers: Any) -> str:
        if not isinstance(sensed_markers, dict) or not sensed_markers:
            return ""
        parts = []
        for marker_type, markers in sensed_markers.items():
            count = len(markers) if isinstance(markers, (list, tuple)) else 1
            name = marker_type.upper()
            if name == "SUCCESS":
                parts.append(f"{count} success trail(s) nearby")
            elif name == "DANGER":
                parts.append(f"{count} danger signal(s) nearby")
            else:
                parts.append(f"{count} {marker_type.lower()} marker(s)")
        return ", ".join(parts)

    def _describe_trust(self, trust: float) -> str:
        if trust >= 0.7:
            return "high"
        if trust >= 0.4:
            return "moderate"
        return "low"

    def _describe_organism(self, endocrine: dict, phi: float, agent_count: int) -> str:
        stressed = endocrine.get("is_stressed", False)
        resting = endocrine.get("is_resting", False)
        explore = float(endocrine.get("exploration_bias", 0.5) or 0.5)
        trust = float(endocrine.get("trust_level", 0.5) or 0.5)

        if stressed:
            state = "stressed"
        elif resting:
            state = "resting"
        else:
            state = "alert"

        if explore > 0.6:
            drive = "curious and exploratory"
        elif explore > 0.4:
            drive = "balanced between exploring and exploiting"
        else:
            drive = "focused on exploiting what it knows"

        mood = f"The organism is {state} and {drive}. {agent_count} agents working."
        if phi > 0:
            mood += f" Integration coherence (phi): {phi:.2f}."
        if trust > 0.65:
            mood += " Agent trust is high."
        elif trust < 0.3:
            mood += " Agent trust is low."
        return mood

    # ── System state readers ──────────────────────────────────────────────────

    def _read_phase(self, systems: dict) -> str:
        circ = systems.get("circadian")
        if circ:
            for method in ("get_statistics", "get_stats"):
                if hasattr(circ, method):
                    try:
                        return str(getattr(circ, method)().get("current_phase", "ACTIVE"))
                    except Exception:
                        pass
        return "ACTIVE"

    def _read_endocrine(self, systems: dict) -> dict:
        es = systems.get("endocrine")
        if es:
            for method in ("get_statistics", "get_stats"):
                if hasattr(es, method):
                    try:
                        return getattr(es, method)() or {}
                    except Exception:
                        pass
        return {}

    def _read_phi(self, systems: dict) -> float:
        meter = systems.get("integration_meter")
        if meter:
            for method in ("get_statistics", "get_stats"):
                if hasattr(meter, method):
                    try:
                        return float(getattr(meter, method)().get("current_phi", 0.0) or 0.0)
                    except Exception:
                        pass
        return 0.0

    def _read_bus(self, systems: dict) -> dict:
        bus = systems.get("event_bus")
        if bus:
            for method in ("get_statistics", "get_stats"):
                if hasattr(bus, method):
                    try:
                        return getattr(bus, method)() or {}
                    except Exception:
                        pass
        return {}

    def _read_pool(self, systems: dict) -> dict:
        pool = systems.get("task_pool")
        if pool:
            for method in ("get_statistics", "get_stats"):
                if hasattr(pool, method):
                    try:
                        return getattr(pool, method)() or {}
                    except Exception:
                        pass
        return {}

    def _write_file(self, path: pathlib.Path, overwrite: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "a"
        try:
            with open(path, mode, encoding="utf-8") as f:
                f.write("\n".join(self._run_lines) + "\n")
        except OSError as e:
            logger.warning("Journal write failed (%s): %s", path, e)
