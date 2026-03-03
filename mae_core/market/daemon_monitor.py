"""Daemon Monitor — Heartbeat writer for MIDGE daemon mode.

Writes a heartbeat file every N steps so external monitoring can detect
if the daemon has stalled. The heartbeat includes process info, timing,
and basic health metrics.

Part of WP-C: Daemon Mode (Always-On MIDGE Wave 1).
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("midge.daemon")

_HEARTBEAT_PATH = Path(__file__).resolve().parents[2] / "data" / "midge" / "heartbeat.json"


class DaemonMonitor:
    """Writes periodic heartbeat files for daemon health monitoring.

    Usage:
        monitor = DaemonMonitor()
        # In step loop:
        monitor.record_step(step_num, model, systems)
    """

    def __init__(
        self,
        heartbeat_path: Optional[Path] = None,
        heartbeat_cadence: int = 100,
    ):
        self._heartbeat_path = heartbeat_path or _HEARTBEAT_PATH
        self._heartbeat_cadence = heartbeat_cadence
        self._start_time = time.monotonic()
        self._start_wall = datetime.now(timezone.utc)
        self._pid = os.getpid()
        self._total_steps = 0
        self._round_num = 0
        self._last_signal_time: Optional[str] = None

    def begin_round(self, round_num: int) -> None:
        """Mark the start of a new daemon round."""
        self._round_num = round_num

    def record_step(self, step: int, systems: dict) -> None:
        """Record a step and write heartbeat if on cadence."""
        self._total_steps += 1

        if self._total_steps % self._heartbeat_cadence != 0:
            return

        self._write_heartbeat(step, systems)

    def _write_heartbeat(self, step: int, systems: dict) -> None:
        """Write heartbeat JSON file with current status."""
        elapsed = time.monotonic() - self._start_time
        step_rate = self._total_steps / max(elapsed, 0.001)

        # Count active agents
        agents = systems.get("agents", [])
        active_agents = len(agents) if agents else 0

        # Get last signal time from sensing hook stats
        sensing_hook = systems.get("_market_sensing_hook")
        if sensing_hook is not None and hasattr(sensing_hook, "get_statistics"):
            try:
                stats = sensing_hook.get_statistics()
                self._last_signal_time = stats.get("last_fetch_source")
            except Exception:
                pass

        heartbeat = {
            "pid": self._pid,
            "started_at": self._start_wall.isoformat(),
            "uptime_seconds": round(elapsed, 1),
            "current_step": step,
            "total_daemon_steps": self._total_steps,
            "daemon_round": self._round_num,
            "step_rate_per_second": round(step_rate, 2),
            "active_agents": active_agents,
            "last_fetch_source": self._last_signal_time,
            "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self._heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._heartbeat_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
            tmp.replace(self._heartbeat_path)
        except Exception:
            logger.debug("Heartbeat write failed", exc_info=True)

    def get_uptime(self) -> float:
        """Return daemon uptime in seconds."""
        return time.monotonic() - self._start_time

    def get_statistics(self) -> dict:
        """Return monitor stats for integration."""
        elapsed = time.monotonic() - self._start_time
        return {
            "pid": self._pid,
            "uptime_seconds": round(elapsed, 1),
            "total_daemon_steps": self._total_steps,
            "daemon_round": self._round_num,
            "heartbeat_cadence": self._heartbeat_cadence,
        }
