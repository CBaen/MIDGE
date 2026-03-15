#!/usr/bin/env python3
"""supervisor.py — MIDGE Ecosystem Supervisor.

Starts, watches, and restarts all MIDGE long-running processes as a single
supervisory shell. Each process runs as an independent OS subprocess so they
are fully isolated — a crash in the post-mortem analyzer never touches the
daemon, and vice versa.

What it does:
  - Starts each registered process as: python -m {module} {args}
  - Polls every 10 seconds; if a process has exited, restarts it after its
    configured restart_delay
  - Logs all lifecycle events (start/death/restart) to data/midge/ecosystem.log
  - Writes a JSON status snapshot every 30 seconds to
    data/midge/ecosystem_status.json
  - Handles SIGINT/SIGTERM: gracefully terminates all children, then exits

Usage:
    # Start everything (daemon + all analysis/mining processes)
    python -m mae_core.market.ecosystem.supervisor

    # Start only analysis + mining — daemon already running separately
    python -m mae_core.market.ecosystem.supervisor --exclude daemon

    # Start only the analysis tier
    python -m mae_core.market.ecosystem.supervisor --only analysis

    # Print current status and exit (daemon does not need to be running)
    python -m mae_core.market.ecosystem.supervisor --status

Windows notes:
  - subprocess.Popen is used, NOT multiprocessing, so each process is a true
    OS child process with its own Python interpreter
  - SIGTERM is not reliable on Windows; we call process.terminate() first,
    then process.kill() after a 5-second wait
  - The supervisor itself can be stopped with Ctrl+C
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# mae_core/market/ecosystem/supervisor.py  →  3 parents up = project root
# parents[0] = ecosystem/, parents[1] = market/, parents[2] = mae_core/, parents[3] = MIDGE/
_PROJECT_ROOT = _THIS_FILE.parents[3]

_DATA_DIR        = _PROJECT_ROOT / "data" / "midge"
_LOG_FILE        = _DATA_DIR / "ecosystem.log"
_STATUS_FILE     = _DATA_DIR / "ecosystem_status.json"

_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("ecosystem")

# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------

@dataclass
class ManagedProcess:
    """Runtime state for one supervised process."""
    name: str
    module: str
    args: List[str]
    tier: str
    restart_delay: int
    critical: bool

    # Runtime fields — not part of the spec
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    pid: Optional[int] = None
    started_at: Optional[float] = None      # time.monotonic() at last start
    status: str = "stopped"                 # "running" | "restarting" | "stopped"
    restart_count: int = 0
    restart_after: Optional[float] = None   # monotonic time when we may restart

    @property
    def uptime_seconds(self) -> float:
        if self.started_at is None or self.status != "running":
            return 0.0
        return time.monotonic() - self.started_at

    def as_status_dict(self) -> dict:
        return {
            "name": self.name,
            "tier": self.tier,
            "pid": self.pid,
            "status": self.status,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "restart_count": self.restart_count,
            "critical": self.critical,
        }


# ---------------------------------------------------------------------------
# Supervisor core
# ---------------------------------------------------------------------------

class EcosystemSupervisor:
    """Manages the full set of MIDGE parallel processes."""

    POLL_INTERVAL   = 10    # seconds between health checks
    STATUS_INTERVAL = 30    # seconds between status file writes

    def __init__(self, specs: List[dict]) -> None:
        self._processes: Dict[str, ManagedProcess] = {}
        for spec in specs:
            mp = ManagedProcess(
                name=spec["name"],
                module=spec["module"],
                args=spec["args"],
                tier=spec["tier"],
                restart_delay=spec["restart_delay"],
                critical=spec["critical"],
            )
            self._processes[spec["name"]] = mp

        self._ecosystem_start = time.monotonic()
        self._shutdown_requested = False
        self._last_status_write = 0.0

        # Register signal handlers (best-effort on Windows)
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame) -> None:  # noqa: ANN001
        log.info("Signal %d received — initiating graceful shutdown.", signum)
        self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _python_cmd(self) -> str:
        """Return the Python executable used to run this supervisor."""
        return sys.executable

    def _start(self, mp: ManagedProcess) -> None:
        """Launch a managed process via subprocess.Popen."""
        cmd = [self._python_cmd(), "-m", mp.module] + mp.args
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                # Inherit stdout/stderr so output flows to the terminal / log
                # The ecosystem log captures supervisor messages; child logs
                # go to their own per-process log files (as designed).
                stdout=None,
                stderr=None,
            )
            mp.proc = proc
            mp.pid = proc.pid
            mp.started_at = time.monotonic()
            mp.status = "running"
            mp.restart_after = None
            log.info(
                "[START] %-22s  pid=%-7d  tier=%s  module=%s",
                mp.name, proc.pid, mp.tier, mp.module,
            )
        except Exception as exc:
            mp.status = "stopped"
            mp.pid = None
            log.error("[ERROR] Failed to start %s: %s", mp.name, exc)

    def _terminate(self, mp: ManagedProcess, timeout: float = 5.0) -> None:
        """Terminate a running process, forcefully if necessary."""
        if mp.proc is None:
            return
        try:
            mp.proc.terminate()
            try:
                mp.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                log.warning(
                    "[KILL] %s did not stop after %.0fs — sending kill.",
                    mp.name, timeout,
                )
                mp.proc.kill()
                mp.proc.wait(timeout=3)
        except Exception as exc:
            log.warning("[KILL] Error terminating %s: %s", mp.name, exc)
        finally:
            mp.proc = None
            mp.pid = None
            mp.status = "stopped"

    # ------------------------------------------------------------------
    # Health check + restart logic
    # ------------------------------------------------------------------

    def _check_one(self, mp: ManagedProcess) -> None:
        """Poll a single process; schedule restart if it has died."""
        if mp.status == "stopped":
            return

        if mp.status == "restarting":
            # Check if the delay has elapsed
            if mp.restart_after is not None and time.monotonic() >= mp.restart_after:
                log.info(
                    "[RESTART] %s  restart_count=%d",
                    mp.name, mp.restart_count + 1,
                )
                mp.restart_count += 1
                self._start(mp)
            return

        # status == "running" — poll the real process
        if mp.proc is None:
            mp.status = "stopped"
            return

        exit_code = mp.proc.poll()
        if exit_code is None:
            return  # Still running — all good

        # Process has exited
        log.warning(
            "[DIED] %-22s  pid=%-7d  exit_code=%s  restart_in=%ds",
            mp.name, mp.pid, exit_code, mp.restart_delay,
        )
        if mp.critical:
            log.critical(
                "!!! CRITICAL PROCESS DIED: %s — ecosystem is degraded !!!",
                mp.name,
            )

        mp.proc = None
        mp.pid = None
        mp.status = "restarting"
        mp.restart_after = time.monotonic() + mp.restart_delay

    def _check_all(self) -> None:
        for mp in self._processes.values():
            self._check_one(mp)

    # ------------------------------------------------------------------
    # Status file
    # ------------------------------------------------------------------

    def _write_status(self) -> None:
        """Write ecosystem_status.json atomically."""
        ecosystem_uptime = round(time.monotonic() - self._ecosystem_start, 1)
        payload = {
            "supervisor_pid": os.getpid(),
            "ecosystem_uptime_seconds": ecosystem_uptime,
            "total_process_count": len(self._processes),
            "running_count": sum(
                1 for mp in self._processes.values() if mp.status == "running"
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processes": [
                mp.as_status_dict() for mp in self._processes.values()
            ],
        }
        tmp = _STATUS_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(_STATUS_FILE)
        except Exception as exc:
            log.warning("Could not write status file: %s", exc)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start_all(self) -> None:
        """Start every registered process."""
        log.info("=== MIDGE Ecosystem Supervisor starting — %d processes ===",
                 len(self._processes))
        for mp in self._processes.values():
            self._start(mp)

    def run(self) -> None:
        """Main supervisor loop — runs until shutdown is requested."""
        self.start_all()

        while not self._shutdown_requested:
            self._check_all()

            now = time.monotonic()
            if now - self._last_status_write >= self.STATUS_INTERVAL:
                self._write_status()
                self._last_status_write = now

            time.sleep(self.POLL_INTERVAL)

        self._shutdown()

    def _shutdown(self) -> None:
        """Gracefully stop all child processes."""
        log.info("=== Shutdown — stopping %d processes ===", len(self._processes))
        for mp in self._processes.values():
            if mp.proc is not None:
                log.info("[STOP] %s (pid=%s)", mp.name, mp.pid)
                self._terminate(mp)
        self._write_status()
        log.info("=== Ecosystem stopped ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_status() -> int:
    """Read and print ecosystem_status.json. Returns exit code."""
    if not _STATUS_FILE.exists():
        print("No status file found. Is the supervisor running?")
        print(f"Expected: {_STATUS_FILE}")
        return 1

    try:
        data = json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read status file: {exc}")
        return 1

    uptime = data.get("ecosystem_uptime_seconds", 0)
    mins, secs = divmod(int(uptime), 60)
    hours, mins = divmod(mins, 60)

    print(f"\nMIDGE Ecosystem — uptime {hours:02d}h {mins:02d}m {secs:02d}s")
    print(f"Supervisor PID : {data.get('supervisor_pid', '?')}")
    print(f"Timestamp      : {data.get('timestamp', '?')}")
    print(
        f"Processes      : {data.get('running_count', '?')} / "
        f"{data.get('total_process_count', '?')} running\n"
    )

    header = f"{'NAME':<24} {'TIER':<10} {'STATUS':<12} {'PID':<8} {'UPTIME':>10} {'RESTARTS':>9}"
    print(header)
    print("-" * len(header))

    for proc in data.get("processes", []):
        uptime_s = proc.get("uptime_seconds", 0)
        pm, ps = divmod(int(uptime_s), 60)
        ph, pm = divmod(pm, 60)
        uptime_str = f"{ph:02d}h {pm:02d}m {ps:02d}s"
        critical_flag = " [CRITICAL]" if proc.get("critical") else ""
        print(
            f"{proc['name']:<24} {proc['tier']:<10} {proc['status']:<12} "
            f"{str(proc.get('pid') or '-'):<8} {uptime_str:>10} "
            f"{proc.get('restart_count', 0):>9}{critical_flag}"
        )

    print()
    return 0


def _build_spec_list(
    all_specs: List[dict],
    exclude_names: List[str],
    only_tier: Optional[str],
) -> List[dict]:
    """Filter the registry down to what should actually be started."""
    result = []
    for spec in all_specs:
        if spec["name"] in exclude_names:
            log.info("Excluding %s (--exclude)", spec["name"])
            continue
        if only_tier is not None and spec["tier"] != only_tier:
            continue
        result.append(spec)
    return result


def main() -> None:
    from mae_core.market.ecosystem.process_registry import (
        ECOSYSTEM_PROCESSES,
        ALL_TIERS,
    )

    parser = argparse.ArgumentParser(
        description="MIDGE Ecosystem Supervisor — manages all parallel processes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m mae_core.market.ecosystem.supervisor
  python -m mae_core.market.ecosystem.supervisor --exclude daemon
  python -m mae_core.market.ecosystem.supervisor --only analysis
  python -m mae_core.market.ecosystem.supervisor --status
        """,
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        metavar="NAME",
        default=[],
        help="Process names to skip (e.g. --exclude midge-daemon)",
    )
    parser.add_argument(
        "--only",
        metavar="TIER",
        default=None,
        choices=sorted(ALL_TIERS),
        help="Start only processes in this tier (core | analysis | mining)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current ecosystem status and exit",
    )
    args = parser.parse_args()

    if args.status:
        sys.exit(_print_status())

    specs = _build_spec_list(
        all_specs=ECOSYSTEM_PROCESSES,
        exclude_names=args.exclude,
        only_tier=args.only,
    )

    if not specs:
        log.error("No processes selected after filtering. Nothing to supervise.")
        sys.exit(1)

    supervisor = EcosystemSupervisor(specs)
    try:
        supervisor.run()
    except KeyboardInterrupt:
        # Signal handler sets the flag; run() will call _shutdown().
        # In case we land here before the flag is set, call it directly.
        supervisor._shutdown_requested = True
        supervisor._shutdown()


if __name__ == "__main__":
    main()
