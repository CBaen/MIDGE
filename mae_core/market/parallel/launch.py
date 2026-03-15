#!/usr/bin/env python3
"""launch.py — Multiprocessing launcher for continuous replay.

Starts the continuous replay process as an independent OS process.
The replay process runs completely independently of the daemon loop —
it reads from the signal archive, not from live EventBus traffic.

Usage:
    python -m mae_core.market.parallel.launch
    python -m mae_core.market.parallel.launch --min-domains 3 --start 2026-01-01

To check if it's running:
    # Windows:
    tasklist | findstr python

To stop it:
    # Windows (from PowerShell):
    Get-Process python | Where-Object {$_.MainWindowTitle -eq ""} | Stop-Process

The PID is written to data/midge/continuous_replay.pid so you can
identify the exact process.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PID_FILE     = _PROJECT_ROOT / "data" / "midge" / "continuous_replay.pid"


def _replay_worker(
    min_domains: int,
    use_thompson: bool,
    start_date_str: str | None,
    outcome_window_days: int,
    dry_run: bool,
) -> None:
    """Target function for the replay subprocess."""
    # Write our PID so the parent (and operators) can find us
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))

    # Resolve optional start date inside the worker
    start_date = None
    if start_date_str:
        try:
            start_date = date.fromisoformat(start_date_str)
        except ValueError:
            pass

    from mae_core.market.parallel.continuous_replay import run_forever
    run_forever(
        min_domains=min_domains,
        use_thompson=use_thompson,
        start_date=start_date,
        outcome_window_days=outcome_window_days,
        dry_run=dry_run,
    )


def launch(
    min_domains: int = 3,
    use_thompson: bool = True,
    start_date: date | None = None,
    outcome_window_days: int = 14,
    dry_run: bool = False,
    daemon: bool = True,
) -> multiprocessing.Process:
    """Start the continuous replay as a background Process and return it.

    Args:
        min_domains:         Minimum convergence domains (default 3).
        use_thompson:        Whether to weight confidence with Thompson distributions.
        start_date:          Only replay days on/after this date.
        outcome_window_days: Days after alert to grade price outcome.
        dry_run:             If True, run replay but don't write results.
        daemon:              If True (default), the process dies when the parent exits.
                             Set to False to let it outlive the parent.

    Returns:
        The started multiprocessing.Process instance.
    """
    start_date_str = start_date.isoformat() if start_date else None

    proc = multiprocessing.Process(
        target=_replay_worker,
        args=(min_domains, use_thompson, start_date_str, outcome_window_days, dry_run),
        name="midge-continuous-replay",
        daemon=daemon,
    )
    proc.start()
    print(
        f"[launch] Continuous replay started — PID {proc.pid} "
        f"(daemon={daemon}, min_domains={min_domains}, thompson={use_thompson})",
        file=sys.stderr,
    )
    return proc


def main() -> None:
    """CLI entry point — starts the process and waits for it (so Ctrl+C works)."""
    parser = argparse.ArgumentParser(
        description="Launch MIDGE continuous historical replay as a background process",
    )
    parser.add_argument(
        "--min-domains", type=int, default=3,
        help="Minimum domains required for convergence alert (default: 3)",
    )
    parser.add_argument(
        "--no-thompson", action="store_true",
        help="Disable Thompson weighting",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Only replay days on/after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--outcome-window", type=int, default=14,
        help="Days after alert to check price for grading (default: 14)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without writing results",
    )
    parser.add_argument(
        "--no-daemon", action="store_true",
        help="Keep process alive after this script exits (default: daemon=True)",
    )
    args = parser.parse_args()

    start_date = None
    if args.start:
        try:
            start_date = date.fromisoformat(args.start)
        except ValueError:
            print(f"Invalid --start date: {args.start}", file=sys.stderr)
            sys.exit(1)

    proc = launch(
        min_domains=args.min_domains,
        use_thompson=not args.no_thompson,
        start_date=start_date,
        outcome_window_days=args.outcome_window,
        dry_run=args.dry_run,
        daemon=not args.no_daemon,
    )

    print(
        f"Replay process running (PID {proc.pid}). "
        f"Press Ctrl+C to stop.",
        file=sys.stderr,
    )

    try:
        proc.join()
    except KeyboardInterrupt:
        print("\nInterrupt received — terminating replay process.", file=sys.stderr)
        proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()

    # Clean up PID file
    try:
        _PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    print(f"Replay process stopped (exit code {proc.exitcode}).", file=sys.stderr)


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    multiprocessing.freeze_support()
    main()
