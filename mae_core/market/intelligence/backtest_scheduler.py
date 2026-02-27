"""BacktestScheduler — Autonomous scheduling of sweep backtest reruns.

Bridge 3: MIDGE schedules her own backtest refreshes without human intervention.

Staleness check: reads run_time from results JSON. Skips if < 24h old.
Non-blocking: ThreadPoolExecutor(1) with skip-if-busy semantics (same pattern
  as MarketSensingHook in sensing_hook.py).
Hypothesis refresh: retires all PROBATION BACKTEST_DERIVED hypotheses before
  re-analyzing. This ensures the rolling yfinance window produces fresh stats
  rather than being blocked by dedup.

Wired into step hook at cadence 5000 (configurable). Runs in background thread.
Does NOT block the step loop at any point.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Default results path — same as BacktestAnalyzer
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_DEFAULT_RESULTS_PATH = _DATA_DIR / "sweep_backtest_results.json"

# Default futures symbols (same as sweep_backtest.py FUTURES)
_DEFAULT_SYMBOLS = ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F"]


class BacktestScheduler:
    """Schedules autonomous sweep backtest reruns.

    Uses ThreadPoolExecutor(1) with skip-if-busy semantics:
    - One background backtest at a time
    - If previous run still going, skip this cadence tick
    - Results written to sweep_backtest_results.json
    - After write, calls refresh_probation() + analyze() on BacktestAnalyzer
    """

    def __init__(
        self,
        backtest_analyzer: Any,
        results_path: Path = None,
        stale_threshold_hours: float = 24.0,
        symbols: List[str] = None,
        bus: Any = None,
    ):
        self._analyzer = backtest_analyzer
        self._results_path = results_path or _DEFAULT_RESULTS_PATH
        self._stale_threshold_hours = stale_threshold_hours
        self._symbols = symbols or list(_DEFAULT_SYMBOLS)
        self._bus = bus

        # Async execution — one background run at a time
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bt-sched"
        )
        self._pending_future: Optional[Future] = None

        # Statistics
        self._runs_scheduled = 0
        self._runs_completed = 0
        self._last_run_time: Optional[str] = None
        self._hypotheses_refreshed = 0

    def check_and_schedule(self) -> None:
        """Step-hook callable. Check staleness, launch if needed.

        Three guards prevent unnecessary work:
        1. Skip if a backtest is already running (busy guard)
        2. Collect results if previous run finished
        3. Skip if results are fresh (staleness guard)
        """
        # Guard 1: collect completed future
        if self._pending_future is not None:
            if self._pending_future.done():
                self._collect_result()
            else:
                logger.debug("Backtest still running, skipping schedule check")
                return

        # Guard 2: skip if results are fresh
        if not self._is_stale():
            logger.debug("Backtest results fresh, skipping rerun")
            return

        # Launch background backtest
        logger.info("Backtest results stale — scheduling rerun (%d symbols)",
                     len(self._symbols))
        self._pending_future = self._executor.submit(
            self._run_backtest_and_refresh
        )
        self._runs_scheduled += 1

    def _is_stale(self) -> bool:
        """Check if backtest results are older than the staleness threshold.

        Returns True if:
        - Results file doesn't exist
        - run_time field is missing or unparseable
        - run_time is older than stale_threshold_hours
        """
        if not self._results_path.exists():
            return True

        try:
            with open(self._results_path) as f:
                data = json.load(f)
            run_time_str = data.get("run_time")
            if not run_time_str:
                return True
            last_run = datetime.fromisoformat(run_time_str)
            age_seconds = (datetime.now() - last_run).total_seconds()
            return age_seconds > self._stale_threshold_hours * 3600
        except (json.JSONDecodeError, ValueError, OSError):
            return True

    def _run_backtest_and_refresh(self) -> dict:
        """Run backtest and refresh hypotheses. Executes in background thread.

        Returns a summary dict for statistics tracking.
        """
        try:
            from mae_core.market.edge.sweep_backtest import SweepBacktester

            bt = SweepBacktester(interval="5m", days=59)
            trades = bt.run(self._symbols)

            if not trades:
                logger.warning("Backtest produced no trades")
                return {"trades": 0, "refreshed": 0}

            # Write results (same format as sweep_backtest.py CLI main())
            self._write_results(trades)

            # Refresh hypotheses: retire stale PROBATION, create fresh ones
            refreshed = 0
            if self._analyzer is not None:
                refreshed = self._analyzer.refresh_probation()
                new_hyps = self._analyzer.analyze()
                logger.info(
                    "Backtest refresh: %d trades, %d probation retired, "
                    "%d new hypotheses",
                    len(trades), refreshed, len(new_hyps),
                )

            # Publish completion event
            if self._bus is not None:
                try:
                    self._bus.publish("market.intel.backtest_refreshed", {
                        "symbols_run": len(self._symbols),
                        "total_trades": len(trades),
                        "hypotheses_refreshed": refreshed,
                        "new_hypotheses": len(new_hyps) if self._analyzer else 0,
                        "run_time": datetime.now().isoformat(),
                    })
                except Exception:
                    logger.debug("Failed to publish backtest_refreshed", exc_info=True)

            return {"trades": len(trades), "refreshed": refreshed}

        except Exception:
            logger.exception("Backtest rerun failed")
            return {"trades": 0, "refreshed": 0, "error": True}

    def _write_results(self, trades) -> None:
        """Serialize trades to JSON (same format as sweep_backtest.py main())."""
        self._results_path.parent.mkdir(parents=True, exist_ok=True)

        wins = [t for t in trades if t.result == "win_2r"]
        results = {
            "run_time": datetime.now().isoformat(),
            "config": {
                "interval": "5m",
                "days": 59,
                "symbols": list({t.symbol for t in trades}),
            },
            "summary": {
                "total_trades": len(trades),
                "wins": len(wins),
                "losses": len(trades) - len(wins),
                "win_rate": len(wins) / len(trades) * 100 if trades else 0,
                "avg_r": sum(t.r_captured for t in trades) / len(trades),
                "hit_1r_count": len([t for t in trades if t.hit_1r]),
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "session_swept": t.session_swept,
                    "entry_price": t.entry_price,
                    "stop_price": t.stop_price,
                    "target_2r": t.target_2r,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "result": t.result,
                    "r_captured": t.r_captured,
                    "hit_1r": t.hit_1r,
                    "risk_pts": t.risk_pts,
                    "displacement_score": t.displacement_score,
                    "fvg_atr_ratio": t.fvg_atr_ratio,
                    "kill_zone_score": t.kill_zone_score,
                    "quality_score": t.quality_score,
                }
                for t in trades
            ],
        }

        with open(self._results_path, "w") as f:
            json.dump(results, f, indent=2)

        self._last_run_time = results["run_time"]

    def _collect_result(self) -> None:
        """Collect result from completed background future."""
        try:
            result = self._pending_future.result()
            self._runs_completed += 1
            self._hypotheses_refreshed += result.get("refreshed", 0)
            if "error" not in result:
                logger.info("Backtest rerun completed: %d trades", result.get("trades", 0))
            else:
                logger.warning("Backtest rerun completed with errors")
        except Exception:
            logger.exception("Backtest future raised exception")
        finally:
            self._pending_future = None

    def get_statistics(self) -> dict:
        """Summary stats for HolonProxy monitoring."""
        return {
            "runs_scheduled": self._runs_scheduled,
            "runs_completed": self._runs_completed,
            "last_run_time": self._last_run_time or "",
            "hypotheses_refreshed": self._hypotheses_refreshed,
            "stale_threshold_hours": self._stale_threshold_hours,
            "symbols_count": len(self._symbols),
            "is_running": (
                self._pending_future is not None
                and not self._pending_future.done()
            ),
        }
