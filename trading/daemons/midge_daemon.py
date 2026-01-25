#!/usr/bin/env python3
"""
midge_daemon.py - Master MIDGE Orchestrator

The brain that coordinates all MIDGE processes:
- Data collection (every 5 minutes)
- Evolution cycles (every 1 hour)
- Daily reports (every 24 hours)

Usage:
    # Dry run - show what would happen
    python trading/daemons/midge_daemon.py --dry-run

    # Run continuously (production mode)
    python trading/daemons/midge_daemon.py --continuous

    # Run once (all components)
    python trading/daemons/midge_daemon.py --once

    # Status check
    python trading/daemons/midge_daemon.py --status
"""

import os
import sys
import json
import time
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread, Event
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
STATE_FILE = PROJECT_ROOT / ".claude" / "midge_daemon_state.json"
LOG_FILE = PROJECT_ROOT / ".claude" / "midge_daemon.log"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Timing (in seconds)
DATA_COLLECTION_INTERVAL = 300    # 5 minutes
EVOLUTION_CYCLE_INTERVAL = 3600   # 1 hour
DAILY_REPORT_INTERVAL = 86400     # 24 hours

# Default symbols to track
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "LMT", "BA", "RTX", "NVDA", "SPY", "QQQ"]


@dataclass
class DaemonState:
    """Persistent state for the daemon."""
    start_time: str
    cycles_run: int
    data_collections: int
    reports_generated: int
    last_data_collection: Optional[str]
    last_evolution_cycle: Optional[str]
    last_daily_report: Optional[str]
    errors: int
    status: str  # running, stopped, error


class MIDGEDaemon:
    """
    Master orchestrator for MIDGE.

    Coordinates:
    - DataCollector: Fresh price/signal data every 5 min
    - MIDGEEvolution: Full evolution cycle every 1 hour
    - DailyReport: Summary report every 24 hours
    """

    def __init__(self, symbols: List[str] = None, dry_run: bool = False):
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.dry_run = dry_run
        self.stop_event = Event()

        # Ensure directories exist
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load or create state
        self.state = self._load_state()

        # Components (lazy loaded)
        self._data_collector = None
        self._evolution = None
        self._paper_trading = None

    def _load_state(self) -> DaemonState:
        """Load daemon state from disk."""
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                return DaemonState(**data)
        except Exception:
            pass

        return DaemonState(
            start_time=datetime.now().isoformat(),
            cycles_run=0,
            data_collections=0,
            reports_generated=0,
            last_data_collection=None,
            last_evolution_cycle=None,
            last_daily_report=None,
            errors=0,
            status="initialized"
        )

    def _save_state(self):
        """Persist daemon state to disk."""
        try:
            STATE_FILE.write_text(json.dumps(asdict(self.state), indent=2))
        except Exception as e:
            self._log(f"Failed to save state: {e}", "ERROR")

    def _log(self, msg: str, level: str = "INFO"):
        """Log to stdout and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"
        print(line)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _get_data_collector(self):
        """Lazy load data collector."""
        if self._data_collector is None:
            try:
                from trading.daemons.data_collector import run_collection_cycle, DataCollectorState
                self._data_collector = {
                    "run": run_collection_cycle,
                    "state": DataCollectorState()
                }
            except ImportError as e:
                self._log(f"Data collector not available: {e}", "WARN")
        return self._data_collector

    def _get_evolution(self):
        """Lazy load evolution engine."""
        if self._evolution is None:
            try:
                from core.evolution import MIDGEEvolution
                self._evolution = MIDGEEvolution()
            except ImportError as e:
                self._log(f"Evolution engine not available: {e}", "WARN")
        return self._evolution

    def _get_paper_trading(self):
        """Lazy load paper trading session."""
        if self._paper_trading is None:
            try:
                from trading.outcome_tracker import PaperTradingSession
                self._paper_trading = PaperTradingSession()
            except ImportError as e:
                self._log(f"Paper trading not available: {e}", "WARN")
        return self._paper_trading

    # =========================================================================
    # Data Collection (every 5 minutes)
    # =========================================================================

    def run_data_collection(self) -> Dict:
        """Run one data collection cycle."""
        self._log("DATA COLLECTION: Starting...")

        if self.dry_run:
            self._log("  [DRY RUN] Would collect data for: " + ", ".join(self.symbols))
            return {"dry_run": True, "symbols": self.symbols}

        collector = self._get_data_collector()
        if not collector:
            self._log("  Data collector not available", "WARN")
            return {"error": "Data collector not available"}

        try:
            result = collector["run"](self.symbols, collector["state"])
            self.state.data_collections += 1
            self.state.last_data_collection = datetime.now().isoformat()
            self._save_state()

            self._log(f"  Collected data for {result['symbols_succeeded']}/{result['symbols_processed']} symbols")
            self._log(f"  Stored {result['total_stored']} signals to Qdrant")

            return result

        except Exception as e:
            self.state.errors += 1
            self._save_state()
            self._log(f"  Data collection failed: {e}", "ERROR")
            return {"error": str(e)}

    # =========================================================================
    # Evolution Cycle (every 1 hour)
    # =========================================================================

    def run_evolution_cycle(self) -> Dict:
        """Run one complete evolution cycle."""
        self._log("EVOLUTION CYCLE: Starting...")

        if self.dry_run:
            self._log("  [DRY RUN] Would run evolution cycle")
            return {"dry_run": True}

        evolution = self._get_evolution()
        if not evolution:
            self._log("  Evolution engine not available", "WARN")
            return {"error": "Evolution engine not available"}

        try:
            cycle = evolution.run_cycle(symbols=self.symbols)
            self.state.cycles_run += 1
            self.state.last_evolution_cycle = datetime.now().isoformat()
            self._save_state()

            self._log(f"  Cycle #{evolution.cycle_count} complete")
            self._log(f"  Found {cycle.findings.get('patterns', 0)} patterns")
            self._log(f"  Made {cycle.findings.get('predictions', 0)} predictions")

            # Update paper trading if we have predictions
            paper = self._get_paper_trading()
            if paper and cycle.findings.get('predictions', 0) > 0:
                self._log(f"  Paper trading capital: ${paper.cash:.2f}")

            return asdict(cycle)

        except Exception as e:
            self.state.errors += 1
            self._save_state()
            self._log(f"  Evolution cycle failed: {e}", "ERROR")
            return {"error": str(e)}

    # =========================================================================
    # Daily Report (every 24 hours)
    # =========================================================================

    def generate_daily_report(self) -> str:
        """Generate a daily summary report."""
        self._log("DAILY REPORT: Generating...")

        if self.dry_run:
            self._log("  [DRY RUN] Would generate daily report")
            return "dry_run"

        report_date = datetime.now().strftime("%Y-%m-%d")
        report_file = REPORTS_DIR / f"midge_report_{report_date}.md"

        try:
            # Gather stats
            evolution = self._get_evolution()
            paper = self._get_paper_trading()

            report_lines = [
                f"# MIDGE Daily Report - {report_date}",
                "",
                "## System Status",
                f"- **Daemon Status**: {self.state.status}",
                f"- **Running Since**: {self.state.start_time}",
                f"- **Total Evolution Cycles**: {self.state.cycles_run}",
                f"- **Total Data Collections**: {self.state.data_collections}",
                f"- **Total Errors**: {self.state.errors}",
                "",
            ]

            # Evolution stats
            if evolution:
                memory = evolution.memory
                report_lines.extend([
                    "## Evolution Memory",
                    f"- **Total Cycles**: {memory.get('cycle_count', 0)}",
                    f"- **Successful Patterns**: {len(memory.get('successful_patterns', []))}",
                    f"- **Lessons Learned**: {len(memory.get('lessons', []))}",
                    "",
                ])

                # Last research
                if memory.get("last_research"):
                    research = memory["last_research"]
                    report_lines.extend([
                        "### Last Research Insights",
                        f"*Generated: {research.get('timestamp', 'unknown')}*",
                        "",
                    ])
                    for q in research.get("questions", [])[:3]:
                        report_lines.append(f"- {q}")
                    report_lines.append("")

            # Paper trading stats
            if paper:
                stats = paper.get_stats()
                report_lines.extend([
                    "## Paper Trading",
                    f"- **Starting Capital**: ${paper.starting_capital:,.2f}",
                    f"- **Current Cash**: ${paper.cash:,.2f}",
                    f"- **Open Positions**: {len(paper.positions)}",
                    f"- **Total Trades**: {stats.get('total_trades', 0)}",
                    f"- **Win Rate**: {stats.get('win_rate', 0):.1%}",
                    f"- **Total PnL**: ${stats.get('total_pnl', 0):,.2f}",
                    "",
                ])

                # Open positions
                if paper.positions:
                    report_lines.append("### Open Positions")
                    for symbol, pos in paper.positions.items():
                        report_lines.append(f"- {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
                    report_lines.append("")

            # Outcome tracking
            try:
                from trading.outcome_tracker import OutcomeTracker
                tracker = OutcomeTracker()
                overall = tracker.get_overall_stats()

                report_lines.extend([
                    "## Prediction Performance",
                    f"- **Total Predictions**: {overall.get('total_predictions', 0)}",
                    f"- **Resolved**: {overall.get('resolved', 0)}",
                    f"- **Accuracy**: {overall.get('accuracy', 0):.1%}",
                    f"- **Avg Return**: {overall.get('avg_return', 0):.2%}",
                    "",
                ])

                # Signal performance
                signal_perf = tracker.get_signal_performance()
                if signal_perf:
                    report_lines.append("### Signal Performance")
                    sorted_signals = sorted(signal_perf.items(),
                                          key=lambda x: x[1].get('accuracy', 0),
                                          reverse=True)
                    for signal, perf in sorted_signals[:5]:
                        acc = perf.get('accuracy', 0)
                        count = perf.get('predictions', 0)
                        report_lines.append(f"- **{signal}**: {acc:.1%} ({count} predictions)")
                    report_lines.append("")
            except Exception as e:
                self._log(f"  Could not get outcome stats: {e}", "WARN")

            # Footer
            report_lines.extend([
                "---",
                f"*Report generated at {datetime.now().isoformat()}*",
            ])

            # Write report
            report_content = "\n".join(report_lines)
            report_file.write_text(report_content, encoding="utf-8")

            self.state.reports_generated += 1
            self.state.last_daily_report = datetime.now().isoformat()
            self._save_state()

            self._log(f"  Report saved to: {report_file}")
            return str(report_file)

        except Exception as e:
            self.state.errors += 1
            self._save_state()
            self._log(f"  Report generation failed: {e}", "ERROR")
            return f"error: {e}"

    # =========================================================================
    # Main Loop
    # =========================================================================

    def run_once(self):
        """Run all components once."""
        self._log("=" * 60)
        self._log("MIDGE DAEMON - Running once")
        self._log("=" * 60)

        self.run_data_collection()
        self.run_evolution_cycle()
        self.generate_daily_report()

        self._log("=" * 60)
        self._log("MIDGE DAEMON - Complete")
        self._log("=" * 60)

    def run_continuous(self):
        """
        Run MIDGE continuously with proper scheduling.

        Schedule:
        - Data collection: every 5 minutes
        - Evolution cycle: every 1 hour
        - Daily report: every 24 hours
        """
        self._log("=" * 60)
        self._log("MIDGE DAEMON - Starting continuous operation")
        self._log(f"Symbols: {', '.join(self.symbols)}")
        self._log(f"Data collection: every {DATA_COLLECTION_INTERVAL}s")
        self._log(f"Evolution cycle: every {EVOLUTION_CYCLE_INTERVAL}s")
        self._log(f"Daily report: every {DAILY_REPORT_INTERVAL}s")
        self._log("=" * 60)

        self.state.status = "running"
        self._save_state()

        # Track when each task was last run
        last_data_collection = 0
        last_evolution = 0
        last_report = 0

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            self._log("\nReceived shutdown signal...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Initial runs
            self.run_data_collection()
            last_data_collection = time.time()

            self.run_evolution_cycle()
            last_evolution = time.time()

            self.generate_daily_report()
            last_report = time.time()

            # Main loop
            while not self.stop_event.is_set():
                now = time.time()

                # Check if data collection is due
                if now - last_data_collection >= DATA_COLLECTION_INTERVAL:
                    self.run_data_collection()
                    last_data_collection = now

                # Check if evolution cycle is due
                if now - last_evolution >= EVOLUTION_CYCLE_INTERVAL:
                    self.run_evolution_cycle()
                    last_evolution = now

                # Check if daily report is due
                if now - last_report >= DAILY_REPORT_INTERVAL:
                    self.generate_daily_report()
                    last_report = now

                # Sleep briefly to avoid busy-waiting
                time.sleep(10)

        except Exception as e:
            self._log(f"Daemon error: {e}", "ERROR")
            self.state.status = "error"
            self.state.errors += 1
        finally:
            self.state.status = "stopped"
            self._save_state()
            self._log("=" * 60)
            self._log("MIDGE DAEMON - Stopped")
            self._log(f"  Total cycles: {self.state.cycles_run}")
            self._log(f"  Total data collections: {self.state.data_collections}")
            self._log(f"  Total reports: {self.state.reports_generated}")
            self._log(f"  Total errors: {self.state.errors}")
            self._log("=" * 60)

    def show_status(self):
        """Show current daemon status."""
        print("\n" + "=" * 60)
        print("MIDGE DAEMON STATUS")
        print("=" * 60)

        print(f"\nStatus: {self.state.status}")
        print(f"Started: {self.state.start_time}")
        print(f"\nActivity:")
        print(f"  Evolution cycles: {self.state.cycles_run}")
        print(f"  Data collections: {self.state.data_collections}")
        print(f"  Reports generated: {self.state.reports_generated}")
        print(f"  Errors: {self.state.errors}")

        print(f"\nLast Activity:")
        print(f"  Data collection: {self.state.last_data_collection or 'never'}")
        print(f"  Evolution cycle: {self.state.last_evolution_cycle or 'never'}")
        print(f"  Daily report: {self.state.last_daily_report or 'never'}")

        # Show recent reports
        if REPORTS_DIR.exists():
            reports = sorted(REPORTS_DIR.glob("midge_report_*.md"), reverse=True)[:3]
            if reports:
                print(f"\nRecent Reports:")
                for r in reports:
                    print(f"  - {r.name}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MIDGE Master Daemon - 24/7 Trading Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Show status
    python trading/daemons/midge_daemon.py --status

    # Dry run (show what would happen)
    python trading/daemons/midge_daemon.py --dry-run --once

    # Run once (all components)
    python trading/daemons/midge_daemon.py --once

    # Run continuously (production)
    python trading/daemons/midge_daemon.py --continuous

    # With custom symbols
    python trading/daemons/midge_daemon.py --continuous --symbols AAPL MSFT GOOGL
        """
    )

    parser.add_argument("--once", action="store_true", help="Run all components once")
    parser.add_argument("--continuous", action="store_true", help="Run continuously (24/7)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    parser.add_argument("--symbols", nargs="+", help="Symbols to track")
    parser.add_argument("--report", action="store_true", help="Generate daily report only")
    parser.add_argument("--collect", action="store_true", help="Run data collection only")
    parser.add_argument("--evolve", action="store_true", help="Run evolution cycle only")

    args = parser.parse_args()

    daemon = MIDGEDaemon(
        symbols=args.symbols,
        dry_run=args.dry_run
    )

    if args.status:
        daemon.show_status()
    elif args.report:
        daemon.generate_daily_report()
    elif args.collect:
        daemon.run_data_collection()
    elif args.evolve:
        daemon.run_evolution_cycle()
    elif args.once:
        daemon.run_once()
    elif args.continuous:
        daemon.run_continuous()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
