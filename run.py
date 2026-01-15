#!/usr/bin/env python3
"""
run.py - MIDGE Main Entry Point

Orchestrates all MIDGE services:
1. Data ingestion workers (SEC, contracts)
2. Evolution loop (pattern detection, learning)
3. Dashboard server

Usage:
    python run.py                    # Start all services
    python run.py --evolution-only   # Only run evolution loop
    python run.py --ingest-only      # Only run ingestion workers
    python run.py --dashboard-only   # Only run dashboard

Environment variables:
    MIDGE_NO_DASHBOARD=1    Skip dashboard server
    MIDGE_NO_INGEST=1       Skip ingestion workers
    MIDGE_NO_EVOLUTION=1    Skip evolution loop
"""

import os
import sys
import time
import signal
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Service configurations
SERVICES = {
    "technical_analysis": {
        "script": "scripts/ingest_technicals.py",
        "args": ["--continuous", "--interval", "600"],
        "description": "Technical indicators (every 10 min)"
    },
    "sec_ingestion": {
        "script": "scripts/ingest_sec_filings.py",
        "args": ["--continuous", "--interval", "3600"],
        "description": "SEC Form 4 ingestion (hourly)"
    },
    "contract_ingestion": {
        "script": "scripts/ingest_contracts.py",
        "args": ["--continuous", "--interval", "7200"],
        "description": "Government contract ingestion (every 2 hours)"
    },
    "evolution": {
        "script": "core/evolution.py",
        "args": ["--continuous", "--delay", "600"],
        "description": "Evolution loop (every 10 min)"
    },
    "dashboard": {
        "script": "dashboard/midge_server.py",
        "args": [],
        "description": "Dashboard server (port 8080)"
    }
}


class ServiceManager:
    """Manages MIDGE background services."""

    def __init__(self):
        self.processes = {}
        self.running = True

    def start_service(self, name: str, config: dict) -> bool:
        """Start a single service."""
        script_path = PROJECT_ROOT / config["script"]

        if not script_path.exists():
            print(f"  [ERROR] Script not found: {script_path}")
            return False

        print(f"  Starting {name}: {config['description']}")

        try:
            process = subprocess.Popen(
                [sys.executable, str(script_path)] + config["args"],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True
            )
            self.processes[name] = process
            print(f"    ✓ Started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"    ✗ Failed: {e}")
            return False

    def stop_all(self):
        """Stop all running services."""
        self.running = False
        print("\nStopping services...")

        for name, process in self.processes.items():
            if process.poll() is None:  # Still running
                print(f"  Stopping {name} (PID: {process.pid})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("All services stopped.")

    def monitor_output(self):
        """Monitor and print output from all services."""
        import select

        while self.running:
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    # Process died
                    print(f"\n[WARN] {name} exited with code {process.returncode}")
                    del self.processes[name]
                    continue

                # Try to read output (non-blocking)
                try:
                    line = process.stdout.readline()
                    if line:
                        print(f"[{name}] {line.rstrip()}")
                except:
                    pass

            time.sleep(0.1)

    def health_check(self) -> dict:
        """Check health of all services."""
        status = {}
        for name, process in self.processes.items():
            status[name] = "running" if process.poll() is None else "stopped"
        return status


def check_prerequisites() -> bool:
    """Check that required services are running."""
    print("\nChecking prerequisites...")

    checks = [
        ("Qdrant", "http://localhost:6333/collections"),
        ("Ollama", "http://localhost:11434/api/tags"),
    ]

    all_good = True
    for name, url in checks:
        try:
            import requests
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"  ✓ {name} is running")
            else:
                print(f"  ✗ {name} returned {response.status_code}")
                all_good = False
        except Exception as e:
            print(f"  ✗ {name} not reachable: {e}")
            all_good = False

    return all_good


def main():
    parser = argparse.ArgumentParser(description="MIDGE Service Orchestrator")
    parser.add_argument("--evolution-only", action="store_true", help="Only run evolution loop")
    parser.add_argument("--ingest-only", action="store_true", help="Only run ingestion workers")
    parser.add_argument("--dashboard-only", action="store_true", help="Only run dashboard")
    parser.add_argument("--no-prereq-check", action="store_true", help="Skip prerequisite check")

    args = parser.parse_args()

    print("="*60)
    print("MIDGE - Self-Improving Trading Intelligence")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not args.no_prereq_check:
        if not check_prerequisites():
            print("\n[ERROR] Prerequisites not met. Start Qdrant and Ollama first:")
            print("  docker start qdrant")
            print("  ollama serve")
            print("\nOr run with --no-prereq-check to skip this check.")
            sys.exit(1)

    # Determine which services to run
    services_to_run = []

    if args.evolution_only:
        services_to_run = ["evolution"]
    elif args.ingest_only:
        services_to_run = ["sec_ingestion", "contract_ingestion"]
    elif args.dashboard_only:
        services_to_run = ["dashboard"]
    else:
        # Run all services unless disabled via environment
        if not os.environ.get("MIDGE_NO_INGEST"):
            services_to_run.extend(["sec_ingestion", "contract_ingestion"])
        if not os.environ.get("MIDGE_NO_EVOLUTION"):
            services_to_run.append("evolution")
        if not os.environ.get("MIDGE_NO_DASHBOARD"):
            services_to_run.append("dashboard")

    if not services_to_run:
        print("\n[ERROR] No services selected to run!")
        sys.exit(1)

    # Start service manager
    manager = ServiceManager()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start services
    print("\nStarting services...")
    for name in services_to_run:
        if name in SERVICES:
            manager.start_service(name, SERVICES[name])

    print("\n" + "="*60)
    print("MIDGE is running. Press Ctrl+C to stop.")
    print("="*60)

    if "dashboard" in services_to_run:
        print("\nDashboard: http://localhost:8080")

    print("\nService status:")
    for name, status in manager.health_check().items():
        print(f"  {name}: {status}")

    # Monitor output
    print("\n--- Service Output ---\n")
    manager.monitor_output()


if __name__ == "__main__":
    main()
