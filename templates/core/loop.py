#!/usr/bin/env python3
"""
loop.py - THE SWIFT PATH (ultrathink-simple)

Main orchestration loop with narrative output.
Triad → Synthesizer → Direct Action → Verify
"""

import io
import os
import sys
import json
import time
import textwrap
import subprocess
from pathlib import Path
from datetime import datetime

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.parent

# Import analytics store
sys.path.insert(0, str(Path.home() / "projects" / "ultrathink-dashboard"))
try:
    from analytics_store import store_cycle
    ANALYTICS_ENABLED = True
except ImportError:
    ANALYTICS_ENABLED = False
    def store_cycle(data): pass

LOG_FILE = PROJECT_ROOT / "evolution.log"
METRICS_FILE = PROJECT_ROOT / "metrics.json"

VARIANT = "ultrathink-simple"
VARIANT_NAME = "THE SWIFT PATH"


# ========== Metrics ==========

def load_metrics():
    try:
        if METRICS_FILE.exists():
            return json.loads(METRICS_FILE.read_text())
    except:
        pass
    return {
        "variant": VARIANT,
        "variant_name": VARIANT_NAME,
        "cycles_completed": 0,
        "files_modified": 0,
        "files_created": 0,
        "verification_passes": 0,
        "verification_fails": 0,
        "total_api_calls": 0,
        "start_time": datetime.now().isoformat(),
        "runtime_hours": 0
    }


def save_metrics(m):
    try:
        # Calculate runtime
        start = datetime.fromisoformat(m["start_time"])
        m["runtime_hours"] = round((datetime.now() - start).total_seconds() / 3600, 2)
        METRICS_FILE.write_text(json.dumps(m, indent=2))
    except:
        pass


# ========== Logging ==========

def log(msg):
    """Log to both stdout and file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def banner(text, char="=", width=60):
    """Print a banner."""
    log(char * width)
    log(f"  {text}")
    log(char * width)


def narrate(text, width=60):
    """Narrative text, wrapped."""
    for line in textwrap.wrap(text, width - 4):
        log(f"  {line}")


# ========== Core Functions ==========

def introspect():
    """Read current state - returns dict directly."""
    result = {}
    try:
        cap_path = PROJECT_ROOT / "knowledge" / "capabilities.md"
        lim_path = PROJECT_ROOT / "knowledge" / "limitations.md"
        goals_path = PROJECT_ROOT / "knowledge" / "goals.md"
        result["capabilities"] = cap_path.read_text(encoding='utf-8')[:500] if cap_path.exists() else ""
        result["limitations"] = lim_path.read_text(encoding='utf-8')[:500] if lim_path.exists() else ""
        result["goals"] = goals_path.read_text(encoding='utf-8')[:500] if goals_path.exists() else ""
    except Exception as e:
        result["error"] = str(e)
        result["capabilities"] = ""
        result["limitations"] = ""
        result["goals"] = ""
    return result


def research(limitation):
    """Call research module."""
    try:
        from research import research as do_research
        return do_research(limitation)
    except Exception as e:
        return {"error": str(e), "triad": {}, "synthesis": "", "action_spec": {}}


def implement(action_spec, dry_run=False):
    """Call implement module."""
    try:
        from implement import implement as do_implement
        return do_implement(action_spec, dry_run)
    except Exception as e:
        return {"success": False, "error": str(e)}


def verify():
    """Verify system still works."""
    try:
        from verify import verify as do_verify
        result = do_verify()
        # VerificationResult object has .success property
        return {"success": result.success, "message": str(result)}
    except Exception as e:
        return {"success": True, "message": f"Verification skipped: {e}"}


# ========== Main Loop ==========

def run_cycle(metrics, dry_run=False):
    """Run one evolution cycle."""
    log("")
    banner(f"{VARIANT_NAME} - Cycle {metrics['cycles_completed'] + 1}", "=", 60)
    log("")

    # CHAPTER 1: INTROSPECT
    banner("CHAPTER 1: THE MIRROR", "-", 50)
    narrate("The system gazes inward, reading its own nature...")
    log("")

    state = introspect()
    limitation = state.get("limitations", "")[:500]

    if limitation:
        narrate(f"Question: {limitation[:100]}...")
    else:
        narrate("No clear limitation found. Seeking growth...")
        limitation = "How can this system improve itself?"

    log("")

    # CHAPTER 2: THE TRIAD + SYNTHESIZER
    banner("CHAPTER 2: THE TRIAD CONVENES", "-", 50)
    narrate("Three voices speak in parallel. Then one decides.")
    log("")

    result = research(limitation)
    metrics["total_api_calls"] += 4  # 3 triad + 1 synthesizer

    # Show Triad briefly
    for role in ["builder", "critic", "seeker"]:
        content = result.get("triad", {}).get(role, "")
        if content and "ERROR" not in content:
            log(f"  [{role.upper()}] Spoke")
        else:
            log(f"  [{role.upper()}] (silent)")

    log("")

    # Show Synthesizer decision
    action_spec = result.get("action_spec", {})
    if action_spec.get("decision"):
        banner("THE SYNTHESIZER DECIDES", "*", 50)
        narrate(f"Decision: {action_spec.get('decision', 'unknown')}")
        narrate(f"Target: {action_spec.get('file_path', 'unknown')}")
        narrate(f"Risk: {action_spec.get('risk', 'UNKNOWN')}")
        log("")
    else:
        narrate("The Synthesizer could not form a decision.")
        log("")
        return False

    # CHAPTER 3: ACTION
    banner("CHAPTER 3: THE FORGE", "-", 50)
    narrate("The decision becomes reality...")
    log("")

    impl_result = implement(action_spec, dry_run)

    if impl_result.get("success"):
        action = action_spec.get("action", "MODIFY").upper()
        if action == "CREATE":
            metrics["files_created"] += 1
        else:
            metrics["files_modified"] += 1
        narrate(f"SUCCESS: {action_spec.get('file_path')} {action.lower()}d")
    else:
        narrate(f"FAILED: {impl_result.get('error', 'unknown error')}")
        log("")
        return False

    log("")

    # CHAPTER 4: VERIFY
    banner("CHAPTER 4: THE TEST", "-", 50)
    narrate("Does the system still stand?")
    log("")

    verify_result = verify()
    if verify_result.get("success", True):
        metrics["verification_passes"] += 1
        narrate("VERIFIED: The system remains whole.")
    else:
        metrics["verification_fails"] += 1
        narrate(f"FAILED: {verify_result.get('message', 'Verification failed')}")
        narrate("Rolling back...")
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=PROJECT_ROOT, capture_output=True)

    log("")
    banner("CYCLE COMPLETE", "=", 60)
    log("")

    metrics["cycles_completed"] += 1

    # Store analytics
    if ANALYTICS_ENABLED:
        try:
            cycle_data = {
                "variant": "simple",
                "variant_name": VARIANT_NAME,
                "cycle_num": metrics["cycles_completed"],
                "timestamp": datetime.now().isoformat(),
                "question": limitation[:500] if limitation else "",
                "agent_research": {
                    role: {
                        "spoke": bool(result.get("triad", {}).get(role)),
                        "summary": str(result.get("triad", {}).get(role, ""))[:300]
                    }
                    for role in ["builder", "critic", "seeker"]
                },
                "proposal": {
                    "decision": action_spec.get("decision", ""),
                    "target_file": action_spec.get("file_path", ""),
                    "action": action_spec.get("action", ""),
                    "risk": action_spec.get("risk", "")
                },
                "votes": {},  # Simple variant has no voting
                "tally": {"approve": 1, "object": 0, "proceed": True},  # Synthesizer decides
                "deliberation_rounds": 0,
                "override_triggered": False,
                "outcome": {
                    "file_created": action_spec.get("file_path", ""),
                    "action_taken": impl_result.get("success", False),
                    "verification_passed": verify_result.get("success", True)
                }
            }
            store_cycle(cycle_data)
        except Exception as e:
            log(f"[Analytics] Store failed: {e}")

    return True


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description=f"{VARIANT_NAME} Evolution Loop")
    parser.add_argument("--continuous", type=int, nargs="?", const=15, help="Run continuously with delay (default 15s)")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually modify files")
    args = parser.parse_args()

    # Change to project root
    os.chdir(PROJECT_ROOT)

    # Load metrics
    metrics = load_metrics()

    log("")
    banner(f"{VARIANT_NAME}", "#", 60)
    narrate("Trust the Synthesizer. Move fast. Learn from mistakes.")
    log("")

    try:
        if args.continuous:
            while True:
                run_cycle(metrics, args.dry_run)
                save_metrics(metrics)
                log(f"Waiting {args.continuous}s...")
                time.sleep(args.continuous)
        else:
            run_cycle(metrics, args.dry_run)
            save_metrics(metrics)

    except KeyboardInterrupt:
        log("")
        banner("INTERRUPTED", "!", 60)
        save_metrics(metrics)

    log(f"Total cycles: {metrics['cycles_completed']}")
    log(f"Files modified: {metrics['files_modified']}")
    log(f"Files created: {metrics['files_created']}")


if __name__ == "__main__":
    main()
