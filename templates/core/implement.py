#!/usr/bin/env python3
"""
implement.py - THE SWIFT PATH (ultrathink-simple)

Receives ACTION_SPEC directly from Synthesizer.
No re-analysis. Execute immediately.
Safety comes from git, not hesitation.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VARIANT = "ultrathink-simple"


def git_commit(message):
    try:
        subprocess.run(["git", "add", "-A"], cwd=PROJECT_ROOT, capture_output=True)
        result = subprocess.run(["git", "commit", "-m", message], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""


def write_file(path, content):
    try:
        full_path = PROJECT_ROOT / path if not Path(path).is_absolute() else Path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except:
        return False


def implement(action_spec, dry_run=False):
    """
    Execute ACTION_SPEC directly.

    action_spec should contain:
    - decision: what we're doing
    - why: rationale
    - risk: LOW/MEDIUM/HIGH
    - file_path: target file
    - action: CREATE/REPLACE/APPEND
    - content: file content
    """
    results = {
        "success": False,
        "decision": action_spec.get("decision"),
        "target": action_spec.get("file_path"),
        "action": action_spec.get("action"),
        "why": action_spec.get("why"),
        "risk": action_spec.get("risk"),
        "error": None,
        "variant": VARIANT
    }

    # Validate
    if not action_spec.get("file_path"):
        results["error"] = "No file_path in ACTION_SPEC"
        return results
    if not action_spec.get("content"):
        results["error"] = "No content in ACTION_SPEC"
        return results

    print(f"[{VARIANT}] Decision: {action_spec.get('decision', 'unknown')}", file=sys.stderr)
    print(f"[{VARIANT}] Target: {action_spec.get('file_path')}", file=sys.stderr)
    print(f"[{VARIANT}] Action: {action_spec.get('action', 'REPLACE')}", file=sys.stderr)
    print(f"[{VARIANT}] Risk: {action_spec.get('risk', 'UNKNOWN')}", file=sys.stderr)

    if dry_run:
        print(f"[{VARIANT}] DRY RUN - no changes made", file=sys.stderr)
        results["success"] = True
        return results

    # Pre-change snapshot
    decision_short = (action_spec.get("decision") or "change")[:50]
    git_commit(f"[{VARIANT}] Pre: {decision_short}")

    # Apply change
    target_path = PROJECT_ROOT / action_spec["file_path"]
    action = action_spec.get("action", "REPLACE").upper()

    if action == "APPEND":
        existing = read_file(target_path) if target_path.exists() else ""
        success = write_file(action_spec["file_path"], existing + "\n" + action_spec["content"])
    else:
        success = write_file(action_spec["file_path"], action_spec["content"])

    if not success:
        results["error"] = f"Failed to write {action_spec['file_path']}"
        return results

    # Post-change commit
    git_commit(f"[{VARIANT}] Evolved: {decision_short}")

    results["success"] = True
    return results


if __name__ == "__main__":
    if "--rollback" in sys.argv:
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=PROJECT_ROOT)
        print("Rolled back")
    elif "--from-stdin" in sys.argv:
        data = json.loads(sys.stdin.read())
        action_spec = data.get("action_spec", data)
        result = implement(action_spec, "--dry-run" in sys.argv)
        print(json.dumps(result))
        sys.exit(0 if result["success"] else 1)
    else:
        print("Usage: python implement.py --from-stdin [--dry-run] | --rollback")
