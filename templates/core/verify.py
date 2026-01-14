#!/usr/bin/env python3
"""
verify.py - Self-testing module for Emergence

Tests that the system still works after modifications.
Returns pass/fail status for the evolution loop.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CORE_DIR = PROJECT_ROOT / "core"


class VerificationResult:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0

    def add(self, name, passed, message=""):
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    @property
    def success(self):
        return self.failed == 0

    def __str__(self):
        lines = ["=" * 60, "VERIFICATION RESULTS", "=" * 60, ""]
        for test in self.tests:
            status = "PASS" if test["passed"] else "FAIL"
            lines.append(f"[{status}] {test['name']}")
            if test["message"]:
                lines.append(f"       {test['message']}")
        lines.append("")
        lines.append(f"Total: {self.passed} passed, {self.failed} failed")
        lines.append("=" * 60)
        if self.success:
            lines.append("VERIFICATION PASSED")
        else:
            lines.append("VERIFICATION FAILED - ROLLBACK RECOMMENDED")
        return "\n".join(lines)


def test_core_modules_syntax():
    """Test that all core modules have valid Python syntax."""
    results = []
    for py_file in CORE_DIR.glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            # Just check syntax, don't execute
            compile(open(py_file).read(), py_file, 'exec')
            results.append((py_file.name, True, ""))
        except SyntaxError as e:
            results.append((py_file.name, False, str(e)))
        except Exception as e:
            results.append((py_file.name, False, str(e)))
    return results


def test_knowledge_files_exist():
    """Test that required knowledge files exist."""
    required = [
        "knowledge/capabilities.md",
        "knowledge/limitations.md",
        "knowledge/goals.md"
    ]
    results = []
    for path in required:
        full_path = PROJECT_ROOT / path
        exists = full_path.exists()
        results.append((path, exists, "" if exists else "File not found"))
    return results


def test_introspect_runs():
    """Test that introspect.py can run without error."""
    try:
        result = subprocess.run(
            [sys.executable, str(CORE_DIR / "introspect.py"), "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return True, "Introspection successful"
        else:
            return False, f"Exit code {result.returncode}: {result.stderr[:200]}"
    except Exception as e:
        return False, str(e)


def test_git_status():
    """Test that git is working."""
    try:
        result = subprocess.run(
            ["git", "status"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return True, "Git operational"
        else:
            return False, "Git error"
    except Exception as e:
        return False, str(e)


def test_qdrant_connection():
    """Test that Qdrant is reachable."""
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            return True, "Qdrant connected"
        else:
            return False, f"Qdrant returned {response.status_code}"
    except Exception as e:
        return False, str(e)


def test_ollama_connection():
    """Test that Ollama is reachable."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Ollama connected"
        else:
            return False, f"Ollama returned {response.status_code}"
    except Exception as e:
        return False, str(e)


def verify():
    """Run all verification tests."""
    result = VerificationResult()

    # Test core module syntax
    for name, passed, msg in test_core_modules_syntax():
        result.add(f"Syntax: {name}", passed, msg)

    # Test knowledge files
    for name, passed, msg in test_knowledge_files_exist():
        result.add(f"File: {name}", passed, msg)

    # Test introspection
    passed, msg = test_introspect_runs()
    result.add("Introspection runs", passed, msg)

    # Test git
    passed, msg = test_git_status()
    result.add("Git operational", passed, msg)

    # Test Qdrant (non-critical)
    passed, msg = test_qdrant_connection()
    result.add("Qdrant connection", passed, msg)

    # Test Ollama (non-critical)
    passed, msg = test_ollama_connection()
    result.add("Ollama connection", passed, msg)

    return result


if __name__ == "__main__":
    result = verify()
    print(result)

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)
