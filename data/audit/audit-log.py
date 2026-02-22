"""
Mae Audit Log Appender
======================
Agents call this to log findings WITHOUT reading the file.

Usage:
    python data/audit/audit-log.py "AgentName" "CHECK|PASS|FAIL|INFO|WARN" "Detail message"

Examples:
    python data/audit/audit-log.py "test-runner" "PASS" "pytest: 1892/1892 passed, 0 failures"
    python data/audit/audit-log.py "system-checker" "FAIL" "system 'GoalManager' missing from bootstrap"
    python data/audit/audit-log.py "connection-auditor" "INFO" "285 connections found, 206 registered"
"""

import sys
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "mae-audit.md")

def append_entry(agent: str, status: str, detail: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"| {timestamp} | **{agent}** | `{status}` | {detail} |\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"[LOGGED] {timestamp} | {agent} | {status} | {detail}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python audit-log.py <agent> <status> <detail>")
        sys.exit(1)
    append_entry(sys.argv[1], sys.argv[2], " ".join(sys.argv[3:]))
