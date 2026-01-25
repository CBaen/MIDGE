#!/usr/bin/env python3
"""
research_daemon.py - MIDGE Research Daemon

Continuous research generation for trading signals, patterns, and strategies.
Integrates with the lineage Gemini infrastructure.

Usage:
    # Run forever
    python scripts/research_daemon.py --continuous

    # Single batch
    python scripts/research_daemon.py --topics research_topics.txt

    # Generate from evolution questions
    python scripts/research_daemon.py --from-limitations --continuous

    # Check status
    python scripts/research_daemon.py --status

Topics are stored to midge_research collection in Qdrant.

Created: 2026-01-16 for MIDGE
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path.home() / ".claude" / "scripts"
STATE_DIR = PROJECT_ROOT / ".claude"
METRICS_FILE = STATE_DIR / "daemon-metrics.json"
LOG_FILE = STATE_DIR / "daemon.log"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DEFAULT_TOPICS_FILE = PROJECT_ROOT / "research_topics.txt"

# MIDGE-specific config
COLLECTION = "midge_research"
CONTEXT = """MIDGE - AI-powered trading signal research system
Tech Stack: Python, Qdrant vector DB, DeepSeek for Triad research, Gemini for knowledge acquisition
Focus: Options flow analysis, unusual activity detection, sentiment analysis, pattern recognition
Goal: Autonomous trading signal generation with self-improvement capabilities
Architecture: Evolution loop (introspect → research → implement → verify)"""

# Model pools (separate daily quotas)
MODEL_POOLS = [
    {"name": "2.5-flash-lite", "model": "gemini-2.5-flash-lite"},
    {"name": "3.0-flash", "model": "gemini-3-flash-preview"},
    {"name": "2.5-flash", "model": "gemini-2.5-flash"},
]


class DaemonState:
    """Persistent daemon state."""

    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.metrics = self._load()

    def _load(self):
        try:
            if METRICS_FILE.exists():
                return json.loads(METRICS_FILE.read_text())
        except:
            pass
        return {
            "topics_processed": 0,
            "topics_succeeded": 0,
            "topics_failed": 0,
            "chunks_stored": 0,
            "quota_hits": 0,
            "current_pool_index": 0,
            "pool_exhausted_until": {},
            "start_time": datetime.now().isoformat(),
            "last_run": None,
            "evolution_cycles": 0,
        }

    def save(self):
        self.metrics["last_run"] = datetime.now().isoformat()
        try:
            METRICS_FILE.write_text(json.dumps(self.metrics, indent=2))
        except:
            pass

    def get_available_model(self):
        """Get an available model, respecting pool exhaustion times."""
        now = datetime.now()

        for i, pool in enumerate(MODEL_POOLS):
            pool_name = pool["name"]

            if pool_name in self.metrics.get("pool_exhausted_until", {}):
                until_str = self.metrics["pool_exhausted_until"][pool_name]
                until = datetime.fromisoformat(until_str)
                if now < until:
                    continue
                else:
                    del self.metrics["pool_exhausted_until"][pool_name]

            self.metrics["current_pool_index"] = i
            return pool["model"]

        # All exhausted
        soonest = None
        for until_str in self.metrics.get("pool_exhausted_until", {}).values():
            until = datetime.fromisoformat(until_str)
            if soonest is None or until < soonest:
                soonest = until

        return None, soonest

    def mark_pool_exhausted(self, model, reset_seconds=None):
        """Mark a model pool as exhausted."""
        for pool in MODEL_POOLS:
            if pool["model"] == model:
                reset_time = datetime.now() + timedelta(seconds=reset_seconds or 54000)
                self.metrics["pool_exhausted_until"][pool["name"]] = reset_time.isoformat()
                self.metrics["quota_hits"] += 1
                log(f"Pool {pool['name']} exhausted until {reset_time.strftime('%H:%M')}", "WARN")
                break


def log(msg, level="INFO"):
    """Log to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def load_topics(topics_file):
    """Load pending topics from file."""
    topics = []
    try:
        with open(topics_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    topics.append(line)
    except Exception as e:
        log(f"Error loading topics: {e}", "ERROR")
    return topics


def generate_topics_from_limitations():
    """Generate research topics from MIDGE's limitations.md."""
    topics = []
    try:
        lim_file = KNOWLEDGE_DIR / "limitations.md"
        if lim_file.exists():
            content = lim_file.read_text(encoding="utf-8")
            # Extract "I don't know" questions
            for match in re.finditer(r"\*\*I don't know ([^*]+)\*\*", content):
                question = match.group(1).strip()
                topics.append(f"{question} - trading implementation")
            # Also look for TODO items
            for match in re.finditer(r"- \[ \] (.+)", content):
                todo = match.group(1).strip()
                topics.append(f"{todo} - research")
    except Exception as e:
        log(f"Error reading limitations: {e}", "ERROR")
    return topics


def mark_topic_done(topics_file, topic):
    """Mark a topic as done in the file."""
    try:
        content = Path(topics_file).read_text(encoding="utf-8")
        new_content = content.replace(f"\n{topic}\n", f"\n# DONE: {topic}\n")
        new_content = new_content.replace(f"{topic}\n", f"# DONE: {topic}\n")
        Path(topics_file).write_text(new_content, encoding="utf-8")
    except:
        pass


def parse_reset_time(output):
    """Parse quota reset time from error message."""
    match = re.search(r'(\d+)h(\d+)m(\d+)?s?', output)
    if match:
        h, m = int(match.group(1)), int(match.group(2))
        s = int(match.group(3)) if match.group(3) else 0
        return h * 3600 + m * 60 + s
    return None


def run_research(topic, account, model, session):
    """Run a single research query."""
    prompt = f"""DO NOT wrap in markdown. Return ONLY JSON.

You are an EXPERT TRADING SYSTEMS CONSULTANT for MIDGE.

CONTEXT: {CONTEXT}

Topic: {topic}

Return JSON:
{{
  "meta": {{"topic": "{topic}", "perspective": "trading-implementation", "research_type": "expert_consultation", "chunk_count": 3}},
  "summary": {{"text": "Executive summary for trading system implementation", "primary_recommendation": "ONE thing to implement first"}},
  "chunks": [
    {{"id": "chunk-01", "title": "Title", "content": "200-400 words with SPECIFIC trading system guidance", "keywords": [], "action_items": [], "importance": "core"}},
    {{"id": "chunk-02", "title": "Title", "content": "200-400 words", "keywords": [], "action_items": [], "importance": "supporting"}},
    {{"id": "chunk-03", "title": "Title", "content": "200-400 words", "keywords": [], "action_items": [], "importance": "advanced"}}
  ],
  "implementation_plan": {{
    "phases": [{{"phase": 1, "title": "", "tasks": [{{"task": "", "rationale": ""}}]}}],
    "critical_decisions": [],
    "risks": [],
    "success_criteria": []
  }}
}}"""

    script = SCRIPTS_DIR / "gemini-research-store.py"
    cmd = [
        sys.executable, str(script),
        "-a", str(account),
        "-c", COLLECTION,
        "-s", session,
        "-m", model,
        "-q", prompt
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            encoding='utf-8',
            errors='replace'
        )

        output = result.stdout + result.stderr

        if "exhausted your capacity" in output or "exhausted" in output.lower():
            reset_seconds = parse_reset_time(output)
            return {"success": False, "reason": "quota_exhausted", "reset_seconds": reset_seconds}

        try:
            data = json.loads(result.stdout)
            if data.get("success"):
                return {
                    "success": True,
                    "chunks_stored": data.get("chunks_stored", 0),
                    "topic": data.get("topic", topic)
                }
            else:
                return {"success": False, "reason": data.get("error", "unknown")}
        except json.JSONDecodeError:
            if result.returncode == 0:
                return {"success": True, "chunks_stored": 0}
            return {"success": False, "reason": f"Parse error: {output[:200]}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "reason": "timeout"}
    except Exception as e:
        return {"success": False, "reason": str(e)}


def run_daemon(topics_file, from_limitations, delay, continuous):
    """Main daemon loop."""
    state = DaemonState()
    session = f"midge-daemon-{datetime.now().strftime('%Y-%m-%d')}"
    account_cycle = 0

    log("=" * 60)
    log("MIDGE RESEARCH DAEMON")
    log(f"Collection: {COLLECTION}")
    log(f"Mode: {'Continuous' if continuous else 'Batch'}")
    log(f"Source: {'Limitations' if from_limitations else topics_file}")
    log(f"Delay: {delay}s between requests")
    log("=" * 60)

    try:
        while True:
            # Get topics
            if from_limitations:
                topics = generate_topics_from_limitations()
            else:
                topics = load_topics(topics_file) if topics_file else []

            if not topics:
                if continuous:
                    log("No topics available, waiting 5m for new limitations...")
                    time.sleep(300)
                    continue
                else:
                    log("All topics processed")
                    break

            topic = topics[0]

            # Get available model
            model_result = state.get_available_model()

            if isinstance(model_result, tuple):
                model, wake_time = model_result
                if model is None and wake_time:
                    sleep_sec = max(0, (wake_time - datetime.now()).total_seconds())
                    log(f"All pools exhausted. Sleeping {sleep_sec/60:.0f}m until {wake_time.strftime('%H:%M')}")
                    time.sleep(sleep_sec + 60)
                    continue
            else:
                model = model_result

            # Rotate accounts
            account = (account_cycle % 2) + 1
            account_cycle += 1

            log(f"Researching: {topic[:60]}...")
            log(f"  Account {account}, Model: {model}")

            # Run research
            result = run_research(topic, account, model, session)
            state.metrics["topics_processed"] += 1

            if result.get("reason") == "quota_exhausted":
                state.mark_pool_exhausted(model, result.get("reset_seconds"))
                state.save()
                continue

            if result.get("success"):
                state.metrics["topics_succeeded"] += 1
                state.metrics["chunks_stored"] += result.get("chunks_stored", 0)
                if topics_file and not from_limitations:
                    mark_topic_done(topics_file, topic)
                log(f"  SUCCESS: {result.get('chunks_stored', '?')} chunks stored")
            else:
                state.metrics["topics_failed"] += 1
                log(f"  FAILED: {result.get('reason', 'unknown')}", "ERROR")

            state.save()
            time.sleep(delay)

            if not continuous:
                if from_limitations:
                    topics = generate_topics_from_limitations()
                else:
                    topics = load_topics(topics_file) if topics_file else []
                if not topics:
                    break

    except KeyboardInterrupt:
        log("\nINTERRUPTED BY USER")

    # Final report
    log("")
    log("=" * 60)
    log("MIDGE DAEMON STOPPED")
    log(f"  Processed: {state.metrics['topics_processed']}")
    log(f"  Succeeded: {state.metrics['topics_succeeded']}")
    log(f"  Failed: {state.metrics['topics_failed']}")
    log(f"  Chunks: {state.metrics['chunks_stored']}")
    log(f"  Quota hits: {state.metrics['quota_hits']}")
    log("=" * 60)

    state.save()


def main():
    parser = argparse.ArgumentParser(
        description="MIDGE Research Daemon - Continuous trading knowledge generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run forever, generating from limitations.md
  python scripts/research_daemon.py --from-limitations --continuous

  # Run on specific topics file
  python scripts/research_daemon.py --topics research_topics.txt --continuous

  # Single batch from limitations
  python scripts/research_daemon.py --from-limitations

  # Check status
  python scripts/research_daemon.py --status
        """
    )

    parser.add_argument("--topics", help="Topics file (one per line)")
    parser.add_argument("--from-limitations", action="store_true",
                        help="Generate topics from knowledge/limitations.md")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between requests")
    parser.add_argument("--continuous", action="store_true", help="Run forever")
    parser.add_argument("--status", action="store_true", help="Show status and exit")

    args = parser.parse_args()

    if args.status:
        if METRICS_FILE.exists():
            print(METRICS_FILE.read_text())
        else:
            print("No daemon state found")
        return

    if not args.topics and not args.from_limitations:
        # Default to limitations if nothing specified
        args.from_limitations = True

    run_daemon(
        topics_file=args.topics,
        from_limitations=args.from_limitations,
        delay=args.delay,
        continuous=args.continuous
    )


if __name__ == "__main__":
    main()
