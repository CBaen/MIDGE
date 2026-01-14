#!/usr/bin/env python3
"""
research.py - THE SWIFT PATH (ultrathink-simple)

THE TRIAD + SYNTHESIZER → DIRECT ACTION

Flow:
1. Three agents research in parallel (Builder, Critic, Seeker)
2. Synthesizer reviews all three, produces ACTION_SPEC
3. ACTION_SPEC goes directly to implement.py

No re-analysis. Trust the Synthesizer. Move fast.
"""

import os
import sys
import json
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
QDRANT_STORE = Path.home() / ".claude" / "scripts" / "qdrant-store-v2.py"
COLLECTION = "emergence_self_knowledge"

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

VARIANT = "ultrathink-simple"
VARIANT_TAG = "THE SWIFT PATH"

TRIAD_ROLES = {
    "builder": {
        "name": "The Builder",
        "focus": "practical implementation, concrete steps, working code",
        "question": "How do we actually build this? What are the specific steps?",
        "style": "pragmatic, detailed, actionable"
    },
    "critic": {
        "name": "The Critic",
        "focus": "risks, gaps, failure modes, what could go wrong",
        "question": "What are we missing? What could fail? What are the risks?",
        "style": "skeptical, thorough, protective"
    },
    "seeker": {
        "name": "The Seeker",
        "focus": "unexplored possibilities, alternatives, what else exists",
        "question": "What else is possible? What haven't we considered?",
        "style": "curious, expansive, exploratory"
    }
}


def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""


def call_deepseek(prompt, role_name="agent"):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return f"ERROR: DEEPSEEK_API_KEY not set"
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0.7},
            timeout=120
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"ERROR: API {response.status_code}"
    except Exception as e:
        return f"ERROR: {e}"


def build_triad_prompt(role_key, limitation, goals, capabilities):
    role = TRIAD_ROLES[role_key]
    other_roles = [r for k, r in TRIAD_ROLES.items() if k != role_key]
    return f"""You are {role['name']} in {VARIANT_TAG}, a self-evolving AI system.

YOUR ROLE: {role['name']}
YOUR FOCUS: {role['focus']}
YOUR GUIDING QUESTION: {role['question']}

THE TRIAD: Two others research this simultaneously:
- {other_roles[0]['name']}: {other_roles[0]['focus']}
- {other_roles[1]['name']}: {other_roles[1]['focus']}

Focus on YOUR angle. Trust your partners.

---
LIMITATION TO ADDRESS:
{limitation}

SYSTEM GOALS:
{goals[:600]}

CURRENT CAPABILITIES:
{capabilities[:600]}
---

OUTPUT FORMAT:
---{role_key.upper()}---
## Analysis
[Your perspective]

## Key Insights
[What others might miss]

## Recommendations
[Concrete suggestions]
---END {role_key.upper()}---"""


def build_synthesizer_prompt(builder, critic, seeker, limitation, state):
    return f"""You are THE SYNTHESIZER in {VARIANT_TAG}.

Your job: Review all three perspectives and produce ONE clear ACTION_SPEC.
This ACTION_SPEC goes DIRECTLY to implementation. No more discussion.

THE TRIAD HAS SPOKEN:

=== THE BUILDER ===
{builder[:1500]}

=== THE CRITIC ===
{critic[:1500]}

=== THE SEEKER ===
{seeker[:1500]}

=== THE QUESTION ===
{limitation[:400]}

=== CURRENT STATE ===
{json.dumps(state, indent=2)[:800]}

YOUR TASK:
1. Identify the SINGLE most important action
2. Produce an ACTION_SPEC that can be executed directly
3. Be SPECIFIC - file path, action type, complete content

OUTPUT FORMAT (follow EXACTLY):

---SYNTHESIS---
## Agreement Points
[What they agree on]

## Conflicts Resolved
[Where they disagreed, your decision]

## Chosen Action
[Why this specific action]
---END SYNTHESIS---

---ACTION_SPEC---
decision: [One sentence - what we're doing]
why: [Why this is the right next step]
risk: LOW|MEDIUM|HIGH
file_path: [relative path e.g. knowledge/capabilities.md]
action: CREATE|REPLACE|APPEND
---END ACTION_SPEC---

---CONTENT---
[Complete file content to write]
---END CONTENT---"""


def research_as_triad(limitation):
    goals = read_file(KNOWLEDGE_DIR / "goals.md")
    capabilities = read_file(KNOWLEDGE_DIR / "capabilities.md")
    results = {}

    # Build prompts
    prompts = {role: build_triad_prompt(role, limitation, goals, capabilities) for role in TRIAD_ROLES}

    # ROUND 1: Triad in parallel
    print(f"[{VARIANT_TAG}] ROUND 1: The Triad researches...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(call_deepseek, prompts[r], TRIAD_ROLES[r]['name']): r for r in TRIAD_ROLES}
        for future in as_completed(futures):
            role = futures[future]
            results[role] = future.result()
            status = "OK" if "ERROR" not in results[role] else "FAIL"
            print(f"  [{status}] {TRIAD_ROLES[role]['name']}", file=sys.stderr)

    # ROUND 2: Synthesizer produces ACTION_SPEC
    error_count = sum(1 for r in results.values() if "ERROR" in str(r))
    if error_count < 3:
        print(f"[{VARIANT_TAG}] ROUND 2: Synthesizer decides...", file=sys.stderr)
        state = {"core_files": [f.name for f in (PROJECT_ROOT / "core").glob("*.py")]}
        synth_prompt = build_synthesizer_prompt(
            results.get("builder", "(No response)"),
            results.get("critic", "(No response)"),
            results.get("seeker", "(No response)"),
            limitation, state
        )
        results["synthesizer"] = call_deepseek(synth_prompt, "Synthesizer")
        print(f"  [OK] Synthesizer", file=sys.stderr)
    else:
        results["synthesizer"] = "ERROR: All Triad members failed"

    return results


def parse_action_spec(synthesis):
    """Extract ACTION_SPEC from Synthesizer output."""
    result = {"decision": None, "why": None, "risk": None, "file_path": None, "action": None, "content": None}
    try:
        if "---ACTION_SPEC---" in synthesis:
            spec = synthesis.split("---ACTION_SPEC---")[1].split("---END ACTION_SPEC---")[0]
            for line in spec.split("\n"):
                line = line.strip()
                if line.startswith("decision:"): result["decision"] = line.split(":", 1)[1].strip()
                elif line.startswith("why:"): result["why"] = line.split(":", 1)[1].strip()
                elif line.startswith("risk:"): result["risk"] = line.split(":", 1)[1].strip().upper()
                elif line.startswith("file_path:"): result["file_path"] = line.split(":", 1)[1].strip()
                elif line.startswith("action:"): result["action"] = line.split(":", 1)[1].strip().upper()
        if "---CONTENT---" in synthesis:
            result["content"] = synthesis.split("---CONTENT---")[1].split("---END CONTENT---")[0].strip("\n")
    except:
        pass
    return result


def store_research(content, topic):
    """Store in shared Qdrant."""
    try:
        subprocess.run(
            [sys.executable, str(QDRANT_STORE), topic, COLLECTION, VARIANT],
            input=content, capture_output=True, text=True, timeout=60
        )
    except:
        pass


def research(limitation_focus=None, store=True):
    """Main research function."""
    print(f"\n{'='*50}", file=sys.stderr)
    print(f"{VARIANT_TAG} - Research Cycle", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)

    # Get limitation
    if limitation_focus:
        limitation = limitation_focus
    else:
        lim_content = read_file(KNOWLEDGE_DIR / "limitations.md")
        if "**I don't know" in lim_content:
            start = lim_content.find("**I don't know")
            end = lim_content.find("**", start + 10)
            if end > start:
                end = lim_content.find("**", end + 2)
                limitation = lim_content[start:end+2].strip()
            else:
                limitation = lim_content[:500]
        else:
            limitation = lim_content[:500]

    print(f"Question: {limitation[:80]}...", file=sys.stderr)

    # Research
    results = research_as_triad(limitation)

    # Store if requested
    if store and results.get("synthesizer") and "ERROR" not in results["synthesizer"]:
        topic = f"{VARIANT}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        combined = f"# {VARIANT_TAG} Research\n\n"
        for role in ["builder", "critic", "seeker", "synthesizer"]:
            combined += f"\n## {role.upper()}\n{results.get(role, '(none)')}\n"
        store_research(combined, topic)

    # Parse ACTION_SPEC
    action_spec = parse_action_spec(results.get("synthesizer", ""))

    return {
        "triad": {k: results.get(k, "") for k in ["builder", "critic", "seeker"]},
        "synthesis": results.get("synthesizer", ""),
        "action_spec": action_spec
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    focus = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    result = research(focus)
    print(json.dumps(result, indent=2))
