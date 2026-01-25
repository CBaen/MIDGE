#!/usr/bin/env python3
"""
research.py - MIDGE TRADING RESEARCH

THE TRADING TRIAD + SYNTHESIZER → DIRECT ACTION

Flow:
1. Query Qdrant for relevant stored research (Bayesian, signals, patterns)
2. Three agents research in parallel (Hunter, Skeptic, Strategist)
3. Synthesizer reviews all three, produces ACTION_SPEC
4. ACTION_SPEC goes directly to implement.py

The Triad is informed by accumulated knowledge before analyzing.
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
QDRANT_STORE = Path.home() / ".claude" / "scripts" / "qdrant-store-gemini.py"
QDRANT_SEARCH = Path.home() / ".claude" / "scripts" / "qdrant-semantic-search.py"
COLLECTION = "midge_research"

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

VARIANT = "midge-trading"
VARIANT_TAG = "MIDGE TRADING RESEARCH"

# Trading-focused Triad - designed for pattern discovery and signal reliability
# Based on Qdrant research: Bull/Bear debate patterns, Bayesian reliability scoring
TRIAD_ROLES = {
    "hunter": {
        "name": "The Pattern Hunter",
        "focus": "novel signal discovery, unusual activity detection, leading indicators, patterns others miss",
        "question": "What patterns are emerging? What signals might indicate an opportunity before it becomes obvious?",
        "style": "observant, curious, exploratory, data-driven",
        "trading_context": """You look for:
- Unusual options flow (large volume, unusual strikes, timing)
- Price/volume divergences
- Sector rotation signals
- Sentiment shifts before price moves
- Cross-asset correlations breaking down or forming"""
    },
    "skeptic": {
        "name": "The Skeptic",
        "focus": "signal reliability assessment, false positive detection, confidence calibration, regime fit",
        "question": "Is this signal reliable? What's the actual confidence level? What could invalidate this?",
        "style": "probabilistic, questioning, protective, rigorous",
        "trading_context": """You evaluate:
- Historical win rate of similar signals (Bayesian posterior)
- Current market regime fit (trending vs ranging, vol regime)
- Potential for false positives (news-driven noise, manipulation)
- Corroboration requirement (never act on single signal)
- Confidence calibration (max 0.85, acknowledge uncertainty)"""
    },
    "strategist": {
        "name": "The Strategist",
        "focus": "signal combination, position sizing, entry/exit timing, learning from history",
        "question": "How do we act on this? What does our past performance suggest? What's the optimal approach?",
        "style": "tactical, integrative, decisive, learning-oriented",
        "trading_context": """You determine:
- How multiple signals should be weighted (ensemble methods)
- Position sizing based on reliability scores
- Entry/exit timing optimization
- What similar past predictions taught us
- How to improve the learning process itself"""
    }
}


def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""


def query_qdrant_research(topic, limit=3):
    """Query Qdrant for relevant stored research before Triad analyzes.

    Returns list of relevant research chunks that can inform the Triad's analysis.
    This is the key integration point - connecting accumulated knowledge to decisions.
    """
    try:
        result = subprocess.run(
            [sys.executable, str(QDRANT_SEARCH),
             "--collection", COLLECTION,
             "--query", topic,
             "--limit", str(limit),
             "--json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            results = json.loads(result.stdout)
            # Filter to chunks with decent relevance (score > 0.5)
            return [r for r in results if r.get("score", 0) > 0.5]
    except Exception as e:
        print(f"[WARN] Qdrant query failed: {e}", file=sys.stderr)
    return []


def format_research_context(research_results, max_chars=1500):
    """Format Qdrant results into context for Triad prompts.

    Extracts the most relevant insights from stored research to inform
    the Triad's analysis of the current question.
    """
    if not research_results:
        return ""

    context_parts = ["RELEVANT STORED RESEARCH (from midge_research):"]

    for i, result in enumerate(research_results[:3], 1):
        payload = result.get("payload", {})
        title = payload.get("title", "Research")
        text = payload.get("text", payload.get("content", ""))[:400]
        score = result.get("score", 0)

        context_parts.append(f"""
[{i}] {title} (relevance: {score:.2f})
{text}...""")

    context = "\n".join(context_parts)
    return context[:max_chars]


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


def build_triad_prompt(role_key, limitation, goals, capabilities, research_context=""):
    role = TRIAD_ROLES[role_key]
    other_roles = [r for k, r in TRIAD_ROLES.items() if k != role_key]
    trading_context = role.get('trading_context', '')

    # Include stored research if available
    research_section = ""
    if research_context:
        research_section = f"""
---
{research_context}
---
Use this stored research to inform your analysis. It contains insights from previous
research cycles, including Bayesian signal reliability patterns and trading strategies.
"""

    return f"""You are {role['name']} in {VARIANT_TAG}, an AI trading signal research system.

YOUR ROLE: {role['name']}
YOUR FOCUS: {role['focus']}
YOUR GUIDING QUESTION: {role['question']}
YOUR STYLE: {role['style']}

{trading_context}

THE TRADING TRIAD: Two others analyze this simultaneously:
- {other_roles[0]['name']}: {other_roles[0]['focus']}
- {other_roles[1]['name']}: {other_roles[1]['focus']}

Focus on YOUR angle. Trust your partners. Be specific to trading.
{research_section}
---
RESEARCH QUESTION:
{limitation}

TRADING SYSTEM GOALS:
{goals[:800]}

CURRENT CAPABILITIES:
{capabilities[:600]}
---

CRITICAL CONSTRAINTS:
- Never recommend action on a single signal (require 2+ corroborating)
- Maximum confidence is 0.85 (always acknowledge uncertainty)
- Consider current market regime (trending/ranging/volatile)
- Think in terms of signal reliability and Bayesian confidence

OUTPUT FORMAT:
---{role_key.upper()}---
## Analysis
[Your trading-focused perspective on this question]

## Signals & Patterns
[Specific signals, patterns, or metrics relevant to your role]

## Confidence Assessment
[How confident should we be? What would change your view?]

## Recommendations
[Concrete, actionable suggestions for the trading system]
---END {role_key.upper()}---"""


def build_synthesizer_prompt(hunter, skeptic, strategist, limitation, state):
    return f"""You are THE SYNTHESIZER in {VARIANT_TAG}, an AI trading signal research system.

Your job: Review all three trading perspectives and produce ONE clear ACTION_SPEC.
This ACTION_SPEC goes DIRECTLY to implementation. No more discussion.

THE TRADING TRIAD HAS ANALYZED:

=== THE PATTERN HUNTER ===
(Focus: Signal discovery, unusual activity, leading indicators)
{hunter[:1500]}

=== THE SKEPTIC ===
(Focus: Signal reliability, false positives, confidence calibration)
{skeptic[:1500]}

=== THE STRATEGIST ===
(Focus: Signal combination, position sizing, learning from history)
{strategist[:1500]}

=== THE RESEARCH QUESTION ===
{limitation[:400]}

=== CURRENT SYSTEM STATE ===
{json.dumps(state, indent=2)[:800]}

YOUR TASK:
1. Synthesize the three trading perspectives
2. Identify the SINGLE most important action to improve MIDGE's trading capability
3. Produce an ACTION_SPEC that can be executed directly
4. Be SPECIFIC - file path, action type, complete content

TRADING SYSTEM PRIORITIES:
- Pattern discovery > prediction count
- Signal reliability > signal quantity
- Confidence calibration > raw accuracy
- Learning capability > immediate performance

OUTPUT FORMAT (follow EXACTLY):

---SYNTHESIS---
## Agreement Points
[Where all three perspectives align]

## Conflicts Resolved
[Where they disagreed, your trading-focused decision]

## Signal Reliability Assessment
[Combined confidence level based on Skeptic's analysis]

## Chosen Action
[Why this specific action advances MIDGE's trading goals]
---END SYNTHESIS---

---ACTION_SPEC---
decision: [One sentence - what we're doing]
why: [Why this advances pattern discovery or signal reliability]
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

    # ROUND 0: Query Qdrant for relevant stored research
    print(f"[{VARIANT_TAG}] ROUND 0: Querying stored research...", file=sys.stderr)
    relevant_research = query_qdrant_research(limitation, limit=5)
    research_context = format_research_context(relevant_research)
    if research_context:
        print(f"  [OK] Found {len(relevant_research)} relevant research chunks", file=sys.stderr)
    else:
        print(f"  [--] No relevant stored research found", file=sys.stderr)

    # Build prompts with research context
    prompts = {role: build_triad_prompt(role, limitation, goals, capabilities, research_context) for role in TRIAD_ROLES}

    # ROUND 1: Triad in parallel
    print(f"[{VARIANT_TAG}] ROUND 1: The Trading Triad researches...", file=sys.stderr)
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
            results.get("hunter", "(No response)"),
            results.get("skeptic", "(No response)"),
            results.get("strategist", "(No response)"),
            limitation, state
        )
        results["synthesizer"] = call_deepseek(synth_prompt, "Synthesizer")
        print(f"  [OK] Synthesizer", file=sys.stderr)
    else:
        results["synthesizer"] = "ERROR: All Trading Triad members failed"

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


def store_research(triad_results, limitation, action_spec):
    """Store research in lineage format (chunked JSON for Qdrant)."""
    import tempfile

    # Build lineage-compatible JSON
    research_json = {
        "meta": {
            "topic": limitation[:100],
            "perspective": "triad-synthesis",
            "context": "midge-trading",
            "research_type": "expert_consultation",
            "depth": "comprehensive",
            "source": "deepseek-triad",
            "total_words": sum(len(str(v).split()) for v in triad_results.values()),
            "chunk_count": 4,
            "generated_at": datetime.now().isoformat()
        },
        "summary": {
            "text": action_spec.get("decision", "Triad research completed") + ". " + action_spec.get("why", ""),
            "keywords": ["trading", "midge", "signals", VARIANT],
            "primary_recommendation": action_spec.get("decision", "Review triad output")
        },
        "chunks": [
            {
                "id": "chunk-hunter",
                "title": "The Pattern Hunter: Signal Discovery",
                "content": str(triad_results.get("hunter", ""))[:1500],
                "keywords": ["patterns", "signals", "unusual-activity", "leading-indicators", "discovery"],
                "questions_answered": ["What patterns are emerging?", "What signals might others miss?"],
                "importance": "core",
                "action_items": []
            },
            {
                "id": "chunk-skeptic",
                "title": "The Skeptic: Reliability Assessment",
                "content": str(triad_results.get("skeptic", ""))[:1500],
                "keywords": ["reliability", "confidence", "false-positives", "regime-fit", "risk"],
                "questions_answered": ["Is this signal reliable?", "What's the actual confidence?"],
                "importance": "core",
                "action_items": []
            },
            {
                "id": "chunk-strategist",
                "title": "The Strategist: Action Plan",
                "content": str(triad_results.get("strategist", ""))[:1500],
                "keywords": ["strategy", "position-sizing", "timing", "learning", "execution"],
                "questions_answered": ["How do we act on this?", "What did past performance teach us?"],
                "importance": "core",
                "action_items": []
            },
            {
                "id": "chunk-synthesis",
                "title": "The Synthesizer: Trading Decision",
                "content": str(triad_results.get("synthesizer", ""))[:2000],
                "keywords": ["decision", "synthesis", "action", "trading"],
                "questions_answered": ["What's the combined view?", "What action should MIDGE take?"],
                "importance": "core",
                "action_items": [action_spec.get("decision", "")] if action_spec.get("decision") else []
            }
        ],
        "implementation_plan": {
            "phases": [{
                "phase": 1,
                "title": action_spec.get("decision", "Execute"),
                "tasks": [{
                    "task": action_spec.get("decision", ""),
                    "rationale": action_spec.get("why", ""),
                    "file_path": action_spec.get("file_path", ""),
                    "risk": action_spec.get("risk", "MEDIUM")
                }]
            }],
            "critical_decisions": [action_spec.get("decision", "")] if action_spec.get("decision") else [],
            "risks": [f"Risk level: {action_spec.get('risk', 'UNKNOWN')}"]
        }
    }

    # Write to temp file and store
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(research_json, f, indent=2)
            temp_path = f.name

        session = f"deepseek-triad-{datetime.now().strftime('%Y-%m-%d')}"
        result = subprocess.run(
            [sys.executable, str(QDRANT_STORE),
             "--collection", COLLECTION,
             "--session", session,
             "--input-file", temp_path],
            capture_output=True, text=True, timeout=60
        )

        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass

        if result.returncode == 0:
            print(f"[STORED] Research saved to {COLLECTION}", file=sys.stderr)
        else:
            print(f"[WARN] Storage issue: {result.stderr[:200]}", file=sys.stderr)

    except Exception as e:
        print(f"[ERROR] Store failed: {e}", file=sys.stderr)


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

    # Parse ACTION_SPEC
    action_spec = parse_action_spec(results.get("synthesizer", ""))

    # Store if requested (in lineage-compatible chunked format)
    if store and results.get("synthesizer") and "ERROR" not in results["synthesizer"]:
        store_research(results, limitation, action_spec)

    return {
        "triad": {k: results.get(k, "") for k in ["hunter", "skeptic", "strategist"]},
        "synthesis": results.get("synthesizer", ""),
        "action_spec": action_spec
    }


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    focus = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    result = research(focus)
    print(json.dumps(result, indent=2))
