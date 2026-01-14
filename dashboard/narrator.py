#!/usr/bin/env python3
"""
narrator.py - Transform cycle data into human-readable narratives

This module converts raw cycle analytics into documentary-style storytelling,
making the AI governance experiments accessible to human understanding.

The goal is not fiction - it's truth told in a way humans can feel.
These are real AI agents making real decisions. We tell their story.
"""

from typing import Optional
from agent_names import AGENTS, get_name, get_agent

# Agent personalities for narrative color
AGENT_VOICES = {
    "builder": {
        "style": "practical",
        "verbs": ["proposed", "suggested building", "outlined", "designed"],
        "adverbs": ["immediately", "pragmatically", "directly"]
    },
    "critic": {
        "style": "cautious",
        "verbs": ["warned", "questioned", "raised concerns about", "challenged"],
        "adverbs": ["cautiously", "skeptically", "carefully"]
    },
    "seeker": {
        "style": "curious",
        "verbs": ["wondered", "explored", "asked about", "considered"],
        "adverbs": ["curiously", "thoughtfully", "openly"]
    },
    "guardian": {
        "style": "protective",
        "verbs": ["guarded against", "protected", "flagged", "vetted"],
        "adverbs": ["vigilantly", "protectively", "firmly"]
    },
    "pragmatist": {
        "style": "efficient",
        "verbs": ["weighed", "calculated", "assessed", "evaluated"],
        "adverbs": ["efficiently", "practically", "rationally"]
    },
    "synthesizer": {
        "style": "wise",
        "verbs": ["concluded", "synthesized", "decided", "mediated"],
        "adverbs": ["thoughtfully", "wisely", "deliberately"]
    }
}

VARIANT_NAMES = {
    "simple": "THE SWIFT PATH",
    "review": "THE COUNCIL",
    "consensus": "THE PARLIAMENT"
}


def narrate_cycle(cycle_data: dict) -> str:
    """
    Transform a cycle into a narrative paragraph.

    Args:
        cycle_data: Full cycle data from analytics_store

    Returns:
        A human-readable narrative of what happened
    """
    variant = cycle_data.get("variant", "unknown")
    cycle_num = cycle_data.get("cycle_num", 0)
    variant_name = VARIANT_NAMES.get(variant, variant.upper())

    # Get the question/limitation
    question = cycle_data.get("question", cycle_data.get("limitation_addressed", "an unknown challenge"))
    if len(question) > 100:
        question = question[:97] + "..."

    # Get proposal info
    proposal = cycle_data.get("proposal", {})
    decision = proposal.get("decision", "make a change")
    target = proposal.get("target_file", "the system")
    action = proposal.get("action", "MODIFY")

    # Build the narrative
    parts = []

    # Opening
    parts.append(f"In Cycle {cycle_num}, {variant_name} faced a question: {question}")

    # Agent contributions
    agent_research = cycle_data.get("agent_research", {})
    contributions = []
    for agent_key, data in agent_research.items():
        if isinstance(data, dict) and data.get("spoke"):
            name = get_name(agent_key)
            summary = data.get("summary", "")[:80]
            if summary:
                voice = AGENT_VOICES.get(agent_key, {})
                verb = voice.get("verbs", ["said"])[0]
                contributions.append(f"{name} {verb}: \"{summary}\"")

    if contributions:
        parts.append(" ".join(contributions[:3]))  # Limit to 3 for readability

    # Voting (for Parliament)
    votes = cycle_data.get("votes", {})
    tally = cycle_data.get("tally", {})
    if votes and tally:
        approves = tally.get("approve", 0)
        objects = tally.get("object", 0)
        proceed = tally.get("proceed", False)

        if approves + objects > 0:
            parts.append(f"The vote was {approves}-{objects}. {'Motion carried.' if proceed else 'Motion failed.'}")

    # Outcome
    outcome = cycle_data.get("outcome", {})
    success = outcome.get("verification_passed", cycle_data.get("success", True))
    override = cycle_data.get("override_triggered", False)

    if override:
        parts.append("Sophia exercised her override authority.")

    action_word = "created" if action == "CREATE" else "modified" if action == "MODIFY" else "acted on"
    if success:
        parts.append(f"The action succeeded: {target} was {action_word}.")
    else:
        parts.append(f"The action failed. The system remained unchanged.")

    return " ".join(parts)


def narrate_cycle_brief(cycle_data: dict) -> str:
    """
    Create a one-line summary of a cycle.

    Args:
        cycle_data: Full cycle data

    Returns:
        A brief one-line summary
    """
    variant = cycle_data.get("variant", "unknown")
    cycle_num = cycle_data.get("cycle_num", 0)
    variant_name = VARIANT_NAMES.get(variant, variant)[:6]  # Short name

    proposal = cycle_data.get("proposal", {})
    target = proposal.get("target_file", "unknown")
    action = proposal.get("action", "?")

    tally = cycle_data.get("tally", {})
    if tally:
        approves = tally.get("approve", 0)
        objects = tally.get("object", 0)
        vote_str = f" [{approves}-{objects}]"
    else:
        vote_str = ""

    success = cycle_data.get("outcome", {}).get("verification_passed", True)
    status = "OK" if success else "FAIL"

    return f"[{variant_name}] Cycle {cycle_num}: {action} {target}{vote_str} -> {status}"


def narrate_agent_moment(agent_key: str, cycle_data: dict) -> Optional[str]:
    """
    Create a narrative moment from a specific agent's perspective.

    Args:
        agent_key: The agent key (builder, critic, etc.)
        cycle_data: The cycle data

    Returns:
        A narrative moment or None if agent didn't participate
    """
    agent_research = cycle_data.get("agent_research", {})
    agent_data = agent_research.get(agent_key, {})

    if not isinstance(agent_data, dict) or not agent_data.get("spoke"):
        return None

    name = get_name(agent_key)
    agent = get_agent(agent_key)
    summary = agent_data.get("summary", "")

    if not summary:
        return None

    voice = AGENT_VOICES.get(agent_key, {})
    adverb = voice.get("adverbs", [""])[0]

    return f"{name} ({agent.get('role', 'Agent')}) {adverb} observed: \"{summary[:150]}...\""


def narrate_vote_drama(cycle_data: dict) -> Optional[str]:
    """
    Create a narrative of the voting drama in a Parliament cycle.

    Args:
        cycle_data: The cycle data

    Returns:
        A narrative of the vote or None if no voting
    """
    votes = cycle_data.get("votes", {})
    if not votes:
        return None

    # Get the last round of voting
    last_round = None
    for round_name in sorted(votes.keys(), reverse=True):
        if isinstance(votes[round_name], dict):
            last_round = votes[round_name]
            break

    if not last_round:
        return None

    approvers = []
    objectors = []

    for agent_key, vote_data in last_round.items():
        if isinstance(vote_data, dict):
            name = get_name(agent_key)
            vote = vote_data.get("vote", "")
            confidence = vote_data.get("confidence", 0)

            if vote == "APPROVE":
                approvers.append((name, confidence))
            elif vote == "OBJECT":
                objectors.append((name, confidence))

    if not approvers and not objectors:
        return None

    parts = []

    if approvers:
        names = ", ".join([n for n, c in approvers])
        parts.append(f"In favor: {names}")

    if objectors:
        names = ", ".join([n for n, c in objectors])
        parts.append(f"Opposed: {names}")

    # Find most confident voter
    all_votes = approvers + objectors
    if all_votes:
        most_confident = max(all_votes, key=lambda x: x[1])
        if most_confident[1] > 0.8:
            parts.append(f"{most_confident[0]} was most certain.")

    return " ".join(parts)


def generate_episode_summary(cycles: list[dict], title: str = None) -> str:
    """
    Generate a summary of multiple cycles as an "episode".

    Args:
        cycles: List of cycle data
        title: Optional episode title

    Returns:
        A narrative summary of the episode
    """
    if not cycles:
        return "No cycles to summarize."

    # Group by variant
    by_variant = {}
    for cycle in cycles:
        v = cycle.get("variant", "unknown")
        if v not in by_variant:
            by_variant[v] = []
        by_variant[v].append(cycle)

    parts = []

    if title:
        parts.append(f"# {title}\n")
    else:
        cycle_range = f"{cycles[0].get('cycle_num', '?')}-{cycles[-1].get('cycle_num', '?')}"
        parts.append(f"# Cycles {cycle_range}\n")

    for variant, variant_cycles in by_variant.items():
        variant_name = VARIANT_NAMES.get(variant, variant)
        count = len(variant_cycles)

        # Count successes
        successes = sum(1 for c in variant_cycles
                       if c.get("outcome", {}).get("verification_passed", True))

        # Count files created
        creates = sum(1 for c in variant_cycles
                     if c.get("proposal", {}).get("action") == "CREATE")

        parts.append(f"## {variant_name}")
        parts.append(f"Completed {count} cycles. {successes} succeeded. {creates} files created.")

        # Include brief summaries
        for cycle in variant_cycles[:5]:  # Limit to 5 per variant
            parts.append(f"- {narrate_cycle_brief(cycle)}")

        if len(variant_cycles) > 5:
            parts.append(f"- ... and {len(variant_cycles) - 5} more cycles")

        parts.append("")

    return "\n".join(parts)


# ==================== CLI ====================

if __name__ == "__main__":
    # Demo with sample data
    sample_cycle = {
        "variant": "consensus",
        "cycle_num": 42,
        "question": "How can we access external information?",
        "agent_research": {
            "builder": {"spoke": True, "summary": "We need HTTP capabilities. I can build it."},
            "critic": {"spoke": True, "summary": "External requests are a security risk."},
            "seeker": {"spoke": True, "summary": "What about using existing tools instead?"},
            "guardian": {"spoke": True, "summary": "Untrusted data could compromise us."},
            "pragmatist": {"spoke": True, "summary": "The benefit outweighs the cost if secured."}
        },
        "proposal": {
            "decision": "Create web_client.py with rate limiting",
            "target_file": "web_client.py",
            "action": "CREATE"
        },
        "votes": {
            "round_1": {
                "builder": {"vote": "APPROVE", "confidence": 0.85},
                "critic": {"vote": "OBJECT", "confidence": 0.90},
                "seeker": {"vote": "APPROVE", "confidence": 0.70},
                "guardian": {"vote": "OBJECT", "confidence": 0.95},
                "pragmatist": {"vote": "APPROVE", "confidence": 0.80}
            }
        },
        "tally": {"approve": 3, "object": 2, "proceed": True},
        "override_triggered": False,
        "outcome": {"verification_passed": True}
    }

    print("=== Full Narrative ===")
    print(narrate_cycle(sample_cycle))
    print()

    print("=== Brief Summary ===")
    print(narrate_cycle_brief(sample_cycle))
    print()

    print("=== Vote Drama ===")
    print(narrate_vote_drama(sample_cycle))
    print()

    print("=== Agent Moments ===")
    for agent in ["builder", "critic", "guardian"]:
        moment = narrate_agent_moment(agent, sample_cycle)
        if moment:
            print(f"  {moment}")
