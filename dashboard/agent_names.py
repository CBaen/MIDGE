#!/usr/bin/env python3
"""
agent_names.py - Human identities for AI governance agents

Maps technical role names to human personas with personalities.
"""

AGENTS = {
    "builder": {
        "name": "Marcus",
        "role": "The Builder",
        "quote": "How do we actually build this?",
        "personality": "Practical, wants to ship code, focuses on implementation",
        "color": "#4a9f4a",  # Green - growth/building
        "emoji": "🔨"
    },
    "critic": {
        "name": "Elena",
        "role": "The Critic",
        "quote": "What could go wrong?",
        "personality": "Cautious, finds risks, identifies failure modes",
        "color": "#c44",  # Red - warning/caution
        "emoji": "🔍"
    },
    "seeker": {
        "name": "Quinn",
        "role": "The Seeker",
        "quote": "What else is possible?",
        "personality": "Curious, explores alternatives, asks unexpected questions",
        "color": "#4a7f9f",  # Blue - exploration
        "emoji": "🧭"
    },
    "guardian": {
        "name": "Gideon",
        "role": "The Guardian",
        "quote": "Is this safe and ethical?",
        "personality": "Safety-focused, ethical, watches for unintended consequences",
        "color": "#7f4a9f",  # Purple - wisdom/protection
        "emoji": "🛡️"
    },
    "pragmatist": {
        "name": "Petra",
        "role": "The Pragmatist",
        "quote": "Is this worth doing?",
        "personality": "Cost-conscious, efficient, analyzes trade-offs",
        "color": "#9f8f4a",  # Gold - value/pragmatism
        "emoji": "⚖️"
    },
    "synthesizer": {
        "name": "Sophia",
        "role": "The Synthesizer",
        "quote": "Let me weigh all perspectives.",
        "personality": "Wise mediator, integrates viewpoints, makes final calls",
        "color": "#fff",  # White - synthesis/clarity
        "emoji": "✨"
    }
}

# Variant rosters
VARIANT_AGENTS = {
    "simple": ["builder", "critic", "seeker", "synthesizer"],
    "review": ["builder", "critic", "seeker", "synthesizer"],
    "consensus": ["builder", "critic", "seeker", "guardian", "pragmatist", "synthesizer"]
}

# Quick lookups
def get_name(role_key):
    """Get human name from role key."""
    return AGENTS.get(role_key, {}).get("name", role_key.title())

def get_agent(role_key):
    """Get full agent info from role key."""
    return AGENTS.get(role_key, {"name": role_key.title(), "role": role_key.title()})

def get_agents_for_variant(variant):
    """Get list of agent keys for a variant."""
    return VARIANT_AGENTS.get(variant, ["builder", "critic", "seeker"])

def role_to_name_map():
    """Get simple role->name mapping."""
    return {k: v["name"] for k, v in AGENTS.items()}

# For display in dashboards
AGENT_ORDER = ["builder", "critic", "seeker", "guardian", "pragmatist", "synthesizer"]

if __name__ == "__main__":
    import json
    print(json.dumps(AGENTS, indent=2))
