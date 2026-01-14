#!/usr/bin/env python3
"""
analytics_store.py - Store and retrieve cycle analytics from Qdrant

Stores per-cycle data for pattern recognition and historical analysis.
"""

import json
import hashlib
import requests
from datetime import datetime
from pathlib import Path

QDRANT_URL = "http://localhost:6333"
COLLECTION = "ultrathink_analytics"
VECTOR_SIZE = 768

# Import agent names
try:
    from agent_names import get_name, AGENTS
except ImportError:
    # Fallback if imported from different location
    def get_name(role): return role.title()
    AGENTS = {}


def ensure_collection():
    """Create collection if it doesn't exist."""
    try:
        # Check if exists
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}")
        if resp.status_code == 200:
            return True

        # Create collection
        requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            json={
                "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}
            }
        )
        return True
    except Exception as e:
        print(f"[analytics_store] Qdrant not available: {e}")
        return False


def generate_embedding(text):
    """Generate deterministic pseudo-embedding from text hash."""
    # Use SHA256 hash to generate consistent 768-dim vector
    hash_bytes = hashlib.sha256(text.encode()).digest()
    # Extend to 768 dimensions by repeating and normalizing
    extended = []
    for i in range(VECTOR_SIZE):
        byte_idx = i % len(hash_bytes)
        val = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # Normalize to [-1, 1]
        extended.append(val)
    return extended


def store_cycle(cycle_data):
    """
    Store a cycle record in Qdrant.

    cycle_data should include:
    - variant: "simple" | "review" | "consensus"
    - cycle_num: int
    - timestamp: ISO string
    - question: str (the limitation addressed)
    - agent_research: dict of agent responses
    - proposal: dict with decision, target_file, action, risk
    - votes: dict (for consensus/review variants)
    - tally: dict with approve/object/proceed
    - outcome: dict with file_created, verification_passed
    """
    if not ensure_collection():
        return False

    try:
        # Generate unique ID
        cycle_id = f"{cycle_data.get('variant', 'unknown')}-{cycle_data.get('cycle_num', 0)}"
        point_id = abs(hash(cycle_id)) % (2**63)

        # Create searchable text for embedding
        searchable = f"""
        Variant: {cycle_data.get('variant', '')}
        Question: {cycle_data.get('question', '')}
        Proposal: {cycle_data.get('proposal', {}).get('decision', '')}
        Outcome: {'Success' if cycle_data.get('outcome', {}).get('verification_passed') else 'Failed'}
        """

        # Add agent summaries to searchable text
        for role, data in cycle_data.get('agent_research', {}).items():
            name = get_name(role)
            summary = data.get('summary', '') if isinstance(data, dict) else str(data)[:200]
            searchable += f"\n{name}: {summary}"

        embedding = generate_embedding(searchable)

        # Flatten nested dicts for Qdrant payload
        payload = {
            "cycle_id": cycle_id,
            "variant": cycle_data.get('variant', 'unknown'),
            "variant_name": cycle_data.get('variant_name', ''),
            "cycle_num": cycle_data.get('cycle_num', 0),
            "timestamp": cycle_data.get('timestamp', datetime.now().isoformat()),
            "question": cycle_data.get('question', '')[:500],

            # Proposal
            "proposal_decision": cycle_data.get('proposal', {}).get('decision', ''),
            "proposal_file": cycle_data.get('proposal', {}).get('target_file', ''),
            "proposal_action": cycle_data.get('proposal', {}).get('action', ''),
            "proposal_risk": cycle_data.get('proposal', {}).get('risk', ''),

            # Voting results (flatten)
            "tally_approve": cycle_data.get('tally', {}).get('approve', 0),
            "tally_object": cycle_data.get('tally', {}).get('object', 0),
            "tally_proceed": cycle_data.get('tally', {}).get('proceed', False),
            "deliberation_rounds": cycle_data.get('deliberation_rounds', 0),
            "override_triggered": cycle_data.get('override_triggered', False),

            # Outcome
            "file_created": cycle_data.get('outcome', {}).get('file_created', ''),
            "action_taken": cycle_data.get('outcome', {}).get('action_taken', False),
            "verification_passed": cycle_data.get('outcome', {}).get('verification_passed', True),

            # Store full data as JSON string for retrieval
            "full_data": json.dumps(cycle_data, default=str)
        }

        # Store individual agent votes if present
        votes = cycle_data.get('votes', {})
        if votes:
            for round_name, round_votes in votes.items():
                if isinstance(round_votes, dict):
                    for agent, vote_data in round_votes.items():
                        if isinstance(vote_data, dict):
                            payload[f"vote_{agent}"] = vote_data.get('vote', '')
                            payload[f"confidence_{agent}"] = vote_data.get('confidence', 0)

        # Upsert to Qdrant
        resp = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={
                "points": [{
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload
                }]
            }
        )

        return resp.status_code == 200

    except Exception as e:
        print(f"[analytics_store] Error storing cycle: {e}")
        return False


def get_cycles(variant=None, limit=50, offset=0):
    """Retrieve cycles, optionally filtered by variant."""
    if not ensure_collection():
        return []

    try:
        filter_clause = None
        if variant:
            filter_clause = {
                "must": [{"key": "variant", "match": {"value": variant}}]
            }

        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json={
                "limit": limit,
                "offset": offset,
                "filter": filter_clause,
                "with_payload": True,
                "with_vector": False
            }
        )

        if resp.status_code == 200:
            data = resp.json()
            points = data.get("result", {}).get("points", [])
            # Parse full_data JSON back to dict
            cycles = []
            for p in points:
                payload = p.get("payload", {})
                if "full_data" in payload:
                    try:
                        full = json.loads(payload["full_data"])
                        cycles.append(full)
                    except:
                        cycles.append(payload)
                else:
                    cycles.append(payload)
            # Sort by cycle_num descending
            cycles.sort(key=lambda x: x.get("cycle_num", 0), reverse=True)
            return cycles
        return []

    except Exception as e:
        print(f"[analytics_store] Error retrieving cycles: {e}")
        return []


def get_cycle(cycle_id):
    """Get a specific cycle by ID."""
    cycles = get_cycles(limit=1000)  # Get all and filter
    for c in cycles:
        if c.get("cycle_id") == cycle_id:
            return c
        # Also check constructed ID
        variant = c.get("variant", "")
        num = c.get("cycle_num", 0)
        if f"{variant}-{num}" == cycle_id:
            return c
    return None


def get_agent_stats(agent_role, variant=None):
    """Compute stats for a specific agent across cycles."""
    cycles = get_cycles(variant=variant, limit=1000)

    stats = {
        "name": get_name(agent_role),
        "role": agent_role,
        "total_cycles": 0,
        "votes_approve": 0,
        "votes_object": 0,
        "times_spoke": 0,
        "agreement_with": {}
    }

    for cycle in cycles:
        # Check if agent participated
        research = cycle.get("agent_research", {})
        if agent_role in research:
            stats["total_cycles"] += 1
            if research[agent_role].get("spoke"):
                stats["times_spoke"] += 1

        # Check votes
        votes = cycle.get("votes", {})
        for round_votes in votes.values():
            if isinstance(round_votes, dict) and agent_role in round_votes:
                vote = round_votes[agent_role].get("vote", "")
                if vote == "APPROVE":
                    stats["votes_approve"] += 1
                elif vote == "OBJECT":
                    stats["votes_object"] += 1

    # Compute rates
    total_votes = stats["votes_approve"] + stats["votes_object"]
    if total_votes > 0:
        stats["approve_rate"] = round(stats["votes_approve"] / total_votes, 2)
        stats["object_rate"] = round(stats["votes_object"] / total_votes, 2)
    else:
        stats["approve_rate"] = 0.5
        stats["object_rate"] = 0.5

    return stats


def compute_agreement_matrix(variant=None):
    """Compute pairwise agreement between agents."""
    cycles = get_cycles(variant=variant, limit=1000)

    agents = ["builder", "critic", "seeker", "guardian", "pragmatist"]
    matrix = {a: {b: {"agree": 0, "total": 0} for b in agents} for a in agents}

    for cycle in cycles:
        votes = cycle.get("votes", {})
        # Get final round votes
        final_votes = {}
        for round_votes in votes.values():
            if isinstance(round_votes, dict):
                final_votes.update(round_votes)

        # Compare pairs
        for a in agents:
            for b in agents:
                if a >= b:
                    continue  # Only upper triangle
                if a in final_votes and b in final_votes:
                    vote_a = final_votes[a].get("vote", "")
                    vote_b = final_votes[b].get("vote", "")
                    if vote_a and vote_b:
                        matrix[a][b]["total"] += 1
                        matrix[b][a]["total"] += 1
                        if vote_a == vote_b:
                            matrix[a][b]["agree"] += 1
                            matrix[b][a]["agree"] += 1

    # Convert to percentages
    result = {}
    for a in agents:
        result[a] = {}
        for b in agents:
            if a == b:
                result[a][b] = 1.0
            elif matrix[a][b]["total"] > 0:
                result[a][b] = round(matrix[a][b]["agree"] / matrix[a][b]["total"], 2)
            else:
                result[a][b] = 0.5  # No data

    return result


if __name__ == "__main__":
    # Test connection
    print("Testing Qdrant connection...")
    if ensure_collection():
        print(f"Collection '{COLLECTION}' ready")

        # Test store
        test_cycle = {
            "variant": "test",
            "variant_name": "Test Variant",
            "cycle_num": 1,
            "timestamp": datetime.now().isoformat(),
            "question": "Test question",
            "agent_research": {
                "builder": {"spoke": True, "summary": "Test summary"},
            },
            "proposal": {"decision": "Test proposal", "target_file": "test.py", "action": "CREATE", "risk": "LOW"},
            "tally": {"approve": 3, "object": 2, "proceed": True},
            "outcome": {"file_created": "test.py", "action_taken": True, "verification_passed": True}
        }

        if store_cycle(test_cycle):
            print("Test cycle stored successfully")
        else:
            print("Failed to store test cycle")
    else:
        print("Qdrant not available")
