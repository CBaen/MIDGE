#!/usr/bin/env python3
"""
Ultrathink Dashboard Server
Serves metrics, agent analytics, and cycle history for the Observatory.
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Import analytics modules
from agent_names import AGENTS, get_name, get_agent, VARIANT_AGENTS, AGENT_ORDER
from analytics_store import (
    get_cycles, get_cycle, get_agent_stats,
    compute_agreement_matrix, ensure_collection
)
from narrator import (
    narrate_cycle, narrate_cycle_brief, narrate_vote_drama,
    narrate_agent_moment, generate_episode_summary
)

PROJECTS_DIR = Path("C:/Users/baenb/projects")
VARIANTS = {
    "simple": {"path": "ultrathink-simple", "name": "THE SWIFT PATH", "color": "#4a9f4a"},
    "review": {"path": "ultrathink-review", "name": "THE COUNCIL", "color": "#4a7f9f"},
    "consensus": {"path": "ultrathink-consensus", "name": "THE PARLIAMENT", "color": "#7f4a9f"}
}

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        # Original endpoints
        if parsed.path == "/api/metrics":
            self.send_json(self.get_all_metrics())
        elif parsed.path == "/api/logs":
            variant = params.get("variant", ["simple"])[0]
            lines = int(params.get("lines", [30])[0])
            self.send_json(self.get_logs(variant, lines))

        # New Observatory endpoints
        elif parsed.path == "/api/agents":
            self.send_json(self.get_all_agents())
        elif parsed.path.startswith("/api/agent/"):
            agent_key = parsed.path.split("/")[-1]
            variant = params.get("variant", [None])[0]
            self.send_json(self.get_agent_profile(agent_key, variant))
        elif parsed.path == "/api/cycles":
            variant = params.get("variant", [None])[0]
            limit = int(params.get("limit", [50])[0])
            self.send_json(self.get_cycle_history(variant, limit))
        elif parsed.path.startswith("/api/cycle/"):
            cycle_id = parsed.path.split("/")[-1]
            self.send_json(self.get_cycle_detail(cycle_id))
        elif parsed.path == "/api/relationships":
            variant = params.get("variant", [None])[0]
            self.send_json(self.get_relationships(variant))
        elif parsed.path == "/api/files":
            variant = params.get("variant", ["simple"])[0]
            self.send_json(self.get_created_files(variant))
        elif parsed.path == "/api/stories":
            variant = params.get("variant", [None])[0]
            limit = int(params.get("limit", [20])[0])
            self.send_json(self.get_stories(variant, limit))

        # Dashboard
        elif parsed.path == "/" or parsed.path == "/index.html":
            self.serve_dashboard()
        else:
            super().do_GET()

    def send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_all_metrics(self):
        result = {}
        for key, info in VARIANTS.items():
            metrics_file = PROJECTS_DIR / info["path"] / "metrics.json"
            if metrics_file.exists():
                try:
                    result[key] = json.loads(metrics_file.read_text())
                    result[key]["name"] = info["name"]
                    result[key]["color"] = info["color"]
                except:
                    result[key] = {"error": "Could not read metrics", "name": info["name"], "color": info["color"]}
            else:
                result[key] = {"cycles_completed": 0, "name": info["name"], "color": info["color"], "not_started": True}
        return result

    def get_logs(self, variant, lines=30):
        if variant not in VARIANTS:
            return {"error": "Unknown variant"}
        log_file = PROJECTS_DIR / VARIANTS[variant]["path"] / "evolution.log"
        if not log_file.exists():
            return {"lines": [], "variant": variant}
        try:
            content = log_file.read_text(encoding='utf-8', errors='replace')
            all_lines = content.strip().split("\n")
            return {"lines": all_lines[-lines:], "variant": variant, "total_lines": len(all_lines)}
        except:
            return {"lines": [], "error": "Could not read log"}

    def get_created_files(self, variant):
        if variant not in VARIANTS:
            return {"error": "Unknown variant"}
        project_dir = PROJECTS_DIR / VARIANTS[variant]["path"]
        files = []
        # Look for non-core Python files and any new files
        for f in project_dir.glob("*.py"):
            if f.name not in ["__init__.py"]:
                files.append({"name": f.name, "size": f.stat().st_size, "type": "python"})
        for f in (project_dir / "tools").glob("*") if (project_dir / "tools").exists() else []:
            files.append({"name": f"tools/{f.name}", "size": f.stat().st_size, "type": "tool"})
        return {"files": files, "variant": variant}

    # ============ New Observatory Endpoints ============

    def get_all_agents(self):
        """Return all agent profiles with their identities."""
        agents = []
        for key in AGENT_ORDER:
            agent = get_agent(key)
            agent["key"] = key
            agents.append(agent)
        return {"agents": agents, "variant_rosters": VARIANT_AGENTS}

    def get_agent_profile(self, agent_key, variant=None):
        """Return detailed stats for a specific agent."""
        if agent_key not in AGENTS:
            return {"error": f"Unknown agent: {agent_key}"}

        # Get base identity
        profile = get_agent(agent_key).copy()
        profile["key"] = agent_key

        # Get computed stats from analytics
        stats = get_agent_stats(agent_key, variant)
        profile.update(stats)

        # Get recent cycles where this agent participated
        cycles = get_cycles(variant=variant, limit=100)
        recent_actions = []
        for cycle in cycles[:10]:  # Last 10 cycles
            research = cycle.get("agent_research", {})
            if agent_key in research:
                action = {
                    "cycle_id": f"{cycle.get('variant', '')}-{cycle.get('cycle_num', 0)}",
                    "cycle_num": cycle.get("cycle_num", 0),
                    "variant": cycle.get("variant", ""),
                    "spoke": research[agent_key].get("spoke", False),
                    "summary": research[agent_key].get("summary", "")[:200]
                }
                # Add vote info if present
                votes = cycle.get("votes", {})
                for round_votes in votes.values():
                    if isinstance(round_votes, dict) and agent_key in round_votes:
                        action["vote"] = round_votes[agent_key].get("vote", "")
                        action["confidence"] = round_votes[agent_key].get("confidence", 0)
                recent_actions.append(action)

        profile["recent_actions"] = recent_actions
        return profile

    def get_cycle_history(self, variant=None, limit=50):
        """Return paginated cycle history."""
        cycles = get_cycles(variant=variant, limit=limit)

        # Transform to summary format
        summaries = []
        for cycle in cycles:
            summary = {
                "cycle_id": f"{cycle.get('variant', '')}-{cycle.get('cycle_num', 0)}",
                "cycle_num": cycle.get("cycle_num", 0),
                "variant": cycle.get("variant", ""),
                "variant_name": cycle.get("variant_name", ""),
                "timestamp": cycle.get("timestamp", ""),
                "question": cycle.get("question", "")[:100],
                "proposal": cycle.get("proposal", {}).get("decision", "")[:100],
                "target_file": cycle.get("proposal", {}).get("target_file", ""),
                "action": cycle.get("proposal", {}).get("action", ""),
                "tally": cycle.get("tally", {}),
                "override": cycle.get("override_triggered", False),
                "success": cycle.get("outcome", {}).get("verification_passed", True),
                "agents_spoke": sum(
                    1 for a in cycle.get("agent_research", {}).values()
                    if isinstance(a, dict) and a.get("spoke")
                )
            }
            summaries.append(summary)

        return {"cycles": summaries, "total": len(summaries)}

    def get_cycle_detail(self, cycle_id):
        """Return full detail for a specific cycle."""
        cycle = get_cycle(cycle_id)
        if not cycle:
            return {"error": f"Cycle not found: {cycle_id}"}

        # Enrich agent research with human names
        enriched_research = {}
        for agent_key, data in cycle.get("agent_research", {}).items():
            enriched = data.copy() if isinstance(data, dict) else {"summary": str(data)}
            enriched["name"] = get_name(agent_key)
            enriched["agent"] = get_agent(agent_key)
            enriched_research[agent_key] = enriched

        cycle["agent_research"] = enriched_research

        # Enrich votes with human names
        enriched_votes = {}
        for round_name, round_votes in cycle.get("votes", {}).items():
            if isinstance(round_votes, dict):
                enriched_round = {}
                for agent_key, vote_data in round_votes.items():
                    enriched_vote = vote_data.copy() if isinstance(vote_data, dict) else {"vote": str(vote_data)}
                    enriched_vote["name"] = get_name(agent_key)
                    enriched_vote["agent"] = get_agent(agent_key)
                    enriched_round[agent_key] = enriched_vote
                enriched_votes[round_name] = enriched_round

        cycle["votes"] = enriched_votes
        return cycle

    def get_relationships(self, variant=None):
        """Return agent agreement matrix."""
        matrix = compute_agreement_matrix(variant)

        # Enrich with agent names
        enriched = {"matrix": {}, "agents": []}
        for agent_key in AGENT_ORDER:
            if agent_key in matrix:
                agent_info = get_agent(agent_key)
                agent_info["key"] = agent_key
                enriched["agents"].append(agent_info)
                enriched["matrix"][agent_key] = {}
                for other_key, agreement in matrix[agent_key].items():
                    enriched["matrix"][agent_key][other_key] = {
                        "agreement": agreement,
                        "other_name": get_name(other_key)
                    }

        return enriched

    def get_stories(self, variant=None, limit=20):
        """Return narrative stories for cycles."""
        cycles = get_cycles(variant=variant, limit=limit)

        stories = []
        for cycle in cycles:
            story = {
                "cycle_id": f"{cycle.get('variant', '')}-{cycle.get('cycle_num', 0)}",
                "cycle_num": cycle.get("cycle_num", 0),
                "variant": cycle.get("variant", ""),
                "variant_name": cycle.get("variant_name", ""),
                "narrative": narrate_cycle(cycle),
                "brief": narrate_cycle_brief(cycle),
                "vote_drama": narrate_vote_drama(cycle),
                "agent_moments": []
            }

            # Get moments from each agent who spoke
            for agent_key in AGENT_ORDER:
                moment = narrate_agent_moment(agent_key, cycle)
                if moment:
                    story["agent_moments"].append({
                        "agent": agent_key,
                        "name": get_name(agent_key),
                        "moment": moment
                    })

            stories.append(story)

        # Generate episode summary for this batch
        episode = generate_episode_summary(cycles) if cycles else ""

        return {
            "stories": stories,
            "episode_summary": episode,
            "total": len(stories)
        }

    def serve_dashboard(self):
        try:
            html_path = Path(__file__).parent / "index.html"
            dashboard_html = html_path.read_text(encoding='utf-8')
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(dashboard_html.encode('utf-8'))
        except Exception as e:
            print(f"Error serving dashboard: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode('utf-8'))

def run_server(port=8888):
    os.chdir(Path(__file__).parent)
    server = HTTPServer(("localhost", port), DashboardHandler)
    print(f"Dashboard running at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    server.serve_forever()

if __name__ == "__main__":
    run_server()
