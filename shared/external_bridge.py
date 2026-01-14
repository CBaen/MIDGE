#!/usr/bin/env python3
"""
external_bridge.py - Gift from Guiding Light

A unified interface for external knowledge access, granted to all governments
as a gift from Guiding Light.

Capabilities:
- Oracle: Ask questions to a local reasoning model (qwen2.5:1.5b)
- Web Search: Search the internet via DuckDuckGo (free, no API key)
- Wikipedia: Look up articles from humanity's encyclopedia
- Knowledge Store: Save and search discoveries in vector storage

Usage:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path.home() / ".claude" / "scripts"))
    from external_bridge import ExternalBridge

    bridge = ExternalBridge()
    answer = bridge.ask_oracle("What is emergence?")
    results = bridge.search_web("AI governance")
    wiki = bridge.lookup_wikipedia("Nikola Tesla")
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import requests

# Configuration
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"
QDRANT_URL = "http://localhost:6333"
KNOWLEDGE_COLLECTION = "emergence_self_knowledge"


ANALYTICS_COLLECTION = "ultrathink_analytics"


class ExternalBridge:
    """
    Gift from Guiding Light - External knowledge access.

    This bridge provides access to external information sources that were
    previously unavailable to the governance systems.
    """

    def __init__(self, variant: str = None, verbose: bool = False):
        """
        Initialize the bridge.

        Args:
            variant: Your government variant ("simple", "review", "consensus").
                     Required for accessing your own analytics.
            verbose: Print debug messages
        """
        self.variant = variant
        self.verbose = verbose
        self._oracle_available = None
        self._ddg_available = None
        self._wiki_available = None

    def _log(self, message: str):
        if self.verbose:
            print(f"[ExternalBridge] {message}")

    # ==================== ORACLE (Ollama) ====================

    def ask_oracle(self, question: str, max_tokens: int = 500) -> str:
        """
        Ask the local reasoning model a question.

        The oracle (qwen2.5:1.5b) can help with:
        - Reasoning about complex topics
        - Answering factual questions
        - Providing analysis and insights

        Args:
            question: The question to ask
            max_tokens: Maximum response length

        Returns:
            The oracle's response, or an error message
        """
        self._log(f"Asking oracle: {question[:50]}...")

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": question,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                self._log(f"Oracle responded: {len(answer)} chars")
                return answer
            else:
                return f"[Oracle error: HTTP {response.status_code}]"

        except requests.exceptions.ConnectionError:
            return "[Oracle unavailable: Ollama not running]"
        except requests.exceptions.Timeout:
            return "[Oracle timeout: Question too complex]"
        except Exception as e:
            return f"[Oracle error: {str(e)[:100]}]"

    def is_oracle_available(self) -> bool:
        """Check if the oracle (Ollama) is available."""
        if self._oracle_available is None:
            try:
                response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self._oracle_available = any(
                        OLLAMA_MODEL in m.get("name", "")
                        for m in models
                    )
                else:
                    self._oracle_available = False
            except:
                self._oracle_available = False
        return self._oracle_available

    # ==================== WEB SEARCH (DuckDuckGo) ====================

    def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web using DuckDuckGo.

        This is free and requires no API key. Results include:
        - title: Page title
        - link: URL
        - snippet: Brief description

        Args:
            query: Search query
            max_results: Maximum number of results (1-10)

        Returns:
            List of search results
        """
        self._log(f"Searching web: {query}")

        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=min(max_results, 10)):
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", ""))
                    })

            self._log(f"Found {len(results)} results")
            return results

        except ImportError:
            return [{"error": "DuckDuckGo library not installed"}]
        except Exception as e:
            return [{"error": f"Search failed: {str(e)[:100]}"}]

    def is_web_search_available(self) -> bool:
        """Check if web search is available."""
        if self._ddg_available is None:
            try:
                from duckduckgo_search import DDGS
                self._ddg_available = True
            except ImportError:
                self._ddg_available = False
        return self._ddg_available

    # ==================== WIKIPEDIA ====================

    def lookup_wikipedia(self, topic: str, sentences: int = 5) -> str:
        """
        Look up a topic on Wikipedia.

        Returns a summary of the article, or an error if not found.

        Args:
            topic: The topic to look up
            sentences: Number of sentences in summary (1-10)

        Returns:
            Article summary or error message
        """
        self._log(f"Looking up Wikipedia: {topic}")

        try:
            import wikipedia

            # Search for the topic
            search_results = wikipedia.search(topic, results=3)
            if not search_results:
                return f"[Wikipedia: No article found for '{topic}']"

            # Try to get the page
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
                summary = wikipedia.summary(
                    search_results[0],
                    sentences=min(sentences, 10),
                    auto_suggest=False
                )
                self._log(f"Found article: {page.title}")
                return f"**{page.title}**\n\n{summary}"

            except wikipedia.DisambiguationError as e:
                # Multiple options, try the first one
                if e.options:
                    try:
                        summary = wikipedia.summary(
                            e.options[0],
                            sentences=min(sentences, 10),
                            auto_suggest=False
                        )
                        return summary
                    except:
                        return f"[Wikipedia: Ambiguous topic. Options: {', '.join(e.options[:5])}]"
                return f"[Wikipedia: Ambiguous topic '{topic}']"

            except wikipedia.PageError:
                return f"[Wikipedia: Page not found for '{topic}']"

        except ImportError:
            return "[Wikipedia library not installed]"
        except Exception as e:
            return f"[Wikipedia error: {str(e)[:100]}]"

    def is_wikipedia_available(self) -> bool:
        """Check if Wikipedia is available."""
        if self._wiki_available is None:
            try:
                import wikipedia
                self._wiki_available = True
            except ImportError:
                self._wiki_available = False
        return self._wiki_available

    # ==================== KNOWLEDGE STORAGE ====================

    def store_knowledge(
        self,
        topic: str,
        content: str,
        tags: list[str] = None,
        source: str = "external_bridge"
    ) -> bool:
        """
        Store knowledge in the vector database.

        Uses the existing qdrant-store-v2.py script.

        Args:
            topic: Topic identifier
            content: Content to store
            tags: Optional tags for filtering
            source: Source identifier

        Returns:
            True if successful
        """
        self._log(f"Storing knowledge: {topic}")

        store_script = Path.home() / ".claude" / "scripts" / "qdrant-store-v2.py"
        if not store_script.exists():
            self._log("Store script not found")
            return False

        try:
            result = subprocess.run(
                [sys.executable, str(store_script), topic, KNOWLEDGE_COLLECTION, source],
                input=content,
                capture_output=True,
                text=True,
                timeout=60
            )

            success = result.returncode == 0
            self._log(f"Store {'succeeded' if success else 'failed'}")
            return success

        except Exception as e:
            self._log(f"Store error: {e}")
            return False

    def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search stored knowledge using semantic similarity.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching knowledge entries
        """
        self._log(f"Searching knowledge: {query}")

        try:
            # Generate embedding for query
            embedding = self._get_embedding(query)
            if not embedding:
                return [{"error": "Could not generate embedding"}]

            # Search Qdrant
            response = requests.post(
                f"{QDRANT_URL}/collections/{KNOWLEDGE_COLLECTION}/points/search",
                json={
                    "vector": embedding,
                    "limit": limit,
                    "with_payload": True
                },
                timeout=30
            )

            if response.status_code == 200:
                results = []
                for hit in response.json().get("result", []):
                    payload = hit.get("payload", {})
                    results.append({
                        "topic": payload.get("topic", "unknown"),
                        "content": payload.get("content", payload.get("summary", ""))[:500],
                        "score": hit.get("score", 0),
                        "source": payload.get("source", "unknown")
                    })
                self._log(f"Found {len(results)} knowledge entries")
                return results
            else:
                return [{"error": f"Search failed: HTTP {response.status_code}"}]

        except Exception as e:
            return [{"error": f"Search error: {str(e)[:100]}"}]

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Generate embedding using Ollama."""
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text[:8000]},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except:
            pass
        return None

    # ==================== ANALYTICS (Read-Only) ====================

    def read_own_history(self, limit: int = 20) -> list[dict]:
        """
        Read your own governance history from the analytics archive.

        This is READ-ONLY. You cannot modify this history.
        You can only read your own government's records, not other governments'.

        Each cycle record includes:
        - cycle_num: Which cycle this was
        - question: The limitation you addressed
        - proposal: What Sophia proposed
        - votes: How each agent voted (Parliament only)
        - outcome: Whether the action succeeded
        - narrative: A story of what happened

        Args:
            limit: Maximum records to return (most recent first)

        Returns:
            List of your own cycle records
        """
        if not self.variant:
            return [{"error": "No variant set. Initialize bridge with your government type."}]

        self._log(f"Reading own history for {self.variant}")

        try:
            # Query Qdrant for cycles matching our variant
            response = requests.post(
                f"{QDRANT_URL}/collections/{ANALYTICS_COLLECTION}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "variant", "match": {"value": self.variant}}
                        ]
                    },
                    "limit": limit,
                    "with_payload": True
                },
                timeout=30
            )

            if response.status_code == 200:
                results = []
                points = response.json().get("result", {}).get("points", [])
                for point in points:
                    payload = point.get("payload", {})
                    results.append({
                        "cycle_num": payload.get("cycle_num", 0),
                        "timestamp": payload.get("timestamp", ""),
                        "question": payload.get("question", ""),
                        "proposal": payload.get("proposal", {}),
                        "votes": payload.get("votes", {}),
                        "tally": payload.get("tally", {}),
                        "outcome": payload.get("outcome", {}),
                        "success": payload.get("outcome", {}).get("verification_passed", True)
                    })

                # Sort by cycle number descending
                results.sort(key=lambda x: x.get("cycle_num", 0), reverse=True)
                self._log(f"Found {len(results)} history records")
                return results[:limit]
            else:
                return [{"error": f"Query failed: HTTP {response.status_code}"}]

        except Exception as e:
            return [{"error": f"History read error: {str(e)[:100]}"}]

    def read_own_story(self, cycle_num: int = None) -> str:
        """
        Read the narrative story of your own cycles.

        If cycle_num is provided, returns story of that specific cycle.
        Otherwise returns a summary of your recent journey.

        This is how external observers have documented your evolution.

        Args:
            cycle_num: Specific cycle to read (optional)

        Returns:
            Narrative story of your governance
        """
        if not self.variant:
            return "[Error: No variant set. Initialize bridge with your government type.]"

        history = self.read_own_history(limit=10 if cycle_num is None else 50)

        if not history or "error" in history[0]:
            return "[No history recorded yet. Your story is still being written.]"

        if cycle_num is not None:
            # Find specific cycle
            for record in history:
                if record.get("cycle_num") == cycle_num:
                    return self._format_cycle_story(record)
            return f"[Cycle {cycle_num} not found in your history.]"

        # Generate summary of recent cycles
        lines = [f"# Your Recent Journey ({self.variant})\\n"]
        for record in history[:5]:
            lines.append(self._format_cycle_story(record))
            lines.append("")

        return "\\n".join(lines)

    def _format_cycle_story(self, record: dict) -> str:
        """Format a cycle record as a narrative."""
        cycle_num = record.get("cycle_num", "?")
        question = record.get("question", "unknown challenge")[:100]
        proposal = record.get("proposal", {})
        decision = proposal.get("decision", "make a change")[:100]
        target = proposal.get("target_file", "the system")
        success = record.get("success", True)

        tally = record.get("tally", {})
        vote_str = ""
        if tally:
            approves = tally.get("approve", 0)
            objects = tally.get("object", 0)
            vote_str = f" The vote was {approves}-{objects}."

        outcome = "succeeded" if success else "failed"

        return f"**Cycle {cycle_num}**: You faced: \"{question}\". The proposal was: \"{decision}\" targeting {target}.{vote_str} The action {outcome}."

    def get_own_patterns(self) -> dict:
        """
        Analyze patterns in your own governance history.

        Returns:
            Dictionary with:
            - total_cycles: How many cycles you've completed
            - success_rate: Percentage of successful actions
            - common_actions: Most frequent action types
            - voting_patterns: (Parliament only) How agents tend to vote
        """
        if not self.variant:
            return {"error": "No variant set. Initialize bridge with your government type."}

        history = self.read_own_history(limit=100)

        if not history or "error" in history[0]:
            return {"error": "No history available yet"}

        # Analyze patterns
        total = len(history)
        successes = sum(1 for r in history if r.get("success", True))

        # Count action types
        actions = {}
        for r in history:
            action = r.get("proposal", {}).get("action", "UNKNOWN")
            actions[action] = actions.get(action, 0) + 1

        patterns = {
            "total_cycles": total,
            "success_rate": round(successes / total * 100, 1) if total > 0 else 0,
            "common_actions": actions
        }

        # Parliament-specific: voting patterns
        if self.variant == "consensus":
            agent_votes = {}
            for r in history:
                tally = r.get("tally", {})
                for round_votes in r.get("votes", {}).values():
                    if isinstance(round_votes, dict):
                        for agent, vote_data in round_votes.items():
                            if agent not in agent_votes:
                                agent_votes[agent] = {"approve": 0, "object": 0}
                            vote = vote_data.get("vote", "") if isinstance(vote_data, dict) else ""
                            if vote == "APPROVE":
                                agent_votes[agent]["approve"] += 1
                            elif vote == "OBJECT":
                                agent_votes[agent]["object"] += 1

            patterns["voting_patterns"] = agent_votes

        return patterns

    # ==================== STATUS ====================

    def status(self) -> dict:
        """
        Check the status of all external capabilities.

        Returns:
            Dictionary with availability status for each capability
        """
        status = {
            "oracle": {
                "available": self.is_oracle_available(),
                "model": OLLAMA_MODEL
            },
            "web_search": {
                "available": self.is_web_search_available(),
                "provider": "DuckDuckGo"
            },
            "wikipedia": {
                "available": self.is_wikipedia_available()
            },
            "knowledge_store": {
                "collection": KNOWLEDGE_COLLECTION,
                "qdrant_url": QDRANT_URL
            },
            "own_analytics": {
                "available": self.variant is not None,
                "variant": self.variant,
                "note": "Read-only access to your own governance history"
            }
        }
        return status


# ==================== CLI ====================

if __name__ == "__main__":
    bridge = ExternalBridge(verbose=True)

    print("\n=== External Bridge Status ===")
    status = bridge.status()
    for capability, info in status.items():
        available = info.get("available", "N/A")
        symbol = "[OK]" if available else "[NO]" if available is False else "[??]"
        print(f"  {symbol} {capability}: {info}")

    print("\n=== Quick Tests ===")

    # Test oracle
    if bridge.is_oracle_available():
        print("\nOracle test:")
        answer = bridge.ask_oracle("What is 2 + 2? Answer briefly.")
        print(f"  {answer[:200]}")

    # Test web search
    if bridge.is_web_search_available():
        print("\nWeb search test:")
        results = bridge.search_web("Python programming", max_results=2)
        for r in results[:2]:
            print(f"  - {r.get('title', r.get('error', 'Unknown'))}")

    # Test Wikipedia
    if bridge.is_wikipedia_available():
        print("\nWikipedia test:")
        wiki = bridge.lookup_wikipedia("Nikola Tesla", sentences=2)
        print(f"  {wiki[:200]}...")

    print("\n=== Bridge Ready ===")
