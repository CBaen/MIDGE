#!/usr/bin/env python3
"""
World Model — Curated causal chain graph for market ripple-effect detection.

MIDGE's understanding of HOW the economy works. When an event fires
(oil supply disruption, Fed rate hike, chip shortage), this graph lets
MIDGE trace the ripple effects to specific sectors and tickers.

Uses NetworkX DiGraph as backbone. Each edge carries:
  - strength: how reliable the causal link (0-1)
  - lag_days: typical propagation delay
  - direction: bullish/bearish effect on the downstream node
  - evidence: how the link was established (curated/discovered/granger)

Two node types:
  - Events/factors: "oil_price_spike", "fed_rate_hike", "chip_shortage"
  - Tickers/instruments: "XLE", "NVDA", "DAL" (leaf nodes, tradeable)

Key method: find_ripple_effects(trigger_event) → list of (ticker, direction,
  strength, path, lag_days) sorted by combined strength.

Biological analogy: This is MIDGE's world model — the mental map that lets
her reason about consequences, not just observe correlations.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mae_core.market.intelligence.world_model_chains import _get_curated_chains  # noqa: F401

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_WORLD_MODEL_PATH = _DATA_DIR / "world_model_state.json"


@dataclass
class RippleEffect:
    """A predicted downstream effect from a trigger event."""
    ticker: str
    direction: str  # bullish, bearish
    strength: float  # 0-1, combined chain strength
    path: List[str]  # event → intermediate → ... → ticker
    total_lag_days: float  # estimated propagation time
    confidence: float  # based on edge evidence quality


@dataclass
class RootCause:
    """A genesis event traced backward from a known downstream effect."""
    trigger: str           # genesis event name
    direction: str         # bullish/bearish — net effect toward original node
    strength: float        # 0-1, combined chain strength (multiplied backward)
    path: List[str]        # observed_node → intermediate → ... → trigger (reversed)
    total_lag_days: float  # estimated propagation time from trigger to observed effect
    confidence: float      # based on edge evidence quality


@dataclass
class CausalEdge:
    """Metadata for a single causal link."""
    strength: float = 0.7
    lag_days: float = 3.0
    direction: str = "bearish"  # effect on downstream node
    evidence: str = "curated"  # curated, discovered, granger
    hit_count: int = 0  # times this chain was confirmed by outcomes
    miss_count: int = 0  # times this chain was contradicted


class WorldModel:
    """Curated + discovered causal chain graph for market reasoning.

    Seeded with ~80 curated relationships across energy, macro, tech,
    geopolitical, and supply chain domains. Grows as Granger/correlation
    analyzers discover new links.
    """

    def __init__(self, persistence_path: str = None):
        if not HAS_NETWORKX:
            logger.warning("networkx not installed — WorldModel disabled")
            self._graph = None
            return

        self._graph = nx.DiGraph()
        self._persistence_path = Path(persistence_path) if persistence_path else _WORLD_MODEL_PATH
        self._seed_curated_chains()
        self._load_discovered_edges()

    def _seed_curated_chains(self):
        """Seed the graph with known real-world causal relationships."""
        if self._graph is None:
            return

        chains = _get_curated_chains()
        for cause, effect, attrs in chains:
            self._graph.add_edge(cause, effect, **attrs)

        logger.info(
            "WorldModel seeded: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def find_ripple_effects(
        self, trigger: str, min_strength: float = 0.3, max_depth: int = 5
    ) -> List[RippleEffect]:
        """Trace all downstream effects from a trigger event.

        Returns tickers/instruments affected, sorted by combined strength.
        """
        if self._graph is None or trigger not in self._graph:
            return []

        effects = []
        # BFS with strength accumulation
        visited = set()
        # queue: (node, path, cumulative_strength, cumulative_lag, direction_chain)
        queue = [(trigger, [trigger], 1.0, 0.0, None)]

        while queue:
            current, path, cum_strength, cum_lag, last_direction = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            # If this is a ticker (leaf node or known ticker), record it
            if self._is_ticker(current) and current != trigger:
                effects.append(RippleEffect(
                    ticker=current,
                    direction=last_direction or "neutral",
                    strength=cum_strength,
                    path=list(path),
                    total_lag_days=cum_lag,
                    confidence=cum_strength * (0.95 ** (len(path) - 2)),
                ))

            if len(path) > max_depth:
                continue

            for successor in self._graph.successors(current):
                if successor in visited:
                    continue
                edge = self._graph.edges[current, successor]
                new_strength = cum_strength * edge["strength"]
                if new_strength < min_strength:
                    continue
                new_lag = cum_lag + edge["lag_days"]
                queue.append((
                    successor,
                    path + [successor],
                    new_strength,
                    new_lag,
                    edge["direction"],
                ))

        effects.sort(key=lambda e: e.strength, reverse=True)
        return effects

    def find_root_causes(
        self, node: str, min_strength: float = 0.3, max_depth: int = 4
    ) -> List["RootCause"]:
        """Trace all upstream genesis events that causally reach a known node.

        Reverse BFS from ``node`` using predecessor edges. Accumulates
        strength and lag the same way find_ripple_effects does, but walks
        the graph backward.  Only nodes that are genesis triggers (no
        predecessors, or all predecessors below min_strength) are returned.

        Args:
            node: Downstream node (e.g. a ticker like "XLE" or intermediate
                  node like "crude_price_spike") to trace backward from.
            min_strength: Minimum cumulative strength to retain a path.
            max_depth: Maximum backward hops to prevent runaway traversal.

        Returns:
            List of RootCause, sorted by strength descending.
        """
        if self._graph is None or node not in self._graph:
            return []

        causes: List[RootCause] = []
        # queue: (current_node, path_so_far, cum_strength, cum_lag)
        # path is built from node → ... → trigger (reversed at the end)
        queue = [(node, [node], 1.0, 0.0)]
        visited: set = set()

        while queue:
            current, path, cum_strength, cum_lag = queue.pop(0)

            # Avoid revisiting nodes (prevents cycles in discovered-edge graphs)
            if current in visited:
                continue
            visited.add(current)

            predecessors = list(self._graph.predecessors(current))

            # If no predecessors (or all would be too weak), this node IS a root
            if not predecessors and current != node:
                # Determine direction: use the edge direction from this root
                # to its first successor on our path (last element before current)
                direction = "neutral"
                if len(path) >= 2:
                    child = path[-2]  # node before current in reversed path
                    if self._graph.has_edge(current, child):
                        direction = self._graph.edges[current, child].get("direction", "neutral")
                causes.append(RootCause(
                    trigger=current,
                    direction=direction,
                    strength=cum_strength,
                    path=list(reversed(path)),
                    total_lag_days=cum_lag,
                    confidence=cum_strength * (0.95 ** (len(path) - 2)),
                ))
                continue

            if len(path) > max_depth:
                # Report as a root even if we haven't reached a true genesis
                if current != node:
                    direction = "neutral"
                    if len(path) >= 2:
                        child = path[-2]
                        if self._graph.has_edge(current, child):
                            direction = self._graph.edges[current, child].get("direction", "neutral")
                    causes.append(RootCause(
                        trigger=current,
                        direction=direction,
                        strength=cum_strength,
                        path=list(reversed(path)),
                        total_lag_days=cum_lag,
                        confidence=cum_strength * (0.95 ** (len(path) - 2)),
                    ))
                continue

            for pred in predecessors:
                if pred in visited:
                    continue
                edge = self._graph.edges[pred, current]
                new_strength = cum_strength * edge["strength"]
                if new_strength < min_strength:
                    continue
                new_lag = cum_lag + edge["lag_days"]
                queue.append((pred, path + [pred], new_strength, new_lag))

        causes.sort(key=lambda c: c.strength, reverse=True)
        return causes

    def _is_ticker(self, node: str) -> bool:
        """Check if a node represents a tradeable instrument."""
        # Tickers are uppercase, 1-5 chars, no underscores
        return (
            node.isupper()
            and 1 <= len(node) <= 5
            and "_" not in node
        )

    def add_discovered_edge(
        self,
        cause: str,
        effect: str,
        strength: float = 0.5,
        lag_days: float = 5.0,
        direction: str = "bearish",
        evidence: str = "discovered",
    ):
        """Add an edge discovered by Granger/correlation analysis."""
        if self._graph is None:
            return

        if self._graph.has_edge(cause, effect):
            # Strengthen existing edge
            edge = self._graph.edges[cause, effect]
            edge["strength"] = min(1.0, edge["strength"] * 0.8 + strength * 0.2)
            edge["hit_count"] = edge.get("hit_count", 0) + 1
        else:
            self._graph.add_edge(cause, effect, **{
                "strength": strength,
                "lag_days": lag_days,
                "direction": direction,
                "evidence": evidence,
                "hit_count": 0,
                "miss_count": 0,
            })

    def record_outcome(self, trigger: str, ticker: str, was_correct: bool):
        """Update edge strength based on prediction outcome."""
        if self._graph is None:
            return

        # Find path from trigger to ticker
        try:
            path = nx.shortest_path(self._graph, trigger, ticker)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return

        # Update each edge in the chain
        for i in range(len(path) - 1):
            edge = self._graph.edges[path[i], path[i + 1]]
            if was_correct:
                edge["hit_count"] = edge.get("hit_count", 0) + 1
                edge["strength"] = min(1.0, edge["strength"] + 0.02)
            else:
                edge["miss_count"] = edge.get("miss_count", 0) + 1
                edge["strength"] = max(0.1, edge["strength"] - 0.03)

    def map_signal_to_trigger(self, source: str, metadata: dict) -> Optional[str]:
        """Map a MIDGE signal source to a world model trigger node.

        This is the bridge between MIDGE's signal pipeline and the world model.
        When a signal fires, this maps it to the appropriate trigger node so
        find_ripple_effects can trace downstream consequences.
        """
        # EIA energy signals → specific triggers
        if source == "eia_energy":
            series = metadata.get("series_id", "")
            direction = metadata.get("direction", "")
            if "crude" in series.lower():
                return "eia_crude_build" if direction == "bearish" else "eia_crude_draw"
            if "natgas" in series.lower() or "natural_gas" in series.lower():
                return "nat_gas_spike" if direction == "bullish" else None

        # Economic calendar → macro triggers
        if source == "economic_calendar":
            event_type = metadata.get("event_type", "")
            if "FOMC" in event_type or "fed" in event_type.lower():
                return "fed_rate_hike"  # Caller should check direction
            if "CPI" in event_type:
                return "cpi_hot"
            if "NFP" in event_type or "employment" in event_type.lower():
                return "nfp_strong"

        # VIX signals
        if source == "vix_term_structure":
            if metadata.get("strength", 0) > 0.7:
                return "vix_spike"

        # Geopolitical / government signals
        if source in ("congressional", "senate", "congress_legislation"):
            keywords = metadata.get("keywords", [])
            text = " ".join(str(v) for v in metadata.values()).lower()
            if any(w in text for w in ("defense", "military", "pentagon")):
                return "defense_spending_increase"
            if any(w in text for w in ("tariff", "trade war", "import duty")):
                return "china_tariffs"

        # Insider/institutional → no direct world model mapping
        # (these are company-specific, not macro events)
        return None

    # ------------------------------------------------------------------
    # Pattern node / edge methods — Phase 3 Deep Pattern Intelligence
    # ------------------------------------------------------------------

    def add_pattern_node(
        self,
        template_id: str,
        domain_signature: str,
        win_rate: float = 0.0,
    ) -> None:
        """Add a PatternTemplate node to the causal graph.

        Pattern nodes live alongside event/ticker nodes and participate in
        find_pattern_chain() traversal.  They are prefixed with "pattern:"
        so _is_ticker() correctly identifies them as non-tradeable.

        Args:
            template_id: The template identifier from PatternLibrary.
            domain_signature: e.g. "bullish:insider+macro+technical"
            win_rate: Historical win rate (0-1).
        """
        if self._graph is None:
            return
        node_id = f"pattern:{template_id}"
        self._graph.add_node(
            node_id,
            node_type="pattern_template",
            template_id=template_id,
            domain_signature=domain_signature,
            win_rate=win_rate,
        )

    def add_pattern_edge(
        self,
        template_a: str,
        template_b: str,
        lag_days: float,
        win_rate: float = 0.0,
        n_obs: int = 0,
    ) -> None:
        """Add a directed edge from pattern A to pattern B.

        Represents "Pattern A is typically followed by Pattern B after
        lag_days days."  Strength is derived from win_rate * observation
        confidence (clamped to 0.1–1.0).

        Args:
            template_a: Preceding template ID.
            template_b: Following template ID.
            lag_days: Average observed lag in days.
            win_rate: Combined sequence win rate (0-1).
            n_obs: Number of observations supporting this edge.
        """
        if self._graph is None:
            return
        node_a = f"pattern:{template_a}"
        node_b = f"pattern:{template_b}"
        # Ensure nodes exist
        if node_a not in self._graph:
            self.add_pattern_node(template_a, "", win_rate)
        if node_b not in self._graph:
            self.add_pattern_node(template_b, "", win_rate)

        # Strength scales with win_rate and observation confidence
        obs_confidence = min(1.0, n_obs / 10.0)  # Saturates at 10 observations
        strength = max(0.1, win_rate * obs_confidence)

        self._graph.add_edge(
            node_a,
            node_b,
            strength=strength,
            lag_days=float(lag_days),
            direction="bullish",  # Pattern sequences are directionally inherited
            evidence="pattern_sequence",
            win_rate=win_rate,
            n_observations=n_obs,
            hit_count=0,
            miss_count=0,
        )
        logger.debug(
            "WorldModel pattern edge added: %s → %s (lag=%.1f, wr=%.2f, n=%d)",
            template_a, template_b, lag_days, win_rate, n_obs,
        )

    def find_pattern_chain(
        self,
        template_id: str,
        max_depth: int = 5,
        min_strength: float = 0.1,
    ) -> List[dict]:
        """Return the cascading chain of downstream patterns following template_id.

        Performs BFS forward from the pattern node, accumulating strength
        and lag at each step.

        Args:
            template_id: Starting template.
            max_depth: Maximum BFS depth.
            min_strength: Prune paths below this cumulative strength.

        Returns:
            List of dicts, sorted by cumulative strength descending:
              {template_id, depth, cumulative_lag_days, cumulative_strength,
               path, win_rate, n_observations}
        """
        if self._graph is None:
            return []

        start_node = f"pattern:{template_id}"
        if start_node not in self._graph:
            return []

        results = []
        visited: set = set()
        # queue: (node, depth, path, cum_strength, cum_lag)
        queue = [(start_node, 0, [template_id], 1.0, 0.0)]

        while queue:
            current, depth, path, cum_strength, cum_lag = queue.pop(0)

            if current in visited:
                continue
            visited.add(current)

            for successor in self._graph.successors(current):
                edge = self._graph.edges[current, successor]
                if edge.get("evidence") != "pattern_sequence":
                    continue  # Only follow pattern sequence edges

                new_strength = cum_strength * edge.get("strength", 0.5)
                if new_strength < min_strength:
                    continue

                new_lag = cum_lag + edge.get("lag_days", 0.0)
                # Strip "pattern:" prefix for display
                succ_tid = successor.replace("pattern:", "", 1)
                new_path = path + [succ_tid]

                if depth + 1 <= max_depth:
                    results.append({
                        "template_id": succ_tid,
                        "depth": depth + 1,
                        "cumulative_lag_days": round(new_lag, 1),
                        "cumulative_strength": round(new_strength, 4),
                        "path": list(new_path),
                        "win_rate": edge.get("win_rate", 0.0),
                        "n_observations": edge.get("n_observations", 0),
                    })
                    queue.append((successor, depth + 1, new_path, new_strength, new_lag))

        results.sort(key=lambda x: x["cumulative_strength"], reverse=True)
        return results

    def get_statistics(self) -> dict:
        """Return graph statistics."""
        if self._graph is None:
            return {"status": "disabled", "reason": "networkx not installed"}

        ticker_nodes = [n for n in self._graph.nodes if self._is_ticker(n)]
        event_nodes = [n for n in self._graph.nodes if not self._is_ticker(n)]

        return {
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "event_nodes": len(event_nodes),
            "ticker_nodes": len(ticker_nodes),
            "tickers": sorted(ticker_nodes),
            "curated_edges": sum(
                1 for _, _, d in self._graph.edges(data=True)
                if d.get("evidence") == "curated"
            ),
            "discovered_edges": sum(
                1 for _, _, d in self._graph.edges(data=True)
                if d.get("evidence") != "curated"
            ),
        }

    def persist(self):
        """Save discovered edges to disk."""
        if self._graph is None:
            return

        discovered = []
        for u, v, data in self._graph.edges(data=True):
            if data.get("evidence") != "curated":
                discovered.append({
                    "cause": u, "effect": v, **data,
                })

        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persistence_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump({"discovered_edges": discovered}, f, indent=2)
            tmp.replace(self._persistence_path)
        except Exception:
            logger.debug("WorldModel persistence failed", exc_info=True)

    def _load_discovered_edges(self):
        """Load previously discovered edges from disk."""
        if self._graph is None:
            return

        try:
            if self._persistence_path.exists():
                with open(self._persistence_path) as f:
                    data = json.load(f)
                for edge in data.get("discovered_edges", []):
                    cause = edge.pop("cause")
                    effect = edge.pop("effect")
                    self._graph.add_edge(cause, effect, **edge)
                logger.info(
                    "WorldModel loaded %d discovered edges",
                    len(data.get("discovered_edges", [])),
                )
        except Exception:
            logger.debug("WorldModel load failed", exc_info=True)
