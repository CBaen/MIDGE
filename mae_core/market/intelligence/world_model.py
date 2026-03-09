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

        chains = self._get_curated_chains()
        for cause, effect, attrs in chains:
            self._graph.add_edge(cause, effect, **attrs)

        logger.info(
            "WorldModel seeded: %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    @staticmethod
    def _get_curated_chains() -> List[Tuple[str, str, dict]]:
        """Return curated causal relationships.

        Each tuple: (cause, effect, {strength, lag_days, direction, evidence})
        Direction refers to the effect on the downstream node:
        - "bearish" = cause event hurts the downstream node
        - "bullish" = cause event helps the downstream node
        """
        edges = []

        def _add(cause, effect, strength=0.7, lag=3.0, direction="bearish"):
            edges.append((cause, effect, {
                "strength": strength, "lag_days": lag,
                "direction": direction, "evidence": "curated",
                "hit_count": 0, "miss_count": 0,
            }))

        # === ENERGY CHAINS ===

        # Oil supply disruptions
        _add("hurricane_gulf", "oil_supply_disruption", 0.85, 1, "bearish")
        _add("opec_production_cut", "oil_supply_disruption", 0.90, 1, "bearish")
        _add("middle_east_conflict", "oil_supply_risk", 0.75, 1, "bearish")
        _add("oil_supply_risk", "crude_price_spike", 0.80, 2, "bullish")
        _add("oil_supply_disruption", "crude_price_spike", 0.85, 2, "bullish")

        # Crude price effects
        _add("crude_price_spike", "XLE", 0.85, 1, "bullish")
        _add("crude_price_spike", "XOP", 0.85, 1, "bullish")
        _add("crude_price_spike", "USO", 0.90, 0, "bullish")
        _add("crude_price_spike", "airline_fuel_costs", 0.90, 3, "bearish")
        _add("crude_price_spike", "chemical_feedstock_costs", 0.80, 5, "bearish")
        _add("crude_price_spike", "trucking_costs", 0.75, 7, "bearish")

        _add("airline_fuel_costs", "DAL", 0.80, 5, "bearish")
        _add("airline_fuel_costs", "UAL", 0.80, 5, "bearish")
        _add("airline_fuel_costs", "AAL", 0.80, 5, "bearish")
        _add("airline_fuel_costs", "LUV", 0.75, 5, "bearish")

        _add("chemical_feedstock_costs", "DOW", 0.70, 10, "bearish")
        _add("chemical_feedstock_costs", "LYB", 0.70, 10, "bearish")

        # Natural gas chains
        _add("nat_gas_spike", "fertilizer_costs", 0.75, 7, "bearish")
        _add("nat_gas_spike", "UNG", 0.90, 0, "bullish")
        _add("fertilizer_costs", "agricultural_costs", 0.70, 14, "bearish")
        _add("agricultural_costs", "ADM", 0.60, 21, "bearish")
        _add("agricultural_costs", "DE", 0.55, 21, "bullish")  # farm equipment demand

        # EIA inventory signals
        _add("eia_crude_build", "crude_price_pressure_down", 0.70, 1, "bearish")
        _add("crude_price_pressure_down", "XLE", 0.65, 1, "bearish")
        _add("eia_crude_draw", "crude_price_pressure_up", 0.70, 1, "bullish")
        _add("crude_price_pressure_up", "XLE", 0.65, 1, "bullish")

        # === MONETARY / MACRO CHAINS ===

        # Fed rate decisions
        _add("fed_rate_hike", "borrowing_costs_up", 0.95, 1, "bearish")
        _add("fed_rate_hike", "dollar_strengthens", 0.80, 1, "bullish")
        _add("fed_rate_hike", "bank_margins_up", 0.80, 5, "bullish")

        _add("borrowing_costs_up", "housing_demand_down", 0.80, 14, "bearish")
        _add("borrowing_costs_up", "growth_stocks_pressure", 0.75, 3, "bearish")
        _add("housing_demand_down", "XHB", 0.80, 7, "bearish")
        _add("housing_demand_down", "ITB", 0.80, 7, "bearish")
        _add("growth_stocks_pressure", "QQQ", 0.70, 3, "bearish")
        _add("growth_stocks_pressure", "ARKK", 0.80, 3, "bearish")

        _add("bank_margins_up", "XLF", 0.75, 5, "bullish")
        _add("bank_margins_up", "KRE", 0.80, 5, "bullish")

        _add("dollar_strengthens", "export_competitiveness_down", 0.70, 14, "bearish")
        _add("export_competitiveness_down", "multinational_earnings_pressure", 0.65, 30, "bearish")

        # Inflation chains
        _add("cpi_hot", "rate_hike_expectation", 0.85, 1, "bearish")
        _add("rate_hike_expectation", "growth_stocks_pressure", 0.75, 1, "bearish")
        _add("rate_hike_expectation", "bond_prices_down", 0.85, 0, "bearish")
        _add("bond_prices_down", "TLT", 0.90, 0, "bearish")

        # Labor market
        _add("unemployment_spike", "consumer_spending_down", 0.80, 14, "bearish")
        _add("consumer_spending_down", "XRT", 0.75, 7, "bearish")
        _add("consumer_spending_down", "XLY", 0.75, 7, "bearish")

        _add("nfp_strong", "consumer_confidence_up", 0.70, 3, "bullish")
        _add("consumer_confidence_up", "XLY", 0.65, 5, "bullish")

        # === TECHNOLOGY / SEMICONDUCTOR CHAINS ===

        _add("china_taiwan_tension", "chip_supply_risk", 0.80, 1, "bearish")
        _add("chip_supply_risk", "semiconductor_prices_up", 0.75, 7, "bullish")
        _add("semiconductor_prices_up", "TSM", 0.80, 3, "bullish")
        _add("semiconductor_prices_up", "auto_production_disruption", 0.70, 14, "bearish")

        _add("auto_production_disruption", "F", 0.70, 14, "bearish")
        _add("auto_production_disruption", "GM", 0.70, 14, "bearish")
        _add("auto_production_disruption", "TM", 0.65, 14, "bearish")

        _add("ai_capex_surge", "datacenter_demand", 0.85, 7, "bullish")
        _add("datacenter_demand", "NVDA", 0.90, 5, "bullish")
        _add("datacenter_demand", "AMD", 0.80, 5, "bullish")
        _add("datacenter_demand", "electricity_demand_up", 0.75, 30, "bullish")
        _add("electricity_demand_up", "utility_revenue_up", 0.70, 30, "bullish")
        _add("utility_revenue_up", "XLU", 0.65, 14, "bullish")

        # === DEFENSE / GOVERNMENT ===

        _add("defense_spending_increase", "defense_contract_awards", 0.85, 30, "bullish")
        _add("defense_contract_awards", "LMT", 0.80, 14, "bullish")
        _add("defense_contract_awards", "RTX", 0.80, 14, "bullish")
        _add("defense_contract_awards", "NOC", 0.80, 14, "bullish")
        _add("defense_contract_awards", "GD", 0.75, 14, "bullish")

        _add("geopolitical_tension_escalation", "defense_spending_increase", 0.75, 30, "bullish")
        _add("geopolitical_tension_escalation", "gold_demand_up", 0.80, 3, "bullish")
        _add("gold_demand_up", "GLD", 0.85, 1, "bullish")
        _add("gold_demand_up", "GDX", 0.80, 1, "bullish")

        # === HEALTHCARE / PHARMA ===

        _add("fda_approval", "drug_revenue_growth", 0.85, 1, "bullish")
        _add("fda_rejection", "drug_pipeline_setback", 0.90, 0, "bearish")

        _add("healthcare_reform_legislation", "pharma_pricing_pressure", 0.70, 30, "bearish")
        _add("pharma_pricing_pressure", "XLV", 0.65, 14, "bearish")

        # === SUPPLY CHAIN ===

        _add("port_strike", "shipping_disruption", 0.90, 1, "bearish")
        _add("shipping_disruption", "inventory_shortage", 0.80, 14, "bearish")
        _add("inventory_shortage", "retail_margin_pressure", 0.70, 21, "bearish")
        _add("retail_margin_pressure", "WMT", 0.60, 14, "bearish")
        _add("retail_margin_pressure", "TGT", 0.65, 14, "bearish")

        _add("drought_major", "crop_yield_down", 0.80, 30, "bearish")
        _add("crop_yield_down", "food_prices_up", 0.75, 14, "bullish")
        _add("food_prices_up", "DBA", 0.80, 3, "bullish")

        _add("steel_tariffs", "construction_costs_up", 0.75, 14, "bearish")
        _add("construction_costs_up", "housing_starts_down", 0.70, 21, "bearish")
        _add("housing_starts_down", "XHB", 0.70, 14, "bearish")

        # === REGULATORY / ANTITRUST ===

        _add("antitrust_action_tech", "market_power_reduced", 0.70, 30, "bearish")
        _add("market_power_reduced", "mega_tech_pressure", 0.65, 14, "bearish")

        _add("crypto_regulation_positive", "crypto_adoption_up", 0.75, 7, "bullish")
        _add("crypto_regulation_negative", "crypto_selloff", 0.80, 1, "bearish")

        # === GEOPOLITICAL ===

        _add("russia_sanctions", "energy_supply_disruption_eu", 0.80, 3, "bearish")
        _add("energy_supply_disruption_eu", "nat_gas_spike", 0.85, 1, "bullish")

        _add("china_tariffs", "import_costs_up", 0.80, 14, "bearish")
        _add("import_costs_up", "consumer_prices_up", 0.70, 30, "bearish")
        _add("consumer_prices_up", "cpi_hot", 0.75, 30, "bearish")

        # === CROSS-DOMAIN AMPLIFIERS ===
        # These connect domains that can create feedback loops

        _add("vix_spike", "risk_off_rotation", 0.80, 0, "bearish")
        _add("risk_off_rotation", "growth_stocks_pressure", 0.75, 1, "bearish")
        _add("risk_off_rotation", "gold_demand_up", 0.70, 1, "bullish")
        _add("risk_off_rotation", "bond_prices_up", 0.75, 0, "bullish")
        _add("bond_prices_up", "TLT", 0.85, 0, "bullish")

        _add("credit_spread_widening", "recession_fear", 0.75, 7, "bearish")
        _add("recession_fear", "consumer_spending_down", 0.70, 14, "bearish")
        _add("recession_fear", "XLF", 0.70, 3, "bearish")

        return edges

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
