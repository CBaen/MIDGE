"""CirculatorySystem - cardiovascular resource distribution for Mae.

Biological basis: William Harvey's discovery (1628) of blood circulation.
The heart pumps blood carrying nutrients and oxygen to tissues proportional
to demand. Organs that work harder receive more blood flow. Murray's law
(1926) governs branching: the cube of a parent vessel's radius equals the
sum of cubes of its children's radii, minimizing total energy for transport.

Mae's circulatory system distributes three resource types:
- compute: processing capacity (CPU-like)
- memory: working memory allocation
- attention: attentional bandwidth

Each system registers demand signals. The circulatory system distributes
supply proportional to demand using Murray's law scaling, with urgency
as a priority multiplier. Heart rate increases under high total demand
(sympathetic response) and decreases when demand is low (parasympathetic).

Law compliance:
- Law 8 property (1): integration — connects all parts of the organism
  via resource flow, creating a unified metabolic whole
- Law 8 property (5): multi-scale hierarchy — distributes resources
  across fractal levels (agent, subsystem, module, organ)
- Law 6: autopoietic closure — the system replenishes itself each step;
  supply regenerates as resources "return to the heart"

EventBus channels:
- substrate.circulation_update: per-step distribution report
- substrate.supply_low: alert when a resource type drops below 20%
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_CIRCULATION = "substrate.circulation_update"
CH_SUPPLY_LOW = "substrate.supply_low"

# Default supply levels for each resource type
_DEFAULT_SUPPLY: dict[str, float] = {
    "compute": 100.0,
    "memory": 100.0,
    "attention": 100.0,
}

# Replenishment rate per step (fraction of max restored)
_REPLENISH_RATE = 0.15

# Supply low threshold (fraction of max)
_SUPPLY_LOW_THRESHOLD = 0.20

# Heart rate bounds
_MIN_HEART_RATE = 0.5
_MAX_HEART_RATE = 3.0

# Murray's law default exponent
_DEFAULT_MURRAY_EXPONENT = 3.0

# Distribution history cap
_MAX_HISTORY = 20


@dataclass
class ResourcePacket:
    """A single packet of resource being distributed.

    Biological analogy: a red blood cell carrying oxygen to a specific
    tissue, with a limited time-to-live before the resource is "lost"
    (like oxygen dissipating without being consumed).
    """

    resource_type: str  # "compute", "memory", or "attention"
    amount: float
    destination: str  # system name receiving the resource
    priority: float = 0.5  # 0.0 (background) to 1.0 (critical)
    ttl: int = 10  # steps before the packet expires


@dataclass
class DemandSignal:
    """A demand for resources from a system.

    Biological analogy: a tissue releasing vasodilators (like nitric
    oxide) to increase local blood flow in proportion to metabolic need.
    """

    system_name: str
    resource_type: str
    amount_needed: float
    urgency: float = 0.5  # 0.0 (can wait) to 1.0 (critical)


class CirculatorySystem:
    """Mae's cardiovascular system — distributes resources to where they are needed.

    The heart pumps (distributes) resources each step. Supply replenishes
    naturally (blood returning to the heart). Demand signals from systems
    guide proportional distribution using Murray's law scaling.

    Murray's law ensures efficient distribution: each system's allocation
    is proportional to (its demand)^(1/exponent) relative to the total,
    which minimizes transport cost just as biological branching does.
    """

    CH_CIRCULATION = CH_CIRCULATION
    CH_SUPPLY_LOW = CH_SUPPLY_LOW

    def __init__(
        self,
        event_bus: Any = None,
        substrate: Any = None,
        initial_supply: Optional[dict[str, float]] = None,
        murray_exponent: float = _DEFAULT_MURRAY_EXPONENT,
    ) -> None:
        self.event_bus = event_bus
        self._substrate = substrate

        # Supply pool — the "blood reservoir" for each resource type
        self._supply: dict[str, float] = dict(initial_supply or _DEFAULT_SUPPLY)
        # Max supply levels (for replenishment ceiling)
        self._max_supply: dict[str, float] = dict(self._supply)

        # Pending demands indexed by resource type
        self._demands: dict[str, list[DemandSignal]] = {
            rt: [] for rt in self._supply
        }

        # Distribution history (last N rounds for observability)
        self._distribution_history: deque[dict[str, Any]] = deque(maxlen=_MAX_HISTORY)

        # Heart rate: distributions per step (can increase under stress)
        self._heart_rate: float = 1.0

        # Murray's law exponent (3.0 for biological vascular networks)
        self._murray_exponent: float = murray_exponent

        # Step tracking
        self._step_count: int = 0

        # Cumulative statistics
        self._total_distributed: float = 0.0
        self._total_unfulfilled: float = 0.0

        # Subscribe to relevant channels
        if self.event_bus is not None:
            self.event_bus.register_callback(
                "memory.starvation_alert", self._on_starvation_alert
            )

        logger.info(
            "CirculatorySystem initialized: supply=%s, murray_exp=%.1f",
            {k: v for k, v in self._supply.items()},
            self._murray_exponent,
        )

    # =========================================================================
    # Demand Registration
    # =========================================================================

    def request_resource(
        self,
        system_name: str,
        resource_type: str,
        amount: float,
        urgency: float = 0.5,
    ) -> None:
        """Register a demand for resources.

        Biological analogy: a tissue releasing vasodilators to request
        more blood flow. The demand is queued until the next heartbeat
        (distribution step).

        Args:
            system_name: The requesting system's identifier.
            resource_type: "compute", "memory", or "attention".
            amount: Amount of resource needed (positive float).
            urgency: 0.0 (background) to 1.0 (critical).
        """
        if resource_type not in self._demands:
            # Accept unknown resource types gracefully
            self._demands[resource_type] = []
            self._supply.setdefault(resource_type, 0.0)
            self._max_supply.setdefault(resource_type, 100.0)

        urgency = max(0.0, min(1.0, urgency))
        amount = max(0.0, amount)

        signal = DemandSignal(
            system_name=system_name,
            resource_type=resource_type,
            amount_needed=amount,
            urgency=urgency,
        )
        self._demands[resource_type].append(signal)

    # =========================================================================
    # Core Distribution (The Heartbeat)
    # =========================================================================

    def _distribute(self) -> dict[str, Any]:
        """Distribute resources using Murray's law proportional allocation.

        Murray's law: parent_flow^exponent = sum(child_flow^exponent).
        We use this to allocate supply proportionally:

        1. Compute weighted demand for each system:
           weighted = amount * (0.5 + 0.5 * urgency)
        2. Apply Murray scaling: each system's share is proportional to
           weighted_demand^(1/exponent) / sum(all weighted^(1/exponent))
        3. Distribute from available supply.

        Returns a summary dict of what was distributed and what remains.
        """
        total_distributed: float = 0.0
        total_unfulfilled: float = 0.0
        distributions: dict[str, dict[str, float]] = {}

        for resource_type, demands in self._demands.items():
            if not demands:
                continue

            available = self._supply.get(resource_type, 0.0)
            if available <= 0:
                # Nothing to give — all demands are unfulfilled
                for d in demands:
                    total_unfulfilled += d.amount_needed
                continue

            # Compute weighted demands
            weighted: list[tuple[DemandSignal, float]] = []
            for d in demands:
                w = d.amount_needed * (0.5 + 0.5 * d.urgency)
                weighted.append((d, max(w, 1e-9)))

            # Murray scaling: share proportional to demand^(1/exponent)
            exp_inv = 1.0 / self._murray_exponent
            murray_values = [(d, w ** exp_inv) for d, w in weighted]
            murray_total = sum(mv for _, mv in murray_values)

            if murray_total <= 0:
                continue

            # Allocate proportionally, capped by actual demand
            for demand_signal, murray_val in murray_values:
                share_fraction = murray_val / murray_total
                allocated = available * share_fraction
                # Do not give more than requested
                actual = min(allocated, demand_signal.amount_needed)
                actual = max(0.0, actual)

                # Deduct from supply
                self._supply[resource_type] -= actual
                total_distributed += actual

                # Track unfulfilled
                shortfall = demand_signal.amount_needed - actual
                if shortfall > 0.01:
                    total_unfulfilled += shortfall

                # Record distribution
                sys_key = demand_signal.system_name
                if sys_key not in distributions:
                    distributions[sys_key] = {}
                prev = distributions[sys_key].get(resource_type, 0.0)
                distributions[sys_key][resource_type] = prev + actual

        # Update cumulative stats
        self._total_distributed += total_distributed
        self._total_unfulfilled += total_unfulfilled

        return {
            "distributed": round(total_distributed, 4),
            "unfulfilled": round(total_unfulfilled, 4),
            "per_system": distributions,
        }

    def _replenish(self) -> None:
        """Replenish resource supply each step.

        Biological analogy: deoxygenated blood returns to the heart and
        lungs, picks up fresh oxygen and nutrients, and becomes available
        for the next heartbeat. Supply regenerates toward max at a fixed
        rate each step.
        """
        for resource_type in self._supply:
            max_val = self._max_supply.get(resource_type, 100.0)
            current = self._supply[resource_type]
            deficit = max_val - current
            if deficit > 0:
                self._supply[resource_type] += deficit * _REPLENISH_RATE

    def _adapt_heart_rate(self) -> None:
        """Adapt heart rate based on demand pressure.

        Under high demand: heart rate increases (sympathetic response).
        Under low demand: heart rate decreases (parasympathetic response).
        """
        total_demand = sum(
            sum(d.amount_needed for d in demands)
            for demands in self._demands.values()
        )
        total_supply = sum(self._supply.values())

        if total_supply > 0:
            demand_ratio = total_demand / total_supply
        else:
            demand_ratio = 1.0

        # Target heart rate based on demand/supply ratio
        target_hr = 1.0 + demand_ratio
        # Smooth adjustment (EMA)
        self._heart_rate = 0.7 * self._heart_rate + 0.3 * target_hr
        self._heart_rate = max(_MIN_HEART_RATE, min(_MAX_HEART_RATE, self._heart_rate))

    def _check_supply_alerts(self) -> None:
        """Publish supply_low alerts when any resource drops below threshold."""
        if self.event_bus is None:
            return

        for resource_type, current in self._supply.items():
            max_val = self._max_supply.get(resource_type, 100.0)
            if max_val > 0 and (current / max_val) < _SUPPLY_LOW_THRESHOLD:
                self.event_bus.publish(
                    CH_SUPPLY_LOW,
                    {
                        "resource_type": resource_type,
                        "current": round(current, 4),
                        "max": round(max_val, 4),
                        "fraction": round(current / max_val, 4),
                        "step": self._step_count,
                    },
                )

    # =========================================================================
    # Step (called by model each tick)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Execute one circulatory cycle.

        1. Replenish supply (blood returns to heart)
        2. Distribute to pending demands (heartbeat)
        3. Adapt heart rate
        4. Check for supply alerts
        5. Clear fulfilled demands
        6. Publish circulation update

        The number of distribution rounds per step equals floor(heart_rate).
        Higher heart rate = more distribution passes = faster resource
        delivery under stress.
        """
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # 1. Replenish supply
        self._replenish()

        # 2. Adapt heart rate (before distribution, so it affects this step)
        self._adapt_heart_rate()

        # 3. Distribute resources (possibly multiple rounds for high heart rate)
        rounds = max(1, int(self._heart_rate))
        round_results: list[dict[str, Any]] = []
        for _ in range(rounds):
            result = self._distribute()
            round_results.append(result)

        # Aggregate results
        total_dist = sum(r["distributed"] for r in round_results)
        total_unfulfilled = round_results[-1]["unfulfilled"] if round_results else 0.0
        per_system: dict[str, dict[str, float]] = {}
        for r in round_results:
            for sys_name, resources in r.get("per_system", {}).items():
                if sys_name not in per_system:
                    per_system[sys_name] = {}
                for rt, amt in resources.items():
                    per_system[sys_name][rt] = per_system[sys_name].get(rt, 0.0) + amt

        # 4. Check supply alerts
        self._check_supply_alerts()

        # 5. Clear fulfilled demands
        self._demands = {rt: [] for rt in self._supply}

        # 6. Record history
        supply_snapshot = {k: round(v, 4) for k, v in self._supply.items()}
        history_entry = {
            "step": self._step_count,
            "distributed": round(total_dist, 4),
            "unfulfilled": round(total_unfulfilled, 4),
            "heart_rate": round(self._heart_rate, 4),
            "supply_levels": supply_snapshot,
            "distribution_rounds": rounds,
            "per_system": per_system,
        }
        self._distribution_history.append(history_entry)

        # 7. Publish circulation update
        if self.event_bus is not None:
            self.event_bus.publish(
                CH_CIRCULATION,
                {
                    "distributed": round(total_dist, 4),
                    "unfulfilled": round(total_unfulfilled, 4),
                    "heart_rate": round(self._heart_rate, 4),
                    "supply_levels": supply_snapshot,
                    "step": self._step_count,
                },
            )

    # =========================================================================
    # EventBus Handlers
    # =========================================================================

    def _on_starvation_alert(self, channel: str, message: Any) -> None:
        """React to substrate starvation by increasing supply urgency.

        When nodes are starving, the circulatory system increases heart
        rate temporarily (like the sympathetic nervous system responding
        to tissue hypoxia).
        """
        self._heart_rate = min(_MAX_HEART_RATE, self._heart_rate * 1.2)

    # =========================================================================
    # Public Query API
    # =========================================================================

    def get_supply_level(self, resource_type: str) -> float:
        """Get current supply level for a resource type."""
        return self._supply.get(resource_type, 0.0)

    def get_heart_rate(self) -> float:
        """Get current heart rate (distributions per step)."""
        return self._heart_rate

    def is_adequate(self) -> bool:
        """Whether all recent demands have been met within tolerance.

        Checks the last 2 distribution rounds. If unfulfilled demand
        is zero (or negligible) in both, circulation is adequate.
        """
        recent = list(self._distribution_history)[-2:]
        if not recent:
            return True  # No history = no unmet demand
        return all(entry.get("unfulfilled", 0.0) < 0.01 for entry in recent)

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return circulatory system statistics for monitoring."""
        return {
            "supply_levels": {k: round(v, 4) for k, v in self._supply.items()},
            "heart_rate": round(self._heart_rate, 4),
            "total_distributed": round(self._total_distributed, 4),
            "total_unfulfilled": round(self._total_unfulfilled, 4),
            "is_adequate": self.is_adequate(),
            "murray_exponent": self._murray_exponent,
            "step_count": self._step_count,
            "history_length": len(self._distribution_history),
            "pending_demands": {
                rt: len(demands) for rt, demands in self._demands.items()
            },
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict:
        """Serialize circulatory state for persistence."""
        return {
            "supply": dict(self._supply),
            "max_supply": dict(self._max_supply),
            "heart_rate": self._heart_rate,
            "murray_exponent": self._murray_exponent,
            "step_count": self._step_count,
            "total_distributed": self._total_distributed,
            "total_unfulfilled": self._total_unfulfilled,
        }

    def restore(self, data: dict) -> None:
        """Restore circulatory state from serialized data."""
        if not isinstance(data, dict):
            return

        supply = data.get("supply")
        if isinstance(supply, dict):
            self._supply.update(
                {k: float(v) for k, v in supply.items()}
            )

        max_supply = data.get("max_supply")
        if isinstance(max_supply, dict):
            self._max_supply.update(
                {k: float(v) for k, v in max_supply.items()}
            )

        self._heart_rate = float(data.get("heart_rate", self._heart_rate))
        self._murray_exponent = float(
            data.get("murray_exponent", self._murray_exponent)
        )
        self._step_count = int(data.get("step_count", self._step_count))
        self._total_distributed = float(
            data.get("total_distributed", self._total_distributed)
        )
        self._total_unfulfilled = float(
            data.get("total_unfulfilled", self._total_unfulfilled)
        )

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _parse_message(message: Any) -> Optional[dict]:
        """Parse an EventBus message (may be JSON string or dict)."""
        if isinstance(message, dict):
            return message
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return None
