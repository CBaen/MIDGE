"""Integration Meter result dataclasses.

Extracted from integration_meter.py for single-responsibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PhiResult:
    """Result of computing IIT Phi for one holon."""

    holon_id: str
    holon_type: str  # "subsystem", "organ", "organism"
    phi: float  # Phi value (0.0 = fully reducible, >0 = integrated)
    mip: tuple[tuple[str, ...], tuple[str, ...]]  # Minimum information partition
    all_partitions: list[dict[str, Any]]  # phi for each bipartition
    children_ids: list[str]
    buffer_size: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class MarkovBlanketResult:
    """Markov blanket analysis for one subsystem/organ."""

    holon_id: str
    internal_states: list[str]  # Children IDs (inside the blanket)
    blanket_states: list[str]  # Parent + siblings + cross-connected
    external_states: list[str]  # Everything else
    blanket_effectiveness: float  # 0.0 = no isolation, 1.0 = perfect
    cross_connections: int  # Cross-boundary connections found
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntegrationReport:
    """Full measurement report across all scales."""

    step: int
    measurement_number: int
    subsystem_phi: dict[str, PhiResult]
    organ_phi: dict[str, PhiResult]
    organism_phi: Optional[PhiResult]
    markov_blankets: dict[str, MarkovBlanketResult]
    organism_mean_phi: float
    weakest_link: Optional[str]  # Holon with lowest non-zero Phi
    timestamp: float = field(default_factory=time.time)
