"""Stigmergic Environment - Pheromone-based indirect communication.

Agents deposit markers in a spatial environment. Other agents sense
these markers to guide behavior. Markers decay over time.

Biological analogy: Ant pheromone trails, bacterial biofilms.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StigmergicMarker:
    """A pheromone marker deposited in the environment."""

    marker_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    marker_type: str = "generic"
    position: tuple[float, ...] = (0.0, 0.0)
    intensity: float = 1.0
    depositor_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        return time.time() - self.timestamp


class StigmergicEnvironment:
    """Spatial environment where agents deposit and sense pheromone markers.

    Provides the interface expected by StigmergyMixin:
    deposit_marker(), sense_markers(), get_gradient(), get_strongest_marker().
    """

    def __init__(
        self,
        decay_rate: float = 0.05,
        diffusion_rate: float = 0.01,
        min_intensity: float = 0.01,
        grid_resolution: float = 1.0,
    ) -> None:
        self._decay_rate = decay_rate
        self._diffusion_rate = diffusion_rate
        self._min_intensity = min_intensity
        self._grid_resolution = grid_resolution

        self._markers: dict[str, StigmergicMarker] = {}
        self._spatial_index: dict[tuple[int, ...], list[str]] = defaultdict(list)
        self._lock = threading.RLock()

        # Statistics
        self._total_deposited = 0
        self._total_decayed = 0

    def _grid_key(self, position: tuple[float, ...]) -> tuple[int, ...]:
        """Convert continuous position to grid cell."""
        return tuple(int(p / self._grid_resolution) for p in position)

    def deposit_marker(
        self,
        marker_type: str,
        position: tuple[float, ...],
        intensity: float = 1.0,
        depositor_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Deposit a pheromone marker. Returns marker_id."""
        marker = StigmergicMarker(
            marker_type=marker_type,
            position=position,
            intensity=intensity,
            depositor_id=depositor_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._markers[marker.marker_id] = marker
            self._spatial_index[self._grid_key(position)].append(marker.marker_id)
            self._total_deposited += 1

        return marker.marker_id

    def sense_markers(
        self,
        position: tuple[float, ...],
        radius: float = 5.0,
        marker_types: list[str] | None = None,
    ) -> list[StigmergicMarker]:
        """Sense markers within radius of position."""
        self._apply_decay()

        results = []
        grid_pos = self._grid_key(position)
        search_radius = int(radius / self._grid_resolution) + 1

        with self._lock:
            # Search neighboring grid cells
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    key = (grid_pos[0] + dx, grid_pos[1] + dy) if len(grid_pos) >= 2 else (grid_pos[0] + dx,)
                    for marker_id in self._spatial_index.get(key, []):
                        marker = self._markers.get(marker_id)
                        if marker is None:
                            continue
                        if marker_types and marker.marker_type not in marker_types:
                            continue
                        dist = self._distance(position, marker.position)
                        if dist <= radius:
                            results.append(marker)

        # Sort by decayed intensity (strongest first)
        results.sort(
            key=lambda m: m.intensity * math.exp(-self._decay_rate * m.age),
            reverse=True,
        )
        return results

    def get_gradient(
        self,
        position: tuple[float, ...],
        marker_type: str,
        radius: float = 5.0,
    ) -> tuple[float, ...]:
        """Compute gradient direction toward strongest concentration."""
        markers = self.sense_markers(position, radius, [marker_type])
        if not markers:
            return tuple(0.0 for _ in position)

        # Weighted centroid of marker positions
        total_weight = 0.0
        weighted_pos = np.zeros(len(position))

        for marker in markers:
            weight = marker.intensity * math.exp(-self._decay_rate * marker.age)
            dist = self._distance(position, marker.position)
            if dist > 0:
                weight /= dist  # closer markers weigh more
            weighted_pos += np.array(marker.position[:len(position)]) * weight
            total_weight += weight

        if total_weight == 0:
            return tuple(0.0 for _ in position)

        centroid = weighted_pos / total_weight
        gradient = centroid - np.array(position)

        # Normalize
        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient = gradient / norm

        return tuple(float(g) for g in gradient)

    def get_strongest_marker(
        self,
        position: tuple[float, ...],
        marker_type: str | None = None,
        radius: float = 10.0,
    ) -> StigmergicMarker | None:
        """Get the strongest marker near position."""
        marker_types = [marker_type] if marker_type else None
        markers = self.sense_markers(position, radius, marker_types)
        return markers[0] if markers else None

    def _apply_decay(self) -> None:
        """Remove markers below minimum intensity."""
        now = time.time()
        to_remove = []

        with self._lock:
            for marker_id, marker in self._markers.items():
                decayed = marker.intensity * math.exp(-self._decay_rate * (now - marker.timestamp))
                if decayed < self._min_intensity:
                    to_remove.append(marker_id)

            for marker_id in to_remove:
                marker = self._markers.pop(marker_id, None)
                if marker:
                    key = self._grid_key(marker.position)
                    if marker_id in self._spatial_index.get(key, []):
                        self._spatial_index[key].remove(marker_id)
                    self._total_decayed += 1

    @staticmethod
    def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        """Euclidean distance between two positions."""
        dims = min(len(a), len(b))
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dims)))

    def get_statistics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_markers": len(self._markers),
                "total_deposited": self._total_deposited,
                "total_decayed": self._total_decayed,
                "grid_cells": len(self._spatial_index),
                "decay_rate": self._decay_rate,
            }

    def __repr__(self) -> str:
        return f"StigmergicEnvironment(markers={len(self._markers)}, deposited={self._total_deposited})"
