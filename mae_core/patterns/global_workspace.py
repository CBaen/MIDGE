"""Global Workspace Theory (GWT) Competitive Ignition with Capacity Limits.

Implements the competitive ignition mechanism from Global Workspace Theory
(Baars 1988, Dehaene & Naccache 2001). Multiple neural coalitions (pattern
candidates) compete for access to a limited-capacity global workspace.
Only the winner gets broadcast; all others are suppressed.

Capacity model (Miller 1956, adapted via Law 2 - Triadic Generator):
- Workspace holds at most WORKSPACE_CAPACITY (3) active representations.
- After ignition, an attentional blink raises the threshold temporarily.
- Related patterns (same domain, high activation) can be chunked into a
  single workspace slot.

Mathematical identity:
- Law 1 (No Bare Dyads): Winning patterns require triadic verification.
- Law 2 (Triadic Generator): Workspace capacity = 3.
- Law 7 (Rule of 3/5): Minimum 3 candidates before competition occurs.
- Law 8, Property 7: Competition/selection.
- Transfractal Compromise: GWT broadcast is the shortcut mechanism.

Implementation is split across sub-modules for the 500-line cap:
  global_workspace_models.py      -- WorkspaceCandidate, WorkspaceItem,
                                     IgnitionResult
  global_workspace_competition.py -- _GlobalWorkspaceCompetitionMixin
                                     (compete, workspace management, chunking)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from mae_core.patterns.pattern_signal import PatternDomain, PatternSignal
from mae_core.patterns.global_workspace_models import (
    WorkspaceCandidate,
    WorkspaceItem,
    IgnitionResult,
)
from mae_core.patterns.global_workspace_competition import (
    _GlobalWorkspaceCompetitionMixin,
    ACTIVATION_ALPHA,
    IGNITION_THRESHOLD,
    SUPPRESSION_FACTOR,
    REFRACTORY_STEPS,
    MIN_COMPETITORS,
    UNCORROBORATED_PENALTY,
    WORKSPACE_CAPACITY,
    BLINK_DURATION,
    BLINK_THRESHOLD_BOOST,
    CHUNK_ACTIVATION_MIN,
)

logger = logging.getLogger(__name__)


class GlobalWorkspace(_GlobalWorkspaceCompetitionMixin):
    """GWT competitive ignition -- patterns compete for broadcast.

    Biological analogy: Multiple neural coalitions form in parallel.
    Each builds activation strength. When one crosses the ignition
    threshold, it "ignites" -- gets broadcast to the entire cortex
    while others are suppressed.

    Capacity model: The workspace holds at most WORKSPACE_CAPACITY (3)
    active representations. When a new pattern ignites and the workspace
    is full, the oldest representation is evicted. After ignition, an
    attentional blink temporarily raises the threshold, enforcing the
    serial bottleneck of consciousness (Dehaene & Changeux 2011).

    Mathematical identity: Law 8 Property 7 (competition/selection),
    Law 2 (Triadic Generator: capacity = 3).
    """

    def __init__(self) -> None:
        self._candidates: dict[PatternDomain, WorkspaceCandidate] = {}
        self._total_ignitions: int = 0
        self._total_competitions: int = 0
        self._total_suppressions: int = 0
        self._total_default_wins: int = 0

        # Capacity tracking (Miller's Law adapted via Law 2)
        self._workspace_contents: deque[WorkspaceItem] = deque(
            maxlen=WORKSPACE_CAPACITY,
        )
        self._blink_remaining: int = 0
        self._current_step: int = 0

        # Phi-driven threshold offset
        self._phi_threshold_offset: float = 0.0

        # New statistics counters
        self._total_workspace_fills: int = 0
        self._total_evictions: int = 0
        self._total_chunks_formed: int = 0
        self._total_blink_events: int = 0

    def get_workspace_contents(self) -> list[WorkspaceItem]:
        """Return the currently active workspace items.

        Each item represents a pattern that has ignited and is still
        occupying a workspace slot. Items may be single representations
        or chunks of related patterns.
        """
        return list(self._workspace_contents)

    def _get_workspace_content_names(self) -> list[str]:
        """Return domain names of current workspace contents."""
        names = []
        for item in self._workspace_contents:
            if item.is_chunk:
                names.append(f"{item.domain.value}[chunked]")
            else:
                names.append(item.domain.value)
        return names

    # ── Phi-driven modulation ────────────────────────────────────────

    def _on_phi_measurement(self, channel: str, message: Any) -> None:
        """Adjust ignition threshold based on organism integration (phi).

        Low phi (< 0.3) = fragmentation. Lower the ignition threshold
        so patterns broadcast more easily, encouraging information
        integration across disconnected subsystems.

        High phi (> 0.7) = strong integration. Raise the threshold
        slightly so only the most salient patterns ignite.

        Advisory: offset clamped to [-0.05, +0.05]. Never blocks.
        """
        import json as _json
        try:
            msg = _json.loads(message) if isinstance(message, str) else message
        except (TypeError, ValueError):
            return
        if not isinstance(msg, dict):
            return

        phi = msg.get("organism_mean_phi")
        if phi is None or not isinstance(phi, (int, float)):
            return

        if phi < 0.3:
            self._phi_threshold_offset = -0.05
        elif phi > 0.7:
            self._phi_threshold_offset = 0.05
        else:
            self._phi_threshold_offset = 0.0

    # ── Statistics ────────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Return workspace statistics for monitoring."""
        return {
            "total_ignitions": self._total_ignitions,
            "total_competitions": self._total_competitions,
            "total_suppressions": self._total_suppressions,
            "total_default_wins": self._total_default_wins,
            "active_candidates": len([
                c for c in self._candidates.values()
                if c.activation > 0.001
            ]),
            "refractory_domains": self._get_refractory_domains(),
            "activation_map": self._get_activation_map(),
            # Capacity statistics
            "workspace_contents": self._get_workspace_content_names(),
            "workspace_size": len(self._workspace_contents),
            "workspace_capacity": WORKSPACE_CAPACITY,
            "blink_remaining": self._blink_remaining,
            "total_workspace_fills": self._total_workspace_fills,
            "total_evictions": self._total_evictions,
            "total_chunks_formed": self._total_chunks_formed,
            "total_blink_events": self._total_blink_events,
        }

    def __repr__(self) -> str:
        active = len([
            c for c in self._candidates.values()
            if c.activation > 0.001
        ])
        ws_size = len(self._workspace_contents)
        return (
            f"GlobalWorkspace(candidates={active}, "
            f"ignitions={self._total_ignitions}, "
            f"workspace={ws_size}/{WORKSPACE_CAPACITY})"
        )
