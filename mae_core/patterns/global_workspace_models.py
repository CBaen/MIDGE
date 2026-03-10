"""Global Workspace Theory -- data models only.

Extracted from global_workspace.py to keep the core class under the 500-line cap.
Import from mae_core.patterns.global_workspace for all public names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mae_core.patterns.pattern_signal import PatternDomain, PatternSignal


@dataclass
class WorkspaceCandidate:
    """A candidate competing for workspace access.

    Each candidate represents a domain-level coalition: all signals
    from a particular domain aggregated into a single competitor.
    """

    domain: PatternDomain
    activation: float = 0.0
    representative: PatternSignal | None = None
    refractory_remaining: int = 0
    last_salience: float = 0.0

    @property
    def can_ignite(self) -> bool:
        """Whether this candidate is eligible for ignition."""
        return self.refractory_remaining <= 0

    def tick_refractory(self) -> None:
        """Decrement refractory counter by one step."""
        if self.refractory_remaining > 0:
            self.refractory_remaining -= 1


@dataclass
class WorkspaceItem:
    """An active representation occupying a workspace slot.

    Each item in the workspace is either a single candidate or a chunk
    of related candidates that share a domain.
    """

    domain: PatternDomain
    representative: PatternSignal | None
    activation: float
    confidence: float
    is_chunk: bool = False
    chunk_members: list[PatternDomain] = field(default_factory=list)
    entry_step: int = 0


@dataclass
class IgnitionResult:
    """The outcome of a single competition step.

    Attributes:
        winner: The signal that won broadcast access, or None.
        ignited: Whether ignition occurred this step.
        activation_map: Current activation levels by domain.
        candidate_count: How many candidates competed.
        corroborated: Whether the winner had triadic support.
        refractory_domains: Domains currently in refractory period.
        workspace_contents: Currently active workspace items (domain names).
        blink_active: Whether the attentional blink is currently active.
    """

    winner: PatternSignal | None = None
    ignited: bool = False
    activation_map: dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    corroborated: bool = False
    refractory_domains: list[str] = field(default_factory=list)
    workspace_contents: list[str] = field(default_factory=list)
    blink_active: bool = False
