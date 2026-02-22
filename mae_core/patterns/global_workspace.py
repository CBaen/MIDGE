"""Global Workspace Theory (GWT) Competitive Ignition with Capacity Limits.

Implements the competitive ignition mechanism from Global Workspace Theory
(Baars 1988, Dehaene & Naccache 2001). Multiple neural coalitions (pattern
candidates) compete for access to a limited-capacity global workspace.
Only the winner gets broadcast; all others are suppressed.

Capacity model (Miller 1956, adapted via Law 2 - Triadic Generator):
- Workspace holds at most WORKSPACE_CAPACITY (3) active representations.
  Miller's classic 7+-2 applies to human working memory; Mae's workspace
  uses 3 as the triadic minimum -- the atom of stable structure (Law 2).
- After ignition, an attentional blink raises the threshold temporarily
  (Dehaene & Changeux 2011: strict seriality of conscious access, with
  the workspace inhibited post-ignition preventing rapid re-ignition).
- Related patterns (same domain, high activation) can be chunked into a
  single workspace slot, modeling how the brain groups related items to
  overcome capacity limits (Cowan 2001, Miller 1956).

Biological analogy: In the brain, sensory processing happens in parallel
across many cortical areas. Each area builds a "coalition" of active neurons.
These coalitions compete for access to a shared workspace (thought to involve
prefrontal-parietal networks). When one coalition's activation crosses a
threshold, it "ignites" -- entering conscious awareness and getting broadcast
to the entire cortex. All other coalitions are simultaneously suppressed.
The workspace itself has limited capacity: only a few representations can
be simultaneously active (the "bottleneck" of consciousness).

Mathematical identity:
- Law 1 (No Bare Dyads): Winning patterns require triadic verification
  via corroboration from independent domain signals.
- Law 2 (Triadic Generator): Workspace capacity = 3 (minimum stable count).
- Law 7 (Rule of 3/5): Minimum 3 candidates before competition occurs.
- Law 8, Property 7: Competition/selection -- this IS that property.
- Transfractal Compromise: GWT broadcast is the shortcut mechanism
  between fractal modules.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from mae_core.patterns.pattern_signal import PatternDomain, PatternSignal

logger = logging.getLogger(__name__)

# ── Tuning Constants ──────────────────────────────────────────────────

# Exponential moving average smoothing factor for activation accumulation.
# Higher alpha = more responsive to current salience, less memory.
# 0.4 means ~60% weight on history, ~40% on current -- biologically plausible
# for a system stepping at ~100ms intervals.
ACTIVATION_ALPHA = 0.4

# Activation must cross this to ignite. Biologically, this corresponds
# to the threshold where a neural coalition achieves enough synchrony
# to trigger the "ignition" cascade (Dehaene et al. 2003).
IGNITION_THRESHOLD = 0.7

# When a candidate ignites, all others' activation is multiplied by this.
# Models lateral inhibition in the cortex.
SUPPRESSION_FACTOR = 0.3

# After ignition, the winner cannot re-ignite for this many steps.
# Prevents perseveration / fixation. Biologically corresponds to the
# attentional blink and psychological refractory period.
REFRACTORY_STEPS = 3

# Minimum number of candidates required for competition (Rule of 3/5).
# Below this, the strongest wins by default (no real competition).
MIN_COMPETITORS = 3

# When winner has no corroboration (fails triadic verification),
# confidence is reduced by this factor.
UNCORROBORATED_PENALTY = 0.7

# ── Capacity Constants (Miller's Law adapted via Law 2) ──────────────

# Maximum number of active representations in the workspace at once.
# Miller (1956) found 7+-2 for human working memory. Mae uses 3:
# the triadic minimum from Law 2 (Triadic Generator). Three is the
# smallest count that provides stability, emergence, and self-awareness.
WORKSPACE_CAPACITY = 3

# After ignition, the workspace enters an "attentional blink" -- a brief
# period where the ignition threshold is raised, making it harder for
# new patterns to ignite. Models the ~500ms gap observed in attentional
# blink experiments (Raymond, Shapiro & Arnell 1992; Dehaene & Changeux
# 2011 GNW simulations showing strict seriality of conscious access).
BLINK_DURATION = 2

# How much the ignition threshold increases during the blink.
# During the blink, IGNITION_THRESHOLD effectively becomes
# IGNITION_THRESHOLD + BLINK_THRESHOLD_BOOST = 0.85.
BLINK_THRESHOLD_BOOST = 0.15

# Minimum activation for two same-domain patterns to be chunked.
# Chunking models how the brain groups related items into a single
# working memory slot (Cowan 2001). Both patterns must exceed this.
CHUNK_ACTIVATION_MIN = 0.5


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


class GlobalWorkspace:
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

        # Phi-driven threshold offset: lowered when phi is low (fragmentation
        # makes broadcast easier to encourage integration), raised when phi
        # is high (selectivity increases when system is already integrated).
        self._phi_threshold_offset: float = 0.0

        # New statistics counters
        self._total_workspace_fills: int = 0
        self._total_evictions: int = 0
        self._total_chunks_formed: int = 0
        self._total_blink_events: int = 0

    def compete(
        self,
        signals: list[PatternSignal],
        correlated_groups: list[list[PatternSignal]] | None = None,
    ) -> IgnitionResult:
        """Run one step of competitive ignition.

        Args:
            signals: All PatternSignals from this step's digest.
            correlated_groups: Groups of correlated signals from the
                PatternBus, used for triadic verification of the winner.

        Returns:
            IgnitionResult with the winner (if any) and diagnostic data.
        """
        correlated_groups = correlated_groups or []
        self._current_step += 1

        # 1. Tick refractory periods for all existing candidates
        for candidate in self._candidates.values():
            candidate.tick_refractory()

        # 1b. Tick attentional blink
        if self._blink_remaining > 0:
            self._blink_remaining -= 1

        # 2. Build domain-level candidates from this step's signals
        self._update_candidates(signals)

        # 2b. Attempt chunking of same-domain workspace items
        self._attempt_chunking()

        # 3. Get eligible candidates (not in refractory, have activation)
        eligible = [
            c for c in self._candidates.values()
            if c.can_ignite and c.activation > 0.0
        ]

        if not eligible:
            return IgnitionResult(
                activation_map=self._get_activation_map(),
                candidate_count=0,
                refractory_domains=self._get_refractory_domains(),
                workspace_contents=self._get_workspace_content_names(),
                blink_active=self._blink_remaining > 0,
            )

        # 4. Check if we have enough competitors for real competition
        if len(eligible) < MIN_COMPETITORS:
            # Not enough for real competition -- strongest wins by default
            winner_candidate = max(eligible, key=lambda c: c.activation)
            self._total_default_wins += 1

            corroborated = self._check_corroboration(
                winner_candidate.domain, correlated_groups,
            )

            return IgnitionResult(
                winner=winner_candidate.representative,
                ignited=False,  # No ignition, just default selection
                activation_map=self._get_activation_map(),
                candidate_count=len(eligible),
                corroborated=corroborated,
                refractory_domains=self._get_refractory_domains(),
                workspace_contents=self._get_workspace_content_names(),
                blink_active=self._blink_remaining > 0,
            )

        # 5. Real competition: check for ignition
        self._total_competitions += 1

        # Sort by activation (highest first)
        eligible.sort(key=lambda c: c.activation, reverse=True)
        top = eligible[0]

        # Calculate effective threshold (raised during attentional blink,
        # adjusted by phi-driven offset for integration responsiveness)
        effective_threshold = IGNITION_THRESHOLD + self._phi_threshold_offset
        if self._blink_remaining > 0:
            effective_threshold += BLINK_THRESHOLD_BOOST

        if top.activation >= effective_threshold:
            # IGNITION! Winner broadcasts, losers get suppressed
            self._total_ignitions += 1

            # Put winner in refractory
            top.refractory_remaining = REFRACTORY_STEPS

            # Suppress all others
            for candidate in eligible[1:]:
                candidate.activation *= SUPPRESSION_FACTOR
            self._total_suppressions += len(eligible) - 1

            # Check triadic corroboration
            corroborated = self._check_corroboration(
                top.domain, correlated_groups,
            )

            # Add winner to workspace contents (capacity-limited)
            self._add_to_workspace(top, corroborated)

            # Start attentional blink
            self._blink_remaining = BLINK_DURATION
            self._total_blink_events += 1

            logger.debug(
                "GWT ignition: %s (activation=%.3f, corroborated=%s, "
                "blink_started=True, workspace_size=%d/%d)",
                top.domain.value,
                top.activation,
                corroborated,
                len(self._workspace_contents),
                WORKSPACE_CAPACITY,
            )

            return IgnitionResult(
                winner=top.representative,
                ignited=True,
                activation_map=self._get_activation_map(),
                candidate_count=len(eligible),
                corroborated=corroborated,
                refractory_domains=self._get_refractory_domains(),
                workspace_contents=self._get_workspace_content_names(),
                blink_active=True,
            )

        # 6. No ignition yet -- competition ongoing, no winner
        return IgnitionResult(
            activation_map=self._get_activation_map(),
            candidate_count=len(eligible),
            refractory_domains=self._get_refractory_domains(),
            workspace_contents=self._get_workspace_content_names(),
            blink_active=self._blink_remaining > 0,
        )

    # ── Workspace Capacity Management ─────────────────────────────────

    def _add_to_workspace(
        self, candidate: WorkspaceCandidate, corroborated: bool,
    ) -> None:
        """Add an ignited candidate to the workspace contents.

        If the workspace is at capacity, the oldest item is evicted.
        Evicted items lose broadcast status and enter refractory.

        Biological basis: The global workspace has limited capacity
        (Baars 1988). Only a few coalitions can be simultaneously
        active. New ignitions displace the oldest representation,
        modeling the serial bottleneck of conscious access.
        """
        # Check if workspace is already at capacity
        was_full = len(self._workspace_contents) >= WORKSPACE_CAPACITY

        if was_full:
            # Evict the oldest item
            evicted = self._workspace_contents[0]
            # The deque's maxlen will handle the eviction automatically
            # when we append, but we track stats first.
            self._total_evictions += 1

            # Evicted patterns enter refractory (they've been displaced
            # from consciousness and need time to re-ignite)
            evicted_domain = evicted.domain
            if evicted_domain in self._candidates:
                evicted_candidate = self._candidates[evicted_domain]
                if evicted_candidate.refractory_remaining <= 0:
                    evicted_candidate.refractory_remaining = REFRACTORY_STEPS

            logger.debug(
                "GWT eviction: %s displaced from workspace by %s",
                evicted.domain.value,
                candidate.domain.value,
            )

        # Build the workspace item
        confidence = candidate.representative.confidence if candidate.representative else 0.0
        item = WorkspaceItem(
            domain=candidate.domain,
            representative=candidate.representative,
            activation=candidate.activation,
            confidence=confidence,
            entry_step=self._current_step,
        )

        # Append to workspace (deque maxlen handles eviction)
        self._workspace_contents.append(item)

        if len(self._workspace_contents) >= WORKSPACE_CAPACITY:
            self._total_workspace_fills += 1

    def _attempt_chunking(self) -> None:
        """Try to chunk same-domain items in the workspace.

        Chunking models how the brain groups related items into a single
        working memory slot (Cowan 2001, Miller 1956). When two items in
        the workspace share the same domain and both have activation above
        CHUNK_ACTIVATION_MIN, they merge into a single chunked item.

        The chunk inherits the higher confidence and combined salience,
        freeing a workspace slot for a new representation.

        Biological basis: In working memory research, "chunking" is the
        process of grouping related items so they consume only one slot.
        Expert chess players chunk familiar board positions; musicians
        chunk note sequences. Same principle here.
        """
        if len(self._workspace_contents) < 2:
            return

        # Find pairs of same-domain items eligible for chunking
        items = list(self._workspace_contents)
        domain_items: dict[PatternDomain, list[int]] = {}
        for i, item in enumerate(items):
            if item.is_chunk:
                continue  # Already chunked items can't re-chunk
            domain_items.setdefault(item.domain, []).append(i)

        chunked_indices: set[int] = set()

        for domain, indices in domain_items.items():
            if len(indices) < 2:
                continue

            # Check both items have sufficient activation
            candidate = self._candidates.get(domain)
            if candidate is None or candidate.activation < CHUNK_ACTIVATION_MIN:
                continue

            # Check that each item's activation at entry was high enough
            pair = indices[:2]  # Chunk the first two
            item_a = items[pair[0]]
            item_b = items[pair[1]]

            if item_a.activation < CHUNK_ACTIVATION_MIN:
                continue
            if item_b.activation < CHUNK_ACTIVATION_MIN:
                continue

            # Create the chunk
            higher_confidence = max(item_a.confidence, item_b.confidence)
            combined_activation = min(
                1.0, item_a.activation + item_b.activation,
            )

            chunk = WorkspaceItem(
                domain=domain,
                representative=item_a.representative if item_a.activation >= item_b.activation else item_b.representative,
                activation=combined_activation,
                confidence=higher_confidence,
                is_chunk=True,
                chunk_members=[domain, domain],
                entry_step=min(item_a.entry_step, item_b.entry_step),
            )

            chunked_indices.add(pair[0])
            chunked_indices.add(pair[1])

            # Rebuild workspace: remove chunked items, add chunk
            new_contents: list[WorkspaceItem] = []
            for i, item in enumerate(items):
                if i not in chunked_indices:
                    new_contents.append(item)
            new_contents.append(chunk)

            self._workspace_contents.clear()
            for item in new_contents:
                self._workspace_contents.append(item)

            self._total_chunks_formed += 1

            logger.debug(
                "GWT chunk formed: 2x %s merged (confidence=%.3f, "
                "activation=%.3f)",
                domain.value,
                higher_confidence,
                combined_activation,
            )

            # Only one chunking operation per step (avoid cascading)
            break

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

    # ── Candidate Management ──────────────────────────────────────────

    def _update_candidates(self, signals: list[PatternSignal]) -> None:
        """Update candidate activations from this step's signals.

        Each domain gets one candidate. The candidate's representative
        is the highest-salience signal in that domain. Activation
        accumulates via exponential moving average.
        """
        # Group signals by domain
        by_domain: dict[PatternDomain, list[PatternSignal]] = {}
        for sig in signals:
            by_domain.setdefault(sig.domain, []).append(sig)

        # Track which domains are present this step
        present_domains = set(by_domain.keys())

        # Update present domains: EMA accumulation
        for domain, domain_signals in by_domain.items():
            # Representative = highest salience in this domain
            representative = max(domain_signals, key=lambda s: s.salience)

            # Current salience = aggregate of all signals in this domain,
            # capped at 1.0
            current_salience = min(
                1.0, sum(s.salience for s in domain_signals),
            )

            if domain not in self._candidates:
                self._candidates[domain] = WorkspaceCandidate(domain=domain)

            candidate = self._candidates[domain]
            # Exponential moving average
            candidate.activation = (
                ACTIVATION_ALPHA * current_salience
                + (1 - ACTIVATION_ALPHA) * candidate.activation
            )
            candidate.representative = representative
            candidate.last_salience = current_salience

        # Decay absent domains (activation decays but does not reset)
        for domain, candidate in self._candidates.items():
            if domain not in present_domains:
                candidate.activation *= (1 - ACTIVATION_ALPHA)
                candidate.last_salience = 0.0

    def _check_corroboration(
        self,
        winner_domain: PatternDomain,
        correlated_groups: list[list[PatternSignal]],
    ) -> bool:
        """Check if the winner has triadic support (Law 1: No Bare Dyads).

        A winner is corroborated if at least one correlated group
        contains signals from the winner's domain from 2+ different
        source systems. This means multiple independent sources agree
        on the pattern -- triadic verification.
        """
        for group in correlated_groups:
            # Check if any group contains signals matching winner's domain
            matching = [s for s in group if s.domain == winner_domain]
            if len(matching) >= 2:
                # 2+ signals in a correlated group = triadic support
                # (signal A + signal B + the correlation itself = triad)
                return True

        return False

    def _get_activation_map(self) -> dict[str, float]:
        """Return current activation levels by domain name."""
        return {
            domain.value: round(candidate.activation, 4)
            for domain, candidate in self._candidates.items()
            if candidate.activation > 0.001
        }

    def _get_refractory_domains(self) -> list[str]:
        """Return domains currently in refractory period."""
        return [
            domain.value
            for domain, candidate in self._candidates.items()
            if candidate.refractory_remaining > 0
        ]

    # ── Phi-driven modulation ────────────────────────────────────────

    def _on_phi_measurement(self, channel: str, message: Any) -> None:
        """Adjust ignition threshold based on organism integration (phi).

        Low phi (< 0.3) = fragmentation. Lower the ignition threshold
        so patterns broadcast more easily, encouraging information
        integration across disconnected subsystems.

        High phi (> 0.7) = strong integration. Raise the threshold
        slightly so only the most salient patterns ignite — the system
        is already well-integrated and can afford selectivity.

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
