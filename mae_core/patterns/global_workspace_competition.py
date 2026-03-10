"""Global Workspace Theory -- competition engine mixin.

Extracted from global_workspace.py to keep the core class under the 500-line cap.
Import from mae_core.patterns.global_workspace for all public names.
"""

from __future__ import annotations

import logging
from typing import Any

from mae_core.patterns.pattern_signal import PatternDomain, PatternSignal
from mae_core.patterns.global_workspace_models import (
    WorkspaceCandidate,
    WorkspaceItem,
    IgnitionResult,
)

logger = logging.getLogger(__name__)

# ── Tuning Constants (re-exported for backward compatibility) ─────────
ACTIVATION_ALPHA = 0.4
IGNITION_THRESHOLD = 0.7
SUPPRESSION_FACTOR = 0.3
REFRACTORY_STEPS = 3
MIN_COMPETITORS = 3
UNCORROBORATED_PENALTY = 0.7
WORKSPACE_CAPACITY = 3
BLINK_DURATION = 2
BLINK_THRESHOLD_BOOST = 0.15
CHUNK_ACTIVATION_MIN = 0.5


class _GlobalWorkspaceCompetitionMixin:
    """Mixin providing the GWT competition engine, workspace management, and chunking.

    Mixed into GlobalWorkspace. Requires the following attributes:
        _candidates, _workspace_contents, _blink_remaining, _current_step,
        _phi_threshold_offset,
        _total_ignitions, _total_competitions, _total_suppressions,
        _total_default_wins, _total_workspace_fills, _total_evictions,
        _total_chunks_formed, _total_blink_events
    """

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
        """
        # Check if workspace is already at capacity
        was_full = len(self._workspace_contents) >= WORKSPACE_CAPACITY

        if was_full:
            # Evict the oldest item
            evicted = self._workspace_contents[0]
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
        working memory slot (Cowan 2001, Miller 1956).
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

    # ── Candidate Management ──────────────────────────────────────────

    def _update_candidates(self, signals: list[PatternSignal]) -> None:
        """Update candidate activations from this step's signals.

        Each domain gets one candidate. Activation accumulates via
        exponential moving average.
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

            # Current salience = aggregate of all signals in this domain, capped
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
        source systems.
        """
        for group in correlated_groups:
            matching = [s for s in group if s.domain == winner_domain]
            if len(matching) >= 2:
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
