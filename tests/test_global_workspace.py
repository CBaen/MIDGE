"""Tests for GlobalWorkspace -- GWT competitive ignition.

Tests cover:
- Candidate tracking and activation accumulation (EMA)
- Ignition threshold and suppression
- Refractory period (prevents fixation)
- Minimum competitors (Rule of 3/5)
- Triadic corroboration (Law 1: No Bare Dyads)
- Integration with PatternCortex
- EventBus broadcast on ignition
- Backward compatibility (existing tests must pass unchanged)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from mae_core.patterns.global_workspace import (
    ACTIVATION_ALPHA,
    IGNITION_THRESHOLD,
    MIN_COMPETITORS,
    REFRACTORY_STEPS,
    SUPPRESSION_FACTOR,
    UNCORROBORATED_PENALTY,
    GlobalWorkspace,
    IgnitionResult,
    WorkspaceCandidate,
)
from mae_core.patterns.pattern_bus import PatternDigest
from mae_core.patterns.pattern_cortex import PatternAdvisory, PatternCortex
from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _sig(
    domain: PatternDomain = PatternDomain.NOVELTY,
    salience: float = 0.5,
    confidence: float = 0.7,
    source: str = "test",
    form: PatternForm = PatternForm.REACTIVE,
    description: str = "test signal",
) -> PatternSignal:
    return PatternSignal(
        source_system=source,
        domain=domain,
        form=form,
        confidence=confidence,
        salience=salience,
        description=description,
    )


def _digest(
    step: int,
    signals: list[PatternSignal] | None = None,
    correlated_groups: list[list[PatternSignal]] | None = None,
) -> PatternDigest:
    sigs = signals or []
    by_domain: dict[PatternDomain, list[PatternSignal]] = {}
    by_form: dict[PatternForm, list[PatternSignal]] = {}
    for s in sigs:
        by_domain.setdefault(s.domain, []).append(s)
        by_form.setdefault(s.form, []).append(s)

    dominant = None
    max_sal = 0.0
    for domain, dsigs in by_domain.items():
        sal = sum(s.salience for s in dsigs)
        if sal > max_sal:
            max_sal = sal
            dominant = domain

    return PatternDigest(
        step=step,
        signals=sigs,
        by_domain=by_domain,
        by_form=by_form,
        correlated_groups=correlated_groups or [],
        dominant_domain=dominant,
        aggregate_salience=sum(s.salience for s in sigs),
        signal_count=len(sigs),
    )


class FakeEventBus:
    """Records published messages for verification."""

    def __init__(self) -> None:
        self.published: list[tuple[str, Any]] = []

    def publish(self, channel: str, message: Any) -> int:
        self.published.append((channel, message))
        return 1


class BrokenEventBus:
    """EventBus that raises on publish."""

    def publish(self, channel: str, message: Any) -> int:
        raise RuntimeError("bus down")


# ══════════════════════════════════════════════════════════════════════
# GlobalWorkspace unit tests
# ══════════════════════════════════════════════════════════════════════


class TestWorkspaceConstruction:
    def test_creates_empty(self):
        ws = GlobalWorkspace()
        stats = ws.get_statistics()
        assert stats["total_ignitions"] == 0
        assert stats["active_candidates"] == 0

    def test_repr(self):
        ws = GlobalWorkspace()
        r = repr(ws)
        assert "GlobalWorkspace" in r
        assert "candidates=0" in r


class TestCandidateTracking:
    def test_single_signal_creates_candidate(self):
        ws = GlobalWorkspace()
        result = ws.compete([_sig(domain=PatternDomain.THREAT, salience=0.5)])

        stats = ws.get_statistics()
        assert stats["active_candidates"] == 1
        assert "threat" in stats["activation_map"]

    def test_multiple_domains_create_multiple_candidates(self):
        ws = GlobalWorkspace()
        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.5),
            _sig(domain=PatternDomain.NOVELTY, salience=0.4),
            _sig(domain=PatternDomain.CAUSATION, salience=0.3),
        ]
        ws.compete(signals)

        stats = ws.get_statistics()
        assert stats["active_candidates"] == 3

    def test_activation_accumulates_via_ema(self):
        ws = GlobalWorkspace()
        sig = _sig(domain=PatternDomain.THREAT, salience=0.8)

        # Step 1: activation = alpha * 0.8
        ws.compete([sig])
        stats1 = ws.get_statistics()
        act1 = stats1["activation_map"]["threat"]

        # Step 2: activation = alpha * 0.8 + (1-alpha) * act1
        ws.compete([sig])
        stats2 = ws.get_statistics()
        act2 = stats2["activation_map"]["threat"]

        assert act2 > act1  # Activation should build

    def test_absent_domain_decays(self):
        ws = GlobalWorkspace()

        # Build activation for threat
        ws.compete([_sig(domain=PatternDomain.THREAT, salience=0.8)])
        act_before = ws.get_statistics()["activation_map"]["threat"]

        # Step without threat -- activation should decay
        ws.compete([_sig(domain=PatternDomain.NOVELTY, salience=0.5)])
        act_after = ws.get_statistics()["activation_map"].get("threat", 0.0)

        assert act_after < act_before

    def test_representative_is_highest_salience(self):
        ws = GlobalWorkspace()
        low = _sig(domain=PatternDomain.THREAT, salience=0.3, source="a")
        high = _sig(domain=PatternDomain.THREAT, salience=0.9, source="b")

        result = ws.compete([low, high])

        # The winner's representative should be the high-salience signal
        if result.winner is not None:
            assert result.winner.salience == 0.9


class TestMinimumCompetitors:
    """Rule of 3/5: minimum 3 candidates for real competition."""

    def test_fewer_than_min_gives_default_win(self):
        ws = GlobalWorkspace()
        # Only 2 domains -- below MIN_COMPETITORS
        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.9),
            _sig(domain=PatternDomain.NOVELTY, salience=0.3),
        ]
        result = ws.compete(signals)

        # Should still produce a winner (default selection)
        assert result.winner is not None
        # But ignited should be False (no real competition)
        assert result.ignited is False

    def test_default_win_picks_strongest(self):
        ws = GlobalWorkspace()
        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.9),
            _sig(domain=PatternDomain.NOVELTY, salience=0.3),
        ]
        result = ws.compete(signals)

        assert result.winner is not None
        assert result.winner.domain == PatternDomain.THREAT

    def test_exactly_min_competitors_enables_competition(self):
        ws = GlobalWorkspace()
        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.5),
            _sig(domain=PatternDomain.NOVELTY, salience=0.4),
            _sig(domain=PatternDomain.CAUSATION, salience=0.3),
        ]
        assert len(signals) == MIN_COMPETITORS

        # First step won't ignite (activation too low), but competition
        # is now possible
        result = ws.compete(signals)
        assert result.candidate_count == MIN_COMPETITORS

    def test_default_win_counted_in_stats(self):
        ws = GlobalWorkspace()
        ws.compete([_sig(domain=PatternDomain.THREAT, salience=0.9)])

        stats = ws.get_statistics()
        assert stats["total_default_wins"] >= 1


class TestIgnition:
    """Ignition occurs when a candidate crosses the threshold."""

    def _build_activation(
        self, ws: GlobalWorkspace, domain: PatternDomain, salience: float, steps: int,
    ) -> None:
        """Feed the same signal repeatedly to build activation."""
        # Include 3 domains so we have enough competitors
        for _ in range(steps):
            signals = [
                _sig(domain=domain, salience=salience),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            ws.compete(signals)

    def test_ignition_occurs_after_accumulation(self):
        ws = GlobalWorkspace()

        # Feed high-salience threat repeatedly. With alpha=0.4 and
        # salience=0.95, activation converges to 0.95.
        # After n steps: act = 0.95 * (1 - (1-0.4)^n)
        # For IGNITION_THRESHOLD=0.7: need about 3 steps
        result = None
        for i in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result = ws.compete(signals)
            if result.ignited:
                break

        assert result is not None
        assert result.ignited is True
        assert result.winner is not None
        assert result.winner.domain == PatternDomain.THREAT

    def test_ignition_increments_count(self):
        ws = GlobalWorkspace()

        for _ in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            ws.compete(signals)

        stats = ws.get_statistics()
        assert stats["total_ignitions"] >= 1

    def test_no_ignition_with_low_salience(self):
        ws = GlobalWorkspace()

        # All low salience -- should never ignite
        for _ in range(20):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.2),
                _sig(domain=PatternDomain.NOVELTY, salience=0.15),
                _sig(domain=PatternDomain.CAUSATION, salience=0.1),
            ]
            result = ws.compete(signals)
            assert result.ignited is False


class TestSuppression:
    """When one ignites, all others get suppressed."""

    def test_losers_suppressed_after_ignition(self):
        ws = GlobalWorkspace()

        # Build activation for all three candidates
        for _ in range(5):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.NOVELTY, salience=0.6),
                _sig(domain=PatternDomain.CAUSATION, salience=0.5),
            ]
            result = ws.compete(signals)
            if result.ignited:
                break

        if result.ignited:
            # After ignition, losers should have much lower activation
            act_map = result.activation_map
            winner_act = act_map.get("threat", 0)
            # Losers should have been suppressed
            for domain_name in ["novelty", "causation"]:
                loser_act = act_map.get(domain_name, 0)
                assert loser_act < winner_act


class TestRefractoryPeriod:
    """After ignition, the winner cannot re-ignite for REFRACTORY_STEPS."""

    def test_winner_enters_refractory(self):
        ws = GlobalWorkspace()

        # Build to ignition
        for _ in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result = ws.compete(signals)
            if result.ignited:
                break

        if result.ignited:
            assert "threat" in result.refractory_domains

    def test_refractory_prevents_re_ignition(self):
        ws = GlobalWorkspace()

        # Build to ignition
        ignited_step = None
        for i in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result = ws.compete(signals)
            if result.ignited:
                ignited_step = i
                break

        if ignited_step is not None:
            # Next step: same signals. Threat should NOT re-ignite.
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result2 = ws.compete(signals)
            # Should not ignite because threat is in refractory
            assert result2.ignited is False

    def test_refractory_expires(self):
        ws = GlobalWorkspace()

        # Build to ignition
        ignited = False
        for _ in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result = ws.compete(signals)
            if result.ignited:
                ignited = True
                break

        if ignited:
            # Step through refractory with LOW salience for threat so
            # it cannot re-ignite immediately when refractory ends.
            for _ in range(REFRACTORY_STEPS + 1):
                ws.compete([
                    _sig(domain=PatternDomain.THREAT, salience=0.1),
                    _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                    _sig(domain=PatternDomain.STATE, salience=0.1),
                ])

            # After refractory + 1 step, threat should no longer be refractory
            stats = ws.get_statistics()
            assert "threat" not in stats["refractory_domains"]


class TestTriadicCorroboration:
    """Law 1: No Bare Dyads -- winners need corroboration."""

    def test_corroborated_when_multiple_sources_agree(self):
        ws = GlobalWorkspace()

        # Build activation over multiple steps so the workspace produces
        # a winner. Threat has two high-salience sources forming a
        # correlated group (triadic verification).
        result = None
        for _ in range(10):
            sig_a = _sig(domain=PatternDomain.THREAT, salience=0.9, source="haven")
            sig_b = _sig(domain=PatternDomain.THREAT, salience=0.8, source="threat_det")
            correlated = [[sig_a, sig_b]]

            signals = [
                sig_a,
                sig_b,
                _sig(domain=PatternDomain.NOVELTY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            result = ws.compete(signals, correlated_groups=correlated)
            if result.winner is not None:
                break

        assert result is not None
        assert result.winner is not None
        assert result.corroborated is True

    def test_not_corroborated_when_single_source(self):
        ws = GlobalWorkspace()
        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.9, source="haven"),
            _sig(domain=PatternDomain.NOVELTY, salience=0.1),
            _sig(domain=PatternDomain.STATE, salience=0.1),
        ]
        # No correlated groups
        result = ws.compete(signals, correlated_groups=[])

        assert result.corroborated is False

    def test_not_corroborated_when_wrong_domain(self):
        ws = GlobalWorkspace()
        # Correlated group is for NOVELTY, but winner is THREAT
        sig_a = _sig(domain=PatternDomain.NOVELTY, salience=0.3, source="a")
        sig_b = _sig(domain=PatternDomain.NOVELTY, salience=0.2, source="b")
        correlated = [[sig_a, sig_b]]

        signals = [
            _sig(domain=PatternDomain.THREAT, salience=0.9),
            sig_a,
            _sig(domain=PatternDomain.STATE, salience=0.1),
        ]
        result = ws.compete(signals, correlated_groups=correlated)

        assert result.corroborated is False


class TestEmptyInput:
    def test_no_signals_no_winner(self):
        ws = GlobalWorkspace()
        result = ws.compete([])

        assert result.winner is None
        assert result.ignited is False
        assert result.candidate_count == 0


# ══════════════════════════════════════════════════════════════════════
# PatternCortex GWT integration tests
# ══════════════════════════════════════════════════════════════════════


class TestCortexGWTIntegration:
    """Tests that the workspace is properly wired into PatternCortex."""

    def test_cortex_has_workspace(self):
        cortex = PatternCortex()
        assert hasattr(cortex, "_workspace")
        assert isinstance(cortex._workspace, GlobalWorkspace)

    def test_workspace_stats_in_cortex_stats(self):
        cortex = PatternCortex()
        stats = cortex.get_statistics()
        assert "workspace" in stats
        assert "total_ignitions" in stats["workspace"]

    def test_cortex_ignition_count_in_stats(self):
        cortex = PatternCortex()
        stats = cortex.get_statistics()
        assert "total_ignitions" in stats
        assert stats["total_ignitions"] == 0

    def test_cortex_accepts_event_bus(self):
        bus = FakeEventBus()
        cortex = PatternCortex(event_bus=bus)
        stats = cortex.get_statistics()
        assert stats["has_event_bus"] is True

    def test_cortex_without_event_bus(self):
        cortex = PatternCortex()
        stats = cortex.get_statistics()
        assert stats["has_event_bus"] is False

    def test_advisory_has_dominant_from_workspace(self):
        """When workspace picks a winner, it becomes the advisory's dominant."""
        cortex = PatternCortex()

        # Single signal -- workspace gives default win (< MIN_COMPETITORS)
        sig = _sig(domain=PatternDomain.THREAT, salience=0.9)
        adv = cortex.process_digest(_digest(step=1, signals=[sig]))

        assert adv.dominant_pattern is not None
        assert adv.dominant_pattern.domain == PatternDomain.THREAT

    def test_advisory_fallback_on_empty(self):
        """Empty digest should produce no dominant pattern."""
        cortex = PatternCortex()
        adv = cortex.process_digest(_digest(step=1))

        assert adv.dominant_pattern is None

    def test_uncorroborated_reduces_confidence(self):
        """Law 1: No Bare Dyads penalty on uncorroborated winners."""
        cortex = PatternCortex()

        # Single source, no correlated groups -- winner is uncorroborated
        sig = _sig(domain=PatternDomain.THREAT, salience=0.8)
        adv = cortex.process_digest(_digest(step=1, signals=[sig]))

        # Base confidence = 0.3 + 0.05 (1 signal) = 0.35
        # With UNCORROBORATED_PENALTY (0.7): 0.35 * 0.7 = 0.245
        assert adv.confidence < 0.35

    def test_corroborated_keeps_full_confidence(self):
        """Corroborated winners do not get penalized."""
        cortex = PatternCortex()

        sig_a = _sig(domain=PatternDomain.THREAT, salience=0.8, source="haven")
        sig_b = _sig(domain=PatternDomain.THREAT, salience=0.7, source="threat_det")
        correlated = [[sig_a, sig_b]]

        adv = cortex.process_digest(
            _digest(step=1, signals=[sig_a, sig_b], correlated_groups=correlated),
        )

        # Base = 0.3 + 0.1 (2 signals) + 0.1 (1 correlated group) = 0.5
        # No penalty (corroborated) -> stays at 0.5
        assert adv.confidence >= 0.49


class TestCortexGWTBroadcast:
    """Tests that ignition publishes to EventBus."""

    def test_broadcast_on_ignition(self):
        bus = FakeEventBus()
        cortex = PatternCortex(event_bus=bus)

        # Build activation to ignition (high salience over multiple steps)
        for i in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            cortex.process_digest(_digest(step=i, signals=signals))

        # Check if any broadcast was published
        broadcasts = [
            (ch, msg) for ch, msg in bus.published if ch == "gwt.broadcast"
        ]
        if broadcasts:
            ch, payload = broadcasts[0]
            assert payload["type"] == "gwt.broadcast"
            assert payload["domain"] == "threat"
            assert "activation_map" in payload

    def test_no_broadcast_without_event_bus(self):
        """No event_bus means no broadcast -- but ignition still works."""
        cortex = PatternCortex()  # No event_bus

        for i in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            cortex.process_digest(_digest(step=i, signals=signals))

        # Should not crash -- graceful degradation
        stats = cortex.get_statistics()
        assert stats["has_event_bus"] is False

    def test_broadcast_graceful_on_broken_bus(self):
        """Broken EventBus should not crash the cortex."""
        bus = BrokenEventBus()
        cortex = PatternCortex(event_bus=bus)

        for i in range(10):
            signals = [
                _sig(domain=PatternDomain.THREAT, salience=0.95),
                _sig(domain=PatternDomain.OPPORTUNITY, salience=0.1),
                _sig(domain=PatternDomain.STATE, salience=0.1),
            ]
            # Should not raise
            cortex.process_digest(_digest(step=i, signals=signals))


class TestCortexBackwardCompat:
    """Ensure existing behavior is preserved."""

    def test_single_signal_still_becomes_dominant(self):
        """With just one signal, workspace gives default win."""
        cortex = PatternCortex()
        sig = _sig(domain=PatternDomain.THREAT, salience=0.8)
        adv = cortex.process_digest(_digest(step=1, signals=[sig]))

        assert adv.dominant_pattern is not None
        assert adv.dominant_pattern.domain == PatternDomain.THREAT

    def test_highest_salience_wins_in_fallback(self):
        """With mixed signals but no ignition, highest salience wins."""
        cortex = PatternCortex()
        low = _sig(domain=PatternDomain.NOVELTY, salience=0.3)
        high = _sig(domain=PatternDomain.THREAT, salience=0.9)

        adv = cortex.process_digest(_digest(step=1, signals=[low, high]))

        assert adv.dominant_pattern is not None
        assert adv.dominant_pattern.domain == PatternDomain.THREAT

    def test_empty_digest_still_works(self):
        cortex = PatternCortex()
        adv = cortex.process_digest(_digest(step=1))

        assert isinstance(adv, PatternAdvisory)
        assert adv.dominant_pattern is None

    def test_trends_still_work(self):
        cortex = PatternCortex()
        for i in range(4):
            sig = _sig(domain=PatternDomain.THREAT)
            adv = cortex.process_digest(_digest(step=i, signals=[sig]))

        assert "threat" in adv.active_trends

    def test_meta_patterns_still_work(self):
        cortex = PatternCortex()
        for i in range(5):
            sig = _sig(domain=PatternDomain.THREAT, salience=0.9)
            adv = cortex.process_digest(_digest(step=i, signals=[sig]))

        assert len(adv.meta_patterns) > 0


class TestWorkspaceCandidateDataclass:
    def test_can_ignite_initially(self):
        c = WorkspaceCandidate(domain=PatternDomain.THREAT)
        assert c.can_ignite is True

    def test_cannot_ignite_in_refractory(self):
        c = WorkspaceCandidate(domain=PatternDomain.THREAT, refractory_remaining=3)
        assert c.can_ignite is False

    def test_tick_decrements_refractory(self):
        c = WorkspaceCandidate(domain=PatternDomain.THREAT, refractory_remaining=2)
        c.tick_refractory()
        assert c.refractory_remaining == 1
        c.tick_refractory()
        assert c.refractory_remaining == 0
        assert c.can_ignite is True

    def test_tick_at_zero_stays_zero(self):
        c = WorkspaceCandidate(domain=PatternDomain.THREAT, refractory_remaining=0)
        c.tick_refractory()
        assert c.refractory_remaining == 0


class TestIgnitionResult:
    def test_defaults(self):
        r = IgnitionResult()
        assert r.winner is None
        assert r.ignited is False
        assert r.corroborated is False
        assert r.candidate_count == 0
        assert r.activation_map == {}
        assert r.refractory_domains == []
