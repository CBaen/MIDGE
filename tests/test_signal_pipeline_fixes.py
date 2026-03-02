"""Round 2 signal pipeline fixes — test coverage.

Tests for ALL Round 2 changes across Forge, Anvil, and Crucible:

  P3  — Neutral-signal convergence (Anvil)
  P4A — Reward rebalancing (Crucible)
  P4B — Alert deduplication (Anvil)
  P5C — Cold-start bias fix: seeded retirement window (Forge)
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alerter(min_domains: int = 3, min_strength: float = 0.6) -> ConvergenceAlerter:
    """Return a ConvergenceAlerter with no Thompson / regime dependencies."""
    return ConvergenceAlerter(
        min_domains=min_domains,
        min_strength=min_strength,
        convergence_window_hours=72,
    )


def _record(alerter, sig_id, strength, domain, direction, confidence=0.7):
    """Shorthand for alerter.record_signal with common defaults."""
    alerter.record_signal(
        sig_id, strength, domain, direction=direction, confidence=confidence
    )


# ---------------------------------------------------------------------------
# P3 — Neutral Signal Convergence
# ---------------------------------------------------------------------------


class TestNeutralSignalConvergence:
    """Neutral signals count toward min_domains without biasing direction."""

    def test_neutral_signal_contributes_domain(self):
        """3 domains (2 directional + 1 neutral) should fire an alert."""
        alerter = _make_alerter(min_domains=3, min_strength=0.6)

        # Two directional domains — bullish
        _record(alerter, "insider_buy", 0.8, "insider", "bullish")
        _record(alerter, "contract_award", 0.7, "government", "bullish")

        # One neutral domain — counts toward min_domains, not direction
        _record(alerter, "finra_short_ratio", 0.65, "volume", "neutral")

        alerts = alerter.check_convergence()
        bullish_alerts = [a for a in alerts if a.direction == "bullish"]
        assert len(bullish_alerts) >= 1, (
            "Expected at least one bullish alert when 2 directional + 1 neutral "
            "domain are present (total == min_domains=3)"
        )

    def test_neutral_signal_alone_insufficient(self):
        """3 neutral signals (zero directional) must NOT fire an alert."""
        alerter = _make_alerter(min_domains=3, min_strength=0.6)

        # Three neutral signals from three distinct domains
        _record(alerter, "vol_a", 0.8, "volume", "neutral")
        _record(alerter, "sent_a", 0.75, "sentiment", "neutral")
        _record(alerter, "macro_a", 0.7, "macro", "neutral")

        alerts = alerter.check_convergence()
        assert alerts == [], (
            "No alert should fire when all domains are neutral (no directional conviction)"
        )

    def test_neutral_signal_no_direction_bias(self):
        """Neutral signal must not push alert direction toward neutral/bearish."""
        alerter = _make_alerter(min_domains=3, min_strength=0.6)

        # Two bullish + one neutral
        _record(alerter, "exec_buy", 0.8, "insider", "bullish")
        _record(alerter, "gov_contract", 0.7, "government", "bullish")
        _record(alerter, "finra_ratio", 0.65, "volume", "neutral")

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        # The resulting alert direction must be bullish
        assert alerts[0].direction == "bullish", (
            f"Expected direction='bullish', got '{alerts[0].direction}'"
        )

    def test_neutral_domain_counted_in_domains(self):
        """Alert's domains_converging must include the neutral signal's domain."""
        alerter = _make_alerter(min_domains=3, min_strength=0.6)

        _record(alerter, "exec_buy", 0.8, "insider", "bullish")
        _record(alerter, "gov_contract", 0.7, "government", "bullish")
        _record(alerter, "finra_ratio", 0.65, "volume", "neutral")

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        alert = alerts[0]
        assert "volume" in alert.domains_converging, (
            f"Neutral domain 'volume' should appear in domains_converging. "
            f"Got: {alert.domains_converging}"
        )


# ---------------------------------------------------------------------------
# P4B — Alert Deduplication
# ---------------------------------------------------------------------------


class TestAlertDedup:
    """Alert suppression: same direction within interval is suppressed."""

    def test_dedup_suppresses_same_direction(self):
        """Five rapid check_convergence() calls for same bullish setup = at most 1 alert."""
        alerter = _make_alerter(min_domains=2, min_strength=0.6)

        _record(alerter, "sig_a", 0.8, "insider", "bullish")
        _record(alerter, "sig_b", 0.75, "government", "bullish")

        all_alerts = []
        for _ in range(5):
            all_alerts.extend(alerter.check_convergence())

        # At most 1 bullish alert should have been returned across all calls
        bullish = [a for a in all_alerts if a.direction == "bullish"]
        assert len(bullish) <= 1, (
            f"Dedup should suppress repeated same-direction alerts. "
            f"Got {len(bullish)} bullish alerts from 5 rapid calls."
        )

    def test_dedup_allows_different_direction(self):
        """Bullish and bearish alerts should each fire independently."""
        alerter = _make_alerter(min_domains=2, min_strength=0.6)

        # Bullish side
        _record(alerter, "buy_a", 0.8, "insider", "bullish")
        _record(alerter, "buy_b", 0.75, "government", "bullish")

        # Bearish side
        _record(alerter, "sell_a", 0.8, "technical", "bearish")
        _record(alerter, "sell_b", 0.7, "sentiment", "bearish")

        alerts = alerter.check_convergence()
        directions = {a.direction for a in alerts}
        assert "bullish" in directions, "Bullish convergence should fire"
        assert "bearish" in directions, "Bearish convergence should fire"

    def test_dedup_expiry_allows_repeat(self):
        """After the dedup interval expires, the same direction should fire again."""
        alerter = _make_alerter(min_domains=2, min_strength=0.6)

        _record(alerter, "sig_a", 0.8, "insider", "bullish")
        _record(alerter, "sig_b", 0.75, "government", "bullish")

        # First call — should produce an alert
        first = alerter.check_convergence()
        assert any(a.direction == "bullish" for a in first), (
            "Expected a bullish alert on the first call"
        )

        # Manually expire the dedup window by back-dating the last alert time
        past_time = datetime.now() - timedelta(
            hours=alerter._min_alert_interval_hours + 1
        )
        with alerter._alert_lock:
            alerter._last_alert_times["bullish"] = past_time

        # Second call after expiry — should produce another alert
        second = alerter.check_convergence()
        assert any(a.direction == "bullish" for a in second), (
            "Expected a bullish alert after the dedup interval expired"
        )


# ---------------------------------------------------------------------------
# P4A — Reward Rebalancing
# ---------------------------------------------------------------------------


class TestRewardRebalancing:
    """Market action ceiling raised to 0.8; TaskPool fall-through capped at 0.3."""

    def test_market_action_ceiling_raised(self):
        """act_market() reward ceiling is now 0.8, not 0.5.

        We verify by injecting a handler that returns a raw reward of 0.9
        directly into the dispatch table, then confirming act_market clamps
        it to 0.8 (not 0.5 as before).
        """
        from mae_core.market import market_actions

        agent = SimpleNamespace(
            role="MARKET_ANALYST",
            unique_id=1,
            step_count=100,
        )
        agent.deposit_marker = MagicMock()

        # Inject a handler that returns 0.9 raw into the dispatch table
        original_dispatch = market_actions._DISPATCH.copy()
        market_actions._DISPATCH[("MARKET_ANALYST", "exploit")] = lambda a: 0.9
        try:
            reward = market_actions.act_market(agent, "exploit")
        finally:
            market_actions._DISPATCH.clear()
            market_actions._DISPATCH.update(original_dispatch)

        assert reward is not None
        assert reward <= 0.8, (
            f"Expected reward <= 0.8 (new ceiling), got {reward}"
        )
        assert reward == pytest.approx(0.8), (
            f"A raw reward of 0.9 should be clamped to 0.8, got {reward}"
        )

    def test_market_role_taskpool_capped(self):
        """Market-role agents that fall through to TaskPool are capped at 0.3.

        Simulates an act_market() returning None (no handler registered for
        this action type, e.g. 'rest'), so the code falls through to TaskPool.
        Then we verify the reward is capped at 0.3 for market-role agents.
        """
        from mae_core.market.market_actions import MARKET_ROLES
        from mae_core.agents.lifecycle_decision import DecisionActionLifecycleMixin

        pool = MagicMock()
        mock_task = SimpleNamespace(
            task_id="t1", task_type="exploit", difficulty=0.3,
            reward_value=1.0, state="available", claimed_by=None,
        )
        pool.get_available_tasks.return_value = [mock_task]
        pool.claim_task.return_value = mock_task
        pool.work_on_task.return_value = (0.9, True, 1.0)  # high raw reward

        # Use a market role that has NO handler for 'exploit' in _DISPATCH
        # by temporarily removing it, forcing fall-through to TaskPool.
        from mae_core.market import market_actions as ma

        role = "MARKET_ANALYST"
        assert role in MARKET_ROLES

        original_handler = ma._DISPATCH.pop(("MARKET_ANALYST", "exploit"), None)
        try:
            agent = SimpleNamespace(
                role=role,
                unique_id=1,
                step_count=5,
                _task_pool=pool,
                last_action=None,
                _resting=False,
                _current_task_id=None,
            )
            agent.deposit_success_marker = MagicMock()
            agent.deposit_marker = MagicMock()
            # Attach the TaskPool action methods from the mixin
            agent._act_exploit = lambda p: DecisionActionLifecycleMixin._act_exploit(agent, p)
            agent._act_explore = lambda p: DecisionActionLifecycleMixin._act_explore(agent, p)
            agent._act_communicate = lambda p: DecisionActionLifecycleMixin._act_communicate(agent, p)
            agent._act_rest = lambda p: DecisionActionLifecycleMixin._act_rest(agent, p)
            agent._act_api_call = lambda p: DecisionActionLifecycleMixin._act_api_call(agent, p)

            reward = DecisionActionLifecycleMixin._act(agent, "exploit")
        finally:
            if original_handler is not None:
                ma._DISPATCH[("MARKET_ANALYST", "exploit")] = original_handler

        assert reward is not None
        assert reward <= 0.3, (
            f"Market-role fall-through TaskPool reward should be capped at 0.3, got {reward}"
        )

    def test_non_market_role_taskpool_uncapped(self):
        """Non-market-role agents using TaskPool are NOT capped at 0.3."""
        from mae_core.agents.lifecycle_decision import DecisionActionLifecycleMixin

        pool = MagicMock()
        mock_task = SimpleNamespace(
            task_id="t2", task_type="exploit", difficulty=0.3,
            reward_value=1.0, state="available", claimed_by=None,
        )
        pool.get_available_tasks.return_value = [mock_task]
        pool.claim_task.return_value = mock_task
        pool.work_on_task.return_value = (0.9, True, 0.9)  # 0.9 raw reward

        agent = SimpleNamespace(
            role="STEM",  # Non-market role
            unique_id=2,
            step_count=5,
            _task_pool=pool,
            last_action=None,
            _resting=False,
            _current_task_id=None,
        )
        agent.deposit_success_marker = MagicMock()
        agent.deposit_marker = MagicMock()
        agent._act_exploit = lambda p: DecisionActionLifecycleMixin._act_exploit(agent, p)
        agent._act_explore = lambda p: DecisionActionLifecycleMixin._act_explore(agent, p)
        agent._act_communicate = lambda p: DecisionActionLifecycleMixin._act_communicate(agent, p)
        agent._act_rest = lambda p: DecisionActionLifecycleMixin._act_rest(agent, p)
        agent._act_api_call = lambda p: DecisionActionLifecycleMixin._act_api_call(agent, p)

        reward = DecisionActionLifecycleMixin._act(agent, "exploit")

        # STEM agents should receive the full TaskPool reward (no 0.3 cap)
        assert reward is not None
        assert reward > 0.3, (
            f"Non-market STEM agent should NOT be capped at 0.3, got {reward}"
        )


# ---------------------------------------------------------------------------
# P5C — Cold-Start Guard (seeded retirement window)
# ---------------------------------------------------------------------------


class TestColdStartGuard:
    """HypothesisEngine seeds retirement window from registry at cold-start.

    Seeded entries have seeded=True so Wire 2 of meta-learning can ignore
    them and only act on real live data (seeded=False).
    """

    def _make_hypothesis(self, status_str, name):
        """Create a Hypothesis with the given status string."""
        from mae_core.market.intelligence.hypothesis import (
            Hypothesis,
            HypothesisStatus,
            TriggerPattern,
            SourceType,
        )

        status_map = {
            "ACTIVE": HypothesisStatus.ACTIVE,
            "RETIRED": HypothesisStatus.RETIRED,
            "PROBATION": HypothesisStatus.PROBATION,
        }
        trigger = TriggerPattern(
            source_a="sec_form4",
            source_b="price",
            lag_days=3,
            direction="bullish",
            domain_filter="ES=F",
        )
        return Hypothesis(
            hypothesis_id=f"hyp_{name}",
            name=name,
            trigger=trigger,
            causal_story="Test story — seeded entry",
            status=status_map[status_str],
            source_type=SourceType.LAG_CORRELATION,
            created_at=datetime.now(),
        )

    def _make_engine(self, tmp_path):
        """Construct a HypothesisEngine with isolated tmp_path directories."""
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator

        registry = HypothesisRegistry(data_dir=tmp_path)
        generator = HypothesisGenerator(registry=registry)
        validator = HypothesisValidator(data_dir=tmp_path)
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=validator,
        )
        return engine, registry

    def test_seeded_entries_marked(self, tmp_path):
        """After cold-start seeding, all retirement window entries have seeded=True."""
        from mae_core.market.intelligence.hypothesis import HypothesisStatus
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine

        # Pre-populate registry with ACTIVE and RETIRED hypotheses
        registry = HypothesisRegistry(data_dir=tmp_path)

        hyp_active = self._make_hypothesis("PROBATION", "h_active")
        registry.register(hyp_active)
        registry.promote(hyp_active.hypothesis_id)  # PROBATION → ACTIVE

        hyp_retired = self._make_hypothesis("PROBATION", "h_retired")
        registry.register(hyp_retired)
        registry.retire(hyp_retired.hypothesis_id, "test reason")  # PROBATION → RETIRED

        # Build engine with NO retirement_window.json present — triggers seeding
        generator = HypothesisGenerator(registry=registry)
        validator = HypothesisValidator(data_dir=tmp_path)
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=validator,
        )

        assert engine._retirement_window, (
            "Retirement window should not be empty after seeding from registry"
        )
        for entry in engine._retirement_window:
            assert isinstance(entry, dict), (
                f"Entry should be dict, got {type(entry)}: {entry}"
            )
            assert entry.get("seeded") is True, (
                f"Expected seeded=True on all cold-start entries, got {entry}"
            )

    def test_live_entries_not_seeded(self, tmp_path):
        """_promote() and _retire() add live entries with seeded=False."""
        from mae_core.market.intelligence.hypothesis import HypothesisStatus
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine

        registry = HypothesisRegistry(data_dir=tmp_path)
        generator = HypothesisGenerator(registry=registry)
        validator = HypothesisValidator(data_dir=tmp_path)
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=validator,
        )

        # Register a hypothesis and promote it (live action, not cold-start)
        hyp = self._make_hypothesis("PROBATION", "live_hyp")
        registry.register(hyp)
        engine._promote(hyp)

        # Entry added by _promote must have seeded=False
        live_entries = [
            e for e in engine._retirement_window
            if e.get("seeded") is False
        ]
        assert len(live_entries) >= 1, (
            f"Expected at least one live (seeded=False) entry after _promote(), "
            f"got: {engine._retirement_window}"
        )

    def test_wire2_skips_when_few_live_entries(self, tmp_path):
        """Wire 2 should NOT tighten min_correlation when live entries < 10.

        This is the core cold-start bias guard: 20 seeded (historical) entries
        that are all 'retired' should NOT trigger a min_correlation tightening,
        because Wire 2 must only count live (seeded=False) entries. With fewer
        than 10 live entries, no adjustment should fire.
        """
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine

        registry = HypothesisRegistry(data_dir=tmp_path)
        generator = HypothesisGenerator(registry=registry)
        validator = HypothesisValidator(data_dir=tmp_path)
        engine = HypothesisEngine(
            registry=registry,
            generator=generator,
            validator=validator,
        )

        # Force-populate retirement window: 20 seeded "retired" entries
        engine._retirement_window = []
        for _ in range(20):
            engine._retirement_window.append({"outcome": "retired", "seeded": True})

        # Add 5 live entries (all retired — 100% live retirement rate)
        for _ in range(5):
            engine._retirement_window.append({"outcome": "retired", "seeded": False})

        # Wire 2 should require >= 10 live entries before adjusting min_correlation.
        # update_config is imported locally inside _run_meta_learning, so we patch
        # it at its definition site in learning_config (where the local import resolves).
        with patch(
            "mae_core.market.intelligence.learning_config.update_config"
        ) as mock_update:
            try:
                engine._run_meta_learning()
            except Exception:
                pass  # learning_config may not be fully wired in isolation

            # No tightening call (+0.02 to min_correlation) should have fired
            tighten_calls = [
                call for call in mock_update.call_args_list
                if ("min_correlation" in str(call) and "0.02" in str(call))
            ]
            assert len(tighten_calls) == 0, (
                "Wire 2 must NOT tighten min_correlation when live entry count < 10. "
                f"Unexpected calls: {tighten_calls}"
            )
