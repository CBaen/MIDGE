"""Tests for market action dispatch — role-keyed agent behavior.

Tests that market-role agents route to correct handlers and produce
valid reward values. Uses mock agents with configurable refs.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.market_actions import (
    MARKET_ROLES,
    act_market,
    _get_advisory,
    _get_alert,
    _get_alerter,
)


# ── Mock Agent Factory ──────────────────────────────────────────────────


def _make_agent(
    role="STEM",
    advisory=None,
    alerter=None,
    regime_classifier=None,
    hypothesis_engine=None,
    hypothesis_registry=None,
    model_ctx=None,
    step_count=100,
):
    """Create a mock agent with configurable market refs."""
    agent = SimpleNamespace(
        role=role,
        unique_id=42,
        step_count=step_count,
        _prediction_error=0.1,
        risk_score=0.2,
    )
    if advisory is not None:
        agent._market_advisory_ref = advisory
    if alerter is not None:
        agent._convergence_alerter_ref = alerter
    if regime_classifier is not None:
        agent._regime_classifier_ref = regime_classifier
    if hypothesis_engine is not None:
        agent._hypothesis_engine_ref = hypothesis_engine
    if hypothesis_registry is not None:
        agent._hypothesis_registry_ref = hypothesis_registry
    if model_ctx is not None:
        agent._model_ctx_ref = model_ctx

    # deposit_marker mock
    agent.deposit_marker = MagicMock()
    return agent


def _make_alerter(signals=None, domain_status=None):
    """Create a mock ConvergenceAlerter."""
    alerter = SimpleNamespace()
    alerter.signals = signals or {}
    alerter.get_domain_status = MagicMock(
        return_value=domain_status or {}
    )
    return alerter


def _make_advisory(alert=None, ticker_alerts=None, updated_step=100):
    """Create a market advisory dict."""
    adv = {"alert": alert, "updated_step": updated_step}
    if ticker_alerts is not None:
        adv["ticker_alerts"] = ticker_alerts
    return adv


def _make_alert(direction="bullish", strength=0.8, domains=None):
    """Create an alert dict."""
    return {
        "direction": direction,
        "strength": strength,
        "domains": domains or ["insider", "regulatory"],
    }


# ── Core Dispatch Tests ─────────────────────────────────────────────────


class TestActMarketDispatch:
    """Test the act_market() routing logic."""

    def test_stem_agent_returns_none(self):
        """STEM agents should fall through to TaskPool."""
        agent = _make_agent(role="STEM")
        assert act_market(agent, "explore") is None

    def test_unknown_role_returns_none(self):
        agent = _make_agent(role="EXPLORER")
        assert act_market(agent, "explore") is None

    def test_non_dispatched_action_returns_none(self):
        """rest and api_call should fall through for market agents."""
        agent = _make_agent(role="SEC_WATCHER")
        assert act_market(agent, "rest") is None
        assert act_market(agent, "api_call") is None

    def test_reward_clamped_to_range(self):
        """Rewards must be in [0.0, 0.5]."""
        alerter = _make_alerter(signals={"insider": list(range(100))})
        agent = _make_agent(
            role="SEC_WATCHER",
            alerter=alerter,
        )
        reward = act_market(agent, "explore")
        assert reward is not None
        assert 0.0 <= reward <= 0.5

    def test_all_market_roles_have_explore_exploit_communicate(self):
        """Every market role must dispatch explore, exploit, communicate."""
        from mae_core.market.market_actions import _DISPATCH
        for role in MARKET_ROLES:
            for action in ("explore", "exploit", "communicate"):
                assert (role, action) in _DISPATCH, f"Missing ({role}, {action})"

    def test_handler_exception_returns_zero(self):
        """If a handler throws, reward should be 0.0, not crash."""
        agent = _make_agent(role="SEC_WATCHER")
        # No alerter ref → _sec_scan returns 0.0 (graceful)
        reward = act_market(agent, "explore")
        assert reward == 0.0


# ── SEC_WATCHER Tests ────────────────────────────────────────────────────


class TestSecWatcher:
    def test_scan_no_signals(self):
        alerter = _make_alerter(signals={"insider": [], "regulatory": []})
        agent = _make_agent(role="SEC_WATCHER", alerter=alerter)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_scan_with_signals(self):
        alerter = _make_alerter(signals={
            "insider": [1, 2, 3, 4, 5],
            "regulatory": [1, 2, 3],
        })
        agent = _make_agent(role="SEC_WATCHER", alerter=alerter)
        reward = act_market(agent, "explore")
        assert 0.0 < reward <= 0.5
        agent.deposit_marker.assert_called_once()

    def test_scan_no_alerter(self):
        agent = _make_agent(role="SEC_WATCHER")
        reward = act_market(agent, "explore")
        assert reward == 0.0

    def test_deepen_no_alert(self):
        agent = _make_agent(role="SEC_WATCHER")
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.05)

    def test_deepen_with_strong_alert(self):
        alert = _make_alert(direction="bullish", strength=0.85, domains=["insider", "regulatory"])
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="SEC_WATCHER", advisory=advisory)
        reward = act_market(agent, "exploit")
        assert reward > 0.1

    def test_deepen_weak_non_insider_alert(self):
        alert = _make_alert(direction="neutral", strength=0.3, domains=["government"])
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="SEC_WATCHER", advisory=advisory)
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.05)


# ── CONTRACT_TRACKER Tests ──────────────────────────────────────────────


class TestContractTracker:
    def test_scan_no_signals(self):
        alerter = _make_alerter(signals={"government": [], "contracts": []})
        agent = _make_agent(role="CONTRACT_TRACKER", alerter=alerter)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_scan_with_signals(self):
        alerter = _make_alerter(signals={
            "government": [1, 2, 3, 4],
            "contracts": [1, 2],
        })
        agent = _make_agent(role="CONTRACT_TRACKER", alerter=alerter)
        reward = act_market(agent, "explore")
        assert 0.0 < reward <= 0.5

    def test_deepen_with_ticker_alerts(self):
        advisory = _make_advisory(
            alert=_make_alert(strength=0.6),
            ticker_alerts=[{"strength": 0.7, "direction": "bullish"}],
        )
        agent = _make_agent(role="CONTRACT_TRACKER", advisory=advisory)
        reward = act_market(agent, "exploit")
        assert reward > 0.1


# ── MARKET_ANALYST Tests ────────────────────────────────────────────────


class TestMarketAnalyst:
    def test_convergence_scan_empty(self):
        alerter = _make_alerter(domain_status={})
        agent = _make_agent(role="MARKET_ANALYST", alerter=alerter)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_convergence_scan_active_domains(self):
        domain_status = {
            "insider": {"direction": "bullish", "strength": 0.7, "signal_count": 5},
            "government": {"direction": "bullish", "strength": 0.5, "signal_count": 3},
            "regulatory": {"direction": "neutral", "strength": 0.3, "signal_count": 2},
        }
        alerter = _make_alerter(domain_status=domain_status)
        advisory = _make_advisory()
        agent = _make_agent(role="MARKET_ANALYST", alerter=alerter, advisory=advisory)
        reward = act_market(agent, "explore")
        assert reward > 0.05

    def test_convergence_deepen_strong_alert(self):
        alert = _make_alert(strength=0.8, domains=["insider", "government", "regulatory"])
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="MARKET_ANALYST", advisory=advisory)
        reward = act_market(agent, "exploit")
        assert reward > 0.15

    def test_convergence_deepen_weak_alert(self):
        alert = _make_alert(strength=0.2)
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="MARKET_ANALYST", advisory=advisory)
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.05)


# ── HYPOTHESIS_EXPLORER Tests ───────────────────────────────────────────


class TestHypothesisExplorer:
    def test_generate_no_engine(self):
        agent = _make_agent(role="HYPOTHESIS_EXPLORER")
        reward = act_market(agent, "explore")
        assert reward == 0.0

    def test_generate_with_cooldown(self):
        engine = MagicMock()
        engine.request_generation.return_value = 0  # Cooldown active
        agent = _make_agent(role="HYPOTHESIS_EXPLORER", hypothesis_engine=engine)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_generate_success(self, tmp_path):
        engine = MagicMock()
        engine.request_generation.return_value = 3
        agent = _make_agent(role="HYPOTHESIS_EXPLORER", hypothesis_engine=engine)
        with patch("mae_core.market.market_actions._OUTPUT_DIR", tmp_path):
            reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.3)
        # Verify hypothesis activity log is written to tmp_path, not data/midge/
        log_path = tmp_path / "hypothesis_activity.jsonl"
        assert log_path.exists(), "hypothesis_activity.jsonl must be written during generation"
        record = json.loads(log_path.read_text().strip())
        assert record["event"] == "generated"
        assert record["count"] == 3

    def test_deepen_no_registry(self):
        agent = _make_agent(role="HYPOTHESIS_EXPLORER")
        reward = act_market(agent, "exploit")
        assert reward == 0.0

    def test_deepen_no_active(self):
        registry = MagicMock()
        registry.get_active.return_value = []
        agent = _make_agent(role="HYPOTHESIS_EXPLORER", hypothesis_registry=registry)
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.05)

    def test_deepen_with_active(self):
        hyp = SimpleNamespace(name="test_hyp", stats=SimpleNamespace(total_observations=10))
        registry = MagicMock()
        registry.get_active.return_value = [hyp]
        agent = _make_agent(role="HYPOTHESIS_EXPLORER", hypothesis_registry=registry)
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.1)
        agent.deposit_marker.assert_called_once()


# ── HYPOTHESIS_VALIDATOR Tests ──────────────────────────────────────────


class TestHypothesisValidator:
    def test_sample_no_registry(self):
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR")
        reward = act_market(agent, "explore")
        assert reward == 0.0

    def test_sample_no_probation(self):
        registry = MagicMock()
        registry.get_probation.return_value = []
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR", hypothesis_registry=registry)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_sample_validatable(self):
        hyp = SimpleNamespace(stats=SimpleNamespace(total_observations=25))
        registry = MagicMock()
        registry.get_probation.return_value = [hyp]
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR", hypothesis_registry=registry)
        reward = act_market(agent, "explore")
        assert reward == pytest.approx(0.05)

    def test_validate_promoted(self, tmp_path):
        engine = MagicMock()
        engine.request_validation.return_value = "promoted"
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR", hypothesis_engine=engine)
        with patch("mae_core.market.market_actions._OUTPUT_DIR", tmp_path):
            reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.4)
        # Verify hypothesis activity is logged to tmp_path, not data/midge/
        log_path = tmp_path / "hypothesis_activity.jsonl"
        assert log_path.exists(), "hypothesis_activity.jsonl must be written on promotion"
        record = json.loads(log_path.read_text().strip())
        assert record["event"] == "promoted"

    def test_validate_retired(self, tmp_path):
        engine = MagicMock()
        engine.request_validation.return_value = "retired"
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR", hypothesis_engine=engine)
        with patch("mae_core.market.market_actions._OUTPUT_DIR", tmp_path):
            reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.2)
        # Verify hypothesis activity is logged to tmp_path, not data/midge/
        log_path = tmp_path / "hypothesis_activity.jsonl"
        assert log_path.exists(), "hypothesis_activity.jsonl must be written on retirement"
        record = json.loads(log_path.read_text().strip())
        assert record["event"] == "retired"

    def test_validate_busy(self):
        engine = MagicMock()
        engine.request_validation.return_value = "busy"
        agent = _make_agent(role="HYPOTHESIS_VALIDATOR", hypothesis_engine=engine)
        reward = act_market(agent, "exploit")
        assert reward == pytest.approx(0.05)


# ── Broadcast Tests ─────────────────────────────────────────────────────


class TestMarketBroadcast:
    def test_broadcast_no_alert(self):
        agent = _make_agent(role="SEC_WATCHER")
        reward = act_market(agent, "communicate")
        assert reward == pytest.approx(0.05)

    def test_broadcast_with_alert(self, tmp_path):
        alert = _make_alert(strength=0.7)
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="MARKET_ANALYST", advisory=advisory)
        with patch("mae_core.market.market_actions._OUTPUT_DIR", tmp_path):
            reward = act_market(agent, "communicate")
        assert reward == pytest.approx(0.1)
        # Check activity log was written
        log_path = tmp_path / "agent_activity.jsonl"
        assert log_path.exists()
        record = json.loads(log_path.read_text().strip())
        assert record["role"] == "MARKET_ANALYST"
        assert record["action"] == "communicate"


# ── Advisory Helper Tests ───────────────────────────────────────────────


class TestAdvisoryHelpers:
    def test_get_advisory_missing_ref(self):
        agent = SimpleNamespace(role="SEC_WATCHER")
        assert _get_advisory(agent) == {}

    def test_get_advisory_with_ref(self):
        adv = {"alert": {"strength": 0.5}}
        agent = SimpleNamespace(_market_advisory_ref=adv)
        assert _get_advisory(agent) == adv

    def test_get_alert_stale(self):
        alert = _make_alert(strength=0.9)
        advisory = _make_advisory(alert=alert, updated_step=10)
        agent = _make_agent(advisory=advisory, step_count=100)
        assert _get_alert(agent) is None  # 100 - 10 = 90 > 50 threshold

    def test_get_alert_fresh(self):
        alert = _make_alert(strength=0.9)
        advisory = _make_advisory(alert=alert, updated_step=80)
        agent = _make_agent(advisory=advisory, step_count=100)
        result = _get_alert(agent)
        assert result is not None
        assert result["strength"] == 0.9

    def test_get_alerter_missing(self):
        agent = SimpleNamespace()
        assert _get_alerter(agent) is None


# ── Market Awareness Integration ────────────────────────────────────────


class TestMarketAwareness:
    """Test market_awareness.py functions used by the action system."""

    def test_market_stimulus_encoding(self):
        from mae_core.market.market_awareness import get_market_context_for_router
        agent = _make_agent(role="MARKET_ANALYST")
        ctx = get_market_context_for_router(agent)
        assert "market_stimulus" in ctx
        assert ctx["market_stimulus"] == "market:ambient"

    def test_strong_convergence_stimulus(self):
        from mae_core.market.market_awareness import get_market_context_for_router
        alert = _make_alert(direction="bullish", strength=0.9)
        advisory = _make_advisory(alert=alert)
        agent = _make_agent(role="MARKET_ANALYST", advisory=advisory)
        ctx = get_market_context_for_router(agent)
        assert ctx["market_stimulus"] == "convergence:strong:bullish"

    def test_hypothesis_empty_stimulus(self):
        from mae_core.market.market_awareness import get_market_context_for_router
        registry = MagicMock()
        registry.get_statistics.return_value = {"active_count": 0, "best_active_win_rate": 0.0}
        agent = _make_agent(role="HYPOTHESIS_EXPLORER", hypothesis_registry=registry)
        # No convergence, no hypotheses
        ctx = get_market_context_for_router(agent)
        assert ctx["market_stimulus"] == "hypothesis:empty"
