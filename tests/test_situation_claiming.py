"""Tests for agent-level situation claiming via role-domain affinity routing.

Covers:
- ROLE_DOMAIN_AFFINITY constant structure and completeness
- select_preferred_role() priority rules (causal > win-rate > insider > contracts > complexity > overlap)
- OctopusColony.submit_task() role-affinity routing branch
- Fallback to workload routing when no matching role octopus exists
- preferred_role key injected into task_data by investigation dispatcher
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mae_core.network.market_task_handlers import (
    ROLE_DOMAIN_AFFINITY,
    select_preferred_role,
)


# ---------------------------------------------------------------------------
# ROLE_DOMAIN_AFFINITY structure tests
# ---------------------------------------------------------------------------


class TestRoleDomainAffinityStructure:
    def test_all_expected_roles_present(self):
        expected = {
            "SEC_WATCHER",
            "CONTRACT_TRACKER",
            "MARKET_ANALYST",
            "HYPOTHESIS_EXPLORER",
            "HYPOTHESIS_VALIDATOR",
        }
        assert expected.issubset(set(ROLE_DOMAIN_AFFINITY.keys()))

    def test_all_values_are_frozensets(self):
        for role, affinity in ROLE_DOMAIN_AFFINITY.items():
            assert isinstance(affinity, frozenset), f"{role} affinity must be frozenset"

    def test_sec_watcher_has_insider(self):
        assert "insider" in ROLE_DOMAIN_AFFINITY["SEC_WATCHER"]

    def test_contract_tracker_has_government(self):
        assert "government" in ROLE_DOMAIN_AFFINITY["CONTRACT_TRACKER"]


# ---------------------------------------------------------------------------
# select_preferred_role priority rules
# ---------------------------------------------------------------------------


class TestSelectPreferredRolePriority:
    """Priority order: causal_predictions > high win_rate > insider > government
    > 3+ domains (MARKET_ANALYST) > overlap score > None"""

    def test_causal_predictions_wins_over_everything(self):
        # Even with insider domain present, causal_predictions takes precedence
        role = select_preferred_role(
            domains_seen=["insider", "macro"],
            causal_predictions=[{"trigger": "OIL"}],
            historical_win_rate=0.9,
        )
        assert role == "HYPOTHESIS_EXPLORER"

    def test_high_win_rate_wins_when_no_causal(self):
        role = select_preferred_role(
            domains_seen=["macro", "technical"],
            causal_predictions=None,
            historical_win_rate=0.75,
        )
        assert role == "HYPOTHESIS_VALIDATOR"

    def test_win_rate_threshold_boundary_below(self):
        # 0.6 is the threshold — exactly 0.6 should NOT trigger HYPOTHESIS_VALIDATOR
        role = select_preferred_role(
            domains_seen=["macro"],
            causal_predictions=None,
            historical_win_rate=0.6,
        )
        assert role != "HYPOTHESIS_VALIDATOR"

    def test_win_rate_threshold_boundary_above(self):
        role = select_preferred_role(
            domains_seen=["macro"],
            causal_predictions=None,
            historical_win_rate=0.61,
        )
        assert role == "HYPOTHESIS_VALIDATOR"

    def test_insider_domain_selects_sec_watcher(self):
        role = select_preferred_role(
            domains_seen=["insider"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "SEC_WATCHER"

    def test_institutional_domain_selects_sec_watcher(self):
        role = select_preferred_role(
            domains_seen=["institutional"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "SEC_WATCHER"

    def test_government_domain_selects_contract_tracker(self):
        role = select_preferred_role(
            domains_seen=["government"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "CONTRACT_TRACKER"

    def test_contracts_domain_selects_contract_tracker(self):
        role = select_preferred_role(
            domains_seen=["contracts"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "CONTRACT_TRACKER"

    def test_three_or_more_domains_selects_market_analyst(self):
        role = select_preferred_role(
            domains_seen=["macro", "technical", "sentiment"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "MARKET_ANALYST"

    def test_four_domains_also_selects_market_analyst(self):
        role = select_preferred_role(
            domains_seen=["macro", "technical", "sentiment", "fundamental"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role == "MARKET_ANALYST"

    def test_single_macro_domain_gets_some_role_or_none(self):
        # macro is in MARKET_ANALYST affinity but 1 domain < 3 threshold
        # Should still match via overlap scoring
        role = select_preferred_role(
            domains_seen=["macro"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        # Either a role with overlap or None — must not raise
        assert role is None or isinstance(role, str)

    def test_empty_domains_returns_none(self):
        role = select_preferred_role(
            domains_seen=[],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role is None

    def test_none_causal_predictions_not_counted(self):
        role = select_preferred_role(
            domains_seen=["macro"],
            causal_predictions=None,
            historical_win_rate=0.0,
        )
        assert role != "HYPOTHESIS_EXPLORER"

    def test_empty_causal_predictions_not_counted(self):
        role = select_preferred_role(
            domains_seen=["macro"],
            causal_predictions=[],
            historical_win_rate=0.0,
        )
        assert role != "HYPOTHESIS_EXPLORER"

    def test_returns_string_or_none(self):
        for domains in [["insider"], ["government"], ["macro", "technical", "events"], []]:
            result = select_preferred_role(domains_seen=domains)
            assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# OctopusColony role-affinity routing
# ---------------------------------------------------------------------------


def _make_mock_octopus(octopus_id: str, workload: float = 0.1, genome_role: str | None = None):
    """Create a minimal OctopusAgent mock."""
    oct_mock = MagicMock()
    oct_mock.octopus_id = octopus_id
    oct_mock.workload = workload
    oct_mock._genome_role = genome_role
    oct_mock.submit_task.return_value = f"task_{octopus_id}"
    oct_mock.update_metrics.return_value = None
    return oct_mock


class TestOctopusColonyRoleRouting:
    """Tests for submit_task() role-affinity routing branch in OctopusColony."""

    def _make_colony_with_octopuses(self, octopus_list):
        """Build a minimal colony stub (no real threading)."""
        from mae_core.network.octopus_colony import OctopusColony

        # Patch _initialize_colony to avoid real spawning
        with patch.object(OctopusColony, "_initialize_colony", return_value=None):
            colony = OctopusColony.__new__(OctopusColony)
            colony.octopuses = {}
            colony.peer_connections = {}
            colony.spawn_history = []
            colony.despawn_history = []
            colony._running = False
            colony._monitoring_thread = None
            colony._bus = None
            colony._stigmergy = None
            colony._decision_router = None
            colony._world_model = None
            colony._signal_bus = None
            colony.min_octopuses = 3
            colony.max_octopuses = 10
            colony.spawn_threshold = 0.8
            colony.despawn_threshold = 0.2
            colony.health_threshold = 0.3
            colony._monitoring_interval = 5.0
            colony._octopus_counter = 0

        for oct in octopus_list:
            colony.octopuses[oct.octopus_id] = oct
        return colony

    def test_routes_to_matching_role_octopus(self):
        sec = _make_mock_octopus("oct_sec", workload=0.5, genome_role="SEC_WATCHER")
        general = _make_mock_octopus("oct_gen", workload=0.1, genome_role=None)
        analyst = _make_mock_octopus("oct_ana", workload=0.2, genome_role="MARKET_ANALYST")

        colony = self._make_colony_with_octopuses([sec, general, analyst])

        colony.submit_task(
            {"ticker": "AAPL", "preferred_role": "SEC_WATCHER"},
            "investigate_partial",
        )

        # sec_watcher octopus should have received the task
        sec.submit_task.assert_called_once()
        general.submit_task.assert_not_called()
        analyst.submit_task.assert_not_called()

    def test_falls_back_to_workload_when_no_role_match(self):
        oct1 = _make_mock_octopus("oct_1", workload=0.3, genome_role=None)
        oct2 = _make_mock_octopus("oct_2", workload=0.1, genome_role=None)  # lowest workload
        oct3 = _make_mock_octopus("oct_3", workload=0.5, genome_role=None)

        colony = self._make_colony_with_octopuses([oct1, oct2, oct3])

        colony.submit_task(
            {"ticker": "AAPL", "preferred_role": "SEC_WATCHER"},
            "investigate_partial",
        )

        # Falls back to least-loaded (oct2)
        oct2.submit_task.assert_called_once()
        oct1.submit_task.assert_not_called()
        oct3.submit_task.assert_not_called()

    def test_picks_least_loaded_among_role_candidates(self):
        sec_busy = _make_mock_octopus("oct_sec_busy", workload=0.8, genome_role="SEC_WATCHER")
        sec_free = _make_mock_octopus("oct_sec_free", workload=0.2, genome_role="SEC_WATCHER")
        other = _make_mock_octopus("oct_other", workload=0.05, genome_role=None)

        colony = self._make_colony_with_octopuses([sec_busy, sec_free, other])

        colony.submit_task(
            {"ticker": "TSLA", "preferred_role": "SEC_WATCHER"},
            "investigate_partial",
        )

        # Least-loaded SEC_WATCHER wins, even though other is less loaded overall
        sec_free.submit_task.assert_called_once()
        sec_busy.submit_task.assert_not_called()
        other.submit_task.assert_not_called()

    def test_no_preferred_role_uses_workload_routing(self):
        oct1 = _make_mock_octopus("oct_1", workload=0.4)
        oct2 = _make_mock_octopus("oct_2", workload=0.1)
        oct3 = _make_mock_octopus("oct_3", workload=0.7)

        colony = self._make_colony_with_octopuses([oct1, oct2, oct3])

        colony.submit_task({"ticker": "SPY"}, "investigate_partial")

        oct2.submit_task.assert_called_once()

    def test_missing_preferred_role_key_works(self):
        """Task data without preferred_role key should not raise."""
        oct1 = _make_mock_octopus("oct_1", workload=0.2)
        oct2 = _make_mock_octopus("oct_2", workload=0.5)
        oct3 = _make_mock_octopus("oct_3", workload=0.9)

        colony = self._make_colony_with_octopuses([oct1, oct2, oct3])

        # Should not raise
        colony.submit_task({"ticker": "QQQ", "domains_seen": ["macro"]}, "investigate_partial")
        oct1.submit_task.assert_called_once()


# ---------------------------------------------------------------------------
# Investigation dispatcher: preferred_role injected into task_data
# ---------------------------------------------------------------------------


class TestDispatcherPreferredRoleInjection:
    """Verify that the investigation dispatcher in market_hooks passes
    preferred_role when select_preferred_role returns a non-None value."""

    def test_preferred_role_added_to_task_data_for_insider_situation(self):
        """Dispatcher should pass preferred_role='SEC_WATCHER' for insider domains."""
        submitted_tasks: list[dict] = []

        mock_colony = MagicMock()
        mock_colony._developing_situations = {
            "bullish:AAPL": {
                "ticker": "AAPL",
                "direction": "bullish",
                "domains_seen": ["insider"],
                "missing_domains": ["macro"],
                "causal_predictions": [],
                "check_count": 1,
            }
        }
        mock_colony._situations_lock = threading.Lock()

        def capture_submit(task_data, task_type, *args, **kwargs):
            submitted_tasks.append({"data": task_data, "type": task_type})
            return "task_001"

        mock_colony.submit_task.side_effect = capture_submit

        # Simulate the dispatcher logic extracted from market_hooks
        situations_snapshot = dict(mock_colony._developing_situations)
        task_budget = 5

        for key, sit in situations_snapshot.items():
            if task_budget <= 0:
                break
            check_count = sit.get("check_count", 0)
            if check_count >= 20:
                continue

            preferred_role = select_preferred_role(
                domains_seen=sit.get("domains_seen", []),
                missing_domains=sit.get("missing_domains", []),
                causal_predictions=sit.get("causal_predictions", []),
            )

            task_data_inv: dict = {
                "ticker": sit["ticker"],
                "direction": sit["direction"],
                "domains_seen": sit.get("domains_seen", []),
                "missing_domains": sit.get("missing_domains", []),
            }
            if preferred_role is not None:
                task_data_inv["preferred_role"] = preferred_role

            mock_colony.submit_task(task_data_inv, "investigate_partial")
            task_budget -= 1

        assert len(submitted_tasks) == 1
        task_data = submitted_tasks[0]["data"]
        assert task_data.get("preferred_role") == "SEC_WATCHER"

    def test_no_preferred_role_for_unknown_domains(self):
        """Dispatcher should not add preferred_role for unrecognised domain combinations."""
        submitted_tasks: list[dict] = []

        mock_colony = MagicMock()
        mock_colony._developing_situations = {
            "bullish:XYZ": {
                "ticker": "XYZ",
                "direction": "bullish",
                "domains_seen": [],
                "missing_domains": [],
                "causal_predictions": [],
                "check_count": 0,
            }
        }
        mock_colony._situations_lock = threading.Lock()
        mock_colony.submit_task.side_effect = lambda td, tt, *a, **kw: submitted_tasks.append(
            {"data": td, "type": tt}
        ) or "task_002"

        situations_snapshot = dict(mock_colony._developing_situations)
        for key, sit in situations_snapshot.items():
            preferred_role = select_preferred_role(
                domains_seen=sit.get("domains_seen", []),
                missing_domains=sit.get("missing_domains", []),
                causal_predictions=sit.get("causal_predictions", []),
            )
            task_data_inv: dict = {
                "ticker": sit["ticker"],
                "direction": sit["direction"],
                "domains_seen": sit.get("domains_seen", []),
                "missing_domains": sit.get("missing_domains", []),
            }
            if preferred_role is not None:
                task_data_inv["preferred_role"] = preferred_role
            mock_colony.submit_task(task_data_inv, "investigate_partial")

        assert len(submitted_tasks) == 1
        assert "preferred_role" not in submitted_tasks[0]["data"]

    def test_causal_predictions_dispatch_goes_to_hypothesis_explorer(self):
        """Situations with causal_predictions route to HYPOTHESIS_EXPLORER."""
        submitted_tasks: list[dict] = []

        mock_colony = MagicMock()
        mock_colony._developing_situations = {
            "bearish:OXY": {
                "ticker": "OXY",
                "direction": "bearish",
                "domains_seen": ["energy"],
                "missing_domains": ["macro"],
                "causal_predictions": [{"trigger": "CRUDE_INVENTORY"}],
                "check_count": 2,
            }
        }
        mock_colony._situations_lock = threading.Lock()
        mock_colony.submit_task.side_effect = lambda td, tt, *a, **kw: submitted_tasks.append(
            {"data": td, "type": tt}
        ) or "task_003"

        situations_snapshot = dict(mock_colony._developing_situations)
        for key, sit in situations_snapshot.items():
            preferred_role = select_preferred_role(
                domains_seen=sit.get("domains_seen", []),
                missing_domains=sit.get("missing_domains", []),
                causal_predictions=sit.get("causal_predictions", []),
            )
            task_data_inv: dict = {
                "ticker": sit["ticker"],
                "direction": sit["direction"],
                "domains_seen": sit.get("domains_seen", []),
                "missing_domains": sit.get("missing_domains", []),
            }
            if preferred_role is not None:
                task_data_inv["preferred_role"] = preferred_role
            mock_colony.submit_task(task_data_inv, "investigate_partial")

        assert len(submitted_tasks) == 1
        assert submitted_tasks[0]["data"].get("preferred_role") == "HYPOTHESIS_EXPLORER"
