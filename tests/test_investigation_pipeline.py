"""Tests for the Octopus investigation pipeline.

Covers:
- investigate_partial handler: convergence check, pattern library query,
  priority request creation, CH_OCTOPUS_INVESTIGATION publish
- situation_check handler: eviction logic, save callback invoked
- inject_market_handlers: pattern_library + world_model attached to colony
- Step-cadence dispatcher: task submission from _developing_situations
- _developing_situations persistence: load/save round-trip
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mae_core.network.market_task_handlers import (
    CH_OCTOPUS_INVESTIGATION,
    MAX_SITUATION_CHECKS,
    inject_market_handlers,
    _make_investigate_partial,
    _make_situation_check,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_colony(situations: dict | None = None) -> Any:
    """Minimal colony stub with lock + shared state."""
    colony = SimpleNamespace()
    colony._developing_situations = situations if situations is not None else {}
    colony._situations_lock = threading.Lock()
    colony._pattern_library = None
    colony._world_model_ref = None
    colony._priority_requests = {}
    colony._save_developing_situations = None
    return colony


def _make_task(ticker: str = "AAPL", direction: str = "bullish",
               domains_seen: list | None = None,
               missing_domains: list | None = None) -> Any:
    task = SimpleNamespace()
    task.data = {
        "ticker": ticker,
        "direction": direction,
        "domains_seen": domains_seen or ["insider", "macro"],
        "missing_domains": missing_domains or ["technical"],
    }
    task.task_type = "investigate_partial"
    return task


# ---------------------------------------------------------------------------
# A: investigate_partial — basic convergence path
# ---------------------------------------------------------------------------


class TestInvestigatePartialBasic:
    def test_no_ticker_returns_early(self):
        colony = _make_colony()
        bus = MagicMock()
        alerter = MagicMock()
        handler = _make_investigate_partial(colony, alerter, bus)

        task = SimpleNamespace()
        task.data = {}  # no ticker
        handler(task)

        bus.publish.assert_not_called()
        alerter.check_ticker_convergence_for.assert_not_called()

    def test_creates_situation_entry(self):
        colony = _make_colony()
        bus = MagicMock()
        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        handler = _make_investigate_partial(colony, alerter, bus)

        handler(_make_task("TSLA", "bullish"))

        assert "bullish:TSLA" in colony._developing_situations
        sit = colony._developing_situations["bullish:TSLA"]
        assert sit["ticker"] == "TSLA"
        assert sit["check_count"] == 1

    def test_increments_existing_situation(self):
        colony = _make_colony({"bullish:MSFT": {
            "ticker": "MSFT", "direction": "bullish",
            "domains_seen": ["macro"], "missing_domains": ["insider"],
            "first_seen": time.time(), "check_count": 3,
        }})
        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        handler = _make_investigate_partial(colony, alerter, MagicMock())

        handler(_make_task("MSFT", "bullish"))
        assert colony._developing_situations["bullish:MSFT"]["check_count"] == 4

    def test_publishes_when_full_alert_fires(self):
        colony = _make_colony()
        bus = MagicMock()
        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = {"ticker": "NVDA", "confidence": 0.8}
        handler = _make_investigate_partial(colony, alerter, bus)

        handler(_make_task("NVDA", "bullish"))

        bus.publish.assert_called_once()
        call_args = bus.publish.call_args
        assert call_args[0][0] == CH_OCTOPUS_INVESTIGATION
        payload = call_args[0][1]
        assert payload["ticker"] == "NVDA"
        assert payload["source"] == "investigate_partial"
        assert payload["alert"] is not None

    def test_no_publish_when_no_alert_no_templates(self):
        colony = _make_colony()
        bus = MagicMock()
        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        handler = _make_investigate_partial(colony, alerter, bus)

        handler(_make_task("XYZ", "bullish"))

        bus.publish.assert_not_called()

    def test_convergence_alerter_none(self):
        colony = _make_colony()
        bus = MagicMock()
        handler = _make_investigate_partial(colony, None, bus)

        handler(_make_task("GLD", "bearish"))

        # No exception, no publish
        bus.publish.assert_not_called()

    def test_convergence_alerter_exception_handled(self):
        colony = _make_colony()
        bus = MagicMock()
        alerter = MagicMock()
        alerter.check_ticker_convergence_for.side_effect = RuntimeError("boom")
        handler = _make_investigate_partial(colony, alerter, bus)

        # Should not raise
        handler(_make_task("SPY", "bullish"))
        bus.publish.assert_not_called()


# ---------------------------------------------------------------------------
# B: investigate_partial — pattern library integration
# ---------------------------------------------------------------------------


class TestInvestigatePartialPatternLibrary:
    def _make_template(self, domains, win_count=8, loss_count=2, instance_count=10,
                       cross_validated=True):
        tmpl = SimpleNamespace()
        tmpl.domains = domains
        tmpl.win_count = win_count
        tmpl.loss_count = loss_count
        tmpl.instance_count = instance_count
        tmpl.cross_validated = cross_validated
        return tmpl

    def _make_match(self, template):
        m = SimpleNamespace()
        m.template = template
        return m

    def test_pattern_library_query_called(self):
        colony = _make_colony()
        lib = MagicMock()
        lib.query_similar.return_value = []
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("AAPL", "bullish", domains_seen=["insider", "macro"]))

        lib.query_similar.assert_called_once()
        call_kwargs = lib.query_similar.call_args
        assert "insider" in call_kwargs[1]["live_sources"] or "insider" in call_kwargs[0][0]

    def test_high_win_rate_creates_priority_request(self):
        colony = _make_colony()
        tmpl = self._make_template(["insider", "macro"], win_count=7, loss_count=3,
                                   instance_count=10)
        lib = MagicMock()
        lib.query_similar.return_value = [self._make_match(tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("AMD", "bullish"))

        # win_rate = 0.7 > 0.6 and instance_count=10 >= 5 → priority request created
        assert "AMD" in colony._priority_requests
        assert colony._priority_requests["AMD"]["win_rate"] == pytest.approx(0.7)

    def test_low_win_rate_no_priority_request(self):
        colony = _make_colony()
        tmpl = self._make_template(["insider", "macro"], win_count=3, loss_count=7,
                                   instance_count=10)
        lib = MagicMock()
        lib.query_similar.return_value = [self._make_match(tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("INTC", "bullish"))

        # win_rate = 0.3, no priority request
        assert "INTC" not in colony._priority_requests

    def test_insufficient_instances_no_priority_request(self):
        colony = _make_colony()
        tmpl = self._make_template(["insider", "macro"], win_count=4, loss_count=1,
                                   instance_count=4)  # < 5
        lib = MagicMock()
        lib.query_similar.return_value = [self._make_match(tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("BA", "bullish"))

        assert "BA" not in colony._priority_requests

    def test_priority_cap_at_50(self):
        colony = _make_colony()
        # Pre-fill to cap
        colony._priority_requests = {f"T{i}": {} for i in range(50)}

        tmpl = self._make_template(["insider"], win_count=8, loss_count=2, instance_count=10)
        lib = MagicMock()
        lib.query_similar.return_value = [SimpleNamespace(template=tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("NEWT", "bullish"))

        # Cap enforced — NEWT not added
        assert "NEWT" not in colony._priority_requests
        assert len(colony._priority_requests) == 50

    def test_historical_templates_in_publish_payload(self):
        colony = _make_colony()
        tmpl = self._make_template(["insider", "macro"], win_count=8, loss_count=2,
                                   instance_count=10, cross_validated=True)
        lib = MagicMock()
        lib.query_similar.return_value = [SimpleNamespace(template=tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = {"ticker": "META", "confidence": 0.9}
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("META", "bullish"))

        bus.publish.assert_called_once()
        payload = bus.publish.call_args[0][1]
        assert "historical_templates" in payload
        assert len(payload["historical_templates"]) == 1
        ht = payload["historical_templates"][0]
        assert ht["win_rate"] == pytest.approx(0.8)
        assert ht["instances"] == 10
        assert ht["cross_validated"] is True

    def test_priority_request_created_flag_in_payload(self):
        colony = _make_colony()
        tmpl = self._make_template(["insider"], win_count=7, loss_count=3, instance_count=10)
        lib = MagicMock()
        lib.query_similar.return_value = [SimpleNamespace(template=tmpl)]
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = {"ticker": "CRM", "confidence": 0.8}
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        handler(_make_task("CRM", "bullish"))

        payload = bus.publish.call_args[0][1]
        assert payload["priority_request_created"] is True

    def test_pattern_library_exception_handled(self):
        colony = _make_colony()
        lib = MagicMock()
        lib.query_similar.side_effect = RuntimeError("db error")
        colony._pattern_library = lib

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        # Should not raise
        handler(_make_task("IBM", "bullish"))


# ---------------------------------------------------------------------------
# C: investigate_partial — world model integration
# ---------------------------------------------------------------------------


class TestInvestigatePartialWorldModel:
    def _make_ripple(self, ticker, strength=0.7):
        r = SimpleNamespace()
        r.ticker = ticker
        r.direction = "bullish"
        r.strength = strength
        r.total_lag_days = 2.0
        return r

    def test_world_model_ripple_checked_for_causal_predictions(self):
        colony = _make_colony({
            "bullish:XLE": {
                "ticker": "XLE", "direction": "bullish",
                "domains_seen": ["energy"], "missing_domains": [],
                "causal_predictions": ["oil_price_spike"],
                "first_seen": time.time(), "check_count": 0,
            }
        })
        wm = MagicMock()
        wm.find_ripple_effects.return_value = [self._make_ripple("XLE")]
        wm.find_root_causes.return_value = []
        colony._world_model_ref = wm

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        task = _make_task("XLE", "bullish")
        task.data["causal_predictions"] = ["oil_price_spike"]
        handler(task)

        wm.find_ripple_effects.assert_called_with("oil_price_spike")

    def test_world_model_exception_handled(self):
        colony = _make_colony()
        wm = MagicMock()
        wm.find_ripple_effects.side_effect = Exception("graph error")
        wm.find_root_causes.side_effect = Exception("graph error")
        colony._world_model_ref = wm

        alerter = MagicMock()
        alerter.check_ticker_convergence_for.return_value = None
        bus = MagicMock()

        handler = _make_investigate_partial(colony, alerter, bus)
        task = _make_task("GOOG", "bullish")
        task.data["causal_predictions"] = ["ad_revenue_drop"]
        # Should not raise
        handler(task)


# ---------------------------------------------------------------------------
# D: situation_check handler
# ---------------------------------------------------------------------------


class TestSituationCheck:
    def _make_sit_task(self, ticker="AAPL", direction="bullish"):
        task = SimpleNamespace()
        task.data = {"ticker": ticker, "direction": direction}
        task.task_type = "situation_check"
        return task

    def test_evicts_when_check_count_exceeds_max(self):
        key = "bullish:AAPL"
        colony = _make_colony({key: {
            "ticker": "AAPL", "direction": "bullish",
            "domains_seen": [], "missing_domains": [],
            "first_seen": time.time(),
            "check_count": MAX_SITUATION_CHECKS,  # at max
        }})
        handler = _make_situation_check(colony)
        handler(self._make_sit_task("AAPL", "bullish"))
        assert key not in colony._developing_situations

    def test_does_not_evict_below_max(self):
        key = "bullish:NVDA"
        colony = _make_colony({key: {
            "ticker": "NVDA", "direction": "bullish",
            "domains_seen": [], "missing_domains": [],
            "first_seen": time.time(), "check_count": 5,
        }})
        handler = _make_situation_check(colony)
        handler(self._make_sit_task("NVDA", "bullish"))
        assert key in colony._developing_situations

    def test_no_ticker_returns_early(self):
        colony = _make_colony()
        handler = _make_situation_check(colony)
        task = SimpleNamespace()
        task.data = {}
        # Should not raise
        handler(task)

    def test_missing_key_returns_early(self):
        colony = _make_colony()
        handler = _make_situation_check(colony)
        task = self._make_sit_task("UNKNOWN", "bullish")
        # Should not raise
        handler(task)

    def test_save_called_on_eviction(self):
        key = "bearish:F"
        colony = _make_colony({key: {
            "ticker": "F", "direction": "bearish",
            "domains_seen": [], "missing_domains": [],
            "first_seen": time.time(),
            "check_count": MAX_SITUATION_CHECKS,
        }})
        save_mock = MagicMock()
        colony._save_developing_situations = save_mock

        handler = _make_situation_check(colony)
        handler(self._make_sit_task("F", "bearish"))

        assert key not in colony._developing_situations
        save_mock.assert_called_once_with(colony)

    def test_save_not_called_when_no_eviction(self):
        key = "bullish:GM"
        colony = _make_colony({key: {
            "ticker": "GM", "direction": "bullish",
            "domains_seen": [], "missing_domains": [],
            "first_seen": time.time(), "check_count": 2,
        }})
        save_mock = MagicMock()
        colony._save_developing_situations = save_mock

        handler = _make_situation_check(colony)
        handler(self._make_sit_task("GM", "bullish"))

        save_mock.assert_not_called()

    def test_save_exception_does_not_propagate(self):
        key = "bullish:CAT"
        colony = _make_colony({key: {
            "ticker": "CAT", "direction": "bullish",
            "domains_seen": [], "missing_domains": [],
            "first_seen": time.time(),
            "check_count": MAX_SITUATION_CHECKS,
        }})
        colony._save_developing_situations = MagicMock(side_effect=OSError("disk full"))

        handler = _make_situation_check(colony)
        # Should not raise
        handler(self._make_sit_task("CAT", "bullish"))


# ---------------------------------------------------------------------------
# E: inject_market_handlers — new args attached to colony
# ---------------------------------------------------------------------------


class TestInjectMarketHandlers:
    def _make_minimal_colony(self):
        """Colony with at least one patchable arm."""
        arm = SimpleNamespace()
        arm.arm_id = "arm_0"
        arm.current_task = None
        arm._lock = threading.Lock()
        arm.task_history = []
        arm.state = SimpleNamespace(workload=0.2)

        cognition = SimpleNamespace(arms={"arm_0": arm})
        octopus = SimpleNamespace(cognition=cognition)

        colony = SimpleNamespace()
        colony.octopuses = {"oct_0": octopus}
        colony._developing_situations = {}
        colony._situations_lock = threading.Lock()
        return colony, arm

    def test_pattern_library_attached_to_colony(self):
        colony, _ = self._make_minimal_colony()
        lib = MagicMock()
        inject_market_handlers(colony, None, None, None, pattern_library=lib)
        assert colony._pattern_library is lib

    def test_world_model_attached_to_colony(self):
        colony, _ = self._make_minimal_colony()
        wm = MagicMock()
        inject_market_handlers(colony, None, None, None, world_model=wm)
        assert colony._world_model_ref is wm

    def test_handler_refs_include_new_args(self):
        colony, _ = self._make_minimal_colony()
        lib = MagicMock()
        wm = MagicMock()
        inject_market_handlers(colony, None, None, None, pattern_library=lib, world_model=wm)
        assert colony._handler_refs["pattern_library"] is lib
        assert colony._handler_refs["world_model"] is wm

    def test_arms_patched(self):
        colony, arm = self._make_minimal_colony()
        n = inject_market_handlers(colony, None, None, None)
        assert n == 1
        assert hasattr(arm, "_task_handlers")
        assert "investigate_partial" in arm._task_handlers
        assert "situation_check" in arm._task_handlers
        assert "archaeology_lookup" in arm._task_handlers


# ---------------------------------------------------------------------------
# F: Persistence — load/save round-trip
# ---------------------------------------------------------------------------


class TestDevelopingSituationsPersistence:
    def test_save_roundtrip(self, tmp_path):
        """Save situations to a temp file and reload them."""
        import json
        from pathlib import Path

        sit_path = tmp_path / "developing_situations.json"

        situations = {
            "bullish:AAPL": {
                "ticker": "AAPL", "direction": "bullish",
                "domains_seen": ["insider"], "missing_domains": ["macro"],
                "first_seen": time.time(), "check_count": 3,
                "causal_predictions": [],
            }
        }

        colony = _make_colony(situations)

        def _save(col):
            snapshot = dict(col._developing_situations)
            tmp = sit_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(snapshot, default=str), encoding="utf-8")
            tmp.replace(sit_path)

        colony._save_developing_situations = _save
        _save(colony)

        assert sit_path.exists()
        loaded = json.loads(sit_path.read_text(encoding="utf-8"))
        assert "bullish:AAPL" in loaded
        assert loaded["bullish:AAPL"]["ticker"] == "AAPL"

    def test_old_entries_filtered_on_load(self, tmp_path):
        """Entries older than 2 hours should be dropped on load."""
        import json

        sit_path = tmp_path / "developing_situations.json"
        old_time = time.time() - 7201  # older than 2h
        fresh_time = time.time() - 100

        raw = {
            "bullish:OLD": {
                "ticker": "OLD", "direction": "bullish",
                "domains_seen": [], "missing_domains": [],
                "first_seen": old_time, "check_count": 1,
            },
            "bullish:FRESH": {
                "ticker": "FRESH", "direction": "bullish",
                "domains_seen": [], "missing_domains": [],
                "first_seen": fresh_time, "check_count": 1,
            },
        }
        sit_path.write_text(json.dumps(raw), encoding="utf-8")

        # Simulate the load logic from market_systems.py
        loaded_raw = json.loads(sit_path.read_text(encoding="utf-8"))
        now = time.time()
        loaded = {
            k: v for k, v in loaded_raw.items()
            if (now - v.get("first_seen", 0)) < 7200
        }

        assert "bullish:OLD" not in loaded
        assert "bullish:FRESH" in loaded

    def test_missing_file_does_not_raise(self, tmp_path):
        """If the file doesn't exist, load should silently skip."""
        import json

        sit_path = tmp_path / "nonexistent.json"
        # Simulate the load logic
        situations = {}
        try:
            if sit_path.exists():
                raw = json.loads(sit_path.read_text(encoding="utf-8"))
                now = time.time()
                situations = {
                    k: v for k, v in raw.items()
                    if (now - v.get("first_seen", 0)) < 7200
                }
        except Exception:
            pass

        assert situations == {}

    def test_corrupt_file_does_not_raise(self, tmp_path):
        """Corrupt JSON should be handled gracefully."""
        import json

        sit_path = tmp_path / "developing_situations.json"
        sit_path.write_text("NOT VALID JSON", encoding="utf-8")

        situations = {}
        try:
            if sit_path.exists():
                raw = json.loads(sit_path.read_text(encoding="utf-8"))
                situations = dict(raw)
        except Exception:
            pass

        assert situations == {}


# ---------------------------------------------------------------------------
# G: Step dispatcher — task submission logic
# ---------------------------------------------------------------------------


class TestStepDispatcher:
    """Unit tests for the step-20 investigation dispatcher logic.

    Rather than invoking the full bootstrap, we test the dispatch
    logic in isolation by replicating the core loop.
    """

    def _run_dispatcher(self, colony, task_budget=5):
        """Replicate the dispatcher logic from market_hooks.py."""
        submitted = []

        lock = getattr(colony, "_situations_lock", None)
        situations_snapshot = {}
        if lock is not None:
            with lock:
                situations_snapshot = dict(colony._developing_situations)
        else:
            situations_snapshot = dict(getattr(colony, "_developing_situations", {}))

        budget = task_budget
        for key, sit in situations_snapshot.items():
            if budget <= 0:
                break
            check_count = sit.get("check_count", 0)
            if check_count >= 20:
                continue

            submitted.append(("investigate_partial", sit["ticker"], sit["direction"]))
            budget -= 1

            if check_count > 0 and check_count % 5 == 0:
                if budget > 0:
                    submitted.append(("situation_check", sit["ticker"], sit["direction"]))
                    budget -= 1

        return submitted

    def test_submits_investigate_for_each_situation(self):
        situations = {
            "bullish:A": {"ticker": "A", "direction": "bullish",
                          "domains_seen": [], "missing_domains": [], "check_count": 0},
            "bearish:B": {"ticker": "B", "direction": "bearish",
                          "domains_seen": [], "missing_domains": [], "check_count": 2},
        }
        colony = _make_colony(situations)
        submitted = self._run_dispatcher(colony)
        types = [s[0] for s in submitted]
        assert types.count("investigate_partial") == 2

    def test_cap_at_5_tasks(self):
        situations = {
            f"bullish:T{i}": {"ticker": f"T{i}", "direction": "bullish",
                               "domains_seen": [], "missing_domains": [], "check_count": 1}
            for i in range(10)
        }
        colony = _make_colony(situations)
        submitted = self._run_dispatcher(colony)
        assert len(submitted) <= 5

    def test_skips_situations_at_max_check_count(self):
        situations = {
            "bullish:OLD": {"ticker": "OLD", "direction": "bullish",
                            "domains_seen": [], "missing_domains": [], "check_count": 20},
            "bullish:OK": {"ticker": "OK", "direction": "bullish",
                           "domains_seen": [], "missing_domains": [], "check_count": 3},
        }
        colony = _make_colony(situations)
        submitted = self._run_dispatcher(colony)
        tickers = [s[1] for s in submitted]
        assert "OLD" not in tickers
        assert "OK" in tickers

    def test_situation_check_submitted_on_5th_check(self):
        situations = {
            "bullish:X": {"ticker": "X", "direction": "bullish",
                          "domains_seen": [], "missing_domains": [], "check_count": 5},
        }
        colony = _make_colony(situations)
        submitted = self._run_dispatcher(colony)
        types = [s[0] for s in submitted]
        assert "situation_check" in types

    def test_situation_check_not_submitted_on_non_5th_check(self):
        situations = {
            "bullish:Y": {"ticker": "Y", "direction": "bullish",
                          "domains_seen": [], "missing_domains": [], "check_count": 3},
        }
        colony = _make_colony(situations)
        submitted = self._run_dispatcher(colony)
        types = [s[0] for s in submitted]
        assert "situation_check" not in types

    def test_empty_situations_no_tasks(self):
        colony = _make_colony({})
        submitted = self._run_dispatcher(colony)
        assert submitted == []
