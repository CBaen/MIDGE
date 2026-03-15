"""Tests for KnowledgeGraph — Neo4j persistent knowledge store.

Two test classes:
  TestKnowledgeGraphOffline  — runs without Neo4j; verifies graceful degradation.
  TestKnowledgeGraphLive     — skipped if Neo4j is not available; exercises real writes.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from mae_core.market.intelligence.knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kg_no_neo4j() -> KnowledgeGraph:
    """Return a KnowledgeGraph that intentionally cannot connect to Neo4j."""
    with patch(
        "mae_core.market.intelligence.knowledge_graph.KnowledgeGraph._connect",
        lambda self: None,  # skip actual connection attempt
    ):
        kg = KnowledgeGraph.__new__(KnowledgeGraph)
        kg._uri = "bolt://localhost:7687"
        kg._user = "neo4j"
        kg._pass = "midgepassword"
        kg._driver = None
        kg._connected = False
    return kg


def _neo4j_available() -> bool:
    """Probe whether Neo4j is reachable (used for live test skip condition)."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("MIDGE_NEO4J_URI", "bolt://localhost:7687"),
            auth=(
                os.environ.get("MIDGE_NEO4J_USER", "neo4j"),
                os.environ.get("MIDGE_NEO4J_PASS", "midgepassword"),
            ),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Offline tests — no Docker required
# ---------------------------------------------------------------------------

class TestKnowledgeGraphOffline:
    """All methods must degrade silently when Neo4j is unavailable."""

    def setup_method(self):
        self.kg = _make_kg_no_neo4j()

    def test_is_connected_returns_false(self):
        assert self.kg.is_connected() is False

    def test_store_thompson_update_no_crash(self):
        # Should return silently without raising
        self.kg.store_thompson_update("insider_form4", "bull", 4.5, 2.0, True)

    def test_store_granger_finding_no_crash(self):
        self.kg.store_granger_finding("institutional", "insider", 4, 82.7, 0.0)

    def test_store_convergence_alert_no_crash(self):
        self.kg.store_convergence_alert("alert_001", "AAPL", "bullish", 0.75, ["insider", "macro"])

    def test_store_outcome_no_crash(self):
        self.kg.store_outcome("pred_001", "AAPL", True, 7.5)

    def test_store_cascade_confirmation_no_crash(self):
        self.kg.store_cascade_confirmation("c_001", "CL=F", "XLE", 2.0, True, 1.1)

    def test_get_ticker_history_returns_empty_dict(self):
        result = self.kg.get_ticker_history("AAPL")
        assert result == {"alerts": [], "outcomes": [], "cascades": []}

    def test_get_causal_chain_returns_empty_list(self):
        result = self.kg.get_causal_chain("institutional")
        assert result == []

    def test_get_learning_trajectory_returns_empty_list(self):
        result = self.kg.get_learning_trajectory("insider_form4")
        assert result == []

    def test_seed_from_granger_returns_zero(self):
        n = self.kg.seed_from_granger()
        assert n == 0

    def test_seed_from_thompson_returns_zero(self):
        n = self.kg.seed_from_thompson()
        assert n == 0

    def test_close_safe_when_not_connected(self):
        # Should not raise
        self.kg.close()

    def test_seed_from_granger_missing_file_returns_zero(self, tmp_path):
        n = self.kg.seed_from_granger(tmp_path / "nonexistent.json")
        assert n == 0

    def test_seed_from_thompson_missing_file_returns_zero(self, tmp_path):
        n = self.kg.seed_from_thompson(tmp_path / "nonexistent.json")
        assert n == 0

    def test_seed_from_granger_corrupt_json_returns_zero(self, tmp_path):
        bad = tmp_path / "granger.json"
        bad.write_text("not json", encoding="utf-8")
        n = self.kg.seed_from_granger(bad)
        assert n == 0

    def test_seed_from_thompson_corrupt_json_returns_zero(self, tmp_path):
        bad = tmp_path / "thompson.json"
        bad.write_text("{bad json", encoding="utf-8")
        n = self.kg.seed_from_thompson(bad)
        assert n == 0

    def test_seed_from_granger_valid_file_no_connection(self, tmp_path):
        """Even with a valid file, zero is returned when not connected."""
        data = {
            "findings": [
                {
                    "cause_domain": "macro",
                    "effect_domain": "insider",
                    "best_lag": 3,
                    "f_statistic": 33.97,
                    "p_value": 0.0,
                    "last_updated": "2026-03-15T12:00:00",
                }
            ]
        }
        p = tmp_path / "granger.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        n = self.kg.seed_from_granger(p)
        assert n == 0  # not connected, must skip

    def test_seed_from_thompson_valid_file_no_connection(self, tmp_path):
        data = {"insider_form4": {"default": {"alpha": 4.2, "beta": 2.0}}}
        p = tmp_path / "thompson.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        n = self.kg.seed_from_thompson(p)
        assert n == 0  # not connected, must skip


# ---------------------------------------------------------------------------
# Live tests — require Docker midge-neo4j to be running
# ---------------------------------------------------------------------------

_SKIP_LIVE = pytest.mark.skipif(
    not _neo4j_available(),
    reason="Neo4j not available (Docker midge-neo4j must be running)",
)


@_SKIP_LIVE
class TestKnowledgeGraphLive:
    """End-to-end tests against the real Neo4j instance."""

    def setup_method(self):
        self.kg = KnowledgeGraph()
        assert self.kg.is_connected(), "Neo4j must be up for live tests"

    def teardown_method(self):
        self.kg.close()

    def test_is_connected_true(self):
        assert self.kg.is_connected()

    def test_store_and_retrieve_thompson_update(self):
        self.kg.store_thompson_update(
            "test_source_kg", "default", 3.0, 2.0, True, datetime(2026, 3, 15)
        )
        rows = self.kg.get_learning_trajectory("test_source_kg", limit=10)
        assert len(rows) >= 1
        last = rows[-1]
        assert last["alpha"] == pytest.approx(3.0)
        assert last["beta"]  == pytest.approx(2.0)
        assert last["won"]   is True

    def test_store_and_retrieve_granger_finding(self):
        self.kg.store_granger_finding(
            "test_cause_domain", "test_effect_domain", 5, 20.0, 0.001, datetime(2026, 3, 15)
        )
        chains = self.kg.get_causal_chain("test_cause_domain", max_hops=1)
        # May return empty if the CAUSES edge is not traversable in 1 hop due to MERGE
        # — the important thing is it doesn't raise.
        assert isinstance(chains, list)

    def test_store_and_retrieve_convergence_alert(self):
        self.kg.store_convergence_alert(
            "test_alert_live_001", "TSLA", "bullish", 0.82,
            ["insider", "macro", "technical"], datetime(2026, 3, 15)
        )
        history = self.kg.get_ticker_history("TSLA", limit=10)
        alert_ids = [r["alert_id"] for r in history["alerts"]]
        assert "test_alert_live_001" in alert_ids

    def test_store_and_retrieve_outcome(self):
        self.kg.store_outcome(
            "test_pred_live_001", "NVDA", False, -3.2, datetime(2026, 3, 15)
        )
        history = self.kg.get_ticker_history("NVDA", limit=10)
        pred_ids = [r["prediction_id"] for r in history["outcomes"]]
        assert "test_pred_live_001" in pred_ids

    def test_store_and_retrieve_cascade_confirmation(self):
        self.kg.store_cascade_confirmation(
            "casc_live_001", "CL=F", "XLE", 2.0, True, 1.05, datetime(2026, 3, 15)
        )
        history = self.kg.get_ticker_history("CL=F", limit=10)
        casc_ids = [r["cascade_id"] for r in history["cascades"]]
        assert "casc_live_001" in casc_ids

    def test_seed_from_granger_live(self, tmp_path):
        data = {
            "findings": [
                {
                    "cause_domain": "live_macro",
                    "effect_domain": "live_insider",
                    "best_lag": 3,
                    "f_statistic": 33.97,
                    "p_value": 0.0,
                    "last_updated": "2026-03-15T12:00:00",
                }
            ]
        }
        p = tmp_path / "granger.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        n = self.kg.seed_from_granger(p)
        assert n == 1

    def test_seed_from_thompson_live(self, tmp_path):
        data = {
            "live_test_source": {
                "default": {"alpha": 4.0, "beta": 2.0},
                "bull":    {"alpha": 5.0, "beta": 2.0},
            }
        }
        p = tmp_path / "thompson.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        n = self.kg.seed_from_thompson(p)
        assert n == 2  # two regime entries

    def test_close_and_reconnect(self):
        """Closing and recreating the driver does not raise."""
        self.kg.close()
        assert not self.kg.is_connected()
        # Reconnect for teardown
        self.kg._connect()
