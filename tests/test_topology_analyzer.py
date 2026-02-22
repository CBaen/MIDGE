"""Tests for TopologyAnalyzer - network structure measurement.

Verifies clustering coefficient, average path length, small-world sigma,
graph density, and cadenced step behavior.
"""

from collections import deque
from unittest.mock import MagicMock

import pytest

from mae_core.backbone.topology_analyzer import TopologyAnalyzer


def _make_registry_with_connections(connections: list[tuple[str, str]]):
    """Build a mock ConnectionRegistry with the given (source, target) pairs."""
    from types import SimpleNamespace

    registry = MagicMock()
    triads = {}
    for src, tgt in connections:
        conn_id = f"{src}->{tgt}"
        triads[conn_id] = SimpleNamespace(source=src, target=tgt, witnesses=[])
    registry._connections = triads
    return registry


class TestEmptyGraph:
    """Empty registry should return safe defaults."""

    def test_empty_returns_zeros(self):
        registry = _make_registry_with_connections([])
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()  # cadence=1, triggers immediately
        report = analyzer.get_latest_report()
        assert report is not None
        assert report.vertices == 0
        assert report.edges == 0
        assert report.clustering_coefficient == 0.0
        assert report.avg_path_length is None
        assert report.small_world_sigma is None
        assert report.density == 0.0
        assert report.components == 0


class TestCompleteTriangle:
    """K3 (complete triangle) has clustering=1.0 and path_length=1.0."""

    def test_k3_clustering(self):
        connections = [("A", "B"), ("B", "C"), ("A", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.vertices == 3
        assert report.edges == 3
        assert report.clustering_coefficient == pytest.approx(1.0)

    def test_k3_path_length(self):
        connections = [("A", "B"), ("B", "C"), ("A", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.avg_path_length == pytest.approx(1.0)

    def test_k3_density(self):
        connections = [("A", "B"), ("B", "C"), ("A", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.density == pytest.approx(1.0)

    def test_k3_single_component(self):
        connections = [("A", "B"), ("B", "C"), ("A", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.components == 1


class TestLineGraph:
    """Path graph (A-B-C) has clustering=0.0."""

    def test_line_clustering_zero(self):
        connections = [("A", "B"), ("B", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.clustering_coefficient == pytest.approx(0.0)

    def test_line_path_length(self):
        connections = [("A", "B"), ("B", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        # A->B=1, A->C=2, B->C=1, avg = (1+2+1+2+1+1)/6 = 8/6 = 4/3
        assert report.avg_path_length == pytest.approx(4.0 / 3.0)


class TestDensity:
    """Density for known small graphs."""

    def test_single_edge_density(self):
        connections = [("A", "B")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        # 1 edge / (2*1/2) = 1/1 = 1.0
        assert report.density == pytest.approx(1.0)

    def test_four_node_density(self):
        # K4 has 6 edges, max = 4*3/2 = 6, density = 1.0
        connections = [("A", "B"), ("A", "C"), ("A", "D"),
                       ("B", "C"), ("B", "D"), ("C", "D")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.density == pytest.approx(1.0)

    def test_sparse_density(self):
        # 4 nodes, 2 edges: density = 2/6 = 1/3
        connections = [("A", "B"), ("C", "D")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.density == pytest.approx(1.0 / 3.0)


class TestCadence:
    """Only computes at cadence intervals."""

    def test_no_report_before_cadence(self):
        registry = _make_registry_with_connections([("A", "B")])
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=5)
        for _ in range(4):
            analyzer.step()
        assert analyzer.get_latest_report() is None

    def test_report_at_cadence(self):
        registry = _make_registry_with_connections([("A", "B")])
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=5)
        for _ in range(5):
            analyzer.step()
        assert analyzer.get_latest_report() is not None
        assert analyzer.get_latest_report().measurement_number == 1

    def test_multiple_cadences(self):
        registry = _make_registry_with_connections([("A", "B")])
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=3)
        for _ in range(9):
            analyzer.step()
        assert analyzer.get_latest_report().measurement_number == 3


class TestStatistics:
    """get_statistics() returns expected keys."""

    def test_stats_before_measurement(self):
        registry = _make_registry_with_connections([])
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=55)
        stats = analyzer.get_statistics()
        assert stats["step_count"] == 0
        assert stats["measurement_count"] == 0
        assert stats["cadence"] == 55
        assert "last_measurement" not in stats

    def test_stats_after_measurement(self):
        connections = [("A", "B"), ("B", "C"), ("A", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        stats = analyzer.get_statistics()
        assert "last_measurement" in stats
        lm = stats["last_measurement"]
        assert lm["vertices"] == 3
        assert lm["edges"] == 3
        assert lm["clustering_coefficient"] == pytest.approx(1.0)
        assert lm["avg_path_length"] == pytest.approx(1.0)

    def test_get_state(self):
        connections = [("A", "B"), ("B", "C")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        state = analyzer.get_state()
        assert state["has_report"] is True
        assert "clustering_coefficient" in state
        assert "avg_path_length" in state
        assert "density" in state


class TestDisconnectedGraph:
    """Disconnected graph handles path length correctly."""

    def test_two_components(self):
        connections = [("A", "B"), ("C", "D")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.components == 2
        # Path length should average only reachable pairs
        # A->B=1, B->A=1, C->D=1, D->C=1: avg = 4/4 = 1.0
        assert report.avg_path_length == pytest.approx(1.0)


class TestEventBusPublish:
    """EventBus receives topology.analysis events."""

    def test_publish_on_measurement(self):
        connections = [("A", "B"), ("B", "C")]
        registry = _make_registry_with_connections(connections)
        bus = MagicMock()
        analyzer = TopologyAnalyzer(registry, event_bus=bus, cadence=1)
        analyzer.step()
        bus.publish.assert_called_once()
        args = bus.publish.call_args
        assert args[0][0] == "topology.analysis"
        payload = args[0][1]
        assert "clustering_coefficient" in payload
        assert "avg_path_length" in payload
        assert "small_world_sigma" in payload
        assert "density" in payload


class TestSmallWorldSigma:
    """Small-world sigma computation edge cases."""

    def test_sigma_none_for_tiny_graph(self):
        connections = [("A", "B")]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        assert report.small_world_sigma is None

    def test_sigma_computable_for_larger_graph(self):
        # Build a small-world-ish graph: K4 with extra node
        connections = [
            ("A", "B"), ("A", "C"), ("A", "D"),
            ("B", "C"), ("B", "D"), ("C", "D"),
            ("D", "E"), ("E", "A"),
        ]
        registry = _make_registry_with_connections(connections)
        analyzer = TopologyAnalyzer(registry, event_bus=None, cadence=1)
        analyzer.step()
        report = analyzer.get_latest_report()
        # Should be computable (5 nodes, 8 edges, connected, non-zero clustering)
        assert report.small_world_sigma is not None
        assert report.small_world_sigma > 0
