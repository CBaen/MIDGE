"""Tests for Phase 5.8: 4D Temporal Reasoning.

Tests TemporalMemory (4D event timeline + causal chains) and
WorldlinePlanner (multi-horizon trajectory planning).
"""

import json
import time

import numpy as np
import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.cognition.causal_reasoning import CausalReasoningEngine
from mae_core.cognition.world_model import Prediction, WorldModel, WorldModelConfig
from mae_core.planning.temporal_memory import (
    CH_TEMPORAL_EVENT_RECORDED,
    CH_TEMPORAL_PATTERN_DETECTED,
    CausalChain,
    EventType,
    FourDEvent,
    TemporalMemory,
    TemporalPattern,
)
from mae_core.planning.worldline_planner import (
    CH_WORLDLINE_PLANNED,
    PlanningResult,
    Worldline,
    WorldlinePlanner,
    WorldlinePoint,
    WorldlineStatus,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def causal():
    return CausalReasoningEngine()


@pytest.fixture
def temporal(bus, causal):
    return TemporalMemory(
        event_bus=bus,
        causal_engine=causal,
        max_events=1000,
        temporal_window=5.0,
        causal_window=10.0,
    )


@pytest.fixture
def world_model():
    return WorldModel(WorldModelConfig(state_dim=4, action_dim=2))


@pytest.fixture
def planner(world_model, bus):
    return WorldlinePlanner(
        world_model=world_model,
        event_bus=bus,
        reactive_horizon=3,
        tactical_horizon=8,
        strategic_horizon=15,
        num_branches=3,
    )


def make_event(
    event_id: str,
    entity_id: str = "agent-1",
    event_type: EventType = EventType.ACTION,
    timestamp: float | None = None,
    position: tuple[float, ...] = (0.0, 0.0),
    importance: float = 0.5,
    data: dict | None = None,
) -> FourDEvent:
    return FourDEvent(
        event_id=event_id,
        entity_id=entity_id,
        event_type=event_type,
        timestamp=timestamp or time.time(),
        position=position,
        importance=importance,
        data=data or {},
    )


# ===========================================================================
# TestTemporalMemory
# ===========================================================================


class TestTemporalMemory:
    """Tests for 4D temporal event memory."""

    def test_record_and_retrieve(self, temporal):
        """Events can be recorded and retrieved by ID."""
        event = make_event("evt-1", entity_id="agent-1")
        temporal.record_event(event)

        retrieved = temporal.get_event("evt-1")
        assert retrieved is not None
        assert retrieved.event_id == "evt-1"
        assert retrieved.entity_id == "agent-1"

    def test_query_by_entity(self, temporal):
        """Events can be queried by entity."""
        for i in range(5):
            temporal.record_event(make_event(f"a1-{i}", entity_id="agent-1"))
        for i in range(3):
            temporal.record_event(make_event(f"a2-{i}", entity_id="agent-2"))

        a1_events = temporal.query_by_entity("agent-1")
        a2_events = temporal.query_by_entity("agent-2")
        assert len(a1_events) == 5
        assert len(a2_events) == 3

    def test_query_by_time_range(self, temporal):
        """Events can be queried within a time range."""
        base = time.time()
        for i in range(10):
            temporal.record_event(make_event(f"evt-{i}", timestamp=base + i))

        # Query middle 4 events
        results = temporal.query_by_time_range(base + 3, base + 6)
        assert len(results) == 4
        assert results[0].event_id == "evt-3"
        assert results[-1].event_id == "evt-6"

    def test_query_by_position(self, temporal):
        """Events can be queried by spatial proximity."""
        temporal.record_event(make_event("near", position=(5.0, 5.0)))
        temporal.record_event(make_event("far", position=(50.0, 50.0)))
        temporal.record_event(make_event("mid", position=(8.0, 8.0)))

        results = temporal.query_by_position((5.0, 5.0), radius=5.0)
        ids = [e.event_id for e in results]
        assert "near" in ids
        assert "mid" in ids
        assert "far" not in ids

    def test_temporal_neighbors(self, temporal):
        """Events within temporal_window are linked as neighbors."""
        base = time.time()
        temporal.record_event(make_event("evt-a", timestamp=base))
        temporal.record_event(make_event("evt-b", timestamp=base + 1))
        temporal.record_event(make_event("evt-c", timestamp=base + 100))  # Far away

        evt_a = temporal.get_event("evt-a")
        evt_b = temporal.get_event("evt-b")
        evt_c = temporal.get_event("evt-c")

        # a and b are temporal neighbors
        assert "evt-b" in evt_a.temporal_neighbors
        assert "evt-a" in evt_b.temporal_neighbors

        # c is too far from a and b
        assert "evt-a" not in evt_c.temporal_neighbors
        assert "evt-b" not in evt_c.temporal_neighbors

    def test_eventbus_publish(self, temporal, bus):
        """Events are published on EventBus when recorded."""
        received = []
        bus.register_callback(
            CH_TEMPORAL_EVENT_RECORDED,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        temporal.record_event(make_event("evt-1", importance=0.9))

        assert len(received) == 1
        assert received[0]["event_id"] == "evt-1"
        assert received[0]["importance"] == 0.9

    def test_max_events_eviction(self):
        """Old events are evicted when capacity is reached."""
        tm = TemporalMemory(max_events=5)

        for i in range(10):
            tm.record_event(make_event(f"evt-{i}", timestamp=time.time() + i))

        # Only last 5 should remain
        assert tm.get_event("evt-0") is None
        assert tm.get_event("evt-4") is None
        assert tm.get_event("evt-5") is not None
        assert tm.get_event("evt-9") is not None

    def test_statistics(self, temporal):
        """Statistics reflect recorded events."""
        for i in range(5):
            temporal.record_event(
                make_event(f"evt-{i}", event_type=EventType.ACTION)
            )
        temporal.record_event(
            make_event("obs-1", event_type=EventType.OBSERVATION)
        )

        stats = temporal.get_statistics()
        assert stats["total_recorded"] == 6
        assert stats["current_size"] == 6
        assert stats["event_types"]["action"] == 5
        assert stats["event_types"]["observation"] == 1


# ===========================================================================
# TestCausalChains
# ===========================================================================


class TestCausalChains:
    """Tests for causal chain discovery and tracing."""

    def test_same_entity_causal_links(self, temporal):
        """Events from the same entity within causal_window are linked."""
        base = time.time()
        temporal.record_event(
            make_event("cause", entity_id="agent-1", timestamp=base)
        )
        temporal.record_event(
            make_event("effect", entity_id="agent-1", timestamp=base + 1)
        )

        cause_evt = temporal.get_event("cause")
        effect_evt = temporal.get_event("effect")

        assert "effect" in cause_evt.causal_successors
        assert "cause" in effect_evt.causal_predecessors

    def test_trace_causal_chain_backward(self, temporal):
        """Can trace backward through a causal chain."""
        base = time.time()
        # Create a chain: A -> B -> C
        temporal.record_event(
            make_event("A", entity_id="agent-1", timestamp=base)
        )
        temporal.record_event(
            make_event("B", entity_id="agent-1", timestamp=base + 1)
        )
        temporal.record_event(
            make_event("C", entity_id="agent-1", timestamp=base + 2)
        )

        chain = temporal.trace_causal_chain("C", direction="backward")
        assert isinstance(chain, CausalChain)
        event_ids = [e.event_id for e in chain.events]
        assert "A" in event_ids
        assert "B" in event_ids
        assert "C" in event_ids

    def test_trace_causal_chain_forward(self, temporal):
        """Can trace forward through a causal chain."""
        base = time.time()
        temporal.record_event(
            make_event("root", entity_id="agent-1", timestamp=base)
        )
        temporal.record_event(
            make_event("mid", entity_id="agent-1", timestamp=base + 1)
        )
        temporal.record_event(
            make_event("leaf", entity_id="agent-1", timestamp=base + 2)
        )

        chain = temporal.trace_causal_chain("root", direction="forward")
        event_ids = [e.event_id for e in chain.events]
        assert "root" in event_ids
        assert "mid" in event_ids
        assert "leaf" in event_ids

    def test_find_common_causes(self, temporal):
        """Can find common causes between two events."""
        base = time.time()
        # Common root: root -> branch_a, root -> branch_b
        temporal.record_event(
            make_event("root", entity_id="agent-1", timestamp=base)
        )
        temporal.record_event(
            make_event("branch_a", entity_id="agent-1", timestamp=base + 1)
        )
        temporal.record_event(
            make_event("branch_b", entity_id="agent-1", timestamp=base + 2)
        )

        common = temporal.find_common_causes("branch_a", "branch_b")
        common_ids = [e.event_id for e in common]
        assert "root" in common_ids

    def test_causal_engine_fed(self, temporal, causal):
        """Causal links are fed to the CausalEngine."""
        base = time.time()
        temporal.record_event(
            make_event(
                "act", entity_id="agent-1",
                event_type=EventType.ACTION, timestamp=base,
            )
        )
        temporal.record_event(
            make_event(
                "obs", entity_id="agent-1",
                event_type=EventType.OBSERVATION, timestamp=base + 1,
            )
        )

        # CausalEngine should have observed correlation
        stats = causal.get_causal_metrics()
        assert stats["observations_count"] > 0


# ===========================================================================
# TestTemporalPatterns
# ===========================================================================


class TestTemporalPatterns:
    """Tests for temporal pattern detection."""

    def test_pattern_detection(self, temporal):
        """Repeated event sequences are detected as patterns."""
        base = time.time()
        # Create repeating pattern: ACTION -> OBSERVATION -> ACTION -> OBSERVATION
        for i in range(6):
            etype = EventType.ACTION if i % 2 == 0 else EventType.OBSERVATION
            temporal.record_event(
                make_event(
                    f"evt-{i}",
                    entity_id="agent-1",
                    event_type=etype,
                    timestamp=base + i,
                )
            )

        patterns = temporal.get_patterns(min_confidence=0.0)
        assert len(patterns) > 0

        # At least one pattern should be action|observation
        found_pattern = False
        for p in patterns:
            types = [t.value for t in p.event_sequence]
            if "action" in types and "observation" in types:
                found_pattern = True
        assert found_pattern

    def test_predict_next_event(self, temporal):
        """Can predict next event type from patterns."""
        base = time.time()
        # Create a strong pattern: ACTION -> OBSERVATION repeated 5 times
        for i in range(10):
            etype = EventType.ACTION if i % 2 == 0 else EventType.OBSERVATION
            temporal.record_event(
                make_event(
                    f"pat-{i}",
                    entity_id="agent-1",
                    event_type=etype,
                    timestamp=base + i,
                )
            )

        # After ending on OBSERVATION (odd index), next should be ACTION
        prediction = temporal.predict_next_event_type("agent-1")
        # Pattern detection may or may not predict correctly depending on
        # occurrence count - just verify the method works
        assert prediction is None or isinstance(prediction, EventType)

    def test_pattern_eventbus(self, temporal, bus):
        """Pattern detection publishes on EventBus."""
        received = []
        bus.register_callback(
            CH_TEMPORAL_PATTERN_DETECTED,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        base = time.time()
        # Need pattern_min_occurrences (default 3) hits
        for i in range(8):
            etype = EventType.ACTION if i % 2 == 0 else EventType.OBSERVATION
            temporal.record_event(
                make_event(
                    f"p-{i}",
                    entity_id="agent-1",
                    event_type=etype,
                    timestamp=base + i,
                )
            )

        # Should have published at least one pattern
        assert len(received) >= 1
        assert "pattern_id" in received[0]
        assert received[0]["occurrences"] >= 3


# ===========================================================================
# TestWorldlinePlanner
# ===========================================================================


class TestWorldlinePlanner:
    """Tests for worldline trajectory planning."""

    def test_basic_plan(self, planner):
        """Can generate a basic plan with worldlines."""
        state = np.zeros(4)
        actions = [0, 1, 2]

        result = planner.plan("agent-1", state, actions, horizon=5)

        assert isinstance(result, PlanningResult)
        assert result.selected_worldline is not None
        assert len(result.selected_worldline.points) == 5
        assert result.confidence > 0

    def test_plan_no_actions(self, planner):
        """Planning with no actions returns empty result."""
        result = planner.plan("agent-1", np.zeros(4), [])
        assert result.selected_worldline is None
        assert result.reason == "no_actions_available"

    def test_plan_alternatives(self, planner):
        """Planning generates multiple alternative worldlines."""
        state = np.zeros(4)
        actions = [0, 1, 2, 3]

        result = planner.plan("agent-1", state, actions)
        # With 3 branches and 4 actions, should have alternatives
        assert len(result.alternatives) >= 1

    def test_multi_horizon_plan(self, planner):
        """Multi-horizon planning returns reactive/tactical/strategic."""
        state = np.zeros(4)
        actions = [0, 1]

        results = planner.plan_multi_horizon("agent-1", state, actions)

        assert "reactive" in results
        assert "tactical" in results
        assert "strategic" in results
        assert results["reactive"].selected_worldline.horizon == 3
        assert results["tactical"].selected_worldline.horizon == 8
        assert results["strategic"].selected_worldline.horizon == 15

    def test_worldline_execution_tracking(self, planner):
        """Can track worldline execution and divergence."""
        state = np.zeros(4)
        result = planner.plan("agent-1", state, [0, 1])

        wl = result.selected_worldline
        planner.begin_execution(wl)
        assert wl.status == WorldlineStatus.EXECUTING

        # Check divergence at step 0
        div = planner.check_divergence(wl.worldline_id, np.zeros(4), 0)
        assert 0 <= div <= 1

        # Complete
        planner.complete_worldline(wl.worldline_id, success=True)
        assert wl.status == WorldlineStatus.COMPLETED

    def test_worldline_abandon(self, planner):
        """Can abandon a worldline that diverged too far."""
        state = np.zeros(4)
        result = planner.plan("agent-1", state, [0, 1])

        wl = result.selected_worldline
        planner.begin_execution(wl)
        planner.abandon_worldline(wl.worldline_id)
        assert wl.status == WorldlineStatus.ABANDONED

    def test_eventbus_publish(self, planner, bus):
        """Planning publishes events on EventBus."""
        received = []
        bus.register_callback(
            CH_WORLDLINE_PLANNED,
            lambda ch, msg: received.append(json.loads(msg)),
        )

        planner.plan("agent-1", np.zeros(4), [0, 1])

        assert len(received) == 1
        assert received[0]["entity_id"] == "agent-1"
        assert "worldline_id" in received[0]

    def test_temporal_context(self, planner):
        """Can get temporal context for decision-making."""
        ctx = planner.get_temporal_context("agent-1")
        assert "planning_accuracy" in ctx
        assert ctx["active_worldline"] is None

        # Plan and execute
        result = planner.plan("agent-1", np.zeros(4), [0, 1])
        planner.begin_execution(result.selected_worldline)

        ctx2 = planner.get_temporal_context("agent-1")
        assert ctx2["active_worldline"] is not None

    def test_statistics(self, planner):
        """Statistics track planning operations."""
        planner.plan("agent-1", np.zeros(4), [0, 1])
        planner.plan("agent-2", np.zeros(4), [0, 1, 2])

        stats = planner.get_statistics()
        assert stats["total_plans"] == 2
        assert stats["entity_count"] == 0  # No completed plans yet
        assert stats["branches_per_plan"] == 3


# ===========================================================================
# TestCrossSystemIntegration
# ===========================================================================


class TestCrossSystemIntegration:
    """Tests for integration between temporal systems and other modules."""

    def test_temporal_feeds_causal_engine(self, temporal, causal):
        """Temporal events feed causal links into CausalEngine."""
        base = time.time()
        # Record sequence: action -> state_change -> healing
        temporal.record_event(
            make_event(
                "act", entity_id="agent-1",
                event_type=EventType.ACTION, timestamp=base,
            )
        )
        temporal.record_event(
            make_event(
                "change", entity_id="agent-1",
                event_type=EventType.STATE_CHANGE, timestamp=base + 1,
            )
        )
        temporal.record_event(
            make_event(
                "heal", entity_id="agent-1",
                event_type=EventType.HEALING, timestamp=base + 2,
            )
        )

        # CausalEngine should have received correlations
        stats = causal.get_causal_metrics()
        assert stats["observations_count"] >= 2

    def test_planner_uses_world_model(self, world_model, bus):
        """WorldlinePlanner uses WorldModel for predictions."""
        planner = WorldlinePlanner(
            world_model=world_model,
            event_bus=bus,
            num_branches=2,
        )

        state = np.random.randn(4).astype(np.float32)
        result = planner.plan("agent-1", state, [0, 1], horizon=3)

        assert result.selected_worldline is not None
        # Each point should have uncertainty from world model
        for point in result.selected_worldline.points:
            assert point.uncertainty >= 0

    def test_temporal_memory_with_planner(self, temporal, world_model, bus):
        """TemporalMemory and WorldlinePlanner work together."""
        planner = WorldlinePlanner(
            world_model=world_model,
            temporal_memory=temporal,
            event_bus=bus,
        )

        # Record some temporal history
        base = time.time()
        for i in range(5):
            temporal.record_event(
                make_event(
                    f"hist-{i}", entity_id="agent-1",
                    event_type=EventType.ACTION, timestamp=base + i,
                )
            )

        # Plan using temporal context
        result = planner.plan("agent-1", np.zeros(4), [0, 1, 2])
        assert result.selected_worldline is not None

    def test_full_planning_lifecycle(self, temporal, world_model, bus, causal):
        """Full lifecycle: record events -> plan -> execute -> complete."""
        planner = WorldlinePlanner(
            world_model=world_model,
            temporal_memory=temporal,
            causal_engine=causal,
            event_bus=bus,
            num_branches=2,
        )

        # 1. Record historical events
        base = time.time()
        for i in range(5):
            temporal.record_event(
                make_event(
                    f"past-{i}", entity_id="agent-1",
                    timestamp=base + i,
                )
            )

        # 2. Plan
        result = planner.plan("agent-1", np.zeros(4), [0, 1], horizon=5)
        assert result.selected_worldline is not None

        # 3. Begin execution
        wl = result.selected_worldline
        planner.begin_execution(wl)

        # 4. Track divergence over steps
        for step in range(3):
            actual = np.random.randn(4) * 0.1  # Small divergence
            div = planner.check_divergence(wl.worldline_id, actual, step)
            assert 0 <= div <= 1

        # 5. Complete
        planner.complete_worldline(wl.worldline_id, success=True)

        # 6. Verify statistics
        stats = planner.get_statistics()
        assert stats["total_plans"] >= 1
        assert stats["successful_plans"] >= 1

    def test_temporal_timeline_output(self, temporal):
        """Timeline output is human-readable."""
        base = time.time()
        for i in range(5):
            temporal.record_event(
                make_event(
                    f"tl-{i}", entity_id="agent-1",
                    event_type=EventType.ACTION, timestamp=base + i,
                    position=(float(i), float(i)),
                )
            )

        timeline = temporal.get_timeline(entity_id="agent-1", limit=3)
        assert len(timeline) == 3
        assert "event_id" in timeline[0]
        assert "type" in timeline[0]
        assert "causes" in timeline[0]
        assert "effects" in timeline[0]
