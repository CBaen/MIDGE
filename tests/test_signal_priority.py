"""Tests for Signal Priority Protocol — thalamus-like signal triage."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.cognition.decision_router import DecisionTier
from mae_core.communication.signal_bus import Signal, SignalBus
from mae_core.communication.signal_priority import (
    PriorityConfig,
    PrioritizedSignal,
    SignalPriorityResolver,
    StepReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return SignalBus(EventBus())


@pytest.fixture
def resolver(bus):
    return SignalPriorityResolver(agent_id="test_agent", signal_bus=bus)


def _emit(bus, signal_type="DANGER", priority=0.5, payload=None, sender="a1"):
    bus.emit_signal(
        signal_type=signal_type,
        payload=payload or {},
        sender_id=sender,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Unit Tests — Basic Queue and Dispatch
# ---------------------------------------------------------------------------

class TestBasicQueueDispatch:
    def test_enqueue_and_process(self, bus, resolver):
        """Signals queue on emit and dispatch during process()."""
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))
        _emit(bus, "DANGER", priority=0.5)

        assert len(received) == 0  # Not yet dispatched
        report = resolver.process()
        assert len(received) == 1
        assert report.step_number == 1
        assert len(report.processed) == 1

    def test_empty_inbox(self, resolver):
        """Processing empty inbox returns empty report."""
        report = resolver.process()
        assert report.step_number == 1
        assert len(report.processed) == 0
        assert len(report.deferred) == 0
        assert len(report.dropped) == 0

    def test_no_double_dispatch(self, bus, resolver):
        """Processed signals are not dispatched again on next process()."""
        count = []
        resolver.register_handler("DANGER", lambda s: count.append(1))
        _emit(bus, "DANGER")

        resolver.process()
        assert len(count) == 1

        resolver.process()
        assert len(count) == 1  # Still 1 — not dispatched again

    def test_expired_ttl_filtered(self, bus, resolver):
        """Expired signals (TTL exceeded) are not enqueued."""
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))

        # Emit with very short TTL, then wait
        bus.emit_signal(
            signal_type="DANGER",
            payload={},
            sender_id="a1",
            priority=0.5,
            ttl=0.001,
        )
        time.sleep(0.01)

        report = resolver.process()
        assert len(received) == 0
        assert len(report.processed) == 0


# ---------------------------------------------------------------------------
# Unit Tests — Priority Ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_higher_priority_first(self, bus, resolver):
        """Higher priority signals are processed before lower ones."""
        order = []
        resolver.register_handler("DANGER", lambda s: order.append(("DANGER", s.priority)))
        resolver.register_handler("KNOWLEDGE_SHARE", lambda s: order.append(("KNOW", s.priority)))

        _emit(bus, "KNOWLEDGE_SHARE", priority=0.2)
        _emit(bus, "DANGER", priority=0.8)

        resolver.process()
        assert len(order) == 2
        assert order[0][0] == "DANGER"
        assert order[1][0] == "KNOW"

    def test_same_type_different_priority(self, bus, resolver):
        """Multiple signals of same type sorted by computed score."""
        scores = []

        def capture(s):
            scores.append(s.priority)

        resolver.register_handler("OPPORTUNITY", capture)
        # Coalescing is on by default, so same-type signals get merged.
        # Disable coalescing to test raw ordering.
        resolver._config.coalesce_same_type = False

        _emit(bus, "OPPORTUNITY", priority=0.3)
        _emit(bus, "OPPORTUNITY", priority=0.7)
        _emit(bus, "OPPORTUNITY", priority=0.1)

        resolver.process()
        assert len(scores) == 3
        # Should be descending (highest first)
        assert scores[0] == 0.7
        assert scores[2] == 0.1


# ---------------------------------------------------------------------------
# Unit Tests — Budget and Deferral
# ---------------------------------------------------------------------------

class TestBudgetAndDeferral:
    def test_budget_enforcement(self, bus):
        """Only budget_per_step signals are processed."""
        config = PriorityConfig(budget_per_step=3)
        resolver = SignalPriorityResolver("a1", bus, config=config)
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))

        for i in range(5):
            _emit(bus, "DANGER", priority=0.4 + i * 0.01)

        # Coalescing merges same-type, so disable it for this test
        resolver._config.coalesce_same_type = False
        report = resolver.process()

        assert len(received) == 3
        assert len(report.processed) == 3
        assert len(report.deferred) == 2

    def test_deferred_carry_over(self, bus):
        """Deferred signals carry over to the next step."""
        config = PriorityConfig(budget_per_step=2, coalesce_same_type=False)
        resolver = SignalPriorityResolver("a1", bus, config=config)
        received = []
        resolver.register_handler("OPPORTUNITY", lambda s: received.append(s))

        for i in range(4):
            _emit(bus, "OPPORTUNITY", priority=0.3 + i * 0.05)

        report1 = resolver.process()
        assert len(report1.processed) == 2
        assert len(report1.deferred) == 2

        # Next step — deferred signals should be processed
        report2 = resolver.process()
        assert len(report2.processed) == 2
        assert len(received) == 4

    def test_deferred_expiry(self, bus):
        """Deferred signals expire after max age steps."""
        config = PriorityConfig(
            budget_per_step=1,
            defer_max_age_steps=1,
            coalesce_same_type=False,
        )
        resolver = SignalPriorityResolver("a1", bus, config=config)
        received = []
        resolver.register_handler("OPPORTUNITY", lambda s: received.append(s))

        for i in range(3):
            _emit(bus, "OPPORTUNITY", priority=0.3 + i * 0.05)

        # Step 1: process 1, defer 2
        resolver.process()
        assert len(received) == 1

        # Step 2: process 1 deferred, defer 1
        resolver.process()
        assert len(received) == 2

        # Step 3: last deferred has age=2 (enqueued step 1, now step 3), max=1 → dropped
        report3 = resolver.process()
        assert len(received) == 2  # Nothing new processed
        assert len(report3.dropped) >= 1

    def test_overflow_dropped_when_defer_disabled(self, bus):
        """With defer_overflow=False, overflow signals are dropped."""
        config = PriorityConfig(
            budget_per_step=2,
            defer_overflow=False,
            coalesce_same_type=False,
        )
        resolver = SignalPriorityResolver("a1", bus, config=config)
        resolver.register_handler("DANGER", lambda s: None)

        for i in range(4):
            _emit(bus, "DANGER", priority=0.3 + i * 0.01)

        report = resolver.process()
        assert len(report.processed) == 2
        assert len(report.dropped) == 2
        assert len(report.deferred) == 0


# ---------------------------------------------------------------------------
# Unit Tests — Coalescing
# ---------------------------------------------------------------------------

class TestCoalescing:
    def test_same_type_coalesced(self, bus, resolver):
        """Multiple signals of same type merge into one."""
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))

        _emit(bus, "DANGER", priority=0.6, sender="a1")
        _emit(bus, "DANGER", priority=0.4, sender="a2")
        _emit(bus, "DANGER", priority=0.5, sender="a3")

        report = resolver.process()
        assert len(received) == 1  # 3 signals → 1 coalesced
        assert report.coalesced_count == 2  # 3 - 1 = 2 merged

    def test_coalesced_preserves_senders(self, bus, resolver):
        """Coalesced signal tracks all original senders."""
        resolver.register_handler("DANGER", lambda s: None)

        _emit(bus, "DANGER", priority=0.5, sender="a1")
        _emit(bus, "DANGER", priority=0.5, sender="a2")

        report = resolver.process()
        ps = report.processed[0]
        assert ps.coalesced_count == 2
        assert "a1" in ps.coalesced_senders
        assert "a2" in ps.coalesced_senders

    def test_coalesced_preserves_payloads(self, bus, resolver):
        """Coalesced signal tracks all original payloads."""
        resolver.register_handler("DANGER", lambda s: None)

        bus.emit_signal("DANGER", {"risk": 0.9}, sender_id="a1", priority=0.5)
        bus.emit_signal("DANGER", {"risk": 0.3}, sender_id="a2", priority=0.5)

        report = resolver.process()
        ps = report.processed[0]
        assert len(ps.coalesced_payloads) == 2
        risks = [p.get("risk") for p in ps.coalesced_payloads]
        assert 0.9 in risks
        assert 0.3 in risks

    def test_different_types_not_coalesced(self, bus, resolver):
        """Different signal types remain separate."""
        received = []
        resolver.register_handler("DANGER", lambda s: received.append("D"))
        resolver.register_handler("OPPORTUNITY", lambda s: received.append("O"))

        _emit(bus, "DANGER", priority=0.5)
        _emit(bus, "OPPORTUNITY", priority=0.5)

        resolver.process()
        assert len(received) == 2


# ---------------------------------------------------------------------------
# Unit Tests — Preemption
# ---------------------------------------------------------------------------

class TestPreemption:
    def test_preemption_enabled(self, bus, resolver):
        """CRITICAL signals (>= threshold) dispatch immediately."""
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))

        _emit(bus, "DANGER", priority=0.95)

        # Signal should have been dispatched immediately (not queued)
        assert len(received) == 1

        # Process should have nothing to dispatch
        report = resolver.process()
        assert len(report.processed) == 0

    def test_preemption_disabled(self, bus):
        """With preemption disabled, critical signals queue normally."""
        config = PriorityConfig(enable_preemption=False)
        resolver = SignalPriorityResolver("a1", bus, config=config)
        received = []
        resolver.register_handler("DANGER", lambda s: received.append(s))

        _emit(bus, "DANGER", priority=0.95)

        assert len(received) == 0  # Not dispatched immediately
        resolver.process()
        assert len(received) == 1  # Dispatched during process()


# ---------------------------------------------------------------------------
# Unit Tests — Tier Mapping
# ---------------------------------------------------------------------------

class TestTierMapping:
    def test_reflex_tier(self, bus, resolver):
        """Score >= 0.8 maps to REFLEX."""
        resolver.register_handler("DANGER", lambda s: None)
        _emit(bus, "DANGER", priority=0.85)  # DANGER urgency=1.0, high priority

        report = resolver.process()
        assert len(report.processed) == 1
        assert report.processed[0].decision_tier == DecisionTier.REFLEX

    def test_habit_tier(self, bus, resolver):
        """Score in [0.5, 0.8) maps to HABIT."""
        resolver.register_handler("OPPORTUNITY", lambda s: None)
        _emit(bus, "OPPORTUNITY", priority=0.5)

        report = resolver.process()
        assert len(report.processed) == 1
        assert report.processed[0].decision_tier == DecisionTier.HABIT

    def test_prefrontal_tier(self, bus, resolver):
        """Score < 0.5 maps to PREFRONTAL."""
        resolver.register_handler("KNOWLEDGE_SHARE", lambda s: None)
        _emit(bus, "KNOWLEDGE_SHARE", priority=0.1)

        report = resolver.process()
        assert len(report.processed) == 1
        assert report.processed[0].decision_tier == DecisionTier.PREFRONTAL


# ---------------------------------------------------------------------------
# Unit Tests — Statistics and Reporting
# ---------------------------------------------------------------------------

class TestStatisticsAndReporting:
    def test_statistics_tracking(self, bus, resolver):
        """Cumulative statistics are tracked correctly."""
        resolver.register_handler("DANGER", lambda s: None)
        _emit(bus, "DANGER", priority=0.5)
        resolver.process()

        stats = resolver.get_statistics()
        assert stats["agent_id"] == "test_agent"
        assert stats["current_step"] == 1
        assert stats["total_received"] == 1
        assert stats["total_processed"] == 1
        assert stats["handler_count"] == 1

    def test_step_report_contents(self, bus, resolver):
        """StepReport contains expected fields."""
        resolver.register_handler("DANGER", lambda s: None)
        _emit(bus, "DANGER", priority=0.5)

        report = resolver.process()
        assert isinstance(report, StepReport)
        assert report.step_number == 1
        assert report.total_received == 1
        assert len(report.processed) == 1
        assert len(report.deferred) == 0
        assert len(report.dropped) == 0
        assert len(report.preempted) == 0


# ---------------------------------------------------------------------------
# Unit Tests — Custom Configuration
# ---------------------------------------------------------------------------

class TestCustomConfig:
    def test_custom_urgency_map(self, bus):
        """Custom urgency map overrides defaults."""
        resolver = SignalPriorityResolver(
            "a1", bus,
            urgency_map={"CUSTOM_TYPE": 0.99},
        )
        resolver.register_handler("CUSTOM_TYPE", lambda s: None)
        _emit(bus, "CUSTOM_TYPE", priority=0.5)

        report = resolver.process()
        ps = report.processed[0]
        # High urgency (0.99) should push score toward REFLEX
        assert ps.computed_score > 0.5


# ---------------------------------------------------------------------------
# Unit Tests — Error Isolation
# ---------------------------------------------------------------------------

class TestErrorIsolation:
    def test_handler_exception_isolated(self, bus, resolver):
        """Exception in one handler does not crash process()."""
        def bad_handler(s):
            raise ValueError("boom")

        received = []
        resolver.register_handler("DANGER", bad_handler)
        resolver.register_handler("OPPORTUNITY", lambda s: received.append(s))

        _emit(bus, "DANGER", priority=0.5)
        _emit(bus, "OPPORTUNITY", priority=0.5)

        report = resolver.process()
        # OPPORTUNITY should still be dispatched despite DANGER handler failing
        assert len(received) == 1
        assert len(report.processed) == 2


# ---------------------------------------------------------------------------
# Unit Tests — Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_serialize_restore(self, bus, resolver, tmp_path):
        """Cumulative statistics survive serialize/restore."""
        resolver.register_handler("DANGER", lambda s: None)

        for _ in range(5):
            _emit(bus, "DANGER", priority=0.5)
            resolver.process()

        meta = resolver.serialize(tmp_path)
        assert meta["current_step"] == 5
        assert meta["total_processed"] == 5

        resolver2 = SignalPriorityResolver("test_agent", bus)
        resolver2.restore(tmp_path, meta)
        assert resolver2._current_step == 5
        assert resolver2._total_processed == 5


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_agent_processes_signals_in_priority_order(self, bus):
        """Full flow: emit signals, agent step processes them in order."""
        from mae_core.agents.mycelial_agent import MycelialAgent

        model = MagicMock()
        model.schedule = MagicMock()
        model.schedule.agents = []

        agent = MycelialAgent(model=model, signal_bus=bus)
        assert agent._signal_resolver is not None

        # Track dispatch order via the resolver's last report
        bus.emit_signal("KNOWLEDGE_SHARE", {"k": 1}, sender_id="peer1", priority=0.2)
        bus.emit_signal("DANGER", {"risk": 0.9}, sender_id="peer2", priority=0.8)
        bus.emit_signal("OPPORTUNITY", {"val": 5}, sender_id="peer3", priority=0.5)

        agent.step()

        report = agent._signal_resolver.get_last_report()
        assert report is not None
        assert len(report.processed) == 3

        # DANGER should be first (highest computed score)
        types = [ps.signal.signal_type for ps in report.processed]
        assert types[0] == "DANGER"

    def test_agent_without_signal_bus_backward_compatible(self):
        """Agent created without signal_bus works normally."""
        model = MagicMock()
        model.schedule = MagicMock()
        model.schedule.agents = []

        from mae_core.agents.mycelial_agent import MycelialAgent
        agent = MycelialAgent(model=model, signal_bus=None)
        assert agent._signal_resolver is None

        # Step should work without errors
        agent.step()

    def test_multi_type_concurrent_sorted(self, bus):
        """Multiple signal types arrive simultaneously, sorted correctly."""
        resolver = SignalPriorityResolver("a1", bus)
        order = []
        resolver.register_handler("DANGER", lambda s: order.append("DANGER"))
        resolver.register_handler("OPPORTUNITY", lambda s: order.append("OPP"))
        resolver.register_handler("KNOWLEDGE_SHARE", lambda s: order.append("KNOW"))
        resolver.register_handler("CONVERGENCE", lambda s: order.append("CONV"))

        # Emit in reverse priority order
        _emit(bus, "KNOWLEDGE_SHARE", priority=0.2)
        _emit(bus, "CONVERGENCE", priority=0.4)
        _emit(bus, "OPPORTUNITY", priority=0.5)
        _emit(bus, "DANGER", priority=0.8)

        resolver.process()

        # DANGER has highest urgency (1.0) + highest priority (0.8)
        assert order[0] == "DANGER"
        # KNOWLEDGE_SHARE has lowest urgency (0.3) + lowest priority (0.2)
        assert order[-1] == "KNOW"

    def test_coalesced_danger_beats_single_opportunity(self, bus):
        """3 DANGER signals coalesced should outrank 1 OPPORTUNITY."""
        resolver = SignalPriorityResolver("a1", bus)
        order = []
        resolver.register_handler("DANGER", lambda s: order.append("DANGER"))
        resolver.register_handler("OPPORTUNITY", lambda s: order.append("OPP"))

        # Emit 3 danger signals (will coalesce) and 1 opportunity
        _emit(bus, "DANGER", priority=0.4, sender="a1")
        _emit(bus, "DANGER", priority=0.4, sender="a2")
        _emit(bus, "DANGER", priority=0.4, sender="a3")
        _emit(bus, "OPPORTUNITY", priority=0.6)

        report = resolver.process()
        assert report.coalesced_count == 2  # 3→1 = 2 merged

        # Coalesced DANGER (boosted) should come before OPPORTUNITY
        assert order[0] == "DANGER"
        assert order[1] == "OPP"
