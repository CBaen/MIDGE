"""Tests for autopoietic closure at subsystem, organ, and organism scales.

Law 6: Autopoietic Closure — At every scale, components produce the
processes that produce the components.

Test coverage:
- Subsystem closure: monitors child systems, heals degraded ones
- Organ closure: monitors child subsystems, triggers healing
- Organism closure: monitors organs, advises on rebalancing
- Closure coordinator: orchestrates all scales together
- Advisory mode: closure observes and reports, never blocks
"""

import pytest

from mae_core.backbone.autopoietic_closure import (
    AutopoieticMonitor,
    ClosureCoordinator,
    ClosureReport,
    OrganClosure,
    OrganismClosure,
    SubsystemClosure,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.holon_protocol import HolonRegistry


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def holon_registry():
    """Create a fresh HolonRegistry."""
    return HolonRegistry()


@pytest.fixture
def event_bus():
    """Create a fresh EventBus."""
    return EventBus()


@pytest.fixture
def simple_hierarchy(holon_registry):
    """Create a simple 3-level hierarchy for testing.

    Structure:
        mae (organism)
        ├── nervous-system (organ)
        │   ├── core-backbone (subsystem)
        │   │   ├── event_bus (system)
        │   │   ├── holon_registry (system)
        │   │   └── connection_registry (system)
        │   ├── enforcement (subsystem)
        │   │   ├── enforcer (system)
        │   │   ├── watchdog (system)
        │   │   └── auditor (system)
        │   └── rhythm (subsystem)
        │       ├── circadian (system)
        │       ├── endocrine (system)
        │       └── awareness_pulse (system)
        └── sensory-system (organ)
            ├── fabric (subsystem)
            │   ├── substrate (system)
            │   ├── physarum (system)
            │   └── signal_bus (system)
            ├── routing (subsystem)
            │   ├── gnn_communicator (system)
            │   ├── stigmergy (system)
            │   └── predictive_field (system)
            └── consensus (subsystem)
                ├── quorum_space (system)
                ├── nociception (system)
                └── proprioception (system)
    """
    # Register mae
    holon_registry.register("mae", holon_type="organism", parent_id=None)

    # Register nervous-system organ
    holon_registry.register(
        "nervous-system", holon_type="organ", parent_id="mae"
    )

    # Register nervous-system subsystems
    holon_registry.register(
        "core-backbone", holon_type="subsystem", parent_id="nervous-system"
    )
    holon_registry.register(
        "enforcement", holon_type="subsystem", parent_id="nervous-system"
    )
    holon_registry.register(
        "rhythm", holon_type="subsystem", parent_id="nervous-system"
    )

    # Register core-backbone systems
    holon_registry.register(
        "event_bus", holon_type="system", parent_id="core-backbone"
    )
    holon_registry.register(
        "holon_registry", holon_type="system", parent_id="core-backbone"
    )
    holon_registry.register(
        "connection_registry", holon_type="system", parent_id="core-backbone"
    )

    # Register enforcement systems
    holon_registry.register(
        "enforcer", holon_type="system", parent_id="enforcement"
    )
    holon_registry.register(
        "watchdog", holon_type="system", parent_id="enforcement"
    )
    holon_registry.register(
        "auditor", holon_type="system", parent_id="enforcement"
    )

    # Register rhythm systems
    holon_registry.register(
        "circadian", holon_type="system", parent_id="rhythm"
    )
    holon_registry.register(
        "endocrine", holon_type="system", parent_id="rhythm"
    )
    holon_registry.register(
        "awareness_pulse", holon_type="system", parent_id="rhythm"
    )

    # Register sensory-system organ
    holon_registry.register(
        "sensory-system", holon_type="organ", parent_id="mae"
    )

    # Register sensory-system subsystems
    holon_registry.register(
        "fabric", holon_type="subsystem", parent_id="sensory-system"
    )
    holon_registry.register(
        "routing", holon_type="subsystem", parent_id="sensory-system"
    )
    holon_registry.register(
        "consensus", holon_type="subsystem", parent_id="sensory-system"
    )

    # Register fabric systems
    holon_registry.register(
        "substrate", holon_type="system", parent_id="fabric"
    )
    holon_registry.register(
        "physarum", holon_type="system", parent_id="fabric"
    )
    holon_registry.register(
        "signal_bus", holon_type="system", parent_id="fabric"
    )

    # Register routing systems
    holon_registry.register(
        "gnn_communicator", holon_type="system", parent_id="routing"
    )
    holon_registry.register(
        "stigmergy", holon_type="system", parent_id="routing"
    )
    holon_registry.register(
        "predictive_field", holon_type="system", parent_id="routing"
    )

    # Register consensus systems
    holon_registry.register(
        "quorum_space", holon_type="system", parent_id="consensus"
    )
    holon_registry.register(
        "nociception", holon_type="system", parent_id="consensus"
    )
    holon_registry.register(
        "proprioception", holon_type="system", parent_id="consensus"
    )

    return holon_registry


# =====================================================================
# Tests: AutopoieticMonitor (base class)
# =====================================================================


class TestAutopoieticMonitor:
    """Tests for the base AutopoieticMonitor advisory pattern."""

    def test_monitor_initialization(self, holon_registry, event_bus):
        """Monitor initializes correctly."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            event_bus=event_bus,
            health_threshold=0.5,
            cadence=5,
        )
        assert monitor._threshold == 0.5
        assert monitor._cadence == 5
        assert monitor._cycle_count == 0

    def test_cadence_skips_checks(self, holon_registry, simple_hierarchy):
        """Monitor skips checks until cadence interval."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            cadence=5,  # Check every 5 steps
        )

        # Steps 1-4 should be skipped
        assert monitor.step(1, "nervous-system", "organ") is None
        assert monitor.step(2, "nervous-system", "organ") is None
        assert monitor.step(3, "nervous-system", "organ") is None
        assert monitor.step(4, "nervous-system", "organ") is None

        # Step 5 should run
        report = monitor.step(5, "nervous-system", "organ")
        assert report is not None
        assert report.scale == "organ"
        assert report.monitored_id == "nervous-system"

    def test_cycle_idempotency(self, holon_registry, simple_hierarchy):
        """Monitor doesn't re-check in the same step."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            cadence=5,
        )

        # First call at step 5 runs
        report1 = monitor.step(5, "nervous-system", "organ")
        assert report1 is not None

        # Second call at step 5 doesn't run (already checked this step)
        report2 = monitor.step(5, "nervous-system", "organ")
        assert report2 is None

        # Step 10 runs again
        report3 = monitor.step(10, "nervous-system", "organ")
        assert report3 is not None

    def test_monitor_healthy_children(self, holon_registry, simple_hierarchy):
        """Monitor counts healthy children correctly."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            health_threshold=0.5,
            cadence=1,
        )

        # All child systems in core-backbone should have default health 0.75
        report = monitor._monitor_cycle("core-backbone", "subsystem", 1)

        assert report.healthy_children == 3  # All 3 systems healthy
        assert report.degraded_children == 0
        assert report.restructure_advice == ""

    def test_monitor_generates_report(self, holon_registry, simple_hierarchy):
        """Monitor generates report structure correctly."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            health_threshold=0.5,
            cadence=1,
        )

        report = monitor._monitor_cycle("core-backbone", "subsystem", 1)

        # Report structure is always created
        assert report is not None
        assert report.scale == "subsystem"
        assert report.monitored_id == "core-backbone"
        assert report.healthy_children >= 0
        assert report.degraded_children >= 0
        assert isinstance(report.metadata, dict)

    def test_monitor_statistics(self, holon_registry, simple_hierarchy):
        """Monitor maintains statistics correctly."""
        monitor = AutopoieticMonitor(
            holon_registry=holon_registry,
            cadence=1,
        )

        # Run a few cycles
        monitor._monitor_cycle("core-backbone", "subsystem", 1)
        monitor._monitor_cycle("enforcement", "subsystem", 2)
        monitor._monitor_cycle("rhythm", "subsystem", 3)

        stats = monitor.get_statistics()

        assert stats["cycles"] == 3
        assert "recent_reports" in stats
        assert len(stats["recent_reports"]) == 3


# =====================================================================
# Tests: SubsystemClosure
# =====================================================================


class TestSubsystemClosure:
    """Tests for subsystem-level autopoietic closure."""

    def test_subsystem_closure_initialization(self, holon_registry):
        """SubsystemClosure initializes correctly."""
        closure = SubsystemClosure(holon_registry)
        assert closure._monitor is not None

    def test_subsystem_cadence(self, holon_registry, simple_hierarchy):
        """SubsystemClosure uses correct cadence (5 steps)."""
        closure = SubsystemClosure(holon_registry)

        # CADENCE_SUBSYSTEM = 5
        # Steps 1-4: skipped
        assert closure.step(1, "core-backbone") is None
        assert closure.step(2, "core-backbone") is None
        assert closure.step(3, "core-backbone") is None
        assert closure.step(4, "core-backbone") is None

        # Step 5: runs
        report = closure.step(5, "core-backbone")
        assert report is not None
        assert report.scale == "subsystem"

    def test_subsystem_closure_generates_report(
        self, holon_registry, simple_hierarchy
    ):
        """SubsystemClosure generates reports correctly."""
        closure = SubsystemClosure(holon_registry)

        report = closure.step(5, "core-backbone")

        assert report is not None
        assert report.scale == "subsystem"
        assert report.monitored_id == "core-backbone"
        # core-backbone has 3 child systems
        assert report.healthy_children + report.degraded_children == 3

    def test_subsystem_closure_statistics(self, holon_registry, simple_hierarchy):
        """SubsystemClosure reports statistics."""
        closure = SubsystemClosure(holon_registry)

        closure.step(5, "core-backbone")
        closure.step(10, "enforcement")

        stats = closure.get_statistics()
        assert "cycles" in stats
        assert stats["cycles"] == 2


# =====================================================================
# Tests: OrganClosure
# =====================================================================


class TestOrganClosure:
    """Tests for organ-level autopoietic closure."""

    def test_organ_closure_initialization(self, holon_registry):
        """OrganClosure initializes correctly."""
        closure = OrganClosure(holon_registry)
        assert closure._monitor is not None

    def test_organ_cadence(self, holon_registry, simple_hierarchy):
        """OrganClosure uses correct cadence (8 steps)."""
        closure = OrganClosure(holon_registry)

        # CADENCE_ORGAN = 8
        # Steps 1-7: skipped
        for step in range(1, 8):
            assert closure.step(step, "nervous-system") is None

        # Step 8: runs
        report = closure.step(8, "nervous-system")
        assert report is not None
        assert report.scale == "organ"

    def test_organ_closure_monitors_subsystems(
        self, holon_registry, simple_hierarchy
    ):
        """OrganClosure monitors organ's child subsystems."""
        closure = OrganClosure(holon_registry)

        # nervous-system has 3 subsystems: core-backbone, enforcement, rhythm
        report = closure.step(8, "nervous-system")

        assert report is not None
        assert report.monitored_id == "nervous-system"
        assert report.healthy_children == 3  # All subsystems healthy

    def test_organ_closure_generates_report(
        self, holon_registry, simple_hierarchy
    ):
        """OrganClosure generates reports correctly."""
        closure = OrganClosure(holon_registry)

        report = closure.step(8, "nervous-system")

        assert report is not None
        assert report.scale == "organ"
        assert report.monitored_id == "nervous-system"
        # nervous-system has 3 child subsystems
        assert report.healthy_children + report.degraded_children == 3

    def test_organ_closure_statistics(self, holon_registry, simple_hierarchy):
        """OrganClosure reports statistics."""
        closure = OrganClosure(holon_registry)

        closure.step(8, "nervous-system")
        closure.step(16, "sensory-system")

        stats = closure.get_statistics()
        assert stats["cycles"] == 2


# =====================================================================
# Tests: OrganismClosure
# =====================================================================


class TestOrganismClosure:
    """Tests for organism-level autopoietic closure."""

    def test_organism_closure_initialization(self, holon_registry):
        """OrganismClosure initializes correctly."""
        closure = OrganismClosure(holon_registry)
        assert closure._monitor is not None

    def test_organism_cadence(self, holon_registry, simple_hierarchy):
        """OrganismClosure uses correct cadence (13 steps)."""
        closure = OrganismClosure(holon_registry)

        # CADENCE_ORGANISM = 13
        # Steps 1-12: skipped
        for step in range(1, 13):
            assert closure.step(step) is None

        # Step 13: runs
        report = closure.step(13)
        assert report is not None
        assert report.scale == "organism"

    def test_organism_closure_monitors_organs(
        self, holon_registry, simple_hierarchy
    ):
        """OrganismClosure monitors Mae's child organs."""
        closure = OrganismClosure(holon_registry)

        # mae has 2 organs: nervous-system, sensory-system
        report = closure.step(13)

        assert report is not None
        assert report.monitored_id == "mae"
        assert report.healthy_children == 2  # Both organs healthy

    def test_organism_closure_generates_report(
        self, holon_registry, simple_hierarchy
    ):
        """OrganismClosure generates reports correctly."""
        closure = OrganismClosure(holon_registry)

        report = closure.step(13)

        assert report is not None
        assert report.scale == "organism"
        assert report.monitored_id == "mae"
        # mae has 2 child organs (nervous-system, sensory-system)
        assert report.healthy_children + report.degraded_children == 2

    def test_organism_closure_advice_structure(
        self, holon_registry, simple_hierarchy
    ):
        """OrganismClosure report includes advice field."""
        closure = OrganismClosure(holon_registry)

        report = closure.step(13)

        assert report is not None
        # Advice field is always present (may be empty)
        assert hasattr(report, "restructure_advice")
        assert isinstance(report.restructure_advice, str)

    def test_organism_closure_statistics(self, holon_registry, simple_hierarchy):
        """OrganismClosure reports statistics."""
        closure = OrganismClosure(holon_registry)

        closure.step(13)
        closure.step(26)

        stats = closure.get_statistics()
        assert stats["cycles"] == 2


# =====================================================================
# Tests: ClosureCoordinator
# =====================================================================


class TestClosureCoordinator:
    """Tests for closure coordination across all scales."""

    def test_coordinator_initialization(self, holon_registry):
        """ClosureCoordinator initializes all sub-monitors."""
        coordinator = ClosureCoordinator(holon_registry)

        assert coordinator._subsystem_closure is not None
        assert coordinator._organ_closure is not None
        assert coordinator._organism_closure is not None

    def test_coordinator_runs_all_scales(self, holon_registry, simple_hierarchy):
        """ClosureCoordinator runs checks at all scales that have cadence."""
        coordinator = ClosureCoordinator(holon_registry, None)

        # At step 5: subsystems check (cadence 5)
        # At step 8: organs check (cadence 8)
        # At step 13: organisms check (cadence 13)

        # Step 5: subsystems only
        reports = coordinator.step(5)
        subsystem_reports = [r for r in reports if r.scale == "subsystem"]
        assert len(subsystem_reports) > 0  # Multiple subsystems

        # Step 8: organs (and subsystems again if they align)
        reports = coordinator.step(8)
        organ_reports = [r for r in reports if r.scale == "organ"]
        assert len(organ_reports) > 0

        # Step 13: organisms (and possibly others)
        reports = coordinator.step(13)
        organism_reports = [r for r in reports if r.scale == "organism"]
        assert len(organism_reports) > 0

    def test_coordinator_publishes_reports(
        self, holon_registry, simple_hierarchy, event_bus
    ):
        """ClosureCoordinator publishes reports on EventBus."""
        coordinator = ClosureCoordinator(holon_registry, event_bus)

        # Capture published messages
        messages = []

        def capture(channel, data):
            messages.append((channel, data))

        # Register callback for closure reports (exact channel)
        event_bus.register_callback("closure.subsystem", capture)
        event_bus.register_callback("closure.organ", capture)
        event_bus.register_callback("closure.organism", capture)

        # Run coordinator
        coordinator.step(5)

        # Should have published closure reports
        assert len(messages) > 0

    def test_coordinator_statistics(self, holon_registry, simple_hierarchy):
        """ClosureCoordinator aggregates statistics across scales."""
        coordinator = ClosureCoordinator(holon_registry)

        # Run several steps to trigger different scales
        coordinator.step(5)   # Subsystems
        coordinator.step(8)   # Organs
        coordinator.step(13)  # Organisms

        stats = coordinator.get_statistics()

        assert stats["total_steps_coordinated"] == 3
        assert "subsystem_closure" in stats
        assert "organ_closure" in stats
        assert "organism_closure" in stats

    def test_coordinator_advisory_never_blocks(
        self, holon_registry, simple_hierarchy
    ):
        """ClosureCoordinator is advisory — never blocks or modifies state."""
        coordinator = ClosureCoordinator(holon_registry)

        # Degrade all holons
        for holon_id, entry in holon_registry._holons.items():
            proxy = holon_registry.get_proxy(holon_id)
            proxy._health = 0.1

        # Coordinator still completes without exceptions
        reports = coordinator.step(13)

        # All reports should be advisory
        for report in reports:
            assert report.scale in ["subsystem", "organ", "organism"]
            assert isinstance(report, ClosureReport)


# =====================================================================
# Integration tests: Closure Report Details
# =====================================================================


class TestClosureReport:
    """Tests for ClosureReport dataclass."""

    def test_closure_report_structure(self):
        """ClosureReport has all required fields."""
        report = ClosureReport(
            scale="subsystem",
            monitored_id="test-subsystem",
            healthy_children=2,
            degraded_children=1,
            children_healed=0,
            restructure_advice="Monitor closely",
        )

        assert report.scale == "subsystem"
        assert report.monitored_id == "test-subsystem"
        assert report.healthy_children == 2
        assert report.degraded_children == 1
        assert report.children_healed == 0
        assert report.restructure_advice == "Monitor closely"

    def test_closure_report_metadata(self):
        """ClosureReport stores and retrieves metadata."""
        metadata = {"custom_key": "custom_value", "step": 42}
        report = ClosureReport(
            scale="organ",
            monitored_id="test-organ",
            metadata=metadata,
        )

        assert report.metadata["custom_key"] == "custom_value"
        assert report.metadata["step"] == 42

    def test_closure_report_timestamp(self):
        """ClosureReport auto-generates timestamp."""
        report1 = ClosureReport(
            scale="organism",
            monitored_id="mae",
        )
        report2 = ClosureReport(
            scale="organism",
            monitored_id="mae",
        )

        assert report1.timestamp > 0
        assert report2.timestamp > 0
        # Timestamps should be close (within reasonable tolerance)
        assert abs(report2.timestamp - report1.timestamp) < 1.0
