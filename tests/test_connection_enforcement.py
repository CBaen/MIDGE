"""Tests for Blocking Triad Enforcement — enforcement modes on ConnectionRegistry."""

import logging
from unittest.mock import MagicMock

import pytest

from mae_core.backbone.connection_registry import (
    CH_CONNECTION_BLOCKED,
    CH_CONNECTION_SEALED,
    ConnectionCriticality,
    ConnectionRegistry,
    ConnectionType,
    EnforcementMode,
)
from mae_core.backbone.event_bus import EventBus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def somatic_map():
    """Minimal SomaticMap mock with get_system_info."""
    sm = MagicMock()
    sm.get_system_info.return_value = None  # Default: system not found
    return sm


@pytest.fixture
def registry(bus, somatic_map):
    """Default registry in ADVISORY mode (unsealed)."""
    return ConnectionRegistry(
        event_bus=bus,
        somatic_map=somatic_map,
        verify_interval=25,
    )


def _register_healthy(registry, source="sys_a", target="sys_b", witness="enforcer"):
    """Helper to register a healthy, witnessed connection."""
    return registry.register_connection(
        source=source,
        target=target,
        connection_type=ConnectionType.DIRECT_REFERENCE,
        witness=witness,
    )


# ---------------------------------------------------------------------------
# EnforcementMode Tests
# ---------------------------------------------------------------------------

class TestEnforcementMode:
    def test_starts_permissive_before_seal(self, registry):
        """Registry starts in PERMISSIVE mode before seal()."""
        assert registry.enforcement_mode == EnforcementMode.PERMISSIVE
        assert not registry.sealed

    def test_seal_transitions_to_configured_mode(self, registry):
        """seal() transitions from PERMISSIVE to the configured post-seal mode."""
        registry.seal()
        assert registry.enforcement_mode == EnforcementMode.ADVISORY
        assert registry.sealed

    def test_seal_is_idempotent(self, registry):
        """Calling seal() twice is a no-op."""
        registry.seal()
        registry.set_enforcement_mode(EnforcementMode.BLOCKING)
        registry.seal()  # Should not reset to ADVISORY
        assert registry.enforcement_mode == EnforcementMode.BLOCKING

    def test_runtime_mode_switching(self, registry):
        """set_enforcement_mode() changes the active mode."""
        registry.seal()
        assert registry.enforcement_mode == EnforcementMode.ADVISORY

        registry.set_enforcement_mode(EnforcementMode.BLOCKING)
        assert registry.enforcement_mode == EnforcementMode.BLOCKING

        registry.set_enforcement_mode(EnforcementMode.PERMISSIVE)
        assert registry.enforcement_mode == EnforcementMode.PERMISSIVE

    def test_default_post_seal_is_advisory(self, bus, somatic_map):
        """Default enforcement mode after seal is ADVISORY."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.seal()
        assert reg.enforcement_mode == EnforcementMode.ADVISORY

    def test_blocking_configurable_at_init(self, bus, somatic_map):
        """Can configure BLOCKING mode at construction time."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()
        assert reg.enforcement_mode == EnforcementMode.BLOCKING


# ---------------------------------------------------------------------------
# is_connection_allowed Tests
# ---------------------------------------------------------------------------

class TestIsConnectionAllowed:
    def test_permissive_allows_everything(self, registry):
        """Before seal, everything is allowed."""
        allowed, reason = registry.is_connection_allowed("unknown_a", "unknown_b")
        assert allowed is True
        assert reason == "permissive"

    def test_advisory_allows_unregistered(self, registry):
        """ADVISORY mode allows unregistered connections (with log)."""
        registry.seal()
        allowed, reason = registry.is_connection_allowed("unknown_a", "unknown_b")
        assert allowed is True
        assert reason == "unregistered_advisory"

    def test_advisory_allows_unhealthy(self, registry):
        """ADVISORY mode allows unhealthy connections (with log)."""
        triad = _register_healthy(registry)
        triad.healthy = False
        registry.seal()

        allowed, reason = registry.is_connection_allowed("sys_a", "sys_b")
        assert allowed is True
        assert reason == "unhealthy_advisory"

    def test_blocking_rejects_unregistered(self, bus, somatic_map):
        """BLOCKING mode rejects unregistered connections."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()

        allowed, reason = reg.is_connection_allowed("unknown_a", "unknown_b")
        assert allowed is False
        assert reason == "unregistered"

    def test_blocking_rejects_unhealthy(self, bus, somatic_map):
        """BLOCKING mode rejects unhealthy connections."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg)
        reg._connections["sys_a->sys_b"].healthy = False
        reg.seal()

        allowed, reason = reg.is_connection_allowed("sys_a", "sys_b")
        assert allowed is False
        assert reason == "unhealthy"

    def test_blocking_rejects_bare_dyad(self, bus, somatic_map):
        """BLOCKING mode rejects bare dyads (connections without witness)."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        # Register while still PERMISSIVE (before seal), manually remove witness
        triad = _register_healthy(reg, witness="enforcer")
        triad.witnesses.clear()
        reg.seal()

        allowed, reason = reg.is_connection_allowed("sys_a", "sys_b")
        assert allowed is False
        assert reason == "bare_dyad"

    def test_blocking_allows_healthy(self, bus, somatic_map):
        """BLOCKING mode allows healthy, witnessed connections."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg)
        reg.seal()

        allowed, reason = reg.is_connection_allowed("sys_a", "sys_b")
        assert allowed is True
        assert reason == "ok"

    def test_channel_fallback(self, registry):
        """Lookup without channel falls back to base connection ID."""
        _register_healthy(registry)
        registry.seal()

        # Query with channel that doesn't exist — falls back to sys_a->sys_b
        allowed, reason = registry.is_connection_allowed(
            "sys_a", "sys_b", channel="some.channel"
        )
        assert allowed is True
        assert reason == "ok"


# ---------------------------------------------------------------------------
# Registration Enforcement Tests
# ---------------------------------------------------------------------------

class TestRegistrationEnforcement:
    def test_blocking_rejects_bare_dyad_registration(self, bus):
        """BLOCKING mode raises ConnectionError for bare dyad registration."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=None,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()

        # Force _assign_witnesses to return empty (normally returns 2 fallbacks)
        reg._assign_witnesses = lambda s, t, count=2, exclude=None: []

        with pytest.raises(ConnectionError, match="bare dyad"):
            reg.register_connection(
                source="sys_a",
                target="sys_b",
                connection_type=ConnectionType.DIRECT_REFERENCE,
            )

    def test_advisory_allows_bare_dyad_registration(self, bus):
        """ADVISORY mode allows bare dyad registration (logs warning)."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=None)
        reg.seal()

        # Should not raise
        triad = reg.register_connection(
            source="sys_a",
            target="sys_b",
            connection_type=ConnectionType.DIRECT_REFERENCE,
        )
        # Still registered but witness is from fallback "enforcer"
        # (somatic_map=None → _assign_witnesses returns nervous system fallbacks)
        assert triad is not None

    def test_permissive_allows_bare_dyad_registration(self, bus):
        """PERMISSIVE mode allows anything (before seal)."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=None,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        # NOT sealed → still PERMISSIVE

        triad = reg.register_connection(
            source="sys_a",
            target="sys_b",
            connection_type=ConnectionType.DIRECT_REFERENCE,
        )
        assert triad is not None

    def test_blocking_accepts_witnessed_registration(self, bus, somatic_map):
        """BLOCKING mode accepts registration when witness is provided."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()

        triad = reg.register_connection(
            source="sys_a",
            target="sys_b",
            connection_type=ConnectionType.DIRECT_REFERENCE,
            witness="sys_c",
        )
        assert triad.witness == "sys_c"

    def test_existing_registrations_survive_mode_switch(self, registry):
        """Connections registered before mode change stay registered."""
        _register_healthy(registry)
        registry.seal()
        registry.set_enforcement_mode(EnforcementMode.BLOCKING)

        triad = registry.get_connection("sys_a->sys_b")
        assert triad is not None
        assert triad.witness == "enforcer"


# ---------------------------------------------------------------------------
# Bootstrap Grace Period Tests
# ---------------------------------------------------------------------------

class TestBootstrapGracePeriod:
    def test_allowed_before_seal(self, registry):
        """All is_connection_allowed calls return True before seal."""
        allowed, reason = registry.is_connection_allowed("x", "y")
        assert allowed is True
        assert reason == "permissive"

    def test_seal_enables_enforcement(self, bus, somatic_map):
        """After seal, enforcement kicks in."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        # Before seal: permissive
        allowed, _ = reg.is_connection_allowed("x", "y")
        assert allowed is True

        reg.seal()

        # After seal: blocking
        allowed, reason = reg.is_connection_allowed("x", "y")
        assert allowed is False
        assert reason == "unregistered"

    def test_bootstrap_connections_unaffected(self, bus, somatic_map):
        """Connections registered during bootstrap work after seal."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg)
        reg.seal()

        allowed, reason = reg.is_connection_allowed("sys_a", "sys_b")
        assert allowed is True
        assert reason == "ok"

    def test_seal_publishes_event(self, bus, somatic_map):
        """seal() publishes CH_CONNECTION_SEALED event."""
        events = []
        bus.register_callback(CH_CONNECTION_SEALED, lambda ch, msg: events.append(msg))

        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.seal()

        assert len(events) == 1


# ---------------------------------------------------------------------------
# Verification Enforcement Tests
# ---------------------------------------------------------------------------

class TestVerificationEnforcement:
    def test_verify_publishes_blocked_in_blocking_mode(self, bus, somatic_map):
        """verify_all() publishes CH_CONNECTION_BLOCKED for unhealthy in BLOCKING mode."""
        import json as _json

        blocked_events = []
        bus.register_callback(
            CH_CONNECTION_BLOCKED,
            lambda ch, msg: blocked_events.append(_json.loads(msg)),
        )

        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        triad = _register_healthy(reg)
        triad.healthy = False
        reg.seal()

        reg.verify_all()
        assert len(blocked_events) >= 1
        assert blocked_events[0]["reason"] == "unhealthy"

    def test_unhealthy_blocked_via_query(self, bus, somatic_map):
        """Unhealthy connection is blocked via is_connection_allowed after verify."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg)
        reg.seal()

        # Force unhealthy
        reg._connections["sys_a->sys_b"].healthy = False

        allowed, reason = reg.is_connection_allowed("sys_a", "sys_b")
        assert allowed is False
        assert reason == "unhealthy"

    def test_statistics_include_enforcement(self, registry):
        """get_statistics() includes enforcement fields."""
        registry.seal()
        stats = registry.get_statistics()
        assert "enforcement_mode" in stats
        assert stats["enforcement_mode"] == "advisory"
        assert "sealed" in stats
        assert stats["sealed"] is True
        assert "blocked_count" in stats

    def test_blocked_count_increments(self, bus, somatic_map):
        """blocked_count increments on each blocked query."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()

        reg.is_connection_allowed("x", "y")
        reg.is_connection_allowed("a", "b")

        stats = reg.get_statistics()
        assert stats["blocked_count"] == 2


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestEnforcementIntegration:
    def test_full_bootstrap_all_connections_allowed(self, bus, somatic_map):
        """After registering connections and sealing, all are allowed."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg, "sys_a", "sys_b", "enforcer")
        _register_healthy(reg, "sys_c", "sys_d", "watchdog")
        _register_healthy(reg, "sys_e", "sys_f", "auditor")
        reg.seal()

        for src, tgt in [("sys_a", "sys_b"), ("sys_c", "sys_d"), ("sys_e", "sys_f")]:
            allowed, reason = reg.is_connection_allowed(src, tgt)
            assert allowed is True, f"{src}->{tgt} should be allowed, got {reason}"

    def test_unhealthy_blocked_healthy_allowed(self, bus, somatic_map):
        """In BLOCKING: unhealthy blocked, healthy allowed side by side."""
        reg = ConnectionRegistry(
            event_bus=bus,
            somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        _register_healthy(reg, "good_a", "good_b", "enforcer")
        triad = _register_healthy(reg, "bad_a", "bad_b", "watchdog")
        triad.healthy = False
        reg.seal()

        allowed_good, _ = reg.is_connection_allowed("good_a", "good_b")
        allowed_bad, reason_bad = reg.is_connection_allowed("bad_a", "bad_b")

        assert allowed_good is True
        assert allowed_bad is False
        assert reason_bad == "unhealthy"


# ---------------------------------------------------------------------------
# EventBus Advisory Witnessing Tests
# ---------------------------------------------------------------------------

class TestEventBusAdvisoryWitnessing:
    """Tests for EventBus.publish() advisory witnessing via ConnectionRegistry."""

    def test_publish_works_without_registry(self, bus):
        """EventBus delivers messages normally when no registry is set."""
        received = []
        bus.register_callback("test.channel", lambda ch, msg: received.append(msg))
        count = bus.publish("test.channel", {"hello": "world"})
        assert count == 1
        assert len(received) == 1
        assert bus._connection_registry is None

    def test_publish_delivers_with_registry_set(self, bus, somatic_map):
        """Messages still deliver when registry is wired (advisory never blocks)."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.seal()
        bus.set_connection_registry(reg)

        received = []
        bus.register_callback("test.channel", lambda ch, msg: received.append(msg))
        count = bus.publish("test.channel", {"data": 1})

        assert count == 1
        assert len(received) == 1

    def test_advisory_logs_unregistered_channel(self, bus, somatic_map, caplog):
        """Advisory mode logs a warning for unregistered channels."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.seal()
        bus.set_connection_registry(reg)

        with caplog.at_level(logging.WARNING, logger="mae_core.backbone.connection_registry"):
            bus.publish("unknown.channel", {"data": 1})

        assert any("unregistered" in r.message.lower() for r in caplog.records)

    def test_advisory_does_not_log_registered_channel(self, bus, somatic_map, caplog):
        """No advisory warning for a properly registered connection."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.register_connection(
            source="memory",
            target="event_bus",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="memory.consolidation_complete",
            witness="enforcer",
        )
        reg.seal()
        bus.set_connection_registry(reg)

        with caplog.at_level(logging.WARNING, logger="mae_core.backbone.connection_registry"):
            bus.publish("memory.consolidation_complete", {"data": 1})

        advisory_warnings = [
            r for r in caplog.records
            if "unregistered" in r.message.lower()
            and "memory" in r.message.lower()
        ]
        assert len(advisory_warnings) == 0

    def test_publish_survives_registry_error(self, bus):
        """If registry raises, publish still delivers (never breaks hot path)."""
        class BrokenRegistry:
            sealed = True
            def is_connection_allowed(self, *a, **kw):
                raise RuntimeError("registry exploded")

        bus.set_connection_registry(BrokenRegistry())

        received = []
        bus.register_callback("test.channel", lambda ch, msg: received.append(msg))
        count = bus.publish("test.channel", "hello")

        assert count == 1
        assert len(received) == 1

    def test_registry_not_consulted_before_seal(self, bus, somatic_map, caplog):
        """Registry is not consulted if not sealed (bootstrap grace period)."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        # NOT sealed
        bus.set_connection_registry(reg)

        with caplog.at_level(logging.WARNING, logger="mae_core.backbone.connection_registry"):
            bus.publish("unknown.channel", {"data": 1})

        advisory_warnings = [
            r for r in caplog.records
            if "unregistered" in r.message.lower()
        ]
        assert len(advisory_warnings) == 0


class TestEventBusBlockingDrop:
    """Tests for EventBus.publish() dropping messages in BLOCKING mode."""

    def test_blocking_drops_unregistered_message(self, bus, somatic_map):
        """BLOCKING mode drops messages on unregistered channels."""
        reg = ConnectionRegistry(
            event_bus=bus, somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()
        bus.set_connection_registry(reg)

        received = []
        bus.register_callback("unknown.channel", lambda ch, msg: received.append(msg))
        count = bus.publish("unknown.channel", {"data": 1})

        assert count == 0
        assert len(received) == 0
        assert bus._dropped_count == 1

    def test_advisory_never_drops(self, bus, somatic_map):
        """ADVISORY mode never drops messages, even for unregistered channels."""
        reg = ConnectionRegistry(event_bus=bus, somatic_map=somatic_map)
        reg.seal()  # default ADVISORY
        bus.set_connection_registry(reg)

        received = []
        bus.register_callback("unknown.channel", lambda ch, msg: received.append(msg))
        count = bus.publish("unknown.channel", {"data": 1})

        assert count >= 1
        assert len(received) == 1
        assert bus._dropped_count == 0

    def test_blocking_allows_registered_channel(self, bus, somatic_map):
        """BLOCKING mode allows messages on properly registered channels."""
        reg = ConnectionRegistry(
            event_bus=bus, somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.register_connection(
            source="memory", target="event_bus",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="memory.test",
            witness="enforcer",
        )
        reg.seal()
        bus.set_connection_registry(reg)

        received = []
        bus.register_callback("memory.test", lambda ch, msg: received.append(msg))
        count = bus.publish("memory.test", {"data": 1})

        assert count >= 1
        assert len(received) == 1
        assert bus._dropped_count == 0

    def test_dropped_count_in_stats(self, bus, somatic_map):
        """get_stats() includes dropped_count."""
        reg = ConnectionRegistry(
            event_bus=bus, somatic_map=somatic_map,
            enforcement_mode=EnforcementMode.BLOCKING,
        )
        reg.seal()
        bus.set_connection_registry(reg)

        bus.publish("unknown.channel", {"data": 1})
        bus.publish("also.unknown", {"data": 2})

        stats = bus.get_stats()
        assert "dropped_count" in stats
        assert stats["dropped_count"] == 2
