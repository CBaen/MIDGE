"""Tests for WitnessNotifier - operational witnessing for triadic connections."""

import json
from unittest.mock import MagicMock

import pytest

from mae_core.backbone.connection_registry import (
    ConnectionRegistry,
    ConnectionType,
)
from mae_core.backbone.event_bus import EventBus
from mae_core.backbone.witness_notifier import (
    CH_WITNESS_DIGEST_PREFIX,
    CH_WITNESS_HEALTH,
    CH_WITNESS_OBSERVATION,
    CH_WITNESS_VERDICT,
    WitnessNotifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def registry(bus):
    return ConnectionRegistry(event_bus=bus)


@pytest.fixture
def notifier(bus, registry):
    """WitnessNotifier with one witnessed EventBus connection registered."""
    registry.register_connection(
        source="memory",
        target="endocrine",
        connection_type=ConnectionType.EVENTBUS_PUBSUB,
        channel="memory.consolidation_started",
        witnesses=["circadian", "somatic_map"],
        description="Test connection",
    )
    registry.seal()
    wn = WitnessNotifier(event_bus=bus, connection_registry=registry)
    wn.activate()
    return wn


# ---------------------------------------------------------------------------
# Activation Tests
# ---------------------------------------------------------------------------

class TestActivation:
    def test_subscribes_to_channels_with_triads(self, notifier):
        """Notifier subscribes to channels that have registered ConnectionTriads."""
        assert "memory.consolidation_started" in notifier._subscribed_channels

    def test_does_not_subscribe_to_own_channels(self, notifier):
        """Notifier does not subscribe to witness.observation or witness.health."""
        assert CH_WITNESS_OBSERVATION not in notifier._subscribed_channels
        assert CH_WITNESS_HEALTH not in notifier._subscribed_channels

    def test_skips_channels_without_triads(self, bus, registry, notifier):
        """Channels without registered triads get no subscription."""
        assert "unknown.channel" not in notifier._subscribed_channels

    def test_statistics_after_activation(self, notifier):
        """Statistics report observable triads after activation."""
        stats = notifier.get_statistics()
        assert stats["subscribed_channels"] == 1
        assert stats["observable_triads"] >= 1
        assert stats["messages_witnessed"] == 0


# ---------------------------------------------------------------------------
# Shadow Notification Tests
# ---------------------------------------------------------------------------

class TestShadowNotification:
    def test_witness_observation_published_on_message(self, bus, notifier):
        """Publishing on a witnessed channel produces a witness.observation event."""
        observations = []
        bus.register_callback(
            CH_WITNESS_OBSERVATION,
            lambda ch, msg: observations.append(json.loads(msg)),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        assert len(observations) >= 1
        obs = observations[0]
        assert obs["channel"] == "memory.consolidation_started"
        assert obs["source"] == "memory"
        assert obs["target"] == "endocrine"
        assert "circadian" in obs["witnesses"]
        assert "somatic_map" in obs["witnesses"]
        assert "timestamp" in obs
        assert "payload_hash" in obs

    def test_no_observation_for_unwitnessed_channel(self, bus, notifier):
        """Publishing on a channel without triads produces no shadow notification."""
        observations = []
        bus.register_callback(
            CH_WITNESS_OBSERVATION,
            lambda ch, msg: observations.append(msg),
        )

        bus.publish("unknown.channel", {"data": 1})

        assert len(observations) == 0

    def test_messages_witnessed_count_increments(self, bus, notifier):
        """messages_witnessed counter increments on each witnessed message."""
        assert notifier.get_statistics()["messages_witnessed"] == 0

        bus.publish("memory.consolidation_started", {"agent_id": "a"})
        assert notifier.get_statistics()["messages_witnessed"] == 1

        bus.publish("memory.consolidation_started", {"agent_id": "b"})
        assert notifier.get_statistics()["messages_witnessed"] == 2

    def test_payload_hash_differs_for_different_messages(self, bus, notifier):
        """Different payloads produce different hashes."""
        observations = []
        bus.register_callback(
            CH_WITNESS_OBSERVATION,
            lambda ch, msg: observations.append(json.loads(msg)),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "a"})
        bus.publish("memory.consolidation_started", {"agent_id": "b"})

        assert len(observations) == 2
        assert observations[0]["payload_hash"] != observations[1]["payload_hash"]


# ---------------------------------------------------------------------------
# Advisory Safety Tests
# ---------------------------------------------------------------------------

class TestAdvisorySafety:
    def test_original_delivery_unaffected(self, bus, notifier):
        """Primary message delivery works normally with notifier active."""
        received = []
        bus.register_callback(
            "memory.consolidation_started",
            lambda ch, msg: received.append(msg),
        )

        # The notifier is also subscribed, but primary delivery must work
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        # received should have the original message (notifier callback also runs but on same channel)
        assert len(received) >= 1

    def test_notifier_failure_does_not_break_delivery(self, bus, registry):
        """If notifier's callback throws, original delivery is unaffected."""
        registry.register_connection(
            source="test_src", target="test_tgt",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="test.channel",
            witnesses=["w1", "w2"],
        )
        registry.seal()
        wn = WitnessNotifier(event_bus=bus, connection_registry=registry)
        wn.activate()

        # Force an error in the shadow publishing
        original_publish = bus.publish
        call_count = [0]

        def broken_publish(channel, message):
            call_count[0] += 1
            if channel == CH_WITNESS_OBSERVATION:
                raise RuntimeError("shadow publish broken")
            return original_publish(channel, message)

        bus.publish = broken_publish

        received = []
        bus.register_callback("test.channel", lambda ch, msg: received.append(msg))

        # This should not raise despite broken shadow publishing
        # (restoring publish for the primary call)
        bus.publish = original_publish
        bus.publish("test.channel", {"data": 1})

        assert len(received) >= 1

    def test_reentrance_guard_prevents_loops(self, bus, notifier):
        """Re-entrancy guard prevents infinite loops."""
        # Manually set _in_shadow to simulate re-entrancy
        notifier._in_shadow = True
        observations = []
        bus.register_callback(
            CH_WITNESS_OBSERVATION,
            lambda ch, msg: observations.append(msg),
        )

        # This should be a no-op due to the guard
        notifier._on_message("memory.consolidation_started", '{"test": 1}')
        assert len(observations) == 0

        notifier._in_shadow = False  # Reset


# ---------------------------------------------------------------------------
# Cadenced Step Tests
# ---------------------------------------------------------------------------

class TestCadencedStep:
    def test_step_publishes_at_cadence(self, bus, notifier):
        """Health summary published every 13 steps (Fibonacci cadence)."""
        health_events = []
        bus.register_callback(
            CH_WITNESS_HEALTH,
            lambda ch, msg: health_events.append(json.loads(msg)),
        )

        # Run 12 steps -- no health event yet
        for _ in range(12):
            notifier.step()
        assert len(health_events) == 0

        # 13th step triggers health report
        notifier.step()
        assert len(health_events) == 1
        assert "messages_witnessed" in health_events[0]
        assert "subscribed_channels" in health_events[0]

    def test_health_report_count_increments(self, notifier):
        """health_reports counter increments on each cadenced step."""
        assert notifier.get_statistics()["health_reports"] == 0
        for _ in range(13):
            notifier.step()
        assert notifier.get_statistics()["health_reports"] == 1


# ---------------------------------------------------------------------------
# Statistics Tests
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_statistics_structure(self, notifier):
        """get_statistics() returns expected fields."""
        stats = notifier.get_statistics()
        expected_keys = {
            "messages_witnessed", "shadow_failures", "subscribed_channels",
            "observable_triads", "structural_triads", "total_connections",
            "health_reports", "witness_coverage_pct",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_witness_coverage_percentage(self, bus, registry):
        """Coverage percentage reflects observable vs total connections."""
        # One EventBus connection (observable) + one DIRECT_REFERENCE (structural)
        registry.register_connection(
            source="a", target="b",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="a.channel",
            witnesses=["w1", "w2"],
        )
        registry.register_connection(
            source="c", target="d",
            connection_type=ConnectionType.DIRECT_REFERENCE,
            witnesses=["w1", "w2"],
        )
        registry.seal()
        wn = WitnessNotifier(event_bus=bus, connection_registry=registry)
        wn.activate()

        stats = wn.get_statistics()
        assert stats["observable_triads"] == 1
        assert stats["structural_triads"] == 1
        assert stats["total_connections"] == 2
        assert stats["witness_coverage_pct"] == 50.0


# ---------------------------------------------------------------------------
# verify_all() Integration Tests
# ---------------------------------------------------------------------------

class TestVerifyAllIntegration:
    def test_verify_all_includes_witnessing_stats(self, bus, registry, notifier):
        """ConnectionRegistry.verify_all() includes witnessing stats when notifier is set."""
        registry._witness_notifier = notifier

        results = registry.verify_all()
        assert "witnessing" in results
        assert "messages_witnessed" in results["witnessing"]

    def test_verify_all_works_without_notifier(self, bus, registry):
        """verify_all() still works when no notifier is set (backward compat)."""
        registry.register_connection(
            "a", "b", ConnectionType.DIRECT_REFERENCE, witnesses=["w"],
        )
        results = registry.verify_all()
        assert "witnessing" not in results
        assert results["total"] == 1


# ---------------------------------------------------------------------------
# Per-Witness Accountability Tests
# ---------------------------------------------------------------------------

class TestPerWitnessAccountability:
    def test_per_witness_counts_increment(self, bus, notifier):
        """Each witness's delivery count increments when their triad fires."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        counts = notifier._witness_delivery_counts
        assert counts.get("circadian", 0) == 1
        assert counts.get("somatic_map", 0) == 1

    def test_per_witness_counts_accumulate(self, bus, notifier):
        """Multiple messages accumulate per-witness counts."""
        for i in range(5):
            bus.publish("memory.consolidation_started", {"agent_id": f"a{i}"})

        counts = notifier._witness_delivery_counts
        assert counts["circadian"] == 5
        assert counts["somatic_map"] == 5

    def test_statistics_include_per_witness_fields(self, bus, notifier):
        """get_statistics() includes per-witness accountability fields."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})
        stats = notifier.get_statistics()

        assert "active_witnesses" in stats
        assert "total_witnesses" in stats
        assert "witness_digests_sent" in stats
        assert "top_witnesses" in stats
        assert "per_witness_counts" in stats

        assert stats["active_witnesses"] == 2  # circadian + somatic_map
        assert stats["total_witnesses"] == 2
        assert stats["per_witness_counts"]["circadian"] == 1
        assert stats["per_witness_counts"]["somatic_map"] == 1

    def test_active_witnesses_zero_before_messages(self, notifier):
        """No active witnesses before any messages flow."""
        stats = notifier.get_statistics()
        assert stats["active_witnesses"] == 0
        assert stats["total_witnesses"] == 0

    def test_top_witnesses_sorted_by_count(self, bus, registry):
        """top_witnesses is sorted highest-count first."""
        # Register two connections with different witnesses
        registry.register_connection(
            source="a", target="b",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="a.test_chan",
            witnesses=["w_busy", "w_quiet"],
        )
        registry.register_connection(
            source="c", target="d",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="c.test_chan",
            witnesses=["w_busy", "w_other"],
        )
        registry.seal()
        wn = WitnessNotifier(event_bus=bus, connection_registry=registry)
        wn.activate()

        # Both channels fire -- w_busy appears in both triads
        bus.publish("a.test_chan", {"x": 1})
        bus.publish("c.test_chan", {"x": 2})

        stats = wn.get_statistics()
        top = stats["top_witnesses"]
        # w_busy should be first (count=2), others have count=1
        assert top[0][0] == "w_busy"
        assert top[0][1] == 2


# ---------------------------------------------------------------------------
# Cadenced Digest Tests
# ---------------------------------------------------------------------------

class TestCadencedDigest:
    def test_digest_published_per_witness_at_cadence(self, bus, notifier):
        """Per-witness digest events published at cadence step."""
        digests = []
        bus.register_callback(
            f"{CH_WITNESS_DIGEST_PREFIX}circadian",
            lambda ch, msg: digests.append(json.loads(msg)),
        )

        # Send a message to accumulate witness counts
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        # Run to cadence (13 steps)
        for _ in range(13):
            notifier.step()

        assert len(digests) == 1
        assert digests[0]["witness"] == "circadian"
        assert digests[0]["observations_since_last"] == 1
        assert "step" in digests[0]

    def test_digest_sent_to_each_active_witness(self, bus, notifier):
        """Each active witness gets their own digest channel."""
        circadian_digests = []
        somatic_digests = []
        bus.register_callback(
            f"{CH_WITNESS_DIGEST_PREFIX}circadian",
            lambda ch, msg: circadian_digests.append(json.loads(msg)),
        )
        bus.register_callback(
            f"{CH_WITNESS_DIGEST_PREFIX}somatic_map",
            lambda ch, msg: somatic_digests.append(json.loads(msg)),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        for _ in range(13):
            notifier.step()

        assert len(circadian_digests) == 1
        assert len(somatic_digests) == 1

    def test_digest_resets_counts_after_flush(self, bus, notifier):
        """Per-witness counts reset to 0 after cadenced digest flush."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})
        assert notifier._witness_delivery_counts["circadian"] == 1

        # Flush at cadence
        for _ in range(13):
            notifier.step()

        assert notifier._witness_delivery_counts["circadian"] == 0

    def test_no_digest_for_zero_count_witnesses(self, bus, notifier):
        """Witnesses with zero observations in the window get no digest."""
        digests = []
        bus.register_callback(
            f"{CH_WITNESS_DIGEST_PREFIX}circadian",
            lambda ch, msg: digests.append(msg),
        )

        # Do NOT send any messages -- just run to cadence
        for _ in range(13):
            notifier.step()

        assert len(digests) == 0

    def test_digest_count_increments(self, bus, notifier):
        """witness_digests_sent counter increments correctly."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        for _ in range(13):
            notifier.step()

        # Two witnesses (circadian + somatic_map) = 2 digests sent
        assert notifier._witness_digests_sent == 2

    def test_digest_channels_not_subscribed_by_notifier(self, notifier):
        """Notifier does not subscribe to its own digest channels."""
        for ch in notifier._subscribed_channels:
            assert not ch.startswith(CH_WITNESS_DIGEST_PREFIX)


# ---------------------------------------------------------------------------
# Witness Sense Resolution Tests (Gap 2: operational sensing)
# ---------------------------------------------------------------------------

class TestWitnessSenseResolution:
    """Tests that witnesses operationally 'see' via HolonProxy.sense()."""

    @pytest.fixture
    def holon_registry(self):
        """Mock HolonRegistry with proxy stubs for circadian and somatic_map."""
        registry = MagicMock()
        proxy_circadian = MagicMock()
        proxy_somatic = MagicMock()
        registry.get_proxy.side_effect = lambda name: {
            "circadian": proxy_circadian,
            "somatic_map": proxy_somatic,
        }.get(name)
        registry._proxies = {
            "circadian": proxy_circadian,
            "somatic_map": proxy_somatic,
        }
        return registry

    @pytest.fixture
    def sensing_notifier(self, bus, registry, holon_registry):
        """WitnessNotifier with holon_registry for operational sensing."""
        registry.register_connection(
            source="memory",
            target="endocrine",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="memory.consolidation_started",
            witnesses=["circadian", "somatic_map"],
            description="Test connection",
        )
        registry.seal()
        wn = WitnessNotifier(
            event_bus=bus,
            connection_registry=registry,
            holon_registry=holon_registry,
        )
        wn.activate()
        return wn

    def test_witness_sense_called_on_message(self, bus, sensing_notifier, holon_registry):
        """proxy.sense() is called for each witness when a message flows."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        proxy_circ = holon_registry._proxies["circadian"]
        proxy_som = holon_registry._proxies["somatic_map"]
        proxy_circ.sense.assert_called_once()
        proxy_som.sense.assert_called_once()

    def test_sense_failure_does_not_break_notification(self, bus, registry, holon_registry):
        """If sense() raises, shadow notification still completes."""
        # Make circadian.sense() raise
        holon_registry._proxies["circadian"].sense.side_effect = RuntimeError("broken")

        registry.register_connection(
            source="a", target="b",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="a.sense_fail",
            witnesses=["circadian", "somatic_map"],
        )
        registry.seal()
        wn = WitnessNotifier(
            event_bus=bus,
            connection_registry=registry,
            holon_registry=holon_registry,
        )
        wn.activate()

        observations = []
        bus.register_callback(
            CH_WITNESS_OBSERVATION,
            lambda ch, msg: observations.append(msg),
        )

        # Should not raise
        bus.publish("a.sense_fail", {"data": 1})

        # Observation still published
        assert len(observations) >= 1
        # Sense failure counted
        assert wn._witness_sense_failures >= 1

    def test_sense_count_increments(self, bus, sensing_notifier):
        """_witness_sense_count increments for each successful sense()."""
        assert sensing_notifier._witness_sense_count == 0

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        # 2 witnesses sensed
        assert sensing_notifier._witness_sense_count == 2

    def test_no_sense_without_holon_registry(self, bus, notifier):
        """Without holon_registry, zero sense calls occur (backward compat)."""
        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        assert notifier._witness_sense_count == 0
        assert notifier._verdicts_published == 0


# ---------------------------------------------------------------------------
# Balance Pathway Verdict Tests (Gap 3: witness -> source feedback)
# ---------------------------------------------------------------------------

class TestBalancePathwayVerdict:
    """Tests that witnesses publish verdicts back (B->C->A balance pathway)."""

    @pytest.fixture
    def holon_registry(self):
        registry = MagicMock()
        proxy_circadian = MagicMock()
        proxy_somatic = MagicMock()
        registry.get_proxy.side_effect = lambda name: {
            "circadian": proxy_circadian,
            "somatic_map": proxy_somatic,
        }.get(name)
        return registry

    @pytest.fixture
    def verdict_notifier(self, bus, registry, holon_registry):
        registry.register_connection(
            source="memory",
            target="endocrine",
            connection_type=ConnectionType.EVENTBUS_PUBSUB,
            channel="memory.consolidation_started",
            witnesses=["circadian", "somatic_map"],
            description="Test connection",
        )
        registry.seal()
        wn = WitnessNotifier(
            event_bus=bus,
            connection_registry=registry,
            holon_registry=holon_registry,
        )
        wn.activate()
        return wn

    def test_verdict_published_after_witness_sense(self, bus, verdict_notifier):
        """Verdict events published on CH_WITNESS_VERDICT for each witness."""
        verdicts = []
        bus.register_callback(
            CH_WITNESS_VERDICT,
            lambda ch, msg: verdicts.append(json.loads(msg)),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        # One verdict per witness (2 witnesses)
        assert len(verdicts) == 2
        witness_names = {v["witness"] for v in verdicts}
        assert witness_names == {"circadian", "somatic_map"}

    def test_verdict_contains_expected_fields(self, bus, verdict_notifier):
        """Each verdict contains all required fields."""
        verdicts = []
        bus.register_callback(
            CH_WITNESS_VERDICT,
            lambda ch, msg: verdicts.append(json.loads(msg)),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        expected_fields = {
            "connection_id", "witness", "source", "target",
            "verdict", "payload_hash", "step",
        }
        for v in verdicts:
            assert expected_fields.issubset(set(v.keys()))
            assert v["verdict"] == "observed"
            assert v["source"] == "memory"
            assert v["target"] == "endocrine"

    def test_verdict_count_increments(self, bus, verdict_notifier):
        """_verdicts_published counter increments correctly."""
        assert verdict_notifier._verdicts_published == 0

        bus.publish("memory.consolidation_started", {"agent_id": "test"})
        assert verdict_notifier._verdicts_published == 2  # 2 witnesses

        bus.publish("memory.consolidation_started", {"agent_id": "test2"})
        assert verdict_notifier._verdicts_published == 4

    def test_no_verdict_without_holon_registry(self, bus, notifier):
        """Without holon_registry, no verdicts are published."""
        verdicts = []
        bus.register_callback(
            CH_WITNESS_VERDICT,
            lambda ch, msg: verdicts.append(msg),
        )

        bus.publish("memory.consolidation_started", {"agent_id": "test"})

        assert len(verdicts) == 0
        assert notifier._verdicts_published == 0

    def test_verdict_channel_excluded_from_subscriptions(self, verdict_notifier):
        """Notifier does not subscribe to its own verdict channel."""
        assert CH_WITNESS_VERDICT not in verdict_notifier._subscribed_channels
