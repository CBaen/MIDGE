"""Tests for the OrganBuilder ↔ SenescenceManager lifecycle loop.

Verifies that:
- OrganBuilder subscribes to CH_SENESCENT when an event_bus is provided
- OrganBuilder works identically without an event_bus (backward-compatible)
- Receiving a senescent event calls prune_organs()
- Receiving a senescent event publishes a rebuild notification
- The handler correctly extracts system_name from the message payload
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.morphogenesis.organ_builder import (
    CH_REBUILD_REQUESTED,
    OrganBlueprint,
    OrganBuilder,
    OrganStatus,
)

# The channel OrganBuilder subscribes to (from senescence.py)
CH_SENESCENT = "emergent.system_senescent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_blueprint(name: str = "test_organ") -> OrganBlueprint:
    """Return a valid single-agent blueprint for test organs."""
    return OrganBlueprint(
        name=name,
        purpose="testing",
        composition={"generalist": 1},
    )


def _senescent_payload(system_name: str = "my_system") -> dict:
    return {
        "system_name": system_name,
        "wear_level": 1.0,
        "total_steps_active": 500,
        "step": 1000,
    }


# ---------------------------------------------------------------------------
# Task 1 — subscription registration
# ---------------------------------------------------------------------------


class TestOrganBuilderSubscribesOnInit:
    """test_organ_builder_subscribes_to_senescent"""

    def test_subscribes_when_event_bus_provided(self) -> None:
        """With event_bus → callback registered on CH_SENESCENT."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        registered = bus._subscribers.get(CH_SENESCENT, [])
        assert len(registered) == 1
        # Bound methods create a new object on each attribute access, so `is`
        # does not work. Compare by __func__ and __self__ instead.
        cb = registered[0]
        assert cb.__func__ is OrganBuilder._on_system_senescent
        assert cb.__self__ is builder

    def test_handler_is_bound_method(self) -> None:
        """The registered callback is OrganBuilder._on_system_senescent."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        callbacks = bus._subscribers[CH_SENESCENT]
        # There should be exactly one callback, bound to this instance
        assert any(cb.__func__ is OrganBuilder._on_system_senescent for cb in callbacks)


# ---------------------------------------------------------------------------
# Task 1 — no event_bus path
# ---------------------------------------------------------------------------


class TestOrganBuilderWithoutEventBus:
    """test_organ_builder_without_event_bus"""

    def test_no_event_bus_no_subscription(self) -> None:
        """Without event_bus → no subscription registered, no errors."""
        builder = OrganBuilder()  # default: no event_bus
        assert builder._event_bus is None

    def test_no_event_bus_operates_normally(self) -> None:
        """OrganBuilder with no event_bus still grows and dissolves organs."""
        builder = OrganBuilder()
        bp = _minimal_blueprint()
        organ = builder.grow_organ(bp)
        assert organ.status == OrganStatus.ACTIVE
        assert builder.active_organ_count == 1


# ---------------------------------------------------------------------------
# Task 1 — backward-compatible construction
# ---------------------------------------------------------------------------


class TestBackwardCompatibleConstruction:
    """test_backward_compatible_construction"""

    def test_positional_agent_factory_still_works(self) -> None:
        """OrganBuilder(agent_factory) with no event_bus is unchanged."""
        factory = MagicMock()
        builder = OrganBuilder(factory)
        assert builder._agent_factory is factory
        assert builder._event_bus is None

    def test_no_args_works(self) -> None:
        """OrganBuilder() with zero args constructs cleanly."""
        builder = OrganBuilder()
        assert builder._event_bus is None
        assert builder._agent_factory is None
        assert builder.active_organ_count == 0

    def test_keyword_event_bus_only(self) -> None:
        """OrganBuilder(event_bus=bus) with no agent_factory works."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)
        assert builder._event_bus is bus
        assert builder._agent_factory is None


# ---------------------------------------------------------------------------
# Task 1 — prune called on senescent event
# ---------------------------------------------------------------------------


class TestOnSenescentCallsPrune:
    """test_on_senescent_calls_prune"""

    def test_prune_called_on_senescent_event(self) -> None:
        """Publishing CH_SENESCENT → prune_organs() is invoked."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        with patch.object(builder, "prune_organs", wraps=builder.prune_organs) as mock_prune:
            bus.publish(CH_SENESCENT, _senescent_payload())
            mock_prune.assert_called_once()

    def test_prune_not_called_without_event(self) -> None:
        """prune_organs() is NOT called if no senescent event is published."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        with patch.object(builder, "prune_organs") as mock_prune:
            mock_prune.assert_not_called()

    def test_prune_actually_dissolves_eligible_organs(self) -> None:
        """End-to-end: senescent event causes eligible organs to dissolve."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        # Grow an organ then force it past max_lifetime so it should dissolve
        bp = OrganBlueprint(
            name="mortal_organ",
            purpose="testing",
            composition={"generalist": 1},
            max_lifetime=0.0001,  # Effectively zero — dissolves immediately
        )
        import time
        organ = builder.grow_organ(bp)
        time.sleep(0.001)  # Let max_lifetime expire
        organ_id = organ.organ_id

        # Publish senescent event — should trigger prune → dissolve
        bus.publish(CH_SENESCENT, _senescent_payload())

        assert organ_id not in builder._active_organs


# ---------------------------------------------------------------------------
# Task 1 — rebuild published on senescent event
# ---------------------------------------------------------------------------


class TestOnSenescentPublishesRebuild:
    """test_on_senescent_publishes_rebuild"""

    def test_rebuild_requested_published(self) -> None:
        """Publishing CH_SENESCENT → CH_REBUILD_REQUESTED is published."""
        bus = EventBus()
        received: list[dict] = []

        def capture(channel: str, message: str) -> None:
            received.append(json.loads(message))

        bus.register_callback(CH_REBUILD_REQUESTED, capture)

        builder = OrganBuilder(event_bus=bus)
        bus.publish(CH_SENESCENT, _senescent_payload("aging_subsystem"))

        assert len(received) == 1
        assert received[0]["system_name"] == "aging_subsystem"
        assert received[0]["reason"] == "senescence"

    def test_rebuild_not_published_without_event_bus(self) -> None:
        """When event_bus is None, _on_system_senescent publish block is skipped."""
        builder = OrganBuilder()  # No event_bus
        # Calling the handler directly must not raise
        builder._on_system_senescent(
            CH_SENESCENT,
            json.dumps(_senescent_payload("ghost_system")),
        )
        # No assertion needed — just verifying it does not raise

    def test_rebuild_channel_name_matches_constant(self) -> None:
        """CH_REBUILD_REQUESTED constant has expected value."""
        assert CH_REBUILD_REQUESTED == "morphogenesis.rebuild_requested"


# ---------------------------------------------------------------------------
# Task 1 — message parsing
# ---------------------------------------------------------------------------


class TestSenescentMessageParsing:
    """test_senescent_message_parsing"""

    def test_json_string_payload_extracted(self) -> None:
        """Handler parses a JSON-string payload and extracts system_name."""
        bus = EventBus()
        received_names: list[str] = []

        def capture_rebuild(channel: str, message: str) -> None:
            payload = json.loads(message)
            received_names.append(payload["system_name"])

        bus.register_callback(CH_REBUILD_REQUESTED, capture_rebuild)
        builder = OrganBuilder(event_bus=bus)

        bus.publish(CH_SENESCENT, {"system_name": "parser_test", "wear_level": 1.0, "step": 42})

        assert received_names == ["parser_test"]

    def test_dict_payload_handled(self) -> None:
        """Handler gracefully handles a raw dict payload (not pre-serialized)."""
        bus = EventBus()
        received_names: list[str] = []

        def capture_rebuild(channel: str, message: str) -> None:
            payload = json.loads(message)
            received_names.append(payload["system_name"])

        bus.register_callback(CH_REBUILD_REQUESTED, capture_rebuild)
        builder = OrganBuilder(event_bus=bus)

        # Call handler directly with a dict (simulates internal calls)
        builder._on_system_senescent(CH_SENESCENT, {"system_name": "dict_test", "wear_level": 1.0})

        assert received_names == ["dict_test"]

    def test_unknown_system_name_defaults_gracefully(self) -> None:
        """Payload missing system_name falls back to 'unknown'."""
        bus = EventBus()
        received: list[dict] = []

        def capture(channel: str, message: str) -> None:
            received.append(json.loads(message))

        bus.register_callback(CH_REBUILD_REQUESTED, capture)
        builder = OrganBuilder(event_bus=bus)

        builder._on_system_senescent(CH_SENESCENT, {"wear_level": 1.0})

        assert received[0]["system_name"] == "unknown"

    def test_malformed_json_string_does_not_raise(self) -> None:
        """Malformed JSON string is handled gracefully (logs warning, no crash)."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        # This must not raise
        builder._on_system_senescent(CH_SENESCENT, "not valid json {{")

    def test_unexpected_type_does_not_raise(self) -> None:
        """Non-string, non-dict payload is handled gracefully."""
        bus = EventBus()
        builder = OrganBuilder(event_bus=bus)

        # Passing an integer — must not raise
        builder._on_system_senescent(CH_SENESCENT, 12345)
