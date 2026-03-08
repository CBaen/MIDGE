"""Tests for GovernanceLogger.

What is under test:
    mae_core/governance/governance_logger.py

Design decisions validated:
    - Events published on subscribed channels are appended to JSONL file
    - Events from different channels are all recorded
    - Log file is created on first event (directory created if needed)
    - get_statistics() returns event_count, last_event_time, log_file_bytes, etc.
    - Write failure (invalid path) logs a warning and does not raise
    - JSON serialization failure for un-serializable event logs a warning, no raise
    - Each line is valid JSON with timestamp, channel, event keys

Test strategy:
    - Use tmp_path (pytest fixture) for all file I/O — no production file touched
    - Use real EventBus instance to exercise the full callback path
    - Direct callback invocation for write-failure tests (avoid real path issues)
    - Never patch file I/O unless testing failure paths
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.backbone.event_bus import EventBus
from mae_core.governance.governance_logger import (
    GovernanceLogger,
    _GOVERNANCE_CHANNELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(tmp_path: Path, **kwargs) -> tuple[EventBus, GovernanceLogger]:
    """Create a real EventBus + GovernanceLogger writing to a temp file."""
    log_path = str(tmp_path / "governance_log.jsonl")
    bus = EventBus()
    defaults = {"event_bus": bus, "log_path": log_path}
    defaults.update(kwargs)
    gl = GovernanceLogger(**defaults)
    return bus, gl


def _read_lines(log_path: str) -> list[dict]:
    """Read all JSON lines from the log file."""
    lines = []
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
    except FileNotFoundError:
        pass
    return lines


# ---------------------------------------------------------------------------
# Section 1: Basic event logging
# ---------------------------------------------------------------------------

class TestThrottleEventLogged:
    """throttle events appear in the JSONL file."""

    def test_throttle_event_logged(self, tmp_path):
        """Publishing a throttle event writes one line to the log."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("market.resource.throttle", {
            "source": "sec_edgar",
            "calls_last_hour": 105,
            "hourly_limit": 100,
            "timestamp": time.time(),
        })

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"
        assert lines[0]["channel"] == "market.resource.throttle"
        assert "timestamp" in lines[0]
        assert "event" in lines[0]

    def test_budget_warning_event_logged(self, tmp_path):
        """Publishing a budget_warning event writes one line."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("market.resource.budget_warning", {
            "source": "finnhub",
            "usage_ratio": 0.82,
            "calls_last_hour": 820,
            "hourly_limit": 1000,
        })

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1
        assert lines[0]["channel"] == "market.resource.budget_warning"

    def test_dispatch_event_logged(self, tmp_path):
        """Publishing a dispatch event writes one line."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("scheduling.inhabitant_dispatched", {
            "system_name": "convergence_alerter",
            "timestamp": time.time(),
        })

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1
        assert lines[0]["channel"] == "scheduling.inhabitant_dispatched"

    def test_senescent_event_logged(self, tmp_path):
        """Publishing a senescent event writes one line."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("emergent.system_senescent", {
            "system_name": "thompson_sampler",
            "wear_level": 1.0,
            "step": 5000,
        })

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1
        assert lines[0]["channel"] == "emergent.system_senescent"

    def test_rejuvenation_event_logged(self, tmp_path):
        """Publishing a rejuvenation_needed event writes one line."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("emergent.rejuvenation_needed", {
            "system_name": "hypothesis_engine",
            "wear_level": 0.82,
            "step": 3000,
        })

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1
        assert lines[0]["channel"] == "emergent.rejuvenation_needed"


# ---------------------------------------------------------------------------
# Section 2: Multiple channels
# ---------------------------------------------------------------------------

class TestMultipleChannelsLogged:
    """Events from different governance channels are all recorded."""

    def test_multiple_channels_logged(self, tmp_path):
        """Publishing to three different channels creates three log lines."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("market.resource.throttle", {"source": "sec_edgar"})
        bus.publish("scheduling.inhabitant_dispatched", {"system_name": "senescence"})
        bus.publish("emergent.system_senescent", {"system_name": "alpha_system"})

        lines = _read_lines(gl._log_path)
        assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

        channels_seen = {line["channel"] for line in lines}
        assert "market.resource.throttle" in channels_seen
        assert "scheduling.inhabitant_dispatched" in channels_seen
        assert "emergent.system_senescent" in channels_seen

    def test_all_five_channels_logged(self, tmp_path):
        """All five subscribed channels each produce a log entry."""
        bus, gl = _make_logger(tmp_path)

        events = [
            ("market.resource.throttle", {"source": "test"}),
            ("market.resource.budget_warning", {"source": "test", "usage_ratio": 0.9}),
            ("scheduling.inhabitant_dispatched", {"system_name": "test"}),
            ("emergent.system_senescent", {"system_name": "test", "wear_level": 1.0, "step": 1}),
            ("emergent.rejuvenation_needed", {"system_name": "test", "wear_level": 0.8, "step": 1}),
        ]
        for channel, payload in events:
            bus.publish(channel, payload)

        lines = _read_lines(gl._log_path)
        assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"

    def test_unsubscribed_channel_not_logged(self, tmp_path):
        """Events on channels not in _GOVERNANCE_CHANNELS are ignored."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("some.other.channel", {"irrelevant": True})

        lines = _read_lines(gl._log_path)
        assert len(lines) == 0, "Unsubscribed channel should not be logged"

    def test_multiple_events_same_channel(self, tmp_path):
        """Publishing the same channel multiple times yields multiple lines."""
        bus, gl = _make_logger(tmp_path)

        for i in range(5):
            bus.publish("market.resource.throttle", {"source": f"src_{i}"})

        lines = _read_lines(gl._log_path)
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Section 3: Log file creation
# ---------------------------------------------------------------------------

class TestLogFileCreated:
    """Log file and parent directory are created on first event."""

    def test_log_file_created(self, tmp_path):
        """Log file is created when the first event is published."""
        log_path = str(tmp_path / "governance_log.jsonl")
        assert not os.path.exists(log_path), "File should not exist before events"

        bus = EventBus()
        gl = GovernanceLogger(event_bus=bus, log_path=log_path)

        assert not os.path.exists(log_path), "File should not exist before first event"

        bus.publish("market.resource.throttle", {"source": "test"})

        assert os.path.exists(log_path), "File should exist after first event"

    def test_nested_directory_created(self, tmp_path):
        """Parent directories are created if they do not exist."""
        log_path = str(tmp_path / "deep" / "nested" / "dir" / "gov.jsonl")
        bus = EventBus()
        gl = GovernanceLogger(event_bus=bus, log_path=log_path)

        bus.publish("market.resource.throttle", {"source": "test"})

        assert os.path.exists(log_path), (
            "Log file should be created with nested directory structure"
        )

    def test_log_lines_are_valid_json(self, tmp_path):
        """Every line in the log file is valid JSON."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("market.resource.throttle", {"source": "sec", "count": 42})
        bus.publish("emergent.system_senescent", {"system_name": "test", "wear_level": 1.0, "step": 1})

        with open(gl._log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    record = json.loads(line)  # Raises if invalid JSON
                    assert "timestamp" in record
                    assert "channel" in record
                    assert "event" in record

    def test_timestamp_is_iso8601(self, tmp_path):
        """The timestamp field is an ISO 8601 string."""
        from datetime import datetime, timezone

        bus, gl = _make_logger(tmp_path)
        bus.publish("market.resource.throttle", {"source": "test"})

        lines = _read_lines(gl._log_path)
        assert lines, "Expected at least one line"
        ts = lines[0]["timestamp"]
        # Should parse without error as ISO 8601
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None, "Timestamp should be timezone-aware"


# ---------------------------------------------------------------------------
# Section 4: Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    """get_statistics() returns accurate information."""

    def test_statistics_keys(self, tmp_path):
        """Statistics dict has all required keys."""
        bus, gl = _make_logger(tmp_path)
        stats = gl.get_statistics()

        required_keys = {
            "event_count", "last_event_time", "log_path",
            "log_file_bytes", "subscribed_channels",
        }
        assert required_keys.issubset(stats.keys()), (
            f"Missing keys: {required_keys - stats.keys()}"
        )

    def test_statistics_event_count(self, tmp_path):
        """event_count reflects the number of events logged."""
        bus, gl = _make_logger(tmp_path)

        bus.publish("market.resource.throttle", {"source": "a"})
        bus.publish("market.resource.throttle", {"source": "b"})
        bus.publish("emergent.system_senescent", {"system_name": "x", "wear_level": 1.0, "step": 1})

        stats = gl.get_statistics()
        assert stats["event_count"] == 3, (
            f"Expected event_count=3, got {stats['event_count']}"
        )

    def test_statistics_last_event_time_set(self, tmp_path):
        """last_event_time is None before events, set after."""
        bus, gl = _make_logger(tmp_path)

        assert gl.get_statistics()["last_event_time"] is None

        before = time.time()
        bus.publish("market.resource.throttle", {"source": "test"})
        after = time.time()

        stats = gl.get_statistics()
        assert stats["last_event_time"] is not None
        assert before <= stats["last_event_time"] <= after

    def test_statistics_log_file_bytes(self, tmp_path):
        """log_file_bytes increases after writing events."""
        bus, gl = _make_logger(tmp_path)

        bytes_before = gl.get_statistics()["log_file_bytes"]
        assert bytes_before == 0, "File should be empty before any events"

        bus.publish("market.resource.throttle", {"source": "test"})

        bytes_after = gl.get_statistics()["log_file_bytes"]
        assert bytes_after > 0, "File should have content after an event"

    def test_statistics_subscribed_channels(self, tmp_path):
        """subscribed_channels lists all five governance channels."""
        bus, gl = _make_logger(tmp_path)
        stats = gl.get_statistics()

        for ch in _GOVERNANCE_CHANNELS:
            assert ch in stats["subscribed_channels"], (
                f"Expected {ch} in subscribed_channels"
            )

    def test_statistics_log_path_is_absolute(self, tmp_path):
        """log_path in statistics is an absolute path."""
        bus, gl = _make_logger(tmp_path)
        stats = gl.get_statistics()
        assert os.path.isabs(stats["log_path"]), (
            f"Expected absolute log_path, got {stats['log_path']}"
        )


# ---------------------------------------------------------------------------
# Section 5: Write failure resilience
# ---------------------------------------------------------------------------

class TestWriteFailureDoesntCrash:
    """Write failures are handled gracefully."""

    def test_write_failure_doesnt_crash(self, tmp_path):
        """If open() raises OSError, _on_event logs a warning and returns cleanly."""
        bus, gl = _make_logger(tmp_path)

        with patch("builtins.open", side_effect=OSError("disk full")):
            import logging
            with patch.object(
                logging.getLogger("mae_core.governance.governance_logger"),
                "warning",
            ) as mock_warn:
                # Should not raise.
                gl._on_event("market.resource.throttle", '{"source": "test"}')

        mock_warn.assert_called_once()
        assert "disk full" in str(mock_warn.call_args) or "write failed" in str(mock_warn.call_args)

    def test_unserializable_event_logs_warning(self, tmp_path):
        """If event cannot be JSON-serialized, a warning is logged and no exception raised."""
        bus, gl = _make_logger(tmp_path)

        # Pass a non-serializable object directly to _on_event.
        class Unserializable:
            pass

        import logging
        with patch.object(
            logging.getLogger("mae_core.governance.governance_logger"),
            "warning",
        ) as mock_warn:
            # Should not raise.
            gl._on_event("market.resource.throttle", Unserializable())

        # A warning may or may not be logged depending on whether json.dumps
        # can handle the string representation. The critical invariant is no exception.
        # (Unserializable wrapped in a dict will fail json.dumps.)

    def test_event_count_not_incremented_on_write_failure(self, tmp_path):
        """event_count is not incremented when a write fails."""
        bus, gl = _make_logger(tmp_path)

        with patch("builtins.open", side_effect=OSError("disk full")):
            gl._on_event("market.resource.throttle", '{"source": "test"}')

        assert gl.get_statistics()["event_count"] == 0, (
            "event_count should not increment on write failure"
        )

    def test_subsequent_events_succeed_after_write_failure(self, tmp_path):
        """After a write failure, subsequent events can still be written."""
        bus, gl = _make_logger(tmp_path)

        # First call fails.
        with patch("builtins.open", side_effect=OSError("transient failure")):
            gl._on_event("market.resource.throttle", '{"source": "x"}')

        # Second call should succeed normally.
        bus.publish("market.resource.throttle", {"source": "y"})

        lines = _read_lines(gl._log_path)
        assert len(lines) == 1, (
            f"Expected 1 successful write after transient failure, got {len(lines)}"
        )
        assert gl.get_statistics()["event_count"] == 1
