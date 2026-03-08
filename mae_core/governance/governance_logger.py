"""GovernanceLogger — append-only JSONL record of governance events.

Subscribes to governance-related EventBus channels and writes each event
as a JSON line. This creates an immutable audit trail of resource throttling,
budget warnings, dispatch activity, and senescence events.

Law 6 compliance: governance events are organism-internal. The logger is a
passive observer — it records what happens, never intervenes.

Subscribed channels:
    "market.resource.throttle"      — API budget exceeded for a source
    "market.resource.budget_warning" — approaching hourly API limit
    "scheduling.inhabitant_dispatched" — a bio-system was dispatched
    "emergent.system_senescent"     — a system reached end-of-life wear
    "emergent.rejuvenation_needed"  — a system needs repair

Constructor signature (for Builder 1 bootstrapping):
    GovernanceLogger(event_bus, log_path="data/market/governance_log.jsonl")

Usage:
    logger = GovernanceLogger(event_bus=ctx.bus)
    # Events are now being recorded automatically via callbacks.
    stats = logger.get_statistics()
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Channels to observe (all governance-relevant)
_GOVERNANCE_CHANNELS = [
    "market.resource.throttle",
    "market.resource.budget_warning",
    "scheduling.inhabitant_dispatched",
    "emergent.system_senescent",
    "emergent.rejuvenation_needed",
]


class GovernanceLogger:
    """Passive observer that appends governance events to a JSONL file.

    Each line written is a JSON object:
        {"timestamp": "<iso8601>", "channel": "<channel>", "event": <data>}

    Write failures are logged as warnings and never re-raised. The observer
    must never block or crash the organism.

    Args:
        event_bus: EventBus to subscribe to. Must not be None.
        log_path: Path to the append-only JSONL file.
    """

    def __init__(
        self,
        event_bus: Any,
        log_path: str = "data/market/governance_log.jsonl",
    ) -> None:
        self._bus = event_bus
        self._log_path = log_path
        self._event_count = 0
        self._last_event_time: float | None = None

        # Ensure the parent directory exists before any write attempt.
        parent = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(parent, exist_ok=True)

        # Subscribe to all governance channels.
        for channel in _GOVERNANCE_CHANNELS:
            self._bus.register_callback(channel, self._on_event)

        logger.info(
            "GovernanceLogger: subscribed to %d channels, writing to %s",
            len(_GOVERNANCE_CHANNELS), log_path,
        )

    # =========================================================================
    # EventBus callback
    # =========================================================================

    def _on_event(self, channel: str, event_data: Any) -> None:
        """Receive an event and append it to the log file.

        Called synchronously by EventBus on every matching publish().
        Must never raise — all errors are caught and logged.
        """
        now = datetime.now(timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "channel": channel,
            "event": event_data,
        }

        try:
            line = json.dumps(record) + "\n"
        except (TypeError, ValueError) as exc:
            logger.warning(
                "GovernanceLogger: could not serialize event on %s: %s",
                channel, exc,
            )
            return

        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
        except OSError as exc:
            logger.warning(
                "GovernanceLogger: write failed for %s: %s", self._log_path, exc
            )
            return

        self._event_count += 1
        self._last_event_time = time.time()

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return event count, last event time, and log file size.

        Returns:
            dict with keys:
                event_count: total events appended this session
                last_event_time: Unix timestamp of last write, or None
                log_path: absolute path to the log file
                log_file_bytes: file size in bytes, or 0 if file does not exist
                subscribed_channels: list of monitored channel names
        """
        try:
            file_bytes = os.path.getsize(self._log_path)
        except OSError:
            file_bytes = 0

        return {
            "event_count": self._event_count,
            "last_event_time": self._last_event_time,
            "log_path": os.path.abspath(self._log_path),
            "log_file_bytes": file_bytes,
            "subscribed_channels": list(_GOVERNANCE_CHANNELS),
        }

    def __repr__(self) -> str:
        return (
            f"GovernanceLogger("
            f"events={self._event_count}, "
            f"path={self._log_path!r})"
        )
