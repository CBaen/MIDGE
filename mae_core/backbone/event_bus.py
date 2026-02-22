"""EventBus - In-process pub/sub replacing Redis.

Provides channel-based message passing between agents without
requiring an external server. All communication stays within
the Python process.

Drop-in replacement for RedisClient's pub/sub interface.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Callable, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types from learning/memory systems."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class EventBus:
    """In-process event bus replacing Redis pub/sub + streams.

    Supports:
    - Channel-based pub/sub (publish, subscribe, listen)
    - Stream operations (write_to_stream, read_from_stream)
    - Synchronous callback-based delivery
    """

    def __init__(self) -> None:
        # Pub/sub state
        self._subscribers: dict[str, list[Callable[[str, Any], None]]] = defaultdict(list)
        self._message_queue: deque[dict[str, Any]] = deque()
        self._subscribed_channels: set[str] = set()
        self._listening = False

        # Stream state (replaces Redis streams)
        self._streams: dict[str, deque[tuple[str, dict[str, Any]]]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._stream_counters: dict[str, int] = defaultdict(int)

        # Advisory witnessing (wired after bootstrap seal)
        self._connection_registry: Any = None

        # Thread safety for parallel agent stepping.
        # RLock (reentrant) because callbacks may re-publish during delivery.
        self._lock = threading.RLock()

        # Statistics
        self._publish_count = 0
        self._deliver_count = 0
        self._dropped_count = 0

    def set_connection_registry(self, registry: Any) -> None:
        """Wire the ConnectionRegistry for advisory witnessing on publish().

        Called after bootstrap seal so that every EventBus message is
        checked against registered triadic connections. Advisory only --
        never blocks delivery.
        """
        self._connection_registry = registry

    def publish(self, channel: str, message: Any) -> int:
        """Publish a message to a channel. Returns number of subscribers that received it.

        Thread-safe: protected by lock for parallel agent stepping.
        """
        if isinstance(message, (dict, list)):
            serialized = json.dumps(message, cls=_NumpySafeEncoder)
        elif not isinstance(message, str):
            serialized = str(message)
        else:
            serialized = message

        with self._lock:
            self._publish_count += 1

            # Triadic witnessing: check if this channel has a registered triad
            # Skip check for system channels (connection.blocked, witness_notifier.*)
            # to prevent recursive cascades when blocking publishes notifications.
            if (
                self._connection_registry is not None
                and not channel.startswith("connection.")
                and not channel.startswith("witness_notifier.")
            ):
                try:
                    reg = self._connection_registry
                    if reg.sealed:
                        source = channel.split(".")[0] if "." in channel else "unknown"
                        allowed, reason = reg.is_connection_allowed(
                            source, "event_bus", channel=channel,
                        )
                        if not allowed:
                            self._dropped_count += 1
                            logger.warning(
                                "EventBus: message dropped on %s (%s)", channel, reason,
                            )
                            return 0
                except Exception:
                    pass  # Never break message delivery

            delivered = 0

            # Deliver to callback subscribers
            for callback in self._subscribers.get(channel, []):
                try:
                    callback(channel, serialized)
                    delivered += 1
                except Exception:
                    logger.exception("Error in subscriber callback for channel %s", channel)

            # Queue for listen()-based subscribers
            if channel in self._subscribed_channels:
                self._message_queue.append({
                    "type": "message",
                    "channel": channel,
                    "data": serialized,
                })
                delivered += 1

            self._deliver_count += delivered
            return delivered

    def subscribe(self, channels: list[str]) -> None:
        """Subscribe to channels for listen()-based consumption."""
        for channel in channels:
            self._subscribed_channels.add(channel)

    def unsubscribe(self, channels: Optional[list[str]] = None) -> None:
        """Unsubscribe from channels."""
        if channels is None:
            self._subscribed_channels.clear()
        else:
            for channel in channels:
                self._subscribed_channels.discard(channel)

    def listen(self) -> Generator[dict[str, Any], None, None]:
        """Yield messages from subscribed channels (non-blocking)."""
        self._listening = True
        while self._message_queue:
            yield self._message_queue.popleft()
        self._listening = False

    def register_callback(
        self, channel: str, callback: Callable[[str, Any], None]
    ) -> None:
        """Register a callback for a specific channel."""
        self._subscribers[channel].append(callback)

    def unregister_callback(
        self, channel: str, callback: Callable[[str, Any], None]
    ) -> None:
        """Remove a callback from a channel."""
        if channel in self._subscribers:
            try:
                self._subscribers[channel].remove(callback)
            except ValueError:
                pass

    # --- Stream Operations (replaces Redis Streams) ---

    def write_to_stream(
        self,
        stream_name: str,
        data: dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        """Append data to a named stream. Returns stream entry ID.

        Thread-safe: protected by lock for parallel agent stepping.
        """
        with self._lock:
            self._stream_counters[stream_name] += 1
            entry_id = f"{int(time.time() * 1000)}-{self._stream_counters[stream_name]}"

            stream = self._streams[stream_name]
            if maxlen is not None:
                while len(stream) >= maxlen:
                    stream.popleft()

            stream.append((entry_id, data))
            return entry_id

    def read_from_stream(
        self,
        stream_name: str,
        count: Optional[int] = None,
        block: Optional[int] = None,
        last_id: str = "0-0",
    ) -> list[tuple[str, dict[str, Any]]]:
        """Read entries from a stream after last_id."""
        stream = self._streams.get(stream_name, deque())
        results = []

        for entry_id, data in stream:
            if entry_id > last_id:
                results.append((entry_id, data))
                if count and len(results) >= count:
                    break

        return results

    def get_stream_length(self, stream_name: str) -> int:
        """Get number of entries in a stream."""
        return len(self._streams.get(stream_name, deque()))

    # --- Utility ---

    def ping(self) -> bool:
        """Always returns True (no connection to check)."""
        return True

    def close(self) -> None:
        """Clean up resources."""
        self._subscribers.clear()
        self._subscribed_channels.clear()
        self._message_queue.clear()
        self._streams.clear()
        self._listening = False

    def flushdb(self) -> None:
        """Clear all data (for testing)."""
        self.close()
        self._publish_count = 0
        self._deliver_count = 0
        self._dropped_count = 0
        self._stream_counters.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return bus statistics."""
        return {
            "publish_count": self._publish_count,
            "deliver_count": self._deliver_count,
            "dropped_count": self._dropped_count,
            "subscribed_channels": len(self._subscribed_channels),
            "callback_channels": len(self._subscribers),
            "pending_messages": len(self._message_queue),
            "active_streams": len(self._streams),
        }
