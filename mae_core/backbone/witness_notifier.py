"""Witness Notifier - Operational witnessing for Mae's triadic connections.

Biological analogy: The vagus nerve. Every organ sends status signals to
the brainstem, which monitors without controlling. The vagus nerve doesn't
decide what the heart does - it watches, records, and reports anomalies.

Before this system: witnesses were names in lists (declarative).
After this system: witnesses receive shadow notifications when messages
flow through connections they observe (operational).

The Connection Law requires three pathways for every A-B connection:
  - Primary pathway: A -> B (direct signal) -- already works via EventBus
  - Verification pathway: A -> C -> B (witness sees the signal) -- THIS MODULE
  - Balance pathway: B -> C -> A (feedback through witness) -- THIS MODULE

This module implements the verification pathway in two layers:
  1. Per-message: shadow metadata published to witness_notifier.observation
     when any witnessed EventBus message flows (existing).
  2. Per-witness cadenced digests: at Fibonacci 13, each individual witness
     receives a witness.digest.{name} summary of what they observed. This
     makes each witness INDIVIDUALLY ACCOUNTABLE -- we track per-witness
     observation counts and can report which witnesses are active vs silent.

Connection points:
- Created after ConnectionRegistry.seal() in bootstrap wiring
- Subscribes to all EventBus channels that have registered triads
- Publishes shadow notifications on witness_notifier.observation
- Publishes per-witness digests on witness_notifier.digest.{name} (cadenced)
- Cadenced health summaries on witness_notifier.health (Fibonacci 13)
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# EventBus channels (owned by WitnessNotifier)
CH_WITNESS_OBSERVATION = "witness_notifier.observation"
CH_WITNESS_HEALTH = "witness_notifier.health"
CH_WITNESS_DIGEST_PREFIX = "witness_notifier.digest."  # + witness_name
CH_WITNESS_VERDICT = "witness_notifier.verdict"


class WitnessNotifier:
    """Publishes shadow notifications to witnesses when messages flow.

    Subscribes to every EventBus channel that has a registered
    ConnectionTriad. When a message arrives, publishes lightweight
    metadata to witness.observation so witnesses are operationally
    informed of message flow.

    Advisory only -- never blocks, never slows the primary pathway.
    Think of it as a security camera system: it records what flows
    through the building's corridors without stopping anyone.
    """

    def __init__(
        self,
        event_bus: Any,
        connection_registry: Any,
        holon_registry: Any = None,
    ) -> None:
        self._bus = event_bus
        self._registry = connection_registry
        self._holon_registry = holon_registry

        # channel -> list of ConnectionTriad references
        self._channel_triads: dict[str, list[Any]] = {}
        self._subscribed_channels: set[str] = set()

        # Re-entrancy guard (prevents infinite recursion if someone
        # accidentally registers a triad with witness.observation as channel)
        self._in_shadow = False

        # Statistics
        self._messages_witnessed = 0
        self._shadow_failures = 0
        self._step_counter = 0
        self._health_reports = 0

        # Per-witness accountability: witness_name -> observation count
        self._witness_delivery_counts: dict[str, int] = {}
        self._witness_digests_sent = 0

        # Operational sensing + balance pathway counters
        self._witness_sense_count = 0
        self._witness_sense_failures = 0
        self._verdicts_published = 0
        self._verdict_failures = 0

        # Cadence: Fibonacci 13 (aligns with Mae's cadence system)
        self._cadence = 13

        logger.info("WitnessNotifier initialized")

    def activate(self) -> None:
        """Subscribe to all channels that have registered triadic connections.

        Must be called AFTER register_all_connections() and seal().
        Builds a channel-to-triads index and registers callbacks.
        """
        # Build index: channel -> [triad, triad, ...]
        with self._registry._lock:
            for triad in self._registry._connections.values():
                if triad.channel and triad.channel not in (
                    CH_WITNESS_OBSERVATION, CH_WITNESS_HEALTH,
                    CH_WITNESS_VERDICT,
                ) and not triad.channel.startswith(CH_WITNESS_DIGEST_PREFIX):
                    if triad.channel not in self._channel_triads:
                        self._channel_triads[triad.channel] = []
                    self._channel_triads[triad.channel].append(triad)

        # Subscribe to each indexed channel
        for channel in self._channel_triads:
            if channel not in self._subscribed_channels:
                self._bus.register_callback(channel, self._on_message)
                self._subscribed_channels.add(channel)

        observable = len(self._channel_triads)
        total = len(self._registry._connections)
        structural = total - sum(len(v) for v in self._channel_triads.values())

        logger.info(
            "WitnessNotifier activated: %d channels subscribed, "
            "%d observable triads, %d structural (no channel)",
            len(self._subscribed_channels), observable, structural,
        )

    def _on_message(self, channel: str, serialized: str) -> None:
        """Shadow callback: notify witnesses for this channel's triads.

        Called synchronously by EventBus after primary delivery.
        Publishes lightweight metadata to witness.observation.
        Never raises -- all exceptions caught.
        """
        if self._in_shadow:
            return  # Re-entrancy guard

        triads = self._channel_triads.get(channel)
        if not triads:
            return

        self._in_shadow = True
        try:
            # Compute payload hash (not the full content -- performance)
            payload_hash = hashlib.md5(
                serialized.encode() if isinstance(serialized, str)
                else str(serialized).encode()
            ).hexdigest()[:12]

            timestamp = time.time()

            for triad in triads:
                try:
                    self._bus.publish(CH_WITNESS_OBSERVATION, {
                        "channel": channel,
                        "source": triad.source,
                        "target": triad.target,
                        "witnesses": triad.witnesses,
                        "connection_id": triad.connection_id,
                        "timestamp": timestamp,
                        "payload_hash": payload_hash,
                    })
                    self._messages_witnessed += 1
                    # Per-witness accountability: count each witness informed
                    for w in triad.witnesses:
                        self._witness_delivery_counts[w] = (
                            self._witness_delivery_counts.get(w, 0) + 1
                        )

                    # Operational sensing + balance pathway verdict
                    if self._holon_registry is not None:
                        for w in triad.witnesses:
                            try:
                                proxy = self._holon_registry.get_proxy(w)
                                if proxy is not None:
                                    proxy.sense()
                                    self._witness_sense_count += 1
                                    # Balance pathway: witness publishes verdict
                                    self._bus.publish(CH_WITNESS_VERDICT, {
                                        "connection_id": triad.connection_id,
                                        "witness": w,
                                        "source": triad.source,
                                        "target": triad.target,
                                        "verdict": "observed",
                                        "payload_hash": payload_hash,
                                        "step": self._step_counter,
                                    })
                                    self._verdicts_published += 1
                            except Exception:
                                self._witness_sense_failures += 1
                except Exception:
                    self._shadow_failures += 1
                    logger.debug(
                        "Shadow notification failed for %s",
                        triad.connection_id, exc_info=True,
                    )
        except Exception:
            self._shadow_failures += 1
            logger.debug("WitnessNotifier._on_message failed", exc_info=True)
        finally:
            self._in_shadow = False

    def step(self) -> None:
        """Cadenced health summary + per-witness digests (Fibonacci 13)."""
        self._step_counter += 1
        if self._step_counter % self._cadence != 0:
            return

        stats = self.get_statistics()
        self._health_reports += 1

        if self._bus:
            self._bus.publish(CH_WITNESS_HEALTH, stats)

            # Per-witness digest: notify each witness individually
            self._in_shadow = True
            try:
                for witness, count in self._witness_delivery_counts.items():
                    if count > 0:
                        try:
                            self._bus.publish(
                                f"{CH_WITNESS_DIGEST_PREFIX}{witness}",
                                {
                                    "witness": witness,
                                    "observations_since_last": count,
                                    "total_observations": count,
                                    "step": self._step_counter,
                                },
                            )
                            self._witness_digests_sent += 1
                        except Exception:
                            logger.debug(
                                "Witness digest failed for %s",
                                witness, exc_info=True,
                            )
                # Reset counts for next cadence window
                self._witness_delivery_counts = {
                    w: 0 for w in self._witness_delivery_counts
                }
            finally:
                self._in_shadow = False

        logger.info(
            "WitnessNotifier health: %d witnessed, %d failures, "
            "%d channels, %d witnesses active",
            stats["messages_witnessed"],
            stats["shadow_failures"],
            stats["subscribed_channels"],
            stats["active_witnesses"],
        )

    def get_statistics(self) -> dict[str, Any]:
        """Return witnessing statistics for observability."""
        total_connections = len(self._registry._connections)
        observable_triads = sum(len(v) for v in self._channel_triads.values())

        # Per-witness accountability summary
        active_witnesses = sum(
            1 for c in self._witness_delivery_counts.values() if c > 0
        )
        total_witnesses = len(self._witness_delivery_counts)
        top_witnesses = sorted(
            self._witness_delivery_counts.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]

        return {
            "messages_witnessed": self._messages_witnessed,
            "shadow_failures": self._shadow_failures,
            "subscribed_channels": len(self._subscribed_channels),
            "observable_triads": observable_triads,
            "structural_triads": total_connections - observable_triads,
            "total_connections": total_connections,
            "health_reports": self._health_reports,
            "witness_coverage_pct": (
                (observable_triads / max(total_connections, 1)) * 100
            ),
            "active_witnesses": active_witnesses,
            "total_witnesses": total_witnesses,
            "witness_digests_sent": self._witness_digests_sent,
            "top_witnesses": top_witnesses,
            "per_witness_counts": dict(self._witness_delivery_counts),
            "witness_sense_count": self._witness_sense_count,
            "witness_sense_failures": self._witness_sense_failures,
            "verdicts_published": self._verdicts_published,
            "verdict_failures": self._verdict_failures,
        }
