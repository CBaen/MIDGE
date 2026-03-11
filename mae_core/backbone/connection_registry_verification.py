"""Connection Registry Verification — periodic health checks.

Extracted from connection_registry.py to keep it under the 500-line cap.

Provides verify_all() and step() for periodic connection health monitoring.
Used as a mixin by ConnectionRegistry.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ConnectionVerificationMixin:
    """Mixin providing verify_all() and step() for ConnectionRegistry.

    Requires the host class to have:
      self._connections, self._lock, self._somatic_map, self._active_mode,
      self._bus, self._total_verifications, self._witness_notifier,
      self._step_counter, self._verify_interval, self.get_bare_dyads,
      self.get_euler_statistics
    """

    def verify_all(self) -> dict[str, Any]:
        """Verify all registered connections are healthy.

        Checks that source, target, and witness systems are all
        known to SomaticMap (i.e., alive and registered).
        """
        from mae_core.backbone.connection_registry_models import EnforcementMode

        results = {"total": 0, "healthy": 0, "unhealthy": 0, "bare_dyads": 0}

        with self._lock:
            for conn_id, triad in self._connections.items():
                results["total"] += 1
                healthy = True

                if self._somatic_map:
                    # Check source and target exist in SomaticMap
                    for system_id in (triad.source, triad.target):
                        node = self._somatic_map.get_system_info(system_id)
                        if node is None:
                            healthy = False

                    # Check all witnesses exist
                    if triad.witnesses:
                        for w in triad.witnesses:
                            witness_node = self._somatic_map.get_system_info(w)
                            if witness_node is None:
                                healthy = False
                    else:
                        results["bare_dyads"] += 1

                triad.healthy = healthy
                triad.last_verified = time.time()

                if healthy:
                    results["healthy"] += 1
                else:
                    results["unhealthy"] += 1
                    if self._active_mode == EnforcementMode.BLOCKING and self._bus:
                        from mae_core.backbone.connection_registry import CH_CONNECTION_BLOCKED
                        self._bus.publish(CH_CONNECTION_BLOCKED, {
                            "connection_id": conn_id,
                            "reason": "unhealthy",
                            "source": triad.source,
                            "target": triad.target,
                        })

            self._total_verifications += 1

        # Topological invariant (advisory — never blocks)
        results["euler"] = self.get_euler_statistics()

        # Operational witnessing stats (if WitnessNotifier is active)
        if self._witness_notifier is not None:
            try:
                results["witnessing"] = self._witness_notifier.get_statistics()
            except Exception:
                pass  # Advisory — never break verification

        if self._bus:
            from mae_core.backbone.connection_registry import CH_CONNECTION_VERIFIED
            self._bus.publish(CH_CONNECTION_VERIFIED, results)

        return results

    def step(self) -> None:
        """Step hook for periodic verification."""
        from mae_core.backbone.connection_registry_models import EnforcementMode
        from mae_core.backbone.connection_registry import CH_CONNECTION_BARE_DYAD

        self._step_counter += 1
        if self._step_counter % self._verify_interval == 0:
            results = self.verify_all()
            bare = self.get_bare_dyads()
            if bare:
                if self._active_mode == EnforcementMode.BLOCKING:
                    logger.error(
                        "BLOCKING: %d bare dyads detected", len(bare),
                    )
                else:
                    logger.warning(
                        "ConnectionRegistry: %d bare dyads detected", len(bare),
                    )
                if self._bus:
                    self._bus.publish(CH_CONNECTION_BARE_DYAD, {
                        "count": len(bare),
                        "connections": [b.connection_id for b in bare],
                        "enforcement": self._active_mode.value,
                    })
