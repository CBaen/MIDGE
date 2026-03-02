"""Hypothesis Registry — Event-sourced lifecycle management.

Persists hypotheses to data/market/hypotheses.jsonl (append-only events).
Loads state by replaying events on startup. Provides register, update,
promote, retire, hibernate, and query operations.

Event types:
  - CREATED: new hypothesis registered
  - UPDATED: stats or causal story changed
  - PROMOTED: probation → active
  - RETIRED: any → retired
  - HIBERNATED: active → hibernated
  - REACTIVATED: hibernated → active
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


class HypothesisRegistry:
    """Manages hypothesis lifecycle with event-sourced persistence.

    All mutations append an event to hypotheses.jsonl. State is rebuilt
    from the event log on construction. This gives full audit trail
    and supports any future replay/debugging needs.
    """

    _SNAPSHOT_INTERVAL = 200  # events between snapshot saves

    def __init__(self, data_dir: Path = None):
        self._data_dir = data_dir or DATA_DIR
        self._events_path = self._data_dir / "hypotheses.jsonl"
        self._snapshot_path = self._data_dir / "hypotheses_snapshot.json"
        self._hypotheses: dict[str, Hypothesis] = {}
        self._events_since_snapshot: int = 0
        self._load_events()

    def register(self, hypothesis: Hypothesis) -> str:
        """Register a new hypothesis (enters PROBATION status).

        Returns the hypothesis_id.
        """
        if hypothesis.hypothesis_id in self._hypotheses:
            logger.debug("Hypothesis %s already registered", hypothesis.hypothesis_id)
            return hypothesis.hypothesis_id

        hypothesis.status = HypothesisStatus.PROBATION
        self._hypotheses[hypothesis.hypothesis_id] = hypothesis
        self._append_event("CREATED", hypothesis)
        logger.info(
            "Hypothesis registered: %s [%s → %s, lag=%d]",
            hypothesis.name, hypothesis.trigger.source_a,
            hypothesis.trigger.source_b, hypothesis.trigger.lag_days,
        )
        return hypothesis.hypothesis_id

    def update_stats(self, hypothesis_id: str, hypothesis: Hypothesis) -> None:
        """Update a hypothesis's stats after validation."""
        if hypothesis_id not in self._hypotheses:
            return
        self._hypotheses[hypothesis_id] = hypothesis
        self._append_event("UPDATED", hypothesis)

    def promote(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Promote probation → active. Returns updated hypothesis or None."""
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None or hyp.status != HypothesisStatus.PROBATION:
            return None

        if not hyp.causal_story:
            logger.warning(
                "Cannot promote %s — no causal story (anti-overfitting gate)",
                hypothesis_id,
            )
            return None

        hyp.status = HypothesisStatus.ACTIVE
        hyp.promoted_at = datetime.now()
        self._append_event("PROMOTED", hyp)
        logger.info("Hypothesis promoted: %s (%s)", hyp.name, hypothesis_id)
        return hyp

    def retire(self, hypothesis_id: str, reason: str = "") -> Optional[Hypothesis]:
        """Retire a hypothesis permanently. Returns updated hypothesis or None."""
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None or hyp.status == HypothesisStatus.RETIRED:
            return None

        hyp.status = HypothesisStatus.RETIRED
        hyp.retired_at = datetime.now()
        hyp.retired_reason = reason
        self._append_event("RETIRED", hyp)
        logger.info("Hypothesis retired: %s — %s", hyp.name, reason)
        return hyp

    def hibernate(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Hibernate an active hypothesis (regime mismatch). Reversible."""
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None or hyp.status != HypothesisStatus.ACTIVE:
            return None

        hyp.status = HypothesisStatus.HIBERNATED
        self._append_event("HIBERNATED", hyp)
        logger.info("Hypothesis hibernated: %s", hyp.name)
        return hyp

    def reactivate(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Reactivate a hibernated hypothesis."""
        hyp = self._hypotheses.get(hypothesis_id)
        if hyp is None or hyp.status != HypothesisStatus.HIBERNATED:
            return None

        hyp.status = HypothesisStatus.ACTIVE
        self._append_event("REACTIVATED", hyp)
        logger.info("Hypothesis reactivated: %s", hyp.name)
        return hyp

    def get(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def get_active(self) -> List[Hypothesis]:
        """Get all active hypotheses."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.ACTIVE
        ]

    def get_probation(self) -> List[Hypothesis]:
        """Get all hypotheses in probation."""
        return [
            h for h in self._hypotheses.values()
            if h.status == HypothesisStatus.PROBATION
        ]

    def get_all(self) -> List[Hypothesis]:
        """Get all hypotheses (any status)."""
        return list(self._hypotheses.values())

    def find_by_trigger(self, source_a: str, source_b: str, lag_days: int) -> Optional[Hypothesis]:
        """Find an existing hypothesis matching a trigger pattern (for dedup)."""
        for hyp in self._hypotheses.values():
            if hyp.status == HypothesisStatus.RETIRED:
                continue
            if (hyp.trigger.source_a == source_a
                    and hyp.trigger.source_b == source_b
                    and abs(hyp.trigger.lag_days - lag_days) <= 5):
                return hyp
        return None

    def get_statistics(self) -> dict:
        """Summary stats for HolonProxy.sense() and monitoring."""
        by_status = {}
        for hyp in self._hypotheses.values():
            by_status[hyp.status.value] = by_status.get(hyp.status.value, 0) + 1
        active = self.get_active()
        best_wr = max((h.stats.win_rate for h in active), default=0.0)
        return {
            "total_hypotheses": len(self._hypotheses),
            "by_status": by_status,
            "active_count": len(active),
            "best_active_win_rate": round(best_wr, 4),
        }

    # ── Persistence ─────────────────────────────────────────────────

    def _append_event(self, event_type: str, hypothesis: Hypothesis) -> None:
        """Append event to hypotheses.jsonl and trigger snapshot if threshold crossed."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis.to_dict(),
        }
        try:
            with open(self._events_path, "a") as f:
                f.write(json.dumps(event) + "\n")
            self._events_since_snapshot += 1
            if self._events_since_snapshot >= self._SNAPSHOT_INTERVAL:
                self._save_snapshot()
        except Exception as e:
            logger.warning("Failed to persist hypothesis event: %s", e)

    def _save_snapshot(self) -> None:
        """Serialize current hypothesis state to hypotheses_snapshot.json.

        Uses atomic tmp+replace so a crash mid-write never corrupts the snapshot.
        The JSONL event log is never truncated — it remains the full audit trail.
        """
        tmp = self._snapshot_path.with_suffix(".tmp")
        try:
            snapshot = {
                "saved_at": datetime.now().isoformat(),
                "event_count": self._events_since_snapshot,
                "hypotheses": {
                    hid: hyp.to_dict()
                    for hid, hyp in self._hypotheses.items()
                },
            }
            tmp.write_text(json.dumps(snapshot, indent=2))
            os.replace(tmp, self._snapshot_path)
            self._events_since_snapshot = 0
            logger.debug(
                "Hypothesis snapshot saved (%d hypotheses)", len(self._hypotheses)
            )
        except Exception as e:
            logger.warning("Failed to save hypothesis snapshot: %s", e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _load_events(self) -> None:
        """Rebuild state from snapshot (if present) + incremental JSONL replay.

        If a snapshot exists, hypotheses are loaded from it and only JSONL events
        newer than snapshot.saved_at are replayed. This keeps startup fast even
        when the event log grows large.

        If no snapshot exists, the full event log is replayed (backward compat).
        """
        snapshot_cutoff: datetime | None = None

        if self._snapshot_path.exists():
            try:
                snap = json.loads(self._snapshot_path.read_text())
                snapshot_cutoff = datetime.fromisoformat(snap["saved_at"])
                for hid, hdata in snap.get("hypotheses", {}).items():
                    try:
                        hyp = Hypothesis.from_dict(hdata)
                        self._hypotheses[hyp.hypothesis_id] = hyp
                    except Exception:
                        continue
                logger.debug(
                    "Loaded hypothesis snapshot (%d hypotheses, cutoff=%s)",
                    len(self._hypotheses), snapshot_cutoff.isoformat(),
                )
            except Exception as e:
                logger.warning(
                    "Failed to load hypothesis snapshot, falling back to full replay: %s", e
                )
                self._hypotheses.clear()
                snapshot_cutoff = None

        if not self._events_path.exists():
            return

        try:
            with open(self._events_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if snapshot_cutoff is not None:
                            # Skip events already captured in the snapshot
                            event_ts_raw = event.get("timestamp", "")
                            try:
                                event_ts = datetime.fromisoformat(event_ts_raw)
                            except (ValueError, TypeError):
                                event_ts = None
                            if event_ts is not None and event_ts <= snapshot_cutoff:
                                continue
                        hyp = Hypothesis.from_dict(event.get("hypothesis", {}))
                        self._hypotheses[hyp.hypothesis_id] = hyp
                    except (json.JSONDecodeError, Exception):
                        continue
        except Exception as e:
            logger.warning("Failed to load hypothesis events: %s", e)
