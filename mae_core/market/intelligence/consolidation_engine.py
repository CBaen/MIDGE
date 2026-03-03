"""Sleep Cycle / Memory Consolidation — Gift 6.

Periodic consolidation of learned patterns: pruning weak Thompson
distributions, archiving stale hypotheses, compressing signal history.
Like biological sleep: replay important memories, discard noise,
reorganize for efficiency.

Without consolidation, Thompson distributions accumulate indefinitely,
hypothesis registries grow without bound, and discovery logs bloat.
MIDGE becomes sluggish and old experiences crowd out recent learning.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# Consolidation thresholds
_MIN_SAMPLES_TO_KEEP = 3  # Distributions with < 3 samples are prunable
_STALE_DAYS = 30  # Distributions untouched for 30+ days are prunable
_HYPOTHESIS_ARCHIVE_DAYS = 14  # RETIRED hypotheses > 14 days → archive
_DISCOVERY_ARCHIVE_DAYS = 90  # Discovery log entries > 90 days → summary


@dataclass
class ConsolidationReport:
    """Summary of a consolidation cycle."""

    distributions_pruned: int = 0
    hypotheses_archived: int = 0
    log_entries_compressed: int = 0
    duration_ms: float = 0.0
    timestamp: str = ""


class ConsolidationEngine:
    """Periodic memory consolidation for the market intelligence system.

    Prunes weak Thompson distributions, archives stale hypotheses,
    and compresses old discovery log entries. Runs at low cadence
    (every 5000 steps) to avoid interfering with active processing.
    """

    def __init__(
        self,
        thompson_sampler=None,
        hypothesis_registry=None,
        hypothesis_engine=None,
        event_bus=None,
    ):
        self._thompson_sampler = thompson_sampler
        self._hypothesis_registry = hypothesis_registry
        self._hypothesis_engine = hypothesis_engine
        self._event_bus = event_bus
        self._consolidation_cadence: int = 5000
        self._last_consolidation: int = 0
        self._reports: list = []

    def consolidate(self) -> ConsolidationReport:
        """Run a full consolidation cycle.

        Prunes Thompson distributions, archives hypotheses, and
        compresses old discovery log entries.
        """
        start = time.time()
        report = ConsolidationReport(
            timestamp=datetime.now().isoformat(),
        )

        report.distributions_pruned = self._prune_thompson()
        report.hypotheses_archived = self._archive_hypotheses()
        report.log_entries_compressed = self._compress_discovery_log()

        elapsed = (time.time() - start) * 1000
        report.duration_ms = round(elapsed, 1)

        self._reports.append(report)
        if len(self._reports) > 20:
            self._reports = self._reports[-20:]

        logger.info(
            "Consolidation complete: %d distributions pruned, "
            "%d hypotheses archived, %d log entries compressed (%.1f ms)",
            report.distributions_pruned,
            report.hypotheses_archived,
            report.log_entries_compressed,
            report.duration_ms,
        )

        # Publish consolidation event
        if self._event_bus is not None:
            try:
                from mae_core.market.channels import CH_CONSOLIDATION_COMPLETE

                self._event_bus.publish(
                    CH_CONSOLIDATION_COMPLETE,
                    {
                        "distributions_pruned": report.distributions_pruned,
                        "hypotheses_archived": report.hypotheses_archived,
                        "log_entries_compressed": report.log_entries_compressed,
                        "duration_ms": report.duration_ms,
                    },
                )
            except Exception:
                logger.debug("Failed to publish consolidation event", exc_info=True)

        return report

    def _prune_thompson(self) -> int:
        """Remove Thompson distributions with < min samples and stale for > threshold days."""
        if self._thompson_sampler is None:
            return 0

        pruned = 0
        try:
            distributions = getattr(self._thompson_sampler, "_distributions", None)
            if distributions is None or not isinstance(distributions, dict):
                return 0

            stale_cutoff = datetime.now() - timedelta(days=_STALE_DAYS)
            keys_to_prune = []

            for key, dist in distributions.items():
                samples = getattr(dist, "samples", 0)
                last_update = getattr(dist, "last_update", None)

                if samples < _MIN_SAMPLES_TO_KEEP:
                    # Check staleness: only prune if also stale
                    if last_update is not None:
                        try:
                            if isinstance(last_update, str):
                                update_dt = datetime.fromisoformat(last_update)
                            else:
                                update_dt = last_update
                            if update_dt < stale_cutoff:
                                keys_to_prune.append(key)
                        except (ValueError, TypeError):
                            # Can't parse date — keep it
                            pass
                    else:
                        # No last_update and thin data — prune
                        keys_to_prune.append(key)

            for key in keys_to_prune:
                try:
                    del distributions[key]
                    pruned += 1
                except (KeyError, RuntimeError):
                    pass

        except Exception:
            logger.debug("Thompson pruning failed", exc_info=True)

        return pruned

    def _archive_hypotheses(self) -> int:
        """Move RETIRED hypotheses older than threshold to archive file."""
        if self._hypothesis_registry is None:
            return 0

        archived = 0
        try:
            registry = self._hypothesis_registry
            hypotheses = getattr(registry, "_hypotheses", None)
            if hypotheses is None or not isinstance(hypotheses, dict):
                return 0

            archive_cutoff = datetime.now() - timedelta(days=_HYPOTHESIS_ARCHIVE_DAYS)
            archive_path = _DATA_DIR / "hypotheses_archive.jsonl"
            keys_to_archive = []

            for hyp_id, hyp in hypotheses.items():
                status = getattr(hyp, "status", "")
                if status != "RETIRED":
                    continue

                retired_at = getattr(hyp, "retired_at", None) or getattr(
                    hyp, "updated_at", None
                )
                if retired_at is None:
                    continue

                try:
                    if isinstance(retired_at, str):
                        ret_dt = datetime.fromisoformat(retired_at)
                    else:
                        ret_dt = retired_at

                    if ret_dt < archive_cutoff:
                        keys_to_archive.append(hyp_id)
                except (ValueError, TypeError):
                    pass

            # Write to archive, then remove from registry
            if keys_to_archive:
                _DATA_DIR.mkdir(parents=True, exist_ok=True)
                with open(archive_path, "a", encoding="utf-8") as f:
                    for hyp_id in keys_to_archive:
                        hyp = hypotheses[hyp_id]
                        try:
                            record = {
                                "hypothesis_id": hyp_id,
                                "archived_at": datetime.now().isoformat(),
                            }
                            # Serialize what we can
                            for attr in (
                                "trigger_source",
                                "target_source",
                                "lag_days",
                                "status",
                                "win_rate",
                                "observations",
                            ):
                                val = getattr(hyp, attr, None)
                                if val is not None:
                                    record[attr] = val
                            f.write(json.dumps(record) + "\n")
                            archived += 1
                        except Exception:
                            pass

                # Remove archived from registry
                for hyp_id in keys_to_archive:
                    try:
                        del hypotheses[hyp_id]
                    except (KeyError, RuntimeError):
                        pass

        except Exception:
            logger.debug("Hypothesis archiving failed", exc_info=True)

        return archived

    def _compress_discovery_log(self) -> int:
        """Compress discovery log entries older than threshold."""
        compressed = 0
        discovery_path = _DATA_DIR / "discovery_log.jsonl"

        if not discovery_path.exists():
            return 0

        try:
            cutoff = datetime.now() - timedelta(days=_DISCOVERY_ARCHIVE_DAYS)
            keep_lines = []
            compressed_count = 0

            with open(discovery_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts_str = entry.get("timestamp", entry.get("discovered_at", ""))
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str)
                            if ts < cutoff:
                                compressed_count += 1
                                continue  # Skip old entries
                    except (ValueError, json.JSONDecodeError):
                        pass
                    keep_lines.append(line)

            if compressed_count > 0:
                # Rewrite file with only recent entries
                with open(discovery_path, "w", encoding="utf-8") as f:
                    for line in keep_lines:
                        f.write(line + "\n")
                compressed = compressed_count

        except Exception:
            logger.debug("Discovery log compression failed", exc_info=True)

        return compressed

    def should_consolidate(self, step: int) -> bool:
        """Check if consolidation should run at this step."""
        if step - self._last_consolidation >= self._consolidation_cadence:
            self._last_consolidation = step
            return True
        return False

    def get_statistics(self) -> dict:
        """Return consolidation statistics for holon protocol."""
        last_report = self._reports[-1] if self._reports else None
        return {
            "total_consolidations": len(self._reports),
            "last_consolidation": (
                last_report.timestamp if last_report else None
            ),
            "last_pruned": (
                last_report.distributions_pruned if last_report else 0
            ),
            "last_archived": (
                last_report.hypotheses_archived if last_report else 0
            ),
        }
