"""SituationBoard — shared workspace for analyst findings and system state.

Replaces the ad-hoc ctx._market_advisory dict with a typed, thread-safe
structure. Analysts publish findings, the heartbeat writer reads snapshots,
and the paper trading gate can query for ticker-specific intelligence.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("midge.market.situation_board")


@dataclass
class AnalystFinding:
    analyst_id: str           # "deep_analyst", "convergence", "cascade", etc.
    ticker: str               # "" for global findings
    direction: str            # "bullish" | "bearish" | "neutral"
    confidence: float         # 0.0-1.0
    summary: str              # Human-readable one-line summary
    evidence: list            # Top 3-5 evidence items (dicts)
    domains: list             # Contributing domains
    timestamp: str            # ISO format
    step: int                 # Step number when published
    decay_hours: float = 4.0  # How long this finding stays relevant


class SituationBoard:
    """Typed, thread-safe board where analysts post findings.

    Key operations:
    - publish(): analyst posts a finding (upsert by key)
    - get_findings(): query findings with optional filters (expired excluded)
    - get_snapshot(): heartbeat-safe JSON-serializable view
    - get_ticker_consensus(): multi-analyst agreement on one ticker
    - prune_expired(): housekeeping (auto-called in get_snapshot)
    - save() / load(): atomic persistence to JSON
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Key: (analyst_id, ticker, direction)
        self._findings: Dict[Tuple[str, str, str], AnalystFinding] = {}
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Write side
    # ------------------------------------------------------------------

    def publish(self, finding: AnalystFinding) -> None:
        """Add or replace a finding keyed by (analyst_id, ticker, direction)."""
        key = (finding.analyst_id, finding.ticker, finding.direction)
        with self._lock:
            self._findings[key] = finding
            self._current_step = max(self._current_step, finding.step)

    # ------------------------------------------------------------------
    # Read side
    # ------------------------------------------------------------------

    def get_findings(
        self,
        ticker: Optional[str] = None,
        direction: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[AnalystFinding]:
        """Return non-expired findings matching optional filters."""
        now = datetime.now()
        with self._lock:
            results = []
            for finding in self._findings.values():
                if self._is_expired(finding, now):
                    continue
                if ticker is not None and finding.ticker != ticker:
                    continue
                if direction is not None and finding.direction != direction:
                    continue
                if finding.confidence < min_confidence:
                    continue
                results.append(finding)
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def get_snapshot(self) -> dict:
        """Return all non-expired findings as a JSON-serializable dict.

        Prunes expired entries as a side-effect. Safe to call from the
        heartbeat writer on any thread.
        """
        with self._lock:
            self.prune_expired()
            findings_list = [asdict(f) for f in self._findings.values()]
        return {
            "step": self._current_step,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "finding_count": len(findings_list),
            "findings": findings_list,
        }

    def get_ticker_consensus(self, ticker: str) -> dict:
        """Summarise all analysts' current views on a single ticker.

        Returns:
            direction: majority vote ("bullish" | "bearish" | "neutral" | "split")
            avg_confidence: mean confidence across all analysts
            analyst_count: number of analysts with non-expired views
            agreement_pct: fraction of analysts agreeing with majority
        """
        findings = self.get_findings(ticker=ticker)
        if not findings:
            return {
                "direction": "neutral",
                "avg_confidence": 0.0,
                "analyst_count": 0,
                "agreement_pct": 0.0,
            }

        direction_counts: Dict[str, int] = {}
        for f in findings:
            direction_counts[f.direction] = direction_counts.get(f.direction, 0) + 1

        majority_dir = max(direction_counts, key=lambda d: direction_counts[d])
        majority_count = direction_counts[majority_dir]
        total = len(findings)
        agreement_pct = majority_count / total if total > 0 else 0.0

        # "split" if top two directions are tied
        sorted_dirs = sorted(direction_counts.values(), reverse=True)
        if len(sorted_dirs) >= 2 and sorted_dirs[0] == sorted_dirs[1]:
            majority_dir = "split"

        avg_conf = sum(f.confidence for f in findings) / total

        return {
            "direction": majority_dir,
            "avg_confidence": round(avg_conf, 4),
            "analyst_count": total,
            "agreement_pct": round(agreement_pct, 4),
        }

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Remove findings older than their decay_hours. Returns count removed.

        Must be called while holding self._lock (or from get_snapshot which holds it).
        """
        now = datetime.now()
        expired_keys = [
            k for k, f in self._findings.items()
            if self._is_expired(f, now)
        ]
        for k in expired_keys:
            del self._findings[k]
        return len(expired_keys)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Atomically write current (non-expired) findings to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        snapshot = self.get_snapshot()
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, indent=2, default=str)
            tmp.replace(path)
        except Exception:
            logger.debug("SituationBoard.save failed", exc_info=True)
            tmp.unlink(missing_ok=True)

    def load(self, path: Path) -> int:
        """Load findings from a JSON file written by save(). Returns count loaded."""
        path = Path(path)
        if not path.exists():
            return 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            loaded = 0
            with self._lock:
                for raw in data.get("findings", []):
                    try:
                        finding = AnalystFinding(**raw)
                        key = (finding.analyst_id, finding.ticker, finding.direction)
                        self._findings[key] = finding
                        loaded += 1
                    except Exception:
                        logger.debug("Skipping malformed finding during load", exc_info=True)
            logger.info("SituationBoard: loaded %d findings from %s", loaded, path)
            return loaded
        except Exception:
            logger.debug("SituationBoard.load failed for %s", path, exc_info=True)
            return 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_expired(finding: AnalystFinding, now: datetime) -> bool:
        """Return True if finding is older than its decay_hours."""
        try:
            ts = datetime.fromisoformat(finding.timestamp)
            return (now - ts) > timedelta(hours=finding.decay_hours)
        except Exception:
            return False  # If we can't parse, keep it

    def __len__(self) -> int:
        with self._lock:
            return len(self._findings)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SituationBoard(findings={len(self._findings)}, step={self._current_step})"
