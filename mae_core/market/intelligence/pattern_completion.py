"""Pattern Completion Engine — Gift 10 (Council Gift).

Active pattern seeking. When Gift 8 identifies a partial archetype match
(e.g., "accumulation: insider buying + range-bound price, but missing
volume decline"), the Completion Engine watches for the missing piece.

This is completion, not prediction. MIDGE already has the archetype
template (Gift 8). She already sees partial matches. The Completion
Engine says: "Given what I've seen, signal X should appear next if this
pattern is real. Let me look for it."

When the missing signal appears, the pattern is confirmed with much
higher confidence than if X had been noticed by chance.

"A missing tool that allows patterns to be truly understood."
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_MAX_HUNTS = 5
_DEFAULT_TTL_HOURS = 72.0
_MIN_PARTIAL_SCORE = 0.3  # Minimum partial match score to spawn a hunt


@dataclass
class CompletionHunt:
    """An active hunt for missing archetype signals."""

    hunt_id: str
    archetype_id: str
    ticker: str
    missing_signals: List[str]  # list of domain/source types still needed
    created_at: datetime
    expires_at: datetime
    partial_match_score: float  # how much of the archetype is already matched
    matched_signals: List[str] = field(default_factory=list)  # domains already matched


@dataclass
class CompletionEvent:
    """Fired when a hunt completes — the missing signal arrived."""

    hunt_id: str
    archetype_id: str
    ticker: str
    completing_signal_domain: str
    completion_score: float  # overall archetype match after completion
    total_archetype_match: float
    confidence_boost: float = 0.15  # sought-and-found > discovered-by-chance


class PatternCompletionEngine:
    """Active pattern seeking — watch for missing archetype signals.

    Lifecycle:
    1. review_partial_matches() reads from archetype_engine.get_partial_matches()
    2. For each partial match with score > threshold and < 1.0, creates a hunt
    3. Each step, check_completions() compares incoming signals against hunts
    4. When a missing signal arrives, fires CH_PATTERN_COMPLETED
    5. Completed/expired hunts are pruned
    """

    def __init__(
        self,
        archetype_engine=None,
        event_bus=None,
        max_concurrent_hunts: int = _DEFAULT_MAX_HUNTS,
        hunt_ttl_hours: float = _DEFAULT_TTL_HOURS,
    ):
        self._archetype_engine = archetype_engine
        self._bus = event_bus
        self._max_concurrent_hunts = max_concurrent_hunts
        self._hunt_ttl_hours = hunt_ttl_hours

        # Active hunts: hunt_id -> CompletionHunt
        self._active_hunts: Dict[str, CompletionHunt] = {}

    def review_partial_matches(self) -> List[CompletionHunt]:
        """Create hunts from partial archetype matches.

        Called at cadence 100 in sensing_hook. Reads partial matches
        from the archetype engine and spawns hunts for the most
        promising incomplete patterns.

        Returns newly created hunts.
        """
        if self._archetype_engine is None:
            return []

        # Prune expired hunts first
        self._prune_expired()

        try:
            partials = self._archetype_engine.get_partial_matches()
        except Exception:
            logger.debug("Pattern completion: failed to get partial matches", exc_info=True)
            return []

        new_hunts = []
        now = datetime.now()

        for ticker, partial in partials.items():
            # Don't exceed max concurrent hunts
            if len(self._active_hunts) >= self._max_concurrent_hunts:
                break

            # Skip if we already have a hunt for this ticker+archetype
            existing = any(
                h.ticker == ticker and h.archetype_id == partial.archetype_id
                for h in self._active_hunts.values()
            )
            if existing:
                continue

            score = getattr(partial, "match_score", 0.0)
            if score < _MIN_PARTIAL_SCORE or score >= 1.0:
                continue

            # Build list of missing signal domains
            missing = getattr(partial, "missing_signals", [])
            matched = getattr(partial, "matched_signals", [])
            if not missing:
                continue

            hunt = CompletionHunt(
                hunt_id=f"HUNT-{uuid.uuid4().hex[:8]}",
                archetype_id=partial.archetype_id,
                ticker=ticker,
                missing_signals=list(missing),
                created_at=now,
                expires_at=now + timedelta(hours=self._hunt_ttl_hours),
                partial_match_score=score,
                matched_signals=list(matched),
            )
            self._active_hunts[hunt.hunt_id] = hunt
            new_hunts.append(hunt)

            logger.info(
                "Pattern completion: hunt created %s for %s (%s) — missing %s",
                hunt.hunt_id, ticker, partial.archetype_id,
                ", ".join(missing),
            )

        return new_hunts

    def check_completions(self, recent_signals: list) -> List[CompletionEvent]:
        """Check incoming signals against active hunts.

        Called every step during _collect_one() with the batch of
        signals just collected. When a signal matches a hunt's missing
        piece, fire a completion event.

        Returns list of completion events.
        """
        if not self._active_hunts or not recent_signals:
            return []

        events = []

        for sig in recent_signals:
            ticker = getattr(sig, "symbol", "") or getattr(sig, "metadata", {}).get("symbol", "")
            domain = getattr(sig, "domain", "")
            source = getattr(sig, "source", "")

            if not ticker:
                continue

            # Check each hunt for this ticker
            completed_ids = []
            for hunt_id, hunt in self._active_hunts.items():
                if hunt.ticker != ticker:
                    continue

                # Check if signal's domain or source matches any missing signal
                matched_missing = None
                for missing in hunt.missing_signals:
                    if missing in (domain, source):
                        matched_missing = missing
                        break

                if matched_missing is None:
                    continue

                # Completion found!
                hunt.missing_signals.remove(matched_missing)
                hunt.matched_signals.append(matched_missing)

                # Compute new match score
                total_needed = len(hunt.matched_signals) + len(hunt.missing_signals)
                matched_count = len(hunt.matched_signals)
                new_score = matched_count / total_needed if total_needed > 0 else 0.0

                event = CompletionEvent(
                    hunt_id=hunt_id,
                    archetype_id=hunt.archetype_id,
                    ticker=ticker,
                    completing_signal_domain=matched_missing,
                    completion_score=new_score,
                    total_archetype_match=new_score,
                )
                events.append(event)

                logger.info(
                    "Pattern completion: %s found for %s (%s) — score %.2f→%.2f",
                    matched_missing, ticker, hunt.archetype_id,
                    hunt.partial_match_score, new_score,
                )

                # If fully completed, mark for removal
                if not hunt.missing_signals:
                    completed_ids.append(hunt_id)

                # Publish to EventBus
                if self._bus is not None:
                    try:
                        from mae_core.market.channels import CH_PATTERN_COMPLETED
                        self._bus.publish(CH_PATTERN_COMPLETED, {
                            "hunt_id": hunt_id,
                            "archetype_id": hunt.archetype_id,
                            "ticker": ticker,
                            "completing_domain": matched_missing,
                            "completion_score": new_score,
                            "confidence_boost": event.confidence_boost,
                        })
                    except Exception:
                        logger.debug("Pattern completion publish failed", exc_info=True)

            # Remove completed hunts
            for hid in completed_ids:
                del self._active_hunts[hid]

        return events

    def _prune_expired(self) -> int:
        """Remove hunts that have exceeded their TTL."""
        now = datetime.now()
        expired = [
            hid for hid, h in self._active_hunts.items()
            if h.expires_at < now
        ]
        for hid in expired:
            logger.debug("Pattern completion: hunt %s expired", hid)
            del self._active_hunts[hid]
        return len(expired)

    def get_active_hunts(self) -> List[CompletionHunt]:
        """Return list of active hunts."""
        self._prune_expired()
        return list(self._active_hunts.values())

    def get_statistics(self) -> dict:
        """Return pattern completion stats."""
        return {
            "active_hunts": len(self._active_hunts),
            "max_concurrent": self._max_concurrent_hunts,
            "hunt_ttl_hours": self._hunt_ttl_hours,
            "hunts": [
                {
                    "hunt_id": h.hunt_id,
                    "archetype": h.archetype_id,
                    "ticker": h.ticker,
                    "missing": h.missing_signals,
                    "score": h.partial_match_score,
                }
                for h in self._active_hunts.values()
            ],
        }
