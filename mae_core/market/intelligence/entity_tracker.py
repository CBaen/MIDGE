#!/usr/bin/env python3
"""Entity Tracker — cross-pattern actor profiling.

Tracks individual actors (insiders, politicians, funds, activists) across
all pattern templates and predictions.  When the SAME person or fund
appears repeatedly in high-win-rate patterns, that is structural signal —
not coincidence.

MIDGE surfaces this as: "When Senator X trades tech stocks, there is a
72% win rate in the following 14 days across 23 historical instances."

Biological analogy: this is MIDGE's social memory — who the influential
actors are, what their track record looks like, and when their fingerprint
appears in a live pattern.

Key concepts:
  - EntityProfile: everything known about one actor
  - track_entity(): called when an actor appears in a signal/pattern
  - get_top_actors(): who appears most often with the best outcomes?
  - get_entity_track_record(): full statistical profile for one actor

Data flow:
  PrecursorSignal.entity_metadata carries actor names
    → EntityTracker.track_entity() called during pattern matching
    → profiles updated with each new appearance
    → EntityAuditor / PatternAuditor can query diversity per template

Persistence: JSONL at data/midge/entity_profiles.jsonl
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "midge" / "entity_profiles.jsonl"

# Recognised entity types
ENTITY_TYPES = frozenset({"insider", "politician", "fund", "activist", "unknown"})

# Maximum notable trades retained per profile
_MAX_NOTABLE_TRADES = 5


@dataclass
class EntityAppearance:
    """A single appearance of an entity in a pattern or signal."""
    template_id: str
    ticker: str
    appeared_on: str          # ISO date
    outcome: str = "pending"  # "win", "loss", "pending"
    move_pct: float = 0.0     # Measured price move (% if graded)
    source: str = ""          # e.g. "sec_form4", "congressional"
    notes: str = ""


@dataclass
class EntityProfile:
    """Full historical profile of one market actor.

    Aggregates all appearances across patterns and tracks their
    predictive value.  An entity with high win_rate across many
    independent templates is a structural edge signal.
    """
    name: str
    entity_type: str                        # "insider", "politician", "fund", "activist"
    patterns_involved: List[str] = field(default_factory=list)   # template_ids
    total_appearances: int = 0
    wins: int = 0
    losses: int = 0
    total_move_pct: float = 0.0
    notable_trades: List[dict] = field(default_factory=list)     # Top 5
    first_seen: str = ""
    last_seen: str = ""
    tickers: List[str] = field(default_factory=list)             # All symbols seen on

    @property
    def win_rate(self) -> float:
        graded = self.wins + self.losses
        return round(self.wins / graded, 4) if graded > 0 else 0.0

    @property
    def avg_move_pct(self) -> float:
        graded = self.wins + self.losses
        return round(self.total_move_pct / graded, 2) if graded > 0 else 0.0

    @property
    def unique_templates(self) -> int:
        return len(set(self.patterns_involved))

    @property
    def unique_tickers(self) -> int:
        return len(set(self.tickers))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "patterns_involved": self.patterns_involved,
            "total_appearances": self.total_appearances,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_move_pct": self.avg_move_pct,
            "total_move_pct": self.total_move_pct,
            "notable_trades": self.notable_trades,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "tickers": sorted(set(self.tickers)),
            "unique_templates": self.unique_templates,
            "unique_tickers": self.unique_tickers,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EntityProfile":
        return cls(
            name=d.get("name", ""),
            entity_type=d.get("entity_type", "unknown"),
            patterns_involved=d.get("patterns_involved", []),
            total_appearances=d.get("total_appearances", 0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            total_move_pct=d.get("total_move_pct", 0.0),
            notable_trades=d.get("notable_trades", []),
            first_seen=d.get("first_seen", ""),
            last_seen=d.get("last_seen", ""),
            tickers=d.get("tickers", []),
        )


class EntityTracker:
    """Tracks and profiles individual market actors across all patterns.

    Usage:
        tracker = EntityTracker()
        # When a signal fires with entity metadata:
        tracker.track_entity("Nancy Pelosi", "politician", "tmpl_abc123",
                             ticker="NVDA", appeared_on="2026-02-14")
        # When the prediction is graded:
        tracker.update_outcome("Nancy Pelosi", "tmpl_abc123", "win", 8.5)
        # Query:
        profile = tracker.get_profile("Nancy Pelosi")
        top_actors = tracker.get_top_actors(limit=10)
    """

    def __init__(self, path: str = None):
        self._path = Path(path) if path else _DEFAULT_PATH
        self._profiles: Dict[str, EntityProfile] = {}   # name -> EntityProfile
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def track_entity(
        self,
        name: str,
        entity_type: str,
        template_id: str,
        ticker: str = "",
        appeared_on: str = "",
        source: str = "",
        outcome: str = "pending",
        move_pct: float = 0.0,
        notes: str = "",
    ) -> EntityProfile:
        """Record an entity appearance in a pattern.

        Creates a new profile if this is the first time we've seen the actor.
        Always idempotent — calling twice with the same appearance is fine
        (it will not double-count; template_id deduplication is per-profile).

        Args:
            name: Display name (e.g. "Nancy Pelosi", "Cathie Wood", "Tim Cook").
            entity_type: One of "insider", "politician", "fund", "activist".
            template_id: The pattern template this appearance is associated with.
            ticker: The symbol involved.
            appeared_on: ISO date string.
            source: Original signal source (e.g. "sec_form4", "congressional").
            outcome: "win", "loss", or "pending".
            move_pct: If graded, the measured price move.
            notes: Any additional context.

        Returns:
            The updated EntityProfile.
        """
        if not name:
            return EntityProfile(name="", entity_type="unknown")

        entity_type = entity_type if entity_type in ENTITY_TYPES else "unknown"
        today = date.today().isoformat()
        appearance_date = appeared_on or today

        if name not in self._profiles:
            self._profiles[name] = EntityProfile(
                name=name,
                entity_type=entity_type,
                first_seen=appearance_date,
                last_seen=appearance_date,
            )

        profile = self._profiles[name]
        profile.total_appearances += 1
        profile.last_seen = max(profile.last_seen, appearance_date) if profile.last_seen else appearance_date
        profile.first_seen = min(profile.first_seen, appearance_date) if profile.first_seen else appearance_date

        # Track templates (allow duplicates — a person can appear multiple times in one template)
        profile.patterns_involved.append(template_id)

        # Track tickers
        if ticker:
            profile.tickers.append(ticker)

        # Update outcome stats immediately if graded
        if outcome == "win":
            profile.wins += 1
            profile.total_move_pct += move_pct
        elif outcome == "loss":
            profile.losses += 1
            profile.total_move_pct += move_pct  # Can be negative for losses

        # Notable trades: keep the 5 most significant by |move_pct|
        if outcome != "pending" and abs(move_pct) > 0:
            trade_entry = {
                "template_id": template_id,
                "ticker": ticker,
                "appeared_on": appearance_date,
                "outcome": outcome,
                "move_pct": move_pct,
                "source": source,
            }
            profile.notable_trades.append(trade_entry)
            # Sort by |move_pct| descending, keep top 5
            profile.notable_trades.sort(key=lambda t: abs(t.get("move_pct", 0)), reverse=True)
            profile.notable_trades = profile.notable_trades[:_MAX_NOTABLE_TRADES]

        return profile

    def update_outcome(
        self,
        name: str,
        template_id: str,
        outcome: str,
        move_pct: float = 0.0,
    ) -> None:
        """Update the outcome for the most recent pending appearance of this entity.

        Called when a prediction associated with this entity is graded.

        Args:
            name: Entity name.
            template_id: Template associated with the prediction.
            outcome: "win" or "loss".
            move_pct: Measured price move.
        """
        if name not in self._profiles:
            return
        if outcome not in ("win", "loss"):
            return

        profile = self._profiles[name]
        if outcome == "win":
            profile.wins += 1
            profile.total_move_pct += move_pct
        else:
            profile.losses += 1
            profile.total_move_pct += move_pct

    def get_profile(self, name: str) -> Optional[EntityProfile]:
        """Full profile of this entity across all patterns.

        Returns None if entity is unknown.
        """
        return self._profiles.get(name)

    def get_top_actors(self, limit: int = 20, min_appearances: int = 3) -> List[dict]:
        """Most frequently appearing entities across all patterns.

        Sorted by total_appearances descending.  Entities with fewer than
        min_appearances are excluded.

        Args:
            limit: Maximum number of results.
            min_appearances: Minimum appearance count to include.

        Returns:
            List of profile dicts.
        """
        results = [
            p.to_dict() for p in self._profiles.values()
            if p.total_appearances >= min_appearances
        ]
        results.sort(key=lambda p: p["total_appearances"], reverse=True)
        return results[:limit]

    def get_entity_track_record(self, name: str) -> dict:
        """When this person/fund is involved, what happens?

        Returns a statistical summary including win rate, average move,
        timing, and regime breakdown.

        Args:
            name: Entity name.

        Returns:
            dict with keys:
              name, entity_type, total_appearances, win_rate, avg_move_pct,
              unique_templates, unique_tickers, first_seen, last_seen,
              notable_trades, verdict (plain-language assessment)
        """
        profile = self._profiles.get(name)
        if not profile:
            return {
                "name": name,
                "found": False,
                "total_appearances": 0,
            }

        # Verdict logic
        graded = profile.wins + profile.losses
        if graded < 3:
            verdict = "insufficient data"
        elif profile.win_rate >= 0.70:
            verdict = "strong signal — high win rate"
        elif profile.win_rate >= 0.55:
            verdict = "positive edge — above-average win rate"
        elif profile.win_rate >= 0.45:
            verdict = "neutral — no significant edge"
        else:
            verdict = "negative signal — below-average win rate (contrarian?)"

        result = profile.to_dict()
        result["found"] = True
        result["graded_appearances"] = graded
        result["verdict"] = verdict
        return result

    def get_entities_for_template(self, template_id: str, min_appearances: int = 1) -> List[dict]:
        """Return all entities that appear in a specific template.

        Useful for PatternAuditor entity diversity scoring.

        Args:
            template_id: Template to query.
            min_appearances: Minimum appearances in this template.

        Returns:
            List of {name, entity_type, appearances_in_template, win_rate}
        """
        results = []
        for name, profile in self._profiles.items():
            appearances_in_tmpl = profile.patterns_involved.count(template_id)
            if appearances_in_tmpl >= min_appearances:
                results.append({
                    "name": name,
                    "entity_type": profile.entity_type,
                    "appearances_in_template": appearances_in_tmpl,
                    "win_rate": profile.win_rate,
                    "total_appearances": profile.total_appearances,
                })
        results.sort(key=lambda x: x["appearances_in_template"], reverse=True)
        return results

    def get_statistics(self) -> dict:
        """Summary statistics for reporting."""
        total = len(self._profiles)
        by_type: dict = defaultdict(int)
        for p in self._profiles.values():
            by_type[p.entity_type] += 1

        high_confidence = sum(
            1 for p in self._profiles.values()
            if (p.wins + p.losses) >= 5 and p.win_rate >= 0.65
        )

        return {
            "total_entities": total,
            "by_type": dict(by_type),
            "high_confidence_actors": high_confidence,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all entity profiles to JSONL."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                for profile in self._profiles.values():
                    f.write(json.dumps(profile.to_dict(), default=str) + "\n")
            tmp.replace(self._path)
            logger.debug(
                "EntityTracker saved %d entity profiles", len(self._profiles)
            )
        except Exception:
            logger.warning("EntityTracker save failed", exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _load(self) -> None:
        """Load entity profiles from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = d.get("name", "")
                    if name:
                        self._profiles[name] = EntityProfile.from_dict(d)
            logger.info(
                "EntityTracker loaded %d entity profiles", len(self._profiles)
            )
        except Exception:
            logger.warning("EntityTracker load failed", exc_info=True)
