"""Fingerprint dataclasses — the DNA of a historical market move.

Three levels of abstraction:

1. PrecursorSignal — a single signal observed before a move
2. MoveFingerprint — the full signal configuration before ONE specific move
                     (symbol-bound: "what preceded NVDA's 10% move on 2024-01-15")
3. PatternTemplate — the symbol-AGNOSTIC pattern extracted from many fingerprints
                     (cross-validated: "insider+macro+technical bullish seen on 47 symbols")

PatternTemplate is the key abstraction. A pattern that only works for NVDA is
worthless. Real patterns transfer across the entire market. Patterns validate
each other through cross-symbol observation. The PatternWatcher matches live
signals against templates, not individual fingerprints.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


# Lag buckets: how far before the move the signal appeared
LAG_BUCKETS = {
    "immediate": (0, 2),    # 0-2 days before
    "short": (3, 5),        # 3-5 days before
    "medium": (6, 10),      # 6-10 days before
    "long": (11, 20),       # 11-20 days before
    "extended": (21, 30),   # 21-30 days before
}


def lag_bucket_for_days(days: int) -> str:
    """Return the lag bucket name for a given number of days before the move."""
    for name, (lo, hi) in LAG_BUCKETS.items():
        if lo <= days <= hi:
            return name
    if days > 30:
        return "extended"
    return "immediate"


@dataclass
class PrecursorSignal:
    """A single signal observed before a historical price move."""
    source: str             # e.g. "sec_form4", "fred_macro"
    domain: str             # e.g. "insider", "macro", "price"
    direction: str          # "bullish" or "bearish"
    strength: float         # Signal strength (0-1)
    lag_days: int           # Days before the move
    lag_bucket: str         # Categorized lag ("immediate", "short", etc.)
    signal_id: str = ""     # Original signal ID from archive

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "domain": self.domain,
            "direction": self.direction,
            "strength": self.strength,
            "lag_days": self.lag_days,
            "lag_bucket": self.lag_bucket,
            "signal_id": self.signal_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PrecursorSignal:
        return cls(
            source=d["source"],
            domain=d["domain"],
            direction=d["direction"],
            strength=d.get("strength", 0.0),
            lag_days=d.get("lag_days", 0),
            lag_bucket=d.get("lag_bucket", "immediate"),
            signal_id=d.get("signal_id", ""),
        )


@dataclass
class MoveFingerprint:
    """The signature of signals that preceded a known historical price move.

    This is what the Pattern Watcher matches against live signals.
    """
    fingerprint_id: str                      # Unique ID (hash of symbol+direction+move_date)
    symbol: str                              # e.g. "NVDA"
    direction: str                           # "bullish" or "bearish"
    move_date: str                           # ISO date of the move (YYYY-MM-DD)
    move_pct: float                          # Magnitude of the move (%)
    regime: str = "default"                  # Market regime at time of move
    lookback_days: int = 30                  # How far back we looked for precursors
    precursor_signals: list[PrecursorSignal] = field(default_factory=list)
    domain_signature: str = ""               # Sorted domains present (e.g. "events+insider+macro")
    lag_profile: dict[str, int] = field(default_factory=dict)  # lag_bucket -> signal count
    created_at: str = ""                     # When this fingerprint was created

    # Stats (updated by pattern library as outcomes are graded)
    wins: int = 0
    losses: int = 0
    total_activations: int = 0
    last_activation: str = ""

    def __post_init__(self):
        if not self.fingerprint_id:
            self.fingerprint_id = self._compute_id()
        if not self.domain_signature:
            self.domain_signature = self._compute_domain_signature()
        if not self.lag_profile:
            self.lag_profile = self._compute_lag_profile()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def _compute_id(self) -> str:
        raw = f"{self.symbol}:{self.direction}:{self.move_date}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compute_domain_signature(self) -> str:
        domains = sorted(set(s.domain for s in self.precursor_signals))
        return "+".join(domains)

    def _compute_lag_profile(self) -> dict[str, int]:
        profile: dict[str, int] = {}
        for s in self.precursor_signals:
            profile[s.lag_bucket] = profile.get(s.lag_bucket, 0) + 1
        return profile

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def source_set(self) -> set[str]:
        """Set of unique sources in this fingerprint — used for independence checks."""
        return set(s.source for s in self.precursor_signals)

    def to_dict(self) -> dict:
        return {
            "fingerprint_id": self.fingerprint_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "move_date": self.move_date,
            "move_pct": self.move_pct,
            "regime": self.regime,
            "lookback_days": self.lookback_days,
            "precursor_signals": [s.to_dict() for s in self.precursor_signals],
            "domain_signature": self.domain_signature,
            "lag_profile": self.lag_profile,
            "created_at": self.created_at,
            "wins": self.wins,
            "losses": self.losses,
            "total_activations": self.total_activations,
            "last_activation": self.last_activation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MoveFingerprint:
        precursors = [PrecursorSignal.from_dict(p) for p in d.get("precursor_signals", [])]
        return cls(
            fingerprint_id=d.get("fingerprint_id", ""),
            symbol=d["symbol"],
            direction=d["direction"],
            move_date=d["move_date"],
            move_pct=d.get("move_pct", 0.0),
            regime=d.get("regime", "default"),
            lookback_days=d.get("lookback_days", 30),
            precursor_signals=precursors,
            domain_signature=d.get("domain_signature", ""),
            lag_profile=d.get("lag_profile", {}),
            created_at=d.get("created_at", ""),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            total_activations=d.get("total_activations", 0),
            last_activation=d.get("last_activation", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
