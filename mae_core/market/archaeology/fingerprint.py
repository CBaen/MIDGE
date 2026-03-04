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

    @property
    def template_key(self) -> str:
        """Key for grouping into PatternTemplates: direction + domain_signature."""
        return f"{self.direction}:{self.domain_signature}"


@dataclass
class TemplateInstance:
    """One observation of a PatternTemplate on a specific symbol."""
    symbol: str
    move_date: str
    move_pct: float
    fingerprint_id: str
    regime: str = "default"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "move_date": self.move_date,
            "move_pct": self.move_pct,
            "fingerprint_id": self.fingerprint_id,
            "regime": self.regime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TemplateInstance:
        return cls(
            symbol=d["symbol"],
            move_date=d["move_date"],
            move_pct=d.get("move_pct", 0.0),
            fingerprint_id=d.get("fingerprint_id", ""),
            regime=d.get("regime", "default"),
        )


@dataclass
class PatternTemplate:
    """Symbol-agnostic pattern — the transferable truth across the market.

    A PatternTemplate groups MoveFingerprints that share the same
    direction + domain_signature. "insider+macro+technical bullish"
    is one template — it doesn't matter if it was NVDA, AAPL, or AMD.

    Cross-validation: a template seen on 3+ different symbols is
    structurally stronger than one seen on a single symbol. The market
    is telling us this pattern is REAL, not a coincidence.
    """
    template_id: str                         # hash(direction + domain_signature)
    direction: str                           # "bullish" or "bearish"
    domain_signature: str                    # "insider+macro+technical"
    domains: list[str] = field(default_factory=list)  # sorted domain list
    lag_profile_normalized: dict[str, float] = field(default_factory=dict)
    source_examples: dict[str, list[str]] = field(default_factory=dict)

    # Cross-symbol validation
    instances: list[TemplateInstance] = field(default_factory=list)
    symbols_seen: list[str] = field(default_factory=list)  # list for JSON serialization
    n_instances: int = 0
    avg_move_pct: float = 0.0

    # Stats (updated as outcomes are graded)
    wins: int = 0
    losses: int = 0
    created_at: str = ""

    def __post_init__(self):
        if not self.template_id:
            self.template_id = self._compute_id()
        if not self.domains and self.domain_signature:
            self.domains = self.domain_signature.split("+")
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def _compute_id(self) -> str:
        raw = f"{self.direction}:{self.domain_signature}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def unique_symbols(self) -> set[str]:
        return set(self.symbols_seen)

    @property
    def cross_validated(self) -> bool:
        """Template is cross-validated if seen on 3+ different symbols."""
        return len(self.unique_symbols) >= 3

    @property
    def confidence_multiplier(self) -> float:
        """Cross-symbol validation boosts confidence.

        1 symbol = 1.0 (no boost, might be coincidence)
        3 symbols = 1.15 (pattern recurring across stocks)
        5 symbols = 1.3 (strong cross-validation)
        10+ symbols = 1.5 (market-wide pattern)
        """
        n = len(self.unique_symbols)
        if n >= 10:
            return 1.5
        if n >= 5:
            return 1.3
        if n >= 3:
            return 1.15
        return 1.0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total

    @property
    def domain_set(self) -> set[str]:
        return set(self.domains)

    def add_instance(self, fingerprint: MoveFingerprint) -> None:
        """Register a new fingerprint observation for this template."""
        inst = TemplateInstance(
            symbol=fingerprint.symbol,
            move_date=fingerprint.move_date,
            move_pct=fingerprint.move_pct,
            fingerprint_id=fingerprint.fingerprint_id,
            regime=fingerprint.regime,
        )
        self.instances.append(inst)
        if fingerprint.symbol not in self.symbols_seen:
            self.symbols_seen.append(fingerprint.symbol)
        self.n_instances = len(self.instances)
        self._recompute_averages()

    def _recompute_averages(self) -> None:
        if not self.instances:
            return
        self.avg_move_pct = sum(abs(i.move_pct) for i in self.instances) / len(self.instances)

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "direction": self.direction,
            "domain_signature": self.domain_signature,
            "domains": self.domains,
            "lag_profile_normalized": self.lag_profile_normalized,
            "source_examples": self.source_examples,
            "instances": [i.to_dict() for i in self.instances],
            "symbols_seen": self.symbols_seen,
            "n_instances": self.n_instances,
            "avg_move_pct": self.avg_move_pct,
            "wins": self.wins,
            "losses": self.losses,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternTemplate:
        instances = [TemplateInstance.from_dict(i) for i in d.get("instances", [])]
        return cls(
            template_id=d.get("template_id", ""),
            direction=d["direction"],
            domain_signature=d["domain_signature"],
            domains=d.get("domains", []),
            lag_profile_normalized=d.get("lag_profile_normalized", {}),
            source_examples=d.get("source_examples", {}),
            instances=instances,
            symbols_seen=d.get("symbols_seen", []),
            n_instances=d.get("n_instances", 0),
            avg_move_pct=d.get("avg_move_pct", 0.0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            created_at=d.get("created_at", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_fingerprint(cls, fingerprint: MoveFingerprint) -> PatternTemplate:
        """Create a new template from the first fingerprint observation."""
        # Compute normalized lag profile
        total_signals = sum(fingerprint.lag_profile.values()) or 1
        normalized = {
            bucket: count / total_signals
            for bucket, count in fingerprint.lag_profile.items()
        }

        # Build source examples by domain
        source_examples: dict[str, list[str]] = {}
        for sig in fingerprint.precursor_signals:
            if sig.domain not in source_examples:
                source_examples[sig.domain] = []
            if sig.source not in source_examples[sig.domain]:
                source_examples[sig.domain].append(sig.source)

        template = cls(
            template_id="",  # auto-computed
            direction=fingerprint.direction,
            domain_signature=fingerprint.domain_signature,
            domains=sorted(set(s.domain for s in fingerprint.precursor_signals)),
            lag_profile_normalized=normalized,
            source_examples=source_examples,
        )
        template.add_instance(fingerprint)
        return template
