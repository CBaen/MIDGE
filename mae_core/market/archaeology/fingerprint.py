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
    entity_metadata: dict = field(default_factory=dict)  # WHO: insider names, fund names, congress members

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "domain": self.domain,
            "direction": self.direction,
            "strength": self.strength,
            "lag_days": self.lag_days,
            "lag_bucket": self.lag_bucket,
            "signal_id": self.signal_id,
            "entity_metadata": self.entity_metadata,
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
            entity_metadata=d.get("entity_metadata", {}),
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
    lag_profile_raw: dict[str, int] = field(default_factory=dict)  # accumulated bucket counts
    lag_profile_normalized: dict[str, float] = field(default_factory=dict)
    source_examples: dict[str, list[str]] = field(default_factory=dict)

    # Cross-symbol validation
    instances: list[TemplateInstance] = field(default_factory=list)
    symbols_seen: list[str] = field(default_factory=list)  # list for JSON serialization
    n_instances: int = 0
    avg_move_pct: float = 0.0

    # Incremental stats
    _move_pct_sum: float = 0.0  # Sum of abs(move_pct) for incremental avg

    # Entity intelligence — WHO keeps appearing in these patterns
    recurring_entities: dict = field(default_factory=dict)  # {entity_name: {count, role, domains}}
    entity_weight_factors: dict = field(default_factory=dict)  # CEO > director, committee chair > backbencher

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

    # Bucket midpoints for computing expected move window
    _LAG_BUCKET_CENTERS = {
        "immediate": 1.0, "short": 4.0, "medium": 8.0,
        "long": 15.0, "extended": 25.0,
    }

    @property
    def expected_move_window_days(self) -> int:
        """Expected days until the move, derived from historical lag data.

        The lag_profile tells us how far BEFORE the move the signals appeared.
        The weighted mean gives us how long to wait after detecting signals.
        """
        if not self.lag_profile_normalized:
            return 14  # fallback
        weighted_sum = 0.0
        weight_total = 0.0
        for bucket, fraction in self.lag_profile_normalized.items():
            center = self._LAG_BUCKET_CENTERS.get(bucket, 8.0)
            weighted_sum += center * fraction
            weight_total += fraction
        if weight_total == 0:
            return 14
        mean_days = weighted_sum / weight_total
        window = int(mean_days * 1.2) + 1  # +20% buffer
        return max(3, min(30, window))

    # Max instances to keep in memory/serialized (recent only — fingerprints are the archive)
    _MAX_INSTANCES = 200

    # Entity role weight — higher-authority roles carry more weight
    _ENTITY_ROLE_WEIGHTS = {
        "insider_name": 1.0,    # base: insider
        "filer_name": 0.9,      # institutional filer
        "fund_name": 0.9,       # hedge fund
        "representative": 0.8,  # congress member
        "filer_title": 0.0,     # title, not a name — don't track as entity
        "party": 0.0,
        "committee": 0.5,       # committee association
        "delta_owned_pct": 0.0,
        "bill_title": 0.0,
        "policy_area": 0.0,
        "amount_range": 0.0,
        "total_value": 0.0,
    }

    # Higher weights for CEO/Chairman vs director-level roles
    _TITLE_WEIGHT_MAP = {
        "ceo": 1.5, "chief executive": 1.5, "chairman": 1.4,
        "cfo": 1.3, "chief financial": 1.3,
        "coo": 1.2, "president": 1.2,
        "director": 1.0, "officer": 1.0,
        "vp": 0.9, "vice president": 0.9,
        "10% owner": 1.1,
    }

    def _compute_entity_weight(self, entity_name: str, role_key: str, filer_title: str = "") -> float:
        """Compute authority weight for an entity based on role and title."""
        base = self._ENTITY_ROLE_WEIGHTS.get(role_key, 1.0)
        if base == 0.0:
            return 0.0
        # Boost for high-authority titles
        title_lower = filer_title.lower() if filer_title else ""
        for keyword, multiplier in self._TITLE_WEIGHT_MAP.items():
            if keyword in title_lower:
                return base * multiplier
        return base

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
        if len(self.instances) > self._MAX_INSTANCES:
            self.instances = self.instances[-self._MAX_INSTANCES:]
        if fingerprint.symbol not in self.symbols_seen:
            self.symbols_seen.append(fingerprint.symbol)
        # Accumulate raw lag counts from this fingerprint
        for bucket, count in fingerprint.lag_profile.items():
            self.lag_profile_raw[bucket] = self.lag_profile_raw.get(bucket, 0) + count
        self.n_instances += 1
        # Incremental avg — O(1) instead of O(N)
        self._move_pct_sum += abs(fingerprint.move_pct)
        self.avg_move_pct = self._move_pct_sum / self.n_instances
        # Recompute normalized lag from accumulated raw counts
        if self.lag_profile_raw:
            total = sum(self.lag_profile_raw.values()) or 1
            self.lag_profile_normalized = {
                bucket: count / total
                for bucket, count in self.lag_profile_raw.items()
            }
        # Accumulate entity names from precursor signals
        for precursor in fingerprint.precursor_signals:
            em = precursor.entity_metadata
            if not em:
                continue
            filer_title = str(em.get("filer_title", ""))
            for key in ("insider_name", "representative", "filer_name", "fund_name"):
                name = em.get(key)
                if not name:
                    continue
                name = str(name).strip()
                if not name:
                    continue
                weight = self._compute_entity_weight(name, key, filer_title)
                if weight == 0.0:
                    continue
                if name not in self.recurring_entities:
                    self.recurring_entities[name] = {
                        "count": 0,
                        "role": key,
                        "domains": set(),
                    }
                self.recurring_entities[name]["count"] += 1
                self.recurring_entities[name]["domains"].add(precursor.domain)
                # Store highest weight seen for this entity
                if name not in self.entity_weight_factors:
                    self.entity_weight_factors[name] = weight
                else:
                    self.entity_weight_factors[name] = max(
                        self.entity_weight_factors[name], weight
                    )

    def to_dict(self) -> dict:
        # Serialize recurring_entities: convert set to list for JSON
        serialized_entities: dict = {}
        for name, info in self.recurring_entities.items():
            entry = dict(info)
            if isinstance(entry.get("domains"), set):
                entry["domains"] = sorted(entry["domains"])
            serialized_entities[name] = entry
        return {
            "template_id": self.template_id,
            "direction": self.direction,
            "domain_signature": self.domain_signature,
            "domains": self.domains,
            "lag_profile_raw": self.lag_profile_raw,
            "lag_profile_normalized": self.lag_profile_normalized,
            "source_examples": self.source_examples,
            "instances": [i.to_dict() for i in self.instances],
            "symbols_seen": self.symbols_seen,
            "n_instances": self.n_instances,
            "avg_move_pct": self.avg_move_pct,
            "_move_pct_sum": self._move_pct_sum,
            "recurring_entities": serialized_entities,
            "entity_weight_factors": self.entity_weight_factors,
            "wins": self.wins,
            "losses": self.losses,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternTemplate:
        instances = [TemplateInstance.from_dict(i) for i in d.get("instances", [])]
        n_instances = d.get("n_instances", 0)
        avg_move_pct = d.get("avg_move_pct", 0.0)
        # Backward compat: recompute _move_pct_sum if missing
        move_pct_sum = d.get("_move_pct_sum", 0.0)
        if move_pct_sum == 0.0 and n_instances > 0 and avg_move_pct > 0:
            move_pct_sum = avg_move_pct * n_instances
        # Deserialize recurring_entities: restore domain lists as sets
        raw_entities = d.get("recurring_entities", {})
        recurring_entities: dict = {}
        for name, info in raw_entities.items():
            entry = dict(info)
            domains = entry.get("domains", [])
            entry["domains"] = set(domains) if isinstance(domains, list) else domains
            recurring_entities[name] = entry
        return cls(
            template_id=d.get("template_id", ""),
            direction=d["direction"],
            domain_signature=d["domain_signature"],
            domains=d.get("domains", []),
            lag_profile_raw=d.get("lag_profile_raw", {}),
            lag_profile_normalized=d.get("lag_profile_normalized", {}),
            source_examples=d.get("source_examples", {}),
            instances=instances,
            symbols_seen=d.get("symbols_seen", []),
            n_instances=n_instances,
            avg_move_pct=avg_move_pct,
            _move_pct_sum=move_pct_sum,
            recurring_entities=recurring_entities,
            entity_weight_factors=d.get("entity_weight_factors", {}),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            created_at=d.get("created_at", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_fingerprint(cls, fingerprint: MoveFingerprint) -> PatternTemplate:
        """Create a new template from the first fingerprint observation."""
        # Build source examples by domain
        source_examples: dict[str, list[str]] = {}
        for sig in fingerprint.precursor_signals:
            if sig.domain not in source_examples:
                source_examples[sig.domain] = []
            if sig.source not in source_examples[sig.domain]:
                source_examples[sig.domain].append(sig.source)

        # lag_profile_raw and lag_profile_normalized start empty;
        # add_instance() will accumulate the first fingerprint's lag data
        # and _recompute_averages() will normalize it.
        template = cls(
            template_id="",  # auto-computed
            direction=fingerprint.direction,
            domain_signature=fingerprint.domain_signature,
            domains=sorted(set(s.domain for s in fingerprint.precursor_signals)),
            source_examples=source_examples,
        )
        template.add_instance(fingerprint)
        return template
