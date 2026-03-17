"""Pattern story dataclasses — narrative structure for pattern intelligence.

A PatternStory is the human-readable, statistically-grounded account of a
recurring market pattern. It answers: What is this pattern? Who keeps appearing
in it? When does it play out? How reliable is it?

Three levels:
  PatternChapter — one phase of the story (early / mid / late signals)
  PatternStory   — the complete narrative for one PatternTemplate

Stories are built by PatternStoryBuilder and saved to
data/midge/pattern_stories.jsonl for consumption by daily_narrative.py,
plain_language.py, and any alert that wants historical context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class PatternChapter:
    """One phase in the pattern's lifecycle.

    A story is divided into chapters by temporal order: which domains tend to
    fire first (earliest signals), which confirm in the middle, and which appear
    just before the move resolves.
    """

    chapter_number: int = 0                  # 1 = first / earliest, 2 = middle, 3 = late
    phase_name: str = ""                     # "early warning" / "confirmation" / "resolution"
    domains: list[str] = field(default_factory=list)   # Domains active in this phase
    avg_lag_days: float = 0.0                # Average days before move for this phase
    description: str = ""                    # Plain-English description of this phase
    signal_count: int = 0                    # How many signals in this phase across all instances

    def to_dict(self) -> dict:
        return {
            "chapter_number": self.chapter_number,
            "phase_name": self.phase_name,
            "domains": self.domains,
            "avg_lag_days": round(self.avg_lag_days, 1),
            "description": self.description,
            "signal_count": self.signal_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternChapter:
        return cls(
            chapter_number=d.get("chapter_number", 0),
            phase_name=d.get("phase_name", ""),
            domains=d.get("domains", []),
            avg_lag_days=d.get("avg_lag_days", 0.0),
            description=d.get("description", ""),
            signal_count=d.get("signal_count", 0),
        )


@dataclass
class PatternStory:
    """The complete narrative for one PatternTemplate.

    Built by PatternStoryBuilder from the template's fingerprints, this
    carries everything needed to explain a pattern in plain English:
    - Who the recurring characters are (executives, congress members, funds)
    - What sequence events unfold in
    - How statistically reliable it is
    - What regimes it performs best in
    - A one-sentence human-readable title and summary
    """

    # Identity
    story_id: str = ""               # Same as template_id
    template_id: str = ""
    direction: str = ""              # "bullish" / "bearish"
    domain_signature: str = ""       # e.g. "insider+macro+technical"

    # Title and summary (auto-generated)
    title: str = ""                  # e.g. "CEO Buying Before Tech Macro Surge"
    summary: str = ""                # 1-2 sentence plain-English summary

    # Narrative chapters (temporal phases)
    chapters: list[PatternChapter] = field(default_factory=list)

    # Recurring characters
    recurring_entities: dict = field(default_factory=dict)  # {name: {count, role, domains, weight}}
    top_entities: list[str] = field(default_factory=list)   # Sorted by count, top 5

    # Statistical foundation
    n_instances: int = 0             # Total fingerprints in template
    n_symbols: int = 0               # Unique symbols observed on
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    ci_lower: float = 0.0            # Clopper-Pearson 95% CI lower bound
    ci_upper: float = 1.0            # Clopper-Pearson 95% CI upper bound
    p_value: float = 1.0             # Binomial test p-value vs 50% null
    is_significant: bool = False     # p < 0.05

    # Timing
    avg_move_pct: float = 0.0        # Average magnitude of the move
    expected_window_days: int = 14   # Expected days until move resolves
    dominant_lag_bucket: str = ""    # Most common lag bucket ("short", "medium", etc.)

    # Regime performance
    regime_win_rates: dict = field(default_factory=dict)  # {regime: win_rate}
    best_regime: str = ""            # Regime with highest win rate

    # Source coverage
    source_examples: dict = field(default_factory=dict)  # {domain: [source_names]}

    # Metadata
    built_at: str = ""
    fingerprint_ids_sampled: list[str] = field(default_factory=list)  # IDs used to build story

    def __post_init__(self):
        if not self.story_id and self.template_id:
            self.story_id = self.template_id
        if not self.built_at:
            self.built_at = datetime.now().isoformat()

    @property
    def has_statistical_signal(self) -> bool:
        """True if there are enough outcomes to trust the win rate."""
        return self.wins + self.losses >= 5

    @property
    def quality_tier(self) -> str:
        """Human label for how strong this story is.

        Based on cross-symbol count and statistical significance.
        """
        if self.n_symbols >= 10 and self.is_significant:
            return "high"
        if self.n_symbols >= 3 and self.n_instances >= 10:
            return "medium"
        return "low"

    def to_dict(self) -> dict:
        # Serialize recurring_entities: set → list
        serialized_entities: dict = {}
        for name, info in self.recurring_entities.items():
            entry = dict(info)
            if isinstance(entry.get("domains"), set):
                entry["domains"] = sorted(entry["domains"])
            serialized_entities[name] = entry
        return {
            "story_id": self.story_id,
            "template_id": self.template_id,
            "direction": self.direction,
            "domain_signature": self.domain_signature,
            "title": self.title,
            "summary": self.summary,
            "chapters": [c.to_dict() for c in self.chapters],
            "recurring_entities": serialized_entities,
            "top_entities": self.top_entities,
            "n_instances": self.n_instances,
            "n_symbols": self.n_symbols,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "avg_move_pct": round(self.avg_move_pct, 2),
            "expected_window_days": self.expected_window_days,
            "dominant_lag_bucket": self.dominant_lag_bucket,
            "regime_win_rates": self.regime_win_rates,
            "best_regime": self.best_regime,
            "source_examples": self.source_examples,
            "built_at": self.built_at,
            "fingerprint_ids_sampled": self.fingerprint_ids_sampled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PatternStory:
        chapters = [PatternChapter.from_dict(c) for c in d.get("chapters", [])]
        # Deserialize recurring_entities: list → set for domains
        raw_entities = d.get("recurring_entities", {})
        recurring_entities: dict = {}
        for name, info in raw_entities.items():
            entry = dict(info)
            domains = entry.get("domains", [])
            entry["domains"] = set(domains) if isinstance(domains, list) else domains
            recurring_entities[name] = entry
        return cls(
            story_id=d.get("story_id", ""),
            template_id=d.get("template_id", ""),
            direction=d.get("direction", ""),
            domain_signature=d.get("domain_signature", ""),
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            chapters=chapters,
            recurring_entities=recurring_entities,
            top_entities=d.get("top_entities", []),
            n_instances=d.get("n_instances", 0),
            n_symbols=d.get("n_symbols", 0),
            wins=d.get("wins", 0),
            losses=d.get("losses", 0),
            win_rate=d.get("win_rate", 0.0),
            ci_lower=d.get("ci_lower", 0.0),
            ci_upper=d.get("ci_upper", 1.0),
            p_value=d.get("p_value", 1.0),
            is_significant=d.get("is_significant", False),
            avg_move_pct=d.get("avg_move_pct", 0.0),
            expected_window_days=d.get("expected_window_days", 14),
            dominant_lag_bucket=d.get("dominant_lag_bucket", ""),
            regime_win_rates=d.get("regime_win_rates", {}),
            best_regime=d.get("best_regime", ""),
            source_examples=d.get("source_examples", {}),
            built_at=d.get("built_at", ""),
            fingerprint_ids_sampled=d.get("fingerprint_ids_sampled", []),
        )

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict())
