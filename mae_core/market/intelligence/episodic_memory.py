#!/usr/bin/env python3
"""
episodic_memory.py — MIDGE's autobiography.

Statistical memory (Thompson) knows signal reliability.
Pattern memory (templates) knows recurring shapes.
Episodic memory knows what ACTUALLY happened — specific situations, context, resolution.

"What happened the last time institutional + insider converged on a semiconductor stock?"
This file answers that question.

Storage: append-only JSONL at data/midge/episodic_memory.jsonl
Each line is one Episode record. Resolved episodes are written as updates
(new line with same episode_id overwrites in query; latest-wins resolution).
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[4] / "data" / "midge" / "episodic_memory.jsonl"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A single episodic memory — one specific situation MIDGE observed.

    All fields have defaults so Episode() can be constructed with partial info
    and filled in later. This prevents TypeError in call sites that only supply
    the fields known at signal time.
    """

    episode_id: str = ""           # Unique ID (uuid4 hex)
    event_type: str = "convergence"  # "convergence" | "cascade" | "anomaly" | "pattern_stack" | "failure"
    timestamp: str = ""            # ISO-8601 when it started
    resolved_at: str = ""          # ISO-8601 when outcome was known (empty = pending)

    # What happened
    tickers: List[str] = field(default_factory=list)    # Involved tickers
    domains: List[str] = field(default_factory=list)    # Involved domains
    direction: str = ""            # "bullish" | "bearish" | "neutral"
    confidence: float = 0.0        # MIDGE's confidence at signal time

    # Context at the time
    regime: str = ""               # Market regime: "bull" | "bear" | "volatile" | "sideways"
    concurrent_events: List[str] = field(default_factory=list)  # episode_ids active at the same time
    macro_context: str = ""        # Brief free-text macro state

    # What MIDGE did
    action_taken: str = "watched"  # "paper_trade" | "alert_sent" | "watched" | "ignored"

    # Resolution (empty until resolved)
    outcome: str = "pending"       # "correct" | "wrong" | "partially_correct" | "expired" | "pending"
    outcome_details: str = ""      # Plain-text description of what actually happened
    price_at_signal: float = 0.0   # Price when signal fired (0.0 = not recorded)
    price_at_resolution: float = 0.0  # Price at outcome check (0.0 = pending)
    move_pct: float = 0.0          # Actual price move % (positive = up, negative = down)

    # Learning
    lessons: List[str] = field(default_factory=list)          # What was learned from this episode
    similar_episodes: List[str] = field(default_factory=list) # episode_ids of similar past episodes

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            episode_id=d.get("episode_id", ""),
            event_type=d.get("event_type", "convergence"),
            timestamp=d.get("timestamp", ""),
            resolved_at=d.get("resolved_at", ""),
            tickers=d.get("tickers", []),
            domains=d.get("domains", []),
            direction=d.get("direction", ""),
            confidence=d.get("confidence", 0.0),
            regime=d.get("regime", ""),
            concurrent_events=d.get("concurrent_events", []),
            macro_context=d.get("macro_context", ""),
            action_taken=d.get("action_taken", "watched"),
            outcome=d.get("outcome", "pending"),
            outcome_details=d.get("outcome_details", ""),
            price_at_signal=d.get("price_at_signal", 0.0),
            price_at_resolution=d.get("price_at_resolution", 0.0),
            move_pct=d.get("move_pct", 0.0),
            lessons=d.get("lessons", []),
            similar_episodes=d.get("similar_episodes", []),
        )


# ---------------------------------------------------------------------------
# EpisodicMemory
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """MIDGE's structured store of specific past situations.

    Append-only JSONL — every write is a new line.
    Reads build an in-memory index (episode_id → latest Episode).
    This keeps the file as an audit trail while queries are O(1).

    Usage:
        em = EpisodicMemory()
        ep = Episode(episode_id=str(uuid.uuid4().hex[:12]), ...)
        em.record_episode(ep)
        em.resolve_episode(ep.episode_id, outcome="correct", ...)
        similar = em.query_similar(["insider", "institutional"], "bullish", regime="bull")
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # episode_id -> Episode (latest version wins on load)
        self._index: dict[str, Episode] = {}
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load JSONL into memory index. Latest record per episode_id wins."""
        if not self.path.exists():
            return
        loaded = 0
        errors = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        ep = Episode.from_dict(d)
                        self._index[ep.episode_id] = ep
                        loaded += 1
                    except Exception:
                        errors += 1
        except Exception as exc:
            logger.warning("EpisodicMemory: could not load %s: %s", self.path, exc)
        if loaded:
            logger.debug("EpisodicMemory: loaded %d episodes (%d errors)", loaded, errors)

    def _append(self, episode: Episode) -> None:
        """Write one episode to the JSONL file."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(episode.to_dict()) + "\n")
            self._index[episode.episode_id] = episode
        except Exception as exc:
            logger.error("EpisodicMemory: write failed: %s", exc)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def record_episode(self, episode: Episode) -> None:
        """Append a new episode. Call this when a signal fires."""
        if not episode.episode_id:
            episode.episode_id = uuid.uuid4().hex[:12]
        if not episode.timestamp:
            episode.timestamp = self._now_iso()
        # Link to currently-active episodes as concurrent context
        if not episode.concurrent_events:
            active_ids = [ep.episode_id for ep in self.get_active(limit=20)
                          if ep.episode_id != episode.episode_id]
            episode.concurrent_events = active_ids[:10]
        self._append(episode)
        logger.debug(
            "EpisodicMemory: recorded %s [%s %s domains=%s conf=%.2f]",
            episode.episode_id, episode.direction,
            episode.event_type, episode.domains, episode.confidence,
        )

    def resolve_episode(
        self,
        episode_id: str,
        outcome: str,
        details: str,
        price_at_resolution: float = 0.0,
        lessons: Optional[List[str]] = None,
    ) -> Optional[Episode]:
        """Update an episode with its outcome. Returns updated episode or None."""
        ep = self._index.get(episode_id)
        if ep is None:
            logger.warning("EpisodicMemory: resolve called on unknown id %s", episode_id)
            return None

        ep.outcome = outcome
        ep.outcome_details = details
        ep.resolved_at = self._now_iso()
        ep.price_at_resolution = price_at_resolution

        if ep.price_at_signal and price_at_resolution:
            ep.move_pct = round(
                (price_at_resolution - ep.price_at_signal) / ep.price_at_signal * 100, 2
            )

        if lessons:
            ep.lessons = lessons
        else:
            ep.lessons = [self.learn_from_resolution(ep)]

        # Cross-link similar resolved episodes
        similar = self.query_similar(ep.domains, ep.direction, regime=ep.regime, limit=5)
        ep.similar_episodes = [s.episode_id for s in similar if s.episode_id != episode_id]

        self._append(ep)
        logger.info(
            "EpisodicMemory: resolved %s → %s (%.1f%% move)",
            episode_id, outcome, ep.move_pct,
        )
        return ep

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def query_similar(
        self,
        domains: List[str],
        direction: str,
        regime: Optional[str] = None,
        limit: int = 5,
    ) -> List[Episode]:
        """Return resolved episodes with similar domain set + direction + (optional) regime.

        Similarity = Jaccard overlap of domain lists, then filtered by direction.
        Regime match scores a bonus. Returns top-N by similarity score.
        """
        target_domains = set(domains)
        scored: list[tuple[float, Episode]] = []

        for ep in self._index.values():
            if ep.outcome == "pending":
                continue
            if ep.direction != direction:
                continue
            ep_domains = set(ep.domains)
            if not ep_domains or not target_domains:
                continue
            # Jaccard similarity
            overlap = len(target_domains & ep_domains)
            union = len(target_domains | ep_domains)
            score = overlap / union if union else 0.0
            if score == 0.0:
                continue
            if regime and ep.regime == regime:
                score += 0.2  # regime match bonus
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:limit]]

    def query_ticker(self, ticker: str, limit: int = 10) -> List[Episode]:
        """All episodes involving this ticker, newest first."""
        ticker = ticker.upper()
        results = [
            ep for ep in self._index.values()
            if ticker in [t.upper() for t in ep.tickers]
        ]
        results.sort(key=lambda ep: ep.timestamp, reverse=True)
        return results[:limit]

    def query_failures(self, limit: int = 10) -> List[Episode]:
        """Episodes where MIDGE was wrong, newest first, with lessons."""
        results = [
            ep for ep in self._index.values()
            if ep.outcome == "wrong"
        ]
        results.sort(key=lambda ep: ep.resolved_at or ep.timestamp, reverse=True)
        return results[:limit]

    def get_active(self, limit: int = 20) -> List[Episode]:
        """Unresolved episodes (pending outcomes), newest first."""
        results = [ep for ep in self._index.values() if ep.outcome == "pending"]
        results.sort(key=lambda ep: ep.timestamp, reverse=True)
        return results[:limit]

    def get_recent(self, days: int = 7, limit: int = 20) -> List[Episode]:
        """Episodes from the last N days, newest first."""
        cutoff = datetime.now(timezone.utc)
        results = []
        for ep in self._index.values():
            try:
                ts_str = ep.timestamp
                if ts_str:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    delta = (cutoff - ts).days
                    if delta <= days:
                        results.append(ep)
            except Exception:
                continue
        results.sort(key=lambda ep: ep.timestamp, reverse=True)
        return results[:limit]

    def count(self) -> dict:
        """Summary counts by outcome."""
        counts: dict[str, int] = {}
        for ep in self._index.values():
            counts[ep.outcome] = counts.get(ep.outcome, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn_from_resolution(self, episode: Episode) -> str:
        """Generate a plain-English lesson string from a resolved episode.

        Called automatically by resolve_episode. Returns a one-paragraph narrative
        that MIDGE (and future instances) can read to understand what happened.
        """
        tickers_str = ", ".join(episode.tickers) if episode.tickers else "unknown ticker"
        domains_str = " + ".join(episode.domains) if episode.domains else "unknown domains"

        # Determine the lead domain (first in sequence if available, else first in list)
        lead_domain = episode.domains[0] if episode.domains else "signal"

        # Timing information
        days_elapsed = 0
        if episode.timestamp and episode.resolved_at:
            try:
                t0 = datetime.fromisoformat(episode.timestamp)
                t1 = datetime.fromisoformat(episode.resolved_at)
                if t0.tzinfo is None:
                    t0 = t0.replace(tzinfo=timezone.utc)
                if t1.tzinfo is None:
                    t1 = t1.replace(tzinfo=timezone.utc)
                days_elapsed = max(0, (t1 - t0).days)
            except Exception:
                pass

        move_str = f"{abs(episode.move_pct):.1f}%" if episode.move_pct else "an unknown amount"
        direction_word = "up" if episode.move_pct >= 0 else "down"

        if episode.outcome == "correct":
            return (
                f"I predicted {tickers_str} would go {episode.direction} based on "
                f"{domains_str}. It moved {direction_word} {move_str} in {days_elapsed} days. "
                f"The {lead_domain} signal fired first, validating the thesis. "
                f"Regime: {episode.regime}. Confidence at signal: {episode.confidence:.0%}."
            )

        elif episode.outcome == "wrong":
            opposite = "down" if episode.direction == "bullish" else "up"
            reasons = []
            if episode.regime in ("volatile",):
                reasons.append("regime was volatile — signals less reliable")
            if episode.confidence < 0.5:
                reasons.append("confidence was below 50% — marginal signal")
            if len(episode.domains) < 3:
                reasons.append("fewer than 3 domains — below minimum convergence threshold")
            if not reasons:
                reasons.append("domains may have been correlated, not independent")
            reason_str = "; ".join(reasons)
            return (
                f"I predicted {tickers_str} would go {episode.direction} but it went "
                f"{opposite} {move_str} in {days_elapsed} days. "
                f"Possible reasons: {reason_str}. "
                f"Regime: {episode.regime}. Confidence at signal: {episode.confidence:.0%}."
            )

        elif episode.outcome == "partially_correct":
            window_days = days_elapsed
            return (
                f"I was right about direction ({episode.direction}) for {tickers_str} "
                f"but timing was off — the move of {move_str} happened "
                f"{window_days} days into my observation window. "
                f"Domains: {domains_str}. Regime: {episode.regime}."
            )

        elif episode.outcome == "expired":
            return (
                f"My {episode.direction} signal on {tickers_str} ({domains_str}) "
                f"expired without a decisive move after {days_elapsed} days. "
                f"The thesis may have been absorbed by the market or timing was wrong. "
                f"Regime at signal: {episode.regime}."
            )

        else:
            return (
                f"Episode for {tickers_str} ({episode.direction}, {domains_str}) "
                f"resolved as '{episode.outcome}' after {days_elapsed} days."
            )


# ---------------------------------------------------------------------------
# Integration TODOs (hooks to wire in bootstrap / market_hooks.py)
# ---------------------------------------------------------------------------

# TODO (convergence_alerter.py / market_hooks.py — _on_convergence_alert):
#   After a ConvergenceAlert fires, create and record an Episode:
#
#   ep = Episode(
#       episode_id=alert.alert_id,
#       event_type="convergence",
#       timestamp=alert.timestamp.isoformat(),
#       resolved_at="",
#       tickers=list(ctx._alert_tickers.get(alert.alert_id, [])),
#       domains=alert.domains_converging,
#       direction=alert.direction,
#       confidence=alert.confidence,
#       regime=ctx.regime_classifier.current_regime if ctx.regime_classifier else "",
#       concurrent_events=[],     # auto-filled by record_episode
#       macro_context="",         # TODO: pull from ctx.world_model or somatic_state
#       action_taken="paper_trade" if qualified_for_trade else "alert_sent",
#       outcome="pending",
#       outcome_details="", price_at_signal=0.0, price_at_resolution=0.0,
#       move_pct=0.0, lessons=[], similar_episodes=[],
#   )
#   ctx.episodic_memory.record_episode(ep)

# TODO (outcome_collector.py — _evaluate_one / _on_outcome):
#   After a prediction is graded, resolve the matching episode:
#
#   ctx.episodic_memory.resolve_episode(
#       episode_id=prediction_id,    # same as alert_id if wired above
#       outcome="correct" if won else "wrong",
#       details=f"{ticker} moved {move_pct:.1f}% in {days} days",
#       price_at_resolution=current_price,
#   )

# TODO (convergence_confidence.py — check_convergence, before generating alert):
#   Query episodic memory for historical context before emitting a new alert:
#
#   similar = ctx.episodic_memory.query_similar(
#       domains=domains_converging,
#       direction=direction,
#       regime=current_regime,
#       limit=3,
#   )
#   if similar:
#       logger.info("Historical precedents: %s", [s.episode_id for s in similar])
#       # Optionally adjust confidence based on historical win rate from similar episodes
