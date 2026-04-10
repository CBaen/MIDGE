#!/usr/bin/env python3
"""Pattern Sequence Tracker — template-to-template timing relationships.

Tracks the observation: "Pattern A often precedes Pattern B by N days."
This adds a temporal dimension on top of co-occurrence — it asks not just
WHAT patterns stack, but in WHAT ORDER and with WHAT DELAY.

Biological analogy: this is MIDGE's sense of causal rhythm. Not just "these
two events correlate" but "event A always happens before event B, and the
gap tells you how much time you have to act."

Key concepts:
  - Sequence: (template_a, template_b) — A precedes B
  - Lag: days between A activating and B activating on the SAME symbol
  - Chain: the downstream sequence starting from a given template

Data flow:
  PatternWatcher fires template match
    → on_template_activated() called
    → records lag against any existing watches for that ticker
    → starts watching for future templates on that ticker
    → confirms sequences in _sequences dict

Persistence: JSONL at data/market/pattern_sequences.jsonl
  Each record is either:
    {"type": "sequence", "template_a": ..., "template_b": ..., "observed_lags": [...]}
    {"type": "watch", "ticker": ..., "template_id": ..., "activated_date": ...}
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pattern_sequences.jsonl"

# Maximum days to watch for a follow-on pattern before expiring the watch
_MAX_WATCH_DAYS: int = 30
# Minimum observations before we report a confirmed sequence
_MIN_OBSERVATIONS: int = 3


@dataclass
class SequenceRecord:
    """Statistics for a confirmed template-to-template sequence."""
    template_a: str
    template_b: str
    observed_lags: List[int] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)  # "win"/"loss"/"pending"
    first_seen: str = ""
    last_seen: str = ""

    @property
    def avg_lag_days(self) -> float:
        if not self.observed_lags:
            return 0.0
        return round(sum(self.observed_lags) / len(self.observed_lags), 1)

    @property
    def n_observations(self) -> int:
        return len(self.observed_lags)

    @property
    def win_rate(self) -> float:
        graded = [o for o in self.outcomes if o in ("win", "loss")]
        if not graded:
            return 0.0
        wins = sum(1 for o in graded if o == "win")
        return round(wins / len(graded), 4)


@dataclass
class ActiveWatch:
    """A template activation we're watching for follow-on patterns."""
    template_id: str
    ticker: str
    activated_date: str  # ISO date string
    direction: str = ""


class PatternSequenceTracker:
    """Tracks template-to-template timing relationships.

    Maintains two data structures:
      _active_watches:  {ticker: [ActiveWatch]}  — templates recently active
      _sequences:       {(a, b): SequenceRecord} — confirmed sequences

    The core loop:
      1. on_template_activated() is called when PatternWatcher fires a match
      2. We check: is any other template already being watched for this ticker?
      3. If so, we have a sequence observation: record the lag
      4. We also add this new activation to the watch list
      5. Expired watches (>30 days old) are pruned on each call

    This produces the "temporal chain" — the order in which templates arrive
    before a confirmed move, which is more predictive than mere co-occurrence.
    """

    def __init__(self, path: str = None):
        self._path = Path(path) if path else _DEFAULT_PATH
        self._active_watches: Dict[str, List[ActiveWatch]] = defaultdict(list)
        self._sequences: Dict[Tuple[str, str], SequenceRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_template_activated(
        self,
        template_id: str,
        ticker: str,
        activated_date: str,
        direction: str = "",
    ) -> List[Tuple[str, int]]:
        """Record a template activation and check for sequence completions.

        Called when PatternWatcher fires a template match.
        Checks if other templates are already being watched for this ticker —
        if so, records the lag between the earlier activation and this one.

        Args:
            template_id: The template that just activated.
            ticker: The symbol this activation is for.
            activated_date: ISO date string (YYYY-MM-DD).
            direction: "bullish" or "bearish" (optional, for filtering).

        Returns:
            List of (preceding_template_id, lag_days) for any sequences found.
        """
        self._prune_expired_watches(activated_date)

        sequences_found: List[Tuple[str, int]] = []
        try:
            act_date = date.fromisoformat(activated_date)
        except (ValueError, TypeError):
            act_date = date.today()

        # Check existing watches for this ticker — any earlier activation?
        for watch in self._active_watches.get(ticker, []):
            if watch.template_id == template_id:
                continue  # Same template re-firing — skip

            try:
                watch_date = date.fromisoformat(watch.activated_date)
            except (ValueError, TypeError):
                continue

            lag = (act_date - watch_date).days
            if lag <= 0:
                continue  # Must be strictly after the preceding template

            # Record this sequence
            seq_key = (watch.template_id, template_id)
            if seq_key not in self._sequences:
                self._sequences[seq_key] = SequenceRecord(
                    template_a=watch.template_id,
                    template_b=template_id,
                    first_seen=activated_date,
                )
            rec = self._sequences[seq_key]
            rec.observed_lags.append(lag)
            rec.last_seen = activated_date
            sequences_found.append((watch.template_id, lag))

            logger.debug(
                "Sequence observed: %s → %s on %s, lag=%d days",
                watch.template_id, template_id, ticker, lag,
            )

        # Add this activation to watches
        self._active_watches[ticker].append(
            ActiveWatch(
                template_id=template_id,
                ticker=ticker,
                activated_date=activated_date,
                direction=direction,
            )
        )

        return sequences_found

    def record_sequence_outcome(
        self,
        template_a: str,
        template_b: str,
        outcome: str,
    ) -> None:
        """Update outcome for a sequence (when a pattern stack's prediction is graded).

        Args:
            template_a: The preceding template in the sequence.
            template_b: The following template in the sequence.
            outcome: "win" or "loss"
        """
        key = (template_a, template_b)
        if key in self._sequences:
            self._sequences[key].outcomes.append(outcome)

    def get_sequences(self, min_observations: int = _MIN_OBSERVATIONS) -> List[dict]:
        """Return confirmed template-to-template sequences.

        Args:
            min_observations: Minimum number of observed instances.

        Returns:
            List of dicts with keys:
              template_a, template_b, avg_lag_days, win_rate, n_observations
        """
        results = []
        for (ta, tb), rec in self._sequences.items():
            if rec.n_observations < min_observations:
                continue
            results.append({
                "template_a": ta,
                "template_b": tb,
                "avg_lag_days": rec.avg_lag_days,
                "win_rate": rec.win_rate,
                "n_observations": rec.n_observations,
                "observed_lags": rec.observed_lags,
                "first_seen": rec.first_seen,
                "last_seen": rec.last_seen,
            })
        results.sort(key=lambda x: x["n_observations"], reverse=True)
        return results

    def get_chain(self, template_id: str, min_observations: int = _MIN_OBSERVATIONS) -> List[dict]:
        """Return the downstream chain following this template.

        "What typically follows template X?"

        Returns list of dicts (sorted by n_observations descending):
          {next_template, avg_lag_days, win_rate, n_observations}
        """
        chain = []
        for (ta, tb), rec in self._sequences.items():
            if ta != template_id:
                continue
            if rec.n_observations < min_observations:
                continue
            chain.append({
                "next_template": tb,
                "avg_lag_days": rec.avg_lag_days,
                "win_rate": rec.win_rate,
                "n_observations": rec.n_observations,
            })
        chain.sort(key=lambda x: x["n_observations"], reverse=True)
        return chain

    def get_predecessors(self, template_id: str, min_observations: int = _MIN_OBSERVATIONS) -> List[dict]:
        """Return templates that typically PRECEDE this template.

        "What fires before template X?"
        """
        preds = []
        for (ta, tb), rec in self._sequences.items():
            if tb != template_id:
                continue
            if rec.n_observations < min_observations:
                continue
            preds.append({
                "preceding_template": ta,
                "avg_lag_days": rec.avg_lag_days,
                "win_rate": rec.win_rate,
                "n_observations": rec.n_observations,
            })
        preds.sort(key=lambda x: x["n_observations"], reverse=True)
        return preds

    def get_statistics(self) -> dict:
        """Summary statistics for reporting."""
        total_seqs = len(self._sequences)
        confirmed = sum(
            1 for r in self._sequences.values()
            if r.n_observations >= _MIN_OBSERVATIONS
        )
        total_watches = sum(len(v) for v in self._active_watches.values())
        return {
            "total_sequences_observed": total_seqs,
            "confirmed_sequences": confirmed,
            "active_watches": total_watches,
            "tickers_watched": len(self._active_watches),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist sequences and active watches to JSONL."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                # Save confirmed sequences
                for (ta, tb), rec in self._sequences.items():
                    record = {
                        "type": "sequence",
                        "template_a": ta,
                        "template_b": tb,
                        "observed_lags": rec.observed_lags,
                        "outcomes": rec.outcomes,
                        "first_seen": rec.first_seen,
                        "last_seen": rec.last_seen,
                    }
                    f.write(json.dumps(record, default=str) + "\n")

                # Save active watches (non-expired)
                today_str = date.today().isoformat()
                for ticker, watches in self._active_watches.items():
                    for w in watches:
                        record = {
                            "type": "watch",
                            "template_id": w.template_id,
                            "ticker": w.ticker,
                            "activated_date": w.activated_date,
                            "direction": w.direction,
                        }
                        f.write(json.dumps(record, default=str) + "\n")

            tmp.replace(self._path)
            logger.debug(
                "PatternSequenceTracker saved %d sequences, %d active watches",
                len(self._sequences),
                sum(len(v) for v in self._active_watches.values()),
            )
        except Exception:
            logger.warning("PatternSequenceTracker save failed", exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _load(self) -> None:
        """Load sequences and watches from disk."""
        if not self._path.exists():
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rtype = rec.get("type", "")
                    if rtype == "sequence":
                        ta = rec.get("template_a", "")
                        tb = rec.get("template_b", "")
                        if ta and tb:
                            self._sequences[(ta, tb)] = SequenceRecord(
                                template_a=ta,
                                template_b=tb,
                                observed_lags=rec.get("observed_lags", []),
                                outcomes=rec.get("outcomes", []),
                                first_seen=rec.get("first_seen", ""),
                                last_seen=rec.get("last_seen", ""),
                            )
                    elif rtype == "watch":
                        tid = rec.get("template_id", "")
                        ticker = rec.get("ticker", "")
                        activated_date = rec.get("activated_date", "")
                        if tid and ticker and activated_date:
                            self._active_watches[ticker].append(
                                ActiveWatch(
                                    template_id=tid,
                                    ticker=ticker,
                                    activated_date=activated_date,
                                    direction=rec.get("direction", ""),
                                )
                            )
            logger.info(
                "PatternSequenceTracker loaded %d sequences, %d watches across %d tickers",
                len(self._sequences),
                sum(len(v) for v in self._active_watches.values()),
                len(self._active_watches),
            )
        except Exception:
            logger.warning("PatternSequenceTracker load failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_expired_watches(self, reference_date_str: str) -> None:
        """Remove watches older than _MAX_WATCH_DAYS."""
        try:
            ref_date = date.fromisoformat(reference_date_str)
        except (ValueError, TypeError):
            ref_date = date.today()

        cutoff = ref_date - timedelta(days=_MAX_WATCH_DAYS)

        for ticker in list(self._active_watches.keys()):
            valid = []
            for w in self._active_watches[ticker]:
                try:
                    w_date = date.fromisoformat(w.activated_date)
                    if w_date >= cutoff:
                        valid.append(w)
                except (ValueError, TypeError):
                    pass  # Drop malformed dates
            if valid:
                self._active_watches[ticker] = valid
            else:
                del self._active_watches[ticker]
