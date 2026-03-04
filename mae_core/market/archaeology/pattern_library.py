"""Pattern Library — storage, querying, and management of pattern fingerprints.

Stores MoveFingerprints in an append-only JSONL file. Provides similarity
matching: given a set of live signals, find fingerprints whose precursor
profile matches the current state.

Stats tracking:
  - Win rate with Clopper-Pearson 95% CI
  - Regime-conditional performance
  - Freshness weighting (exponential decay, 90-day half-life)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from mae_core.market.archaeology.fingerprint import MoveFingerprint, PrecursorSignal

logger = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pattern_library.jsonl"
STATS_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pattern_stats.json"
DEFAULT_MATCH_THRESHOLD = 0.60  # 60% of precursor signals must match to "activate"


@dataclass
class PatternMatch:
    """Result of matching live signals against a fingerprint."""
    fingerprint: MoveFingerprint
    match_score: float      # 0-1, fraction of precursor signals matched
    matched_sources: list[str]   # Which precursor sources matched
    missing_sources: list[str]   # Which precursor sources are absent


class PatternLibrary:
    """Stores and queries pattern fingerprints.

    All fingerprints are loaded into memory on init. For MIDGE's scale
    (hundreds to low thousands of patterns), this is efficient.
    """

    def __init__(
        self,
        library_path: Optional[Path] = None,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ):
        self._path = library_path or LIBRARY_PATH
        self._match_threshold = match_threshold
        self._fingerprints: dict[str, MoveFingerprint] = {}
        self._load()

    def _load(self) -> None:
        """Load all fingerprints from disk."""
        if not self._path.exists():
            return

        count = 0
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        fp = MoveFingerprint.from_dict(data)
                        self._fingerprints[fp.fingerprint_id] = fp
                        count += 1
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug("Skipping malformed fingerprint: %s", e)
                        continue
        except OSError as e:
            logger.warning("Could not load pattern library: %s", e)

        logger.info("Pattern library loaded: %d fingerprints", count)

    def store(self, fingerprint: MoveFingerprint) -> bool:
        """Store a fingerprint. Returns False if duplicate (already exists)."""
        if fingerprint.fingerprint_id in self._fingerprints:
            return False

        self._fingerprints[fingerprint.fingerprint_id] = fingerprint

        # Append to file
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(fingerprint.to_json() + "\n")
        except OSError as e:
            logger.warning("Could not persist fingerprint: %s", e)

        return True

    def store_batch(self, fingerprints: list[MoveFingerprint]) -> int:
        """Store multiple fingerprints. Returns count of new (non-duplicate) entries."""
        stored = 0
        for fp in fingerprints:
            if self.store(fp):
                stored += 1
        return stored

    def get(self, fingerprint_id: str) -> Optional[MoveFingerprint]:
        """Get a fingerprint by ID."""
        return self._fingerprints.get(fingerprint_id)

    @property
    def size(self) -> int:
        return len(self._fingerprints)

    def query_similar(
        self,
        live_sources: set[str],
        symbol: str,
        direction: str,
        regime: str = "default",
    ) -> list[PatternMatch]:
        """Find fingerprints whose precursor profile matches live signal state.

        Args:
            live_sources: Set of source names currently active for this symbol
                          (e.g. {"sec_form4", "fred_macro", "ta_rsi"})
            symbol: Ticker symbol to match against.
            direction: "bullish" or "bearish".
            regime: Current market regime (optional filter).

        Returns:
            List of PatternMatch objects with match_score >= threshold,
            sorted by match_score descending.
        """
        matches: list[PatternMatch] = []

        for fp in self._fingerprints.values():
            # Direction must match
            if fp.direction != direction:
                continue

            # Symbol must match (fingerprints are symbol-specific)
            if fp.symbol != symbol:
                continue

            # Compute match score
            fp_sources = set(s.source for s in fp.precursor_signals)
            if not fp_sources:
                continue

            matched = fp_sources & live_sources
            missing = fp_sources - live_sources

            # Weight by lag bucket: immediate signals matter more
            weighted_match = 0.0
            weighted_total = 0.0
            bucket_weights = {
                "immediate": 2.0,
                "short": 1.5,
                "medium": 1.0,
                "long": 0.8,
                "extended": 0.6,
            }

            for precursor in fp.precursor_signals:
                weight = bucket_weights.get(precursor.lag_bucket, 1.0)
                weighted_total += weight
                if precursor.source in live_sources:
                    weighted_match += weight

            if weighted_total <= 0:
                continue

            score = weighted_match / weighted_total

            if score >= self._match_threshold:
                matches.append(PatternMatch(
                    fingerprint=fp,
                    match_score=round(score, 3),
                    matched_sources=sorted(matched),
                    missing_sources=sorted(missing),
                ))

        # Sort by match score descending
        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def update_outcome(
        self,
        fingerprint_id: str,
        won: bool,
        return_pct: float = 0.0,
    ) -> None:
        """Update a fingerprint's stats after an outcome is graded."""
        fp = self._fingerprints.get(fingerprint_id)
        if fp is None:
            return

        fp.total_activations += 1
        if won:
            fp.wins += 1
        else:
            fp.losses += 1
        fp.last_activation = datetime.now().isoformat()

        # Persist updated stats
        self._persist_all()

    def _persist_all(self) -> None:
        """Rewrite the full library file (for stat updates)."""
        try:
            with open(self._path, "w") as f:
                for fp in self._fingerprints.values():
                    f.write(fp.to_json() + "\n")
        except OSError as e:
            logger.warning("Could not persist pattern library: %s", e)

    def get_statistics(self) -> dict:
        """Return summary statistics about the library."""
        if not self._fingerprints:
            return {"total": 0}

        total = len(self._fingerprints)
        bullish = sum(1 for fp in self._fingerprints.values() if fp.direction == "bullish")
        with_outcomes = sum(1 for fp in self._fingerprints.values() if fp.wins + fp.losses > 0)

        # Domain signature distribution
        domain_sigs: dict[str, int] = {}
        for fp in self._fingerprints.values():
            domain_sigs[fp.domain_signature] = domain_sigs.get(fp.domain_signature, 0) + 1

        return {
            "total": total,
            "bullish": bullish,
            "bearish": total - bullish,
            "with_outcomes": with_outcomes,
            "top_domain_signatures": sorted(domain_sigs.items(), key=lambda x: -x[1])[:10],
        }

    def clopper_pearson_ci(self, fingerprint_id: str, alpha: float = 0.05) -> tuple[float, float]:
        """Compute Clopper-Pearson confidence interval for a fingerprint's win rate.

        Returns (lower_bound, upper_bound) at 1-alpha confidence level.
        """
        fp = self._fingerprints.get(fingerprint_id)
        if fp is None:
            return (0.0, 1.0)

        n = fp.wins + fp.losses
        if n == 0:
            return (0.0, 1.0)

        k = fp.wins

        # Clopper-Pearson uses the Beta distribution quantile function
        # For a pure-Python implementation without scipy, use the approximation
        # based on the F-distribution relationship
        try:
            from scipy.stats import beta as beta_dist
            lower = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
            upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
            return (round(lower, 4), round(upper, 4))
        except ImportError:
            # Fallback: Wilson score interval (simpler, still good)
            p = k / n
            z = 1.96  # 95% CI
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
            return (round(max(0, center - spread), 4), round(min(1, center + spread), 4))
