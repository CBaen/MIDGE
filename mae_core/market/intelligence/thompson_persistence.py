"""Thompson Sampler persistence, seeding, and history-replay utilities.

Contains:
- BetaDistribution, SamplingResult, UpdateResult dataclasses
- _load_distributions, _save_distributions_locked, _save_distributions
- _seed_from_reliability, replay_from_history, seed_combo_distributions
- _log_update
"""

import json
import logging
import os
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Persistence path constants
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
DISTRIBUTIONS_FILE = DATA_DIR / "thompson_distributions.json"
HISTORY_FILE = DATA_DIR / "thompson_history.jsonl"

DEFAULT_PRIOR_SCALE = 20


@dataclass
class BetaDistribution:
    """Beta distribution parameters for a signal."""
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Expected value of the distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def samples(self) -> int:
        """Approximate number of observations."""
        return int(self.alpha + self.beta - 2)  # Subtract initial prior


@dataclass
class SamplingResult:
    """Result of a Thompson sampling selection."""
    timestamp: str
    signal_id: str
    sampled_score: float
    distribution: BetaDistribution
    regime: str


@dataclass
class UpdateResult:
    """Result of a distribution update."""
    timestamp: str
    signal_id: str
    success: bool
    regime: str
    old_alpha: float
    old_beta: float
    new_alpha: float
    new_beta: float
    old_mean: float
    new_mean: float


class ThompsonPersistenceMixin:
    """Persistence, seeding, and history-replay methods for ThompsonSampler.

    Expects the following instance attributes to be set before use:
        self.persistence_path: Path
        self.history_path: Path
        self.prior_scale: float
        self._lock: threading.Lock
        self.distributions: Dict[str, Dict[str, Dict[str, float]]]
    """

    def _load_distributions(self) -> None:
        """Load distributions from disk."""
        if self.persistence_path.exists():
            try:
                data = json.loads(self.persistence_path.read_text())
                if isinstance(data, dict):
                    self.distributions = data
                else:
                    logger.warning(
                        "Thompson distributions file has unexpected format — treating as empty"
                    )
                    self.distributions = {}
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    "Thompson distributions file is corrupt or unreadable — treating as empty"
                )
                self.distributions = {}

    def _save_distributions_locked(self) -> None:
        """Atomic write — assumes caller already holds self._lock."""
        tmp = self.persistence_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self.distributions, indent=2))
            os.replace(tmp, self.persistence_path)
        except Exception:
            logger.warning("Failed to persist Thompson distributions", exc_info=True)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _save_distributions(self) -> None:
        """Atomic write with lock."""
        with self._lock:
            self._save_distributions_locked()

    def _seed_from_reliability(self) -> None:
        """Seed distributions from existing reliability scores.

        Converts reliability r to Beta(alpha, beta) where:
        alpha = r * scale
        beta = (1-r) * scale
        """
        try:
            from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
            SOURCE_RELIABILITY = LEARNING_CONFIG.get("source_reliability", {})

            for signal_id, reliability in SOURCE_RELIABILITY.items():
                alpha = reliability * self.prior_scale
                beta = (1 - reliability) * self.prior_scale

                self.distributions[signal_id] = {
                    "default": {"alpha": alpha, "beta": beta}
                }

            self._save_distributions()
        except ImportError:
            # No reliability scores available
            pass

    def replay_from_history(self, history_path: Optional[Path] = None) -> int:
        """Rebuild distributions by replaying all 'update' events from history log.

        Reads thompson_history.jsonl and re-applies every recorded outcome
        (success/failure) to reconstruct the learned state. Existing in-memory
        distributions are CLEARED first so the replay is authoritative.

        This is the recovery mechanism for Bug 2: if distributions.json is lost
        or corrupt, the history log is the source of truth.

        Args:
            history_path: Path to thompson_history.jsonl (default: self.history_path)

        Returns:
            Number of update events replayed.
        """
        path = history_path or self.history_path
        if not path.exists():
            logger.warning("replay_from_history: history file not found at %s", path)
            return 0

        # Clear in-memory state — replay is authoritative
        self.distributions = {}

        replayed = 0
        skipped = 0
        with open(path, "r") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                # Only replay 'update' events — forgetting events are intentionally
                # not replayed (they represent time-based decay that is already
                # encoded in the update sequence through the cadence).
                # Fields come from UpdateResult.asdict():
                #   signal_id, success, regime, timestamp, old_alpha/beta, new_alpha/beta
                if "signal_id" not in entry or "success" not in entry:
                    continue

                signal_id = entry["signal_id"]
                success = bool(entry["success"])
                regime = entry.get("regime", "default")

                # Apply the Bayesian update directly — do NOT use self.update()
                # to avoid writing to the history file again (would double the log).
                if signal_id not in self.distributions:
                    self.distributions[signal_id] = {}
                if regime not in self.distributions[signal_id]:
                    self.distributions[signal_id][regime] = {"alpha": 1.0, "beta": 1.0}

                params = self.distributions[signal_id][regime]
                if success:
                    params["alpha"] += 1
                else:
                    params["beta"] += 1
                replayed += 1

        if replayed:
            with self._lock:
                self._save_distributions_locked()
            logger.info(
                "replay_from_history: replayed %d updates, skipped %d malformed lines.",
                replayed, skipped,
            )
        else:
            logger.warning(
                "replay_from_history: no update events found in %s (skipped %d lines).",
                path, skipped,
            )

        return replayed

    def seed_combo_distributions(self, replay_path: Path) -> int:
        """Seed combo Thompson distributions from replay results file.

        Reads domain_combos from replay_results.json and creates Beta
        distributions for each combo with total >= 3 observations.
        Idempotent: skips combos where live samples already exceed replay total.
        """
        if not replay_path.exists():
            return 0
        results = json.loads(replay_path.read_text())
        combos = results.get("report", {}).get("domain_combos", {})
        seeded = 0
        for combo_str, stats in combos.items():
            total = stats.get("total", 0)
            if total < 3:
                continue
            key = f"combo:{combo_str}"
            if self.get_distribution(key).samples >= total:
                continue
            for _ in range(stats.get("wins", 0)):
                self.update(key, success=True)
            for _ in range(total - stats.get("wins", 0)):
                self.update(key, success=False)
            seeded += 1
        return seeded

    def _log_update(self, result: UpdateResult) -> None:
        """Append update to history file."""
        with open(self.history_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
