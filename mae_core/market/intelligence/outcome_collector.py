#!/usr/bin/env python3
"""
outcome_collector.py - Bridge between signal archives and the Thompson feedback loop.

The gap this closes:
  midge_scan.py stores signals in JSONL archives → but nothing ever tells
  OutcomeTracker to track them as predictions → so Thompson never learns from
  real market outcomes.

OutcomeCollector:
  1. Registers qualifying scan signals as predictions (with per-type windows)
  2. Evaluates matured predictions (price checks → Thompson updates)
  3. Can retroactively process old signal archives

Per-type outcome windows (from triadic research):
  - sec_form4: 45 days (Lakonishok & Lee 2001: insider alpha persists 3-12 months)
  - insider_cluster: 60 days (Alldredge 2019: cluster alpha peaks 40-80 days)
  - congressional: 14 days (from disclosure date — trade itself is already stale)
  - sec_form8k: 5 days (market prices binary events in hours/days)
  - contract_prediction: 90 days (pre-announcement thesis needs time)
  - contract: 45 days (post-award drift resolves in weeks)
  - hiring: 90 days (hiring leads contract by 60-120 days)
  - sam_gov: 90 days (competition periods last months)
  - correlation: 21 days (correlation anomalies resolve)

Success threshold: 5% price move in predicted direction (not 2% — the old
threshold barely exceeded random baseline at $50K+ trade sizes after costs).
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mae_core.market.signal import MarketSignal

logger = logging.getLogger(__name__)

# Per-signal-type outcome windows (days)
OUTCOME_WINDOWS = {
    "sec_form4": 45,
    "insider_cluster": 60,
    "congressional": 14,
    "sec_form8k": 5,
    "contract_prediction": 90,
    "contract": 45,
    "hiring_tracker": 90,
    "sam_gov": 90,
    "correlation": 21,
}

# 5% move required — 2% was too close to random noise after transaction costs
SUCCESS_THRESHOLD_PCT = 5.0

# Data directory — same resolution as OutcomeTracker
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


class OutcomeCollector:
    """
    Bridges signal archives → OutcomeTracker prediction pipeline → Thompson Sampler.

    Usage:
        collector = OutcomeCollector(price_fetcher, thompson_sampler)
        registered = collector.register_signals(scan_signals)
        evaluated = collector.evaluate()
    """

    def __init__(
        self,
        price_fetcher,
        thompson_sampler,
        regime_classifier=None,
        data_dir: Path = None,
    ):
        from mae_core.market.outcome_tracker import OutcomeTracker

        self._data_dir = data_dir or DATA_DIR
        self.tracker = OutcomeTracker(
            price_fetcher, thompson_sampler, regime_classifier,
            data_dir=self._data_dir,
        )
        self.tracker.min_price_move_pct = SUCCESS_THRESHOLD_PCT

        self._registered_path = self._data_dir / "registered_signals.json"
        self._registered: set = self._load_registered()

    # ── Registration ──────────────────────────────────────────────────

    def register_signals(self, signals: "List[MarketSignal]") -> int:
        """
        Register qualifying scan signals as predictions for outcome tracking.

        A signal qualifies if it has a symbol (ticker) and isn't already registered.
        Direction-neutral signals are registered as direction-agnostic predictions
        (Thompson evaluates by magnitude alone).

        Returns count of newly registered predictions.
        """
        count = 0
        for sig in signals:
            if not sig.symbol or sig.signal_id in self._registered:
                continue

            direction = {"bullish": "up", "bearish": "down"}.get(sig.direction, "")
            window = OUTCOME_WINDOWS.get(sig.source, 14)

            self.tracker.record_prediction(
                source=sig.source,
                symbol=sig.symbol,
                direction=direction,
                outcome_window_days=window,
                metadata={"original_signal_id": sig.signal_id},
            )
            self._registered.add(sig.signal_id)
            count += 1

        if count > 0:
            self._save_registered()
            logger.info(f"Registered {count} signals as predictions (window varies by type)")

        return count

    def evaluate(self) -> int:
        """
        Evaluate matured predictions — the actual feedback loop.

        Delegates to OutcomeTracker.check_pending_outcomes() which:
        1. Checks price movement for predictions past their window
        2. Updates Thompson Sampler (success/failure)
        3. Writes to outcomes.jsonl
        """
        return self.tracker.check_pending_outcomes()

    def collect_from_archives(self, signals_dir: Path) -> int:
        """
        Retroactively register signals from JSONL archives.

        Reads all data/midge/signals/*.jsonl files and registers any unregistered
        signals that have a symbol. Use this once to bootstrap prediction tracking
        from existing signal history.
        """
        count = 0
        for jsonl_file in sorted(signals_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        signal_id = record.get("signal_id", "")
                        symbol = record.get("symbol", "")

                        if not symbol or not signal_id or signal_id in self._registered:
                            continue

                        direction_raw = record.get("direction", "")
                        direction = {"bullish": "up", "bearish": "down"}.get(direction_raw, "")
                        source = record.get("source", "unknown")
                        window = OUTCOME_WINDOWS.get(source, 14)

                        self.tracker.record_prediction(
                            source=source,
                            symbol=symbol,
                            direction=direction,
                            outcome_window_days=window,
                            metadata={"original_signal_id": signal_id, "archive": jsonl_file.name},
                        )
                        self._registered.add(signal_id)
                        count += 1
            except Exception as e:
                logger.warning(f"Failed to process archive {jsonl_file.name}: {e}")

        if count > 0:
            self._save_registered()
            logger.info(f"Registered {count} signals from archives")

        return count

    def get_statistics(self) -> dict:
        """Summary statistics for reporting."""
        tracker_stats = self.tracker.get_statistics()
        return {
            "registered_signals": len(self._registered),
            "pending_predictions": tracker_stats["pending_predictions"],
            "total_evaluated": tracker_stats["total_evaluated"],
            "success_threshold_pct": SUCCESS_THRESHOLD_PCT,
        }

    # ── Persistence ───────────────────────────────────────────────────

    def _load_registered(self) -> set:
        """Load set of already-registered signal IDs."""
        if self._registered_path.exists():
            try:
                data = json.loads(self._registered_path.read_text())
                return set(data)
            except (json.JSONDecodeError, TypeError):
                pass
        return set()

    def _save_registered(self) -> None:
        """Persist registered signal IDs."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._registered_path.write_text(json.dumps(sorted(self._registered), indent=0))
        except Exception as e:
            logger.warning(f"Failed to persist registered signals: {e}")
