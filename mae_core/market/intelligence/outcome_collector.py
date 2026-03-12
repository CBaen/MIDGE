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
import threading
from datetime import datetime
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
    "convergence_combo": 14,  # Combo-level convergence alert outcome window
    "pattern_stack": 14,      # Pattern archaeology stacking detection
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
        self._registered: dict = self._load_registered()
        self._registered_lock = threading.Lock()
        self._pattern_library = None  # Set via set_pattern_library() for feedback
        self.tracker.on_outcome = self._on_outcome_graded

    def set_pattern_library(self, library) -> None:
        """Wire PatternLibrary for template outcome feedback."""
        self._pattern_library = library

    def _on_outcome_graded(self, pred: dict, success: bool, pct_change: float) -> None:
        """Callback fired by OutcomeTracker for each graded outcome.

        For pattern_stack predictions, updates each template's win/loss stats
        in PatternLibrary — closing the feedback loop so templates improve.
        """
        source = pred.get("source", "")
        if not source.startswith("pattern_stack:") or self._pattern_library is None:
            return
        metadata = pred.get("metadata", {})
        template_ids = metadata.get("template_ids", [])
        for tid in template_ids:
            try:
                self._pattern_library.update_outcome(template_id=tid, won=success)
            except Exception:
                logger.debug("Template outcome update failed for %s", tid, exc_info=True)
        if template_ids:
            logger.info(
                "Pattern stack outcome: %s, %d templates updated (%s %.1f%%)",
                "WIN" if success else "LOSS", len(template_ids), source, pct_change,
            )

    # ── Registration ──────────────────────────────────────────────────

    def register_signals(self, signals: "List[MarketSignal]") -> int:
        """
        Register qualifying scan signals as predictions for outcome tracking.

        A signal qualifies if it has a symbol (ticker) and isn't already registered.
        Direction-neutral signals are registered as direction-agnostic predictions
        (Thompson evaluates by magnitude alone).

        Returns count of newly registered predictions.
        """
        with self._registered_lock:
            return self._register_signals_locked(signals)

    def _register_signals_locked(self, signals: "List[MarketSignal]") -> int:
        """Inner — caller must hold self._registered_lock."""
        # Prune stale entries (older than 90 days) before processing
        now = datetime.now()
        self._registered = {
            k: v for k, v in self._registered.items()
            if (now - v).days <= 90
        }

        count = 0
        for sig in signals:
            if not sig.symbol or sig.signal_id in self._registered:
                continue

            direction = {"bullish": "up", "bearish": "down"}.get(sig.direction, "")
            window = OUTCOME_WINDOWS.get(sig.source, 14)

            sig_ts = sig.timestamp.isoformat() if sig.timestamp else ""
            self.tracker.record_prediction(
                source=sig.source,
                symbol=sig.symbol,
                direction=direction,
                outcome_window_days=window,
                metadata={"original_signal_id": sig.signal_id},
                timestamp=sig_ts,
            )
            self._registered[sig.signal_id] = datetime.now()
            count += 1

        if count > 0:
            self._save_registered()
            logger.info(f"Registered {count} signals as predictions (window varies by type)")

        return count

    def register_convergence_alert(self, alert, primary_symbol: str) -> bool:
        """
        Register a convergence alert as a combo-level prediction for Thompson feedback.

        The combo key (e.g. "combo:events+macro+price") flows through OutcomeTracker
        to Thompson Sampler — when the outcome resolves, Thompson learns which domain
        combinations actually predict price movement.
        """
        with self._registered_lock:
            return self._register_convergence_alert_locked(alert, primary_symbol)

    def _register_convergence_alert_locked(self, alert, primary_symbol: str) -> bool:
        """Inner — caller must hold self._registered_lock."""
        if not primary_symbol:
            return False

        combo_key = "combo:" + "+".join(alert.domains_converging)
        alert_id = f"combo_{alert.alert_id}"

        if alert_id in self._registered:
            return False

        direction = {"bullish": "up", "bearish": "down"}.get(alert.direction, "")
        window = OUTCOME_WINDOWS.get("convergence_combo", 14)

        ts = alert.timestamp.isoformat() if alert.timestamp else ""
        self.tracker.record_prediction(
            source=combo_key,
            symbol=primary_symbol,
            direction=direction,
            outcome_window_days=window,
            metadata={"combo_key": combo_key, "alert_id": alert.alert_id},
            timestamp=ts,
        )
        self._registered[alert_id] = datetime.now()
        self._save_registered()
        logger.info(
            "Registered combo prediction: %s for %s (%s)",
            combo_key, primary_symbol, alert.direction,
        )
        return True

    def register_pattern_stack(self, stack, primary_symbol: str) -> bool:
        """Register a pattern stack as a prediction for Thompson feedback.

        The stack key (e.g. "pattern_stack:insider+macro+technical") flows
        through OutcomeTracker to Thompson Sampler. Template IDs are stored
        in metadata so graded outcomes can update template win/loss stats.
        """
        with self._registered_lock:
            return self._register_pattern_stack_locked(stack, primary_symbol)

    def _register_pattern_stack_locked(self, stack, primary_symbol: str) -> bool:
        """Inner — caller must hold self._registered_lock."""
        if not primary_symbol:
            return False

        # Build domain key from the stack's activations
        domains = set()
        template_ids = []
        for act in stack.activations:
            domains.update(act.template.domains)
            template_ids.append(act.template.template_id)
        stack_key = "pattern_stack:" + "+".join(sorted(domains))
        from datetime import date as _date_cls
        dedup_id = f"pstack_{primary_symbol}_{stack.direction}_{len(stack.activations)}_{_date_cls.today().isoformat()}"

        if dedup_id in self._registered:
            return False

        direction = {"bullish": "up", "bearish": "down"}.get(stack.direction, "")

        # Compute dynamic window from pattern timing data
        windows = []
        for act in stack.activations:
            tmpl = act.template
            if hasattr(tmpl, "expected_move_window_days"):
                windows.append(tmpl.expected_move_window_days)
        if windows:
            windows.sort()
            window = windows[len(windows) // 2]  # median
            window_source = "dynamic"
        else:
            window = OUTCOME_WINDOWS.get("pattern_stack", 14)
            window_source = "fallback"

        self.tracker.record_prediction(
            source=stack_key,
            symbol=primary_symbol,
            direction=direction,
            outcome_window_days=window,
            metadata={
                "stack_key": stack_key,
                "template_ids": template_ids,
                "tier": stack.tier,
                "stack_confidence": stack.stack_confidence,
                "n_patterns": len(stack.activations),
                "independent_pairs": stack.independent_pairs,
                "outcome_window_source": window_source,
            },
            timestamp=stack.created_at,
        )
        self._registered[dedup_id] = datetime.now()
        self._save_registered()
        logger.info(
            "Registered pattern stack prediction: %s for %s (%s, %d patterns, window=%dd/%s)",
            stack_key, primary_symbol, stack.direction, len(stack.activations),
            window, window_source,
        )
        return True

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
        with self._registered_lock:
            return self._collect_from_archives_locked(signals_dir)

    def _collect_from_archives_locked(self, signals_dir: Path) -> int:
        """Inner — caller must hold self._registered_lock."""
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
                            timestamp=record.get("timestamp", ""),
                        )
                        self._registered[signal_id] = datetime.now()
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

    def _load_registered(self) -> dict:
        """Load dict of signal_id → registration datetime.

        Backward compatible: if the persisted data is a list or set (old format),
        each entry is migrated with a fallback timestamp of 90 days ago so they
        pass one more prune cycle before expiring naturally.
        """
        if self._registered_path.exists():
            try:
                data = json.loads(self._registered_path.read_text())
                if isinstance(data, dict):
                    # New format: {signal_id: iso_timestamp}
                    result = {}
                    for k, v in data.items():
                        try:
                            result[k] = datetime.fromisoformat(v)
                        except (ValueError, TypeError):
                            result[k] = datetime.now()
                    return result
                elif isinstance(data, list):
                    # Old format: [signal_id, ...] — migrate with fallback timestamp
                    fallback = datetime.now()
                    return {sid: fallback for sid in data if isinstance(sid, str)}
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _save_registered(self) -> None:
        """Persist registered signal IDs with their registration timestamps."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            serialized = {k: v.isoformat() for k, v in self._registered.items()}
            self._registered_path.write_text(json.dumps(serialized, indent=0))
        except Exception as e:
            logger.warning(f"Failed to persist registered signals: {e}")
