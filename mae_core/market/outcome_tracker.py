#!/usr/bin/env python3
"""
outcome_tracker.py - Bayesian Feedback Loop Closer

Reads predictions.jsonl, checks price movement after outcome_window_days
has elapsed, and updates ThompsonSampler with success/failure outcomes.

Without this, the Thompson Sampler's Beta distributions never update from
real market results — the Bayesian brain never learns.

Price logic:
- Entry price: closing price on signal_date (via PriceFetcher.get_historical_price)
- Exit price: closing price on signal_date + window_days
- Success: abs(pct_change) >= min_price_move_pct AND direction matches (if specified)

Prediction record format (from predictions.jsonl):
    {
        "signal_id": "uuid",
        "source": "sec_edgar",
        "symbol": "AAPL",
        "direction": "up" | "down" | "",   # optional
        "timestamp": "2026-02-01T10:00:00",
        "outcome_window_days": 14,
        "outcome_symbol": "AAPL"            # optional override
    }
"""

import json
import logging
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Data directory — same resolution pattern as thompson_sampler.py
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


class OutcomeTracker:
    """
    Closes the Bayesian feedback loop.

    Reads predictions.jsonl, checks prices after outcome_window_days,
    updates ThompsonSampler with success/failure.

    Injected dependencies:
        price_fetcher: PriceFetcher instance (from mae_core.market.apis.price_fetcher)
        thompson_sampler: ThompsonSampler instance
    """

    def __init__(self, price_fetcher, thompson_sampler, regime_classifier=None, data_dir: Path = None):
        """
        Args:
            price_fetcher: PriceFetcher instance with get_historical_price() support
            thompson_sampler: ThompsonSampler instance with update() method
            regime_classifier: Optional RegimeClassifier for regime-aware updates
            data_dir: Override for data directory (default: DATA_DIR)
        """
        self.price_fetcher = price_fetcher
        self.thompson_sampler = thompson_sampler
        self.regime_classifier = regime_classifier
        self.data_dir = data_dir or DATA_DIR
        self.predictions_path = self.data_dir / "predictions.jsonl"
        self.outcomes_path = self.data_dir / "outcomes.jsonl"
        self.min_price_move_pct = 2.0  # 2% move = successful prediction
        self._logger = logger

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record_prediction(
        self,
        source: str,
        symbol: str,
        direction: str = "",
        outcome_window_days: int = 14,
        metadata: dict = None,
        timestamp: str = "",
    ) -> str:
        """
        Record a prediction for later outcome evaluation.

        Called when a TradeSignal or convergence alert is generated.
        The prediction is written to predictions.jsonl and will be
        evaluated by check_pending_outcomes() once the window elapses.

        Args:
            source: Signal source (e.g. "sec_edgar", "convergence")
            symbol: Stock ticker to track
            direction: "up", "down", or "" (direction-agnostic)
            outcome_window_days: Days to wait before evaluating
            metadata: Optional extra context
            timestamp: ISO timestamp override (for archive backfill).
                       Uses datetime.now() if empty.

        Returns:
            signal_id (UUID) for tracking
        """
        signal_id = str(uuid.uuid4())
        record = {
            "signal_id": signal_id,
            "source": source,
            "symbol": symbol.upper(),
            "direction": direction,
            "timestamp": timestamp or datetime.now().isoformat(),
            "outcome_window_days": outcome_window_days,
            "outcome_symbol": symbol.upper(),
        }
        if metadata:
            record["metadata"] = metadata

        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.predictions_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            self._logger.warning(f"Failed to write prediction: {e}")

        return signal_id

    def check_pending_outcomes(self) -> int:
        """
        Check all predictions where outcome_window_days has elapsed.

        For each mature prediction:
          1. Fetch entry price at signal_date and exit price at outcome_date
          2. Compute percentage change
          3. Determine success (magnitude + direction match if direction provided)
          4. Update ThompsonSampler
          5. Append to outcomes.jsonl
          6. Remove from predictions.jsonl

        Returns:
            Count of outcomes evaluated this call.
        """
        if not self.predictions_path.exists():
            return 0

        evaluated = 0
        today = datetime.now()
        remaining_predictions = []

        # Load already-evaluated signal_ids to prevent duplication across runs
        already_evaluated = self._load_evaluated_ids()

        # Read all predictions
        predictions = []
        with open(self.predictions_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        for pred in predictions:
            try:
                # Handle both new format ("timestamp") and legacy ("predicted_at")
                ts_str = pred.get("timestamp") or pred.get("predicted_at", "")
                if not ts_str:
                    self._logger.warning("Prediction missing timestamp, skipping")
                    continue
                signal_timestamp = datetime.fromisoformat(ts_str)
                window_days = pred.get("outcome_window_days", 14)
                outcome_date = signal_timestamp + timedelta(days=window_days)

                if outcome_date > today:
                    # Window not elapsed — keep for later
                    remaining_predictions.append(pred)
                    continue

                # Skip if already evaluated (prevents duplication across runs)
                signal_id = pred.get("signal_id", "")
                if pred.get("evaluated", False) or signal_id in already_evaluated:
                    continue

                outcome_symbol = pred.get("outcome_symbol") or pred.get("symbol", "")
                if not outcome_symbol:
                    self._logger.warning("Prediction missing symbol, skipping")
                    continue

                price_result = self._check_price_movement(
                    outcome_symbol,
                    signal_timestamp,
                    outcome_date,
                    pred.get("direction", ""),
                )

                if price_result is None:
                    # Price unavailable — keep for retry
                    remaining_predictions.append(pred)
                    continue

                price_change_pct, success = price_result

                # Update Thompson Sampler (regime-aware)
                source = pred.get("source", "unknown")
                regime = "default"
                if self.regime_classifier is not None:
                    try:
                        regime = self.regime_classifier.classify()
                    except Exception:
                        pass
                try:
                    self.thompson_sampler.update(source, success=success, regime=regime)
                except Exception as e:
                    self._logger.warning(f"Thompson update failed for {source}: {e}")

                # Record outcome
                outcome = {
                    "signal_id": pred.get("signal_id", ""),
                    "source": source,
                    "symbol": outcome_symbol,
                    "direction": pred.get("direction", ""),
                    "predicted_at": pred.get("timestamp", ""),
                    "evaluated_at": today.isoformat(),
                    "window_days": window_days,
                    "price_change_pct": round(price_change_pct, 4),
                    "success": success,
                    "min_move_threshold": self.min_price_move_pct,
                }
                self._append_outcome(outcome)
                evaluated += 1

            except (KeyError, ValueError) as e:
                self._logger.warning(f"Skipping malformed prediction: {e}")
                remaining_predictions.append(pred)
                continue

        # Rewrite predictions file with only unprocessed entries
        if evaluated > 0:
            with open(self.predictions_path, "w") as f:
                for pred in remaining_predictions:
                    f.write(json.dumps(pred) + "\n")

        if evaluated > 0:
            self._logger.info(f"OutcomeTracker evaluated {evaluated} predictions")

        return evaluated

    def _check_price_movement(
        self,
        symbol: str,
        signal_date: datetime,
        outcome_date: datetime,
        direction: str,
    ) -> Optional[Tuple[float, bool]]:
        """
        Compute price movement from signal_date to outcome_date.

        Uses PriceFetcher.get_historical_price() for both entry and exit.
        Falls back to get_current_price() for the exit price when outcome_date
        is today (i.e., window just elapsed).

        Args:
            symbol: Stock ticker
            signal_date: When the prediction was made
            outcome_date: signal_date + window_days
            direction: "up", "down", or "" (empty = direction-agnostic)

        Returns:
            (price_change_pct, success) or None if prices unavailable.
            price_change_pct is positive for gains, negative for losses.
            success = magnitude >= min_price_move_pct AND direction matches.
        """
        try:
            entry_date_str = signal_date.strftime("%Y-%m-%d")
            exit_date_str = outcome_date.strftime("%Y-%m-%d")

            entry_data = self.price_fetcher.get_historical_price(symbol, entry_date_str)
            if entry_data is None or entry_data.price == 0:
                self._logger.debug(f"No entry price for {symbol} on {entry_date_str}")
                return None

            # For exit: try historical first, fall back to current if same-day
            exit_data = self.price_fetcher.get_historical_price(symbol, exit_date_str)
            if exit_data is None or exit_data.price == 0:
                # Outcome date is today or very recent — use current price
                exit_data = self.price_fetcher.get_current_price(symbol)
            if exit_data is None or exit_data.price == 0:
                self._logger.debug(f"No exit price for {symbol} on {exit_date_str}")
                return None

            entry_price = entry_data.price
            exit_price = exit_data.price
            pct_change = ((exit_price - entry_price) / entry_price) * 100.0

            # Determine success
            magnitude_ok = abs(pct_change) >= self.min_price_move_pct

            if direction == "up":
                direction_ok = pct_change > 0
            elif direction == "down":
                direction_ok = pct_change < 0
            else:
                # No direction specified — magnitude alone determines success
                direction_ok = True

            success = magnitude_ok and direction_ok
            return (pct_change, success)

        except Exception as e:
            self._logger.debug(f"Price check failed for {symbol}: {e}")
            return None

    def _load_evaluated_ids(self) -> set:
        """Load signal_ids already recorded in outcomes.jsonl to prevent duplication."""
        evaluated = set()
        if not self.outcomes_path.exists():
            return evaluated
        try:
            with open(self.outcomes_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        sid = record.get("signal_id", "")
                        if sid:
                            evaluated.add(sid)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            self._logger.warning(f"Failed to read outcomes for dedup: {e}")
        return evaluated

    def _append_outcome(self, outcome: dict) -> None:
        """Append outcome record to outcomes.jsonl."""
        try:
            with open(self.outcomes_path, "a") as f:
                f.write(json.dumps(outcome) + "\n")
        except Exception as e:
            self._logger.warning(f"Failed to write outcome: {e}")

    def _count_pending(self) -> int:
        """Count predictions still awaiting evaluation."""
        if not self.predictions_path.exists():
            return 0
        count = 0
        with open(self.predictions_path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _count_evaluated(self) -> int:
        """Count total evaluated outcomes."""
        if not self.outcomes_path.exists():
            return 0
        count = 0
        with open(self.outcomes_path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "pending_predictions": self._count_pending(),
            "total_evaluated": self._count_evaluated(),
        }
