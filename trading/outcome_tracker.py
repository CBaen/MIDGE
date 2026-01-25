#!/usr/bin/env python3
"""
outcome_tracker.py - Record and evaluate prediction outcomes

The feedback loop that enables learning:
1. Record predictions when made
2. Check outcomes when due
3. Calculate accuracy
4. Feed results to learning loop
"""

import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

from trading.storage import PredictionPayload, QDRANT_URL


# Local storage for predictions (JSON file backup)
PREDICTIONS_FILE = Path("C:/Users/baenb/projects/MIDGE/data/predictions.jsonl")
OUTCOMES_FILE = Path("C:/Users/baenb/projects/MIDGE/data/outcomes.jsonl")


@dataclass
class PredictionOutcome:
    """Result of evaluating a prediction."""
    prediction_id: str
    symbol: str
    direction: str  # bullish or bearish
    predicted_confidence: float
    entry_price: float
    outcome_price: float
    was_correct: bool
    return_pct: float
    contributing_signals: List[str]
    predicted_at: str
    outcome_at: str
    timeframe: str


class OutcomeTracker:
    """
    Tracks predictions and their outcomes.

    Flow:
    1. record_prediction() - Store a new prediction
    2. check_outcomes() - Periodically check if predictions are due
    3. record_outcome() - Store the outcome
    4. get_signal_performance() - Calculate signal reliability
    """

    def __init__(self, storage_dir: str = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("C:/Users/baenb/projects/MIDGE/data")

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.storage_dir / "predictions.jsonl"
        self.outcomes_file = self.storage_dir / "outcomes.jsonl"

        # In-memory caches
        self._predictions: Dict[str, PredictionPayload] = {}
        self._outcomes: Dict[str, PredictionOutcome] = {}

        # Load existing data
        self._load_predictions()

    def _load_predictions(self):
        """Load predictions from disk."""
        if self.predictions_file.exists():
            try:
                for line in self.predictions_file.read_text().strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        pred = PredictionPayload(**data)
                        self._predictions[pred.prediction_id] = pred
            except Exception as e:
                print(f"Error loading predictions: {e}")

        if self.outcomes_file.exists():
            try:
                for line in self.outcomes_file.read_text().strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        outcome = PredictionOutcome(**data)
                        self._outcomes[outcome.prediction_id] = outcome
            except Exception as e:
                print(f"Error loading outcomes: {e}")

    def record_prediction(self, prediction: PredictionPayload) -> str:
        """
        Record a new prediction.

        Returns prediction_id for tracking.
        """
        # Store in memory
        self._predictions[prediction.prediction_id] = prediction

        # Persist to disk
        with open(self.predictions_file, "a") as f:
            f.write(json.dumps(asdict(prediction)) + "\n")

        return prediction.prediction_id

    def get_pending_predictions(self) -> List[PredictionPayload]:
        """Get predictions that haven't had outcomes recorded yet."""
        pending = []
        for pred in self._predictions.values():
            if not pred.outcome_recorded:
                pending.append(pred)
        return pending

    def get_due_predictions(self) -> List[PredictionPayload]:
        """Get predictions that are due for outcome checking."""
        now = datetime.now()
        due = []

        for pred in self._predictions.values():
            if pred.outcome_recorded:
                continue

            try:
                outcome_due = datetime.fromisoformat(pred.outcome_due)
                if now >= outcome_due:
                    due.append(pred)
            except:
                continue

        return due

    def record_outcome(self,
                      prediction_id: str,
                      outcome_price: float,
                      outcome_date: str = None) -> Optional[PredictionOutcome]:
        """
        Record the outcome for a prediction.

        Args:
            prediction_id: ID of the prediction
            outcome_price: Actual price at outcome time
            outcome_date: When the outcome was recorded

        Returns:
            PredictionOutcome with accuracy calculation
        """
        if prediction_id not in self._predictions:
            print(f"Prediction not found: {prediction_id}")
            return None

        pred = self._predictions[prediction_id]

        if outcome_date is None:
            outcome_date = datetime.now().isoformat()

        # Calculate result
        if pred.entry_price > 0:
            return_pct = (outcome_price - pred.entry_price) / pred.entry_price * 100
        else:
            return_pct = 0.0

        # Determine if prediction was correct
        if pred.direction == "bullish":
            was_correct = outcome_price > pred.entry_price
        elif pred.direction == "bearish":
            was_correct = outcome_price < pred.entry_price
        else:
            was_correct = False

        # Update prediction record
        pred.outcome_recorded = True
        pred.outcome_price = outcome_price
        pred.outcome_date = outcome_date
        pred.was_correct = was_correct
        pred.return_pct = return_pct

        # Create outcome record
        outcome = PredictionOutcome(
            prediction_id=prediction_id,
            symbol=pred.symbol,
            direction=pred.direction,
            predicted_confidence=pred.confidence,
            entry_price=pred.entry_price,
            outcome_price=outcome_price,
            was_correct=was_correct,
            return_pct=return_pct,
            contributing_signals=pred.contributing_signals,
            predicted_at=pred.predicted_at,
            outcome_at=outcome_date,
            timeframe=pred.timeframe
        )

        self._outcomes[prediction_id] = outcome

        # Persist outcome
        with open(self.outcomes_file, "a") as f:
            f.write(json.dumps(asdict(outcome)) + "\n")

        return outcome

    def check_and_record_outcomes(self, price_fetcher=None) -> List[PredictionOutcome]:
        """
        Check all due predictions and record outcomes.

        Args:
            price_fetcher: Function(symbol) -> current_price
                          If None, outcomes must be recorded manually

        Returns:
            List of newly recorded outcomes
        """
        due = self.get_due_predictions()
        recorded = []

        for pred in due:
            if price_fetcher:
                try:
                    current_price = price_fetcher(pred.symbol)
                    outcome = self.record_outcome(pred.prediction_id, current_price)
                    if outcome:
                        recorded.append(outcome)
                except Exception as e:
                    print(f"Could not fetch price for {pred.symbol}: {e}")
            else:
                print(f"Due: {pred.prediction_id} ({pred.symbol}) - needs manual outcome recording")

        return recorded

    def get_signal_performance(self) -> Dict[str, Dict]:
        """
        Calculate performance by signal source.

        Returns dict mapping signal_id -> {
            predictions: int,
            correct: int,
            accuracy: float,
            avg_confidence: float,
            avg_return: float
        }
        """
        signal_stats: Dict[str, Dict] = {}

        for outcome in self._outcomes.values():
            for signal_id in outcome.contributing_signals:
                if signal_id not in signal_stats:
                    signal_stats[signal_id] = {
                        "predictions": 0,
                        "correct": 0,
                        "total_confidence": 0.0,
                        "total_return": 0.0
                    }

                stats = signal_stats[signal_id]
                stats["predictions"] += 1
                if outcome.was_correct:
                    stats["correct"] += 1
                stats["total_confidence"] += outcome.predicted_confidence
                stats["total_return"] += outcome.return_pct

        # Calculate averages
        for signal_id, stats in signal_stats.items():
            n = stats["predictions"]
            stats["accuracy"] = stats["correct"] / n if n > 0 else 0
            stats["avg_confidence"] = stats["total_confidence"] / n if n > 0 else 0
            stats["avg_return"] = stats["total_return"] / n if n > 0 else 0

        return signal_stats

    def get_overall_stats(self) -> Dict:
        """Get overall prediction statistics."""
        total = len(self._outcomes)
        correct = sum(1 for o in self._outcomes.values() if o.was_correct)

        return {
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": correct / total if total > 0 else 0,
            "pending_predictions": len(self.get_pending_predictions()),
            "avg_return": (
                sum(o.return_pct for o in self._outcomes.values()) / total
                if total > 0 else 0
            )
        }

    def get_performance_by_timeframe(self) -> Dict[str, Dict]:
        """Get accuracy by timeframe."""
        by_tf: Dict[str, Dict] = {}

        for outcome in self._outcomes.values():
            tf = outcome.timeframe
            if tf not in by_tf:
                by_tf[tf] = {"total": 0, "correct": 0}

            by_tf[tf]["total"] += 1
            if outcome.was_correct:
                by_tf[tf]["correct"] += 1

        for tf, stats in by_tf.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        return by_tf

    def get_performance_by_direction(self) -> Dict[str, Dict]:
        """Get accuracy by prediction direction (bullish vs bearish)."""
        by_dir: Dict[str, Dict] = {}

        for outcome in self._outcomes.values():
            direction = outcome.direction
            if direction not in by_dir:
                by_dir[direction] = {"total": 0, "correct": 0, "total_return": 0}

            by_dir[direction]["total"] += 1
            if outcome.was_correct:
                by_dir[direction]["correct"] += 1
            by_dir[direction]["total_return"] += outcome.return_pct

        for d, stats in by_dir.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            stats["avg_return"] = stats["total_return"] / stats["total"] if stats["total"] > 0 else 0

        return by_dir


def create_prediction(symbol: str,
                     direction: str,
                     confidence: float,
                     entry_price: float,
                     reasoning: str,
                     contributing_signals: List[str] = None,
                     timeframe: str = "1d") -> PredictionPayload:
    """
    Convenience function to create a prediction.

    Args:
        symbol: Stock/crypto symbol
        direction: "bullish" or "bearish"
        confidence: 0.0 - 1.0
        entry_price: Current price
        reasoning: Why this prediction
        contributing_signals: List of signal IDs that informed this
        timeframe: "1h", "4h", "1d", "1w"

    Returns:
        PredictionPayload ready for tracking
    """
    # Calculate outcome_due based on timeframe
    tf_hours = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
    hours = tf_hours.get(timeframe, 24)
    outcome_due = datetime.now() + timedelta(hours=hours)

    return PredictionPayload(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        entry_price=entry_price,
        reasoning=reasoning,
        contributing_signals=contributing_signals or [],
        timeframe=timeframe,
        outcome_due=outcome_due.isoformat()
    )


class PaperTradingSession:
    """
    Simulates trading with fake capital for paper trading.

    Tracks positions, calculates PnL, and provides portfolio metrics.
    Integrates with OutcomeTracker for prediction-based position management.
    """

    def __init__(self, starting_capital: float = 10000.0, storage_dir: str = None):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {shares, entry_price, prediction_id, direction}
        self.trades: List[Dict] = []  # History of all trades
        self.portfolio_history: List[Dict] = []  # Portfolio value over time

        # Storage
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path("C:/Users/baenb/projects/midge/data")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.paper_trading_file = self.storage_dir / "paper_trading.jsonl"

        # Load existing state
        self._load_state()

    def _load_state(self):
        """Load trading state from disk."""
        if self.paper_trading_file.exists():
            try:
                lines = self.paper_trading_file.read_text().strip().split("\n")
                if lines and lines[0]:
                    # Last line contains current state
                    for line in lines:
                        if line:
                            data = json.loads(line)
                            if data.get("type") == "state":
                                self.cash = data.get("cash", self.starting_capital)
                                self.positions = data.get("positions", {})
                            elif data.get("type") == "trade":
                                self.trades.append(data)
            except Exception as e:
                print(f"Error loading paper trading state: {e}")

    def _save_state(self):
        """Save current state to disk."""
        state = {
            "type": "state",
            "timestamp": datetime.now().isoformat(),
            "cash": self.cash,
            "positions": self.positions,
            "starting_capital": self.starting_capital
        }
        with open(self.paper_trading_file, "a") as f:
            f.write(json.dumps(state) + "\n")

    def open_position(self,
                     prediction_id: str,
                     symbol: str,
                     direction: str,
                     confidence: float,
                     current_price: float,
                     max_position_pct: float = 0.1) -> Optional[Dict]:
        """
        Open a position based on a prediction.

        Position size is based on confidence (simplified Kelly criterion).
        Max 10% of portfolio per position by default.

        Args:
            prediction_id: ID of the prediction driving this trade
            symbol: Stock ticker
            direction: "bullish" (long) or "bearish" (short)
            confidence: 0.0 - 1.0
            current_price: Current market price
            max_position_pct: Maximum position size as % of portfolio

        Returns:
            Trade record or None if position couldn't be opened
        """
        # Don't open if already have position in this symbol
        if symbol in self.positions:
            print(f"Already have position in {symbol}")
            return None

        # Calculate position size (confidence-weighted, capped at max_position_pct)
        position_pct = min(confidence * 0.15, max_position_pct)  # e.g., 0.75 confidence = 11.25%, capped at 10%
        position_value = self.cash * position_pct

        if position_value < 1:
            print(f"Insufficient cash for position")
            return None

        # Calculate shares (round down)
        shares = int(position_value / current_price)
        if shares < 1:
            shares = 1

        actual_cost = shares * current_price

        if actual_cost > self.cash:
            print(f"Insufficient cash: need ${actual_cost:.2f}, have ${self.cash:.2f}")
            return None

        # Open position
        self.cash -= actual_cost
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": current_price,
            "prediction_id": prediction_id,
            "direction": direction,
            "opened_at": datetime.now().isoformat()
        }

        # Record trade
        trade = {
            "type": "trade",
            "action": "OPEN",
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "price": current_price,
            "value": actual_cost,
            "prediction_id": prediction_id,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.trades.append(trade)

        with open(self.paper_trading_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

        self._save_state()

        print(f"Opened {direction.upper()} position: {shares} {symbol} @ ${current_price:.2f} = ${actual_cost:.2f}")
        return trade

    def close_position(self, symbol: str, current_price: float, reason: str = "manual") -> Optional[Dict]:
        """
        Close an existing position.

        Args:
            symbol: Stock ticker
            current_price: Current market price
            reason: Why closing (manual, stop_loss, take_profit, prediction_due)

        Returns:
            Trade record with PnL or None if no position
        """
        if symbol not in self.positions:
            print(f"No position in {symbol}")
            return None

        position = self.positions[symbol]
        shares = position["shares"]
        entry_price = position["entry_price"]
        direction = position["direction"]

        # Calculate PnL
        if direction == "bullish":
            pnl = (current_price - entry_price) * shares
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:  # bearish (short)
            pnl = (entry_price - current_price) * shares
            pnl_pct = (entry_price - current_price) / entry_price * 100

        # Update cash
        exit_value = shares * current_price
        self.cash += exit_value

        # Record trade
        trade = {
            "type": "trade",
            "action": "CLOSE",
            "symbol": symbol,
            "direction": direction,
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": current_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "prediction_id": position["prediction_id"],
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self.trades.append(trade)

        with open(self.paper_trading_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

        # Remove position
        del self.positions[symbol]
        self._save_state()

        result_str = "PROFIT" if pnl > 0 else "LOSS"
        print(f"Closed {symbol}: {result_str} ${abs(pnl):.2f} ({pnl_pct:+.2f}%)")
        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Args:
            current_prices: Dict mapping symbol -> current price

        Returns:
            Total portfolio value
        """
        total = self.cash

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total += position["shares"] * current_prices[symbol]
            else:
                # Use entry price if current price not available
                total += position["shares"] * position["entry_price"]

        return total

    def get_stats(self, current_prices: Dict[str, float] = None) -> Dict:
        """Get portfolio statistics."""
        if current_prices is None:
            current_prices = {}

        portfolio_value = self.get_portfolio_value(current_prices)
        total_pnl = portfolio_value - self.starting_capital
        total_return = (total_pnl / self.starting_capital) * 100

        # Trade statistics
        winning_trades = [t for t in self.trades if t.get("action") == "CLOSE" and t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("action") == "CLOSE" and t.get("pnl", 0) < 0]
        closed_trades = [t for t in self.trades if t.get("action") == "CLOSE"]

        win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

        return {
            "starting_capital": self.starting_capital,
            "current_cash": self.cash,
            "portfolio_value": portfolio_value,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "open_positions": len(self.positions),
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "positions": self.positions
        }

    def print_status(self, current_prices: Dict[str, float] = None):
        """Print current portfolio status."""
        stats = self.get_stats(current_prices)

        print("\n" + "=" * 50)
        print("PAPER TRADING STATUS")
        print("=" * 50)
        print(f"Starting Capital: ${stats['starting_capital']:.2f}")
        print(f"Current Cash: ${stats['current_cash']:.2f}")
        print(f"Portfolio Value: ${stats['portfolio_value']:.2f}")
        print(f"Total P&L: ${stats['total_pnl']:+.2f} ({stats['total_return_pct']:+.2f}%)")
        print(f"Win Rate: {stats['win_rate']:.1%} ({stats['winning_trades']}/{stats['total_trades']})")

        if self.positions:
            print(f"\nOpen Positions ({len(self.positions)}):")
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f} ({pos['direction']})")


if __name__ == "__main__":
    print("Outcome Tracker - Prediction Feedback System")
    print("=" * 50)

    tracker = OutcomeTracker()

    # Show current stats
    stats = tracker.get_overall_stats()
    print(f"\nCurrent Statistics:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  Correct: {stats['correct_predictions']}")
    print(f"  Accuracy: {stats['accuracy']:.1%}")
    print(f"  Pending: {stats['pending_predictions']}")

    # Show due predictions
    due = tracker.get_due_predictions()
    if due:
        print(f"\n{len(due)} predictions due for outcome recording:")
        for pred in due[:5]:
            print(f"  - {pred.symbol} ({pred.direction}) - Entry: ${pred.entry_price:.2f}")

    # Example: Create and record a prediction
    print("\n" + "=" * 50)
    print("Example: Creating a test prediction")

    pred = create_prediction(
        symbol="TEST",
        direction="bullish",
        confidence=0.75,
        entry_price=100.0,
        reasoning="Test prediction for system validation",
        contributing_signals=["signal_1", "signal_2"],
        timeframe="1d"
    )

    pred_id = tracker.record_prediction(pred)
    print(f"Recorded prediction: {pred_id}")

    # Simulate outcome
    outcome = tracker.record_outcome(pred_id, outcome_price=105.0)
    if outcome:
        print(f"Outcome recorded: {'Correct' if outcome.was_correct else 'Wrong'}")
        print(f"  Return: {outcome.return_pct:.1f}%")
