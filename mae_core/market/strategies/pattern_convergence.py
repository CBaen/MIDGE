"""
pattern_convergence.py — Math-first convergence for crypto trading.

Fires when 3+ independently validated strategies agree on the same
ticker + direction. This is the crypto equivalent of ConvergenceAlerter
but using algorithmic consensus instead of data-source diversity.

Instead of requiring signals from 3 different data domains (SEC + insider +
technical), this engine requires 3 different mathematical strategies (RSI +
MACD + Bollinger) to independently agree on the same ticker and direction.
Each strategy must have a validated backtest record on that symbol before
its vote counts.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from mae_core.market.strategies.models import PatternConvergenceAlert, StrategyResult
from mae_core.market.strategies.crypto_ohlcv import get_ohlcv
from mae_core.market.strategies.strategy_library import ALL_STRATEGIES

logger = logging.getLogger("midge.market.strategies")


class PatternConvergenceEngine:
    """
    Evaluates crypto symbols against all validated strategies.
    Fires PatternConvergenceAlert when 3+ agree.

    The engine is deliberately dependency-minimal: price_fetcher, thompson_sampler,
    and event_bus are all optional so it can be unit-tested without a running
    Mae bootstrap. When event_bus is provided, alerts are also published to
    the EventBus for downstream subscribers.
    """

    MIN_STRATEGIES = 3      # Law 7 minimum — triadic consensus
    CONFIDENCE_GATE = 0.50  # Aggregate confidence must clear this floor
    DEDUP_HOURS = 4.0       # Suppress repeat alerts for same symbol+direction

    def __init__(
        self,
        strategy_registry,      # StrategyRegistry
        price_fetcher=None,     # PriceFetcher (optional, fallback to crypto_ohlcv)
        thompson_sampler=None,  # ThompsonSampler (optional)
        event_bus=None,         # EventBus (optional)
    ) -> None:
        self._registry = strategy_registry
        self._price_fetcher = price_fetcher
        self._thompson = thompson_sampler
        self._bus = event_bus
        self._last_alert_times: Dict[str, datetime] = {}  # "symbol:direction" -> datetime
        self._alert_counter = 0

    # ------------------------------------------------------------------
    # Primary evaluation interface
    # ------------------------------------------------------------------

    def evaluate(self, symbol: str) -> Optional[PatternConvergenceAlert]:
        """Evaluate a single symbol against all validated strategies.

        Algorithm
        ---------
        1. Fetch OHLCV for symbol (90 days, 5-min TTL cache via crypto_ohlcv).
        2. Get the list of strategy names validated for this symbol from the registry.
        3. Run each validated strategy function against the OHLCV DataFrame.
        4. Group non-None results by direction ("bullish" / "bearish").
        5. For any direction with >= MIN_STRATEGIES results:
           a. Check deduplication: skip if same symbol+direction alerted < DEDUP_HOURS ago.
           b. Compute aggregate confidence (geometric mean of per-strategy confidences).
           c. Skip if aggregate confidence < CONFIDENCE_GATE.
           d. Compute aggregate strength (arithmetic mean of per-strategy strengths).
           e. stop_loss: widest (most conservative) level across all strategies.
           f. take_profit: nearest (most conservative) level across all strategies.
           g. Construct and return PatternConvergenceAlert.
        6. When both directions qualify, return the one with more strategies agreeing.
           Ties resolved by higher aggregate confidence.

        Parameters
        ----------
        symbol : str
            Yahoo Finance ticker (e.g. "BTC-USD").

        Returns
        -------
        PatternConvergenceAlert or None
        """
        df = get_ohlcv(symbol, days=90)
        if df is None:
            logger.debug("evaluate(%s): no OHLCV data — skipping", symbol)
            return None

        validated_names = self._registry.get_validated_strategies(symbol)
        if len(validated_names) < self.MIN_STRATEGIES:
            logger.debug(
                "evaluate(%s): only %d validated strategies (need %d) — skipping",
                symbol, len(validated_names), self.MIN_STRATEGIES,
            )
            return None

        # Build a quick lookup from name to callable
        strategy_map = {name: fn for name, fn in ALL_STRATEGIES}

        # Run each validated strategy
        results_by_direction: Dict[str, List[StrategyResult]] = {
            "bullish": [],
            "bearish": [],
        }
        for name in validated_names:
            fn = strategy_map.get(name)
            if fn is None:
                logger.warning("evaluate(%s): strategy '%s' not found in ALL_STRATEGIES", symbol, name)
                continue
            # Apply the cross-market confidence prior from the registry
            confidence_prior = self._registry.get_confidence_prior(name)
            try:
                result = fn(symbol, df)
            except Exception:
                logger.debug("Strategy %s raised on %s", name, symbol, exc_info=True)
                continue
            if result is None:
                continue
            # Overwrite the default 0.55 prior with the registry's cross-market estimate
            result.confidence = confidence_prior
            results_by_direction[result.direction].append(result)

        # Find qualifying directions
        candidates: List[PatternConvergenceAlert] = []
        for direction, results in results_by_direction.items():
            if len(results) < self.MIN_STRATEGIES:
                continue

            dedup_key = f"{symbol}:{direction}"
            if self._is_deduped(dedup_key):
                logger.debug(
                    "evaluate(%s): dedup suppressed %s alert (within %.1fh window)",
                    symbol, direction, self.DEDUP_HOURS,
                )
                continue

            aggregate_confidence = self._geometric_mean_confidence(results)
            if aggregate_confidence < self.CONFIDENCE_GATE:
                logger.debug(
                    "evaluate(%s): %s confidence %.3f below gate %.3f",
                    symbol, direction, aggregate_confidence, self.CONFIDENCE_GATE,
                )
                continue

            aggregate_strength = sum(r.strength for r in results) / len(results)
            sl, tp = self._conservative_stops(results, direction)

            self._alert_counter += 1
            alert = PatternConvergenceAlert(
                alert_id=f"PCE-{symbol}-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}",
                symbol=symbol,
                direction=direction,
                strategy_names=[r.strategy_name for r in results],
                confidence=round(aggregate_confidence, 4),
                strength=round(aggregate_strength, 4),
                stop_loss=sl,
                take_profit=tp,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "per_strategy_confidence": {r.strategy_name: round(r.confidence, 4) for r in results},
                    "per_strategy_strength": {r.strategy_name: round(r.strength, 4) for r in results},
                    "strategy_count": len(results),
                },
            )
            candidates.append(alert)

        if not candidates:
            return None

        # Pick the best candidate: most strategies > higher confidence
        best = max(
            candidates,
            key=lambda a: (len(a.strategy_names), a.confidence),
        )

        # Record dedup timestamp and optionally publish
        self._last_alert_times[f"{best.symbol}:{best.direction}"] = datetime.now()
        self._maybe_publish(best)

        logger.info(
            "PatternConvergence: %s %s — %d strategies agree (conf=%.3f str=%.3f)",
            best.symbol, best.direction.upper(), len(best.strategy_names),
            best.confidence, best.strength,
        )
        return best

    def evaluate_many(self, symbols: List[str]) -> List[PatternConvergenceAlert]:
        """Run evaluate() on a list of symbols and return all non-None alerts.

        Failures on individual symbols are logged at DEBUG and do not abort
        the batch — this mirrors ConvergenceAlerter's resilience pattern.

        Parameters
        ----------
        symbols : list[str]
            Yahoo Finance tickers to evaluate (e.g. ["BTC-USD", "ETH-USD"]).

        Returns
        -------
        list[PatternConvergenceAlert]
            All alerts generated across all symbols (may be empty).
        """
        alerts: List[PatternConvergenceAlert] = []
        for sym in symbols:
            try:
                alert = self.evaluate(sym)
                if alert:
                    alerts.append(alert)
            except Exception:
                logger.debug("Pattern convergence eval failed for %s", sym, exc_info=True)
        return alerts

    # ------------------------------------------------------------------
    # ConvergenceAlert bridge
    # ------------------------------------------------------------------

    def to_convergence_alert(self, alert: PatternConvergenceAlert):
        """Convert PatternConvergenceAlert to a standard ConvergenceAlert.

        This allows PatternConvergenceAlerts to flow through the existing
        _run_paper_trading_gate in market_hooks.py without modification.

        Alert ID format
        ---------------
        MUST start with "TCKR-" — the Session 13 filter in market_hooks.py
        extracts the ticker from convergence alert IDs using this prefix.
        Format: TCKR-{symbol}-{YYYYMMDD}-{counter:04d}

        Signal construction
        -------------------
        Each contributing strategy becomes one Signal. All signals share:
        - domain = "technical"  (all strategies are technical analysis)
        - metadata["symbol"]   (REQUIRED — ticker extraction depends on this)
        - source = "strategy_{strategy_name}"

        The combo_key follows the pattern used by ConvergenceAlerter's Thompson
        sampler so that combo-level Beta distributions accumulate over time.
        """
        # Import here to keep the module import graph clean — convergence_models
        # is only needed by callers that bridge into the existing pipeline.
        from mae_core.market.intelligence.convergence_models import ConvergenceAlert, Signal

        now = datetime.now()
        signals = []
        for sname in alert.strategy_names:
            signals.append(Signal(
                signal_id=f"strategy:{sname}:{alert.symbol}",
                strength=alert.strength,
                domain="technical",
                direction=alert.direction,
                timestamp=now,
                metadata={"symbol": alert.symbol, "strategy": sname},
                source=f"strategy_{sname}",
                confidence=alert.confidence,
            ))

        self._alert_counter += 1
        sorted_names = sorted(alert.strategy_names)
        combo_key = "combo:" + "+".join(sorted_names)

        return ConvergenceAlert(
            alert_id=f"TCKR-{alert.symbol}-{now.strftime('%Y%m%d')}-{self._alert_counter:04d}",
            timestamp=now,
            direction=alert.direction,
            strength=alert.strength,
            confidence=alert.confidence,
            domains_converging=[f"strategy_{s}" for s in alert.strategy_names],
            signals=signals,
            cross_domain_count=len(alert.strategy_names),
            summary=(
                f"PATTERN {alert.symbol} {alert.direction.upper()}: "
                f"{len(alert.strategy_names)} strategies agree "
                f"({', '.join(alert.strategy_names)})"
            ),
            urgency="hours",  # crypto resolves in hours, not days
            combo_key=combo_key,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return a health-check snapshot for the daemon heartbeat logger."""
        return {
            "alert_counter": self._alert_counter,
            "dedup_entries": len(self._last_alert_times),
            "min_strategies": self.MIN_STRATEGIES,
            "confidence_gate": self.CONFIDENCE_GATE,
            "dedup_hours": self.DEDUP_HOURS,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_deduped(self, dedup_key: str) -> bool:
        """Return True if an alert for this key was fired within DEDUP_HOURS."""
        last = self._last_alert_times.get(dedup_key)
        if last is None:
            return False
        return (datetime.now() - last) < timedelta(hours=self.DEDUP_HOURS)

    @staticmethod
    def _geometric_mean_confidence(results: List[StrategyResult]) -> float:
        """Compute the geometric mean of per-strategy confidence values.

        Geometric mean is used rather than arithmetic mean because confidence
        values are probabilities: a single very-low-confidence strategy should
        drag the aggregate down multiplicatively, not just additionally.
        Any zero-confidence strategy makes the aggregate zero (veto semantics).
        """
        if not results:
            return 0.0
        log_sum = sum(math.log(max(r.confidence, 1e-9)) for r in results)
        return math.exp(log_sum / len(results))

    @staticmethod
    def _conservative_stops(
        results: List[StrategyResult],
        direction: str,
    ) -> tuple[float, float]:
        """Select the most conservative stop-loss and take-profit across strategies.

        Conservative = protecting capital over maximising reward:
        - Bullish: widest SL = lowest stop_loss value; nearest TP = lowest take_profit value.
        - Bearish: widest SL = highest stop_loss value; nearest TP = highest take_profit value.

        Strategies that returned 0.0 for stops (insufficient data) are excluded
        to avoid pulling the consensus toward a degenerate zero value.

        Returns (stop_loss, take_profit) as absolute price levels.
        """
        valid_sl = [r.stop_loss for r in results if r.stop_loss != 0.0]
        valid_tp = [r.take_profit for r in results if r.take_profit != 0.0]

        if not valid_sl or not valid_tp:
            return 0.0, 0.0

        if direction == "bullish":
            # Widest SL = furthest below price = minimum value
            # Nearest TP = closest above price = minimum value
            return min(valid_sl), min(valid_tp)
        else:
            # Widest SL = furthest above price = maximum value
            # Nearest TP = closest below price = maximum value
            return max(valid_sl), max(valid_tp)

    def _maybe_publish(self, alert: PatternConvergenceAlert) -> None:
        """Publish alert to EventBus if one is wired in."""
        if self._bus is None:
            return
        try:
            self._bus.publish(
                "CH_PATTERN_CONVERGENCE",
                {
                    "alert_id": alert.alert_id,
                    "symbol": alert.symbol,
                    "direction": alert.direction,
                    "strategy_names": alert.strategy_names,
                    "confidence": alert.confidence,
                    "strength": alert.strength,
                    "stop_loss": alert.stop_loss,
                    "take_profit": alert.take_profit,
                    "timestamp": alert.timestamp,
                },
            )
        except Exception:
            logger.debug("EventBus publish failed for PatternConvergenceAlert", exc_info=True)
