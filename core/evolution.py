#!/usr/bin/env python3
"""
evolution.py - Unified Self-Improving Trading Intelligence

Merges MIDGE's pattern detection with Emergence's self-improvement loop.

The cycle:
1. GATHER - Collect trading data (SEC, USASpending, prices)
2. DETECT - Find patterns (politician/contract correlations, technicals)
3. PREDICT - Record predictions with contributing signals
4. OBSERVE - Fetch outcomes when due
5. LEARN - Update signal weights based on accuracy
6. RESEARCH - Use LLM to analyze what's working/failing
7. EVOLVE - Modify strategies based on research
8. REPEAT

This is the brain that learns from itself.
"""

import json
import os
import sys
import time
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# LLM Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_SCRIPT = Path.home() / ".claude" / "scripts" / "gemini-account.sh"

# MIDGE components
from trading.edge.politician_tracker import PoliticianTracker
from trading.outcome_tracker import OutcomeTracker, create_prediction
from trading.learning_loop import LearningLoop
from trading.storage import TradingVectorStore, SignalPayload

# Try to import optional components
try:
    from trading.apis.price_fetcher import PriceFetcher, price_fetcher_for_outcomes
    PRICE_FETCHER_AVAILABLE = True
except ImportError:
    PRICE_FETCHER_AVAILABLE = False
    print("Price fetcher not available - install yfinance")


@dataclass
class EvolutionCycle:
    """Record of one evolution cycle."""
    cycle_id: str
    timestamp: str
    phase: str  # gather, detect, predict, observe, learn, research, evolve
    success: bool
    findings: Dict
    changes_made: List[str]
    lessons: List[str]
    error: Optional[str] = None


class MIDGEEvolution:
    """
    Unified evolution loop for MIDGE.

    Combines:
    - MIDGE's pattern detection and prediction tracking
    - Emergence's self-improvement and research patterns
    """

    def __init__(self, memory_path: str = None):
        # Core components
        self.politician_tracker = PoliticianTracker()
        self.outcome_tracker = OutcomeTracker()
        self.learning_loop = LearningLoop()

        # Optional components
        self.price_fetcher = PriceFetcher() if PRICE_FETCHER_AVAILABLE else None
        self.vector_store = None
        try:
            self.vector_store = TradingVectorStore()
        except:
            print("Vector store not available - Qdrant may not be running")

        # Evolution state
        if memory_path:
            self.memory_path = Path(memory_path)
        else:
            self.memory_path = project_root / ".claude" / "evolution" / "MEMORY.json"
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)

        self.memory = self._load_memory()
        self.cycle_count = self.memory.get("cycle_count", 0)

    def _load_memory(self) -> Dict:
        """Load evolution memory from disk."""
        if self.memory_path.exists():
            try:
                return json.loads(self.memory_path.read_text())
            except:
                pass
        return {
            "cycle_count": 0,
            "lessons": [],
            "successful_patterns": [],
            "failed_patterns": [],
            "signal_history": {},
            "last_research": None
        }

    def _save_memory(self):
        """Persist evolution memory to disk."""
        self.memory["cycle_count"] = self.cycle_count
        self.memory_path.write_text(json.dumps(self.memory, indent=2))

    def _log(self, message: str, level: str = "INFO"):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")

    def _query_deepseek(self, prompt: str, max_tokens: int = 4096) -> Optional[str]:
        """
        Query DeepSeek API (cheap, designed for 24/7 operation).

        Args:
            prompt: The question/prompt to send
            max_tokens: Maximum response length

        Returns:
            DeepSeek's response or None if failed
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            self._log("DEEPSEEK_API_KEY not set", "WARN")
            return None

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=180
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                self._log(f"DeepSeek API error: {response.status_code}", "WARN")
                return None

        except Exception as e:
            self._log(f"DeepSeek error: {e}", "WARN")
            return None

    def _query_gemini(self, prompt: str, account: int = 1) -> Optional[str]:
        """
        Query Gemini using the lineage wrapper script (fallback).

        Args:
            prompt: The question/prompt to send
            account: Which Gemini account to use (1 or 2)

        Returns:
            Gemini's response or None if failed
        """
        if not GEMINI_SCRIPT.exists():
            self._log("Gemini script not found", "WARN")
            return None

        try:
            result = subprocess.run(
                ["bash", str(GEMINI_SCRIPT), str(account), prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path.home())
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self._log(f"Gemini error: {result.stderr[:200]}", "WARN")
                return None

        except subprocess.TimeoutExpired:
            self._log("Gemini query timed out", "WARN")
            return None
        except Exception as e:
            self._log(f"Gemini query failed: {e}", "WARN")
            return None

    def _query_llm(self, prompt: str, prefer: str = "deepseek") -> Optional[str]:
        """
        Query an LLM with fallback.

        Args:
            prompt: The question/prompt to send
            prefer: Preferred LLM ("deepseek" or "gemini")

        Returns:
            LLM response or None if all failed
        """
        if prefer == "deepseek":
            # Try DeepSeek first (cheap, 24/7)
            result = self._query_deepseek(prompt)
            if result:
                return result
            # Fall back to Gemini
            self._log("  DeepSeek unavailable, trying Gemini...", "INFO")
            return self._query_gemini(prompt)
        else:
            # Try Gemini first
            result = self._query_gemini(prompt)
            if result:
                return result
            # Fall back to DeepSeek
            self._log("  Gemini unavailable, trying DeepSeek...", "INFO")
            return self._query_deepseek(prompt)

    # =========================================================================
    # PHASE 1: GATHER
    # =========================================================================

    def gather_data(self, symbols: List[str] = None) -> Dict:
        """
        Gather fresh trading data from all sources.

        Returns dict with data from each source.
        """
        self._log("GATHER: Collecting trading data...")

        if symbols is None:
            symbols = ["LMT", "RTX", "BA", "MSFT", "AAPL"]

        results = {
            "prices": {},
            "insider_trades": [],
            "contracts": [],
            "gathered_at": datetime.now().isoformat()
        }

        # Gather prices
        if self.price_fetcher:
            try:
                for symbol in symbols:
                    price_data = self.price_fetcher.get_current_price(symbol)
                    if price_data:
                        results["prices"][symbol] = price_data.price
                self._log(f"  Gathered prices for {len(results['prices'])} symbols")
            except Exception as e:
                self._log(f"  Price gathering failed: {e}", "WARN")

        # Note: Insider trades and contracts are gathered in detect phase
        # to avoid duplicate API calls

        return results

    # =========================================================================
    # PHASE 2: DETECT
    # =========================================================================

    def detect_patterns(self, symbols: List[str] = None) -> List[Dict]:
        """
        Detect trading patterns from ALL signal sources.

        Queries:
        - midge_signals collection for technical indicators, insider trades, contracts
        - Politician tracker for direct correlations
        - Aggregates multiple aligned signals into stronger patterns

        Returns list of detected patterns with confidence scores.
        """
        self._log("DETECT: Finding patterns...")

        if symbols is None:
            symbols = ["LMT", "RTX", "BA", "MSFT", "AAPL"]

        patterns = []

        # =====================================================================
        # 1. Query ALL signals from Qdrant (midge_signals collection)
        # =====================================================================
        signal_counts = {"technical": 0, "insider": 0, "contract": 0, "politician": 0}

        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host="localhost", port=6333)

            # Get recent signals (last 24 hours) for our symbols
            from qdrant_client.models import Filter, FieldCondition, MatchAny, Range
            from datetime import datetime, timedelta

            cutoff = (datetime.now() - timedelta(hours=24)).isoformat()

            # Group signals by symbol for pattern aggregation
            symbol_signals = {s: [] for s in symbols}

            # Query for each symbol
            for symbol in symbols:
                try:
                    results = client.scroll(
                        collection_name="midge_signals",
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(key="symbol", match=MatchAny(any=[symbol])),
                            ]
                        ),
                        limit=100,
                        with_payload=True
                    )

                    for point in results[0]:
                        payload = point.payload
                        signal_counts[payload.get("signal_source", "unknown")] = \
                            signal_counts.get(payload.get("signal_source", "unknown"), 0) + 1
                        symbol_signals[symbol].append(payload)

                except Exception as e:
                    self._log(f"  Query for {symbol} failed: {e}", "WARN")

            self._log(f"  Retrieved signals: {dict(signal_counts)}")

            # =====================================================================
            # 2. Aggregate signals into patterns
            # =====================================================================
            for symbol, signals in symbol_signals.items():
                if not signals:
                    continue

                # Count bullish vs bearish signals
                bullish_signals = [s for s in signals if s.get("direction") == "bullish"]
                bearish_signals = [s for s in signals if s.get("direction") == "bearish"]

                # Calculate weighted confidence (more signals = higher confidence)
                if len(bullish_signals) >= 2:
                    # Multiple bullish signals align
                    avg_confidence = sum(s.get("confidence", 0.5) for s in bullish_signals) / len(bullish_signals)
                    # Boost for signal alignment (more signals = stronger pattern)
                    alignment_boost = min(0.2, 0.05 * len(bullish_signals))
                    final_confidence = min(0.95, avg_confidence + alignment_boost)

                    signal_types = list(set(s.get("indicator_type", s.get("signal_source", "unknown"))
                                           for s in bullish_signals))

                    patterns.append({
                        "type": "multi_signal_alignment",
                        "symbol": symbol,
                        "direction": "bullish",
                        "confidence": final_confidence,
                        "signals": signal_types,
                        "description": f"{symbol}: {len(bullish_signals)} bullish signals align ({', '.join(signal_types[:3])})",
                        "signal_count": len(bullish_signals)
                    })

                if len(bearish_signals) >= 2:
                    # Multiple bearish signals align
                    avg_confidence = sum(s.get("confidence", 0.5) for s in bearish_signals) / len(bearish_signals)
                    alignment_boost = min(0.2, 0.05 * len(bearish_signals))
                    final_confidence = min(0.95, avg_confidence + alignment_boost)

                    signal_types = list(set(s.get("indicator_type", s.get("signal_source", "unknown"))
                                           for s in bearish_signals))

                    patterns.append({
                        "type": "multi_signal_alignment",
                        "symbol": symbol,
                        "direction": "bearish",
                        "confidence": final_confidence,
                        "signals": signal_types,
                        "description": f"{symbol}: {len(bearish_signals)} bearish signals align ({', '.join(signal_types[:3])})",
                        "signal_count": len(bearish_signals)
                    })

            self._log(f"  Found {len(patterns)} multi-signal patterns from Qdrant")

        except ImportError:
            self._log("  Qdrant client not installed, skipping signal aggregation", "WARN")
        except Exception as e:
            self._log(f"  Signal aggregation failed: {e}", "WARN")

        # =====================================================================
        # 3. Politician/contract correlations (legacy pattern type)
        # =====================================================================
        try:
            correlations = self.politician_tracker.find_correlations(
                symbols=symbols,
                days_lookback=90,
                min_trade_value=25000
            )

            for corr in correlations:
                patterns.append({
                    "type": "politician_contract",
                    "symbol": corr.symbol,
                    "direction": "bullish" if corr.trade_type == "buy" else "bearish",
                    "confidence": corr.confidence,
                    "signals": ["politician_trade", "government_contract"],
                    "description": corr.to_plain_language(),
                    "oversight_match": corr.oversight_match
                })

            self._log(f"  Found {len(correlations)} politician/contract correlations")

        except Exception as e:
            self._log(f"  Correlation detection failed: {e}", "WARN")

        self._log(f"  TOTAL PATTERNS: {len(patterns)}")
        return patterns

    # =========================================================================
    # PHASE 3: PREDICT
    # =========================================================================

    def make_predictions(self, patterns: List[Dict]) -> List[str]:
        """
        Convert patterns into tracked predictions.

        Returns list of prediction IDs.
        """
        self._log("PREDICT: Recording predictions...")

        prediction_ids = []

        for pattern in patterns:
            if pattern["confidence"] < 0.5:
                continue  # Skip low-confidence patterns

            # Get current price for entry
            entry_price = 0.0
            if self.price_fetcher:
                try:
                    price_data = self.price_fetcher.get_current_price(pattern["symbol"])
                    if price_data:
                        entry_price = price_data.price
                except:
                    pass

            if entry_price == 0:
                continue  # Can't predict without entry price

            # Create prediction
            pred = create_prediction(
                symbol=pattern["symbol"],
                direction=pattern["direction"],
                confidence=pattern["confidence"],
                entry_price=entry_price,
                reasoning=pattern["description"],
                contributing_signals=pattern["signals"],
                timeframe="1d"
            )

            pred_id = self.outcome_tracker.record_prediction(pred)
            prediction_ids.append(pred_id)

            self._log(f"  Predicted {pattern['symbol']} {pattern['direction']} "
                     f"@ ${entry_price:.2f} (conf: {pattern['confidence']:.0%})")

        self._log(f"  Made {len(prediction_ids)} predictions")
        return prediction_ids

    # =========================================================================
    # PHASE 4: OBSERVE
    # =========================================================================

    def observe_outcomes(self) -> List[Dict]:
        """
        Check predictions that are due and record outcomes.

        Returns list of recorded outcomes.
        """
        self._log("OBSERVE: Checking due predictions...")

        outcomes = []

        if self.price_fetcher:
            try:
                recorded = self.outcome_tracker.check_and_record_outcomes(
                    price_fetcher=price_fetcher_for_outcomes
                )
                outcomes = [asdict(o) for o in recorded]
                self._log(f"  Recorded {len(outcomes)} outcomes")

                for o in recorded:
                    status = "CORRECT" if o.was_correct else "WRONG"
                    self._log(f"    {o.symbol}: {status} ({o.return_pct:+.1f}%)")

            except Exception as e:
                self._log(f"  Outcome observation failed: {e}", "WARN")
        else:
            due = self.outcome_tracker.get_due_predictions()
            if due:
                self._log(f"  {len(due)} predictions due but no price fetcher available")

        return outcomes

    # =========================================================================
    # PHASE 5: LEARN
    # =========================================================================

    def learn_from_outcomes(self) -> Dict:
        """
        Run learning cycle to update signal weights.

        Returns learning cycle results.
        """
        self._log("LEARN: Updating signal weights...")

        try:
            result = self.learning_loop.run_learning_cycle(min_predictions=3)

            self._log(f"  Analyzed {result.predictions_analyzed} predictions")
            self._log(f"  Summary: {result.summary}")

            # Store lessons in memory
            if result.reliability_updates:
                for signal, new_score in result.reliability_updates.items():
                    self.memory.setdefault("signal_history", {})[signal] = {
                        "score": new_score,
                        "updated_at": datetime.now().isoformat()
                    }

            return {
                "predictions_analyzed": result.predictions_analyzed,
                "reliability_updates": result.reliability_updates,
                "summary": result.summary
            }

        except Exception as e:
            self._log(f"  Learning failed: {e}", "WARN")
            return {"error": str(e)}

    # =========================================================================
    # PHASE 6: RESEARCH (Emergence pattern)
    # =========================================================================

    def research_improvements(self, use_llm: bool = True, prefer_llm: str = "deepseek") -> Dict:
        """
        Analyze what's working and what's not.
        Generate research questions and optionally query LLM for answers.

        This is the Emergence "introspect + research" pattern.

        Args:
            use_llm: Whether to query LLM for answers (default True)
            prefer_llm: Which LLM to prefer ("deepseek" or "gemini")
        """
        self._log("RESEARCH: Analyzing performance...")

        # Get current stats
        stats = self.outcome_tracker.get_overall_stats()
        signal_perf = self.outcome_tracker.get_signal_performance()
        by_direction = self.outcome_tracker.get_performance_by_direction()

        research = {
            "overall_accuracy": stats.get("accuracy", 0),
            "total_predictions": stats.get("total_predictions", 0),
            "best_signals": [],
            "worst_signals": [],
            "questions": [],
            "answers": [],
            "recommendations": []
        }

        # Identify best and worst performing signals
        for signal_id, perf in signal_perf.items():
            if perf["predictions"] >= 3:
                if perf["accuracy"] >= 0.7:
                    research["best_signals"].append({
                        "signal": signal_id,
                        "accuracy": perf["accuracy"]
                    })
                elif perf["accuracy"] < 0.4:
                    research["worst_signals"].append({
                        "signal": signal_id,
                        "accuracy": perf["accuracy"]
                    })

        # Generate research questions
        if research["best_signals"]:
            best = research["best_signals"][0]["signal"]
            research["questions"].append(
                f"Why is {best} performing well? What market conditions favor it?"
            )

        if research["worst_signals"]:
            worst = research["worst_signals"][0]["signal"]
            research["questions"].append(
                f"Why is {worst} underperforming? Should we reduce its weight or remove it?"
            )

        if stats.get("accuracy", 0) < 0.5:
            research["questions"].append(
                "Overall accuracy below 50%. Are we missing key signals or is timing off?"
            )

        # Check direction bias
        if by_direction:
            bullish = by_direction.get("bullish", {}).get("accuracy", 0)
            bearish = by_direction.get("bearish", {}).get("accuracy", 0)
            if abs(bullish - bearish) > 0.2:
                bias = "bullish" if bullish > bearish else "bearish"
                research["questions"].append(
                    f"Strong {bias} bias detected. Are we missing signals for the other direction?"
                )

        self._log(f"  Generated {len(research['questions'])} research questions")

        # Query LLM for answers if enabled
        if use_llm and research["questions"]:
            llm_name = "DeepSeek" if prefer_llm == "deepseek" else "Gemini"
            self._log(f"  Querying {llm_name} for insights...")

            # Build context prompt
            context = f"""You are a trading strategy analyst for MIDGE, a self-improving pattern recognition system.

Current performance:
- Overall accuracy: {stats.get('accuracy', 0):.1%}
- Total predictions: {stats.get('total_predictions', 0)}
- Best signals: {', '.join(s['signal'] for s in research['best_signals']) or 'None yet'}
- Worst signals: {', '.join(s['signal'] for s in research['worst_signals']) or 'None yet'}

Questions to analyze:
{chr(10).join(f'- {q}' for q in research['questions'])}

Provide concise, actionable recommendations (2-3 sentences each). Focus on:
1. What patterns might explain the performance
2. Specific weight adjustments to consider
3. New signals that might help"""

            answer = self._query_llm(context, prefer=prefer_llm)
            if answer:
                research["answers"].append(answer)
                self._log(f"  LLM provided insights ({len(answer)} chars)")

                # Parse recommendations from LLM's answer
                if "reduce" in answer.lower() or "decrease" in answer.lower():
                    research["recommendations"].append(
                        "LLM suggests reducing weight of underperforming signals"
                    )
                if "add" in answer.lower() or "include" in answer.lower():
                    research["recommendations"].append(
                        "LLM suggests adding new signal types"
                    )
            else:
                self._log("  LLM unavailable, using heuristics only")

        # Generate heuristic recommendations
        if research["worst_signals"]:
            research["recommendations"].append(
                f"Consider reducing weight of {research['worst_signals'][0]['signal']}"
            )

        if stats.get("total_predictions", 0) < 10:
            research["recommendations"].append(
                "Need more predictions for reliable learning. Expand symbol coverage."
            )

        self._log(f"  Generated {len(research['recommendations'])} recommendations")

        # Store in memory
        self.memory["last_research"] = {
            "timestamp": datetime.now().isoformat(),
            "questions": research["questions"],
            "answers": research["answers"],
            "recommendations": research["recommendations"]
        }

        return research

    # =========================================================================
    # PHASE 7: EVOLVE
    # =========================================================================

    def evolve_strategy(self, research: Dict) -> List[str]:
        """
        Apply evolutionary changes based on research.

        This is the Emergence "implement" pattern.
        Currently automatic adjustments only - more complex evolution
        would require LLM integration.
        """
        self._log("EVOLVE: Applying improvements...")

        changes = []

        # Auto-adjust based on recommendations
        for rec in research.get("recommendations", []):
            if "reducing weight" in rec.lower():
                # Find the signal and reduce its weight
                for worst in research.get("worst_signals", []):
                    signal = worst["signal"]
                    current = self.learning_loop.get_signal_reliability(signal)
                    new_score = max(0.1, current * 0.9)  # Reduce by 10%
                    self.learning_loop.reliability_scores[signal] = new_score
                    changes.append(f"Reduced {signal} weight: {current:.3f} → {new_score:.3f}")
                    self._log(f"  {changes[-1]}")

        # Store successful patterns
        for best in research.get("best_signals", []):
            if best not in self.memory.get("successful_patterns", []):
                self.memory.setdefault("successful_patterns", []).append(best)

        # Store lessons
        for question in research.get("questions", []):
            lesson = {
                "timestamp": datetime.now().isoformat(),
                "type": "research_question",
                "content": question
            }
            self.memory.setdefault("lessons", []).append(lesson)

        # Limit lessons to last 100
        if len(self.memory.get("lessons", [])) > 100:
            self.memory["lessons"] = self.memory["lessons"][-100:]

        self._save_memory()

        return changes

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run_cycle(self, symbols: List[str] = None) -> EvolutionCycle:
        """
        Run one complete evolution cycle.

        Returns EvolutionCycle record.
        """
        self.cycle_count += 1
        cycle_id = f"cycle_{self.cycle_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self._log(f"{'='*60}")
        self._log(f"EVOLUTION CYCLE #{self.cycle_count}")
        self._log(f"{'='*60}")

        findings = {}
        changes = []
        lessons = []
        error = None
        phase = "unknown"

        try:
            # Phase 1: Gather
            phase = "gather"
            data = self.gather_data(symbols)
            findings["gathered"] = {"symbols": list(data.get("prices", {}).keys())}

            # Phase 2: Detect
            phase = "detect"
            patterns = self.detect_patterns(symbols)
            findings["patterns"] = len(patterns)

            # Phase 3: Predict
            phase = "predict"
            predictions = self.make_predictions(patterns)
            findings["predictions"] = len(predictions)

            # Phase 4: Observe
            phase = "observe"
            outcomes = self.observe_outcomes()
            findings["outcomes"] = len(outcomes)

            # Phase 5: Learn
            phase = "learn"
            learning = self.learn_from_outcomes()
            findings["learning"] = learning

            # Phase 6: Research
            phase = "research"
            use_llm = getattr(self, 'use_llm', True)
            prefer_llm = getattr(self, 'prefer_llm', 'deepseek')
            research = self.research_improvements(use_llm=use_llm, prefer_llm=prefer_llm)
            findings["research"] = {
                "questions": len(research.get("questions", [])),
                "recommendations": len(research.get("recommendations", []))
            }
            lessons = research.get("questions", [])

            # Phase 7: Evolve
            phase = "evolve"
            changes = self.evolve_strategy(research)
            findings["changes"] = len(changes)

            self._log(f"\nCycle #{self.cycle_count} complete!")
            success = True

        except Exception as e:
            self._log(f"\nCycle failed in {phase}: {e}", "ERROR")
            error = str(e)
            success = False

        # Create cycle record
        cycle = EvolutionCycle(
            cycle_id=cycle_id,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            success=success,
            findings=findings,
            changes_made=changes,
            lessons=lessons,
            error=error
        )

        # Save memory
        self._save_memory()

        return cycle

    def run_continuously(self, cycle_delay: int = 3600, max_cycles: int = None):
        """
        Run evolution loop continuously.

        Args:
            cycle_delay: Seconds between cycles (default 1 hour)
            max_cycles: Maximum cycles to run (None = infinite)
        """
        self._log(f"Starting continuous evolution (delay: {cycle_delay}s)")

        cycles_run = 0

        try:
            while max_cycles is None or cycles_run < max_cycles:
                cycle = self.run_cycle()
                cycles_run += 1

                if not cycle.success:
                    self._log(f"Cycle failed, waiting longer before retry...")
                    time.sleep(cycle_delay * 2)
                else:
                    self._log(f"Waiting {cycle_delay}s until next cycle...")
                    time.sleep(cycle_delay)

        except KeyboardInterrupt:
            self._log("Evolution stopped by user")


def main():
    """Entry point for MIDGE evolution."""
    import argparse

    parser = argparse.ArgumentParser(description="MIDGE Self-Improving Trading Intelligence")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    parser.add_argument("--delay", type=int, default=3600, help="Seconds between cycles")
    parser.add_argument("--symbols", nargs="+", help="Symbols to track")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--llm", choices=["deepseek", "gemini", "none"], default="deepseek",
                       help="LLM for research phase (default: deepseek)")

    args = parser.parse_args()

    evolution = MIDGEEvolution()
    evolution.prefer_llm = args.llm if args.llm != "none" else None
    evolution.use_llm = args.llm != "none"

    if args.continuous:
        evolution.run_continuously(cycle_delay=args.delay)
    else:
        for i in range(args.cycles):
            evolution.run_cycle(symbols=args.symbols)


if __name__ == "__main__":
    main()
