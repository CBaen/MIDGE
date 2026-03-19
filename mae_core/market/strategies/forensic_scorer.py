"""forensic_scorer.py — Real-time strategy combination scoring.

Analyzes recent 5-minute data to find which strategy combinations are
WINNING RIGHT NOW. Updates every 10 minutes. The trader uses this
scorecard to weight convergence decisions.

This is MIDGE's learning loop — she learns from hours ago, not weeks.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("midge.crypto_trader")

SCORECARD_PATH = Path("data/market/forensic_scorecard.json")


class ForensicScorer:
    """Analyzes recent price data to score strategy combinations."""

    OUTCOME_BARS = 36       # 36 × 5min = 3 hours to judge outcome
    MIN_WINDOW = 50         # Min bars before evaluating strategies
    COMBO_MIN_SAMPLES = 3   # Min co-occurrences to trust a combo
    REFRESH_SECONDS = 600   # Re-analyze every 10 minutes

    def __init__(self) -> None:
        self._scorecard: Dict[str, Dict[str, Any]] = {}  # combo_key → {wins, losses, wr}
        self._last_refresh: float = 0
        self._load()

    def needs_refresh(self) -> bool:
        return (time.time() - self._last_refresh) > self.REFRESH_SECONDS

    def refresh(self, symbols: List[str], all_strategies: list) -> None:
        """Re-analyze recent 5-minute data across all symbols."""
        import yfinance as yf

        combo_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})
        total_firings = 0

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="7d", interval="5m")
                if df is None or len(df) < 100:
                    continue
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

                # Walk through bars, find strategy firings and outcomes
                by_window: Dict[int, list] = defaultdict(list)

                for i in range(self.MIN_WINDOW, len(df) - self.OUTCOME_BARS, 3):
                    for name, fn in all_strategies:
                        try:
                            r = fn(symbol, df.iloc[:i])
                            if r is None:
                                continue
                            entry = float(df["Close"].iloc[i])
                            exit_price = float(df["Close"].iloc[i + self.OUTCOME_BARS])
                            pct = (exit_price - entry) / entry * 100
                            won = (pct > 0.25 and r.direction == "bullish") or \
                                  (pct < -0.25 and r.direction == "bearish")
                            by_window[i // 3].append({
                                "strategy": name,
                                "direction": r.direction,
                                "won": won,
                            })
                            total_firings += 1
                        except Exception:
                            pass

                # Find co-occurring pairs and their outcomes
                for window_key, group in by_window.items():
                    for direction in ("bullish", "bearish"):
                        strats = sorted(set(
                            f["strategy"] for f in group if f["direction"] == direction
                        ))
                        if len(strats) < 2:
                            continue
                        relevant = [f for f in group if f["direction"] == direction]
                        avg_won = sum(1 for f in relevant if f["won"]) / max(len(relevant), 1)

                        for s1_idx in range(len(strats)):
                            for s2_idx in range(s1_idx + 1, min(len(strats), s1_idx + 4)):
                                key = f"{strats[s1_idx]}+{strats[s2_idx]}"
                                if avg_won > 0.5:
                                    combo_stats[key]["wins"] += 1
                                else:
                                    combo_stats[key]["losses"] += 1

            except Exception as e:
                logger.debug("Forensic scan failed for %s: %s", symbol, e)

        # Build scorecard
        self._scorecard = {}
        for combo_key, stats in combo_stats.items():
            total = stats["wins"] + stats["losses"]
            if total >= self.COMBO_MIN_SAMPLES:
                wr = stats["wins"] / total
                self._scorecard[combo_key] = {
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "total": total,
                    "win_rate": round(wr, 3),
                    "hot": wr >= 0.60,  # "hot" combos get priority
                }

        self._last_refresh = time.time()
        self._save()
        logger.info(
            "Forensic refresh: %d firings, %d combos scored, %d hot (≥60%% WR)",
            total_firings,
            len(self._scorecard),
            sum(1 for v in self._scorecard.values() if v["hot"]),
        )

    def score_convergence(self, strategy_names: List[str]) -> Tuple[float, bool]:
        """Score a set of converging strategies against the forensic scorecard.

        Returns (combo_win_rate, is_hot).
        combo_win_rate: average WR of all pairs found in the scorecard, or 0.5 if unknown.
        is_hot: True if any pair in the combo is marked hot (≥60% WR recently).
        """
        if not self._scorecard or len(strategy_names) < 2:
            return 0.5, False

        pair_wrs = []
        any_hot = False
        names = sorted(strategy_names)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                key = f"{names[i]}+{names[j]}"
                if key in self._scorecard:
                    entry = self._scorecard[key]
                    pair_wrs.append(entry["win_rate"])
                    if entry["hot"]:
                        any_hot = True

        if not pair_wrs:
            return 0.5, False

        avg_wr = sum(pair_wrs) / len(pair_wrs)
        return round(avg_wr, 3), any_hot

    def get_hot_combos(self) -> List[Dict[str, Any]]:
        """Return all combos with ≥60% recent win rate."""
        return [
            {"combo": k, **v}
            for k, v in sorted(
                self._scorecard.items(),
                key=lambda x: -x[1]["win_rate"],
            )
            if v["hot"]
        ]

    def _save(self) -> None:
        try:
            SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
            SCORECARD_PATH.write_text(json.dumps({
                "updated_at": datetime.now().isoformat(),
                "combos": self._scorecard,
            }, indent=2))
        except Exception:
            pass

    def _load(self) -> None:
        try:
            if SCORECARD_PATH.exists():
                data = json.loads(SCORECARD_PATH.read_text())
                self._scorecard = data.get("combos", {})
                logger.info("Forensic scorecard loaded: %d combos", len(self._scorecard))
        except Exception:
            pass
