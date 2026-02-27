"""Backtest Analyzer — Converts sweep backtest results into formal hypotheses.

Bridge 1: MIDGE reads her own backtest performance and turns it into
trackable, validatable hypotheses. Slices trade data into aggregates
(per symbol, direction, session, and combinations), filters by sample
size, pre-populates HypothesisStats, and registers qualifying patterns
in the hypothesis registry with ICT-theory causal stories.

The backtest IS the validation evidence. Hypotheses enter PROBATION with
pre-populated stats. The validator recognizes BACKTEST_DERIVED and uses
these stats directly (computing DSR from the global trial counter).
Strong aggregates get promoted. Weak ones get retired. The lifecycle
is respected — no shortcut.

Generation cadence: every 500 steps (driven by HypothesisEngine),
but dedup ensures only new findings produce hypotheses.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# ── Thresholds ────────────────────────────────────────────────────────

MIN_AGGREGATE_N = 20       # Minimum trades per aggregate to generate hypothesis
BACKTEST_RESULTS_PATH = DATA_DIR / "sweep_backtest_results.json"

# ── ICT Causal Story Templates ────────────────────────────────────────
# Real market microstructure reasoning per dimension. No generic fallbacks.

_STORY_BY_SYMBOL = {
    "CL=F": (
        "Crude oil futures session sweeps are amplified by the commodity's "
        "sensitivity to geopolitical news flow concentrated outside US hours. "
        "Asia session lows are frequently established by algorithmic stop runs "
        "targeting prior-session range extremes before institutional order flow "
        "reverses during the London-US handoff. Crude's high ATR-to-spread "
        "ratio makes FVG entries geometrically clean with defined risk."
    ),
    "GC=F": (
        "Gold futures session sweeps exploit the London-to-US handoff window "
        "where bullion bank positioning is restructured. Asia range extremes "
        "serve as premium liquidity pools swept before gold's RTH directional "
        "move. Gold's mean-reverting intraday profile and low noise ratio "
        "relative to ATR makes inverted FVG entries high-probability."
    ),
    "YM=F": (
        "Dow Jones futures are index-cap-weighted toward 30 mega-caps, making "
        "them sensitive to overnight futures premium compression. Session "
        "sweeps on YM target prior-session extremes where retail stops cluster, "
        "then reverse as institutional flow enters during kill zone transitions. "
        "YM's lower tick noise versus NQ produces cleaner displacement."
    ),
    "ES=F": (
        "S&P 500 E-mini futures are the most liquid equity index contract. "
        "Session sweep patterns work but with lower edge due to tighter "
        "spreads and higher efficiency. Institutional algorithms arbitrage "
        "ES continuously, reducing the persistence of liquidity pool sweeps "
        "compared to less liquid contracts."
    ),
    "NQ=F": (
        "Nasdaq 100 futures carry high beta to tech sector overnight news. "
        "Session sweeps target prior-session extremes but NQ's wide intraday "
        "range and gap-prone behavior can invalidate FVG zones before entry. "
        "Higher noise-to-signal ratio reduces sweep pattern reliability "
        "versus lower-volatility contracts."
    ),
    "RTY=F": (
        "Russell 2000 futures have the widest spreads and lowest institutional "
        "participation among major equity index futures. Session sweep patterns "
        "underperform because liquidity pool formation is less consistent — "
        "retail stop clustering is sparser and institutional sweep algorithms "
        "are less active in small-cap futures."
    ),
}

_STORY_BY_DIRECTION = {
    "bearish": (
        "Bearish session sweeps exploit the structural tendency of algorithmic "
        "systems to hunt buy-side stops above prior session highs before "
        "distributing into institutional sell orders. ICT's Power of 3 concept "
        "(accumulation, manipulation, distribution) places the manipulation "
        "sweep above session highs before the actual downward leg."
    ),
    "bullish": (
        "Bullish session sweeps target sell-side stops below prior session "
        "lows. Market makers acquire long positions at stop-clustered lows "
        "before marking price up. The sweep-to-FVG reversal structure is "
        "less reliable on the long side because downward sweeps tend to "
        "have wider displacement, stretching FVG entry zones."
    ),
}

_STORY_BY_SESSION = {
    "asia": (
        "Asia session (20:00-22:00 ET) sweeps establish the day's initial "
        "liquidity reference points. Lower volume during Asia creates "
        "narrower ranges with tighter stop clustering. Sweeps of these "
        "tight ranges by London-open algorithms produce high-quality "
        "displacement and inverted FVG entries."
    ),
    "london": (
        "London session (02:00-05:00 ET) sweeps target Asia session extremes "
        "with the highest directional conviction. London open is the most "
        "active non-US session with institutional bank flow setting the "
        "day's directional bias. FVG entries during London have strong "
        "follow-through into US RTH."
    ),
}


def _build_story(agg: dict) -> str:
    """Compose a causal story from dimension-specific templates."""
    parts = []

    symbol = agg.get("symbol")
    direction = agg.get("direction")
    session = agg.get("session")

    if symbol and symbol in _STORY_BY_SYMBOL:
        parts.append(_STORY_BY_SYMBOL[symbol])
    if direction and direction in _STORY_BY_DIRECTION:
        parts.append(_STORY_BY_DIRECTION[direction])
    if session and session in _STORY_BY_SESSION:
        parts.append(_STORY_BY_SESSION[session])

    if not parts:
        # Fallback for aggregates with unknown dimensions — should not happen
        # with the current 6 symbols / 2 directions / 2 sessions
        parts.append(
            "ICT session sweep + IFVG pattern on futures. Liquidity pool "
            "formation at session extremes creates predictable reversal "
            "zones when swept by institutional algorithms."
        )

    wr = agg.get("win_rate", 0)
    n = agg.get("total", 0)
    avg_r = agg.get("avg_r", 0)
    parts.append(
        f"Backtest evidence: {wr:.1%} win rate over {n} trades, "
        f"average {avg_r:+.3f}R per trade."
    )

    return " ".join(parts)


class BacktestAnalyzer:
    """Converts sweep backtest results into formal hypotheses.

    Reads data/market/sweep_backtest_results.json, slices trades into
    statistical aggregates, filters by sample size, and registers
    qualifying patterns in the hypothesis registry.

    One job: backtest results -> PROBATION hypotheses with pre-populated
    stats. The validator handles promotion/retirement on next cycle.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        backtest_path: Path = None,
    ):
        self._registry = registry
        self._backtest_path = backtest_path or BACKTEST_RESULTS_PATH
        self._last_run_time: Optional[str] = None
        self._aggregates_built = 0
        self._hypotheses_created = 0
        self._hypotheses_refreshed = 0

    # ── Public API ────────────────────────────────────────────────────

    def analyze(self) -> List[Hypothesis]:
        """Read backtest results and generate qualifying hypotheses.

        Returns list of newly registered hypotheses.
        """
        data = self._load_backtest()
        if not data:
            return []

        trades = data.get("trades", [])
        if not trades:
            return []

        self._last_run_time = data.get("run_time", "")
        aggregates = self._build_aggregates(trades)
        self._aggregates_built = len(aggregates)

        new_hypotheses: List[Hypothesis] = []
        for agg in aggregates:
            hyp = self._aggregate_to_hypothesis(agg)
            if hyp is not None:
                self._registry.register(hyp)
                new_hypotheses.append(hyp)

        self._hypotheses_created += len(new_hypotheses)

        if new_hypotheses:
            logger.info(
                "BacktestAnalyzer: %d hypotheses from %d aggregates "
                "(backtest: %s)",
                len(new_hypotheses),
                len(aggregates),
                self._last_run_time,
            )
        return new_hypotheses

    def get_statistics(self) -> dict:
        """Summary stats for monitoring."""
        return {
            "last_backtest_run_time": self._last_run_time or "",
            "aggregates_built": self._aggregates_built,
            "hypotheses_created": self._hypotheses_created,
            "hypotheses_refreshed": self._hypotheses_refreshed,
            "backtest_file_exists": self._backtest_path.exists(),
        }

    # ── Loading ───────────────────────────────────────────────────────

    def _load_backtest(self) -> dict:
        """Load sweep_backtest_results.json. Returns {} on failure."""
        if not self._backtest_path.exists():
            logger.debug("No backtest results at %s", self._backtest_path)
            return {}

        try:
            data = json.loads(self._backtest_path.read_text())
            if isinstance(data, dict) and "trades" in data:
                return data
            return {}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to load backtest results: %s", e)
            return {}

    # ── Aggregation ───────────────────────────────────────────────────

    def _build_aggregates(self, trades: list) -> List[dict]:
        """Slice trades into statistical aggregates.

        Produces:
          - Per-symbol (6 aggregates)
          - Per-direction (2 aggregates)
          - Per-session (2 aggregates)
          - Per-symbol+direction (12 combos)
        Total: up to 22 potential aggregates, filtered by MIN_AGGREGATE_N.
        """
        aggregates = []

        # Group by single dimensions
        by_symbol: dict[str, list] = {}
        by_direction: dict[str, list] = {}
        by_session: dict[str, list] = {}
        by_sym_dir: dict[tuple, list] = {}

        for t in trades:
            sym = t.get("symbol", "")
            dirn = t.get("direction", "")
            sess = t.get("session_swept", "")

            if sym:
                by_symbol.setdefault(sym, []).append(t)
            if dirn:
                by_direction.setdefault(dirn, []).append(t)
            if sess:
                by_session.setdefault(sess, []).append(t)
            if sym and dirn:
                by_sym_dir.setdefault((sym, dirn), []).append(t)

        # Build aggregate dicts
        for sym, group in by_symbol.items():
            agg = self._compute_aggregate(group, "symbol", symbol=sym)
            if agg["total"] >= MIN_AGGREGATE_N:
                aggregates.append(agg)

        for dirn, group in by_direction.items():
            agg = self._compute_aggregate(group, "direction", direction=dirn)
            if agg["total"] >= MIN_AGGREGATE_N:
                aggregates.append(agg)

        for sess, group in by_session.items():
            agg = self._compute_aggregate(group, "session", session=sess)
            if agg["total"] >= MIN_AGGREGATE_N:
                aggregates.append(agg)

        for (sym, dirn), group in by_sym_dir.items():
            agg = self._compute_aggregate(
                group, "symbol+direction", symbol=sym, direction=dirn,
            )
            if agg["total"] >= MIN_AGGREGATE_N:
                aggregates.append(agg)

        return aggregates

    def _compute_aggregate(
        self,
        trades: list,
        dimension: str,
        *,
        symbol: str = "",
        direction: str = "",
        session: str = "",
    ) -> dict:
        """Compute stats for a group of trades."""
        wins = sum(1 for t in trades if t.get("result") == "win_2r")
        timeouts = sum(1 for t in trades if t.get("result") == "timeout")
        total = len(trades)
        losses = total - wins  # includes timeouts

        r_values = [t.get("r_captured", 0.0) for t in trades]
        avg_r = sum(r_values) / total if total else 0.0
        win_rate = wins / total if total else 0.0

        return {
            "dimension": dimension,
            "symbol": symbol or None,
            "direction": direction or None,
            "session": session or None,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "total": total,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "r_values": r_values,
        }

    # ── Hypothesis Construction ───────────────────────────────────────

    def _aggregate_to_hypothesis(self, agg: dict) -> Optional[Hypothesis]:
        """Convert a qualifying aggregate to a Hypothesis, or None."""
        trigger = self._build_trigger(agg)

        # Dedup: check if we already have this backtest hypothesis
        if self._is_duplicate(trigger):
            return None

        # Build stats from backtest data
        sharpe = self._compute_sharpe(agg["r_values"])

        stats = HypothesisStats(
            wins=agg["wins"],
            losses=agg["losses"],
            total_observations=agg["total"],
            cumulative_return_pct=sum(agg["r_values"]),
            sharpe_ratio=sharpe,
            last_evaluated=datetime.now(),
        )

        name = self._build_name(agg)
        story = _build_story(agg)

        return Hypothesis(
            name=name,
            trigger=trigger,
            stats=stats,
            causal_story=story,
            source_type=SourceType.BACKTEST_DERIVED,
            parent_lag_finding=self._agg_to_finding(agg),
        )

    def _build_trigger(self, agg: dict) -> TriggerPattern:
        """Build TriggerPattern encoding the backtest aggregate.

        source_a = "session_sweep" (the detection mechanism)
        source_b = "price_outcome" (what it predicts)
        lag_days = 0 (same-session, not cross-source lag)
        domain_filter = encoded dimension string
        """
        return TriggerPattern(
            source_a="session_sweep",
            source_b="price_outcome",
            lag_days=0,
            direction=agg.get("direction") or "",
            domain_filter=self._encode_domain_filter(agg),
        )

    def _encode_domain_filter(self, agg: dict) -> str:
        """Encode aggregate conditions as a colon-separated string.

        Examples: "CL=F", "bearish", "asia", "CL=F:bearish"
        """
        parts = []
        if agg.get("symbol"):
            parts.append(agg["symbol"])
        if agg.get("direction"):
            parts.append(agg["direction"])
        if agg.get("session"):
            parts.append(agg["session"])
        return ":".join(parts) if parts else "all"

    def _build_name(self, agg: dict) -> str:
        """Human-readable hypothesis name."""
        conditions = self._encode_domain_filter(agg)
        wr = agg.get("win_rate", 0)
        n = agg.get("total", 0)
        return f"Sweep:{conditions} ({wr:.1%} win, n={n})"

    def _agg_to_finding(self, agg: dict) -> dict:
        """Convert aggregate to a serializable finding dict."""
        return {
            "dimension": agg["dimension"],
            "symbol": agg.get("symbol"),
            "direction": agg.get("direction"),
            "session": agg.get("session"),
            "wins": agg["wins"],
            "losses": agg["losses"],
            "timeouts": agg["timeouts"],
            "total": agg["total"],
            "win_rate": agg["win_rate"],
            "avg_r": agg["avg_r"],
        }

    # ── Sharpe Computation ────────────────────────────────────────────

    def _compute_sharpe(self, r_values: list) -> float:
        """Compute Sharpe ratio from actual R-values.

        Uses raw R-captured values (not binary win/loss). Annualized
        assuming ~252 trading days, ~3 trades/day based on backtest cadence.
        """
        if len(r_values) < 2:
            return 0.0

        mean_r = sum(r_values) / len(r_values)
        variance = sum((r - mean_r) ** 2 for r in r_values) / (len(r_values) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0

        if std_r == 0.0:
            return 0.0

        # Daily Sharpe (per-trade) * sqrt(annualization factor)
        # ~756 trades/year estimate (3/day * 252 days)
        trades_per_year = 756
        return (mean_r / std_r) * math.sqrt(trades_per_year)

    # ── Dedup ─────────────────────────────────────────────────────────

    def _is_duplicate(self, trigger: TriggerPattern) -> bool:
        """Check if a non-retired BACKTEST_DERIVED hypothesis with this domain_filter exists."""
        for hyp in self._registry.get_all():
            if hyp.status == HypothesisStatus.RETIRED:
                continue
            if (
                hyp.source_type == SourceType.BACKTEST_DERIVED
                and hyp.trigger.domain_filter == trigger.domain_filter
            ):
                return True
        return False

    def refresh_probation(self) -> int:
        """Retire all PROBATION BACKTEST_DERIVED hypotheses for fresh re-analysis.

        Called before analyze() when backtest results have been refreshed.
        PROBATION means 'not yet validated by real market outcomes' — retiring
        and recreating with updated stats loses nothing of value. ACTIVE/RETIRED
        hypotheses are never touched.

        Returns count of hypotheses retired.
        """
        count = 0
        for hyp in self._registry.get_probation():
            if hyp.source_type == SourceType.BACKTEST_DERIVED:
                self._registry.retire(
                    hyp.hypothesis_id,
                    reason="stale_backtest_refresh",
                )
                count += 1
        self._hypotheses_refreshed += count
        return count
