"""Deep Analyst — MIDGE's thinking layer.

Reads all historical data and synthesizes a ranked list of "most likely
near-term moves" with full evidence trails.  Runs on demand (including
at startup).  Six scoring components: Thompson reliability, pattern
template match, World Model causal position, lag leading indicators,
recency-weighted signal density, and historical outcome win rate.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from mae_core.market.archaeology.fingerprint import _SOURCE_DOMAIN_MAP
from mae_core.market.archaeology.pattern_library import PatternLibrary
from mae_core.market.intelligence.signal_archive_reader import SignalArchiveReader
from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
from mae_core.market.intelligence.world_model import WorldModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SIGNALS_DIR = str(_PROJECT_ROOT / "data" / "midge" / "signals")
_DEFAULT_DATA_DIR = str(_PROJECT_ROOT / "data" / "market")

# Weights for the six scoring components
_WEIGHTS = {
    "thompson":  0.25,
    "template":  0.20,
    "world_model": 0.15,
    "lag_lead":  0.15,
    "density":   0.15,
    "outcome":   0.10,
}

# Minimum independent domains to consider a ticker
_MIN_DOMAINS = 3

# Exponential decay half-life in days (recent signals matter more)
_DECAY_HALF_LIFE = 7.0


@dataclass
class Inevitability:
    """A ranked candidate move with full evidence trail."""
    ticker: str
    direction: str                          # "bullish" or "bearish"
    score: float                            # 0-1 combined inevitability score
    domains: List[str]                      # which domains are signaling
    signals: List[dict]                     # individual signals with source/strength/timestamp
    thompson_score: float                   # reliability from Thompson distributions
    template_match: Optional[str]           # matched template id if any
    template_win_rate: Optional[float]      # template historical win rate
    world_model_chain: Optional[List[str]]  # causal chain if applicable
    leading_indicators: List[dict]          # lag correlations that support this
    historical_win_rate: Optional[float]    # from outcomes data for this ticker+direction
    evidence_summary: str                   # human-readable evidence paragraph
    expected_window_days: int               # estimated time horizon
    signal_count: int                       # total number of supporting signals
    earliest_signal: str                    # ISO timestamp of first signal
    latest_signal: str                      # ISO timestamp of most recent signal


class DeepAnalyst:
    """Full-picture synthesizer — standalone or bootstrap-wired."""

    def __init__(
        self,
        signals_dir: str = _DEFAULT_SIGNALS_DIR,
        data_dir: str = _DEFAULT_DATA_DIR,
        thompson_sampler: Optional[ThompsonSampler] = None,
        pattern_library: Optional[PatternLibrary] = None,
        world_model: Optional[WorldModel] = None,
    ):
        self._signals_dir = Path(signals_dir)
        self._data_dir = Path(data_dir)

        # Dependencies — accept injected or load standalone
        self._sampler = thompson_sampler
        self._library = pattern_library
        self._world_model = world_model

        self._lag_correlations: List[dict] = []
        self._outcome_index: Dict[str, Dict[str, list]] = {}  # ticker -> dir -> [was_correct]
        self._combo_stats: Dict[str, dict] = {}               # from post_mortem_insights

        self._last_analysis_time: Optional[datetime] = None
        self._last_signal_count: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(self, lookback_days: int = 30, top_n: int = 20) -> List[Inevitability]:
        """Read all data, synthesize, return top N inevitabilities."""
        t_start = datetime.now()
        self._ensure_dependencies()

        # Step 1 — load recent signals
        reader = SignalArchiveReader(archive_dir=self._signals_dir)
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        loaded = reader.load_range(start_date, end_date)
        self._last_signal_count = loaded

        # Step 2 — load ancillary data
        self._load_lag_correlations()
        self._load_outcomes()
        self._load_combo_stats()

        logger.info(
            "DeepAnalyst: loaded %d signals (%d days), %d templates, %d correlations",
            loaded, lookback_days,
            self._library.template_count if self._library else 0,
            len(self._lag_correlations),
        )

        # Step 3 — group signals by (ticker, direction)
        candidates = self._group_signals(reader)

        # Step 4 — score each candidate
        inevitabilities = []
        today = datetime.now()
        for (ticker, direction), recs in candidates.items():
            domains = sorted({_SOURCE_DOMAIN_MAP.get(r.source, "unknown") for r in recs} - {"unknown"})
            if len(domains) < _MIN_DOMAINS:
                continue

            iv = self._score_candidate(ticker, direction, domains, recs, today)
            inevitabilities.append(iv)

        # Step 5 — rank and return top N
        inevitabilities.sort(key=lambda x: x.score, reverse=True)
        result = inevitabilities[:top_n]

        elapsed = (datetime.now() - t_start).total_seconds()
        self._last_analysis_time = datetime.now()
        logger.info(
            "DeepAnalyst: scored %d candidates, returning top %d (%.1fs)",
            len(inevitabilities), len(result), elapsed,
        )
        return result

    def summarize(self, results: Optional[List[Inevitability]] = None, lookback_days: int = 30) -> str:
        """Plain-language report of top findings. Calls analyze() if results is None."""
        if results is None:
            results = self.analyze(lookback_days=lookback_days)

        lines = [
            f"=== MIDGE Deep Analysis ({date.today()}) ===",
            f"Analyzed: {self._last_signal_count:,} signals across {lookback_days} days, "
            f"{self._count_unique_tickers(results)} tickers\n",
        ]

        if not results:
            lines.append("No candidates met the minimum evidence threshold (3+ independent domains).")
            return "\n".join(lines)

        for i, iv in enumerate(results, 1):
            lines.append(f"#{i}. {iv.ticker} — {iv.direction.upper()} (score: {iv.score:.2f}, {len(iv.domains)} domains)")
            lines.append(f"    {iv.evidence_summary}")
            if iv.template_win_rate is not None:
                lines.append(
                    f"    Pattern match: '{iv.template_match}' — "
                    f"{iv.template_win_rate:.0%} historical WR"
                )
            if iv.world_model_chain:
                chain_str = " → ".join(iv.world_model_chain[:5])
                lines.append(f"    Causal chain: {chain_str}")
            for li in iv.leading_indicators[:2]:
                lines.append(
                    f"    Leading indicator: {li['source_a']} leads by ~{li['lag_days']}d "
                    f"(r={li['correlation']:.2f})"
                )
            if iv.historical_win_rate is not None:
                lines.append(f"    Historical WR for {iv.ticker} {iv.direction}: {iv.historical_win_rate:.0%}")
            lines.append(f"    Expected window: {iv.expected_window_days} days | "
                         f"Signals: {iv.signal_count} | "
                         f"First: {iv.earliest_signal[:10]} | Last: {iv.latest_signal[:10]}")
            lines.append("")

        return "\n".join(lines)

    def _ensure_dependencies(self) -> None:
        """Load sampler/library/world_model from disk if not injected."""
        if self._sampler is None:
            try:
                self._sampler = ThompsonSampler()
            except Exception as e:
                logger.warning("DeepAnalyst: could not load ThompsonSampler: %s", e)

        if self._library is None:
            try:
                self._library = PatternLibrary()
            except Exception as e:
                logger.warning("DeepAnalyst: could not load PatternLibrary: %s", e)

        if self._world_model is None:
            try:
                self._world_model = WorldModel()
            except Exception as e:
                logger.warning("DeepAnalyst: could not load WorldModel: %s", e)

    def _load_lag_correlations(self) -> None:
        path = self._data_dir / "lag_correlations.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                self._lag_correlations = json.load(f)
        except Exception as e:
            logger.warning("DeepAnalyst: could not load lag_correlations: %s", e)

    def _load_outcomes(self) -> None:
        path = self._data_dir / "outcomes.jsonl"
        if not path.exists():
            return
        index: Dict[str, Dict[str, list]] = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sym = rec.get("symbol", "")
                        dirn = rec.get("direction", "")
                        correct = rec.get("was_correct", False)
                        if sym and dirn:
                            index.setdefault(sym, {}).setdefault(dirn, []).append(correct)
                    except (json.JSONDecodeError, KeyError):
                        pass
        except Exception as e:
            logger.warning("DeepAnalyst: could not load outcomes: %s", e)
        self._outcome_index = index

    def _load_combo_stats(self) -> None:
        path = self._data_dir / "post_mortem_insights.json"
        if not path.exists():
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._combo_stats = data.get("combo_stats", {})
        except Exception as e:
            logger.warning("DeepAnalyst: could not load post_mortem_insights: %s", e)

    def _group_signals(self, reader: SignalArchiveReader) -> Dict:
        """Group all loaded records by (ticker, direction)."""
        groups: Dict = {}
        for symbol, recs in reader._by_symbol.items():
            for rec in recs:
                if rec.direction in ("neutral", ""):
                    continue
                key = (symbol, rec.direction)
                groups.setdefault(key, []).append(rec)
        return groups

    def _score_candidate(
        self,
        ticker: str,
        direction: str,
        domains: List[str],
        recs: list,
        today: datetime,
    ) -> Inevitability:
        sources = {r.source for r in recs}

        # (a) Thompson reliability
        t_score = self._thompson_score(sources)

        # (b) Template match
        template_match, template_wr, template_score, expected_days = self._template_score(sources, direction)

        # (c) World model causal score
        wm_score, wm_chain = self._world_model_score(ticker)

        # (d) Lag leading indicator score
        lag_score, lead_indicators = self._lag_score(sources)

        # (e) Signal density (recency-weighted)
        density_score = self._density_score(recs, today)

        # (f) Historical outcome score
        hist_wr = self._historical_win_rate(ticker, direction)
        hist_score = hist_wr if hist_wr is not None else 0.5

        # Combo-level bonus from post-mortem
        combo_boost = self._combo_boost(domains)

        combined = (
            _WEIGHTS["thompson"]   * t_score +
            _WEIGHTS["template"]   * template_score +
            _WEIGHTS["world_model"] * wm_score +
            _WEIGHTS["lag_lead"]   * lag_score +
            _WEIGHTS["density"]    * density_score +
            _WEIGHTS["outcome"]    * hist_score
        ) * combo_boost

        combined = min(1.0, max(0.0, combined))

        # Build signal list (capped at 50 for memory)
        signal_dicts = [
            {"source": r.source, "strength": r.strength, "timestamp": r.timestamp.isoformat()}
            for r in sorted(recs, key=lambda x: x.timestamp, reverse=True)[:50]
        ]

        timestamps = [r.timestamp for r in recs]
        earliest = min(timestamps).isoformat() if timestamps else ""
        latest = max(timestamps).isoformat() if timestamps else ""

        summary = self._build_evidence_summary(
            ticker, direction, domains, t_score, template_match,
            template_wr, wm_chain, hist_wr,
        )

        return Inevitability(
            ticker=ticker,
            direction=direction,
            score=round(combined, 4),
            domains=domains,
            signals=signal_dicts,
            thompson_score=round(t_score, 4),
            template_match=template_match,
            template_win_rate=round(template_wr, 3) if template_wr is not None else None,
            world_model_chain=wm_chain,
            leading_indicators=lead_indicators,
            historical_win_rate=round(hist_wr, 3) if hist_wr is not None else None,
            evidence_summary=summary,
            expected_window_days=expected_days,
            signal_count=len(recs),
            earliest_signal=earliest,
            latest_signal=latest,
        )

    def _thompson_score(self, sources: set) -> float:
        """Weighted geometric mean of Thompson means across active sources."""
        if not self._sampler or not sources:
            return 0.5
        log_sum = 0.0
        count = 0
        for src in sources:
            try:
                dist = self._sampler.get_distribution(src)
                mean = dist.mean
                if mean > 0:
                    log_sum += math.log(mean)
                    count += 1
            except Exception:
                pass
        if count == 0:
            return 0.5
        return math.exp(log_sum / count)

    def _template_score(
        self, sources: set, direction: str
    ):
        """Return (template_id, win_rate, score 0-1, expected_days)."""
        if not self._library:
            return None, None, 0.0, 14
        try:
            matches = self._library.query_similar(sources, direction=direction)
        except Exception:
            return None, None, 0.0, 14
        if not matches:
            return None, None, 0.0, 14
        best = matches[0]
        t = best.template
        wr = t.win_rate if (t.wins + t.losses) > 0 else None
        expected_days = t.expected_move_window_days
        # Score: match_score, boosted if template has real win rate data
        score = best.match_score
        if wr is not None:
            score = score * (0.5 + 0.5 * wr)
        return t.template_id, wr, min(1.0, score), expected_days

    def _world_model_score(self, ticker: str):
        """Return (score, causal_chain) — checks if ticker is a known downstream node."""
        if not self._world_model:
            return 0.0, None
        try:
            causes = self._world_model.find_root_causes(ticker)
        except Exception:
            return 0.0, None
        if not causes:
            return 0.0, None
        best = causes[0]
        chain = list(reversed(best.path))
        score = min(1.0, best.strength)
        return score, chain

    def _lag_score(self, sources: set):
        """Score based on lag correlations — are known leading indicators present?"""
        if not self._lag_correlations:
            return 0.0, []
        relevant = []
        for entry in self._lag_correlations:
            a = entry.get("source_a", "")
            if a in sources:
                corr = abs(entry.get("correlation", 0.0))
                if corr >= 0.5:
                    relevant.append(entry)
        if not relevant:
            return 0.0, []
        # Score = mean abs correlation of relevant leading indicators, capped at 1
        score = min(1.0, sum(abs(e.get("correlation", 0)) for e in relevant) / len(relevant))
        return score, relevant[:5]  # cap at 5 for memory

    def _density_score(self, recs: list, today: datetime) -> float:
        """Recency-weighted signal density score."""
        if not recs:
            return 0.0
        weight_sum = 0.0
        for r in recs:
            age_days = max(0.0, (today - r.timestamp).total_seconds() / 86400.0)
            decay = math.exp(-age_days * math.log(2) / _DECAY_HALF_LIFE)
            weight_sum += r.strength * decay
        # Normalize: 20+ weighted units = score of 1.0
        return min(1.0, weight_sum / 20.0)

    def _historical_win_rate(self, ticker: str, direction: str) -> Optional[float]:
        """Win rate for this ticker+direction from outcomes history."""
        outcomes = self._outcome_index.get(ticker, {}).get(direction, [])
        if len(outcomes) < 3:
            return None
        return sum(1 for x in outcomes if x) / len(outcomes)

    def _combo_boost(self, domains: List[str]) -> float:
        """Look up post-mortem combo stats and apply a multiplier (0.8–1.25)."""
        if not self._combo_stats:
            return 1.0
        key = "signals:" + "+".join(sorted(domains))
        entry = self._combo_stats.get(key)
        if entry is None:
            return 1.0
        wr = entry.get("win_rate", 0.5)
        n = entry.get("n", 0)
        if n < 3:
            return 1.0
        # Above 0.5 WR → boost up to 1.25; below → penalty down to 0.8
        return max(0.8, min(1.25, 0.8 + 0.9 * wr))

    def _build_evidence_summary(
        self,
        ticker: str,
        direction: str,
        domains: List[str],
        t_score: float,
        template_match: Optional[str],
        template_wr: Optional[float],
        wm_chain: Optional[List[str]],
        hist_wr: Optional[float],
    ) -> str:
        parts = []
        parts.append(f"Domains active: {', '.join(domains)}.")
        parts.append(f"Source reliability (Thompson): {t_score:.0%}.")
        if template_match and template_wr is not None:
            parts.append(f"Matched pattern '{template_match}' with {template_wr:.0%} historical WR.")
        if wm_chain:
            parts.append(f"World Model causal path: {' → '.join(wm_chain[:4])}.")
        if hist_wr is not None:
            parts.append(f"Past {ticker} {direction} predictions: {hist_wr:.0%} correct.")
        return " ".join(parts)

    def _count_unique_tickers(self, results: List[Inevitability]) -> int:
        return len({iv.ticker for iv in results})
