#!/usr/bin/env python3
"""
kelly_position_sizer.py - Kelly Criterion Position Sizing for MIDGE

Translates signal reliability (Thompson distribution mean) and historical
win/loss ratios (from outcomes.jsonl) into a recommended position fraction.

Kelly formula:
    f = (b*p - q) / b
    where:
        p = probability of winning (Thompson distribution mean)
        q = 1 - p
        b = average win / average loss ratio (from outcomes.jsonl)

Always applies half-Kelly (f/2) for safety, then caps at MAX_KELLY_FRACTION.
When fewer than MIN_OBSERVATIONS outcomes exist for a source, falls back to
DEFAULT_WIN_LOSS_RATIO rather than fabricating a ratio from thin data.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_WIN_LOSS_RATIO = 1.5    # Used when < MIN_OBSERVATIONS outcomes exist
MIN_OBSERVATIONS = 10           # Required before using historical b
MAX_KELLY_FRACTION = 0.05       # Never recommend > 5% of bankroll
HALF_KELLY_FACTOR = 0.5         # Always use half-Kelly

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
SIZING_LOG_PATH = DATA_DIR / "position_sizing_log.jsonl"


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class PositionRecommendation:
    """A complete Kelly-sized position recommendation for one signal + symbol."""
    signal_source: str
    symbol: str
    kelly_full: float           # Raw Kelly fraction before safety adjustments
    kelly_half: float           # kelly_full * HALF_KELLY_FACTOR
    kelly_capped: float         # min(kelly_half, MAX_KELLY_FRACTION)
    p_win: float                # Thompson distribution mean for signal_source
    win_loss_ratio: float       # b parameter used in Kelly formula
    confidence_in_sizing: str   # "low" / "medium" / "high"
    observations_used: int      # Outcome records that informed win_loss_ratio
    generated_at: str           # ISO timestamp


# ── Main class ─────────────────────────────────────────────────────────────────

class KellyPositionSizer:
    """
    Converts signal reliability into actionable position sizes.

    Dependencies injected at construction:
        thompson_sampler: ThompsonSampler instance — provides p_win via
            get_distribution(signal_id, regime).mean
        data_dir: Directory containing outcomes.jsonl (default: DATA_DIR)

    Usage:
        sizer = KellyPositionSizer(thompson_sampler)
        rec = sizer.recommend("sec_form4", "AAPL", regime="bull")
        print(f"Size {rec.kelly_capped:.1%} of bankroll in {rec.symbol}")
    """

    def __init__(
        self,
        thompson_sampler,
        data_dir: Optional[Path] = None,
        max_fraction: float = MAX_KELLY_FRACTION,
    ):
        """
        Args:
            thompson_sampler: ThompsonSampler with get_distribution(id, regime)
            data_dir: Override for data directory (default: DATA_DIR)
            max_fraction: Hard cap on recommended fraction (default: 0.05)
        """
        self.thompson = thompson_sampler
        self.data_dir = data_dir or DATA_DIR
        self.max_fraction = max_fraction
        self.outcomes_path = self.data_dir / "outcomes.jsonl"

        # Cache: {source: (win_loss_ratio, observation_count)}
        self._ratio_cache: Dict[str, Tuple[float, int]] = {}

        self._load_historical_ratios()

    # ── Public API ─────────────────────────────────────────────────────────────

    def recommend(
        self,
        signal_source: str,
        symbol: str,
        regime: str = "default",
    ) -> PositionRecommendation:
        """
        Compute a Kelly-sized position recommendation.

        Args:
            signal_source: Thompson distribution key (e.g. "sec_form4")
            symbol: Ticker symbol this recommendation is for
            regime: Market regime passed through to ThompsonSampler

        Returns:
            PositionRecommendation with full, half, and capped Kelly fractions
        """
        # p from Thompson distribution mean
        try:
            dist = self.thompson.get_distribution(signal_source, regime)
            p_win = dist.mean
        except Exception as e:
            logger.warning(f"KellyPositionSizer: Thompson lookup failed for {signal_source}: {e}")
            p_win = 0.5  # Uninformative fallback

        q_lose = 1.0 - p_win

        # b from historical outcomes
        win_loss_ratio, obs_count = self._get_win_loss_ratio(signal_source)

        # Kelly formula: f = (b*p - q) / b
        kelly_full = (win_loss_ratio * p_win - q_lose) / win_loss_ratio

        # Clamp to [0, 1] — negative Kelly means "don't trade"
        kelly_full = max(0.0, min(1.0, kelly_full))

        kelly_half = kelly_full * HALF_KELLY_FACTOR
        kelly_capped = min(kelly_half, self.max_fraction)

        # Confidence in the sizing
        if obs_count >= 50:
            confidence = "high"
        elif obs_count >= MIN_OBSERVATIONS:
            confidence = "medium"
        else:
            confidence = "low"

        rec = PositionRecommendation(
            signal_source=signal_source,
            symbol=symbol,
            kelly_full=round(kelly_full, 6),
            kelly_half=round(kelly_half, 6),
            kelly_capped=round(kelly_capped, 6),
            p_win=round(p_win, 6),
            win_loss_ratio=round(win_loss_ratio, 4),
            confidence_in_sizing=confidence,
            observations_used=obs_count,
            generated_at=datetime.now().isoformat(),
        )

        self._log_recommendation(rec)

        logger.info(
            f"Kelly sizing | source={signal_source} symbol={symbol} regime={regime} "
            f"p_win={p_win:.3f} b={win_loss_ratio:.2f} "
            f"full={kelly_full:.4f} half={kelly_half:.4f} capped={kelly_capped:.4f} "
            f"confidence={confidence} obs={obs_count}"
        )

        return rec

    def refresh_ratios(self) -> None:
        """Clear the ratio cache and reload from outcomes.jsonl."""
        self._ratio_cache.clear()
        self._load_historical_ratios()
        logger.info("KellyPositionSizer: ratio cache refreshed")

    def get_statistics(self) -> dict:
        """HolonProxy.sense() delegation — summary of cached ratios."""
        return {
            "sources_with_historical_data": len(self._ratio_cache),
            "outcomes_path": str(self.outcomes_path),
            "max_kelly_fraction": self.max_fraction,
            "min_observations_required": MIN_OBSERVATIONS,
            "default_win_loss_ratio": DEFAULT_WIN_LOSS_RATIO,
            "cached_sources": [
                {
                    "source": src,
                    "win_loss_ratio": round(ratio, 4),
                    "observations": obs,
                }
                for src, (ratio, obs) in sorted(self._ratio_cache.items())
            ],
        }

    def save(self) -> None:
        """No-op: recommendations are logged to JSONL as they are generated."""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_win_loss_ratio(self, source: str) -> Tuple[float, int]:
        """
        Return (win_loss_ratio, observation_count) for a signal source.

        Checks the in-memory cache first. If not cached, reads outcomes.jsonl
        directly and filters by source. Falls back to DEFAULT_WIN_LOSS_RATIO
        when fewer than MIN_OBSERVATIONS records exist or when there are no
        wins or losses (which would cause division by zero).

        Args:
            source: Signal source string matching outcomes.jsonl "source" field

        Returns:
            Tuple of (ratio, count). ratio is always > 0.
        """
        if source in self._ratio_cache:
            return self._ratio_cache[source]

        wins: List[float] = []
        losses: List[float] = []

        if self.outcomes_path.exists():
            try:
                with open(self.outcomes_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if record.get("source") != source:
                            continue

                        pct = record.get("price_change_pct")
                        success = record.get("success")

                        if pct is None or success is None:
                            continue

                        abs_pct = abs(float(pct))
                        if success:
                            wins.append(abs_pct)
                        else:
                            losses.append(abs_pct)

            except Exception as e:
                logger.warning(f"KellyPositionSizer: failed reading outcomes for {source}: {e}")

        total_obs = len(wins) + len(losses)

        if total_obs < MIN_OBSERVATIONS or not wins or not losses:
            return (DEFAULT_WIN_LOSS_RATIO, total_obs)

        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        if avg_loss == 0.0:
            ratio = DEFAULT_WIN_LOSS_RATIO
        else:
            ratio = avg_win / avg_loss

        self._ratio_cache[source] = (ratio, total_obs)
        return (ratio, total_obs)

    def _load_historical_ratios(self) -> None:
        """
        Pre-populate the cache from outcomes.jsonl on startup.

        Builds a per-source dictionary of (wins_list, losses_list), then
        only caches sources that have reached MIN_OBSERVATIONS. Sources below
        the threshold still fall back to DEFAULT_WIN_LOSS_RATIO at call time.
        """
        if not self.outcomes_path.exists():
            return

        by_source: Dict[str, Tuple[List[float], List[float]]] = {}

        try:
            with open(self.outcomes_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    source = record.get("source")
                    pct = record.get("price_change_pct")
                    success = record.get("success")

                    if not source or pct is None or success is None:
                        continue

                    if source not in by_source:
                        by_source[source] = ([], [])

                    abs_pct = abs(float(pct))
                    if success:
                        by_source[source][0].append(abs_pct)
                    else:
                        by_source[source][1].append(abs_pct)

        except Exception as e:
            logger.warning(f"KellyPositionSizer: failed pre-loading historical ratios: {e}")
            return

        cached = 0
        for source, (wins, losses) in by_source.items():
            total_obs = len(wins) + len(losses)
            if total_obs < MIN_OBSERVATIONS or not wins or not losses:
                continue

            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            ratio = (avg_win / avg_loss) if avg_loss > 0 else DEFAULT_WIN_LOSS_RATIO

            self._ratio_cache[source] = (ratio, total_obs)
            cached += 1

        if cached > 0:
            logger.info(f"KellyPositionSizer: pre-loaded historical ratios for {cached} sources")

    def _log_recommendation(self, rec: PositionRecommendation) -> None:
        """Append the recommendation to position_sizing_log.jsonl."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "signal_source": rec.signal_source,
            "symbol": rec.symbol,
            "kelly_full": rec.kelly_full,
            "kelly_half": rec.kelly_half,
            "kelly_capped": rec.kelly_capped,
            "p_win": rec.p_win,
            "win_loss_ratio": rec.win_loss_ratio,
            "confidence_in_sizing": rec.confidence_in_sizing,
            "observations_used": rec.observations_used,
            "generated_at": rec.generated_at,
        }
        try:
            with open(SIZING_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"KellyPositionSizer: failed to log recommendation: {e}")
