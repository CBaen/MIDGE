"""LagCorrelationAnalyzer — cross-correlation across time lags.

Answers: "Does a surge in congressional trades TODAY predict a price move
in 14 days?" Cross-correlation(A[t], B[t+lag]) for lags 1-90 days.

Separate from CorrelationTracker (which does same-time Pearson on streams).
This module works on historical JSONL archives, day-granularity.

Biological analogy: Memory-based association. The organism learns that
"congressional buying THEN price surge" happens 3 weeks later on average.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_LAG_FINDINGS_PATH = _DATA_DIR / "lag_correlations.json"


@dataclass
class LagFinding:
    """A statistically significant lag-correlation finding."""
    source_a: str          # Leading indicator source
    source_b: str          # Lagging outcome source
    lag_days: int          # A leads B by this many days
    correlation: float     # Pearson r at this lag
    p_value_approx: float  # Approximate p-value (Fisher Z transform)
    n_pairs: int           # Number of aligned day pairs
    direction: str         # "positive" or "negative"
    last_updated: str      # ISO timestamp


class LagCorrelationAnalyzer:
    """
    Detects leading indicator relationships across signal sources.

    Workflow:
    1. Load signal archives via SignalArchiveReader
    2. Build daily aggregated time series per source (mean strength per day)
    3. Cross-correlate each pair at lags 1-90 days
    4. Report findings with p < 0.05

    Results persist to data/market/lag_correlations.json for continuity
    across runs. Analysis runs on cadence (every 500 steps) when enough
    archive data exists (>= min_observations days).
    """

    def __init__(
        self,
        archive_reader,
        lag_range: Tuple[int, int] = (1, 90),
        min_observations: int = 20,
        significance_threshold: float = 0.05,
        persistence_path: Optional[Path] = None,
    ):
        self._reader = archive_reader
        self._lag_min, self._lag_max = lag_range
        self._min_obs = min_observations
        self._sig_threshold = significance_threshold
        self._persistence_path = persistence_path or _LAG_FINDINGS_PATH
        self._findings: List[LagFinding] = []
        self._last_analysis_date: Optional[date] = None
        self._total_analyses: int = 0
        self._load_findings()

    def analyze(self, lookback_days: int = 90) -> List[LagFinding]:
        """Run full lag-correlation analysis over last N days of archives.

        Returns list of significant findings (correlation significant at
        p < significance_threshold). Call on cadence — reads from
        already-loaded archives after initial load_range().

        Args:
            lookback_days: How many days of history to analyze
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        # Ensure archives are loaded for this range
        self._reader.load_range(start, end)

        sources = self._reader.available_sources()
        if len(sources) < 2:
            logger.debug(
                "LagCorrelation: need at least 2 sources, have %d",
                len(sources),
            )
            return []

        # Build daily aggregated series per source
        daily_series = self._build_daily_series(sources, start, end)

        findings = []
        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                series_a = daily_series.get(src_a, {})
                series_b = daily_series.get(src_b, {})
                if not series_a or not series_b:
                    continue

                # Test A leads B, and B leads A
                for leader, follower, s_lead, s_follow in [
                    (src_a, src_b, series_a, series_b),
                    (src_b, src_a, series_b, series_a),
                ]:
                    for lag in range(self._lag_min, self._lag_max + 1):
                        finding = self._compute_lag_correlation(
                            leader, follower, lag, s_lead, s_follow,
                        )
                        if finding is not None:
                            findings.append(finding)

        # Sort by absolute correlation strength
        findings.sort(key=lambda f: abs(f.correlation), reverse=True)
        self._findings = findings
        self._last_analysis_date = end
        self._total_analyses += 1
        self._save_findings()

        logger.info(
            "LagCorrelation: analysis complete — %d significant findings "
            "across %d sources",
            len(findings),
            len(sources),
        )
        return findings

    def _build_daily_series(
        self,
        sources: List[str],
        start: date,
        end: date,
    ) -> Dict[str, Dict[date, float]]:
        """Aggregate signals to daily mean strength per source.

        Returns {source: {day: mean_strength}} dict. Day granularity
        is appropriate for multi-week lag analysis.
        """
        result: Dict[str, Dict[date, float]] = {}
        for src in sources:
            records = self._reader.query_source(src, start=start, end=end)
            daily: Dict[date, List[float]] = defaultdict(list)
            for rec in records:
                day = rec.timestamp.date()
                daily[day].append(rec.strength)
            # Collapse to mean per day
            result[src] = {
                d: sum(v) / len(v) for d, v in daily.items()
            }
        return result

    def _compute_lag_correlation(
        self,
        src_a: str,
        src_b: str,
        lag: int,
        series_a: Dict[date, float],
        series_b: Dict[date, float],
    ) -> Optional[LagFinding]:
        """Compute Pearson r between A[t] and B[t+lag].

        Returns LagFinding if p-value < threshold, else None.
        """
        # Build aligned pairs: (a_val, b_val) where b is measured lag days later
        pairs = []
        for day_a, val_a in series_a.items():
            day_b = day_a + timedelta(days=lag)
            val_b = series_b.get(day_b)
            if val_b is not None:
                pairs.append((val_a, val_b))

        n = len(pairs)
        if n < self._min_obs:
            return None

        # Pearson r
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r = _pearson(xs, ys)
        if r is None:
            return None

        # Approximate p-value via Fisher Z transform
        p_val = _fisher_p(r, n)
        if p_val >= self._sig_threshold:
            return None

        direction = "positive" if r > 0 else "negative"
        return LagFinding(
            source_a=src_a,
            source_b=src_b,
            lag_days=lag,
            correlation=round(r, 4),
            p_value_approx=round(p_val, 6),
            n_pairs=n,
            direction=direction,
            last_updated=datetime.now().isoformat(),
        )

    def get_top_findings(self, n: int = 10) -> List[LagFinding]:
        """Top N findings by absolute correlation."""
        return sorted(
            self._findings,
            key=lambda f: abs(f.correlation),
            reverse=True,
        )[:n]

    def get_leading_indicators(self, target_source: str) -> List[LagFinding]:
        """Find all sources that lead a specific target source."""
        return [
            f for f in self._findings if f.source_b == target_source
        ]

    def get_statistics(self) -> dict:
        """HolonProxy.sense() delegation."""
        return {
            "total_analyses": self._total_analyses,
            "significant_findings": len(self._findings),
            "last_analysis_date": (
                self._last_analysis_date.isoformat()
                if self._last_analysis_date
                else None
            ),
            "top_findings": [
                {
                    "source_a": f.source_a,
                    "source_b": f.source_b,
                    "lag_days": f.lag_days,
                    "correlation": f.correlation,
                }
                for f in self.get_top_findings(5)
            ],
        }

    def _save_findings(self):
        """Persist findings to JSON."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = [asdict(f) for f in self._findings]
            self._persistence_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception as e:
            logger.warning("LagCorrelation: could not save findings: %s", e)

    def _load_findings(self):
        """Load prior findings from JSON."""
        if self._persistence_path and self._persistence_path.exists():
            try:
                raw = json.loads(
                    self._persistence_path.read_text(encoding="utf-8"),
                )
                self._findings = [LagFinding(**r) for r in raw]
            except Exception:
                self._findings = []

    def save(self):
        """Public persist method for bootstrap shutdown hooks."""
        self._save_findings()


# ── Pure-Python statistics (no scipy dependency) ────────────────────


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    """Pearson correlation coefficient. Returns None if variance is zero."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    r = num / math.sqrt(var_x * var_y)
    return max(-1.0, min(1.0, r))


def _fisher_p(r: float, n: int) -> float:
    """Approximate p-value for Pearson r via Fisher Z transform.

    z = arctanh(r), se = 1/sqrt(n-3), z_score = z / se
    Two-tailed p from standard normal.
    """
    if abs(r) >= 1.0 or n <= 3:
        return 1.0
    try:
        z = math.atanh(r)
        se = 1.0 / math.sqrt(n - 3)
        z_score = abs(z / se)
        p_one_tail = _norm_sf(z_score)
        return min(1.0, 2 * p_one_tail)
    except (ValueError, ZeroDivisionError):
        return 1.0


def _norm_sf(z: float) -> float:
    """Survival function of standard normal (1 - CDF).

    Rational approximation (Abramowitz & Stegun 26.2.17).
    """
    if z < 0:
        return 1.0 - _norm_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (
        0.319381530
        + t * (
            -0.356563782
            + t * (
                1.781477937
                + t * (-1.821255978 + t * 1.330274429)
            )
        )
    )
    return poly * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
