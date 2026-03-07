"""GrangerAnalyzer — directional causality detection between signal sources.

Complements LagCorrelationAnalyzer which answers "do they correlate at some
lag?" This module answers the harder question: "does knowing A's past
*improve* prediction of B beyond what B's own past already predicts?"

Key difference from lag correlation:
- Lag correlation finds co-movement (possibly spurious from autocorrelation).
- Granger causality controls for the target's own history.
  If price trends (autocorrelation), lag-correlation finds false leading
  indicators. Granger won't — it asks "does A add information?"

Uses statsmodels.tsa.stattools.grangercausalitytests (bivariate F-test).
Series are first-differenced for stationarity before testing.

Results persist to data/market/granger_causality.json and feed into
ConvergenceAlerter as causal weight modifiers — sources that Granger-cause
price movement get higher effective weight when predicting that asset.
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
_GRANGER_PATH = _DATA_DIR / "granger_causality.json"

# Lazy import — statsmodels is heavy, only load when needed
_grangercausalitytests = None


def _get_granger_func():
    """Lazy-load statsmodels Granger test to avoid import-time cost."""
    global _grangercausalitytests
    if _grangercausalitytests is None:
        from statsmodels.tsa.stattools import grangercausalitytests
        _grangercausalitytests = grangercausalitytests
    return _grangercausalitytests


@dataclass
class GrangerFinding:
    """A Granger causality finding between two signal sources."""
    cause_source: str        # Hypothesized cause (X)
    effect_source: str       # Hypothesized effect (Y)
    best_lag: int            # Lag (days) with strongest evidence
    f_statistic: float       # F-test statistic at best lag
    p_value: float           # p-value at best lag
    direction: str           # "causal" or "not_causal"
    all_lags_tested: int     # Number of lags tested
    significant_lags: int    # Number of lags with p < threshold
    min_p_value: float       # Smallest p-value across all lags
    n_observations: int      # Length of time series used
    last_updated: str        # ISO timestamp


class GrangerAnalyzer:
    """Detects directional causal relationships between signal sources.

    Workflow:
    1. Load daily aggregated time series from SignalArchiveReader
    2. First-difference for stationarity
    3. Run Granger causality test (bivariate F-test) for each pair
    4. Report findings with p < significance_threshold
    5. Persist to granger_causality.json

    Integration: Results feed into ConvergenceAlerter — sources that
    Granger-cause price movement get a causal weight boost.
    """

    def __init__(
        self,
        archive_reader,
        max_lag: int = 30,
        min_observations: int = 40,
        significance_threshold: float = 0.05,
        persistence_path: Optional[Path] = None,
    ):
        self._reader = archive_reader
        self._max_lag = max_lag
        self._min_obs = min_observations
        self._sig_threshold = significance_threshold
        self._persistence_path = persistence_path or _GRANGER_PATH
        self._findings: List[GrangerFinding] = []
        self._causal_pairs: Dict[Tuple[str, str], float] = {}
        self._last_analysis_date: Optional[date] = None
        self._total_analyses: int = 0
        self._load_findings()

    def analyze(self, lookback_days: int = 180) -> List[GrangerFinding]:
        """Run Granger causality analysis over signal archive history.

        Tests each ordered pair (A→B and B→A separately) to detect
        directional causation. Longer lookback than lag_correlation
        because Granger tests need more data for stable estimates.

        Args:
            lookback_days: How many days of history to analyze.
                           180 recommended (need 2x max_lag minimum).

        Returns:
            List of significant GrangerFinding objects.
        """
        end = date.today()
        start = end - timedelta(days=lookback_days)

        self._reader.load_range(start, end)

        sources = self._reader.available_sources()
        if len(sources) < 2:
            logger.debug(
                "GrangerAnalyzer: need at least 2 sources, have %d",
                len(sources),
            )
            return []

        daily_series = self._build_daily_series(sources, start, end)

        findings = []
        pairs_tested = 0
        for i, src_a in enumerate(sources):
            for src_b in sources[i + 1:]:
                series_a = daily_series.get(src_a)
                series_b = daily_series.get(src_b)
                if series_a is None or series_b is None:
                    continue

                # Test both directions: A→B and B→A
                for cause, effect, s_cause, s_effect in [
                    (src_a, src_b, series_a, series_b),
                    (src_b, src_a, series_b, series_a),
                ]:
                    finding = self._test_granger(
                        cause, effect, s_cause, s_effect,
                    )
                    if finding is not None:
                        findings.append(finding)
                    pairs_tested += 1

        findings.sort(key=lambda f: f.p_value)
        self._findings = findings
        self._rebuild_causal_pairs()
        self._last_analysis_date = end
        self._total_analyses += 1
        self._save_findings()

        logger.info(
            "GrangerAnalyzer: %d causal relationships found across "
            "%d directional pairs tested",
            len(findings),
            pairs_tested,
        )
        return findings

    def _build_daily_series(
        self,
        sources: List[str],
        start: date,
        end: date,
    ) -> Dict[str, Dict[date, float]]:
        """Aggregate signals to daily mean strength per source."""
        result: Dict[str, Dict[date, float]] = {}
        for src in sources:
            records = self._reader.query_source(src, start=start, end=end)
            daily: Dict[date, List[float]] = defaultdict(list)
            for rec in records:
                day = rec.timestamp.date()
                daily[day].append(rec.strength)
            if len(daily) >= self._min_obs:
                result[src] = {
                    d: sum(v) / len(v) for d, v in daily.items()
                }
        return result

    def _test_granger(
        self,
        cause_source: str,
        effect_source: str,
        cause_series: Dict[date, float],
        effect_series: Dict[date, float],
    ) -> Optional[GrangerFinding]:
        """Test whether cause_source Granger-causes effect_source.

        Steps:
        1. Align both series to common dates
        2. First-difference for stationarity
        3. Run statsmodels grangercausalitytests
        4. Return finding if any lag is significant
        """
        # Align to common dates (sorted)
        common_dates = sorted(
            set(cause_series.keys()) & set(effect_series.keys())
        )
        if len(common_dates) < self._min_obs:
            return None

        # Build aligned arrays
        y_vals = [effect_series[d] for d in common_dates]
        x_vals = [cause_series[d] for d in common_dates]

        # First-difference for stationarity
        y_diff = [y_vals[i] - y_vals[i - 1] for i in range(1, len(y_vals))]
        x_diff = [x_vals[i] - x_vals[i - 1] for i in range(1, len(x_vals))]

        n = len(y_diff)
        if n < self._min_obs:
            return None

        # Max lag must leave enough observations
        max_lag = min(self._max_lag, n // 3)
        if max_lag < 1:
            return None

        try:
            import numpy as np
            granger_func = _get_granger_func()

            # grangercausalitytests expects shape (n, 2) with [y, x]
            data = np.column_stack([y_diff, x_diff])

            results = granger_func(data, maxlag=max_lag, verbose=False)

            # Extract F-test results per lag
            best_lag = None
            best_p = 1.0
            best_f = 0.0
            significant_count = 0

            for lag in range(1, max_lag + 1):
                lag_results = results[lag]
                # lag_results is a tuple: (test_dict, [ols_results])
                test_dict = lag_results[0]
                f_stat, p_val, df_denom, df_num = test_dict["ssr_ftest"]

                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
                    best_lag = lag

                if p_val < self._sig_threshold:
                    significant_count += 1

            if best_p >= self._sig_threshold:
                return None

            # Bonferroni correction for multiple lag tests
            corrected_p = min(1.0, best_p * max_lag)
            if corrected_p >= self._sig_threshold:
                return None

            return GrangerFinding(
                cause_source=cause_source,
                effect_source=effect_source,
                best_lag=best_lag,
                f_statistic=round(best_f, 4),
                p_value=round(corrected_p, 6),
                direction="causal",
                all_lags_tested=max_lag,
                significant_lags=significant_count,
                min_p_value=round(best_p, 6),
                n_observations=n,
                last_updated=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.debug(
                "GrangerAnalyzer: test failed for %s→%s: %s",
                cause_source, effect_source, e,
            )
            return None

    def _rebuild_causal_pairs(self):
        """Build lookup dict of causal pair strengths.

        Maps (cause, effect) → causal_strength where strength is
        derived from p-value: -log10(p) normalized to [0, 1].
        Stronger evidence (lower p) → higher strength.
        """
        self._causal_pairs = {}
        for f in self._findings:
            # -log10(p) gives a positive number; p=0.001 → 3, p=0.05 → 1.3
            # Normalize: cap at 5 (p=0.00001), divide by 5
            if f.p_value > 0:
                raw = -math.log10(f.p_value)
            else:
                raw = 5.0
            strength = min(1.0, raw / 5.0)
            key = (f.cause_source, f.effect_source)
            # Keep strongest if multiple findings for same pair
            if strength > self._causal_pairs.get(key, 0.0):
                self._causal_pairs[key] = strength

    def get_causal_strength(
        self, cause_source: str, effect_source: str
    ) -> float:
        """Get causal strength from cause to effect.

        Returns:
            Float in [0, 1]. 0 = no evidence of causation.
            Higher = stronger Granger-causal evidence.
        """
        return self._causal_pairs.get((cause_source, effect_source), 0.0)

    def get_causes_of(self, target_source: str) -> List[GrangerFinding]:
        """Find all sources that Granger-cause the target."""
        return [
            f for f in self._findings
            if f.effect_source == target_source
        ]

    def get_effects_of(self, source: str) -> List[GrangerFinding]:
        """Find all sources that this source Granger-causes."""
        return [
            f for f in self._findings
            if f.cause_source == source
        ]

    def get_bidirectional_pairs(self) -> List[Tuple[GrangerFinding, GrangerFinding]]:
        """Find pairs where A→B AND B→A (feedback loops)."""
        cause_map: Dict[Tuple[str, str], GrangerFinding] = {
            (f.cause_source, f.effect_source): f for f in self._findings
        }
        bidirectional = []
        seen = set()
        for f in self._findings:
            reverse = (f.effect_source, f.cause_source)
            if reverse in cause_map and reverse not in seen:
                seen.add((f.cause_source, f.effect_source))
                seen.add(reverse)
                bidirectional.append((f, cause_map[reverse]))
        return bidirectional

    def get_statistics(self) -> dict:
        """HolonProxy.sense() delegation."""
        return {
            "total_analyses": self._total_analyses,
            "causal_findings": len(self._findings),
            "causal_pairs": len(self._causal_pairs),
            "bidirectional_pairs": len(self.get_bidirectional_pairs()),
            "last_analysis_date": (
                self._last_analysis_date.isoformat()
                if self._last_analysis_date
                else None
            ),
            "top_findings": [
                {
                    "cause": f.cause_source,
                    "effect": f.effect_source,
                    "lag": f.best_lag,
                    "p_value": f.p_value,
                }
                for f in self._findings[:5]
            ],
        }

    def _save_findings(self):
        """Persist findings to JSON."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = [asdict(f) for f in self._findings]
            tmp = self._persistence_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self._persistence_path)
        except Exception as e:
            logger.warning("GrangerAnalyzer: could not save findings: %s", e)

    def _load_findings(self):
        """Load prior findings from JSON."""
        if self._persistence_path and self._persistence_path.exists():
            try:
                raw = json.loads(
                    self._persistence_path.read_text(encoding="utf-8"),
                )
                self._findings = [GrangerFinding(**r) for r in raw]
                self._rebuild_causal_pairs()
            except Exception:
                self._findings = []

    def save(self):
        """Public persist method for bootstrap shutdown hooks."""
        self._save_findings()
