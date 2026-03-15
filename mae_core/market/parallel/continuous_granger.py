"""Continuous Granger causality discovery — independent parallel process.

Reads ALL signals from the MIDGE signal archive, groups them by domain,
and continuously tests every domain pair for directional Granger causality.
Discovers which domains PRECEDE other domains across the full 874K+ signal
history, writing results to data/market/granger_continuous.json.

Runs completely independently of the daemon loop:
    python -m mae_core.market.parallel.continuous_granger

Why domain-level rather than source-level:
  The daemon's GrangerAnalyzer tests individual sources (e.g. sec_form4 →
  yfinance_price). This process tests DOMAINS (e.g. insider → technical),
  which is a higher signal-to-noise view — individual sources are sparse,
  domains are dense. Both views are valuable; this one complements.

Design:
  - Loads entire archive once per cycle into domain → {date: [strengths]} dict
  - For each ordered domain pair (A→B and B→A): runs bivariate Granger F-test
    at lags 1–14 days (market-relevant window)
  - Bonferroni correction across tested lags (same as daemon GrangerAnalyzer)
  - Merges new significant findings with any existing ones in the output file
    (keeps the strongest p-value per pair across runs)
  - Sleeps 60 seconds, then re-loads to pick up newly arrived signals
  - Logs to data/market/granger_continuous.log with rotation

Statistical notes (same methods as granger_analyzer.py):
  - First-differencing for stationarity before testing
  - statsmodels grangercausalitytests bivariate F-test
  - Significance threshold p < 0.05 after Bonferroni correction
  - Minimum 40 common observations required per pair
"""

import json
import logging
import logging.handlers
import math
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SIGNALS_DIR = _PROJECT_ROOT / "data" / "midge" / "signals"
_DATA_DIR = _PROJECT_ROOT / "data" / "market"
_OUTPUT_PATH = _DATA_DIR / "granger_continuous.json"
_LOG_PATH = _DATA_DIR / "granger_continuous.log"

# ── Configuration ──────────────────────────────────────────────────────────────
_SLEEP_SECONDS = 60
_MAX_LAG = 14           # Test lags 1-14 days (market-relevant)
_MIN_OBS = 40           # Minimum aligned observations per pair
_SIG_THRESHOLD = 0.05   # p-value significance threshold (pre-correction)
_LOOKBACK_DAYS = 365    # How far back to read (1 year default; use all if fewer)

# Source → domain mapping (mirrors PatternLibrary._SOURCE_DOMAIN_MAP)
_SOURCE_DOMAIN_MAP: Dict[str, str] = {
    "sec_form4": "insider", "openinsider_purchase": "insider",
    "insider_cluster": "insider", "openinsider_cluster": "insider",
    "finviz_insider": "insider",
    "sec_form8k": "events", "sec_efts": "events",
    "finnhub_earnings": "events", "finnhub_news": "events",
    "finnhub_realtime": "events", "finnhub_earnings_calendar": "events",
    "massive_snapshot": "events", "hiring_tracker": "events",
    "yahoo_rss": "events", "kalshi_market": "events",
    "fred_macro": "macro", "economic_calendar": "macro",
    "finnhub_economic": "macro", "fred_yields": "macro",
    "usda_agriculture": "macro",
    "ta_rsi": "technical", "ta_macd": "technical", "ta_bollinger": "technical",
    "ta_structure": "technical", "ta_candle": "technical",
    "session_sweep": "technical", "session_sweep_ifvg": "technical",
    "fractal_resonance": "technical", "order_flow": "technical",
    "finviz_unusual_volume": "technical", "finviz_short_squeeze": "technical",
    "yfinance_price": "technical", "polygon": "technical",
    "social_sentiment": "sentiment", "google_trends": "sentiment",
    "stocktwits_sentiment": "sentiment", "social_text": "sentiment",
    "congressional": "government", "congress_legislation": "government",
    "senate": "government",
    "contract_award": "contracts", "sam_gov": "contracts",
    "contract_prediction": "contracts",
    "finnhub_analyst": "fundamentals",
    "cot_positioning": "positioning", "binance_funding": "positioning",
    "vix_term_structure": "volatility",
    "crypto_coingecko": "crypto", "crypto_coincap": "crypto",
    "activist_13d": "institutional", "institutional_13f": "institutional",
    "finra_short": "institutional",
    "eia_energy": "energy",
}

# ── Logging ────────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    """Configure rotating file + console logging."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("continuous_granger")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Rotating file handler — 5 MB max, 3 backups
        fh = logging.handlers.RotatingFileHandler(
            _LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s [continuous_granger] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ContinuousGrangerFinding:
    """A Granger causality finding between two DOMAINS (not sources)."""
    cause_domain: str       # Hypothesized cause domain (X)
    effect_domain: str      # Hypothesized effect domain (Y)
    best_lag: int           # Lag (days) with strongest evidence
    f_statistic: float      # F-test statistic at best lag
    p_value: float          # Bonferroni-corrected p-value at best lag
    direction: str          # Always "causal"
    all_lags_tested: int    # Number of lags tested (1 to max_lag)
    significant_lags: int   # Number of lags with uncorrected p < threshold
    min_p_value: float      # Smallest uncorrected p-value across all lags
    n_observations: int     # Number of aligned daily observations used
    signal_count_cause: int # Total signals in cause domain
    signal_count_effect: int# Total signals in effect domain
    first_detected: str     # ISO timestamp when first found
    last_updated: str       # ISO timestamp of most recent confirmation


# ── Archive reader ─────────────────────────────────────────────────────────────

def _load_archive(lookback_days: int, logger: logging.Logger) -> Dict[str, Dict[str, List[float]]]:
    """Read signal archive and group by domain → date_str → [strengths].

    Returns:
        dict mapping domain_name → {date_iso_str → [strength_floats]}

    Uses the `domain` field directly from signal records (already mapped
    at write time). Falls back to source-based mapping for records that
    have `source` but a blank or missing `domain`.
    """
    if not _SIGNALS_DIR.exists():
        logger.warning("Signal archive not found: %s", _SIGNALS_DIR)
        return {}

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    # domain → date_str → list of strengths
    domain_daily: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    files_read = 0
    signals_read = 0
    skipped_old = 0
    skipped_unknown = 0

    for jsonl_path in sorted(_SIGNALS_DIR.glob("*.jsonl")):
        # Fast path: skip by filename date before opening
        fname = jsonl_path.stem  # e.g. "2026-03-01"
        if fname < cutoff_str:
            skipped_old += 1
            continue

        try:
            with open(jsonl_path, encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        rec = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue

                    # Resolve domain
                    domain = rec.get("domain", "")
                    if not domain or domain == "unknown":
                        source = rec.get("source", "")
                        domain = _SOURCE_DOMAIN_MAP.get(source, "")

                    if not domain or domain in ("unknown", "meta", "convergence"):
                        skipped_unknown += 1
                        continue

                    # Resolve date from timestamp or received_at
                    ts_str = rec.get("timestamp") or rec.get("received_at", "")
                    if not ts_str:
                        continue
                    # Accept ISO strings: "2026-03-01T..." → "2026-03-01"
                    date_str = ts_str[:10]
                    if date_str < cutoff_str:
                        continue

                    strength = float(rec.get("strength", 0.5))
                    domain_daily[domain][date_str].append(strength)
                    signals_read += 1

            files_read += 1

        except Exception as exc:
            logger.debug("Error reading %s: %s", jsonl_path.name, exc)

    logger.info(
        "Archive loaded: %d files, %d signals, %d domains | "
        "skipped_old=%d skipped_unknown=%d",
        files_read, signals_read, len(domain_daily),
        skipped_old, skipped_unknown,
    )
    return domain_daily


def _build_domain_series(
    domain_daily: Dict[str, Dict[str, List[float]]],
    min_obs: int,
) -> Dict[str, Dict[str, float]]:
    """Convert domain_daily to domain → {date_str → mean_strength}.

    Only keeps domains with at least min_obs distinct dates.
    """
    result: Dict[str, Dict[str, float]] = {}
    for domain, daily in domain_daily.items():
        if len(daily) >= min_obs:
            result[domain] = {
                d: sum(vals) / len(vals)
                for d, vals in daily.items()
            }
    return result


# ── Granger test ───────────────────────────────────────────────────────────────

def _test_domain_pair(
    cause_domain: str,
    effect_domain: str,
    cause_series: Dict[str, float],
    effect_series: Dict[str, float],
    min_obs: int,
    max_lag: int,
    sig_threshold: float,
    signal_counts: Dict[str, int],
) -> Optional[ContinuousGrangerFinding]:
    """Run Granger causality test for cause_domain → effect_domain.

    Steps mirror granger_analyzer.py:
    1. Align to common dates (sorted)
    2. First-difference for stationarity
    3. Run statsmodels grangercausalitytests
    4. Bonferroni correction over lags tested
    5. Return finding if corrected p < sig_threshold
    """
    # Align to common dates
    common_dates = sorted(set(cause_series.keys()) & set(effect_series.keys()))
    if len(common_dates) < min_obs:
        return None

    y_vals = [effect_series[d] for d in common_dates]
    x_vals = [cause_series[d] for d in common_dates]

    # First-difference for stationarity
    y_diff = [y_vals[i] - y_vals[i - 1] for i in range(1, len(y_vals))]
    x_diff = [x_vals[i] - x_vals[i - 1] for i in range(1, len(x_vals))]

    n = len(y_diff)
    if n < min_obs:
        return None

    # Cap max_lag to leave enough degrees of freedom
    effective_max_lag = min(max_lag, n // 3)
    if effective_max_lag < 1:
        return None

    try:
        data = np.column_stack([y_diff, x_diff])
        results = grangercausalitytests(data, maxlag=effective_max_lag, verbose=False)

        best_lag: Optional[int] = None
        best_p = 1.0
        best_f = 0.0
        significant_count = 0

        for lag in range(1, effective_max_lag + 1):
            test_dict = results[lag][0]
            f_stat, p_val, _df_denom, _df_num = test_dict["ssr_ftest"]

            if p_val < best_p:
                best_p = p_val
                best_f = f_stat
                best_lag = lag

            if p_val < sig_threshold:
                significant_count += 1

        if best_p >= sig_threshold or best_lag is None:
            return None

        # Bonferroni correction: multiply best p-value by number of lags tested
        corrected_p = min(1.0, best_p * effective_max_lag)
        if corrected_p >= sig_threshold:
            return None

        now_iso = datetime.utcnow().isoformat()
        return ContinuousGrangerFinding(
            cause_domain=cause_domain,
            effect_domain=effect_domain,
            best_lag=best_lag,
            f_statistic=round(best_f, 4),
            p_value=round(corrected_p, 6),
            direction="causal",
            all_lags_tested=effective_max_lag,
            significant_lags=significant_count,
            min_p_value=round(best_p, 6),
            n_observations=n,
            signal_count_cause=signal_counts.get(cause_domain, 0),
            signal_count_effect=signal_counts.get(effect_domain, 0),
            first_detected=now_iso,
            last_updated=now_iso,
        )

    except Exception:
        return None


# ── Persistence ────────────────────────────────────────────────────────────────

def _load_existing_findings() -> Dict[Tuple[str, str], ContinuousGrangerFinding]:
    """Load existing findings from granger_continuous.json.

    Returns dict keyed by (cause_domain, effect_domain).
    """
    if not _OUTPUT_PATH.exists():
        return {}
    try:
        raw = json.loads(_OUTPUT_PATH.read_text(encoding="utf-8"))
        result: Dict[Tuple[str, str], ContinuousGrangerFinding] = {}
        for item in raw.get("findings", []):
            f = ContinuousGrangerFinding(**item)
            result[(f.cause_domain, f.effect_domain)] = f
        return result
    except Exception:
        return {}


def _merge_findings(
    existing: Dict[Tuple[str, str], ContinuousGrangerFinding],
    new_findings: List[ContinuousGrangerFinding],
) -> Dict[Tuple[str, str], ContinuousGrangerFinding]:
    """Merge new findings into existing, keeping strongest p-value per pair.

    For pairs already known:
      - Keep the lower (stronger) p_value
      - Preserve first_detected from the older record
      - Update last_updated to now
    New pairs are added as-is.
    """
    merged = dict(existing)  # shallow copy
    for f in new_findings:
        key = (f.cause_domain, f.effect_domain)
        if key in merged:
            prior = merged[key]
            if f.p_value < prior.p_value:
                # New finding is stronger — update most fields, preserve first_detected
                updated = ContinuousGrangerFinding(
                    cause_domain=f.cause_domain,
                    effect_domain=f.effect_domain,
                    best_lag=f.best_lag,
                    f_statistic=f.f_statistic,
                    p_value=f.p_value,
                    direction=f.direction,
                    all_lags_tested=f.all_lags_tested,
                    significant_lags=f.significant_lags,
                    min_p_value=f.min_p_value,
                    n_observations=f.n_observations,
                    signal_count_cause=f.signal_count_cause,
                    signal_count_effect=f.signal_count_effect,
                    first_detected=prior.first_detected,  # preserve
                    last_updated=datetime.utcnow().isoformat(),
                )
                merged[key] = updated
            else:
                # Existing is stronger — just freshen last_updated
                merged[key] = ContinuousGrangerFinding(
                    **{**asdict(prior), "last_updated": datetime.utcnow().isoformat()}
                )
        else:
            merged[key] = f
    return merged


def _save_findings(
    findings: Dict[Tuple[str, str], ContinuousGrangerFinding],
    cycle: int,
    pairs_tested: int,
    domains_active: int,
    logger: logging.Logger,
):
    """Write findings to granger_continuous.json (atomic replace)."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        sorted_findings = sorted(
            findings.values(), key=lambda f: f.p_value
        )
        output = {
            "meta": {
                "last_updated": datetime.utcnow().isoformat(),
                "cycle": cycle,
                "pairs_tested_this_cycle": pairs_tested,
                "domains_active": domains_active,
                "total_findings": len(sorted_findings),
                "description": (
                    "Domain-level Granger causality. "
                    "Significant means: knowing domain A's past improves "
                    "prediction of domain B beyond B's own history."
                ),
            },
            "findings": [asdict(f) for f in sorted_findings],
        }
        tmp = _OUTPUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
        tmp.replace(_OUTPUT_PATH)
    except Exception as exc:
        logger.error("Failed to save findings: %s", exc)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run_continuous(
    lookback_days: int = _LOOKBACK_DAYS,
    sleep_seconds: int = _SLEEP_SECONDS,
    max_lag: int = _MAX_LAG,
    min_obs: int = _MIN_OBS,
    sig_threshold: float = _SIG_THRESHOLD,
):
    """Run the continuous Granger discovery loop.

    This function never returns unless interrupted (KeyboardInterrupt or
    SIGTERM). Each cycle:
      1. Load archive signals into domain time series
      2. Test every ordered domain pair for Granger causality
      3. Merge findings with prior results (keep strongest per pair)
      4. Write to granger_continuous.json
      5. Sleep, then repeat
    """
    logger = _setup_logging()
    logger.info(
        "Starting continuous Granger causality discovery | "
        "lookback=%dd max_lag=%d min_obs=%d sig=%.2f sleep=%ds",
        lookback_days, max_lag, min_obs, sig_threshold, sleep_seconds,
    )

    cycle = 0
    accumulated_findings: Dict[Tuple[str, str], ContinuousGrangerFinding] = (
        _load_existing_findings()
    )
    if accumulated_findings:
        logger.info(
            "Loaded %d existing findings from %s",
            len(accumulated_findings), _OUTPUT_PATH.name,
        )

    while True:
        cycle += 1
        cycle_start = time.monotonic()
        logger.info("=== Cycle %d starting ===", cycle)

        # ── Step 1: Load archive ───────────────────────────────────────────────
        try:
            domain_daily = _load_archive(lookback_days, logger)
        except Exception as exc:
            logger.error("Archive load failed: %s — sleeping and retrying", exc)
            time.sleep(sleep_seconds)
            continue

        if len(domain_daily) < 2:
            logger.warning(
                "Fewer than 2 domains found in archive — nothing to test. "
                "Archive path: %s", _SIGNALS_DIR
            )
            time.sleep(sleep_seconds)
            continue

        # ── Step 2: Build aligned domain series ───────────────────────────────
        domain_series = _build_domain_series(domain_daily, min_obs)
        domains = sorted(domain_series.keys())

        if len(domains) < 2:
            logger.warning(
                "Only %d domain(s) have >= %d observations: %s",
                len(domains), min_obs, domains,
            )
            time.sleep(sleep_seconds)
            continue

        # Signal count per domain (for metadata)
        signal_counts: Dict[str, int] = {
            domain: sum(len(v) for v in daily.items())
            for domain, daily in domain_daily.items()
        }

        logger.info(
            "Domains eligible for testing (%d): %s",
            len(domains), ", ".join(domains),
        )

        # ── Step 3: Test every ordered pair ───────────────────────────────────
        new_findings: List[ContinuousGrangerFinding] = []
        pairs_tested = 0
        pairs_significant = 0

        for i, domain_a in enumerate(domains):
            for domain_b in domains[i + 1:]:
                series_a = domain_series[domain_a]
                series_b = domain_series[domain_b]

                # Test both directions: A→B and B→A
                for cause, effect, s_cause, s_effect in [
                    (domain_a, domain_b, series_a, series_b),
                    (domain_b, domain_a, series_b, series_a),
                ]:
                    finding = _test_domain_pair(
                        cause_domain=cause,
                        effect_domain=effect,
                        cause_series=s_cause,
                        effect_series=s_effect,
                        min_obs=min_obs,
                        max_lag=max_lag,
                        sig_threshold=sig_threshold,
                        signal_counts=signal_counts,
                    )
                    pairs_tested += 1
                    if finding is not None:
                        new_findings.append(finding)
                        pairs_significant += 1
                        logger.info(
                            "  FOUND: %s → %s | lag=%d p=%.5f (corrected) "
                            "F=%.3f n=%d",
                            cause, effect,
                            finding.best_lag, finding.p_value,
                            finding.f_statistic, finding.n_observations,
                        )

        # ── Step 4: Merge and save ─────────────────────────────────────────────
        accumulated_findings = _merge_findings(accumulated_findings, new_findings)
        _save_findings(
            accumulated_findings,
            cycle=cycle,
            pairs_tested=pairs_tested,
            domains_active=len(domains),
            logger=logger,
        )

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "=== Cycle %d complete | %.1fs | tested=%d significant=%d "
            "total_accumulated=%d ===",
            cycle, elapsed, pairs_tested, pairs_significant,
            len(accumulated_findings),
        )

        # ── Step 5: Sleep ──────────────────────────────────────────────────────
        logger.info("Sleeping %ds before next cycle...", sleep_seconds)
        time.sleep(sleep_seconds)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Continuous domain-level Granger causality discovery for MIDGE. "
            "Tests every domain pair from the full signal archive. "
            "Runs indefinitely, writing results to data/market/granger_continuous.json."
        )
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=_LOOKBACK_DAYS,
        help=f"Days of signal history to use per cycle (default: {_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=_SLEEP_SECONDS,
        help=f"Seconds to sleep between cycles (default: {_SLEEP_SECONDS})",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=_MAX_LAG,
        help=f"Maximum lag in days to test (default: {_MAX_LAG})",
    )
    parser.add_argument(
        "--min-obs",
        type=int,
        default=_MIN_OBS,
        help=f"Minimum aligned observations required per domain pair (default: {_MIN_OBS})",
    )
    parser.add_argument(
        "--sig",
        type=float,
        default=_SIG_THRESHOLD,
        help=f"Significance threshold before Bonferroni correction (default: {_SIG_THRESHOLD})",
    )
    args = parser.parse_args()

    try:
        run_continuous(
            lookback_days=args.lookback,
            sleep_seconds=args.sleep,
            max_lag=args.max_lag,
            min_obs=args.min_obs,
            sig_threshold=args.sig,
        )
    except KeyboardInterrupt:
        print("\n[continuous_granger] Interrupted by user. Exiting.")
