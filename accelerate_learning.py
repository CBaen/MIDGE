#!/usr/bin/env python3
"""accelerate_learning.py — Feed MIDGE her own history.

Retroactively evaluates historical signals against historical prices,
accelerating Thompson Sampling and lag correlation without waiting for
organic maturation.

Three phases:
  1. EVALUATE: Register archived signals as predictions via OutcomeCollector
  2. RESOLVE:  Check matured predictions → price lookups → Thompson updates
  3. CORRELATE: Run lag-correlation analysis on the full archive

All existing infrastructure is reused. No new systems are created.

Usage:
    python accelerate_learning.py                      # All phases
    python accelerate_learning.py --phase evaluate      # Register signals only
    python accelerate_learning.py --phase resolve       # Resolve pending only
    python accelerate_learning.py --phase correlate     # Run lag analysis only
    python accelerate_learning.py --lookback 365        # Override lookback window
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midge.accelerate")

PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_ROOT / "data" / "midge" / "signals"
MARKET_DATA_DIR = PROJECT_ROOT / "data" / "market"


def phase_evaluate():
    """Phase 1: Register all archived signals as predictions."""
    logger.info("=" * 60)
    logger.info("Phase 1: EVALUATE — registering archived signals as predictions")
    logger.info("=" * 60)

    from mae_core.market.apis.price_fetcher import PriceFetcher
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    from mae_core.market.intelligence.outcome_collector import OutcomeCollector

    price_fetcher = PriceFetcher()
    thompson = ThompsonSampler()
    collector = OutcomeCollector(price_fetcher, thompson, data_dir=MARKET_DATA_DIR)

    before_stats = collector.get_statistics()
    logger.info("Before: %d registered signals, %d pending, %d evaluated",
                before_stats["registered_signals"],
                before_stats["pending_predictions"],
                before_stats["total_evaluated"])

    count = collector.collect_from_archives(SIGNALS_DIR)

    after_stats = collector.get_statistics()
    logger.info("After:  %d registered signals, %d pending, %d evaluated",
                after_stats["registered_signals"],
                after_stats["pending_predictions"],
                after_stats["total_evaluated"])
    logger.info("Newly registered: %d signals", count)

    # Thompson auto-saves on each update via OutcomeTracker
    return count


def phase_resolve():
    """Phase 2: Resolve matured predictions → price lookups → Thompson updates."""
    logger.info("=" * 60)
    logger.info("Phase 2: RESOLVE — evaluating matured predictions against prices")
    logger.info("=" * 60)

    from mae_core.market.apis.price_fetcher import PriceFetcher
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    from mae_core.market.intelligence.outcome_collector import OutcomeCollector

    price_fetcher = PriceFetcher()
    thompson = ThompsonSampler()
    collector = OutcomeCollector(price_fetcher, thompson, data_dir=MARKET_DATA_DIR)

    before_stats = collector.get_statistics()
    logger.info("Pending predictions: %d", before_stats["pending_predictions"])

    total_evaluated = 0
    batch = 0

    # Run evaluate in batches — each call processes what's matured
    while True:
        batch += 1
        evaluated = collector.evaluate()
        if evaluated == 0:
            break
        total_evaluated += evaluated
        logger.info("  Batch %d: evaluated %d predictions (%d total)",
                    batch, evaluated, total_evaluated)

    after_stats = collector.get_statistics()
    logger.info("Total evaluated this run: %d", total_evaluated)
    logger.info("Remaining pending: %d", after_stats["pending_predictions"])

    # Save Thompson state
    thompson.save()

    # Print Thompson summary
    _print_thompson_summary(thompson)
    return total_evaluated


def phase_correlate(lookback_days: int = 365):
    """Phase 3: Run lag-correlation analysis on the full archive."""
    logger.info("=" * 60)
    logger.info("Phase 3: CORRELATE — analyzing %d days of cross-domain lag patterns",
                lookback_days)
    logger.info("=" * 60)

    from mae_core.market.intelligence.signal_archive_reader import SignalArchiveReader
    from mae_core.market.intelligence.lag_correlation_analyzer import LagCorrelationAnalyzer

    reader = SignalArchiveReader(archive_dir=SIGNALS_DIR)
    analyzer = LagCorrelationAnalyzer(reader, min_observations=20)

    start_time = time.time()
    findings = analyzer.analyze(lookback_days=lookback_days)
    elapsed = time.time() - start_time

    logger.info("Analysis complete in %.1f seconds", elapsed)
    logger.info("Findings: %d significant lag correlations", len(findings))

    if findings:
        logger.info("")
        logger.info("Top findings:")
        # Sort by absolute correlation
        sorted_findings = sorted(findings, key=lambda f: abs(f.correlation), reverse=True)
        for f in sorted_findings[:15]:
            logger.info("  %s -> %s: r=%.3f, lag=%dd, n=%d pairs, %s",
                        f.source_a, f.source_b,
                        f.correlation, f.lag_days, f.n_pairs, f.direction)

    return findings


def _print_thompson_summary(thompson):
    """Print top Thompson distributions by sample count."""
    try:
        dists_path = MARKET_DATA_DIR / "thompson_distributions.json"
        if dists_path.exists():
            data = json.loads(dists_path.read_text())
            logger.info("")
            logger.info("Thompson Sampling distributions (top 15 by samples):")
            sorted_dists = sorted(
                data.items(),
                key=lambda x: x[1].get("alpha", 1) + x[1].get("beta", 1),
                reverse=True
            )
            for key, dist in sorted_dists[:15]:
                alpha = dist.get("alpha", 1)
                beta = dist.get("beta", 1)
                samples = alpha + beta - 2  # Subtract initial prior
                mean = alpha / (alpha + beta)
                logger.info("  %-30s  samples=%5.0f  mean=%.3f  alpha=%.1f  beta=%.1f",
                            key, samples, mean, alpha, beta)
    except Exception as e:
        logger.warning("Could not print Thompson summary: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Accelerate MIDGE's learning from historical data")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["evaluate", "resolve", "correlate", "all"],
                        help="Which phase to run (default: all)")
    parser.add_argument("--lookback", type=int, default=365,
                        help="Lookback days for lag correlation (default: 365)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MIDGE Learning Acceleration Pipeline")
    logger.info("Phase: %s | Lookback: %d days", args.phase, args.lookback)
    logger.info("=" * 60)

    start_time = time.time()

    if args.phase in ("evaluate", "all"):
        registered = phase_evaluate()
        logger.info("")

    if args.phase in ("resolve", "all"):
        evaluated = phase_resolve()
        logger.info("")

    if args.phase in ("correlate", "all"):
        findings = phase_correlate(lookback_days=args.lookback)
        logger.info("")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
