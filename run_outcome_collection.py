#!/usr/bin/env python3
"""
run_outcome_collection.py - One-shot script to close Thompson's learning loop.

Registers all archived signals as predictions (preserving original timestamps),
then evaluates every matured prediction against actual price data via yfinance.
Thompson Sampler distributions update in real-time as outcomes resolve.

Safe to run multiple times — deduplicates via registered_signals.json.
"""

import logging
import sys
from pathlib import Path

# Project root on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("outcome_collection")

# Suppress noisy yfinance/urllib3 output
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)


def main():
    from mae_core.market.apis.price_fetcher import PriceFetcher
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    from mae_core.market.intelligence.outcome_collector import OutcomeCollector

    signals_dir = Path("data/midge/signals")
    data_dir = Path("data/market")

    if not signals_dir.exists():
        logger.error("Signal archive not found at %s", signals_dir)
        return

    print("=" * 60)
    print("MIDGE Outcome Collection")
    print("=" * 60)

    # Initialize dependencies
    print("\nInitializing components...")
    price_fetcher = PriceFetcher()
    thompson = ThompsonSampler(persistence_path=data_dir / "thompson_distributions.json")

    print(f"  Thompson distributions loaded: {len(thompson.distributions)} sources")
    print(f"  Signal archive: {signals_dir}")

    # Create collector
    collector = OutcomeCollector(
        price_fetcher=price_fetcher,
        thompson_sampler=thompson,
        regime_classifier=None,
        data_dir=data_dir,
    )

    # Phase 1: Register archive signals as predictions
    print(f"\nPhase 1: Registering archive signals as predictions...")
    pre_stats = collector.get_statistics()
    print(f"  Already registered: {pre_stats['registered_signals']}")
    print(f"  Pending predictions: {pre_stats['pending_predictions']}")

    registered = collector.collect_from_archives(signals_dir)
    print(f"  Newly registered: {registered}")

    post_stats = collector.get_statistics()
    print(f"  Total registered: {post_stats['registered_signals']}")
    print(f"  Total pending: {post_stats['pending_predictions']}")

    # Phase 2: Evaluate matured predictions
    print(f"\nPhase 2: Evaluating matured predictions (this calls yfinance)...")
    print("  This may take a while — one price lookup per prediction pair...")
    evaluated = collector.evaluate()

    final_stats = collector.get_statistics()
    print(f"\n  Outcomes evaluated this run: {evaluated}")
    print(f"  Total outcomes (all time): {final_stats['total_evaluated']}")
    print(f"  Remaining pending: {final_stats['pending_predictions']}")

    # Phase 3: Show updated Thompson distributions
    print(f"\nPhase 3: Thompson distribution state after updates:")
    for source_key, regimes in sorted(thompson.distributions.items()):
        default = regimes.get("default", {})
        alpha = default.get("alpha", 1.0)
        beta = default.get("beta", 1.0)
        mean = alpha / (alpha + beta)
        total_obs = alpha + beta - 2.0  # subtract the 2 priors
        if total_obs > 0.1:  # only show sources with observations
            print(f"  {source_key:<25} alpha={alpha:.3f}  beta={beta:.3f}  "
                  f"mean={mean:.3f}  obs~{total_obs:.0f}")

    print(f"\n{'=' * 60}")
    print(f"Done. Thompson distributions saved to {data_dir / 'thompson_distributions.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
