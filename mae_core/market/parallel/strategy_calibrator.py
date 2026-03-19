"""
strategy_calibrator.py — Offline backtest runner for all strategies.

Backtests every strategy in strategy_library.ALL_STRATEGIES against every
symbol in data/market/crypto_watchlist.json.  Results are persisted to
data/market/strategy_registry.json via StrategyRegistry.

Runs are skipped when a backtest record already exists for a (strategy, symbol)
pair, so the script is safe to re-run — it only does new work.

Usage
-----
    python -m mae_core.market.parallel.strategy_calibrator
    python -m mae_core.market.parallel.strategy_calibrator --symbol BTC-USD
    python -m mae_core.market.parallel.strategy_calibrator --strategy rsi_oversold_14
    python -m mae_core.market.parallel.strategy_calibrator --days 365

Output
------
    data/market/strategy_registry.json  — backtest records for all tested pairs

A 1-second sleep is inserted between symbol fetches to be gentle with yfinance
rate limits.  Execution is sequential (no threading) for the same reason.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the script works from any cwd
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]   # mae_core/market/parallel/… -> MIDGE root
_CRYPTO_WATCHLIST = _PROJECT_ROOT / "data" / "market" / "crypto_watchlist.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CALIBRATOR] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("midge.market.strategies.calibrator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_watchlist(path: Path, symbol_filter: Optional[str]) -> list[str]:
    """Load the crypto watchlist JSON.  Optionally filter to a single symbol."""
    if not path.exists():
        logger.error("Crypto watchlist not found at %s", path)
        sys.exit(1)
    try:
        symbols = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load watchlist: %s", exc)
        sys.exit(1)
    if not isinstance(symbols, list) or not symbols:
        logger.error("Watchlist is empty or not a list: %s", path)
        sys.exit(1)
    if symbol_filter:
        symbols = [s for s in symbols if s.upper() == symbol_filter.upper()]
        if not symbols:
            logger.error("Symbol %s not found in watchlist", symbol_filter)
            sys.exit(1)
    return symbols


def _print_summary(registry, strategies: list[str], symbols: list[str]) -> None:
    """Print a formatted summary table of all backtest results."""
    print()
    print("=" * 72)
    print(f"{'STRATEGY':<30} {'SYMBOL':<12} {'WIN RATE':>9} {'TRADES':>7} {'STATUS':<12}")
    print("-" * 72)

    total = 0
    validated_count = 0

    for strategy_name in strategies:
        for symbol in symbols:
            if registry.needs_backtest(strategy_name, symbol):
                # No record (skipped due to filter or error)
                continue
            # Fetch the stored record for display
            key = f"{strategy_name}:{symbol}"
            record = registry._records.get(key)  # noqa: SLF001 — internal access for display
            if record is None:
                continue
            total += 1
            status = "VALIDATED" if record.validated else "not validated"
            if record.validated:
                validated_count += 1
            print(
                f"{strategy_name:<30} {symbol:<12} "
                f"{record.win_rate:>8.1%} {record.total_trades:>7}  {status:<12}"
            )

    print("=" * 72)
    print(
        f"Total tested: {total}  |  Validated: {validated_count}  |  "
        f"Validation rate: {validated_count / total:.1%}" if total else
        "No results recorded."
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MIDGE Strategy Calibrator — backtests all strategies against all "
            "crypto symbols and writes results to strategy_registry.json."
        )
    )
    parser.add_argument(
        "--symbol",
        metavar="TICKER",
        default=None,
        help="Limit run to a single symbol (e.g. BTC-USD).",
    )
    parser.add_argument(
        "--strategy",
        metavar="NAME",
        default=None,
        help="Limit run to a single strategy by name (e.g. rsi_oversold_14).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Calendar days of history to backtest (default: 730 = 2 years).",
    )
    args = parser.parse_args()

    # -- Imports (deferred so argparse --help works without heavy imports) --
    from mae_core.market.strategies.strategy_library import ALL_STRATEGIES
    from mae_core.market.strategies.strategy_registry import StrategyRegistry
    from mae_core.market.strategies.strategy_backtester import StrategyBacktester

    # -- Setup --
    symbols = _load_watchlist(_CRYPTO_WATCHLIST, args.symbol)
    registry = StrategyRegistry()
    backtester = StrategyBacktester()

    # Filter strategies if requested
    if args.strategy:
        strategies = [(n, fn) for n, fn in ALL_STRATEGIES if n == args.strategy]
        if not strategies:
            logger.error(
                "Strategy '%s' not found in ALL_STRATEGIES.  Valid names:\n  %s",
                args.strategy,
                "\n  ".join(n for n, _ in ALL_STRATEGIES),
            )
            sys.exit(1)
    else:
        strategies = list(ALL_STRATEGIES)

    total_pairs = len(strategies) * len(symbols)
    logger.info(
        "Calibrator starting: %d strategies x %d symbols = %d pairs | days=%d",
        len(strategies), len(symbols), total_pairs, args.days,
    )

    stats = registry.get_statistics()
    logger.info(
        "Registry state: %d existing records, %d validated",
        stats["total_records"], stats["validated_records"],
    )

    # -- Main loop --
    done = 0
    new_records = 0
    errors = 0

    for strategy_name, strategy_fn in strategies:
        for idx, symbol in enumerate(symbols):
            done += 1
            pair_label = f"{strategy_name}:{symbol}"

            if not registry.needs_backtest(strategy_name, symbol):
                logger.debug("[%d/%d] %s — already in registry, skipping", done, total_pairs, pair_label)
                # Still sleep briefly between skips to avoid hammering the filesystem
                continue

            logger.info("[%d/%d] Backtesting %s ...", done, total_pairs, pair_label)

            try:
                record = backtester.backtest(
                    strategy_fn=strategy_fn,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    days=args.days,
                )
            except Exception as exc:
                logger.error("[%d/%d] %s — unhandled error: %s", done, total_pairs, pair_label, exc)
                errors += 1
                time.sleep(1)
                continue

            if record is None:
                logger.info(
                    "[%d/%d] %s — no result (data unavailable or too few trades)",
                    done, total_pairs, pair_label,
                )
            else:
                registry.register_result(record)
                new_records += 1
                status = "VALIDATED" if record.validated else "not validated"
                logger.info(
                    "[%d/%d] %s — win_rate=%.1%% trades=%d sharpe=%.2f [%s]",
                    done, total_pairs, pair_label,
                    record.win_rate, record.total_trades, record.sharpe_ratio, status,
                )

            # Rate-limit: sleep between symbol fetches to be gentle with yfinance
            if idx < len(symbols) - 1:
                time.sleep(1)

    # -- Summary --
    logger.info(
        "Calibration complete. New records: %d | Errors: %d",
        new_records, errors,
    )
    all_strategy_names = [n for n, _ in strategies]
    _print_summary(registry, all_strategy_names, symbols)

    final_stats = registry.get_statistics()
    logger.info(
        "Registry final state: %d total records, %d validated",
        final_stats["total_records"], final_stats["validated_records"],
    )


if __name__ == "__main__":
    main()
