"""crypto_trader.py — Standalone crypto trading loop.

No Mesa. No agents. No LLM. No Ollama. No 33-layer bootstrap.
Just: fetch prices → evaluate 31 strategies → trade if 3+ agree → repeat.

Usage:
    python -m mae_core.market.strategies.crypto_trader
    python -m mae_core.market.strategies.crypto_trader --dry-run
    python -m mae_core.market.strategies.crypto_trader --interval 300

This is the script that makes money.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRADER] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("midge.crypto_trader")

# Paths
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
WATCHLIST_PATH = DATA_DIR / "market" / "crypto_watchlist.json"
TRADES_PATH = DATA_DIR / "midge" / "paper_trades.jsonl"
TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)

# Graceful shutdown
_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    logger.info("Shutdown signal received — finishing current cycle")
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def load_watchlist() -> list[str]:
    """Load crypto symbols from watchlist JSON."""
    if not WATCHLIST_PATH.exists():
        return ["BTC-USD", "ETH-USD", "SOL-USD"]
    return json.loads(WATCHLIST_PATH.read_text())


def evaluate_symbol(symbol: str, registry, all_strategies) -> Optional[dict]:
    """Run all validated strategies on a symbol. Return trade dict if 3+ agree."""
    from mae_core.market.strategies.crypto_ohlcv import get_ohlcv

    df = get_ohlcv(symbol, days=90)
    if df is None or len(df) < 50:
        return None

    validated_names = set(registry.get_validated_strategies(symbol))
    if len(validated_names) < 3:
        logger.debug("%s: only %d validated strategies — skipping", symbol, len(validated_names))
        return None

    # Run all validated strategies
    bullish = []
    bearish = []
    for name, fn in all_strategies:
        if name not in validated_names:
            continue
        try:
            result = fn(symbol, df)
            if result is None:
                continue
            if result.direction == "bullish" and result.signal == 1:
                bullish.append(result)
            elif result.direction == "bearish" and result.signal == -1:
                bearish.append(result)
        except Exception:
            pass  # Strategy error — skip silently

    # Check for convergence (3+ agree)
    for direction, results in [("bullish", bullish), ("bearish", bearish)]:
        if len(results) >= 3:
            # Aggregate
            names = [r.strategy_name for r in results]
            avg_strength = sum(r.strength for r in results) / len(results)
            avg_confidence = sum(r.confidence for r in results) / len(results)

            # Conservative stops: widest SL, nearest TP
            if direction == "bullish":
                sl = min(r.stop_loss for r in results if r.stop_loss > 0) if any(r.stop_loss > 0 for r in results) else 0
                tp = min(r.take_profit for r in results if r.take_profit > 0) if any(r.take_profit > 0 for r in results) else 0
            else:
                sl = max(r.stop_loss for r in results if r.stop_loss > 0) if any(r.stop_loss > 0 for r in results) else 0
                tp = max(r.take_profit for r in results if r.take_profit > 0) if any(r.take_profit > 0 for r in results) else 0

            last_close = float(df["Close"].iloc[-1])

            return {
                "symbol": symbol,
                "direction": direction,
                "side": "buy" if direction == "bullish" else "sell",
                "strategies": names,
                "strategy_count": len(results),
                "strength": round(avg_strength, 3),
                "confidence": round(avg_confidence, 3),
                "entry_price": last_close,
                "stop_loss": round(sl, 2),
                "take_profit": round(tp, 2),
                "timestamp": datetime.now().isoformat(),
            }

    return None


def write_trade(trade: dict) -> None:
    """Append trade to paper_trades.jsonl."""
    with open(TRADES_PATH, "a") as f:
        f.write(json.dumps(trade) + "\n")


def submit_to_alpaca(trade: dict, dry_run: bool = False) -> bool:
    """Submit a paper trade to Alpaca. Returns True if successful."""
    if dry_run:
        logger.info("DRY RUN: would trade %s %s (%d strategies agree)",
                     trade["side"].upper(), trade["symbol"], trade["strategy_count"])
        return True

    try:
        from mae_core.market.apis.alpaca_client import AlpacaClient
        client = AlpacaClient()
        if not client.connected:
            logger.warning("Alpaca not connected — skipping trade")
            return False

        symbol = trade["symbol"]
        # Convert to Alpaca format: BTC-USD → BTC/USD
        base = symbol.replace("-USD", "")
        alpaca_symbol = f"{base}/USD"

        # Check if already holding
        positions = client.get_positions()
        if any(p.symbol == alpaca_symbol for p in positions):
            logger.info("Already holding %s — skipping", alpaca_symbol)
            return False

        # Position sizing: 10% of equity
        account = client.get_account()
        if account is None:
            return False
        dollar_amount = account.equity * 0.10
        if dollar_amount < 1.0:
            return False

        qty = round(dollar_amount / trade["entry_price"], 6)
        if qty <= 0:
            return False

        # Simple market order (no bracket for crypto)
        result = client.submit_market_order(
            symbol=alpaca_symbol,
            qty=qty,
            side=trade["side"],
            metadata={
                "source": "crypto_trader",
                "strategies": trade["strategies"],
                "confidence": trade["confidence"],
            },
        )

        if result:
            logger.info(
                "ALPACA TRADE: %s %.6f %s @ ~$%.2f | %d strategies: %s",
                trade["side"].upper(), qty, alpaca_symbol, trade["entry_price"],
                trade["strategy_count"], ", ".join(trade["strategies"]),
            )
            return True
    except Exception as e:
        logger.error("Alpaca submission failed: %s", e)
    return False


def run_cycle(registry, all_strategies, dry_run: bool = False) -> int:
    """Run one evaluation cycle across all crypto symbols. Returns trade count."""
    symbols = load_watchlist()
    trades = 0

    for symbol in symbols:
        try:
            trade = evaluate_symbol(symbol, registry, all_strategies)
            if trade:
                logger.info(
                    "CONVERGENCE: %s %s — %d strategies agree: %s",
                    trade["direction"].upper(), trade["symbol"],
                    trade["strategy_count"], ", ".join(trade["strategies"]),
                )
                write_trade(trade)
                if submit_to_alpaca(trade, dry_run=dry_run):
                    trades += 1
        except Exception as e:
            logger.debug("Error evaluating %s: %s", symbol, e)

    return trades


def main():
    parser = argparse.ArgumentParser(description="MIDGE Standalone Crypto Trader")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between evaluation cycles (default: 300 = 5 min)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log trades without submitting to Alpaca")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    args = parser.parse_args()

    # Load strategy library and registry
    from mae_core.market.strategies.strategy_library import ALL_STRATEGIES
    from mae_core.market.strategies.strategy_registry import StrategyRegistry

    registry = StrategyRegistry()
    stats = registry.get_statistics()

    logger.info("MIDGE Crypto Trader starting")
    logger.info("  Strategies: %d loaded, %d validated", len(ALL_STRATEGIES), stats["validated_records"])
    logger.info("  Symbols: %s", ", ".join(load_watchlist()))
    logger.info("  Interval: %ds | Dry run: %s", args.interval, args.dry_run)
    logger.info("  Alpaca: %s", "DRY RUN" if args.dry_run else "PAPER TRADING")

    cycle = 0
    total_trades = 0

    while _RUNNING:
        cycle += 1
        start = time.time()

        logger.info("--- Cycle %d ---", cycle)
        trades = run_cycle(registry, ALL_STRATEGIES, dry_run=args.dry_run)
        total_trades += trades
        elapsed = time.time() - start

        logger.info("Cycle %d complete: %d trades (%.1fs) | Total: %d trades",
                     cycle, trades, elapsed, total_trades)

        if args.once:
            break

        # Sleep until next cycle
        sleep_time = max(1, args.interval - elapsed)
        logger.info("Next cycle in %ds...", int(sleep_time))
        # Sleep in small increments so shutdown signal is responsive
        for _ in range(int(sleep_time)):
            if not _RUNNING:
                break
            time.sleep(1)

    logger.info("MIDGE Crypto Trader stopped. Total trades: %d across %d cycles", total_trades, cycle)


if __name__ == "__main__":
    main()
