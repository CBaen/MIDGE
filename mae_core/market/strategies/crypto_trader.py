"""crypto_trader.py — Standalone crypto trading loop.

No Mesa. No agents. No LLM. No Ollama. No 33-layer bootstrap.
Just: fetch prices -> evaluate 31 strategies -> trade if 2+ agree -> manage exits -> repeat.

Usage:
    python -m mae_core.market.strategies.crypto_trader
    python -m mae_core.market.strategies.crypto_trader --dry-run
    python -m mae_core.market.strategies.crypto_trader --timeframe 5m --interval 60

This is the script that makes money.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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
POSITIONS_PATH = DATA_DIR / "market" / "open_positions.json"
TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)

# Position management
MAX_POSITION_PCT = 0.05       # 5% of equity per trade (was 10%)
MAX_TOTAL_EXPOSURE_PCT = 0.30  # 30% of equity in crypto total
MAX_HOLD_HOURS = 24            # Close stale positions after 24 hours
STALE_MOVE_PCT = 0.01          # Less than 1% move = stale

# Trading
MIN_STRATEGIES = 2
MIN_MOVE_PCT = 2.0  # Minimum expected price move to overcome 0.5% round-trip fees

# Graceful shutdown
_RUNNING = True


def _handle_signal(signum, frame):
    global _RUNNING
    logger.info("Shutdown signal received — finishing current cycle")
    _RUNNING = False


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Watchlist ────────────────────────────────────────────────────────────────

def load_watchlist() -> list[str]:
    """Load crypto symbols from watchlist JSON."""
    if not WATCHLIST_PATH.exists():
        return ["BTC-USD", "ETH-USD", "SOL-USD"]
    return json.loads(WATCHLIST_PATH.read_text())


# ── Position tracking ────────────────────────────────────────────────────────

def load_tracked_positions() -> dict:
    """Load tracked position data from disk."""
    if POSITIONS_PATH.exists():
        try:
            return json.loads(POSITIONS_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def save_tracked_positions(tracked: dict) -> None:
    """Persist tracked position data to disk."""
    POSITIONS_PATH.write_text(json.dumps(tracked, indent=2))


def record_entry(tracked: dict, trade: dict, qty: float) -> None:
    """Record a new position entry for exit management."""
    alpaca_symbol = trade["symbol"].replace("-USD", "") + "/USD"
    tracked[alpaca_symbol] = {
        "entry_price": trade["entry_price"],
        "stop_loss": trade["stop_loss"],
        "take_profit": trade["take_profit"],
        "direction": trade["direction"],
        "strategies": trade["strategies"],
        "entry_time": trade["timestamp"],
        "qty": qty,
    }
    save_tracked_positions(tracked)


# ── Alpaca client ────────────────────────────────────────────────────────────

def create_alpaca_client():
    """Create a persistent Alpaca client. Returns None if unavailable."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from mae_core.market.apis.alpaca_client import AlpacaClient
    client = AlpacaClient(
        api_key=os.environ.get("ALPACA_API_KEY"),
        secret_key=os.environ.get("ALPACA_SECRET_KEY"),
        paper=True,
    )
    return client if client.connected else None


# ── Exit management ──────────────────────────────────────────────────────────

def check_exits(client, tracked: dict, registry, all_strategies,
                timeframe: str, cycle: int) -> int:
    """Check all open positions for exit conditions. Returns exit count."""
    if client is None:
        return 0

    positions = client.get_positions()
    if not positions:
        return 0

    exits = 0

    for pos in positions:
        # Only manage crypto positions (format: BTC/USD)
        if "/" not in pos.symbol:
            continue

        entry = tracked.get(pos.symbol)
        if not entry:
            # Position on Alpaca but not tracked — add with defaults so we can manage it
            logger.info("Untracked position found: %s — adding with defaults", pos.symbol)
            tracked[pos.symbol] = {
                "entry_price": pos.avg_entry_price,
                "stop_loss": 0,
                "take_profit": 0,
                "direction": "bullish",
                "strategies": [],
                "entry_time": datetime.now().isoformat(),
                "qty": pos.qty,
            }
            entry = tracked[pos.symbol]

        # Current price from Alpaca position data
        current_price = pos.market_value / pos.qty if pos.qty > 0 else 0
        if current_price <= 0:
            continue

        exit_reason = None

        # Exit 1: Stop-loss hit
        sl = entry.get("stop_loss", 0)
        if sl > 0 and current_price <= sl:
            exit_reason = f"STOP-LOSS (${current_price:.2f} <= ${sl:.2f})"

        # Exit 2: Take-profit hit
        tp = entry.get("take_profit", 0)
        if exit_reason is None and tp > 0 and current_price >= tp:
            exit_reason = f"TAKE-PROFIT (${current_price:.2f} >= ${tp:.2f})"

        # Exit 3: Strategy reversal — check every 5 cycles (~5 min) to save API calls
        if exit_reason is None and cycle % 5 == 0:
            symbol_yf = pos.symbol.replace("/", "-")  # BTC/USD -> BTC-USD
            try:
                from mae_core.market.strategies.crypto_ohlcv import get_ohlcv
                df = get_ohlcv(symbol_yf, days=7, interval=timeframe)
                if df is not None and len(df) >= 50:
                    validated = set(registry.get_validated_strategies(symbol_yf))
                    bearish_count = 0
                    bullish_count = 0
                    for name, fn in all_strategies:
                        if name not in validated:
                            continue
                        try:
                            result = fn(symbol_yf, df)
                            if result and result.signal == -1:
                                bearish_count += 1
                            elif result and result.signal == 1:
                                bullish_count += 1
                        except Exception:
                            pass
                    if bearish_count >= 2 and bearish_count > bullish_count:
                        exit_reason = f"REVERSAL ({bearish_count} bearish vs {bullish_count} bullish)"
            except Exception:
                logger.debug("Strategy reversal check failed for %s", pos.symbol, exc_info=True)

        # Exit 4: Stale position — held too long with no movement
        if exit_reason is None:
            entry_time_str = entry.get("entry_time", "")
            if entry_time_str:
                try:
                    entry_dt = datetime.fromisoformat(entry_time_str)
                    hours_held = (datetime.now() - entry_dt).total_seconds() / 3600
                    entry_price = entry.get("entry_price", 0)
                    if hours_held > MAX_HOLD_HOURS and entry_price > 0:
                        pct_move = abs(current_price - entry_price) / entry_price
                        if pct_move < STALE_MOVE_PCT:
                            exit_reason = f"STALE ({hours_held:.1f}h, {pct_move*100:.2f}% move)"
                except (ValueError, TypeError):
                    pass

        # Execute exit
        if exit_reason:
            pl_str = f"${pos.unrealized_pl:+.2f} ({pos.unrealized_plpc*100:+.1f}%)"
            logger.info("EXIT %s: %s | P&L: %s", pos.symbol, exit_reason, pl_str)
            if client.close_position(pos.symbol):
                tracked.pop(pos.symbol, None)
                exits += 1
                # Log exit to trades file
                exit_record = {
                    "symbol": pos.symbol,
                    "action": "EXIT",
                    "reason": exit_reason,
                    "entry_price": entry.get("entry_price", 0),
                    "exit_price": current_price,
                    "unrealized_pl": pos.unrealized_pl,
                    "unrealized_plpc": pos.unrealized_plpc,
                    "hours_held": 0,
                    "timestamp": datetime.now().isoformat(),
                }
                if entry.get("entry_time"):
                    try:
                        h = (datetime.now() - datetime.fromisoformat(entry["entry_time"])).total_seconds() / 3600
                        exit_record["hours_held"] = round(h, 2)
                    except (ValueError, TypeError):
                        pass
                write_trade(exit_record)

    if exits > 0:
        save_tracked_positions(tracked)

    return exits


# ── Strategy evaluation ──────────────────────────────────────────────────────

def evaluate_symbol(symbol: str, registry, all_strategies,
                    interval: str = "5m", lookback_days: int = 7,
                    scorer=None) -> Optional[dict]:
    """Run all validated strategies on a symbol. Return trade dict if 2+ agree."""
    from mae_core.market.strategies.crypto_ohlcv import get_ohlcv

    df = get_ohlcv(symbol, days=lookback_days, interval=interval)
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

    # Only enter bullish trades (can't short crypto on Alpaca)
    if len(bullish) >= MIN_STRATEGIES:
        results = bullish
        direction = "bullish"

        names = [r.strategy_name for r in results]
        avg_strength = sum(r.strength for r in results) / len(results)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        # Conservative stops: widest SL, nearest TP
        sl = min(r.stop_loss for r in results if r.stop_loss > 0) if any(r.stop_loss > 0 for r in results) else 0
        tp = min(r.take_profit for r in results if r.take_profit > 0) if any(r.take_profit > 0 for r in results) else 0

        last_close = float(df["Close"].iloc[-1])

        # Fee filter: skip trades where expected move < 2%
        if tp > 0 and last_close > 0:
            expected_move_pct = abs(tp - last_close) / last_close * 100
            if expected_move_pct < MIN_MOVE_PCT:
                logger.debug("%s: expected move %.1f%% < %.1f%% minimum — skipping",
                             symbol, expected_move_pct, MIN_MOVE_PCT)
                return None

        # Forensic scoring — boost confidence if this combo recently won
        combo_wr, is_hot = (0.5, False)
        if scorer is not None:
            combo_wr, is_hot = scorer.score_convergence(names)
            if is_hot:
                avg_confidence = min(0.95, avg_confidence * 1.3)

        return {
            "symbol": symbol,
            "direction": direction,
            "side": "buy",
            "strategies": names,
            "strategy_count": len(results),
            "strength": round(avg_strength, 3),
            "confidence": round(avg_confidence, 3),
            "forensic_wr": combo_wr,
            "forensic_hot": is_hot,
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


# ── Order submission ─────────────────────────────────────────────────────────

def submit_to_alpaca(client, trade: dict, tracked: dict,
                     dry_run: bool = False) -> bool:
    """Submit a paper trade to Alpaca with position sizing limits."""
    if dry_run:
        logger.info("DRY RUN: would BUY %s (%d strategies agree)",
                     trade["symbol"], trade["strategy_count"])
        return True

    if client is None:
        logger.warning("Alpaca not connected — skipping trade")
        return False

    try:
        symbol = trade["symbol"]
        base = symbol.replace("-USD", "")
        alpaca_symbol = f"{base}/USD"

        # Skip if already holding this symbol
        positions = client.get_positions()
        if any(p.symbol == alpaca_symbol for p in positions):
            logger.info("Already holding %s — skipping", alpaca_symbol)
            return False

        # Account check
        account = client.get_account()
        if account is None:
            return False

        # Position sizing: 5% of equity per trade
        dollar_amount = account.equity * MAX_POSITION_PCT

        # Total exposure check: don't exceed 30% of equity in crypto
        crypto_exposure = sum(
            p.market_value for p in positions if "/" in p.symbol
        )
        remaining_budget = (account.equity * MAX_TOTAL_EXPOSURE_PCT) - crypto_exposure
        if remaining_budget <= 0:
            logger.info("Total crypto exposure at %.0f%% — no more trades",
                        (crypto_exposure / account.equity * 100) if account.equity > 0 else 0)
            return False

        # Don't exceed remaining budget
        dollar_amount = min(dollar_amount, remaining_budget)

        # Cash check: don't trade if cash is negative or insufficient
        if account.cash < dollar_amount:
            if account.cash <= 0:
                logger.info("Cash is $%.2f — cannot trade", account.cash)
                return False
            dollar_amount = account.cash * 0.9  # Use 90% of remaining cash

        if dollar_amount < 1.0:
            return False

        qty = round(dollar_amount / trade["entry_price"], 6)
        if qty <= 0:
            return False

        # Simple market order (no bracket for crypto on Alpaca)
        result = client.submit_market_order(
            symbol=alpaca_symbol,
            qty=qty,
            side="buy",
            metadata={
                "source": "crypto_trader",
                "strategies": trade["strategies"],
                "confidence": trade["confidence"],
            },
        )

        if result:
            logger.info(
                "TRADE: BUY %.6f %s @ ~$%.2f ($%.2f) | %d strategies: %s",
                qty, alpaca_symbol, trade["entry_price"], dollar_amount,
                trade["strategy_count"], ", ".join(trade["strategies"]),
            )
            # Record for exit management
            record_entry(tracked, trade, qty)
            return True

    except Exception as e:
        logger.error("Alpaca submission failed: %s", e)
    return False


# ── Main loop ────────────────────────────────────────────────────────────────

def run_cycle(client, tracked: dict, registry, all_strategies,
              dry_run: bool = False, timeframe: str = "5m",
              scorer=None, cycle: int = 0) -> tuple[int, int]:
    """Run one evaluation cycle. Returns (entries, exits)."""
    # Check exits FIRST — protect capital before seeking new trades
    exits = check_exits(client, tracked, registry, all_strategies, timeframe, cycle)

    # Then evaluate for new entries
    symbols = load_watchlist()
    entries = 0

    for symbol in symbols:
        try:
            trade = evaluate_symbol(symbol, registry, all_strategies,
                                    interval=timeframe, lookback_days=7,
                                    scorer=scorer)
            if trade:
                hot_tag = " [HOT COMBO]" if trade.get("forensic_hot") else ""
                logger.info(
                    "CONVERGENCE: %s %s — %d strategies agree: %s | forensic WR=%.0f%%%s",
                    trade["direction"].upper(), trade["symbol"],
                    trade["strategy_count"], ", ".join(trade["strategies"]),
                    trade.get("forensic_wr", 0.5) * 100, hot_tag,
                )
                write_trade(trade)
                if submit_to_alpaca(client, trade, tracked, dry_run=dry_run):
                    entries += 1
        except Exception as e:
            logger.debug("Error evaluating %s: %s", symbol, e)

    return entries, exits


def main():
    parser = argparse.ArgumentParser(description="MIDGE Standalone Crypto Trader")
    parser.add_argument("--interval", type=int, default=60,
                        help="Seconds between evaluation cycles (default: 60)")
    parser.add_argument("--timeframe", type=str, default="5m",
                        help="Bar interval for strategies (default: 5m)")
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

    # Forensic scorer — learns which strategy combos win on recent data
    from mae_core.market.strategies.forensic_scorer import ForensicScorer
    scorer = ForensicScorer()

    # Persistent Alpaca client
    client = create_alpaca_client() if not args.dry_run else None

    # Load tracked positions (survives restarts)
    tracked = load_tracked_positions()

    logger.info("MIDGE Crypto Trader starting")
    logger.info("  Strategies: %d loaded, %d validated", len(ALL_STRATEGIES), stats["validated_records"])
    logger.info("  Symbols: %s", ", ".join(load_watchlist()))
    logger.info("  Timeframe: %s | Cycle: %ds | Dry run: %s",
                args.timeframe, args.interval, args.dry_run)
    logger.info("  Position limits: %.0f%% per trade, %.0f%% total exposure",
                MAX_POSITION_PCT * 100, MAX_TOTAL_EXPOSURE_PCT * 100)
    logger.info("  Exit rules: SL/TP from strategies, reversal check every 5 cycles, stale after %dh",
                MAX_HOLD_HOURS)
    logger.info("  Tracked positions: %d", len(tracked))
    logger.info("  Forensic scorer: %d combos loaded", len(scorer._scorecard))

    if client:
        account = client.get_account()
        if account:
            positions = client.get_positions()
            crypto_positions = [p for p in positions if "/" in p.symbol]
            crypto_value = sum(p.market_value for p in crypto_positions)
            logger.info("  Account: $%.2f equity, $%.2f cash, %d crypto positions ($%.2f)",
                        account.equity, account.cash, len(crypto_positions), crypto_value)

    cycle = 0
    total_entries = 0
    total_exits = 0

    while _RUNNING:
        cycle += 1
        start = time.time()

        # Refresh forensic scorecard every 10 minutes
        if scorer.needs_refresh():
            logger.info("Forensic refresh starting...")
            try:
                scorer.refresh(load_watchlist()[:5], ALL_STRATEGIES)
                hot = scorer.get_hot_combos()
                if hot:
                    logger.info("HOT combos: %s", ", ".join(
                        f"{c['combo']} ({c['win_rate']*100:.0f}%)" for c in hot[:5]
                    ))
            except Exception:
                logger.debug("Forensic refresh failed", exc_info=True)

        logger.info("--- Cycle %d ---", cycle)
        entries, exits = run_cycle(
            client, tracked, registry, ALL_STRATEGIES,
            dry_run=args.dry_run, timeframe=args.timeframe,
            scorer=scorer, cycle=cycle,
        )
        total_entries += entries
        total_exits += exits
        elapsed = time.time() - start

        logger.info("Cycle %d: %d entries, %d exits (%.1fs) | Total: %d entries, %d exits",
                     cycle, entries, exits, elapsed, total_entries, total_exits)

        if args.once:
            break

        # Sleep until next cycle
        sleep_time = max(1, args.interval - elapsed)
        logger.info("Next cycle in %ds...", int(sleep_time))
        for _ in range(int(sleep_time)):
            if not _RUNNING:
                break
            time.sleep(1)

    # Save tracked positions on shutdown
    save_tracked_positions(tracked)
    logger.info("MIDGE Crypto Trader stopped. %d entries, %d exits across %d cycles",
                total_entries, total_exits, cycle)


if __name__ == "__main__":
    main()
