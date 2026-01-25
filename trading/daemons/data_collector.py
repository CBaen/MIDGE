#!/usr/bin/env python3
"""
data_collector.py - Continuous Data Collection for MIDGE

Fetches price data, generates signals, and stores to Qdrant.
Runs continuously to keep the signal database fresh.

Usage:
    # Run once
    python trading/daemons/data_collector.py --once

    # Run continuously (every 5 minutes)
    python trading/daemons/data_collector.py --continuous --interval 300

    # Check status
    python trading/daemons/data_collector.py --status
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading.apis.price_fetcher import PriceFetcher, PriceData
from trading.technical.signals import SignalGenerator, TradingSignal, SignalDirection, SignalStrength
from trading.storage import TradingVectorStore, SignalPayload, DECAY_RATES

# Configuration
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "LMT", "BA", "RTX", "NVDA", "SPY", "QQQ"]
STATE_FILE = PROJECT_ROOT / ".claude" / "data_collector_state.json"
LOG_FILE = PROJECT_ROOT / ".claude" / "data_collector.log"


class DataCollectorState:
    """Persistent state for the data collector."""

    def __init__(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    def _load(self):
        try:
            if STATE_FILE.exists():
                return json.loads(STATE_FILE.read_text())
        except:
            pass
        return {
            "cycles_run": 0,
            "signals_stored": 0,
            "errors": 0,
            "last_run": None,
            "start_time": datetime.now().isoformat()
        }

    def save(self):
        self.state["last_run"] = datetime.now().isoformat()
        try:
            STATE_FILE.write_text(json.dumps(self.state, indent=2))
        except:
            pass


def log(msg, level="INFO"):
    """Log to stdout and file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def trading_signal_to_payload(signal: TradingSignal, symbol: str) -> SignalPayload:
    """Convert a TradingSignal to a SignalPayload for Qdrant storage."""
    # Map signal direction to strength value
    strength_map = {
        SignalDirection.BULLISH: signal.confidence,
        SignalDirection.BEARISH: -signal.confidence,
        SignalDirection.NEUTRAL: 0.0
    }

    return SignalPayload(
        topic=f"{symbol} {signal.indicator} signal",
        symbol=symbol,
        data_type="signal",
        signal_source="technical",
        signal_type=signal.indicator,
        signal_strength=strength_map.get(signal.direction, 0.0),
        decay_rate=DECAY_RATES["technical"],
        confidence=signal.confidence,
        content=f"{signal.headline}\n\n{signal.explanation}\n\nAction: {signal.action}",
        question=f"What does {signal.indicator} indicate for {symbol}?",
        tags=[signal.indicator, str(signal.direction.value), str(signal.strength.value), symbol],
        domain="technical_analysis",
        subdomain=signal.indicator
    )


def collect_data_for_symbol(fetcher: PriceFetcher, symbol: str, store: TradingVectorStore) -> dict:
    """Fetch price data and generate signals for a single symbol."""
    result = {
        "symbol": symbol,
        "success": False,
        "price": None,
        "signals_generated": 0,
        "signals_stored": 0,
        "error": None
    }

    try:
        # Fetch current price
        price_data = fetcher.get_current_price(symbol)
        if not price_data:
            result["error"] = "Could not fetch price"
            return result

        result["price"] = price_data.price

        # Get historical data for indicators (need at least 50 days for most indicators)
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < 20:
                result["error"] = "Insufficient historical data"
                return result

            # Convert to format expected by SignalGenerator
            closes = hist["Close"].tolist()
            highs = hist["High"].tolist()
            lows = hist["Low"].tolist()
            volumes = hist["Volume"].tolist()

        except ImportError:
            result["error"] = "yfinance not installed"
            return result

        # Generate signals from price data
        signal_gen = SignalGenerator(closes, highs, lows, volumes, symbol=symbol)
        summary = signal_gen.generate_all()
        signals = summary.signals

        result["signals_generated"] = len(signals)

        # Store significant signals to Qdrant (filter out weak/neutral)
        stored_count = 0
        for signal in signals:
            # Only store signals with meaningful strength
            if signal.strength == SignalStrength.WEAK and signal.direction == SignalDirection.NEUTRAL:
                continue

            # Only store if confidence > 0.5
            if signal.confidence < 0.5:
                continue

            payload = trading_signal_to_payload(signal, symbol)
            point_id = store.store(payload)
            if point_id:
                stored_count += 1

        result["signals_stored"] = stored_count
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def run_collection_cycle(symbols: list, state: DataCollectorState) -> dict:
    """Run one complete data collection cycle."""
    log(f"Starting collection cycle for {len(symbols)} symbols...")

    fetcher = PriceFetcher()
    store = TradingVectorStore()

    cycle_results = {
        "timestamp": datetime.now().isoformat(),
        "symbols_processed": 0,
        "symbols_succeeded": 0,
        "total_signals": 0,
        "total_stored": 0,
        "errors": []
    }

    for symbol in symbols:
        result = collect_data_for_symbol(fetcher, symbol, store)
        cycle_results["symbols_processed"] += 1

        if result["success"]:
            cycle_results["symbols_succeeded"] += 1
            cycle_results["total_signals"] += result["signals_generated"]
            cycle_results["total_stored"] += result["signals_stored"]
            log(f"  {symbol}: ${result['price']:.2f} - {result['signals_stored']} signals stored")
        else:
            cycle_results["errors"].append({"symbol": symbol, "error": result["error"]})
            log(f"  {symbol}: FAILED - {result['error']}", "ERROR")

        # Small delay between symbols to avoid rate limits
        time.sleep(0.5)

    # Update state
    state.state["cycles_run"] += 1
    state.state["signals_stored"] += cycle_results["total_stored"]
    state.state["errors"] += len(cycle_results["errors"])
    state.save()

    log(f"Cycle complete: {cycle_results['symbols_succeeded']}/{cycle_results['symbols_processed']} symbols, "
        f"{cycle_results['total_stored']} signals stored")

    return cycle_results


def run_continuous(symbols: list, interval: int):
    """Run data collection continuously."""
    state = DataCollectorState()

    log("=" * 60)
    log("MIDGE DATA COLLECTOR - Starting continuous mode")
    log(f"Symbols: {', '.join(symbols)}")
    log(f"Interval: {interval} seconds")
    log("=" * 60)

    try:
        while True:
            run_collection_cycle(symbols, state)
            log(f"Sleeping {interval}s until next cycle...")
            time.sleep(interval)

    except KeyboardInterrupt:
        log("\nStopped by user")

    # Final report
    log("")
    log("=" * 60)
    log("DATA COLLECTOR STOPPED")
    log(f"  Total cycles: {state.state['cycles_run']}")
    log(f"  Total signals: {state.state['signals_stored']}")
    log(f"  Total errors: {state.state['errors']}")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="MIDGE Data Collector - Continuous price and signal collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run once
    python trading/daemons/data_collector.py --once

    # Run continuously every 5 minutes
    python trading/daemons/data_collector.py --continuous --interval 300

    # With custom symbols
    python trading/daemons/data_collector.py --once --symbols AAPL,MSFT,GOOGL

    # Check status
    python trading/daemons/data_collector.py --status
        """
    )

    parser.add_argument("--once", action="store_true", help="Run one collection cycle")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between cycles (default: 300)")
    parser.add_argument("--symbols", help="Comma-separated list of symbols")
    parser.add_argument("--status", action="store_true", help="Show status and exit")

    args = parser.parse_args()

    if args.status:
        if STATE_FILE.exists():
            print(STATE_FILE.read_text())
        else:
            print("No collector state found")
        return

    # Parse symbols
    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS

    if args.continuous:
        run_continuous(symbols, args.interval)
    elif args.once:
        state = DataCollectorState()
        run_collection_cycle(symbols, state)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
