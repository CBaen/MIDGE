#!/usr/bin/env python3
"""
sweep_backtest.py - ICT Session Sweep + IFVG Backtest Engine

Re-export hub. All logic lives in sub-modules:
  sweep_backtest_models.py  — Level, FVGZone, SweepEvent, Trade dataclasses
  sweep_backtest_engine.py  — fetch, detection, FVG/IFVG, scoring helpers
  sweep_backtest_runner.py  — simulate_trade, backtest_symbol, run
  sweep_backtest_report.py  — report() statistics function

Usage:
    python -m mae_core.market.edge.sweep_backtest
    python -m mae_core.market.edge.sweep_backtest --symbols ES=F NQ=F
    python -m mae_core.market.edge.sweep_backtest --futures-only
    python -m mae_core.market.edge.sweep_backtest --equities-only --interval 15m
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

# Re-export dataclasses
from mae_core.market.edge.sweep_backtest_models import (  # noqa: F401
    Level, FVGZone, SweepEvent, Trade,
)

# Re-export engine and runner
from mae_core.market.edge.sweep_backtest_engine import SweepBacktestEngine  # noqa: F401
from mae_core.market.edge.sweep_backtest_runner import SweepBacktester  # noqa: F401

# Re-export report
from mae_core.market.edge.sweep_backtest_report import report  # noqa: F401

try:
    HAS_DEPS = True
    import pandas  # noqa: F401
    import numpy   # noqa: F401
except ImportError:
    HAS_DEPS = False

try:
    import yfinance  # noqa: F401
    HAS_YF = True
except ImportError:
    HAS_YF = False

# Symbol lists (kept here for CLI use)
FUTURES = ["ES=F", "NQ=F", "YM=F", "RTY=F", "GC=F", "CL=F"]
EQUITIES = [
    "SPY", "QQQ", "IWM", "DIA",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
    "AMD", "NFLX", "JPM", "V", "MA", "COST", "CRM", "AVGO",
]


def main():
    parser = argparse.ArgumentParser(
        description="ICT Session Sweep + IFVG Backtest",
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--futures-only", action="store_true")
    parser.add_argument("--equities-only", action="store_true")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--days", type=int, default=59)
    parser.add_argument("--min-quality", type=float, default=0.0)
    args = parser.parse_args()

    if not HAS_DEPS or not HAS_YF:
        print("Missing dependencies: pip install yfinance pandas numpy")
        return

    if args.symbols:
        symbols = args.symbols
    elif args.futures_only:
        symbols = FUTURES
    elif args.equities_only:
        symbols = EQUITIES
    else:
        symbols = FUTURES + EQUITIES

    bt = SweepBacktester(
        interval=args.interval, days=args.days, min_quality=args.min_quality,
    )
    trades = bt.run(symbols)
    output = report(trades)
    print(output)

    if trades:
        results_path = Path("data/market/sweep_backtest_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "run_time": datetime.now().isoformat(),
            "config": {
                "interval": args.interval,
                "days": args.days,
                "symbols": [t.symbol for t in trades],
            },
            "summary": {
                "total_trades": len(trades),
                "wins": len([t for t in trades if t.result == "win_2r"]),
                "losses": len([t for t in trades if t.result == "loss"]),
                "win_rate": (
                    len([t for t in trades if t.result == "win_2r"])
                    / len(trades) * 100
                ),
                "avg_r": sum(t.r_captured for t in trades) / len(trades),
                "hit_1r_count": len([t for t in trades if t.hit_1r]),
            },
            "trades": [
                {
                    "symbol": t.symbol,
                    "direction": t.direction,
                    "session_swept": t.session_swept,
                    "entry_price": t.entry_price,
                    "stop_price": t.stop_price,
                    "target_2r": t.target_2r,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "exit_price": t.exit_price,
                    "result": t.result,
                    "r_captured": t.r_captured,
                    "hit_1r": t.hit_1r,
                    "risk_pts": t.risk_pts,
                    "displacement_score": t.displacement_score,
                    "fvg_atr_ratio": t.fvg_atr_ratio,
                    "kill_zone_score": t.kill_zone_score,
                    "quality_score": t.quality_score,
                }
                for t in trades
            ],
        }
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
