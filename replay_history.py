#!/usr/bin/env python3
"""replay_history.py — Replay MIDGE's signal archives through the convergence engine.

Discovers historical convergence patterns by feeding archived signals
chronologically through the ConvergenceAlerter with simulated time.
Then grades discovered alerts against actual price movement.

Three phases:
  1. REPLAY:  Feed archived signals through convergence engine (time-simulated)
  2. GRADE:   Evaluate discovered alerts against actual price data
  3. REPORT:  Summarize findings (win rates, domain combos, etc.)

Usage:
    python replay_history.py                           # Full pipeline
    python replay_history.py --start 2025-01-01        # From date
    python replay_history.py --end 2025-12-31          # To date
    python replay_history.py --phase replay             # Replay only
    python replay_history.py --phase grade              # Grade existing results
    python replay_history.py --phase report             # Report only
    python replay_history.py --min-domains 4            # Stricter convergence
    python replay_history.py --no-thompson              # Raw convergence (no Thompson)
"""

import argparse
import json
import logging
import math
import time as time_module
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midge.replay")

PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_ROOT / "data" / "midge" / "signals"
MARKET_DATA_DIR = PROJECT_ROOT / "data" / "market"
RESULTS_DIR = PROJECT_ROOT / "data" / "midge"
RESULTS_PATH = RESULTS_DIR / "replay_results.json"

# Default outcome window for convergence alerts (days)
DEFAULT_OUTCOME_WINDOW_DAYS = 14

# Success threshold — matches OutcomeCollector.SUCCESS_THRESHOLD_PCT
SUCCESS_THRESHOLD_PCT = 5.0

# Rate limit between yfinance lookups (seconds)
PRICE_LOOKUP_DELAY = 0.3


# ---------------------------------------------------------------
# Time simulation
# ---------------------------------------------------------------

class SimulatedDateTime:
    """Drop-in replacement for the datetime class with controllable now().

    Replaces the ``datetime`` name in convergence_alerter's module namespace
    so that all ``datetime.now()`` calls return the simulated time instead
    of wall-clock time.  Everything else (fromisoformat, min, constructors)
    delegates to the real datetime class.
    """

    def __init__(self, frozen_time: datetime):
        self._frozen = frozen_time
        self._real_datetime = datetime

    def now(self, tz=None):
        """Return the simulated time instead of wall-clock time."""
        if tz is not None:
            return self._frozen.replace(tzinfo=tz)
        return self._frozen

    def __call__(self, *args, **kwargs):
        """Support ``datetime(year, month, day, ...)`` constructor calls."""
        return self._real_datetime(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate everything else to the real datetime class."""
        return getattr(self._real_datetime, name)


# ---------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------

def _load_day_signals(day: date, signals_dir: Path = None) -> List[dict]:
    """Load all signals from a single day's archive file."""
    sdir = signals_dir or SIGNALS_DIR
    path = sdir / f"{day.isoformat()}.jsonl"
    if not path.exists():
        return []

    signals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return signals


def _parse_timestamp(ts_str: str) -> datetime:
    """Parse ISO timestamp from archive record."""
    if not ts_str:
        return datetime.min
    try:
        return datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return datetime.min


def _group_into_windows(
    signals: List[dict],
    window_hours: float = 4.0,
) -> List[List[dict]]:
    """Group pre-sorted signals into time windows of *window_hours* hours."""
    if not signals:
        return []

    windows: List[List[dict]] = []
    current_window: List[dict] = []
    window_start = _parse_timestamp(signals[0].get("timestamp", ""))
    window_delta = timedelta(hours=window_hours)

    for sig in signals:
        ts = _parse_timestamp(sig.get("timestamp", ""))
        if ts - window_start >= window_delta and current_window:
            windows.append(current_window)
            current_window = [sig]
            window_start = ts
        else:
            current_window.append(sig)

    if current_window:
        windows.append(current_window)

    return windows


# ---------------------------------------------------------------
# Symbol extraction from alerts
# ---------------------------------------------------------------

def _extract_symbols(alert) -> List[str]:
    """Extract ticker symbols from a convergence alert's contributing signals."""
    symbols = []
    for sig in alert.signals:
        meta = sig.metadata or {}
        symbol = meta.get("symbol", meta.get("ticker", ""))
        if symbol and symbol not in ("", "N/A", "UNKNOWN"):
            symbols.append(symbol)
    return symbols


def _primary_symbol(alert) -> Optional[str]:
    """Return the most-mentioned symbol across an alert's signals."""
    symbols = _extract_symbols(alert)
    if not symbols:
        return None
    counter = Counter(symbols)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------
# Phase 1: REPLAY
# ---------------------------------------------------------------

def phase_replay(
    start_date: date,
    end_date: date,
    min_domains: int = 3,
    use_thompson: bool = True,
    signals_dir: Path = None,
) -> List[dict]:
    """Feed archived signals through the convergence engine with simulated time.

    Returns a list of discovered alert dicts (JSON-serializable).
    """
    logger.info("=" * 60)
    logger.info("Phase 1: REPLAY — feeding archives through convergence engine")
    logger.info("  Date range: %s to %s", start_date, end_date)
    logger.info("  Min domains: %d | Thompson: %s", min_domains, use_thompson)
    logger.info("=" * 60)

    import mae_core.market.intelligence.convergence_alerter as ca_module
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    # Optionally wire Thompson for realistic confidence weighting
    thompson = None
    if use_thompson:
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        thompson = ThompsonSampler()

    # Fresh alerter — no EventBus, regime, or other live wiring
    alerter = ConvergenceAlerter(
        min_domains=min_domains,
        thompson_sampler=thompson,
    )

    # Save the real datetime reference
    original_datetime = ca_module.datetime

    discovered_alerts: List[dict] = []
    total_signals = 0
    days_processed = 0

    try:
        current = start_date
        while current <= end_date:
            signals = _load_day_signals(current, signals_dir=signals_dir)
            if signals:
                # Sort chronologically within the day
                signals.sort(key=lambda s: s.get("timestamp", ""))

                # Group into 4-hour windows (matches alert dedup interval)
                windows = _group_into_windows(signals, window_hours=4.0)

                for window in windows:
                    if not window:
                        continue

                    # Latest timestamp in this window becomes simulated "now"
                    timestamps = [
                        _parse_timestamp(s.get("timestamp", ""))
                        for s in window
                    ]
                    valid_ts = [t for t in timestamps if t != datetime.min]
                    if not valid_ts:
                        continue
                    window_time = max(valid_ts)

                    # Monkeypatch the clock
                    ca_module.datetime = SimulatedDateTime(window_time)

                    # Feed each signal
                    for sig in window:
                        ts = _parse_timestamp(sig.get("timestamp", ""))
                        if ts == datetime.min:
                            continue

                        metadata = dict(sig.get("metadata", {}) or {})
                        # Guarantee symbol is in metadata for grading later
                        if sig.get("symbol") and "symbol" not in metadata:
                            metadata["symbol"] = sig["symbol"]

                        alerter.record_signal(
                            signal_id=sig.get("signal_id", ""),
                            strength=float(sig.get("strength", 0.0)),
                            domain=sig.get("domain", ""),
                            direction=sig.get("direction", "neutral"),
                            confidence=float(sig.get("confidence", 0.5)),
                            velocity=float(sig.get("velocity", 0.0)),
                            timestamp=ts,
                            metadata=metadata,
                            source=sig.get("source", ""),
                        )
                        total_signals += 1

                    # Check for convergence under simulated time
                    alerts = alerter.check_convergence()
                    for alert in alerts:
                        alert_dict = alert.to_dict()
                        alert_dict["replay_date"] = current.isoformat()
                        alert_dict["primary_symbol"] = _primary_symbol(alert)
                        alert_dict["all_symbols"] = list(
                            set(_extract_symbols(alert))
                        )
                        discovered_alerts.append(alert_dict)

                        logger.info(
                            "  ALERT: %s %s | conf=%.2f str=%.2f | %s | %s",
                            alert.direction.upper(),
                            alert.alert_id,
                            alert.confidence,
                            alert.strength,
                            "+".join(alert.domains_converging),
                            alert_dict["all_symbols"][:3],
                        )

            # Reset dedup between days — each day is independent for dedup,
            # but the signal buffer carries across days (signals within their
            # domain windows survive just like in live operation).
            alerter._last_alert_times.clear()

            days_processed += 1
            if days_processed % 30 == 0:
                logger.info(
                    "  ... %d days, %d signals, %d alerts so far",
                    days_processed, total_signals, len(discovered_alerts),
                )

            current += timedelta(days=1)

    finally:
        # ALWAYS restore the real datetime — critical for system health
        ca_module.datetime = original_datetime

    logger.info(
        "Replay complete: %d days, %d signals fed, %d alerts discovered",
        days_processed, total_signals, len(discovered_alerts),
    )

    _save_results({"alerts": discovered_alerts, "phase": "replay"})
    return discovered_alerts


# ---------------------------------------------------------------
# Magnitude helpers
# ---------------------------------------------------------------

def _compute_excursions(
    symbol: str,
    alert_dt: date,
    window_days: int,
    entry_price: float,
    direction: str,
    get_price_fn,
) -> tuple:
    """Compute Maximum Favorable Excursion and Maximum Adverse Excursion.

    MFE = largest move in the predicted direction during the window.
    MAE = largest move against the predicted direction during the window.
    Both returned as positive percentages.
    """
    prices = []
    for d in range(1, window_days + 1):
        dt = alert_dt + timedelta(days=d)
        p = get_price_fn(symbol, dt)
        if p is not None:
            prices.append(p)

    if not prices:
        return None, None

    pct_moves = [((p - entry_price) / entry_price) * 100 for p in prices]

    if direction == "bullish":
        mfe = max(max(pct_moves), 0)
        mae = max(-min(pct_moves), 0)
    elif direction == "bearish":
        mfe = max(-min(pct_moves), 0)
        mae = max(max(pct_moves), 0)
    else:
        mfe = max(abs(m) for m in pct_moves) if pct_moves else 0
        mae = 0

    return mfe, mae


# ---------------------------------------------------------------
# Phase 2: GRADE
# ---------------------------------------------------------------

def phase_grade(
    alerts: List[dict] = None,
    outcome_window_days: int = DEFAULT_OUTCOME_WINDOW_DAYS,
) -> List[dict]:
    """Grade discovered alerts against actual price movement.

    Returns the same alert list with grade/price fields appended.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: GRADE — evaluating alerts against price data")
    logger.info("=" * 60)

    if alerts is None:
        if RESULTS_PATH.exists():
            data = json.loads(RESULTS_PATH.read_text())
            alerts = data.get("alerts", [])
        else:
            logger.warning("No replay results found at %s", RESULTS_PATH)
            return []

    from mae_core.market.apis.price_fetcher import PriceFetcher

    price_fetcher = PriceFetcher()
    price_cache: Dict[tuple, Optional[float]] = {}

    def get_price(symbol: str, dt: date) -> Optional[float]:
        key = (symbol, dt.isoformat())
        if key not in price_cache:
            try:
                pd = price_fetcher.get_historical_price(symbol, dt.isoformat())
                price_cache[key] = pd.price if pd else None
            except Exception:
                price_cache[key] = None
            time_module.sleep(PRICE_LOOKUP_DELAY)
        return price_cache[key]

    graded = 0
    successes = 0
    failures = 0
    ungraded_count = 0

    for alert in alerts:
        symbol = alert.get("primary_symbol")
        if not symbol:
            alert["grade"] = "ungraded"
            alert["grade_reason"] = "no_symbol"
            ungraded_count += 1
            continue

        try:
            alert_dt = datetime.fromisoformat(alert["timestamp"]).date()
        except (ValueError, KeyError):
            alert["grade"] = "ungraded"
            alert["grade_reason"] = "bad_timestamp"
            ungraded_count += 1
            continue

        outcome_dt = alert_dt + timedelta(days=outcome_window_days)

        entry_price = get_price(symbol, alert_dt)
        exit_price = get_price(symbol, outcome_dt)

        if entry_price is None or exit_price is None:
            alert["grade"] = "ungraded"
            alert["grade_reason"] = "no_price_data"
            alert["symbol_graded"] = symbol
            ungraded_count += 1
            continue

        pct_change = ((exit_price - entry_price) / entry_price) * 100
        alert["entry_price"] = round(entry_price, 2)
        alert["exit_price"] = round(exit_price, 2)
        alert["pct_change"] = round(pct_change, 2)
        alert["outcome_window_days"] = outcome_window_days
        alert["symbol_graded"] = symbol

        # Magnitude tracking: MFE/MAE from daily prices through the window
        direction = alert.get("direction", "neutral")
        mfe, mae = _compute_excursions(
            symbol, alert_dt, outcome_window_days, entry_price,
            direction, get_price,
        )
        if mfe is not None:
            alert["mfe_pct"] = round(mfe, 2)
            alert["mae_pct"] = round(mae, 2)

        if direction == "bullish" and pct_change >= SUCCESS_THRESHOLD_PCT:
            alert["grade"] = "success"
            successes += 1
        elif direction == "bearish" and pct_change <= -SUCCESS_THRESHOLD_PCT:
            alert["grade"] = "success"
            successes += 1
        else:
            alert["grade"] = "failure"
            failures += 1

        graded += 1
        if graded % 20 == 0:
            logger.info(
                "  Graded %d alerts (%d wins, %d losses, %d ungraded)",
                graded, successes, failures, ungraded_count,
            )

    total = graded + ungraded_count
    win_rate = (successes / graded * 100) if graded > 0 else 0
    logger.info("Grading complete: %d/%d graded, %d ungraded", graded, total, ungraded_count)
    logger.info("Win rate: %.1f%% (%d/%d)", win_rate, successes, graded)

    _save_results({"alerts": alerts, "phase": "grade"})
    return alerts


# ---------------------------------------------------------------
# Phase 3: REPORT
# ---------------------------------------------------------------

def phase_report(alerts: List[dict] = None) -> dict:
    """Generate summary report from graded alerts."""
    logger.info("=" * 60)
    logger.info("Phase 3: REPORT — summarizing findings")
    logger.info("=" * 60)

    if alerts is None:
        if RESULTS_PATH.exists():
            data = json.loads(RESULTS_PATH.read_text())
            alerts = data.get("alerts", [])
        else:
            logger.warning("No results found at %s", RESULTS_PATH)
            return {}

    graded = [a for a in alerts if a.get("grade") in ("success", "failure")]
    successes = [a for a in graded if a["grade"] == "success"]
    failures = [a for a in graded if a["grade"] == "failure"]
    ungraded = [a for a in alerts if a.get("grade") not in ("success", "failure")]

    report: dict = {
        "total_alerts": len(alerts),
        "graded": len(graded),
        "ungraded": len(ungraded),
        "successes": len(successes),
        "failures": len(failures),
        "win_rate_pct": round(len(successes) / len(graded) * 100, 1) if graded else 0,
    }

    # Win rate by direction
    for direction in ("bullish", "bearish"):
        dir_graded = [a for a in graded if a.get("direction") == direction]
        dir_wins = [a for a in dir_graded if a["grade"] == "success"]
        report[f"win_rate_{direction}_pct"] = (
            round(len(dir_wins) / len(dir_graded) * 100, 1) if dir_graded else 0
        )
        report[f"count_{direction}"] = len(dir_graded)

    # Win rate by domain combination
    combo_stats: Dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    for alert in graded:
        domains = tuple(sorted(alert.get("domains", [])))
        combo_key = "+".join(domains)
        combo_stats[combo_key]["total"] += 1
        if alert["grade"] == "success":
            combo_stats[combo_key]["wins"] += 1

    report["domain_combos"] = {
        combo: {
            "wins": stats["wins"],
            "total": stats["total"],
            "win_rate_pct": (
                round(stats["wins"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
            ),
        }
        for combo, stats in sorted(
            combo_stats.items(), key=lambda x: x[1]["total"], reverse=True
        )
    }

    # Top 10 strongest alerts by confidence * strength
    scored = sorted(
        alerts,
        key=lambda a: a.get("confidence", 0) * a.get("strength", 0),
        reverse=True,
    )
    report["top_10_alerts"] = [
        {
            "alert_id": a.get("alert_id"),
            "direction": a.get("direction"),
            "strength": a.get("strength"),
            "confidence": a.get("confidence"),
            "domains": a.get("domains"),
            "grade": a.get("grade", "ungraded"),
            "pct_change": a.get("pct_change"),
            "symbol": a.get("symbol_graded") or a.get("primary_symbol"),
        }
        for a in scored[:10]
    ]

    # Monthly distribution
    monthly: Dict[str, int] = defaultdict(int)
    for alert in alerts:
        try:
            dt = datetime.fromisoformat(alert["timestamp"])
            monthly[dt.strftime("%Y-%m")] += 1
        except (ValueError, KeyError):
            pass
    report["monthly_distribution"] = dict(sorted(monthly.items()))

    # Average confidence: winners vs losers
    if successes:
        report["avg_confidence_winners"] = round(
            sum(a.get("confidence", 0) for a in successes) / len(successes), 3
        )
    if failures:
        report["avg_confidence_losers"] = round(
            sum(a.get("confidence", 0) for a in failures) / len(failures), 3
        )

    # --- Magnitude metrics ---
    winner_returns = [a["pct_change"] for a in successes if "pct_change" in a]
    loser_returns = [a["pct_change"] for a in failures if "pct_change" in a]
    all_returns = [a["pct_change"] for a in graded if "pct_change" in a]

    if winner_returns:
        report["avg_return_winners"] = round(sum(winner_returns) / len(winner_returns), 2)
    if loser_returns:
        report["avg_return_losers"] = round(sum(loser_returns) / len(loser_returns), 2)

    # Expectancy: avg_win * win_rate - avg_loss * loss_rate
    if winner_returns and loser_returns and graded:
        win_rate = len(successes) / len(graded)
        loss_rate = len(failures) / len(graded)
        # For bearish successes, pct_change is negative but it's a win — use absolute values
        avg_win = sum(abs(r) for r in winner_returns) / len(winner_returns)
        avg_loss = sum(abs(r) for r in loser_returns) / len(loser_returns)
        report["expectancy_pct"] = round(avg_win * win_rate - avg_loss * loss_rate, 2)
        report["avg_win_pct"] = round(avg_win, 2)
        report["avg_loss_pct"] = round(avg_loss, 2)

    # Sharpe ratio (annualized from outcome window)
    if len(all_returns) >= 2:
        # Use directional returns: bullish = pct_change, bearish = -pct_change
        dir_returns = []
        for a in graded:
            if "pct_change" not in a:
                continue
            r = a["pct_change"]
            if a.get("direction") == "bearish":
                r = -r  # flip so positive = profitable
            dir_returns.append(r)
        if len(dir_returns) >= 2:
            mean_r = sum(dir_returns) / len(dir_returns)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in dir_returns) / (len(dir_returns) - 1))
            if std_r > 0:
                # Annualize: sqrt(trades_per_year) where trades_per_year ~ 252/outcome_window
                outcome_window = graded[0].get("outcome_window_days", DEFAULT_OUTCOME_WINDOW_DAYS)
                ann_factor = math.sqrt(252 / max(outcome_window, 1))
                report["sharpe_ratio"] = round((mean_r / std_r) * ann_factor, 2)

    # Percentile returns (P25, P50, P75, P90)
    def _percentile(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        k = (len(s) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(s) else f
        return s[f] + (k - f) * (s[c] - s[f])

    if winner_returns:
        abs_wins = sorted(abs(r) for r in winner_returns)
        report["winner_return_p25"] = round(_percentile(abs_wins, 25), 2)
        report["winner_return_p50"] = round(_percentile(abs_wins, 50), 2)
        report["winner_return_p75"] = round(_percentile(abs_wins, 75), 2)
        report["winner_return_p90"] = round(_percentile(abs_wins, 90), 2)

    # MFE/MAE summaries
    mfe_vals = [a["mfe_pct"] for a in graded if "mfe_pct" in a]
    mae_vals = [a["mae_pct"] for a in graded if "mae_pct" in a]
    if mfe_vals:
        report["mfe_p50"] = round(_percentile(sorted(mfe_vals), 50), 2)
        report["mfe_p90"] = round(_percentile(sorted(mfe_vals), 90), 2)
    if mae_vals:
        report["mae_p50"] = round(_percentile(sorted(mae_vals), 50), 2)
        report["mae_p90"] = round(_percentile(sorted(mae_vals), 90), 2)

    # Print human-readable summary
    logger.info("")
    logger.info("=== REPLAY RESULTS ===")
    logger.info("Total alerts:    %d", report["total_alerts"])
    logger.info("Graded:          %d  (ungraded: %d)", report["graded"], report["ungraded"])
    logger.info(
        "Win rate:        %.1f%% (%d wins / %d graded)",
        report["win_rate_pct"], report["successes"], report["graded"],
    )
    logger.info("")
    logger.info("By direction:")
    logger.info(
        "  Bullish:  %.1f%% win rate (%d alerts)",
        report.get("win_rate_bullish_pct", 0), report.get("count_bullish", 0),
    )
    logger.info(
        "  Bearish:  %.1f%% win rate (%d alerts)",
        report.get("win_rate_bearish_pct", 0), report.get("count_bearish", 0),
    )
    logger.info("")
    logger.info("By domain combination:")
    for combo, stats in list(report["domain_combos"].items())[:10]:
        logger.info(
            "  %-40s  %.1f%% (%d/%d)",
            combo, stats["win_rate_pct"], stats["wins"], stats["total"],
        )
    if "avg_confidence_winners" in report:
        logger.info("")
        logger.info("Avg confidence (winners): %.3f", report["avg_confidence_winners"])
    if "avg_confidence_losers" in report:
        logger.info("Avg confidence (losers):  %.3f", report["avg_confidence_losers"])

    # Magnitude metrics
    if "expectancy_pct" in report:
        logger.info("")
        logger.info("=== MAGNITUDE METRICS ===")
        logger.info("Avg win:       %.2f%%", report.get("avg_win_pct", 0))
        logger.info("Avg loss:      %.2f%%", report.get("avg_loss_pct", 0))
        logger.info("Expectancy:    %.2f%% per trade", report["expectancy_pct"])
    if "sharpe_ratio" in report:
        logger.info("Sharpe ratio:  %.2f (annualized)", report["sharpe_ratio"])
    if "winner_return_p50" in report:
        logger.info("")
        logger.info("Winner returns: P25=%.1f%%  P50=%.1f%%  P75=%.1f%%  P90=%.1f%%",
                     report["winner_return_p25"], report["winner_return_p50"],
                     report["winner_return_p75"], report["winner_return_p90"])
    if "mfe_p50" in report:
        logger.info("MFE (all):  P50=%.1f%%  P90=%.1f%%", report["mfe_p50"], report["mfe_p90"])
    if "mae_p50" in report:
        logger.info("MAE (all):  P50=%.1f%%  P90=%.1f%%", report["mae_p50"], report["mae_p90"])

    logger.info("")
    logger.info("Monthly distribution:")
    for month, count in report["monthly_distribution"].items():
        logger.info("  %s: %d alerts", month, count)

    _save_results({"alerts": alerts, "report": report, "phase": "report"})
    return report


# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------

def _save_results(data: dict) -> None:
    """Atomically save results to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tmp = RESULTS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(RESULTS_PATH)
    except Exception as e:
        logger.warning("Could not save results: %s", e)


def _get_date_range(signals_dir: Path = None) -> tuple:
    """Discover the available date range from signal archives."""
    sdir = signals_dir or SIGNALS_DIR
    if not sdir.exists():
        return None, None

    dates = []
    for f in sdir.glob("*.jsonl"):
        try:
            d = date.fromisoformat(f.stem)
            dates.append(d)
        except ValueError:
            pass

    if not dates:
        return None, None
    return min(dates), max(dates)


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay MIDGE's signal archives through the convergence engine",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["replay", "grade", "report", "all"],
        help="Which phase to run (default: all)",
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--min-domains", type=int, default=3,
        help="Minimum domains for convergence (default: 3)",
    )
    parser.add_argument(
        "--no-thompson", action="store_true",
        help="Skip Thompson weighting (raw convergence)",
    )
    parser.add_argument(
        "--outcome-window", type=int, default=DEFAULT_OUTCOME_WINDOW_DAYS,
        help="Outcome window in days (default: %d)" % DEFAULT_OUTCOME_WINDOW_DAYS,
    )
    args = parser.parse_args()

    # Determine date range
    archive_start, archive_end = _get_date_range()
    if archive_start is None:
        logger.error("No signal archives found in %s", SIGNALS_DIR)
        return

    start_date = date.fromisoformat(args.start) if args.start else archive_start
    end_date = date.fromisoformat(args.end) if args.end else archive_end

    logger.info("=" * 60)
    logger.info("MIDGE Historical Replay Pipeline")
    logger.info("  Phase: %s", args.phase)
    logger.info("  Archive range: %s to %s", archive_start, archive_end)
    logger.info("  Replay range:  %s to %s", start_date, end_date)
    logger.info(
        "  Min domains: %d | Thompson: %s | Outcome window: %dd",
        args.min_domains, not args.no_thompson, args.outcome_window,
    )
    logger.info("=" * 60)

    start_time = time_module.time()
    alerts = None

    if args.phase in ("replay", "all"):
        alerts = phase_replay(
            start_date=start_date,
            end_date=end_date,
            min_domains=args.min_domains,
            use_thompson=not args.no_thompson,
        )
        logger.info("")

    if args.phase in ("grade", "all"):
        alerts = phase_grade(
            alerts=alerts,
            outcome_window_days=args.outcome_window,
        )
        logger.info("")

    if args.phase in ("report", "all"):
        phase_report(alerts=alerts)
        logger.info("")

    elapsed = time_module.time() - start_time
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
