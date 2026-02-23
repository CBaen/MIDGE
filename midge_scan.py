#!/usr/bin/env python3
"""MIDGE Market Intelligence Scan.

Standalone script that fetches live data from all sources, converts to
MarketSignals, feeds the convergence engine, stores to Qdrant + JSONL,
and writes an intelligence report.

Does NOT require the 33-layer Mae bootstrap. Instantiates only what it needs.

Usage:
    python midge_scan.py                    # Full scan
    python midge_scan.py --dry-run          # Skip Qdrant storage
    python midge_scan.py --watchlist path   # Custom watchlist JSON
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Load .env before any module reads os.environ at import time
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from mae_core.market.signal import (
    MarketSignal,
    from_insider_trade,
    from_form8k_event,
    from_congressional_trade,
    from_hiring_signal,
    from_government_contract,
    from_contract_opportunity,
)
from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
from mae_core.market.intelligence.velocity_detector import VelocityDetector
from mae_core.market.intelligence.outcome_collector import OutcomeCollector
from mae_core.market.edge.filing_time_analyzer import FilingTimeAnalyzer
from mae_core.market.memory import SignalMemory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midge_scan")

# Paths
DATA_DIR = Path(__file__).parent / "data" / "midge"
SCANS_DIR = DATA_DIR / "scans"
SIGNALS_DIR = DATA_DIR / "signals"

# Multi-timeframe convergence: route signals by source → tier
TIER_ROUTING = {
    # Tier 1 Tactical (48h) — fast-decaying signals priced in hours/days
    "sec_form4": "tactical",
    "sec_form8k": "tactical",
    # Tier 2 Strategic (21d) — medium persistence, weeks-scale information
    "congressional": "strategic",
    "contract": "strategic",
    "insider_cluster": "strategic",
    "correlation": "strategic",
    # Tier 3 Thematic (90d) — slow-moving, months-scale patterns
    "sam_gov": "thematic",
    "hiring_tracker": "thematic",
    "contract_prediction": "thematic",
}


# ── Phase 1: Setup ────────────────────────────────────────────────────────

def load_watchlist(path: Path) -> dict:
    """Load watchlist from JSON file."""
    with open(path) as f:
        return json.load(f)


def setup(args):
    """Initialize all components. Returns (clients, alerter, memory, watchlist, velocity_detector, filing_analyzer, outcome_collector)."""
    watchlist_path = Path(args.watchlist) if args.watchlist else DATA_DIR / "watchlist.json"
    watchlist = load_watchlist(watchlist_path)
    logger.info(
        f"Watchlist: {len(watchlist['tickers'])} tickers, "
        f"{len(watchlist['keywords'])} keywords, "
        f"{len(watchlist['companies'])} companies"
    )

    # API clients — no provider injection needed for standalone scan
    from mae_core.market.apis.sec_edgar.client import SECEdgarClient
    from mae_core.market.apis.price_fetcher import PriceFetcher
    from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
    from mae_core.market.apis.job_tracker import JobTracker
    from mae_core.market.apis.usa_spending import USASpendingClient
    from mae_core.market.apis.sam_gov import SAMGovClient

    rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
    sam_key = os.environ.get("SAM_GOV_API_KEY", "")
    av_key = os.environ.get("MAE_ALPHAVANTAGE_API_KEY", "")

    clients = {
        "sec": SECEdgarClient(),
        "price": PriceFetcher(alpha_vantage_key=av_key),
        "congress": HouseStockWatcherClient(rapidapi_key=rapidapi_key),
        "jobs": JobTracker(rapidapi_key=rapidapi_key),
        "usa_spending": USASpendingClient(),
        "sam_gov": SAMGovClient(api_key=sam_key),
    }

    alerter = ConvergenceAlerter(min_domains=2, convergence_window_hours=168)

    memory = SignalMemory()
    if not args.dry_run and memory.is_available():
        memory.ensure_collection()
        logger.info(f"Qdrant connected — {memory.count()} existing signals")
    elif args.dry_run:
        logger.info("Dry run — Qdrant storage skipped")
    else:
        logger.info("Qdrant unavailable — signals will only persist to JSONL")

    velocity_detector = VelocityDetector()
    filing_analyzer = FilingTimeAnalyzer()

    # Outcome collector — bridges signals → predictions → Thompson feedback loop
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    thompson = ThompsonSampler()
    outcome_collector = OutcomeCollector(clients["price"], thompson)
    logger.info(f"Outcome collector: {outcome_collector.get_statistics()['registered_signals']} signals tracked")

    return clients, alerter, memory, watchlist, velocity_detector, filing_analyzer, outcome_collector


# ── Phase 2: Fetch ────────────────────────────────────────────────────────

def fetch_all(clients: dict, watchlist: dict) -> dict:
    """Fetch data from all sources. Returns dict of raw results by source."""
    results = {
        "insider_trades": [],
        "form8k_events": [],
        "congressional_trades": [],
        "hiring_signals": [],
        "contracts": [],
        "opportunities": [],
        "prices": {},
    }

    tickers = watchlist["tickers"]
    keywords = watchlist["keywords"]
    companies = watchlist["companies"]

    # SEC EDGAR — Form 4 insider trades
    from mae_core.market.apis.sec_edgar import get_recent_form4s, get_recent_form8ks

    for ticker in tickers:
        try:
            trades = get_recent_form4s(ticker, days=30)
            results["insider_trades"].extend(trades)
            if trades:
                logger.info(f"  SEC Form 4: {len(trades)} trades for {ticker}")
        except Exception as e:
            logger.warning(f"  SEC Form 4 failed for {ticker}: {e}")

        try:
            events = get_recent_form8ks(ticker, days=30)
            results["form8k_events"].extend(events)
            if events:
                logger.info(f"  SEC Form 8-K: {len(events)} events for {ticker}")
        except Exception as e:
            logger.warning(f"  SEC Form 8-K failed for {ticker}: {e}")

    # Congressional trades
    try:
        trades = clients["congress"].get_recent_trades(days=30)
        results["congressional_trades"] = trades
        logger.info(f"  Congressional trades: {len(trades)}")
    except Exception as e:
        logger.warning(f"  Congressional trades failed: {e}")

    # Job tracker — hiring blitzes
    for company, ticker in companies.items():
        try:
            signal = clients["jobs"].analyze_hiring_activity(company, ticker=ticker)
            results["hiring_signals"].append(signal)
            if signal.is_spike:
                logger.info(f"  HIRING SPIKE: {company} ({ticker}) — {signal.spike_ratio:.1f}x normal")
            else:
                logger.info(f"  Jobs: {company} — {signal.jobs_24h} in 24h (normal)")
        except Exception as e:
            logger.warning(f"  Job tracker failed for {company}: {e}")

    # USASpending — government contracts
    for keyword in keywords:
        try:
            contracts = clients["usa_spending"].search_contracts(keyword=keyword, limit=5)
            results["contracts"].extend(contracts)
            if contracts:
                logger.info(f"  USASpending '{keyword}': {len(contracts)} contracts")
        except Exception as e:
            logger.warning(f"  USASpending failed for '{keyword}': {e}")

    # SAM.gov — contract opportunities
    for keyword in keywords:
        try:
            opps = clients["sam_gov"].search_opportunities(keywords=keyword, limit=5)
            results["opportunities"].extend(opps)
            if opps:
                logger.info(f"  SAM.gov '{keyword}': {len(opps)} opportunities")
        except Exception as e:
            logger.warning(f"  SAM.gov failed for '{keyword}': {e}")

    # Prices for all symbols seen
    all_symbols = set(tickers)
    for trade in results["insider_trades"]:
        if trade.ticker_symbol:
            all_symbols.add(trade.ticker_symbol)
    for trade in results["congressional_trades"]:
        if trade.ticker:
            all_symbols.add(trade.ticker)

    for symbol in sorted(all_symbols):
        try:
            price = clients["price"].get_current_price(symbol)
            if price:
                results["prices"][symbol] = price
        except Exception:
            pass

    return results


# ── Phase 3: Convert to MarketSignals ─────────────────────────────────────

def convert_to_signals(results: dict) -> List[MarketSignal]:
    """Convert raw API results to normalized MarketSignals."""
    signals = []

    for trade in results["insider_trades"]:
        try:
            signals.append(from_insider_trade(trade))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (insider): {e}")

    for event in results["form8k_events"]:
        try:
            signals.append(from_form8k_event(event))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (8-K): {e}")

    for trade in results["congressional_trades"]:
        try:
            # Filter: trades below $50K are economically unactionable noise
            if trade.amount_high < 50_000:
                continue
            signals.append(from_congressional_trade(trade))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (congressional): {e}")

    for signal in results["hiring_signals"]:
        try:
            signals.append(from_hiring_signal(signal))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (hiring): {e}")

    for contract in results["contracts"]:
        try:
            signals.append(from_government_contract(contract))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (contract): {e}")

    for opp in results["opportunities"]:
        try:
            signals.append(from_contract_opportunity(opp))
        except Exception as e:
            logger.debug(f"  Signal conversion failed (opportunity): {e}")

    return signals


# ── Phase 4: Store + Feed ─────────────────────────────────────────────────

def store_and_feed(
    signals: List[MarketSignal],
    alerter: ConvergenceAlerter,
    memory: SignalMemory,
    dry_run: bool,
    velocity_detector: VelocityDetector = None,
    filing_analyzer: FilingTimeAnalyzer = None,
):
    """Store signals to Qdrant + JSONL, feed into convergence engine.

    Applies VelocityDetector (populates velocity field) and FilingTimeAnalyzer
    (applies confidence modifiers for suspicious filing times) before feeding
    signals into the convergence alerter.
    """
    # Qdrant storage
    if not dry_run and memory.is_available():
        success = memory.store_signals(signals)
        if success:
            logger.info(f"Stored {len(signals)} signals to Qdrant")
        else:
            logger.warning("Qdrant storage failed — continuing with JSONL only")

    # JSONL cold storage (always)
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    jsonl_path = SIGNALS_DIR / f"{today}.jsonl"

    with open(jsonl_path, "a") as f:
        for sig in signals:
            record = {
                "signal_id": sig.signal_id,
                "source": sig.source,
                "symbol": sig.symbol,
                "domain": sig.domain,
                "direction": sig.direction,
                "strength": sig.strength,
                "confidence": sig.confidence,
                "velocity": sig.velocity,
                "timestamp": sig.timestamp.isoformat(),
                "received_at": sig.received_at.isoformat(),
                "metadata": sig.metadata,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Appended {len(signals)} signals to {jsonl_path}")

    # Enrich signals with velocity and filing-time modifiers before feeding alerter
    for sig in signals:
        # Populate velocity via VelocityDetector
        if velocity_detector is not None:
            try:
                state = velocity_detector.record(sig.signal_id, sig.strength, sig.timestamp)
                sig.velocity = state.current_velocity
            except Exception:
                pass

        # Apply filing-time confidence modifier for SEC filings
        if filing_analyzer is not None and sig.source in ("sec_form4", "sec_form8k"):
            try:
                filing_dt = sig.received_at or sig.timestamp
                fta_signal = filing_analyzer.analyze_filing_time(
                    ticker=sig.symbol,
                    filer_name=sig.metadata.get("filer_name", ""),
                    filing_date=sig.timestamp.strftime("%Y-%m-%d"),
                    filing_datetime=filing_dt,
                    form_type="4" if sig.source == "sec_form4" else "8-K",
                )
                sig.confidence = max(0.0, min(1.0, sig.confidence + fta_signal.confidence_modifier))
            except Exception:
                pass

    # Feed convergence engine
    for sig in signals:
        alerter.record_signal(
            signal_id=sig.signal_id,
            strength=sig.strength,
            domain=sig.domain,
            direction=sig.direction,
            confidence=sig.confidence,
            velocity=sig.velocity,
            timestamp=sig.timestamp,
            metadata={**sig.metadata, "symbol": sig.symbol},
        )


# ── Phase 5: Analyze ─────────────────────────────────────────────────────

def analyze(alerter: ConvergenceAlerter) -> dict:
    """Run convergence analysis. Returns analysis results dict."""
    alerts = alerter.check_convergence()
    ticker_alerts = alerter.check_ticker_convergence(min_domains=2)
    summary = alerter.get_actionable_summary()
    matrix = alerter.get_convergence_matrix()
    domain_status = alerter.get_domain_status()

    return {
        "alerts": alerts,
        "ticker_alerts": ticker_alerts,
        "summary": summary,
        "matrix": matrix,
        "domain_status": domain_status,
    }


# ── Phase 6: Report ──────────────────────────────────────────────────────

def count_by_source(signals: List[MarketSignal]) -> dict:
    """Count signals by source type."""
    counts = {}
    for sig in signals:
        counts[sig.source] = counts.get(sig.source, 0) + 1
    return counts


def group_by_symbol(signals: List[MarketSignal]) -> dict:
    """Group signals by ticker symbol, sorted by max strength."""
    groups = {}
    for sig in signals:
        if sig.symbol:
            if sig.symbol not in groups:
                groups[sig.symbol] = []
            groups[sig.symbol].append(sig)

    # Sort each group by strength descending
    for sym in groups:
        groups[sym].sort(key=lambda s: s.strength, reverse=True)

    # Sort groups by max strength of their signals
    sorted_groups = dict(
        sorted(groups.items(), key=lambda kv: kv[1][0].strength, reverse=True)
    )
    return sorted_groups


def write_report(
    signals: List[MarketSignal],
    analysis: dict,
    results: dict,
    outcome_stats: dict = None,
) -> Path:
    """Write markdown intelligence report. Returns path to report."""
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d-%H%M") + ".md"
    report_path = SCANS_DIR / filename

    summary = analysis["summary"]
    alerts = analysis["alerts"]
    ticker_alerts = analysis.get("ticker_alerts", [])
    domain_status = analysis["domain_status"]

    source_counts = count_by_source(signals)
    symbol_groups = group_by_symbol(signals)

    lines = []
    lines.append(f"# MIDGE Intelligence Report — {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    rec = summary.get("recommendation", "HOLD")
    conf = summary.get("confidence", 0.0)
    reasoning = summary.get("reasoning", "Insufficient data for recommendation.")
    lines.append(f"**Recommendation:** {rec} (confidence: {conf:.0%})")
    lines.append(f"**Reasoning:** {reasoning}")
    lines.append("")

    bullish = summary.get("bullish_domains", [])
    bearish = summary.get("bearish_domains", [])
    if bullish:
        lines.append(f"Bullish domains: {', '.join(bullish)}")
    if bearish:
        lines.append(f"Bearish domains: {', '.join(bearish)}")
    lines.append(f"Total signals: {summary.get('total_signals', len(signals))}")
    lines.append("")

    # Convergence Alerts
    lines.append("## Convergence Alerts")
    lines.append("")
    if alerts:
        for alert in alerts:
            lines.append(f"### {alert.direction.upper()} — {alert.urgency} urgency")
            lines.append(f"- Strength: {alert.strength:.2f} | Confidence: {alert.confidence:.2f}")
            lines.append(f"- Domains: {', '.join(alert.domains_converging)}")
            lines.append(f"- Signals: {len(alert.signals)} across {alert.cross_domain_count} domains")
            lines.append(f"- Summary: {alert.summary}")
            lines.append("")
    else:
        lines.append("No convergence alerts triggered.")
        lines.append("")

    # Ticker-Level Convergence
    lines.append("## Ticker-Level Convergence")
    lines.append("")
    if ticker_alerts:
        for alert in ticker_alerts:
            lines.append(f"### {alert.summary}")
            lines.append(f"- Strength: {alert.strength:.2f} | Confidence: {alert.confidence:.2f}")
            lines.append(f"- Domains: {', '.join(alert.domains_converging)}")
            lines.append(f"- Urgency: {alert.urgency}")
            lines.append("")
    else:
        lines.append("No per-ticker convergence detected.")
        lines.append("")

    # Signal Counts by Source
    lines.append("## Signal Counts by Source")
    lines.append("")
    lines.append("| Source | Count |")
    lines.append("|--------|-------|")
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")
    lines.append(f"| **Total** | **{len(signals)}** |")
    lines.append("")

    # Domain Status
    lines.append("## Domain Status")
    lines.append("")
    if domain_status:
        lines.append("| Domain | Direction | Strength | Signals |")
        lines.append("|--------|-----------|----------|---------|")
        for domain, status in sorted(domain_status.items()):
            d = status.get("direction", "neutral")
            s = status.get("strength", 0)
            c = status.get("signal_count", 0)
            lines.append(f"| {domain} | {d} | {s:.2f} | {c} |")
    else:
        lines.append("No domain data.")
    lines.append("")

    # Signals by Symbol
    lines.append("## Signals by Symbol")
    lines.append("")
    for symbol, sigs in symbol_groups.items():
        price_data = results["prices"].get(symbol)
        price_str = f" — ${price_data.price:.2f}" if price_data else ""
        lines.append(f"### {symbol}{price_str}")
        lines.append("")
        for sig in sigs[:5]:  # Top 5 per symbol
            meta_summary = ""
            if sig.source == "sec_form4":
                name = sig.metadata.get("filer_name", "")
                val = sig.metadata.get("total_value", 0)
                meta_summary = f" ({name}, ${val:,.0f})" if name else ""
            elif sig.source == "congressional":
                rep = sig.metadata.get("representative", "")
                amt = sig.metadata.get("amount_range", "")
                meta_summary = f" ({rep}, {amt})" if rep else ""
            lines.append(
                f"- [{sig.direction}] {sig.source} "
                f"str={sig.strength:.2f} conf={sig.confidence:.2f}"
                f"{meta_summary}"
            )
        lines.append("")

    # Outcome Tracking
    if outcome_stats:
        lines.append("## Outcome Tracking (Thompson Feedback Loop)")
        lines.append("")
        lines.append(f"- Signals tracked: {outcome_stats.get('registered_signals', 0)}")
        lines.append(f"- Pending predictions: {outcome_stats.get('pending_predictions', 0)}")
        lines.append(f"- Total evaluated: {outcome_stats.get('total_evaluated', 0)}")
        lines.append(f"- Success threshold: {outcome_stats.get('success_threshold_pct', 5.0)}% price move")
        lines.append("")

    # Prices snapshot
    if results["prices"]:
        lines.append("## Price Snapshot")
        lines.append("")
        lines.append("| Symbol | Price | Source |")
        lines.append("|--------|-------|--------|")
        for symbol in sorted(results["prices"].keys()):
            p = results["prices"][symbol]
            lines.append(f"| {symbol} | ${p.price:.2f} | {p.source} |")
        lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    return report_path


def print_summary(signals: List[MarketSignal], analysis: dict, report_path: Path, outcome_stats: dict = None):
    """Print concise summary to stdout."""
    summary = analysis["summary"]
    alerts = analysis["alerts"]
    source_counts = count_by_source(signals)

    print()
    print("=" * 60)
    print("MIDGE SCAN COMPLETE")
    print("=" * 60)
    print()
    print(f"  Signals collected: {len(signals)}")
    for source, count in sorted(source_counts.items()):
        print(f"    {source}: {count}")
    print()
    print(f"  Recommendation: {summary.get('recommendation', 'N/A')}")
    print(f"  Confidence: {summary.get('confidence', 0):.0%}")
    print(f"  Reasoning: {summary.get('reasoning', 'N/A')}")
    print()

    if alerts:
        print(f"  CONVERGENCE ALERTS: {len(alerts)}")
        for alert in alerts:
            print(f"    [{alert.urgency}] {alert.direction} — {alert.summary}")
    else:
        print("  No convergence alerts.")

    ticker_alerts = analysis.get("ticker_alerts", [])
    if ticker_alerts:
        print()
        print(f"  TICKER CONVERGENCE: {len(ticker_alerts)}")
        for alert in ticker_alerts:
            print(f"    [{alert.urgency}] {alert.summary}")

    if outcome_stats:
        print()
        print(f"  OUTCOME TRACKING: {outcome_stats.get('registered_signals', 0)} tracked, "
              f"{outcome_stats.get('pending_predictions', 0)} pending, "
              f"{outcome_stats.get('total_evaluated', 0)} evaluated")

    print()
    print(f"  Report: {report_path}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIDGE Market Intelligence Scan")
    parser.add_argument("--dry-run", action="store_true", help="Skip Qdrant storage")
    parser.add_argument("--watchlist", type=str, help="Path to watchlist JSON")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("MIDGE Market Intelligence Scan")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Phase 1: Setup
    logger.info("Phase 1/7: Setup")
    clients, alerter, memory, watchlist, velocity_detector, filing_analyzer, outcome_collector = setup(args)

    # Phase 2: Fetch
    logger.info("Phase 2/7: Fetching live data from all sources...")
    results = fetch_all(clients, watchlist)

    # Phase 3: Convert
    logger.info("Phase 3/7: Converting to MarketSignals...")
    signals = convert_to_signals(results)
    logger.info(f"  {len(signals)} signals created")

    if not signals:
        print("\nNo signals collected. Check API keys and network connectivity.")
        print("Run test_live_apis.py to diagnose individual sources.")
        sys.exit(1)

    # Phase 4: Store + Feed
    logger.info("Phase 4/7: Storing signals and feeding convergence engine...")
    store_and_feed(
        signals, alerter, memory,
        dry_run=args.dry_run,
        velocity_detector=velocity_detector,
        filing_analyzer=filing_analyzer,
    )

    # Phase 5: Outcome tracking (Thompson feedback loop)
    logger.info("Phase 5/7: Registering predictions and evaluating outcomes...")
    registered = outcome_collector.register_signals(signals)
    evaluated = outcome_collector.evaluate()
    if registered or evaluated:
        logger.info(f"  Registered: {registered} new predictions, Evaluated: {evaluated} matured outcomes")
    outcome_stats = outcome_collector.get_statistics()

    # Phase 6: Analyze
    logger.info("Phase 6/7: Running convergence analysis...")
    analysis = analyze(alerter)

    # Phase 7: Report
    logger.info("Phase 7/7: Writing intelligence report...")
    report_path = write_report(signals, analysis, results, outcome_stats)

    print_summary(signals, analysis, report_path, outcome_stats)


if __name__ == "__main__":
    main()
