#!/usr/bin/env python3
"""backfill_archives.py — Populate MIDGE signal archives with historical data.

Downloads 90 days of market signals from all available free data sources:
  - SEC EDGAR (Form 4 insider trades, EFTS keyword filings)
  - Congressional trades (House + Senate)
  - USASpending government contracts
  - FINRA daily short volume (institutional flow)
  - FRED macro indicators (yield curve, VIX, credit spreads, rates, employment)
  - Finnhub earnings surprises
  - yfinance price/volume (significant daily moves)

Converts them through the same signal.py adapters the live system uses,
and writes them to data/midge/signals/YYYY-MM-DD.jsonl — one file per day.

This feeds the lag-correlation analyzer, Thompson calibrator, and Kelly
position sizer with broad multi-domain temporal data.

Usage:
    python backfill_archives.py                    # Default: 90 days, all sources
    python backfill_archives.py --days 180         # Override lookback
    python backfill_archives.py --sources sec,finra,price  # Subset of sources
    python backfill_archives.py --dry-run          # Count signals, don't write
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("midge.backfill")

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
SIGNALS_DIR = PROJECT_ROOT / "data" / "midge" / "signals"
WATCHLIST_PATH = PROJECT_ROOT / "data" / "midge" / "watchlist.json"

# ── Ticker list for SEC Form 4 backfill ───────────────────────────────────────

BACKFILL_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "CRM", "ORCL",
    # Defense
    "LMT", "RTX", "NOC", "GD", "BA",
    # Finance
    "JPM", "GS", "BAC", "WFC",
    # Pharma
    "JNJ", "PFE", "UNH",
    # Energy
    "XOM", "CVX",
    # Semiconductors
    "INTC", "AMD", "MU", "AVGO",
    # Retail
    "WMT", "COST",
    # Cybersecurity
    "PANW", "CRWD",
]

ALL_SOURCES = [
    "sec", "congress", "contracts", "efts", "finra", "macro", "earnings", "price",
    "cot", "vix", "stocktwits", "trends", "finnhub_economic", "finnhub_analyst", "finnhub_earnings_cal",
]


# ── Signal serialization (matches sensing_hook._store_signals) ────────────────

def _serialize_signal(sig) -> dict:
    """Convert MarketSignal to the JSONL record format the archive reader expects."""
    return {
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


# ── Phase 1: Fetch ───────────────────────────────────────────────────────────

def fetch_sec_form4(tickers: List[str], days: int) -> list:
    """Fetch Form 4 insider trades for all tickers."""
    try:
        from mae_core.market.apis.sec_edgar import get_recent_form4s
    except ImportError:
        logger.warning("SEC EDGAR client not available, skipping Form 4")
        return []

    all_trades = []
    for i, ticker in enumerate(tickers):
        logger.info("  SEC Form 4: %s (%d/%d)", ticker, i + 1, len(tickers))
        try:
            trades = get_recent_form4s(ticker, days=days)
            all_trades.extend(trades)
            logger.info("    -> %d trades", len(trades))
        except Exception as e:
            logger.warning("    -> FAILED: %s", e)
    return all_trades


def fetch_house_trades(days: int) -> list:
    """Fetch House congressional trades."""
    try:
        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
    except ImportError:
        logger.warning("House Stock Watcher client not available, skipping")
        return []

    try:
        client = HouseStockWatcherClient()
        trades = client.get_recent_trades(days=days)
        logger.info("  House trades: %d", len(trades))
        return trades
    except Exception as e:
        logger.warning("  House trades FAILED: %s", e)
        return []


def fetch_senate_trades(days: int) -> list:
    """Fetch Senate stock trades."""
    try:
        from mae_core.market.apis.senate_stock_watcher import SenateStockWatcherClient
    except ImportError:
        logger.warning("Senate Stock Watcher client not available, skipping")
        return []

    try:
        client = SenateStockWatcherClient()
        trades = client.get_recent_trades(days=days)
        logger.info("  Senate trades: %d", len(trades))
        return trades
    except Exception as e:
        logger.warning("  Senate trades FAILED: %s", e)
        return []


def fetch_contracts(days: int) -> list:
    """Fetch large government contracts from USASpending.gov."""
    try:
        from mae_core.market.apis.usa_spending import USASpendingClient
    except ImportError:
        logger.warning("USASpending client not available, skipping")
        return []

    try:
        client = USASpendingClient()
        contracts = client.get_recent_large_contracts(days=days)
        logger.info("  Government contracts: %d", len(contracts))
        return contracts
    except Exception as e:
        logger.warning("  Government contracts FAILED: %s", e)
        return []


def fetch_efts_keywords(days: int) -> list:
    """Fetch SEC EFTS keyword filing hits."""
    try:
        from mae_core.market.apis.sec_edgar.efts import SECFullTextSearchClient
    except ImportError:
        logger.warning("SEC EFTS client not available, skipping")
        return []

    try:
        client = SECFullTextSearchClient()
        hits = client.scan_all_keywords(days=days)
        logger.info("  EFTS keyword hits: %d", len(hits))
        return hits
    except Exception as e:
        logger.warning("  EFTS keyword scan FAILED: %s", e)
        return []


def fetch_finra_short(tickers: List[str], days: int) -> list:
    """Fetch FINRA daily short volume for watchlist tickers over date range."""
    try:
        from mae_core.market.apis.finra_short_interest import FINRAShortInterestClient
    except ImportError:
        logger.warning("FINRA client not available, skipping short interest")
        return []

    try:
        client = FINRAShortInterestClient()
        ticker_set = set(t.upper() for t in tickers)
        all_records = []

        # Iterate trading days and filter for our tickers
        current = datetime.now()
        end_date = current
        start_date = current - timedelta(days=days)
        check_date = end_date

        days_fetched = 0
        while check_date >= start_date:
            # Skip weekends
            if check_date.weekday() >= 5:
                check_date -= timedelta(days=1)
                continue

            date_str = check_date.strftime("%Y-%m-%d")
            day_records = client.get_daily_short_volume(date_str)

            if day_records:
                matched = [r for r in day_records if r.symbol in ticker_set]
                all_records.extend(matched)
                days_fetched += 1

                if days_fetched % 10 == 0:
                    logger.info("  FINRA short: %d trading days fetched, %d records so far",
                                days_fetched, len(all_records))

            check_date -= timedelta(days=1)

        logger.info("  FINRA short interest: %d records across %d trading days",
                     len(all_records), days_fetched)
        return all_records
    except Exception as e:
        logger.warning("  FINRA short interest FAILED: %s", e)
        return []


def fetch_macro_history(days: int) -> list:
    """Fetch FRED macro indicator history (yield curve, VIX, credit spreads, etc.)."""
    if not os.environ.get("FRED_API_KEY"):
        logger.warning("  FRED_API_KEY not set, skipping macro indicators")
        return []

    try:
        from mae_core.market.apis.fred_client import FREDClient
    except ImportError:
        logger.warning("FRED client not available, skipping macro")
        return []

    try:
        client = FREDClient()
        series_ids = ["T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "DFF", "UNRATE"]
        all_indicators = []

        for series_id in series_ids:
            indicators = client.get_historical_series(series_id, days=days)
            all_indicators.extend(indicators)
            logger.info("  FRED %s: %d observations", series_id, len(indicators))

        logger.info("  FRED macro total: %d indicators", len(all_indicators))
        return all_indicators
    except Exception as e:
        logger.warning("  FRED macro FAILED: %s", e)
        return []


def fetch_earnings_history(days: int) -> list:
    """Fetch Finnhub reported earnings surprises."""
    if not os.environ.get("MAE_FINNHUB_API_KEY"):
        logger.warning("  MAE_FINNHUB_API_KEY not set, skipping earnings")
        return []

    try:
        from mae_core.market.apis.finnhub_client import FinnhubClient
    except ImportError:
        logger.warning("Finnhub client not available, skipping earnings")
        return []

    try:
        client = FinnhubClient()
        events = client.get_recent_earnings_surprises(days=days)
        logger.info("  Finnhub earnings: %d reported events", len(events))
        return events
    except Exception as e:
        logger.warning("  Finnhub earnings FAILED: %s", e)
        return []


def fetch_price_history(tickers: List[str], days: int) -> list:
    """Fetch daily OHLCV history for all tickers via yfinance."""
    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher
    except ImportError:
        logger.warning("PriceFetcher not available, skipping price history")
        return []

    try:
        fetcher = PriceFetcher()
        all_prices = []

        for i, ticker in enumerate(tickers):
            logger.info("  Price history: %s (%d/%d)", ticker, i + 1, len(tickers))
            try:
                prices = fetcher.get_daily_history(ticker, days=days)
                all_prices.extend(prices)
            except Exception as e:
                logger.warning("    -> FAILED: %s", e)

        logger.info("  Price history total: %d daily records", len(all_prices))
        return all_prices
    except Exception as e:
        logger.warning("  Price history FAILED: %s", e)
        return []


# ── Layer 6 Sources ──────────────────────────────────────────────────────────

def fetch_cot_history(days: int) -> list:
    """Fetch CFTC COT positioning for all mapped futures contracts (all weekly rows)."""
    try:
        from mae_core.market.apis.cot_client import COTClient
    except ImportError:
        logger.warning("COT client not available, skipping")
        return []

    try:
        client = COTClient()
        # Fetch current year + previous year if days > 365
        current_year = datetime.now().year
        years = [current_year]
        if days > 365:
            years = [current_year - 1, current_year]
        elif days > 30:
            years = [current_year - 1, current_year]  # Always get both for coverage

        positions = client.get_all_positions(years=years)
        # Filter to requested window
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        filtered = [p for p in positions if p.report_date >= cutoff]
        logger.info("  COT positions: %d weekly reports (filtered from %d)", len(filtered), len(positions))
        return filtered
    except Exception as e:
        logger.warning("  COT history FAILED: %s", e)
        return []


def fetch_vix_history(days: int) -> list:
    """Fetch CBOE VIX daily history."""
    try:
        from mae_core.market.apis.vix_client import VIXClient
    except ImportError:
        logger.warning("VIX client not available, skipping")
        return []

    try:
        client = VIXClient()
        signals = client.get_vix_history(days=days)
        logger.info("  VIX history: %d daily readings", len(signals))
        return signals
    except Exception as e:
        logger.warning("  VIX history FAILED: %s", e)
        return []


def fetch_stocktwits_sentiment(tickers: List[str]) -> list:
    """Fetch StockTwits sentiment (snapshot only — no historical API)."""
    try:
        from mae_core.market.apis.stocktwits_client import StockTwitsClient
    except ImportError:
        logger.warning("StockTwits client not available, skipping")
        return []

    try:
        client = StockTwitsClient()
        sentiment = client.get_sentiment(tickers[:20])  # Limit to avoid rate limiting
        logger.info("  StockTwits sentiment: %d tickers", len(sentiment))
        return sentiment
    except Exception as e:
        logger.warning("  StockTwits sentiment FAILED: %s", e)
        return []


def fetch_trends(days: int) -> list:
    """Fetch Google Trends search interest (snapshot only — current 7d window)."""
    try:
        from mae_core.market.apis.trends_client import TrendsClient
    except ImportError:
        logger.warning("Trends client not available, skipping")
        return []

    try:
        client = TrendsClient()
        signals = client.get_interest()
        logger.info("  Google Trends: %d keyword signals", len(signals))
        return signals
    except Exception as e:
        logger.warning("  Google Trends FAILED: %s", e)
        return []


def fetch_finnhub_economic(days: int) -> list:
    """Fetch Finnhub economic calendar events."""
    if not os.environ.get("MAE_FINNHUB_API_KEY"):
        logger.warning("  MAE_FINNHUB_API_KEY not set, skipping economic calendar")
        return []

    try:
        from mae_core.market.apis.finnhub_client import FinnhubClient
    except ImportError:
        logger.warning("Finnhub client not available, skipping economic")
        return []

    try:
        client = FinnhubClient()
        events = client.get_economic_calendar(days=min(days, 90))
        logger.info("  Finnhub economic calendar: %d events", len(events))
        return events
    except Exception as e:
        logger.warning("  Finnhub economic calendar FAILED: %s", e)
        return []


def fetch_finnhub_analyst(tickers: List[str]) -> list:
    """Fetch Finnhub analyst recommendations for all tickers."""
    if not os.environ.get("MAE_FINNHUB_API_KEY"):
        logger.warning("  MAE_FINNHUB_API_KEY not set, skipping analyst recs")
        return []

    try:
        from mae_core.market.apis.finnhub_client import FinnhubClient
    except ImportError:
        logger.warning("Finnhub client not available, skipping analyst")
        return []

    try:
        client = FinnhubClient()
        all_recs = []
        for i, ticker in enumerate(tickers):
            logger.info("  Finnhub analyst: %s (%d/%d)", ticker, i + 1, len(tickers))
            try:
                recs = client.get_analyst_recommendations(ticker)
                all_recs.extend(recs)
            except Exception as e:
                logger.warning("    -> FAILED: %s", e)
        logger.info("  Finnhub analyst recs total: %d", len(all_recs))
        return all_recs
    except Exception as e:
        logger.warning("  Finnhub analyst FAILED: %s", e)
        return []


def fetch_finnhub_earnings_calendar(days: int) -> list:
    """Fetch Finnhub earnings calendar (past + upcoming)."""
    if not os.environ.get("MAE_FINNHUB_API_KEY"):
        logger.warning("  MAE_FINNHUB_API_KEY not set, skipping earnings calendar")
        return []

    try:
        from mae_core.market.apis.finnhub_client import FinnhubClient
    except ImportError:
        logger.warning("Finnhub client not available, skipping earnings calendar")
        return []

    try:
        client = FinnhubClient()
        events = client.get_earnings_calendar(days=min(days, 90))
        logger.info("  Finnhub earnings calendar: %d events", len(events))
        return events
    except Exception as e:
        logger.warning("  Finnhub earnings calendar FAILED: %s", e)
        return []


# ── Phase 2: Convert ─────────────────────────────────────────────────────────

def convert_all(
    form4_trades: list,
    house_trades: list,
    senate_trades: list,
    contracts: list,
    efts_hits: list,
    short_interest: list = None,
    macro_indicators: list = None,
    earnings_events: list = None,
    price_history: list = None,
    cot_positions: list = None,
    vix_readings: list = None,
    stocktwits: list = None,
    trends: list = None,
    finnhub_econ: list = None,
    finnhub_recs: list = None,
    finnhub_earn_cal: list = None,
) -> list:
    """Convert raw API results to MarketSignal objects, deduped by signal_id."""
    from mae_core.market.signal import (
        from_insider_trade,
        from_congressional_trade,
        from_senate_trade,
        from_government_contract,
        from_filing_keyword,
        from_short_interest,
        from_macro_indicator,
        from_earnings_event,
        from_price_data,
        from_cot_positioning,
        from_vix_structure,
        from_stocktwits_sentiment,
        from_trends_signal,
        from_economic_event,
        from_analyst_recommendation,
    )

    signals = []
    seen_ids: Set[str] = set()

    def _add(sig):
        if sig.signal_id not in seen_ids:
            seen_ids.add(sig.signal_id)
            signals.append(sig)

    for trade in form4_trades:
        try:
            _add(from_insider_trade(trade))
        except Exception:
            pass

    for trade in house_trades:
        try:
            _add(from_congressional_trade(trade))
        except Exception:
            pass

    for trade in senate_trades:
        try:
            _add(from_senate_trade(trade))
        except Exception:
            pass

    for contract in contracts:
        try:
            _add(from_government_contract(contract))
        except Exception:
            pass

    for hit in efts_hits:
        try:
            _add(from_filing_keyword(hit))
        except Exception:
            pass

    for si in (short_interest or []):
        try:
            _add(from_short_interest(si))
        except Exception:
            pass

    for mi in (macro_indicators or []):
        try:
            _add(from_macro_indicator(mi))
        except Exception:
            pass

    for ee in (earnings_events or []):
        try:
            _add(from_earnings_event(ee))
        except Exception:
            pass

    for pd in (price_history or []):
        try:
            sig = from_price_data(pd)
            if sig is not None:
                _add(sig)
        except Exception:
            pass

    for pos in (cot_positions or []):
        try:
            _add(from_cot_positioning(pos))
        except Exception:
            pass

    for vix in (vix_readings or []):
        try:
            _add(from_vix_structure(vix))
        except Exception:
            pass

    for st in (stocktwits or []):
        try:
            _add(from_stocktwits_sentiment(st))
        except Exception:
            pass

    for tr in (trends or []):
        try:
            _add(from_trends_signal(tr))
        except Exception:
            pass

    for ev in (finnhub_econ or []):
        try:
            _add(from_economic_event(ev))
        except Exception:
            pass

    for rec in (finnhub_recs or []):
        try:
            _add(from_analyst_recommendation(rec))
        except Exception:
            pass

    # Finnhub earnings calendar uses the same converter as earnings surprises
    for ee in (finnhub_earn_cal or []):
        try:
            _add(from_earnings_event(ee))
        except Exception:
            pass

    return signals


# ── Phase 3: Write ───────────────────────────────────────────────────────────

def load_existing_ids(path: Path) -> Set[str]:
    """Load signal_ids already in a JSONL file for dedup."""
    ids = set()
    if path.exists():
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        sid = record.get("signal_id")
                        if sid:
                            ids.add(sid)
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
    return ids


def write_signals(signals: list, dry_run: bool = False) -> Dict[str, int]:
    """Write signals to daily JSONL files. Returns {date_str: count} written."""
    # Group by event date
    by_date: Dict[str, list] = defaultdict(list)
    for sig in signals:
        day_str = sig.timestamp.strftime("%Y-%m-%d")
        by_date[day_str].append(sig)

    if dry_run:
        logger.info("DRY RUN — would write %d signals across %d days",
                     len(signals), len(by_date))
        return {d: len(sigs) for d, sigs in by_date.items()}

    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    written_counts: Dict[str, int] = {}

    for day_str in sorted(by_date.keys()):
        day_signals = by_date[day_str]
        path = SIGNALS_DIR / f"{day_str}.jsonl"

        # Load existing IDs for idempotent append
        existing_ids = load_existing_ids(path)
        new_signals = [s for s in day_signals if s.signal_id not in existing_ids]

        if not new_signals:
            continue

        with open(path, "a") as f:
            for sig in new_signals:
                record = _serialize_signal(sig)
                f.write(json.dumps(record) + "\n")

        written_counts[day_str] = len(new_signals)

    return written_counts


# ── Main ──────────────────────────────────────────────────────────────────────

def get_tickers() -> List[str]:
    """Load watchlist tickers and merge with BACKFILL_TICKERS."""
    tickers = list(BACKFILL_TICKERS)
    if WATCHLIST_PATH.exists():
        try:
            watchlist = json.loads(WATCHLIST_PATH.read_text())
            for t in watchlist.get("tickers", []):
                if t not in tickers:
                    tickers.append(t)
        except Exception:
            pass
    return tickers


def main():
    parser = argparse.ArgumentParser(description="Backfill MIDGE signal archives")
    parser.add_argument("--days", type=int, default=90, help="Days of history (default: 90)")
    parser.add_argument("--sources", type=str, default="all",
                        help="Comma-separated sources: sec,congress,contracts,efts,finra,macro,earnings,price (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Count signals without writing")
    args = parser.parse_args()

    sources = ALL_SOURCES if args.sources == "all" else [s.strip() for s in args.sources.split(",")]

    logger.info("=" * 60)
    logger.info("MIDGE Archive Backfill — %d days, sources: %s", args.days, sources)
    logger.info("=" * 60)

    tickers = get_tickers()
    logger.info("Tickers for SEC Form 4: %d (%s...)", len(tickers), ", ".join(tickers[:5]))

    # Phase 1: Fetch
    logger.info("")
    logger.info("Phase 1: Fetching from %d sources...", len(sources))

    form4_trades = []
    house_trades = []
    senate_trades = []
    contracts = []
    efts_hits = []
    short_interest = []
    macro_indicators = []
    earnings_events = []
    price_history = []
    cot_positions = []
    vix_readings = []
    stocktwits_data = []
    trends_data = []
    finnhub_econ = []
    finnhub_recs = []
    finnhub_earn_cal = []

    if "sec" in sources:
        form4_trades = fetch_sec_form4(tickers, args.days)

    if "congress" in sources:
        house_trades = fetch_house_trades(args.days)
        senate_trades = fetch_senate_trades(args.days)

    if "contracts" in sources:
        contracts = fetch_contracts(args.days)

    if "efts" in sources:
        efts_hits = fetch_efts_keywords(args.days)

    if "finra" in sources:
        short_interest = fetch_finra_short(tickers, args.days)

    if "macro" in sources:
        macro_indicators = fetch_macro_history(args.days)

    if "earnings" in sources:
        earnings_events = fetch_earnings_history(args.days)

    if "price" in sources:
        price_history = fetch_price_history(tickers, args.days)

    if "cot" in sources:
        cot_positions = fetch_cot_history(args.days)

    if "vix" in sources:
        vix_readings = fetch_vix_history(args.days)

    if "stocktwits" in sources:
        stocktwits_data = fetch_stocktwits_sentiment(tickers)

    if "trends" in sources:
        trends_data = fetch_trends(args.days)

    if "finnhub_economic" in sources:
        finnhub_econ = fetch_finnhub_economic(args.days)

    if "finnhub_analyst" in sources:
        finnhub_recs = fetch_finnhub_analyst(tickers)

    if "finnhub_earnings_cal" in sources:
        finnhub_earn_cal = fetch_finnhub_earnings_calendar(args.days)

    raw_total = (len(form4_trades) + len(house_trades) + len(senate_trades)
                 + len(contracts) + len(efts_hits) + len(short_interest)
                 + len(macro_indicators) + len(earnings_events) + len(price_history)
                 + len(cot_positions) + len(vix_readings) + len(stocktwits_data)
                 + len(trends_data) + len(finnhub_econ) + len(finnhub_recs)
                 + len(finnhub_earn_cal))
    logger.info("")
    logger.info("Phase 1 complete: %d raw records fetched", raw_total)

    if raw_total == 0:
        logger.warning("No data fetched from any source. Check network connectivity.")
        return

    # Phase 2: Convert
    logger.info("")
    logger.info("Phase 2: Converting to MarketSignal format...")
    signals = convert_all(
        form4_trades, house_trades, senate_trades, contracts, efts_hits,
        short_interest=short_interest,
        macro_indicators=macro_indicators,
        earnings_events=earnings_events,
        price_history=price_history,
    )
    logger.info("Phase 2 complete: %d unique signals (deduped from %d raw)", len(signals), raw_total)

    # Source breakdown
    source_counts: Dict[str, int] = defaultdict(int)
    for sig in signals:
        source_counts[sig.source] += 1
    for src, count in sorted(source_counts.items()):
        logger.info("  %s: %d signals", src, count)

    # Phase 3: Write
    logger.info("")
    logger.info("Phase 3: Writing to daily JSONL files...")
    written = write_signals(signals, dry_run=args.dry_run)

    total_written = sum(written.values())
    logger.info("")
    logger.info("=" * 60)
    logger.info("Backfill complete: %d signals written across %d days", total_written, len(written))
    logger.info("Archive directory: %s", SIGNALS_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
