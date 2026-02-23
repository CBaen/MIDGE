#!/usr/bin/env python3
"""Live API connectivity test for MIDGE.

Tests each data source with real keys from .env.
Run from project root: python test_live_apis.py

This is NOT a pytest file — it makes real API calls.
"""

import os
import sys
from pathlib import Path

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")


def test_sec_edgar():
    """SEC EDGAR — free, no key needed."""
    print("\n[1/6] SEC EDGAR (free, no key)")
    try:
        from mae_core.market.apis.sec_edgar.client import SECEdgarClient
        client = SECEdgarClient()
        # Look up Apple's CIK
        cik = client.get_company_cik("AAPL")
        if cik:
            print(f"  OK — AAPL CIK: {cik}")
            return True
        else:
            print("  FAIL — could not resolve AAPL CIK")
            return False
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def test_yfinance():
    """Yahoo Finance — free, no key needed."""
    print("\n[2/6] Yahoo Finance (free, no key)")
    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher, YFINANCE_AVAILABLE
        if not YFINANCE_AVAILABLE:
            print("  SKIP — yfinance not installed (pip install yfinance)")
            return None
        fetcher = PriceFetcher()
        price = fetcher.get_current_price("AAPL")
        if price:
            print(f"  OK — AAPL: ${price.price:.2f} (source: {price.source})")
            return True
        else:
            print("  FAIL — no price returned")
            return False
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def test_alpha_vantage():
    """Alpha Vantage — needs MAE_ALPHAVANTAGE_API_KEY."""
    print("\n[3/6] Alpha Vantage")
    key = os.environ.get("MAE_ALPHAVANTAGE_API_KEY", "")
    if not key:
        print("  SKIP — MAE_ALPHAVANTAGE_API_KEY not set")
        return None
    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher
        fetcher = PriceFetcher(alpha_vantage_key=key)
        price = fetcher._fetch_alpha_vantage("MSFT")
        if price:
            print(f"  OK — MSFT: ${price.price:.2f} (source: {price.source})")
            return True
        else:
            print("  FAIL — no price returned (may be rate limited, 5 calls/min on free tier)")
            return False
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def test_congressional_trades():
    """Congressional trades — needs RAPIDAPI_KEY."""
    print("\n[4/6] Congressional Trades (RapidAPI)")
    key = os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        print("  SKIP — RAPIDAPI_KEY not set")
        return None
    try:
        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
        client = HouseStockWatcherClient(rapidapi_key=key)
        trades = client.get_recent_trades(days=30)
        if trades:
            print(f"  OK — {len(trades)} recent trades found")
            for t in trades[:2]:
                print(f"       {t.to_plain_language()}")
            return True
        else:
            print("  WARN — 0 trades returned (API may be down or subscription inactive)")
            return False
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def test_job_tracker():
    """Job tracker — needs RAPIDAPI_KEY."""
    print("\n[5/6] Job Tracker (RapidAPI)")
    key = os.environ.get("RAPIDAPI_KEY", "")
    if not key:
        print("  SKIP — RAPIDAPI_KEY not set")
        return None
    try:
        from mae_core.market.apis.job_tracker import JobTracker
        tracker = JobTracker(rapidapi_key=key)
        signal = tracker.analyze_hiring_activity("Lockheed Martin", ticker="LMT")
        print(f"  OK — {signal.to_plain_language()}")
        return True
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def test_usa_spending():
    """USASpending — free, no key needed."""
    print("\n[6/6] USASpending.gov (free, no key)")
    try:
        from mae_core.market.apis.usa_spending import USASpendingClient
        client = USASpendingClient()
        contracts = client.search_contracts(keyword="cybersecurity", limit=3)
        if contracts:
            print(f"  OK — {len(contracts)} contracts found")
            for c in contracts[:2]:
                print(f"       {c.to_plain_language()}")
            return True
        else:
            print("  WARN — 0 contracts returned")
            return False
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MIDGE Live API Connectivity Test")
    print("=" * 60)

    results = {}
    results["SEC EDGAR"] = test_sec_edgar()
    results["Yahoo Finance"] = test_yfinance()
    results["Alpha Vantage"] = test_alpha_vantage()
    results["Congressional Trades"] = test_congressional_trades()
    results["Job Tracker"] = test_job_tracker()
    results["USASpending"] = test_usa_spending()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results.items():
        status = "OK" if result else ("SKIP" if result is None else "FAIL")
        icon = {"OK": "+", "SKIP": "-", "FAIL": "!"}[status]
        print(f"  [{icon}] {name}: {status}")

    ok = sum(1 for v in results.values() if v is True)
    fail = sum(1 for v in results.values() if v is False)
    skip = sum(1 for v in results.values() if v is None)
    print(f"\n  {ok} passed, {fail} failed, {skip} skipped")
