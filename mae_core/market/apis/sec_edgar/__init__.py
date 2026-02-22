"""
SEC EDGAR API Package.

Scrapes insider trading data (Form 4) and material events (Form 8-K).
Free data source - no API key required.
Respects SEC's rate limiting (10 requests/second max).

Usage:
    from mae_core.market.apis.sec_edgar import get_recent_form4s, get_recent_form8ks

    trades = get_recent_form4s("AAPL", days=30)
    events = get_recent_form8ks("AAPL", days=30)
"""

from datetime import datetime, timedelta
from typing import List

from .models import Form8KEvent, InsiderTrade
from .client import SECEdgarClient, SEC_USER_AGENT


def get_recent_form8ks(ticker: str, days: int = 30) -> List[Form8KEvent]:
    """
    Get recent Form 8-K filings for a ticker.

    Form 8-K is the "current report" - filed within 4 business days
    of material events. These often PRECEDE insider trading activity.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        days: Number of days to look back

    Returns:
        List of Form8KEvent objects
    """
    client = SECEdgarClient()

    cik = client.get_company_cik(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return []

    print(f"Found CIK {cik} for {ticker}")

    filings = client.get_company_filings(cik, form_type="8-K")
    print(f"Found {len(filings)} Form 8-K filings")

    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent_filings = [f for f in filings if f.get("filing_date", "") >= cutoff_date]
    print(f"Found {len(recent_filings)} filings in last {days} days")

    all_events = []
    for filing in recent_filings[:15]:
        events = client.parse_form8k(
            cik,
            filing["accession_number"],
            filing_date=filing.get("filing_date", ""),
            company_name=filing.get("company_name", ""),
            ticker_symbol=ticker,
            document_url=filing.get("document_url")
        )
        all_events.extend(events)

    return all_events


def get_recent_form4s(ticker: str, days: int = 30) -> List[InsiderTrade]:
    """
    Get recent Form 4 filings for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        days: Number of days to look back

    Returns:
        List of InsiderTrade objects
    """
    client = SECEdgarClient()

    cik = client.get_company_cik(ticker)
    if not cik:
        print(f"Could not find CIK for {ticker}")
        return []

    print(f"Found CIK {cik} for {ticker}")

    filings = client.get_company_filings(cik, form_type="4")
    print(f"Found {len(filings)} Form 4 filings")

    cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent_filings = [f for f in filings if f.get("filing_date", "") >= cutoff_date]
    print(f"Found {len(recent_filings)} filings in last {days} days")

    all_trades = []
    for filing in recent_filings[:10]:
        trades = client.parse_form4(
            cik,
            filing["accession_number"],
            document_url=filing.get("document_url")
        )
        for trade in trades:
            trade.filing_date = filing.get("filing_date", "")
        all_trades.extend(trades)

    return all_trades


def search_politician_trades(politician_name: str = None) -> List[InsiderTrade]:
    """
    Search for trades by politicians (placeholder).

    Note: This would require cross-referencing SEC data with
    Congress member stock disclosures from other sources.
    """
    print("Politician trade search requires additional data sources")
    return []


__all__ = [
    # Models
    'Form8KEvent',
    'InsiderTrade',
    # Client
    'SECEdgarClient',
    'SEC_USER_AGENT',
    # Helper functions
    'get_recent_form4s',
    'get_recent_form8ks',
    'search_politician_trades',
]
