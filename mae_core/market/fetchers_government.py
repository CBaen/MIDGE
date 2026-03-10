"""Government data fetch functions — Congressional trades, legislation, contracts.

Covers STOCK Act disclosures, Congress.gov legislative signals, USASpending,
and SAM.gov federal contracting opportunities.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("midge.market.sensing")


def fetch_congressional(congress_client: Any, converter: Callable) -> list:
    """Fetch congressional stock trades."""
    if congress_client is None:
        return []

    signals = []
    try:
        trades = congress_client.get_recent_trades(days=30)
        for trade in trades:
            try:
                # Filter: trades below $50K are noise
                if trade.amount_high < 50_000:
                    continue
                signals.append(converter(trade))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Congressional trades fetch failed: %s", e)
    return signals


def fetch_senate(senate_client: Any, converter: Callable) -> list:
    """Fetch Senate stock trades."""
    if senate_client is None:
        return []

    signals = []
    try:
        trades = senate_client.get_recent_trades(days=30)
        for trade in trades:
            try:
                if trade.amount_high < 50_000:
                    continue
                signals.append(converter(trade))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Senate trades fetch failed: %s", e)
    return signals


def fetch_congress_legislation(congress_gov_client: Any, converter: Callable) -> list:
    """Fetch advancing legislative indicators from Congress.gov."""
    if congress_gov_client is None:
        return []

    signals = []
    try:
        snapshot = congress_gov_client.get_legislative_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Congress.gov legislation fetch failed: %s", e)
    return signals


def fetch_hiring(job_tracker: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch hiring signals for watchlist companies."""
    if job_tracker is None:
        return []

    signals = []
    companies = watchlist.get("companies", {})
    for company, ticker in companies.items():
        try:
            signal = job_tracker.analyze_hiring_activity(company, ticker=ticker)
            signals.append(converter(signal))
        except Exception as e:
            logger.debug("Hiring fetch failed for %s: %s", company, e)
    return signals


def fetch_usa_spending(usa_spending: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch government contracts from USASpending."""
    if usa_spending is None:
        return []

    signals = []
    for keyword in watchlist.get("keywords", []):
        try:
            contracts = usa_spending.search_contracts(keyword=keyword, limit=5)
            for contract in contracts:
                try:
                    signals.append(converter(contract))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("USASpending fetch failed for '%s': %s", keyword, e)
    return signals


def fetch_sam_gov(sam_gov: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch SAM.gov opportunities."""
    if sam_gov is None:
        return []

    signals = []
    for keyword in watchlist.get("keywords", []):
        try:
            opps = sam_gov.search_opportunities(keywords=keyword, limit=5)
            for opp in opps:
                try:
                    signals.append(converter(opp))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SAM.gov fetch failed for '%s': %s", keyword, e)
    return signals
