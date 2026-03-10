"""house_stock_normalizer.py - Congressional trade normalization helpers.

Extracted from house_stock_watcher.py. Converts raw RapidAPI and House
Stock Watcher JSON trade dicts into a canonical normalized dict that
HouseStockWatcherClient.get_recent_trades() can consume uniformly.
"""

from __future__ import annotations

from typing import Optional


def normalize_trade(item: dict) -> Optional[dict]:
    """Normalize trade data from different API formats.

    Handles both RapidAPI and House Stock Watcher formats. Returns a
    canonical dict with these keys:
        disclosure_date, transaction_date, representative, party,
        district, ticker, asset_description, transaction_type,
        amount_low, amount_high, amount_str, owner, url

    Returns None if the item is empty or malformed beyond recovery.
    """
    if not item:
        return None

    # Detect format by checking for RapidAPI-specific fields
    # API uses snake_case (pub_date, tx_date) or camelCase (pubDate, txDate)
    is_rapidapi = any(k in item for k in (
        "pub_date", "tx_date", "pubDate", "txDate", "politician_id"
    ))

    if is_rapidapi:
        return _normalize_rapidapi(item)
    return _normalize_house_stock_watcher(item)


def _normalize_rapidapi(item: dict) -> dict:
    """Normalize a RapidAPI US Congress Insider Trading trade record."""
    pub_date = item.get("pub_date") or item.get("pubDate", "")
    if pub_date and "T" in pub_date:
        pub_date = pub_date.split("T")[0]

    # Parse ticker from "JNJ:US" format
    ticker_raw = item.get("issuer_ticker") or item.get("issuer_issuerTicker", "")
    ticker = ticker_raw.split(":")[0] if ticker_raw else ""

    # Build representative name
    first = item.get("politician_first_name") or item.get("politician_firstName", "")
    last = item.get("politician_last_name") or item.get("politician_lastName", "")
    rep_name = f"{first} {last}".strip() or "Unknown"

    # Map party
    party_raw = item.get("politician_party", "").lower()
    party = "D" if "democrat" in party_raw else "R" if "republican" in party_raw else ""

    # Get value
    value = item.get("value", 0) or 0

    # State
    state = item.get("politician_state") or item.get("politician_stateId", "")

    return {
        "disclosure_date": pub_date,
        "transaction_date": item.get("tx_date") or item.get("txDate", ""),
        "representative": rep_name,
        "party": party,
        "district": f"{state.upper()} ({item.get('chamber', '')})",
        "ticker": ticker,
        "asset_description": item.get("issuer_name") or item.get("issuer_issuerName", ""),
        "transaction_type": item.get("tx_type") or item.get("txType", "unknown"),
        "amount_low": value,
        "amount_high": value,
        "amount_str": f"${value:,.0f}" if value else "$0",
        "owner": item.get("ownership", "Self"),
        "url": "",
    }


def _normalize_house_stock_watcher(item: dict) -> dict:
    """Normalize a House Stock Watcher (free source) trade record."""
    return {
        "disclosure_date": item.get("disclosure_date", ""),
        "transaction_date": item.get("transaction_date", ""),
        "representative": item.get("representative", "Unknown"),
        "party": item.get("party", ""),
        "district": item.get("district", ""),
        "ticker": item.get("ticker", ""),
        "asset_description": item.get("asset_description", ""),
        "transaction_type": item.get("type", "unknown"),
        "amount_low": 0,
        "amount_high": 0,
        "amount_str": item.get("amount", "$0 - $0"),
        "owner": item.get("owner", "Self"),
        "url": item.get("ptr_link", ""),
    }
