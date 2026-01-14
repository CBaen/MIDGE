"""
Trading API clients for various data sources.
"""

from .sec_edgar import SECEdgarClient, InsiderTrade, get_recent_form4s
from .usa_spending import USASpendingClient, GovernmentContract, get_oversight_committees
from .price_fetcher import PriceFetcher, PriceData, get_price, get_prices, price_fetcher_for_outcomes

__all__ = [
    # SEC Edgar
    "SECEdgarClient",
    "InsiderTrade",
    "get_recent_form4s",
    # USASpending
    "USASpendingClient",
    "GovernmentContract",
    "get_oversight_committees",
    # Price Fetcher
    "PriceFetcher",
    "PriceData",
    "get_price",
    "get_prices",
    "price_fetcher_for_outcomes",
]
