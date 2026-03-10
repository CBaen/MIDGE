"""
raw_store.py - Raw Data Storage Layer (composite entry point).

Persists ALL data from API calls before processing. Each data domain
gets its own SQLite database in data/market/raw/. WAL mode enables
concurrent read/write for daemon operation.

This is MIDGE's long-term memory for raw market data. The signal
pipeline extracts 1-2 processed values per API call; this layer
preserves the other 95% for future pattern discovery.

Split into domain mixins for manageability:
  raw_store_base.py       — SQLite connection management
  raw_store_market_data.py — VIX, COT, EIA, Trends, prices, FRED, USDA
  raw_store_social.py     — StockTwits, Finnhub, Yahoo RSS, ApeWisdom
  raw_store_government.py — Congressional trades, Congress bills, FINRA
  raw_store_insider.py    — SEC Form4, OpenInsider, FinViz, EDGAR 13F/13D
  raw_store_operational.py — Polygon bars, CoinGecko, CoinCap, jobs, SAM.gov
"""

from .raw_store_base import RawStoreBase
from .raw_store_market_data import RawStoreMarketDataMixin
from .raw_store_social import RawStoreSocialMixin
from .raw_store_government import RawStoreGovernmentMixin
from .raw_store_insider import RawStoreInsiderMixin
from .raw_store_operational import RawStoreOperationalMixin


class RawStore(
    RawStoreMarketDataMixin,
    RawStoreSocialMixin,
    RawStoreGovernmentMixin,
    RawStoreInsiderMixin,
    RawStoreOperationalMixin,
    RawStoreBase,
):
    """
    Unified raw data store. All store_* methods live in domain mixins.
    RawStoreBase provides _get_conn() and close().
    """
    pass


__all__ = ["RawStore"]
