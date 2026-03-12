"""Market Sensing Fetchers — thin re-export hub.

All fetch functions are organized into domain-specific sub-modules:
  fetchers_insider.py     — SEC Form 4/8K, OpenInsider, 13F/13D, FinViz insider
  fetchers_government.py  — Congressional trades, Congress.gov, USASpending, SAM.gov, hiring
  fetchers_market_data.py — FRED, COT, VIX, EIA, Massive/Polygon
  fetchers_technical.py   — TA indicators, session sweeps, fractal resonance, order flow
  fetchers_social.py      — ApeWisdom, StockTwits, Google Trends, social text, Yahoo RSS, Finnhub
  fetchers_crypto.py      — CoinGecko, CoinCap, FINRA short, economic calendar

Called by MarketSensingHook._fetch_source() router in sensing_hook.py.
All import paths from sensing_fetchers continue to work via this hub.
"""

from mae_core.market.fetchers_insider import (
    fetch_sec_form4,
    fetch_sec_form8k,
    fetch_sec_efts,
    fetch_openinsider,
    fetch_13f_holdings,
    fetch_finviz,
)
from mae_core.market.fetchers_government import (
    fetch_congressional,
    fetch_senate,
    fetch_hiring,
    fetch_congress_legislation,
    fetch_usa_spending,
    fetch_sam_gov,
)
from mae_core.market.fetchers_market_data import (
    fetch_fred,
    fetch_fred_yields,
    fetch_usda,
    fetch_cot,
    fetch_vix,
    fetch_eia,
    fetch_massive_snapshot,
)
from mae_core.market.fetchers_technical import (
    fetch_ta_indicators,
    fetch_session_sweep,
    fetch_fractal_resonance,
    fetch_order_flow,
)
from mae_core.market.fetchers_social import (
    fetch_social_sentiment,
    fetch_stocktwits,
    fetch_trends,
    fetch_social_text,
    fetch_yahoo_rss,
    fetch_finnhub,
    fetch_finnhub_extras,
)
from mae_core.market.fetchers_crypto import (
    fetch_crypto_prices,
    fetch_crypto_exchange,
    fetch_binance_funding,
    fetch_kalshi_movers,
    fetch_finra_short,
    fetch_economic_calendar,
)

__all__ = [
    "fetch_sec_form4",
    "fetch_sec_form8k",
    "fetch_sec_efts",
    "fetch_openinsider",
    "fetch_13f_holdings",
    "fetch_finviz",
    "fetch_congressional",
    "fetch_senate",
    "fetch_hiring",
    "fetch_congress_legislation",
    "fetch_usa_spending",
    "fetch_sam_gov",
    "fetch_fred",
    "fetch_fred_yields",
    "fetch_usda",
    "fetch_cot",
    "fetch_vix",
    "fetch_eia",
    "fetch_massive_snapshot",
    "fetch_ta_indicators",
    "fetch_session_sweep",
    "fetch_fractal_resonance",
    "fetch_order_flow",
    "fetch_social_sentiment",
    "fetch_stocktwits",
    "fetch_trends",
    "fetch_social_text",
    "fetch_yahoo_rss",
    "fetch_finnhub",
    "fetch_finnhub_extras",
    "fetch_crypto_prices",
    "fetch_crypto_exchange",
    "fetch_binance_funding",
    "fetch_kalshi_movers",
    "fetch_finra_short",
    "fetch_economic_calendar",
]
