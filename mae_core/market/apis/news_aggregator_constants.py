"""
news_aggregator_constants.py — Static data for NewsAggregatorClient.

Contains keyword sets, ticker allowlist, RSS feed definitions, and URL
templates. Kept separate so news_aggregator_client.py stays under the
500-line monolith cap. Edit here when adding new sources or keywords.
"""

from typing import Dict, List, Set

# ---------------------------------------------------------------------------
# Sentiment keyword sets  (keyword-based, no LLM)
# ---------------------------------------------------------------------------

NEGATIVE_KEYWORDS: frozenset = frozenset({
    "crash", "crashes", "plunge", "plunges", "plunging",
    "layoff", "layoffs", "job cuts", "cuts jobs", "mass layoff",
    "recall", "recalled",
    "default", "defaulted",
    "bankruptcy", "bankrupt", "chapter 11", "chapter 7",
    "investigation", "investigated", "probe", "subpoena",
    "fraud", "fraudulent", "ponzi",
    "warning", "warns", "warning sign",
    "downgrade", "downgraded",
    "miss", "misses", "missed estimates",
    "decline", "declines", "declining",
    "loss", "losses",
    "slump", "slumps",
    "shutdown", "shuttered",
    "inflation", "stagflation",
    "recession", "contraction",
    "rate hike", "rate hikes", "tightening",
    "sanction", "sanctions",
    "tariff", "tariffs",
})

POSITIVE_KEYWORDS: frozenset = frozenset({
    "surge", "surges", "surging",
    "record", "record high", "all-time high",
    "upgrade", "upgraded",
    "beat", "beats", "beat expectations", "earnings beat",
    "approval", "approved", "fda approves", "fda approval",
    "launch", "launches", "launched",
    "expansion", "expands", "expanding",
    "partnership", "partnerships",
    "acquisition", "acquires", "merger", "deal",
    "profit", "profits",
    "rally", "rallies", "rallying",
    "breakout",
    "rate cut", "rate cuts", "easing",
    "stimulus",
    "growth", "growing",
    "hiring", "jobs added",
})

# ---------------------------------------------------------------------------
# Financial domain keywords for contextual tagging
# ---------------------------------------------------------------------------

FINANCIAL_KEYWORDS: frozenset = frozenset({
    "fed", "federal reserve", "fomc", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp", "jobs report", "nfp", "unemployment",
    "earnings", "revenue", "guidance", "forecast",
    "ipo", "spac", "merger", "acquisition",
    "sec", "filing", "8-k", "10-k", "10-q",
    "congress", "senate", "legislation", "regulation",
    "oil", "crude", "energy", "natural gas",
    "gold", "silver", "copper",
    "yield", "treasury", "bond", "debt ceiling",
    "bitcoin", "crypto", "cryptocurrency",
    "china", "trade war", "tariff",
    "recession", "contraction", "expansion",
    "stock", "shares", "equities",
    "market", "nasdaq", "s&p", "dow",
    "bank", "banking", "credit",
    "dollar", "yen", "euro", "forex",
})

# ---------------------------------------------------------------------------
# Top-100 S&P 500 tickers for standalone-word extraction
# (cap standalone matches at 3 per headline to suppress false positives)
# ---------------------------------------------------------------------------

SP500_TICKERS: Set[str] = {
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK",
    "UNH", "LLY", "JPM", "V", "AVGO", "XOM", "PG", "MA", "JNJ", "HD",
    "MRK", "ABBV", "CVX", "CRM", "BAC", "COST", "NFLX", "AMD", "PEP", "KO",
    "TMO", "ADBE", "WMT", "ACN", "MCD", "CSCO", "LIN", "ABT", "DHR", "TXN",
    "CMCSA", "VZ", "NEE", "WFC", "PM", "MS", "BMY", "ORCL", "INTC", "T",
    "RTX", "AMGN", "GE", "HON", "QCOM", "IBM", "CAT", "UPS", "COP", "GS",
    "LOW", "SPGI", "DE", "INTU", "ELV", "SBUX", "MDT", "AXP", "ISRG", "PLD",
    "GILD", "CVS", "CI", "SYK", "ADP", "BLK", "TJX", "NOW", "REGN", "ZTS",
    "MO", "DUK", "SO", "PNC", "USB", "MMC", "VRTX", "CB", "SCHW", "ADI",
    "LRCX", "KLAC", "PANW", "SNPS", "CDNS", "MELI", "SHW", "ITW", "APD",
    "GM", "F", "BA", "DAL", "UAL", "AAL", "LUV", "X", "NUE", "CLF",
    "SPY", "QQQ", "IWM", "GLD", "SLV", "USO", "TLT", "HYG",
}

# ---------------------------------------------------------------------------
# RSS feed source definitions
# ---------------------------------------------------------------------------
# Each entry: name (slug), url, label (human-readable).
# Bloomberg is included but commonly returns 403 — handled gracefully.

RSS_SOURCES: List[Dict] = [
    {
        "name": "reuters",
        "url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "label": "Reuters Business",
    },
    {
        "name": "cnbc",
        "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "label": "CNBC Top News",
    },
    {
        "name": "marketwatch",
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "label": "MarketWatch",
    },
    {
        "name": "bloomberg",
        "url": "https://feeds.bloomberg.com/markets/news.rss",
        "label": "Bloomberg Markets",
    },
    {
        "name": "federal_reserve",
        "url": "https://www.federalreserve.gov/feeds/press_all.xml",
        "label": "Federal Reserve",
    },
]

# ---------------------------------------------------------------------------
# SEC EDGAR public endpoints (no API key required)
# ---------------------------------------------------------------------------
# {date} placeholder filled with YYYY-MM-DD at runtime.

EDGAR_8K_URL = (
    "https://efts.sec.gov/LATEST/search-index"
    "?q=%228-K%22&forms=8-K&dateRange=custom"
    "&startdt={date}&enddt={date}"
    "&hits.hits._source=period_of_report,entity_name,file_date,form_type"
)
