"""Market Sensing Constants — routing tables and source rotation config.

Centralizes all module-level constants used across the sensing subsystem:
  TIER_ROUTING          — source name → tactical/strategic/thematic tier
  SOURCE_ROTATION       — ordered list of 36 rotation source names
  _ROTATION_TO_THOMPSON — rotation name → Thompson distribution key
  _ABSENCE_SOURCE_DOMAINS — absence source name → convergence domain
  _DOMAIN_TO_SOURCES    — pre-built reverse map: domain → rotation names
"""

from __future__ import annotations

from typing import Dict, List

# Multi-timeframe convergence: route signals by source → tier
TIER_ROUTING: Dict[str, str] = {
    "sec_form4": "tactical",
    "sec_form8k": "tactical",
    "sec_efts": "tactical",
    "finnhub_news": "tactical",
    "finnhub_earnings": "tactical",
    "congressional": "strategic",
    "senate": "strategic",
    "contract": "strategic",
    "insider_cluster": "strategic",
    "correlation": "strategic",
    "finra_short": "strategic",
    "sam_gov": "thematic",
    "hiring_tracker": "thematic",
    "contract_prediction": "thematic",
    "contract_award": "thematic",
    "social_sentiment": "thematic",
    "fred_macro": "thematic",
    "session_sweep": "tactical",
    "session_sweep_ifvg": "tactical",
    "ta_rsi": "tactical",
    "ta_macd": "tactical",
    "ta_bollinger": "tactical",
    "ta_structure": "tactical",
    "ta_candle": "tactical",
    # Ten Gifts: Wave 1
    "order_flow": "tactical",
    # Ten Gifts: Wave 2
    "fractal_resonance": "strategic",
    "archetype_match": "strategic",
    # New sources (Layer 6)
    "cot_positioning": "strategic",
    "stocktwits_sentiment": "thematic",
    "vix_term_structure": "strategic",
    "google_trends": "thematic",
    "finnhub_economic": "tactical",
    "finnhub_analyst": "strategic",
    "finnhub_earnings_calendar": "tactical",
    # Wave 2: Real-Time + Crypto
    "finnhub_realtime": "tactical",
    "crypto_coingecko": "thematic",
    "crypto_coincap": "thematic",
    # Wave 3: Data Enrichment
    "openinsider_purchase": "tactical",
    "institutional_13f": "strategic",
    "activist_13d": "strategic",
    "finviz_unusual_volume": "tactical",
    "finviz_short_squeeze": "strategic",
    "finviz_insider": "tactical",
    "economic_calendar": "thematic",
    # Massive/Polygon.io
    "massive_snapshot": "tactical",
    # Real-economy: Energy
    "eia_energy": "strategic",
    # Real-economy: Legislative
    "congress_legislation": "strategic",
    # Social text analysis
    "social_text": "thematic",
    # Yahoo Finance RSS — per-ticker headline velocity
    "yahoo_rss": "tactical",
    # USDA agricultural supply/demand
    "usda_agriculture": "strategic",
    # Forex-critical FRED yields / DXY
    "fred_yields": "thematic",
    # Crypto derivatives positioning (Binance perpetual futures funding rates)
    "binance_funding": "tactical",
    # Prediction markets (Kalshi — crowd probability estimates)
    "kalshi_market": "strategic",
}

# Source names for rotation — 36 sources, 8 concurrent per cadence tick
SOURCE_ROTATION: List[str] = [
    "sec_form4",
    "sec_form8k",
    "congressional",
    "senate",
    "hiring",
    "usa_spending",
    "sam_gov_and_prices",
    "social_sentiment",
    "finra_short",
    "sec_efts",
    "finnhub",
    "fred_macro",
    "session_sweep",
    "ta_indicators",
    # Ten Gifts: Wave 1
    "order_flow",
    # Ten Gifts: Wave 2
    "fractal_resonance",
    # New sources (Layer 6)
    "cot_positioning",
    "stocktwits",
    "vix_structure",
    "google_trends",
    "finnhub_extras",
    # Wave 2: Real-Time + Crypto (Always-On)
    "crypto_prices",
    "crypto_exchange",
    # Wave 3: Data Enrichment
    "openinsider",
    "institutional_13f",
    "finviz",
    "economic_calendar",
    # Massive/Polygon.io
    "massive_snapshot",
    # Real-economy: Energy
    "eia_energy",
    # Real-economy: Legislative
    "congress_legislation",
    # Social text analysis (local — no API, just reads SQLite)
    "social_text",
    # Yahoo Finance RSS — per-ticker headline velocity (free, no key)
    "yahoo_rss",
    # USDA agricultural supply/demand
    "usda_agriculture",
    # Forex-critical FRED yields / DXY
    "fred_yields",
    # Crypto derivatives positioning (Binance perpetual futures funding rates)
    "binance_funding",
    # Prediction markets (Kalshi — crowd probability estimates)
    "kalshi_market",
]

# Map rotation source names → Thompson distribution keys for guided selection.
# Sources with multiple Thompson keys use their primary/dominant key.
_ROTATION_TO_THOMPSON: Dict[str, str] = {
    "sec_form4": "sec_form4",
    "sec_form8k": "sec_form8k",
    "congressional": "congressional",
    "senate": "senate",
    "hiring": "hiring_tracker",
    "usa_spending": "contract_award",
    "sam_gov_and_prices": "sam_gov",
    "social_sentiment": "social_sentiment",
    "finra_short": "finra_short",
    "sec_efts": "sec_efts",
    "finnhub": "finnhub_news",
    "fred_macro": "fred_macro",
    "session_sweep": "session_sweep",
    "ta_indicators": "ta_rsi",
    "order_flow": "order_flow",
    "fractal_resonance": "fractal_resonance",
    "cot_positioning": "cot_positioning",
    "stocktwits": "stocktwits_sentiment",
    "vix_structure": "vix_term_structure",
    "google_trends": "google_trends",
    "finnhub_extras": "finnhub_economic",
    "crypto_prices": "crypto_coingecko",
    "crypto_exchange": "crypto_coincap",
    "openinsider": "openinsider_purchase",
    "institutional_13f": "institutional_13f",
    "finviz": "finviz_unusual_volume",
    "economic_calendar": "economic_calendar",
    "massive_snapshot": "massive_snapshot",
    "eia_energy": "eia_energy",
    "congress_legislation": "congress_legislation",
    "social_text": "social_text",
    "yahoo_rss": "yahoo_rss",
    "usda_agriculture": "usda_agriculture",
    "fred_yields": "fred_macro",
    "binance_funding": "binance_funding",
    "kalshi_market": "kalshi_market",
}

# Map absence source names back to convergence domains
_ABSENCE_SOURCE_DOMAINS: Dict[str, str] = {
    "sec_form4": "insider", "sec_form8k": "insider", "sec_efts": "insider",
    "congressional": "government", "senate": "government",
    "finra_short": "institutional", "cot_positioning": "positioning",
    "fred_macro": "macro", "finnhub_earnings": "news",
    "finnhub_news": "news", "hiring": "contracts",
    "usa_spending": "contracts", "sam_gov": "contracts",
    "crypto_coingecko": "crypto", "crypto_coincap": "crypto",
    "openinsider_purchase": "insider", "institutional_13f": "institutional",
    "activist_13d": "institutional",
    "finviz_unusual_volume": "technical", "finviz_short_squeeze": "institutional",
    "finviz_insider": "insider",
    "massive_snapshot": "technical",
    "social_text": "sentiment",
    "eia_energy": "energy",
    "congress_legislation": "government",
    "yahoo_rss": "events",
    "usda_agriculture": "macro",
    "fred_yields": "macro",
    "binance_funding": "positioning",
    "kalshi_market": "prediction_market",
}


def _absence_source_to_domain(source: str) -> str:
    """Map an absence source name to a convergence domain."""
    return _ABSENCE_SOURCE_DOMAINS.get(source, "unknown")


def _build_domain_to_sources() -> Dict[str, List[str]]:
    """Build reverse mapping: domain → list of SOURCE_ROTATION names that serve it.

    Approach:
    1. For each SOURCE_ROTATION name, look up its Thompson key via _ROTATION_TO_THOMPSON.
    2. Look up that Thompson key in _ABSENCE_SOURCE_DOMAINS to find the domain.
    3. Also check if the rotation name itself is in _ABSENCE_SOURCE_DOMAINS directly.
    4. Group rotation names by domain.

    This is a heuristic — a few sources serve multiple domains (e.g., "finviz"
    produces signals across technical + institutional + insider).  We assign
    each rotation source to its PRIMARY domain (via its primary Thompson key).
    """
    result: Dict[str, List[str]] = {}
    for rotation_name in SOURCE_ROTATION:
        # Prefer direct match (rotation_name == absence source key)
        domain = _ABSENCE_SOURCE_DOMAINS.get(rotation_name)
        if domain is None:
            # Try via Thompson key mapping
            thompson_key = _ROTATION_TO_THOMPSON.get(rotation_name)
            if thompson_key:
                domain = _ABSENCE_SOURCE_DOMAINS.get(thompson_key)
        if domain is not None:
            result.setdefault(domain, []).append(rotation_name)
    return result


# Pre-built reverse map: domain → SOURCE_ROTATION names that cover it.
# Used by _launch_thompson_guided() to boost priority-requested sources.
_DOMAIN_TO_SOURCES: Dict[str, List[str]] = _build_domain_to_sources()
