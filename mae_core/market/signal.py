"""Canonical signal types for MIDGE market intelligence.

MarketSignal is the normalized format for all market data flowing through MIDGE.
TradeSignal is the actionable output — what to trade and why.

Adapter functions convert raw source types into MarketSignal.
Each adapter is a standalone function named `from_{source_type}`.

The 34 adapter functions live in the signal_adapters/ subpackage (split by domain)
and are re-exported here for full backward compatibility. All existing imports of
the form `from mae_core.market.signal import from_insider_trade` continue to work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


def _ensure_datetime(value) -> datetime:
    """Coerce a string or datetime to datetime. Returns now() on failure."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                    "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(value.split("T")[0] if "T" in value else value, fmt)
            except ValueError:
                continue
        logger.debug("Could not parse date string %r, using now()", value)
    return datetime.now()


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketSignal:
    """Canonical normalized signal format for all market data flowing through MIDGE."""

    # Identity
    signal_id: str          # UUID or "{source}:{symbol}:{timestamp}"
    source: str             # "sec_form4", "congressional", "hiring_tracker", etc.
    symbol: str             # Ticker. Empty string ("") for macro/pre-ticker signals.
    asset_class: str        # "stock", "crypto", "futures", "commodities", "macro"

    # Classification
    domain: str             # "insider", "congress", "contracts", "government",
                            # "technical", "sentiment", "news", "institutional"
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0–1.0 normalized

    # Reliability
    confidence: float       # Source reliability estimate 0.0–1.0
    decay_rate: float       # Per-day decay

    # Time
    timestamp: datetime     # When the UNDERLYING EVENT occurred
    received_at: datetime   # When MIDGE received/detected the signal

    # Velocity — populated by VelocityDetector
    velocity: float = 0.0  # Per-DAY velocity. Default 0.0 before wiring.

    # Feedback loop
    outcome_symbol: str = ""         # Ticker to check for price outcome
    outcome_window_days: int = 14    # Default 14 days forward

    # Audit trail
    raw_id: str = ""        # Identifier of original record in source system
    raw_type: str = ""      # "InsiderTrade", "Form8KEvent", etc.

    # Pattern discovery context
    metadata: dict = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Actionable output: what to trade, when, and why."""
    signal_id: str
    asset: str              # Ticker or asset name
    asset_class: str        # "stock", "crypto", "futures", "commodities"
    direction: str          # "buy" or "sell"
    confidence: float       # 0.0–1.0
    timeframe_days: int     # Expected holding period
    catalyst: str           # Human-readable reason
    contributing_signals: list = field(default_factory=list)  # MarketSignal IDs
    hit_rate: float = 0.0   # Historical accuracy for this signal type
    generated_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Adapter re-exports — all 34 from_* functions (backward compatibility)
# ---------------------------------------------------------------------------
# The actual implementations live in signal_adapters/ subpackage.
# These imports keep all existing `from mae_core.market.signal import X` working.

from mae_core.market.signal_adapters import (  # noqa: E402
    from_insider_trade,
    from_form8k_event,
    from_filing_keyword,
    from_cluster_signal,
    from_correlation_signal,
    from_congressional_trade,
    from_senate_trade,
    from_short_interest,
    from_news_sentiment,
    from_earnings_event,
    from_macro_indicator,
    from_price_data,
    from_social_sentiment,
    from_ta_signal,
    from_session_sweep,
    from_order_flow,
    from_fractal_resonance,
    from_government_contract,
    from_contract_opportunity,
    from_contract_prediction,
    from_hiring_signal,
    from_cot_positioning,
    from_stocktwits_sentiment,
    from_vix_structure,
    from_trends_signal,
    from_economic_event,
    from_analyst_recommendation,
    from_yahoo_rss_signal,
    from_crypto_signal,
    from_openinsider,
    from_13f_holding,
    from_activist_filing,
    from_finviz_unusual_volume,
    from_finviz_short_squeeze,
    from_finnhub_realtime,
    from_suppression_event,
    from_binance_funding,
    from_energy_indicator,
    from_legislative_indicator,
)

__all__ = [
    # Dataclasses
    "MarketSignal",
    "TradeSignal",
    # Helper
    "_ensure_datetime",
    # Adapter re-exports (regulatory)
    "from_insider_trade",
    "from_form8k_event",
    "from_filing_keyword",
    "from_cluster_signal",
    "from_correlation_signal",
    # Adapter re-exports (political)
    "from_congressional_trade",
    "from_senate_trade",
    # Adapter re-exports (market_data)
    "from_short_interest",
    "from_news_sentiment",
    "from_earnings_event",
    "from_macro_indicator",
    "from_price_data",
    "from_social_sentiment",
    # Adapter re-exports (technical)
    "from_ta_signal",
    "from_session_sweep",
    "from_order_flow",
    "from_fractal_resonance",
    # Adapter re-exports (contracts)
    "from_government_contract",
    "from_contract_opportunity",
    "from_contract_prediction",
    "from_hiring_signal",
    # Adapter re-exports (layer6)
    "from_cot_positioning",
    "from_stocktwits_sentiment",
    "from_vix_structure",
    "from_trends_signal",
    "from_economic_event",
    "from_analyst_recommendation",
    "from_yahoo_rss_signal",
    # Adapter re-exports (wave2_3)
    "from_crypto_signal",
    "from_openinsider",
    "from_13f_holding",
    "from_activist_filing",
    "from_finviz_unusual_volume",
    "from_finviz_short_squeeze",
    "from_finnhub_realtime",
    "from_suppression_event",
    "from_binance_funding",
    # energy + legislation (2)
    "from_energy_indicator",
    "from_legislative_indicator",
]
