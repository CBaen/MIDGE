"""Signal adapter subpackage — all 27 from_* converter functions.

This package splits the original signal.py adapter functions into focused
modules by domain. Backward compatibility is maintained: importing any
from_* function from mae_core.market.signal still works via re-exports there.

Module layout:
    regulatory   — SEC filings, insider clusters, correlation signals (5 functions)
    political    — Congressional and Senate trade disclosures (2 functions)
    market_data  — Short interest, news, earnings, macro, social, price (6 functions)
    technical    — TA indicators, session sweeps, order flow, fractal resonance (4 functions)
    contracts    — Government contracts, SAM.gov, predictions, hiring (4 functions)
    layer6       — COT, StockTwits, VIX, Google Trends, Finnhub extras (6 functions)
"""

from mae_core.market.signal_adapters.regulatory import (
    from_insider_trade,
    from_form8k_event,
    from_filing_keyword,
    from_cluster_signal,
    from_correlation_signal,
)

from mae_core.market.signal_adapters.political import (
    from_congressional_trade,
    from_senate_trade,
)

from mae_core.market.signal_adapters.market_data import (
    from_short_interest,
    from_news_sentiment,
    from_earnings_event,
    from_macro_indicator,
    from_price_data,
    from_social_sentiment,
)

from mae_core.market.signal_adapters.technical import (
    from_ta_signal,
    from_session_sweep,
    from_order_flow,
    from_fractal_resonance,
)

from mae_core.market.signal_adapters.contracts import (
    from_government_contract,
    from_contract_opportunity,
    from_contract_prediction,
    from_hiring_signal,
)

from mae_core.market.signal_adapters.layer6 import (
    from_cot_positioning,
    from_stocktwits_sentiment,
    from_vix_structure,
    from_trends_signal,
    from_economic_event,
    from_analyst_recommendation,
)

__all__ = [
    # regulatory (5)
    "from_insider_trade",
    "from_form8k_event",
    "from_filing_keyword",
    "from_cluster_signal",
    "from_correlation_signal",
    # political (2)
    "from_congressional_trade",
    "from_senate_trade",
    # market_data (6)
    "from_short_interest",
    "from_news_sentiment",
    "from_earnings_event",
    "from_macro_indicator",
    "from_price_data",
    "from_social_sentiment",
    # technical (4)
    "from_ta_signal",
    "from_session_sweep",
    "from_order_flow",
    "from_fractal_resonance",
    # contracts (4)
    "from_government_contract",
    "from_contract_opportunity",
    "from_contract_prediction",
    "from_hiring_signal",
    # layer6 (6)
    "from_cot_positioning",
    "from_stocktwits_sentiment",
    "from_vix_structure",
    "from_trends_signal",
    "from_economic_event",
    "from_analyst_recommendation",
]
