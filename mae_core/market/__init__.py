"""Mae Market Intelligence Package — MIDGE's sensory organs for financial markets."""

from mae_core.market.signal import (
    MarketSignal,
    TradeSignal,
    from_insider_trade,
    from_form8k_event,
    from_congressional_trade,
    from_hiring_signal,
    from_government_contract,
    from_contract_opportunity,
    from_cluster_signal,
    from_contract_prediction,
    from_correlation_signal,
)

__all__ = [
    "MarketSignal",
    "TradeSignal",
    "from_insider_trade",
    "from_form8k_event",
    "from_congressional_trade",
    "from_hiring_signal",
    "from_government_contract",
    "from_contract_opportunity",
    "from_cluster_signal",
    "from_contract_prediction",
    "from_correlation_signal",
]
