"""Signal Enricher — maps ticker-less signals to their natural affected tickers.

Many important signals arrive with symbol="" because they are macro, energy,
volatility, or legislative events that affect categories of instruments rather
than a single ticker. This module expands those signals into per-ticker copies
so they participate in per-ticker convergence detection.

Key insight: a crude oil inventory drawdown (EIA, symbol="") + CL=F session
sweep (symbol="CL=F") + CL=F COT bullish (symbol="CL=F") should converge on
CL=F. Without enrichment they never meet in the ticker-convergence grouper.

Enriched signals get a 15% confidence haircut (they are inferred, not direct).
The original ticker-less signal is preserved unchanged alongside the copies.
"""

from __future__ import annotations

import copy
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain / source → natural tickers
# ---------------------------------------------------------------------------
# Keys are matched against signal.domain first, then signal.source.
# Lists are deduplicated on use; order does not matter.
# Keep this table narrow — only include tickers that are already on the
# MIDGE watchlist or are liquid enough to trade. Do not add speculative names.
# ---------------------------------------------------------------------------
DOMAIN_TICKER_MAP: dict[str, list[str]] = {
    # ---- Macro (FRED, economic calendar) --------------------------------
    "macro": ["SPY", "ES=F", "NQ=F", "QQQ", "DIA"],
    "fred_macro": ["SPY", "ES=F", "NQ=F"],
    # Yield-curve signals map to bond ETFs
    "fred_yields": ["TLT", "IEF", "SHY"],

    # ---- Energy (EIA: crude, natural gas, distillate) -------------------
    "energy": ["XLE", "CL=F", "XOM", "CVX", "OXY"],
    "eia_energy": ["CL=F", "XLE", "XOM", "CVX"],

    # ---- Volatility / options -------------------------------------------
    "volatility": ["SPY", "QQQ", "ES=F", "VXX"],
    "options": ["SPY", "QQQ", "ES=F"],

    # ---- Crypto (fear/greed, sector sentiment) --------------------------
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "crypto_fear_greed": ["BTC-USD", "ETH-USD"],

    # ---- Government / legislation (broad until ticker is parsed) --------
    "government": ["LMT", "RTX", "NOC", "GD", "BA"],
    "congress_legislation": ["SPY"],

    # ---- Institutional synthesis (13F aggregate moves) ------------------
    "institutional": ["SPY", "QQQ"],
    "institutional_synthesis": ["SPY"],

    # ---- Positioning (COT) — futures already ticker-tagged; skip broad --
    "positioning": [],

    # ---- Economic calendar (FOMC/CPI/NFP affect everything) -------------
    "economic_calendar": ["SPY", "ES=F", "NQ=F", "TLT"],
}

# Sources that are domain-less but carry a meaningful natural ticker set.
# Checked only when domain lookup returns nothing.
_SOURCE_TICKER_MAP: dict[str, list[str]] = {
    "fred_macro": ["SPY", "ES=F", "NQ=F"],
    "eia_energy": ["CL=F", "XLE", "XOM", "CVX"],
    "vix_term_structure": ["SPY", "QQQ", "ES=F", "VXX"],
    "economic_calendar": ["SPY", "ES=F", "NQ=F", "TLT"],
    "congress_legislation": ["SPY"],
    "finnhub_economic": ["SPY", "ES=F", "TLT"],
    "crypto_coingecko": ["BTC-USD", "ETH-USD", "SOL-USD"],
    "crypto_coincap": ["BTC-USD", "ETH-USD"],
}

# Confidence penalty applied to every enriched copy (inferred, not direct).
_CONFIDENCE_HAIRCUT = 0.15


def enrich_ticker(signal_dict: dict) -> List[dict]:
    """Expand a ticker-less signal into per-ticker copies.

    Args:
        signal_dict: Keyword-argument dict used to call record_signal().
                     Must contain at least "domain" and "source".
                     The "metadata" key is expected to carry {"symbol": ...}.

    Returns:
        A list of signal dicts ready to be unpacked into record_signal().
        - If the signal already carries a non-empty symbol: returns [signal_dict]
          unchanged (no expansion needed).
        - If the signal has no symbol: returns one copy per mapped ticker, each
          with symbol injected into metadata and a 15% confidence haircut.
          The original ticker-less dict is NOT included (it would produce a
          symbol="" entry that contributes nothing to per-ticker convergence
          and would duplicate the global domain-level contribution).
        - If no tickers are mapped for this domain/source: returns [signal_dict]
          unchanged so the signal still contributes to global convergence.
    """
    metadata = signal_dict.get("metadata") or {}
    symbol = metadata.get("symbol", "")

    # Already has a ticker — pass through unchanged.
    if symbol:
        return [signal_dict]

    domain: str = signal_dict.get("domain", "")
    source: str = signal_dict.get("source", "")

    # Look up natural tickers: domain wins over source.
    tickers = DOMAIN_TICKER_MAP.get(domain) or _SOURCE_TICKER_MAP.get(source) or []

    if not tickers:
        # No mapping — preserve original so it still feeds global convergence.
        return [signal_dict]

    enriched: List[dict] = []
    for ticker in tickers:
        copy_ = copy.deepcopy(signal_dict)

        # Inject ticker into metadata.
        copy_["metadata"] = {**metadata, "symbol": ticker,
                             "enriched_from": "domain_ticker_map",
                             "original_source": source or domain}

        # Apply confidence haircut — inferred association, not a direct signal.
        original_confidence = copy_.get("confidence", 0.5)
        copy_["confidence"] = max(0.0, original_confidence * (1.0 - _CONFIDENCE_HAIRCUT))

        # Suffix signal_id so each copy is unique in the buffer.
        copy_["signal_id"] = f"{copy_['signal_id']}::{ticker}"

        enriched.append(copy_)

    logger.debug(
        "SignalEnricher: expanded ticker-less %s/%s signal into %d tickers: %s",
        domain, source, len(enriched), [t["metadata"]["symbol"] for t in enriched],
    )
    return enriched
