"""Political signal adapters — Congressional and Senate trade disclosures.

Converts House and Senate STOCK Act disclosure data into normalized MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime
from mae_core.market.apis.house_stock_watcher import CongressionalTrade


def from_congressional_trade(trade: CongressionalTrade) -> MarketSignal:
    """Convert a CongressionalTrade to a MarketSignal.

    Critical: timestamp=transaction_date (when the trade occurred, NOT when disclosed).
    received_at=disclosure_date (when MIDGE/public learned of it).
    """
    is_buy = "purchase" in trade.transaction_type.lower()
    direction = "bullish" if is_buy else "bearish"

    # Strength from upper bound of trade amount range
    strength = min(1.0, trade.amount_high / 500_000)

    # transaction_date is the event; disclosure_date is when MIDGE received it
    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.disclosure_date) if trade.disclosure_date else datetime.now()

    symbol = trade.ticker or ""
    signal_id = f"congressional:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="congressional",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=strength,
        confidence=trade.confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.disclosure_url or "",
        raw_type="CongressionalTrade",
        metadata={
            "representative": trade.representative,
            "party": trade.party,
            "district": trade.district,
            "transaction_type": trade.transaction_type,
            "amount_range": trade.amount_range,
            "amount_low": trade.amount_low,
            "amount_high": trade.amount_high,
            "owner": trade.owner,
            "asset_description": trade.asset_description,
        },
    )


def from_senate_trade(trade: CongressionalTrade) -> MarketSignal:
    """Convert a Senate CongressionalTrade to a MarketSignal.

    Same logic as from_congressional_trade but tagged as senate source.
    Uses same CongressionalTrade dataclass — Senate Stock Watcher produces
    the same shape.
    """
    is_buy = "purchase" in trade.transaction_type.lower()
    direction = "bullish" if is_buy else "bearish"

    strength = min(1.0, trade.amount_high / 500_000)

    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.disclosure_date) if trade.disclosure_date else datetime.now()

    symbol = trade.ticker or ""
    signal_id = f"senate:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="senate",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=strength,
        confidence=trade.confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.disclosure_url or "",
        raw_type="CongressionalTrade",
        metadata={
            "representative": trade.representative,
            "party": trade.party,
            "chamber": "Senate",
            "district": trade.district,
            "transaction_type": trade.transaction_type,
            "amount_range": trade.amount_range,
            "amount_low": trade.amount_low,
            "amount_high": trade.amount_high,
            "owner": trade.owner,
        },
    )
