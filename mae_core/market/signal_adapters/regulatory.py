"""Regulatory signal adapters — SEC filings, insider clusters, correlation signals.

Converts raw regulatory data types into normalized MarketSignal objects.
"""

from __future__ import annotations

import math
from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime
from mae_core.market.apis.sec_edgar.models import InsiderTrade, Form8KEvent
from mae_core.market.edge.cluster_detector import ClusterSignal
from mae_core.market.edge.politician_tracker import CorrelationSignal


def from_insider_trade(trade: InsiderTrade) -> MarketSignal:
    """Convert a Form 4 InsiderTrade to a MarketSignal.

    Filters out noise from scheduled compensation transactions:
    - Transaction code "D" (disposition/delivery for RSU vesting): strength * 0.25
    - Transaction code "A" (award/grant): strength * 0.25
    - Transaction code "F" (tax withholding on vesting): strength * 0.25
    - Suspected 10b5-1 plan sales (non-buy with plan indicators): strength * 0.25
    """
    is_buy = trade.is_purchase
    direction = "bullish" if is_buy else "bearish"

    # Log-linear scale: preserves differentiation at high values
    # $100k -> 0.42, $500k -> 0.73, $1M -> 0.83, $5M -> 0.95, $10M -> 1.0
    if is_buy:
        strength = min(1.0, math.log1p(trade.total_value / 100_000) / math.log1p(10))
    else:
        strength = min(1.0, math.log1p(trade.total_value / 50_000) / math.log1p(10))

    confidence = 0.70

    # Detect compensation/plan transactions (noise, not informed trading)
    compensation_codes = {"D", "A", "F", "G", "M"}  # disposition, award, tax, gift, option exercise
    tc = trade.transaction_code.upper().strip() if trade.transaction_code else ""
    is_compensation = tc in compensation_codes
    # 10b5-1 plan sales: scheduled, not informed (Pichai, Kress, etc.)
    is_plan = trade.is_plan_sale and not is_buy

    if is_compensation or is_plan:
        strength *= 0.25
        confidence = 0.40

    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.filing_date) if trade.filing_date else datetime.now()

    symbol = trade.ticker_symbol or ""
    signal_id = f"sec_form4:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_form4",
        symbol=symbol,
        asset_class="stock",
        domain="insider",
        direction=direction,
        strength=strength,
        confidence=confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.accession_number or "",
        raw_type="InsiderTrade",
        metadata={
            "filer_name": trade.filer_name,
            "filer_title": trade.filer_title,
            "transaction_type": trade.transaction_type,
            "shares": trade.shares,
            "price_per_share": trade.price_per_share,
            "total_value": trade.total_value,
            "company_name": trade.company_name,
        },
    )


def from_form8k_event(event: Form8KEvent) -> MarketSignal:
    """Convert a Form 8-K event to a MarketSignal."""
    # Direction from the item code's known impact
    _, impact = Form8KEvent.get_item_info(event.item_code)
    direction = impact if impact in ("bullish", "bearish") else "neutral"

    event_dt = _ensure_datetime(event.event_date)
    received_dt = _ensure_datetime(event.filing_date) if event.filing_date else datetime.now()

    symbol = event.ticker_symbol or ""
    signal_id = f"sec_form8k:{symbol}:{event.event_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_form8k",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=event.confidence,  # 0.50–0.70 range per source model
        confidence=event.confidence,
        decay_rate=event.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=event.accession_number or "",
        raw_type="Form8KEvent",
        metadata={
            "item_code": event.item_code,
            "item_description": event.item_description,
            "event_summary": event.event_summary,
            "material_impact": event.material_impact,
            "company_name": event.company_name,
        },
    )


def from_filing_keyword(hit) -> MarketSignal:
    """Convert a SEC EFTS FilingKeywordHit to a MarketSignal.

    Keyword matches in 8-K filings are event-driven signals.
    "Tender offer" = M&A bullish. "Going concern" = distress bearish.
    """
    from mae_core.market.apis.sec_edgar.efts import SECFullTextSearchClient

    # Direction from keyword category
    keyword_lower = hit.keyword_matched.lower()
    if keyword_lower in [k.lower() for k in SECFullTextSearchClient.BULLISH_KEYWORDS]:
        direction = "bullish"
    elif keyword_lower in [k.lower() for k in SECFullTextSearchClient.BEARISH_KEYWORDS]:
        direction = "bearish"
    else:
        direction = "neutral"

    # M&A keywords are higher strength than general distress
    if "tender offer" in keyword_lower or "merger" in keyword_lower:
        strength = 0.90
    elif "going concern" in keyword_lower or "restatement" in keyword_lower:
        strength = 0.85
    else:
        strength = 0.65

    event_dt = _ensure_datetime(hit.filing_date)

    symbol = hit.ticker or ""
    signal_id = f"sec_efts:{symbol}:{hit.keyword_matched}:{hit.filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_efts",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=hit.confidence,
        decay_rate=hit.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="FilingKeywordHit",
        metadata={
            "keyword_matched": hit.keyword_matched,
            "form_type": hit.form_type,
            "description": hit.description,
            "company_name": hit.company_name,
        },
    )


def from_cluster_signal(cluster: ClusterSignal) -> MarketSignal:
    """Convert an insider ClusterSignal to a MarketSignal."""
    # Clusters are always bullish (cluster_detector only tracks buys)
    direction = "bullish"

    event_dt = _ensure_datetime(cluster.detected_at)

    signal_id = f"insider_cluster:{cluster.symbol}:{cluster.cluster_id}"

    return MarketSignal(
        signal_id=signal_id,
        source="insider_cluster",
        symbol=cluster.symbol,
        asset_class="stock",
        domain="insider",
        direction=direction,
        strength=cluster.confidence,
        confidence=cluster.confidence,
        decay_rate=cluster.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=cluster.symbol,
        raw_id=cluster.cluster_id or "",
        raw_type="ClusterSignal",
        metadata={
            "insider_count": cluster.insider_count,
            "total_value": cluster.total_value,
            "weighted_score": cluster.weighted_score,
            "avg_conviction": cluster.avg_conviction,
            "has_csuite": cluster.has_csuite,
            "window_days": cluster.window_days,
        },
    )


def from_correlation_signal(correlation: CorrelationSignal) -> MarketSignal:
    """Convert a politician/insider CorrelationSignal to a MarketSignal."""
    is_buy = correlation.trade_type == "buy" if hasattr(correlation, "trade_type") else True
    direction = "bullish" if is_buy else "bearish"

    event_dt = _ensure_datetime(correlation.trade_date)

    symbol = correlation.symbol or ""
    signal_id = f"politician_correlation:{symbol}:{correlation.trade_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="politician_correlation",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=correlation.confidence,
        confidence=correlation.confidence,
        decay_rate=0.04,  # ~17 day half-life (combination of trade staleness + political info)
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="CorrelationSignal",
        metadata={
            "trader_name": correlation.trader_name,
            "trade_value": correlation.value,
            "correlation_type": correlation.correlation_type,
            "contract_value": correlation.contract_value,
            "awarding_agency": correlation.awarding_agency,
            "committee": correlation.committee,
            "oversight_match": correlation.oversight_match,
            "days_between": correlation.days_between,
        },
    )
