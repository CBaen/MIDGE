"""wave2_3_insider.py - Insider-group Wave 2+3 signal adapters.

Extracted from wave2_3.py. Converts OpenInsider purchases, SEC 13F holdings,
activist 13D filings, and FinViz insider trades to MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime


def from_openinsider(purchase) -> MarketSignal:
    """Convert an InsiderPurchase (OpenInsider) to a MarketSignal.

    OpenInsider pre-filters RSU grants and 10b5-1 plan exercises, so the
    signal quality is higher than raw SEC EDGAR Form 4 data.  All records
    are purchase-only — direction is always bullish.  Strength scales to
    $500K (a large but realistic single-insider buy).
    """
    value = getattr(purchase, "value", 0.0) or 0.0
    ticker = getattr(purchase, "ticker", "") or ""
    filing_date = getattr(purchase, "filing_date", "") or ""
    trade_date = getattr(purchase, "trade_date", "") or ""

    strength = min(1.0, value / 500_000.0)

    # Use trade_date for the underlying event; fall back to filing_date
    event_dt = _ensure_datetime(trade_date or filing_date)

    signal_id = f"openinsider:{ticker}:{trade_date or filing_date}:{getattr(purchase, 'insider_name', '')}"

    return MarketSignal(
        signal_id=signal_id,
        source="openinsider_purchase",
        symbol=ticker,
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=strength,
        confidence=0.55,
        decay_rate=0.05,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=30,
        raw_id="",
        raw_type="InsiderPurchase",
        metadata={
            "insider_name": getattr(purchase, "insider_name", ""),
            "title": getattr(purchase, "title", ""),
            "trade_type": getattr(purchase, "trade_type", ""),
            "price": getattr(purchase, "price", 0.0),
            "quantity": getattr(purchase, "quantity", 0),
            "value": value,
            "delta_owned_pct": getattr(purchase, "delta_owned_pct", 0.0),
            "company_name": getattr(purchase, "company_name", ""),
        },
    )


def from_13f_holding(holding) -> MarketSignal:
    """Convert an InstitutionalHolding (SEC 13F) to a MarketSignal.

    13F filings have a 45-day lag — confidence is neutral (0.50) to reflect
    that the position may have changed since the reporting period.  Strength
    scales to a $10M position (large institutional conviction).
    value_usd is reported in thousands by SEC convention; multiply by 1000
    for the actual dollar figure.
    """
    ticker = getattr(holding, "ticker", "") or ""
    value_usd = getattr(holding, "value_usd", 0.0) or 0.0
    filing_date = getattr(holding, "filing_date", "") or ""
    period = getattr(holding, "period_of_report", "") or ""

    # value_usd is in thousands — convert to actual dollars for strength calc
    actual_value = value_usd * 1000.0
    strength = min(1.0, actual_value / 10_000_000.0)

    event_dt = _ensure_datetime(period or filing_date)

    signal_id = f"13f:{getattr(holding, 'filer_cik', '')}:{ticker}:{period or filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="institutional_13f",
        symbol=ticker,
        asset_class="stock",
        domain="institutional",
        direction="bullish",
        strength=strength,
        confidence=0.50,
        decay_rate=0.03,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=90,
        raw_id=getattr(holding, "filer_cik", ""),
        raw_type="InstitutionalHolding",
        metadata={
            "filer_name": getattr(holding, "filer_name", ""),
            "filer_cik": getattr(holding, "filer_cik", ""),
            "company_name": getattr(holding, "company_name", ""),
            "shares": getattr(holding, "shares", 0),
            "value_usd_thousands": value_usd,
            "period_of_report": period,
            "filing_date": filing_date,
        },
    )


def from_activist_filing(filing) -> MarketSignal:
    """Convert an ActivistFiling (SC 13D) to a MarketSignal.

    Activist investors acquiring >5% stakes signal intent to unlock value —
    historically bullish for the target company.  Strength is fixed at 0.85
    (activist campaigns are inherently high-conviction events).
    Decay rate is very slow (0.02) because campaigns play out over months.
    """
    subject_ticker = getattr(filing, "subject_ticker", "") or ""
    filing_date = getattr(filing, "filing_date", "") or ""

    event_dt = _ensure_datetime(filing_date)

    signal_id = f"activist13d:{getattr(filing, 'filer_cik', '')}:{subject_ticker}:{filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="activist_13d",
        symbol=subject_ticker,
        asset_class="stock",
        domain="institutional",
        direction="bullish",
        strength=0.85,
        confidence=0.60,
        decay_rate=0.02,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=subject_ticker,
        outcome_window_days=90,
        raw_id=getattr(filing, "filer_cik", ""),
        raw_type="ActivistFiling",
        metadata={
            "filer_name": getattr(filing, "filer_name", ""),
            "filer_cik": getattr(filing, "filer_cik", ""),
            "subject_company": getattr(filing, "subject_company", ""),
            "form_type": getattr(filing, "form_type", "SC 13D"),
            "percent_owned": getattr(filing, "percent_owned", 0.0),
            "purpose": getattr(filing, "purpose", ""),
        },
    )


def from_finviz_insider(trade) -> MarketSignal:
    """Convert a FinVizInsiderTrade (Buy) to a MarketSignal.

    FinViz scrapes the same SEC Form 4 filings as EDGAR and OpenInsider but
    presents them through a different lens (screener context, owner titles).
    Using all three sources gives cross-validation: if all three flag the same
    ticker on the same day, the signal confidence is substantially higher.

    Only Buy transactions should be passed here (filtered in fetch_finviz).
    Strength scales to $1M trade value.
    """
    ticker = getattr(trade, "ticker", "") or ""
    value = getattr(trade, "value", 0.0) or 0.0
    date_str = getattr(trade, "date", "") or ""

    strength = min(1.0, value / 1_000_000.0)

    event_dt = _ensure_datetime(date_str) if date_str else datetime.now()

    signal_id = (
        f"finviz_insider:{ticker}:{date_str}:{getattr(trade, 'owner_name', '')}"
    )

    return MarketSignal(
        signal_id=signal_id,
        source="finviz_insider",
        symbol=ticker,
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=strength,
        confidence=0.50,
        decay_rate=0.05,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=ticker,
        outcome_window_days=30,
        raw_id="",
        raw_type="FinVizInsiderTrade",
        metadata={
            "owner_name": getattr(trade, "owner_name", ""),
            "relationship": getattr(trade, "relationship", ""),
            "transaction_type": getattr(trade, "transaction_type", ""),
            "date": date_str,
            "shares_traded": getattr(trade, "shares_traded", 0),
            "value": value,
            "shares_owned": getattr(trade, "shares_owned", 0),
        },
    )
