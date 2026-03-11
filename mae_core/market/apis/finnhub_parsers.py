#!/usr/bin/env python3
"""
finnhub_parsers.py - Static parser methods for Finnhub API responses.

Extracted from finnhub_client.py. Each function converts a raw Finnhub
JSON payload into strongly-typed dataclasses from finnhub_models.py.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from mae_core.market.apis.finnhub_models import (
    AnalystRec,
    EarningsEvent,
    EconomicEvent,
    NewsSentiment,
)

logger = logging.getLogger(__name__)


def parse_sentiment(symbol: str, data: dict) -> Optional[NewsSentiment]:
    """
    Parse /news-sentiment response into a NewsSentiment dataclass.

    Expected keys:
        buzz.articlesInLastWeek, buzz.weeklyAverage,
        companyNewsScore, sentiment.bullishPercent, sentiment.bearishPercent
    """
    try:
        buzz_block = data.get("buzz", {})
        articles_this_week = float(buzz_block.get("articlesInLastWeek", 0))
        weekly_average     = float(buzz_block.get("weeklyAverage", 1) or 1)
        buzz_score         = articles_this_week / weekly_average

        sentiment_block = data.get("sentiment", {})
        bullish_pct = float(sentiment_block.get("bullishPercent", 0.5))
        bearish_pct = float(sentiment_block.get("bearishPercent", 0.5))
        news_score  = float(data.get("companyNewsScore", 0.0))

        return NewsSentiment(
            ticker=symbol,
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            news_score=news_score,
            buzz_score=buzz_score,
            detected_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.warning("Failed to parse sentiment for %s: %s", symbol, exc)
        return None


def parse_earnings_calendar(data: dict) -> List[EarningsEvent]:
    """
    Parse /calendar/earnings response into EarningsEvent objects.

    Expected structure: {"earningsCalendar": [...]}
    Each entry has: symbol, date, epsActual, epsEstimate, hour,
                    quarter, revenueActual, revenueEstimate, year
    """
    events: List[EarningsEvent] = []
    calendar = data.get("earningsCalendar") or []

    for entry in calendar:
        try:
            symbol = (entry.get("symbol") or "").upper()
            if not symbol:
                continue

            date = entry.get("date") or ""

            # Actual fields are null until the report is released
            eps_actual = entry.get("epsActual")
            eps_estimate = entry.get("epsEstimate")
            rev_actual = entry.get("revenueActual")
            rev_estimate = entry.get("revenueEstimate")

            # Cast non-null numerics; keep None as-is
            eps_actual    = float(eps_actual)    if eps_actual    is not None else None
            eps_estimate  = float(eps_estimate)  if eps_estimate  is not None else None
            rev_actual    = float(rev_actual)    if rev_actual    is not None else None
            rev_estimate  = float(rev_estimate)  if rev_estimate  is not None else None

            hour = (entry.get("hour") or "").lower()  # "bmo" or "amc"

            events.append(EarningsEvent(
                symbol=symbol,
                date=date,
                eps_estimate=eps_estimate,
                eps_actual=eps_actual,
                revenue_estimate=rev_estimate,
                revenue_actual=rev_actual,
                hour=hour,
            ))
        except Exception as exc:
            logger.debug("Skipping malformed earnings entry: %s | %s", entry, exc)
            continue

    return events


def parse_economic_calendar(data: dict) -> List[EconomicEvent]:
    """
    Parse /calendar/economic response into EconomicEvent objects.

    Expected structure: {"economicCalendar": [...]}
    Each entry has: country, event, date, time, impact, actual, estimate, prev, unit
    """
    # Major economies whose central bank / macro events move forex & commodities
    MAJOR_ECONOMIES = {"US", "EU", "JP", "GB", "CN", "CA", "AU"}

    events: List[EconomicEvent] = []
    calendar = data.get("economicCalendar") or []

    for entry in calendar:
        try:
            country = (entry.get("country") or "").upper()
            impact = (entry.get("impact") or "low").lower()

            # US: all impact levels. Others: high-impact only (rate decisions, CPI, GDP)
            if country not in MAJOR_ECONOMIES:
                continue
            if country != "US" and impact != "high":
                continue

            event_name = entry.get("event") or ""
            if not event_name:
                continue

            date = entry.get("date") or ""
            event_time = entry.get("time") or ""

            actual = entry.get("actual")
            estimate = entry.get("estimate")
            previous = entry.get("prev")
            unit = entry.get("unit") or ""

            actual   = float(actual)   if actual   is not None else None
            estimate = float(estimate) if estimate is not None else None
            previous = float(previous) if previous is not None else None

            events.append(EconomicEvent(
                event=event_name,
                country=country,
                date=date,
                time=event_time,
                impact=impact,
                actual=actual,
                estimate=estimate,
                previous=previous,
                unit=unit,
            ))
        except Exception as exc:
            logger.debug("Skipping malformed economic entry: %s | %s", entry, exc)
            continue

    # Sort by date ascending
    events.sort(key=lambda e: e.date)
    return events


def parse_analyst_recommendations(data: list) -> List[AnalystRec]:
    """
    Parse /stock/recommendation response into AnalystRec objects.

    Response is a list of dicts with: buy, hold, period, sell, strongBuy, strongSell, symbol
    """
    results: List[AnalystRec] = []

    for entry in data:
        try:
            symbol = (entry.get("symbol") or "").upper()
            if not symbol:
                continue

            results.append(AnalystRec(
                symbol=symbol,
                period=entry.get("period", ""),
                strong_buy=int(entry.get("strongBuy", 0)),
                buy=int(entry.get("buy", 0)),
                hold=int(entry.get("hold", 0)),
                sell=int(entry.get("sell", 0)),
                strong_sell=int(entry.get("strongSell", 0)),
            ))
        except Exception as exc:
            logger.debug("Skipping malformed analyst entry: %s | %s", entry, exc)
            continue

    # Most recent first
    results.sort(key=lambda r: r.period, reverse=True)
    return results
