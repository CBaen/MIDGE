#!/usr/bin/env python3
"""
finnhub_models.py - Dataclasses for Finnhub market data signals.

Extracted from finnhub_client.py to keep models separate from HTTP logic.
Provides: NewsSentiment, EconomicEvent, AnalystRec, EarningsEvent.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NewsSentiment:
    """
    News sentiment snapshot for a single ticker from Finnhub.

    buzz_score = articles_in_last_week / weekly_average
    Values > 1.0 mean above-average coverage volume.
    """
    ticker: str
    bullish_pct: float          # 0.0 – 1.0  (e.g. 0.65 = 65% bullish)
    bearish_pct: float          # 0.0 – 1.0
    news_score: float           # Finnhub's companyNewsScore (0–1)
    buzz_score: float           # articles_in_last_week / weekly_average
    detected_at: str            # ISO-8601 timestamp when we fetched this
    signal_source: str = "finnhub_news"
    decay_rate: float = 0.20    # ~3-day half-life; sentiment fades fast
    confidence: float = 0.50    # Base confidence before Bayesian update

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        buzz_label = "above-average" if self.buzz_score > 1.0 else "below-average"
        sentiment_label = "bullish" if self.bullish_pct >= 0.5 else "bearish"
        return (
            f"{self.ticker}: {sentiment_label} ({self.bullish_pct:.0%} bullish, "
            f"{self.bearish_pct:.0%} bearish), {buzz_label} buzz "
            f"(score {self.buzz_score:.2f}), news score {self.news_score:.2f}"
        )


@dataclass
class EconomicEvent:
    """
    Economic calendar entry from Finnhub (FOMC, CPI, NFP, etc.).

    High-impact events move markets. actual vs estimate determines surprise.
    """
    event: str                          # "CPI", "FOMC", "Nonfarm Payrolls", etc.
    country: str                        # "US", "EU", etc.
    date: str                           # YYYY-MM-DD
    time: str                           # HH:MM or ""
    impact: str                         # "high", "medium", "low"
    actual: Optional[float]             # None if not yet released
    estimate: Optional[float]
    previous: Optional[float]
    unit: str                           # "%", "K", etc.
    signal_source: str = "finnhub_economic"
    decay_rate: float = 0.30            # Economic events price in fast
    confidence: float = 0.55

    def surprise_pct(self) -> Optional[float]:
        """Percentage surprise vs estimate. None if data missing."""
        if self.actual is None or self.estimate is None:
            return None
        if self.estimate == 0:
            return None
        return (self.actual - self.estimate) / abs(self.estimate)

    def to_plain_language(self) -> str:
        if self.actual is not None:
            surprise = self.surprise_pct()
            s_str = f" (surprise {surprise:+.1%})" if surprise else ""
            return f"{self.event} on {self.date}: actual={self.actual}{self.unit}{s_str}"
        return f"{self.event} on {self.date}: estimate={self.estimate}{self.unit} (upcoming)"


@dataclass
class AnalystRec:
    """
    Analyst recommendation summary from Finnhub for a ticker.

    Aggregates buy/sell/hold/strong_buy/strong_sell counts.
    """
    symbol: str
    period: str                         # YYYY-MM-DD
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    signal_source: str = "finnhub_analyst"
    decay_rate: float = 0.05            # Slow decay — analyst recs are monthly
    confidence: float = 0.50

    @property
    def total(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def buy_ratio(self) -> float:
        """Fraction of analysts with buy/strong_buy. 0.0-1.0."""
        t = self.total
        return (self.strong_buy + self.buy) / max(1, t)

    def to_plain_language(self) -> str:
        return (
            f"{self.symbol}: {self.strong_buy} strong buy, {self.buy} buy, "
            f"{self.hold} hold, {self.sell} sell, {self.strong_sell} strong sell "
            f"(period {self.period})"
        )


@dataclass
class EarningsEvent:
    """
    Single earnings calendar entry from Finnhub.

    eps_actual and revenue_actual are None when the report has not
    yet been released.  Once reported they are floats.
    """
    symbol: str
    date: str                           # YYYY-MM-DD
    eps_estimate: Optional[float]
    eps_actual: Optional[float]         # None if not yet reported
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]     # None if not yet reported
    hour: str                           # "bmo" (before market open) | "amc" (after market close)
    signal_source: str = "finnhub_earnings"
    decay_rate: float = 0.25            # Earnings are priced in fast
    confidence: float = 0.65           # Higher base; hard calendar events

    def is_reported(self) -> bool:
        """True when actual results have come in."""
        return self.eps_actual is not None

    def eps_surprise_pct(self) -> Optional[float]:
        """
        Percentage EPS surprise vs estimate.

        Returns None when either value is missing or estimate is zero.
        """
        if self.eps_actual is None or self.eps_estimate is None:
            return None
        if self.eps_estimate == 0:
            return None
        return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate)

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        timing = "before open" if self.hour == "bmo" else "after close"
        if self.is_reported():
            surprise = self.eps_surprise_pct()
            surprise_str = (
                f" — EPS surprise {surprise:+.1%}" if surprise is not None
                else " — EPS reported"
            )
            return f"{self.symbol} reported on {self.date}{surprise_str}"
        else:
            est_str = f"${self.eps_estimate:.2f}" if self.eps_estimate is not None else "N/A"
            return (
                f"{self.symbol} earnings on {self.date} {timing}, "
                f"EPS estimate {est_str}"
            )
