"""Social and news fetch functions — sentiment, StockTwits, Trends, Finnhub, Yahoo RSS.

Covers social media momentum, news sentiment, economic calendar events,
analyst recommendations, and headline velocity signals.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("midge.market.sensing")


def fetch_social_sentiment(apewisdom: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch Reddit/WSB social sentiment from ApeWisdom."""
    if apewisdom is None:
        return []

    signals = []
    try:
        # Get accelerating tickers (2x+ mention velocity) — these are the signal
        accelerating = apewisdom.get_accelerating_tickers(min_change=2.0, limit=10)
        for sentiment in accelerating:
            try:
                signals.append(converter(sentiment))
            except Exception:
                pass

        # Also check watchlist tickers directly
        for ticker in watchlist.get("tickers", []):
            try:
                sentiment = apewisdom.get_by_ticker(ticker)
                if sentiment is not None and sentiment.mention_change >= 1.5:
                    signals.append(converter(sentiment))
            except Exception:
                pass
    except Exception as e:
        logger.debug("ApeWisdom fetch failed: %s", e)
    return signals


def fetch_stocktwits(stocktwits_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch StockTwits bull/bear sentiment for watchlist tickers."""
    if stocktwits_client is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:10]  # Cap at 10 for rate limits
    try:
        sentiments = stocktwits_client.get_sentiment(tickers)
        for st in sentiments:
            try:
                signals.append(converter(st))
            except Exception:
                pass
    except Exception as e:
        logger.debug("StockTwits fetch failed: %s", e)
    return signals


def fetch_trends(trends_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch Google Trends interest for watchlist tickers + keywords + macro terms.

    Watchlist keywords (cybersecurity, AI, defense, space) are included so
    MIDGE tracks cultural/political attention on her focus sectors, not just
    individual tickers.  The TrendsClient also injects any discovered rising
    queries from previous cycles, creating a self-expanding keyword antenna.
    """
    if trends_client is None:
        return []

    signals = []
    # Mix watchlist tickers + watchlist cultural keywords + macro fear terms
    tickers = watchlist.get("tickers", [])[:5]
    cultural_keywords = watchlist.get("keywords", [])  # e.g. "cybersecurity", "AI"
    macro_terms = ["recession", "market crash", "fed rate"]
    keywords = tickers + cultural_keywords + macro_terms

    try:
        trends = trends_client.get_interest(keywords)
        for trend in trends:
            try:
                signals.append(converter(trend))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Google Trends fetch failed: %s", e)
    return signals


def fetch_social_text(
    social_text_analyzer: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Run text theme analysis on stored StockTwits messages.

    Reads from the StockTwits SQLite DB (populated by StockTwitsClient).
    No API calls — pure local computation.  Runs on every sensing cycle,
    returns signals only when theme thresholds are crossed.
    """
    if social_text_analyzer is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])
    try:
        results = social_text_analyzer.analyze_all(tickers if tickers else None)
        for result in results:
            try:
                signals.append(converter(result))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Social text analysis failed: %s", e)
    return signals


def fetch_yahoo_rss(
    yahoo_rss_client: Any,
    watchlist: dict,
    converter: Callable,
    min_velocity: float = 1.5,
) -> list:
    """Fetch Yahoo Finance RSS headline velocity signals for watchlist tickers.

    Only returns tickers with at least min_velocity headline rate change vs the
    prior window.  This filters out the daily background noise and surfaces
    names where something is genuinely happening right now.

    Yahoo RSS is per-ticker (one feed per symbol), so we cap at 15 tickers per
    cycle to stay polite with Yahoo's servers and stay within the 5-minute cache.

    Args:
        yahoo_rss_client: YahooRSSClient instance.
        watchlist: Watchlist dict with "tickers" list.
        converter: from_yahoo_rss_signal adapter function.
        min_velocity: Minimum headline velocity ratio to emit a signal (default 1.5).

    Returns:
        List of MarketSignal objects for tickers with accelerating headline flow.
    """
    if yahoo_rss_client is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:15]  # Cap per cycle — one RSS call per ticker
    try:
        headline_signals = yahoo_rss_client.get_accelerating(
            tickers, min_velocity=min_velocity
        )
        for hs in headline_signals:
            try:
                signals.append(converter(hs))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Yahoo RSS fetch failed: %s", e)
    return signals


def fetch_finnhub(
    finnhub: Any,
    watchlist: dict,
    news_converter: Callable,
    earnings_converter: Callable,
) -> list:
    """Fetch Finnhub news sentiment + earnings surprises."""
    if finnhub is None:
        return []

    signals = []

    # News sentiment for watchlist tickers
    for ticker in watchlist.get("tickers", []):
        try:
            sentiment = finnhub.get_news_sentiment(ticker)
            if sentiment is not None:
                signals.append(news_converter(sentiment))
        except Exception as e:
            logger.debug("Finnhub news sentiment failed for %s: %s", ticker, e)

    # Recent earnings surprises
    try:
        reported = finnhub.get_recent_earnings_surprises(days=7)
        for event in reported:
            try:
                signals.append(earnings_converter(event))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Finnhub earnings fetch failed: %s", e)

    return signals


def fetch_news_headlines(news_aggregator_client: Any) -> list:
    """Fetch free news headlines from RSS feeds — events domain signals."""
    if news_aggregator_client is None:
        return []
    from datetime import datetime, timezone
    signals = []
    try:
        headlines = news_aggregator_client.get_headlines(max_per_source=10)
        for h in headlines:
            if not h.tickers or h.sentiment == "neutral":
                continue
            direction = "bullish" if h.sentiment == "positive" else "bearish"
            for ticker in h.tickers[:3]:
                signals.append({
                    "source": "news_headlines",
                    "symbol": ticker,
                    "asset_class": "stock",
                    "domain": "events",
                    "direction": direction,
                    "strength": 0.45,
                    "confidence": 0.40,
                    "decay_rate": 0.30,
                    "timestamp": h.published or datetime.now(timezone.utc).isoformat(),
                    "outcome_window_days": 5,
                    "metadata": {
                        "title": h.title[:200],
                        "source_name": h.source,
                        "sentiment": h.sentiment,
                        "keywords": h.keywords[:5],
                        "signal_source": "news_aggregator",
                    },
                })
    except Exception as e:
        logger.debug("News headlines fetch failed: %s", e)
    return signals


def fetch_finnhub_extras(
    finnhub: Any,
    watchlist: dict,
    econ_converter: Callable,
    analyst_converter: Callable,
    surprise_converter: Optional[Callable] = None,
) -> list:
    """Fetch Finnhub economic calendar + analyst recommendations.

    For each economic event, produces:
      - A catalyst/suppression signal via econ_converter (from_economic_event,
        domain="events", fires for all high-impact events)
      - An economic surprise signal via surprise_converter (from_economic_surprise,
        domain="macro", fires only when actual vs estimate exceeds 0.5% threshold)

    The two signals use different domains so they contribute independently to
    convergence scoring.
    """
    if finnhub is None:
        return []

    signals = []

    # Economic calendar — produce both event and surprise signals per entry
    try:
        events = finnhub.get_economic_calendar(days=7)
        for event in events:
            try:
                signals.append(econ_converter(event))
            except Exception:
                pass
            if surprise_converter is not None:
                try:
                    surprise_sig = surprise_converter(event)
                    if surprise_sig is not None:
                        signals.append(surprise_sig)
                except Exception:
                    pass
    except Exception as e:
        logger.debug("Finnhub economic calendar failed: %s", e)

    # Analyst recommendations for watchlist tickers
    tickers = watchlist.get("tickers", [])[:5]
    for ticker in tickers:
        try:
            recs = finnhub.get_analyst_recommendations(ticker)
            if recs:
                # Only use the most recent recommendation period
                signals.append(analyst_converter(recs[0]))
        except Exception as e:
            logger.debug("Finnhub analyst recs failed for %s: %s", ticker, e)

    return signals
