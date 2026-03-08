"""Market Sensing Fetchers — standalone fetch functions for each data source.

Each function is extracted from MarketSensingHook._fetch_*() methods.
All functions take explicit parameters instead of relying on `self`,
making them independently testable and composable.

Called by MarketSensingHook._fetch_source() router in sensing_hook.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("midge.market.sensing")


def fetch_sec_form4(sec_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch SEC Form 4 insider trades for watchlist tickers."""
    if sec_client is None:
        return []

    from mae_core.market.apis.sec_edgar import get_recent_form4s

    signals = []
    for ticker in watchlist.get("tickers", []):
        try:
            trades = get_recent_form4s(ticker, days=30)
            for trade in trades:
                try:
                    signals.append(converter(trade))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SEC Form 4 fetch failed for %s: %s", ticker, e)
    return signals


def fetch_sec_form8k(sec_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch SEC Form 8-K events for watchlist tickers."""
    if sec_client is None:
        return []

    from mae_core.market.apis.sec_edgar import get_recent_form8ks

    signals = []
    for ticker in watchlist.get("tickers", []):
        try:
            events = get_recent_form8ks(ticker, days=30)
            for event in events:
                try:
                    signals.append(converter(event))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SEC Form 8-K fetch failed for %s: %s", ticker, e)
    return signals


def fetch_congressional(congress_client: Any, converter: Callable) -> list:
    """Fetch congressional stock trades."""
    if congress_client is None:
        return []

    signals = []
    try:
        trades = congress_client.get_recent_trades(days=30)
        for trade in trades:
            try:
                # Filter: trades below $50K are noise
                if trade.amount_high < 50_000:
                    continue
                signals.append(converter(trade))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Congressional trades fetch failed: %s", e)
    return signals


def fetch_senate(senate_client: Any, converter: Callable) -> list:
    """Fetch Senate stock trades."""
    if senate_client is None:
        return []

    signals = []
    try:
        trades = senate_client.get_recent_trades(days=30)
        for trade in trades:
            try:
                if trade.amount_high < 50_000:
                    continue
                signals.append(converter(trade))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Senate trades fetch failed: %s", e)
    return signals


def fetch_hiring(job_tracker: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch hiring signals for watchlist companies."""
    if job_tracker is None:
        return []

    signals = []
    companies = watchlist.get("companies", {})
    for company, ticker in companies.items():
        try:
            signal = job_tracker.analyze_hiring_activity(company, ticker=ticker)
            signals.append(converter(signal))
        except Exception as e:
            logger.debug("Hiring fetch failed for %s: %s", company, e)
    return signals


def fetch_usa_spending(usa_spending: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch government contracts from USASpending."""
    if usa_spending is None:
        return []

    signals = []
    for keyword in watchlist.get("keywords", []):
        try:
            contracts = usa_spending.search_contracts(keyword=keyword, limit=5)
            for contract in contracts:
                try:
                    signals.append(converter(contract))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("USASpending fetch failed for '%s': %s", keyword, e)
    return signals


def fetch_sam_gov(sam_gov: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch SAM.gov opportunities."""
    if sam_gov is None:
        return []

    signals = []
    for keyword in watchlist.get("keywords", []):
        try:
            opps = sam_gov.search_opportunities(keywords=keyword, limit=5)
            for opp in opps:
                try:
                    signals.append(converter(opp))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SAM.gov fetch failed for '%s': %s", keyword, e)
    return signals


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


def fetch_finra_short(finra_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch FINRA daily short volume — high short ratio tickers."""
    if finra_client is None:
        return []

    signals = []
    try:
        # Get tickers with >50% short volume ratio
        high_short = finra_client.get_high_short_ratio(min_ratio=0.5)
        # Filter to watchlist + top 10 highest ratios
        watchlist_tickers = set(watchlist.get("tickers", []))
        for i, record in enumerate(high_short):
            try:
                if record.symbol in watchlist_tickers or i < 10:
                    signals.append(converter(record))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FINRA short volume fetch failed: %s", e)
    return signals


def fetch_sec_efts(sec_efts: Any, converter: Callable) -> list:
    """Fetch SEC EFTS full-text search keyword hits."""
    if sec_efts is None:
        return []

    signals = []
    try:
        hits = sec_efts.scan_all_keywords(days=3)
        for hit in hits:
            try:
                signals.append(converter(hit))
            except Exception:
                pass
    except Exception as e:
        logger.debug("SEC EFTS fetch failed: %s", e)
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


def fetch_fred(fred: Any, converter: Callable) -> list:
    """Fetch FRED macroeconomic indicators."""
    if fred is None:
        return []

    signals = []
    try:
        snapshot = fred.get_macro_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FRED macro fetch failed: %s", e)
    return signals


def fetch_session_sweep(session_sweep_detector: Any, converter: Callable) -> list:
    """Fetch ICT session sweep signals for futures.

    Kill-zone time guard: returns early if not within 90 min of a
    kill zone window. Prevents wasting yfinance rate limit during
    dead hours.
    """
    if session_sweep_detector is None:
        return []

    # Time-of-day guard (Eastern time)
    try:
        from zoneinfo import ZoneInfo
        from datetime import time as _time
        now_et = datetime.now(ZoneInfo("America/New_York")).time()
        # Kill zone windows with ±90 min buffer
        kz_windows = [
            (_time(18, 30), _time(23, 59)),  # Asia buffer (evening)
            (_time(0, 0), _time(6, 30)),     # Asia + London buffer
            (_time(5, 30), _time(11, 30)),   # NY kill zone buffer
        ]
        in_window = any(s <= now_et <= e for s, e in kz_windows)
        if not in_window:
            logger.debug("Session sweep: outside kill zone window, skipping")
            return []
    except Exception:
        pass  # If timezone check fails, proceed anyway

    signals = []
    futures_symbols = ["ES=F", "NQ=F"]
    for symbol in futures_symbols:
        try:
            sweeps = session_sweep_detector.detect_sweeps(symbol)
            for sweep in sweeps:
                try:
                    signals.append(converter(sweep))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Session sweep fetch failed for %s: %s", symbol, e)
    return signals


def fetch_ta_indicators(
    ta_indicators: Any,
    price_fetcher: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Compute technical analysis indicators for watchlist tickers.

    Uses price_fetcher.get_daily_history() for OHLCV data, then runs
    RSI, MACD, Bollinger, Market Structure, and Candlestick detection.
    Pure local computation — no external API calls beyond yfinance history.
    """
    if ta_indicators is None or price_fetcher is None:
        return []

    from mae_core.market.edge.ta_indicators import compute_all

    signals = []
    for ticker in watchlist.get("tickers", []):
        try:
            history = price_fetcher.get_daily_history(ticker, days=90)
            if not history:
                continue
            ta_signals = compute_all(ticker, history)
            for ta_sig in ta_signals:
                try:
                    signals.append(converter(ta_sig))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("TA indicators failed for %s: %s", ticker, e)
    return signals


def fetch_cot(cot_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch CFTC Commitments of Traders for futures watchlist."""
    if cot_client is None:
        return []

    signals = []
    # COT is futures-only — use futures tickers from watchlist or defaults
    futures_tickers = [t for t in watchlist.get("tickers", []) if t.endswith("=F")]
    if not futures_tickers:
        futures_tickers = ["ES=F", "NQ=F", "GC=F", "CL=F"]

    try:
        positions = cot_client.get_latest_positions(futures_tickers)
        for pos in positions:
            try:
                signals.append(converter(pos))
            except Exception:
                pass
    except Exception as e:
        logger.debug("COT fetch failed: %s", e)
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


def fetch_vix(vix_client: Any, converter: Callable) -> list:
    """Fetch CBOE VIX term structure."""
    if vix_client is None:
        return []

    signals = []
    try:
        vix = vix_client.get_vix_structure()
        if vix is not None:
            signals.append(converter(vix))
    except Exception as e:
        logger.debug("VIX fetch failed: %s", e)
    return signals


def fetch_trends(trends_client: Any, watchlist: dict, converter: Callable) -> list:
    """Fetch Google Trends interest for watchlist tickers + macro terms."""
    if trends_client is None:
        return []

    signals = []
    # Mix watchlist tickers with macro fear terms
    tickers = watchlist.get("tickers", [])[:5]
    macro_terms = ["recession", "market crash", "fed rate"]
    keywords = tickers + macro_terms

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


def fetch_finnhub_extras(
    finnhub: Any,
    watchlist: dict,
    econ_converter: Callable,
    analyst_converter: Callable,
) -> list:
    """Fetch Finnhub economic calendar + analyst recommendations."""
    if finnhub is None:
        return []

    signals = []

    # Economic calendar
    try:
        events = finnhub.get_economic_calendar(days=7)
        for event in events:
            try:
                signals.append(econ_converter(event))
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


def fetch_fractal_resonance(
    fractal_detector: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch fractal resonance signals — multi-timeframe structure alignment.

    Calls FractalResonanceDetector.detect_resonance() for each ticker.
    Cadence 200 in SOURCE_ROTATION (weekly refresh rate).
    """
    if fractal_detector is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:10]
    for ticker in tickers:
        try:
            result = fractal_detector.detect_resonance(ticker)
            if result is not None and result.resonance_score > 0.3:
                try:
                    signals.append(converter(result))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Fractal resonance fetch failed for %s: %s", ticker, e)
    return signals


def fetch_order_flow(
    order_flow_detector: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch order flow imbalance signals for watchlist tickers.

    Calls OrderFlowDetector.detect_imbalance() for each ticker, then converts
    detected imbalances into MarketSignal via the from_order_flow adapter.
    """
    if order_flow_detector is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])[:10]
    for ticker in tickers:
        try:
            results = order_flow_detector.detect_imbalance(ticker)
            for result in results:
                try:
                    signals.append(converter(result))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Order flow fetch failed for %s: %s", ticker, e)
    return signals


def fetch_crypto_prices(coingecko_client: Any, converter: Callable) -> list:
    """Fetch crypto prices from CoinGecko."""
    if coingecko_client is None:
        return []
    signals = []
    try:
        prices = coingecko_client.get_prices()
        for price in prices:
            try:
                signals.append(converter(price))
            except Exception:
                pass
    except Exception as e:
        logger.debug("CoinGecko fetch failed: %s", e)
    return signals


def fetch_crypto_exchange(coincap_client: Any, converter: Callable) -> list:
    """Fetch crypto from CoinCap."""
    if coincap_client is None:
        return []
    signals = []
    try:
        assets = coincap_client.get_assets(limit=10)
        for asset in assets:
            try:
                signals.append(converter(asset))
            except Exception:
                pass
    except Exception as e:
        logger.debug("CoinCap fetch failed: %s", e)
    return signals


def fetch_openinsider(openinsider_client: Any, converter: Callable) -> list:
    """Fetch pre-filtered insider purchases AND cluster buys from OpenInsider."""
    if openinsider_client is None:
        return []
    signals = []
    try:
        purchases = openinsider_client.get_recent_purchases(days=7)
        for purchase in purchases:
            try:
                signals.append(converter(purchase))
            except Exception:
                pass
    except Exception as e:
        logger.debug("OpenInsider fetch failed: %s", e)
    # Cluster buys — 3+ insiders buying same stock = high-confidence signal
    try:
        clusters = openinsider_client.get_cluster_buys(min_insiders=3)
        for cluster in clusters:
            try:
                from mae_core.market.signal_adapters.market_data import MarketSignal
                from datetime import datetime
                strength = min(1.0, 0.5 + (cluster.insider_count - 3) * 0.1
                               + min(cluster.total_value / 5_000_000, 0.3))
                signals.append(MarketSignal(
                    signal_id=f"openinsider_cluster:{cluster.ticker}:{cluster.date_range}",
                    source="openinsider_cluster",
                    symbol=cluster.ticker,
                    asset_class="stock",
                    domain="insider",
                    direction="bullish",
                    strength=strength,
                    confidence=strength,
                    decay_rate=0.03,
                    timestamp=datetime.now(),
                    received_at=datetime.now(),
                    outcome_symbol=cluster.ticker,
                    raw_id="",
                    raw_type="ClusterBuy",
                    metadata={
                        "insider_count": cluster.insider_count,
                        "total_value": cluster.total_value,
                        "company_name": cluster.company_name,
                        "date_range": cluster.date_range,
                    },
                ))
            except Exception:
                pass
    except Exception as e:
        logger.debug("OpenInsider cluster fetch failed: %s", e)
    return signals


def fetch_13f_holdings(
    edgar_enhanced_client: Any,
    converter: Callable,
    activist_converter: Callable,
) -> list:
    """Fetch 13F holdings + 13D activist filings from SEC EDGAR."""
    if edgar_enhanced_client is None:
        return []
    signals = []
    # 13D activist filings (high signal — someone took >5% stake)
    try:
        filings = edgar_enhanced_client.get_13d_filings(days=30)
        for filing in filings:
            try:
                signals.append(activist_converter(filing))
            except Exception:
                pass
    except Exception as e:
        logger.debug("SEC 13D fetch failed: %s", e)
    return signals


def fetch_finviz(
    finviz_client: Any,
    volume_converter: Callable,
    squeeze_converter: Callable,
) -> list:
    """Fetch FinViz unusual volume + short squeeze candidates."""
    if finviz_client is None:
        return []
    signals = []
    # Unusual volume
    try:
        uv_list = finviz_client.get_unusual_volume()
        for uv in uv_list[:20]:  # Cap at 20
            try:
                signals.append(volume_converter(uv))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FinViz unusual volume fetch failed: %s", e)
    # Short squeeze candidates
    try:
        sq_list = finviz_client.get_high_short_float(min_pct=20)
        for sq in sq_list[:10]:  # Cap at 10
            try:
                signals.append(squeeze_converter(sq))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FinViz short squeeze fetch failed: %s", e)
    return signals


def fetch_economic_calendar(calendar_client: Any, converter: Callable) -> list:
    """Fetch upcoming high-impact economic events as suppression signals."""
    if calendar_client is None:
        return []
    signals = []
    try:
        events = calendar_client.get_upcoming_events(days=7)
        # Only high-impact events
        high_impact = [e for e in events if e.impact == "high"]
        for event in high_impact:
            try:
                signals.append(converter(event))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Economic calendar fetch failed: %s", e)
    return signals


def fetch_eia(eia_client: Any, converter: Callable) -> list:
    """Fetch EIA energy inventory/production indicators."""
    if eia_client is None:
        return []

    signals = []
    try:
        snapshot = eia_client.get_energy_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("EIA energy fetch failed: %s", e)
    return signals


def fetch_congress_legislation(congress_gov_client: Any, converter: Callable) -> list:
    """Fetch advancing legislative indicators from Congress.gov."""
    if congress_gov_client is None:
        return []

    signals = []
    try:
        snapshot = congress_gov_client.get_legislative_snapshot()
        for indicator in snapshot:
            try:
                signals.append(converter(indicator))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Congress.gov legislation fetch failed: %s", e)
    return signals


def fetch_massive_snapshot(
    massive_client: Any,
    watchlist: dict,
    converter: Callable,
) -> list:
    """Fetch Massive/Polygon grouped daily data and generate signals.

    Uses ONE grouped daily call for ALL tickers OHLCV, then fetches 20-day
    historical bars per ticker for accurate volume_ratio calculation.
    Free tier: 5 calls/min — one grouped daily + up to 4 per-ticker bars
    within the rate window. Caps at 4 tickers per cycle to stay under limit.
    """
    if massive_client is None:
        return []

    signals = []
    tickers = watchlist.get("tickers", [])
    if not tickers:
        return []

    # Fetch 20-day historical bars for up to 4 tickers to populate volume_ratio.
    # Capped at 4 because: 1 grouped-daily call + 4 per-ticker = 5 total (free tier limit).
    bars_by_ticker = {}
    for ticker in tickers[:4]:
        try:
            hist = massive_client.get_ticker_bars(ticker, days=20)
            if hist:
                bars_by_ticker[ticker.upper()] = hist
        except Exception as e:
            logger.debug("Massive historical bars failed for %s: %s", ticker, e)

    try:
        snapshots = massive_client.build_snapshots(tickers, bars_by_ticker=bars_by_ticker)
        for snap in snapshots:
            try:
                signals.append(converter(snap))
            except Exception:
                pass
    except Exception as e:
        logger.debug("Massive snapshot fetch failed: %s", e)
    return signals
