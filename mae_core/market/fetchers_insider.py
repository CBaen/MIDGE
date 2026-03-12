"""Insider trading fetch functions — SEC Form 4, OpenInsider, 13F/13D, FinViz.

All three probe the same underlying regulatory filings from different angles,
making their convergence a strong cross-validation signal.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger("midge.market.sensing")


def fetch_sec_form4(
    sec_client: Any,
    watchlist: dict,
    converter: Callable,
    cluster_detector: Any = None,
    cluster_converter: Callable = None,
) -> list:
    """Fetch SEC Form 4 insider trades for watchlist tickers.

    If cluster_detector and cluster_converter are provided, also detects
    insider buying clusters (3+ insiders) and appends ClusterSignal results.
    """
    if sec_client is None:
        return []

    from mae_core.market.apis.sec_edgar import get_recent_form4s

    raw_store = getattr(sec_client, "_raw_store", None)
    signals = []
    for ticker in watchlist.get("tickers", []):
        try:
            trades = get_recent_form4s(ticker, days=30, raw_store=raw_store)
            for trade in trades:
                try:
                    signals.append(converter(trade))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SEC Form 4 fetch failed for %s: %s", ticker, e)

        # Cluster detection: find 3+ insiders buying same stock
        if cluster_detector is not None and cluster_converter is not None:
            try:
                clusters = cluster_detector.find_clusters(ticker, days_back=30, min_insiders=3)
                for cluster in clusters:
                    try:
                        signals.append(cluster_converter(cluster))
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Cluster detection failed for %s: %s", ticker, e)

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
    filer_activity_converter: Callable = None,
) -> list:
    """Fetch 13F filer activity + 13D activist filings from SEC EDGAR.

    Two calls:
    1. get_13d_filings() — activist investors taking >5% stake (high-signal, ticker-specific)
    2. get_recent_13f_filers() — institutional quarterly filer metadata (market-wide activity)

    The 13F filer metadata has no ticker holdings (would require individual filing XML
    downloads), so it produces a market-wide institutional activity signal via
    filer_activity_converter. When many institutions file in a window, it signals
    broad repositioning.
    """
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
    # 13F institutional filer activity (previously built but never called)
    if filer_activity_converter is not None:
        try:
            filers = edgar_enhanced_client.get_recent_13f_filers(days=30)
            for filer in filers:
                try:
                    signals.append(filer_activity_converter(filer))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SEC 13F filer fetch failed: %s", e)
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


def fetch_finviz(
    finviz_client: Any,
    volume_converter: Callable,
    squeeze_converter: Callable,
    insider_converter: Callable,
) -> list:
    """Fetch FinViz unusual volume + short squeeze candidates + insider trades.

    FinViz insider trades are a third independent insider data source alongside
    SEC EDGAR Form 4 and OpenInsider.  All three probe the same underlying
    regulatory filings from different angles, which makes their convergence
    a strong cross-validation signal.
    """
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
    # Insider trades (Buy transactions only — filter on the fly)
    try:
        trades = finviz_client.get_insider_trades()
        for trade in trades[:30]:
            try:
                # Only Buy transactions carry directional signal
                if str(getattr(trade, "transaction_type", "")).lower() in ("buy", "purchase"):
                    signals.append(insider_converter(trade))
            except Exception:
                pass
    except Exception as e:
        logger.debug("FinViz insider trades fetch failed: %s", e)
    return signals
