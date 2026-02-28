"""Market Sensing Hook — Async market data fetcher for MIDGE's step loop.

Biological analogy: The autonomic nervous system. Automatically gathers
sensory data on a cadence without conscious agent involvement. Agents
then process the gathered signals through their lifecycle.

Replaces the monolithic midge_scan.py pipeline with incremental sensing
that runs inside MIDGE's 33-layer bootstrap. Each model step, the hook:
1. Collects results from any completed async fetch
2. Feeds signals into convergence_alerter + tiered alerters
3. Launches the next async fetch (on cadence, rotating through sources)

Threading: One pending future at a time via ThreadPoolExecutor(1).
Same pattern as ApiGateway (proven safe). No race conditions on
convergence_alerter because collection happens in the main thread.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("midge.market.sensing")

# Data paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "midge"
SIGNALS_DIR = DATA_DIR / "signals"

# Multi-timeframe convergence: route signals by source → tier
TIER_ROUTING = {
    "sec_form4": "tactical",
    "sec_form8k": "tactical",
    "sec_efts": "tactical",
    "finnhub_news": "tactical",
    "finnhub_earnings": "tactical",
    "congressional": "strategic",
    "senate": "strategic",
    "contract": "strategic",
    "insider_cluster": "strategic",
    "correlation": "strategic",
    "finra_short": "strategic",
    "sam_gov": "thematic",
    "hiring_tracker": "thematic",
    "contract_prediction": "thematic",
    "contract_award": "thematic",
    "social_sentiment": "thematic",
    "fred_macro": "thematic",
    "session_sweep": "tactical",
    "session_sweep_ifvg": "tactical",
    "ta_rsi": "tactical",
    "ta_macd": "tactical",
    "ta_bollinger": "tactical",
    "ta_structure": "tactical",
    "ta_candle": "tactical",
    # New sources (Layer 6)
    "cot_positioning": "strategic",
    "stocktwits_sentiment": "thematic",
    "vix_term_structure": "strategic",
    "google_trends": "thematic",
    "finnhub_economic": "tactical",
    "finnhub_analyst": "strategic",
    "finnhub_earnings_calendar": "tactical",
}

# Source names for rotation — 19 sources, full cycle every 950 steps
SOURCE_ROTATION = [
    "sec_form4",
    "sec_form8k",
    "congressional",
    "senate",
    "hiring",
    "usa_spending",
    "sam_gov_and_prices",
    "social_sentiment",
    "finra_short",
    "sec_efts",
    "finnhub",
    "fred_macro",
    "session_sweep",
    "ta_indicators",
    # New sources (Layer 6)
    "cot_positioning",
    "stocktwits",
    "vix_structure",
    "google_trends",
    "finnhub_extras",
]

# Map rotation source names → Thompson distribution keys for guided selection.
# Sources with multiple Thompson keys use their primary/dominant key.
_ROTATION_TO_THOMPSON = {
    "sec_form4": "sec_form4",
    "sec_form8k": "sec_form8k",
    "congressional": "congressional",
    "senate": "senate",
    "hiring": "hiring_tracker",
    "usa_spending": "contract_award",
    "sam_gov_and_prices": "sam_gov",
    "social_sentiment": "social_sentiment",
    "finra_short": "finra_short",
    "sec_efts": "sec_efts",
    "finnhub": "finnhub_news",
    "fred_macro": "fred_macro",
    "session_sweep": "session_sweep",
    "ta_indicators": "ta_rsi",
    "cot_positioning": "cot_positioning",
    "stocktwits": "stocktwits_sentiment",
    "vix_structure": "vix_term_structure",
    "google_trends": "google_trends",
    "finnhub_extras": "finnhub_economic",
}


class MarketSensingHook:
    """Async market data fetcher that runs as a step hook.

    Instantiated in bootstrap/market.py and registered via
    ctx.model.add_step_hook(hook.step).
    """

    def __init__(
        self,
        sec_client: Any = None,
        price_fetcher: Any = None,
        congress_client: Any = None,
        senate_client: Any = None,
        job_tracker: Any = None,
        usa_spending: Any = None,
        sam_gov: Any = None,
        apewisdom: Any = None,
        finra_client: Any = None,
        sec_efts: Any = None,
        finnhub: Any = None,
        fred: Any = None,
        convergence_alerter: Any = None,
        velocity_detector: Any = None,
        filing_analyzer: Any = None,
        form8k_sentiment: Any = None,
        session_sweep_detector: Any = None,
        ta_indicators: Any = None,
        cot_client: Any = None,
        stocktwits_client: Any = None,
        vix_client: Any = None,
        trends_client: Any = None,
        outcome_collector: Any = None,
        memory: Any = None,
        thompson_sampler: Any = None,
        tiered_alerters: Optional[dict] = None,
        watchlist: Optional[dict] = None,
        fetch_cadence: int = 50,
        outcome_cadence: int = 200,
    ):
        # API clients (all optional — graceful degradation)
        self._sec_client = sec_client
        self._price_fetcher = price_fetcher
        self._congress_client = congress_client
        self._senate_client = senate_client
        self._job_tracker = job_tracker
        self._usa_spending = usa_spending
        self._sam_gov = sam_gov
        self._apewisdom = apewisdom
        self._finra_client = finra_client
        self._sec_efts = sec_efts
        self._finnhub = finnhub
        self._fred = fred
        self._cot_client = cot_client
        self._stocktwits_client = stocktwits_client
        self._vix_client = vix_client
        self._trends_client = trends_client

        # EventBus (injected by bootstrap for signal bridge)
        self._bus = None

        # Intelligence layer
        self._convergence_alerter = convergence_alerter
        self._velocity_detector = velocity_detector
        self._filing_analyzer = filing_analyzer
        self._form8k_sentiment = form8k_sentiment
        self._session_sweep_detector = session_sweep_detector
        self._ta_indicators = ta_indicators
        self._outcome_collector = outcome_collector
        self._memory = memory
        self._thompson_sampler = thompson_sampler

        # Tiered alerters (tactical/strategic/thematic)
        self._tiered_alerters = tiered_alerters or {}

        # Watchlist
        self._watchlist = watchlist or self._load_watchlist()

        # Async fetch state — 3 concurrent workers for parallel senses
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="mkt-sense")
        self._pending_futures: Dict[str, Future] = {}  # source_name -> future
        self._fetch_queue = deque(SOURCE_ROTATION)
        self._step_counter = 0
        self._fetch_cadence = fetch_cadence
        self._outcome_cadence = outcome_cadence

        # Stats
        self._total_signals_fed = 0
        self._total_fetches = 0
        self._last_fetch_source = None

        # Ensure data dirs exist
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    def step(self):
        """Called every model step. Non-blocking."""
        self._step_counter += 1

        # Collect results from previous async fetch (if ready)
        self._collect_results()

        # Launch next async fetch on cadence
        if self._step_counter % self._fetch_cadence == 0:
            self._launch_next_fetch()

        # Outcome tracking on cadence
        if self._step_counter % self._outcome_cadence == 0:
            self._evaluate_outcomes()

    # ------------------------------------------------------------------
    # Async fetch lifecycle
    # ------------------------------------------------------------------

    def _launch_next_fetch(self):
        """Fill up to 3 concurrent fetch slots using Thompson-guided selection.

        When a Thompson sampler is available, sources are selected
        probabilistically: each source draws from its Beta distribution,
        then the highest-scoring source is picked. This naturally
        balances exploration (new sources with wide posteriors) and
        exploitation (proven sources with high means). Falls back to
        round-robin when no sampler is available.
        """
        # Collect any completed futures first
        self._collect_results()

        # Fill available slots (3 max concurrent)
        slots = 3 - len(self._pending_futures)
        if slots <= 0:
            return

        # Identify eligible sources (not already in-flight)
        eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures]
        if not eligible:
            return

        if self._thompson_sampler is not None:
            self._launch_thompson_guided(eligible, slots)
        else:
            self._launch_round_robin(slots)

    def _launch_thompson_guided(self, eligible: list, slots: int):
        """Select sources via Thompson sampling draws."""
        # Draw from each source's Beta distribution
        scored = []
        for source in eligible:
            thompson_key = _ROTATION_TO_THOMPSON.get(source)
            if thompson_key is None:
                # Unknown source — use uniform prior (maximally uncertain)
                score = random.betavariate(1.0, 1.0)
            else:
                try:
                    dist = self._thompson_sampler.get_distribution(thompson_key)
                    if dist is not None and hasattr(dist, "alpha") and hasattr(dist, "beta"):
                        score = random.betavariate(
                            max(dist.alpha, 0.01), max(dist.beta, 0.01)
                        )
                    else:
                        # No distribution yet — use wide prior (encourages exploration)
                        score = random.betavariate(1.0, 1.0)
                except Exception:
                    score = random.betavariate(1.0, 1.0)
            scored.append((score, source))

        # Sort descending by score, pick top N
        scored.sort(reverse=True)
        for _, source in scored[:slots]:
            self._total_fetches += 1
            logger.info(
                "Market sensing: Thompson-guided fetch [%s] (cycle %d)",
                source, self._total_fetches,
            )
            self._pending_futures[source] = self._executor.submit(
                self._fetch_source, source
            )

    def _launch_round_robin(self, slots: int):
        """Fallback: simple round-robin selection."""
        attempts = 0
        launched = 0
        while launched < slots and attempts < len(SOURCE_ROTATION):
            source = self._fetch_queue[0]
            self._fetch_queue.rotate(-1)
            attempts += 1

            if source in self._pending_futures:
                continue

            self._total_fetches += 1
            logger.info(
                "Market sensing: launching async fetch [%s] (cycle %d)",
                source, self._total_fetches,
            )
            self._pending_futures[source] = self._executor.submit(
                self._fetch_source, source
            )
            launched += 1

    def _collect_results(self):
        """Check for completed futures and process their signals."""
        done = [k for k, f in self._pending_futures.items() if f.done()]
        for source_name in done:
            self._collect_one(source_name)

    def _collect_one(self, source_name: str):
        """Process signals from a single completed fetch."""
        future = self._pending_futures.pop(source_name, None)
        if future is None:
            return

        try:
            signals = future.result()
        except Exception:
            logger.warning("Market sensing: fetch [%s] failed", source_name, exc_info=True)
            signals = []

        if not signals:
            return

        self._last_fetch_source = source_name

        # Signals arrive pre-enriched from background thread (_fetch_source)

        # Feed into convergence engine (global + tiered)
        for sig in signals:
            sig_kwargs = dict(
                signal_id=sig.signal_id,
                strength=sig.strength,
                domain=sig.domain,
                direction=sig.direction,
                confidence=sig.confidence,
                velocity=sig.velocity,
                timestamp=sig.timestamp,
                metadata={**sig.metadata, "symbol": sig.symbol},
                source=sig.source,
            )
            # Global alerter
            if self._convergence_alerter is not None:
                try:
                    self._convergence_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Failed to feed signal to global alerter", exc_info=True)

            # Route to tier
            tier = TIER_ROUTING.get(sig.source, "strategic")
            tier_alerter = self._tiered_alerters.get(tier)
            if tier_alerter is not None:
                try:
                    tier_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Failed to feed signal to %s alerter", tier, exc_info=True)

        self._total_signals_fed += len(signals)
        logger.info(
            "Market sensing: fed %d signals from [%s] (total: %d)",
            len(signals), source_name, self._total_signals_fed,
        )

        # Publish each signal to EventBus (hypothesis engine subscribes here)
        if self._bus is not None:
            from mae_core.market.channels import CH_SIGNAL_INGESTED
            for sig in signals:
                try:
                    self._bus.publish(CH_SIGNAL_INGESTED, {
                        "signal_id": sig.signal_id,
                        "source": sig.source,
                        "symbol": sig.symbol,
                        "domain": sig.domain,
                        "direction": sig.direction,
                        "strength": sig.strength,
                        "confidence": sig.confidence,
                        "velocity": sig.velocity,
                        "timestamp": sig.timestamp.isoformat(),
                    })
                except Exception:
                    logger.debug("Failed to publish signal to EventBus", exc_info=True)

        # Store to Qdrant + JSONL
        self._store_signals(signals)

        # Register with outcome collector
        if self._outcome_collector is not None:
            try:
                registered = self._outcome_collector.register_signals(signals)
                if registered:
                    logger.info("Market sensing: registered %d predictions for outcome tracking", registered)
            except Exception:
                logger.debug("Outcome registration failed", exc_info=True)

    # ------------------------------------------------------------------
    # Fetch sources (runs in background thread)
    # ------------------------------------------------------------------

    def _fetch_source(self, source_name: str) -> list:
        """Fetch from a single source. Runs in ThreadPoolExecutor.

        Returns list of MarketSignal objects.
        """
        from mae_core.market.signal import (
            MarketSignal,
            from_insider_trade,
            from_form8k_event,
            from_congressional_trade,
            from_senate_trade,
            from_hiring_signal,
            from_government_contract,
            from_contract_opportunity,
            from_social_sentiment,
            from_short_interest,
            from_filing_keyword,
            from_news_sentiment,
            from_earnings_event,
            from_macro_indicator,
            from_session_sweep,
            from_ta_signal,
            from_cot_positioning,
            from_stocktwits_sentiment,
            from_vix_structure,
            from_trends_signal,
            from_economic_event,
            from_analyst_recommendation,
        )

        signals = []

        if source_name == "sec_form4":
            signals = self._fetch_sec_form4(from_insider_trade)

        elif source_name == "sec_form8k":
            signals = self._fetch_sec_form8k(from_form8k_event)

        elif source_name == "congressional":
            signals = self._fetch_congressional(from_congressional_trade)

        elif source_name == "senate":
            signals = self._fetch_senate(from_senate_trade)

        elif source_name == "hiring":
            signals = self._fetch_hiring(from_hiring_signal)

        elif source_name == "usa_spending":
            signals = self._fetch_usa_spending(from_government_contract)

        elif source_name == "sam_gov_and_prices":
            signals = self._fetch_sam_gov(from_contract_opportunity)

        elif source_name == "social_sentiment":
            signals = self._fetch_social_sentiment(from_social_sentiment)

        elif source_name == "finra_short":
            signals = self._fetch_finra_short(from_short_interest)

        elif source_name == "sec_efts":
            signals = self._fetch_sec_efts(from_filing_keyword)

        elif source_name == "finnhub":
            signals = self._fetch_finnhub(from_news_sentiment, from_earnings_event)

        elif source_name == "fred_macro":
            signals = self._fetch_fred(from_macro_indicator)

        elif source_name == "session_sweep":
            signals = self._fetch_session_sweep(from_session_sweep)

        elif source_name == "ta_indicators":
            signals = self._fetch_ta_indicators(from_ta_signal)

        elif source_name == "cot_positioning":
            signals = self._fetch_cot(from_cot_positioning)

        elif source_name == "stocktwits":
            signals = self._fetch_stocktwits(from_stocktwits_sentiment)

        elif source_name == "vix_structure":
            signals = self._fetch_vix(from_vix_structure)

        elif source_name == "google_trends":
            signals = self._fetch_trends(from_trends_signal)

        elif source_name == "finnhub_extras":
            signals = self._fetch_finnhub_extras(
                from_economic_event, from_analyst_recommendation
            )

        # Enrich in background thread (velocity, filing-time, Ollama sentiment)
        # Moved from _collect_results() so Ollama's 15s timeout doesn't block
        # the main step loop. Thread-safe: only mutates signal objects.
        for sig in signals:
            self._enrich_signal(sig)

        return signals

    def _fetch_sec_form4(self, converter) -> list:
        """Fetch SEC Form 4 insider trades for watchlist tickers."""
        if self._sec_client is None:
            return []

        from mae_core.market.apis.sec_edgar import get_recent_form4s

        signals = []
        for ticker in self._watchlist.get("tickers", []):
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

    def _fetch_sec_form8k(self, converter) -> list:
        """Fetch SEC Form 8-K events for watchlist tickers."""
        if self._sec_client is None:
            return []

        from mae_core.market.apis.sec_edgar import get_recent_form8ks

        signals = []
        for ticker in self._watchlist.get("tickers", []):
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

    def _fetch_congressional(self, converter) -> list:
        """Fetch congressional stock trades."""
        if self._congress_client is None:
            return []

        signals = []
        try:
            trades = self._congress_client.get_recent_trades(days=30)
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

    def _fetch_hiring(self, converter) -> list:
        """Fetch hiring signals for watchlist companies."""
        if self._job_tracker is None:
            return []

        signals = []
        companies = self._watchlist.get("companies", {})
        for company, ticker in companies.items():
            try:
                signal = self._job_tracker.analyze_hiring_activity(company, ticker=ticker)
                signals.append(converter(signal))
            except Exception as e:
                logger.debug("Hiring fetch failed for %s: %s", company, e)
        return signals

    def _fetch_usa_spending(self, converter) -> list:
        """Fetch government contracts from USASpending."""
        if self._usa_spending is None:
            return []

        signals = []
        for keyword in self._watchlist.get("keywords", []):
            try:
                contracts = self._usa_spending.search_contracts(keyword=keyword, limit=5)
                for contract in contracts:
                    try:
                        signals.append(converter(contract))
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("USASpending fetch failed for '%s': %s", keyword, e)
        return signals

    def _fetch_senate(self, converter) -> list:
        """Fetch Senate stock trades."""
        if self._senate_client is None:
            return []

        signals = []
        try:
            trades = self._senate_client.get_recent_trades(days=30)
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

    def _fetch_sam_gov(self, converter) -> list:
        """Fetch SAM.gov opportunities."""
        if self._sam_gov is None:
            return []

        signals = []
        for keyword in self._watchlist.get("keywords", []):
            try:
                opps = self._sam_gov.search_opportunities(keywords=keyword, limit=5)
                for opp in opps:
                    try:
                        signals.append(converter(opp))
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("SAM.gov fetch failed for '%s': %s", keyword, e)
        return signals

    def _fetch_social_sentiment(self, converter) -> list:
        """Fetch Reddit/WSB social sentiment from ApeWisdom."""
        if self._apewisdom is None:
            return []

        signals = []
        try:
            # Get accelerating tickers (2x+ mention velocity) — these are the signal
            accelerating = self._apewisdom.get_accelerating_tickers(min_change=2.0, limit=10)
            for sentiment in accelerating:
                try:
                    signals.append(converter(sentiment))
                except Exception:
                    pass

            # Also check watchlist tickers directly
            for ticker in self._watchlist.get("tickers", []):
                try:
                    sentiment = self._apewisdom.get_by_ticker(ticker)
                    if sentiment is not None and sentiment.mention_change >= 1.5:
                        signals.append(converter(sentiment))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("ApeWisdom fetch failed: %s", e)
        return signals

    def _fetch_finra_short(self, converter) -> list:
        """Fetch FINRA daily short volume — high short ratio tickers."""
        if self._finra_client is None:
            return []

        signals = []
        try:
            # Get tickers with >50% short volume ratio
            high_short = self._finra_client.get_high_short_ratio(min_ratio=0.5)
            # Filter to watchlist + top 10 highest ratios
            watchlist_tickers = set(self._watchlist.get("tickers", []))
            for record in high_short:
                try:
                    if record.symbol in watchlist_tickers or high_short.index(record) < 10:
                        signals.append(converter(record))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("FINRA short volume fetch failed: %s", e)
        return signals

    def _fetch_sec_efts(self, converter) -> list:
        """Fetch SEC EFTS full-text search keyword hits."""
        if self._sec_efts is None:
            return []

        signals = []
        try:
            hits = self._sec_efts.scan_all_keywords(days=3)
            for hit in hits:
                try:
                    signals.append(converter(hit))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("SEC EFTS fetch failed: %s", e)
        return signals

    def _fetch_finnhub(self, news_converter, earnings_converter) -> list:
        """Fetch Finnhub news sentiment + earnings surprises."""
        if self._finnhub is None:
            return []

        signals = []

        # News sentiment for watchlist tickers
        for ticker in self._watchlist.get("tickers", []):
            try:
                sentiment = self._finnhub.get_news_sentiment(ticker)
                if sentiment is not None:
                    signals.append(news_converter(sentiment))
            except Exception as e:
                logger.debug("Finnhub news sentiment failed for %s: %s", ticker, e)

        # Recent earnings surprises
        try:
            reported = self._finnhub.get_recent_earnings_surprises(days=7)
            for event in reported:
                try:
                    signals.append(earnings_converter(event))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Finnhub earnings fetch failed: %s", e)

        return signals

    def _fetch_fred(self, converter) -> list:
        """Fetch FRED macroeconomic indicators."""
        if self._fred is None:
            return []

        signals = []
        try:
            snapshot = self._fred.get_macro_snapshot()
            for indicator in snapshot:
                try:
                    signals.append(converter(indicator))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("FRED macro fetch failed: %s", e)
        return signals

    def _fetch_session_sweep(self, converter) -> list:
        """Fetch ICT session sweep signals for futures.

        Kill-zone time guard: returns early if not within 90 min of a
        kill zone window. Prevents wasting yfinance rate limit during
        dead hours.
        """
        if self._session_sweep_detector is None:
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
                sweeps = self._session_sweep_detector.detect_sweeps(symbol)
                for sweep in sweeps:
                    try:
                        signals.append(converter(sweep))
                    except Exception:
                        pass
            except Exception as e:
                logger.debug("Session sweep fetch failed for %s: %s", symbol, e)
        return signals

    def _fetch_ta_indicators(self, converter) -> list:
        """Compute technical analysis indicators for watchlist tickers.

        Uses price_fetcher.get_daily_history() for OHLCV data, then runs
        RSI, MACD, Bollinger, Market Structure, and Candlestick detection.
        Pure local computation — no external API calls beyond yfinance history.
        """
        if self._ta_indicators is None or self._price_fetcher is None:
            return []

        from mae_core.market.edge.ta_indicators import compute_all

        signals = []
        for ticker in self._watchlist.get("tickers", []):
            try:
                history = self._price_fetcher.get_daily_history(ticker, days=90)
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

    # ------------------------------------------------------------------
    # New source fetchers (Layer 6)
    # ------------------------------------------------------------------

    def _fetch_cot(self, converter) -> list:
        """Fetch CFTC Commitments of Traders for futures watchlist."""
        if self._cot_client is None:
            return []

        signals = []
        # COT is futures-only — use futures tickers from watchlist or defaults
        futures_tickers = [t for t in self._watchlist.get("tickers", [])
                          if t.endswith("=F")]
        if not futures_tickers:
            futures_tickers = ["ES=F", "NQ=F", "GC=F", "CL=F"]

        try:
            positions = self._cot_client.get_latest_positions(futures_tickers)
            for pos in positions:
                try:
                    signals.append(converter(pos))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("COT fetch failed: %s", e)
        return signals

    def _fetch_stocktwits(self, converter) -> list:
        """Fetch StockTwits bull/bear sentiment for watchlist tickers."""
        if self._stocktwits_client is None:
            return []

        signals = []
        tickers = self._watchlist.get("tickers", [])[:10]  # Cap at 10 for rate limits
        try:
            sentiments = self._stocktwits_client.get_sentiment(tickers)
            for st in sentiments:
                try:
                    signals.append(converter(st))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("StockTwits fetch failed: %s", e)
        return signals

    def _fetch_vix(self, converter) -> list:
        """Fetch CBOE VIX term structure."""
        if self._vix_client is None:
            return []

        signals = []
        try:
            vix = self._vix_client.get_vix_structure()
            if vix is not None:
                signals.append(converter(vix))
        except Exception as e:
            logger.debug("VIX fetch failed: %s", e)
        return signals

    def _fetch_trends(self, converter) -> list:
        """Fetch Google Trends interest for watchlist tickers + macro terms."""
        if self._trends_client is None:
            return []

        signals = []
        # Mix watchlist tickers with macro fear terms
        tickers = self._watchlist.get("tickers", [])[:5]
        macro_terms = ["recession", "market crash", "fed rate"]
        keywords = tickers + macro_terms

        try:
            trends = self._trends_client.get_interest(keywords)
            for trend in trends:
                try:
                    signals.append(converter(trend))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Google Trends fetch failed: %s", e)
        return signals

    def _fetch_finnhub_extras(self, econ_converter, analyst_converter) -> list:
        """Fetch Finnhub economic calendar + analyst recommendations."""
        if self._finnhub is None:
            return []

        signals = []

        # Economic calendar
        try:
            events = self._finnhub.get_economic_calendar(days=7)
            for event in events:
                try:
                    signals.append(econ_converter(event))
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Finnhub economic calendar failed: %s", e)

        # Analyst recommendations for watchlist tickers
        tickers = self._watchlist.get("tickers", [])[:5]
        for ticker in tickers:
            try:
                recs = self._finnhub.get_analyst_recommendations(ticker)
                if recs:
                    # Only use the most recent recommendation period
                    signals.append(analyst_converter(recs[0]))
            except Exception as e:
                logger.debug("Finnhub analyst recs failed for %s: %s", ticker, e)

        return signals

    # ------------------------------------------------------------------
    # Signal enrichment
    # ------------------------------------------------------------------

    def _enrich_signal(self, sig):
        """Apply velocity and filing-time modifiers to a signal."""
        # Populate velocity via VelocityDetector
        if self._velocity_detector is not None:
            try:
                state = self._velocity_detector.record(sig.signal_id, sig.strength, sig.timestamp)
                sig.velocity = state.current_velocity
            except Exception:
                pass

        # Apply filing-time confidence modifier for SEC filings
        if self._filing_analyzer is not None and sig.source in ("sec_form4", "sec_form8k"):
            try:
                filing_dt = sig.received_at or sig.timestamp
                fta_signal = self._filing_analyzer.analyze_filing_time(
                    ticker=sig.symbol,
                    filer_name=sig.metadata.get("filer_name", ""),
                    filing_date=sig.timestamp.strftime("%Y-%m-%d"),
                    filing_datetime=filing_dt,
                    form_type="4" if sig.source == "sec_form4" else "8-K",
                )
                sig.confidence = max(0.0, min(1.0, sig.confidence + fta_signal.confidence_modifier))
            except Exception:
                pass

        # Apply 8-K text sentiment via Ollama (enriches beyond rule-based item codes)
        if self._form8k_sentiment is not None and sig.source == "sec_form8k":
            try:
                event_text = sig.metadata.get("event_summary", "")
                item_code = sig.metadata.get("item_code", "")
                result = self._form8k_sentiment.classify(event_text, item_code)
                if result is not None:
                    # Override direction if sentiment disagrees with rule-based
                    if result.direction != "neutral":
                        sig.direction = result.direction
                    sig.confidence = max(0.0, min(1.0, sig.confidence + result.confidence_modifier))
                    sig.metadata["ollama_sentiment"] = result.direction
                    sig.metadata["ollama_raw"] = result.raw_response[:200]
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store_signals(self, signals: list):
        """Store signals to Qdrant + JSONL archive."""
        # Qdrant (if available)
        if self._memory is not None:
            try:
                if self._memory.is_available():
                    self._memory.store_signals(signals)
            except Exception:
                logger.debug("Qdrant storage failed", exc_info=True)

        # JSONL cold storage (always)
        today = datetime.now().strftime("%Y-%m-%d")
        jsonl_path = SIGNALS_DIR / f"{today}.jsonl"
        try:
            with open(jsonl_path, "a") as f:
                for sig in signals:
                    record = {
                        "signal_id": sig.signal_id,
                        "source": sig.source,
                        "symbol": sig.symbol,
                        "domain": sig.domain,
                        "direction": sig.direction,
                        "strength": sig.strength,
                        "confidence": sig.confidence,
                        "velocity": sig.velocity,
                        "timestamp": sig.timestamp.isoformat(),
                        "received_at": sig.received_at.isoformat(),
                        "metadata": sig.metadata,
                    }
                    f.write(json.dumps(record) + "\n")
        except Exception:
            logger.debug("JSONL archive write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Outcome tracking
    # ------------------------------------------------------------------

    def _evaluate_outcomes(self):
        """Evaluate matured predictions through Thompson feedback loop."""
        if self._outcome_collector is None:
            return
        try:
            evaluated = self._outcome_collector.evaluate()
            if evaluated:
                logger.info("Market sensing: evaluated %d matured outcomes", evaluated)
        except Exception:
            logger.debug("Outcome evaluation failed", exc_info=True)

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def _load_watchlist(self) -> dict:
        """Load watchlist from data/midge/watchlist.json."""
        watchlist_path = DATA_DIR / "watchlist.json"
        try:
            with open(watchlist_path) as f:
                return json.load(f)
        except Exception:
            logger.warning("Could not load watchlist from %s, using defaults", watchlist_path)
            return {
                "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                             "LMT", "RTX", "NOC", "GD", "BA"],
                "keywords": ["cybersecurity", "artificial intelligence", "defense", "space"],
                "companies": {
                    "Lockheed Martin": "LMT",
                    "Raytheon Technologies": "RTX",
                    "Northrop Grumman": "NOC",
                    "General Dynamics": "GD",
                    "Boeing": "BA",
                },
            }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """Return sensing hook stats for monitoring."""
        return {
            "step_counter": self._step_counter,
            "total_signals_fed": self._total_signals_fed,
            "total_fetches": self._total_fetches,
            "last_fetch_source": self._last_fetch_source,
            "fetch_cadence": self._fetch_cadence,
            "pending_fetches": len([f for f in self._pending_futures.values() if not f.done()]),
            "in_flight_sources": list(self._pending_futures.keys()),
        }

    def shutdown(self):
        """Graceful shutdown of thread pool."""
        self._executor.shutdown(wait=False)
