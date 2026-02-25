"""Market Sensing Hook — Async market data fetcher for Mae's step loop.

Biological analogy: The autonomic nervous system. Automatically gathers
sensory data on a cadence without conscious agent involvement. Agents
then process the gathered signals through their lifecycle.

Replaces the monolithic midge_scan.py pipeline with incremental sensing
that runs inside Mae's 33-layer bootstrap. Each model step, the hook:
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
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("mae.market.sensing")

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
}

# Source names for rotation — 12 sources, full cycle every 600 steps
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
]


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
        outcome_collector: Any = None,
        memory: Any = None,
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

        # Intelligence layer
        self._convergence_alerter = convergence_alerter
        self._velocity_detector = velocity_detector
        self._filing_analyzer = filing_analyzer
        self._form8k_sentiment = form8k_sentiment
        self._outcome_collector = outcome_collector
        self._memory = memory

        # Tiered alerters (tactical/strategic/thematic)
        self._tiered_alerters = tiered_alerters or {}

        # Watchlist
        self._watchlist = watchlist or self._load_watchlist()

        # Async fetch state
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mkt-sense")
        self._pending_future: Optional[Future] = None
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
        """Start async fetch from next source in rotation."""
        if self._pending_future is not None and not self._pending_future.done():
            return  # Previous fetch still running — skip this cadence

        source = self._fetch_queue[0]
        self._fetch_queue.rotate(-1)
        self._last_fetch_source = source
        self._total_fetches += 1

        logger.info("Market sensing: launching async fetch [%s] (cycle %d)", source, self._total_fetches)
        self._pending_future = self._executor.submit(self._fetch_source, source)

    def _collect_results(self):
        """Check if async fetch completed. Feed signals into convergence engine."""
        if self._pending_future is None or not self._pending_future.done():
            return

        try:
            signals = self._pending_future.result()
        except Exception:
            logger.warning("Market sensing: fetch failed", exc_info=True)
            signals = []
        finally:
            self._pending_future = None

        if not signals:
            return

        # Enrich signals with velocity and filing-time modifiers
        for sig in signals:
            self._enrich_signal(sig)

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
            len(signals), self._last_fetch_source, self._total_signals_fed,
        )

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
            "pending_fetch": self._pending_future is not None and not self._pending_future.done(),
        }

    def shutdown(self):
        """Graceful shutdown of thread pool."""
        self._executor.shutdown(wait=False)
