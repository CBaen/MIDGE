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

Decomposed into three files:
  sensing_hook.py      — this file: MarketSensingHook class + constants
  sensing_fetchers.py  — 30 standalone fetch functions
  sensing_lifecycle.py — enrich_signal, store_signals, load_watchlist
"""

from __future__ import annotations

import logging
import os
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from mae_core.market.sensing_fetchers import (
    fetch_sec_form4,
    fetch_sec_form8k,
    fetch_congressional,
    fetch_senate,
    fetch_hiring,
    fetch_usa_spending,
    fetch_sam_gov,
    fetch_social_sentiment,
    fetch_finra_short,
    fetch_sec_efts,
    fetch_finnhub,
    fetch_fred,
    fetch_session_sweep,
    fetch_ta_indicators,
    fetch_cot,
    fetch_stocktwits,
    fetch_vix,
    fetch_trends,
    fetch_finnhub_extras,
    fetch_order_flow,
    fetch_fractal_resonance,
    fetch_crypto_prices,
    fetch_crypto_exchange,
    fetch_openinsider,
    fetch_13f_holdings,
    fetch_finviz,
    fetch_economic_calendar,
    fetch_eia,
    fetch_congress_legislation,
    fetch_massive_snapshot,
)
from mae_core.market.sensing_lifecycle import (
    enrich_signal,
    store_signals,
    load_watchlist,
)

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
    # Ten Gifts: Wave 1
    "order_flow": "tactical",
    # Ten Gifts: Wave 2
    "fractal_resonance": "strategic",
    "archetype_match": "strategic",
    # New sources (Layer 6)
    "cot_positioning": "strategic",
    "stocktwits_sentiment": "thematic",
    "vix_term_structure": "strategic",
    "google_trends": "thematic",
    "finnhub_economic": "tactical",
    "finnhub_analyst": "strategic",
    "finnhub_earnings_calendar": "tactical",
    # Wave 2: Real-Time + Crypto
    "finnhub_realtime": "tactical",
    "crypto_coingecko": "thematic",
    "crypto_coincap": "thematic",
    # Wave 3: Data Enrichment
    "openinsider_purchase": "tactical",
    "institutional_13f": "strategic",
    "activist_13d": "strategic",
    "finviz_unusual_volume": "tactical",
    "finviz_short_squeeze": "strategic",
    "economic_calendar": "thematic",
    # Massive/Polygon.io
    "massive_snapshot": "tactical",
    # Real-economy: Energy
    "eia_energy": "strategic",
    # Real-economy: Legislative
    "congress_legislation": "strategic",
}

# Source names for rotation — 30 sources, 8 concurrent per cadence tick
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
    # Ten Gifts: Wave 1
    "order_flow",
    # Ten Gifts: Wave 2
    "fractal_resonance",
    # New sources (Layer 6)
    "cot_positioning",
    "stocktwits",
    "vix_structure",
    "google_trends",
    "finnhub_extras",
    # Wave 2: Real-Time + Crypto (Always-On)
    "crypto_prices",
    "crypto_exchange",
    # Wave 3: Data Enrichment
    "openinsider",
    "institutional_13f",
    "finviz",
    "economic_calendar",
    # Massive/Polygon.io
    "massive_snapshot",
    # Real-economy: Energy
    "eia_energy",
    # Real-economy: Legislative
    "congress_legislation",
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
    "order_flow": "order_flow",
    "fractal_resonance": "fractal_resonance",
    "cot_positioning": "cot_positioning",
    "stocktwits": "stocktwits_sentiment",
    "vix_structure": "vix_term_structure",
    "google_trends": "google_trends",
    "finnhub_extras": "finnhub_economic",
    "crypto_prices": "crypto_coingecko",
    "crypto_exchange": "crypto_coincap",
    "openinsider": "openinsider_purchase",
    "institutional_13f": "institutional_13f",
    "finviz": "finviz_unusual_volume",
    "economic_calendar": "economic_calendar",
    "massive_snapshot": "massive_snapshot",
    "eia_energy": "eia_energy",
    "congress_legislation": "congress_legislation",
}

# Map absence source names back to convergence domains
_ABSENCE_SOURCE_DOMAINS = {
    "sec_form4": "insider", "sec_form8k": "insider", "sec_efts": "insider",
    "congressional": "government", "senate": "government",
    "finra_short": "institutional", "cot_positioning": "positioning",
    "fred_macro": "macro", "finnhub_earnings": "news",
    "finnhub_news": "news", "hiring": "contracts",
    "usa_spending": "contracts", "sam_gov": "contracts",
    "crypto_coingecko": "crypto", "crypto_coincap": "crypto",
    "openinsider_purchase": "insider", "institutional_13f": "institutional",
    "activist_13d": "institutional",
    "finviz_unusual_volume": "technical", "finviz_short_squeeze": "institutional",
    "massive_snapshot": "technical",
    "eia_energy": "energy",
    "congress_legislation": "government",
}


def _absence_source_to_domain(source: str) -> str:
    """Map an absence source name to a convergence domain."""
    return _ABSENCE_SOURCE_DOMAINS.get(source, "unknown")


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
        fetch_cadence: int = 25,
        outcome_cadence: int = 200,
        order_flow_detector: Any = None,
        portfolio_tracker: Any = None,
        catalyst_calendar: Any = None,
        deception_detector: Any = None,
        consolidation_engine: Any = None,
        fractal_resonance_detector: Any = None,
        pattern_archetype_engine: Any = None,
        somatic_anticipation: Any = None,
        pattern_completion_engine: Any = None,
        market_clock: Any = None,
        coingecko_client: Any = None,
        coincap_client: Any = None,
        openinsider_client: Any = None,
        edgar_enhanced_client: Any = None,
        finviz_client: Any = None,
        economic_calendar_client: Any = None,
        finnhub_websocket: Any = None,
        massive_client: Any = None,
        eia_client: Any = None,
        congress_gov_client: Any = None,
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
        self._order_flow_detector = order_flow_detector
        self._portfolio_tracker = portfolio_tracker
        self._catalyst_calendar = catalyst_calendar
        self._deception_detector = deception_detector
        self._consolidation_engine = consolidation_engine
        self._fractal_resonance_detector = fractal_resonance_detector
        self._pattern_archetype_engine = pattern_archetype_engine
        self._somatic_anticipation = somatic_anticipation
        self._pattern_completion_engine = pattern_completion_engine
        self._market_clock = market_clock
        self._coingecko_client = coingecko_client
        self._coincap_client = coincap_client
        self._openinsider_client = openinsider_client
        self._edgar_enhanced_client = edgar_enhanced_client
        self._finviz_client = finviz_client
        self._economic_calendar_client = economic_calendar_client
        self._finnhub_websocket = finnhub_websocket
        self._massive_client = massive_client
        self._eia_client = eia_client
        self._congress_gov_client = congress_gov_client

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
        self._correlation_tracker = None  # Injected by bootstrap
        self._absence_monitor = None  # Injected by bootstrap

        # Tiered alerters (tactical/strategic/thematic)
        self._tiered_alerters = tiered_alerters or {}

        # Watchlist
        self._watchlist = watchlist or load_watchlist()

        # Async fetch state — 8 concurrent workers for parallel senses
        self._executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="mkt-sense")
        self._pending_futures: Dict[str, Future] = {}  # source_name -> future
        self._fetch_queue = deque(SOURCE_ROTATION)
        self._step_counter = 0
        self._fetch_cadence = fetch_cadence
        self._outcome_cadence = outcome_cadence

        # Recent signal domains per ticker (for archetype scanning)
        self._recent_domains: Dict[str, set] = {}

        # Stats
        self._total_signals_fed = 0
        self._total_fetches = 0
        self._last_fetch_source = None

        # Ensure data dirs exist
        SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

    def step(self):
        """Called every model step. Non-blocking."""
        self._step_counter += 1

        # Real-time WebSocket signals bypass rotation — collected EVERY step
        if self._finnhub_websocket is not None:
            try:
                ws_signals = self._finnhub_websocket.get_pending_signals()
                if ws_signals:
                    from mae_core.market.signal import from_finnhub_realtime
                    converted = []
                    for ws_sig in ws_signals:
                        try:
                            converted.append(from_finnhub_realtime(ws_sig))
                        except Exception:
                            pass
                    if converted:
                        # Feed directly into _collect_one style processing
                        self._process_realtime_signals(converted)
            except Exception:
                logger.debug("WebSocket signal collection failed", exc_info=True)

        # Log market session status periodically
        if self._step_counter % 1000 == 0 and self._market_clock is not None:
            try:
                session = self._market_clock.get_session()
                available = self._market_clock.get_available_sources()
                logger.info(
                    "Market clock: session=%s, %d/%d sources available",
                    session, len(available), len(SOURCE_ROTATION),
                )
            except Exception:
                logger.debug("Market clock session check failed", exc_info=True)

        # Collect results from previous async fetch (if ready)
        self._collect_results()

        # Launch next async fetch on cadence
        if self._step_counter % self._fetch_cadence == 0:
            self._launch_next_fetch()

        # Outcome tracking on cadence
        if self._step_counter % self._outcome_cadence == 0:
            self._evaluate_outcomes()

        # Portfolio tracker: mark-to-market + exit signal check (cadence 50)
        if self._step_counter % 50 == 0 and self._portfolio_tracker is not None:
            try:
                self._portfolio_tracker.update_prices()
                exits = self._portfolio_tracker.check_exits()
                if exits and self._convergence_alerter is not None:
                    for exit_sig in exits:
                        try:
                            self._convergence_alerter.record_signal(
                                signal_id=f"exit:{exit_sig.ticker}:{exit_sig.reason}",
                                strength=exit_sig.strength,
                                domain="portfolio",
                                direction=exit_sig.direction,
                                source="portfolio_exit",
                            )
                        except Exception:
                            logger.debug("Portfolio exit signal feed failed", exc_info=True)
            except Exception:
                logger.debug("Portfolio tracker step failed", exc_info=True)

        # Catalyst calendar: refresh upcoming events (cadence 200)
        if self._step_counter % 200 == 0 and self._catalyst_calendar is not None:
            try:
                self._catalyst_calendar.refresh()
            except Exception:
                logger.debug("Catalyst calendar refresh failed", exc_info=True)

        # Absence detection on cadence 100 — check for unexpectedly silent sources
        if self._step_counter % 100 == 0 and self._absence_monitor is not None:
            try:
                absences = self._absence_monitor.check_absences()
                if absences:
                    logger.info(
                        "AbsenceMonitor: %d sources unexpectedly silent",
                        len(absences),
                    )
                    # Feed absence signals to convergence alerter
                    if self._convergence_alerter is not None:
                        for absence in absences:
                            try:
                                from mae_core.market.intelligence.convergence_alerter import Signal as CASignal
                                absence_signal = CASignal(
                                    signal_id=f"absence:{absence.source}",
                                    strength=min(1.0, 0.3 + 0.1 * (absence.silence_ratio - self._absence_monitor._absence_threshold)),
                                    domain=_absence_source_to_domain(absence.source),
                                    direction=absence.direction,
                                    timestamp=datetime.now(),
                                    confidence=0.5,
                                )
                                self._convergence_alerter.record_signal(
                                    absence_signal.signal_id,
                                    absence_signal.strength,
                                    absence_signal.domain,
                                    direction=absence_signal.direction,
                                    source=f"absence_{absence.source}",
                                )
                            except Exception:
                                logger.debug("Absence signal feed failed", exc_info=True)
            except Exception:
                logger.debug("AbsenceMonitor check failed", exc_info=True)

        # Cross-source correlation anomaly scan on cadence 200
        if self._step_counter % 200 == 0 and self._correlation_tracker is not None:
            try:
                anomalies = self._correlation_tracker.detect_cross_domain_anomalies()
                if anomalies:
                    logger.info(
                        "CorrelationTracker: %d cross-domain anomalies detected",
                        len(anomalies),
                    )
            except Exception:
                logger.debug("CorrelationTracker anomaly scan failed", exc_info=True)

        # Consolidation engine: memory pruning (cadence 5000)
        if self._step_counter % 5000 == 0 and self._consolidation_engine is not None:
            try:
                if self._consolidation_engine.should_consolidate(self._step_counter):
                    self._consolidation_engine.consolidate()
            except Exception:
                logger.debug("Consolidation engine step failed", exc_info=True)

        # Somatic anticipation: pre-conscious pattern response (cadence 25)
        if self._step_counter % 25 == 0 and self._somatic_anticipation is not None:
            try:
                events = self._somatic_anticipation.check_anticipation()
                if events:
                    logger.info(
                        "Somatic anticipation: %d tickers building pre-convergence",
                        len(events),
                    )
            except Exception:
                logger.debug("Somatic anticipation check failed", exc_info=True)

        # Pattern archetype scanning: check watchlist tickers (cadence 100)
        if self._step_counter % 100 == 0 and self._pattern_archetype_engine is not None:
            try:
                for ticker in list(self._watchlist.get("tickers", []))[:5]:
                    self._pattern_archetype_engine.scan_for_archetypes(
                        ticker, signal_domains=list(self._recent_domains.get(ticker, [])),
                    )
            except Exception:
                logger.debug("Pattern archetype scan failed", exc_info=True)

        # Pattern completion: review partial archetype matches for active hunts (cadence 100)
        if self._step_counter % 100 == 0 and self._pattern_completion_engine is not None:
            try:
                new_hunts = self._pattern_completion_engine.review_partial_matches()
                if new_hunts:
                    logger.info(
                        "Pattern completion: %d new hunts created",
                        len(new_hunts),
                    )
                # Prune expired hunts
                self._pattern_completion_engine._prune_expired()
            except Exception:
                logger.debug("Pattern completion review failed", exc_info=True)

    def _process_realtime_signals(self, signals: list):
        """Process real-time WebSocket signals — same pipeline as rotation signals."""
        self._last_fetch_source = "finnhub_realtime"

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
            if self._convergence_alerter is not None:
                try:
                    self._convergence_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Realtime signal feed to alerter failed", exc_info=True)

            tier = TIER_ROUTING.get(sig.source, "tactical")
            tier_alerter = self._tiered_alerters.get(tier)
            if tier_alerter is not None:
                try:
                    tier_alerter.record_signal(**sig_kwargs)
                except Exception:
                    logger.debug("Tier alerter %s failed for signal", tier, exc_info=True)

        # Track stats
        self._total_signals_fed += len(signals)

        # Publish to EventBus
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
                    logger.debug("EventBus publish failed for signal %s", sig.signal_id, exc_info=True)

        # Store
        store_signals(signals, self._memory)

    # ------------------------------------------------------------------
    # Async fetch lifecycle
    # ------------------------------------------------------------------

    def _launch_next_fetch(self):
        """Fill up to 8 concurrent fetch slots using Thompson-guided selection.

        When a Thompson sampler is available, sources are selected
        probabilistically: each source draws from its Beta distribution,
        then the highest-scoring source is picked. This naturally
        balances exploration (new sources with wide posteriors) and
        exploitation (proven sources with high means). Falls back to
        round-robin when no sampler is available.
        """
        # Collect any completed futures first
        self._collect_results()

        # Fill available slots (8 max concurrent)
        slots = 8 - len(self._pending_futures)
        if slots <= 0:
            return

        # Identify eligible sources (not already in-flight)
        # Filter by market clock availability (if available)
        if self._market_clock is not None:
            available = set(self._market_clock.get_available_sources())
            eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures and s in available]
            if not eligible:
                # Fall back to always-available sources if nothing matches
                eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures]
        else:
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

        # Track per-ticker signal domains for archetype scanning (Gift 8)
        for sig in signals:
            sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
            if sym:
                if sym not in self._recent_domains:
                    self._recent_domains[sym] = set()
                self._recent_domains[sym].add(sig.domain)

        # Feed deception detector (Gift 5) — track signal patterns for manipulation detection
        if self._deception_detector is not None:
            for sig in signals:
                try:
                    sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
                    if sym:
                        self._deception_detector.record_signal(
                            ticker=sym,
                            domain=sig.domain,
                            direction=sig.direction,
                            strength=sig.strength,
                            timestamp=sig.timestamp,
                        )
                except Exception:
                    logger.debug("Deception detector record failed", exc_info=True)

        # Feed somatic anticipation (Gift 9) — accumulate per-ticker signal state
        if self._somatic_anticipation is not None:
            for sig in signals:
                try:
                    sym = getattr(sig, "symbol", "") or sig.metadata.get("symbol", "")
                    if sym:
                        self._somatic_anticipation.record_signal(
                            ticker=sym,
                            domain=sig.domain,
                            direction=sig.direction,
                            strength=sig.strength,
                            timestamp=sig.timestamp,
                        )
                except Exception:
                    logger.debug("Somatic anticipation record failed", exc_info=True)

        # Check pattern completions (Gift 10) — match signals against active hunts
        if self._pattern_completion_engine is not None:
            try:
                completion_events = self._pattern_completion_engine.check_completions(signals)
                if completion_events:
                    logger.info(
                        "Pattern completion: %d hunts matched by incoming signals",
                        len(completion_events),
                    )
            except Exception:
                logger.debug("Pattern completion check failed", exc_info=True)

        self._total_signals_fed += len(signals)
        logger.info(
            "Market sensing: fed %d signals from [%s] (total: %d)",
            len(signals), source_name, self._total_signals_fed,
        )

        # Record signal arrival for AbsenceMonitor cadence tracking
        if self._absence_monitor is not None and signals:
            try:
                self._absence_monitor.record_arrival(source_name, datetime.now())
            except Exception:
                logger.debug("AbsenceMonitor record_arrival failed", exc_info=True)

        # Feed CorrelationTracker (cross-source anomaly detection)
        if self._correlation_tracker is not None and signals:
            try:
                max_strength = max(sig.strength for sig in signals)
                domain = signals[0].domain if hasattr(signals[0], "domain") else None
                self._correlation_tracker.record(
                    signal_id=source_name,
                    value=max_strength,
                    timestamp=datetime.now(),
                    domain=domain,
                )
            except Exception:
                logger.debug("CorrelationTracker record failed", exc_info=True)

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
        store_signals(signals, self._memory)

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
            from_order_flow,
            from_fractal_resonance,
            from_crypto_signal,
            from_openinsider,
            from_13f_holding,
            from_activist_filing,
            from_finviz_unusual_volume,
            from_finviz_short_squeeze,
            from_suppression_event,
        )
        from mae_core.market.signal_adapters.wave2_3 import from_finviz_insider

        signals = []

        if source_name == "sec_form4":
            signals = fetch_sec_form4(self._sec_client, self._watchlist, from_insider_trade)

        elif source_name == "sec_form8k":
            signals = fetch_sec_form8k(self._sec_client, self._watchlist, from_form8k_event)

        elif source_name == "congressional":
            signals = fetch_congressional(self._congress_client, from_congressional_trade)

        elif source_name == "senate":
            signals = fetch_senate(self._senate_client, from_senate_trade)

        elif source_name == "hiring":
            signals = fetch_hiring(self._job_tracker, self._watchlist, from_hiring_signal)

        elif source_name == "usa_spending":
            signals = fetch_usa_spending(self._usa_spending, self._watchlist, from_government_contract)

        elif source_name == "sam_gov_and_prices":
            signals = fetch_sam_gov(self._sam_gov, self._watchlist, from_contract_opportunity)

        elif source_name == "social_sentiment":
            signals = fetch_social_sentiment(self._apewisdom, self._watchlist, from_social_sentiment)

        elif source_name == "finra_short":
            signals = fetch_finra_short(self._finra_client, self._watchlist, from_short_interest)

        elif source_name == "sec_efts":
            signals = fetch_sec_efts(self._sec_efts, from_filing_keyword)

        elif source_name == "finnhub":
            signals = fetch_finnhub(self._finnhub, self._watchlist, from_news_sentiment, from_earnings_event)

        elif source_name == "fred_macro":
            signals = fetch_fred(self._fred, from_macro_indicator)

        elif source_name == "session_sweep":
            signals = fetch_session_sweep(self._session_sweep_detector, from_session_sweep)

        elif source_name == "ta_indicators":
            signals = fetch_ta_indicators(self._ta_indicators, self._price_fetcher, self._watchlist, from_ta_signal)

        elif source_name == "cot_positioning":
            signals = fetch_cot(self._cot_client, self._watchlist, from_cot_positioning)

        elif source_name == "stocktwits":
            signals = fetch_stocktwits(self._stocktwits_client, self._watchlist, from_stocktwits_sentiment)

        elif source_name == "vix_structure":
            signals = fetch_vix(self._vix_client, from_vix_structure)

        elif source_name == "google_trends":
            signals = fetch_trends(self._trends_client, self._watchlist, from_trends_signal)

        elif source_name == "finnhub_extras":
            signals = fetch_finnhub_extras(
                self._finnhub, self._watchlist, from_economic_event, from_analyst_recommendation
            )

        elif source_name == "order_flow":
            signals = fetch_order_flow(self._order_flow_detector, self._watchlist, from_order_flow)

        elif source_name == "fractal_resonance":
            signals = fetch_fractal_resonance(
                self._fractal_resonance_detector, self._watchlist, from_fractal_resonance
            )

        elif source_name == "crypto_prices":
            signals = fetch_crypto_prices(self._coingecko_client, from_crypto_signal)

        elif source_name == "crypto_exchange":
            signals = fetch_crypto_exchange(self._coincap_client, from_crypto_signal)

        elif source_name == "openinsider":
            signals = fetch_openinsider(self._openinsider_client, from_openinsider)

        elif source_name == "institutional_13f":
            signals = fetch_13f_holdings(self._edgar_enhanced_client, from_13f_holding, from_activist_filing)

        elif source_name == "finviz":
            signals = fetch_finviz(self._finviz_client, from_finviz_unusual_volume, from_finviz_short_squeeze)

        elif source_name == "economic_calendar":
            signals = fetch_economic_calendar(self._economic_calendar_client, from_suppression_event)

        elif source_name == "massive_snapshot":
            from mae_core.market.signal_adapters.wave2_3 import from_massive_snapshot
            signals = fetch_massive_snapshot(self._massive_client, self._watchlist, from_massive_snapshot)

        elif source_name == "eia_energy":
            from mae_core.market.signal_adapters.market_data import from_energy_indicator
            signals = fetch_eia(self._eia_client, from_energy_indicator)

        elif source_name == "congress_legislation":
            from mae_core.market.signal_adapters.market_data import from_legislative_indicator
            signals = fetch_congress_legislation(self._congress_gov_client, from_legislative_indicator)

        # Enrich in background thread (velocity, filing-time, Ollama sentiment)
        # Moved from _collect_results() so Ollama's 15s timeout doesn't block
        # the main step loop. Thread-safe: only mutates signal objects.
        for sig in signals:
            enrich_signal(sig, self._velocity_detector, self._filing_analyzer, self._form8k_sentiment)

        return signals

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
