"""Market Sensing Hook — Async market data fetcher for MIDGE's step loop.

Biological analogy: The autonomic nervous system. Automatically gathers
sensory data on a cadence without conscious agent involvement. Agents
then process the gathered signals through their lifecycle.

Replaces the monolithic midge_scan.py pipeline with incremental sensing
that runs inside MIDGE's 33-layer bootstrap. Each model step, the hook:
1. Collects results from any completed async fetch
2. Feeds signals into convergence_alerter + tiered alerters
3. Launches the next async fetch (on cadence, rotating through sources)

Threading: 12 concurrent workers via ThreadPoolExecutor(12).
Same pattern as ApiGateway (proven safe). No race conditions on
convergence_alerter because collection happens in the main thread.

Decomposed into seven files:
  sensing_hook.py       — this file: MarketSensingHook class (thin orchestrator)
  sensing_constants.py  — TIER_ROUTING, SOURCE_ROTATION, Thompson/domain maps
  sensing_fetchers.py   — thin re-export hub for 35 standalone fetch functions
  sensing_lifecycle.py  — enrich_signal, store_signals, load_watchlist
  sensing_scheduler.py  — SensingSchedulerMixin: Thompson/round-robin scheduling
  sensing_collector.py  — SensingCollectorMixin: result collection + signal pipeline
  sensing_reactive.py   — SensingReactiveMixin: reactive convergence + fetch dispatch
  sensing_step_ops.py   — SensingStepOpsMixin: periodic cadence-based step operations
"""

from __future__ import annotations

import logging
import random
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mae_core.market.sensing_constants import (
    TIER_ROUTING,
    SOURCE_ROTATION,
    _ROTATION_TO_THOMPSON,
    _ABSENCE_SOURCE_DOMAINS,
    _DOMAIN_TO_SOURCES,
    _absence_source_to_domain,
)
from mae_core.market.sensing_lifecycle import (
    enrich_signal,
    store_signals,
    load_watchlist,
)
from mae_core.market.sensing_scheduler import SensingSchedulerMixin
from mae_core.market.sensing_collector import SensingCollectorMixin
from mae_core.market.sensing_reactive import SensingReactiveMixin
from mae_core.market.sensing_step_ops import SensingStepOpsMixin

logger = logging.getLogger("midge.market.sensing")

# Data paths
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "midge"
SIGNALS_DIR = DATA_DIR / "signals"


class MarketSensingHook(SensingSchedulerMixin, SensingCollectorMixin, SensingReactiveMixin, SensingStepOpsMixin):
    """Async market data fetcher that runs as a step hook.

    Instantiated in bootstrap/market.py and registered via
    ctx.model.add_step_hook(hook.step).

    Decomposed via mixins:
      SensingSchedulerMixin — Thompson-guided + round-robin scheduling
      SensingCollectorMixin — result collection + signal pipeline
      SensingReactiveMixin  — reactive convergence + source dispatch table
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
        social_text_analyzer: Any = None,
        yahoo_rss_client: Any = None,
        usda_client: Any = None,
        binance_funding_client: Any = None,
        kalshi_client: Any = None,
        cluster_detector: Any = None,
        politician_tracker: Any = None,
        contract_predictor: Any = None,
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
        self._social_text_analyzer = social_text_analyzer
        self._yahoo_rss_client = yahoo_rss_client
        self._usda_client = usda_client
        self._binance_funding_client = binance_funding_client
        self._kalshi_client = kalshi_client
        self._cluster_detector = cluster_detector
        self._politician_tracker = politician_tracker
        self._contract_predictor = contract_predictor

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

        # Async fetch state — 12 concurrent workers for parallel senses
        self._executor = ThreadPoolExecutor(max_workers=12, thread_name_prefix="mkt-sense")
        self._pending_futures: Dict[str, Future] = {}  # source_name -> future
        self._fetch_queue = deque(SOURCE_ROTATION)
        self._step_counter = 0
        self._fetch_cadence = fetch_cadence
        self._outcome_cadence = outcome_cadence

        # Recent signal domains per ticker (for archetype scanning)
        self._recent_domains: Dict[str, set] = {}

        # Priority polling: market_hooks.py writes here when partial convergence
        # is detected (2 of 3+ domains seen, missing domain identified).
        # Format: {ticker: {"ticker": str, "domains_needed": List[str],
        #                   "priority": str, "expires": float, "source": str}}
        # Cap: 50 entries. Entries expire after 1 hour (3600 seconds).
        self._priority_requests: Dict[str, dict] = {}

        # Stats
        self._total_signals_fed = 0
        self._total_fetches = 0
        self._last_fetch_source = None

        # Reactive convergence cache — alerts fired immediately on signal ingestion.
        # Step-hook convergence check in market_hooks.py reads this to avoid
        # double-processing. List is cleared each step by _collect_results caller.
        self._cached_reactive_alerts: List = []

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
        if self._step_counter % 50 == 0:
            self._run_step_portfolio()

        # Catalyst calendar: refresh upcoming events (cadence 200)
        if self._step_counter % 200 == 0 and self._catalyst_calendar is not None:
            try:
                self._catalyst_calendar.refresh()
            except Exception:
                logger.debug("Catalyst calendar refresh failed", exc_info=True)

        # Absence detection on cadence 100
        if self._step_counter % 100 == 0:
            self._run_step_absence()

        # Cross-source correlation anomaly scan on cadence 200
        if self._step_counter % 200 == 0:
            self._run_step_correlation()

        # Consolidation engine: memory pruning (cadence 5000)
        if self._step_counter % 5000 == 0:
            self._run_step_consolidation()

        # Somatic anticipation: pre-conscious pattern response (cadence 25)
        if self._step_counter % 25 == 0:
            self._run_step_somatic()

        # Pattern archetype scanning: check watchlist tickers (cadence 100)
        if self._step_counter % 100 == 0:
            self._run_step_archetypes()

        # Pattern completion: review partial archetype matches (cadence 100)
        if self._step_counter % 100 == 0:
            self._run_step_pattern_completion()

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
