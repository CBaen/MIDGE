"""Bootstrap Layer 33a: Market system instantiation.

Construct all market system objects on ctx. Every instantiation is wrapped in
try/except — failures set ctx.attr = None (graceful degradation).
Also handles BoundaryMembrane trust and ApiGateway registration.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _instantiate_wave2_3_clients(ctx: SimpleNamespace) -> None:
    """Construct Wave 2+3 API clients (Always-On MIDGE)."""
    import importlib
    for attr, mod_path, cls in [
        ("coingecko_client", "mae_core.market.apis.coingecko_client", "CoinGeckoClient"),
        ("coincap_client", "mae_core.market.apis.coincap_client", "CoinCapClient"),
        ("openinsider_client", "mae_core.market.apis.openinsider_client", "OpenInsiderClient"),
        ("edgar_enhanced_client", "mae_core.market.apis.edgar_enhanced_client", "EdgarEnhancedClient"),
        ("finviz_client", "mae_core.market.apis.finviz_client", "FinVizClient"),
        ("economic_calendar_client", "mae_core.market.apis.economic_calendar_client", "EconomicCalendarClient"),
        ("finnhub_websocket", "mae_core.market.apis.finnhub_websocket", "FinnhubWebSocket"),
        ("massive_client", "mae_core.market.apis.massive_client", "MassiveClient"),
    ]:
        try:
            setattr(ctx, attr, getattr(importlib.import_module(mod_path), cls)())
        except Exception:
            logger.debug("Market: %s failed to construct", attr, exc_info=True)
            setattr(ctx, attr, None)


def _instantiate_market_systems(ctx: SimpleNamespace) -> None:
    """Create all 56 market system objects on ctx."""
    # Warm-start: restore meta-learned config from prior sessions
    try:
        from mae_core.market.intelligence.learning_config import load_snapshot
        if load_snapshot():
            logger.info("Market config snapshot loaded — meta-learned values active from step 1")
    except Exception:
        logger.debug("Config snapshot warm-start skipped", exc_info=True)

    qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
    failures = 0

    # --- Create MarketDataProvider first (injected into all API clients) ---
    provider = None
    try:
        from mae_core.market.apis.market_data_provider import MarketDataProvider
        provider = MarketDataProvider()
        ctx.market_data_provider = provider
    except Exception:
        logger.debug("Market: market_data_provider failed to construct", exc_info=True)
        ctx.market_data_provider = None

    # --- API clients (Market Sensing) — provider injected for gateway routing ---

    try:
        from mae_core.market.apis.sec_edgar.client import SECEdgarClient
        ctx.sec_edgar_client = SECEdgarClient(provider=provider)
    except Exception:
        logger.debug("Market: sec_edgar_client failed to construct", exc_info=True)
        ctx.sec_edgar_client = None
        failures += 1

    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher
        ctx.price_fetcher = PriceFetcher(
            alpha_vantage_key=os.environ.get("MAE_ALPHAVANTAGE_API_KEY", ""),
            provider=provider,
        )
    except Exception:
        logger.debug("Market: price_fetcher failed to construct", exc_info=True)
        ctx.price_fetcher = None
        failures += 1

    try:
        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
        ctx.house_stock_watcher = HouseStockWatcherClient(provider=provider)
    except Exception:
        logger.debug("Market: house_stock_watcher failed to construct", exc_info=True)
        ctx.house_stock_watcher = None
        failures += 1

    try:
        from mae_core.market.apis.job_tracker import JobTracker
        ctx.job_tracker = JobTracker(provider=provider)
    except Exception:
        logger.debug("Market: job_tracker failed to construct", exc_info=True)
        ctx.job_tracker = None
        failures += 1

    try:
        from mae_core.market.apis.usa_spending import USASpendingClient
        ctx.usa_spending_client = USASpendingClient(provider=provider)
    except Exception:
        logger.debug("Market: usa_spending_client failed to construct", exc_info=True)
        ctx.usa_spending_client = None
        failures += 1

    try:
        from mae_core.market.apis.sam_gov import SAMGovClient
        ctx.sam_gov_client = SAMGovClient(provider=provider)
    except Exception:
        logger.debug("Market: sam_gov_client failed to construct", exc_info=True)
        ctx.sam_gov_client = None
        failures += 1

    # --- Phase 2 API clients (new free sources) ---
    try:
        from mae_core.market.apis.senate_stock_watcher import SenateStockWatcherClient
        ctx.senate_stock_watcher = SenateStockWatcherClient(provider=provider)
    except Exception:
        logger.debug("Market: senate_stock_watcher failed to construct", exc_info=True)
        ctx.senate_stock_watcher = None

    try:
        from mae_core.market.apis.apewisdom import ApeWisdomClient
        ctx.apewisdom_client = ApeWisdomClient(provider=provider)
    except Exception:
        logger.debug("Market: apewisdom_client failed to construct", exc_info=True)
        ctx.apewisdom_client = None

    try:
        from mae_core.market.apis.finra_short_interest import FINRAShortInterestClient
        ctx.finra_client = FINRAShortInterestClient(provider=provider)
    except Exception:
        logger.debug("Market: finra_client failed to construct", exc_info=True)
        ctx.finra_client = None

    try:
        from mae_core.market.apis.sec_edgar.efts import SECFullTextSearchClient
        ctx.sec_efts_client = SECFullTextSearchClient(provider=provider)
    except Exception:
        logger.debug("Market: sec_efts_client failed to construct", exc_info=True)
        ctx.sec_efts_client = None

    try:
        from mae_core.market.apis.finnhub_client import FinnhubClient
        ctx.finnhub_client = FinnhubClient(provider=provider)
    except Exception:
        logger.debug("Market: finnhub_client failed to construct", exc_info=True)
        ctx.finnhub_client = None

    try:
        from mae_core.market.apis.fred_client import FREDClient
        ctx.fred_client = FREDClient(provider=provider)
    except Exception:
        logger.debug("Market: fred_client failed to construct", exc_info=True)
        ctx.fred_client = None

    # --- Layer 6 API clients (new free sources) ---
    try:
        from mae_core.market.apis.cot_client import COTClient
        ctx.cot_client = COTClient(provider=provider)
    except Exception:
        logger.debug("Market: cot_client failed to construct", exc_info=True)
        ctx.cot_client = None

    try:
        from mae_core.market.apis.stocktwits_client import StockTwitsClient
        ctx.stocktwits_client = StockTwitsClient(provider=provider)
    except Exception:
        logger.debug("Market: stocktwits_client failed to construct", exc_info=True)
        ctx.stocktwits_client = None

    try:
        from mae_core.market.apis.vix_client import VIXClient
        ctx.vix_client = VIXClient(provider=provider)
    except Exception:
        logger.debug("Market: vix_client failed to construct", exc_info=True)
        ctx.vix_client = None

    try:
        from mae_core.market.apis.trends_client import TrendsClient
        ctx.trends_client = TrendsClient()  # pytrends library, not HTTP provider
    except Exception:
        logger.debug("Market: trends_client failed to construct", exc_info=True)
        ctx.trends_client = None

    # --- Wave 2+3: Real-Time + Crypto + Data Enrichment (Always-On MIDGE) ---
    _instantiate_wave2_3_clients(ctx)

    # --- Edge detectors ---
    try:
        from mae_core.market.edge.cluster_detector import ClusterDetector
        ctx.cluster_detector = ClusterDetector(qdrant_url=qdrant_url)
    except Exception:
        logger.debug("Market: cluster_detector failed to construct", exc_info=True)
        ctx.cluster_detector = None
        failures += 1

    try:
        from mae_core.market.edge.politician_tracker import PoliticianTracker
        ctx.politician_tracker = PoliticianTracker()
    except Exception:
        logger.debug("Market: politician_tracker failed to construct", exc_info=True)
        ctx.politician_tracker = None
        failures += 1

    try:
        from mae_core.market.edge.filing_time_analyzer import FilingTimeAnalyzer
        ctx.filing_time_analyzer = FilingTimeAnalyzer(qdrant_url=qdrant_url)
    except Exception:
        logger.debug("Market: filing_time_analyzer failed to construct", exc_info=True)
        ctx.filing_time_analyzer = None
        failures += 1

    try:
        from mae_core.market.edge.contract_predictor import ContractPredictor
        ctx.contract_predictor = ContractPredictor(qdrant_url=qdrant_url)
    except Exception:
        logger.debug("Market: contract_predictor failed to construct", exc_info=True)
        ctx.contract_predictor = None
        failures += 1

    try:
        from mae_core.market.edge.session_sweep_detector import SessionSweepDetector
        ctx.session_sweep_detector = SessionSweepDetector()
    except Exception:
        logger.debug("Market: session_sweep_detector failed to construct", exc_info=True)
        ctx.session_sweep_detector = None
        failures += 1

    # TA indicators (pure computation — no constructor args needed)
    try:
        from mae_core.market.edge import ta_indicators as _ta_mod
        ctx.ta_indicators = _ta_mod  # Module reference — compute_all() is the entry point
    except Exception:
        logger.debug("Market: ta_indicators failed to import", exc_info=True)
        ctx.ta_indicators = None
        failures += 1

    # --- Intelligence layer ---
    try:
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        ctx.thompson_sampler = ThompsonSampler()
    except Exception:
        logger.debug("Market: thompson_sampler failed to construct", exc_info=True)
        ctx.thompson_sampler = None
        failures += 1

    if ctx.thompson_sampler is not None:  # Seed combo Thompson from replay results
        try:
            _rp = Path(__file__).resolve().parents[2] / "data" / "midge" / "replay_results.json"
            _n = ctx.thompson_sampler.seed_combo_distributions(_rp)
            if _n:
                logger.info("Market: seeded %d combo Thompson distributions from replay", _n)
        except Exception:
            logger.debug("Market: combo Thompson seeding failed", exc_info=True)
    try:
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        ctx.convergence_alerter = ConvergenceAlerter(
            min_domains=3,
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            causal_engine=getattr(ctx, "shared_causal_engine", None),
            event_bus=getattr(ctx, "bus", None),
            economic_calendar=getattr(ctx, "economic_calendar_client", None),
        )
    except Exception:
        logger.debug("Market: convergence_alerter failed to construct", exc_info=True)
        ctx.convergence_alerter = None
        failures += 1

    try:
        from mae_core.market.intelligence.velocity_detector import VelocityDetector
        ctx.velocity_detector = VelocityDetector()
    except Exception:
        logger.debug("Market: velocity_detector failed to construct", exc_info=True)
        ctx.velocity_detector = None
        failures += 1

    try:
        from mae_core.market.intelligence.correlation_tracker import CorrelationTracker
        ctx.correlation_tracker = CorrelationTracker()
    except Exception:
        logger.debug("Market: correlation_tracker failed to construct", exc_info=True)
        ctx.correlation_tracker = None
        failures += 1

    # --- Regime classifier (requires price_fetcher) ---
    try:
        from mae_core.market.intelligence.regime_classifier import RegimeClassifier
        ctx.regime_classifier = RegimeClassifier(price_fetcher=ctx.price_fetcher)
    except Exception:
        logger.debug("Market: regime_classifier failed to construct", exc_info=True)
        ctx.regime_classifier = None
        failures += 1

    # Wire regime classifier into convergence alerter (two-phase init —
    # regime_classifier is constructed after convergence_alerter)
    if getattr(ctx, "convergence_alerter", None) is not None and ctx.regime_classifier is not None:
        ctx.convergence_alerter._regime_classifier = ctx.regime_classifier

    # Warm-start: restore signal buffer from prior session (WP-A persistence)
    if getattr(ctx, "convergence_alerter", None) is not None:
        try:
            restored = ctx.convergence_alerter.load_signal_buffer()
            if restored:
                logger.info("Market: convergence alerter signal buffer restored from prior session (%d signals)", restored)
        except Exception:
            logger.debug("Market: convergence alerter signal buffer restore failed", exc_info=True)

    # --- Feedback loop (requires price_fetcher + thompson_sampler) ---
    try:
        from mae_core.market.outcome_tracker import OutcomeTracker
        ctx.outcome_tracker = OutcomeTracker(
            price_fetcher=ctx.price_fetcher,
            thompson_sampler=ctx.thompson_sampler,
            regime_classifier=ctx.regime_classifier,
        )
    except Exception:
        logger.debug("Market: outcome_tracker failed to construct", exc_info=True)
        ctx.outcome_tracker = None
        failures += 1

    # --- Phase 4: Signal archive, lag-correlation, calibration, Kelly ---
    try:
        from mae_core.market.intelligence.signal_archive_reader import SignalArchiveReader
        ctx.signal_archive_reader = SignalArchiveReader()
    except Exception:
        logger.debug("Market: signal_archive_reader failed to construct", exc_info=True)
        ctx.signal_archive_reader = None
        failures += 1

    try:
        from mae_core.market.intelligence.lag_correlation_analyzer import LagCorrelationAnalyzer
        ctx.lag_correlation_analyzer = (
            LagCorrelationAnalyzer(archive_reader=ctx.signal_archive_reader)
            if ctx.signal_archive_reader is not None else None
        )
    except Exception:
        logger.debug("Market: lag_correlation_analyzer failed to construct", exc_info=True)
        ctx.lag_correlation_analyzer = None
        failures += 1

    try:
        from mae_core.market.intelligence.thompson_calibrator import ThompsonCalibrator
        ctx.thompson_calibrator = (
            ThompsonCalibrator(thompson_sampler=ctx.thompson_sampler)
            if ctx.thompson_sampler is not None else None
        )
    except Exception:
        logger.debug("Market: thompson_calibrator failed to construct", exc_info=True)
        ctx.thompson_calibrator = None
        failures += 1

    try:
        from mae_core.market.intelligence.kelly_position_sizer import KellyPositionSizer
        ctx.kelly_position_sizer = (
            KellyPositionSizer(thompson_sampler=ctx.thompson_sampler)
            if ctx.thompson_sampler is not None else None
        )
    except Exception:
        logger.debug("Market: kelly_position_sizer failed to construct", exc_info=True)
        ctx.kelly_position_sizer = None
        failures += 1

    # --- Hypothesis loop (RSI Layer 2: generator + validator + engine) ---

    try:
        from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
        ctx.hypothesis_registry = HypothesisRegistry()
    except Exception:
        logger.debug("Market: hypothesis_registry failed to construct", exc_info=True)
        ctx.hypothesis_registry = None

    try:
        from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
        ctx.hypothesis_generator = (
            HypothesisGenerator(registry=ctx.hypothesis_registry)
            if ctx.hypothesis_registry is not None else None
        )
    except Exception:
        logger.debug("Market: hypothesis_generator failed to construct", exc_info=True)
        ctx.hypothesis_generator = None

    try:
        from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
        ctx.hypothesis_validator = HypothesisValidator(
            causal_engine=getattr(ctx, "shared_causal_engine", None),
        )
    except Exception:
        logger.debug("Market: hypothesis_validator failed to construct", exc_info=True)
        ctx.hypothesis_validator = None

    try:
        from mae_core.market.intelligence.backtest_analyzer import BacktestAnalyzer
        ctx.backtest_analyzer = (
            BacktestAnalyzer(registry=ctx.hypothesis_registry)
            if ctx.hypothesis_registry is not None else None
        )
    except Exception:
        logger.debug("Market: backtest_analyzer failed to construct", exc_info=True)
        ctx.backtest_analyzer = None

    try:
        from mae_core.market.intelligence.backtest_scheduler import BacktestScheduler
        ctx.backtest_scheduler = (
            BacktestScheduler(
                backtest_analyzer=ctx.backtest_analyzer,
                bus=ctx.bus,
            )
            if ctx.backtest_analyzer is not None else None
        )
    except Exception:
        logger.debug("Market: backtest_scheduler failed to construct", exc_info=True)
        ctx.backtest_scheduler = None

    try:
        from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
        ctx.hypothesis_engine = (
            HypothesisEngine(
                registry=ctx.hypothesis_registry,
                generator=ctx.hypothesis_generator,
                validator=ctx.hypothesis_validator,
                bus=ctx.bus,
                regime_classifier=getattr(ctx, "regime_classifier", None),
                thompson_sampler=getattr(ctx, "thompson_sampler", None),
                backtest_analyzer=getattr(ctx, "backtest_analyzer", None),
                thompson_calibrator=getattr(ctx, "thompson_calibrator", None),
            )
            if (ctx.hypothesis_registry is not None
                and ctx.hypothesis_generator is not None
                and ctx.hypothesis_validator is not None)
            else None
        )
    except Exception:
        logger.debug("Market: hypothesis_engine failed to construct", exc_info=True)
        ctx.hypothesis_engine = None

    # --- AbsenceMonitor (detecting unexpectedly silent sources) ---
    try:
        from mae_core.market.intelligence.absence_monitor import AbsenceMonitor
        ctx.absence_monitor = AbsenceMonitor(
            event_bus=getattr(ctx, "bus", None),
            archive_reader=getattr(ctx, "signal_archive_reader", None),
        )
        # Bootstrap cadences from signal archive (non-blocking, best-effort)
        if ctx.absence_monitor is not None:
            try:
                ctx.absence_monitor.bootstrap_from_archives()
            except Exception:
                logger.debug("AbsenceMonitor archive bootstrap skipped", exc_info=True)
    except Exception:
        logger.debug("Market: absence_monitor failed to construct", exc_info=True)
        ctx.absence_monitor = None

    # --- Ten Gifts: Wave 1-3 (extracted to market_gifts.py to stay under 500 lines) ---
    try:
        from mae_core.bootstrap.market_gifts import _instantiate_gift_systems
        _instantiate_gift_systems(ctx)
    except Exception:
        logger.debug("Market: gift systems instantiation failed", exc_info=True)

    # Warm-start: restore somatic anticipation state (WP-A persistence)
    if getattr(ctx, "somatic_anticipation", None) is not None:
        try:
            ctx.somatic_anticipation.load_state()
            logger.info("Market: somatic anticipation state restored from prior session")
        except Exception:
            logger.debug("Market: somatic anticipation state restore failed", exc_info=True)

    # Warm-start: restore deception detector state (WP-A persistence)
    if getattr(ctx, "deception_detector", None) is not None:
        try:
            ctx.deception_detector.load_state()
            logger.info("Market: deception detector state restored from prior session")
        except Exception:
            logger.debug("Market: deception detector state restore failed", exc_info=True)

    # --- StepTimer (performance metabolism monitoring) ---
    try:
        from mae_core.market.step_timer import StepTimer
        ctx.step_timer = StepTimer()
    except Exception:
        ctx.step_timer = None

    # --- Trust registration with BoundaryMembrane ---
    market_sources = [
        ("sec_edgar", 0.90), ("yfinance", 0.75), ("alpha_vantage", 0.80), ("rapidapi", 0.65),
        ("usa_spending", 0.85), ("sam_gov", 0.80), ("senate_free", 0.80), ("apewisdom", 0.45),
        ("finra_short", 0.85), ("sec_efts", 0.90), ("finnhub", 0.75), ("fred_macro", 0.95),
        ("session_sweep", 0.60),
    ]
    bm = getattr(ctx, "boundary_membrane", None)
    if bm is not None and hasattr(bm, "register_source"):
        for source_name, trust in market_sources:
            try:
                bm.register_source(source_name, trust=trust)
            except Exception:
                logger.debug("Could not register %s with BoundaryMembrane", source_name)

    # --- Register MarketDataProvider with ApiGateway ---
    gateway = getattr(ctx, "api_gateway", None)
    if gateway is not None and ctx.market_data_provider is not None:
        try:
            gateway.register_provider("market_data", ctx.market_data_provider)
        except Exception:
            logger.debug("Could not register MarketDataProvider with ApiGateway", exc_info=True)

    logger.info(
        "Layer 33a - Market systems: %d instantiated (%d failures)", 56 - failures, failures,
    )
