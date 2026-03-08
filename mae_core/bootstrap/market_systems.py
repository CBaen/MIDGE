"""Bootstrap Layer 33a: Market system instantiation.

Construct all market system objects on ctx. Every instantiation is wrapped in
try/except — failures set ctx.attr = None (graceful degradation).
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


def _instantiate_execution_bridges(ctx: SimpleNamespace) -> None:
    """Construct execution bridges (Alpaca, Kalshi)."""
    try:
        from mae_core.market.apis.alpaca_client import AlpacaClient
        ctx.alpaca_client = AlpacaClient(
            api_key=os.environ.get("ALPACA_API_KEY"),
            secret_key=os.environ.get("ALPACA_SECRET_KEY"), paper=True)
    except Exception:
        ctx.alpaca_client = None

_MARKET_SOURCE_TRUST = [
    ("sec_edgar", 0.90), ("yfinance", 0.75), ("alpha_vantage", 0.80), ("rapidapi", 0.65),
    ("usa_spending", 0.85), ("sam_gov", 0.80), ("senate_free", 0.80), ("apewisdom", 0.45),
    ("finra_short", 0.85), ("sec_efts", 0.90), ("finnhub", 0.75), ("fred_macro", 0.95),
    ("session_sweep", 0.60), ("cot_positioning", 0.85), ("stocktwits", 0.40),
    ("vix_structure", 0.80), ("google_trends", 0.50), ("ta_indicators", 0.70),
    ("order_flow", 0.60), ("fractal_resonance", 0.65), ("crypto_coingecko", 0.70),
    ("crypto_coincap", 0.65), ("openinsider", 0.80), ("institutional_13f", 0.85),
    ("finviz", 0.65), ("economic_calendar", 0.80), ("massive_snapshot", 0.90),
    ("eia_energy", 0.95), ("congress_legislation", 0.90),
]


def _register_trust_and_gateway(ctx: SimpleNamespace) -> None:
    """Register all market sources with BoundaryMembrane + ApiGateway."""
    bm = getattr(ctx, "boundary_membrane", None)
    if bm is not None and hasattr(bm, "register_source"):
        for source_name, trust in _MARKET_SOURCE_TRUST:
            try:
                bm.register_source(source_name, trust=trust)
            except Exception:
                logger.debug("Could not register %s with BoundaryMembrane", source_name)
    gateway = getattr(ctx, "api_gateway", None)
    if gateway is not None and getattr(ctx, "market_data_provider", None) is not None:
        try:
            gateway.register_provider("market_data", ctx.market_data_provider)
        except Exception:
            logger.debug("Could not register MarketDataProvider with ApiGateway", exc_info=True)


def _instantiate_market_systems(ctx: SimpleNamespace) -> None:
    """Create all market system objects on ctx."""
    try:
        from mae_core.market.intelligence.learning_config import load_snapshot
        if load_snapshot():
            logger.info("Market config snapshot loaded")
    except Exception:
        pass
    qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
    failures = 0
    provider = None
    try:
        from mae_core.market.apis.market_data_provider import MarketDataProvider
        provider = MarketDataProvider()
        ctx.market_data_provider = provider
    except Exception:
        ctx.market_data_provider = None

    # --- API clients (Market Sensing) — provider injected for gateway routing ---
    import importlib as _imp
    for _attr, _mod, _cls, _kw in [
        ("sec_edgar_client", "mae_core.market.apis.sec_edgar.client", "SECEdgarClient", {"provider": provider}),
        ("house_stock_watcher", "mae_core.market.apis.house_stock_watcher", "HouseStockWatcherClient", {"provider": provider}),
        ("job_tracker", "mae_core.market.apis.job_tracker", "JobTracker", {"provider": provider}),
        ("usa_spending_client", "mae_core.market.apis.usa_spending", "USASpendingClient", {"provider": provider}),
        ("sam_gov_client", "mae_core.market.apis.sam_gov", "SAMGovClient", {"provider": provider}),
    ]:
        try:
            setattr(ctx, _attr, getattr(_imp.import_module(_mod), _cls)(**_kw))
        except Exception:
            logger.debug("Market: %s failed", _attr, exc_info=True)
            setattr(ctx, _attr, None)
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

    # --- Raw data storage (persist ALL API data before processing) ---
    try:
        from mae_core.market.raw_store import RawStore
        raw_store = RawStore()
    except Exception:
        raw_store = None

    # --- Phase 2 + Layer 6 API clients (free sources) ---
    for _attr, _mod, _cls, _kw in [
        ("senate_stock_watcher", "mae_core.market.apis.senate_stock_watcher", "SenateStockWatcherClient", {"provider": provider}),
        ("apewisdom_client", "mae_core.market.apis.apewisdom", "ApeWisdomClient", {"provider": provider}),
        ("finra_client", "mae_core.market.apis.finra_short_interest", "FINRAShortInterestClient", {"provider": provider}),
        ("sec_efts_client", "mae_core.market.apis.sec_edgar.efts", "SECFullTextSearchClient", {"provider": provider}),
        ("finnhub_client", "mae_core.market.apis.finnhub_client", "FinnhubClient", {"provider": provider}),
        ("fred_client", "mae_core.market.apis.fred_client", "FREDClient", {"provider": provider}),
        ("cot_client", "mae_core.market.apis.cot_client", "COTClient", {"provider": provider, "raw_store": raw_store}),
        ("stocktwits_client", "mae_core.market.apis.stocktwits_client", "StockTwitsClient", {"provider": provider}),
        ("vix_client", "mae_core.market.apis.vix_client", "VIXClient", {"provider": provider, "raw_store": raw_store}),
        ("trends_client", "mae_core.market.apis.trends_client", "TrendsClient", {"raw_store": raw_store}),
        ("eia_client", "mae_core.market.apis.eia_client", "EIAClient", {"provider": provider, "raw_store": raw_store}),
        ("congress_gov_client", "mae_core.market.apis.congress_gov_client", "CongressGovClient", {"provider": provider}),
    ]:
        try:
            setattr(ctx, _attr, getattr(_imp.import_module(_mod), _cls)(**_kw))
        except Exception:
            logger.debug("Market: %s failed", _attr, exc_info=True)
            setattr(ctx, _attr, None)

    # --- Wave 2+3: Real-Time + Crypto + Data Enrichment (Always-On MIDGE) ---
    _instantiate_wave2_3_clients(ctx)
    _instantiate_execution_bridges(ctx)

    # --- Edge detectors ---
    for _attr, _mod, _cls, _kw in [
        ("cluster_detector", "mae_core.market.edge.cluster_detector", "ClusterDetector", {"qdrant_url": qdrant_url}),
        ("politician_tracker", "mae_core.market.edge.politician_tracker", "PoliticianTracker", {}),
        ("filing_time_analyzer", "mae_core.market.edge.filing_time_analyzer", "FilingTimeAnalyzer", {"qdrant_url": qdrant_url}),
        ("contract_predictor", "mae_core.market.edge.contract_predictor", "ContractPredictor", {"qdrant_url": qdrant_url}),
        ("session_sweep_detector", "mae_core.market.edge.session_sweep_detector", "SessionSweepDetector", {}),
    ]:
        try:
            import importlib as _imp
            _m = _imp.import_module(_mod)
            setattr(ctx, _attr, getattr(_m, _cls)(**_kw))
        except Exception:
            logger.debug("Market: %s failed to construct", _attr, exc_info=True)
            setattr(ctx, _attr, None)
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
            from pathlib import Path
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
        from pathlib import Path as _Path
        ctx.correlation_tracker = CorrelationTracker()
        _lag_file = _Path(__file__).resolve().parents[2] / "data" / "market" / "lag_correlations.json"
        try:
            _seeded = ctx.correlation_tracker.seed_from_lag_data(str(_lag_file))
            if _seeded:
                logger.info("Market: correlation_tracker seeded %d pairs from lag data", _seeded)
        except Exception:
            logger.debug("Market: correlation_tracker lag seeding failed", exc_info=True)
    except Exception:
        logger.debug("Market: correlation_tracker failed to construct", exc_info=True)
        ctx.correlation_tracker = None
        failures += 1

    # Two-phase wiring: correlation_tracker → convergence_alerter (mirrors regime_classifier pattern)
    if getattr(ctx, "convergence_alerter", None) is not None and ctx.correlation_tracker is not None:
        ctx.convergence_alerter._correlation_tracker = ctx.correlation_tracker

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
        from mae_core.market.intelligence.granger_analyzer import GrangerAnalyzer
        ctx.granger_analyzer = (
            GrangerAnalyzer(archive_reader=ctx.signal_archive_reader)
            if ctx.signal_archive_reader is not None else None
        )
    except Exception:
        logger.debug("Market: granger_analyzer failed to construct", exc_info=True)
        ctx.granger_analyzer = None
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

    # --- Pattern Archaeology (library + watcher + excavation daemon) ---
    ctx.pattern_library = ctx.pattern_watcher = ctx.excavation_daemon = None
    try:
        from mae_core.market.archaeology.pattern_library import PatternLibrary
        from mae_core.market.archaeology.pattern_watcher import PatternWatcher
        from mae_core.market.archaeology.excavator import Excavator
        from mae_core.market.archaeology.historical_fetcher import HistoricalDataFetcher
        from mae_core.market.archaeology.excavation_daemon import ExcavationDaemon

        ctx.pattern_library = PatternLibrary()
        ctx.pattern_watcher = PatternWatcher(
            library=ctx.pattern_library, bus=getattr(ctx, "bus", None),
        )
        fetcher = HistoricalDataFetcher(
            sec_client=getattr(ctx, "sec_edgar_client", None),
            fred_client=getattr(ctx, "fred_client", None),
            cot_client=getattr(ctx, "cot_client", None),
            congress_client=getattr(ctx, "house_stock_watcher", None),
            senate_client=getattr(ctx, "senate_stock_watcher", None),
        )
        ctx.excavation_daemon = ExcavationDaemon(
            library=ctx.pattern_library, excavator=Excavator(fetcher=fetcher),
            price_fetcher=getattr(ctx, "price_fetcher", None),
            bus=getattr(ctx, "bus", None),
        )
        logger.info(
            "Market: Pattern Archaeology initialized (%d fingerprints, %d templates)",
            ctx.pattern_library.size, ctx.pattern_library.template_count,
        )
    except Exception:
        failures += 1
        logger.debug("Market: Pattern Archaeology failed", exc_info=True)

    # --- Active Tracker (continuous monitoring of predicted assets) ---
    ctx.active_tracker = None
    try:
        from mae_core.market.archaeology.active_tracker import ActiveTracker
        ctx.active_tracker = ActiveTracker(
            price_fetcher=getattr(ctx, "price_fetcher", None),
            outcome_collector=getattr(ctx, "outcome_collector", None),
            pattern_library=ctx.pattern_library,
        )
        logger.info(
            "Market: Active Tracker initialized (%d assets tracked)",
            ctx.active_tracker.count,
        )
    except Exception:
        failures += 1
        logger.debug("Market: Active Tracker failed", exc_info=True)

    _register_trust_and_gateway(ctx)

    logger.info(
        "Layer 33a - Market systems: %d instantiated (%d failures)", 56 - failures, failures,
    )
