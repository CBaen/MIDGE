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
    raw_store = getattr(ctx, "_raw_store_ref", None)

    # Clients that accept raw_store as a constructor argument (newly wired)
    _with_raw_store = {
        "coingecko_client", "coincap_client", "openinsider_client",
        "edgar_enhanced_client", "finviz_client", "finnhub_websocket",
        "massive_client",
    }

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
            klass = getattr(importlib.import_module(mod_path), cls)
            kwargs = {"raw_store": raw_store} if (attr in _with_raw_store and raw_store is not None) else {}
            setattr(ctx, attr, klass(**kwargs))
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
    ("social_text", 0.40), ("finviz_insider", 0.55),
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

    # --- Raw data storage (persist ALL API data before processing) ---
    try:
        from mae_core.market.raw_store import RawStore
        raw_store = RawStore()
    except Exception:
        raw_store = None

    # --- API clients (Market Sensing) — provider + raw_store injected ---
    import importlib as _imp
    for _attr, _mod, _cls, _kw in [
        ("sec_edgar_client", "mae_core.market.apis.sec_edgar.client", "SECEdgarClient", {"provider": provider, "raw_store": raw_store}),
        ("house_stock_watcher", "mae_core.market.apis.house_stock_watcher", "HouseStockWatcherClient", {"provider": provider, "raw_store": raw_store}),
        ("job_tracker", "mae_core.market.apis.job_tracker", "JobTracker", {"provider": provider, "raw_store": raw_store}),
        ("usa_spending_client", "mae_core.market.apis.usa_spending", "USASpendingClient", {"provider": provider}),
        ("sam_gov_client", "mae_core.market.apis.sam_gov", "SAMGovClient", {"provider": provider, "raw_store": raw_store}),
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
            provider=provider, raw_store=raw_store,
        )
    except Exception:
        logger.debug("Market: price_fetcher failed to construct", exc_info=True)
        ctx.price_fetcher = None
        failures += 1

    # --- Phase 2 + Layer 6 API clients (free sources) ---
    for _attr, _mod, _cls, _kw in [
        ("senate_stock_watcher", "mae_core.market.apis.senate_stock_watcher", "SenateStockWatcherClient", {"provider": provider, "raw_store": raw_store}),
        ("apewisdom_client", "mae_core.market.apis.apewisdom", "ApeWisdomClient", {"provider": provider, "raw_store": raw_store}),
        ("finra_client", "mae_core.market.apis.finra_short_interest", "FINRAShortInterestClient", {"provider": provider, "raw_store": raw_store}),
        ("sec_efts_client", "mae_core.market.apis.sec_edgar.efts", "SECFullTextSearchClient", {"provider": provider}),
        ("finnhub_client", "mae_core.market.apis.finnhub_client", "FinnhubClient", {"provider": provider, "raw_store": raw_store}),
        ("fred_client", "mae_core.market.apis.fred_client", "FREDClient", {"provider": provider, "raw_store": raw_store}),
        ("cot_client", "mae_core.market.apis.cot_client", "COTClient", {"provider": provider, "raw_store": raw_store}),
        ("stocktwits_client", "mae_core.market.apis.stocktwits_client", "StockTwitsClient", {"provider": provider, "raw_store": raw_store}),
        ("vix_client", "mae_core.market.apis.vix_client", "VIXClient", {"provider": provider, "raw_store": raw_store}),
        ("trends_client", "mae_core.market.apis.trends_client", "TrendsClient", {"raw_store": raw_store}),
        ("eia_client", "mae_core.market.apis.eia_client", "EIAClient", {"provider": provider, "raw_store": raw_store}),
        ("congress_gov_client", "mae_core.market.apis.congress_gov_client", "CongressGovClient", {"provider": provider, "raw_store": raw_store}),
        ("yahoo_rss_client", "mae_core.market.apis.yahoo_rss_client", "YahooRSSClient", {"raw_store": raw_store}),
    ]:
        try:
            setattr(ctx, _attr, getattr(_imp.import_module(_mod), _cls)(**_kw))
        except Exception:
            logger.debug("Market: %s failed", _attr, exc_info=True)
            setattr(ctx, _attr, None)

    # Store raw_store ref on ctx so wave2/3 init can access it
    ctx._raw_store_ref = raw_store

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
    # --- Social text analyzer (reads from RawStore SQLite — no API calls) ---
    try:
        from mae_core.market.intelligence.social_text_analyzer import SocialTextAnalyzer
        ctx.social_text_analyzer = SocialTextAnalyzer(raw_store=raw_store)
    except Exception:
        logger.debug("Market: social_text_analyzer failed to construct", exc_info=True)
        ctx.social_text_analyzer = None
        failures += 1

    try:
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        ctx.convergence_alerter = ConvergenceAlerter(
            min_domains=3,
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            causal_engine=getattr(ctx, "shared_causal_engine", None),
            event_bus=getattr(ctx, "bus", None),
            economic_calendar=getattr(ctx, "economic_calendar_client", None),
            quorum_space=getattr(ctx, "quorum_space", None),
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
        from mae_core.market.intelligence.post_mortem import PostMortemReviewer
        ctx.post_mortem_reviewer = PostMortemReviewer(
            thompson_sampler=ctx.thompson_sampler,
            regime_classifier=getattr(ctx, "regime_classifier", None),
        )
    except Exception:
        logger.debug("Market: post_mortem_reviewer failed to construct", exc_info=True)
        ctx.post_mortem_reviewer = None
        failures += 1

    try:
        from mae_core.market.intelligence.world_model import WorldModel
        ctx.world_model = WorldModel()
        if ctx.convergence_alerter is not None:
            ctx.convergence_alerter._world_model = ctx.world_model
    except Exception:
        logger.debug("Market: world_model failed to construct", exc_info=True)
        ctx.world_model = None
        failures += 1

    try:
        from mae_core.market.intelligence.cascade_tracker import CascadeTracker
        ctx.cascade_tracker = CascadeTracker(
            world_model=getattr(ctx, "world_model", None),
            event_bus=getattr(ctx, "bus", None),
        )
    except Exception:
        logger.debug("Market: cascade_tracker failed to construct", exc_info=True)
        ctx.cascade_tracker = None
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

    # --- Hypothesis loop (RSI Layer 2) + Pattern Archaeology ---
    # (extracted to market_intelligence.py to stay under 500-line cap)
    try:
        from mae_core.bootstrap.market_intelligence import _instantiate_intelligence_systems
        failures += _instantiate_intelligence_systems(ctx)
    except Exception:
        logger.debug("Market: intelligence systems instantiation failed", exc_info=True)

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

    # --- Pattern Discovery: MotifDetector, StreamingAnomalyDetector, DriftDetector ---
    try:
        from mae_core.market.intelligence.motif_detector import MotifDetector
        ctx.motif_detector = MotifDetector(raw_store=raw_store)
    except Exception:
        logger.debug("Market: motif_detector failed to construct", exc_info=True)
        ctx.motif_detector = None

    try:
        from mae_core.market.intelligence.streaming_anomaly import StreamingAnomalyDetector
        ctx.streaming_anomaly = StreamingAnomalyDetector(n_features=4, threshold=0.8)
    except Exception:
        logger.debug("Market: streaming_anomaly failed to construct", exc_info=True)
        ctx.streaming_anomaly = None

    try:
        from mae_core.market.intelligence.drift_detector import DriftDetector
        ctx.drift_detector = DriftDetector(delta=0.002)
    except Exception:
        logger.debug("Market: drift_detector failed to construct", exc_info=True)
        ctx.drift_detector = None

    # --- OctopusColony (ecosystem bridge) ---
    try:
        from mae_core.network.octopus_colony import OctopusColony
        ctx.octopus_colony = OctopusColony(
            event_bus=getattr(ctx, "bus", None),
            min_octopuses=3,
            max_octopuses=7,
            world_model=getattr(ctx, "world_model", None),
            signal_bus=getattr(ctx, "signal_bus", None),
            stigmergy=getattr(ctx, "stigmergy", None),
        )
        # Pre-initialize situation tracking so EventBus callbacks
        # registered in step 7 can safely write before inject_market_handlers
        # runs in step 9.  inject_market_handlers will reuse these if present.
        import threading as _th
        ctx.octopus_colony._developing_situations = {}
        ctx.octopus_colony._situations_lock = _th.Lock()

        # Load persisted developing situations (entries older than 2 hours dropped).
        import json as _json
        import time as _time
        _sit_path = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "data" / "market" / "developing_situations.json"
        )
        try:
            if _sit_path.exists():
                _raw = _json.loads(_sit_path.read_text(encoding="utf-8"))
                _now = _time.time()
                _loaded = {
                    k: v for k, v in _raw.items()
                    if (_now - v.get("first_seen", 0)) < 7200
                }
                ctx.octopus_colony._developing_situations = _loaded
                if _loaded:
                    logger.debug(
                        "Market: loaded %d developing situations from disk", len(_loaded)
                    )
        except Exception:
            logger.debug("Market: failed to load developing_situations.json", exc_info=True)

        # Attach save helper to colony so situation_check handler can call it.
        _sit_path_ref = _sit_path

        def _save_developing_situations(colony_obj: Any) -> None:
            """Atomic write of _developing_situations to disk."""
            try:
                lock = getattr(colony_obj, "_situations_lock", None)
                if lock is not None:
                    with lock:
                        snapshot = dict(colony_obj._developing_situations)
                else:
                    snapshot = dict(getattr(colony_obj, "_developing_situations", {}))
                tmp = _sit_path_ref.with_suffix(".tmp")
                tmp.write_text(
                    _json.dumps(snapshot, default=str), encoding="utf-8"
                )
                tmp.replace(_sit_path_ref)
            except Exception:
                logger.debug("Market: failed to save developing_situations", exc_info=True)

        ctx.octopus_colony._save_developing_situations = _save_developing_situations
    except Exception:
        logger.debug("Market: octopus_colony failed to construct", exc_info=True)
        ctx.octopus_colony = None

    # --- ResourceGovernor (self-governing API budget — Law 6 autopoiesis) ---
    try:
        from mae_core.market.resource_governor import ResourceGovernor
        ctx.resource_governor = ResourceGovernor(event_bus=getattr(ctx, "bus", None))
    except Exception:
        logger.debug("Market: resource_governor failed to construct", exc_info=True)
        ctx.resource_governor = None

    # --- InhabitantScheduler (organism-internal wall-clock cadence dispatch) ---
    try:
        from mae_core.scheduling.inhabitant_scheduler import InhabitantScheduler
        ctx.inhabitant_scheduler = InhabitantScheduler(event_bus=getattr(ctx, "bus", None))
    except Exception:
        logger.debug("Market: inhabitant_scheduler failed to construct", exc_info=True)
        ctx.inhabitant_scheduler = None

    # --- GovernanceLogger (append-only governance event audit trail) ---
    try:
        from mae_core.governance.governance_logger import GovernanceLogger
        ctx.governance_logger = GovernanceLogger(event_bus=getattr(ctx, "bus", None))
    except Exception:
        logger.debug("Market: governance_logger failed to construct", exc_info=True)
        ctx.governance_logger = None

    _register_trust_and_gateway(ctx)

    logger.info(
        "Layer 33a - Market systems: %d instantiated (%d failures)", 59 - failures, failures,
    )
