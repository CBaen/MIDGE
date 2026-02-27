"""Bootstrap Layer 33: Market Intelligence Organ.

Creates 29 market systems, registers holons, wires fractal hierarchy,
registers triadic connections (Group 14-18),
wires EventBus channels, and hooks into the step lifecycle.

Biological analogy: Growing a new sensory organ specialized for
financial market signals. Like an immune system evolved to detect
not pathogens but information asymmetries — insider buying clusters,
congressional trade patterns, contract prediction signals. The organ
synthesizes these into convergence alerts that modulate the organism's
behavioral state through the endocrine system.

Graceful degradation: Every instantiation is wrapped in try/except.
If a market system fails to construct (missing API key, Qdrant down),
that ctx attribute is set to None and the rest of the organ continues.
This mirrors how Mae handles all optional systems — advisory, never blocking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger("mae.bootstrap")


def bootstrap_market(ctx: SimpleNamespace) -> None:
    """Wire Layer 33: Market Intelligence organ (29 systems, 55 connections)."""
    _instantiate_market_systems(ctx)
    _register_market_somatic(ctx)
    _register_market_holons(ctx)
    _register_market_fractal(ctx)
    _register_market_connections(ctx)
    _register_market_stem_roles(ctx)
    _register_market_eventbus(ctx)
    _register_market_step_hooks(ctx)
    _wire_sensing_hook(ctx)
    _differentiate_market_agents(ctx)

    # Count active systems
    market_attrs = [
        "sec_edgar_client", "price_fetcher", "house_stock_watcher",
        "job_tracker", "usa_spending_client", "sam_gov_client",
        "cluster_detector", "politician_tracker", "filing_time_analyzer",
        "contract_predictor", "session_sweep_detector",
        "thompson_sampler", "convergence_alerter",
        "velocity_detector", "correlation_tracker", "outcome_tracker",
        "regime_classifier",
        "signal_archive_reader", "lag_correlation_analyzer",
        "thompson_calibrator", "kelly_position_sizer",
        "hypothesis_registry", "hypothesis_generator",
        "hypothesis_validator", "hypothesis_engine",
        "backtest_analyzer", "backtest_scheduler",
        "ta_indicators", "step_timer",
    ]
    active = sum(1 for a in market_attrs if getattr(ctx, a, None) is not None)
    holon_count = len([
        hid for hid in ctx.holon_registry.get_all_ids()
        if hid.startswith("market") or hid in market_attrs
        or hid in ("sec_edgar_client", "price_fetcher", "house_stock_watcher",
                    "job_tracker", "usa_spending_client", "sam_gov_client",
                    "cluster_detector", "politician_tracker", "filing_time_analyzer",
                    "contract_predictor", "session_sweep_detector",
                    "thompson_sampler", "convergence_alerter",
                    "velocity_detector", "correlation_tracker", "outcome_tracker",
                    "regime_classifier",
                    "signal_archive_reader", "lag_correlation_analyzer",
                    "thompson_calibrator", "kelly_position_sizer",
                    "hypothesis_registry", "hypothesis_generator",
                    "hypothesis_validator", "hypothesis_engine",
                    "backtest_analyzer", "backtest_scheduler",
                    "ta_indicators", "step_timer")
    ])

    logger.info(
        "Layer 33  - Market Intelligence organ complete: %d systems, %d holons, 55 connections",
        active, holon_count,
    )
    logger.info(
        "            NOTE: OutcomeTracker active (Bayesian feedback from real market outcomes)."
    )
    logger.info(
        "            NOTE: MarketDataProvider registered with ApiGateway. "
        "All 6 API clients route through provider when available."
    )


# =========================================================================
# Layer 33a: Instantiate market systems
# =========================================================================

def _instantiate_market_systems(ctx: SimpleNamespace) -> None:
    """Create all 28 market system objects on ctx."""
    import os
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

    try:
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
        ctx.convergence_alerter = ConvergenceAlerter(
            min_domains=3,
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
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
    # (placed after intelligence layer so thompson_sampler is already set)

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
        ctx.hypothesis_validator = HypothesisValidator()
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
            )
            if (ctx.hypothesis_registry is not None
                and ctx.hypothesis_generator is not None
                and ctx.hypothesis_validator is not None)
            else None
        )
    except Exception:
        logger.debug("Market: hypothesis_engine failed to construct", exc_info=True)
        ctx.hypothesis_engine = None

    # --- StepTimer (performance metabolism monitoring) ---
    try:
        from mae_core.market.step_timer import StepTimer
        ctx.step_timer = StepTimer()
    except Exception:
        ctx.step_timer = None

    # --- Trust registration with BoundaryMembrane ---
    market_sources = [
        ("sec_edgar", 0.90), ("yfinance", 0.75), ("alpha_vantage", 0.80),
        ("rapidapi", 0.65), ("usa_spending", 0.85), ("sam_gov", 0.80),
        ("senate_free", 0.80), ("apewisdom", 0.45), ("finra_short", 0.85),
        ("sec_efts", 0.90), ("finnhub", 0.75), ("fred_macro", 0.95),
        ("session_sweep", 0.60),
    ]
    if getattr(ctx, "boundary_membrane", None) is not None:
        for source_name, trust in market_sources:
            try:
                if hasattr(ctx.boundary_membrane, "register_source"):
                    ctx.boundary_membrane.register_source(source_name, trust=trust)
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
        "Layer 33a - Market systems instantiated: %d systems (construction failures: %d)",
        26 - failures, failures,
    )
    logger.info(
        "            Operational dependencies: Qdrant, RAPIDAPI_KEY, ALPHA_VANTAGE_KEY, "
        "SAM_GOV_API_KEY — failures deferred to first use"
    )


# =========================================================================
# Layer 33b: Register with SomaticMap
# =========================================================================

def _register_market_somatic(ctx: SimpleNamespace) -> None:
    """Register market systems with SomaticMap (must precede connections)."""
    somatic = getattr(ctx, "somatic_map", None)
    if somatic is None or not hasattr(somatic, "register_system"):
        return

    market_systems = {
        "sec_edgar_client": ("SECEdgarClient", []),
        "price_fetcher": ("PriceFetcher", []),
        "house_stock_watcher": ("HouseStockWatcher", []),
        "job_tracker": ("JobTracker", []),
        "usa_spending_client": ("USASpendingClient", []),
        "sam_gov_client": ("SAMGovClient", []),
        "cluster_detector": ("ClusterDetector", []),
        "politician_tracker": ("PoliticianTracker", ["sec_edgar_client", "usa_spending_client"]),
        "filing_time_analyzer": ("FilingTimeAnalyzer", []),
        "contract_predictor": ("ContractPredictor", ["job_tracker", "sam_gov_client"]),
        "session_sweep_detector": ("SessionSweepDetector", ["price_fetcher"]),
        "thompson_sampler": ("ThompsonSampler", []),
        "convergence_alerter": ("ConvergenceAlerter", ["thompson_sampler", "velocity_detector", "regime_classifier"]),
        "velocity_detector": ("VelocityDetector", []),
        "correlation_tracker": ("CorrelationTracker", []),
        "outcome_tracker": ("OutcomeTracker", ["price_fetcher", "thompson_sampler"]),
        "regime_classifier": ("RegimeClassifier", ["price_fetcher"]),
        "signal_archive_reader": ("SignalArchiveReader", []),
        "lag_correlation_analyzer": ("LagCorrelationAnalyzer", ["signal_archive_reader"]),
        "thompson_calibrator": ("ThompsonCalibrator", ["thompson_sampler"]),
        "kelly_position_sizer": ("KellyPositionSizer", ["thompson_sampler"]),
        "hypothesis_registry": ("HypothesisRegistry", []),
        "hypothesis_generator": ("HypothesisGenerator", ["hypothesis_registry"]),
        "hypothesis_validator": ("HypothesisValidator", []),
        "hypothesis_engine": ("HypothesisEngine", ["hypothesis_registry", "hypothesis_generator", "hypothesis_validator"]),
        "backtest_analyzer": ("BacktestAnalyzer", ["hypothesis_registry"]),
        "backtest_scheduler": ("BacktestScheduler", ["backtest_analyzer"]),
        "ta_indicators": ("TAIndicators", ["price_fetcher"]),
        "step_timer": ("StepTimer", []),
    }

    for sys_id, (desc, deps) in market_systems.items():
        if getattr(ctx, sys_id, None) is not None:
            try:
                somatic.register_system(
                    system_id=sys_id,
                    description=desc,
                    depends_on=deps,
                )
            except Exception:
                logger.debug("Could not register %s with SomaticMap", sys_id)


# =========================================================================
# Layer 33b: Register holons
# =========================================================================

def _register_market_holons(ctx: SimpleNamespace) -> None:
    """Register all market systems as holons with HolonProxy injection."""
    registry = ctx.holon_registry

    market_systems = [
        "sec_edgar_client", "price_fetcher", "house_stock_watcher",
        "job_tracker", "usa_spending_client", "sam_gov_client",
        "cluster_detector", "politician_tracker", "filing_time_analyzer",
        "contract_predictor", "session_sweep_detector",
        "thompson_sampler", "convergence_alerter",
        "velocity_detector", "correlation_tracker", "outcome_tracker",
        "regime_classifier",
        "signal_archive_reader", "lag_correlation_analyzer",
        "thompson_calibrator", "kelly_position_sizer",
        "hypothesis_registry", "hypothesis_generator",
        "hypothesis_validator", "hypothesis_engine",
        "backtest_analyzer", "backtest_scheduler",
        "ta_indicators", "step_timer",
    ]

    registered = 0
    for sys_id in market_systems:
        system_obj = getattr(ctx, sys_id, None)
        if system_obj is None:
            continue

        try:
            registry.register(sys_id, holon_type="system", parent_id="mae")
            proxy = registry.get_proxy(sys_id)
            if proxy is not None:
                proxy.set_system_ref(system_obj)
            registered += 1
        except Exception:
            logger.debug("Could not register holon: %s", sys_id)

    logger.info("Layer 33b - Market holons registered: %d holons", registered)


# =========================================================================
# Layer 33c: Fractal registration
# =========================================================================

def _register_market_fractal(ctx: SimpleNamespace) -> None:
    """Register fractal hierarchy — three K3 subsystems under one organ."""
    fg = ctx.fractal_generator

    # Subsystem triads
    fg.generate_triad(
        name="market-sensing",
        holon_type="subsystem",
        children_ids=["sec_edgar_client", "price_fetcher", "job_tracker"],
        parent_id="market-intelligence-system",
    )
    fg.generate_triad(
        name="market-edge",
        holon_type="subsystem",
        children_ids=["cluster_detector", "politician_tracker", "contract_predictor"],
        parent_id="market-intelligence-system",
    )
    fg.generate_triad(
        name="market-learning",
        holon_type="subsystem",
        children_ids=["thompson_sampler", "convergence_alerter", "velocity_detector"],
        parent_id="market-intelligence-system",
    )

    # Organ — completes the bare dyad in organ-cluster-cognitive
    fg.generate_triad(
        name="market-intelligence-system",
        holon_type="organ",
        children_ids=["market-sensing", "market-edge", "market-learning"],
        parent_id="organ-cluster-cognitive",
    )

    # Hypothesis K3 subsystem (RSI Layer 2: generator + validator + engine)
    fg.generate_triad(
        name="market-hypothesis",
        holon_type="subsystem",
        children_ids=["hypothesis_generator", "hypothesis_validator", "hypothesis_engine"],
        parent_id="market-intelligence-system",
    )

    # Register remaining systems individually (advisory non-triadic)
    extras = [
        "house_stock_watcher", "filing_time_analyzer",
        "session_sweep_detector",
        "usa_spending_client", "sam_gov_client", "correlation_tracker",
        "outcome_tracker", "regime_classifier",
        "signal_archive_reader", "lag_correlation_analyzer",
        "thompson_calibrator", "kelly_position_sizer",
        "hypothesis_registry", "backtest_analyzer", "backtest_scheduler",
        "ta_indicators", "step_timer",
    ]
    for sys_id in extras:
        if ctx.holon_registry.get_entry(sys_id) is not None:
            try:
                ctx.holon_registry.reparent(sys_id, "market-intelligence-system")
            except Exception:
                logger.debug("Could not reparent %s under market-intelligence-system", sys_id)

    # Verify repair
    children = ctx.holon_registry.get_children("organ-cluster-cognitive")
    logger.info(
        "Layer 33c - Market fractal: organ-cluster-cognitive now has %d children "
        "(K3 requires 3)",
        len(children),
    )


# =========================================================================
# Layer 33d: Triadic connections (Group 14)
# =========================================================================

def _register_market_connections(ctx: SimpleNamespace) -> None:
    """Register 51 triadic connections for market systems (Group 14-17)."""
    from mae_core.backbone.connection_registry import (
        ConnectionType,
        ConnectionCriticality,
    )

    reg = ctx.connection_registry.register_connection
    dr = ConnectionType.DIRECT_REFERENCE
    eb = ConnectionType.EVENTBUS_PUBSUB
    cb = ConnectionType.CALLBACK_REGISTRATION
    sh = ConnectionType.STEP_HOOK

    # --- Market Sensing K3 ---
    reg("sec_edgar_client", "price_fetcher", dr,
        witnesses=["job_tracker", "auditor"],
        description="SEC signals correlate with price movement")
    reg("price_fetcher", "job_tracker", dr,
        witnesses=["sec_edgar_client", "auditor"],
        description="Price data validates hiring signal predictions")
    reg("job_tracker", "sec_edgar_client", dr,
        witnesses=["price_fetcher", "auditor"],
        description="Hiring data cross-references insider filings")

    # --- Market Edge K3 ---
    reg("cluster_detector", "politician_tracker", dr,
        witnesses=["contract_predictor", "auditor"],
        description="Insider clusters may correlate with political trades")
    reg("politician_tracker", "contract_predictor", dr,
        witnesses=["cluster_detector", "auditor"],
        description="Political trades may precede contract awards")
    reg("contract_predictor", "cluster_detector", dr,
        witnesses=["politician_tracker", "auditor"],
        description="Contract predictions validated by insider clusters")

    # --- Market Learning K3 ---
    reg("thompson_sampler", "convergence_alerter", dr,
        witnesses=["velocity_detector", "auditor"],
        description="Thompson reliability weights convergence decisions")
    reg("convergence_alerter", "velocity_detector", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alerts carry velocity context")
    reg("velocity_detector", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Velocity anomalies feed signal exploration priority")

    # --- Cross-subsystem: EventBus publish ---
    reg("cluster_detector", "event_bus", eb,
        channel="market.edge.cluster_detected",
        witnesses=["threat_detector", "auditor"],
        description="Insider cluster signal published")
    reg("politician_tracker", "event_bus", eb,
        channel="market.edge.politician_trade",
        witnesses=["threat_detector", "auditor"],
        description="Political trade correlation published")
    reg("contract_predictor", "event_bus", eb,
        channel="market.edge.contract_predicted",
        witnesses=["threat_detector", "auditor"],
        description="Contract prediction published")
    reg("filing_time_analyzer", "event_bus", eb,
        channel="market.edge.filing_anomaly",
        witnesses=["threat_detector", "auditor"],
        description="Filing time anomaly published")
    reg("session_sweep_detector", "event_bus", eb,
        channel="market.edge.session_sweep",
        witnesses=["threat_detector", "auditor"],
        description="Session sweep signal published")

    # --- Cross-subsystem: EventBus subscribe ---
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.cluster_detected",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to cluster signals")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.politician_trade",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to political trade signals")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.contract_predicted",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to contract predictions")
    reg("convergence_alerter", "event_bus", cb,
        channel="market.edge.session_sweep",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to session sweep signals")

    # --- Integration connections ---
    reg("convergence_alerter", "thompson_sampler", dr,
        witnesses=["velocity_detector", "knowledge_base"],
        description="Convergence queries Thompson for signal reliability")
    reg("velocity_detector", "convergence_alerter", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Velocity enriches convergence signal context")
    reg("convergence_alerter", "knowledge_base", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence stores discoveries to knowledge base")
    reg("sec_edgar_client", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="SEC client validated through boundary membrane")
    reg("price_fetcher", "boundary_membrane", dr,
        witnesses=["input_validator", "threat_detector"],
        criticality=ConnectionCriticality.IMPORTANT,
        description="Price fetcher validated through boundary membrane")
    reg("session_sweep_detector", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Session sweep reads intraday price data via yfinance")

    # --- TA Indicators connections ---
    reg("ta_indicators", "price_fetcher", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="TA indicators compute RSI/MACD/Bollinger from daily OHLCV")
    reg("convergence_alerter", "ta_indicators", dr,
        channel="market.edge.ta_indicators",
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence subscribes to TA indicator signals")
    reg("ta_indicators", "thompson_sampler", dr,
        witnesses=["convergence_alerter", "auditor"],
        description="Thompson learns TA indicator reliability from outcomes")

    # --- Step hook + periodic ---
    reg("convergence_alerter", "model", sh,
        witnesses=["auditor", "connection_registry"],
        description="Convergence alerter runs each step")
    reg("thompson_sampler", "event_bus", eb,
        channel="market.intel.thompson_stats",
        witnesses=["auditor", "connection_registry"],
        description="Thompson stats published periodically")

    # --- Group 15: Phase 4 — Signal Archive, Lag-Correlation, Calibration, Kelly ---
    reg("signal_archive_reader", "lag_correlation_analyzer", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Archive reader provides historical series to lag analyzer")
    reg("lag_correlation_analyzer", "event_bus", eb,
        channel="market.intel.lag_finding",
        witnesses=["correlation_tracker", "auditor"],
        description="Lag findings published to EventBus")
    reg("thompson_calibrator", "signal_archive_reader", dr,
        witnesses=["lag_correlation_analyzer", "auditor"],
        description="Calibrator reads archives to validate distributions")

    reg("thompson_calibrator", "thompson_sampler", dr,
        witnesses=["signal_archive_reader", "auditor"],
        description="Calibrator seeds and diagnoses Thompson distributions")
    reg("thompson_calibrator", "event_bus", eb,
        channel="market.intel.thompson_stats",
        witnesses=["convergence_alerter", "auditor"],
        description="Calibration report published to EventBus")
    reg("convergence_alerter", "thompson_calibrator", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alerter uses calibration for confidence")

    reg("kelly_position_sizer", "thompson_sampler", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Kelly sizer queries Thompson mean for p parameter")
    reg("convergence_alerter", "kelly_position_sizer", dr,
        witnesses=["thompson_sampler", "auditor"],
        description="Convergence alert triggers Kelly sizing recommendation")
    reg("kelly_position_sizer", "event_bus", eb,
        channel="market.intel.kelly_sizing",
        witnesses=["convergence_alerter", "auditor"],
        description="Kelly sizing recommendation published on convergence")

    reg("lag_correlation_analyzer", "correlation_tracker", dr,
        witnesses=["signal_archive_reader", "auditor"],
        description="Lag findings feed leading-indicator pairs to tracker")
    reg("lag_correlation_analyzer", "convergence_alerter", dr,
        witnesses=["correlation_tracker", "auditor"],
        description="Significant lag pairs update convergence domain weights")
    reg("signal_archive_reader", "outcome_tracker", dr,
        witnesses=["thompson_calibrator", "auditor"],
        description="Archive reader provides historical signals to outcome tracker")

    # --- Group 16: Hypothesis loop (RSI Layer 2) ---
    reg("hypothesis_generator", "hypothesis_registry", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Generator registers new hypotheses in the registry")
    reg("hypothesis_validator", "hypothesis_registry", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Validator reads hypotheses for adversarial testing")
    reg("hypothesis_engine", "hypothesis_generator", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Engine drives generation on cadence")
    reg("hypothesis_engine", "hypothesis_validator", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine drives validation on cadence")
    reg("hypothesis_engine", "hypothesis_registry", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine manages lifecycle transitions (promote/retire)")
    reg("hypothesis_generator", "lag_correlation_analyzer", dr,
        witnesses=["hypothesis_registry", "auditor"],
        description="Generator reads lag findings to create hypotheses")
    reg("hypothesis_engine", "thompson_sampler", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Engine registers Thompson key on hypothesis promotion")
    reg("hypothesis_engine", "event_bus", eb,
        channel="market.hypothesis.discovered",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine publishes hypothesis discoveries")
    reg("hypothesis_engine", "event_bus", eb,
        channel="market.hypothesis.promoted",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine publishes hypothesis promotions")
    reg("hypothesis_engine", "event_bus", cb,
        channel="market.sensing.signal_ingested",
        witnesses=["hypothesis_registry", "auditor"],
        description="Engine subscribes to ingested signals for trigger matching")

    # --- Group 17: Backtest bridge (Bridge 1 — RSI Layer 2 self-reading) ---
    reg("backtest_analyzer", "hypothesis_registry", dr,
        witnesses=["hypothesis_validator", "auditor"],
        description="Backtest analyzer registers derived hypotheses in registry")
    reg("backtest_analyzer", "hypothesis_validator", dr,
        witnesses=["hypothesis_registry", "auditor"],
        description="Backtest hypotheses flow to validator for DSR evaluation")
    reg("hypothesis_engine", "backtest_analyzer", dr,
        witnesses=["hypothesis_generator", "auditor"],
        description="Engine drives backtest analysis on generation cadence")

    # --- Bridge 3: Autonomous backtest scheduling ---
    reg("backtest_scheduler", "backtest_analyzer", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="Scheduler triggers backtest rerun and refreshes analyzer")
    reg("backtest_scheduler", "event_bus", eb,
        channel="market.intel.backtest_refreshed",
        witnesses=["hypothesis_engine", "auditor"],
        description="Scheduler publishes refresh completion event")

    # --- Efficiency: Step timer (performance metabolism) ---
    reg("step_timer", "convergence_alerter", dr,
        witnesses=["hypothesis_engine", "auditor"],
        description="StepTimer wraps convergence check to measure latency")
    reg("step_timer", "event_bus", eb,
        channel="market.intel.step_timing",
        witnesses=["convergence_alerter", "auditor"],
        description="StepTimer publishes timing statistics for health monitoring")

    logger.info("Layer 33d - Market connections: 55 triadic connections registered (Group 14-18)")


# =========================================================================
# Layer 33e: Stem cell role registration
# =========================================================================

def _register_market_stem_roles(ctx: SimpleNamespace) -> None:
    """Verify market stem cell roles are available."""
    from mae_core.agents.stem_cell import ROLE_PROFILES

    market_roles = [
        "SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST",
        "HYPOTHESIS_EXPLORER", "HYPOTHESIS_VALIDATOR",
    ]
    for role in market_roles:
        if role not in ROLE_PROFILES:
            logger.warning("Market stem cell role %s not found in ROLE_PROFILES", role)

    logger.info("Layer 33e - Market stem cell roles verified: %s", ", ".join(market_roles))


# =========================================================================
# Layer 33f: EventBus wiring + endocrine coupling
# =========================================================================

def _register_market_eventbus(ctx: SimpleNamespace) -> None:
    """Wire EventBus channels — endocrine coupling for convergence alerts."""
    from mae_core.market.channels import CH_CONVERGENCE

    def _on_market_convergence(channel, data):
        msg = json.loads(data) if isinstance(data, str) else data
        strength = msg.get("strength", 0.0)
        direction = msg.get("direction", "neutral")

        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return

        try:
            from mae_core.coordination.endocrine_system import HormoneType

            if direction == "bullish" and strength > 0.7:
                endocrine.release_hormone(
                    HormoneType.DOPAMINE,
                    min(0.4, strength * 0.4),
                    "market_opportunity",
                )
            elif direction == "bearish" and strength > 0.7:
                endocrine.release_hormone(
                    HormoneType.ADRENALINE,
                    min(0.5, strength * 0.5),
                    "market_threat",
                )
        except Exception:
            logger.debug("Endocrine coupling failed for convergence alert", exc_info=True)

    ctx.bus.register_callback(CH_CONVERGENCE, _on_market_convergence)

    # --- Hypothesis lifecycle endocrine coupling ---
    from mae_core.market.channels import CH_HYPOTHESIS_PROMOTED, CH_HYPOTHESIS_RETIRED

    def _on_hypothesis_promoted(channel, data):
        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return
        try:
            from mae_core.coordination.endocrine_system import HormoneType
            endocrine.release_hormone(HormoneType.DOPAMINE, 0.3, "hypothesis_promoted")
        except Exception:
            pass

    def _on_hypothesis_retired(channel, data):
        msg = json.loads(data) if isinstance(data, str) else data
        if not msg.get("was_active", False):
            return  # Only cortisol if it was actively being used
        endocrine = getattr(ctx, "endocrine", None)
        if endocrine is None:
            return
        try:
            from mae_core.coordination.endocrine_system import HormoneType
            endocrine.release_hormone(HormoneType.CORTISOL, 0.15, "hypothesis_retired_unexpectedly")
        except Exception:
            pass

    ctx.bus.register_callback(CH_HYPOTHESIS_PROMOTED, _on_hypothesis_promoted)
    ctx.bus.register_callback(CH_HYPOTHESIS_RETIRED, _on_hypothesis_retired)

    # --- Hypothesis engine signal ingestion subscription ---
    engine = getattr(ctx, "hypothesis_engine", None)
    if engine is not None:
        from mae_core.market.channels import CH_SIGNAL_INGESTED
        ctx.bus.register_callback(CH_SIGNAL_INGESTED, engine.on_signal_ingested)

    logger.info("Layer 33f - Market EventBus: convergence + hypothesis -> endocrine coupling wired")


# =========================================================================
# Layer 33g: Step hooks with deduplication
# =========================================================================

def _register_market_step_hooks(ctx: SimpleNamespace) -> None:
    """Register step hooks with cadence and deduplication."""
    from mae_core.market.channels import (
        CH_CONVERGENCE, CH_THOMPSON_STATS, CH_VELOCITY_ANOMALY,
    )

    _step_counter = [0]
    _last_convergence_state = [None]  # {"direction": str, "strength": float}
    ctx._cached_alerts = [None]  # Shared: written by _market_sense_hook, read by advisory bridge

    def _get_regime():
        """Get current market regime (cached daily, essentially free)."""
        rc = getattr(ctx, "regime_classifier", None)
        if rc is not None:
            try:
                return rc.classify()
            except Exception:
                pass
        return "default"

    _timer = getattr(ctx, "step_timer", None)

    def _market_sense_hook():
        _step_counter[0] += 1
        step = _step_counter[0]

        # Every step: check convergence (lightweight, pure in-memory)
        alerter = getattr(ctx, "convergence_alerter", None)
        if alerter is not None:
            try:
                if _timer is not None:
                    with _timer.track("convergence_check"):
                        alerts = alerter.check_convergence()
                else:
                    alerts = alerter.check_convergence()
                _cached_alerts[0] = alerts  # Cache for advisory bridge (avoid duplicate call)
                for alert in alerts:
                    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else {}
                    last = _last_convergence_state[0]
                    direction = alert_dict.get("direction", "neutral")
                    strength = alert_dict.get("strength", 0.0)

                    is_new_direction = (last is None or last["direction"] != direction)
                    is_material_change = (
                        last is not None
                        and abs(last["strength"] - strength) > 0.1
                    )

                    if is_new_direction or is_material_change:
                        ctx.bus.publish(CH_CONVERGENCE, alert_dict)
                        _last_convergence_state[0] = {
                            "direction": direction,
                            "strength": strength,
                        }
            except Exception:
                logger.debug("Convergence alerter step failed", exc_info=True)

        # Every 10 steps: Thompson stats (regime-aware)
        if step % 10 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    if _timer is not None:
                        with _timer.track("thompson_stats"):
                            regime = _get_regime()
                            stats = sampler.get_stats(regime)
                    else:
                        regime = _get_regime()
                        stats = sampler.get_stats(regime)
                    stats["regime"] = regime
                    ctx.bus.publish(CH_THOMPSON_STATS, stats)
                except Exception:
                    logger.debug("Thompson sampler stats step failed", exc_info=True)

        # Every 50 steps: velocity anomaly scan
        if step % 50 == 0:
            vd = getattr(ctx, "velocity_detector", None)
            if vd is not None:
                try:
                    if _timer is not None:
                        with _timer.track("velocity_scan"):
                            anomalies = vd.detect_velocity_anomalies()
                    else:
                        anomalies = vd.detect_velocity_anomalies()
                    if anomalies:
                        ctx.bus.publish(CH_VELOCITY_ANOMALY,
                                        {"anomalies": len(anomalies)})
                except Exception:
                    logger.debug("Velocity detector step failed", exc_info=True)

        # Every 100 steps: Bayesian forgetting (decay old evidence)
        if step % 100 == 0:
            sampler = getattr(ctx, "thompson_sampler", None)
            if sampler is not None:
                try:
                    sampler.apply_forgetting(decay_factor=0.99)
                except Exception:
                    logger.debug("Thompson forgetting step failed", exc_info=True)

        # Every 500 steps: lag-correlation analysis
        if step % 500 == 0:
            lag = getattr(ctx, "lag_correlation_analyzer", None)
            if lag is not None:
                try:
                    if _timer is not None:
                        with _timer.track("lag_correlation"):
                            findings = lag.analyze(lookback_days=90)
                    else:
                        findings = lag.analyze(lookback_days=90)
                    if findings and hasattr(ctx, "bus"):
                        ctx.bus.publish("market.intel.lag_finding", {
                            "count": len(findings),
                            "top": [
                                {"a": f.source_a, "b": f.source_b,
                                 "lag": f.lag_days, "r": f.correlation}
                                for f in findings[:3]
                            ],
                        })
                except Exception:
                    logger.debug("Lag correlation step failed", exc_info=True)

        # Every 1000 steps: Thompson calibration diagnostic
        if step % 1000 == 0:
            calibrator = getattr(ctx, "thompson_calibrator", None)
            if calibrator is not None:
                try:
                    if _timer is not None:
                        with _timer.track("thompson_calibration"):
                            calibrator.calibrate()
                    else:
                        calibrator.calibrate()
                except Exception:
                    logger.debug("Thompson calibration step failed", exc_info=True)

        # Every 5000 steps: backtest scheduler staleness check
        if step % 5000 == 0:
            scheduler = getattr(ctx, "backtest_scheduler", None)
            if scheduler is not None:
                try:
                    scheduler.check_and_schedule()
                except Exception:
                    logger.debug("Backtest scheduler check failed", exc_info=True)

        # Every step: hypothesis engine (manages its own cadence internally)
        hyp_engine = getattr(ctx, "hypothesis_engine", None)
        if hyp_engine is not None:
            try:
                if _timer is not None:
                    with _timer.track("hypothesis_engine"):
                        hyp_engine.step()
                else:
                    hyp_engine.step()
            except Exception:
                logger.debug("Hypothesis engine step failed", exc_info=True)

        # Kelly sizing: fires on per-ticker convergence alerts
        if step % 50 == 0:
            sizer = getattr(ctx, "kelly_position_sizer", None)
            alerter = getattr(ctx, "convergence_alerter", None)
            if sizer is not None and alerter is not None:
                try:
                    alerts = alerter.check_ticker_convergence(min_domains=2)
                    for alert in alerts:
                        # Extract symbol from alert signals' metadata
                        symbols = set()
                        for sig in alert.signals:
                            sym = sig.metadata.get("symbol", "")
                            if sym:
                                symbols.add(sym)
                        source = alert.signals[0].source if alert.signals else "unknown"
                        for symbol in symbols:
                            rec = sizer.recommend(source, symbol)
                            if rec.kelly_capped > 0 and hasattr(ctx, "bus"):
                                ctx.bus.publish("market.intel.kelly_sizing", {
                                    "symbol": symbol,
                                    "source": source,
                                    "kelly_capped": rec.kelly_capped,
                                    "p_win": rec.p_win,
                                    "confidence": rec.confidence_in_sizing,
                                })
                except Exception:
                    logger.debug("Kelly sizing step failed", exc_info=True)

    ctx.model.add_step_hook(_market_sense_hook)
    logger.info(
        "Layer 33g - Market step hooks: 1 sense hook registered "
        "(cadence: convergence/1, stats/10, velocity/50, forgetting/100, "
        "lag/500, calibration/1000, backtest/5000)"
    )


# =========================================================================
# Layer 33h: Market Sensing Hook (data → convergence pipeline)
# =========================================================================

def _wire_sensing_hook(ctx: SimpleNamespace) -> None:
    """Wire the MarketSensingHook into the step lifecycle.

    This is the critical missing link: data never flowed into the
    convergence_alerter during agent runs. Everything downstream was
    already wired (endocrine coupling, body state, advisory). This
    function connects the sensory organs to the nervous system.

    Biological analogy: Wiring the optic nerve — the eyes existed,
    the visual cortex existed, but signals never traveled between them.
    """
    from mae_core.market.sensing_hook import MarketSensingHook
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
    from mae_core.market.channels import CH_CONVERGENCE

    # --- Tiered ConvergenceAlerters (tactical/strategic/thematic) ---
    tiered_alerters = {}
    try:
        tiered_alerters["tactical"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=48,
        )
        tiered_alerters["strategic"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=21 * 24,
        )
        tiered_alerters["thematic"] = ConvergenceAlerter(
            min_domains=2, convergence_window_hours=90 * 24,
        )
    except Exception:
        logger.debug("Tiered alerter construction failed", exc_info=True)

    # --- OutcomeCollector (signal → prediction → Thompson feedback) ---
    outcome_collector = None
    try:
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector
        outcome_collector = OutcomeCollector(
            price_fetcher=getattr(ctx, "price_fetcher", None),
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            regime_classifier=getattr(ctx, "regime_classifier", None),
        )
    except Exception:
        logger.debug("OutcomeCollector construction failed", exc_info=True)

    # --- SignalMemory (Qdrant persistence) ---
    memory = None
    try:
        from mae_core.market.memory import SignalMemory
        qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
        memory = SignalMemory(qdrant_url=qdrant_url)
    except Exception:
        logger.debug("SignalMemory construction failed", exc_info=True)

    # --- 8-K text sentiment via Ollama ---
    form8k_sentiment = None
    try:
        from mae_core.market.edge.form8k_sentiment import Form8KSentimentAnalyzer
        form8k_sentiment = Form8KSentimentAnalyzer()
    except Exception:
        logger.debug("Form8KSentimentAnalyzer construction failed", exc_info=True)

    # --- Instantiate the sensing hook ---
    try:
        hook = MarketSensingHook(
            sec_client=getattr(ctx, "sec_edgar_client", None),
            price_fetcher=getattr(ctx, "price_fetcher", None),
            congress_client=getattr(ctx, "house_stock_watcher", None),
            senate_client=getattr(ctx, "senate_stock_watcher", None),
            job_tracker=getattr(ctx, "job_tracker", None),
            usa_spending=getattr(ctx, "usa_spending_client", None),
            sam_gov=getattr(ctx, "sam_gov_client", None),
            apewisdom=getattr(ctx, "apewisdom_client", None),
            finra_client=getattr(ctx, "finra_client", None),
            sec_efts=getattr(ctx, "sec_efts_client", None),
            finnhub=getattr(ctx, "finnhub_client", None),
            fred=getattr(ctx, "fred_client", None),
            convergence_alerter=getattr(ctx, "convergence_alerter", None),
            velocity_detector=getattr(ctx, "velocity_detector", None),
            filing_analyzer=getattr(ctx, "filing_time_analyzer", None),
            form8k_sentiment=form8k_sentiment,
            session_sweep_detector=getattr(ctx, "session_sweep_detector", None),
            ta_indicators=getattr(ctx, "ta_indicators", None),
            outcome_collector=outcome_collector,
            memory=memory,
            tiered_alerters=tiered_alerters,
        )
    except Exception:
        logger.warning("MarketSensingHook construction failed — agents will not sense market data", exc_info=True)
        return

    # --- Market advisory dict (Channel B: supplements endocrine Channel A) ---
    # Separate from _latest_advisory which PatternCortex overwrites every step.
    # Market-role agents read this in their decision cascade.
    ctx._market_advisory = {"alert": None, "updated_step": 0, "active_hypotheses": 0}

    # Wire convergence alerts into the advisory dict
    _sensing_step_counter = [0]
    original_step = hook.step

    def _sensing_step_with_advisory():
        """Wrap the sensing hook step to also update the market advisory."""
        _sensing_step_counter[0] += 1
        original_step()

        # Reuse cached convergence alerts (written by _market_sense_hook)
        alerts = _cached_alerts[0] or []
        if alerts:
            try:
                strongest = max(alerts, key=lambda a: a.strength)
                ctx._market_advisory["alert"] = (
                    strongest.to_dict() if hasattr(strongest, "to_dict")
                    else {"direction": strongest.direction, "strength": strongest.strength}
                )
                ctx._market_advisory["updated_step"] = _sensing_step_counter[0]
            except Exception:
                logger.debug("Advisory bridge failed", exc_info=True)

    # Register the wrapped step hook
    ctx.model.add_step_hook(_sensing_step_with_advisory)

    # Inject EventBus for signal bridge (Phase 1 of hypothesis loop)
    hook._bus = ctx.bus

    # Store hook reference on ctx for monitoring
    ctx._market_sensing_hook = hook

    logger.info(
        "Layer 33h - Market sensing hook wired: "
        "async fetch (cadence=50), outcome tracking (cadence=200), "
        "tiered alerters (%d), advisory bridge active",
        len(tiered_alerters),
    )


# =========================================================================
# Layer 33i: Agent differentiation into market roles
# =========================================================================

def _differentiate_market_agents(ctx: SimpleNamespace) -> None:
    """Differentiate agents into triadic groups per Law 2.

    Progressive differentiation by agent count:
      6 agents: K3 general (1-3 STEM) + K3 market (4-6)
      9 agents: K3 general + K3 market + K3 intelligence (7-9)

    Law 5: same genome, specialized epigenome.
    Law 2: K3 is the atom of all structure — groups of 3, always.
    """
    from mae_core.agents.stem_cell import redifferentiate

    agents = getattr(ctx, "agents", [])
    if len(agents) < 6:
        logger.info(
            "Layer 33i - Need at least 6 agents for market differentiation "
            "(K3 general + K3 market per Law 2), have %d — skipping",
            len(agents),
        )
        return

    registry = getattr(ctx, "stem_cell_registry", None)
    market_advisory = getattr(ctx, "_market_advisory", None)

    # --- K3 Market: agents[-6], [-5], [-4] (or [-3], [-2], [-1] when only 6) ---
    # With 6 agents: indices 3, 4, 5. With 9: indices 3, 4, 5.
    market_roles = [
        ("SEC_WATCHER", 3),
        ("CONTRACT_TRACKER", 4),
        ("MARKET_ANALYST", 5),
    ]

    differentiated = 0
    for role, idx in market_roles:
        if idx >= len(agents):
            break
        try:
            agent = agents[idx]
            redifferentiate(agent, role, registry=registry, step=0)
            if market_advisory is not None:
                agent._market_advisory_ref = market_advisory
            # Attach live market system refs for market_awareness.py functions
            agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
            agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
            differentiated += 1
            logger.info(
                "Market differentiation: agent %s → %s",
                getattr(agent, "unique_id", id(agent)), role,
            )
        except Exception:
            logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    # --- K3 Intelligence: agents 6, 7, 8 (only when 9+ agents) ---
    if len(agents) >= 9:
        intel_roles = [
            ("EXPLORER", 6),
            ("LEARNER", 7),
            ("LLM_SPECIALIST", 8),
        ]
        for role, idx in intel_roles:
            try:
                agent = agents[idx]
                redifferentiate(agent, role, registry=registry, step=0)
                if market_advisory is not None:
                    agent._market_advisory_ref = market_advisory
                agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
                agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
                differentiated += 1
                logger.info(
                    "Intelligence differentiation: agent %s → %s",
                    getattr(agent, "unique_id", id(agent)), role,
                )
            except Exception:
                logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    # --- K3 Hypothesis: agents 9, 10, 11 (only when 12+ agents) ---
    if len(agents) >= 12:
        hyp_roles = [
            ("HYPOTHESIS_EXPLORER", 9),
            ("HYPOTHESIS_VALIDATOR", 10),
            ("MARKET_ANALYST", 11),  # Third member of the hypothesis triad
        ]
        for role, idx in hyp_roles:
            try:
                agent = agents[idx]
                redifferentiate(agent, role, registry=registry, step=0)
                if market_advisory is not None:
                    agent._market_advisory_ref = market_advisory
                agent._convergence_alerter_ref = getattr(ctx, "convergence_alerter", None)
                agent._regime_classifier_ref = getattr(ctx, "regime_classifier", None)
                # Attach hypothesis engine/registry refs for direct interaction
                agent._hypothesis_engine_ref = getattr(ctx, "hypothesis_engine", None)
                agent._hypothesis_registry_ref = getattr(ctx, "hypothesis_registry", None)
                differentiated += 1
                logger.info(
                    "Hypothesis differentiation: agent %s → %s",
                    getattr(agent, "unique_id", id(agent)), role,
                )
            except Exception:
                logger.debug("Failed to differentiate agent[%d] as %s", idx, role, exc_info=True)

    triads = (
        1
        + (1 if len(agents) >= 6 else 0)
        + (1 if len(agents) >= 9 else 0)
        + (1 if len(agents) >= 12 else 0)
    )
    logger.info(
        "Layer 33i - Agent differentiation: %d agents across %d K3 triads "
        "(general + %s)",
        differentiated, triads,
        "market + intelligence + hypothesis" if len(agents) >= 12
        else ("market + intelligence" if len(agents) >= 9 else "market"),
    )
