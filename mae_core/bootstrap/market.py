"""Bootstrap Layer 33: Market Intelligence Organ.

Creates 16 market systems, registers holons, wires fractal hierarchy,
registers triadic connections (Group 14), wires EventBus channels,
and hooks into the step lifecycle.

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
    """Wire Layer 33: Market Intelligence organ (16 systems, 23 connections)."""
    _instantiate_market_systems(ctx)
    _register_market_somatic(ctx)
    _register_market_holons(ctx)
    _register_market_fractal(ctx)
    _register_market_connections(ctx)
    _register_market_stem_roles(ctx)
    _register_market_eventbus(ctx)
    _register_market_step_hooks(ctx)

    # Count active systems
    market_attrs = [
        "sec_edgar_client", "price_fetcher", "house_stock_watcher",
        "job_tracker", "usa_spending_client", "sam_gov_client",
        "cluster_detector", "politician_tracker", "filing_time_analyzer",
        "contract_predictor", "thompson_sampler", "convergence_alerter",
        "velocity_detector", "correlation_tracker", "outcome_tracker",
        "regime_classifier",
    ]
    active = sum(1 for a in market_attrs if getattr(ctx, a, None) is not None)
    holon_count = len([
        hid for hid in ctx.holon_registry.get_all_ids()
        if hid.startswith("market") or hid in market_attrs
        or hid in ("sec_edgar_client", "price_fetcher", "house_stock_watcher",
                    "job_tracker", "usa_spending_client", "sam_gov_client",
                    "cluster_detector", "politician_tracker", "filing_time_analyzer",
                    "contract_predictor", "thompson_sampler", "convergence_alerter",
                    "velocity_detector", "correlation_tracker", "outcome_tracker",
                    "regime_classifier")
    ])

    logger.info(
        "Layer 33  - Market Intelligence organ complete: %d systems, %d holons, 23 connections",
        active, holon_count,
    )
    logger.info(
        "            NOTE: OutcomeTracker active (Bayesian feedback from real market outcomes)."
    )
    logger.info(
        "            NOTE: MarketDataProvider registered with ApiGateway. "
        "Client migration to gateway routing is incremental."
    )


# =========================================================================
# Layer 33a: Instantiate market systems
# =========================================================================

def _instantiate_market_systems(ctx: SimpleNamespace) -> None:
    """Create all 16 market system objects on ctx."""
    qdrant_url = getattr(ctx, "qdrant_url", "http://localhost:6333")
    failures = 0

    # --- API clients (Market Sensing) ---

    try:
        from mae_core.market.apis.sec_edgar.client import SECEdgarClient
        ctx.sec_edgar_client = SECEdgarClient()
    except Exception:
        logger.debug("Market: sec_edgar_client failed to construct", exc_info=True)
        ctx.sec_edgar_client = None
        failures += 1

    try:
        from mae_core.market.apis.price_fetcher import PriceFetcher
        ctx.price_fetcher = PriceFetcher()
    except Exception:
        logger.debug("Market: price_fetcher failed to construct", exc_info=True)
        ctx.price_fetcher = None
        failures += 1

    try:
        from mae_core.market.apis.house_stock_watcher import HouseStockWatcherClient
        ctx.house_stock_watcher = HouseStockWatcherClient()
    except Exception:
        logger.debug("Market: house_stock_watcher failed to construct", exc_info=True)
        ctx.house_stock_watcher = None
        failures += 1

    try:
        from mae_core.market.apis.job_tracker import JobTracker
        ctx.job_tracker = JobTracker()
    except Exception:
        logger.debug("Market: job_tracker failed to construct", exc_info=True)
        ctx.job_tracker = None
        failures += 1

    try:
        from mae_core.market.apis.usa_spending import USASpendingClient
        ctx.usa_spending_client = USASpendingClient()
    except Exception:
        logger.debug("Market: usa_spending_client failed to construct", exc_info=True)
        ctx.usa_spending_client = None
        failures += 1

    try:
        from mae_core.market.apis.sam_gov import SAMGovClient
        ctx.sam_gov_client = SAMGovClient()
    except Exception:
        logger.debug("Market: sam_gov_client failed to construct", exc_info=True)
        ctx.sam_gov_client = None
        failures += 1

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
        ctx.convergence_alerter = ConvergenceAlerter(min_domains=3)
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

    # --- Trust registration with BoundaryMembrane ---
    market_sources = [
        ("sec_edgar", 0.90), ("yfinance", 0.75), ("alpha_vantage", 0.80),
        ("rapidapi", 0.65), ("usa_spending", 0.85), ("sam_gov", 0.80),
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
    if gateway is not None:
        try:
            from mae_core.market.apis.market_data_provider import MarketDataProvider
            provider = MarketDataProvider()
            gateway.register_provider("market_data", provider)
            ctx.market_data_provider = provider
        except Exception:
            logger.debug("Could not register MarketDataProvider with ApiGateway", exc_info=True)
            ctx.market_data_provider = None
    else:
        ctx.market_data_provider = None

    logger.info(
        "Layer 33a - Market systems instantiated: %d systems (construction failures: %d)",
        16 - failures, failures,
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
        "thompson_sampler": ("ThompsonSampler", []),
        "convergence_alerter": ("ConvergenceAlerter", ["thompson_sampler", "velocity_detector"]),
        "velocity_detector": ("VelocityDetector", []),
        "correlation_tracker": ("CorrelationTracker", []),
        "outcome_tracker": ("OutcomeTracker", ["price_fetcher", "thompson_sampler"]),
        "regime_classifier": ("RegimeClassifier", ["price_fetcher"]),
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
        "contract_predictor", "thompson_sampler", "convergence_alerter",
        "velocity_detector", "correlation_tracker", "outcome_tracker",
        "regime_classifier",
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

    # Register remaining systems individually (advisory non-triadic)
    extras = [
        "house_stock_watcher", "filing_time_analyzer",
        "usa_spending_client", "sam_gov_client", "correlation_tracker",
        "outcome_tracker", "regime_classifier",
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
    """Register 23 triadic connections for market systems (Group 14)."""
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

    # --- Step hook + periodic ---
    reg("convergence_alerter", "model", sh,
        witnesses=["auditor", "connection_registry"],
        description="Convergence alerter runs each step")
    reg("thompson_sampler", "event_bus", eb,
        channel="market.intel.thompson_stats",
        witnesses=["auditor", "connection_registry"],
        description="Thompson stats published periodically")

    logger.info("Layer 33d - Market connections: 23 triadic connections registered (Group 14)")


# =========================================================================
# Layer 33e: Stem cell role registration
# =========================================================================

def _register_market_stem_roles(ctx: SimpleNamespace) -> None:
    """Verify market stem cell roles are available."""
    from mae_core.agents.stem_cell import ROLE_PROFILES

    market_roles = ["SEC_WATCHER", "CONTRACT_TRACKER", "MARKET_ANALYST"]
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
    logger.info("Layer 33f - Market EventBus: convergence -> endocrine coupling wired")


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

    def _get_regime():
        """Get current market regime (cached daily, essentially free)."""
        rc = getattr(ctx, "regime_classifier", None)
        if rc is not None:
            try:
                return rc.classify()
            except Exception:
                pass
        return "default"

    def _market_sense_hook():
        _step_counter[0] += 1
        step = _step_counter[0]

        # Every step: check convergence (lightweight, pure in-memory)
        alerter = getattr(ctx, "convergence_alerter", None)
        if alerter is not None:
            try:
                alerts = alerter.check_convergence()
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

    ctx.model.add_step_hook(_market_sense_hook)
    logger.info(
        "Layer 33g - Market step hooks: 1 sense hook registered "
        "(cadence: convergence/1, stats/10, velocity/50, forgetting/100)"
    )
