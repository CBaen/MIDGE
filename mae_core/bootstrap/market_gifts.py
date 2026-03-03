"""Bootstrap Layer 33a-gifts: Ten Gifts system instantiation.

Extracted from market_systems.py to stay under the 500-line cap.
Constructs Wave 1-3 gift systems on ctx with graceful degradation.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _instantiate_gift_systems(ctx: SimpleNamespace) -> None:
    """Construct Ten Gifts systems on ctx (Wave 1-3)."""

    # --- Wave 1 (Foundation) ---

    try:
        from mae_core.market.intelligence.portfolio_tracker import PortfolioTracker
        ctx.portfolio_tracker = PortfolioTracker(
            price_fetcher=getattr(ctx, "price_fetcher", None),
        )
    except Exception:
        logger.debug("Market: portfolio_tracker failed to construct", exc_info=True)
        ctx.portfolio_tracker = None

    try:
        from mae_core.market.edge.order_flow_detector import OrderFlowDetector
        ctx.order_flow_detector = OrderFlowDetector(
            price_fetcher=getattr(ctx, "price_fetcher", None),
        )
    except Exception:
        logger.debug("Market: order_flow_detector failed to construct", exc_info=True)
        ctx.order_flow_detector = None

    try:
        from mae_core.market.intelligence.catalyst_calendar import CatalystCalendar
        ctx.catalyst_calendar = CatalystCalendar(
            finnhub_client=getattr(ctx, "finnhub_client", None),
        )
    except Exception:
        logger.debug("Market: catalyst_calendar failed to construct", exc_info=True)
        ctx.catalyst_calendar = None

    try:
        from mae_core.market.intelligence.cross_asset_confirmer import CrossAssetConfirmer
        ctx.cross_asset_confirmer = CrossAssetConfirmer(
            price_fetcher=getattr(ctx, "price_fetcher", None),
        )
    except Exception:
        logger.debug("Market: cross_asset_confirmer failed to construct", exc_info=True)
        ctx.cross_asset_confirmer = None

    # Wire Wave 1 gifts into convergence alerter (two-phase init — alerter exists before gifts)
    if getattr(ctx, "convergence_alerter", None) is not None:
        if getattr(ctx, "catalyst_calendar", None) is not None:
            ctx.convergence_alerter._catalyst_calendar = ctx.catalyst_calendar
        if getattr(ctx, "cross_asset_confirmer", None) is not None:
            ctx.convergence_alerter._cross_asset_confirmer = ctx.cross_asset_confirmer

    # --- Wave 2 (Intelligence) ---

    try:
        from mae_core.market.edge.deception_detector import DeceptionDetector
        ctx.deception_detector = DeceptionDetector(
            event_bus=getattr(ctx, "bus", None),
        )
    except Exception:
        logger.debug("Market: deception_detector failed to construct", exc_info=True)
        ctx.deception_detector = None

    try:
        from mae_core.market.intelligence.consolidation_engine import ConsolidationEngine
        ctx.consolidation_engine = ConsolidationEngine(
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            hypothesis_registry=getattr(ctx, "hypothesis_registry", None),
            hypothesis_engine=getattr(ctx, "hypothesis_engine", None),
            event_bus=getattr(ctx, "bus", None),
        )
    except Exception:
        logger.debug("Market: consolidation_engine failed to construct", exc_info=True)
        ctx.consolidation_engine = None

    try:
        from mae_core.market.edge.fractal_resonance import FractalResonanceDetector
        ctx.fractal_resonance_detector = FractalResonanceDetector(
            price_fetcher=getattr(ctx, "price_fetcher", None),
        )
    except Exception:
        logger.debug("Market: fractal_resonance_detector failed to construct", exc_info=True)
        ctx.fractal_resonance_detector = None

    try:
        from mae_core.market.intelligence.pattern_archetypes import PatternArchetypeEngine
        ctx.pattern_archetype_engine = PatternArchetypeEngine(
            price_fetcher=getattr(ctx, "price_fetcher", None),
        )
    except Exception:
        logger.debug("Market: pattern_archetype_engine failed to construct", exc_info=True)
        ctx.pattern_archetype_engine = None

    # Wire Wave 2 gifts into convergence alerter (two-phase init)
    if getattr(ctx, "convergence_alerter", None) is not None:
        if getattr(ctx, "deception_detector", None) is not None:
            ctx.convergence_alerter._deception_detector = ctx.deception_detector
        if getattr(ctx, "pattern_archetype_engine", None) is not None:
            ctx.convergence_alerter._pattern_archetype_engine = ctx.pattern_archetype_engine
