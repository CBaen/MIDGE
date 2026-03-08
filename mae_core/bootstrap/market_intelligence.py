"""Bootstrap Layer 33a helpers: RSI Layer 2 (hypothesis loop) + Pattern Archaeology.

Extracted from market_systems.py to stay under the 500-line monolith cap.
Called by _instantiate_market_systems() via _instantiate_intelligence_systems().

Constructs:
  - HypothesisRegistry, HypothesisGenerator, HypothesisValidator
  - BacktestAnalyzer, BacktestScheduler, HypothesisEngine
  - PatternLibrary, PatternWatcher, ExcavationDaemon
  - ActiveTracker
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

logger = logging.getLogger("midge.bootstrap")


def _instantiate_hypothesis_loop(ctx: SimpleNamespace) -> None:
    """Construct RSI Layer 2: hypothesis registry, generator, validator, engine."""
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


def _instantiate_archaeology(ctx: SimpleNamespace) -> int:
    """Construct Pattern Archaeology + Active Tracker. Returns failure count."""
    failures = 0

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

    return failures


def _instantiate_intelligence_systems(ctx: SimpleNamespace) -> int:
    """Entry point called by market_systems.py. Returns failure count."""
    _instantiate_hypothesis_loop(ctx)
    return _instantiate_archaeology(ctx)
