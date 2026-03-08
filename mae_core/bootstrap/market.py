"""Bootstrap Layer 33: Market Intelligence Organ.

Creates 51 market systems, registers holons, wires fractal hierarchy,
registers triadic connections (Group 14-32),
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

Sub-module responsibilities:
  market_systems.py       — Layer 33a: construct all market objects on ctx
  market_registration.py — Layer 33b-e: somatic, holons, fractal, stem roles
  market_connections.py  — Layer 33d: 103 triadic connections (Groups 14-32)
  market_hooks.py        — Layer 33f-h: EventBus callbacks, step hooks, sensing hook
  market_agents.py       — Layer 33i: agent differentiation + market reflexes
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from mae_core.bootstrap.market_systems import _instantiate_market_systems
from mae_core.bootstrap.market_registration import (
    _register_market_somatic,
    _register_market_holons,
    _register_market_fractal,
    _register_market_stem_roles,
)
from mae_core.bootstrap.market_connections import _register_market_connections
from mae_core.bootstrap.market_hooks import (
    _register_market_eventbus,
    _register_market_step_hooks,
    _wire_sensing_hook,
)
from mae_core.bootstrap.market_agents import _differentiate_market_agents

logger = logging.getLogger("midge.bootstrap")


def bootstrap_market(ctx: SimpleNamespace) -> None:
    """Wire Layer 33: Market Intelligence organ (51 systems, 103 connections).

    Call order is load-bearing — sub-modules have dependencies:
      1. _instantiate_market_systems: build all objects first
      2. _register_market_somatic: somatic map (must precede connections)
      3. _register_market_holons: holon registry
      4. _register_market_fractal: K3 fractal hierarchy
      5. _register_market_connections: triadic connections (requires somatic)
      6. _register_market_stem_roles: verify stem cell roles
      7. _register_market_eventbus: endocrine + hypothesis lifecycle coupling
      8. _register_market_step_hooks: cadenced step logic, writes ctx._cached_alerts
      9. _wire_sensing_hook: reads ctx._cached_alerts (must follow step hooks)
     10. _differentiate_market_agents: assign roles + reflexes (agents must exist)
    """
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

    # Count active systems for final log
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
        "absence_monitor",
        # Ten Gifts: Wave 1
        "portfolio_tracker", "order_flow_detector",
        "catalyst_calendar", "cross_asset_confirmer",
        # Ten Gifts: Wave 2
        "deception_detector", "consolidation_engine",
        "fractal_resonance_detector", "pattern_archetype_engine",
        # Ten Gifts: Wave 3
        "somatic_anticipation", "pattern_completion_engine",
        # Layer 6
        "cot_client", "stocktwits_client", "vix_client", "trends_client",
        # WP-B: Market Clock
        "market_clock",
        # Pattern Archaeology
        "pattern_library", "pattern_watcher", "excavation_daemon",
        # Ecosystem Bridge
        "octopus_colony",
    ]
    active = sum(1 for a in market_attrs if getattr(ctx, a, None) is not None)
    holon_count = len([
        hid for hid in ctx.holon_registry.get_all_ids()
        if hid.startswith("market") or hid in market_attrs
    ])

    logger.info(
        "Layer 33  - Market Intelligence organ complete: %d systems, %d holons, 103 connections",
        active, holon_count,
    )
    logger.info(
        "            NOTE: OutcomeTracker active (Bayesian feedback from real market outcomes)."
    )
    logger.info(
        "            NOTE: MarketDataProvider registered with ApiGateway. "
        "All 6 API clients route through provider when available."
    )
