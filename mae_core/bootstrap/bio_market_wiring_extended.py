"""Bootstrap Layer 33k extended: Tier 4+5 biological system market activation.

Wire remaining biological systems to market EventBus channels. Each system
gets a real market job that maps its biological metaphor to actual market
intelligence work.

Tier 4 (market jobs exist but less direct):
  DigestiveSystem, CirculatorySystem, LymphaticSystem, Microbiome,
  RenalFilter, SenescenceManager, MorphogenesisCoordinator,
  ReproductiveSystem, PearlDefense

Tier 5 (purpose emerges through market connection):
  RespiratorySystem, ThermoregulationSystem, VestibularSystem,
  ProprioceptionSystem, EnergyReserve, PredictiveField

Note: GenerativeReplayMemory is not bootstrapped (needs state_dim/action_dim
design decisions). Deferred to a separate build.

Implementation lives in:
  - bio_market_wiring_extended_a.py — Tier 4 (DigestiveSystem through ReproductiveSystem)
  - bio_market_wiring_extended_b.py — Tier 4 (PearlDefense) + all Tier 5
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("midge.bootstrap")


def _parse(data: Any) -> dict:
    """Safe parse of EventBus message to dict."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def wire_bio_systems_extended(ctx: SimpleNamespace) -> int:
    """Wire Tier 4+5 biological systems to market channels.

    Returns the number of systems successfully wired.
    """
    bus = getattr(ctx, "bus", None)
    if bus is None:
        return 0

    from mae_core.bootstrap.bio_market_wiring_extended_a import (
        _wire_digestive,
        _wire_circulatory,
        _wire_lymphatic,
        _wire_microbiome,
        _wire_renal_filter,
        _wire_senescence,
        _wire_morphogenesis,
        _wire_reproductive,
    )
    from mae_core.bootstrap.bio_market_wiring_extended_b import (
        _wire_pearl_defense,
        _wire_respiratory,
        _wire_thermoregulation,
        _wire_vestibular,
        _wire_proprioception,
        _wire_energy_reserve,
        _wire_predictive_field,
    )

    count = 0
    # Tier 4: market jobs exist but less direct
    count += _wire_digestive(ctx, bus)
    count += _wire_circulatory(ctx, bus)
    count += _wire_lymphatic(ctx, bus)
    count += _wire_microbiome(ctx, bus)
    count += _wire_renal_filter(ctx, bus)
    count += _wire_senescence(ctx, bus)
    count += _wire_morphogenesis(ctx, bus)
    count += _wire_reproductive(ctx, bus)
    count += _wire_pearl_defense(ctx, bus)
    # Tier 5: purpose emerges through connection
    count += _wire_respiratory(ctx, bus)
    count += _wire_thermoregulation(ctx, bus)
    count += _wire_vestibular(ctx, bus)
    count += _wire_proprioception(ctx, bus)
    count += _wire_energy_reserve(ctx, bus)
    count += _wire_predictive_field(ctx, bus)

    logger.info(
        "Layer 33k extended - Tier 4+5 bio activation: %d systems wired", count,
    )
    return count
