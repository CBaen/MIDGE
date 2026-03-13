"""Bootstrap Layer 33k: Biological system market activation.

Wire biological systems to market EventBus channels, giving each a real
market job. Each wiring function subscribes a bio system to market channels
and translates market events into system-appropriate inputs.

Tier 2 (one hop away — already hormone-connected via EndocrineSystem):
  EmotionalSystem, HomeostasisRegulator, ArousalRegulator,
  CuriosityDrive, NociceptionSystem

Tier 3 (clear market jobs, need EventBus subscriptions):
  CircadianRhythm, MetacognitionMonitor, ThreatDetector, HAVEN,
  InhibitionSystem, MemoryConsolidator, CollectiveDreamPlanner,
  QuorumSpace, Stigmergy

Implementation lives in:
  - bio_market_wiring_a.py — Tier 2 + MetacognitionMonitor + ThreatDetector
  - bio_market_wiring_b.py — QuorumSpace, CircadianRhythm, HAVEN,
                             InhibitionSystem, MemoryConsolidator,
                             CollectiveDreamPlanner, Stigmergy
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


def wire_bio_systems_to_market(ctx: SimpleNamespace) -> None:
    """Activate biological systems with market intelligence jobs.

    Called after both core bio systems (Layers 2-30) and market systems
    (Layer 33) are fully initialized. All wiring is additive — existing
    bio system behavior is unchanged, market signals provide new inputs.
    """
    bus = getattr(ctx, "bus", None)
    if bus is None:
        return

    from mae_core.bootstrap.bio_market_wiring_a import (
        _wire_emotional_system,
        _wire_homeostasis,
        _wire_arousal,
        _wire_curiosity,
        _wire_nociception,
        _wire_metacognition,
        _wire_threat_detector,
    )
    from mae_core.bootstrap.bio_market_wiring_b import (
        _wire_quorum,
        _wire_circadian,
        _wire_haven,
        _wire_inhibition,
        _wire_memory_consolidator,
        _wire_collective_dream,
        _wire_stigmergy,
    )

    count = 0
    # Tier 2: one hop away
    count += _wire_emotional_system(ctx, bus)
    count += _wire_homeostasis(ctx, bus)
    count += _wire_arousal(ctx, bus)
    count += _wire_curiosity(ctx, bus)
    count += _wire_nociception(ctx, bus)
    # Tier 3: clear market jobs
    count += _wire_metacognition(ctx, bus)
    count += _wire_threat_detector(ctx, bus)
    count += _wire_quorum(ctx, bus)
    count += _wire_circadian(ctx, bus)
    count += _wire_haven(ctx, bus)
    count += _wire_inhibition(ctx, bus)
    count += _wire_memory_consolidator(ctx, bus)
    count += _wire_collective_dream(ctx, bus)
    count += _wire_stigmergy(ctx, bus)

    # Tier 4+5: extended wiring (separate file to prevent monolith)
    try:
        from mae_core.bootstrap.bio_market_wiring_extended import (
            wire_bio_systems_extended,
        )
        count += wire_bio_systems_extended(ctx)
    except Exception:
        logger.debug("Tier 4+5 bio wiring failed", exc_info=True)

    # Endocrine → ResourceGovernor cortisol coupling.
    # MIDGE: disabled — fictional physiology harms trading daemon.
    # High cortisol (market stress events: deception detected, failed predictions)
    # was reducing EXPLORE-tier API source budgets by the cortisol factor (up to
    # 40% reduction at cortisol=0.6). Market stress is exactly when MIDGE should
    # fetch MORE data, not less. The ResourceGovernor itself (real API rate limits)
    # remains fully operational — only the cortisol multiplier is disabled.
    # To re-enable: uncomment the block below.
    #
    # if hasattr(ctx, "resource_governor") and ctx.resource_governor is not None:
    #     endocrine = getattr(ctx, "endocrine", None)
    #     if endocrine is not None and hasattr(endocrine, "register_resource_governor"):
    #         try:
    #             endocrine.register_resource_governor(ctx.resource_governor)
    #             count += 1
    #             logger.debug(
    #                 "Layer 33k - Endocrine → ResourceGovernor cortisol coupling wired"
    #             )
    #         except Exception:
    #             logger.debug(
    #                 "Layer 33k - Endocrine → ResourceGovernor coupling failed",
    #                 exc_info=True,
    #             )
    logger.debug("Layer 33k - Cortisol → ResourceGovernor coupling DISABLED (fictional physiology)")

    logger.info(
        "Layer 33k - Bio-market activation: %d systems wired to market channels",
        count,
    )
