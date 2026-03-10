"""Bootstrap Layer 33a: Market infrastructure systems.

OctopusColony, ResourceGovernor, InhabitantScheduler, GovernanceLogger,
StepTimer, pattern discovery (MotifDetector, StreamingAnomaly, DriftDetector),
and Risk Architecture (DrawdownMonitor, SystemHealthMonitor, SelfMonitor).

Extracted from market_systems.py to stay under 500-line cap.
"""
from __future__ import annotations

import json as _json
import logging
import time as _time
import threading as _th
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger("midge.bootstrap")


def _instantiate_infrastructure(ctx: SimpleNamespace) -> None:
    """Construct infrastructure systems on ctx."""
    raw_store = getattr(ctx, "_raw_store_ref", None)

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
        ctx.octopus_colony._developing_situations = {}
        ctx.octopus_colony._situations_lock = _th.Lock()

        # Load persisted developing situations (entries older than 2 hours dropped).
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

    # --- Risk Architecture (DrawdownMonitor, SystemHealthMonitor, SelfMonitor) ---
    try:
        from mae_core.market.intelligence.drawdown_monitor import DrawdownMonitor
        ctx.drawdown_monitor = DrawdownMonitor(
            event_bus=getattr(ctx, "bus", None),
            data_dir="data/market",
        )
        ctx.drawdown_monitor.load_state()
    except Exception:
        logger.debug("Market: drawdown_monitor failed to construct", exc_info=True)
        ctx.drawdown_monitor = None

    try:
        from mae_core.market.system_health_monitor import SystemHealthMonitor
        ctx.system_health_monitor = SystemHealthMonitor(
            event_bus=getattr(ctx, "bus", None),
            step_timer=getattr(ctx, "step_timer", None),
        )
    except Exception:
        logger.debug("Market: system_health_monitor failed to construct", exc_info=True)
        ctx.system_health_monitor = None

    try:
        from mae_core.market.intelligence.self_monitor import SelfMonitor
        ctx.self_monitor = SelfMonitor(event_bus=getattr(ctx, "bus", None))
    except Exception:
        logger.debug("Market: self_monitor failed to construct", exc_info=True)
        ctx.self_monitor = None
