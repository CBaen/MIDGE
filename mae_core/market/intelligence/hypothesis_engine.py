"""Hypothesis Engine — RSI Layer 2 orchestrator.

The engine manages the full hypothesis lifecycle:
  discover → formalize → validate → promote/retire → monitor

It owns the generator, validator, and registry references.
Driven by the step loop, it runs generation on cadence,
validation on cadence, and monitors incoming signals against
active hypothesis triggers via EventBus subscription.

Step cadence:
  - Generation: every 500 steps (generate new hypotheses from lag findings)
  - Validation: every 1000 steps (validate probation hypotheses)
  - Regime check: every 100 steps (hibernate/reactivate based on regime)

EventBus:
  - Subscribes to CH_SIGNAL_INGESTED (matches incoming signals to triggers)
  - Publishes CH_HYPOTHESIS_DISCOVERED, CH_HYPOTHESIS_PROMOTED,
    CH_HYPOTHESIS_RETIRED, CH_HYPOTHESIS_FIRED

Implementation split:
  - HypothesisLifecycleMixin     (hypothesis_lifecycle.py)
      step, request_generation, request_validation, on_signal_ingested,
      _run_generation, _launch_validation, _collect_validation_results,
      _run_validation, _promote, _retire, get_persistence_stats
  - HypothesisMetaLearningMixin  (hypothesis_meta_learning.py)
      _save_retirement_window, _load_retirement_window,
      _seed_retirement_window_from_registry, _check_regime,
      _review_gates, _run_meta_learning, get_statistics
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

_RETIREMENT_WINDOW_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "retirement_window.json"

from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
from mae_core.market.intelligence.hypothesis_lifecycle import HypothesisLifecycleMixin
from mae_core.market.intelligence.hypothesis_meta_learning import HypothesisMetaLearningMixin

logger = logging.getLogger(__name__)


class HypothesisEngine(HypothesisLifecycleMixin, HypothesisMetaLearningMixin):
    """Orchestrates hypothesis lifecycle — the RSI Layer 2 brain.

    Biological analogy: The prefrontal cortex. It doesn't sense signals
    directly (that's the sensing hook) or store memories (that's the
    registry). It decides what patterns to investigate, tests them
    adversarially, and promotes or kills them based on evidence.
    """

    def __init__(
        self,
        registry: HypothesisRegistry,
        generator: HypothesisGenerator,
        validator: HypothesisValidator,
        bus: Any = None,
        regime_classifier: Any = None,
        thompson_sampler: Any = None,
        backtest_analyzer: Any = None,
        archaeological_analyzer: Any = None,
        thompson_calibrator: Any = None,
        generation_cadence: int = 500,
        validation_cadence: int = 1000,
        regime_cadence: int = 100,
    ):
        self._registry = registry
        self._generator = generator
        self._validator = validator
        self._bus = bus
        self._regime_classifier = regime_classifier
        self._thompson_sampler = thompson_sampler
        self._backtest_analyzer = backtest_analyzer
        self._archaeological_analyzer = archaeological_analyzer
        self._thompson_calibrator = thompson_calibrator

        self._generation_cadence = generation_cadence
        self._validation_cadence = validation_cadence
        self._regime_cadence = regime_cadence
        self._step_counter = 0
        self._last_generation_step = 0

        # Signal match tracking
        self._signals_matched = 0
        self._hypotheses_generated = 0
        self._hypotheses_promoted = 0
        self._hypotheses_retired = 0

        # Background validation — skip-if-busy pattern
        self._validation_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hyp-val")
        self._validation_future: Optional[Future] = None

        # Bridge 4: meta-tracking for dynamic gate review
        self._meta_promoted_total = 0
        self._meta_retired_after_active = 0
        self._gate_review_cadence = 2000
        self._gate_cooldowns: dict[str, int] = {}  # key → step when last adjusted

        # Bridge 5: meta-learning for RSI Layer 3
        # Meta-learning persistence verified: config snapshot warm-starts from
        # market_systems.py load_snapshot() before any system is constructed,
        # retirement window loads in __init__ via _load_retirement_window() then
        # seeds from registry if empty via _seed_retirement_window_from_registry(),
        # pair outcomes load in generator __init__ via _load_pair_outcomes().
        # Meta-learning fires at step 3000 regardless of session number (cadence-
        # based counter resets to 0 on restart — absolute step is irrelevant).
        self._meta_learning_cadence = 3000
        # Ring buffer of outcome dicts: {"outcome": "promoted"/"retired", "seeded": bool}
        # seeded=True entries come from _seed_retirement_window_from_registry() and
        # represent historical state reconstructed at cold-start. Wire 2 of meta-
        # learning only counts live (seeded=False) entries so historical data cannot
        # bias the threshold adjustments before any real session data accumulates.
        self._retirement_window: list[dict] = []
        self._retirement_window_max = 50

        # Derive retirement window path from registry's data_dir so tests using
        # tmp_path registries get isolated persistence (no cross-test contamination).
        registry_data_dir = getattr(registry, "_data_dir", None)
        if registry_data_dir is not None:
            self._retirement_window_path = Path(registry_data_dir) / "retirement_window.json"
        else:
            self._retirement_window_path = _RETIREMENT_WINDOW_PATH

        self._load_retirement_window()

        # Cold-start: seed from registry if persistence file didn't exist
        if not self._retirement_window:
            self._seed_retirement_window_from_registry()
