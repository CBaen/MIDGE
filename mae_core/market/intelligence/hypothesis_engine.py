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
"""

from __future__ import annotations

import logging
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator

logger = logging.getLogger(__name__)


class HypothesisEngine:
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

        self._generation_cadence = generation_cadence
        self._validation_cadence = validation_cadence
        self._regime_cadence = regime_cadence
        self._step_counter = 0

        # Signal match tracking
        self._signals_matched = 0
        self._hypotheses_generated = 0
        self._hypotheses_promoted = 0
        self._hypotheses_retired = 0

        # Background validation — skip-if-busy pattern
        self._validation_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hyp-val")
        self._validation_future: Optional[Future] = None

    def step(self) -> None:
        """Called every model step. Runs lifecycle operations on cadence."""
        self._step_counter += 1

        # Check if background validation completed
        self._collect_validation_results()

        if self._step_counter % self._regime_cadence == 0:
            self._check_regime()

        if self._step_counter % self._generation_cadence == 0:
            self._run_generation()

        if self._step_counter % self._validation_cadence == 0:
            self._launch_validation()

    def on_signal_ingested(self, channel: str, data: Any) -> None:
        """EventBus callback — match incoming signals against active triggers.

        When a signal matches a hypothesis trigger's source_a, we record
        a "fire" event. The hypothesis engine tracks these for validation.
        """
        if isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except Exception:
                return

        source = data.get("source", "")
        symbol = data.get("symbol", "")

        for hyp in self._registry.get_active():
            if hyp.trigger.source_a == source:
                self._signals_matched += 1
                if self._bus is not None:
                    from mae_core.market.channels import CH_HYPOTHESIS_FIRED
                    try:
                        self._bus.publish(CH_HYPOTHESIS_FIRED, {
                            "hypothesis_id": hyp.hypothesis_id,
                            "hypothesis_name": hyp.name,
                            "trigger_source": source,
                            "trigger_symbol": symbol,
                            "expected_lag_days": hyp.trigger.lag_days,
                            "expected_direction": hyp.trigger.direction,
                            "timestamp": datetime.now().isoformat(),
                        })
                    except Exception:
                        pass

    def _run_generation(self) -> None:
        """Generate new hypotheses from lag findings + backtest results."""
        all_new = []

        # Path 1: Lag-correlation hypotheses
        try:
            lag_hypotheses = self._generator.generate()
            all_new.extend(lag_hypotheses)
        except Exception:
            logger.debug("Lag-correlation generation failed", exc_info=True)

        # Path 2: Backtest-derived hypotheses (Bridge 1)
        if self._backtest_analyzer is not None:
            try:
                bt_hypotheses = self._backtest_analyzer.analyze()
                all_new.extend(bt_hypotheses)
            except Exception:
                logger.debug("Backtest analysis failed", exc_info=True)

        self._hypotheses_generated += len(all_new)

        if all_new and self._bus is not None:
            from mae_core.market.channels import CH_HYPOTHESIS_DISCOVERED
            for hyp in all_new:
                try:
                    self._bus.publish(CH_HYPOTHESIS_DISCOVERED, {
                        "hypothesis_id": hyp.hypothesis_id,
                        "name": hyp.name,
                        "source_a": hyp.trigger.source_a,
                        "source_b": hyp.trigger.source_b,
                        "lag_days": hyp.trigger.lag_days,
                        "has_causal_story": bool(
                            hyp.causal_story
                            and "REQUIRES MANUAL REVIEW" not in hyp.causal_story
                        ),
                    })
                except Exception:
                    pass

        if all_new:
            logger.info(
                "HypothesisEngine: generated %d new hypotheses",
                len(all_new),
            )

    def _run_validation(self) -> None:
        """Validate hypotheses in probation — promote or retire."""
        probation = self._registry.get_probation()
        if not probation:
            return

        for hyp in probation:
            try:
                result = self._validator.validate(hyp)

                # Update stats on the hypothesis
                hyp.stats.wins = result.wins
                hyp.stats.losses = result.losses
                hyp.stats.total_observations = result.total_observations
                hyp.stats.sharpe_ratio = result.sharpe_ratio
                hyp.stats.deflated_sharpe_ratio = result.deflated_sharpe_ratio
                hyp.stats.last_evaluated = datetime.now()
                self._registry.update_stats(hyp.hypothesis_id, hyp)

                if result.recommend_promote:
                    self._promote(hyp)
                elif result.recommend_retire:
                    self._retire(hyp, result.retire_reason)

            except Exception:
                logger.debug(
                    "Validation failed for %s", hyp.name, exc_info=True
                )

        # Also re-validate active hypotheses (check for degradation)
        for hyp in self._registry.get_active():
            try:
                result = self._validator.validate(hyp)
                hyp.stats.wins = result.wins
                hyp.stats.losses = result.losses
                hyp.stats.total_observations = result.total_observations
                hyp.stats.sharpe_ratio = result.sharpe_ratio
                hyp.stats.deflated_sharpe_ratio = result.deflated_sharpe_ratio
                hyp.stats.last_evaluated = datetime.now()
                self._registry.update_stats(hyp.hypothesis_id, hyp)

                if result.recommend_retire:
                    self._retire(hyp, result.retire_reason)
            except Exception:
                logger.debug(
                    "Re-validation failed for %s", hyp.name, exc_info=True
                )

    def _promote(self, hyp: Hypothesis) -> None:
        """Promote a hypothesis — update registry, Thompson, endocrine."""
        promoted = self._registry.promote(hyp.hypothesis_id)
        if promoted is None:
            return

        self._hypotheses_promoted += 1

        # Register a Thompson key for the promoted hypothesis
        if self._thompson_sampler is not None:
            try:
                if not hasattr(self._thompson_sampler, 'distributions'):
                    pass
                elif hyp.source_type == SourceType.BACKTEST_DERIVED:
                    # Granular key encoding the specific pattern
                    key = f"sweep_bt:{hyp.trigger.domain_filter}"
                    if key not in self._thompson_sampler.distributions:
                        # Seed with backtest evidence + conservative prior
                        wins = hyp.stats.wins
                        losses = hyp.stats.losses
                        self._thompson_sampler.distributions[key] = {
                            "default": {
                                "alpha": wins + 1.1,
                                "beta": losses + 0.9,
                            }
                        }
                        self._thompson_sampler._save_distributions()
                else:
                    key = f"hyp_{hyp.trigger.source_a}_{hyp.trigger.source_b}"
                    if key not in self._thompson_sampler.distributions:
                        self._thompson_sampler.distributions[key] = {
                            "default": {"alpha": 1.0, "beta": 1.0}
                        }
                        self._thompson_sampler._save_distributions()
            except Exception:
                pass

        # Publish promotion event
        if self._bus is not None:
            from mae_core.market.channels import CH_HYPOTHESIS_PROMOTED
            try:
                self._bus.publish(CH_HYPOTHESIS_PROMOTED, {
                    "hypothesis_id": hyp.hypothesis_id,
                    "name": hyp.name,
                    "win_rate": hyp.stats.win_rate,
                    "dsr": hyp.stats.deflated_sharpe_ratio,
                    "observations": hyp.stats.total_observations,
                })
            except Exception:
                pass

        logger.info(
            "HYPOTHESIS PROMOTED: %s (win_rate=%.1f%%, DSR=%.3f)",
            hyp.name,
            hyp.stats.win_rate * 100,
            hyp.stats.deflated_sharpe_ratio,
        )

    def _retire(self, hyp: Hypothesis, reason: str) -> None:
        """Retire a hypothesis — update registry, publish event."""
        was_active = hyp.status == HypothesisStatus.ACTIVE
        retired = self._registry.retire(hyp.hypothesis_id, reason)
        if retired is None:
            return

        self._hypotheses_retired += 1

        if self._bus is not None:
            from mae_core.market.channels import CH_HYPOTHESIS_RETIRED
            try:
                self._bus.publish(CH_HYPOTHESIS_RETIRED, {
                    "hypothesis_id": hyp.hypothesis_id,
                    "name": hyp.name,
                    "reason": reason,
                    "was_active": was_active,
                    "win_rate": hyp.stats.win_rate,
                })
            except Exception:
                pass

    def _check_regime(self) -> None:
        """Check market regime — hibernate/reactivate hypotheses as needed."""
        if self._regime_classifier is None:
            return

        try:
            current_regime = self._regime_classifier.classify()
        except Exception:
            return

        # Hibernate active hypotheses created in a different regime
        for hyp in self._registry.get_active():
            if (hyp.regime_created_in != "default"
                    and hyp.regime_created_in != current_regime):
                self._registry.hibernate(hyp.hypothesis_id)

        # Reactivate hibernated hypotheses matching current regime
        for hyp in self._registry.get_all():
            if hyp.status == HypothesisStatus.HIBERNATED:
                if (hyp.regime_created_in == current_regime
                        or hyp.regime_created_in == "default"):
                    self._registry.reactivate(hyp.hypothesis_id)

    def get_statistics(self) -> dict:
        """Summary stats for HolonProxy.sense() and monitoring."""
        registry_stats = self._registry.get_statistics()
        return {
            "step_counter": self._step_counter,
            "signals_matched": self._signals_matched,
            "hypotheses_generated": self._hypotheses_generated,
            "hypotheses_promoted": self._hypotheses_promoted,
            "hypotheses_retired": self._hypotheses_retired,
            **registry_stats,
        }
