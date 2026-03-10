"""HypothesisLifecycleMixin — step, generation, validation, promote/retire.

Used by HypothesisEngine as a mixin. Expects the following instance attributes:
    self._registry, self._generator, self._validator, self._bus,
    self._thompson_sampler, self._backtest_analyzer, self._archaeological_analyzer,
    self._validation_executor, self._validation_future,
    self._step_counter, self._last_generation_step,
    self._generation_cadence, self._validation_cadence,
    self._signals_matched, self._hypotheses_generated,
    self._hypotheses_promoted, self._hypotheses_retired,
    self._meta_promoted_total, self._meta_retired_after_active,
    self._retirement_window, self._retirement_window_max,
    self._save_retirement_window()  (provided by HypothesisMetaLearningMixin)
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from datetime import datetime
from typing import Any, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
)

logger = logging.getLogger(__name__)


class HypothesisLifecycleMixin:
    """Step, generation, validation, promote/retire lifecycle methods."""

    def step(self) -> None:
        """Called every model step. Runs lifecycle operations on cadence."""
        self._step_counter += 1

        # Check if background validation completed
        self._collect_validation_results()

        if self._step_counter % self._regime_cadence == 0:
            self._check_regime()

        if self._step_counter % self._generation_cadence == 0:
            self._run_generation()
            self._last_generation_step = self._step_counter

        if self._step_counter % self._validation_cadence == 0:
            self._launch_validation()

        if self._step_counter % self._gate_review_cadence == 0:
            self._review_gates()

        if self._step_counter % self._meta_learning_cadence == 0:
            self._run_meta_learning()

    # -----------------------------------------------------------------
    # Agent-triggered methods (Phase 3a — market action dispatch)
    # -----------------------------------------------------------------

    def request_generation(self) -> int:
        """Agent-triggered hypothesis generation with cooldown.

        Returns the number of new hypotheses generated. Respects a 100-step
        cooldown to prevent over-generation. The cadence-based step() clock
        continues as a floor — agents can accelerate but not replace it.
        """
        last_gen = getattr(self, "_last_generation_step", 0)
        if self._step_counter - last_gen < 100:
            return 0
        before = self._hypotheses_generated
        self._run_generation()
        self._last_generation_step = self._step_counter
        return self._hypotheses_generated - before

    def request_validation(self) -> str:
        """Agent-triggered validation with skip-if-busy guard.

        Returns:
            'promoted' — a hypothesis was promoted
            'retired' — a hypothesis was retired (caught a bad one)
            'busy' — background validation already in-flight
            'none' — no validatable hypotheses exist
        """
        if self._validation_future is not None and not self._validation_future.done():
            return "busy"

        probation = self._registry.get_probation()
        validatable = [
            h for h in probation
            if hasattr(h, "stats") and h.stats.total_observations >= 20
        ]
        if not validatable:
            return "none"

        # Validate the most-ready hypothesis synchronously (agent is waiting)
        target = max(validatable, key=lambda h: h.stats.total_observations)
        try:
            result = self._validator.validate(target)
            target.stats.wins = result.wins
            target.stats.losses = result.losses
            target.stats.total_observations = result.total_observations
            target.stats.sharpe_ratio = result.sharpe_ratio
            target.stats.deflated_sharpe_ratio = result.deflated_sharpe_ratio
            target.stats.last_evaluated = datetime.now()
            self._registry.update_stats(target.hypothesis_id, target)

            if result.recommend_promote:
                self._promote(target)
                return "promoted"
            elif result.recommend_retire:
                self._retire(target, result.retire_reason)
                return "retired"
            return "none"
        except Exception:
            logger.debug("Agent-triggered validation failed for %s", target.name, exc_info=True)
            return "none"

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

        # Path 3: Archaeological hypotheses (reverse-engineered from historical moves)
        if self._archaeological_analyzer is not None:
            try:
                arch_hypotheses = self._archaeological_analyzer.analyze()
                all_new.extend(arch_hypotheses)
            except Exception:
                logger.debug("Archaeological analysis failed", exc_info=True)

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

    def _launch_validation(self) -> None:
        """Launch validation in background thread. Skip if already running."""
        if self._validation_future is not None and not self._validation_future.done():
            return  # Previous validation still running
        self._validation_future = self._validation_executor.submit(
            self._run_validation)

    def _collect_validation_results(self) -> None:
        """Check if background validation completed."""
        if self._validation_future is None or not self._validation_future.done():
            return
        try:
            self._validation_future.result()
        except Exception:
            logger.debug("Background validation failed", exc_info=True)
        finally:
            self._validation_future = None

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
        self._meta_promoted_total += 1

        # Bridge 5: track outcome in retirement window + pair quality
        self._retirement_window.append({"outcome": "promoted", "seeded": False})
        if len(self._retirement_window) > self._retirement_window_max:
            self._retirement_window.pop(0)
        try:
            self._generator.record_outcome(
                hyp.trigger.source_a, hyp.trigger.source_b, "promoted")
        except Exception:
            pass
        self._save_retirement_window()

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
        if was_active:
            self._meta_retired_after_active += 1

        # Bridge 5: track outcome in retirement window + pair quality
        self._retirement_window.append({"outcome": "retired", "seeded": False})
        if len(self._retirement_window) > self._retirement_window_max:
            self._retirement_window.pop(0)
        try:
            self._generator.record_outcome(
                hyp.trigger.source_a, hyp.trigger.source_b, "retired")
        except Exception:
            pass
        self._save_retirement_window()

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

    def get_persistence_stats(self) -> dict:
        """Return persistence state for reporting and post-mortem diagnostics."""
        return {
            "retirement_window_size": len(self._retirement_window),
            "retirement_window_path": str(self._retirement_window_path),
            "meta_promoted_total": self._meta_promoted_total,
            "meta_retired_after_active": self._meta_retired_after_active,
        }
