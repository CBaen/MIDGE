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
        self._meta_learning_cadence = 3000
        self._retirement_window: list[str] = []  # ring buffer of "promoted"/"retired"
        self._retirement_window_max = 50

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
        self._retirement_window.append("promoted")
        if len(self._retirement_window) > self._retirement_window_max:
            self._retirement_window.pop(0)
        try:
            self._generator.record_outcome(
                hyp.trigger.source_a, hyp.trigger.source_b, "promoted")
        except Exception:
            pass

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
        self._retirement_window.append("retired")
        if len(self._retirement_window) > self._retirement_window_max:
            self._retirement_window.pop(0)
        try:
            self._generator.record_outcome(
                hyp.trigger.source_a, hyp.trigger.source_b, "retired")
        except Exception:
            pass

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

    def _review_gates(self) -> None:
        """Bridge 4: Review and adjust hypothesis promotion/retirement gates.

        Examines false-positive rate (promoted → retired) and promotion rate.
        If gates are too loose (high FP), tightens promote_win_rate.
        If gates are too tight (zero promotions with available candidates), loosens.
        All adjustments are clamped to _bounds and rate-limited by cooldown.
        """
        try:
            from mae_core.market.intelligence.learning_config import (
                LEARNING_CONFIG, update_config,
            )
        except ImportError:
            return

        gates = LEARNING_CONFIG.get("hypothesis_gates", {})
        bounds = gates.get("_bounds", {})

        # Compute false positive rate
        fp_rate = (
            self._meta_retired_after_active / max(1, self._meta_promoted_total)
            if self._meta_promoted_total > 0 else 0.0
        )

        # Check if any probation hypotheses exist (gate-tightness indicator)
        probation_count = 0
        try:
            probation_count = len(self._registry.get_probation())
        except Exception:
            pass

        adjustments = []

        # Case A: too many false positives → tighten promotion gate
        if fp_rate > 0.30 and self._meta_promoted_total >= 3:
            adjustments.append(("promote_win_rate", +0.01))

        # Case B: zero promotions but candidates exist → loosen gate
        elif (self._meta_promoted_total == 0
              and probation_count >= 3
              and self._step_counter > self._gate_review_cadence * 2):
            adjustments.append(("promote_win_rate", -0.01))

        # Apply adjustments with cooldown and bounds
        cooldown_steps = self._gate_review_cadence * 5  # 5 review cycles
        changed = False

        for key, delta in adjustments:
            last_adjusted = self._gate_cooldowns.get(key, 0)
            if self._step_counter - last_adjusted < cooldown_steps:
                continue  # Cooldown active

            current = gates.get(key, 0.0)
            new_val = current + delta

            # Clamp to bounds
            key_bounds = bounds.get(key, [None, None])
            if key_bounds[0] is not None:
                new_val = max(key_bounds[0], new_val)
            if key_bounds[1] is not None:
                new_val = min(key_bounds[1], new_val)

            if abs(new_val - current) < 1e-6:
                continue  # No effective change after clamping

            update_config(
                f"hypothesis_gates.{key}",
                round(new_val, 4),
                modified_by="gate_reviewer",
            )
            self._gate_cooldowns[key] = self._step_counter
            changed = True
            logger.info(
                "GATE ADJUSTED: %s %.4f → %.4f (fp_rate=%.2f, promoted=%d, probation=%d)",
                key, current, new_val, fp_rate,
                self._meta_promoted_total, probation_count,
            )

        if changed and self._bus is not None:
            from mae_core.market.channels import CH_GATE_ADJUSTED
            try:
                self._bus.publish(CH_GATE_ADJUSTED, {
                    "step": self._step_counter,
                    "fp_rate": fp_rate,
                    "promoted_total": self._meta_promoted_total,
                    "probation_count": probation_count,
                })
            except Exception:
                pass

    def _run_meta_learning(self) -> None:
        """Bridge 5: RSI Layer 3 — improve how the system discovers patterns.

        Three wires:
          Wire 1: Calibration → source_reliability. If ThompsonCalibrator reports
            overconfident sources, reduce their reliability in learning_config.
          Wire 2: Retirement rate → generator thresholds. If too many hypotheses
            get retired, tighten min_correlation to raise the quality bar.
          Wire 3: Pair quality feedback is handled by _promote/_retire calling
            generator.record_outcome() directly (not deferred to this method).
        """
        try:
            from mae_core.market.intelligence.learning_config import (
                LEARNING_CONFIG, update_config,
            )
        except ImportError:
            return

        changed = False

        # Wire 1: Calibration feedback → source_reliability adjustment
        if self._thompson_calibrator is not None:
            try:
                feedback = self._thompson_calibrator.get_calibration_feedback()
                reliability = LEARNING_CONFIG.get("source_reliability", {})

                for item in feedback.get("overconfident", []):
                    key = item["key"]
                    delta = item["delta"]  # Negative (reducing)
                    if key in reliability:
                        current = reliability[key]
                        new_val = max(0.10, min(0.95, current + delta))
                        if abs(new_val - current) > 1e-6:
                            update_config(
                                f"source_reliability.{key}",
                                round(new_val, 4),
                                modified_by="meta_learner_calibration",
                            )
                            changed = True
                            logger.info(
                                "META-LEARNING: source_reliability.%s %.4f → %.4f "
                                "(overconfident, gap=%.4f)",
                                key, current, new_val, item["gap"],
                            )

                for item in feedback.get("underconfident", []):
                    key = item["key"]
                    delta = item["delta"]  # Positive (increasing)
                    if key in reliability:
                        current = reliability[key]
                        new_val = max(0.10, min(0.95, current + delta))
                        if abs(new_val - current) > 1e-6:
                            update_config(
                                f"source_reliability.{key}",
                                round(new_val, 4),
                                modified_by="meta_learner_calibration",
                            )
                            changed = True
                            logger.info(
                                "META-LEARNING: source_reliability.%s %.4f → %.4f "
                                "(underconfident, gap=%.4f)",
                                key, current, new_val, item["gap"],
                            )
            except Exception:
                logger.debug("Meta-learning Wire 1 (calibration) failed", exc_info=True)

        # Wire 2: Retirement rate → generator threshold adaptation
        if len(self._retirement_window) >= 20:
            try:
                gen_thresholds = LEARNING_CONFIG.get("generator_thresholds", {})
                gen_bounds = gen_thresholds.get("_bounds", {})
                retired_count = self._retirement_window.count("retired")
                retirement_rate = retired_count / len(self._retirement_window)

                current_min_corr = gen_thresholds.get("min_correlation", 0.6)

                if retirement_rate > 0.70:
                    # Too many hypotheses being retired → raise the bar
                    new_val = current_min_corr + 0.02
                elif retirement_rate < 0.20:
                    # Very few retirements → can afford to explore more
                    new_val = current_min_corr - 0.01
                else:
                    new_val = current_min_corr  # No change

                # Clamp to bounds
                corr_bounds = gen_bounds.get("min_correlation", [0.4, 0.85])
                new_val = max(corr_bounds[0], min(corr_bounds[1], new_val))

                if abs(new_val - current_min_corr) > 1e-6:
                    update_config(
                        "generator_thresholds.min_correlation",
                        round(new_val, 4),
                        modified_by="meta_learner_retirement",
                    )
                    changed = True
                    logger.info(
                        "META-LEARNING: min_correlation %.4f → %.4f "
                        "(retirement_rate=%.2f, window=%d)",
                        current_min_corr, new_val,
                        retirement_rate, len(self._retirement_window),
                    )
            except Exception:
                logger.debug("Meta-learning Wire 2 (retirement rate) failed", exc_info=True)

        if changed and self._bus is not None:
            from mae_core.market.channels import CH_META_ADJUSTED
            try:
                self._bus.publish(CH_META_ADJUSTED, {
                    "step": self._step_counter,
                    "retirement_window_size": len(self._retirement_window),
                    "retirement_rate": (
                        self._retirement_window.count("retired") / len(self._retirement_window)
                        if self._retirement_window else 0.0
                    ),
                })
            except Exception:
                pass

    def get_statistics(self) -> dict:
        """Summary stats for HolonProxy.sense() and monitoring."""
        registry_stats = self._registry.get_statistics()
        return {
            "step_counter": self._step_counter,
            "signals_matched": self._signals_matched,
            "hypotheses_generated": self._hypotheses_generated,
            "hypotheses_promoted": self._hypotheses_promoted,
            "hypotheses_retired": self._hypotheses_retired,
            "validation_in_progress": (
                self._validation_future is not None
                and not self._validation_future.done()
            ),
            **registry_stats,
        }
