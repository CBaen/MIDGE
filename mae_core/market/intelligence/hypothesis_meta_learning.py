"""HypothesisMetaLearningMixin — persistence, regime, gate review, meta-learning.

Used by HypothesisEngine as a mixin. Expects the following instance attributes:
    self._registry, self._generator, self._validator, self._bus,
    self._regime_classifier, self._thompson_calibrator,
    self._step_counter, self._gate_review_cadence, self._gate_cooldowns,
    self._meta_promoted_total, self._meta_retired_after_active,
    self._retirement_window, self._retirement_window_max,
    self._retirement_window_path, self._meta_learning_cadence
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from mae_core.market.intelligence.hypothesis import HypothesisStatus

logger = logging.getLogger(__name__)


class HypothesisMetaLearningMixin:
    """Retirement window persistence, regime checks, gate review, meta-learning."""

    def _save_retirement_window(self) -> None:
        """Persist retirement window and meta-counters to disk.

        Does NOT persist _gate_cooldowns — those are session-scoped step indices
        that become meaningless after a restart. Non-critical: failures are logged
        but never raised.
        """
        tmp = self._retirement_window_path.with_suffix(".tmp")
        try:
            self._retirement_window_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "saved_at": datetime.now().isoformat(),
                "retirement_window": list(self._retirement_window),
                "meta_promoted_total": self._meta_promoted_total,
                "meta_retired_after_active": self._meta_retired_after_active,
            }
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._retirement_window_path)
        except Exception as e:
            logger.debug("Failed to save retirement window: %s", e)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _load_retirement_window(self) -> None:
        """Restore retirement window and meta-counters from disk.

        Trims the loaded window to _retirement_window_max in case the max was
        lowered since the last save. Safe to call when the file is absent.
        """
        if not self._retirement_window_path.exists():
            return
        try:
            data = json.loads(self._retirement_window_path.read_text())
            raw_window = data.get("retirement_window", [])
            # Backward compatibility: old format stored plain strings.
            # Treat all old string entries as seeded=True because we cannot
            # distinguish live from historical entries retrospectively.
            normalized: list[dict] = []
            for entry in raw_window:
                if isinstance(entry, str):
                    normalized.append({"outcome": entry, "seeded": True})
                elif isinstance(entry, dict):
                    normalized.append(entry)
                # Silently skip any other unexpected types
            # Trim to current max
            self._retirement_window = normalized[-self._retirement_window_max:]
            self._meta_promoted_total = int(data.get("meta_promoted_total", 0))
            self._meta_retired_after_active = int(data.get("meta_retired_after_active", 0))
            logger.info(
                "Loaded retirement window: %d entries, promoted=%d, retired_after_active=%d",
                len(self._retirement_window),
                self._meta_promoted_total,
                self._meta_retired_after_active,
            )
        except Exception as e:
            logger.warning("Failed to load retirement window: %s", e)

    def _seed_retirement_window_from_registry(self) -> None:
        """Cold-start fix: populate retirement window from registry state.

        Called when no persistence file exists so the retirement window isn't
        empty at startup. An empty window means Wire 2 of meta-learning has no
        signal to act on until enough live promotions/retirements accumulate,
        which can take many sessions. By reconstructing from the persisted
        registry we get a meaningful prior immediately.

        ACTIVE hypotheses were promoted at some point → "promoted".
        RETIRED hypotheses failed validation → "retired".
        PROBATION/HIBERNATED are excluded — their outcome is still unknown.

        Entries are sorted by created_at (oldest first) before appending so the
        window reflects the historical order of events rather than registry
        iteration order.
        """
        try:
            all_hyps = self._registry.get_all()
        except Exception:
            logger.debug("_seed_retirement_window_from_registry: get_all() failed", exc_info=True)
            return

        candidates = []
        for hyp in all_hyps:
            if hyp.status == HypothesisStatus.ACTIVE:
                candidates.append((hyp.created_at, "promoted"))
            elif hyp.status == HypothesisStatus.RETIRED:
                candidates.append((hyp.created_at, "retired"))
            # PROBATION and HIBERNATED are skipped — outcome unknown

        if not candidates:
            return

        # Sort oldest-first so the window reflects historical progression
        candidates.sort(key=lambda x: x[0])

        # Trim to max before appending (take the most recent N)
        if len(candidates) > self._retirement_window_max:
            candidates = candidates[-self._retirement_window_max:]

        for _, outcome in candidates:
            self._retirement_window.append({"outcome": outcome, "seeded": True})

        logger.info(
            "HypothesisEngine: seeded retirement window from registry — "
            "%d entries (%d promoted, %d retired)",
            len(self._retirement_window),
            sum(1 for e in self._retirement_window if e.get("outcome") == "promoted"),
            sum(1 for e in self._retirement_window if e.get("outcome") == "retired"),
        )

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

        # Case C: retire gate too conservative — loosen if fp_rate very low
        if (self._meta_promoted_total >= 5
                and fp_rate < 0.05
                and self._meta_retired_after_active == 0):
            adjustments.append(("retire_win_rate", -0.01))

        # Case D: couple promote_dsr to promote_win_rate direction
        for key, delta in list(adjustments):
            if key == "promote_win_rate" and delta > 0:
                adjustments.append(("promote_dsr", +0.05))
            elif key == "promote_win_rate" and delta < 0:
                adjustments.append(("promote_dsr", -0.05))

        # Case E: min_observations boundary retirement — if >40% of retirements
        # happen at exactly the min_observations threshold, the gate is too low:
        # hypotheses are being judged before they've accumulated enough evidence.
        # Raise min_observations by +5 to demand more data before any verdict.
        try:
            min_obs_gate = int(gates.get("min_observations", 20))
            retired_hyps = [
                h for h in self._registry.get_all()
                if h.status == HypothesisStatus.RETIRED
            ]
            if len(retired_hyps) >= 5:
                boundary_retirements = sum(
                    1 for h in retired_hyps
                    if h.stats.total_observations == min_obs_gate
                )
                boundary_fraction = boundary_retirements / len(retired_hyps)
                if boundary_fraction > 0.40:
                    adjustments.append(("min_observations", +5))
        except Exception:
            pass

        # Apply adjustments with cooldown and bounds
        cooldown_steps = self._gate_review_cadence * 5  # 5 review cycles
        changed = False

        for key, delta in adjustments:
            last_adjusted = self._gate_cooldowns.get(key)
            if last_adjusted is not None and self._step_counter - last_adjusted < cooldown_steps:
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
        # Only count live (seeded=False) entries so cold-start historical data
        # cannot bias min_correlation before real session outcomes accumulate.
        live_entries = [e for e in self._retirement_window if not e.get("seeded", False)]
        if len(live_entries) >= 10:
            try:
                gen_thresholds = LEARNING_CONFIG.get("generator_thresholds", {})
                gen_bounds = gen_thresholds.get("_bounds", {})
                retired_count = sum(
                    1 for e in live_entries if e.get("outcome") == "retired"
                )
                retirement_rate = retired_count / len(live_entries)

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
                        "(retirement_rate=%.2f, live_window=%d, total_window=%d)",
                        current_min_corr, new_val,
                        retirement_rate, len(live_entries), len(self._retirement_window),
                    )
            except Exception:
                logger.debug("Meta-learning Wire 2 (retirement rate) failed", exc_info=True)

        if changed and self._bus is not None:
            from mae_core.market.channels import CH_META_ADJUSTED
            try:
                live_window = [
                    e for e in self._retirement_window if not e.get("seeded", False)
                ]
                self._bus.publish(CH_META_ADJUSTED, {
                    "step": self._step_counter,
                    "retirement_window_size": len(self._retirement_window),
                    "live_window_size": len(live_window),
                    "retirement_rate": (
                        sum(1 for e in live_window if e.get("outcome") == "retired")
                        / len(live_window)
                        if live_window else 0.0
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
