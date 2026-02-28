"""Tests for Bridge 4 — Dynamic Quality Gates.

Tests that hypothesis promotion/retirement thresholds are read from
learning_config at runtime, adjustable via update_config(), clamped
to bounds, and self-tuned by _review_gates().
"""

import pytest
from unittest.mock import MagicMock, patch

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine
from mae_core.market.intelligence.hypothesis_validator import (
    HypothesisValidator,
    ValidationResult,
    _get_gate,
    _GATE_FALLBACKS,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


@pytest.fixture
def engine(registry):
    return HypothesisEngine(
        registry=registry,
        generator=MagicMock(generate=MagicMock(return_value=[])),
        validator=MagicMock(),
    )


# ── _get_gate reads from config ─────────────────────────────────────


class TestGetGate:
    def test_returns_fallback_when_config_missing(self):
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {},
            create=True,
        ):
            for key, expected in _GATE_FALLBACKS.items():
                result = _get_gate(key)
                assert result == float(expected), f"{key}: expected {expected}, got {result}"

    def test_reads_from_learning_config(self):
        mock_config = {
            "hypothesis_gates": {
                "promote_win_rate": 0.60,
                "_regime_deltas": {"default": {}},
            }
        }
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            mock_config,
            create=True,
        ):
            assert _get_gate("promote_win_rate") == 0.60

    def test_applies_regime_delta(self):
        mock_config = {
            "hypothesis_gates": {
                "promote_win_rate": 0.52,
                "_regime_deltas": {
                    "volatile": {"promote_win_rate": 0.03},
                    "default": {},
                },
            }
        }
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            mock_config,
            create=True,
        ):
            assert _get_gate("promote_win_rate", regime="volatile") == pytest.approx(0.55)

    def test_default_regime_has_no_delta(self):
        mock_config = {
            "hypothesis_gates": {
                "promote_dsr": 0.5,
                "_regime_deltas": {
                    "volatile": {"promote_dsr": 0.1},
                    "default": {},
                },
            }
        }
        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            mock_config,
            create=True,
        ):
            assert _get_gate("promote_dsr", regime="default") == 0.5


# ── _review_gates self-tuning ────────────────────────────────────────


class TestReviewGates:
    def test_tightens_on_high_false_positive_rate(self, engine):
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 5  # FP rate = 50% > 30%
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()
            mock_update.assert_called_once()
            call_args = mock_update.call_args
            assert call_args[0][0] == "hypothesis_gates.promote_win_rate"
            assert call_args[0][1] == pytest.approx(0.53)
            assert call_args[1]["modified_by"] == "gate_reviewer"

    def test_loosens_when_zero_promotions_and_candidates_exist(self, engine, registry):
        engine._meta_promoted_total = 0
        engine._step_counter = 5000
        engine._gate_review_cadence = 2000

        # Create 3 probation hypotheses so the gate recognizes candidates
        for i in range(3):
            hyp = Hypothesis(
                name=f"test_{i}",
                trigger=TriggerPattern(source_a=f"a{i}", source_b=f"b{i}", lag_days=10),
                causal_story="Test.",
            )
            registry.register(hyp)

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.55,
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()
            mock_update.assert_called_once()
            assert mock_update.call_args[0][1] == pytest.approx(0.54)

    def test_no_change_in_healthy_range(self, engine):
        engine._meta_promoted_total = 5
        engine._meta_retired_after_active = 1  # FP rate = 20%, healthy
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()
            mock_update.assert_not_called()

    def test_clamps_to_bounds(self, engine):
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 5
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.65,  # Already at upper bound
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()
            # Should not adjust because 0.65 + 0.01 = 0.66 clamps back to 0.65
            mock_update.assert_not_called()

    def test_cooldown_prevents_rapid_adjustment(self, engine):
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 5
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000
        # Set cooldown as if we just adjusted 1000 steps ago
        engine._gate_cooldowns["promote_win_rate"] = 3000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ) as mock_update:
            engine._review_gates()
            mock_update.assert_not_called()

    def test_publishes_ch_gate_adjusted(self, registry):
        bus = MagicMock()
        engine = HypothesisEngine(
            registry=registry,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
            bus=bus,
        )
        engine._meta_promoted_total = 10
        engine._meta_retired_after_active = 5
        engine._step_counter = 4000
        engine._gate_review_cadence = 2000

        with patch(
            "mae_core.market.intelligence.learning_config.LEARNING_CONFIG",
            {
                "hypothesis_gates": {
                    "promote_win_rate": 0.52,
                    "_bounds": {"promote_win_rate": [0.50, 0.65]},
                    "_regime_deltas": {"default": {}},
                },
            },
            create=True,
        ), patch(
            "mae_core.market.intelligence.learning_config.update_config",
        ):
            engine._review_gates()
            # Find the CH_GATE_ADJUSTED publish call
            gate_calls = [
                c for c in bus.publish.call_args_list
                if "gate_adjusted" in str(c)
            ]
            assert len(gate_calls) == 1


# ── Promote/retire track meta counters ───────────────────────────────


class TestMetaTracking:
    def test_promote_increments_meta_promoted_total(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._promote(hyp)
        assert engine._meta_promoted_total == 1

    def test_retire_active_increments_retired_after_active(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        registry.promote(hyp.hypothesis_id)
        hyp.status = HypothesisStatus.ACTIVE
        engine._retire(hyp, "Poor performance")
        assert engine._meta_retired_after_active == 1

    def test_retire_probation_does_not_increment_retired_after_active(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._retire(hyp, "Poor performance")
        assert engine._meta_retired_after_active == 0

    def test_gate_review_cadence_fires_in_step(self, engine):
        engine._gate_review_cadence = 10
        review_count = [0]
        original = engine._review_gates

        def counting():
            review_count[0] += 1
            original()

        engine._review_gates = counting
        for _ in range(20):
            engine.step()
        assert review_count[0] == 2


# ── Legacy constants still importable ────────────────────────────────


class TestBackwardCompatibility:
    def test_legacy_constants_exist(self):
        from mae_core.market.intelligence.hypothesis_validator import (
            MIN_OBSERVATIONS,
            PROMOTE_WIN_RATE,
            PROMOTE_DSR,
            RETIRE_WIN_RATE,
            RETIRE_DSR,
        )
        assert MIN_OBSERVATIONS == 20
        assert PROMOTE_WIN_RATE == 0.52
        assert PROMOTE_DSR == 0.5
        assert RETIRE_WIN_RATE == 0.45
        assert RETIRE_DSR == 0.0
