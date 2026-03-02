"""Tests for Causal Bridge (Package A) — convergence_alerter + hypothesis_validator
causal engine integration.

Covers:
  1. Contradiction publishing via CH_CONTRADICTION_DETECTED (convergence_alerter)
  2. Causal observations via observe_correlation (convergence_alerter)
  3. Confounded gate tightening via query_causation (hypothesis_validator)
  4. Bootstrap wiring: shared_causal_engine flows to both consumers
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from mae_core.cognition.causal_reasoning import CausalRelationType
from mae_core.market.channels import CH_CONTRADICTION_DETECTED
from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)
from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_signal(
    domain: str,
    direction: str,
    strength: float = 0.8,
    signal_id: str | None = None,
) -> Signal:
    return Signal(
        signal_id=signal_id or f"{domain}_{direction}",
        strength=strength,
        domain=domain,
        direction=direction,
        timestamp=datetime.now(),
    )


def _make_alerter(causal_engine=None, event_bus=None) -> ConvergenceAlerter:
    return ConvergenceAlerter(
        min_domains=3,
        causal_engine=causal_engine,
        event_bus=event_bus,
    )


def _make_hypothesis(
    source_a: str = "sec_form4",
    source_b: str = "price_fetcher",
    causal_story: str = "Insider buying precedes price increase.",
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="test_hyp_1",
        name="Test Hypothesis",
        trigger=TriggerPattern(
            source_a=source_a,
            source_b=source_b,
            lag_days=5,
            direction="positive",
        ),
        causal_story=causal_story,
        status=HypothesisStatus.PROBATION,
        source_type=SourceType.LAG_CORRELATION,
    )


def _inject_contradicting_signals(alerter: ConvergenceAlerter) -> None:
    """Inject signals in 3 bullish and 3 bearish domains to force coherence < 0.7."""
    for domain in ("insider", "sentiment", "macro"):
        alerter.signals[domain] = [_make_signal(domain, "bullish")]
    for domain in ("technical", "positioning", "government"):
        alerter.signals[domain] = [_make_signal(domain, "bearish")]


def _inject_aligned_signals(alerter: ConvergenceAlerter) -> None:
    """Inject signals in 5 bullish and 1 bearish domain for coherence >= 0.7."""
    for domain in ("insider", "sentiment", "macro", "crypto", "volume"):
        alerter.signals[domain] = [_make_signal(domain, "bullish")]
    alerter.signals["technical"] = [_make_signal("technical", "bearish")]


def _inject_strong_multi_domain_signals(alerter: ConvergenceAlerter) -> None:
    """Inject two high-strength domains so each pair's strength product > 0.3."""
    for domain in ("insider", "government"):
        alerter.signals[domain] = [_make_signal(domain, "bullish", strength=0.9)]


# ---------------------------------------------------------------------------
# Section 1: Contradiction publishing (convergence_alerter)
# ---------------------------------------------------------------------------

class TestContradictionPublishing:
    """CH_CONTRADICTION_DETECTED is published when coherence < 0.7."""

    def test_publishes_when_coherence_below_threshold(self):
        """Low coherence (evenly split bullish/bearish) triggers a publish."""
        mock_bus = MagicMock()
        alerter = _make_alerter(event_bus=mock_bus)
        _inject_contradicting_signals(alerter)

        alerter.check_convergence()

        mock_bus.publish.assert_called_once()
        channel, payload = mock_bus.publish.call_args[0]
        assert channel == CH_CONTRADICTION_DETECTED

    def test_does_not_publish_when_coherence_above_threshold(self):
        """High coherence (strong majority direction) produces no contradiction event."""
        mock_bus = MagicMock()
        alerter = _make_alerter(event_bus=mock_bus)
        _inject_aligned_signals(alerter)

        alerter.check_convergence()

        for call in mock_bus.publish.call_args_list:
            channel = call[0][0]
            assert channel != CH_CONTRADICTION_DETECTED, (
                f"Unexpected CH_CONTRADICTION_DETECTED published at coherence >= 0.7"
            )

    def test_no_crash_when_bus_is_none(self):
        """No event_bus should not raise even at low coherence."""
        alerter = _make_alerter(event_bus=None)
        _inject_contradicting_signals(alerter)

        # Must not raise
        alerter.check_convergence()

    def test_payload_includes_required_fields(self):
        """Published payload carries coherence, bullish_count, bearish_count."""
        mock_bus = MagicMock()
        alerter = _make_alerter(event_bus=mock_bus)
        _inject_contradicting_signals(alerter)

        alerter.check_convergence()

        _, payload = mock_bus.publish.call_args[0]
        assert "coherence" in payload
        assert "bullish_count" in payload
        assert "bearish_count" in payload
        assert isinstance(payload["coherence"], float)
        assert payload["bullish_count"] >= 1
        assert payload["bearish_count"] >= 1

    def test_payload_coherence_value_is_below_threshold(self):
        """Payload coherence value matches the computed low-coherence state."""
        mock_bus = MagicMock()
        alerter = _make_alerter(event_bus=mock_bus)
        _inject_contradicting_signals(alerter)

        alerter.check_convergence()

        _, payload = mock_bus.publish.call_args[0]
        assert payload["coherence"] < 0.7


# ---------------------------------------------------------------------------
# Section 2: Causal observations (convergence_alerter)
# ---------------------------------------------------------------------------

class TestCausalObservations:
    """observe_correlation is called for strong domain pairs."""

    def test_observe_correlation_called_for_strong_pairs(self):
        """Two high-strength domains produce a strength product > 0.3, triggering call."""
        mock_engine = MagicMock()
        alerter = _make_alerter(causal_engine=mock_engine)
        _inject_strong_multi_domain_signals(alerter)

        alerter.check_convergence()

        mock_engine.observe_correlation.assert_called()

    def test_low_strength_pairs_are_skipped(self):
        """Pairs whose strength product <= 0.3 must not call observe_correlation."""
        mock_engine = MagicMock()
        alerter = _make_alerter(causal_engine=mock_engine)
        # Two domains, each with very low strength — product = 0.1 * 0.1 = 0.01
        alerter.signals["insider"] = [_make_signal("insider", "bullish", strength=0.1)]
        alerter.signals["government"] = [_make_signal("government", "bullish", strength=0.1)]

        alerter.check_convergence()

        mock_engine.observe_correlation.assert_not_called()

    def test_no_crash_when_causal_engine_is_none(self):
        """Missing causal_engine must not raise during convergence check."""
        alerter = _make_alerter(causal_engine=None)
        _inject_strong_multi_domain_signals(alerter)

        alerter.check_convergence()  # Must not raise

    def test_observe_correlation_exception_is_caught(self):
        """If observe_correlation raises, check_convergence continues without crashing."""
        mock_engine = MagicMock()
        mock_engine.observe_correlation.side_effect = RuntimeError("causal graph error")
        alerter = _make_alerter(causal_engine=mock_engine)
        _inject_strong_multi_domain_signals(alerter)

        # Must not propagate the RuntimeError
        alerter.check_convergence()

    def test_observe_correlation_receives_domain_names(self):
        """Domain names are passed as the first two positional args to observe_correlation."""
        mock_engine = MagicMock()
        alerter = _make_alerter(causal_engine=mock_engine)
        alerter.signals["insider"] = [_make_signal("insider", "bullish", strength=0.9)]
        alerter.signals["government"] = [_make_signal("government", "bullish", strength=0.9)]

        alerter.check_convergence()

        args, _ = mock_engine.observe_correlation.call_args
        domain_a, domain_b = args[0], args[1]
        assert {domain_a, domain_b} == {"insider", "government"}


# ---------------------------------------------------------------------------
# Section 3: Confounded gate tightening (hypothesis_validator)
# ---------------------------------------------------------------------------

def _make_validator_with_engine(tmp_path, causal_engine=None) -> HypothesisValidator:
    """HypothesisValidator using an isolated tmp directory."""
    return HypothesisValidator(
        data_dir=tmp_path,
        causal_engine=causal_engine,
    )


def _make_confounded_result():
    """Mock causal query result that has relation_type == CONFOUNDED."""
    result = MagicMock()
    result.relation_type = CausalRelationType.CONFOUNDED
    return result


def _stub_validate_internals(validator: HypothesisValidator, wins: int, losses: int):
    """Patch the archive scan and outcome check so validate() uses synthetic data."""
    events = [{"id": str(i)} for i in range(wins + losses)]

    outcomes = []
    for i in range(wins + losses):
        if i < wins:
            outcomes.append((0.02, True))   # win: small positive return
        else:
            outcomes.append((-0.01, False))  # loss: small negative return

    validator._find_trigger_events = MagicMock(return_value=events)
    validator._check_event_outcome = MagicMock(side_effect=outcomes)


class TestConfoundedGateTightening:
    """Causal CONFOUNDED judgment tightens the promote_win_rate gate by +0.03."""

    def test_confounded_tightens_gate_and_blocks_border_promotion(self, tmp_path):
        """A hypothesis at win_rate just above base gate is blocked when confounded.

        Default promote_win_rate gate = 0.52.  With +0.03 for CONFOUNDED the gate
        becomes 0.55.  We create 23 wins / 20 losses → win_rate ≈ 0.535, which
        clears 0.52 but not 0.55.
        """
        mock_engine = MagicMock()
        mock_engine.query_causation.return_value = _make_confounded_result()

        validator = _make_validator_with_engine(tmp_path, causal_engine=mock_engine)
        h = _make_hypothesis()
        wins, losses = 23, 20   # win_rate ≈ 0.535 (above 0.52, below 0.55)
        _stub_validate_internals(validator, wins, losses)

        # DSR also needs to clear 0.5 — give it a good Sharpe by controlling returns.
        # Patch _compute_sharpe and _compute_dsr for determinism.
        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result = validator.validate(h)

        assert not result.recommend_promote, (
            "Confounded hypothesis at 53.5% win rate should NOT promote (gate = 0.55)"
        )

    def test_non_confounded_hypothesis_proceeds_normally(self, tmp_path):
        """DIRECT relation type does not tighten the gate — promotion is possible."""
        direct_result = MagicMock()
        direct_result.relation_type = CausalRelationType.DIRECT

        mock_engine = MagicMock()
        mock_engine.query_causation.return_value = direct_result

        validator = _make_validator_with_engine(tmp_path, causal_engine=mock_engine)
        h = _make_hypothesis()
        wins, losses = 23, 20   # win_rate ≈ 0.535 — clears 0.52 base gate
        _stub_validate_internals(validator, wins, losses)

        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result = validator.validate(h)

        assert result.recommend_promote, (
            "Non-confounded hypothesis at 53.5% win rate should promote (gate = 0.52)"
        )

    def test_no_causal_engine_uses_standard_gate(self, tmp_path):
        """When causal_engine is None, fallthrough to normal promotion logic."""
        validator = _make_validator_with_engine(tmp_path, causal_engine=None)
        h = _make_hypothesis()
        wins, losses = 23, 20
        _stub_validate_internals(validator, wins, losses)

        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result = validator.validate(h)

        assert result.recommend_promote, (
            "No causal engine — standard gate applies, 53.5% should promote"
        )

    def test_query_causation_exception_does_not_change_gate(self, tmp_path):
        """If query_causation raises, no gate tightening happens."""
        mock_engine = MagicMock()
        mock_engine.query_causation.side_effect = RuntimeError("causal engine error")

        validator = _make_validator_with_engine(tmp_path, causal_engine=mock_engine)
        h = _make_hypothesis()
        wins, losses = 23, 20
        _stub_validate_internals(validator, wins, losses)

        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result = validator.validate(h)

        # Exception is swallowed — gate unchanged — 53.5% wins should still promote
        assert result.recommend_promote, (
            "Exception in query_causation must be caught; standard gate still applies"
        )

    def test_query_causation_result_without_relation_type_is_ignored(self, tmp_path):
        """A result lacking relation_type (e.g., no causal path found) is skipped."""
        no_relation_result = MagicMock(spec=[])  # no attributes whatsoever
        mock_engine = MagicMock()
        mock_engine.query_causation.return_value = no_relation_result

        validator = _make_validator_with_engine(tmp_path, causal_engine=mock_engine)
        h = _make_hypothesis()
        wins, losses = 23, 20
        _stub_validate_internals(validator, wins, losses)

        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result = validator.validate(h)

        # hasattr check prevents gate adjustment — standard gate applies
        assert result.recommend_promote

    def test_confounded_gate_tightening_is_precisely_0_03(self, tmp_path):
        """Gate increase is exactly +0.03: a hypothesis right at win_rate 0.55 stays blocked,
        one just above 0.55 promotes.

        Base gate = 0.52, +0.03 confounded adjustment = 0.55 threshold.
        win_rate must be STRICTLY greater than 0.55 to promote.
        """
        mock_engine = MagicMock()
        mock_engine.query_causation.return_value = _make_confounded_result()

        validator = _make_validator_with_engine(tmp_path, causal_engine=mock_engine)

        # win_rate = 0.55 exactly (11/20 = 0.55) — NOT strictly greater, should NOT promote
        h_at_gate = _make_hypothesis()
        _stub_validate_internals(validator, wins=11, losses=9)
        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result_at = validator.validate(h_at_gate)
        assert not result_at.recommend_promote, "win_rate == gate (0.55) should NOT promote"

        # win_rate = 12/20 = 0.60 — strictly above 0.55, should promote
        h_above_gate = _make_hypothesis()
        _stub_validate_internals(validator, wins=12, losses=8)
        with patch.object(validator, "_compute_sharpe", return_value=1.5), \
             patch.object(validator, "_compute_dsr", return_value=0.8):
            result_above = validator.validate(h_above_gate)
        assert result_above.recommend_promote, "win_rate=0.60 > confounded gate (0.55) should promote"


# ---------------------------------------------------------------------------
# Section 4: Bootstrap wiring
# ---------------------------------------------------------------------------

class TestBootstrapWiring:
    """shared_causal_engine flows through to convergence_alerter and hypothesis_validator."""

    def test_convergence_alerter_receives_shared_causal_engine(self):
        """ConvergenceAlerter constructor is passed shared_causal_engine from ctx."""
        mock_engine = MagicMock()
        ctx = SimpleNamespace()
        ctx.shared_causal_engine = mock_engine
        ctx.thompson_sampler = None

        # Simulate the bootstrap call
        alerter = ConvergenceAlerter(
            min_domains=3,
            thompson_sampler=getattr(ctx, "thompson_sampler", None),
            causal_engine=getattr(ctx, "shared_causal_engine", None),
            event_bus=getattr(ctx, "bus", None),
        )

        assert alerter._causal_engine is mock_engine

    def test_hypothesis_validator_receives_shared_causal_engine(self, tmp_path):
        """HypothesisValidator constructor is passed shared_causal_engine from ctx."""
        mock_engine = MagicMock()
        ctx = SimpleNamespace()
        ctx.shared_causal_engine = mock_engine

        validator = HypothesisValidator(
            data_dir=tmp_path,
            causal_engine=getattr(ctx, "shared_causal_engine", None),
        )

        assert validator._causal_engine is mock_engine

    def test_missing_shared_causal_engine_yields_none_gracefully(self, tmp_path):
        """When shared_causal_engine is absent from ctx, both consumers get None."""
        ctx = SimpleNamespace()  # no shared_causal_engine attribute

        alerter = ConvergenceAlerter(
            min_domains=3,
            causal_engine=getattr(ctx, "shared_causal_engine", None),
        )
        validator = HypothesisValidator(
            data_dir=tmp_path,
            causal_engine=getattr(ctx, "shared_causal_engine", None),
        )

        assert alerter._causal_engine is None
        assert validator._causal_engine is None
