"""Tests for HypothesisEngine — RSI Layer 2 orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator
from mae_core.market.intelligence.hypothesis_engine import HypothesisEngine


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def signals_dir(tmp_path):
    d = tmp_path / "signals"
    d.mkdir()
    return d


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


@pytest.fixture
def generator(registry, tmp_data_dir):
    return HypothesisGenerator(
        registry=registry,
        lag_data_path=tmp_data_dir / "lag_correlations.json",
    )


@pytest.fixture
def validator(tmp_data_dir, signals_dir):
    return HypothesisValidator(
        signals_dir=signals_dir,
        data_dir=tmp_data_dir,
    )


@pytest.fixture
def engine(registry, generator, validator):
    return HypothesisEngine(
        registry=registry,
        generator=generator,
        validator=validator,
    )


class TestStepCadence:
    def test_step_increments_counter(self, engine):
        engine.step()
        assert engine._step_counter == 1
        engine.step()
        assert engine._step_counter == 2

    def test_generation_fires_on_cadence(self, engine):
        engine._generation_cadence = 5
        gen_call_count = [0]
        original = engine._run_generation

        def counting_gen():
            gen_call_count[0] += 1
            original()

        engine._run_generation = counting_gen
        for _ in range(10):
            engine.step()
        assert gen_call_count[0] == 2  # step 5 and step 10

    def test_validation_fires_on_cadence(self, engine):
        engine._validation_cadence = 10
        val_call_count = [0]
        original = engine._run_validation

        def counting_val():
            val_call_count[0] += 1
            original()

        engine._run_validation = counting_val
        for _ in range(20):
            engine.step()
        # Validation runs in background thread — wait for completion
        if engine._validation_future is not None:
            engine._validation_future.result(timeout=5)
        # Collect so second launch isn't blocked
        engine._collect_validation_results()
        # One more step to trigger collection of the last future
        engine.step()
        if engine._validation_future is not None:
            engine._validation_future.result(timeout=5)
        assert val_call_count[0] == 2


class TestSignalMatching:
    def test_on_signal_ingested_matches_active(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="sec_form4", source_b="finnhub_earnings",
                lag_days=45, direction="positive",
            ),
            causal_story="Test story.",
        )
        registry.register(hyp)
        registry.promote(hyp.hypothesis_id)

        engine.on_signal_ingested("market.sensing.signal_ingested", {
            "source": "sec_form4",
            "symbol": "AAPL",
        })
        assert engine._signals_matched == 1

    def test_on_signal_ingested_ignores_non_matching(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="sec_form4", source_b="finnhub_earnings",
                lag_days=45, direction="positive",
            ),
            causal_story="Test story.",
        )
        registry.register(hyp)
        registry.promote(hyp.hypothesis_id)

        engine.on_signal_ingested("market.sensing.signal_ingested", {
            "source": "congressional",  # doesn't match sec_form4
            "symbol": "AAPL",
        })
        assert engine._signals_matched == 0


class TestLifecycleTransitions:
    def test_promote_increments_counter(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="sec_form4", source_b="finnhub_earnings",
                lag_days=45,
            ),
            causal_story="Insiders trade before earnings.",
        )
        registry.register(hyp)
        engine._promote(hyp)
        assert engine._hypotheses_promoted == 1
        assert registry.get(hyp.hypothesis_id).status == HypothesisStatus.ACTIVE

    def test_retire_increments_counter(self, engine, registry):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(source_a="a", source_b="b", lag_days=10),
            causal_story="Test.",
        )
        registry.register(hyp)
        engine._retire(hyp, "Win rate too low")
        assert engine._hypotheses_retired == 1
        assert registry.get(hyp.hypothesis_id).status == HypothesisStatus.RETIRED


class TestStatistics:
    def test_get_statistics(self, engine):
        stats = engine.get_statistics()
        assert "step_counter" in stats
        assert "signals_matched" in stats
        assert "total_hypotheses" in stats
        assert "active_count" in stats


class TestEventBusIntegration:
    def test_signal_ingested_publishes_fired(self, registry):
        bus = MagicMock()
        engine = HypothesisEngine(
            registry=registry,
            generator=MagicMock(),
            validator=MagicMock(),
            bus=bus,
        )

        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="sec_form4", source_b="finnhub_earnings",
                lag_days=45,
            ),
            causal_story="Test.",
        )
        registry.register(hyp)
        registry.promote(hyp.hypothesis_id)

        engine.on_signal_ingested("ch", {"source": "sec_form4", "symbol": "AAPL"})
        bus.publish.assert_called()
        call_args = bus.publish.call_args
        assert "hypothesis.fired" in call_args[0][0]


class TestBacktestAnalyzerIntegration:
    """Tests for Bridge 1 — backtest_analyzer wiring in HypothesisEngine."""

    def test_backtest_analyzer_called_in_generation(self, registry):
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        engine = HypothesisEngine(
            registry=registry,
            generator=MagicMock(generate=MagicMock(return_value=[])),
            validator=MagicMock(),
            backtest_analyzer=mock_analyzer,
        )
        engine._run_generation()
        mock_analyzer.analyze.assert_called_once()

    def test_granular_thompson_key_on_promote(self, registry):
        mock_sampler = MagicMock()
        mock_sampler.distributions = {}
        engine = HypothesisEngine(
            registry=registry,
            generator=MagicMock(),
            validator=MagicMock(),
            thompson_sampler=mock_sampler,
        )
        hyp = Hypothesis(
            name="Sweep:CL=F:bearish",
            trigger=TriggerPattern(
                source_a="session_sweep", source_b="price_outcome",
                lag_days=0, direction="bearish",
                domain_filter="CL=F:bearish",
            ),
            stats=HypothesisStats(wins=13, losses=9, total_observations=22),
            causal_story="ICT session sweep pattern.",
            source_type=SourceType.BACKTEST_DERIVED,
        )
        registry.register(hyp)
        engine._promote(hyp)
        # Thompson key should be granular: sweep_bt:CL=F:bearish
        assert "sweep_bt:CL=F:bearish" in mock_sampler.distributions
        dist = mock_sampler.distributions["sweep_bt:CL=F:bearish"]["default"]
        assert dist["alpha"] == 13 + 1.1  # wins + conservative prior
        assert dist["beta"] == 9 + 0.9    # losses + conservative prior

    def test_lag_correlation_path_unchanged(self, registry):
        """Existing lag-correlation generation still works with backtest_analyzer present."""
        mock_generator = MagicMock()
        mock_generator.generate.return_value = []
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = []
        engine = HypothesisEngine(
            registry=registry,
            generator=mock_generator,
            validator=MagicMock(),
            backtest_analyzer=mock_analyzer,
        )
        engine._run_generation()
        mock_generator.generate.assert_called_once()
        mock_analyzer.analyze.assert_called_once()
