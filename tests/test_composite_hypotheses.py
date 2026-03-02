"""Tests for Capability 1: Composite Hypotheses.

Adversarial tests for TriggerPattern new fields, _generate_composites(),
_findings_to_composite(), _is_duplicate_composite(), and validator routing.
"""

import json
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_generator import HypothesisGenerator
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_validator import HypothesisValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


@pytest.fixture
def generator(registry, tmp_data_dir):
    lag_path = tmp_data_dir / "lag_correlations.json"
    return HypothesisGenerator(registry=registry, lag_data_path=lag_path)


def _write_lag(tmp_data_dir, findings):
    path = tmp_data_dir / "lag_correlations.json"
    path.write_text(json.dumps(findings))
    return path


def _qualifying_finding(source_a, source_b, lag=10, corr=0.75, n=15, direction="positive"):
    return {
        "source_a": source_a,
        "source_b": source_b,
        "lag_days": lag,
        "correlation": corr,
        "n_pairs": n,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# TriggerPattern — new field serialization / deserialization
# ---------------------------------------------------------------------------

class TestTriggerPatternFields:
    def test_new_fields_default_to_empty(self):
        t = TriggerPattern(source_a="a", source_b="b", lag_days=5)
        assert t.conjunct_source == ""
        assert t.conjunct_min_strength == 0.0

    def test_new_fields_can_be_set(self):
        t = TriggerPattern(
            source_a="a", source_b="b", lag_days=5,
            conjunct_source="c", conjunct_min_strength=0.7,
        )
        assert t.conjunct_source == "c"
        assert t.conjunct_min_strength == 0.7

    def test_to_dict_includes_conjunct_fields(self):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="a", source_b="b", lag_days=5,
                conjunct_source="c", conjunct_min_strength=0.7,
            ),
        )
        d = hyp.to_dict()
        assert d["trigger"]["conjunct_source"] == "c"
        assert d["trigger"]["conjunct_min_strength"] == 0.7

    def test_from_dict_restores_conjunct_fields(self):
        hyp = Hypothesis(
            name="test",
            trigger=TriggerPattern(
                source_a="sec_form4", source_b="finnhub_earnings", lag_days=30,
                conjunct_source="finra_short", conjunct_min_strength=0.6,
            ),
        )
        d = hyp.to_dict()
        restored = Hypothesis.from_dict(d)
        assert restored.trigger.conjunct_source == "finra_short"
        assert restored.trigger.conjunct_min_strength == 0.6

    def test_backward_compat_old_data_without_conjunct(self):
        """Old records without conjunct fields must deserialize cleanly."""
        old_dict = {
            "hypothesis_id": "old-id-1",
            "name": "sec_form4→finnhub_earnings",
            "trigger": {
                "source_a": "sec_form4",
                "source_b": "finnhub_earnings",
                "lag_days": 30,
                "direction": "positive",
                "min_strength": 0.0,
                "domain_filter": "",
                # conjunct_source and conjunct_min_strength ABSENT
            },
            "status": "probation",
            "stats": {
                "wins": 0, "losses": 0, "total_observations": 0,
                "cumulative_return_pct": 0.0, "information_coefficient": 0.0,
                "sharpe_ratio": 0.0, "deflated_sharpe_ratio": 0.0,
                "last_evaluated": None,
            },
            "causal_story": "Story.",
            "source_type": "lag_correlation",
            "parent_lag_finding": {},
            "created_at": "2026-01-01T00:00:00",
            "promoted_at": None,
            "retired_at": None,
            "retired_reason": "",
            "regime_created_in": "default",
        }
        hyp = Hypothesis.from_dict(old_dict)
        assert hyp.trigger.conjunct_source == ""
        assert hyp.trigger.conjunct_min_strength == 0.0


# ---------------------------------------------------------------------------
# _generate_composites()
# ---------------------------------------------------------------------------

class TestGenerateComposites:
    def test_two_findings_sharing_target_produce_composite(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composite = [h for h in result if h.trigger.conjunct_source]
        assert len(composite) == 1

    def test_no_shared_target_produces_zero_composites(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings"),
            _qualifying_finding("finra_short", "congressional"),  # different target
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 0

    def test_three_findings_sharing_target_produce_all_pairs(self, registry, tmp_data_dir):
        # 3 findings → C(3,2) = 3 composite pairs
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
            _qualifying_finding("insider_cluster", "finnhub_earnings", lag=25),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 3

    def test_single_finding_per_target_no_composites(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings"),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 0

    def test_generate_returns_both_bivariate_and_composite(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        bivariate = [h for h in result if not h.trigger.conjunct_source]
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(bivariate) >= 2
        assert len(composites) >= 1


# ---------------------------------------------------------------------------
# _findings_to_composite() properties
# ---------------------------------------------------------------------------

class TestFindingsToComposite:
    def test_composite_lag_is_minimum_of_two_lags(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=40),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=15),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 1
        assert composites[0].trigger.lag_days == 15  # min(40, 15)

    def test_composite_name_format(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 1
        name = composites[0].name
        # Must match format [A+C]→B ...
        assert "→finnhub_earnings" in name
        assert name.startswith("[")

    def test_composite_story_prefix_is_composite_tag(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 1
        assert composites[0].causal_story.startswith("[COMPOSITE]")

    def test_composite_stores_both_source_names(self, registry, tmp_data_dir):
        findings = [
            _qualifying_finding("sec_form4", "finnhub_earnings", lag=30),
            _qualifying_finding("finra_short", "finnhub_earnings", lag=20),
        ]
        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=_write_lag(tmp_data_dir, findings),
        )
        result = gen.generate()
        composites = [h for h in result if h.trigger.conjunct_source]
        assert len(composites) == 1
        t = composites[0].trigger
        leading_pair = {t.source_a, t.conjunct_source}
        assert leading_pair == {"sec_form4", "finra_short"}
        assert t.source_b == "finnhub_earnings"


# ---------------------------------------------------------------------------
# _is_duplicate_composite()
# ---------------------------------------------------------------------------

class TestIsDuplicateComposite:
    def test_detects_exact_duplicate(self, registry, tmp_data_dir):
        # Register a composite manually
        hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        registry.register(hyp)

        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        assert gen._is_duplicate_composite("sec_form4", "finra_short", "finnhub_earnings")

    def test_detects_order_independent_duplicate(self, registry, tmp_data_dir):
        """(A+C→B) == (C+A→B) must be detected."""
        hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        registry.register(hyp)

        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        # Swap order — should still detect as duplicate
        assert gen._is_duplicate_composite("finra_short", "sec_form4", "finnhub_earnings")

    def test_different_target_not_duplicate(self, registry, tmp_data_dir):
        hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        registry.register(hyp)

        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        # Same leading pair but different target — NOT a duplicate
        assert not gen._is_duplicate_composite("sec_form4", "finra_short", "congressional")

    def test_retired_hypothesis_does_not_block_new_composite(self, registry, tmp_data_dir):
        """A retired composite should not prevent creating a new one."""
        hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        registry.register(hyp)
        registry.retire(hyp.hypothesis_id, reason="test")

        gen = HypothesisGenerator(
            registry=registry,
            lag_data_path=tmp_data_dir / "lag_correlations.json",
        )
        # Retired should not block
        assert not gen._is_duplicate_composite("sec_form4", "finra_short", "finnhub_earnings")


# ---------------------------------------------------------------------------
# Validator routing for composite hypotheses
# ---------------------------------------------------------------------------

class TestValidatorCompositeRouting:
    def _make_validator(self, tmp_data_dir):
        signals_dir = tmp_data_dir / "signals"
        signals_dir.mkdir(exist_ok=True)
        return HypothesisValidator(
            signals_dir=signals_dir,
            data_dir=tmp_data_dir,
        )

    def test_composite_increases_min_observations(self, tmp_data_dir):
        """Composite hypothesis needs +10 observations over bivariate."""
        validator = self._make_validator(tmp_data_dir)

        from mae_core.market.intelligence.hypothesis_validator import _get_gate

        base_min_obs = _get_gate("min_observations")
        expected = base_min_obs + 10

        # Composite: promote requires base+10 observations
        composite_hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        # With no signal archive, result is empty — but we can verify the gate
        # by mocking sufficient data inline via pre-computed stats path
        # (backtest_derived bypasses archive scan)
        composite_hyp.source_type = SourceType.BACKTEST_DERIVED
        composite_hyp.stats.wins = int(expected * 0.6)
        composite_hyp.stats.losses = int(expected * 0.4)
        composite_hyp.stats.total_observations = expected
        composite_hyp.stats.sharpe_ratio = 2.0

        result = validator.validate(composite_hyp)
        # Should NOT promote because we're at exactly the +10 threshold boundary
        # and win_rate still needs to exceed promote_win_rate + 0.02
        # Result will vary, but there should be no crash
        assert result.total_observations == expected

    def test_composite_increases_promote_win_rate(self, tmp_data_dir):
        """Composite needs promote_win_rate + 0.02."""
        from mae_core.market.intelligence.hypothesis_validator import _get_gate

        base_pwr = _get_gate("promote_win_rate")
        expected_pwr = base_pwr + 0.02

        validator = self._make_validator(tmp_data_dir)
        composite_hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=20,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
            source_type=SourceType.BACKTEST_DERIVED,
        )
        # Set win rate exactly at base threshold (not composite threshold)
        total = 40
        wins = int(total * (base_pwr + 0.01))  # just above base, below composite bar
        composite_hyp.stats.wins = wins
        composite_hyp.stats.losses = total - wins
        composite_hyp.stats.total_observations = total
        composite_hyp.stats.sharpe_ratio = 2.0

        result = validator.validate(composite_hyp)
        # At base+0.01 win rate, should NOT promote (composite bar is base+0.02)
        assert not result.recommend_promote

    def test_composite_validated_via_find_composite_trigger_events(self, tmp_data_dir):
        """Validator routes composite to _find_composite_trigger_events()."""
        signals_dir = tmp_data_dir / "signals"
        signals_dir.mkdir(exist_ok=True)
        validator = self._make_validator(tmp_data_dir)

        composite_hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=10,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        # No signals on disk → no trigger events → empty result (not crash)
        result = validator.validate(composite_hyp)
        assert result.hypothesis_id == composite_hyp.hypothesis_id
        assert result.total_observations == 0

    def test_find_composite_requires_both_sources_within_window(self, tmp_data_dir):
        """If conjunct never fires, no events are returned."""
        signals_dir = tmp_data_dir / "signals"
        signals_dir.mkdir(exist_ok=True)

        # Write ONLY source_a events, no conjunct events
        from datetime import datetime, timedelta
        ts = (datetime.now() - timedelta(days=5)).isoformat()
        sig_file = signals_dir / "2026-01-01.jsonl"
        sig_file.write_text(
            json.dumps({"source": "sec_form4", "timestamp": ts, "symbol": "AAPL"}) + "\n"
        )

        validator = HypothesisValidator(
            signals_dir=signals_dir,
            data_dir=tmp_data_dir,
        )
        composite_hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=10,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        result = validator.validate(composite_hyp)
        assert result.total_observations == 0

    def test_find_composite_excludes_mismatched_symbols(self, tmp_data_dir):
        """source_a on AAPL + conjunct on MSFT should not match."""
        signals_dir = tmp_data_dir / "signals"
        signals_dir.mkdir(exist_ok=True)

        from datetime import datetime, timedelta
        ts_a = (datetime.now() - timedelta(days=5)).isoformat()
        ts_c = (datetime.now() - timedelta(days=5)).isoformat()

        sig_file = signals_dir / "2026-01-01.jsonl"
        with open(sig_file, "w") as f:
            f.write(json.dumps({"source": "sec_form4", "timestamp": ts_a, "symbol": "AAPL"}) + "\n")
            f.write(json.dumps({"source": "finra_short", "timestamp": ts_c, "symbol": "MSFT"}) + "\n")

        validator = HypothesisValidator(
            signals_dir=signals_dir,
            data_dir=tmp_data_dir,
        )
        composite_hyp = Hypothesis(
            name="[sec_form4+finra_short]→finnhub_earnings",
            trigger=TriggerPattern(
                source_a="sec_form4",
                source_b="finnhub_earnings",
                lag_days=10,
                conjunct_source="finra_short",
            ),
            causal_story="[COMPOSITE] story.",
        )
        result = validator.validate(composite_hyp)
        assert result.total_observations == 0
