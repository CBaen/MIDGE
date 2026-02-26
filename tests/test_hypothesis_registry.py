"""Tests for HypothesisRegistry — event-sourced hypothesis lifecycle."""

import json
import tempfile
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStatus,
    SourceType,
    TriggerPattern,
)
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for test isolation."""
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    """Fresh HypothesisRegistry backed by temp dir."""
    return HypothesisRegistry(data_dir=tmp_data_dir)


def _make_hypothesis(name="test_hyp", source_a="sec_form4", source_b="finnhub_earnings", lag=45):
    return Hypothesis(
        name=name,
        trigger=TriggerPattern(
            source_a=source_a, source_b=source_b,
            lag_days=lag, direction="positive",
        ),
        causal_story="Insiders trade before earnings.",
        source_type=SourceType.LAG_CORRELATION,
    )


class TestHypothesisDataclass:
    def test_to_dict_round_trip(self):
        hyp = _make_hypothesis()
        d = hyp.to_dict()
        restored = Hypothesis.from_dict(d)
        assert restored.name == hyp.name
        assert restored.trigger.source_a == "sec_form4"
        assert restored.trigger.lag_days == 45
        assert restored.status == HypothesisStatus.PROBATION

    def test_win_rate_zero_when_no_observations(self):
        hyp = _make_hypothesis()
        assert hyp.stats.win_rate == 0.0

    def test_win_rate_computes_correctly(self):
        hyp = _make_hypothesis()
        hyp.stats.wins = 7
        hyp.stats.losses = 3
        hyp.stats.total_observations = 10
        assert abs(hyp.stats.win_rate - 0.7) < 0.01


class TestHypothesisRegistry:
    def test_register(self, registry):
        hyp = _make_hypothesis()
        hid = registry.register(hyp)
        assert hid == hyp.hypothesis_id
        assert registry.get(hid) is not None
        assert registry.get(hid).status == HypothesisStatus.PROBATION

    def test_register_dedup(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        registry.register(hyp)  # same ID
        assert len(registry.get_all()) == 1

    def test_promote_requires_causal_story(self, registry):
        hyp = _make_hypothesis()
        hyp.causal_story = ""
        registry.register(hyp)
        result = registry.promote(hyp.hypothesis_id)
        assert result is None
        assert registry.get(hyp.hypothesis_id).status == HypothesisStatus.PROBATION

    def test_promote_with_causal_story(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        promoted = registry.promote(hyp.hypothesis_id)
        assert promoted is not None
        assert promoted.status == HypothesisStatus.ACTIVE
        assert promoted.promoted_at is not None

    def test_retire(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        retired = registry.retire(hyp.hypothesis_id, "Win rate too low")
        assert retired.status == HypothesisStatus.RETIRED
        assert retired.retired_reason == "Win rate too low"

    def test_hibernate_and_reactivate(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        registry.promote(hyp.hypothesis_id)
        registry.hibernate(hyp.hypothesis_id)
        assert registry.get(hyp.hypothesis_id).status == HypothesisStatus.HIBERNATED
        registry.reactivate(hyp.hypothesis_id)
        assert registry.get(hyp.hypothesis_id).status == HypothesisStatus.ACTIVE

    def test_get_active(self, registry):
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2", source_a="congressional")
        registry.register(h1)
        registry.register(h2)
        registry.promote(h1.hypothesis_id)
        assert len(registry.get_active()) == 1
        assert registry.get_active()[0].name == "h1"

    def test_find_by_trigger_dedup(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        found = registry.find_by_trigger("sec_form4", "finnhub_earnings", 43)
        assert found is not None  # lag 43 is within ±5 of 45

    def test_find_by_trigger_no_match(self, registry):
        hyp = _make_hypothesis()
        registry.register(hyp)
        found = registry.find_by_trigger("sec_form4", "finnhub_earnings", 100)
        assert found is None  # lag 100 is too far from 45

    def test_persistence_reload(self, tmp_data_dir):
        reg1 = HypothesisRegistry(data_dir=tmp_data_dir)
        hyp = _make_hypothesis()
        reg1.register(hyp)
        reg1.promote(hyp.hypothesis_id)

        # New registry from same dir should load events
        reg2 = HypothesisRegistry(data_dir=tmp_data_dir)
        assert len(reg2.get_all()) == 1
        assert reg2.get(hyp.hypothesis_id).status == HypothesisStatus.ACTIVE

    def test_get_statistics(self, registry):
        h1 = _make_hypothesis("h1")
        h2 = _make_hypothesis("h2", source_a="congressional")
        registry.register(h1)
        registry.register(h2)
        registry.promote(h1.hypothesis_id)
        stats = registry.get_statistics()
        assert stats["total_hypotheses"] == 2
        assert stats["active_count"] == 1
