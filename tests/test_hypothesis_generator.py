"""Tests for HypothesisGenerator — lag findings → formal hypotheses."""

import json
from pathlib import Path

import pytest

from mae_core.market.intelligence.hypothesis import HypothesisStatus
from mae_core.market.intelligence.hypothesis_registry import HypothesisRegistry
from mae_core.market.intelligence.hypothesis_generator import (
    HypothesisGenerator,
    CAUSAL_STORY_TEMPLATES,
    MIN_CORRELATION,
    MIN_PAIRS,
    _get_causal_story,
)


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path


@pytest.fixture
def registry(tmp_data_dir):
    return HypothesisRegistry(data_dir=tmp_data_dir)


def _write_lag_findings(tmp_path, findings):
    path = tmp_path / "lag_correlations.json"
    path.write_text(json.dumps(findings))
    return path


class TestCausalStoryLookup:
    def test_known_pair(self):
        story = _get_causal_story("sec_form4", "finnhub_earnings")
        assert "Insider" in story or "insider" in story

    def test_known_pair_reversed(self):
        story = _get_causal_story("finnhub_earnings", "sec_form4")
        assert "insider" in story.lower()

    def test_unknown_pair_requires_review(self):
        story = _get_causal_story("foo_source", "bar_source")
        assert "REQUIRES MANUAL REVIEW" in story

    def test_all_templates_are_nonempty(self):
        for key, story in CAUSAL_STORY_TEMPLATES.items():
            assert len(story) > 20, f"Template for {key} is too short"


class TestHypothesisGenerator:
    def test_generates_from_qualifying_findings(self, registry, tmp_data_dir):
        findings = [
            {
                "source_a": "sec_form4",
                "source_b": "finnhub_earnings",
                "lag_days": 57,
                "correlation": 0.91,
                "n_pairs": 12,
                "direction": "positive",
            }
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        new = gen.generate()
        assert len(new) == 1
        assert new[0].trigger.source_a == "sec_form4"
        assert new[0].trigger.lag_days == 57
        assert new[0].status == HypothesisStatus.PROBATION

    def test_filters_low_correlation(self, registry, tmp_data_dir):
        findings = [
            {
                "source_a": "foo", "source_b": "bar",
                "lag_days": 10, "correlation": 0.3,
                "n_pairs": 20, "direction": "positive",
            }
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        assert len(gen.generate()) == 0

    def test_filters_low_n_pairs(self, registry, tmp_data_dir):
        findings = [
            {
                "source_a": "foo", "source_b": "bar",
                "lag_days": 10, "correlation": 0.9,
                "n_pairs": 3, "direction": "positive",
            }
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        assert len(gen.generate()) == 0

    def test_deduplicates_against_registry(self, registry, tmp_data_dir):
        findings = [
            {
                "source_a": "sec_form4", "source_b": "finnhub_earnings",
                "lag_days": 57, "correlation": 0.91,
                "n_pairs": 12, "direction": "positive",
            }
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        gen.generate()
        # Second call should produce nothing (already registered)
        assert len(gen.generate()) == 0

    def test_unknown_pair_gets_review_flag(self, registry, tmp_data_dir):
        findings = [
            {
                "source_a": "mystery_a", "source_b": "mystery_b",
                "lag_days": 30, "correlation": 0.85,
                "n_pairs": 15, "direction": "positive",
            }
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        new = gen.generate()
        assert len(new) == 1
        assert "REQUIRES MANUAL REVIEW" in new[0].causal_story

    def test_statistics(self, registry, tmp_data_dir):
        findings = [
            {"source_a": "a", "source_b": "b", "lag_days": 10,
             "correlation": 0.8, "n_pairs": 15, "direction": "positive"},
            {"source_a": "c", "source_b": "d", "lag_days": 20,
             "correlation": 0.3, "n_pairs": 5, "direction": "positive"},
        ]
        lag_path = _write_lag_findings(tmp_data_dir, findings)
        gen = HypothesisGenerator(registry=registry, lag_data_path=lag_path)
        stats = gen.get_statistics()
        assert stats["total_lag_findings"] == 2
        assert stats["qualifying_findings"] == 1
