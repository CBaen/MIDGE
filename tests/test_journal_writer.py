"""Tests for JournalWriter — narrative journal of Mae's internal experience."""
import pathlib
import tempfile

import pytest

from mae_core.backbone.journal_writer import JournalWriter


# ── Formatter unit tests ──────────────────────────────────────────────────────

class TestSurpriseDescriptions:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_low_surprise(self):
        desc = self.jw._describe_surprise(0.10)
        assert "Low surprise" in desc
        assert "0.10" in desc

    def test_moderate_surprise(self):
        desc = self.jw._describe_surprise(0.55)
        assert "Moderate surprise" in desc

    def test_high_surprise(self):
        desc = self.jw._describe_surprise(0.75)
        assert "High surprise" in desc

    def test_extreme_surprise(self):
        desc = self.jw._describe_surprise(0.95)
        assert "Extreme surprise" in desc

    def test_exact_boundary_low(self):
        # 0.20 is the boundary for "Low"
        desc = self.jw._describe_surprise(0.20)
        assert "Low surprise" in desc

    def test_just_above_boundary(self):
        # 0.21 should be "Mild"
        desc = self.jw._describe_surprise(0.21)
        assert "Mild surprise" in desc


class TestBodyDescriptions:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_energized_calm(self):
        desc = self.jw._describe_body({"energy_level": 0.9, "emotional_valence": 0.5})
        assert "Energized" in desc
        assert "calm" in desc

    def test_exhausted_stressed(self):
        desc = self.jw._describe_body({"energy_level": 0.1, "emotional_valence": -0.7})
        assert "Exhausted" in desc
        assert "stressed" in desc

    def test_moderate_energy_neutral(self):
        desc = self.jw._describe_body({"energy_level": 0.6, "emotional_valence": 0.0})
        assert "Moderate energy" in desc
        assert "neutral" in desc

    def test_low_energy_uneasy(self):
        desc = self.jw._describe_body({"energy_level": 0.3, "emotional_valence": -0.3})
        assert "Low energy" in desc
        assert "uneasy" in desc

    def test_non_dict_graceful(self):
        desc = self.jw._describe_body(None)
        assert "unknown" in desc.lower()

    def test_empty_dict_graceful(self):
        desc = self.jw._describe_body({})
        # Should use defaults and not crash
        assert isinstance(desc, str)
        assert len(desc) > 0


class TestInhibitionDescriptions:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_with_veto_sources(self):
        desc = self.jw._describe_inhibition("INHIBIT: NoGo > Go", ["surprise:0.75", "risk:0.82"])
        assert "surprise" in desc or "risk" in desc
        assert "wait" in desc

    def test_with_reason_only(self):
        desc = self.jw._describe_inhibition("INHIBIT: NoGo(0.6) > Go(0.3)", [])
        assert "wait" in desc
        # Should strip the "INHIBIT:" prefix
        assert "NoGo(0.6) > Go(0.3)" in desc

    def test_empty_reason(self):
        desc = self.jw._describe_inhibition("", [])
        assert "wait" in desc

    def test_limits_veto_sources_to_three(self):
        sources = ["a:0.1", "b:0.2", "c:0.3", "d:0.4", "e:0.5"]
        desc = self.jw._describe_inhibition("reason", sources)
        # Should not include all 5 sources
        assert "d:0.4" not in desc or "e:0.5" not in desc


class TestMarkerDescriptions:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_success_markers(self):
        desc = self.jw._describe_markers({"SUCCESS": ["m1", "m2"]})
        assert "2 success trail(s)" in desc

    def test_danger_markers(self):
        desc = self.jw._describe_markers({"DANGER": ["d1"]})
        assert "1 danger signal(s)" in desc

    def test_mixed_markers(self):
        desc = self.jw._describe_markers({"SUCCESS": ["m1"], "DANGER": ["d1", "d2"]})
        assert "success" in desc
        assert "danger" in desc

    def test_empty_markers(self):
        desc = self.jw._describe_markers({})
        assert desc == ""

    def test_none_markers(self):
        desc = self.jw._describe_markers(None)
        assert desc == ""


class TestOrganismDescriptions:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_stressed_organism(self):
        endocrine = {"is_stressed": True, "is_resting": False, "exploration_bias": 0.4, "trust_level": 0.5}
        desc = self.jw._describe_organism(endocrine, 0.0, 3)
        assert "stressed" in desc

    def test_resting_organism(self):
        endocrine = {"is_stressed": False, "is_resting": True, "exploration_bias": 0.4, "trust_level": 0.5}
        desc = self.jw._describe_organism(endocrine, 0.0, 3)
        assert "resting" in desc

    def test_phi_included_when_positive(self):
        endocrine = {"is_stressed": False, "is_resting": False, "exploration_bias": 0.5, "trust_level": 0.5}
        desc = self.jw._describe_organism(endocrine, 0.42, 3)
        assert "0.42" in desc

    def test_phi_omitted_when_zero(self):
        endocrine = {"is_stressed": False, "is_resting": False, "exploration_bias": 0.5, "trust_level": 0.5}
        desc = self.jw._describe_organism(endocrine, 0.0, 3)
        assert "phi" not in desc.lower()

    def test_agent_count_included(self):
        endocrine = {"is_stressed": False, "exploration_bias": 0.5, "trust_level": 0.5}
        desc = self.jw._describe_organism(endocrine, 0.0, 7)
        assert "7" in desc

    def test_trust_label_high(self):
        desc = self.jw._describe_trust(0.8)
        assert desc == "high"

    def test_trust_label_moderate(self):
        desc = self.jw._describe_trust(0.5)
        assert desc == "moderate"

    def test_trust_label_low(self):
        desc = self.jw._describe_trust(0.2)
        assert desc == "low"


class TestActionNormalization:
    def setup_method(self):
        self.jw = JournalWriter()

    def test_string_action(self):
        assert self.jw._normalize_action("explore") == "explore"

    def test_dict_action(self):
        assert self.jw._normalize_action({"type": "rest"}) == "rest"

    def test_none_action(self):
        assert self.jw._normalize_action(None) == "idle"

    def test_dict_missing_type(self):
        assert self.jw._normalize_action({}) == "idle"


# ── Integration tests (file I/O) ──────────────────────────────────────────────

class _MockAgent:
    """Minimal mock agent for record_step testing."""
    def __init__(self, uid, role="STEM", action="explore", reward=0.5, pred_error=0.2):
        self.unique_id = uid
        self.role = role
        self.agent_config = {"preferred_provider": "groq"}
        self.last_action = action
        self.last_reward = reward
        self._prediction_error = pred_error
        self._body_state = {"energy_level": 0.8, "emotional_valence": 0.3}
        self._sensed_markers = {"SUCCESS": ["m1"]}
        self._inhibited_this_step = False
        self._last_inhibit_reason = ""
        self._inhibit_veto_sources = []


class TestJournalFileOutput:
    def test_begin_run_creates_file(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=30, agent_count=3)
        assert (tmp_path / "mae-journal.md").exists()

    def test_journal_contains_header(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=30, agent_count=3)
        content = (tmp_path / "mae-journal.md").read_text()
        assert "Mae Run Journal" in content
        assert "3 agents" in content

    def test_record_step_writes_step_header(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=10, agent_count=1)
        agents = [_MockAgent(uid=1)]
        jw.record_step(step=1, agents=agents, systems={})
        content = (tmp_path / "mae-journal.md").read_text()
        assert "## Step 1" in content

    def test_record_step_writes_agent_section(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=10, agent_count=1)
        agents = [_MockAgent(uid=42, role="COORDINATOR")]
        jw.record_step(step=1, agents=agents, systems={})
        content = (tmp_path / "mae-journal.md").read_text()
        assert "Agent 42" in content
        assert "COORDINATOR" in content

    def test_inhibited_agent_logged(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=10, agent_count=1)
        agent = _MockAgent(uid=1)
        agent._inhibited_this_step = True
        agent._last_inhibit_reason = "INHIBIT: high surprise"
        jw.record_step(step=1, agents=[agent], systems={})
        content = (tmp_path / "mae-journal.md").read_text()
        assert "PAUSED" in content

    def test_end_run_appends_to_log(self, tmp_path):
        jw = JournalWriter(output_dir=str(tmp_path))
        jw.begin_run(step_count=5, agent_count=1)
        jw.record_step(step=1, agents=[_MockAgent(uid=1)], systems={})
        jw.end_run(steps_completed=1)
        assert (tmp_path / "journal-log.md").exists()
        log_content = (tmp_path / "journal-log.md").read_text()
        assert "Mae Run Journal" in log_content
        assert "1 steps completed" in log_content

    def test_multiple_runs_append_to_log(self, tmp_path):
        for _ in range(2):
            jw = JournalWriter(output_dir=str(tmp_path))
            jw.begin_run(step_count=3, agent_count=1)
            jw.record_step(step=1, agents=[_MockAgent(uid=1)], systems={})
            jw.end_run(steps_completed=1)
        log_content = (tmp_path / "journal-log.md").read_text()
        # Should appear twice
        assert log_content.count("Mae Run Journal") == 2

    def test_journal_overwritten_each_run(self, tmp_path):
        # First run
        jw1 = JournalWriter(output_dir=str(tmp_path))
        jw1.begin_run(step_count=5, agent_count=1)
        jw1.end_run(steps_completed=0)

        # Second run — journal should be fresh
        jw2 = JournalWriter(output_dir=str(tmp_path))
        jw2.begin_run(step_count=5, agent_count=2)
        jw2.end_run(steps_completed=0)

        content = (tmp_path / "mae-journal.md").read_text()
        # Should only contain one "Mae Run Journal" header
        assert content.count("Mae Run Journal") == 1
        # Should reflect second run's agent count
        assert "2 agents" in content
