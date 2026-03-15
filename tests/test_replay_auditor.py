"""Tests for mae_core.market.parallel.replay_auditor.

All tests are fully offline — no yfinance calls, no real file I/O for
audit output.  Price fetching is monkey-patched to return controlled values.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — patch paths before module import so tests use temp dirs
# ---------------------------------------------------------------------------

import mae_core.market.parallel.replay_auditor as auditor_mod
from mae_core.market.parallel.replay_auditor import (
    MIN_OUTCOMES_PER_COMBO,
    SUCCESS_THRESHOLD_PCT,
    _build_combo_key,
    _infer_batch_id,
    _read_graded_records,
    _tier1_sample_verify,
    _tier2_consistency,
    _tier3_cross_worker,
    _apply_blocked_ids,
    run_audit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_record(
    grade: str = "success",
    ticker: str = "AAPL",
    direction: str = "bullish",
    domains: list | None = None,
    timestamp: str = "2026-01-10T12:00:00",
    pct_change: float | None = None,
    replay_ts: str = "2026-03-15T10:00",
    regime: str = "default",
) -> dict:
    if domains is None:
        domains = ["insider", "technical"]
    rec = {
        "grade":          grade,
        "primary_symbol": ticker,
        "direction":      direction,
        "domains":        domains,
        "timestamp":      timestamp,
        "replay_ts":      replay_ts,
        "regime":         regime,
        "confidence":     0.7,
        "strength":       0.6,
        "outcome_window_days": 14,
    }
    if pct_change is not None:
        rec["pct_change"] = pct_change
    return rec


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# _build_combo_key
# ---------------------------------------------------------------------------


class TestBuildComboKey:
    def test_sorted_and_joined(self):
        assert _build_combo_key(["technical", "insider"]) == "insider+technical"

    def test_single_domain(self):
        assert _build_combo_key(["macro"]) == "macro"

    def test_empty_list(self):
        assert _build_combo_key([]) == ""

    def test_none_returns_empty(self):
        assert _build_combo_key(None) == ""

    def test_filters_falsy_entries(self):
        assert _build_combo_key(["insider", "", None, "technical"]) == "insider+technical"

    def test_stable_regardless_of_input_order(self):
        a = _build_combo_key(["a", "b", "c"])
        b = _build_combo_key(["c", "a", "b"])
        assert a == b


# ---------------------------------------------------------------------------
# _infer_batch_id
# ---------------------------------------------------------------------------


class TestInferBatchId:
    def test_uses_replay_ts_truncated_to_minute(self):
        rec = {"replay_ts": "2026-03-15T14:23:45.123456"}
        assert _infer_batch_id(rec) == "2026-03-15T14:23"

    def test_falls_back_to_date_when_no_replay_ts(self):
        rec = {"timestamp": "2026-01-10T12:00:00"}
        assert _infer_batch_id(rec) == "2026-01-10"

    def test_unknown_when_both_missing(self):
        assert _infer_batch_id({}) == "unknown"


# ---------------------------------------------------------------------------
# _read_graded_records
# ---------------------------------------------------------------------------


class TestReadGradedRecords:
    def test_reads_success_and_failure(self, tmp_path):
        path = tmp_path / "results.jsonl"
        records = [
            _make_record(grade="success"),
            _make_record(grade="failure"),
        ]
        _write_jsonl(path, records)
        result = _read_graded_records(path)
        assert len(result) == 2

    def test_skips_cycle_summary(self, tmp_path):
        path = tmp_path / "results.jsonl"
        _write_jsonl(path, [
            {"type": "cycle_summary", "cycle": 1},
            _make_record(grade="success"),
        ])
        result = _read_graded_records(path)
        assert len(result) == 1

    def test_skips_pending_and_ungraded(self, tmp_path):
        path = tmp_path / "results.jsonl"
        _write_jsonl(path, [
            _make_record(grade="pending"),
            _make_record(grade="ungraded"),
            _make_record(grade="success"),
        ])
        result = _read_graded_records(path)
        assert len(result) == 1

    def test_skips_bad_json(self, tmp_path):
        path = tmp_path / "results.jsonl"
        with open(path, "w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_make_record(grade="success")) + "\n")
        result = _read_graded_records(path)
        assert len(result) == 1

    def test_returns_empty_when_file_missing(self, tmp_path):
        result = _read_graded_records(tmp_path / "missing.jsonl")
        assert result == []

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "results.jsonl"
        with open(path, "w") as fh:
            fh.write("\n\n")
            fh.write(json.dumps(_make_record(grade="success")) + "\n")
        result = _read_graded_records(path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _tier2_consistency
# ---------------------------------------------------------------------------


class TestTier2Consistency:
    def test_approves_combo_with_enough_outcomes(self):
        records = [_make_record(grade="success") for _ in range(MIN_OUTCOMES_PER_COMBO)]
        buckets, flagged = _tier2_consistency(records, min_outcomes=5)
        key = "insider+technical|default"
        assert key in buckets
        assert buckets[key]["approved"] is True
        assert buckets[key]["wins"] == 5
        assert len(flagged) == 0

    def test_does_not_approve_below_minimum(self):
        records = [_make_record(grade="success") for _ in range(3)]
        buckets, flagged = _tier2_consistency(records, min_outcomes=5)
        key = "insider+technical|default"
        assert buckets[key]["approved"] is False

    def test_flags_bullish_win_with_negative_return(self):
        rec = _make_record(grade="success", direction="bullish", pct_change=-8.0)
        buckets, flagged = _tier2_consistency([rec], min_outcomes=1)
        assert len(flagged) == 1
        assert flagged[0]["reason"] == "grade_contradicts_return"

    def test_flags_bearish_win_with_positive_return(self):
        rec = _make_record(grade="success", direction="bearish", pct_change=7.0)
        _, flagged = _tier2_consistency([rec], min_outcomes=1)
        assert len(flagged) == 1
        assert flagged[0]["ticker"] == "AAPL"

    def test_does_not_flag_failure_with_any_return(self):
        # Failures may have any sign — only wins are checked
        rec = _make_record(grade="failure", direction="bullish", pct_change=-3.0)
        _, flagged = _tier2_consistency([rec], min_outcomes=1)
        assert flagged == []

    def test_separates_combos_by_domain_set(self):
        r1 = _make_record(grade="success", domains=["insider", "technical"])
        r2 = _make_record(grade="success", domains=["macro", "events"])
        for _ in range(4):
            r1_copy = dict(r1)
            r2_copy = dict(r2)
        records = [dict(r1) for _ in range(5)] + [dict(r2) for _ in range(5)]
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        assert "insider+technical|default" in buckets
        assert "events+macro|default" in buckets

    def test_regime_creates_separate_bucket(self):
        r_bull = _make_record(grade="success", regime="bull")
        r_bear = _make_record(grade="success", regime="bear")
        records = [dict(r_bull) for _ in range(5)] + [dict(r_bear) for _ in range(5)]
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        assert "insider+technical|bull" in buckets
        assert "insider+technical|bear" in buckets

    def test_counts_wins_and_losses_correctly(self):
        records = (
            [_make_record(grade="success") for _ in range(6)] +
            [_make_record(grade="failure") for _ in range(4)]
        )
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        key = "insider+technical|default"
        assert buckets[key]["wins"]   == 6
        assert buckets[key]["losses"] == 4

    def test_skips_records_with_no_domains(self):
        rec = _make_record(grade="success", domains=[])
        buckets, _ = _tier2_consistency([rec], min_outcomes=1)
        assert buckets == {}

    def test_does_not_flag_bullish_win_with_positive_return(self):
        rec = _make_record(grade="success", direction="bullish", pct_change=9.0)
        _, flagged = _tier2_consistency([rec], min_outcomes=1)
        assert flagged == []

    def test_does_not_flag_bearish_win_with_negative_return(self):
        rec = _make_record(grade="success", direction="bearish", pct_change=-9.0)
        _, flagged = _tier2_consistency([rec], min_outcomes=1)
        assert flagged == []


# ---------------------------------------------------------------------------
# _tier3_cross_worker
# ---------------------------------------------------------------------------


class TestTier3CrossWorker:
    def test_no_conflict_when_single_batch(self):
        records = [_make_record() for _ in range(5)]
        blocked, flagged = _tier3_cross_worker(records)
        assert blocked == set()
        assert flagged == []

    def test_no_conflict_when_batches_agree(self):
        r1 = _make_record(grade="success", replay_ts="2026-03-15T10:00:00")
        r2 = _make_record(grade="success", replay_ts="2026-03-15T11:00:00")
        blocked, flagged = _tier3_cross_worker([r1, r2])
        assert blocked == set()
        assert flagged == []

    def test_detects_conflict_when_batches_disagree(self):
        r1 = _make_record(
            grade="success", ticker="TSLA",
            timestamp="2026-01-10T12:00:00",
            replay_ts="2026-03-15T10:00:00",
        )
        r2 = _make_record(
            grade="failure", ticker="TSLA",
            timestamp="2026-01-10T12:00:00",
            replay_ts="2026-03-15T11:00:00",
        )
        blocked, flagged = _tier3_cross_worker([r1, r2])
        assert ("TSLA", "2026-01-10") in blocked
        assert len(flagged) == 1
        assert flagged[0]["reason"] == "cross_worker_disagreement"

    def test_different_tickers_no_conflict(self):
        r1 = _make_record(grade="success", ticker="AAPL", replay_ts="2026-03-15T10:00:00")
        r2 = _make_record(grade="failure", ticker="MSFT", replay_ts="2026-03-15T11:00:00")
        blocked, flagged = _tier3_cross_worker([r1, r2])
        assert blocked == set()
        assert flagged == []

    def test_different_dates_no_conflict(self):
        r1 = _make_record(
            grade="success", ticker="AAPL",
            timestamp="2026-01-10T12:00:00",
            replay_ts="2026-03-15T10:00:00",
        )
        r2 = _make_record(
            grade="failure", ticker="AAPL",
            timestamp="2026-01-11T12:00:00",
            replay_ts="2026-03-15T11:00:00",
        )
        blocked, flagged = _tier3_cross_worker([r1, r2])
        assert blocked == set()

    def test_skips_records_with_no_ticker(self):
        r = {
            "grade": "success",
            "primary_symbol": None,
            "ticker": None,
            "timestamp": "2026-01-10T12:00:00",
            "replay_ts": "2026-03-15T10:00",
            "domains": ["insider"],
        }
        blocked, flagged = _tier3_cross_worker([r])
        assert blocked == set()


# ---------------------------------------------------------------------------
# _apply_blocked_ids
# ---------------------------------------------------------------------------


class TestApplyBlockedIds:
    def test_removes_blocked_outcomes_from_bucket(self):
        # 5 successes for insider+technical
        records = [
            _make_record(grade="success", ticker="AAPL",
                         timestamp="2026-01-10T12:00:00")
            for _ in range(5)
        ]
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        assert buckets["insider+technical|default"]["wins"] == 5

        blocked = {("AAPL", "2026-01-10")}
        updated, removed = _apply_blocked_ids(buckets, records, blocked)

        assert removed == 5
        assert updated["insider+technical|default"]["wins"] == 0

    def test_no_removal_when_blocked_set_empty(self):
        records = [_make_record(grade="success") for _ in range(5)]
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        updated, removed = _apply_blocked_ids(buckets, records, set())
        assert removed == 0
        assert updated["insider+technical|default"]["wins"] == 5

    def test_re_unapproves_combo_after_removal(self):
        # Exactly 5 records, all from a blocked ticker+date
        records = [
            _make_record(grade="success", ticker="AAPL",
                         timestamp="2026-01-10T12:00:00")
            for _ in range(5)
        ]
        buckets, _ = _tier2_consistency(records, min_outcomes=5)
        assert buckets["insider+technical|default"]["approved"] is True

        blocked = {("AAPL", "2026-01-10")}
        updated, _ = _apply_blocked_ids(buckets, records, blocked)
        # After removal 0 outcomes remain — below minimum
        assert updated["insider+technical|default"]["approved"] is False


# ---------------------------------------------------------------------------
# _tier1_sample_verify (mocked price fetching)
# ---------------------------------------------------------------------------


class TestTier1SampleVerify:
    def _records_with_known_prices(self, n: int = 20) -> list:
        """Return n records all past their outcome window."""
        base = datetime(2025, 6, 1, 12, 0, 0)
        records = []
        for i in range(n):
            ts = (base + timedelta(days=i)).isoformat()
            records.append(_make_record(
                grade="success",
                ticker=f"T{i:03d}",
                direction="bullish",
                timestamp=ts,
                pct_change=10.0,  # Worker says +10%
            ))
        return records

    def test_all_agree_batch_ok(self):
        records = self._records_with_known_prices(20)
        # Auditor independently confirms success (bullish, pct >= threshold)
        with patch.object(auditor_mod, "_independent_grade", return_value="success"):
            batch_ok, flagged, verified, mismatches = _tier1_sample_verify(
                records, sample_rate=1.0
            )
        assert batch_ok is True
        assert mismatches == 0
        assert len(flagged) == 0

    def test_high_mismatch_rate_fails_batch(self):
        records = self._records_with_known_prices(20)
        # All independent grades contradict the worker
        with patch.object(auditor_mod, "_independent_grade", return_value="failure"):
            batch_ok, flagged, verified, mismatches = _tier1_sample_verify(
                records, sample_rate=1.0
            )
        assert batch_ok is False
        assert mismatches == verified

    def test_none_grade_is_not_counted_as_mismatch(self):
        records = self._records_with_known_prices(20)
        # All independent grades are None (price unavailable)
        with patch.object(auditor_mod, "_independent_grade", return_value=None):
            batch_ok, flagged, verified, mismatches = _tier1_sample_verify(
                records, sample_rate=1.0
            )
        assert batch_ok is True
        assert verified == 0
        assert mismatches == 0

    def test_low_mismatch_rate_passes(self):
        # 20 records, only 1 mismatch = 5% exactly at threshold → ok
        records = self._records_with_known_prices(20)
        call_count = {"n": 0}

        def mock_grade(rec, outcome_window_days=14):
            call_count["n"] += 1
            # Return mismatch for the very first record only
            if call_count["n"] == 1:
                return "failure"   # worker says "success"
            return "success"

        with patch.object(auditor_mod, "_independent_grade", side_effect=mock_grade):
            batch_ok, flagged, verified, mismatches = _tier1_sample_verify(
                records, sample_rate=1.0
            )
        # 1 mismatch out of 20 = 5.0% — at threshold, should PASS (<=)
        assert batch_ok is True
        assert mismatches == 1

    def test_empty_records_returns_batch_ok(self):
        batch_ok, flagged, verified, mismatches = _tier1_sample_verify(
            [], sample_rate=0.1
        )
        assert batch_ok is True
        assert verified == 0


# ---------------------------------------------------------------------------
# run_audit — integration-style (no real file I/O, no real prices)
# ---------------------------------------------------------------------------


class TestRunAudit:
    def _setup(self, tmp_path, records):
        """Redirect auditor paths to tmp_path and write records."""
        results_path = tmp_path / "continuous_replay_results.jsonl"
        audit_path   = tmp_path / "replay_audit.json"
        _write_jsonl(results_path, records)

        # Monkeypatch module-level paths
        auditor_mod.RESULTS_JSONL = results_path
        auditor_mod.AUDIT_PATH    = audit_path

        return results_path, audit_path

    def test_no_results_writes_empty_audit(self, tmp_path):
        results_path = tmp_path / "missing.jsonl"
        audit_path   = tmp_path / "replay_audit.json"
        auditor_mod.RESULTS_JSONL = results_path
        auditor_mod.AUDIT_PATH    = audit_path

        output = run_audit(skip_sample_verify=True)

        assert output["total_reviewed"] == 0
        assert output["approved_combos"] == []

    def test_approved_combos_with_sufficient_evidence(self, tmp_path):
        records = [_make_record(grade="success") for _ in range(5)]
        _, audit_path = self._setup(tmp_path, records)

        output = run_audit(min_outcomes=5, skip_sample_verify=True)

        assert len(output["approved_combos"]) == 1
        assert output["approved_combos"][0]["combo"] == "insider+technical"
        assert output["approved_combos"][0]["approved"] is True
        assert audit_path.exists()

    def test_insufficient_evidence_produces_no_approved_combos(self, tmp_path):
        records = [_make_record(grade="success") for _ in range(3)]
        self._setup(tmp_path, records)

        output = run_audit(min_outcomes=5, skip_sample_verify=True)

        assert len(output["approved_combos"]) == 0

    def test_sign_contradiction_creates_flagged_item(self, tmp_path):
        records = [
            _make_record(grade="success", direction="bullish", pct_change=-8.0)
        ] + [_make_record(grade="success") for _ in range(5)]
        self._setup(tmp_path, records)

        output = run_audit(min_outcomes=5, skip_sample_verify=True)

        reasons = [f["reason"] for f in output["flagged_items"]]
        assert "grade_contradicts_return" in reasons

    def test_cross_worker_conflict_reduces_approved_count(self, tmp_path):
        # Two batches disagree on AAPL 2026-01-10 — that record should be removed
        conflict_r1 = _make_record(
            grade="success", ticker="AAPL",
            timestamp="2026-01-10T12:00:00",
            replay_ts="2026-03-15T10:00:00",
        )
        conflict_r2 = _make_record(
            grade="failure", ticker="AAPL",
            timestamp="2026-01-10T12:00:00",
            replay_ts="2026-03-15T11:00:00",
        )
        # Add 5 clean successes from a different date to ensure combo has some evidence
        clean = [
            _make_record(
                grade="success",
                timestamp=f"2026-02-{10+i:02d}T12:00:00",
                replay_ts="2026-03-15T10:00:00",
            )
            for i in range(5)
        ]
        records = [conflict_r1, conflict_r2] + clean
        self._setup(tmp_path, records)

        output = run_audit(min_outcomes=5, skip_sample_verify=True)

        reasons = [f["reason"] for f in output["flagged_items"]]
        assert "cross_worker_disagreement" in reasons

    def test_audit_file_written_atomically(self, tmp_path):
        records = [_make_record(grade="success") for _ in range(5)]
        _, audit_path = self._setup(tmp_path, records)

        run_audit(skip_sample_verify=True)

        assert audit_path.exists()
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        assert "last_audit" in data
        assert "approved_combos" in data

    def test_batch_suspect_blocks_all_combos(self, tmp_path):
        records = [_make_record(grade="success") for _ in range(10)]
        self._setup(tmp_path, records)

        # Tier 1 returns batch_ok=False — all combos should be blocked
        with patch.object(
            auditor_mod, "_tier1_sample_verify",
            return_value=(False, [{"reason": "sample_verification_mismatch"}], 10, 6)
        ):
            output = run_audit(min_outcomes=5, skip_sample_verify=False)

        assert output["batch_ok"] is False
        assert len(output["approved_combos"]) == 0
        reasons = [f["reason"] for f in output["flagged_items"]]
        assert "batch_suspect_high_mismatch_rate" in reasons

    def test_total_reviewed_matches_record_count(self, tmp_path):
        n = 12
        records = [_make_record(grade="success") for _ in range(n)]
        self._setup(tmp_path, records)

        output = run_audit(skip_sample_verify=True)
        assert output["total_reviewed"] == n

    def test_run_audit_never_raises_on_bad_data(self, tmp_path):
        """Auditor must not crash even when records are malformed."""
        bad_path = tmp_path / "continuous_replay_results.jsonl"
        with open(bad_path, "w") as fh:
            fh.write("}{bad json\n")
            fh.write("{}\n")  # missing all fields
        audit_path = tmp_path / "replay_audit.json"
        auditor_mod.RESULTS_JSONL = bad_path
        auditor_mod.AUDIT_PATH    = audit_path

        # Should not raise
        output = run_audit(skip_sample_verify=True)
        assert isinstance(output, dict)
