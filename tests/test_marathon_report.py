"""Tests for marathon_report.py — post-mortem report generator.

Covers:
- Empty data directory (graceful degradation with "unavailable" strings)
- Thompson distributions rendering (source keys + win rates)
- Convergence state rendering (regime + direction)
- Hypothesis pipeline rendering (event replay + status counts)
- Append-not-overwrite behaviour of generate_report()
- Alert filtering by confidence threshold in _write_alerts()
- Gate comparison rendering (changed vs unchanged gates)
"""

import json
import pytest

import marathon_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Test 1: empty data directory → "unavailable" strings, no exception
# ---------------------------------------------------------------------------

def test_report_with_no_data_files(tmp_path, monkeypatch):
    """generate_report() must not raise even when all data files are absent.

    Every section should fall back to its "unavailable" sentinel string so
    that the report is always readable even after a fresh install.
    """
    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)
    monkeypatch.setattr(marathon_report, "DATA_MIDGE", tmp_path)

    out = tmp_path / "report.md"
    # Should complete without raising
    marathon_report.generate_report(output_path=str(out))

    assert out.exists(), "generate_report() must create the output file"

    content = out.read_text(encoding="utf-8")
    assert "unavailable" in content.lower(), (
        "At least one 'unavailable' fallback string must appear when data is missing"
    )


# ---------------------------------------------------------------------------
# Test 2: Thompson distributions → source keys + win rates
# ---------------------------------------------------------------------------

def test_report_reads_thompson_distributions(tmp_path, monkeypatch):
    """_render_bayesian_learning() must show source keys and win rates from
    thompson_distributions.json when the file contains enough samples.
    """
    # Build a minimal distribution with 20 samples so it clears the >= 10 gate
    distributions = {
        "sec_form4": {
            "default": {"alpha": 12.0, "beta": 8.0}   # mean 0.60, samples = 18
        },
        "finra_short": {
            "default": {"alpha": 8.0, "beta": 12.0}   # mean 0.40, samples = 18
        },
    }
    _write_json(tmp_path / "thompson_distributions.json", distributions)
    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)

    output = marathon_report._render_bayesian_learning()

    assert "sec_form4" in output, "Source key sec_form4 must appear in output"
    assert "finra_short" in output, "Source key finra_short must appear in output"
    # Win rates rendered as percentages — check at least one percentage symbol
    assert "%" in output, "Win rates must be rendered as percentages"
    # The Thompson section itself must not show its own unavailable fallback.
    # (calibration_report.json is absent so "*Calibration report unavailable.*" is
    # expected; we only verify the Thompson distributions block rendered correctly.)
    assert "Thompson distributions unavailable" not in output


# ---------------------------------------------------------------------------
# Test 3: Convergence state → regime + direction
# ---------------------------------------------------------------------------

def test_report_reads_convergence_state(tmp_path, monkeypatch):
    """_render_market_intelligence() must include the regime and global
    direction values read from convergence_state.json.
    """
    convergence_state = {
        "regime": "bull",
        "global": {
            "direction": "LONG",
            "strength": 0.72,
            "domains": 4,
        },
    }
    _write_json(tmp_path / "convergence_state.json", convergence_state)

    # discovery_log.jsonl must exist or the section will still work (it has
    # its own try/except), but create an empty one for a cleaner test
    (tmp_path / "discovery_log.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)
    monkeypatch.setattr(marathon_report, "DATA_MIDGE", tmp_path)

    output = marathon_report._render_market_intelligence()

    assert "bull" in output, "Regime 'bull' must appear in the output"
    assert "LONG" in output, "Direction 'LONG' must appear in the output"


# ---------------------------------------------------------------------------
# Test 4: Hypothesis JSONL replay → status counts
# ---------------------------------------------------------------------------

def test_report_reads_hypotheses_jsonl(tmp_path, monkeypatch):
    """_render_hypothesis_pipeline() must replay hypothesis events and show
    accurate per-status counts (active, probation, retired).
    """
    # Three distinct hypotheses with final statuses: active, active, retired
    events = [
        {"event": "CREATED",  "hypothesis": {"hypothesis_id": "h1", "status": "probation"}},
        {"event": "PROMOTED", "hypothesis": {"hypothesis_id": "h1", "status": "active"}},
        {"event": "CREATED",  "hypothesis": {"hypothesis_id": "h2", "status": "probation"}},
        {"event": "PROMOTED", "hypothesis": {"hypothesis_id": "h2", "status": "active"}},
        {"event": "CREATED",  "hypothesis": {"hypothesis_id": "h3", "status": "probation"}},
        {"event": "RETIRED",  "hypothesis": {"hypothesis_id": "h3", "status": "retired"}},
    ]
    _write_jsonl(tmp_path / "hypotheses.jsonl", events)
    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)
    monkeypatch.setattr(marathon_report, "DATA_MIDGE", tmp_path)

    output = marathon_report._render_hypothesis_pipeline()

    # Total tracked must be 3
    assert "3" in output, "Total hypothesis count (3) must appear in output"
    # Active count = 2
    lines = output.splitlines()
    active_line = next((l for l in lines if l.startswith("| Active")), None)
    assert active_line is not None, "Active row must be present"
    assert "| 2 |" in active_line, f"Active count must be 2, got: {active_line}"
    # Retired count = 1
    retired_line = next((l for l in lines if l.startswith("| Retired")), None)
    assert retired_line is not None, "Retired row must be present"
    assert "| 1 |" in retired_line, f"Retired count must be 1, got: {retired_line}"


# ---------------------------------------------------------------------------
# Test 5: generate_report() appends, never overwrites
# ---------------------------------------------------------------------------

def test_report_appends_not_overwrites(tmp_path, monkeypatch):
    """Calling generate_report() twice must produce a file with TWO report
    sections separated by '---' dividers, not just the latest one.
    """
    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)
    monkeypatch.setattr(marathon_report, "DATA_MIDGE", tmp_path)

    out = tmp_path / "report.md"
    marathon_report.generate_report(output_path=str(out))
    marathon_report.generate_report(output_path=str(out))

    content = out.read_text(encoding="utf-8")
    # Each call writes one "---" separator as the very first token of a section
    separator_count = content.count("\n---\n")
    assert separator_count >= 2, (
        f"Expected at least 2 '---' separators (one per report), found {separator_count}"
    )


# ---------------------------------------------------------------------------
# Test 6: _write_alerts() filters by confidence >= 0.65
# ---------------------------------------------------------------------------

def test_alert_filtering_by_confidence(tmp_path, monkeypatch):
    """_write_alerts() must write only ticker alerts whose confidence >= 0.65
    and must skip those below the threshold.
    """
    import importlib
    import sys

    # Patch the paths used INSIDE main._write_alerts() via monkeypatch of
    # pathlib.Path so the function writes into tmp_path.
    state_path = tmp_path / "convergence_state.json"
    alerts_path = tmp_path / "alerts.jsonl"

    state = {
        "ticker_alerts": {
            "AAPL": {"confidence": 0.80, "direction": "LONG"},   # above threshold
            "TSLA": {"confidence": 0.50, "direction": "SHORT"},  # below threshold
            "MSFT": {"confidence": 0.65, "direction": "LONG"},   # exactly at threshold
            "GOOG": {"confidence": 0.30, "direction": "SHORT"},  # below threshold
        }
    }
    _write_json(state_path, state)

    # Import main module; redirect its internal path constants by patching
    # pathlib.Path constructor is too broad — instead patch at the import-site
    # by temporarily changing the working directory so relative paths resolve
    # to tmp_path, which is simpler and avoids touching production code.
    import os
    orig_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # Ensure data/midge/ sub-structure exists for _write_alerts
        (tmp_path / "data" / "midge").mkdir(parents=True, exist_ok=True)
        # Copy state file to the relative path _write_alerts expects
        import shutil
        shutil.copy(state_path, tmp_path / "data" / "midge" / "convergence_state.json")

        # Import (or reload) main so its module-level logger resolves
        if "main" in sys.modules:
            import main as main_mod
        else:
            import importlib
            main_mod = importlib.import_module("main")

        # Redirect alerts output to tmp_path
        import pathlib
        real_alerts = pathlib.Path("data/midge/alerts.jsonl")

        main_mod._write_alerts()

        written_path = tmp_path / "data" / "midge" / "alerts.jsonl"
        assert written_path.exists(), "alerts.jsonl must be created"

        records = [json.loads(l) for l in written_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        tickers_written = {r["ticker"] for r in records}

        assert "AAPL" in tickers_written, "AAPL (confidence 0.80) must be written"
        assert "MSFT" in tickers_written, "MSFT (confidence 0.65) must be written"
        assert "TSLA" not in tickers_written, "TSLA (confidence 0.50) must be filtered out"
        assert "GOOG" not in tickers_written, "GOOG (confidence 0.30) must be filtered out"
    finally:
        os.chdir(orig_cwd)


# ---------------------------------------------------------------------------
# Test 7: Gate comparison → "YES" for changed gates
# ---------------------------------------------------------------------------

def test_report_shows_gate_comparison(tmp_path, monkeypatch):
    """_render_hypothesis_pipeline() must show 'YES' in the Changed column
    for any hypothesis gate whose value differs from the module-level default.
    """
    # Use a gate value that differs from _GATE_DEFAULTS["promote_win_rate"] = 0.52
    config_snapshot = {
        "hypothesis_gates": {
            "min_observations": 20,        # same as default → "no"
            "promote_win_rate": 0.60,      # changed from 0.52 → "YES"
            "promote_dsr": 0.5,            # same as default → "no"
            "retire_win_rate": 0.45,       # same as default → "no"
            "retire_dsr": 0.0,             # same as default → "no"
        }
    }
    _write_json(tmp_path / "config_snapshot.json", config_snapshot)

    # hypotheses.jsonl must exist (empty is fine — the section has its own guard)
    (tmp_path / "hypotheses.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(marathon_report, "DATA_MARKET", tmp_path)
    monkeypatch.setattr(marathon_report, "DATA_MIDGE", tmp_path)

    output = marathon_report._render_hypothesis_pipeline()

    assert "YES" in output, (
        "Changed gate (promote_win_rate 0.52→0.60) must produce 'YES' in the table"
    )
    # The unchanged gate should stay "no"
    assert "no" in output, "Unchanged gates must show 'no'"
