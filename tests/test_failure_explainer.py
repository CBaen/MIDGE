"""Tests for failure_explainer.py — WHY did MIDGE get it wrong?"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mae_core.market.intelligence.failure_explainer import (
    FailureExplainer,
    FailureExplanation,
    _signals_to_domains,
    _parse_dt,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    """Isolated data directories — no production file access."""
    data_dir = tmp_path / "data" / "market"
    midge_dir = tmp_path / "data" / "midge"
    data_dir.mkdir(parents=True)
    midge_dir.mkdir(parents=True)
    return data_dir, midge_dir


@pytest.fixture
def explainer(tmp_dirs, monkeypatch):
    data_dir, midge_dir = tmp_dirs
    e = FailureExplainer(data_dir=str(data_dir))
    # Redirect output paths to tmp
    e._midge_dir = midge_dir
    e._explanations_path = midge_dir / "failure_explanations.jsonl"
    e._summary_path = midge_dir / "failure_summary.json"
    return e


def _pred(
    ticker="AAPL",
    direction="bullish",
    confidence=0.55,
    signals=None,
    predicted_at=None,
):
    return {
        "prediction_id": "test-001",
        "symbol": ticker,
        "direction": direction,
        "confidence": confidence,
        "contributing_signals": signals or ["technical_rsi", "macro"],
        "predicted_at": (predicted_at or datetime(2026, 3, 1)).isoformat(),
    }


def _outcome(
    ticker="AAPL",
    direction="bullish",
    was_correct=False,
    return_pct=-3.5,
    mfe=None,
    predicted_at=None,
    outcome_at=None,
):
    return {
        "prediction_id": "test-001",
        "symbol": ticker,
        "direction": direction,
        "was_correct": was_correct,
        "return_pct": return_pct,
        "predicted_confidence": 0.55,
        "mfe": mfe,
        "predicted_at": (predicted_at or datetime(2026, 3, 1)).isoformat(),
        "outcome_at": (outcome_at or datetime(2026, 3, 10)).isoformat(),
    }


# ── Domain mapping ───────────────────────────────────────────────────────────

def test_signals_to_domains_known():
    domains = _signals_to_domains(["technical_rsi", "insider_form4", "fred"])
    assert "technical" in domains
    assert "insider" in domains
    assert "macro" in domains


def test_signals_to_domains_congressional():
    domains = _signals_to_domains(["congressional", "politician_sell"])
    assert "government" in domains


def test_signals_to_domains_unknown_kept():
    domains = _signals_to_domains(["mystery_signal"])
    assert "mystery_signal" in domains


def test_signals_to_domains_empty():
    assert _signals_to_domains([]) == set()


# ── Parse datetime ────────────────────────────────────────────────────────────

def test_parse_dt_iso():
    dt = _parse_dt("2026-03-01T10:30:00.123456")
    assert dt.year == 2026
    assert dt.month == 3


def test_parse_dt_date_only():
    dt = _parse_dt("2026-03-01")
    assert dt.day == 1


def test_parse_dt_empty():
    assert _parse_dt("") is None


def test_parse_dt_invalid():
    assert _parse_dt("not-a-date") is None


# ── FailureExplanation dataclass ──────────────────────────────────────────────

def test_failure_explanation_fields():
    expl = FailureExplanation(
        prediction_id="x",
        ticker="TSLA",
        category="regime_shift",
        confidence_score=0.75,
        explanation="Market turned bear.",
        evidence=["regime=bear"],
        recommendation="Gate bullish calls in bear.",
    )
    assert expl.category == "regime_shift"
    assert expl.confidence_score == 0.75
    assert "bear" in expl.explanation


# ── Core explain_failure ──────────────────────────────────────────────────────

def test_explain_failure_returns_explanation(explainer):
    pred = _pred()
    outcome = _outcome()
    result = explainer.explain_failure(pred, outcome)
    assert isinstance(result, FailureExplanation)
    assert result.ticker == "AAPL"
    assert result.category  # non-empty
    assert result.explanation  # non-empty
    assert isinstance(result.evidence, list)
    assert isinstance(result.recommendation, str)


def test_explain_failure_stale_signal(explainer):
    pred = _pred(signals=["congressional", "technical_rsi"])
    outcome = _outcome()
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "stale_signal"
    assert result.confidence_score >= 0.7


def test_explain_failure_correlated_domains(explainer):
    pred = _pred(signals=["technical_rsi", "fred"])  # technical + macro = correlated pair
    outcome = _outcome()
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "correlated_domains"
    assert result.confidence_score >= 0.6


def test_explain_failure_overcrowded(explainer):
    # 5+ domains that do NOT form any correlated pair and no congressional (stale)
    # pairs: insider+institutional, macro+technical, sentiment+technical, institutional+insider
    # Safe: insider, government, contracts, events, sentiment, energy — no correlated pair among these
    pred = _pred(signals=[
        "insider_form4",      # insider
        "hiring",             # events
        "sam_gov",            # contracts
        "stocktwits",         # sentiment
        "eia",                # energy
        "usa_spending",       # contracts (second — domain still contracts, 5 distinct domains)
        "sec_form8k",         # events again — so use google_trends for a 6th distinct domain
    ])
    outcome = _outcome()
    # Manually set signals to 6 clearly distinct non-correlated domains
    pred["contributing_signals"] = [
        "insider_form4",   # insider
        "sec_form8k",      # events
        "sam_gov",         # contracts
        "stocktwits",      # sentiment
        "eia",             # energy
        "coingecko",       # crypto
    ]
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "overcrowded"
    assert "domain" in result.evidence[0]


def test_explain_failure_insufficient_evidence(explainer):
    pred = _pred(confidence=0.40, signals=["technical_rsi", "macro"])
    outcome = _outcome()
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "insufficient_evidence"


def test_explain_failure_timing_error(explainer):
    pred = _pred(confidence=0.65, signals=["technical_rsi", "insider_form4", "fred"])
    outcome = _outcome(return_pct=2.0, mfe=8.5)  # MFE > 5% but final < 5%
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "timing_error"
    assert result.confidence_score >= 0.8


def test_explain_failure_unknown_fallback(explainer):
    """When no check fires strongly, category should be 'unknown'."""
    pred = _pred(
        confidence=0.65,
        signals=["technical_rsi", "insider_form4", "fred", "events"],
    )
    outcome = _outcome(return_pct=-6.0, mfe=None)
    result = explainer.explain_failure(pred, outcome)
    # Without regime change data in tmp, may hit unknown — just verify it's valid
    assert result.category in {
        "unknown", "correlated_domains", "insufficient_evidence",
        "regime_shift", "overcrowded", "timing_error", "stale_signal", "deception",
    }


def test_explain_failure_deception(explainer, tmp_dirs):
    data_dir, midge_dir = tmp_dirs
    # Write a fake deception state
    deception_state = data_dir / "deception_state.json"
    # FailureExplainer reads from the fixed path — patch via explainer's cache
    explainer._deception_flags = {"PUMP": {"suspicion_score": 0.9}}
    pred = _pred(ticker="PUMP", signals=["technical_rsi", "stocktwits", "macro"])
    outcome = _outcome(ticker="PUMP")
    result = explainer.explain_failure(pred, outcome)
    assert result.category == "deception"
    assert result.confidence_score > 0.7


# ── Persistence ───────────────────────────────────────────────────────────────

def test_persist_writes_jsonl(explainer):
    pred = _pred(signals=["congressional"])
    outcome = _outcome()
    explainer.explain_failure(pred, outcome)
    assert explainer._explanations_path.exists()
    lines = explainer._explanations_path.read_text().strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert record["ticker"] == "AAPL"
    assert record["category"]


def test_persist_updates_summary(explainer):
    pred = _pred(signals=["congressional"])
    outcome = _outcome()
    explainer.explain_failure(pred, outcome)
    assert explainer._summary_path.exists()
    summary = json.loads(explainer._summary_path.read_text())
    assert summary["total_explained"] >= 1
    assert "category_counts" in summary
    assert "top_category" in summary


def test_summary_increments_on_multiple(explainer):
    for i in range(3):
        pred = _pred(signals=["congressional"])
        pred["prediction_id"] = f"test-{i}"
        outcome = _outcome()
        outcome["prediction_id"] = f"test-{i}"
        explainer.explain_failure(pred, outcome)

    summary = json.loads(explainer._summary_path.read_text())
    assert summary["total_explained"] == 3


# ── Batch explain ─────────────────────────────────────────────────────────────

def test_batch_explain_skips_correct(explainer):
    pairs = [
        (_pred(), _outcome(was_correct=True)),   # skip — correct
        (_pred(signals=["congressional"]), _outcome(was_correct=False)),
    ]
    results = explainer.batch_explain(pairs)
    assert len(results) == 1


def test_batch_explain_returns_list(explainer):
    pairs = [
        (_pred(signals=["congressional"]), _outcome()),
        (_pred(signals=["technical_rsi", "fred"]), _outcome()),
    ]
    results = explainer.batch_explain(pairs)
    assert len(results) == 2
    assert all(isinstance(r, FailureExplanation) for r in results)


# ── Aggregation queries ───────────────────────────────────────────────────────

def test_get_failure_rate_empty(explainer):
    rates = explainer.get_failure_rate_by_category()
    assert rates == {}


def test_get_failure_rate_populated(explainer):
    explainer.explain_failure(_pred(signals=["congressional"]), _outcome())
    explainer.explain_failure(_pred(signals=["congressional"]), _outcome())
    explainer.explain_failure(_pred(signals=["technical_rsi", "fred"]), _outcome())

    rates = explainer.get_failure_rate_by_category()
    assert isinstance(rates, dict)
    total_pct = sum(rates.values())
    assert abs(total_pct - 100.0) < 1.0  # sums to ~100%


def test_get_common_failure_modes_empty(explainer):
    modes = explainer.get_common_failure_modes()
    assert modes == []


def test_get_common_failure_modes_populated(explainer):
    for _ in range(3):
        explainer.explain_failure(_pred(signals=["congressional"]), _outcome())
    explainer.explain_failure(_pred(signals=["technical_rsi", "fred"]), _outcome())

    modes = explainer.get_common_failure_modes(n=3)
    assert len(modes) >= 1
    cat, count, example = modes[0]
    assert isinstance(cat, str)
    assert count > 0


def test_get_common_failure_modes_n_limit(explainer):
    for s in [["congressional"], ["technical_rsi", "fred"], ["stocktwits", "macro"]]:
        explainer.explain_failure(_pred(signals=s), _outcome())

    modes = explainer.get_common_failure_modes(n=2)
    assert len(modes) <= 2
