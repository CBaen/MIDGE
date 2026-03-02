"""Tests for Session Sweep Direct-Output Bypass path (P5A).

Verifies that _check_sweep_bypass:
  - Fires only for eligible sources (session_sweep_ifvg)
  - Applies quality >= 0.65 and confidence >= 0.55 gates
  - Writes to paper_trades_bypass.jsonl (NOT paper_trades.jsonl)
  - Includes bypass_reason field
  - Deduplicates within 4-hour window
  - Skips ineligible sources

Uses chdir to tmp_path so Path("data/midge/...") resolves inside the temp dir.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlert,
    Signal,
)
from mae_core.bootstrap.market_hooks import (
    BYPASS_ELIGIBLE_SOURCES,
    _check_sweep_bypass,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_signal(source="session_sweep_ifvg", symbol="ES=F",
                 direction="bullish", quality=0.70):
    return Signal(
        signal_id=f"sig-{symbol}-{direction}",
        strength=0.75,
        domain="technical",
        direction=direction,
        timestamp=datetime.now(),
        metadata={"symbol": symbol, "quality": quality, "asset_class": "futures"},
        source=source,
    )


def _make_alert(
    direction="bullish",
    confidence=0.65,
    strength=0.75,
    symbol="ES=F",
    source="session_sweep_ifvg",
    quality=0.70,
    alert_id="TCKR-ES=F-20260301-0001",
):
    sig = _make_signal(source=source, symbol=symbol, direction=direction, quality=quality)
    return ConvergenceAlert(
        alert_id=alert_id,
        timestamp=datetime.now(),
        direction=direction,
        strength=strength,
        confidence=confidence,
        domains_converging=["technical"],
        signals=[sig],
        cross_domain_count=1,
        summary=f"TICKER {symbol} {direction.upper()}: bypass test",
        urgency="immediate",
    )


def _make_alerter(alerts=None):
    """Create a mock alerter that returns pre-built ticker alerts."""
    alerter = MagicMock()
    alerter.check_ticker_convergence.return_value = alerts or []
    return alerter


def _make_ctx():
    ctx = SimpleNamespace()
    ctx._bypass_dedup = {}
    return ctx


def _run_bypass_in(alerts, ctx, tmp_path):
    """Run _check_sweep_bypass with mocked alerter, writing to tmp_path/data/midge/.

    Uses os.chdir so Path("data/midge/paper_trades_bypass.jsonl") resolves
    inside tmp_path rather than the project root.
    """
    alerter = _make_alerter(alerts)
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _check_sweep_bypass(alerter, ctx)
    finally:
        os.chdir(original_cwd)
    return tmp_path / "data" / "midge" / "paper_trades_bypass.jsonl"


# ── Tests ──────────────────────────────────────────────────────────────────


class TestBypassEligibleSources:
    def test_eligible_sources_initial_set(self):
        """BYPASS_ELIGIBLE_SOURCES contains only 'session_sweep_ifvg' at init."""
        assert BYPASS_ELIGIBLE_SOURCES == {"session_sweep_ifvg"}

    def test_session_sweep_ifvg_is_eligible(self):
        """session_sweep_ifvg is in the bypass eligible set."""
        assert "session_sweep_ifvg" in BYPASS_ELIGIBLE_SOURCES

    def test_session_sweep_plain_is_not_eligible(self):
        """session_sweep (plain, no IFVG) is NOT in bypass eligible set."""
        assert "session_sweep" not in BYPASS_ELIGIBLE_SOURCES


class TestBypassFiring:
    def test_bypass_fires_for_eligible_source(self, tmp_path):
        """session_sweep_ifvg alert with quality >= 0.65 → bypass trade written."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        assert bypass_path.exists(), "Bypass file should be created"
        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1

    def test_bypass_skips_low_quality(self, tmp_path):
        """Alert with quality=0.50 (below 0.65 gate) → no bypass written."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.50, confidence=0.65, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        if bypass_path.exists():
            assert bypass_path.read_text().strip() == "", "No trade should be written"

    def test_bypass_confidence_gate(self, tmp_path):
        """Alert with confidence=0.40 (below 0.55 gate) → no bypass written."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.40, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        if bypass_path.exists():
            assert bypass_path.read_text().strip() == "", "No trade should be written"

    def test_bypass_skips_ineligible_source(self, tmp_path):
        """Source not in BYPASS_ELIGIBLE_SOURCES → no bypass written."""
        ctx = _make_ctx()
        # sec_form4 is not bypass eligible
        alert = _make_alert(quality=0.80, confidence=0.70, source="sec_form4")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        if bypass_path.exists():
            assert bypass_path.read_text().strip() == "", "No trade should be written"

    def test_bypass_at_quality_boundary(self, tmp_path):
        """Quality exactly at 0.65 gate → trade written (boundary inclusive)."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.65, confidence=0.60, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        # 0.65 >= 0.65 should pass
        assert bypass_path.exists(), "Boundary quality 0.65 should pass the gate"
        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1


class TestBypassOutput:
    def test_bypass_writes_separate_file(self, tmp_path):
        """Bypass writes to paper_trades_bypass.jsonl, NOT paper_trades.jsonl."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)
        main_path = tmp_path / "data" / "midge" / "paper_trades.jsonl"

        # Bypass file should be written
        assert bypass_path.exists(), "paper_trades_bypass.jsonl should be created"
        # Main paper trades should NOT be written by bypass path
        assert not main_path.exists(), "Bypass should NOT write to paper_trades.jsonl"

    def test_bypass_includes_reason(self, tmp_path):
        """Written bypass record includes bypass_reason: 'backtest_validated'."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        assert bypass_path.exists()
        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record.get("bypass_reason") == "backtest_validated"

    def test_bypass_record_has_required_fields(self, tmp_path):
        """Bypass record includes signal_id, direction, confidence, quality, generated_at."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg")

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        assert bypass_path.exists()
        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        for field in ("signal_id", "direction", "confidence", "quality",
                      "generated_at", "bypass_reason", "asset_class"):
            assert field in record, f"Missing required field: {field}"


class TestBypassDedup:
    def test_bypass_dedup_suppresses_second_write(self, tmp_path):
        """Same direction+ticker within 3h → only 1 bypass trade written."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg",
                            symbol="ES=F", direction="bullish")

        # First write
        bypass_path = _run_bypass_in([alert], ctx, tmp_path)
        # Second write immediately after (dedup should suppress)
        _run_bypass_in([alert], ctx, tmp_path)

        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1, "Dedup should suppress the second bypass write"

    def test_bypass_dedup_expiry_allows_second_write(self, tmp_path):
        """Same direction+ticker, 5h apart → second bypass write is allowed."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg",
                            symbol="NQ=F", direction="bearish")

        # Pre-populate dedup with an old timestamp (5 hours ago)
        ctx._bypass_dedup["bearish:NQ=F"] = datetime.now() - timedelta(hours=5)

        bypass_path = _run_bypass_in([alert], ctx, tmp_path)

        lines = [l for l in bypass_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1, "Expired dedup window should allow write"

    def test_bypass_dedup_key_set_after_write(self, tmp_path):
        """After a successful bypass write, the dedup key is updated in ctx."""
        ctx = _make_ctx()
        alert = _make_alert(quality=0.70, confidence=0.65, source="session_sweep_ifvg",
                            symbol="GC=F", direction="bullish")

        _run_bypass_in([alert], ctx, tmp_path)

        # The dedup dict should have been updated
        assert "bullish:GC=F" in ctx._bypass_dedup

    def test_bypass_dedup_initialized_on_ctx(self):
        """_bypass_dedup initialized as empty dict in _wire_sensing_hook."""
        import inspect
        from mae_core.bootstrap import market_hooks
        source = inspect.getsource(market_hooks._wire_sensing_hook)
        assert "_bypass_dedup" in source, "_bypass_dedup should be initialized in _wire_sensing_hook"
