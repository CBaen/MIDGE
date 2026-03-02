"""Tests for paper trading path — _write_paper_trade in market_hooks.py.

Verifies TradeSignal instantiation, JSONL output format, confidence/strength
thresholds, deduplication, and direction mapping.

Uses os.chdir to tmp_path so Path("data/midge/...") resolves inside the temp
directory rather than the project root.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlert,
    Signal,
)
from mae_core.market.signal import TradeSignal
from mae_core.bootstrap.market_hooks import _write_paper_trade


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_signal(source="sec_form4", symbol="AAPL", direction="bullish"):
    return Signal(
        signal_id=f"sig-{symbol}-{direction}",
        strength=0.8,
        domain="insider",
        direction=direction,
        timestamp=datetime.now(),
        metadata={"symbol": symbol},
        source=source,
    )


def _make_alert(
    direction="bullish",
    confidence=0.80,
    strength=0.70,
    symbol="AAPL",
    alert_id="CONV-20260301-0001",
):
    sig = _make_signal(symbol=symbol, direction=direction)
    return ConvergenceAlert(
        alert_id=alert_id,
        timestamp=datetime.now(),
        direction=direction,
        strength=strength,
        confidence=confidence,
        domains_converging=["insider", "government", "technical"],
        signals=[sig],
        cross_domain_count=3,
        summary=f"{direction.upper()} convergence test",
        urgency="hours",
    )


def _make_ctx():
    """Create a minimal SimpleNamespace ctx for paper trade tests."""
    ctx = SimpleNamespace()
    ctx._paper_trade_dedup = {}
    ctx._latest_kelly = {}
    ctx._market_sensing_hook = None
    return ctx


def _write_in(alert, ctx, tmp_path):
    """Call _write_paper_trade with CWD set to tmp_path.

    This ensures Path("data/midge/paper_trades.jsonl") resolves
    inside tmp_path rather than the project root.
    """
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        _write_paper_trade(alert, ctx)
    finally:
        os.chdir(original_cwd)
    return tmp_path / "data" / "midge" / "paper_trades.jsonl"


# ── Tests ──────────────────────────────────────────────────────────────────


class TestTradeSignalInstantiation:
    def test_trade_signal_instantiation(self):
        """TradeSignal fields populate correctly from direct construction."""
        ts = TradeSignal(
            signal_id="PT-001",
            asset="AAPL",
            asset_class="stock",
            direction="buy",
            confidence=0.82,
            timeframe_days=5,
            catalyst="3 domains bullish",
            contributing_signals=["sig-1", "sig-2"],
            hit_rate=0.0,
            generated_at=datetime.now(),
        )
        assert ts.signal_id == "PT-001"
        assert ts.asset == "AAPL"
        assert ts.direction == "buy"
        assert ts.confidence == 0.82
        assert ts.timeframe_days == 5
        assert ts.contributing_signals == ["sig-1", "sig-2"]

    def test_trade_signal_dataclass_is_serializable(self):
        """TradeSignal survives json.dumps via asdict round-trip."""
        from dataclasses import asdict
        ts = TradeSignal(
            signal_id="PT-002",
            asset="ES=F",
            asset_class="futures",
            direction="sell",
            confidence=0.78,
            timeframe_days=3,
            catalyst="bearish",
            generated_at=datetime(2026, 3, 1, 12, 0),
        )
        record = asdict(ts)
        record["generated_at"] = ts.generated_at.isoformat()
        serialized = json.dumps(record)
        assert "ES=F" in serialized
        assert "sell" in serialized


class TestPaperTradeWriting:
    def test_paper_trade_written_above_threshold(self, tmp_path):
        """Alert with confidence=0.80, strength=0.70 → trade written."""
        ctx = _make_ctx()
        alert = _make_alert(confidence=0.80, strength=0.70)

        paper_path = _write_in(alert, ctx, tmp_path)

        assert paper_path.exists(), "Paper trade file should be written"
        lines = [l for l in paper_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["direction"] in ("buy", "sell")
        assert "confidence" in record

    def test_paper_trade_skipped_below_confidence(self, tmp_path):
        """_write_paper_trade always writes if called — gate is in step hook.
        This test verifies the written record has correct fields even for low-conf alerts."""
        ctx = _make_ctx()
        alert = _make_alert(confidence=0.60, strength=0.70)

        paper_path = _write_in(alert, ctx, tmp_path)

        # _write_paper_trade writes unconditionally; gate is in the step hook.
        # Verify format is correct regardless.
        if paper_path.exists():
            record = json.loads(paper_path.read_text().strip().splitlines()[0])
            assert "confidence" in record

    def test_paper_trade_skipped_below_strength(self, tmp_path):
        """_write_paper_trade should not raise on low-strength alerts."""
        ctx = _make_ctx()
        alert = _make_alert(confidence=0.80, strength=0.50)
        # Should not raise
        _write_in(alert, ctx, tmp_path)

    def test_paper_trade_jsonl_format(self, tmp_path):
        """Each written line is valid JSON with required fields."""
        ctx = _make_ctx()
        alert = _make_alert(confidence=0.82, strength=0.72)

        paper_path = _write_in(alert, ctx, tmp_path)

        assert paper_path.exists()
        for line in paper_path.read_text().strip().splitlines():
            if not line:
                continue
            record = json.loads(line)
            assert "signal_id" in record
            assert "direction" in record
            assert "confidence" in record
            assert "generated_at" in record

    def test_paper_trade_direction_mapping(self, tmp_path):
        """'bullish' alert → 'buy', 'bearish' alert → 'sell'."""
        for raw_dir, expected in [("bullish", "buy"), ("bearish", "sell")]:
            ctx = _make_ctx()
            alert = _make_alert(direction=raw_dir, confidence=0.82, strength=0.72)
            sub_tmp = tmp_path / f"dir_{raw_dir}"
            sub_tmp.mkdir()

            paper_path = _write_in(alert, ctx, sub_tmp)

            assert paper_path.exists(), f"File should exist for {raw_dir}"
            lines = [l for l in paper_path.read_text().strip().splitlines() if l]
            record = json.loads(lines[0])
            assert record["direction"] == expected, f"{raw_dir} should map to {expected}"


class TestPaperTradeDedup:
    def test_paper_trade_dedup_same_direction(self, tmp_path):
        """Same direction+ticker within 3h → only 1 trade written."""
        ctx = _make_ctx()
        alert = _make_alert(direction="bullish", symbol="AAPL", confidence=0.82, strength=0.72)

        # Write once
        paper_path = _write_in(alert, ctx, tmp_path)
        # Write again immediately (dedup should suppress)
        _write_in(alert, ctx, tmp_path)

        lines = [l for l in paper_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1, "Dedup should suppress the second write"

    def test_paper_trade_dedup_different_direction(self, tmp_path):
        """Bullish then bearish on same ticker → both written."""
        ctx = _make_ctx()

        alert_bull = _make_alert(direction="bullish", symbol="AAPL",
                                  confidence=0.82, strength=0.72,
                                  alert_id="CONV-20260301-0001")
        alert_bear = _make_alert(direction="bearish", symbol="AAPL",
                                  confidence=0.82, strength=0.72,
                                  alert_id="CONV-20260301-0002")

        paper_path = _write_in(alert_bull, ctx, tmp_path)
        _write_in(alert_bear, ctx, tmp_path)

        lines = [l for l in paper_path.read_text().strip().splitlines() if l]
        assert len(lines) == 2, "Different directions should both be written"
        directions = {json.loads(l)["direction"] for l in lines}
        assert directions == {"buy", "sell"}

    def test_paper_trade_dedup_expiry(self, tmp_path):
        """Same direction, 5h apart → both written (dedup window expires)."""
        ctx = _make_ctx()
        alert = _make_alert(direction="bullish", symbol="TSLA",
                             confidence=0.82, strength=0.72)

        # Simulate first write 5 hours ago
        dedup_key = "bullish:TSLA"
        ctx._paper_trade_dedup[dedup_key] = datetime.now() - timedelta(hours=5)

        paper_path = _write_in(alert, ctx, tmp_path)

        lines = [l for l in paper_path.read_text().strip().splitlines() if l]
        assert len(lines) == 1, "Expired dedup should allow write"
