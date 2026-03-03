"""Tests for Wave 1 WP-A: Signal Persistence.

Covers save/load round-trips for:
  - ConvergenceAlerter  (save_signal_buffer / load_signal_buffer)
  - SomaticAnticipation (save_state / load_state)
  - DeceptionDetector   (save_state / load_state)

All file I/O is redirected to tmp_path so real data/market/ files are
never touched.  Each test is fully independent.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now()
_FRESH = _NOW  # within every domain window
_STALE_72H = _NOW - timedelta(hours=96)   # beyond the 72-h default window
_STALE_48H = _NOW - timedelta(hours=50)   # beyond SomaticAnticipation's 48-h window


# ===========================================================================
# ConvergenceAlerter persistence
# ===========================================================================

import mae_core.market.intelligence.convergence_alerter as ca_module
from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlerter,
    Signal,
)


def _make_alerter(tmp_path: Path) -> ConvergenceAlerter:
    """Return an alerter whose file I/O is redirected to tmp_path."""
    alerter = ConvergenceAlerter()
    return alerter


def _patch_ca_dir(monkeypatch, tmp_path: Path) -> None:
    """Redirect convergence_alerter._DATA_DIR to tmp_path."""
    monkeypatch.setattr(ca_module, "_DATA_DIR", tmp_path)


def _make_signal(
    signal_id: str = "sig1",
    domain: str = "insider",
    direction: str = "bullish",
    strength: float = 0.8,
    timestamp: datetime | None = None,
    source: str = "sec_form4",
) -> Signal:
    return Signal(
        signal_id=signal_id,
        strength=strength,
        domain=domain,
        direction=direction,
        timestamp=timestamp or _FRESH,
        source=source,
    )


# --- ConvergenceAlerter tests ------------------------------------------------

class TestConvergenceAlerterPersistence:

    def test_save_load_round_trip_empty(self, tmp_path, monkeypatch):
        """Save with no signals; load into a new instance; both signal dicts empty."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert sum(len(v) for v in alerter2.signals.values()) == 0

    def test_save_load_round_trip_with_signals(self, tmp_path, monkeypatch):
        """Record signals, save, create new instance, load — signals are restored."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        alerter.record_signal("sig_insider", 0.8, "insider", direction="bullish",
                              timestamp=_FRESH)
        alerter.record_signal("sig_sentiment", 0.7, "sentiment", direction="bullish",
                              timestamp=_FRESH)
        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        total = sum(len(v) for v in alerter2.signals.values())
        assert total == 2

    def test_expired_signals_pruned_on_load(self, tmp_path, monkeypatch):
        """Signals whose timestamp is beyond the domain window are dropped on load."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        # Default window is 72 h; use a 96-h-old timestamp
        alerter.record_signal("old_sig", 0.8, "insider", direction="bullish",
                              timestamp=_STALE_72H)
        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert sum(len(v) for v in alerter2.signals.values()) == 0

    def test_fresh_signals_preserved_on_load(self, tmp_path, monkeypatch):
        """Recent signals (within convergence window) survive a save/load cycle."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        alerter.record_signal("fresh_sig", 0.9, "sentiment", direction="bearish",
                              timestamp=_FRESH)
        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert len(alerter2.signals["sentiment"]) == 1
        assert alerter2.signals["sentiment"][0].signal_id == "fresh_sig"

    def test_alert_counter_restored(self, tmp_path, monkeypatch):
        """_alert_counter value persists across save/load."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)
        alerter._alert_counter = 42

        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert alerter2._alert_counter == 42

    def test_last_alert_times_restored(self, tmp_path, monkeypatch):
        """_last_alert_times dict persists across save/load."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)
        alerter._last_alert_times["AAPL_bullish"] = _FRESH

        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert "AAPL_bullish" in alerter2._last_alert_times
        restored = alerter2._last_alert_times["AAPL_bullish"]
        assert isinstance(restored, datetime)
        # Within 1 second of original (isoformat round-trip)
        assert abs((restored - _FRESH).total_seconds()) < 1

    def test_atomic_write_pattern(self, tmp_path, monkeypatch):
        """The .tmp intermediate file must not be present after a successful save."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)
        alerter.record_signal("sig1", 0.7, "insider", timestamp=_FRESH)

        alerter.save_signal_buffer()

        assert not (tmp_path / "signal_buffer.tmp").exists()
        assert not (tmp_path / "alerter_state.tmp").exists()
        assert (tmp_path / "signal_buffer.json").exists()
        assert (tmp_path / "alerter_state.json").exists()

    def test_load_with_missing_files(self, tmp_path, monkeypatch):
        """load_signal_buffer returns gracefully when no files exist on disk."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        # Should not raise; signals remain empty
        alerter.load_signal_buffer()

        assert sum(len(v) for v in alerter.signals.values()) == 0
        assert alerter._alert_counter == 0

    def test_load_with_corrupt_json(self, tmp_path, monkeypatch):
        """Corrupt JSON in signal_buffer.json is handled gracefully — no crash."""
        _patch_ca_dir(monkeypatch, tmp_path)
        (tmp_path / "signal_buffer.json").write_text("{NOT VALID JSON", encoding="utf-8")

        alerter = _make_alerter(tmp_path)
        # Must not raise
        alerter.load_signal_buffer()

        assert sum(len(v) for v in alerter.signals.values()) == 0

    def test_save_creates_directory(self, tmp_path, monkeypatch):
        """save_signal_buffer creates the data directory if it does not exist."""
        nested = tmp_path / "deep" / "nested"
        _patch_ca_dir(monkeypatch, tmp_path)
        # Override _DATA_DIR to a not-yet-existing sub-path
        monkeypatch.setattr(ca_module, "_DATA_DIR", nested)

        alerter = _make_alerter(tmp_path)
        alerter.save_signal_buffer()

        assert nested.exists()
        assert (nested / "signal_buffer.json").exists()

    def test_domain_window_filtering(self, tmp_path, monkeypatch):
        """Signals within their domain window survive; those beyond it are pruned."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        # "government" domain window is 7 days (168 h)
        # A signal 5 days old (120 h) should survive; 10 days old should not.
        recent_gov = _NOW - timedelta(hours=120)
        old_gov = _NOW - timedelta(hours=250)

        alerter.record_signal("gov_recent", 0.7, "government", timestamp=recent_gov)
        alerter.record_signal("gov_old", 0.7, "government", timestamp=old_gov)
        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        gov_ids = {s.signal_id for s in alerter2.signals.get("government", [])}
        assert "gov_recent" in gov_ids
        assert "gov_old" not in gov_ids

    def test_multiple_domains_round_trip(self, tmp_path, monkeypatch):
        """Signals across multiple domains all persist correctly."""
        _patch_ca_dir(monkeypatch, tmp_path)
        alerter = _make_alerter(tmp_path)

        for i, domain in enumerate(["insider", "sentiment", "technical", "macro"]):
            alerter.record_signal(f"sig_{domain}", 0.7 + i * 0.02, domain,
                                  timestamp=_FRESH)
        alerter.save_signal_buffer()

        alerter2 = _make_alerter(tmp_path)
        alerter2.load_signal_buffer()

        assert len(alerter2.signals["insider"]) == 1
        assert len(alerter2.signals["sentiment"]) == 1
        assert len(alerter2.signals["technical"]) == 1
        assert len(alerter2.signals["macro"]) == 1


# ===========================================================================
# SomaticAnticipation persistence
# ===========================================================================

import mae_core.market.intelligence.somatic_anticipation as sa_module
from mae_core.market.intelligence.somatic_anticipation import SomaticAnticipation


def _patch_sa_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sa_module, "_STATE_DIR", tmp_path)


def _make_somatic(tmp_path: Path) -> SomaticAnticipation:
    return SomaticAnticipation()


class TestSomaticAnticipationPersistence:

    def test_save_load_empty_state(self, tmp_path, monkeypatch):
        """No ticker states — round trip works without error."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        assert len(sa2._ticker_states) == 0
        assert len(sa2._last_fired) == 0

    def test_save_load_ticker_states(self, tmp_path, monkeypatch):
        """Record signals to build TickerState, save, load, verify count."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.record_signal("AAPL", "insider", "bullish", 0.8, _FRESH)
        sa.record_signal("AAPL", "sentiment", "bullish", 0.7, _FRESH)
        sa.record_signal("MSFT", "technical", "bearish", 0.6, _FRESH)
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        assert "AAPL" in sa2._ticker_states
        assert "MSFT" in sa2._ticker_states

    def test_expired_states_pruned(self, tmp_path, monkeypatch):
        """States whose last_signal_time is older than 48 h are dropped on load."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.record_signal("OLD", "insider", "bullish", 0.8, _STALE_48H)
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        assert "OLD" not in sa2._ticker_states

    def test_set_and_counter_types_restored(self, tmp_path, monkeypatch):
        """domains_active is a set and directions is a Counter after deserialization."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.record_signal("TSLA", "insider", "bullish", 0.9, _FRESH)
        sa.record_signal("TSLA", "sentiment", "bullish", 0.8, _FRESH)
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        state = sa2._ticker_states["TSLA"]
        assert isinstance(state.domains_active, set)
        assert isinstance(state.directions, Counter)
        assert "insider" in state.domains_active
        assert "sentiment" in state.domains_active
        assert state.directions["bullish"] == 2

    def test_datetime_fields_restored(self, tmp_path, monkeypatch):
        """first_signal_time and last_signal_time are datetime objects after load."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.record_signal("NVDA", "macro", "bullish", 0.75, _FRESH)
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        state = sa2._ticker_states["NVDA"]
        assert isinstance(state.first_signal_time, datetime)
        assert isinstance(state.last_signal_time, datetime)

    def test_last_fired_restored(self, tmp_path, monkeypatch):
        """_last_fired dict (ticker -> float timestamp) survives round trip."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        import time
        sa._last_fired["AAPL"] = time.time()
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        assert "AAPL" in sa2._last_fired
        assert isinstance(sa2._last_fired["AAPL"], float)

    def test_missing_file_graceful(self, tmp_path, monkeypatch):
        """load_state with no file on disk does not crash."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.load_state()  # no file exists

        assert len(sa._ticker_states) == 0

    def test_corrupt_json_graceful(self, tmp_path, monkeypatch):
        """load_state with invalid JSON does not crash."""
        _patch_sa_dir(monkeypatch, tmp_path)
        (tmp_path / "somatic_state.json").write_text("NOTJSON{{{", encoding="utf-8")

        sa = _make_somatic(tmp_path)
        sa.load_state()  # must not raise

        assert len(sa._ticker_states) == 0

    def test_atomic_write(self, tmp_path, monkeypatch):
        """The .json.tmp intermediate file is not present after successful save."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)
        sa.record_signal("GLD", "macro", "neutral", 0.5, _FRESH)

        sa.save_state()

        assert not (tmp_path / "somatic_state.json.tmp").exists()
        assert (tmp_path / "somatic_state.json").exists()

    def test_partial_expiry(self, tmp_path, monkeypatch):
        """Fresh ticker states survive; expired ones are filtered on load."""
        _patch_sa_dir(monkeypatch, tmp_path)
        sa = _make_somatic(tmp_path)

        sa.record_signal("FRESH_TICKER", "insider", "bullish", 0.8, _FRESH)
        sa.record_signal("OLD_TICKER", "sentiment", "bearish", 0.7, _STALE_48H)
        sa.save_state()

        sa2 = _make_somatic(tmp_path)
        sa2.load_state()

        assert "FRESH_TICKER" in sa2._ticker_states
        assert "OLD_TICKER" not in sa2._ticker_states


# ===========================================================================
# DeceptionDetector persistence
# ===========================================================================

import mae_core.market.edge.deception_detector as dd_module
from mae_core.market.edge.deception_detector import DeceptionDetector, _MAX_HISTORY


def _patch_dd_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dd_module, "_STATE_DIR", tmp_path)


def _make_detector(tmp_path: Path) -> DeceptionDetector:
    return DeceptionDetector()


class TestDeceptionDetectorPersistence:

    def test_save_load_empty(self, tmp_path, monkeypatch):
        """No signal history — round trip works without error."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        assert len(dd2._signal_history) == 0

    def test_save_load_with_history(self, tmp_path, monkeypatch):
        """Record signals, save, load into new instance — history is restored."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        dd.record_signal("AAPL", "sentiment", "bullish", 0.8, timestamp=_FRESH)
        dd.record_signal("AAPL", "insider", "bearish", 0.7, timestamp=_FRESH)
        dd.record_signal("MSFT", "technical", "bullish", 0.6, timestamp=_FRESH)
        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        assert "AAPL" in dd2._signal_history
        assert len(dd2._signal_history["AAPL"]) == 2
        assert "MSFT" in dd2._signal_history
        assert len(dd2._signal_history["MSFT"]) == 1

    def test_expired_records_pruned(self, tmp_path, monkeypatch):
        """Records older than _DETECTION_WINDOW_HOURS (48 h) are dropped on load."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        old_ts = _NOW - timedelta(hours=50)  # beyond 48-h window
        dd.record_signal("GOOG", "sentiment", "bullish", 0.8, timestamp=old_ts)
        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        # Ticker may not be present at all, or deque is empty
        total = sum(len(q) for q in dd2._signal_history.values())
        assert total == 0

    def test_deque_maxlen_preserved(self, tmp_path, monkeypatch):
        """Deques maintain _MAX_HISTORY maxlen after deserialization."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        dd.record_signal("META", "sentiment", "bullish", 0.5, timestamp=_FRESH)
        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        dq = dd2._signal_history["META"]
        assert dq.maxlen == _MAX_HISTORY

    def test_missing_file_graceful(self, tmp_path, monkeypatch):
        """load_state with no file on disk does not crash."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        dd.load_state()  # no file

        assert len(dd._signal_history) == 0

    def test_corrupt_json_graceful(self, tmp_path, monkeypatch):
        """load_state with corrupt JSON does not crash."""
        _patch_dd_dir(monkeypatch, tmp_path)
        (tmp_path / "deception_state.json").write_text("!!!BAD JSON!!!", encoding="utf-8")

        dd = _make_detector(tmp_path)
        dd.load_state()  # must not raise

        assert len(dd._signal_history) == 0

    def test_signal_record_fields(self, tmp_path, monkeypatch):
        """All _SignalRecord fields are preserved through save/load."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        ts = _FRESH
        dd.record_signal("NVDA", "insider", "bearish", 0.92, timestamp=ts)
        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        record = dd2._signal_history["NVDA"][0]
        assert record.ticker == "NVDA"
        assert record.domain == "insider"
        assert record.direction == "bearish"
        assert abs(record.strength - 0.92) < 1e-6
        assert isinstance(record.timestamp, datetime)
        assert abs((record.timestamp - ts).total_seconds()) < 1

    def test_multiple_tickers_round_trip(self, tmp_path, monkeypatch):
        """Signal history for multiple tickers all persists correctly."""
        _patch_dd_dir(monkeypatch, tmp_path)
        dd = _make_detector(tmp_path)

        tickers = ["AAPL", "TSLA", "AMZN", "NVDA"]
        for ticker in tickers:
            dd.record_signal(ticker, "sentiment", "bullish", 0.75, timestamp=_FRESH)
        dd.save_state()

        dd2 = _make_detector(tmp_path)
        dd2.load_state()

        for ticker in tickers:
            assert ticker in dd2._signal_history
            assert len(dd2._signal_history[ticker]) == 1
