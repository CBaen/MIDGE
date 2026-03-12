"""Tests for startup_warmup and archive_scanner.

Covers:
- warm_up_from_archive: loads signals, skips expired, deduplicates, handles
  empty/missing archives, verifies convergence is NOT checked during warmup,
  handles partial archives (missing days), domain distribution.
- scan_archive_state: correct counts, domain coverage, multi-domain tickers,
  empty archive, missing directory.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.startup_warmup import warm_up_from_archive, _DOMAIN_WINDOWS, _DEFAULT_WINDOW
from mae_core.market.intelligence.archive_scanner import scan_archive_state, ArchiveState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alerter(existing_ids: list[str] | None = None):
    """Create a minimal mock ConvergenceAlerter."""
    alerter = MagicMock()
    # Simulate pre-populated buffer from load_signal_buffer()
    existing_ids = existing_ids or []
    existing_signals = [MagicMock(signal_id=sid) for sid in existing_ids]
    alerter.signals = defaultdict(list)
    if existing_signals:
        alerter.signals["technical"] = existing_signals  # put them in a domain
    # record_signal must be trackable
    alerter.record_signal = MagicMock()
    # _prune_old_signals is called inside record_signal on real object;
    # our mock doesn't need it.
    return alerter


def _write_signal(tmp_path: Path, date_str: str, signal: dict) -> None:
    """Append a signal JSON line to the archive file for the given date."""
    f = tmp_path / f"{date_str}.jsonl"
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(signal) + "\n")


def _make_signal(
    signal_id: str,
    domain: str = "technical",
    direction: str = "bullish",
    strength: float = 0.8,
    timestamp: datetime | None = None,
    symbol: str = "AAPL",
    source: str = "ta_rsi",
) -> dict:
    ts = timestamp or datetime.now()
    return {
        "signal_id": signal_id,
        "source": source,
        "symbol": symbol,
        "domain": domain,
        "direction": direction,
        "strength": strength,
        "confidence": 0.7,
        "velocity": 0.0,
        "timestamp": ts.isoformat(),
        "received_at": ts.isoformat(),
        "metadata": {"test": True},
    }


# ---------------------------------------------------------------------------
# warm_up_from_archive tests
# ---------------------------------------------------------------------------

class TestWarmUpFromArchive:

    def test_loads_signals_from_archive(self, tmp_path):
        """Warmup reads signals from archive files and injects them."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        _write_signal(tmp_path, date_str, _make_signal("sig_1", timestamp=now - timedelta(hours=1)))
        _write_signal(tmp_path, date_str, _make_signal("sig_2", timestamp=now - timedelta(hours=2)))

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)

        assert injected == 2
        assert alerter.record_signal.call_count == 2

    def test_skips_signals_older_than_domain_window(self, tmp_path):
        """Signals beyond their domain-specific window are NOT injected."""
        now = datetime.now()
        date_str = (now - timedelta(days=4)).strftime("%Y-%m-%d")
        # Technical domain has 72h window — 4 days ago is expired
        _write_signal(
            tmp_path, date_str,
            _make_signal("old_technical", domain="technical", timestamp=now - timedelta(hours=100)),
        )
        # Positioning domain has 14-day window — 4 days ago is fine
        _write_signal(
            tmp_path, date_str,
            _make_signal("recent_positioning", domain="positioning", timestamp=now - timedelta(days=4)),
        )

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)

        assert injected == 1
        call_ids = [c.kwargs["signal_id"] for c in alerter.record_signal.call_args_list]
        assert "recent_positioning" in call_ids
        assert "old_technical" not in call_ids

    def test_deduplication_skips_existing_buffer_signals(self, tmp_path):
        """Signals already restored from load_signal_buffer() are not re-injected."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        _write_signal(tmp_path, date_str, _make_signal("already_here", timestamp=now - timedelta(hours=1)))
        _write_signal(tmp_path, date_str, _make_signal("brand_new", timestamp=now - timedelta(hours=2)))

        alerter = _make_alerter(existing_ids=["already_here"])
        injected = warm_up_from_archive(alerter, tmp_path, days=7)

        assert injected == 1
        call_ids = [c.kwargs["signal_id"] for c in alerter.record_signal.call_args_list]
        assert "brand_new" in call_ids
        assert "already_here" not in call_ids

    def test_empty_archive_does_not_crash(self, tmp_path):
        """Empty signals directory returns 0 without errors."""
        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        assert injected == 0
        alerter.record_signal.assert_not_called()

    def test_missing_directory_does_not_crash(self, tmp_path):
        """Non-existent signals directory returns 0 without errors."""
        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path / "nonexistent", days=7)
        assert injected == 0

    def test_signal_count_matches_expected(self, tmp_path):
        """Exact count of injected signals equals valid, non-expired, non-duplicate records."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        for i in range(10):
            _write_signal(tmp_path, date_str, _make_signal(f"sig_{i}", timestamp=now - timedelta(hours=i)))

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        assert injected == 10

    def test_domain_distribution_is_correct(self, tmp_path):
        """Signals from different domains are all injected and routed correctly."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        domains = ["technical", "insider", "macro", "events", "sentiment"]
        for i, domain in enumerate(domains):
            _write_signal(
                tmp_path, date_str,
                _make_signal(f"sig_{domain}", domain=domain, timestamp=now - timedelta(hours=i)),
            )

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)

        assert injected == 5
        injected_domains = {c.kwargs["domain"] for c in alerter.record_signal.call_args_list}
        assert injected_domains == set(domains)

    def test_convergence_not_checked_during_warmup(self, tmp_path):
        """check_convergence() is NEVER called during warmup."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        for i in range(5):
            _write_signal(tmp_path, date_str, _make_signal(f"sig_{i}", timestamp=now - timedelta(hours=i)))

        alerter = _make_alerter()
        warm_up_from_archive(alerter, tmp_path, days=7)

        alerter.check_convergence.assert_not_called()

    def test_partial_archive_missing_days_handled(self, tmp_path):
        """Days with missing archive files are silently skipped — no crash.

        Uses positioning domain (14-day window) so signals from 3 and 5 days
        ago are still within their domain window.
        """
        now = datetime.now()
        # Write only day 3 and day 5 — days 1, 2, 4, 6, 7 are missing
        for day_offset in [3, 5]:
            date_str = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            _write_signal(
                tmp_path, date_str,
                _make_signal(
                    f"sig_day{day_offset}",
                    domain="positioning",  # 14-day window — 5 days ago is fine
                    timestamp=now - timedelta(days=day_offset, hours=1),
                ),
            )

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        assert injected == 2

    def test_files_outside_days_window_excluded(self, tmp_path):
        """Archive files older than the `days` parameter are never read."""
        now = datetime.now()
        # Within window: 3 days ago
        recent_date = (now - timedelta(days=3)).strftime("%Y-%m-%d")
        _write_signal(tmp_path, recent_date, _make_signal("in_window", timestamp=now - timedelta(hours=50)))
        # Outside window: 10 days ago (but within the 72h domain window if we only look at timestamp)
        # The file date check should exclude this file entirely
        old_date = (now - timedelta(days=10)).strftime("%Y-%m-%d")
        _write_signal(tmp_path, old_date, _make_signal("out_of_window", timestamp=now - timedelta(hours=1)))

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        # out_of_window file is from 10 days ago — excluded by file-date filter
        # in_window file: signal timestamp is 50h ago, within 72h window
        assert injected == 1
        call_ids = [c.kwargs["signal_id"] for c in alerter.record_signal.call_args_list]
        assert "in_window" in call_ids
        assert "out_of_window" not in call_ids

    def test_malformed_lines_skipped_gracefully(self, tmp_path):
        """JSON parse errors in archive lines do not abort the warmup."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        filepath = tmp_path / f"{date_str}.jsonl"
        filepath.write_text(
            "not valid json\n"
            + json.dumps(_make_signal("valid_sig", timestamp=now - timedelta(hours=1))) + "\n"
            + "{broken\n",
            encoding="utf-8",
        )

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        assert injected == 1

    def test_government_domain_uses_7day_window(self, tmp_path):
        """Government domain signals from 6 days ago are within the 7-day window."""
        now = datetime.now()
        date_str = (now - timedelta(days=6)).strftime("%Y-%m-%d")
        _write_signal(
            tmp_path, date_str,
            _make_signal("gov_6days", domain="government", timestamp=now - timedelta(days=6, hours=1)),
        )

        alerter = _make_alerter()
        injected = warm_up_from_archive(alerter, tmp_path, days=7)
        assert injected == 1

    def test_record_signal_kwargs_match_archive_fields(self, tmp_path):
        """record_signal() is called with the exact fields from the archive record."""
        now = datetime.now()
        ts = now - timedelta(hours=5)
        date_str = now.strftime("%Y-%m-%d")
        sig = _make_signal("precise_sig", domain="insider", direction="bearish",
                           strength=0.95, timestamp=ts, symbol="TSLA", source="sec_form4")
        _write_signal(tmp_path, date_str, sig)

        alerter = _make_alerter()
        warm_up_from_archive(alerter, tmp_path, days=7)

        assert alerter.record_signal.call_count == 1
        kwargs = alerter.record_signal.call_args.kwargs
        assert kwargs["signal_id"] == "precise_sig"
        assert kwargs["domain"] == "insider"
        assert kwargs["direction"] == "bearish"
        assert abs(kwargs["strength"] - 0.95) < 1e-6
        assert kwargs["source"] == "sec_form4"
        assert "symbol" in kwargs["metadata"]
        assert kwargs["metadata"]["symbol"] == "TSLA"


# ---------------------------------------------------------------------------
# scan_archive_state tests
# ---------------------------------------------------------------------------

class TestScanArchiveState:

    def test_returns_archive_state_dataclass(self, tmp_path):
        """scan_archive_state returns an ArchiveState dataclass."""
        result = scan_archive_state(tmp_path, days=30)
        assert isinstance(result, ArchiveState)

    def test_empty_archive_returns_zero_totals(self, tmp_path):
        """Empty signals directory returns zero counts."""
        state = scan_archive_state(tmp_path, days=30)
        assert state.total_signals == 0
        assert state.domain_coverage == []
        assert state.tickers_with_multi_domain == []

    def test_missing_directory_returns_zero_totals(self, tmp_path):
        """Missing signals directory returns zero counts without crash."""
        state = scan_archive_state(tmp_path / "nonexistent", days=30)
        assert state.total_signals == 0

    def test_counts_signals_correctly(self, tmp_path):
        """Total signal count matches number of records written."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        for i in range(7):
            _write_signal(tmp_path, date_str, _make_signal(f"s{i}", timestamp=now - timedelta(hours=i)))

        state = scan_archive_state(tmp_path, days=30)
        assert state.total_signals == 7

    def test_domain_coverage_lists_all_present_domains(self, tmp_path):
        """domain_coverage includes every domain that appeared in the archive."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        _write_signal(tmp_path, date_str, _make_signal("s1", domain="technical"))
        _write_signal(tmp_path, date_str, _make_signal("s2", domain="insider"))
        _write_signal(tmp_path, date_str, _make_signal("s3", domain="macro"))

        state = scan_archive_state(tmp_path, days=30)
        assert "technical" in state.domain_coverage
        assert "insider" in state.domain_coverage
        assert "macro" in state.domain_coverage

    def test_tickers_with_multi_domain_threshold(self, tmp_path):
        """Tickers appearing in 3+ domains are flagged; those in 1-2 are not."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        # AAPL in 3 domains → should appear
        for domain in ["technical", "insider", "macro"]:
            _write_signal(tmp_path, date_str, _make_signal(f"aapl_{domain}", domain=domain, symbol="AAPL"))
        # MSFT in 2 domains → should NOT appear
        for domain in ["technical", "sentiment"]:
            _write_signal(tmp_path, date_str, _make_signal(f"msft_{domain}", domain=domain, symbol="MSFT"))

        state = scan_archive_state(tmp_path, days=30)
        assert "AAPL" in state.tickers_with_multi_domain
        assert "MSFT" not in state.tickers_with_multi_domain

    def test_signals_by_domain_counts_per_domain(self, tmp_path):
        """signals_by_domain maps each domain to its signal count."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        for _ in range(5):
            _write_signal(tmp_path, date_str, _make_signal("t", domain="technical"))
        for _ in range(3):
            _write_signal(tmp_path, date_str, _make_signal("i", domain="insider"))

        # Use unique signal_ids
        filepath = tmp_path / f"{date_str}.jsonl"
        filepath.unlink()
        for i in range(5):
            _write_signal(tmp_path, date_str, _make_signal(f"t_{i}", domain="technical"))
        for i in range(3):
            _write_signal(tmp_path, date_str, _make_signal(f"i_{i}", domain="insider"))

        state = scan_archive_state(tmp_path, days=30)
        assert state.signals_by_domain.get("technical", 0) == 5
        assert state.signals_by_domain.get("insider", 0) == 3

    def test_top_50_tickers_in_signals_by_ticker(self, tmp_path):
        """signals_by_ticker contains at most 50 entries."""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        for i in range(60):
            _write_signal(
                tmp_path, date_str,
                _make_signal(f"sig_{i}", symbol=f"TICK{i:03d}"),
            )

        state = scan_archive_state(tmp_path, days=30)
        assert len(state.signals_by_ticker) <= 50
