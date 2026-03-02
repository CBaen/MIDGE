"""Tests for PortfolioTracker — paper position tracking and exit signal generation.

All tests are self-contained. No real price fetches are made.
price_fetcher is mocked or omitted throughout.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mae_core.market.intelligence.portfolio_tracker import (
    ClosedTrade,
    ExitSignal,
    PortfolioState,
    PortfolioTracker,
    Position,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_trade_record(
    ticker="AAPL",
    direction="buy",
    confidence=0.80,
    strength=0.70,
    kelly_fraction=0.03,
    position_size_usd=1500.0,
    timestamp=None,
    entry_price=None,
):
    """Build a paper_trades.jsonl record dict."""
    rec = {
        "timestamp": (timestamp or datetime.now().isoformat()),
        "ticker": ticker,
        "direction": direction,
        "confidence": confidence,
        "strength": strength,
        "domains": ["insider", "technical"],
        "kelly_fraction": kelly_fraction,
        "position_size_usd": position_size_usd,
    }
    if entry_price is not None:
        rec["entry_price"] = entry_price
    return rec


def _write_trades(path: Path, records: list) -> None:
    """Write a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_tracker(tmp_path: Path, records=None, price_fetcher=None) -> PortfolioTracker:
    """Create a PortfolioTracker backed by a temp directory."""
    midge_dir = tmp_path / "midge"
    midge_dir.mkdir(parents=True, exist_ok=True)
    if records is not None:
        _write_trades(midge_dir / "paper_trades.jsonl", records)
    return PortfolioTracker(data_dir=midge_dir, price_fetcher=price_fetcher)


def _mock_fetcher(price: float = 150.0) -> MagicMock:
    """Price fetcher that always returns a fixed price."""
    pd = MagicMock()
    pd.price = price
    fetcher = MagicMock()
    fetcher.get_current_price.return_value = pd
    return fetcher


# ── TestPositionLoading ────────────────────────────────────────────────────────

class TestPositionLoading:
    def test_load_single_position(self, tmp_path):
        records = [_make_trade_record("AAPL", entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        count = tracker.load_positions()
        assert count == 1
        assert tracker.has_position("AAPL")

    def test_load_multiple_positions(self, tmp_path):
        records = [
            _make_trade_record("AAPL", entry_price=150.0),
            _make_trade_record("MSFT", direction="sell", entry_price=400.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        count = tracker.load_positions()
        assert count == 2

    def test_dedup_last_entry_wins(self, tmp_path):
        """When the same ticker appears twice, the last record wins."""
        early = _make_trade_record("AAPL", direction="buy", entry_price=100.0)
        late = _make_trade_record("AAPL", direction="sell", entry_price=200.0)
        tracker = _make_tracker(tmp_path, [early, late])
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        assert pos is not None
        assert pos.direction == "sell"
        assert pos.entry_price == 200.0

    def test_empty_file_returns_zero(self, tmp_path):
        tracker = _make_tracker(tmp_path, [])
        count = tracker.load_positions()
        assert count == 0

    def test_missing_file_returns_zero(self, tmp_path):
        tracker = _make_tracker(tmp_path, records=None)  # no file written
        count = tracker.load_positions()
        assert count == 0

    def test_malformed_lines_are_skipped(self, tmp_path):
        midge_dir = tmp_path / "midge"
        midge_dir.mkdir(parents=True, exist_ok=True)
        path = midge_dir / "paper_trades.jsonl"
        with open(path, "w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps(_make_trade_record("AAPL", entry_price=150.0)) + "\n")
            fh.write('{"ticker": ""}\n')  # missing direction — invalid
        tracker = PortfolioTracker(data_dir=midge_dir)
        count = tracker.load_positions()
        assert count == 1  # only AAPL
        assert tracker.has_position("AAPL")

    def test_record_without_ticker_is_ignored(self, tmp_path):
        record = {"direction": "buy", "position_size_usd": 1000.0}
        midge_dir = tmp_path / "midge"
        midge_dir.mkdir()
        _write_trades(midge_dir / "paper_trades.jsonl", [record])
        tracker = PortfolioTracker(data_dir=midge_dir)
        count = tracker.load_positions()
        assert count == 0

    def test_entry_price_from_record(self, tmp_path):
        records = [_make_trade_record("TSLA", entry_price=250.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("TSLA")
        assert pos.entry_price == 250.0

    def test_entry_price_falls_back_to_fetcher(self, tmp_path):
        """If no entry_price in record, fetcher provides current price as proxy."""
        fetcher = _mock_fetcher(price=175.0)
        records = [_make_trade_record("NVDA")]  # no entry_price field
        tracker = _make_tracker(tmp_path, records, price_fetcher=fetcher)
        tracker.load_positions()
        pos = tracker.get_position("NVDA")
        assert pos.entry_price == 175.0

    def test_load_populates_size_usd(self, tmp_path):
        records = [_make_trade_record("AAPL", position_size_usd=3000.0, entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        assert pos.size_usd == 3000.0


# ── TestPriceUpdates ───────────────────────────────────────────────────────────

class TestPriceUpdates:
    def test_mark_to_market_long(self, tmp_path):
        fetcher = _mock_fetcher(price=165.0)
        records = [_make_trade_record("AAPL", direction="buy",
                                      entry_price=150.0, position_size_usd=1500.0)]
        tracker = _make_tracker(tmp_path, records, price_fetcher=fetcher)
        tracker.load_positions()
        tracker.update_prices()
        pos = tracker.get_position("AAPL")
        assert pos.current_price == 165.0
        # unrealized_pnl: (165-150)/150 * 1500 = 150
        assert abs(pos.unrealized_pnl - 150.0) < 0.01

    def test_mark_to_market_short(self, tmp_path):
        fetcher = _mock_fetcher(price=90.0)
        records = [_make_trade_record("TSLA", direction="sell",
                                      entry_price=100.0, position_size_usd=1000.0)]
        tracker = _make_tracker(tmp_path, records, price_fetcher=fetcher)
        tracker.load_positions()
        tracker.update_prices()
        pos = tracker.get_position("TSLA")
        # Short: price fell → profit. (100-90)/100 * 1000 = 100
        assert pos.unrealized_pnl > 0

    def test_price_fetcher_returns_none(self, tmp_path):
        fetcher = MagicMock()
        fetcher.get_current_price.return_value = None
        records = [_make_trade_record("AAPL", entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records, price_fetcher=fetcher)
        tracker.load_positions()
        tracker.update_prices()
        pos = tracker.get_position("AAPL")
        # Should leave current_price and unrealized_pnl unchanged (not crash)
        assert pos is not None

    def test_no_fetcher_does_not_crash(self, tmp_path):
        records = [_make_trade_record("AAPL", entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records, price_fetcher=None)
        tracker.load_positions()
        tracker.update_prices()  # Must not raise

    def test_unrealized_pnl_pct_property_long(self, tmp_path):
        records = [_make_trade_record("AAPL", direction="buy", entry_price=100.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        pos.current_price = 110.0
        assert abs(pos.unrealized_pnl_pct - 10.0) < 0.01

    def test_unrealized_pnl_pct_property_short(self, tmp_path):
        records = [_make_trade_record("AAPL", direction="sell", entry_price=100.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        pos.current_price = 90.0
        # Short: price fell 10% → +10% P&L
        assert abs(pos.unrealized_pnl_pct - 10.0) < 0.01


# ── TestExitSignals ────────────────────────────────────────────────────────────

class TestExitSignals:
    def _tracker_with_pnl(self, tmp_path, direction="buy", pnl_pct=0.0) -> PortfolioTracker:
        """Helper: create a tracker with a position at a given P&L %."""
        entry = 100.0
        if direction == "buy":
            current = entry * (1 + pnl_pct / 100.0)
        else:
            current = entry * (1 - pnl_pct / 100.0)

        records = [_make_trade_record("AAPL", direction=direction,
                                      entry_price=entry, position_size_usd=1000.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        pos.current_price = current
        return tracker

    def test_stop_loss_long(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "buy", pnl_pct=-6.0)
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert exits[0].reason == "stop_loss"
        assert exits[0].ticker == "AAPL"
        assert exits[0].direction == "sell"
        assert exits[0].urgency == "immediate"

    def test_stop_loss_short(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "sell", pnl_pct=-6.0)
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert exits[0].reason == "stop_loss"
        assert exits[0].direction == "buy"

    def test_take_profit_long(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "buy", pnl_pct=16.0)
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert exits[0].reason == "take_profit"
        assert exits[0].urgency == "hours"

    def test_take_profit_short(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "sell", pnl_pct=16.0)
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert exits[0].reason == "take_profit"
        assert exits[0].direction == "buy"

    def test_no_exit_within_bounds(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "buy", pnl_pct=3.0)
        exits = tracker.check_exits()
        assert exits == []

    def test_time_decay_triggers(self, tmp_path):
        records = [_make_trade_record("AAPL", direction="buy",
                                      entry_price=100.0, position_size_usd=1000.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        # Artificially age the position past the threshold
        pos = tracker.get_position("AAPL")
        pos.entry_time = datetime.now() - timedelta(days=35)
        pos.current_price = 101.0  # Within stop/profit bounds
        exits = tracker.check_exits()
        assert any(e.reason == "time_decay" for e in exits)

    def test_stop_loss_takes_priority_over_time_decay(self, tmp_path):
        records = [_make_trade_record("AAPL", direction="buy",
                                      entry_price=100.0, position_size_usd=1000.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        pos = tracker.get_position("AAPL")
        pos.entry_time = datetime.now() - timedelta(days=40)
        pos.current_price = 93.0  # -7% → stop loss
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert exits[0].reason == "stop_loss"

    def test_exit_strength_is_bounded(self, tmp_path):
        tracker = self._tracker_with_pnl(tmp_path, "buy", pnl_pct=-20.0)
        exits = tracker.check_exits()
        assert len(exits) == 1
        assert 0.0 <= exits[0].strength <= 1.0


# ── TestPortfolioState ─────────────────────────────────────────────────────────

class TestPortfolioState:
    def test_position_count(self, tmp_path):
        records = [
            _make_trade_record("AAPL", entry_price=150.0),
            _make_trade_record("MSFT", entry_price=400.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        assert state.position_count == 2

    def test_net_exposure_all_long(self, tmp_path):
        records = [
            _make_trade_record("AAPL", direction="buy",
                                position_size_usd=5000.0, entry_price=150.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        # All long → positive exposure
        assert state.net_exposure > 0

    def test_net_exposure_all_short(self, tmp_path):
        records = [
            _make_trade_record("TSLA", direction="sell",
                                position_size_usd=5000.0, entry_price=200.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        assert state.net_exposure < 0

    def test_net_exposure_balanced(self, tmp_path):
        records = [
            _make_trade_record("AAPL", direction="buy",
                                position_size_usd=5000.0, entry_price=150.0),
            _make_trade_record("TSLA", direction="sell",
                                position_size_usd=5000.0, entry_price=200.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        assert abs(state.net_exposure) < 0.01

    def test_largest_position_pct(self, tmp_path):
        records = [
            _make_trade_record("AAPL", position_size_usd=10000.0, entry_price=150.0),
            _make_trade_record("MSFT", position_size_usd=5000.0, entry_price=400.0),
        ]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        # 10000 / 50000 = 20%
        assert abs(state.largest_position_pct - 20.0) < 0.1

    def test_empty_portfolio_state(self, tmp_path):
        tracker = _make_tracker(tmp_path, [])
        tracker.load_positions()
        state = tracker.get_portfolio_state()
        assert state.position_count == 0
        assert state.unrealized_pnl == 0.0
        assert state.net_exposure == 0.0


# ── TestPositionLimits ─────────────────────────────────────────────────────────

class TestPositionLimits:
    def test_has_position_true(self, tmp_path):
        records = [_make_trade_record("AAPL", entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        assert tracker.has_position("AAPL") is True

    def test_has_position_false(self, tmp_path):
        tracker = _make_tracker(tmp_path, [])
        tracker.load_positions()
        assert tracker.has_position("AAPL") is False

    def test_is_max_exposed_by_position_count(self, tmp_path):
        tickers = [f"T{i}" for i in range(10)]
        records = [_make_trade_record(t, entry_price=100.0) for t in tickers]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        # 10 positions == _max_positions (10) → any new ticker is over-exposed
        assert tracker.is_max_exposed("NEW") is True

    def test_is_max_exposed_by_single_position_pct(self, tmp_path):
        # Single position at 20% of 50000 = 10000
        records = [_make_trade_record("AAPL", position_size_usd=10000.0, entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        # Adding more to AAPL at or above the 20% limit
        assert tracker.is_max_exposed("AAPL") is True

    def test_is_max_exposed_false_under_limits(self, tmp_path):
        records = [_make_trade_record("AAPL", position_size_usd=1000.0, entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        assert tracker.is_max_exposed("MSFT") is False

    def test_get_position_returns_none_for_unknown(self, tmp_path):
        tracker = _make_tracker(tmp_path, [])
        tracker.load_positions()
        assert tracker.get_position("AAPL") is None

    def test_statistics_keys(self, tmp_path):
        records = [_make_trade_record("AAPL", entry_price=150.0)]
        tracker = _make_tracker(tmp_path, records)
        tracker.load_positions()
        stats = tracker.get_statistics()
        assert "position_count" in stats
        assert "unrealized_pnl" in stats
        assert "net_exposure" in stats
        assert "total_value" in stats
