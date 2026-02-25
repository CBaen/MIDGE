"""Tests for backfill_archives.py

Covers: _serialize_signal, load_existing_ids, write_signals, get_tickers,
and convert_all. No real API calls are made — all external clients are mocked.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.signal import MarketSignal


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_signal():
    """A minimal but complete MarketSignal for use across tests."""
    return MarketSignal(
        signal_id="test:AAPL:2026-01-15",
        source="sec_form4",
        symbol="AAPL",
        asset_class="stock",
        domain="insider",
        direction="bullish",
        strength=0.73,
        confidence=0.70,
        decay_rate=0.035,
        timestamp=datetime(2026, 1, 15),
        received_at=datetime(2026, 1, 20),
        metadata={"filer_name": "Test"},
    )


# ---------------------------------------------------------------------------
# _serialize_signal
# ---------------------------------------------------------------------------

class TestSerializeSignal:
    def test_serialize_signal_has_required_keys(self, sample_signal):
        """All JSONL archive keys must be present in the serialized dict."""
        import backfill_archives
        record = backfill_archives._serialize_signal(sample_signal)

        required_keys = {
            "signal_id", "source", "symbol", "domain", "direction",
            "strength", "confidence", "velocity", "timestamp", "received_at",
            "metadata",
        }
        assert required_keys.issubset(record.keys()), (
            f"Missing keys: {required_keys - record.keys()}"
        )

    def test_serialize_signal_timestamps_are_iso(self, sample_signal):
        """timestamp and received_at must be ISO 8601 strings, not datetime objects."""
        import backfill_archives
        record = backfill_archives._serialize_signal(sample_signal)

        # Must be strings
        assert isinstance(record["timestamp"], str), "timestamp must be a string"
        assert isinstance(record["received_at"], str), "received_at must be a string"

        # Must round-trip as datetime (valid ISO 8601)
        parsed_ts = datetime.fromisoformat(record["timestamp"])
        parsed_ra = datetime.fromisoformat(record["received_at"])
        assert parsed_ts == sample_signal.timestamp
        assert parsed_ra == sample_signal.received_at

    def test_serialize_signal_values_match_source(self, sample_signal):
        """Scalar values in the record must match those on the signal object."""
        import backfill_archives
        record = backfill_archives._serialize_signal(sample_signal)

        assert record["signal_id"] == sample_signal.signal_id
        assert record["source"] == sample_signal.source
        assert record["symbol"] == sample_signal.symbol
        assert record["domain"] == sample_signal.domain
        assert record["direction"] == sample_signal.direction
        assert record["strength"] == sample_signal.strength
        assert record["confidence"] == sample_signal.confidence
        assert record["velocity"] == sample_signal.velocity
        assert record["metadata"] == sample_signal.metadata


# ---------------------------------------------------------------------------
# load_existing_ids
# ---------------------------------------------------------------------------

class TestLoadExistingIds:
    def test_load_existing_ids_from_jsonl(self, tmp_path):
        """Reading a JSONL with 3 records must return exactly those 3 signal_ids."""
        import backfill_archives

        jsonl_file = tmp_path / "2026-01-15.jsonl"
        records = [
            {"signal_id": "id:A:2026-01-15", "source": "sec_form4", "symbol": "AAPL"},
            {"signal_id": "id:B:2026-01-15", "source": "congressional", "symbol": "MSFT"},
            {"signal_id": "id:C:2026-01-15", "source": "contract_award", "symbol": "LMT"},
        ]
        jsonl_file.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

        ids = backfill_archives.load_existing_ids(jsonl_file)

        assert ids == {"id:A:2026-01-15", "id:B:2026-01-15", "id:C:2026-01-15"}
        assert isinstance(ids, set)

    def test_load_existing_ids_missing_file(self, tmp_path):
        """A path that does not exist must return an empty set, not raise."""
        import backfill_archives

        nonexistent = tmp_path / "does_not_exist.jsonl"
        ids = backfill_archives.load_existing_ids(nonexistent)

        assert ids == set()

    def test_load_existing_ids_skips_malformed_lines(self, tmp_path):
        """Lines that are not valid JSON must be silently skipped."""
        import backfill_archives

        jsonl_file = tmp_path / "mixed.jsonl"
        jsonl_file.write_text(
            '{"signal_id": "good:id:1"}\n'
            "THIS IS NOT JSON\n"
            '{"signal_id": "good:id:2"}\n',
            encoding="utf-8",
        )

        ids = backfill_archives.load_existing_ids(jsonl_file)
        assert ids == {"good:id:1", "good:id:2"}


# ---------------------------------------------------------------------------
# write_signals
# ---------------------------------------------------------------------------

class TestWriteSignals:
    def _make_signal(self, signal_id: str, timestamp: datetime) -> MarketSignal:
        return MarketSignal(
            signal_id=signal_id,
            source="sec_form4",
            symbol="AAPL",
            asset_class="stock",
            domain="insider",
            direction="bullish",
            strength=0.5,
            confidence=0.70,
            decay_rate=0.035,
            timestamp=timestamp,
            received_at=timestamp,
            metadata={},
        )

    def test_write_signals_creates_daily_files(self, tmp_path):
        """Signals on 2 different dates must produce exactly 2 JSONL files."""
        import backfill_archives

        signals = [
            self._make_signal("id:1:2026-01-10", datetime(2026, 1, 10)),
            self._make_signal("id:2:2026-01-10", datetime(2026, 1, 10)),
            self._make_signal("id:3:2026-01-12", datetime(2026, 1, 12)),
        ]

        with patch.object(backfill_archives, "SIGNALS_DIR", tmp_path):
            written = backfill_archives.write_signals(signals, dry_run=False)

        assert set(written.keys()) == {"2026-01-10", "2026-01-12"}
        assert written["2026-01-10"] == 2
        assert written["2026-01-12"] == 1

        assert (tmp_path / "2026-01-10.jsonl").exists()
        assert (tmp_path / "2026-01-12.jsonl").exists()

    def test_write_signals_dry_run(self, tmp_path):
        """dry_run=True must return counts but create no files on disk."""
        import backfill_archives

        signals = [
            self._make_signal("id:1:2026-01-10", datetime(2026, 1, 10)),
            self._make_signal("id:2:2026-01-11", datetime(2026, 1, 11)),
        ]

        with patch.object(backfill_archives, "SIGNALS_DIR", tmp_path):
            written = backfill_archives.write_signals(signals, dry_run=True)

        # Counts are returned
        assert written["2026-01-10"] == 1
        assert written["2026-01-11"] == 1

        # No files written to disk
        files_created = list(tmp_path.glob("*.jsonl"))
        assert files_created == [], f"Expected no files but found: {files_created}"

    def test_write_signals_dedup(self, tmp_path):
        """Writing the same signals twice must not produce duplicate lines."""
        import backfill_archives

        signals = [
            self._make_signal("id:dedup:2026-01-15", datetime(2026, 1, 15)),
            self._make_signal("id:unique:2026-01-15", datetime(2026, 1, 15)),
        ]

        with patch.object(backfill_archives, "SIGNALS_DIR", tmp_path):
            # First write
            backfill_archives.write_signals(signals, dry_run=False)
            # Second write with the same signals (idempotent append)
            backfill_archives.write_signals(signals, dry_run=False)

        jsonl_path = tmp_path / "2026-01-15.jsonl"
        lines = [l for l in jsonl_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 2, (
            f"Expected 2 unique records after double write, got {len(lines)}"
        )
        ids_in_file = {json.loads(l)["signal_id"] for l in lines}
        assert ids_in_file == {"id:dedup:2026-01-15", "id:unique:2026-01-15"}


# ---------------------------------------------------------------------------
# get_tickers
# ---------------------------------------------------------------------------

class TestGetTickers:
    def test_get_tickers_includes_backfill(self):
        """Every ticker in BACKFILL_TICKERS must appear in the result."""
        import backfill_archives

        tickers = backfill_archives.get_tickers()
        for t in backfill_archives.BACKFILL_TICKERS:
            assert t in tickers, f"BACKFILL_TICKERS entry '{t}' missing from get_tickers()"

    def test_get_tickers_no_duplicates(self):
        """get_tickers must return a list with no duplicate entries."""
        import backfill_archives

        tickers = backfill_archives.get_tickers()
        assert len(tickers) == len(set(tickers)), "Duplicate tickers returned"

    def test_get_tickers_merges_watchlist(self, tmp_path):
        """Extra tickers in watchlist.json must be appended without duplicates."""
        import backfill_archives

        watchlist = {"tickers": ["TSLA", "AAPL"]}   # AAPL already in BACKFILL_TICKERS
        watchlist_file = tmp_path / "watchlist.json"
        watchlist_file.write_text(json.dumps(watchlist), encoding="utf-8")

        with patch.object(backfill_archives, "WATCHLIST_PATH", watchlist_file):
            tickers = backfill_archives.get_tickers()

        assert "TSLA" in tickers
        assert tickers.count("AAPL") == 1   # no duplicate from watchlist


# ---------------------------------------------------------------------------
# convert_all
# ---------------------------------------------------------------------------

class TestConvertAll:
    def _mock_insider_trade(self, signal_id_suffix: str):
        """Build a minimal mock InsiderTrade that from_insider_trade can consume."""
        trade = MagicMock()
        trade.is_purchase = True
        trade.total_value = 500_000
        trade.transaction_code = "P"
        trade.is_plan_sale = False
        trade.transaction_date = "2026-01-15"
        trade.filing_date = "2026-01-20"
        trade.ticker_symbol = "AAPL"
        trade.filer_name = "Test Insider"
        trade.filer_title = "CFO"
        trade.transaction_type = "Purchase"
        trade.shares = 1000
        trade.price_per_share = 500.0
        trade.company_name = "Apple Inc"
        trade.accession_number = f"0001234-{signal_id_suffix}"
        trade.decay_rate = 0.035
        return trade

    def test_convert_all_deduplicates(self):
        """Passing two insider trades with the same signal_id must yield only 1 signal."""
        import backfill_archives

        # Both trades produce the same signal_id because they share ticker + date
        trade_a = self._mock_insider_trade("A")
        trade_b = self._mock_insider_trade("B")
        # trade_b has same ticker_symbol and transaction_date — identical signal_id

        signals = backfill_archives.convert_all(
            form4_trades=[trade_a, trade_b],
            house_trades=[],
            senate_trades=[],
            contracts=[],
            efts_hits=[],
        )

        assert len(signals) == 1, (
            f"Expected 1 deduplicated signal, got {len(signals)}"
        )

    def test_convert_all_empty_inputs(self):
        """All-empty inputs must return an empty list without raising."""
        import backfill_archives

        signals = backfill_archives.convert_all(
            form4_trades=[],
            house_trades=[],
            senate_trades=[],
            contracts=[],
            efts_hits=[],
        )

        assert signals == []

    def test_convert_all_bad_record_skipped(self):
        """A malformed raw record that raises in the adapter must be silently skipped."""
        import backfill_archives

        bad_trade = MagicMock()
        bad_trade.is_purchase = True
        # Missing required attributes — from_insider_trade will raise AttributeError
        del bad_trade.total_value

        # Should not propagate the exception
        signals = backfill_archives.convert_all(
            form4_trades=[bad_trade],
            house_trades=[],
            senate_trades=[],
            contracts=[],
            efts_hits=[],
        )

        assert signals == []

    def test_convert_all_with_short_interest(self):
        """ShortInterestData records must produce finra_short signals."""
        import backfill_archives
        from mae_core.market.apis.finra_short_interest import ShortInterestData

        si = ShortInterestData(
            symbol="NVDA",
            date="2026-01-15",
            short_volume=5_000_000,
            total_volume=8_000_000,
            short_ratio=0.625,
        )

        signals = backfill_archives.convert_all(
            form4_trades=[], house_trades=[], senate_trades=[],
            contracts=[], efts_hits=[], short_interest=[si],
        )

        assert len(signals) == 1
        assert signals[0].source == "finra_short"
        assert signals[0].symbol == "NVDA"
        assert signals[0].domain == "institutional"

    def test_convert_all_with_macro_indicators(self):
        """MacroIndicator records must produce fred_macro signals."""
        import backfill_archives
        from mae_core.market.apis.fred_client import MacroIndicator

        mi = MacroIndicator(
            series_id="T10Y2Y",
            series_name="10Y-2Y Treasury Spread",
            value=-0.15,
            date="2026-01-10",
            signal_type="yield_curve",
            direction="bearish",
        )

        signals = backfill_archives.convert_all(
            form4_trades=[], house_trades=[], senate_trades=[],
            contracts=[], efts_hits=[], macro_indicators=[mi],
        )

        assert len(signals) == 1
        assert signals[0].source == "fred_macro"
        assert signals[0].domain == "macro"
        assert signals[0].direction == "bearish"

    def test_convert_all_with_price_above_threshold(self):
        """PriceData with |change_pct| > 1.5% must produce a price signal."""
        import backfill_archives
        from mae_core.market.apis.price_fetcher import PriceData

        pd = PriceData(
            symbol="AAPL",
            price=185.0,
            timestamp="2026-01-15",
            source="yfinance_history",
            open=180.0,
            high=186.0,
            low=179.0,
            volume=50_000_000,
            change_pct=2.78,
        )

        signals = backfill_archives.convert_all(
            form4_trades=[], house_trades=[], senate_trades=[],
            contracts=[], efts_hits=[], price_history=[pd],
        )

        assert len(signals) == 1
        assert signals[0].source == "yfinance_price"
        assert signals[0].domain == "price"
        assert signals[0].direction == "bullish"
        assert signals[0].strength == pytest.approx(2.78 / 5.0, abs=0.01)

    def test_convert_all_with_price_below_threshold(self):
        """PriceData with |change_pct| < 1.5% must be filtered out (no signal)."""
        import backfill_archives
        from mae_core.market.apis.price_fetcher import PriceData

        pd = PriceData(
            symbol="AAPL",
            price=180.5,
            timestamp="2026-01-15",
            source="yfinance_history",
            open=180.0,
            change_pct=0.28,
        )

        signals = backfill_archives.convert_all(
            form4_trades=[], house_trades=[], senate_trades=[],
            contracts=[], efts_hits=[], price_history=[pd],
        )

        assert len(signals) == 0, "Sub-threshold price moves should be filtered out"

    def test_fetch_finra_handles_import_error(self):
        """If FINRA client import fails, fetch_finra_short must return []."""
        import backfill_archives

        with patch.dict("sys.modules", {"mae_core.market.apis.finra_short_interest": None}):
            # Force reimport to trigger ImportError
            import importlib
            importlib.reload(backfill_archives)
            result = backfill_archives.fetch_finra_short(["AAPL"], 30)

        assert result == []
        # Reload normally to not break other tests
        importlib.reload(backfill_archives)

    def test_fetch_macro_skips_without_key(self):
        """If FRED_API_KEY is not set, fetch_macro_history must return []."""
        import backfill_archives

        with patch.dict("os.environ", {}, clear=True):
            result = backfill_archives.fetch_macro_history(30)

        assert result == []
