"""Tests for enhanced COT signal: week-over-week change and COT Index.

Covers:
- COTSignal dataclass new fields (change_commercial_net, cot_index)
- COTClient._compute_derived_metrics() logic
- RawStore.get_cot_history() query
- from_cot_positioning() signal adapter metadata + strength modifiers
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict

from mae_core.market.apis.cot_client import COTSignal, COTClient, TICKER_TO_CFTC
from mae_core.market.signal_adapters.layer6 import from_cot_positioning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    commercial_net: int = 50_000,
    open_interest: int = 500_000,
    pct_commercial_long: float = 0.60,
    change_commercial_net=None,
    cot_index=None,
) -> COTSignal:
    half = abs(commercial_net) // 2
    comm_long = 250_000 + half if commercial_net >= 0 else 250_000 - half
    comm_short = comm_long - commercial_net
    return COTSignal(
        ticker="ES=F",
        contract_name="E-MINI S&P 500",
        commercial_long=comm_long,
        commercial_short=comm_short,
        commercial_net=commercial_net,
        noncommercial_long=100_000,
        noncommercial_short=80_000,
        noncommercial_net=20_000,
        small_trader_net=10_000,
        open_interest=open_interest,
        pct_commercial_long=pct_commercial_long,
        report_date="2026-03-04",
        change_commercial_net=change_commercial_net,
        cot_index=cot_index,
    )


def _make_raw_store(history: list) -> MagicMock:
    store = MagicMock()
    store.get_cot_history.return_value = history
    return store


def _make_history_rows(nets: list[int], base_long=250_000) -> list[dict]:
    """Build raw_store history rows from a list of net positions."""
    rows = []
    for i, net in enumerate(nets):
        rows.append({
            "report_date": f"2025-{i+1:02d}-01",
            "commercial_long": base_long + net // 2,
            "commercial_short": base_long - net // 2,
            "noncommercial_long": 100_000,
            "noncommercial_short": 80_000,
            "open_interest": 500_000,
        })
    return rows


# ---------------------------------------------------------------------------
# COTSignal dataclass
# ---------------------------------------------------------------------------

class TestCOTSignalDataclass:
    def test_default_fields_are_none(self):
        sig = _make_signal()
        assert sig.change_commercial_net is None
        assert sig.cot_index is None

    def test_fields_populated(self):
        sig = _make_signal(change_commercial_net=5_000, cot_index=0.80)
        assert sig.change_commercial_net == 5_000
        assert sig.cot_index == 0.80

    def test_to_plain_language_no_history(self):
        sig = _make_signal()
        text = sig.to_plain_language()
        assert "ES=F" in text
        assert "WoW" not in text
        assert "COT Index" not in text

    def test_to_plain_language_with_history(self):
        sig = _make_signal(change_commercial_net=10_000, cot_index=0.90)
        text = sig.to_plain_language()
        assert "WoW change" in text
        assert "+10,000" in text
        assert "COT Index" in text
        assert "90%" in text

    def test_to_plain_language_negative_change(self):
        sig = _make_signal(change_commercial_net=-3_000)
        text = sig.to_plain_language()
        assert "-3,000" in text
        assert "WoW change" in text

    def test_dataclass_serialisable(self):
        sig = _make_signal(change_commercial_net=1_000, cot_index=0.50)
        d = asdict(sig)
        assert d["change_commercial_net"] == 1_000
        assert d["cot_index"] == 0.50


# ---------------------------------------------------------------------------
# COTClient._compute_derived_metrics()
# ---------------------------------------------------------------------------

class TestComputeDerivedMetrics:
    def _client_with_store(self, history):
        client = COTClient(raw_store=_make_raw_store(history))
        return client

    def test_no_raw_store_returns_none(self):
        client = COTClient()
        change, idx = client._compute_derived_metrics("GOLD", 50_000)
        assert change is None
        assert idx is None

    def test_empty_history_returns_none(self):
        client = self._client_with_store([])
        change, idx = client._compute_derived_metrics("GOLD", 50_000)
        assert change is None
        assert idx is None

    def test_single_history_row_gives_change(self):
        history = _make_history_rows([40_000])  # previous week net = 40k
        client = self._client_with_store(history)
        change, idx = client._compute_derived_metrics("GOLD", 50_000)
        # change = current - last stored = 50k - 40k = 10k
        assert change == 10_000

    def test_single_history_row_gives_index_midpoint(self):
        # Only one stored week + current week have same value → flat → 0.5
        history = _make_history_rows([50_000])
        client = self._client_with_store(history)
        _, idx = client._compute_derived_metrics("GOLD", 50_000)
        assert idx == 0.5  # identical range → midpoint

    def test_cot_index_at_top(self):
        # current_net is maximum of all nets → index = 1.0
        history = _make_history_rows([10_000, 20_000, 30_000])
        client = self._client_with_store(history)
        _, idx = client._compute_derived_metrics("GOLD", 40_000)
        assert idx == 1.0

    def test_cot_index_at_bottom(self):
        # current_net is minimum → index = 0.0
        history = _make_history_rows([20_000, 30_000, 40_000])
        client = self._client_with_store(history)
        _, idx = client._compute_derived_metrics("GOLD", 10_000)
        assert idx == 0.0

    def test_cot_index_mid_range(self):
        # nets: 0, 50k, 100k → current 50k → (50-0)/(100-0) = 0.5
        history = _make_history_rows([0, 100_000])
        client = self._client_with_store(history)
        _, idx = client._compute_derived_metrics("GOLD", 50_000)
        assert idx == pytest.approx(0.5, abs=0.001)

    def test_wow_change_is_signed(self):
        # Previous net 60k, current 50k → change = -10k
        history = _make_history_rows([60_000])
        client = self._client_with_store(history)
        change, _ = client._compute_derived_metrics("GOLD", 50_000)
        assert change == -10_000

    def test_raw_store_failure_returns_none(self):
        store = MagicMock()
        store.get_cot_history.side_effect = RuntimeError("db error")
        client = COTClient(raw_store=store)
        change, idx = client._compute_derived_metrics("GOLD", 50_000)
        assert change is None
        assert idx is None

    def test_history_oldest_first_ordering(self):
        # history must be oldest-first (store returns oldest-first);
        # last element = most recent previous week
        history = _make_history_rows([10_000, 30_000, 50_000])  # ascending
        client = self._client_with_store(history)
        change, _ = client._compute_derived_metrics("GOLD", 60_000)
        # last stored net = 50k; change = 60k - 50k = 10k
        assert change == 10_000


# ---------------------------------------------------------------------------
# RawStore.get_cot_history()
# ---------------------------------------------------------------------------

class TestGetCOTHistory:
    @pytest.fixture
    def store(self, tmp_path):
        from mae_core.market.raw_store import RawStore
        s = RawStore(base_dir=tmp_path)
        yield s
        s.close()

    def _seed_history(self, store, rows: list[tuple]):
        """Seed cot_weekly table directly via SQL."""
        conn = store._get_conn("cot")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cot_weekly (
                report_date TEXT,
                contract_name TEXT,
                commercial_long INTEGER,
                commercial_short INTEGER,
                noncommercial_long INTEGER,
                noncommercial_short INTEGER,
                open_interest INTEGER,
                ingested_at TEXT,
                PRIMARY KEY (report_date, contract_name)
            )
        """)
        conn.executemany(
            "INSERT OR REPLACE INTO cot_weekly VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()

    def test_returns_empty_when_table_missing(self, store):
        result = store.get_cot_history("GOLD", weeks=10)
        assert result == []

    def test_returns_rows_for_matching_contract(self, store):
        self._seed_history(store, [
            ("2026-01-07", "GOLD", 50_000, 40_000, 100_000, 80_000, 300_000, "now"),
            ("2026-01-14", "GOLD", 55_000, 42_000, 105_000, 82_000, 310_000, "now"),
            ("2026-01-21", "GOLD", 60_000, 44_000, 110_000, 84_000, 320_000, "now"),
        ])
        rows = store.get_cot_history("GOLD", weeks=10)
        assert len(rows) == 3
        # Oldest-first ordering
        assert rows[0]["report_date"] == "2026-01-07"
        assert rows[-1]["report_date"] == "2026-01-21"

    def test_commercial_net_derivable(self, store):
        self._seed_history(store, [
            ("2026-01-07", "GOLD", 50_000, 40_000, 100_000, 80_000, 300_000, "now"),
        ])
        rows = store.get_cot_history("GOLD", weeks=10)
        net = rows[0]["commercial_long"] - rows[0]["commercial_short"]
        assert net == 10_000

    def test_weeks_limit_respected(self, store):
        rows_data = [
            (f"2025-{w:02d}-01", "GOLD", 50_000, 40_000, 100_000, 80_000, 300_000, "now")
            for w in range(1, 53)  # 52 rows
        ]
        self._seed_history(store, rows_data)
        result = store.get_cot_history("GOLD", weeks=10)
        assert len(result) == 10

    def test_partial_match_on_contract_name(self, store):
        # Real CFTC names often have exchange suffix
        self._seed_history(store, [
            ("2026-01-07", "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE",
             100_000, 80_000, 200_000, 150_000, 500_000, "now"),
        ])
        rows = store.get_cot_history("E-MINI S&P 500", weeks=10)
        assert len(rows) == 1

    def test_does_not_return_other_contracts(self, store):
        self._seed_history(store, [
            ("2026-01-07", "GOLD", 50_000, 40_000, 100_000, 80_000, 300_000, "now"),
            ("2026-01-07", "SILVER", 30_000, 25_000, 60_000, 55_000, 200_000, "now"),
        ])
        rows = store.get_cot_history("GOLD", weeks=10)
        assert all(r["commercial_long"] in (50_000,) for r in rows)
        assert len(rows) == 1

    def test_dict_keys_present(self, store):
        self._seed_history(store, [
            ("2026-01-07", "GOLD", 50_000, 40_000, 100_000, 80_000, 300_000, "now"),
        ])
        rows = store.get_cot_history("GOLD", weeks=10)
        assert set(rows[0].keys()) == {
            "report_date", "commercial_long", "commercial_short",
            "noncommercial_long", "noncommercial_short", "open_interest",
        }


# ---------------------------------------------------------------------------
# Signal adapter: from_cot_positioning()
# ---------------------------------------------------------------------------

class TestFromCOTPositioningAdapter:
    def test_metadata_includes_new_fields(self):
        sig = _make_signal(change_commercial_net=5_000, cot_index=0.80)
        ms = from_cot_positioning(sig)
        assert "change_commercial_net" in ms.metadata
        assert "cot_index" in ms.metadata
        assert ms.metadata["change_commercial_net"] == 5_000
        assert ms.metadata["cot_index"] == 0.80

    def test_metadata_includes_none_when_missing(self):
        sig = _make_signal()
        ms = from_cot_positioning(sig)
        assert ms.metadata["change_commercial_net"] is None
        assert ms.metadata["cot_index"] is None

    def test_existing_metadata_preserved(self):
        sig = _make_signal(change_commercial_net=1_000, cot_index=0.50)
        ms = from_cot_positioning(sig)
        assert "contract_name" in ms.metadata
        assert "commercial_net" in ms.metadata
        assert "pct_commercial_long" in ms.metadata
        assert "noncommercial_net" in ms.metadata
        assert "open_interest" in ms.metadata

    def test_direction_bullish_without_history(self):
        sig = _make_signal(commercial_net=50_000, pct_commercial_long=0.60)
        ms = from_cot_positioning(sig)
        assert ms.direction == "bullish"

    def test_direction_bearish_without_history(self):
        sig = _make_signal(
            commercial_net=-50_000,
            open_interest=500_000,
            pct_commercial_long=0.40,
        )
        ms = from_cot_positioning(sig)
        assert ms.direction == "bearish"

    def test_direction_neutral(self):
        sig = _make_signal(commercial_net=5_000, pct_commercial_long=0.52)
        ms = from_cot_positioning(sig)
        assert ms.direction == "neutral"

    def test_strength_boosted_by_cot_index_extreme_bullish(self):
        # COT Index at 0.90 (extreme bullish extreme) should boost strength
        sig_base = _make_signal(commercial_net=50_000, pct_commercial_long=0.60)
        sig_extreme = _make_signal(
            commercial_net=50_000, pct_commercial_long=0.60, cot_index=0.90
        )
        ms_base = from_cot_positioning(sig_base)
        ms_extreme = from_cot_positioning(sig_extreme)
        assert ms_extreme.strength > ms_base.strength

    def test_strength_boosted_by_cot_index_extreme_bearish(self):
        sig_base = _make_signal(
            commercial_net=-50_000, open_interest=500_000, pct_commercial_long=0.40
        )
        sig_extreme = _make_signal(
            commercial_net=-50_000, open_interest=500_000,
            pct_commercial_long=0.40, cot_index=0.10,
        )
        ms_base = from_cot_positioning(sig_base)
        ms_extreme = from_cot_positioning(sig_extreme)
        assert ms_extreme.strength > ms_base.strength

    def test_strength_damped_by_contrarian_cot_index(self):
        # Net long but COT Index at extreme low → contrarian caution
        sig_base = _make_signal(
            commercial_net=50_000, pct_commercial_long=0.60, cot_index=None
        )
        sig_contrarian = _make_signal(
            commercial_net=50_000, pct_commercial_long=0.60, cot_index=0.10
        )
        ms_base = from_cot_positioning(sig_base)
        ms_contrarian = from_cot_positioning(sig_contrarian)
        assert ms_contrarian.strength < ms_base.strength

    def test_strength_boosted_by_positive_wow_bullish(self):
        # Large positive WoW change on a bullish signal boosts strength
        oi = 500_000
        sig_no_wow = _make_signal(commercial_net=50_000, open_interest=oi, pct_commercial_long=0.60)
        sig_wow = _make_signal(
            commercial_net=50_000, open_interest=oi,
            pct_commercial_long=0.60,
            change_commercial_net=50_000,  # 10% of OI positive change
        )
        ms_no_wow = from_cot_positioning(sig_no_wow)
        ms_wow = from_cot_positioning(sig_wow)
        assert ms_wow.strength > ms_no_wow.strength

    def test_strength_damped_by_reversal_wow(self):
        # Negative WoW change on bullish signal weakens it
        oi = 500_000
        sig_no_wow = _make_signal(commercial_net=50_000, open_interest=oi, pct_commercial_long=0.60)
        sig_reversal = _make_signal(
            commercial_net=50_000, open_interest=oi,
            pct_commercial_long=0.60,
            change_commercial_net=-50_000,  # large reversal
        )
        ms_no_wow = from_cot_positioning(sig_no_wow)
        ms_reversal = from_cot_positioning(sig_reversal)
        assert ms_reversal.strength < ms_no_wow.strength

    def test_strength_clamped_zero_to_one(self):
        # Even with large boosts, strength must stay in [0, 1]
        sig = _make_signal(
            commercial_net=500_000, open_interest=500_000, pct_commercial_long=0.90,
            change_commercial_net=500_000, cot_index=0.99,
        )
        ms = from_cot_positioning(sig)
        assert 0.0 <= ms.strength <= 1.0

    def test_source_and_domain_unchanged(self):
        sig = _make_signal(change_commercial_net=1_000, cot_index=0.50)
        ms = from_cot_positioning(sig)
        assert ms.source == "cot_positioning"
        assert ms.domain == "positioning"
        assert ms.asset_class == "futures"

    def test_signal_id_format(self):
        sig = _make_signal()
        ms = from_cot_positioning(sig)
        assert ms.signal_id.startswith("cot:ES=F:")
