"""Tests for Pattern Archaeology: Excavator + DigSites + Fingerprint + Templates."""

import json
import pytest
from datetime import datetime
from pathlib import Path

from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PrecursorSignal,
    PatternTemplate,
    TemplateInstance,
    lag_bucket_for_days,
)
from mae_core.market.archaeology.dig_sites import DigSite, find_dig_sites
from mae_core.market.archaeology.excavator import Excavator


# ── Fingerprint tests ──────────────────────────────────────────────────


class TestPrecursorSignal:
    def test_roundtrip(self):
        sig = PrecursorSignal(
            source="sec_form4", domain="insider", direction="bullish",
            strength=0.8, lag_days=3, lag_bucket="short", signal_id="test-1",
        )
        d = sig.to_dict()
        restored = PrecursorSignal.from_dict(d)
        assert restored.source == "sec_form4"
        assert restored.lag_bucket == "short"
        assert restored.strength == 0.8


class TestLagBucket:
    def test_immediate(self):
        assert lag_bucket_for_days(0) == "immediate"
        assert lag_bucket_for_days(2) == "immediate"

    def test_short(self):
        assert lag_bucket_for_days(3) == "short"
        assert lag_bucket_for_days(5) == "short"

    def test_medium(self):
        assert lag_bucket_for_days(6) == "medium"
        assert lag_bucket_for_days(10) == "medium"

    def test_long(self):
        assert lag_bucket_for_days(11) == "long"
        assert lag_bucket_for_days(20) == "long"

    def test_extended(self):
        assert lag_bucket_for_days(21) == "extended"
        assert lag_bucket_for_days(30) == "extended"
        assert lag_bucket_for_days(50) == "extended"


class TestMoveFingerprint:
    def test_auto_id_generation(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-11-20", move_pct=23.5,
        )
        assert fp.fingerprint_id != ""
        assert len(fp.fingerprint_id) == 16

    def test_deterministic_id(self):
        fp1 = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-11-20", move_pct=23.5,
        )
        fp2 = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-11-20", move_pct=50.0,  # Different pct, same key
        )
        assert fp1.fingerprint_id == fp2.fingerprint_id

    def test_domain_signature(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="AAPL", direction="bullish",
            move_date="2024-01-15", move_pct=5.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bullish", 0.5, 3, "short"),
                PrecursorSignal("fred_macro", "macro", "bullish", 0.4, 7, "medium"),
                PrecursorSignal("ta_rsi", "technical", "bullish", 0.6, 1, "immediate"),
            ],
        )
        assert fp.domain_signature == "insider+macro+technical"

    def test_lag_profile(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="AAPL", direction="bullish",
            move_date="2024-01-15", move_pct=5.0,
            precursor_signals=[
                PrecursorSignal("a", "d1", "bullish", 0.5, 1, "immediate"),
                PrecursorSignal("b", "d2", "bullish", 0.5, 4, "short"),
                PrecursorSignal("c", "d3", "bullish", 0.5, 4, "short"),
            ],
        )
        assert fp.lag_profile["immediate"] == 1
        assert fp.lag_profile["short"] == 2

    def test_source_set(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="MSFT", direction="bearish",
            move_date="2024-06-01", move_pct=-7.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bearish", 0.5, 2, "immediate"),
                PrecursorSignal("fred_macro", "macro", "bearish", 0.4, 8, "medium"),
            ],
        )
        assert fp.source_set == {"sec_form4", "fred_macro"}

    def test_win_rate(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="X", direction="bullish",
            move_date="2024-01-01", move_pct=5.0,
            wins=3, losses=7,
        )
        assert fp.win_rate == pytest.approx(0.3)

    def test_win_rate_zero(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="X", direction="bullish",
            move_date="2024-01-01", move_pct=5.0,
        )
        assert fp.win_rate == 0.0

    def test_template_key(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_pct=10.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bullish", 0.5, 3, "short"),
                PrecursorSignal("fred_macro", "macro", "bullish", 0.5, 7, "medium"),
            ],
        )
        assert fp.template_key == "bullish:insider+macro"

    def test_roundtrip_json(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="TSLA", direction="bearish",
            move_date="2025-03-01", move_pct=-12.3,
            precursor_signals=[
                PrecursorSignal("src_a", "dom_a", "bearish", 0.7, 5, "short"),
            ],
            wins=5, losses=3,
        )
        json_str = fp.to_json()
        d = json.loads(json_str)
        restored = MoveFingerprint.from_dict(d)
        assert restored.symbol == "TSLA"
        assert restored.wins == 5
        assert restored.losses == 3
        assert len(restored.precursor_signals) == 1
        assert restored.precursor_signals[0].source == "src_a"


# ── PatternTemplate tests ─────────────────────────────────────────────


class TestPatternTemplate:
    def test_auto_id(self):
        t = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
        )
        assert t.template_id != ""
        assert len(t.template_id) == 16

    def test_deterministic_id(self):
        t1 = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
        )
        t2 = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
        )
        assert t1.template_id == t2.template_id

    def test_different_direction_different_id(self):
        t1 = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
        )
        t2 = PatternTemplate(
            template_id="", direction="bearish",
            domain_signature="insider+macro",
        )
        assert t1.template_id != t2.template_id

    def test_from_fingerprint(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_pct=10.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bullish", 0.8, 3, "short"),
                PrecursorSignal("fred_macro", "macro", "bullish", 0.5, 7, "medium"),
            ],
        )
        t = PatternTemplate.from_fingerprint(fp)
        assert t.direction == "bullish"
        assert t.domain_signature == "insider+macro"
        assert t.n_instances == 1
        assert "NVDA" in t.unique_symbols
        assert t.avg_move_pct == pytest.approx(10.0)

    def test_add_instance(self):
        fp1 = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_pct=10.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bullish", 0.8, 3, "short"),
            ],
        )
        t = PatternTemplate.from_fingerprint(fp1)
        assert t.n_instances == 1

        fp2 = MoveFingerprint(
            fingerprint_id="", symbol="AAPL", direction="bullish",
            move_date="2024-02-15", move_pct=8.0,
        )
        t.add_instance(fp2)
        assert t.n_instances == 2
        assert "AAPL" in t.unique_symbols
        assert t.avg_move_pct == pytest.approx(9.0)

    def test_cross_validated(self):
        fp1 = MoveFingerprint(fingerprint_id="", symbol="NVDA", direction="bullish", move_date="2024-01-15", move_pct=10.0)
        t = PatternTemplate.from_fingerprint(fp1)
        assert not t.cross_validated

        fp2 = MoveFingerprint(fingerprint_id="", symbol="AAPL", direction="bullish", move_date="2024-02-15", move_pct=8.0)
        t.add_instance(fp2)
        assert not t.cross_validated  # Only 2 symbols

        fp3 = MoveFingerprint(fingerprint_id="", symbol="MSFT", direction="bullish", move_date="2024-03-15", move_pct=6.0)
        t.add_instance(fp3)
        assert t.cross_validated  # 3 symbols!
        assert t.confidence_multiplier == 1.15

    def test_confidence_multiplier_scaling(self):
        t = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
        )
        assert t.confidence_multiplier == 1.0  # 0 symbols
        t.symbols_seen = ["A"]
        assert t.confidence_multiplier == 1.0
        t.symbols_seen = ["A", "B", "C"]
        assert t.confidence_multiplier == 1.15
        t.symbols_seen = ["A", "B", "C", "D", "E"]
        assert t.confidence_multiplier == 1.3
        t.symbols_seen = list("ABCDEFGHIJ")
        assert t.confidence_multiplier == 1.5

    def test_win_rate(self):
        t = PatternTemplate(
            template_id="", direction="bullish",
            domain_signature="insider+macro",
            wins=7, losses=3,
        )
        assert t.win_rate == pytest.approx(0.7)

    def test_roundtrip_json(self):
        fp = MoveFingerprint(
            fingerprint_id="", symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_pct=10.0,
            precursor_signals=[
                PrecursorSignal("sec_form4", "insider", "bullish", 0.8, 3, "short"),
            ],
        )
        t = PatternTemplate.from_fingerprint(fp)
        t.wins = 5
        t.losses = 3
        json_str = t.to_json()
        d = json.loads(json_str)
        restored = PatternTemplate.from_dict(d)
        assert restored.template_id == t.template_id
        assert restored.direction == "bullish"
        assert restored.domain_signature == t.domain_signature
        assert restored.wins == 5
        assert restored.losses == 3
        assert restored.n_instances == 1


class TestTemplateInstance:
    def test_roundtrip(self):
        inst = TemplateInstance(
            symbol="NVDA", move_date="2024-01-15",
            move_pct=10.0, fingerprint_id="abc123",
        )
        d = inst.to_dict()
        restored = TemplateInstance.from_dict(d)
        assert restored.symbol == "NVDA"
        assert restored.move_pct == 10.0
        assert restored.fingerprint_id == "abc123"


# ── DigSites tests ─────────────────────────────────────────────────────


class _FakePrice:
    def __init__(self, price, timestamp):
        self.price = price
        self.timestamp = timestamp


class TestFindDigSites:
    def _make_prices(self, prices_with_dates):
        return [_FakePrice(p, d) for p, d in prices_with_dates]

    def test_single_day_big_move(self):
        prices = self._make_prices([
            (100.0, "2024-01-01"), (100.0, "2024-01-02"), (106.0, "2024-01-03"),
            (106.0, "2024-01-04"), (106.0, "2024-01-05"),
        ])
        sites = find_dig_sites(prices, "TEST", min_single_day_pct=5.0)
        assert len(sites) == 1
        assert sites[0].direction == "bullish"
        assert sites[0].move_pct == pytest.approx(6.0, abs=0.1)

    def test_bearish_move(self):
        prices = self._make_prices([
            (100.0, "2024-01-01"), (100.0, "2024-01-02"), (93.0, "2024-01-03"),
        ])
        sites = find_dig_sites(prices, "TEST", min_single_day_pct=5.0)
        assert len(sites) == 1
        assert sites[0].direction == "bearish"

    def test_no_significant_moves(self):
        prices = self._make_prices([
            (100.0, "2024-01-01"), (101.0, "2024-01-02"),
            (100.5, "2024-01-03"), (101.5, "2024-01-04"),
        ])
        sites = find_dig_sites(prices, "TEST", min_single_day_pct=5.0)
        assert len(sites) == 0

    def test_min_gap_between_sites(self):
        prices = self._make_prices([
            (100.0, "2024-01-01"), (106.0, "2024-01-02"),
            (106.0, "2024-01-03"), (112.36, "2024-01-04"),  # Another 6% but too close
            (112.36, "2024-01-05"), (112.36, "2024-01-06"),
            (112.36, "2024-01-07"), (112.36, "2024-01-08"),
            (112.36, "2024-01-09"), (119.1, "2024-01-10"),  # Far enough
        ])
        sites = find_dig_sites(prices, "TEST", min_single_day_pct=5.0, min_gap_days=5)
        assert len(sites) == 2

    def test_empty_history(self):
        assert find_dig_sites([], "TEST") == []

    def test_single_price(self):
        prices = self._make_prices([(100.0, "2024-01-01")])
        assert find_dig_sites(prices, "TEST") == []


# ── Excavator tests ────────────────────────────────────────────────────


class TestExcavator:
    def test_excavate_with_signals(self, tmp_path):
        # Create a signal archive
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        # Signal 5 days before the move
        sig_file = signal_dir / "2024-01-10.jsonl"
        sig_file.write_text(json.dumps({
            "signal_id": "test-1", "source": "sec_form4", "symbol": "NVDA",
            "domain": "insider", "direction": "bullish", "strength": 0.8,
            "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
        }) + "\n" + json.dumps({
            "signal_id": "test-2", "source": "fred_macro", "symbol": "",
            "domain": "macro", "direction": "bullish", "strength": 0.5,
            "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
        }) + "\n")

        # Signal 2 days before the move
        sig_file2 = signal_dir / "2024-01-13.jsonl"
        sig_file2.write_text(json.dumps({
            "signal_id": "test-3", "source": "ta_rsi", "symbol": "NVDA",
            "domain": "technical", "direction": "bullish", "strength": 0.6,
            "timestamp": "2024-01-13T00:00:00", "received_at": "2024-01-13T00:00:00",
        }) + "\n")

        site = DigSite(
            symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_start="2024-01-15",
            move_pct=10.0,
        )

        excavator = Excavator(signal_dir=signal_dir, min_precursors=2)
        fp = excavator.excavate(site, price_history=[], lookback_days=10)

        assert fp is not None
        assert fp.symbol == "NVDA"
        assert fp.direction == "bullish"
        assert len(fp.precursor_signals) == 3  # 2 from day -5 + 1 from day -2
        assert fp.domain_signature == "insider+macro+technical"

    def test_excavate_too_few_signals(self, tmp_path):
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        # Only 1 signal — below min_precursors
        sig_file = signal_dir / "2024-01-10.jsonl"
        sig_file.write_text(json.dumps({
            "signal_id": "test-1", "source": "sec_form4", "symbol": "NVDA",
            "domain": "insider", "direction": "bullish", "strength": 0.8,
            "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
        }) + "\n")

        site = DigSite(
            symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_start="2024-01-15",
            move_pct=10.0,
        )

        excavator = Excavator(signal_dir=signal_dir, min_precursors=3)
        fp = excavator.excavate(site, price_history=[], lookback_days=10)
        assert fp is None

    def test_look_ahead_bias_filter(self, tmp_path):
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        # Signal received AFTER the move — should be excluded
        sig_file = signal_dir / "2024-01-14.jsonl"
        sig_file.write_text(json.dumps({
            "signal_id": "future-1", "source": "sec_form4", "symbol": "NVDA",
            "domain": "insider", "direction": "bullish", "strength": 0.8,
            "timestamp": "2024-01-14T00:00:00", "received_at": "2024-01-16T00:00:00",
        }) + "\n")

        site = DigSite(
            symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_start="2024-01-15",
            move_pct=10.0,
        )

        excavator = Excavator(signal_dir=signal_dir, min_precursors=1)
        fp = excavator.excavate(site, price_history=[], lookback_days=10)
        assert fp is None  # The only signal was future-biased

    def test_symbol_agnostic_signals_included(self, tmp_path):
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        sig_file = signal_dir / "2024-01-10.jsonl"
        lines = [
            json.dumps({
                "signal_id": "macro-1", "source": "fred_macro", "symbol": "",
                "domain": "macro", "direction": "bullish", "strength": 0.5,
                "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
            }),
            json.dumps({
                "signal_id": "event-1", "source": "economic_calendar", "symbol": "",
                "domain": "events", "direction": "bullish", "strength": 0.6,
                "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
            }),
            json.dumps({
                "signal_id": "insider-1", "source": "sec_form4", "symbol": "NVDA",
                "domain": "insider", "direction": "bullish", "strength": 0.8,
                "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
            }),
        ]
        sig_file.write_text("\n".join(lines) + "\n")

        site = DigSite(
            symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_start="2024-01-15",
            move_pct=10.0,
        )

        excavator = Excavator(signal_dir=signal_dir, min_precursors=2)
        fp = excavator.excavate(site, price_history=[], lookback_days=10)
        assert fp is not None
        assert len(fp.precursor_signals) == 3  # macro + events + insider

    def test_wrong_symbol_excluded(self, tmp_path):
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        sig_file = signal_dir / "2024-01-10.jsonl"
        sig_file.write_text(json.dumps({
            "signal_id": "wrong-1", "source": "sec_form4", "symbol": "AAPL",
            "domain": "insider", "direction": "bullish", "strength": 0.8,
            "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
        }) + "\n")

        site = DigSite(
            symbol="NVDA", direction="bullish",
            move_date="2024-01-15", move_start="2024-01-15",
            move_pct=10.0,
        )

        excavator = Excavator(signal_dir=signal_dir, min_precursors=1)
        fp = excavator.excavate(site, price_history=[], lookback_days=10)
        assert fp is None  # AAPL signal doesn't match NVDA dig site

    def test_batch_excavation(self, tmp_path):
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        sig_file = signal_dir / "2024-01-10.jsonl"
        lines = [
            json.dumps({
                "signal_id": f"sig-{i}", "source": "sec_form4", "symbol": "NVDA",
                "domain": "insider", "direction": "bullish", "strength": 0.5,
                "timestamp": "2024-01-10T00:00:00", "received_at": "2024-01-10T00:00:00",
            })
            for i in range(5)
        ]
        sig_file.write_text("\n".join(lines) + "\n")

        sites = [
            DigSite("NVDA", "bullish", "2024-01-15", "2024-01-15", 10.0),
            DigSite("AAPL", "bullish", "2024-01-15", "2024-01-15", 8.0),  # No matching signals
        ]

        excavator = Excavator(signal_dir=signal_dir, min_precursors=3)
        fps = excavator.excavate_batch(sites, [], lookback_days=10)
        assert len(fps) == 1  # Only NVDA has enough matching signals
        assert fps[0].symbol == "NVDA"

    def test_template_accumulation(self, tmp_path):
        """Verify excavator groups fingerprints into PatternTemplates."""
        signal_dir = tmp_path / "signals"
        signal_dir.mkdir()

        # Signals for two symbols with same domains
        for sym, dates in [("NVDA", "2024-01-10"), ("AAPL", "2024-02-10")]:
            sig_file = signal_dir / f"{dates}.jsonl"
            lines = [
                json.dumps({
                    "signal_id": f"{sym}-insider", "source": "sec_form4", "symbol": sym,
                    "domain": "insider", "direction": "bullish", "strength": 0.8,
                    "timestamp": f"{dates}T00:00:00", "received_at": f"{dates}T00:00:00",
                }),
                json.dumps({
                    "signal_id": f"{sym}-macro", "source": "fred_macro", "symbol": "",
                    "domain": "macro", "direction": "bullish", "strength": 0.5,
                    "timestamp": f"{dates}T00:00:00", "received_at": f"{dates}T00:00:00",
                }),
                json.dumps({
                    "signal_id": f"{sym}-ta", "source": "ta_rsi", "symbol": sym,
                    "domain": "technical", "direction": "bullish", "strength": 0.6,
                    "timestamp": f"{dates}T00:00:00", "received_at": f"{dates}T00:00:00",
                }),
            ]
            sig_file.write_text("\n".join(lines) + "\n")

        excavator = Excavator(signal_dir=signal_dir, min_precursors=2)

        site1 = DigSite("NVDA", "bullish", "2024-01-15", "2024-01-15", 10.0)
        fp1 = excavator.excavate(site1, price_history=[], lookback_days=10)
        assert fp1 is not None

        site2 = DigSite("AAPL", "bullish", "2024-02-15", "2024-02-15", 8.0)
        fp2 = excavator.excavate(site2, price_history=[], lookback_days=10)
        assert fp2 is not None

        # Same domain_signature → same template
        assert fp1.template_key == fp2.template_key

        # Template should have 2 instances from 2 symbols
        templates = excavator.templates
        assert len(templates) == 1
        key = fp1.template_key
        assert templates[key].n_instances == 2
        assert len(templates[key].unique_symbols) == 2
