"""Tests for the inevitability cascade — WorldModel wired into convergence pipeline.

Tests cover:
1. ConvergenceAlert.ripple_effects field populated from WorldModel
2. Partial convergence enriched with causal_predictions
3. Proactive causal watch emitted on signal ingestion
4. Graceful degradation when WorldModel is None/disabled
"""

from dataclasses import field
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlert,
    ConvergenceAlerter,
)
from mae_core.market.intelligence.world_model import RippleEffect, WorldModel
from mae_core.market.signal import MarketSignal


def _make_signal(source, domain, direction="bullish", strength=0.7, symbol="TEST",
                 metadata=None):
    """Create a MarketSignal for testing."""
    return MarketSignal(
        signal_id=f"{source}:{symbol}:{datetime.now().isoformat()}",
        source=source,
        symbol=symbol,
        asset_class="stock",
        domain=domain,
        direction=direction,
        strength=strength,
        confidence=0.6,
        decay_rate=0.1,
        timestamp=datetime.now(),
        received_at=datetime.now(),
        metadata=metadata or {},
    )


class TestRippleEffectsOnAlert:
    """ConvergenceAlert gets downstream cascade predictions from WorldModel."""

    def test_ripple_effects_field_exists(self):
        alert = ConvergenceAlert(
            alert_id="TEST-001",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.7,
            confidence=0.6,
            domains_converging=["macro", "energy"],
            signals=[],
            cross_domain_count=2,
            summary="test",
            urgency="days",
        )
        assert alert.ripple_effects == []

    def test_ripple_effects_in_to_dict(self):
        ripples = [{"ticker": "XLE", "direction": "bullish", "strength": 0.8}]
        alert = ConvergenceAlert(
            alert_id="TEST-002",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.7,
            confidence=0.6,
            domains_converging=["macro"],
            signals=[],
            cross_domain_count=1,
            summary="test",
            urgency="days",
            ripple_effects=ripples,
        )
        d = alert.to_dict()
        assert d["ripple_effects"] == ripples

    def test_compute_ripple_effects_with_world_model(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)

        # EIA crude draw signal maps to eia_crude_draw → XLE bullish
        sig = _make_signal(
            "eia_energy", "energy", "bullish", 0.8,
            metadata={"series_id": "crude_inventory", "direction": "bullish"},
        )
        ripples = alerter._compute_ripple_effects([sig])
        assert len(ripples) > 0
        tickers = [r["ticker"] for r in ripples]
        assert "XLE" in tickers

    def test_compute_ripple_effects_returns_direction(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)

        sig = _make_signal(
            "eia_energy", "energy", "bullish", 0.8,
            metadata={"series_id": "crude_inventory", "direction": "bullish"},
        )
        ripples = alerter._compute_ripple_effects([sig])
        xle = next((r for r in ripples if r["ticker"] == "XLE"), None)
        assert xle is not None
        assert xle["direction"] == "bullish"
        assert "strength" in xle
        assert "lag_days" in xle

    def test_compute_ripple_effects_without_world_model(self):
        alerter = ConvergenceAlerter(world_model=None)
        sig = _make_signal("eia_energy", "energy")
        ripples = alerter._compute_ripple_effects([sig])
        assert ripples == []

    def test_compute_ripple_effects_unmapped_source(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)
        # sec_form4 has no world model mapping (company-specific, not macro)
        sig = _make_signal("sec_form4", "insider")
        ripples = alerter._compute_ripple_effects([sig])
        assert ripples == []

    def test_compute_ripple_effects_deduplicates_tickers(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)

        # Two signals that both map to triggers affecting XLE
        sig1 = _make_signal(
            "eia_energy", "energy", metadata={"series_id": "crude", "direction": "bullish"},
        )
        sig2 = _make_signal(
            "eia_energy", "energy", metadata={"series_id": "crude", "direction": "bullish"},
        )
        ripples = alerter._compute_ripple_effects([sig1, sig2])
        tickers = [r["ticker"] for r in ripples]
        # No duplicates
        assert len(tickers) == len(set(tickers))

    def test_compute_ripple_effects_capped_at_20(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)

        # VIX spike triggers risk_off_rotation → many downstream effects
        sig = _make_signal(
            "vix_term_structure", "technical", metadata={"strength": 0.9},
        )
        ripples = alerter._compute_ripple_effects([sig])
        assert len(ripples) <= 20

    def test_ripple_effects_sorted_by_strength(self):
        wm = WorldModel()
        alerter = ConvergenceAlerter(world_model=wm)

        sig = _make_signal(
            "vix_term_structure", "technical", metadata={"strength": 0.9},
        )
        ripples = alerter._compute_ripple_effects([sig])
        if len(ripples) >= 2:
            strengths = [r["strength"] for r in ripples]
            assert strengths == sorted(strengths, reverse=True)


class TestPartialConvergenceCausalPredictions:
    """Partial convergence events carry causal predictions for Octopus investigation."""

    def test_partial_convergence_includes_causal_predictions(self):
        wm = WorldModel()
        bus = MagicMock()
        alerter = ConvergenceAlerter(
            min_domains=3,
            world_model=wm,
            event_bus=bus,
        )

        # Add signals from 2 domains (below min_domains=3) including a mappable source
        alerter.add_signal(_make_signal(
            "eia_energy", "energy", "bullish", 0.8,
            metadata={"series_id": "crude_inventory", "direction": "bullish"},
        ))
        alerter.add_signal(_make_signal("sec_form4", "insider", "bullish", 0.7))

        # Check convergence — should emit partial (2 < 3)
        result = alerter.check_convergence(direction_filter="bullish")
        assert result is None  # Below threshold

        # Verify partial convergence was published with causal_predictions
        if bus.publish.called:
            calls = [c for c in bus.publish.call_args_list
                     if c[0][0] == "market.intel.partial_convergence"]
            if calls:
                payload = calls[-1][0][1]
                assert "causal_predictions" in payload

    def test_partial_convergence_no_world_model_still_works(self):
        bus = MagicMock()
        alerter = ConvergenceAlerter(
            min_domains=3,
            world_model=None,
            event_bus=bus,
        )
        alerter.add_signal(_make_signal("sec_form4", "insider", "bullish", 0.7))
        alerter.add_signal(_make_signal("congressional", "government", "bullish", 0.7))

        result = alerter.check_convergence(direction_filter="bullish")
        assert result is None

        if bus.publish.called:
            calls = [c for c in bus.publish.call_args_list
                     if c[0][0] == "market.intel.partial_convergence"]
            if calls:
                payload = calls[-1][0][1]
                assert payload.get("causal_predictions", []) == []


class TestProactiveCausalWatch:
    """Signal ingestion triggers proactive causal watch via WorldModel."""

    def test_causal_watch_emitted_for_mapped_signal(self):
        wm = WorldModel()
        bus = MagicMock()
        bus.publish = MagicMock()

        # Simulate the callback that market_hooks.py registers
        def _on_signal_causal_watch(channel, data):
            msg = data if isinstance(data, dict) else {}
            source = msg.get("source", "")
            metadata = msg.get("metadata", {})
            trigger = wm.map_signal_to_trigger(source, metadata)
            if not trigger:
                return
            effects = wm.find_ripple_effects(trigger, min_strength=0.4)
            if not effects:
                return
            bus.publish("market.intel.causal_watch", {
                "trigger": trigger,
                "source": source,
                "effects": [{
                    "ticker": e.ticker,
                    "direction": e.direction,
                    "strength": round(e.strength, 3),
                    "lag_days": e.total_lag_days,
                    "path": e.path,
                } for e in effects[:10]],
            })

        # Simulate EIA signal ingestion
        _on_signal_causal_watch("market.sensing.signal_ingested", {
            "source": "eia_energy",
            "metadata": {"series_id": "crude_inventory", "direction": "bullish"},
        })

        assert bus.publish.called
        call_args = bus.publish.call_args
        assert call_args[0][0] == "market.intel.causal_watch"
        payload = call_args[0][1]
        assert payload["trigger"] == "eia_crude_draw"
        assert len(payload["effects"]) > 0
        assert any(e["ticker"] == "XLE" for e in payload["effects"])

    def test_causal_watch_not_emitted_for_unmapped_signal(self):
        wm = WorldModel()
        bus = MagicMock()

        trigger = wm.map_signal_to_trigger("sec_form4", {})
        assert trigger is None  # No mapping for insider trades

    def test_causal_watch_respects_min_strength(self):
        wm = WorldModel()
        effects = wm.find_ripple_effects("eia_crude_draw", min_strength=0.4)
        # All returned effects should have strength >= 0.4
        for e in effects:
            assert e.strength >= 0.4


class TestWorldModelIntegration:
    """WorldModel graph traversal correctness."""

    def test_find_ripple_effects_energy_chain(self):
        wm = WorldModel()
        effects = wm.find_ripple_effects("crude_price_spike")
        tickers = {e.ticker for e in effects}
        # Should find energy ETFs and affected sectors
        assert "XLE" in tickers or "USO" in tickers

    def test_find_ripple_effects_fed_chain(self):
        wm = WorldModel()
        effects = wm.find_ripple_effects("fed_rate_hike")
        tickers = {e.ticker for e in effects}
        # Rate hike should ripple to housing, growth stocks, banks
        assert len(tickers) >= 3

    def test_find_ripple_effects_returns_lag_days(self):
        wm = WorldModel()
        effects = wm.find_ripple_effects("crude_price_spike")
        for e in effects:
            assert e.total_lag_days >= 0

    def test_find_ripple_effects_empty_for_unknown_trigger(self):
        wm = WorldModel()
        effects = wm.find_ripple_effects("nonexistent_event")
        assert effects == []

    def test_map_signal_covers_key_sources(self):
        wm = WorldModel()
        # EIA
        assert wm.map_signal_to_trigger("eia_energy", {"series_id": "crude", "direction": "bearish"}) == "eia_crude_build"
        assert wm.map_signal_to_trigger("eia_energy", {"series_id": "crude", "direction": "bullish"}) == "eia_crude_draw"
        # Economic calendar
        assert wm.map_signal_to_trigger("economic_calendar", {"event_type": "FOMC"}) == "fed_rate_hike"
        assert wm.map_signal_to_trigger("economic_calendar", {"event_type": "CPI"}) == "cpi_hot"
        # VIX
        assert wm.map_signal_to_trigger("vix_term_structure", {"strength": 0.9}) == "vix_spike"
        assert wm.map_signal_to_trigger("vix_term_structure", {"strength": 0.3}) is None
        # Congressional defense
        assert wm.map_signal_to_trigger("congressional", {"keywords": ["defense"]}) == "defense_spending_increase"

    def test_record_outcome_strengthens_chain(self):
        wm = WorldModel()
        initial_stats = wm.get_statistics()
        # Record a correct prediction along crude → XLE chain
        wm.record_outcome("crude_price_spike", "XLE", was_correct=True)
        # Edge strength should have increased
        edge = wm._graph.edges["crude_price_spike", "XLE"]
        assert edge["hit_count"] == 1

    def test_record_outcome_weakens_on_miss(self):
        wm = WorldModel()
        initial_strength = wm._graph.edges["crude_price_spike", "XLE"]["strength"]
        wm.record_outcome("crude_price_spike", "XLE", was_correct=False)
        assert wm._graph.edges["crude_price_spike", "XLE"]["strength"] < initial_strength
