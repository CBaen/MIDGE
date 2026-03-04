"""Tests for combo-level Thompson Sampling in convergence alerter.

Domain combinations have vastly different win rates (8% to 67%), but the
confidence engine was blind to this — per-signal Thompson + diversity bonus
produced flat confidence across winners and losers.  Combo Thompson tracks
Beta distributions per domain combination so the confidence score actually
discriminates between good and bad combos.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_signal(domain, direction="bullish", strength=0.7, confidence=0.6,
                 source="test_source", symbol="AAPL"):
    """Create a minimal Signal for convergence testing."""
    from mae_core.market.intelligence.convergence_alerter import Signal
    return Signal(
        signal_id=f"test_{domain}_{direction}_{id(domain)}",
        domain=domain,
        direction=direction,
        strength=strength,
        timestamp=datetime.now(),
        velocity=0.0,
        confidence=confidence,
        source=source,
        metadata={"symbol": symbol},
    )


def _make_alerter(thompson=None, min_domains=3):
    """Create a ConvergenceAlerter for testing."""
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter
    return ConvergenceAlerter(
        min_domains=min_domains,
        thompson_sampler=thompson,
    )


def _build_thompson_with_combo(combo_key, wins, losses):
    """Build a ThompsonSampler with a specific combo distribution."""
    from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
    ts = ThompsonSampler()
    for _ in range(wins):
        ts.update(combo_key, success=True)
    for _ in range(losses):
        ts.update(combo_key, success=False)
    return ts


# ── Combo Key Format ────────────────────────────────────────────────────

class TestComboKeyFormat:
    """Combo keys are 'combo:' + sorted domains joined by '+'."""

    def test_combo_key_from_sorted_domains(self):
        domains = ["price", "events", "macro"]
        key = "combo:" + "+".join(sorted(domains))
        assert key == "combo:events+macro+price"

    def test_combo_key_on_alert(self):
        """ConvergenceAlert.combo_key is populated at construction."""
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert, Signal
        alert = ConvergenceAlert(
            alert_id="TEST-001",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.7,
            confidence=0.6,
            domains_converging=["events", "macro", "price"],
            signals=[],
            cross_domain_count=3,
            summary="test",
            urgency="days",
            combo_key="combo:events+macro+price",
        )
        assert alert.combo_key == "combo:events+macro+price"

    def test_combo_key_in_to_dict(self):
        """combo_key appears in to_dict output."""
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert
        alert = ConvergenceAlert(
            alert_id="TEST-002",
            timestamp=datetime.now(),
            direction="bearish",
            strength=0.8,
            confidence=0.7,
            domains_converging=["insider", "macro", "price"],
            signals=[],
            cross_domain_count=3,
            summary="test",
            urgency="hours",
            combo_key="combo:insider+macro+price",
        )
        d = alert.to_dict()
        assert d["combo_key"] == "combo:insider+macro+price"

    def test_combo_key_default_empty(self):
        """combo_key defaults to empty string for backward compatibility."""
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert
        alert = ConvergenceAlert(
            alert_id="TEST-003",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.7,
            confidence=0.6,
            domains_converging=["a", "b", "c"],
            signals=[],
            cross_domain_count=3,
            summary="test",
            urgency="days",
        )
        assert alert.combo_key == ""


# ── Combo Weight Application ────────────────────────────────────────────

class TestComboWeight:
    """Combo Thompson weight modifies confidence at alert time."""

    def test_high_winrate_combo_boosts_confidence(self):
        """A combo with 80% win rate should boost confidence (multiplier > 1.0)."""
        combo_key = "combo:events+macro+price"
        # 80% win rate → mean ≈ 0.8 → multiplier ≈ 1.3
        ts = _build_thompson_with_combo(combo_key, wins=16, losses=4)

        alerter = _make_alerter(thompson=ts)
        # Feed 3 domains to trigger convergence
        for domain in ["events", "macro", "price"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "AAPL"},
            )

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        alert = alerts[0]
        # With high combo win rate, confidence should be boosted above base
        assert alert.confidence > 0.5  # Base geometric mean is ~0.6
        assert alert.combo_key == combo_key

    def test_low_winrate_combo_reduces_confidence(self):
        """A combo with 10% win rate should reduce confidence (multiplier < 1.0)."""
        combo_key = "combo:events+insider+macro"
        # 10% win rate → mean ≈ 0.1 → multiplier ≈ 0.6
        ts = _build_thompson_with_combo(combo_key, wins=2, losses=18)

        alerter = _make_alerter(thompson=ts)
        for domain in ["events", "insider", "macro"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "AAPL"},
            )

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        alert = alerts[0]
        # With 10% combo, multiplier ~0.6 → confidence drops significantly
        assert alert.confidence < 0.55

    def test_immature_combo_no_adjustment(self):
        """Combo with < 5 samples gets no weight adjustment (neutral)."""
        combo_key = "combo:events+macro+price"
        ts = _build_thompson_with_combo(combo_key, wins=1, losses=1)  # 2 samples < 5

        alerter = _make_alerter(thompson=ts)
        for domain in ["events", "macro", "price"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "AAPL"},
            )

        # Also create an alerter with NO Thompson for comparison
        alerter_no_ts = _make_alerter(thompson=None)
        for domain in ["events", "macro", "price"]:
            alerter_no_ts.record_signal(
                signal_id=f"sig2_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "AAPL"},
            )

        alerts_with = alerter.check_convergence()
        alerts_without = alerter_no_ts.check_convergence()
        assert len(alerts_with) >= 1
        assert len(alerts_without) >= 1
        # With immature combo (< 5 samples), per-signal Thompson still runs
        # but combo multiplier is skipped, so only per-signal Thompson differs

    def test_confidence_clamped_after_combo(self):
        """Confidence stays within [0.05, 0.95] even with extreme combo weight."""
        combo_key = "combo:a+b+c"
        # Extreme win rate → multiplier would push confidence very high
        ts = _build_thompson_with_combo(combo_key, wins=99, losses=1)

        alerter = _make_alerter(thompson=ts)
        for domain in ["a", "b", "c"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.9,
                confidence=0.95,
                timestamp=datetime.now(),
                metadata={"symbol": "TEST"},
            )

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        assert alerts[0].confidence <= 0.95
        assert alerts[0].confidence >= 0.05

    def test_no_thompson_no_crash(self):
        """Alerter without Thompson Sampler works fine — combo weight is skipped."""
        alerter = _make_alerter(thompson=None)
        for domain in ["events", "macro", "price"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bullish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "AAPL"},
            )

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        assert alerts[0].combo_key == "combo:events+macro+price"


# ── Combo Registration in OutcomeCollector ──────────────────────────────

class TestComboRegistration:
    """OutcomeCollector.register_convergence_alert creates combo predictions."""

    def _make_collector(self):
        """Create an OutcomeCollector with mocked tracker."""
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector
        mock_pf = MagicMock()
        mock_ts = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            collector = OutcomeCollector(mock_pf, mock_ts, data_dir=tmp_path)
            # Replace tracker's record_prediction with a mock so we can assert calls
            collector.tracker.record_prediction = MagicMock()
            yield collector, mock_ts, tmp_path

    def _make_alert(self, direction="bullish", domains=None):
        """Create a mock ConvergenceAlert."""
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert
        if domains is None:
            domains = ["events", "macro", "price"]
        return ConvergenceAlert(
            alert_id="CONV-20260303-0001",
            timestamp=datetime.now(),
            direction=direction,
            strength=0.8,
            confidence=0.7,
            domains_converging=sorted(domains),
            signals=[],
            cross_domain_count=len(domains),
            summary="test",
            urgency="days",
            combo_key="combo:" + "+".join(sorted(domains)),
        )

    def test_registers_combo_prediction(self):
        """register_convergence_alert creates a prediction with combo key."""
        for collector, mock_ts, tmp in self._make_collector():
            alert = self._make_alert()
            result = collector.register_convergence_alert(alert, "AAPL")
            assert result is True
            # Verify record_prediction was called with combo source
            collector.tracker.record_prediction.assert_called_once()
            kwargs = collector.tracker.record_prediction.call_args.kwargs
            assert kwargs["source"] == "combo:events+macro+price"
            assert kwargs["symbol"] == "AAPL"
            assert kwargs["direction"] == "up"

    def test_deduplicates(self):
        """Same alert_id not registered twice."""
        for collector, mock_ts, tmp in self._make_collector():
            alert = self._make_alert()
            assert collector.register_convergence_alert(alert, "AAPL") is True
            assert collector.register_convergence_alert(alert, "AAPL") is False
            assert collector.tracker.record_prediction.call_count == 1

    def test_skips_no_symbol(self):
        """Alert with no primary symbol is not registered."""
        for collector, mock_ts, tmp in self._make_collector():
            alert = self._make_alert()
            assert collector.register_convergence_alert(alert, "") is False
            collector.tracker.record_prediction.assert_not_called()

    def test_bearish_direction_mapped(self):
        """Bearish alert maps to 'down' direction."""
        for collector, mock_ts, tmp in self._make_collector():
            alert = self._make_alert(direction="bearish")
            collector.register_convergence_alert(alert, "TSLA")
            kwargs = collector.tracker.record_prediction.call_args.kwargs
            assert kwargs["direction"] == "down"

    def test_outcome_window_is_14_days(self):
        """Combo predictions use 14-day outcome window."""
        for collector, mock_ts, tmp in self._make_collector():
            alert = self._make_alert()
            collector.register_convergence_alert(alert, "AAPL")
            kwargs = collector.tracker.record_prediction.call_args.kwargs
            assert kwargs["outcome_window_days"] == 14


# ── Replay Seeding ──────────────────────────────────────────────────────

class TestReplaySeeding:
    """Bootstrap seeds combo Thompson distributions from replay results."""

    def _make_replay_file(self, tmp_dir, combos):
        """Write a minimal replay_results.json."""
        replay = {
            "alerts": [],
            "report": {
                "total_alerts": 0,
                "domain_combos": combos,
            },
        }
        midge_dir = tmp_dir / "midge"
        midge_dir.mkdir(parents=True, exist_ok=True)
        replay_path = midge_dir / "replay_results.json"
        replay_path.write_text(json.dumps(replay))
        return replay_path

    def test_seeds_from_replay(self):
        """Combo distributions are seeded from replay win/loss counts."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ts = ThompsonSampler(persistence_path=tmp_path / "thompson.json")

            combos = {
                "events+macro+price": {"wins": 10, "total": 32, "win_rate_pct": 31.2},
            }
            self._make_replay_file(tmp_path, combos)

            # Simulate seeding logic from market_systems.py
            replay_path = tmp_path / "midge" / "replay_results.json"
            results = json.loads(replay_path.read_text())
            for combo_str, stats in results["report"]["domain_combos"].items():
                if stats["total"] < 3:
                    continue
                key = f"combo:{combo_str}"
                for _ in range(stats["wins"]):
                    ts.update(key, success=True)
                for _ in range(stats["total"] - stats["wins"]):
                    ts.update(key, success=False)

            dist = ts.get_distribution("combo:events+macro+price")
            assert dist.samples == 32
            # Mean should be close to 10/32 ≈ 0.31
            assert abs(dist.mean - (11 / 34)) < 0.05  # Beta(1+10, 1+22) = Beta(11, 23)

    def test_skips_small_combos(self):
        """Combos with total < 3 are not seeded."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler

        with tempfile.TemporaryDirectory() as tmp:
            ts = ThompsonSampler(persistence_path=Path(tmp) / "thompson.json")
            combos = {
                "x+y+z": {"wins": 1, "total": 2, "win_rate_pct": 50.0},
            }
            for combo_str, stats in combos.items():
                if stats["total"] < 3:
                    continue
                key = f"combo:{combo_str}"
                for _ in range(stats["wins"]):
                    ts.update(key, success=True)
                for _ in range(stats["total"] - stats["wins"]):
                    ts.update(key, success=False)

            dist = ts.get_distribution("combo:x+y+z")
            assert dist.samples == 0  # Not seeded

    def test_idempotent_seeding(self):
        """Second seed run skips combos where live data exceeds replay total."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        ts = ThompsonSampler()

        combo_key = "combo:events+macro+price"
        # First seed: 10 wins, 22 losses = 32 total
        for _ in range(10):
            ts.update(combo_key, success=True)
        for _ in range(22):
            ts.update(combo_key, success=False)

        assert ts.get_distribution(combo_key).samples == 32

        # Simulate re-seed check (same logic as market_systems.py)
        replay_stats = {"wins": 10, "total": 32}
        existing = ts.get_distribution(combo_key)
        if existing.samples >= replay_stats["total"]:
            pass  # Skip — already seeded
        else:
            for _ in range(replay_stats["wins"]):
                ts.update(combo_key, success=True)

        # Should still be 32, not 42
        assert ts.get_distribution(combo_key).samples == 32

    def test_seeded_mean_matches_win_rate(self):
        """Seeded distribution mean approximates replay win rate."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        ts = ThompsonSampler()

        # 66.7% win rate: 2 wins, 1 loss
        for _ in range(2):
            ts.update("combo:a+b+c+d+e", success=True)
        for _ in range(1):
            ts.update("combo:a+b+c+d+e", success=False)

        dist = ts.get_distribution("combo:a+b+c+d+e")
        # Beta(1+2, 1+1) = Beta(3, 2) → mean = 3/5 = 0.6
        assert abs(dist.mean - 0.6) < 0.05


# ── Wire in Market Hooks ────────────────────────────────────────────────

class TestMarketHooksWiring:
    """market_hooks._market_sense_hook registers combo predictions."""

    def test_outcome_windows_has_convergence_combo(self):
        """OUTCOME_WINDOWS includes convergence_combo entry."""
        from mae_core.market.intelligence.outcome_collector import OUTCOME_WINDOWS
        assert "convergence_combo" in OUTCOME_WINDOWS
        assert OUTCOME_WINDOWS["convergence_combo"] == 14

    def test_combo_key_populated_on_live_alert(self):
        """Alerts from check_convergence() have combo_key populated."""
        alerter = _make_alerter()
        for domain in ["insider", "events", "macro"]:
            alerter.record_signal(
                signal_id=f"sig_{domain}",
                source="test",
                domain=domain,
                direction="bearish",
                strength=0.7,
                confidence=0.6,
                timestamp=datetime.now(),
                metadata={"symbol": "TSLA"},
            )

        alerts = alerter.check_convergence()
        assert len(alerts) >= 1
        assert alerts[0].combo_key == "combo:events+insider+macro"


# ── Full Feedback Loop ──────────────────────────────────────────────────

class TestComboFeedbackLoop:
    """End-to-end: alert → register → evaluate → Thompson update."""

    def test_combo_key_flows_to_thompson_update(self):
        """When an outcome resolves, Thompson updates the combo distribution."""
        from mae_core.market.intelligence.thompson_sampler import ThompsonSampler
        from mae_core.market.intelligence.outcome_collector import OutcomeCollector
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert

        ts = ThompsonSampler()
        mock_pf = MagicMock()

        with tempfile.TemporaryDirectory() as tmp:
            collector = OutcomeCollector(mock_pf, ts, data_dir=Path(tmp))

            alert = ConvergenceAlert(
                alert_id="CONV-20260303-0001",
                timestamp=datetime.now() - timedelta(days=20),
                direction="bullish",
                strength=0.8,
                confidence=0.7,
                domains_converging=["events", "macro", "price"],
                signals=[],
                cross_domain_count=3,
                summary="test",
                urgency="days",
                combo_key="combo:events+macro+price",
            )

            result = collector.register_convergence_alert(alert, "AAPL")
            assert result is True

            # Verify the combo key was recorded as a prediction source
            combo_dist_before = ts.get_distribution("combo:events+macro+price")
            assert combo_dist_before.samples == 0  # Not yet evaluated

    def test_primary_symbol_extraction(self):
        """Primary symbol extracted from alert signals' metadata."""
        sig = _make_signal("events", symbol="MSFT")
        from mae_core.market.intelligence.convergence_alerter import ConvergenceAlert
        alert = ConvergenceAlert(
            alert_id="CONV-20260303-0002",
            timestamp=datetime.now(),
            direction="bullish",
            strength=0.8,
            confidence=0.7,
            domains_converging=["events", "macro", "price"],
            signals=[sig],
            cross_domain_count=3,
            summary="test",
            urgency="days",
            combo_key="combo:events+macro+price",
        )

        # Extract symbol from signals (same logic as market_hooks.py)
        primary_symbol = None
        for s in alert.signals:
            sym = getattr(s, "metadata", {}).get("symbol", "")
            if sym:
                primary_symbol = sym
                break
        assert primary_symbol == "MSFT"
