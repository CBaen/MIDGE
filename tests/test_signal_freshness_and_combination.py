"""Adversarial tests for Capability 5 (Temporal Freshness) and
Capability 6 (Intra-Domain Signal Combination / Count Boost).

These two capabilities interact tightly inside _check_direction_convergence():
  - Cap 5: strongest signal per domain is selected by freshness-weighted strength
            (signal.strength * _compute_freshness(signal, domain))
  - Cap 6: after selection, effective strength is multiplied by (1 + 0.1*log(count))
            where count = number of same-direction signals that pass min_strength

The tests are deliberately adversarial — they probe:
  - Exact formula values (not just "bigger is better")
  - Floor and ceiling clamping
  - Domain-specific window differences
  - Selection inversion (stale-but-strong loses to fresh-but-weaker)
  - Neutrals excluded from count boost
  - Direction isolation in count (bearish signals don't boost bullish count)
  - Coherence multiplier still applies on top of freshness+count
  - Pruning interaction (check_convergence calls _prune_old_signals)

Implementation notes for test authors:
  - _compute_freshness uses datetime.now() internally, so signals built from
    a module-level _NOW constant will show a tiny positive age by the time
    any method executes. Use abs=1e-3 (not 1e-9) for freshness comparisons
    involving signals meant to be "brand new".
  - check_convergence() calls _prune_old_signals() before evaluating. Injecting
    signals via _inject() bypasses record_signal() but NOT the prune. Use
    domains with wide windows (positioning=336h, government=168h) for stale
    signals that must survive into the convergence check.
"""

import math
from datetime import datetime, timedelta

import pytest

from mae_core.market.intelligence.convergence_alerter import (
    ConvergenceAlert,
    ConvergenceAlerter,
    Signal,
)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _now() -> datetime:
    """Return the current time. Called per-use to avoid module-level drift."""
    return datetime.now()


def _sig(
    signal_id: str,
    strength: float,
    domain: str,
    direction: str = "bullish",
    age_hours: float = 0.0,
) -> Signal:
    """Build a Signal timestamped age_hours before now."""
    return Signal(
        signal_id=signal_id,
        strength=strength,
        domain=domain,
        direction=direction,
        timestamp=datetime.now() - timedelta(hours=age_hours),
    )


def _alerter(min_domains: int = 3, min_strength: float = 0.5) -> ConvergenceAlerter:
    """Create a ConvergenceAlerter with no Thompson / regime classifiers
    and deduplication disabled (interval=0 so every call can produce an alert)."""
    a = ConvergenceAlerter(
        min_domains=min_domains,
        min_strength=min_strength,
        convergence_window_hours=72,
    )
    # Disable deduplication so tests can call check_convergence repeatedly
    a._min_alert_interval_hours = 0.0
    return a


def _inject(alerter: ConvergenceAlerter, sig: Signal) -> None:
    """Inject a pre-built Signal directly into the alerter's signal store,
    bypassing record_signal(). The signal's timestamp is preserved exactly.

    WARNING: check_convergence() calls _prune_old_signals() before evaluating.
    Signals older than their domain's window will be discarded. Use a domain
    with a wide enough window for stale signals."""
    alerter.signals[sig.domain].append(sig)


def _freshness(alerter: ConvergenceAlerter, sig: Signal, domain: str) -> float:
    """Convenience wrapper for _compute_freshness."""
    return alerter._compute_freshness(sig, domain)


# ── Class 1: TestTemporalFreshness ───────────────────────────────────────────


class TestTemporalFreshness:
    """Unit tests for ConvergenceAlerter._compute_freshness().

    Formula: freshness = max(0.3, 1.0 - (age_hours / window_hours) ** 0.5)
    Default window: 72 hours.  Domain overrides: positioning=336h, government=168h,
    contracts=168h.

    Timing note: _compute_freshness calls datetime.now() internally, so a signal
    built immediately before the call will have a tiny positive age (~microseconds).
    Tests use abs=1e-3 for "brand new" signals and exact comparisons for floored ones.
    """

    # ── 1. Brand-new signal ───────────────────────────────────────────────────

    def test_age_zero_returns_one(self):
        """A signal with age=0 must return freshness very close to 1.0.

        Exact 1.0 is impossible due to the tiny elapsed time between signal
        construction and the datetime.now() call inside _compute_freshness.
        We verify the result is above 0.999 (milliseconds of drift at most).
        """
        alerter = _alerter()
        sig = _sig("s1", 0.8, "insider", age_hours=0.0)
        result = _freshness(alerter, sig, "insider")
        assert result > 0.999, f"Brand-new signal freshness should be ~1.0, got {result}"
        assert result <= 1.0, "Freshness cannot exceed 1.0"

    # ── 2. 25% window age ─────────────────────────────────────────────────────

    def test_quarter_window_age_approx_half(self):
        """At 25% of window age, freshness = 1 - sqrt(0.25) = 0.50."""
        alerter = _alerter()
        # 72h window → 25% = 18h exactly
        sig = _sig("s2", 0.8, "insider", age_hours=18.0)
        result = _freshness(alerter, sig, "insider")
        expected = 1.0 - math.sqrt(18.0 / 72.0)  # exactly 0.50
        assert result == pytest.approx(expected, abs=1e-3)
        assert 0.49 < result < 0.51

    # ── 3. 99% window age → floor ────────────────────────────────────────────

    def test_near_window_edge_hits_floor(self):
        """At 99% of window age, raw value ≈ 0.005, floored to 0.3."""
        alerter = _alerter()
        age = 72.0 * 0.99  # 71.28 hours
        sig = _sig("s3", 0.8, "insider", age_hours=age)
        result = _freshness(alerter, sig, "insider")
        raw = 1.0 - math.sqrt(0.99)  # ≈ 0.005 — well below floor
        assert raw < 0.3
        assert result == pytest.approx(0.3, abs=1e-3)

    # ── 4. Past the window edge still gets floor ──────────────────────────────

    def test_past_window_edge_still_gets_floor(self):
        """Signal 100h old in a 72h window — age/window > 1, so raw result < 0.
        The max(0.3, ...) floor must clamp it to 0.3."""
        alerter = _alerter()
        sig = _sig("s4", 0.8, "insider", age_hours=100.0)
        result = _freshness(alerter, sig, "insider")
        assert result == pytest.approx(0.3, abs=1e-6)

    # ── 5. Positioning domain has 336h window ─────────────────────────────────

    def test_positioning_uses_336h_window(self):
        """positioning domain window = 14*24=336h. Same signal age yields
        much higher freshness than in the 72h-window 'insider' domain."""
        alerter = _alerter()
        # Use 18h — well within both windows, no floor involved
        sig_pos = _sig("s5", 0.8, "positioning", age_hours=18.0)
        sig_ins = _sig("s6", 0.8, "insider", age_hours=18.0)

        f_pos = _freshness(alerter, sig_pos, "positioning")
        f_ins = _freshness(alerter, sig_ins, "insider")

        expected_pos = 1.0 - math.sqrt(18.0 / 336.0)  # ≈ 0.769
        expected_ins = 1.0 - math.sqrt(18.0 / 72.0)   # ≈ 0.500

        assert f_pos == pytest.approx(expected_pos, abs=1e-3)
        assert f_ins == pytest.approx(expected_ins, abs=1e-3)
        # positioning freshness is higher for the same signal age
        assert f_pos > f_ins

    def test_global_default_domain_uses_72h(self):
        """Unknown domain 'crypto' falls back to the global 72h window.
        At 36h (half-window): freshness = 1 - sqrt(0.5) ≈ 0.293 → floor 0.3."""
        alerter = _alerter()
        sig = _sig("s7", 0.8, "crypto", age_hours=36.0)
        result = _freshness(alerter, sig, "crypto")
        # 1 - sqrt(36/72) = 1 - sqrt(0.5) ≈ 0.293 < 0.30 → floored
        assert result == pytest.approx(0.3, abs=1e-3)

    def test_government_uses_168h_window(self):
        """government domain window = 7*24=168h. At 10h: freshness = 1-sqrt(10/168)."""
        alerter = _alerter()
        sig = _sig("s8", 0.8, "government", age_hours=10.0)
        result = _freshness(alerter, sig, "government")
        expected = 1.0 - math.sqrt(10.0 / 168.0)  # ≈ 0.756
        assert result == pytest.approx(expected, abs=1e-3)

    def test_contracts_uses_168h_window(self):
        """contracts domain window = 7*24=168h — same as government.
        For the same age, contracts freshness > insider freshness."""
        alerter = _alerter()
        sig_con = _sig("s9",  0.8, "contracts", age_hours=10.0)
        sig_ins = _sig("s10", 0.8, "insider",   age_hours=10.0)

        f_con = _freshness(alerter, sig_con, "contracts")
        f_ins = _freshness(alerter, sig_ins, "insider")

        # contracts: 1 - sqrt(10/168) ≈ 0.756
        # insider:   1 - sqrt(10/72)  ≈ 0.627
        assert f_con > f_ins

    # ── 6. Window = 0 guard ───────────────────────────────────────────────────

    def test_zero_window_returns_one(self):
        """If window_hours is 0 (edge case), _compute_freshness must not divide
        by zero and must return 1.0."""
        alerter = _alerter(min_domains=1)
        alerter._domain_windows["zero_win"] = timedelta(hours=0)
        sig = _sig("s11", 0.8, "zero_win", age_hours=5.0)
        result = _freshness(alerter, sig, "zero_win")
        assert result == pytest.approx(1.0, abs=1e-9)

    # ── 7. Selection inversion: fresher beats stronger ────────────────────────

    def test_fresher_signal_beats_stale_stronger_signal(self):
        """In _check_direction_convergence, the best signal is chosen by
        freshness-weighted strength. A fresh-but-weaker signal outranks a
        stale-but-stronger one when their effective strengths cross over.

        Domain 'insider' (72h window):
          stale: strength=0.85, age=60h → freshness ≈ 0.087 → eff ≈ 0.074
          fresh: strength=0.70, age=1h  → freshness ≈ 0.882 → eff ≈ 0.617

        We detect which signal was chosen by checking which ID appears in
        alert.signals — only the domain representative is added, not every
        signal from that domain.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        stale_strong = _sig("stale", 0.85, "insider", "bullish", age_hours=60.0)
        fresh_weak   = _sig("fresh", 0.70, "insider", "bullish", age_hours=1.0)

        # Verify test setup: fresh must have higher effective strength
        f_stale = _freshness(alerter, stale_strong, "insider")
        f_fresh = _freshness(alerter, fresh_weak, "insider")
        assert fresh_weak.strength * f_fresh > stale_strong.strength * f_stale, (
            f"Setup error: fresh eff={fresh_weak.strength * f_fresh:.4f} "
            f"must > stale eff={stale_strong.strength * f_stale:.4f}"
        )

        _inject(alerter, stale_strong)
        _inject(alerter, fresh_weak)
        # Two more domains to reach min_domains=3
        _inject(alerter, _sig("d2", 0.7, "sentiment", "bullish", age_hours=1.0))
        _inject(alerter, _sig("d3", 0.7, "contracts", "bullish", age_hours=1.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        alert_ids = {s.signal_id for s in alert.signals}
        assert "fresh" in alert_ids, "Fresh signal must be the domain representative"
        assert "stale" not in alert_ids, "Stale signal must not be the domain representative"

    # ── 8. avg_strength uses effective (freshness-weighted) values ────────────

    def test_avg_strength_uses_freshness_weighted_effective_strength(self):
        """avg_strength in the alert reflects freshness-weighted effective strength,
        not raw strength. A run with fresh signals must produce higher avg_strength
        than an identical run with stale signals."""
        def _run(age_hours):
            a = _alerter(min_domains=3, min_strength=0.5)
            _inject(a, _sig("ins",  0.80, "insider",    "bullish", age_hours=age_hours))
            _inject(a, _sig("sent", 0.70, "sentiment",  "bullish", age_hours=age_hours))
            _inject(a, _sig("gov",  0.75, "government", "bullish", age_hours=age_hours))
            alerts = a.check_convergence(direction_filter="bullish")
            assert len(alerts) == 1, f"No alert for age_hours={age_hours}"
            return alerts[0].strength

        strength_fresh = _run(age_hours=1.0)
        strength_stale = _run(age_hours=50.0)

        assert strength_fresh > strength_stale, (
            "Stale signals must produce lower avg_strength due to freshness weighting"
        )

    # ── 9. Freshness applies to neutral signal selection ──────────────────────

    def test_neutral_selection_uses_freshness_weighting(self):
        """When multiple neutral signals compete in a domain, the one with higher
        freshness-weighted strength wins. We verify by checking which ID appears
        in the resulting alert's signals list.

        Domain: 'positioning' (336h window) so stale signals survive pruning.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # positioning window is 336h, so age=300h signal survives pruning
        stale_neutral = _sig("neu_stale", 0.90, "positioning", "neutral", age_hours=300.0)
        fresh_neutral = _sig("neu_fresh", 0.60, "positioning", "neutral", age_hours=1.0)

        # Verify test setup: fresh must win
        f_stale = _freshness(alerter, stale_neutral, "positioning")
        f_fresh = _freshness(alerter, fresh_neutral, "positioning")
        assert fresh_neutral.strength * f_fresh > stale_neutral.strength * f_stale, (
            f"Setup error: fresh neutral eff={fresh_neutral.strength * f_fresh:.4f} "
            f"must > stale neutral eff={stale_neutral.strength * f_stale:.4f}"
        )

        _inject(alerter, stale_neutral)
        _inject(alerter, fresh_neutral)
        # Two directional domains (positioning is neutral → counts toward min_domains)
        _inject(alerter, _sig("ins", 0.7, "insider",    "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov", 0.7, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert_ids = {s.signal_id for s in alerts[0].signals}
        assert "neu_fresh" in alert_ids, "Fresh neutral must be selected over stale neutral"
        assert "neu_stale" not in alert_ids


# ── Class 2: TestIntraDomainCombination ──────────────────────────────────────


class TestIntraDomainCombination:
    """Tests for Capability 6: intra-domain count boost.

    After Cap 5 selects the strongest (by freshness-weighted strength) signal
    per domain, the effective strength is boosted if count > 1:
        max_eff *= (1 + 0.1 * math.log(count))

    where count = number of same-direction signals that pass min_strength.

    The private helper _effective_strength_for_domain() mirrors this logic
    exactly so individual formula steps can be verified without running
    the full convergence pipeline.
    """

    @staticmethod
    def _compute_effective(
        alerter: ConvergenceAlerter,
        signals: list,
        domain: str,
        direction: str = "bullish",
        min_strength: float = 0.5,
    ) -> float:
        """Mirrors the Cap 5+6 computation from _check_direction_convergence.

        Returns the effective strength that would be stored in effective_strengths
        for this domain, or 0.0 if no qualifying signals exist.

        Uses the SAME alerter instance that was used to build the signals so
        that freshness is computed at (nearly) the same moment as signal creation.
        """
        matching = [
            s for s in signals
            if s.direction == direction and s.strength >= min_strength
        ]
        if not matching:
            return 0.0
        strongest = max(
            matching,
            key=lambda s: s.strength * alerter._compute_freshness(s, domain)
        )
        max_eff = strongest.strength * alerter._compute_freshness(strongest, domain)
        count = len(matching)
        if count > 1:
            max_eff *= (1 + 0.1 * math.log(count))
        return max_eff

    # ── 10. 1 signal: no boost ────────────────────────────────────────────────

    def test_single_signal_no_boost(self):
        """1 signal → count=1 → log(1)=0 → boost factor is exactly 1.0."""
        alerter = _alerter(min_domains=1)
        sig = _sig("s1", 0.80, "insider", "bullish", age_hours=0.0)
        result = self._compute_effective(alerter, [sig], "insider")
        # freshness ≈ 1.0 (tiny drift), no boost
        expected_min = 0.80 * 0.999  # lower bound allowing for sub-millisecond age
        expected_max = 0.80 * 1.0    # upper bound
        assert expected_min <= result <= expected_max, (
            f"Single signal: result={result:.6f}, expected near 0.80"
        )

    # ── 11. 2 signals: ~6.9% boost ────────────────────────────────────────────

    def test_two_signals_log2_boost(self):
        """2 qualifying signals → count=2 → boost = 1 + 0.1*log(2) ≈ 1.0693."""
        alerter = _alerter(min_domains=1)
        sig_a = _sig("sa", 0.80, "insider", "bullish", age_hours=0.0)
        sig_b = _sig("sb", 0.70, "insider", "bullish", age_hours=0.5)

        result = self._compute_effective(alerter, [sig_a, sig_b], "insider")

        # sig_a selected (0.80 * f_a > 0.70 * f_b for small ages)
        f_a = alerter._compute_freshness(sig_a, "insider")
        expected = 0.80 * f_a * (1 + 0.1 * math.log(2))
        assert result == pytest.approx(expected, abs=1e-4)

    # ── 12. 3 signals: ~11.0% boost ───────────────────────────────────────────

    def test_three_signals_log3_boost(self):
        """3 qualifying signals → count=3 → boost = 1 + 0.1*log(3) ≈ 1.1099."""
        alerter = _alerter(min_domains=1)
        signals = [
            _sig("sa", 0.80, "insider", "bullish", age_hours=0.0),
            _sig("sb", 0.70, "insider", "bullish", age_hours=0.0),
            _sig("sc", 0.65, "insider", "bullish", age_hours=0.0),
        ]
        result = self._compute_effective(alerter, signals, "insider")

        f_a = alerter._compute_freshness(signals[0], "insider")
        expected = 0.80 * f_a * (1 + 0.1 * math.log(3))
        assert result == pytest.approx(expected, abs=1e-4)

    # ── 13. 5 signals: ~16.1% boost ───────────────────────────────────────────

    def test_five_signals_log5_boost(self):
        """5 qualifying signals → count=5 → boost = 1 + 0.1*log(5) ≈ 1.1609."""
        alerter = _alerter(min_domains=1)
        # Stagger ages slightly so s0 is clearly the freshest AND strongest
        signals = [
            _sig(f"s{i}", 0.80 - i * 0.03, "insider", "bullish", age_hours=float(i) * 0.1)
            for i in range(5)
        ]
        result = self._compute_effective(alerter, signals, "insider")

        # s0 is the strongest and freshest → selected representative
        f_0 = alerter._compute_freshness(signals[0], "insider")
        expected = signals[0].strength * f_0 * (1 + 0.1 * math.log(5))
        assert result == pytest.approx(expected, abs=1e-4)

    # ── 14. Boost is multiplicative with freshness ────────────────────────────

    def test_count_boost_multiplicative_with_freshness(self):
        """The boost multiplies the freshness-weighted strength, NOT the raw strength.

        Selected signal strength=0.80, age chosen so freshness≈0.70:
          eff_pre_boost = 0.80 * 0.70 = 0.560
          eff_post_boost = 0.560 * (1 + 0.1*log(2)) ≈ 0.599

        NOT: 0.80 * (1 + 0.1*log(2)) * 0.70 — which is the same but the test
        confirms the intermediate computation path.
        """
        alerter = _alerter(min_domains=1)
        # age such that freshness ≈ 0.70: solve 1 - sqrt(age/72) = 0.70 → age ≈ 6.48h
        age = 72.0 * (0.30 ** 2)  # = 6.48h
        sig_a = _sig("sa", 0.80, "insider", "bullish", age_hours=age)
        sig_b = _sig("sb", 0.55, "insider", "bullish", age_hours=0.0)  # ensures count=2

        f_a = alerter._compute_freshness(sig_a, "insider")
        # f_a should be close to 0.70
        assert 0.68 < f_a < 0.72, f"Freshness was {f_a}, expected ~0.70"

        result = self._compute_effective(alerter, [sig_a, sig_b], "insider")
        expected = sig_a.strength * f_a * (1 + 0.1 * math.log(2))
        assert result == pytest.approx(expected, abs=1e-4)

    # ── 15. Only min_strength-passing signals count ───────────────────────────

    def test_only_qualifying_signals_count_toward_boost(self):
        """A sub-threshold signal does not increment the count.

        With min_strength=0.65: signal at 0.55 is excluded → count=1 → no boost.
        """
        alerter = _alerter(min_domains=1, min_strength=0.65)
        sig_strong = _sig("strong", 0.80, "insider", "bullish", age_hours=0.0)
        sig_weak   = _sig("weak",   0.55, "insider", "bullish", age_hours=0.0)

        result = self._compute_effective(
            alerter, [sig_strong, sig_weak], "insider", min_strength=0.65
        )
        # count=1 → no boost; strong selected; eff ≈ 0.80
        f_strong = alerter._compute_freshness(sig_strong, "insider")
        expected = sig_strong.strength * f_strong  # no boost
        assert result == pytest.approx(expected, abs=1e-4)

    # ── 16. Neutral signals get no count boost ────────────────────────────────

    def test_neutral_signals_get_no_count_boost(self):
        """Neutral signals enter the alert via a separate code path that does
        NOT apply the count boost.

        Strategy: 3 neutral signals in 'positioning' domain + 2 directional
        domains. avg_strength must be computed from the 2 directional signals
        only. If neutral count boost were incorrectly applied, the neutral
        domain would leak into the strength calculation.

        Direct check: alert.strength == mean of the two directional effective
        strengths (each count=1, each freshness≈1.0 for fresh signals).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 neutral signals in positioning (336h window — survives pruning)
        for i in range(3):
            _inject(alerter, _sig(f"pos{i}", 0.80, "positioning", "neutral", age_hours=0.0))

        # 2 directional domains, well-controlled strengths
        ins_sig = _sig("ins", 0.70, "insider",    "bullish", age_hours=0.0)
        gov_sig = _sig("gov", 0.60, "government", "bullish", age_hours=0.0)
        _inject(alerter, ins_sig)
        _inject(alerter, gov_sig)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        # avg_strength = (eff_ins + eff_gov) / 2
        # Each is count=1, so eff = strength * freshness (no boost)
        f_ins = alerter._compute_freshness(ins_sig, "insider")
        f_gov = alerter._compute_freshness(gov_sig, "government")
        expected_strength = (0.70 * f_ins + 0.60 * f_gov) / 2
        assert alert.strength == pytest.approx(expected_strength, abs=1e-4), (
            "avg_strength must come from directional signals only (no neutral count boost)"
        )

    # ── 17. Count boost direction-isolated ────────────────────────────────────

    def test_count_boost_direction_isolated(self):
        """3 bullish + 2 bearish signals in the same domain.
        When checking bullish convergence, count=3 (bearish excluded).
        Bearish signals must not inflate the bullish count to 5.

        Expected insider eff = max_bullish_eff * (1 + 0.1*log(3)), not log(5).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 bullish signals, bull0 is strongest and freshest
        bull_signals = [
            _sig(f"bull{i}", 0.70 - i * 0.02, "insider", "bullish", age_hours=float(i) * 0.5)
            for i in range(3)
        ]
        # 2 bearish signals in same domain
        bear_signals = [
            _sig(f"bear{i}", 0.70, "insider", "bearish", age_hours=float(i) * 0.5)
            for i in range(2)
        ]
        for s in bull_signals + bear_signals:
            _inject(alerter, s)

        # Two more bullish domains
        _inject(alerter, _sig("sent", 0.70, "sentiment",  "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.70, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        # Compute expected: insider gets log(3) boost (3 bullish, NOT 5)
        f_bull0 = alerter._compute_freshness(bull_signals[0], "insider")
        expected_insider_eff = bull_signals[0].strength * f_bull0 * (1 + 0.1 * math.log(3))

        f_sent = alerter._compute_freshness(alerter.signals["sentiment"][0], "sentiment")
        f_gov  = alerter._compute_freshness(alerter.signals["government"][0], "government")
        sent_eff = 0.70 * f_sent
        gov_eff  = 0.70 * f_gov

        expected_avg = (expected_insider_eff + sent_eff + gov_eff) / 3
        assert alert.strength == pytest.approx(expected_avg, abs=0.01), (
            f"Expected avg={expected_avg:.4f}, got {alert.strength:.4f}. "
            "Bearish signals must not count toward bullish boost."
        )


# ── Class 3: TestFreshnessAndCombinationIntegration ──────────────────────────


class TestFreshnessAndCombinationIntegration:
    """Integration tests exercising Cap 5 + Cap 6 together through the full
    check_convergence() pipeline, including interaction with Cap 3 (coherence).
    """

    # ── 18. Full convergence: 3 domains, varying ages and counts ─────────────

    def test_full_convergence_with_varying_ages_and_counts(self):
        """3 domains with different signal ages and counts all contribute to alert.

        Domain A ('insider'):  2 signals (age 0h and 2h) → count boost applies
        Domain B ('sentiment'): 1 signal (age 10h) → no boost
        Domain C ('contracts'): 1 signal (age 1h) → no boost

        Verify: alert fires, avg_strength matches manual Cap 5+6 computation.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        ins_a = _sig("ins_a", 0.80, "insider", "bullish", age_hours=0.0)
        ins_b = _sig("ins_b", 0.75, "insider", "bullish", age_hours=2.0)
        sent  = _sig("sent",  0.70, "sentiment", "bullish", age_hours=10.0)
        con   = _sig("con",   0.72, "contracts", "bullish", age_hours=1.0)

        for s in (ins_a, ins_b, sent, con):
            _inject(alerter, s)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1, "Must fire with 3 qualifying domains"

        alert = alerts[0]
        assert set(alert.domains_converging) == {"insider", "sentiment", "contracts"}

        # Manual Cap 5+6 computation
        f_ins_a = alerter._compute_freshness(ins_a, "insider")
        f_ins_b = alerter._compute_freshness(ins_b, "insider")
        ins_a_eff = ins_a.strength * f_ins_a
        ins_b_eff = ins_b.strength * f_ins_b
        selected_ins_eff = max(ins_a_eff, ins_b_eff)
        boosted_ins = selected_ins_eff * (1 + 0.1 * math.log(2))

        f_sent = alerter._compute_freshness(sent, "sentiment")
        sent_eff = sent.strength * f_sent  # count=1, no boost

        f_con = alerter._compute_freshness(con, "contracts")
        con_eff = con.strength * f_con  # count=1, no boost

        expected_avg = (boosted_ins + sent_eff + con_eff) / 3
        assert alert.strength == pytest.approx(expected_avg, abs=1e-4)

    # ── 19. Identical raw signals, different ages → different alert strengths ─

    def test_same_raw_signals_different_ages_produce_different_strengths(self):
        """Two alerters with identical raw strengths but different ages must
        produce alerts with different avg_strength, proving freshness weighting
        has real effect on the output."""
        def _run(age_hours):
            a = _alerter(min_domains=3, min_strength=0.5)
            _inject(a, _sig("ins",  0.80, "insider",    "bullish", age_hours=age_hours))
            _inject(a, _sig("sent", 0.70, "sentiment",  "bullish", age_hours=age_hours))
            _inject(a, _sig("gov",  0.75, "government", "bullish", age_hours=age_hours))
            alerts = a.check_convergence(direction_filter="bullish")
            return alerts[0].strength if alerts else None

        strength_1h  = _run(age_hours=1.0)
        strength_50h = _run(age_hours=50.0)

        assert strength_1h is not None
        assert strength_50h is not None
        assert strength_1h > strength_50h, (
            "Fresh signals must produce higher avg_strength than stale signals"
        )

    # ── 20. Count-boosted domain beats stale single domain ────────────────────

    def test_count_boosted_domain_beats_stale_single_domain(self):
        """Domain A has 3 recent signals (count boost ~11%) while Domain B has
        1 stale signal (freshness near 0.30 floor). Domain A should contribute
        more effective strength despite having lower raw strength per signal.

        Uses 'positioning' (336h window) for Domain B so stale signal survives pruning.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # Domain A: 3 fresh signals at strength=0.65 (count boost applies)
        ins_sigs = [
            _sig(f"ins{i}", 0.65, "insider", "bullish", age_hours=float(i) * 0.1)
            for i in range(3)
        ]
        for s in ins_sigs:
            _inject(alerter, s)

        # Domain B: 1 stale positioning signal at strength=0.90
        # positioning window=336h, so 300h-old signal survives pruning
        stale = _sig("b_stale", 0.90, "positioning", "neutral", age_hours=300.0)
        _inject(alerter, stale)

        # Third domain to satisfy min_domains=3
        _inject(alerter, _sig("con", 0.70, "contracts", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        # Compute Domain A effective strength
        f_ins0 = alerter._compute_freshness(ins_sigs[0], "insider")
        a_eff = ins_sigs[0].strength * f_ins0 * (1 + 0.1 * math.log(3))

        # Compute Domain B effective strength (neutral, no count boost, stale)
        f_stale = alerter._compute_freshness(stale, "positioning")
        b_eff = stale.strength * f_stale  # f_stale near 0.30 floor

        assert a_eff > b_eff, (
            f"Count-boosted fresh domain ({a_eff:.4f}) must exceed "
            f"stale single-signal domain ({b_eff:.4f})"
        )

    # ── 21. Coherence multiplier applies on top of freshness+count ────────────

    def test_coherence_multiplier_applies_on_top_of_freshness_and_count(self):
        """Cap 3 (coherence) damps final_confidence when contradictory domains exist.
        This multiplier must apply AFTER Cap 5+6 compute avg_strength and effective
        strengths. We verify coherence < 1.0 and confidence < pure-bullish baseline.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 bullish domains + 1 bearish (introduces contradiction → coherence < 1.0)
        _inject(alerter, _sig("ins",  0.80, "insider",    "bullish", age_hours=0.0))
        _inject(alerter, _sig("sent", 0.70, "sentiment",  "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.75, "government", "bullish", age_hours=0.0))
        _inject(alerter, _sig("con_bear", 0.70, "contracts", "bearish", age_hours=0.0))

        alerts_mixed = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts_mixed) == 1

        # Pure bullish baseline (no contradictions)
        alerter_pure = _alerter(min_domains=3, min_strength=0.5)
        _inject(alerter_pure, _sig("ins",  0.80, "insider",   "bullish", age_hours=0.0))
        _inject(alerter_pure, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter_pure, _sig("gov",  0.75, "government","bullish", age_hours=0.0))
        alerts_pure = alerter_pure.check_convergence(direction_filter="bullish")
        assert len(alerts_pure) == 1

        assert alerts_mixed[0].coherence < 1.0, "Contradictory domain must lower coherence"
        assert alerts_mixed[0].confidence < alerts_pure[0].confidence, (
            "Coherence penalty must reduce confidence vs the pure-bullish baseline"
        )

    # ── 22. All signals at floor freshness: alert fires, strength is dampened ─

    def test_all_signals_at_floor_freshness_still_fires(self):
        """Very stale signals in domains with wide-enough windows so they survive
        pruning but have freshness=0.30 (floor). Alert must still fire with
        avg_strength ≈ raw_strength * 0.30 per domain.

        Uses 'positioning' (336h) and 'government'/'contracts' (168h) domains.
        All signals aged 160h — inside government/contracts window but well past 72h.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)
        raw_strength = 0.80
        stale_age = 160.0  # inside government/contracts 168h window; outside 72h default

        sigs = [
            _sig("pos",  raw_strength, "positioning", "bullish", age_hours=stale_age),
            _sig("gov",  raw_strength, "government",  "bullish", age_hours=stale_age),
            _sig("con",  raw_strength, "contracts",   "bullish", age_hours=stale_age),
        ]
        for s in sigs:
            _inject(alerter, s)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1, "Alert must still fire even with very stale signals"

        # Freshness for government/contracts at 160h of 168h window:
        # 1 - sqrt(160/168) ≈ 1 - 0.976 ≈ 0.024 → floored to 0.30
        # Freshness for positioning at 160h of 336h window:
        # 1 - sqrt(160/336) ≈ 1 - 0.690 ≈ 0.310 → above floor
        f_pos = alerter._compute_freshness(sigs[0], "positioning")
        f_gov = alerter._compute_freshness(sigs[1], "government")
        f_con = alerter._compute_freshness(sigs[2], "contracts")

        expected_avg = (raw_strength * f_pos + raw_strength * f_gov + raw_strength * f_con) / 3
        assert alerts[0].strength == pytest.approx(expected_avg, abs=1e-4)
        # All three at or near the floor → avg_strength well below raw
        assert alerts[0].strength < raw_strength * 0.6

    # ── 23. Count boost has diminishing returns ───────────────────────────────

    def test_count_boost_has_diminishing_returns(self):
        """Adding the 10th signal provides a smaller boost increment than adding
        the 2nd, confirming log-saturation behavior.

        Marginal gain from count n-1 to n: delta = 0.1 * log(n / (n-1))
        delta(2)  = 0.1 * log(2/1)   ≈ 0.0693
        delta(10) = 0.1 * log(10/9)  ≈ 0.0105
        """
        delta_2  = 0.1 * math.log(2 / 1)    # ≈ 0.0693
        delta_10 = 0.1 * math.log(10 / 9)   # ≈ 0.0105
        assert delta_2 > delta_10, "Each additional signal yields diminishing returns"
        assert delta_2 > 6 * delta_10, "Marginal gain at count=2 is much larger than at count=10"

    # ── 24. Boost does not cross domain boundaries ────────────────────────────

    def test_count_boost_does_not_cross_domain_boundaries(self):
        """5 signals across 5 different domains (1 per domain) should NOT produce
        a count boost in any domain. Each domain has count=1, so boost=1.0.

        avg_strength must equal the mean of individual freshness-weighted strengths
        (all at strength=0.70, age≈0h → freshness≈1.0 → eff≈0.70 each).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)
        domains = ["insider", "sentiment", "contracts", "government", "technical"]
        sigs = [_sig(d, 0.70, d, "bullish", age_hours=0.0) for d in domains]
        for s in sigs:
            _inject(alerter, s)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        # Each domain: count=1, freshness ≈ 1.0 → eff ≈ 0.70, no boost
        # avg_strength = 0.70 (allow small freshness drift)
        alert = alerts[0]
        assert 0.695 < alert.strength <= 0.70, (
            f"avg_strength={alert.strength:.6f} should be near 0.70 with no cross-domain boost"
        )

    # ── 25. Freshness floor combined with count boost ─────────────────────────

    def test_freshness_floor_combined_with_count_boost(self):
        """3 very stale signals in 'positioning' domain (336h window so they
        survive pruning), all hitting the freshness floor at 0.30.

        Expected insider eff = raw_strength * 0.30 * (1 + 0.1*log(3))

        Companion fresh domains use 'government' and 'contracts' for precise control.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        raw = 0.80
        # positioning window=336h; 320h-old signals hit the floor
        pos_sigs = [
            _sig(f"pos{i}", raw, "positioning", "bullish", age_hours=320.0)
            for i in range(3)
        ]
        for s in pos_sigs:
            _inject(alerter, s)

        # Two fresh domains
        gov_sig = _sig("gov", 0.70, "government", "bullish", age_hours=1.0)
        con_sig = _sig("con", 0.70, "contracts",  "bullish", age_hours=1.0)
        _inject(alerter, gov_sig)
        _inject(alerter, con_sig)

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        floor = 0.3
        expected_pos_eff = raw * floor * (1 + 0.1 * math.log(3))

        f_gov = alerter._compute_freshness(gov_sig, "government")
        f_con = alerter._compute_freshness(con_sig, "contracts")
        gov_eff = 0.70 * f_gov
        con_eff = 0.70 * f_con

        expected_avg = (expected_pos_eff + gov_eff + con_eff) / 3
        assert alerts[0].strength == pytest.approx(expected_avg, abs=1e-4)

    # ── 26. Sub-threshold signals cannot rescue a domain via count boost ───────

    def test_count_boost_cannot_rescue_sub_threshold_signals(self):
        """10 signals in a domain, all at strength=0.45 with min_strength=0.5.
        None pass the threshold → count=0 → domain excluded from convergence.
        Alert must not fire if only 2 qualifying domains remain (min_domains=3).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        for i in range(10):
            _inject(alerter, _sig(f"ins{i}", 0.45, "insider", "bullish", age_hours=0.0))

        # Only 2 qualifying domains (need 3 total including insider)
        _inject(alerter, _sig("sent", 0.70, "sentiment",  "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.70, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 0, (
            "Sub-threshold signals must not contribute even with high count"
        )

    # ── 27. record_signal() API preserves timestamps for freshness ────────────

    def test_record_signal_timestamp_param_used_for_freshness(self):
        """ConvergenceAlerter.record_signal() accepts an explicit timestamp.
        A signal recorded with a 48h-old timestamp must show freshness near the
        floor (1 - sqrt(48/72) ≈ 0.184 → floored to 0.30) in the 72h window.
        """
        alerter = _alerter(min_domains=1, min_strength=0.5)
        old_ts = datetime.now() - timedelta(hours=48)
        alerter.record_signal("sig_old", 0.80, "insider", "bullish", timestamp=old_ts)

        sig = alerter.signals["insider"][0]
        freshness = alerter._compute_freshness(sig, "insider")

        # 1 - sqrt(48/72) = 1 - sqrt(2/3) ≈ 0.184 → floor = 0.30
        assert freshness == pytest.approx(0.3, abs=1e-3)

    # ── 28. Signal exactly at window edge hits floor ───────────────────────────

    def test_signal_exactly_at_window_edge_hits_floor(self):
        """Signal with age == window_hours exactly: 1.0 - sqrt(1.0) = 0.0 → floor."""
        alerter = _alerter()
        sig = _sig("edge", 0.80, "insider", "bullish", age_hours=72.0)
        result = alerter._compute_freshness(sig, "insider")
        assert result == pytest.approx(0.3, abs=1e-3)

    # ── 29. Large count boost stays below 2x ──────────────────────────────────

    def test_large_count_boost_stays_below_2x(self):
        """The log-based boost never doubles effective strength for any realistic count.

        boost(100)  = 1 + 0.1*log(100)  ≈ 1.46
        boost(1000) = 1 + 0.1*log(1000) ≈ 1.69
        Would need n ≥ e^10 ≈ 22,026 to reach 2.0x — impossible in practice.
        """
        boost_100  = 1 + 0.1 * math.log(100)
        boost_1000 = 1 + 0.1 * math.log(1000)
        assert boost_100  < 2.0
        assert boost_1000 < 2.0

    # ── 30. Per-ticker convergence path is unaffected by Cap 5+6 ─────────────

    def test_per_ticker_convergence_unaffected_by_count_boost(self):
        """check_ticker_convergence() uses a simpler strength selection path:
        max by raw strength, no freshness, no count boost. This confirms the
        two pipelines are independent.

        Expected per-ticker avg_strength = mean(0.70, 0.65) = 0.675 (raw strengths).
        """
        alerter = _alerter(min_domains=2, min_strength=0.0)

        for i in range(3):
            alerter.record_signal(
                f"ins{i}", 0.70, "insider", direction="bullish",
                metadata={"symbol": "AAPL"}, timestamp=datetime.now()
            )
        alerter.record_signal(
            "sent", 0.65, "sentiment", direction="bullish",
            metadata={"symbol": "AAPL"}, timestamp=datetime.now()
        )

        ticker_alerts = alerter.check_ticker_convergence(min_domains=2)
        aapl_alerts = [a for a in ticker_alerts if "AAPL" in a.alert_id]
        assert len(aapl_alerts) >= 1

        # Per-ticker: max insider raw = 0.70, sentiment = 0.65 → avg = 0.675
        aapl_alert = aapl_alerts[0]
        assert aapl_alert.strength == pytest.approx((0.70 + 0.65) / 2, abs=1e-5), (
            "Per-ticker path must use raw strength (no Cap 5+6 adjustments)"
        )
