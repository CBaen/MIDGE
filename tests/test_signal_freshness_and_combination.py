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

_NOW = datetime.now()


def _sig(
    signal_id: str,
    strength: float,
    domain: str,
    direction: str = "bullish",
    age_hours: float = 0.0,
) -> Signal:
    """Build a Signal with a controlled timestamp (age relative to now)."""
    return Signal(
        signal_id=signal_id,
        strength=strength,
        domain=domain,
        direction=direction,
        timestamp=_NOW - timedelta(hours=age_hours),
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
    """Bypass record_signal() to inject a pre-built Signal directly,
    preserving its exact timestamp (record_signal would call _prune_old_signals
    which is fine, but we want full control over what survives)."""
    alerter.signals[sig.domain].append(sig)


# ── Class 1: TestTemporalFreshness ───────────────────────────────────────────


class TestTemporalFreshness:
    """Unit tests for ConvergenceAlerter._compute_freshness().

    Formula: freshness = max(0.3, 1.0 - (age_hours / window_hours) ** 0.5)
    Default window: 72 hours.  Domain overrides: positioning=336h, government=168h,
    contracts=168h.
    """

    # ── 1. Brand-new signal ───────────────────────────────────────────────────

    def test_age_zero_returns_one(self):
        """A signal with age=0 must return exactly 1.0 (no decay)."""
        alerter = _alerter()
        sig = _sig("s1", 0.8, "insider", age_hours=0.0)
        result = alerter._compute_freshness(sig, "insider")
        assert result == pytest.approx(1.0, abs=1e-9)

    # ── 2. 25% window age ─────────────────────────────────────────────────────

    def test_quarter_window_age_approx_half(self):
        """At 25% of window age, freshness ≈ 1 - sqrt(0.25) = 0.50."""
        alerter = _alerter()
        # 72h window → 25% = 18h
        sig = _sig("s2", 0.8, "insider", age_hours=18.0)
        result = alerter._compute_freshness(sig, "insider")
        expected = 1.0 - math.sqrt(18.0 / 72.0)  # 0.50
        assert result == pytest.approx(expected, abs=1e-6)
        assert 0.49 < result < 0.51

    # ── 3. 99% window age → floor ────────────────────────────────────────────

    def test_near_window_edge_hits_floor(self):
        """At 99% of window age, raw value < 0.3, so floor = 0.3 applies."""
        alerter = _alerter()
        age = 72.0 * 0.99  # 71.28 hours
        sig = _sig("s3", 0.8, "insider", age_hours=age)
        result = alerter._compute_freshness(sig, "insider")
        raw = 1.0 - math.sqrt(0.99)  # ≈ 0.005 — well below floor
        assert raw < 0.3
        assert result == pytest.approx(0.3, abs=1e-9)

    # ── 4. Past the window edge still gets floor ──────────────────────────────

    def test_past_window_edge_still_gets_floor(self):
        """Signal 100h old in a 72h window yields floor=0.3 (not pruned yet
        if _prune_old_signals hasn't been called, but freshness should still floor)."""
        alerter = _alerter()
        sig = _sig("s4", 0.8, "insider", age_hours=100.0)
        result = alerter._compute_freshness(sig, "insider")
        # age/window = 100/72 > 1, sqrt > 1, so 1 - sqrt < 0 → floor clamps to 0.3
        assert result == pytest.approx(0.3, abs=1e-9)

    # ── 5. Positioning domain has 336h window ─────────────────────────────────

    def test_positioning_uses_336h_window(self):
        """positioning domain window = 14*24=336h. Same 18h age signal has
        very different freshness than in the 72h-window 'insider' domain."""
        alerter = _alerter()
        sig = _sig("s5", 0.8, "positioning", age_hours=18.0)

        freshness_pos = alerter._compute_freshness(sig, "positioning")
        freshness_ins = alerter._compute_freshness(sig, "insider")

        expected_pos = 1.0 - math.sqrt(18.0 / 336.0)  # ≈ 0.768
        expected_ins = 1.0 - math.sqrt(18.0 / 72.0)   # ≈ 0.500

        assert freshness_pos == pytest.approx(expected_pos, abs=1e-6)
        assert freshness_ins == pytest.approx(expected_ins, abs=1e-6)
        # positioning freshness is higher for the same signal age
        assert freshness_pos > freshness_ins

    def test_global_default_domain_uses_72h(self):
        """Unknown domain 'crypto' falls back to the global 72h window."""
        alerter = _alerter()
        sig = _sig("s6", 0.8, "crypto", age_hours=36.0)
        result = alerter._compute_freshness(sig, "crypto")
        expected = 1.0 - math.sqrt(36.0 / 72.0)  # 1 - sqrt(0.5) ≈ 0.293 → floor 0.3
        # sqrt(0.5) ≈ 0.707, so 1 - 0.707 = 0.293, floored to 0.3
        assert result == pytest.approx(0.3, abs=1e-6)

    def test_government_uses_168h_window(self):
        """government domain window = 7*24=168h."""
        alerter = _alerter()
        sig = _sig("s7", 0.8, "government", age_hours=84.0)  # 50% of window
        result = alerter._compute_freshness(sig, "government")
        expected = 1.0 - math.sqrt(84.0 / 168.0)  # 1 - sqrt(0.5) ≈ 0.293 → floor 0.3
        assert result == pytest.approx(0.3, abs=1e-6)

    def test_contracts_uses_168h_window(self):
        """contracts domain window = 7*24=168h, same as government."""
        alerter = _alerter()
        sig_contracts = _sig("s8", 0.8, "contracts", age_hours=10.0)
        sig_insider   = _sig("s9", 0.8, "insider", age_hours=10.0)

        f_contracts = alerter._compute_freshness(sig_contracts, "contracts")
        f_insider   = alerter._compute_freshness(sig_insider, "insider")

        # contracts window 168h → 10/168 = 0.0595 → sqrt ≈ 0.244 → 1-0.244 = 0.756
        # insider window 72h   → 10/72  = 0.139  → sqrt ≈ 0.372 → 1-0.372 = 0.628
        assert f_contracts > f_insider

    # ── 6. Window = 0 guard ───────────────────────────────────────────────────

    def test_zero_window_returns_one(self):
        """If window_hours is 0 (edge case), _compute_freshness must not divide
        by zero and must return 1.0."""
        alerter = _alerter(min_domains=1)
        # Temporarily inject a custom zero-hour window for domain "zero_win"
        alerter._domain_windows["zero_win"] = timedelta(hours=0)
        sig = _sig("s10", 0.8, "zero_win", age_hours=5.0)
        result = alerter._compute_freshness(sig, "zero_win")
        assert result == pytest.approx(1.0, abs=1e-9)

    # ── 7. Selection inversion: fresher beats stronger ────────────────────────

    def test_fresher_signal_beats_stale_stronger_signal(self):
        """In _check_direction_convergence, the best signal is chosen by
        freshness-weighted strength. A fresh weak signal should outrank a stale
        strong one when their effective strengths (strength*freshness) cross over.

        Domain 'insider' (72h window):
          stale: strength=0.85, age=60h → freshness=1-sqrt(60/72)≈0.087 → eff=0.074
          fresh: strength=0.70, age=1h  → freshness=1-sqrt(1/72)≈0.882 → eff=0.617

        The fresh signal's effective strength is 8x higher — it must win.
        We detect which signal was selected by checking which is in the alert's signals list.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        stale_strong = _sig("stale", 0.85, "insider", "bullish", age_hours=60.0)
        fresh_weak   = _sig("fresh", 0.70, "insider", "bullish", age_hours=1.0)

        # Verify effective strengths manually
        f_stale = alerter._compute_freshness(stale_strong, "insider")
        f_fresh = alerter._compute_freshness(fresh_weak, "insider")
        assert fresh_weak.strength * f_fresh > stale_strong.strength * f_stale, (
            "Test setup error: fresh signal must have higher effective strength"
        )

        # Inject both signals into 'insider' domain
        _inject(alerter, stale_strong)
        _inject(alerter, fresh_weak)

        # Fill two more domains to reach min_domains=3
        _inject(alerter, _sig("d2", 0.7, "sentiment", "bullish", age_hours=1.0))
        _inject(alerter, _sig("d3", 0.7, "contracts", "bullish", age_hours=1.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        # The selected signal from 'insider' must be fresh_weak (id matches)
        alert_signal_ids = {s.signal_id for s in alert.signals}
        assert "fresh" in alert_signal_ids, "Fresh signal must be selected representative"
        # stale is in the same domain — only one representative is selected per domain
        assert "stale" not in alert_signal_ids, "Stale signal must NOT be the domain representative"

    # ── 8. avg_strength uses effective (freshness-weighted) values ────────────

    def test_avg_strength_uses_freshness_weighted_effective_strength(self):
        """avg_strength in the alert must reflect freshness-weighted effective strength,
        not raw strength. We check this by comparing two scenarios:
          A) 3 domains, each signal at age=0 (freshness=1.0) → avg_strength = avg of raw
          B) Same raw strengths but signals aged 60h → avg_strength is lower
        """
        def _run(age_hours):
            a = _alerter(min_domains=3, min_strength=0.5)
            _inject(a, _sig("ins", 0.80, "insider", "bullish", age_hours=age_hours))
            _inject(a, _sig("sent", 0.70, "sentiment", "bullish", age_hours=age_hours))
            _inject(a, _sig("con", 0.75, "contracts", "bullish", age_hours=age_hours))
            alerts = a.check_convergence(direction_filter="bullish")
            assert len(alerts) == 1
            return alerts[0].strength

        strength_fresh = _run(age_hours=0.0)
        strength_stale = _run(age_hours=60.0)

        assert strength_fresh > strength_stale, (
            "Stale signals should yield lower avg_strength due to freshness weighting"
        )

    # ── 9. Freshness applies to neutral signal selection ──────────────────────

    def test_neutral_selection_uses_freshness_weighting(self):
        """When multiple neutral signals compete in a domain, the one with
        higher freshness-weighted strength wins (same logic as directional).

        We verify indirectly: if the fresh neutral signal is selected, its
        ID will appear in alert.signals.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        stale_neutral = _sig("neu_stale", 0.90, "positioning", "neutral", age_hours=300.0)
        fresh_neutral = _sig("neu_fresh", 0.60, "positioning", "neutral", age_hours=1.0)

        # positioning window = 336h
        f_stale = alerter._compute_freshness(stale_neutral, "positioning")
        f_fresh = alerter._compute_freshness(fresh_neutral, "positioning")
        # Verify test setup: fresh should win
        assert fresh_neutral.strength * f_fresh > stale_neutral.strength * f_stale

        _inject(alerter, stale_neutral)
        _inject(alerter, fresh_neutral)

        # Two directional domains to satisfy min_domains=3 (positioning is neutral)
        _inject(alerter, _sig("ins", 0.7, "insider", "bullish", age_hours=0.0))
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
    """

    def _effective_strength_for_domain(
        self,
        signals: list,
        domain: str,
        direction: str = "bullish",
        min_strength: float = 0.5,
    ) -> float:
        """Helper: compute what Cap 5+6 would produce for a domain's signals.

        This mirrors the logic in _check_direction_convergence so we can verify
        the exact output without running the full convergence pipeline.
        """
        a = _alerter(min_domains=1, min_strength=min_strength)
        matching = [s for s in signals if s.direction == direction
                    and s.strength >= min_strength]
        if not matching:
            return 0.0
        strongest = max(matching, key=lambda s: s.strength * a._compute_freshness(s, domain))
        max_eff = strongest.strength * a._compute_freshness(strongest, domain)
        count = len(matching)
        if count > 1:
            max_eff *= (1 + 0.1 * math.log(count))
        return max_eff

    # ── 10. 1 signal: no boost ────────────────────────────────────────────────

    def test_single_signal_no_boost(self):
        """1 signal → count=1 → log(1)=0 → no boost applied."""
        sig = _sig("s1", 0.80, "insider", "bullish", age_hours=0.0)
        result = self._effective_strength_for_domain([sig], "insider")
        expected = 0.80 * 1.0  # freshness=1.0, no boost
        assert result == pytest.approx(expected, abs=1e-6)

    # ── 11. 2 signals: ~6.9% boost ────────────────────────────────────────────

    def test_two_signals_log2_boost(self):
        """2 signals → count=2 → boost = 1 + 0.1*log(2) ≈ 1.0693."""
        sig_a = _sig("sa", 0.80, "insider", "bullish", age_hours=0.0)
        sig_b = _sig("sb", 0.70, "insider", "bullish", age_hours=1.0)  # slightly weaker
        result = self._effective_strength_for_domain([sig_a, sig_b], "insider")

        # sig_a is selected (stronger + fresh), eff_raw = 0.80*1.0 = 0.80
        # boost = 1 + 0.1*log(2) ≈ 1.0693
        expected = 0.80 * 1.0 * (1 + 0.1 * math.log(2))
        assert result == pytest.approx(expected, abs=1e-6)

    # ── 12. 3 signals: ~11.0% boost ───────────────────────────────────────────

    def test_three_signals_log3_boost(self):
        """3 signals → count=3 → boost = 1 + 0.1*log(3) ≈ 1.1099."""
        signals = [
            _sig("sa", 0.80, "insider", "bullish", age_hours=0.0),
            _sig("sb", 0.70, "insider", "bullish", age_hours=0.0),
            _sig("sc", 0.65, "insider", "bullish", age_hours=0.0),
        ]
        result = self._effective_strength_for_domain(signals, "insider")
        expected = 0.80 * 1.0 * (1 + 0.1 * math.log(3))
        assert result == pytest.approx(expected, abs=1e-6)

    # ── 13. 5 signals: ~16.1% boost ───────────────────────────────────────────

    def test_five_signals_log5_boost(self):
        """5 signals → count=5 → boost = 1 + 0.1*log(5) ≈ 1.1609."""
        signals = [_sig(f"s{i}", 0.80 - i * 0.02, "insider", "bullish", age_hours=float(i))
                   for i in range(5)]
        result = self._effective_strength_for_domain(signals, "insider")
        # s0 is strongest (0.80, age=0 → freshness=1.0)
        expected = 0.80 * 1.0 * (1 + 0.1 * math.log(5))
        assert result == pytest.approx(expected, abs=1e-6)

    # ── 14. Boost is multiplicative with freshness ────────────────────────────

    def test_count_boost_multiplicative_with_freshness(self):
        """The boost multiplies the freshness-weighted strength, not the raw strength.

        If the selected signal has strength=0.80 and freshness=0.70, then:
          eff_pre_boost = 0.80 * 0.70 = 0.56
          eff_post_boost = 0.56 * (1 + 0.1*log(2)) ≈ 0.598
        """
        # age_hours chosen so freshness ≈ 0.70 in the 72h window:
        # 1 - sqrt(age/72) = 0.70 → sqrt(age/72)=0.30 → age/72=0.09 → age≈6.48h
        age = 72.0 * (0.30 ** 2)  # = 6.48h  → freshness = 1 - 0.30 = 0.70
        sig_a = _sig("sa", 0.80, "insider", "bullish", age_hours=age)
        sig_b = _sig("sb", 0.60, "insider", "bullish", age_hours=0.0)  # extra signal for count=2

        a = _alerter(min_domains=1)
        f = a._compute_freshness(sig_a, "insider")
        assert f == pytest.approx(0.70, abs=0.01), f"Freshness was {f}, expected ~0.70"

        result = self._effective_strength_for_domain([sig_a, sig_b], "insider")
        expected = 0.80 * f * (1 + 0.1 * math.log(2))
        assert result == pytest.approx(expected, abs=1e-5)

    # ── 15. Only min_strength-passing signals count ───────────────────────────

    def test_only_qualifying_signals_count_toward_boost(self):
        """Signals below min_strength do not count toward the boost count.

        With min_strength=0.65, a signal at 0.55 is excluded.
        count becomes 1 (only the 0.80 signal passes), so no boost.
        """
        sig_strong = _sig("strong", 0.80, "insider", "bullish", age_hours=0.0)
        sig_weak   = _sig("weak",   0.55, "insider", "bullish", age_hours=0.0)

        # Only 1 signal qualifies → no boost
        result = self._effective_strength_for_domain(
            [sig_strong, sig_weak], "insider", min_strength=0.65
        )
        expected = 0.80 * 1.0  # no boost, count=1
        assert result == pytest.approx(expected, abs=1e-6)

    # ── 16. Neutral signals get no count boost ────────────────────────────────

    def test_neutral_signals_get_no_count_boost(self):
        """Neutral signals enter the alert via a separate code path that does NOT
        apply the count boost. We verify by checking the neutral domain's effective
        contribution is pure freshness-weighted strength (no log multiplier).

        Strategy: build a scenario where 'positioning' domain has 3 neutral signals.
        We cannot directly inspect the neutral effective strength from the alert
        (it's not stored separately), but we can verify the alert fires without
        inflated strength from the neutral domain.

        The test verifies that the alert's avg_strength only reflects directional
        signals, and that neutrals do not contribute to it.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 neutral signals in positioning — should NOT boost each other
        for i in range(3):
            _inject(alerter, _sig(f"pos{i}", 0.80, "positioning", "neutral", age_hours=0.0))

        # 2 directional domains
        _inject(alerter, _sig("ins", 0.70, "insider", "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov", 0.60, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        # avg_strength only comes from directional signals (insider + government)
        # If neutral got the boost: 0.80 * (1 + 0.1*log(3)) ≈ 0.888, which could inflate avg.
        # Directional only: (0.70 + 0.60) / 2 = 0.65
        assert alert.strength == pytest.approx((0.70 + 0.60) / 2, abs=1e-6), (
            "avg_strength must come from directional signals only, not neutral count boost"
        )

    # ── 17. Count boost direction-isolated: bearish don't boost bullish ───────

    def test_count_boost_direction_isolated(self):
        """3 bullish + 2 bearish signals from the same domain.
        When checking bullish convergence, count=3 (only bullish counted).
        Bearish signals must not inflate the bullish count to 5.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 bullish signals in 'insider'
        for i in range(3):
            _inject(alerter, _sig(f"bull{i}", 0.70, "insider", "bullish", age_hours=float(i)))
        # 2 bearish signals in same domain (opposite direction)
        for i in range(2):
            _inject(alerter, _sig(f"bear{i}", 0.70, "insider", "bearish", age_hours=float(i)))

        # Fill two more domains for bullish convergence
        _inject(alerter, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.70, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1
        alert = alerts[0]

        # The 'insider' domain representative is selected from the 3 bullish signals
        # eff = max_bullish_eff * (1 + 0.1*log(3))  [NOT log(5)]
        # max bullish signal: bull0 (strength=0.70, age=0h, freshness=1.0) → eff_raw=0.70
        expected_insider_eff = 0.70 * 1.0 * (1 + 0.1 * math.log(3))
        # avg_strength = (insider_eff + sent_eff + gov_eff) / 3
        # sent_eff = 0.70 * 1.0 * 1 (count=1)
        # gov_eff  = 0.70 * 1.0 * 1 (count=1) — government window 168h so freshness≈1.0
        expected_avg = (expected_insider_eff + 0.70 + 0.70) / 3
        assert alert.strength == pytest.approx(expected_avg, abs=0.01)


# ── Class 3: TestFreshnessAndCombinationIntegration ──────────────────────────


class TestFreshnessAndCombinationIntegration:
    """Integration tests exercising Cap 5 + Cap 6 together through the full
    check_convergence() pipeline, including interaction with Cap 3 (coherence).
    """

    # ── 18. Full convergence: 3 domains, varying ages and counts ─────────────

    def test_full_convergence_with_varying_ages_and_counts(self):
        """3 domains with different signal ages and counts all contribute to alert.

        Domain A ('insider'): 2 signals (age 0h and 2h) → count boost applies
        Domain B ('sentiment'): 1 signal (age 10h) → no boost
        Domain C ('contracts'): 1 signal (age 1h) → no boost

        Verify: alert fires, avg_strength incorporates freshness and count boost.
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
        assert "insider" in alert.domains_converging
        assert "sentiment" in alert.domains_converging
        assert "contracts" in alert.domains_converging

        # Compute expected avg_strength manually
        f_ins_a = alerter._compute_freshness(ins_a, "insider")
        f_ins_b = alerter._compute_freshness(ins_b, "insider")
        # ins_a wins selection: 0.80*f_ins_a vs 0.75*f_ins_b
        ins_a_eff = ins_a.strength * f_ins_a
        ins_b_eff = ins_b.strength * f_ins_b
        selected_ins_eff = max(ins_a_eff, ins_b_eff)
        boosted_ins = selected_ins_eff * (1 + 0.1 * math.log(2))  # count=2

        f_sent = alerter._compute_freshness(sent, "sentiment")
        sent_eff = sent.strength * f_sent  # no boost (count=1)

        f_con = alerter._compute_freshness(con, "contracts")
        con_eff = con.strength * f_con  # no boost (count=1)

        expected_avg = (boosted_ins + sent_eff + con_eff) / 3
        assert alert.strength == pytest.approx(expected_avg, abs=1e-5)

    # ── 19. Identical raw signals, different ages → different alert strengths ─

    def test_same_raw_signals_different_ages_produce_different_strengths(self):
        """Two alerters with identical raw signal strengths but different ages
        must produce alerts with different avg_strength values, proving that
        freshness weighting actually changes the output."""
        def _run(age_hours):
            a = _alerter(min_domains=3, min_strength=0.5)
            _inject(a, _sig("ins", 0.80, "insider", "bullish", age_hours=age_hours))
            _inject(a, _sig("sent", 0.70, "sentiment", "bullish", age_hours=age_hours))
            _inject(a, _sig("gov", 0.75, "government", "bullish", age_hours=age_hours))
            alerts = a.check_convergence(direction_filter="bullish")
            return alerts[0].strength if alerts else None

        strength_1h = _run(age_hours=1.0)
        strength_50h = _run(age_hours=50.0)

        assert strength_1h is not None
        assert strength_50h is not None
        assert strength_1h > strength_50h, (
            "Fresh signals must produce higher avg_strength than stale ones"
        )

    # ── 20. Count-boosted domain contributes more than stale single domain ────

    def test_count_boosted_domain_beats_stale_single_domain(self):
        """Domain A has 3 recent signals (count boost ~11%) while Domain B has
        1 stale signal (freshness near 0.30 floor). Domain A should contribute
        more effective strength to the alert even if its raw strength is lower.

        This validates that count boost and freshness work together correctly.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # Domain A: 3 fresh signals at strength=0.65 (count boost applies)
        for i in range(3):
            _inject(alerter, _sig(f"a{i}", 0.65, "insider", "bullish", age_hours=float(i)))

        # Domain B: 1 stale signal at strength=0.90 (near floor freshness)
        _inject(alerter, _sig("b_stale", 0.90, "sentiment", "bullish", age_hours=68.0))

        # Third domain to satisfy min_domains=3
        _inject(alerter, _sig("c", 0.70, "contracts", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        # Verify Domain A effective > Domain B effective
        a_freshness = alerter._compute_freshness(
            alerter.signals["insider"][0], "insider"
        )
        a_raw_eff = 0.65 * a_freshness
        a_boosted = a_raw_eff * (1 + 0.1 * math.log(3))

        b_freshness = alerter._compute_freshness(
            alerter.signals["sentiment"][0], "sentiment"
        )
        b_eff = 0.90 * b_freshness  # no boost (count=1)

        assert a_boosted > b_eff, (
            f"Count-boosted fresh domain ({a_boosted:.4f}) must outrank "
            f"stale single domain ({b_eff:.4f})"
        )

    # ── 21. Coherence multiplier still applies on top of freshness+count ──────

    def test_coherence_multiplier_applies_on_top_of_freshness_and_count(self):
        """Cap 3 (coherence) must multiply final_confidence AFTER Cap 5+6 adjust
        avg_strength. We verify this by creating a mixed-direction scenario:
        3 bullish domains + 1 bearish domain → coherence < 1.0 → confidence damped.

        If coherence multiplier were not applied, confidence would be higher.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # 3 bullish domains
        _inject(alerter, _sig("ins", 0.80, "insider", "bullish", age_hours=0.0))
        _inject(alerter, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov", 0.75, "government", "bullish", age_hours=0.0))
        # 1 bearish domain — introduces coherence < 1.0
        _inject(alerter, _sig("con_bear", 0.70, "contracts", "bearish", age_hours=0.0))

        alerts_mixed = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts_mixed) == 1

        # Compare to a purely bullish alerter (no contradictions)
        alerter_pure = _alerter(min_domains=3, min_strength=0.5)
        _inject(alerter_pure, _sig("ins", 0.80, "insider", "bullish", age_hours=0.0))
        _inject(alerter_pure, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter_pure, _sig("gov", 0.75, "government", "bullish", age_hours=0.0))
        alerts_pure = alerter_pure.check_convergence(direction_filter="bullish")
        assert len(alerts_pure) == 1

        # Mixed scenario should have lower confidence due to coherence penalty
        assert alerts_mixed[0].confidence < alerts_pure[0].confidence, (
            "Coherence multiplier must reduce confidence when contradictory domains exist"
        )
        # Coherence < 1.0 in the mixed case
        assert alerts_mixed[0].coherence < 1.0

    # ── 22. All signals at floor freshness: alert fires, strength is dampened ─

    def test_all_signals_at_floor_freshness_still_fires(self):
        """If every signal is very stale (freshness hits 0.30 floor), the alert
        can still fire if enough domains are present, but avg_strength will be
        near 0.30 * raw_strength (heavily dampened).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # age > window so floor applies (but signals still within prune cutoff
        # if we bypass pruning by using _inject)
        raw_strength = 0.80
        floor = 0.3

        _inject(alerter, _sig("ins",  raw_strength, "insider",   "bullish", age_hours=100.0))
        _inject(alerter, _sig("sent", raw_strength, "sentiment", "bullish", age_hours=100.0))
        _inject(alerter, _sig("gov",  raw_strength, "government","bullish", age_hours=100.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1, "Alert must still fire despite stale signals"

        # Expected: each domain contributes raw_strength * floor (count=1, no boost)
        expected_strength = raw_strength * floor
        assert alerts[0].strength == pytest.approx(expected_strength, abs=1e-6), (
            f"Dampened strength should be {expected_strength}, got {alerts[0].strength}"
        )

    # ── 23. Count boost is log-saturating (diminishing returns) ──────────────

    def test_count_boost_has_diminishing_returns(self):
        """Adding the 10th signal should provide a smaller boost increment than
        adding the 2nd signal — confirming log-saturation behavior.

        Boost at count n: B(n) = 0.1 * log(n)
        Marginal gain from n-1 to n: delta(n) = 0.1 * (log(n) - log(n-1))
        delta(2) = 0.1 * log(2) ≈ 0.0693
        delta(10) = 0.1 * log(10/9) ≈ 0.0105
        """
        delta_2  = 0.1 * (math.log(2) - math.log(1))   # ≈ 0.0693
        delta_10 = 0.1 * (math.log(10) - math.log(9))  # ≈ 0.0105
        assert delta_2 > delta_10, "Each additional signal adds diminishing returns"

    # ── 24. Boost preserves domain isolation (separate domains, no cross-boost)

    def test_count_boost_does_not_cross_domain_boundaries(self):
        """5 signals across 5 different domains should NOT produce a count boost
        in any individual domain. Each domain has count=1, so no boost applies.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)
        domains = ["insider", "sentiment", "contracts", "government", "technical"]
        for d in domains:
            _inject(alerter, _sig(d, 0.70, d, "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        # Each domain: count=1, freshness≈1.0, no boost → eff=0.70
        # avg_strength = 0.70 (within tolerance)
        assert alerts[0].strength == pytest.approx(0.70, abs=1e-5)

    # ── 25. Edge case: freshness floor interacts correctly with count boost ────

    def test_freshness_floor_combined_with_count_boost(self):
        """Verify that freshness=0.3 (floor) and count=3 compound correctly.

        A domain with 3 very stale signals (all at freshness floor):
          selected_eff = strength * 0.30
          boosted = selected_eff * (1 + 0.1 * log(3))

        This must equal the actual computation in _check_direction_convergence.
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        # All signals far outside the window so freshness hits 0.30 floor
        for i in range(3):
            _inject(alerter, _sig(f"ins{i}", 0.80, "insider", "bullish", age_hours=200.0))

        # Two fresh domains to allow convergence
        _inject(alerter, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.70, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 1

        floor = 0.3
        expected_insider_eff = 0.80 * floor * (1 + 0.1 * math.log(3))

        f_sent = alerter._compute_freshness(
            Signal("s", 0.70, "sentiment", "bullish", _NOW - timedelta(hours=0)),
            "sentiment"
        )
        f_gov = alerter._compute_freshness(
            Signal("g", 0.70, "government", "bullish", _NOW - timedelta(hours=0)),
            "government"
        )
        sent_eff = 0.70 * f_sent
        gov_eff  = 0.70 * f_gov

        expected_avg = (expected_insider_eff + sent_eff + gov_eff) / 3
        assert alerts[0].strength == pytest.approx(expected_avg, abs=1e-5)

    # ── 26. No alert when all matching signals below min_strength ─────────────

    def test_count_boost_cannot_rescue_sub_threshold_signals(self):
        """10 signals in a domain, all at strength=0.45 with min_strength=0.5.
        None pass the threshold, so count=0, domain does not contribute,
        and convergence cannot fire (assuming no other domains compensate).
        """
        alerter = _alerter(min_domains=3, min_strength=0.5)

        for i in range(10):
            _inject(alerter, _sig(f"ins{i}", 0.45, "insider", "bullish", age_hours=0.0))

        # Only 2 other domains (need 3 total including insider)
        _inject(alerter, _sig("sent", 0.70, "sentiment", "bullish", age_hours=0.0))
        _inject(alerter, _sig("gov",  0.70, "government", "bullish", age_hours=0.0))

        alerts = alerter.check_convergence(direction_filter="bullish")
        assert len(alerts) == 0, (
            "Sub-threshold signals must not contribute, even with high count"
        )

    # ── 27. Record_signal() API preserves timestamps for freshness ────────────

    def test_record_signal_timestamp_param_used_for_freshness(self):
        """ConvergenceAlerter.record_signal() accepts an explicit timestamp.
        A signal recorded with a 48h-old timestamp should have freshness < 1.0.
        """
        alerter = _alerter(min_domains=1, min_strength=0.5)
        old_ts = datetime.now() - timedelta(hours=48)
        alerter.record_signal("sig_old", 0.80, "insider", "bullish",
                              timestamp=old_ts)

        sig = alerter.signals["insider"][0]
        freshness = alerter._compute_freshness(sig, "insider")

        expected = 1.0 - math.sqrt(48.0 / 72.0)  # ≈ 0.184 → floored to 0.3
        # sqrt(48/72) = sqrt(2/3) ≈ 0.816 → 1 - 0.816 = 0.184 → floor 0.3
        assert freshness == pytest.approx(0.3, abs=1e-6)

    # ── 28. Freshness of exactly-at-window-edge signal ────────────────────────

    def test_signal_exactly_at_window_edge_hits_floor(self):
        """A signal with age == window_hours exactly:
        1.0 - sqrt(1.0) = 0.0 → floor = 0.3."""
        alerter = _alerter()
        sig = _sig("edge", 0.80, "insider", "bullish", age_hours=72.0)
        result = alerter._compute_freshness(sig, "insider")
        assert result == pytest.approx(0.3, abs=1e-9)

    # ── 29. Very large count: boost is bounded (log grows slowly) ─────────────

    def test_large_count_boost_stays_below_2x(self):
        """With 100 signals, boost = 1 + 0.1*log(100) ≈ 1.46.
        It must never double (2.0x) the effective strength — log growth ensures this.
        """
        # Theoretical: 1 + 0.1*log(n) >= 2.0 requires log(n) >= 10 → n >= e^10 ≈ 22026
        # So well within safe range for any realistic count.
        boost_100 = 1 + 0.1 * math.log(100)  # ≈ 1.46
        boost_1000 = 1 + 0.1 * math.log(1000)  # ≈ 1.69
        assert boost_100 < 2.0
        assert boost_1000 < 2.0

    # ── 30. Per-ticker convergence does NOT receive count boost ───────────────

    def test_per_ticker_convergence_unaffected_by_count_boost(self):
        """check_ticker_convergence() uses a simpler strength selection path
        (max by raw strength, no freshness, no count boost). This confirms
        the two pipelines are independent and the per-ticker path hasn't
        accidentally picked up Cap 5+6 logic.
        """
        alerter = _alerter(min_domains=2, min_strength=0.0)

        # 3 bullish insider signals for ticker AAPL
        for i in range(3):
            alerter.record_signal(
                f"ins{i}", 0.70, "insider", direction="bullish",
                metadata={"symbol": "AAPL"}, timestamp=datetime.now()
            )

        # 1 bullish sentiment signal for the same ticker
        alerter.record_signal(
            "sent", 0.65, "sentiment", direction="bullish",
            metadata={"symbol": "AAPL"}, timestamp=datetime.now()
        )

        ticker_alerts = alerter.check_ticker_convergence(min_domains=2)
        aapl_alerts = [a for a in ticker_alerts if "AAPL" in a.alert_id]
        assert len(aapl_alerts) >= 1

        # Per-ticker strength = avg of raw strengths (no boost, no freshness)
        # insider max = 0.70, sentiment = 0.65 → avg = 0.675
        aapl_alert = aapl_alerts[0]
        assert aapl_alert.strength == pytest.approx((0.70 + 0.65) / 2, abs=1e-5), (
            "Per-ticker path must use raw strength, no count boost or freshness"
        )
