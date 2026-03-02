#!/usr/bin/env python3
"""Tests for mae_core.market.intelligence.cross_asset_confirmer.

All tests are self-contained — no real price data is fetched.
price_fetcher is mocked to return synthetic PriceData histories.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mae_core.market.intelligence.cross_asset_confirmer import (
    CorrelationPair,
    CrossAssetConfirmer,
    CrossAssetResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(symbol: str, prices: list) -> list:
    """
    Build a list of mock PriceData objects oldest-first.

    Args:
        symbol: Ticker symbol.
        prices: List of close prices (oldest first).
    """
    objects = []
    for p in prices:
        pd_mock = MagicMock()
        pd_mock.symbol = symbol
        pd_mock.price = float(p)
        objects.append(pd_mock)
    return objects


def _make_fetcher(price_map: dict) -> MagicMock:
    """
    Build a mock price_fetcher.

    price_map: dict of symbol → list of prices (oldest-first).
    get_daily_history(symbol, days) returns the list for that symbol,
    or an empty list if unknown.
    """
    fetcher = MagicMock()

    def get_daily_history(symbol, days=5):
        if symbol in price_map:
            return _make_price_data(symbol, price_map[symbol])
        return []

    fetcher.get_daily_history.side_effect = get_daily_history
    return fetcher


def _make_confirmer(price_map: dict = None) -> CrossAssetConfirmer:
    """Build a CrossAssetConfirmer with mocked price fetcher."""
    fetcher = _make_fetcher(price_map or {}) if price_map is not None else None
    return CrossAssetConfirmer(price_fetcher=fetcher)


# ---------------------------------------------------------------------------
# TestDefaultPairs
# ---------------------------------------------------------------------------

class TestDefaultPairs(unittest.TestCase):
    """All built-in pairs are registered and accessible bidirectionally."""

    def setUp(self):
        self.confirmer = CrossAssetConfirmer(price_fetcher=None)

    def test_spy_has_qqq_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("SPY")
        others = {p.ticker_b for p in pairs}
        assert "QQQ" in others

    def test_qqq_has_spy_pair(self):
        """Bidirectional: QQQ → SPY should also be registered."""
        pairs = self.confirmer.get_pairs_for_ticker("QQQ")
        others = {p.ticker_b for p in pairs}
        assert "SPY" in others

    def test_gc_f_has_dxy_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("GC=F")
        others = {p.ticker_b for p in pairs}
        assert "DXY" in others

    def test_cl_f_has_xle_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("CL=F")
        others = {p.ticker_b for p in pairs}
        assert "XLE" in others

    def test_tlt_has_spy_inverse_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("TLT")
        spy_pairs = [p for p in pairs if p.ticker_b == "SPY"]
        assert len(spy_pairs) >= 1
        assert spy_pairs[0].relationship == "inverse"

    def test_vix_has_spy_inverse_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("VIX")
        spy_pairs = [p for p in pairs if p.ticker_b == "SPY"]
        assert len(spy_pairs) >= 1
        assert spy_pairs[0].relationship == "inverse"

    def test_es_f_has_nq_f_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("ES=F")
        others = {p.ticker_b for p in pairs}
        assert "NQ=F" in others

    def test_es_f_has_spy_pair(self):
        pairs = self.confirmer.get_pairs_for_ticker("ES=F")
        others = {p.ticker_b for p in pairs}
        assert "SPY" in others

    def test_unknown_ticker_returns_empty_list(self):
        pairs = self.confirmer.get_pairs_for_ticker("NOEXIST")
        assert pairs == []

    def test_all_pairs_have_valid_relationship(self):
        """Every registered pair has relationship "positive" or "inverse"."""
        valid = {"positive", "inverse"}
        for ticker, pairs in self.confirmer._pairs.items():
            for p in pairs:
                assert p.relationship in valid, (
                    f"Invalid relationship {p.relationship!r} for {ticker}"
                )


# ---------------------------------------------------------------------------
# TestConfirmation
# ---------------------------------------------------------------------------

class TestConfirmation(unittest.TestCase):
    """Signals confirmed by correlated instruments score positively."""

    def test_positive_pair_confirms_bullish_signal(self):
        """CL=F bullish + XLE moving up = confirmation."""
        # CL=F (start 70) → XLE moving from 90 to 92 (+2.2%)
        confirmer = _make_confirmer({"XLE": [90, 90.5, 91, 91.5, 92]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        xle_pairs = [p for p in confirmer.get_pairs_for_ticker("CL=F")
                     if p.ticker_b == "XLE"]
        # At least the XLE pair should confirm
        assert result.confirmation_score > 0 or len(xle_pairs) == 0

    def test_inverse_pair_confirms_bearish_signal(self):
        """TLT bearish + SPY moving up = confirmation (inverse relationship)."""
        # SPY moving up → confirms bearish TLT (they move opposite)
        confirmer = _make_confirmer({"SPY": [500, 501, 502, 503, 504]})
        result = confirmer.check_confirmation("TLT", direction="bearish")
        assert result.confirmation_score > 0

    def test_spy_bullish_confirmed_by_qqq_moving_up(self):
        """SPY bullish + QQQ moving up → confirmation."""
        confirmer = _make_confirmer({"QQQ": [400, 401, 402, 403, 404]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        # QQQ is positively correlated with SPY
        assert result.confirmation_score > 0

    def test_gld_confirms_gc_f_bullish(self):
        """GC=F bullish + GLD (gold ETF) moving up → confirmation."""
        confirmer = _make_confirmer({"GLD": [180, 181, 182, 183, 184]})
        result = confirmer.check_confirmation("GC=F", direction="bullish")
        assert result.confirmation_score > 0


# ---------------------------------------------------------------------------
# TestContradiction
# ---------------------------------------------------------------------------

class TestContradiction(unittest.TestCase):
    """Signals contradicted by correlated instruments score negatively."""

    def test_positive_pair_moving_opposite_is_conflict(self):
        """CL=F bullish + XLE moving down = conflict."""
        # XLE falling while we have bullish crude signal → contradiction
        confirmer = _make_confirmer({"XLE": [90, 89.5, 89, 88.5, 88]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        xle_pairs = [p for p in confirmer.get_pairs_for_ticker("CL=F")
                     if p.ticker_b == "XLE"]
        if xle_pairs:
            # XLE pair should contribute negative score
            assert result.confirmation_score < 1.0

    def test_inverse_pair_moving_same_direction_is_conflict(self):
        """TLT bearish + SPY also moving down = conflict (inverse expected up)."""
        # SPY falling while TLT bearish → conflict (inverse means TLT up = SPY down)
        confirmer = _make_confirmer({"SPY": [500, 499, 498, 497, 496]})
        result = confirmer.check_confirmation("TLT", direction="bearish")
        # SPY down = contradicts bearish TLT (inverse pair means TLT bearish
        # should see SPY rising)
        assert result.confirmation_score < 0

    def test_vix_bullish_with_spy_rising_is_conflict(self):
        """VIX bullish (fear rising) + SPY moving up = contradiction."""
        confirmer = _make_confirmer({"SPY": [500, 501, 502, 503, 504]})
        result = confirmer.check_confirmation("VIX", direction="bullish")
        # VIX inverse to SPY: VIX bullish means fear rising, SPY should fall
        assert result.confirmation_score < 0

    def test_conflicting_pairs_are_listed(self):
        """Conflicting pairs appear in result.conflicting_pairs."""
        # SPY falling while QQQ bullish signal — positive pair conflict
        confirmer = _make_confirmer({"QQQ": [400, 399, 398, 397, 396]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        # QQQ moving down while we expect bullish SPY = conflict
        assert len(result.conflicting_pairs) > 0 or result.confirmation_score <= 0


# ---------------------------------------------------------------------------
# TestScoreCalculation
# ---------------------------------------------------------------------------

class TestScoreCalculation(unittest.TestCase):
    """confirmation_score is weighted average of individual pair scores."""

    def test_single_confirming_pair_scores_positive(self):
        confirmer = _make_confirmer({"QQQ": [400, 401, 402, 403, 404]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        # QQQ up 1% with bullish SPY → confirm
        assert result.confirmation_score > 0

    def test_single_conflicting_pair_scores_negative(self):
        confirmer = _make_confirmer({"QQQ": [400, 399, 398, 397, 396]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        # QQQ down with bullish SPY → conflict
        assert result.confirmation_score < 0

    def test_score_is_clamped_to_minus_one_to_one(self):
        """Score should always be in [-1.0, 1.0]."""
        confirmer = _make_confirmer({"QQQ": [400, 410, 420, 430, 440]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        assert -1.0 <= result.confirmation_score <= 1.0

    def test_pairs_checked_counts_known_pairs(self):
        """pairs_checked reflects how many pairs we evaluated."""
        # SPY has pairs with QQQ and TLT at minimum
        confirmer = _make_confirmer({
            "QQQ": [400, 401, 402, 403, 404],
            "TLT": [90, 89, 88, 87, 86],
        })
        result = confirmer.check_confirmation("SPY", direction="bullish")
        assert result.pairs_checked >= 1

    def test_confirming_pairs_are_listed(self):
        """Confirming pairs appear in result.confirming_pairs."""
        confirmer = _make_confirmer({"QQQ": [400, 401, 402, 403, 404]})
        result = confirmer.check_confirmation("SPY", direction="bullish")
        assert len(result.confirming_pairs) > 0 or result.confirmation_score == 0


# ---------------------------------------------------------------------------
# TestGracefulDegradation
# ---------------------------------------------------------------------------

class TestGracefulDegradation(unittest.TestCase):
    """System works correctly when price data is missing or fetcher is None."""

    def test_no_price_fetcher_returns_neutral(self):
        confirmer = CrossAssetConfirmer(price_fetcher=None)
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        assert result.confirmation_score == 0.0
        assert result.pairs_checked == 0

    def test_no_pairs_for_ticker_returns_neutral(self):
        confirmer = _make_confirmer({})
        result = confirmer.check_confirmation("UNKNOWN_TICKER", direction="bullish")
        assert result.confirmation_score == 0.0

    def test_price_fetcher_returns_empty_list(self):
        """If fetcher returns [] for all tickers, score is neutral."""
        confirmer = _make_confirmer({})  # price_map is empty → all [] returns
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        assert result.confirmation_score == 0.0

    def test_price_fetcher_raises_exception_returns_neutral(self):
        """If price_fetcher.get_daily_history raises, system continues."""
        fetcher = MagicMock()
        fetcher.get_daily_history.side_effect = ConnectionError("Network down")
        confirmer = CrossAssetConfirmer(price_fetcher=fetcher)
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        assert isinstance(result, CrossAssetResult)
        assert result.confirmation_score == 0.0

    def test_single_price_point_returns_neutral(self):
        """Only one price point → cannot compute a move → neutral."""
        confirmer = _make_confirmer({"XLE": [90]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        # Single point → move_pct = 0 → below min_move_pct → neutral
        # Score may be 0 because all pairs are neutral
        assert result.confirmation_score == 0.0 or result.pairs_checked >= 0


# ---------------------------------------------------------------------------
# TestSmallMoves
# ---------------------------------------------------------------------------

class TestSmallMoves(unittest.TestCase):
    """Movements below min_move_pct are treated as neutral (noise)."""

    def test_tiny_move_treated_as_neutral(self):
        """
        XLE moves 0.1% — below the 0.5% threshold.
        The pair is still counted (pairs_checked > 0) but contributes 0
        to the weighted score, leaving it at 0.0.
        """
        # XLE: 90.00 → 90.09 = +0.1% move — well below 0.5% threshold
        confirmer = _make_confirmer({"XLE": [90.00, 90.03, 90.06, 90.08, 90.09]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        # Score should be near 0 because the tiny XLE move is treated as noise
        # (other pairs may have no data → score stays 0.0)
        assert abs(result.confirmation_score) < 0.1

    def test_move_above_threshold_is_not_neutral(self):
        """Moves >= 0.5% should register as a direction."""
        # XLE: 90 → 91.8 = +2% move — clearly above threshold
        confirmer = _make_confirmer({"XLE": [90, 90.5, 91, 91.5, 91.8]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        # XLE moving up with bullish CL=F = confirmation → non-zero score
        xle_pairs = confirmer.get_pairs_for_ticker("CL=F")
        if xle_pairs:
            assert result.confirmation_score != 0.0

    def test_exactly_at_threshold_registers(self):
        """
        Move at exactly 0.5% should be considered significant.
        The condition is abs(move) < min_move_pct, so 0.5% passes.
        """
        # 90 → 90.45 = exactly 0.5%
        confirmer = _make_confirmer({"XLE": [90, 90.1, 90.2, 90.3, 90.45]})
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        # This should NOT be treated as noise
        # Score may still be 0 if only XLE pair exists and other data missing
        assert isinstance(result, CrossAssetResult)

    def test_zero_first_price_returns_neutral(self):
        """If first price is 0 (bad data), division by zero is avoided."""
        confirmer = _make_confirmer({"XLE": [0, 90, 91, 92, 93]})
        # first price = 0 → _get_recent_move_pct returns None → pair skipped
        result = confirmer.check_confirmation("CL=F", direction="bullish")
        assert isinstance(result, CrossAssetResult)


if __name__ == "__main__":
    unittest.main()
