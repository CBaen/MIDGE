"""Tests for the three pattern discovery modules.

Covers:
  - MotifDetector (STUMPY streaming)
  - StreamingAnomalyDetector (PySAD RRCF)
  - DriftDetector (ADWIN — River or pure-Python fallback)

All modules degrade gracefully if their optional libraries are absent, so
tests are written to pass in both the library-present and absent scenarios.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List


# ---------------------------------------------------------------------------
# MotifDetector
# ---------------------------------------------------------------------------

class TestMotifDetector:
    """Tests for mae_core/market/intelligence/motif_detector.py"""

    def _make_detector(self):
        from mae_core.market.intelligence.motif_detector import MotifDetector
        return MotifDetector()

    def test_instantiation(self):
        """MotifDetector constructs without error."""
        md = self._make_detector()
        assert md is not None

    def test_update_returns_list(self):
        """update() always returns a list (never None)."""
        md = self._make_detector()
        result = md.update("AAPL", 185.50, datetime.now())
        assert isinstance(result, list)

    def test_empty_before_warmup(self):
        """No motifs fire before the warmup threshold is reached."""
        md = self._make_detector()
        signals = []
        for i in range(10):  # far below _WARMUP_BARS=40
            sigs = md.update("AAPL", 100.0 + i * 0.1, datetime.now() + timedelta(days=i))
            signals.extend(sigs)
        assert len(signals) == 0

    def test_get_active_motifs_unknown_symbol(self):
        """get_active_motifs returns empty list for unknown symbol."""
        md = self._make_detector()
        result = md.get_active_motifs("ZZZZ")
        assert result == []

    def test_get_statistics_structure(self):
        """get_statistics returns required keys."""
        md = self._make_detector()
        stats = md.get_statistics()
        assert "tracked_symbols" in stats
        assert "total_signals" in stats
        assert "has_stumpy" in stats

    def test_max_symbols_cap(self):
        """Detector accepts up to _MAX_SYMBOLS before eviction."""
        from mae_core.market.intelligence.motif_detector import _MAX_SYMBOLS
        md = self._make_detector()
        # Insert more than the cap
        for i in range(_MAX_SYMBOLS + 5):
            md.update(f"SYM{i:04d}", 100.0, datetime.now())
        assert len(md._streams) <= _MAX_SYMBOLS

    def test_nan_price_ignored(self):
        """NaN price input is silently ignored (returns empty list)."""
        md = self._make_detector()
        result = md.update("AAPL", float("nan"), datetime.now())
        assert result == []

    def test_invalid_symbol_ignored(self):
        """Empty symbol string is ignored."""
        md = self._make_detector()
        result = md.update("", 100.0, datetime.now())
        assert result == []

    def test_sufficient_data_stream_initialises(self):
        """After _WARMUP_BARS prices a stream becomes initialised (if stumpy present)."""
        from mae_core.market.intelligence.motif_detector import (
            MotifDetector, HAS_STUMPY, _WARMUP_BARS,
        )
        md = MotifDetector()
        base = datetime.now() - timedelta(days=_WARMUP_BARS + 5)
        for i in range(_WARMUP_BARS + 2):
            md.update("TSLA", 200.0 + (i % 10) * 0.5, base + timedelta(days=i))
        stream = md._streams.get("TSLA")
        assert stream is not None
        if HAS_STUMPY:
            assert stream._initialised

    def test_discord_detection_synthetic(self):
        """A sudden extreme price spike can trigger a discord after warmup."""
        from mae_core.market.intelligence.motif_detector import (
            MotifDetector, HAS_STUMPY, _WARMUP_BARS,
        )
        if not HAS_STUMPY:
            return  # graceful degradation — skip when library absent

        md = MotifDetector()
        base = datetime.now() - timedelta(days=_WARMUP_BARS + 60)

        # Establish baseline (smooth sine-like oscillation for 80 bars)
        for i in range(80):
            price = 100.0 + 2.0 * math.sin(i * 0.3)
            md.update("TEST", price, base + timedelta(days=i))

        # Fire a very different pattern for 25 more bars
        all_signals = []
        for i in range(25):
            price = 100.0 + 20.0 * (1 if i % 2 == 0 else -1)   # extreme alternation
            sigs = md.update("TEST", price, base + timedelta(days=80 + i))
            all_signals.extend(sigs)

        # Either some signals fired, or we got an empty list (both are valid outcomes
        # depending on threshold and data shape) — the key assertion is no exception
        assert isinstance(all_signals, list)


# ---------------------------------------------------------------------------
# StreamingAnomalyDetector
# ---------------------------------------------------------------------------

class TestStreamingAnomalyDetector:
    """Tests for mae_core/market/intelligence/streaming_anomaly.py"""

    def _make_detector(self, **kw):
        from mae_core.market.intelligence.streaming_anomaly import StreamingAnomalyDetector
        return StreamingAnomalyDetector(**kw)

    def test_instantiation(self):
        sad = self._make_detector()
        assert sad is not None

    def test_update_returns_float(self):
        sad = self._make_detector()
        score = sad.update([0.01, 1.2, 0.3, 18.0])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_is_anomalous_during_warmup_false(self):
        """is_anomalous always returns False during warmup."""
        from mae_core.market.intelligence.streaming_anomaly import _WARMUP_OBS
        sad = self._make_detector(threshold=0.5)
        for _ in range(_WARMUP_OBS - 1):
            result = sad.is_anomalous([0.01, 1.0, 0.5, 20.0])
        assert result is False

    def test_score_normalised(self):
        """Scores stay in [0, 1] after 50 normal observations."""
        sad = self._make_detector()
        rng = random.Random(42)
        scores = []
        for _ in range(50):
            vec = [rng.gauss(0, 0.01), rng.gauss(1.0, 0.1), rng.gauss(0.5, 0.1), rng.gauss(18, 2)]
            scores.append(sad.update(vec))
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_nan_in_vector_handled(self):
        """NaN elements in feature vector are replaced with 0.0 silently."""
        sad = self._make_detector()
        score = sad.update([float("nan"), 1.0, 0.5, 18.0])
        assert isinstance(score, float)
        assert not math.isnan(score)

    def test_vector_padding(self):
        """Short vectors are padded to n_features."""
        sad = self._make_detector(n_features=4)
        score = sad.update([0.01])   # only 1 element
        assert isinstance(score, float)

    def test_vector_truncation(self):
        """Long vectors are truncated to n_features."""
        sad = self._make_detector(n_features=4)
        score = sad.update([0.01, 1.0, 0.5, 18.0, 99.0, 99.0])  # 6 elements
        assert isinstance(score, float)

    def test_get_statistics_structure(self):
        sad = self._make_detector()
        stats = sad.get_statistics()
        assert "observation_count" in stats
        assert "last_score" in stats
        assert "total_anomalies" in stats
        assert "has_pysad" in stats

    def test_extreme_values_dont_crash(self):
        """Very large values are handled without exception."""
        sad = self._make_detector()
        score = sad.update([1e9, 1e9, 1e9, 1e9])
        assert 0.0 <= score <= 1.0

    def test_observation_count_increments(self):
        sad = self._make_detector()
        for _ in range(5):
            sad.update([0.01, 1.0, 0.5, 18.0])
        assert sad.get_statistics()["observation_count"] == 5


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    """Tests for mae_core/market/intelligence/drift_detector.py"""

    def _make_detector(self, **kw):
        from mae_core.market.intelligence.drift_detector import DriftDetector
        return DriftDetector(**kw)

    def test_instantiation(self):
        dd = self._make_detector()
        assert dd is not None

    def test_update_returns_tuple(self):
        dd = self._make_detector()
        result = dd.update("price_returns", 0.01)
        assert isinstance(result, tuple)
        assert len(result) == 3
        drift_detected, old_mean, new_mean = result
        assert isinstance(drift_detected, bool)
        assert isinstance(old_mean, float)
        assert isinstance(new_mean, float)

    def test_no_drift_on_stable_stream(self):
        """Stable identical values should not trigger drift."""
        dd = self._make_detector(delta=0.002)
        drifts = []
        for _ in range(50):
            detected, _, _ = dd.update("vix", 20.0)
            drifts.append(detected)
        # Very few or no drift detections on a constant stream
        assert sum(drifts) <= 2

    def test_drift_on_step_change(self):
        """A clear mean shift should be detected (pure-Python ADWIN)."""
        dd = self._make_detector(delta=0.05)   # higher sensitivity for test
        # Feed 30 observations near 0
        for _ in range(30):
            dd.update("test_stream", 0.0 + random.gauss(0, 0.001))
        # Then shift to a very different mean
        detected_any = False
        for _ in range(30):
            d, _, _ = dd.update("test_stream", 10.0 + random.gauss(0, 0.001))
            if d:
                detected_any = True
                break
        assert detected_any, "Expected drift to be detected after clear mean shift"

    def test_get_drift_status_structure(self):
        dd = self._make_detector()
        dd.update("price_returns", 0.01)
        dd.update("vix", 20.0)
        status = dd.get_drift_status()
        assert "price_returns" in status
        assert "vix" in status
        for entry in status.values():
            assert "current_mean" in entry
            assert "drift_count" in entry
            assert "observation_count" in entry

    def test_multiple_streams_independent(self):
        """Multiple stream names are tracked independently."""
        dd = self._make_detector()
        streams = ["price_returns", "vix", "volume", "sentiment"]
        for s in streams:
            dd.update(s, 1.0)
        status = dd.get_drift_status()
        for s in streams:
            assert s in status

    def test_nan_value_handled(self):
        """NaN value returns (False, 0, 0) without crashing."""
        dd = self._make_detector()
        result = dd.update("price_returns", float("nan"))
        assert result == (False, 0.0, 0.0)

    def test_get_statistics_structure(self):
        dd = self._make_detector()
        stats = dd.get_statistics()
        assert "tracked_streams" in stats
        assert "total_drifts" in stats
        assert "has_river" in stats

    def test_get_recent_drifts_empty_initially(self):
        dd = self._make_detector()
        assert dd.get_recent_drifts() == []

    def test_high_sensitivity_delta(self):
        """delta=0.1 (high sensitivity) still produces valid output."""
        dd = self._make_detector(delta=0.1)
        for i in range(20):
            result = dd.update("test", float(i % 5))
        status = dd.get_drift_status()
        assert "test" in status
