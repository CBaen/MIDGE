"""Streaming Anomaly Detector — PySAD RRCF for composite signal vectors.

Uses PySAD's RobustRandomCutForest (RRCF) to score composite feature vectors
in real time. Each call to update() feeds one observation and returns an
anomaly score.

Input design: a 4-element feature vector captures the most informative
market conditions that have no single-source home —
  [price_change, volume_ratio, sentiment_score, vix_level]

These four inputs are designed to complement VelocityDetector (which tracks
single-signal rates of change). RRCF is unsupervised and tree-based so it
requires zero labelled data.

Graceful degradation: if pysad or rrcf are not installed, the detector
returns a score of 0.0 and is_anomalous() always returns False.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from pysad.models import RobustRandomCutForest as _RRCF

    HAS_PYSAD = True
except ImportError:
    HAS_PYSAD = False
    logger.warning("pysad/rrcf not installed — StreamingAnomalyDetector runs in no-op mode")

_DEFAULT_N_TREES = 40
_DEFAULT_SHINGLE = 4   # how many consecutive observations each tree sees
_DEFAULT_TREE_SIZE = 256
_WARMUP_OBS = 20       # observations before scores are meaningful


@dataclass
class AnomalyEvent:
    """A scored anomaly observation."""

    score: float           # 0-1, higher = more anomalous
    feature_vector: List[float]
    timestamp: datetime = field(default_factory=datetime.now)
    exceeded_threshold: bool = False


class StreamingAnomalyDetector:
    """Unsupervised streaming anomaly detector over composite feature vectors.

    Constructor args:
        n_features: expected vector length (default 4)
        threshold:  score above which is_anomalous() returns True (default 0.8)

    Usage:
        sad = StreamingAnomalyDetector()
        score = sad.update([0.02, 1.5, 0.3, 18.0])
        if sad.is_anomalous([0.02, 1.5, 0.3, 18.0]):
            ...
    """

    def __init__(self, n_features: int = 4, threshold: float = 0.8) -> None:
        self.n_features = n_features
        self.threshold = threshold
        self._model: Optional[object] = None
        self._observation_count = 0
        self._last_score: float = 0.0
        self._score_history: list = []  # last 200 scores for normalisation
        self._max_history = 200
        self._total_anomalies = 0

        if HAS_PYSAD:
            try:
                self._model = _RRCF(
                    num_trees=_DEFAULT_N_TREES,
                    shingle_size=_DEFAULT_SHINGLE,
                    tree_size=_DEFAULT_TREE_SIZE,
                )
            except Exception:
                logger.debug("RRCF init failed", exc_info=True)
                self._model = None

    def _normalise(self, raw_score: float) -> float:
        """Normalise raw RRCF score to [0, 1] using rolling max."""
        self._score_history.append(raw_score)
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)
        max_seen = max(self._score_history) if self._score_history else 1.0
        if max_seen <= 0:
            return 0.0
        return min(1.0, raw_score / max_seen)

    def _validate_vector(self, feature_vector: Sequence[float]) -> Optional[List[float]]:
        """Validate and pad/truncate feature vector to expected length."""
        try:
            vec = [float(v) for v in feature_vector]
        except (TypeError, ValueError):
            return None

        # Replace NaN/inf with 0
        vec = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in vec]

        # Pad or truncate to n_features
        if len(vec) < self.n_features:
            vec = vec + [0.0] * (self.n_features - len(vec))
        elif len(vec) > self.n_features:
            vec = vec[: self.n_features]

        return vec

    def update(self, feature_vector: Sequence[float]) -> float:
        """Feed one observation. Returns normalised anomaly score [0, 1].

        Args:
            feature_vector: sequence of floats, e.g. [price_change, volume_ratio,
                            sentiment_score, vix_level]. Length is padded/truncated
                            to n_features automatically.

        Returns:
            float in [0, 1]. Scores above threshold indicate anomalies.
        """
        vec = self._validate_vector(feature_vector)
        if vec is None:
            return self._last_score

        self._observation_count += 1

        if not HAS_PYSAD or self._model is None:
            return 0.0

        try:
            arr = np.array(vec, dtype=float)
            self._model.fit_partial(arr)
            raw_score = float(self._model.score_partial(arr))
        except Exception:
            logger.debug("RRCF score_partial failed", exc_info=True)
            return self._last_score

        # Raw RRCF scores are displacement-based (unbounded). Normalise to [0,1].
        score = self._normalise(raw_score)
        self._last_score = score

        # Only flag anomalies after warmup
        if self._observation_count >= _WARMUP_OBS and score >= self.threshold:
            self._total_anomalies += 1
            logger.info(
                "StreamingAnomaly: anomaly detected (score=%.3f threshold=%.2f obs=%d)",
                score, self.threshold, self._observation_count,
            )

        return score

    def is_anomalous(self, feature_vector: Sequence[float]) -> bool:
        """Update the model and return True if the observation is anomalous.

        Convenience wrapper around update() that applies the configured threshold.
        Scores during warmup (< _WARMUP_OBS observations) always return False.
        """
        score = self.update(feature_vector)
        return (
            self._observation_count >= _WARMUP_OBS
            and score >= self.threshold
        )

    def get_statistics(self) -> dict:
        """For HolonProxy.sense() delegation."""
        return {
            "observation_count": self._observation_count,
            "last_score": round(self._last_score, 4),
            "total_anomalies": self._total_anomalies,
            "threshold": self.threshold,
            "has_pysad": HAS_PYSAD,
            "in_warmup": self._observation_count < _WARMUP_OBS,
        }
