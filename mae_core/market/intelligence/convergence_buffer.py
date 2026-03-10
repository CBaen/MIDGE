"""ConvergenceBufferMixin — signal buffer persistence, record_signal, regime cache.

Used by ConvergenceConfidenceMixin as a mixin. Expects the following instance
attributes:
    self.signals, self.convergence_window, self._domain_windows,
    self._regime_classifier, self._cached_regime,
    self._alert_counter, self._last_alert_times
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from mae_core.market.intelligence.convergence_models import Signal

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


class ConvergenceBufferMixin:
    """Signal recording, buffer persistence, and regime caching."""

    @staticmethod
    def _json_default(obj):
        """JSON serializer for objects not handled by default encoder."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return obj.total_seconds()
        if isinstance(obj, set):
            return list(obj)
        return str(obj)

    def save_signal_buffer(self) -> int:
        """Persist in-memory signal buffer and alerter state to disk.

        Saves only signals that are still within their domain convergence
        window — expired signals are not worth carrying across restarts.
        Uses atomic writes (write to .tmp then rename) for thread safety,
        matching the pattern used by ThompsonSampler.

        Returns the number of signals saved (0 if buffer was empty).
        """
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        total_in_memory = sum(len(v) for v in self.signals.values())
        logger.info(
            "Signal buffer save starting: %d signals across %d domains in memory",
            total_in_memory,
            len(self.signals),
        )

        # Build serialisable buffer — only keep live (non-expired) signals.
        buffer: Dict[str, list] = {}
        for domain, sigs in list(self.signals.items()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            live = []
            for s in sigs:
                try:
                    if s.timestamp >= cutoff:
                        live.append({
                            "signal_id": s.signal_id,
                            "strength": s.strength,
                            "domain": s.domain,
                            "direction": s.direction,
                            "timestamp": s.timestamp.isoformat(),
                            "metadata": s.metadata,
                            "velocity": s.velocity,
                            "confidence": s.confidence,
                            "source": s.source,
                        })
                except Exception:
                    logger.debug("Skipping non-serializable signal in %s", domain)
            if live:
                buffer[domain] = live

        # Atomically write signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        tmp_buf = buf_path.with_suffix(".tmp")
        try:
            tmp_buf.write_text(json.dumps(buffer, indent=2, default=self._json_default))
            os.replace(tmp_buf, buf_path)
        except Exception:
            logger.warning("Failed to save signal buffer", exc_info=True)
            if tmp_buf.exists():
                tmp_buf.unlink(missing_ok=True)
            return 0

        # Atomically write alerter state (counter + dedup times).
        state = {
            "alert_counter": self._alert_counter,
            "last_alert_times": {
                k: (v.isoformat() if isinstance(v, datetime) else str(v))
                for k, v in self._last_alert_times.items()
            },
        }
        state_path = _DATA_DIR / "alerter_state.json"
        tmp_state = state_path.with_suffix(".tmp")
        try:
            tmp_state.write_text(json.dumps(state, indent=2))
            os.replace(tmp_state, state_path)
        except Exception:
            logger.warning("Failed to save alerter state", exc_info=True)
            if tmp_state.exists():
                tmp_state.unlink(missing_ok=True)
            return 0

        total_signals = sum(len(v) for v in buffer.values())
        logger.info(
            "Signal buffer saved: %d signals across %d domains",
            total_signals,
            len(buffer),
        )
        return total_signals

    def load_signal_buffer(self) -> int:
        """Restore signal buffer and alerter state from disk.

        Signals older than their domain convergence window are discarded
        on load — only signals that are still relevant are restored.
        Missing or corrupt files are handled gracefully.

        Returns the number of signals restored (0 if none found).
        """
        now = datetime.now()

        # Restore signal buffer.
        buf_path = _DATA_DIR / "signal_buffer.json"
        if buf_path.exists():
            try:
                raw: Dict[str, list] = json.loads(buf_path.read_text())
                loaded = 0
                for domain, entries in raw.items():
                    window = self._domain_windows.get(domain, self.convergence_window)
                    cutoff = now - window
                    for entry in entries:
                        ts = datetime.fromisoformat(entry["timestamp"])
                        if ts < cutoff:
                            continue  # expired — skip
                        self.signals[domain].append(
                            Signal(
                                signal_id=entry["signal_id"],
                                strength=entry["strength"],
                                domain=entry["domain"],
                                direction=entry["direction"],
                                timestamp=ts,
                                metadata=entry.get("metadata", {}),
                                velocity=entry.get("velocity", 0.0),
                                confidence=entry.get("confidence", 0.5),
                                source=entry.get("source", ""),
                            )
                        )
                        loaded += 1
                domain_count = len(self.signals)
                logger.info(
                    "Signal buffer loaded: %d signals across %d domains",
                    loaded,
                    domain_count,
                )
            except Exception:
                logger.warning("Failed to load signal buffer — starting fresh", exc_info=True)
                loaded = 0
        else:
            logger.debug("No signal buffer file found — starting with empty buffer")
            loaded = 0

        # Restore alerter state.
        state_path = _DATA_DIR / "alerter_state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self._alert_counter = int(state.get("alert_counter", 0))
                raw_times = state.get("last_alert_times", {})
                self._last_alert_times = {
                    k: datetime.fromisoformat(v) for k, v in raw_times.items()
                }
                logger.debug(
                    "Alerter state loaded: counter=%d, dedup_entries=%d",
                    self._alert_counter,
                    len(self._last_alert_times),
                )
            except Exception:
                logger.warning("Failed to load alerter state — using defaults", exc_info=True)
        else:
            logger.debug("No alerter state file found — using defaults")

        return loaded

    def record_signal(
        self,
        signal_id: str,
        strength: float,
        domain: str,
        direction: str = "neutral",
        confidence: float = 0.5,
        velocity: float = 0.0,
        timestamp: datetime = None,
        metadata: dict = None,
        source: str = "",
    ):
        """Record a signal observation.

        Args:
            signal_id: Unique signal identifier
            strength: Signal strength 0-1 (higher = stronger)
            domain: Domain category (insider, crypto, sentiment, etc.)
            direction: bullish, bearish, or neutral
            confidence: Reliability estimate 0-1
            velocity: Rate of change (from VelocityDetector)
            timestamp: When observed
            metadata: Additional context
            source: Original source type (e.g. "sec_form4") for Thompson lookup
        """
        strength = max(0.0, min(1.0, strength))
        confidence = max(0.0, min(1.0, confidence))
        if direction not in ("bullish", "bearish", "neutral"):
            direction = "neutral"

        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        signal = Signal(
            signal_id=signal_id,
            strength=strength,
            domain=domain,
            direction=direction,
            timestamp=timestamp,
            metadata=metadata,
            velocity=velocity,
            confidence=confidence,
            source=source,
        )

        self.signals[domain].append(signal)
        self._prune_old_signals()

    def _prune_old_signals(self):
        """Remove signals outside the convergence window.

        Each domain may have its own extended window (self._domain_windows).
        Domains not in the dict use the global convergence_window (72h default).
        Domains whose signal list empties after pruning are removed entirely to
        keep self.signals clean.
        """
        now = datetime.now()
        for domain in list(self.signals.keys()):
            window = self._domain_windows.get(domain, self.convergence_window)
            cutoff = now - window
            self.signals[domain] = [
                s for s in self.signals[domain]
                if s.timestamp >= cutoff
            ]
            if not self.signals[domain]:
                del self.signals[domain]

    def _get_regime(self) -> str:
        """Get current market regime with 60s cache to avoid repeated calls."""
        if self._regime_classifier is None:
            return "default"
        now = time.monotonic()
        regime, cached_at = self._cached_regime
        if now - cached_at < 60.0:
            return regime
        try:
            regime = self._regime_classifier.classify()
        except Exception:
            regime = "default"
        self._cached_regime = (regime, now)
        return regime
