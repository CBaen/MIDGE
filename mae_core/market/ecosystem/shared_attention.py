"""Shared ecosystem whiteboard for MIDGE's independent processes.

Any process can read and any process can contribute. Not a message queue —
a shared state snapshot so each process knows what the others have found.

Writes are atomic (tmp → rename) and safe for concurrent processes.
Reads handle missing or corrupt files gracefully.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("midge.ecosystem")

_DEFAULT_PATH = Path("data/market/ecosystem_attention.json")

_EMPTY_STATE: dict[str, Any] = {
    "last_updated": "",
    "hot_tickers": {},
    "active_cascades": [],
    "regime": {"current": "unknown", "changed_at": "", "vestibular_triggered": False},
    "causal_discoveries": [],
    "cross_market_alerts": [],
    "crypto_sentiment": {},
    "process_heartbeats": {},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class SharedAttention:
    """Shared whiteboard for inter-process ecosystem awareness.

    Each process reads the full state, merges its update, and writes back
    atomically. No locking — last write wins per section, which is fine
    for coarse-grained state snapshots.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self) -> dict[str, Any]:
        """Return the full shared state. Returns empty skeleton on any error."""
        try:
            if not self._path.exists():
                return dict(_EMPTY_STATE)
            text = self._path.read_text(encoding="utf-8")
            if not text.strip():
                return dict(_EMPTY_STATE)
            return json.loads(text)
        except Exception:
            logger.debug("SharedAttention read failed — returning empty state", exc_info=True)
            return dict(_EMPTY_STATE)

    # ── Write helpers ─────────────────────────────────────────────────────────

    def _write(self, state: dict[str, Any]) -> None:
        """Atomically write state to disk (tmp → rename)."""
        state["last_updated"] = _now_iso()
        try:
            parent = self._path.parent
            fd, tmp_path = tempfile.mkstemp(dir=parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, default=str)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            os.replace(tmp_path, self._path)
        except Exception:
            logger.debug("SharedAttention write failed", exc_info=True)

    # ── Public update methods ─────────────────────────────────────────────────

    def update_hot_ticker(
        self, ticker: str, reason: str, confidence: float, domains: int
    ) -> None:
        """Add or refresh a hot ticker entry."""
        state = self.read()
        hot = state.setdefault("hot_tickers", {})
        hot[ticker] = {
            "reason": reason,
            "confidence": round(float(confidence), 4),
            "domains": int(domains),
            "updated": _now_iso(),
        }
        self._write(state)

    def update_cascade(self, chain_desc: str, confirmed: int, total: int) -> None:
        """Add or refresh an active cascade entry (matched by chain description)."""
        state = self.read()
        cascades: list[dict] = state.setdefault("active_cascades", [])
        for entry in cascades:
            if entry.get("chain") == chain_desc:
                entry["confirmed_links"] = confirmed
                entry["total_links"] = total
                entry["updated"] = _now_iso()
                self._write(state)
                return
        cascades.append({
            "chain": chain_desc,
            "confirmed_links": confirmed,
            "total_links": total,
            "updated": _now_iso(),
        })
        # Keep list bounded
        if len(cascades) > 50:
            cascades[:] = cascades[-50:]
        self._write(state)

    def update_regime(self, regime: str, vestibular_triggered: bool = False) -> None:
        """Update current market regime."""
        state = self.read()
        prev = state.get("regime", {}).get("current", "unknown")
        changed = prev != regime
        state["regime"] = {
            "current": regime,
            "changed_at": _now_iso() if changed else state.get("regime", {}).get("changed_at", _now_iso()),
            "vestibular_triggered": vestibular_triggered,
        }
        self._write(state)

    def add_causal_discovery(
        self, cause: str, effect: str, lag_days: float, confidence: float
    ) -> None:
        """Append a new causal discovery (deduplicates by cause+effect pair)."""
        state = self.read()
        discoveries: list[dict] = state.setdefault("causal_discoveries", [])
        for entry in discoveries:
            if entry.get("cause") == cause and entry.get("effect") == effect:
                entry["lag_days"] = round(float(lag_days), 2)
                entry["confidence"] = round(float(confidence), 4)
                entry["updated"] = _now_iso()
                self._write(state)
                return
        discoveries.append({
            "cause": cause,
            "effect": effect,
            "lag_days": round(float(lag_days), 2),
            "confidence": round(float(confidence), 4),
            "updated": _now_iso(),
        })
        if len(discoveries) > 100:
            discoveries[:] = discoveries[-100:]
        self._write(state)

    def add_cross_market_alert(
        self, alert_type: str, tickers: list[str], strength: float
    ) -> None:
        """Append a cross-market anomaly alert."""
        state = self.read()
        alerts: list[dict] = state.setdefault("cross_market_alerts", [])
        alerts.append({
            "type": alert_type,
            "tickers": list(tickers),
            "strength": round(float(strength), 4),
            "updated": _now_iso(),
        })
        if len(alerts) > 50:
            alerts[:] = alerts[-50:]
        self._write(state)

    def update_crypto_sentiment(
        self, value: int | float, classification: str, trend: str
    ) -> None:
        """Update the crypto fear & greed snapshot."""
        state = self.read()
        state["crypto_sentiment"] = {
            "fear_greed": int(value),
            "classification": classification,
            "trend": trend,
            "updated": _now_iso(),
        }
        self._write(state)

    def heartbeat(self, process_name: str, **kwargs: Any) -> None:
        """Record that a process is alive, with optional metadata."""
        state = self.read()
        beats: dict = state.setdefault("process_heartbeats", {})
        entry: dict = {"updated": _now_iso()}
        entry.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})
        beats[process_name] = entry
        self._write(state)

    # ── Query methods ─────────────────────────────────────────────────────────

    def get_hot_tickers(self, min_confidence: float = 0.5) -> list[tuple[str, dict]]:
        """Return list of (ticker, info) for tickers above min_confidence threshold."""
        state = self.read()
        return [
            (ticker, info)
            for ticker, info in state.get("hot_tickers", {}).items()
            if info.get("confidence", 0.0) >= min_confidence
        ]

    def get_stale_processes(self, max_age_seconds: int = 600) -> list[str]:
        """Return process names that haven't sent a heartbeat within max_age_seconds."""
        state = self.read()
        now = datetime.now(timezone.utc)
        stale = []
        for name, info in state.get("process_heartbeats", {}).items():
            updated_str = info.get("updated", "")
            if not updated_str:
                stale.append(name)
                continue
            try:
                updated_dt = datetime.fromisoformat(updated_str)
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
                age = (now - updated_dt).total_seconds()
                if age > max_age_seconds:
                    stale.append(name)
            except Exception:
                stale.append(name)
        return stale
