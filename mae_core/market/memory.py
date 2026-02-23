"""Persistent signal storage via Qdrant.

Abstracts the raw HTTP Qdrant calls used by edge detectors into a single
SignalMemory class. Stores MarketSignal objects as searchable vectors
with full payload metadata.

Uses raw HTTP (no qdrant-client dependency) matching the existing pattern
in cluster_detector.py, filing_time_analyzer.py, and contract_predictor.py.
"""

import hashlib
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from mae_core.market.signal import MarketSignal

logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
SIGNALS_COLLECTION = "midge_signals"

# Vector dimensions: we encode key numeric fields into a fixed-size vector
# for Qdrant storage. The actual search is payload-filtered (not vector-based).
VECTOR_SIZE = 8


def _direction_code(direction: str) -> float:
    """Encode direction as float for vector embedding."""
    return {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(direction, 0.0)


# Stable domain -> float mapping for vector encoding
_DOMAIN_CODES = {
    "insider": 0.1, "congress": 0.2, "contracts": 0.3,
    "government": 0.4, "events": 0.5, "institutional": 0.6,
    "institutional_synthesis": 0.7, "sentiment": 0.8,
    "news": 0.9, "technical": 1.0, "crypto": 0.15,
    "volume": 0.25, "reddit": 0.35, "fundamentals": 0.45,
    "macro": 0.55,
}


def _domain_code(domain: str) -> float:
    """Encode domain as float for vector embedding."""
    return _DOMAIN_CODES.get(domain, 0.0)


def _signal_to_vector(signal: MarketSignal) -> List[float]:
    """Convert MarketSignal to a fixed-size numeric vector."""
    return [
        signal.strength,
        signal.confidence,
        signal.velocity,
        _direction_code(signal.direction),
        _domain_code(signal.domain),
        signal.decay_rate,
        signal.outcome_window_days / 30.0,  # normalize to ~1.0
        min(1.0, signal.strength * signal.confidence),  # composite score
    ]


def _signal_to_payload(signal: MarketSignal) -> dict:
    """Convert MarketSignal to Qdrant payload dict."""
    return {
        "signal_id": signal.signal_id,
        "source": signal.source,
        "symbol": signal.symbol,
        "asset_class": signal.asset_class,
        "domain": signal.domain,
        "direction": signal.direction,
        "strength": signal.strength,
        "confidence": signal.confidence,
        "decay_rate": signal.decay_rate,
        "velocity": signal.velocity,
        "timestamp": signal.timestamp.isoformat(),
        "received_at": signal.received_at.isoformat(),
        "outcome_symbol": signal.outcome_symbol,
        "outcome_window_days": signal.outcome_window_days,
        "raw_id": signal.raw_id,
        "raw_type": signal.raw_type,
        "metadata": signal.metadata,
        "signal_source": signal.source,  # compat with cluster_detector filter key
    }


def _point_id(signal: MarketSignal) -> int:
    """Deterministic point ID from signal_id (matches cluster_detector pattern)."""
    return int(hashlib.md5(signal.signal_id.encode()).hexdigest(), 16) % (10**18)


class SignalMemory:
    """Qdrant-backed persistent storage for MarketSignals.

    Gracefully degrades when Qdrant is unavailable — callers should
    check is_available() or handle store failures.
    """

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = SIGNALS_COLLECTION,
    ):
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection = collection
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            resp = requests.get(f"{self.qdrant_url}/collections", timeout=3)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def ensure_collection(self) -> bool:
        """Create the collection if it doesn't exist. Returns True on success."""
        try:
            # Check if collection already exists
            resp = requests.get(
                f"{self.qdrant_url}/collections/{self.collection}",
                timeout=5,
            )
            if resp.status_code == 200:
                return True

            # Create it
            resp = requests.put(
                f"{self.qdrant_url}/collections/{self.collection}",
                json={
                    "vectors": {
                        "size": VECTOR_SIZE,
                        "distance": "Cosine",
                    }
                },
                timeout=10,
            )
            success = resp.status_code in (200, 201)
            if success:
                logger.info(f"[SignalMemory] Created collection '{self.collection}'")
            else:
                logger.warning(
                    f"[SignalMemory] Failed to create collection: {resp.status_code} {resp.text}"
                )
            return success
        except Exception as e:
            logger.error(f"[SignalMemory] ensure_collection error: {e}")
            return False

    def store_signal(self, signal: MarketSignal) -> bool:
        """Store a single MarketSignal to Qdrant. Returns True on success."""
        return self.store_signals([signal])

    def store_signals(self, signals: List[MarketSignal]) -> bool:
        """Batch-store MarketSignals to Qdrant. Returns True on success."""
        if not signals:
            return True

        points = []
        for sig in signals:
            points.append({
                "id": _point_id(sig),
                "vector": _signal_to_vector(sig),
                "payload": _signal_to_payload(sig),
            })

        try:
            resp = requests.put(
                f"{self.qdrant_url}/collections/{self.collection}/points",
                json={"points": points},
                timeout=30,
            )
            success = resp.status_code in (200, 201)
            if success:
                logger.info(f"[SignalMemory] Stored {len(points)} signals")
            else:
                logger.warning(
                    f"[SignalMemory] Store failed: {resp.status_code} {resp.text[:200]}"
                )
            return success
        except Exception as e:
            logger.error(f"[SignalMemory] store_signals error: {e}")
            return False

    def query_recent(
        self,
        hours: int = 72,
        domain: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> List[dict]:
        """Query signals from the last N hours, optionally filtered."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        must_filters = []
        if domain:
            must_filters.append({"key": "domain", "match": {"value": domain}})
        if symbol:
            must_filters.append({"key": "symbol", "match": {"value": symbol}})

        try:
            resp = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                json={
                    "filter": {"must": must_filters} if must_filters else None,
                    "limit": limit,
                    "with_payload": True,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            points = resp.json().get("result", {}).get("points", [])

            # Client-side date filter (Qdrant doesn't support range on strings)
            results = []
            for point in points:
                payload = point.get("payload", {})
                ts = payload.get("timestamp", "")
                if ts >= cutoff:
                    results.append(payload)

            return results
        except Exception as e:
            logger.error(f"[SignalMemory] query_recent error: {e}")
            return []

    def query_by_symbol(self, symbol: str, limit: int = 50) -> List[dict]:
        """Query all signals for a specific symbol."""
        try:
            resp = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "symbol", "match": {"value": symbol}},
                        ]
                    },
                    "limit": limit,
                    "with_payload": True,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                return []

            return [
                p.get("payload", {})
                for p in resp.json().get("result", {}).get("points", [])
            ]
        except Exception as e:
            logger.error(f"[SignalMemory] query_by_symbol error: {e}")
            return []

    def count(self) -> int:
        """Return total number of stored signals."""
        try:
            resp = requests.get(
                f"{self.qdrant_url}/collections/{self.collection}",
                timeout=5,
            )
            if resp.status_code == 200:
                return resp.json().get("result", {}).get("points_count", 0)
        except Exception:
            pass
        return 0
