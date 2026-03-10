"""
relationship_tracker.py - Insider trading relationship tracker + storage helpers

RelationshipTracker class, store_cluster_signal(), and scan_all_symbols().
Extracted from cluster_detector.py to keep each file under 500 lines.
"""

import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional

from mae_core.market.edge.cluster_detector import (
    ClusterSignal, InsiderRelationship,
    QDRANT_URL, SIGNALS_COLLECTION, OLLAMA_URL,
)

logger = logging.getLogger(__name__)


class RelationshipTracker:
    """Tracks insider trading relationships over time.

    Builds a graph of which insiders trade together (within 48h windows).
    When insider A trades, predicts which other insiders will follow.
    """

    def __init__(self, qdrant_url: str = QDRANT_URL, collection: str = SIGNALS_COLLECTION):
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.relationships: Dict[str, InsiderRelationship] = {}

    def _relationship_key(self, a: str, b: str) -> str:
        """Generate consistent key regardless of order."""
        return "|".join(sorted([a, b]))

    def track_insider_relationships(
        self,
        symbol: str,
        days_back: int = 365,
        time_window_hours: int = 48,
    ) -> List[InsiderRelationship]:
        """Build relationship graph for a symbol.

        Finds all insider pairs who traded within time_window_hours of each other.
        """
        trades = self._query_all_trades(symbol, days_back)

        if len(trades) < 2:
            return []

        trades.sort(key=lambda t: t.get("details", {}).get("transaction_date", ""))

        for i, trade_a in enumerate(trades):
            details_a = trade_a.get("details", {})
            name_a = details_a.get("filer_name", "")
            date_a = details_a.get("transaction_date", "")

            if not name_a or not date_a:
                continue

            try:
                dt_a = datetime.fromisoformat(date_a.replace("Z", ""))
            except Exception:
                continue

            for trade_b in trades[i + 1:]:
                details_b = trade_b.get("details", {})
                name_b = details_b.get("filer_name", "")
                date_b = details_b.get("transaction_date", "")

                if not name_b or not date_b or name_a == name_b:
                    continue

                try:
                    dt_b = datetime.fromisoformat(date_b.replace("Z", ""))
                except Exception:
                    continue

                delta = abs((dt_b - dt_a).total_seconds() / 3600)
                if delta <= time_window_hours:
                    self._record_relationship(name_a, name_b, symbol, delta)

        return list(self.relationships.values())

    def _query_all_trades(self, symbol: str, days_back: int) -> List[dict]:
        """Query all Form 4 trades (buys and sells) for relationship analysis."""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        try:
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "signal_source", "match": {"value": "sec_form4"}},
                            {"key": "symbol", "match": {"value": symbol}},
                        ]
                    },
                    "limit": 500,
                    "with_payload": True,
                },
                timeout=30,
            )

            if response.status_code != 200:
                return []

            points = response.json().get("result", {}).get("points", [])

            trades = []
            for point in points:
                payload = point.get("payload", {})
                timestamp = payload.get("timestamp", "")
                if timestamp >= cutoff:
                    trades.append(payload)

            return trades

        except Exception as e:
            logger.error(f"[RelationshipTracker] Query error: {e}")
            return []

    def _record_relationship(self, name_a: str, name_b: str, symbol: str, delta_hours: float):
        """Record or update a relationship between two insiders."""
        key = self._relationship_key(name_a, name_b)

        if key not in self.relationships:
            self.relationships[key] = InsiderRelationship(
                insider_a=name_a,
                insider_b=name_b,
                symbols_traded=[symbol],
                trades_together=1,
                avg_time_delta_hours=delta_hours,
                correlation_score=0.5,
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat(),
            )
        else:
            rel = self.relationships[key]
            if symbol not in rel.symbols_traded:
                rel.symbols_traded.append(symbol)
            rel.trades_together += 1
            rel.avg_time_delta_hours = (
                (rel.avg_time_delta_hours * (rel.trades_together - 1) + delta_hours)
                / rel.trades_together
            )
            rel.last_seen = datetime.now().isoformat()
            rel.correlation_score = min(0.95, 0.5 + (rel.trades_together * 0.1))

    def find_coordinated_trades(
        self,
        insider_name: str,
        min_correlation: float = 0.6,
    ) -> List[InsiderRelationship]:
        """Find insiders who reliably trade with the given insider."""
        matches = []

        for rel in self.relationships.values():
            if insider_name in (rel.insider_a, rel.insider_b):
                if rel.correlation_score >= min_correlation:
                    matches.append(rel)

        matches.sort(key=lambda r: r.correlation_score, reverse=True)
        return matches

    def build_multi_symbol_graph(
        self,
        symbols: List[str],
        days_back: int = 365,
    ) -> List[InsiderRelationship]:
        """Build relationship graph across multiple symbols.

        Insiders who trade together across MULTIPLE stocks have stronger signal.
        """
        for symbol in symbols:
            self.track_insider_relationships(symbol, days_back)

        all_rels = list(self.relationships.values())
        all_rels.sort(key=lambda r: r.trades_together, reverse=True)
        return all_rels


def store_cluster_signal(cluster: ClusterSignal, qdrant_url: str = QDRANT_URL) -> bool:
    """Store a cluster signal to Qdrant."""
    import hashlib

    text = cluster.to_plain_language()
    insider_names = ", ".join(i["name"] for i in cluster.insiders[:5])
    text += f" Insiders: {insider_names}"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": text},
            timeout=30,
        )
        if response.status_code != 200:
            logger.warning("[ClusterDetector] Could not get embedding")
            return False
        embedding = response.json().get("embedding")
    except Exception as e:
        logger.error(f"[ClusterDetector] Embedding error: {e}")
        return False

    payload = asdict(cluster)
    payload["text"] = text
    payload["direction"] = "bullish"

    deterministic_id = int(hashlib.md5(cluster.cluster_id.encode()).hexdigest(), 16) % (10**18)

    try:
        response = requests.put(
            f"{qdrant_url}/collections/{SIGNALS_COLLECTION}/points",
            json={
                "points": [{
                    "id": deterministic_id,
                    "vector": embedding,
                    "payload": payload,
                }]
            },
            timeout=10,
        )
        success = response.status_code in (200, 201)
        if success:
            logger.info(f"[ClusterDetector] Stored cluster signal: {cluster.to_plain_language()}")
        return success
    except Exception as e:
        logger.error(f"[ClusterDetector] Storage error: {e}")
        return False


def scan_all_symbols(
    symbols: List[str], days_back: int = 30, min_insiders: int = 3,
) -> List[ClusterSignal]:
    """Scan multiple symbols for clusters. Convenience for batch processing."""
    from mae_core.market.edge.cluster_detector import ClusterDetector
    detector = ClusterDetector()
    all_clusters = []

    for symbol in symbols:
        clusters = detector.find_clusters(symbol, days_back, min_insiders)
        all_clusters.extend(clusters)

    return all_clusters
