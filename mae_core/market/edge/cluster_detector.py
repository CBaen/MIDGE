#!/usr/bin/env python3
"""
cluster_detector.py - Detect insider buying clusters

When 3+ insiders buy the same stock within 30 days, this is a high-confidence
signal that outperforms single insider trades.

Research backing (from universal_vault_v2 predictions):
- [high] Cluster buys (3+ insiders within 30 days) produce higher returns
- [high] Open-market purchases (Code P) outperform other transaction types
- [high] 40%+ holdings increase indicates high conviction

Sub-modules:
  relationship_tracker.py — RelationshipTracker, store_cluster_signal,
                             scan_all_symbols (extracted for size)
"""

import uuid
import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

# InsiderRelationship lives in relationship_tracker to avoid circular imports.
from mae_core.market.edge.relationship_tracker import (  # noqa: F401
    InsiderRelationship, RelationshipTracker,
    store_cluster_signal, scan_all_symbols,
    QDRANT_URL, SIGNALS_COLLECTION, OLLAMA_URL,
)

logger = logging.getLogger(__name__)

ROLE_WEIGHTS = {
    "ceo": 3.0,
    "cfo": 3.0,
    "chief": 2.5,
    "president": 2.5,
    "officer": 2.0,
    "director": 1.5,
    "10%": 1.0,
    "owner": 1.0,
}


@dataclass
class InsiderInCluster:
    """Single insider's contribution to a cluster."""
    name: str
    role: str
    relationship: str
    trade_date: str
    shares: float
    value: float
    role_weight: float
    conviction_score: float


@dataclass
class ClusterSignal:
    """A cluster of insider buying activity."""
    cluster_id: str = ""
    symbol: str = ""
    insider_count: int = 0
    insiders: List[dict] = field(default_factory=list)
    window_days: int = 30
    total_value: float = 0.0
    weighted_score: float = 0.0
    avg_conviction: float = 0.0
    has_csuite: bool = False
    confidence: float = 0.70
    compression_score: float = 0.0
    detected_at: str = ""
    signal_source: str = "insider_cluster"
    decay_rate: float = 0.025

    def __post_init__(self):
        if not self.cluster_id:
            self.cluster_id = str(uuid.uuid4())
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        c_suite_note = " (including C-suite)" if self.has_csuite else ""
        compression_note = ""
        if self.compression_score >= 0.8:
            compression_note = " [TIGHT cluster]"
        elif self.compression_score >= 0.5:
            compression_note = " [moderate spread]"
        return (
            f"{self.insider_count} insiders bought {self.symbol} "
            f"within {self.window_days} days{c_suite_note}{compression_note}. "
            f"Total value: ${self.total_value:,.0f}. "
            f"Confidence: {self.confidence:.0%}"
        )


class ClusterDetector:
    """Detects clusters of insider buying activity.

    Queries Qdrant for recent Form 4 signals and identifies when
    multiple insiders are buying the same stock.
    """

    def __init__(self, qdrant_url: str = QDRANT_URL, collection: str = SIGNALS_COLLECTION):
        self.qdrant_url = qdrant_url
        self.collection = collection

    def find_clusters(
        self, symbol: str, days_back: int = 30, min_insiders: int = 3,
    ) -> List[ClusterSignal]:
        """Find insider buying clusters for a symbol."""
        trades = self._query_recent_trades(symbol, days_back)
        if not trades:
            return []

        filtered = self._filter_transactions(trades)
        if not filtered:
            return []

        by_insider = self._aggregate_by_insider(filtered)
        if len(by_insider) < min_insiders:
            return []

        cluster = self._generate_cluster_signal(symbol, by_insider, days_back)
        return [cluster] if cluster else []

    def _query_recent_trades(self, symbol: str, days_back: int) -> List[dict]:
        """Query Qdrant for recent Form 4 signals."""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        try:
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "signal_source", "match": {"value": "sec_form4"}},
                            {"key": "symbol", "match": {"value": symbol}},
                            {"key": "direction", "match": {"value": "bullish"}},
                        ]
                    },
                    "limit": 100,
                    "with_payload": True,
                },
                timeout=10,
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
            logger.error(f"[ClusterDetector] Error querying Qdrant: {e}")
            return []

    def _filter_transactions(self, trades: List[dict]) -> List[dict]:
        """Filter to valid open-market purchases only."""
        filtered = []

        for trade in trades:
            details = trade.get("details", {})

            if trade.get("direction") != "bullish":
                continue
            if details.get("is_10b5_1_plan", False):
                continue

            trans_code = details.get("transaction_code", "").upper()
            if trans_code in ["G", "F", "M"]:
                continue

            filtered.append(trade)

        return filtered

    def _aggregate_by_insider(self, trades: List[dict]) -> Dict[str, List[dict]]:
        """Group trades by unique insider name."""
        by_insider = defaultdict(list)
        for trade in trades:
            details = trade.get("details", {})
            filer_name = details.get("filer_name", "Unknown")
            by_insider[filer_name].append(trade)
        return dict(by_insider)

    def _calculate_role_weight(self, role: str, relationship: str) -> float:
        """Calculate weight based on insider's role."""
        combined = f"{role} {relationship}".lower()
        for key, weight in ROLE_WEIGHTS.items():
            if key in combined:
                return weight
        return 1.0

    def _calculate_conviction(self, trade: dict) -> float:
        """Calculate conviction score (0.0-1.0) based on trade significance."""
        details = trade.get("details", {})
        shares_after = details.get("shares_owned_after", 0)
        shares_traded = details.get("shares", 0)
        total_value = details.get("total_value", 0)

        if shares_after > 0 and shares_traded > 0:
            shares_before = shares_after - shares_traded
            if shares_before > 0:
                pct_increase = shares_traded / shares_before
                conviction = min(1.0, pct_increase / 2.0)
            else:
                conviction = 0.7
        else:
            if total_value >= 500000:
                conviction = 0.8
            elif total_value >= 100000:
                conviction = 0.6
            elif total_value >= 50000:
                conviction = 0.5
            else:
                conviction = 0.4

        return conviction

    def _is_csuite(self, role: str, relationship: str) -> bool:
        """Check if insider is C-suite."""
        combined = f"{role} {relationship}".lower()
        return any(title in combined for title in ["ceo", "cfo", "chief", "president"])

    def _generate_cluster_signal(
        self, symbol: str, by_insider: Dict[str, List[dict]], window_days: int,
    ) -> Optional[ClusterSignal]:
        """Generate a ClusterSignal from aggregated insider trades."""
        insiders_data = []
        total_value = 0.0
        total_weighted_score = 0.0
        total_conviction = 0.0
        has_csuite = False

        for insider_name, trades in by_insider.items():
            best_trade = max(trades, key=lambda t: t.get("details", {}).get("total_value", 0))
            details = best_trade.get("details", {})

            role = details.get("filer_title", "")
            relationship = details.get("filer_relationship", "")

            role_weight = self._calculate_role_weight(role, relationship)
            conviction = self._calculate_conviction(best_trade)
            value = details.get("total_value", 0)

            if self._is_csuite(role, relationship):
                has_csuite = True

            insiders_data.append({
                "name": insider_name,
                "role": role,
                "relationship": relationship,
                "trade_date": details.get("transaction_date", ""),
                "shares": details.get("shares", 0),
                "value": value,
                "role_weight": role_weight,
                "conviction": conviction,
            })

            total_value += value
            total_weighted_score += role_weight * conviction
            total_conviction += conviction

        insider_count = len(insiders_data)
        avg_conviction = total_conviction / insider_count if insider_count > 0 else 0

        trade_dates = []
        for ins in insiders_data:
            td = ins.get("trade_date", "")
            if td:
                try:
                    trade_dates.append(datetime.fromisoformat(td.replace("Z", "")))
                except (ValueError, TypeError):
                    pass

        compression_score = 0.0
        if len(trade_dates) >= 2:
            span_days = (max(trade_dates) - min(trade_dates)).total_seconds() / 86400
            compression_score = max(0.0, 1.0 - span_days / max(1, window_days))

        confidence = 0.70
        confidence += min(0.15, (insider_count - 3) * 0.05)
        if has_csuite:
            confidence += 0.05
        if avg_conviction > 0.4:
            confidence += 0.05
        confidence += compression_score * 0.10
        confidence = min(0.95, confidence)

        return ClusterSignal(
            symbol=symbol,
            insider_count=insider_count,
            insiders=insiders_data,
            window_days=window_days,
            total_value=total_value,
            weighted_score=total_weighted_score,
            avg_conviction=avg_conviction,
            has_csuite=has_csuite,
            confidence=confidence,
            compression_score=round(compression_score, 3),
        )


if __name__ == "__main__":
    import sys

    test_symbols = ["LMT", "MSFT", "AAPL", "NVDA"]

    if len(sys.argv) > 1 and sys.argv[1] == "relationships":
        print("Building insider relationship graph...")
        tracker = RelationshipTracker()

        relationships = tracker.build_multi_symbol_graph(test_symbols, days_back=365)

        if relationships:
            print(f"\nFound {len(relationships)} insider relationships:")
            for rel in relationships[:10]:
                print(f"  {rel.to_plain_language()}")
        else:
            print("  No relationships found (need Form 4 data in Qdrant)")

        if relationships:
            test_insider = relationships[0].insider_a
            print(f"\nWho trades with {test_insider}?")
            coordinated = tracker.find_coordinated_trades(test_insider)
            for rel in coordinated[:5]:
                other = rel.insider_b if rel.insider_a == test_insider else rel.insider_a
                print(f"  {other}: {rel.correlation_score:.0%} correlation")

    else:
        print("Scanning for insider clusters...")
        detector = ClusterDetector()

        for symbol in test_symbols:
            print(f"\n{symbol}:")
            clusters = detector.find_clusters(symbol)
            if clusters:
                for cluster in clusters:
                    print(f"  {cluster.to_plain_language()}")
            else:
                print("  No clusters found")

        print("\nTip: Run with 'relationships' argument to test relationship tracking")
