#!/usr/bin/env python3
"""
cluster_detector.py - Detect insider buying clusters

When 3+ insiders buy the same stock within 30 days, this is a high-confidence
signal that outperforms single insider trades.

Research backing (from universal_vault_v2 predictions):
- [high] Cluster buys (3+ insiders within 30 days) produce higher returns
- [high] Open-market purchases (Code P) outperform other transaction types
- [high] 40%+ holdings increase indicates high conviction
"""

import uuid
import logging
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

QDRANT_URL = "http://localhost:6333"
SIGNALS_COLLECTION = "midge_signals"
OLLAMA_URL = "http://localhost:11434"

# Role weights for insider hierarchy
# CEO/CFO > Directors > 10% Owners
ROLE_WEIGHTS = {
    "ceo": 3.0,
    "cfo": 3.0,
    "chief": 2.5,  # Other C-suite
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
    conviction_score: float  # % of holdings this trade represents


@dataclass
class InsiderRelationship:
    """
    Tracks coordinated trading between two insiders.

    If A and B consistently trade within 48 hours across multiple stocks,
    when A trades, we can PREDICT B will follow.
    """
    insider_a: str
    insider_b: str
    symbols_traded: List[str] = field(default_factory=list)  # Stocks they both traded
    trades_together: int = 0  # Number of coordinated trade events
    avg_time_delta_hours: float = 0.0  # Average time between their trades
    correlation_score: float = 0.0  # How reliably B follows A (0-1)
    first_seen: str = ""
    last_seen: str = ""

    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now().isoformat()
        if not self.last_seen:
            self.last_seen = datetime.now().isoformat()

    def to_plain_language(self) -> str:
        """Format for dashboard."""
        return (
            f"{self.insider_a} and {self.insider_b} traded together "
            f"{self.trades_together} times across {len(self.symbols_traded)} stocks. "
            f"Avg lag: {self.avg_time_delta_hours:.1f}h. "
            f"Correlation: {self.correlation_score:.0%}"
        )


@dataclass
class ClusterSignal:
    """A cluster of insider buying activity."""
    cluster_id: str = ""
    symbol: str = ""
    insider_count: int = 0
    insiders: List[dict] = field(default_factory=list)
    window_days: int = 30
    total_value: float = 0.0
    weighted_score: float = 0.0  # Sum of (role_weight * conviction)
    avg_conviction: float = 0.0
    has_csuite: bool = False
    confidence: float = 0.70
    compression_score: float = 0.0  # 1.0 = all trades within 48h, 0.0 = spread across full window
    detected_at: str = ""
    signal_source: str = "insider_cluster"
    decay_rate: float = 0.025  # ~28 day half-life (clusters persist longer, Alldredge 2019)

    def __post_init__(self):
        if not self.cluster_id:
            self.cluster_id = str(uuid.uuid4())
        if not self.detected_at:
            self.detected_at = datetime.now().isoformat()

    def to_plain_language(self) -> str:
        """Format for dashboard display."""
        c_suite_note = " (including C-suite)" if self.has_csuite else ""
        return (
            f"{self.insider_count} insiders bought {self.symbol} "
            f"within {self.window_days} days{c_suite_note}. "
            f"Total value: ${self.total_value:,.0f}. "
            f"Confidence: {self.confidence:.0%}"
        )


class ClusterDetector:
    """
    Detects clusters of insider buying activity.

    Queries Qdrant for recent Form 4 signals and identifies when
    multiple insiders are buying the same stock.
    """

    def __init__(self, qdrant_url: str = QDRANT_URL, collection: str = SIGNALS_COLLECTION):
        self.qdrant_url = qdrant_url
        self.collection = collection

    def find_clusters(
        self,
        symbol: str,
        days_back: int = 30,
        min_insiders: int = 3
    ) -> List[ClusterSignal]:
        """
        Find insider buying clusters for a symbol.

        Args:
            symbol: Stock ticker to analyze
            days_back: Time window to search (default 30 days)
            min_insiders: Minimum unique insiders for a cluster (default 3)

        Returns:
            List of ClusterSignal objects (usually 0 or 1)
        """
        # Query recent bullish insider signals
        trades = self._query_recent_trades(symbol, days_back)

        if not trades:
            return []

        # Filter to valid open-market purchases
        filtered = self._filter_transactions(trades)

        if not filtered:
            return []

        # Aggregate by unique insider
        by_insider = self._aggregate_by_insider(filtered)

        # Check if we have enough unique insiders
        if len(by_insider) < min_insiders:
            return []

        # Generate cluster signal
        cluster = self._generate_cluster_signal(symbol, by_insider, days_back)

        return [cluster] if cluster else []

    def _query_recent_trades(self, symbol: str, days_back: int) -> List[dict]:
        """Query Qdrant for recent Form 4 signals."""
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        try:
            # Scroll through all matching points
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection}/points/scroll",
                json={
                    "filter": {
                        "must": [
                            {"key": "signal_source", "match": {"value": "sec_form4"}},
                            {"key": "symbol", "match": {"value": symbol}},
                            {"key": "direction", "match": {"value": "bullish"}}
                        ]
                    },
                    "limit": 100,
                    "with_payload": True
                },
                timeout=10
            )

            if response.status_code != 200:
                return []

            points = response.json().get("result", {}).get("points", [])

            # Filter by date (Qdrant doesn't support date range filters easily)
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
        """
        Filter to valid open-market purchases only.

        Excludes:
        - Sales (disposition)
        - Gifts
        - Option exercises (when detectable)
        - Rule 10b5-1 plan trades (when detectable)
        """
        filtered = []

        for trade in trades:
            details = trade.get("details", {})

            # Only bullish direction (already filtered in query but double-check)
            if trade.get("direction") != "bullish":
                continue

            # Check for 10b5-1 plan indicator (if available in details)
            if details.get("is_10b5_1_plan", False):
                continue

            # Check transaction code if available
            trans_code = details.get("transaction_code", "").upper()
            if trans_code in ["G", "F", "M"]:  # Gift, tax withholding, options
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
        """
        Calculate weight based on insider's role.

        CEO/CFO = 3.0x
        Other C-suite = 2.5x
        Officers = 2.0x
        Directors = 1.5x
        10% Owners = 1.0x
        """
        combined = f"{role} {relationship}".lower()

        for key, weight in ROLE_WEIGHTS.items():
            if key in combined:
                return weight

        # Default weight
        return 1.0

    def _calculate_conviction(self, trade: dict) -> float:
        """
        Calculate conviction score based on trade significance.

        Factors:
        - Percentage of holdings this trade represents
        - Absolute dollar value

        Returns 0.0 to 1.0
        """
        details = trade.get("details", {})

        shares_after = details.get("shares_owned_after", 0)
        shares_traded = details.get("shares", 0)
        total_value = details.get("total_value", 0)

        # If we have holdings data, calculate % increase
        if shares_after > 0 and shares_traded > 0:
            shares_before = shares_after - shares_traded
            if shares_before > 0:
                pct_increase = shares_traded / shares_before
                # 40%+ increase = high conviction (0.8+)
                # Tripling (200%+) = very high conviction (1.0)
                conviction = min(1.0, pct_increase / 2.0)
            else:
                # New position - moderately high conviction
                conviction = 0.7
        else:
            # Fall back to dollar value heuristic
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
        self,
        symbol: str,
        by_insider: Dict[str, List[dict]],
        window_days: int
    ) -> Optional[ClusterSignal]:
        """Generate a ClusterSignal from aggregated insider trades."""

        insiders_data = []
        total_value = 0.0
        total_weighted_score = 0.0
        total_conviction = 0.0
        has_csuite = False

        for insider_name, trades in by_insider.items():
            # Use the most recent/largest trade for this insider
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
                "conviction": conviction
            })

            total_value += value
            total_weighted_score += role_weight * conviction
            total_conviction += conviction

        insider_count = len(insiders_data)
        avg_conviction = total_conviction / insider_count if insider_count > 0 else 0

        # Calculate confidence
        # Base: 0.70 for 3 insiders
        # +0.05 per additional insider (up to 0.85)
        # +0.05 if C-suite present
        # +0.05 if avg conviction > 40%
        confidence = 0.70
        confidence += min(0.15, (insider_count - 3) * 0.05)  # Extra insiders
        if has_csuite:
            confidence += 0.05
        if avg_conviction > 0.4:
            confidence += 0.05
        confidence = min(0.95, confidence)  # Cap at 95%

        return ClusterSignal(
            symbol=symbol,
            insider_count=insider_count,
            insiders=insiders_data,
            window_days=window_days,
            total_value=total_value,
            weighted_score=total_weighted_score,
            avg_conviction=avg_conviction,
            has_csuite=has_csuite,
            confidence=confidence
        )


class RelationshipTracker:
    """
    Tracks insider trading relationships over time.

    Builds a graph of which insiders trade together (within 48h windows).
    When insider A trades, predicts which other insiders will follow.
    """

    def __init__(self, qdrant_url: str = QDRANT_URL, collection: str = SIGNALS_COLLECTION):
        self.qdrant_url = qdrant_url
        self.collection = collection
        self.relationships: Dict[str, InsiderRelationship] = {}  # key = "A|B"

    def _relationship_key(self, a: str, b: str) -> str:
        """Generate consistent key regardless of order."""
        return "|".join(sorted([a, b]))

    def track_insider_relationships(
        self,
        symbol: str,
        days_back: int = 365,
        time_window_hours: int = 48
    ) -> List[InsiderRelationship]:
        """
        Build relationship graph for a symbol.

        Finds all insider pairs who traded within time_window_hours of each other.

        Args:
            symbol: Stock ticker to analyze
            days_back: Historical lookback period
            time_window_hours: Max hours between trades to count as "together"

        Returns:
            List of InsiderRelationship objects
        """
        # Query all insider trades for this symbol
        trades = self._query_all_trades(symbol, days_back)

        if len(trades) < 2:
            return []

        # Sort by date
        trades.sort(key=lambda t: t.get("details", {}).get("transaction_date", ""))

        # Find pairs trading within the window
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

                # Check if within window
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
                            {"key": "symbol", "match": {"value": symbol}}
                        ]
                    },
                    "limit": 500,
                    "with_payload": True
                },
                timeout=30
            )

            if response.status_code != 200:
                return []

            points = response.json().get("result", {}).get("points", [])

            # Filter by date
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
                correlation_score=0.5,  # Initial score
                first_seen=datetime.now().isoformat(),
                last_seen=datetime.now().isoformat()
            )
        else:
            rel = self.relationships[key]
            # Update existing relationship
            if symbol not in rel.symbols_traded:
                rel.symbols_traded.append(symbol)
            rel.trades_together += 1
            # Running average of time delta
            rel.avg_time_delta_hours = (
                (rel.avg_time_delta_hours * (rel.trades_together - 1) + delta_hours)
                / rel.trades_together
            )
            rel.last_seen = datetime.now().isoformat()
            # Increase correlation with more co-trades
            rel.correlation_score = min(0.95, 0.5 + (rel.trades_together * 0.1))

    def find_coordinated_trades(
        self,
        insider_name: str,
        min_correlation: float = 0.6
    ) -> List[InsiderRelationship]:
        """
        Find insiders who reliably trade with the given insider.

        When insider_name trades, these are the people likely to follow.

        Args:
            insider_name: Name of the insider who just traded
            min_correlation: Minimum correlation score to include

        Returns:
            List of relationships sorted by correlation (highest first)
        """
        matches = []

        for rel in self.relationships.values():
            if insider_name in (rel.insider_a, rel.insider_b):
                if rel.correlation_score >= min_correlation:
                    matches.append(rel)

        # Sort by correlation descending
        matches.sort(key=lambda r: r.correlation_score, reverse=True)
        return matches

    def build_multi_symbol_graph(
        self,
        symbols: List[str],
        days_back: int = 365
    ) -> List[InsiderRelationship]:
        """
        Build relationship graph across multiple symbols.

        Insiders who trade together across MULTIPLE stocks have stronger signal.
        """
        for symbol in symbols:
            self.track_insider_relationships(symbol, days_back)

        # Sort by trades_together (strongest relationships first)
        all_rels = list(self.relationships.values())
        all_rels.sort(key=lambda r: r.trades_together, reverse=True)
        return all_rels


def store_cluster_signal(cluster: ClusterSignal, qdrant_url: str = QDRANT_URL) -> bool:
    """Store a cluster signal to Qdrant."""
    import hashlib

    # Create searchable text
    text = cluster.to_plain_language()
    insider_names = ", ".join(i["name"] for i in cluster.insiders[:5])
    text += f" Insiders: {insider_names}"

    # Get embedding
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": text},
            timeout=30
        )
        if response.status_code != 200:
            logger.warning("[ClusterDetector] Could not get embedding")
            return False
        embedding = response.json().get("embedding")
    except Exception as e:
        logger.error(f"[ClusterDetector] Embedding error: {e}")
        return False

    # Build payload
    payload = asdict(cluster)
    payload["text"] = text
    payload["direction"] = "bullish"

    # Deterministic ID from cluster_id using MD5 (hash() is process-randomized)
    deterministic_id = int(hashlib.md5(cluster.cluster_id.encode()).hexdigest(), 16) % (10**18)

    # Store to Qdrant
    try:
        response = requests.put(
            f"{qdrant_url}/collections/{SIGNALS_COLLECTION}/points",
            json={
                "points": [{
                    "id": deterministic_id,
                    "vector": embedding,
                    "payload": payload
                }]
            },
            timeout=10
        )
        success = response.status_code in (200, 201)
        if success:
            logger.info(f"[ClusterDetector] Stored cluster signal: {cluster.to_plain_language()}")
        return success
    except Exception as e:
        logger.error(f"[ClusterDetector] Storage error: {e}")
        return False


def scan_all_symbols(symbols: List[str], days_back: int = 30, min_insiders: int = 3) -> List[ClusterSignal]:
    """
    Scan multiple symbols for clusters.

    Convenience function for batch processing.
    """
    detector = ClusterDetector()
    all_clusters = []

    for symbol in symbols:
        clusters = detector.find_clusters(symbol, days_back, min_insiders)
        all_clusters.extend(clusters)

    return all_clusters


if __name__ == "__main__":
    import sys

    test_symbols = ["LMT", "MSFT", "AAPL", "NVDA"]

    # Check if we're testing relationships
    if len(sys.argv) > 1 and sys.argv[1] == "relationships":
        print("Building insider relationship graph...")
        tracker = RelationshipTracker()

        relationships = tracker.build_multi_symbol_graph(test_symbols, days_back=365)

        if relationships:
            print(f"\nFound {len(relationships)} insider relationships:")
            for rel in relationships[:10]:  # Top 10
                print(f"  {rel.to_plain_language()}")
        else:
            print("  No relationships found (need Form 4 data in Qdrant)")

        # Test prediction
        if relationships:
            test_insider = relationships[0].insider_a
            print(f"\nWho trades with {test_insider}?")
            coordinated = tracker.find_coordinated_trades(test_insider)
            for rel in coordinated[:5]:
                other = rel.insider_b if rel.insider_a == test_insider else rel.insider_a
                print(f"  {other}: {rel.correlation_score:.0%} correlation")

    else:
        # Default: cluster detection
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
        print("  python cluster_detector.py relationships")
