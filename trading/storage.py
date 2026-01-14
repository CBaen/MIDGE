#!/usr/bin/env python3
"""
storage.py - Qdrant vector storage with decay metadata

Uses MIDGE's HTTP pattern for cross-platform compatibility.
Enhanced with:
- Signal decay rates per type
- Source reliability scoring
- Confidence tracking
- Chunk indexing for retrieval
"""

import uuid
import math
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION = "trading_research"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

# Signal decay rates (per day) - empirically derived from research
DECAY_RATES = {
    "news": 0.5,           # Half-life: ~1.4 days (fast)
    "sentiment": 0.3,      # Half-life: ~2.3 days
    "technical": 0.1,      # Half-life: ~7 days
    "insider": 0.05,       # Half-life: ~14 days
    "institutional": 0.03, # Half-life: ~23 days (13F filings)
    "politician": 0.04,    # Half-life: ~17 days (STOCK Act delay)
    "contract": 0.02,      # Half-life: ~35 days (slow pricing)
    "research": 0.01,      # Half-life: ~69 days (knowledge base)
}

# Source reliability scores (0-1)
SOURCE_RELIABILITY = {
    "sec_edgar": 0.95,
    "13f_filing": 0.90,
    "form_4": 0.90,
    "capitol_trades": 0.85,
    "unusual_whales": 0.80,
    "polygon": 0.95,
    "twitter_verified": 0.60,
    "reddit": 0.30,
    "stocktwits": 0.50,
    "gemini": 0.75,
    "deepseek": 0.70,
    "unknown": 0.50,
}


@dataclass
class PredictionPayload:
    """Schema for tracking predictions and their outcomes."""
    # Identity
    prediction_id: str = ""
    symbol: str = ""
    direction: str = ""  # "bullish" or "bearish"

    # Prediction details
    confidence: float = 0.5  # 0.0 - 1.0
    entry_price: float = 0.0  # Price when predicted
    target_price: float = 0.0  # Optional target
    stop_loss: float = 0.0  # Optional stop

    # Reasoning
    reasoning: str = ""  # Why this prediction
    contributing_signals: list = None  # Signal IDs that informed this

    # Timing
    predicted_at: str = ""
    outcome_due: str = ""  # When to check result
    timeframe: str = "1d"  # 1h, 4h, 1d, 1w

    # Outcome (filled later)
    outcome_recorded: bool = False
    outcome_price: float = 0.0
    outcome_date: str = ""
    was_correct: bool = False
    return_pct: float = 0.0

    # Source tracking
    prediction_source: str = "midge"  # midge | manual | technical | fundamental

    def __post_init__(self):
        if not self.prediction_id:
            self.prediction_id = str(uuid.uuid4())
        if not self.predicted_at:
            self.predicted_at = datetime.now().isoformat()
        if self.contributing_signals is None:
            self.contributing_signals = []
        if not self.outcome_due:
            # Default: check outcome in 1 day
            outcome_date = datetime.now() + timedelta(days=1)
            self.outcome_due = outcome_date.isoformat()


@dataclass
class SignalPayload:
    """Enhanced payload schema for trading signals."""
    # Identity
    topic: str
    symbol: Optional[str] = None
    timestamp: str = ""
    data_type: str = "research"  # signal | research | pattern | event

    # Signal Classification
    signal_source: str = "research"  # technical | fundamental | sentiment | insider | institutional | contract
    signal_type: str = ""
    signal_strength: float = 0.0  # -1.0 to 1.0

    # Temporal Decay
    decay_rate: float = 0.01
    effective_until: str = ""

    # Confidence & Source
    source_reliability: float = 0.70
    confidence: float = 0.5
    historical_accuracy: float = 0.5

    # Cross-Reference IDs
    person_id: Optional[str] = None
    company_id: Optional[str] = None
    contract_id: Optional[str] = None
    pattern_id: Optional[str] = None

    # Content
    domain: str = ""
    subdomain: str = ""
    tags: list = None
    content: str = ""
    question: str = ""
    sources: list = None

    # Chunk Info
    chunk_index: int = 0
    total_chunks: int = 1

    # Embedding
    embedding_model: str = EMBEDDING_MODEL
    embedding_dims: int = EMBEDDING_DIM

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.sources is None:
            self.sources = []
        if not self.decay_rate:
            self.decay_rate = DECAY_RATES.get(self.signal_source, 0.01)
        if not self.effective_until:
            # Calculate effective_until based on 90% decay
            days_to_90_decay = -math.log(0.1) / self.decay_rate
            effective_date = datetime.now() + timedelta(days=days_to_90_decay)
            self.effective_until = effective_date.isoformat()


class TradingVectorStore:
    """Vector storage with decay-aware retrieval."""

    def __init__(self, qdrant_url: str = QDRANT_URL, ollama_url: str = OLLAMA_URL):
        self.qdrant_url = qdrant_url
        self.ollama_url = ollama_url
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        response = requests.get(f"{self.qdrant_url}/collections/{COLLECTION}")
        if response.status_code != 200:
            requests.put(
                f"{self.qdrant_url}/collections/{COLLECTION}",
                json={"vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"}}
            )

    def get_embedding(self, text: str) -> Optional[list]:
        """Generate embedding using Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text[:8000]},
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
        return None

    def store(self, payload: SignalPayload, text_for_embedding: str = None) -> Optional[str]:
        """Store a signal with embedding."""
        if text_for_embedding is None:
            text_for_embedding = f"{payload.topic}\n{payload.content[:4000]}"

        embedding = self.get_embedding(text_for_embedding)
        if not embedding:
            return None

        point_id = str(uuid.uuid4())

        response = requests.put(
            f"{self.qdrant_url}/collections/{COLLECTION}/points",
            json={
                "points": [{
                    "id": point_id,
                    "vector": embedding,
                    "payload": asdict(payload)
                }]
            },
            timeout=10
        )

        if response.status_code == 200:
            return point_id
        return None

    def search(self, query: str, limit: int = 10,
               filter_domain: str = None,
               filter_signal_source: str = None,
               apply_decay: bool = True) -> list:
        """Search with optional decay weighting."""
        embedding = self.get_embedding(query)
        if not embedding:
            return []

        # Build filter
        must_conditions = []
        if filter_domain:
            must_conditions.append({"key": "domain", "match": {"value": filter_domain}})
        if filter_signal_source:
            must_conditions.append({"key": "signal_source", "match": {"value": filter_signal_source}})

        search_params = {
            "vector": embedding,
            "limit": limit * 2 if apply_decay else limit,  # Get more, then filter
            "with_payload": True
        }

        if must_conditions:
            search_params["filter"] = {"must": must_conditions}

        response = requests.post(
            f"{self.qdrant_url}/collections/{COLLECTION}/points/search",
            json=search_params,
            timeout=30
        )

        if response.status_code != 200:
            return []

        results = response.json().get("result", [])

        if apply_decay:
            # Apply decay weighting
            now = datetime.now()
            weighted_results = []
            for r in results:
                payload = r.get("payload", {})
                try:
                    timestamp = datetime.fromisoformat(payload.get("timestamp", now.isoformat()))
                    days_elapsed = (now - timestamp).total_seconds() / 86400
                    decay_rate = payload.get("decay_rate", 0.01)
                    source_reliability = payload.get("source_reliability", 0.5)

                    # Relevance = base_score * reliability * decay
                    decay_factor = math.exp(-decay_rate * days_elapsed)
                    adjusted_score = r["score"] * source_reliability * decay_factor

                    weighted_results.append({
                        "score": adjusted_score,
                        "original_score": r["score"],
                        "decay_factor": decay_factor,
                        "payload": payload
                    })
                except:
                    weighted_results.append({
                        "score": r["score"],
                        "original_score": r["score"],
                        "decay_factor": 1.0,
                        "payload": payload
                    })

            # Sort by adjusted score and limit
            weighted_results.sort(key=lambda x: x["score"], reverse=True)
            return weighted_results[:limit]

        return results

    def get_stats(self) -> dict:
        """Get collection statistics."""
        response = requests.get(f"{self.qdrant_url}/collections/{COLLECTION}")
        if response.status_code == 200:
            result = response.json().get("result", {})
            return {
                "points_count": result.get("points_count", 0),
                "vector_size": result.get("config", {}).get("params", {}).get("vectors", {}).get("size", 0)
            }
        return {"error": "Could not get stats"}


if __name__ == "__main__":
    # Test
    store = TradingVectorStore()
    print(f"Stats: {store.get_stats()}")

    # Test search with decay
    results = store.search("RSI indicator momentum", limit=3, apply_decay=True)
    for r in results:
        print(f"Score: {r['score']:.3f} (decay: {r['decay_factor']:.3f}) - {r['payload'].get('topic', 'unknown')}")
