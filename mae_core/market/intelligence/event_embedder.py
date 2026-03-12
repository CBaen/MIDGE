"""EventEmbedder — converts market events to vectors stored in Qdrant.

MIDGE's semantic memory layer. Takes structured market events, converts them
to rich natural language, embeds via Ollama (mxbai-embed-large), and stores
in Qdrant with metadata for filtering.

Key capability: find_historical_precedents() — given live signals, find
historical situations that looked similar. This is MIDGE's pattern memory.

Degrades gracefully: if Qdrant or Ollama is unavailable, embedding is skipped
and a warning is logged. The daemon continues without interruption.

Split:
  event_descriptions.py  — text generation (one job: structured → string)
  event_embedder.py       — embedding + storage + retrieval (this file)
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

import requests

from mae_core.market.intelligence.event_descriptions import (
    describe_convergence_alert,
    describe_market_signal,
    describe_pattern_template,
    describe_insider_trade,
    describe_economic_event,
    describe_congressional_trade,
    describe_contract_award,
)

logger = logging.getLogger(__name__)

# --- Constants ---
COLLECTION_NAME = "midge_events"
VECTOR_DIM = 1024          # mxbai-embed-large output dimension
OLLAMA_MODEL = "mxbai-embed-large"
_CONNECT_TIMEOUT = 3.0     # seconds — fast fail if services are down
_EMBED_TIMEOUT = 15.0      # seconds — embedding can be slow on first call


class EventEmbedder:
    """Converts market events to vectors and stores them in Qdrant for semantic search.

    The three operations are: describe → embed → store.
    Retrieval inverts: embed query → nearest-neighbor search → return payloads.

    All methods return None and log warnings on service unavailability —
    they never raise exceptions to callers.
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        ollama_host: str = "http://localhost:11434",
        collection_name: str = COLLECTION_NAME,
    ):
        self.qdrant_base = f"http://{qdrant_host}:{qdrant_port}"
        self.ollama_host = ollama_host.rstrip("/")
        self.collection_name = collection_name
        self._available = False   # Set True after successful health check
        self._initialized = False  # Collection created

        self._check_services()

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    def _check_services(self) -> None:
        """Probe Qdrant and Ollama. Set _available flag."""
        try:
            r = requests.get(f"{self.qdrant_base}/healthz", timeout=_CONNECT_TIMEOUT)
            if r.status_code != 200:
                logger.warning("EventEmbedder: Qdrant health check failed (%d)", r.status_code)
                return
        except Exception as e:
            logger.warning("EventEmbedder: Qdrant not reachable (%s). Embedding disabled.", e)
            return

        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=_CONNECT_TIMEOUT)
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(OLLAMA_MODEL in m for m in models):
                logger.warning(
                    "EventEmbedder: %s not found in Ollama (available: %s). Embedding disabled.",
                    OLLAMA_MODEL, models,
                )
                return
        except Exception as e:
            logger.warning("EventEmbedder: Ollama not reachable (%s). Embedding disabled.", e)
            return

        self._available = True
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't already exist."""
        if self._initialized:
            return
        try:
            r = requests.get(
                f"{self.qdrant_base}/collections/{self.collection_name}",
                timeout=_CONNECT_TIMEOUT,
            )
            if r.status_code == 200:
                self._initialized = True
                logger.info("EventEmbedder: using existing collection '%s'", self.collection_name)
                return

            # Create it
            payload = {
                "vectors": {
                    "size": VECTOR_DIM,
                    "distance": "Cosine",
                },
            }
            r = requests.put(
                f"{self.qdrant_base}/collections/{self.collection_name}",
                json=payload,
                timeout=_CONNECT_TIMEOUT,
            )
            if r.status_code in (200, 201):
                self._initialized = True
                logger.info(
                    "EventEmbedder: created Qdrant collection '%s' (dim=%d, Cosine)",
                    self.collection_name, VECTOR_DIM,
                )
            else:
                logger.warning("EventEmbedder: collection creation returned %d", r.status_code)
        except Exception as e:
            logger.warning("EventEmbedder: collection setup failed: %s", e)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[list[float]]:
        """Embed text via Ollama. Returns None on failure."""
        if not self._available:
            return None
        try:
            r = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=_EMBED_TIMEOUT,
            )
            r.raise_for_status()
            vector = r.json().get("embedding", [])
            if len(vector) != VECTOR_DIM:
                logger.warning(
                    "EventEmbedder: unexpected vector dim %d (expected %d)",
                    len(vector), VECTOR_DIM,
                )
                return None
            return vector
        except Exception as e:
            logger.warning("EventEmbedder: embedding failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _store(self, point_id: str, vector: list[float], payload: dict) -> bool:
        """Upsert a single point into Qdrant. Returns True on success."""
        if not self._available or not self._initialized:
            return False
        try:
            # Qdrant point IDs must be UUID or unsigned int — use UUID5 from string ID
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
            body = {
                "points": [
                    {"id": qdrant_id, "vector": vector, "payload": payload}
                ]
            }
            r = requests.put(
                f"{self.qdrant_base}/collections/{self.collection_name}/points",
                json=body,
                timeout=_EMBED_TIMEOUT,
            )
            if r.status_code not in (200, 201):
                logger.debug(
                    "EventEmbedder: upsert failed for %s: HTTP %d", point_id, r.status_code
                )
                return False
            return True
        except Exception as e:
            logger.debug("EventEmbedder: store failed: %s", e)
            return False

    def _describe_and_store(
        self,
        description: str,
        event_type: str,
        point_id: str,
        metadata: dict,
    ) -> Optional[str]:
        """Embed description, store with payload. Returns point_id or None."""
        vector = self._embed(description)
        if vector is None:
            return None

        payload = {
            "event_type": event_type,
            "description": description[:2000],  # cap for payload size
            "stored_at": datetime.now().isoformat(),
            **metadata,
        }
        success = self._store(point_id, vector, payload)
        return point_id if success else None

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def embed_convergence_alert(self, alert) -> Optional[str]:
        """Embed and store a ConvergenceAlert. Returns point ID or None."""
        try:
            description = describe_convergence_alert(alert)
            alert_id = getattr(alert, "alert_id", str(uuid.uuid4()))
            ts = getattr(alert, "timestamp", datetime.now())
            direction = getattr(alert, "direction", "neutral")
            confidence = getattr(alert, "confidence", 0.0)
            strength = getattr(alert, "strength", 0.0)
            domains = getattr(alert, "domains_converging", [])
            ticker = _extract_ticker(alert)

            metadata = {
                "alert_id": alert_id,
                "ticker": ticker,
                "direction": direction,
                "confidence": round(confidence, 3),
                "strength": round(strength, 3),
                "domains": domains,
                "domain_count": len(domains),
                "urgency": getattr(alert, "urgency", "days"),
                "coherence": round(getattr(alert, "coherence", 1.0), 3),
                "date": ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts),
                "timestamp_iso": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            }

            point_id = f"convergence:{alert_id}"
            return self._describe_and_store(description, "convergence_alert", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_convergence_alert failed: %s", e)
            return None

    def embed_market_signal(self, signal) -> Optional[str]:
        """Embed and store a MarketSignal. Returns point ID or None."""
        try:
            description = describe_market_signal(signal)
            signal_id = getattr(signal, "signal_id", str(uuid.uuid4()))
            ts = getattr(signal, "timestamp", datetime.now())
            symbol = getattr(signal, "symbol", "")
            domain = getattr(signal, "domain", "")
            direction = getattr(signal, "direction", "neutral")
            source = getattr(signal, "source", "")

            metadata = {
                "signal_id": signal_id,
                "ticker": symbol,
                "domain": domain,
                "direction": direction,
                "source": source,
                "strength": round(getattr(signal, "strength", 0.0), 3),
                "confidence": round(getattr(signal, "confidence", 0.5), 3),
                "asset_class": getattr(signal, "asset_class", "stock"),
                "date": ts.strftime("%Y-%m-%d") if isinstance(ts, datetime) else str(ts),
                "timestamp_iso": ts.isoformat() if isinstance(ts, datetime) else str(ts),
            }

            point_id = f"signal:{signal_id}"
            return self._describe_and_store(description, "market_signal", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_market_signal failed: %s", e)
            return None

    def embed_pattern_template(self, template) -> Optional[str]:
        """Embed and store a PatternTemplate. Returns point ID or None."""
        try:
            description = describe_pattern_template(template)
            template_id = getattr(template, "template_id", "")
            if not template_id:
                template_id = hashlib.sha256(
                    getattr(template, "domain_signature", "unknown").encode()
                ).hexdigest()[:16]

            wins = getattr(template, "wins", 0)
            losses = getattr(template, "losses", 0)
            total = wins + losses
            win_rate = wins / total if total > 0 else 0.0

            metadata = {
                "template_id": template_id,
                "direction": getattr(template, "direction", "neutral"),
                "domain_signature": getattr(template, "domain_signature", ""),
                "domains": getattr(template, "domains", []),
                "n_instances": getattr(template, "n_instances", 0),
                "symbols_seen_count": len(set(getattr(template, "symbols_seen", []))),
                "avg_move_pct": round(getattr(template, "avg_move_pct", 0.0), 2),
                "win_rate": round(win_rate, 3),
                "cross_validated": getattr(template, "cross_validated", False),
                "expected_window_days": getattr(template, "expected_move_window_days", 14),
                "created_at": getattr(template, "created_at", ""),
            }

            point_id = f"template:{template_id}"
            return self._describe_and_store(description, "pattern_template", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_pattern_template failed: %s", e)
            return None

    def embed_insider_trade(self, trade: dict) -> Optional[str]:
        """Embed and store an insider trade record. Returns point ID or None."""
        try:
            description = describe_insider_trade(trade)
            # Stable ID from key fields
            ticker = trade.get("ticker", trade.get("symbol", ""))
            name = trade.get("insider_name", trade.get("name", ""))
            date_str = trade.get("date", trade.get("transaction_date", ""))
            raw = f"{ticker}:{name}:{date_str}"
            point_id = f"insider:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

            value = float(trade.get("value", trade.get("total_value", 0)) or 0)
            tx_type = trade.get("transaction_type", trade.get("type", "P"))
            is_buy = "P" in str(tx_type).upper() or "buy" in str(tx_type).lower()

            metadata = {
                "ticker": ticker,
                "insider_name": name,
                "role": trade.get("relationship", trade.get("role", "")),
                "direction": "bullish" if is_buy else "bearish",
                "transaction_type": str(tx_type),
                "value": round(value, 2),
                "date": date_str,
                "source": trade.get("source", "sec_form4"),
            }

            return self._describe_and_store(description, "insider_trade", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_insider_trade failed: %s", e)
            return None

    def embed_economic_event(self, event: dict) -> Optional[str]:
        """Embed and store an economic event. Returns point ID or None."""
        try:
            description = describe_economic_event(event)
            event_name = event.get("event", event.get("name", "economic"))
            date_str = event.get("date", event.get("release_date", ""))
            raw = f"{event_name}:{date_str}"
            point_id = f"econ:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

            metadata = {
                "event_name": event_name,
                "date": date_str,
                "country": event.get("country", "US"),
                "impact": event.get("impact", "medium"),
                "actual": str(event.get("actual", "")),
                "forecast": str(event.get("forecast", "")),
                "series_id": event.get("series_id", ""),
                "direction": _econ_direction(event),
            }

            return self._describe_and_store(description, "economic_event", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_economic_event failed: %s", e)
            return None

    def embed_congressional_trade(self, trade: dict) -> Optional[str]:
        """Embed and store a congressional trade. Returns point ID or None."""
        try:
            description = describe_congressional_trade(trade)
            member = trade.get("representative", trade.get("member", trade.get("senator", "")))
            ticker = trade.get("ticker", trade.get("symbol", ""))
            date_str = trade.get("transaction_date", trade.get("date", ""))
            raw = f"{member}:{ticker}:{date_str}"
            point_id = f"congress:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

            tx_type = trade.get("type", trade.get("transaction_type", "Purchase"))
            is_buy = "purchase" in str(tx_type).lower() or "buy" in str(tx_type).lower()

            metadata = {
                "member": member,
                "ticker": ticker,
                "chamber": trade.get("chamber", ""),
                "state": trade.get("state", ""),
                "committee": trade.get("committee", ""),
                "direction": "bullish" if is_buy else "bearish",
                "amount_range": trade.get("amount", trade.get("amount_range", "")),
                "date": date_str,
            }

            return self._describe_and_store(description, "congressional_trade", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_congressional_trade failed: %s", e)
            return None

    def embed_contract_award(self, contract: dict) -> Optional[str]:
        """Embed and store a government contract award. Returns point ID or None."""
        try:
            description = describe_contract_award(contract)
            company = contract.get("recipient_name", contract.get("company", ""))
            agency = contract.get("awarding_agency", contract.get("agency", ""))
            date_str = contract.get("award_date", contract.get("date", ""))
            raw = f"{company}:{agency}:{date_str}"
            point_id = f"contract:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

            amount = float(contract.get("amount", contract.get("base_and_all_options_value", 0)) or 0)

            metadata = {
                "company": company,
                "ticker": contract.get("ticker", ""),
                "agency": agency,
                "amount": round(amount, 2),
                "date": date_str,
                "direction": "bullish",  # contract awards are always bullish for recipient
            }

            return self._describe_and_store(description, "contract_award", point_id, metadata)
        except Exception as e:
            logger.debug("EventEmbedder: embed_contract_award failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query_text: str,
        limit: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search: find events similar to a text description.

        filters: optional dict with keys:
          - ticker: str — filter by ticker symbol
          - event_type: str or list[str] — filter by event type
          - direction: str — "bullish" or "bearish"
          - date_from: str — ISO date (YYYY-MM-DD) lower bound
          - date_to: str — ISO date (YYYY-MM-DD) upper bound

        Returns list of dicts with keys: score, event_type, description, metadata.
        """
        if not self._available:
            return []

        vector = self._embed(query_text)
        if vector is None:
            return []

        search_body: dict[str, Any] = {
            "vector": vector,
            "limit": limit,
            "with_payload": True,
        }

        qdrant_filter = _build_qdrant_filter(filters)
        if qdrant_filter:
            search_body["filter"] = qdrant_filter

        try:
            r = requests.post(
                f"{self.qdrant_base}/collections/{self.collection_name}/points/search",
                json=search_body,
                timeout=_EMBED_TIMEOUT,
            )
            r.raise_for_status()
            results = r.json().get("result", [])
            return [
                {
                    "score": item.get("score", 0.0),
                    "event_type": item.get("payload", {}).get("event_type", ""),
                    "description": item.get("payload", {}).get("description", ""),
                    "metadata": {
                        k: v for k, v in item.get("payload", {}).items()
                        if k not in ("description", "event_type", "stored_at")
                    },
                    "stored_at": item.get("payload", {}).get("stored_at", ""),
                }
                for item in results
            ]
        except Exception as e:
            logger.debug("EventEmbedder: find_similar search failed: %s", e)
            return []

    def find_similar_to_event(self, point_id: str, limit: int = 10) -> list[dict]:
        """Find events similar to an already-stored event (by point ID string)."""
        if not self._available:
            return []

        # Fetch the stored vector
        qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
        try:
            r = requests.get(
                f"{self.qdrant_base}/collections/{self.collection_name}/points/{qdrant_id}",
                timeout=_CONNECT_TIMEOUT,
            )
            r.raise_for_status()
            point = r.json().get("result", {})
            description = point.get("payload", {}).get("description", "")
            if not description:
                return []
            return self.find_similar(description, limit=limit)
        except Exception as e:
            logger.debug("EventEmbedder: find_similar_to_event failed: %s", e)
            return []

    def find_historical_precedents(
        self,
        current_signals: list,
        ticker: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Given current live signals, find historical situations that looked similar.

        This is MIDGE's primary pattern memory call. Converts the live signal
        list to a natural language query, searches for historical matches,
        and returns ranked results with outcomes.

        current_signals: list of Signal or MarketSignal objects, or dicts.
        """
        if not current_signals:
            return []

        # Build a query description from the live signals
        parts = []
        if ticker:
            parts.append(f"Situation for {ticker}:")
        else:
            parts.append("Market-wide situation:")

        domain_counts: dict[str, int] = {}
        directions: list[str] = []

        for sig in current_signals:
            domain = _get_attr(sig, "domain", "")
            direction = _get_attr(sig, "direction", "neutral")
            source = _get_attr(sig, "source", "")
            strength = float(_get_attr(sig, "strength", 0.0))

            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            if direction and direction != "neutral":
                directions.append(direction)
            if domain or source:
                parts.append(
                    f"{direction} signal from {source or domain} "
                    f"(strength {strength:.2f})"
                )

        if domain_counts:
            domain_list = ", ".join(
                f"{d} ({c} signal{'s' if c > 1 else ''})"
                for d, c in sorted(domain_counts.items())
            )
            parts.append(f"Active domains: {domain_list}.")

        dominant_direction = "bullish" if directions.count("bullish") > directions.count("bearish") else "bearish"
        parts.append(f"Overall bias: {dominant_direction}.")

        query = " ".join(parts)

        filters: dict = {"direction": dominant_direction}
        if ticker:
            # Search for same ticker first, fall back to similar patterns
            results = self.find_similar(query, limit=limit * 2, filters={**filters, "ticker": ticker})
            if len(results) < limit:
                # Broaden to cross-ticker if not enough ticker-specific results
                cross_results = self.find_similar(query, limit=limit, filters=filters)
                existing_ids = {r.get("metadata", {}).get("alert_id", "") for r in results}
                for cr in cross_results:
                    if cr.get("metadata", {}).get("alert_id", "") not in existing_ids:
                        results.append(cr)
        else:
            results = self.find_similar(query, limit=limit, filters=filters)

        return results[:limit]

    @property
    def is_available(self) -> bool:
        return self._available


# ------------------------------------------------------------------
# Filter builder
# ------------------------------------------------------------------

def _build_qdrant_filter(filters: Optional[dict]) -> Optional[dict]:
    """Convert a simple filter dict to Qdrant filter DSL."""
    if not filters:
        return None

    must_clauses = []

    if "ticker" in filters and filters["ticker"]:
        must_clauses.append({
            "key": "ticker",
            "match": {"value": filters["ticker"]},
        })

    if "direction" in filters and filters["direction"]:
        must_clauses.append({
            "key": "direction",
            "match": {"value": filters["direction"]},
        })

    if "event_type" in filters and filters["event_type"]:
        et = filters["event_type"]
        if isinstance(et, list):
            must_clauses.append({"key": "event_type", "match": {"any": et}})
        else:
            must_clauses.append({"key": "event_type", "match": {"value": et}})

    # Date range filtering (Qdrant range on string fields — lexicographic works for ISO dates)
    range_filter: dict = {}
    if "date_from" in filters and filters["date_from"]:
        range_filter["gte"] = filters["date_from"]
    if "date_to" in filters and filters["date_to"]:
        range_filter["lte"] = filters["date_to"]
    if range_filter:
        must_clauses.append({"key": "date", "range": range_filter})

    if not must_clauses:
        return None
    return {"must": must_clauses}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_ticker(alert) -> str:
    ticker = getattr(alert, "ticker", None)
    if ticker:
        return ticker
    for sig in getattr(alert, "signals", []):
        sym = getattr(sig, "metadata", {}).get("symbol", "")
        if not sym:
            parts = str(getattr(sig, "signal_id", "")).split(":")
            if len(parts) >= 2 and parts[1].isupper():
                sym = parts[1]
        if sym:
            return sym
    return ""


def _econ_direction(event: dict) -> str:
    """Rough direction heuristic for economic events."""
    actual = event.get("actual", "")
    forecast = event.get("forecast", "")
    try:
        a = float(str(actual).replace("%", "").replace(",", ""))
        f = float(str(forecast).replace("%", "").replace(",", ""))
        return "bullish" if a > f else ("bearish" if a < f else "neutral")
    except (ValueError, TypeError):
        return "neutral"


def _get_attr(obj, key: str, default):
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
