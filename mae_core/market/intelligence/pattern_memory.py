"""PatternMemory — MIDGE's semantic memory for market patterns.

High-level interface over EventEmbedder. Other systems call this to ask
"have we seen this before?" without needing to know about Qdrant or Ollama.

Three key questions PatternMemory answers:
  1. remember_event()       — "store this for later recall"
  2. recall_similar()       — "what do we know that looks like this?"
  3. find_precedents()      — "show me past situations that felt like now"
  4. get_pattern_context()  — "for this convergence alert, what's the historical context?"

Integration point: bootstrap wires this onto ctx.pattern_memory.
Any system that wants historical context calls ctx.pattern_memory.find_precedents().
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class PatternMemory:
    """MIDGE's semantic memory for market patterns.

    Uses EventEmbedder for storage/retrieval. Provides high-level methods
    that other systems call to ask 'have we seen this before?'

    Degrades gracefully when embedder is unavailable — returns empty lists
    and logs at DEBUG level.
    """

    def __init__(self, embedder):
        """
        Args:
            embedder: EventEmbedder instance. Can be None (memory will no-op).
        """
        self.embedder = embedder
        self._available = embedder is not None and getattr(embedder, "is_available", False)

        if self._available:
            logger.info("PatternMemory: online (Qdrant + Ollama available)")
        else:
            logger.warning(
                "PatternMemory: offline — semantic memory disabled. "
                "Start Qdrant (port 6333) and Ollama (port 11434) to enable."
            )

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def remember_event(self, event_type: str, event_data: dict) -> Optional[str]:
        """Store an event in semantic memory.

        event_type: one of 'convergence_alert', 'market_signal', 'pattern_template',
                    'insider_trade', 'economic_event', 'congressional_trade',
                    'contract_award', or any custom string.
        event_data: raw dict representation of the event.

        Returns point ID string or None if storage failed/unavailable.
        """
        if not self._available:
            return None

        _dispatch = {
            "insider_trade": self.embedder.embed_insider_trade,
            "economic_event": self.embedder.embed_economic_event,
            "congressional_trade": self.embedder.embed_congressional_trade,
            "contract_award": self.embedder.embed_contract_award,
        }

        fn = _dispatch.get(event_type)
        if fn is not None:
            try:
                return fn(event_data)
            except Exception as e:
                logger.debug("PatternMemory.remember_event failed for %s: %s", event_type, e)
                return None

        # For object-based event types, callers should use the typed methods directly.
        logger.debug(
            "PatternMemory.remember_event: no handler for event_type '%s'. "
            "Use embed_convergence_alert / embed_market_signal / embed_pattern_template directly.",
            event_type,
        )
        return None

    def remember_convergence_alert(self, alert) -> Optional[str]:
        """Store a ConvergenceAlert in semantic memory. Returns point ID."""
        if not self._available:
            return None
        try:
            return self.embedder.embed_convergence_alert(alert)
        except Exception as e:
            logger.debug("PatternMemory.remember_convergence_alert failed: %s", e)
            return None

    def remember_market_signal(self, signal) -> Optional[str]:
        """Store a MarketSignal in semantic memory. Returns point ID."""
        if not self._available:
            return None
        try:
            return self.embedder.embed_market_signal(signal)
        except Exception as e:
            logger.debug("PatternMemory.remember_market_signal failed: %s", e)
            return None

    def remember_pattern_template(self, template) -> Optional[str]:
        """Store a PatternTemplate in semantic memory. Returns point ID."""
        if not self._available:
            return None
        try:
            return self.embedder.embed_pattern_template(template)
        except Exception as e:
            logger.debug("PatternMemory.remember_pattern_template failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def recall_similar(self, description: str, limit: int = 5) -> list[dict]:
        """What do we remember that looks like this?

        Takes a free-text description and returns the most similar stored events.
        Each result has: score, event_type, description, metadata, stored_at.
        """
        if not self._available:
            return []
        try:
            return self.embedder.find_similar(description, limit=limit)
        except Exception as e:
            logger.debug("PatternMemory.recall_similar failed: %s", e)
            return []

    def find_precedents(
        self,
        ticker: str,
        signals: list,
        limit: int = 5,
    ) -> list[dict]:
        """Given a developing situation, find historical precedents.

        Returns ranked list of past events similar to the current signal
        configuration, prioritizing same-ticker history then cross-ticker.

        Each result: {score, event_type, description, metadata, stored_at}
        """
        if not self._available:
            return []
        try:
            return self.embedder.find_historical_precedents(
                current_signals=signals,
                ticker=ticker,
                limit=limit,
            )
        except Exception as e:
            logger.debug("PatternMemory.find_precedents failed: %s", e)
            return []

    def get_pattern_context(self, alert) -> dict:
        """For a convergence alert, find historical context.

        Returns:
          {
            "similar_alerts": [...],     # Past convergence alerts with similar structure
            "similar_signals": [...],    # Individual signals that match current domains
            "similar_templates": [...],  # Pattern templates matching domain combination
            "temporal_note": str,        # Observation about timing patterns
          }
        """
        if not self._available:
            return _empty_context()

        try:
            from mae_core.market.intelligence.event_descriptions import describe_convergence_alert
            alert_desc = describe_convergence_alert(alert)
        except Exception:
            alert_desc = getattr(alert, "summary", "convergence alert")

        similar_alerts = self.embedder.find_similar(
            alert_desc,
            limit=5,
            filters={"event_type": "convergence_alert"},
        )

        # Find matching templates by domain signature
        domains = getattr(alert, "domains_converging", [])
        direction = getattr(alert, "direction", "neutral")
        domain_sig = "+".join(sorted(domains))
        template_query = (
            f"{direction} pattern template with domain signature {domain_sig}. "
            f"Domains: {', '.join(domains)}."
        )
        similar_templates = self.embedder.find_similar(
            template_query,
            limit=3,
            filters={"event_type": "pattern_template", "direction": direction},
        )

        # Find similar individual signals
        ticker = _extract_ticker_from_alert(alert)
        signal_filters: dict = {"event_type": "market_signal"}
        if ticker:
            signal_filters["ticker"] = ticker
        signals = getattr(alert, "signals", [])
        signal_desc = " ".join(
            f"{getattr(s, 'direction', '')} {getattr(s, 'domain', '')} "
            f"from {getattr(s, 'source', '')}"
            for s in signals[:3]
        )
        similar_signals = self.embedder.find_similar(
            signal_desc or alert_desc,
            limit=4,
            filters=signal_filters,
        ) if signal_desc else []

        # Temporal observation
        temporal_note = _make_temporal_note(similar_alerts)

        return {
            "similar_alerts": similar_alerts,
            "similar_signals": similar_signals,
            "similar_templates": similar_templates,
            "temporal_note": temporal_note,
        }

    def search_by_ticker(self, ticker: str, limit: int = 10) -> list[dict]:
        """Retrieve all stored events for a specific ticker."""
        if not self._available:
            return []
        try:
            return self.embedder.find_similar(
                f"events related to {ticker} stock",
                limit=limit,
                filters={"ticker": ticker},
            )
        except Exception as e:
            logger.debug("PatternMemory.search_by_ticker failed: %s", e)
            return []

    def search_insider_buys(self, ticker: str = None, limit: int = 10) -> list[dict]:
        """Find insider purchase events, optionally filtered by ticker."""
        if not self._available:
            return []
        query = f"insider purchase bullish signal{' for ' + ticker if ticker else ''}"
        filters: dict = {"event_type": "insider_trade", "direction": "bullish"}
        if ticker:
            filters["ticker"] = ticker
        try:
            return self.embedder.find_similar(query, limit=limit, filters=filters)
        except Exception as e:
            logger.debug("PatternMemory.search_insider_buys failed: %s", e)
            return []

    def search_high_confidence_alerts(
        self, direction: str = None, limit: int = 10
    ) -> list[dict]:
        """Find historically high-confidence convergence alerts."""
        if not self._available:
            return []
        query = (
            f"high confidence {direction or ''} convergence alert multiple domains agree "
            "strong signal reliable historical win rate"
        )
        filters: dict = {"event_type": "convergence_alert"}
        if direction:
            filters["direction"] = direction
        try:
            return self.embedder.find_similar(query, limit=limit, filters=filters)
        except Exception as e:
            logger.debug("PatternMemory.search_high_confidence_alerts failed: %s", e)
            return []

    @property
    def is_available(self) -> bool:
        return self._available


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _empty_context() -> dict:
    return {
        "similar_alerts": [],
        "similar_signals": [],
        "similar_templates": [],
        "temporal_note": "Pattern memory offline — semantic search unavailable.",
    }


def _extract_ticker_from_alert(alert) -> str:
    ticker = getattr(alert, "ticker", None)
    if ticker:
        return ticker
    for sig in getattr(alert, "signals", []):
        sym = getattr(sig, "metadata", {}).get("symbol", "")
        if sym:
            return sym
    return ""


def _make_temporal_note(similar_alerts: list[dict]) -> str:
    """Generate a plain-language note about temporal patterns in similar alerts."""
    if not similar_alerts:
        return "No historical precedents found in memory."

    dates = []
    for a in similar_alerts:
        date_str = a.get("metadata", {}).get("date", "")
        if date_str:
            try:
                dates.append(datetime.strptime(date_str[:10], "%Y-%m-%d"))
            except ValueError:
                pass

    if not dates:
        return f"Found {len(similar_alerts)} similar historical alert(s) — dates unavailable."

    if len(dates) == 1:
        return f"One similar alert found, dated {dates[0].strftime('%Y-%m-%d')}."

    dates.sort()
    oldest = dates[0].strftime("%Y-%m-%d")
    newest = dates[-1].strftime("%Y-%m-%d")
    avg_score = sum(a.get("score", 0) for a in similar_alerts) / len(similar_alerts)
    return (
        f"Found {len(similar_alerts)} similar historical alerts "
        f"(range: {oldest} to {newest}). "
        f"Average semantic similarity: {avg_score:.2f}."
    )
