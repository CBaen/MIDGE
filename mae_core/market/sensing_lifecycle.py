"""Market Sensing Lifecycle — signal enrichment, storage, and watchlist loading.

Extracted from MarketSensingHook: _enrich_signal(), _store_signals(), _load_watchlist().
All functions take explicit parameters instead of relying on `self`.

Called by MarketSensingHook in sensing_hook.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List

logger = logging.getLogger("midge.market.sensing")

# Data paths — same as sensing_hook.py so both modules resolve consistently
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "midge"
SIGNALS_DIR = DATA_DIR / "signals"


def enrich_signal(
    sig: Any,
    velocity_detector: Any,
    filing_analyzer: Any,
    form8k_sentiment: Any,
) -> None:
    """Apply velocity and filing-time modifiers to a signal (mutates in place)."""
    # Populate velocity via VelocityDetector
    if velocity_detector is not None:
        try:
            state = velocity_detector.record(sig.signal_id, sig.strength, sig.timestamp)
            sig.velocity = state.current_velocity
        except Exception:
            pass

    # Apply filing-time confidence modifier for SEC filings
    if filing_analyzer is not None and sig.source in ("sec_form4", "sec_form8k"):
        try:
            filing_dt = sig.received_at or sig.timestamp
            fta_signal = filing_analyzer.analyze_filing_time(
                ticker=sig.symbol,
                filer_name=sig.metadata.get("filer_name", ""),
                filing_date=sig.timestamp.strftime("%Y-%m-%d"),
                filing_datetime=filing_dt,
                form_type="4" if sig.source == "sec_form4" else "8-K",
            )
            sig.confidence = max(0.0, min(1.0, sig.confidence + fta_signal.confidence_modifier))
        except Exception:
            pass

    # Apply 8-K text sentiment via Ollama (enriches beyond rule-based item codes)
    if form8k_sentiment is not None and sig.source == "sec_form8k":
        try:
            event_text = sig.metadata.get("event_summary", "")
            item_code = sig.metadata.get("item_code", "")
            result = form8k_sentiment.classify(event_text, item_code)
            if result is not None:
                # Override direction if sentiment disagrees with rule-based
                if result.direction != "neutral":
                    sig.direction = result.direction
                sig.confidence = max(0.0, min(1.0, sig.confidence + result.confidence_modifier))
                sig.metadata["ollama_sentiment"] = result.direction
                sig.metadata["ollama_raw"] = result.raw_response[:200]
        except Exception:
            pass


def store_signals(signals: list, memory: Any) -> None:
    """Store signals to Qdrant + JSONL archive."""
    # Qdrant (if available)
    if memory is not None:
        try:
            if memory.is_available():
                memory.store_signals(signals)
        except Exception:
            logger.debug("Qdrant storage failed", exc_info=True)

    # JSONL cold storage (always)
    today = datetime.now().strftime("%Y-%m-%d")
    jsonl_path = SIGNALS_DIR / f"{today}.jsonl"
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for sig in signals:
                record = {
                    "signal_id": sig.signal_id,
                    "source": sig.source,
                    "symbol": sig.symbol,
                    "domain": sig.domain,
                    "direction": sig.direction,
                    "strength": sig.strength,
                    "confidence": sig.confidence,
                    "velocity": sig.velocity,
                    "timestamp": sig.timestamp.isoformat(),
                    "received_at": sig.received_at.isoformat(),
                    "metadata": sig.metadata,
                }
                f.write(json.dumps(record) + "\n")
    except Exception:
        logger.debug("JSONL archive write failed", exc_info=True)


def load_watchlist() -> dict:
    """Load watchlist from data/midge/watchlist.json."""
    watchlist_path = DATA_DIR / "watchlist.json"
    try:
        with open(watchlist_path) as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not load watchlist from %s, using defaults", watchlist_path)
        return {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
                         "LMT", "RTX", "NOC", "GD", "BA"],
            "keywords": ["cybersecurity", "artificial intelligence", "defense", "space"],
            "companies": {
                "Lockheed Martin": "LMT",
                "Raytheon Technologies": "RTX",
                "Northrop Grumman": "NOC",
                "General Dynamics": "GD",
                "Boeing": "BA",
            },
        }
