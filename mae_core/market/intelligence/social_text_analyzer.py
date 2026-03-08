"""
social_text_analyzer.py - Lightweight StockTwits Message Text Analysis

Reads raw StockTwits messages stored in RawStore's SQLite and extracts
cultural/thematic signals from the text WITHOUT heavy NLP dependencies.

The approach: smart keyword bucketing.  The same pattern-matching logic
that finds code patterns finds cultural patterns — frequency shifts and
intensity spikes are the signal, not semantic understanding.

Theme buckets:
  options_flow    — puts / calls / options / premium / IV / theta / gamma
  short_squeeze   — short / squeeze / gamma / float / SI
  earnings_play   — earnings / ER / guidance / beat / miss / report
  macro_fear      — fed / fomc / cpi / recession / crash / rate / inflation
  breakout        — breakout / ATH / all-time / resistance / squeeze / rip
  capitulation    — down / crash / dump / sell / rug / panic / bag / hold

Signal fired when:
  - A theme bucket's share of messages exceeds THEME_DOMINANCE_THRESHOLD
  - Intensity (exclamation + all-caps ratio) exceeds INTENSITY_THRESHOLD
  - Message count is above MIN_MESSAGE_COUNT (noise floor)

Output: SocialTextSignal dataclass — fed into signal pipeline as source
"social_text", domain "sentiment".
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("midge.market.social_text")

# --- Config ---
MIN_MESSAGE_COUNT = 5            # Ignore tickers with fewer messages
THEME_DOMINANCE_THRESHOLD = 0.40 # 40% of messages mention a theme → signal
INTENSITY_THRESHOLD = 0.25       # 25% of tokens are exclamation/all-caps → high intensity
LOOKBACK_HOURS = 24              # Only analyze messages from last 24h
RAW_DATA_DIR = Path("data/market/raw")

# --- Theme keyword buckets ---
THEME_BUCKETS: Dict[str, List[str]] = {
    "options_flow": [
        "put", "puts", "call", "calls", "option", "options", "premium",
        "iv", "theta", "gamma", "delta", "strike", "expiry", "expiration",
        "otm", "itm", "atm", "yolo", "weeklies",
    ],
    "short_squeeze": [
        "short", "squeeze", "gamma", "float", "si", "short interest",
        "days to cover", "dtc", "naked", "ftd", "fails to deliver",
        "short seller", "short sellers",
    ],
    "earnings_play": [
        "earnings", "er", "guidance", "beat", "miss", "eps", "report",
        "q1", "q2", "q3", "q4", "revenue", "catalyst", "print",
        "after hours", "pre market", "afterhours",
    ],
    "macro_fear": [
        "fed", "fomc", "cpi", "ppi", "inflation", "recession", "crash",
        "rate", "rates", "hike", "cut", "powell", "treasury", "yield",
        "tariff", "tariffs", "war", "geopolitical",
    ],
    "breakout": [
        "breakout", "break out", "ath", "all-time high", "resistance",
        "support", "rip", "ripping", "moon", "mooning", "rocket",
        "momentum", "trend", "run",
    ],
    "capitulation": [
        "down", "crash", "dump", "sell", "selling", "rug", "rugpull",
        "panic", "bag", "bagholder", "bagholding", "hold", "hodl",
        "underwater", "loss", "losses", "red",
    ],
}

# Pre-compile patterns for speed
_THEME_PATTERNS: Dict[str, re.Pattern] = {
    theme: re.compile(
        r"\b(" + "|".join(re.escape(kw) for kw in keywords) + r")\b",
        re.IGNORECASE,
    )
    for theme, keywords in THEME_BUCKETS.items()
}

# Pattern for all-caps words (3+ chars, not common abbreviations like "CEO")
_ALLCAPS_PATTERN = re.compile(r"\b[A-Z]{3,}\b")
_EXCLAMATION_PATTERN = re.compile(r"!")
_WORD_PATTERN = re.compile(r"\b\w{2,}\b")


@dataclass
class SocialTextSignal:
    """Theme-based text analysis result for a single ticker."""

    ticker: str
    dominant_theme: str             # Which theme bucket dominates
    theme_score: float              # 0.0–1.0 share of messages mentioning theme
    intensity: float                # 0.0–1.0 exclamation/all-caps ratio
    message_count: int
    themes: Dict[str, float]        # All theme scores for metadata
    direction: str                  # "bullish", "bearish", "neutral"
    strength: float                 # 0.0–1.0

    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signal_source: str = "social_text"
    decay_rate: float = 0.60        # Text sentiment fades within hours
    confidence: float = 0.40        # Lower prior — crowd text is noisy

    def to_plain_language(self) -> str:
        return (
            f"{self.ticker}: {self.dominant_theme} chatter "
            f"({self.theme_score:.0%} of {self.message_count} msgs, "
            f"intensity {self.intensity:.0%}) → {self.direction}"
        )


# Themes that lean bullish vs bearish
_BULLISH_THEMES = {"options_flow", "short_squeeze", "earnings_play", "breakout"}
_BEARISH_THEMES = {"macro_fear", "capitulation"}


def _analyze_messages(messages: List[Tuple[str, str, int]]) -> Optional[SocialTextSignal]:
    """
    Analyze a list of (ticker, body, likes) tuples.

    Returns a SocialTextSignal if the theme/intensity thresholds are met,
    otherwise None.
    """
    if not messages:
        return None

    ticker = messages[0][0]
    bodies = [body for _, body, _ in messages if body]
    n = len(bodies)

    if n < MIN_MESSAGE_COUNT:
        return None

    # --- Theme scoring ---
    theme_hit_counts: Dict[str, int] = {theme: 0 for theme in THEME_BUCKETS}
    total_intensity_score = 0.0

    for body in bodies:
        # Theme presence (boolean per message — prevents one message dominating)
        for theme, pattern in _THEME_PATTERNS.items():
            if pattern.search(body):
                theme_hit_counts[theme] += 1

        # Intensity: ratio of (exclamation marks + all-caps words) to total words
        words = _WORD_PATTERN.findall(body)
        word_count = max(len(words), 1)
        exclamations = len(_EXCLAMATION_PATTERN.findall(body))
        allcaps = len(_ALLCAPS_PATTERN.findall(body))
        total_intensity_score += (exclamations + allcaps) / word_count

    theme_scores = {theme: count / n for theme, count in theme_hit_counts.items()}
    avg_intensity = total_intensity_score / n

    # Find dominant theme
    dominant_theme, dominant_score = max(theme_scores.items(), key=lambda x: x[1])

    if dominant_score < THEME_DOMINANCE_THRESHOLD and avg_intensity < INTENSITY_THRESHOLD:
        return None  # Nothing actionable

    # Direction from theme
    if dominant_theme in _BULLISH_THEMES:
        direction = "bullish"
    elif dominant_theme in _BEARISH_THEMES:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength: weighted by theme dominance + intensity
    strength = min(1.0, (dominant_score + avg_intensity) / 2.0)

    return SocialTextSignal(
        ticker=ticker,
        dominant_theme=dominant_theme,
        theme_score=round(dominant_score, 4),
        intensity=round(avg_intensity, 4),
        message_count=n,
        themes={k: round(v, 4) for k, v in theme_scores.items()},
        direction=direction,
        strength=round(strength, 4),
    )


class SocialTextAnalyzer:
    """
    Reads StockTwits messages from RawStore SQLite and emits theme signals.

    Designed to run as a cadenced task (every N steps).  Reads messages
    from the last LOOKBACK_HOURS hours and returns one SocialTextSignal
    per ticker where a theme threshold is crossed.

    No external dependencies beyond stdlib + sqlite3.
    """

    def __init__(self, raw_store=None, base_dir: Optional[Path] = None):
        self._raw_store = raw_store
        self._base_dir = Path(base_dir) if base_dir else RAW_DATA_DIR
        self._db_path = self._base_dir / "stocktwits.db"

    def _get_conn(self) -> Optional[sqlite3.Connection]:
        """Open a read-only connection to the StockTwits DB if it exists."""
        if not self._db_path.exists():
            return None
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            return conn
        except Exception as exc:
            logger.debug("SocialTextAnalyzer: cannot open DB: %s", exc)
            return None

    def analyze_ticker(self, ticker: str) -> Optional[SocialTextSignal]:
        """
        Analyze recent StockTwits messages for a single ticker.

        Returns a SocialTextSignal or None if no meaningful theme found.
        """
        conn = self._get_conn()
        if conn is None:
            return None

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat()

        try:
            cur = conn.execute(
                """
                SELECT ticker, body, likes
                FROM stocktwits_messages
                WHERE ticker = ?
                  AND ingested_at >= ?
                ORDER BY ingested_at DESC
                LIMIT 100
                """,
                (ticker.upper(), cutoff),
            )
            rows = cur.fetchall()
        except Exception as exc:
            logger.debug("SocialTextAnalyzer DB read failed for %s: %s", ticker, exc)
            return None
        finally:
            conn.close()

        return _analyze_messages(rows)

    def analyze_all(self, tickers: Optional[List[str]] = None) -> List[SocialTextSignal]:
        """
        Analyze recent messages for all (or specified) tickers.

        If `tickers` is None, analyzes every ticker that has messages in
        the last LOOKBACK_HOURS window.
        """
        conn = self._get_conn()
        if conn is None:
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).isoformat()

        try:
            if tickers:
                placeholders = ",".join("?" * len(tickers))
                cur = conn.execute(
                    f"""
                    SELECT ticker, body, likes
                    FROM stocktwits_messages
                    WHERE ticker IN ({placeholders})
                      AND ingested_at >= ?
                    ORDER BY ticker, ingested_at DESC
                    """,
                    [t.upper() for t in tickers] + [cutoff],
                )
            else:
                cur = conn.execute(
                    """
                    SELECT ticker, body, likes
                    FROM stocktwits_messages
                    WHERE ingested_at >= ?
                    ORDER BY ticker, ingested_at DESC
                    """,
                    (cutoff,),
                )
            rows = cur.fetchall()
        except Exception as exc:
            logger.debug("SocialTextAnalyzer bulk read failed: %s", exc)
            return []
        finally:
            conn.close()

        # Group rows by ticker
        ticker_messages: Dict[str, List[Tuple[str, str, int]]] = {}
        for ticker, body, likes in rows:
            if ticker not in ticker_messages:
                ticker_messages[ticker] = []
            ticker_messages[ticker].append((ticker, body or "", likes or 0))

        signals = []
        for tkr, msgs in ticker_messages.items():
            try:
                sig = _analyze_messages(msgs)
                if sig is not None:
                    signals.append(sig)
            except Exception as exc:
                logger.debug("SocialTextAnalyzer analysis failed for %s: %s", tkr, exc)

        logger.debug(
            "SocialTextAnalyzer: analyzed %d tickers, %d signals above threshold",
            len(ticker_messages), len(signals),
        )
        return signals
