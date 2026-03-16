"""Narrative feedback loop — closes the dead end in MIDGE's daily letter.

The daily LLM-generated letter contains synthesized conclusions (e.g.
"institutional moves lead insider buying by 4 days") that currently exist
ONLY in the letter text and are never fed back into the learning systems.

This module:
  1. Parses the narrative text with regex/keyword rules (no LLM — fast, deterministic).
  2. Extracts structured NarrativeInsight records.
  3. Feeds them back into SharedAttention and WorldModel.
  4. Persists all insights to data/midge/narrative_insights.jsonl.

Usage (called at the end of generate_daily_narrative):
    feedback = NarrativeFeedback(shared_attention=sa, world_model=wm)
    insights = feedback.extract_insights(narrative_text)
    feedback.feed_back(insights)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("midge.narrative_feedback")

_INSIGHTS_PATH = Path("data/midge/narrative_insights.jsonl")

# ── Confidence language → numeric mapping ─────────────────────────────────────
# Matches the _confidence_words() scale in daily_narrative.py (reversed here).
_CONFIDENCE_MAP: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bvery confident\b", re.I), 0.87),
    (re.compile(r"\bvery sure\b", re.I), 0.87),
    (re.compile(r"\bfairly sure\b", re.I), 0.70),
    (re.compile(r"\bfairly confident\b", re.I), 0.70),
    (re.compile(r"\bforming\b|\bneed more\b", re.I), 0.50),
    (re.compile(r"\bstill watching\b|\bstill forming\b", re.I), 0.45),
    (re.compile(r"\bearly\b|\bnot ready\b|\bnot yet\b", re.I), 0.40),
    (re.compile(r"\buncertain\b|\bneed more information\b", re.I), 0.35),
]

# ── Direction word patterns ────────────────────────────────────────────────────
_BULLISH_WORDS = re.compile(
    r"\b(rise|rising|rise|up|bull(?:ish)?|long|climb(?:ing)?|rally(?:ing)?)\b", re.I
)
_BEARISH_WORDS = re.compile(
    r"\b(fall(?:ing)?|drop(?:ping)?|down|bear(?:ish)?|short|declin(?:e|ing)?)\b", re.I
)

# ── Ticker pattern: 1–5 uppercase letters, not a common English word ──────────
_COMMON_WORDS = {
    "A", "I", "THE", "AND", "OR", "FOR", "IN", "ON", "AT", "TO", "BE", "IS",
    "IT", "BY", "AN", "AS", "US", "AM", "MY", "UP", "OK", "NO", "DO", "GO",
    "ALL", "BUT", "NOT", "NEW", "OUT", "NOW", "HOW", "WHO", "WHY", "WHEN",
    "WHAT", "WITH", "FROM", "INTO", "HAVE", "THAT", "THIS", "THEY", "WERE",
    "BEEN", "MORE", "THAN", "THEN", "WILL", "ALSO", "JUST", "EVEN", "SOME",
    "BOTH", "INTO", "OVER", "EACH", "MOST", "ONLY", "VERY", "LIKE", "WELL",
    "STILL", "ABOUT", "AFTER", "WHICH", "COULD", "THOSE", "THEIR", "THESE",
    "MIDGE", "SUBJECT", "LAYER", "WHAT", "THINGS",
}
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# ── Causal claim patterns ─────────────────────────────────────────────────────
# Matches: "X leads Y by N days", "X precedes Y", "when X happens, Y follows"
_CAUSAL_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"([\w\s]+?)\s+leads?\s+([\w\s]+?)\s+by\s+(?:about\s+)?(\d+(?:\.\d+)?)\s+days?",
        re.I,
    ),
    re.compile(
        r"([\w\s]+?)\s+precedes?\s+([\w\s]+?)\s+by\s+(?:about\s+)?(\d+(?:\.\d+)?)\s+days?",
        re.I,
    ),
    re.compile(
        r"when\s+([\w\s]+?)\s+(?:happens?|moves?|changes?|fires?),?\s+([\w\s]+?)\s+(?:follows?|tends?\s+to\s+follow)",
        re.I,
    ),
]

# ── Learning pattern markers ──────────────────────────────────────────────────
_LEARNED_MARKERS = re.compile(
    r"(I (?:learned|discovered|noticed|found|realized)|interesting connection|I have a theory|"
    r"I'm learning that|discovered that|I noticed that)",
    re.I,
)

# ── Failure pattern markers ───────────────────────────────────────────────────
_WRONG_MARKERS = re.compile(
    r"(I (?:was wrong|got wrong|missed)|didn't work|didn't pan out|it moved the other way|"
    r"wrong direction|failed to|I thought it would)",
    re.I,
)

# ── Watching markers ─────────────────────────────────────────────────────────
_WATCHING_MARKERS = re.compile(
    r"(I'?m watching|still watching|I'?m (?:still )?looking at|"
    r"my attention (?:keeps |going back to)|something is forming|"
    r"not ready to call|I noticed something)",
    re.I,
)

# ── Section headings ─────────────────────────────────────────────────────────
_SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$", re.M)


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class NarrativeInsight:
    insight_type: str   # "ticker_call", "causal_claim", "watching", "learned", "wrong"
    content: str        # The extracted text
    ticker: str         # If applicable, else ""
    direction: str      # "bullish", "bearish", or ""
    confidence: float   # Mapped from language (0.0–1.0)
    source_section: str # Which section heading, e.g. "WHAT I LEARNED"
    extracted_at: str   # ISO timestamp


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_direction(text: str) -> str:
    b = len(_BULLISH_WORDS.findall(text))
    br = len(_BEARISH_WORDS.findall(text))
    if b > br:
        return "bullish"
    if br > b:
        return "bearish"
    return ""


def _detect_confidence(text: str) -> float:
    for pattern, score in _CONFIDENCE_MAP:
        if pattern.search(text):
            return score
    return 0.55  # Default: moderate confidence when no language matched


def _extract_tickers(text: str) -> list[str]:
    return [m for m in _TICKER_RE.findall(text) if m not in _COMMON_WORDS]


def _current_section(pos: int, section_spans: list[tuple[int, str]]) -> str:
    """Return the most recent section heading before character position `pos`."""
    current = "GENERAL"
    for start, heading in section_spans:
        if start <= pos:
            current = heading
        else:
            break
    return current


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-level chunks."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\n•*-]+", text) if s.strip()]


# ── Main class ────────────────────────────────────────────────────────────────

class NarrativeFeedback:
    """Extracts structured insights from daily narrative text and feeds them
    back into MIDGE's learning systems (SharedAttention + WorldModel)."""

    def __init__(
        self,
        shared_attention=None,
        world_model=None,
    ):
        self._sa = shared_attention
        self._wm = world_model

    # ── Extraction ───────────────────────────────────────────────────────────

    def extract_insights(self, narrative_text: str) -> list[NarrativeInsight]:
        """Parse narrative_text and return a list of NarrativeInsight records.

        Extraction is purely regex/keyword — no LLM, no network calls.
        Fast and deterministic.
        """
        insights: list[NarrativeInsight] = []
        now_iso = datetime.now().isoformat()

        # Build section index (position → heading) for attribution
        section_spans: list[tuple[int, str]] = []
        for m in _SECTION_RE.finditer(narrative_text):
            section_spans.append((m.start(), m.group(1).strip().upper()))

        sentences = _split_sentences(narrative_text)

        for sentence in sentences:
            if not sentence:
                continue
            # Find sentence position in original text for section lookup
            pos = narrative_text.find(sentence)
            section = _current_section(pos, section_spans)

            # ── 1. Causal claims ─────────────────────────────────────────
            _is_causal = any(cpat.search(sentence) for cpat in _CAUSAL_PATTERNS)
            if _is_causal:
                confidence = _detect_confidence(sentence)
                insights.append(NarrativeInsight(
                    insight_type="causal_claim",
                    content=sentence,
                    ticker="",
                    direction="",
                    confidence=confidence,
                    source_section=section,
                    extracted_at=now_iso,
                ))

            # ── 2. Watching items ────────────────────────────────────────
            elif _WATCHING_MARKERS.search(sentence):
                tickers = _extract_tickers(sentence)
                direction = _detect_direction(sentence)
                confidence = _detect_confidence(sentence)
                for ticker in tickers[:2]:  # cap at 2 tickers per sentence
                    insights.append(NarrativeInsight(
                        insight_type="watching",
                        content=sentence,
                        ticker=ticker,
                        direction=direction,
                        confidence=confidence,
                        source_section=section,
                        extracted_at=now_iso,
                    ))
                if not tickers:
                    insights.append(NarrativeInsight(
                        insight_type="watching",
                        content=sentence,
                        ticker="",
                        direction=direction,
                        confidence=confidence,
                        source_section=section,
                        extracted_at=now_iso,
                    ))

            # ── 3. Learning observations ─────────────────────────────────
            elif _LEARNED_MARKERS.search(sentence):
                insights.append(NarrativeInsight(
                    insight_type="learned",
                    content=sentence,
                    ticker="",
                    direction="",
                    confidence=_detect_confidence(sentence),
                    source_section=section,
                    extracted_at=now_iso,
                ))

            # ── 4. Failure observations ──────────────────────────────────
            elif _WRONG_MARKERS.search(sentence):
                tickers = _extract_tickers(sentence)
                direction = _detect_direction(sentence)
                insights.append(NarrativeInsight(
                    insight_type="wrong",
                    content=sentence,
                    ticker=tickers[0] if tickers else "",
                    direction=direction,
                    confidence=_detect_confidence(sentence),
                    source_section=section,
                    extracted_at=now_iso,
                ))

            # ── 5. Ticker calls (direction + ticker in same sentence) ────
            else:
                tickers = _extract_tickers(sentence)
                direction = _detect_direction(sentence)
                if tickers and direction:
                    confidence = _detect_confidence(sentence)
                    for ticker in tickers[:3]:
                        insights.append(NarrativeInsight(
                            insight_type="ticker_call",
                            content=sentence,
                            ticker=ticker,
                            direction=direction,
                            confidence=confidence,
                            source_section=section,
                            extracted_at=now_iso,
                        ))

        logger.info(
            "NarrativeFeedback: extracted %d insights (%s causal, %s ticker_calls, "
            "%s watching, %s learned, %s wrong)",
            len(insights),
            sum(1 for i in insights if i.insight_type == "causal_claim"),
            sum(1 for i in insights if i.insight_type == "ticker_call"),
            sum(1 for i in insights if i.insight_type == "watching"),
            sum(1 for i in insights if i.insight_type == "learned"),
            sum(1 for i in insights if i.insight_type == "wrong"),
        )
        return insights

    # ── Feed back ────────────────────────────────────────────────────────────

    def feed_back(self, insights: list[NarrativeInsight]) -> None:
        """Push insights into learning systems and persist to JSONL.

        - ticker_calls + watching → SharedAttention.update_hot_ticker()
        - causal_claims (confidence > 0.6) → WorldModel.add_discovered_edge()
        - All insights → narrative_insights.jsonl
        """
        if not insights:
            return

        for insight in insights:
            try:
                if insight.insight_type in ("ticker_call", "watching") and insight.ticker:
                    self._push_to_shared_attention(insight)
                elif insight.insight_type == "causal_claim" and insight.confidence > 0.60:
                    self._push_to_world_model(insight)
            except Exception:
                logger.debug(
                    "NarrativeFeedback feed_back error for insight %s",
                    insight.insight_type,
                    exc_info=True,
                )

        self._persist(insights)

    # ── Internal pushers ─────────────────────────────────────────────────────

    def _push_to_shared_attention(self, insight: NarrativeInsight) -> None:
        if self._sa is None:
            return
        reason = (
            f"narrative:{insight.insight_type}:{insight.source_section.lower()}"
        )
        self._sa.update_hot_ticker(
            ticker=insight.ticker,
            reason=reason,
            confidence=insight.confidence,
            domains=1,  # Letter is a synthesis — counts as 1 meta-domain
        )
        logger.debug(
            "SharedAttention updated: %s (%.2f) from narrative",
            insight.ticker,
            insight.confidence,
        )

    def _push_to_world_model(self, insight: NarrativeInsight) -> None:
        """Parse causal claim text and add a discovered edge to WorldModel.

        Handles patterns like:
          "X leads Y by N days"
          "X precedes Y by N days"
          "when X happens, Y follows"
        """
        if self._wm is None:
            return

        text = insight.content

        # Try lag-bearing patterns first
        for cpat in _CAUSAL_PATTERNS[:2]:
            m = cpat.search(text)
            if m:
                cause = m.group(1).strip().lower().replace(" ", "_")
                effect = m.group(2).strip().lower().replace(" ", "_")
                lag = float(m.group(3)) if m.lastindex and m.lastindex >= 3 else 5.0
                # Trim to reasonable domain-level strings
                cause = cause[:60]
                effect = effect[:60]
                self._wm.add_discovered_edge(
                    cause=cause,
                    effect=effect,
                    strength=insight.confidence,
                    lag_days=lag,
                    direction="neutral",
                    evidence="narrative_synthesis",
                )
                logger.debug(
                    "WorldModel edge added: %s → %s (lag=%.1fd, strength=%.2f)",
                    cause, effect, lag, insight.confidence,
                )
                return

        # Fallback: "when X, Y follows" — no lag info, use default 5d
        m = _CAUSAL_PATTERNS[2].search(text)
        if m:
            cause = m.group(1).strip().lower().replace(" ", "_")[:60]
            effect = m.group(2).strip().lower().replace(" ", "_")[:60]
            self._wm.add_discovered_edge(
                cause=cause,
                effect=effect,
                strength=insight.confidence,
                lag_days=5.0,
                direction="neutral",
                evidence="narrative_synthesis",
            )
            logger.debug(
                "WorldModel edge added (no-lag): %s → %s (strength=%.2f)",
                cause, effect, insight.confidence,
            )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist(self, insights: list[NarrativeInsight]) -> None:
        """Append all insights to narrative_insights.jsonl."""
        try:
            _INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_INSIGHTS_PATH, "a", encoding="utf-8") as f:
                for insight in insights:
                    f.write(json.dumps(asdict(insight)) + "\n")
            logger.info(
                "NarrativeFeedback: persisted %d insights to %s",
                len(insights),
                _INSIGHTS_PATH,
            )
        except Exception:
            logger.debug("NarrativeFeedback: could not persist insights", exc_info=True)
