"""Form 8-K Text Sentiment Analyzer via Ollama.

Enriches 8-K signals beyond rule-based item codes. Two "item 2.02"
filings can be wildly different — one announces record revenue, the
other warns of declining margins. This classifier reads the actual
text and assigns direction + confidence.

Uses Ollama (local LLM, no API key) for classification. Follows
the same pattern as ClusterDetector's Ollama embedding calls.
Degrades gracefully when Ollama is unavailable.

Usage:
    analyzer = Form8KSentimentAnalyzer()
    result = analyzer.classify(event_summary, item_code)
    # result.direction = "bullish" / "bearish" / "neutral"
    # result.confidence_modifier = -0.15 to +0.15
"""

import logging
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"  # Fast, good enough for classification
OLLAMA_TIMEOUT = 15  # seconds — don't block the step loop

# Classification prompt — structured output for reliable parsing
_CLASSIFICATION_PROMPT = """Classify this SEC Form 8-K filing event as bullish, bearish, or neutral for the stock price.

Item code: {item_code}
Event text: {text}

Respond with EXACTLY one line in this format:
DIRECTION: <bullish|bearish|neutral> CONFIDENCE: <low|medium|high>

Rules:
- Revenue/earnings beat, new contracts, positive guidance = bullish
- Revenue/earnings miss, lawsuits, executive departures, impairments = bearish
- Routine filings, amendments, compensation changes = neutral
- If uncertain, say neutral with low confidence"""


@dataclass
class SentimentResult:
    """Result of 8-K text sentiment analysis."""
    direction: str  # bullish, bearish, neutral
    confidence_modifier: float  # -0.15 to +0.15 adjustment to signal confidence
    raw_response: str = ""  # Ollama's response for audit
    source: str = "ollama"


class Form8KSentimentAnalyzer:
    """Classifies 8-K filing text via local Ollama LLM.

    Graceful degradation: if Ollama is down or model unavailable,
    returns None (caller falls back to rule-based item code).
    """

    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self._url = ollama_url
        self._model = model
        self._timeout = timeout
        self._available = None  # lazy check

    def classify(
        self,
        event_text: str,
        item_code: str = "",
    ) -> SentimentResult | None:
        """Classify 8-K event text for market sentiment.

        Args:
            event_text: The event summary or document text
            item_code: SEC item code (e.g., "2.02") for context

        Returns:
            SentimentResult or None if Ollama unavailable
        """
        if not event_text or len(event_text.strip()) < 10:
            return None

        if not self._check_available():
            return None

        prompt = _CLASSIFICATION_PROMPT.format(
            item_code=item_code or "unknown",
            text=event_text[:2000],  # Truncate to keep prompt reasonable
        )

        try:
            resp = requests.post(
                f"{self._url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 50},
                },
                timeout=self._timeout,
            )
            if resp.status_code != 200:
                logger.debug("Ollama returned %d for 8-K sentiment", resp.status_code)
                return None

            raw = resp.json().get("response", "").strip()
            return self._parse_response(raw)

        except requests.exceptions.ConnectionError:
            self._available = False
            logger.debug("Ollama not reachable for 8-K sentiment")
            return None
        except requests.exceptions.Timeout:
            logger.debug("Ollama timed out for 8-K sentiment")
            return None
        except Exception:
            logger.debug("8-K sentiment classification failed", exc_info=True)
            return None

    def _parse_response(self, raw: str) -> SentimentResult | None:
        """Parse Ollama's structured response into a SentimentResult."""
        raw_lower = raw.lower()

        # Extract direction
        if "bullish" in raw_lower:
            direction = "bullish"
        elif "bearish" in raw_lower:
            direction = "bearish"
        else:
            direction = "neutral"

        # Extract confidence level → modifier
        if "high" in raw_lower:
            modifier_magnitude = 0.15
        elif "medium" in raw_lower:
            modifier_magnitude = 0.08
        else:
            modifier_magnitude = 0.03

        # Direction determines sign of modifier
        if direction == "bullish":
            confidence_modifier = modifier_magnitude
        elif direction == "bearish":
            confidence_modifier = -modifier_magnitude
        else:
            confidence_modifier = 0.0

        return SentimentResult(
            direction=direction,
            confidence_modifier=confidence_modifier,
            raw_response=raw,
        )

    def _check_available(self) -> bool:
        """Lazy check if Ollama is reachable. Caches result."""
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(f"{self._url}/api/tags", timeout=3)
            self._available = resp.status_code == 200
        except Exception:
            self._available = False

        if not self._available:
            logger.debug("Ollama not available for 8-K sentiment analysis")

        return self._available

    def reset_availability(self):
        """Reset cached availability (e.g., after Docker restart)."""
        self._available = None
