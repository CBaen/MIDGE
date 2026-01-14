#!/usr/bin/env python3
"""
engine.py - Trading Research Engine

Multi-LLM research engine:
- DeepSeek for continuous research (cheap, 24/7)
- Gemini for deep analysis (1M context)

Uses MIDGE's HTTP patterns for cross-platform compatibility.
"""

import os
import time
import requests
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Literal

from .storage import TradingVectorStore, SignalPayload, DECAY_RATES, SOURCE_RELIABILITY

# API Configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

# Chunk settings
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""
    name: str
    api_key: str
    model: str
    max_tokens: int
    cost_per_1k_input: float
    cost_per_1k_output: float


class TradingResearchEngine:
    """
    Multi-LLM research engine for trading intelligence.

    Usage:
        engine = TradingResearchEngine()
        result = engine.research(
            topic="RSI Indicator",
            question="Explain RSI calculation and divergence trading",
            domain="technical-analysis",
            use_llm="deepseek"  # or "gemini"
        )
    """

    # Domain-specific prompt templates
    DOMAIN_PROMPTS = {
        "technical-analysis": """You are a technical analysis research expert.
REQUIREMENTS:
1. BE EXHAUSTIVE - Cover every aspect: calculation formulas, settings, interpretation
2. INCLUDE SPECIFICS - Exact numbers, thresholds, historical examples
3. CITE SOURCES - Academic papers, industry standards, authoritative texts
4. PRACTICAL APPLICATION - Entry/exit rules, timeframes, confirmation signals
5. STATISTICAL RELIABILITY - Success rates where known, backtesting considerations
6. MINIMUM 2000 WORDS - Comprehensive coverage, no summarizing""",

        "institutional": """You are an institutional finance research expert.
REQUIREMENTS:
1. BE EXHAUSTIVE - Cover all relevant filings, patterns, players
2. INCLUDE DATA - Specific holdings, dates, dollar amounts
3. CITE SOURCES - SEC filings, academic studies, industry reports
4. TRACK RECORD - Historical accuracy, notable examples
5. TIMING PATTERNS - Lead times, disclosure delays, seasonal patterns
6. MINIMUM 2000 WORDS - Comprehensive coverage""",

        "options": """You are a derivatives and options research expert.
REQUIREMENTS:
1. BE EXHAUSTIVE - All strategies, Greeks, mechanics
2. INCLUDE FORMULAS - Exact calculations, pricing models
3. CITE SOURCES - Black-Scholes, academic papers, CBOE documentation
4. PRACTICAL APPLICATION - When to use each strategy, risk/reward
5. MARKET MAKER PERSPECTIVE - How MMs hedge, gamma exposure
6. MINIMUM 2000 WORDS - Comprehensive coverage""",

        "sentiment": """You are a sentiment analysis and alternative data expert.
REQUIREMENTS:
1. BE EXHAUSTIVE - All data sources, processing methods, signals
2. INCLUDE SPECIFICS - APIs, NLP models, accuracy rates
3. CITE SOURCES - Academic studies on sentiment prediction
4. SIGNAL QUALITY - Latency, decay rates, reliability
5. PRACTICAL APPLICATION - Filtering noise, weighting sources
6. MINIMUM 2000 WORDS - Comprehensive coverage""",

        "default": """You are a comprehensive research expert.
REQUIREMENTS:
1. BE EXHAUSTIVE - Use your full context capacity
2. INCLUDE SOURCES - Cite where information comes from
3. STRUCTURE CLEARLY - Headers, bullet points, numbered lists
4. INCLUDE SPECIFICS - Names, dates, numbers, formulas
5. COVER ALL ANGLES - Historical, current, future implications
6. MINIMUM 2000 WORDS - Do not truncate or summarize"""
    }

    def __init__(self):
        self.store = TradingVectorStore()
        self._deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self._gemini_key = os.environ.get("GOOGLE_API_KEY")

    def _build_prompt(self, question: str, topic: str, domain: str) -> str:
        """Build domain-optimized prompt."""
        domain_template = self.DOMAIN_PROMPTS.get(domain, self.DOMAIN_PROMPTS["default"])

        return f"""{domain_template}

TOPIC: {topic}
QUESTION: {question}

This research will be stored in a vector database for an AI trading intelligence system.
The more comprehensive and sourced your response, the more valuable it is.

BEGIN YOUR COMPREHENSIVE RESPONSE:"""

    def _query_deepseek(self, prompt: str, max_tokens: int = 4096) -> Optional[str]:
        """Query DeepSeek API."""
        if not self._deepseek_key:
            print("ERROR: DEEPSEEK_API_KEY not set")
            return None

        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {self._deepseek_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=180
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"DeepSeek API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"DeepSeek error: {e}")
            return None

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Query Gemini API."""
        if not self._gemini_key:
            print("ERROR: GOOGLE_API_KEY not set")
            return None

        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={self._gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 8192,
                        "temperature": 0.7
                    }
                },
                timeout=300
            )

            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Gemini API error: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    def _chunk_text(self, text: str) -> list:
        """Split text into chunks with overlap."""
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            paragraphs = text.split('\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(para) > CHUNK_SIZE:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunks.append(para[i:i + CHUNK_SIZE])
            elif len(current_chunk) + len(para) < CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        if not chunks and len(text) > 0:
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(text[i:i + CHUNK_SIZE])

        return chunks

    def research(
        self,
        topic: str,
        question: str,
        domain: str,
        tags: list = None,
        use_llm: Literal["deepseek", "gemini"] = "deepseek",
        signal_source: str = "research",
        symbol: str = None,
        store_result: bool = True,
        retries: int = 3,
        retry_delay: int = 10
    ) -> dict:
        """
        Execute research query and store in Qdrant.

        Args:
            topic: Research topic identifier
            question: The research question
            domain: Research domain (technical-analysis, institutional, options, sentiment)
            tags: Optional tags for filtering
            use_llm: Which LLM to use ("deepseek" or "gemini")
            signal_source: Signal type for decay calculation
            symbol: Optional stock symbol
            store_result: Whether to store in Qdrant
            retries: Number of retry attempts
            retry_delay: Seconds between retries

        Returns:
            dict with status, content, chunks_stored
        """
        if tags is None:
            tags = []

        print(f"Researching: {topic}")
        print(f"  Domain: {domain}")
        print(f"  LLM: {use_llm}")
        print(f"  Tags: {tags}")

        # Build enhanced prompt
        prompt = self._build_prompt(question, topic, domain)

        # Query LLM with retries
        content = None
        for attempt in range(retries):
            print(f"  Attempt {attempt + 1}: Querying {use_llm}...")

            if use_llm == "deepseek":
                content = self._query_deepseek(prompt)
            else:
                content = self._query_gemini(prompt)

            if content and len(content) > 100:
                print(f"  Got {len(content)} characters")
                break

            print(f"  Attempt {attempt + 1}: Empty response, retrying in {retry_delay}s...")
            time.sleep(retry_delay)

        if not content:
            return {
                "status": "error",
                "error": f"Failed to get response from {use_llm} after {retries} attempts",
                "chunks_stored": 0
            }

        # Store in Qdrant if requested
        chunks_stored = 0
        if store_result:
            print("  Chunking and storing...")
            chunks = self._chunk_text(content)
            print(f"  Created {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                payload = SignalPayload(
                    topic=topic,
                    symbol=symbol,
                    data_type="research",
                    signal_source=signal_source,
                    decay_rate=DECAY_RATES.get(signal_source, 0.01),
                    source_reliability=SOURCE_RELIABILITY.get(use_llm, 0.70),
                    domain=domain,
                    tags=tags,
                    content=chunk,
                    question=question,
                    sources=[use_llm],
                    chunk_index=i,
                    total_chunks=len(chunks)
                )

                point_id = self.store.store(payload)
                if point_id:
                    chunks_stored += 1
                    print(f"  Stored chunk {i + 1}/{len(chunks)}")

        print(f"  SUCCESS: Stored {chunks_stored} chunks")
        return {
            "status": "success",
            "topic": topic,
            "content": content,
            "content_length": len(content),
            "chunks_stored": chunks_stored,
            "llm_used": use_llm
        }

    def search(self, query: str, limit: int = 5, domain: str = None) -> list:
        """Search stored research with decay-adjusted scoring."""
        return self.store.search(query, limit=limit, filter_domain=domain, apply_decay=True)

    def get_stats(self) -> dict:
        """Get storage statistics."""
        return self.store.get_stats()


if __name__ == "__main__":
    engine = TradingResearchEngine()
    print(f"Stats: {engine.get_stats()}")

    # Test search
    results = engine.search("candlestick patterns", limit=3)
    for r in results:
        print(f"Score: {r['score']:.3f} - {r['payload'].get('topic', 'unknown')}")
