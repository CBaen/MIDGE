"""
MIDGE Trading Intelligence System

Multi-agent stock prediction with:
- DeepSeek for continuous research (cheap, 24/7)
- Gemini for deep analysis (1M context)
- Qdrant for vector storage with decay metadata
- Specialist agents with weighted consensus
"""

from .engine import TradingResearchEngine
from .storage import TradingVectorStore

__all__ = ['TradingResearchEngine', 'TradingVectorStore']
