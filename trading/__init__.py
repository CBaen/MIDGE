"""
MIDGE Trading Intelligence System

Self-improving stock prediction system with:
- DeepSeek for continuous research (cheap, 24/7)
- Gemini for deep analysis via subagent (1M context)
- Qdrant for vector storage with decay metadata
- Curiosity-driven pattern discovery (find patterns others miss)
- Self-modifying learning parameters
- Gamification for intrinsic motivation

Architecture:
- engine.py         : TradingResearchEngine (research → vector storage)
- storage.py        : TradingVectorStore + SignalPayload schemas
- edge/            : Edge discovery (options flow, Form 4, on-chain)
- agents/          : Specialist agents (technical, sentiment, institutional, risk)
- self_improve/    : Dual-agent self-modification (researcher + developer)
- gamification/    : Achievement system, curiosity scores
- knowledge/       : Self-awareness files (capabilities, limitations, goals)
- config/          : Learning parameters (self-modifiable)
- logs/            : Evolution history, predictions, outcomes
"""

from .engine import TradingResearchEngine
from .storage import TradingVectorStore

__all__ = ['TradingResearchEngine', 'TradingVectorStore']
