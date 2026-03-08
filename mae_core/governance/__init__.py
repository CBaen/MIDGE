"""Governance package — organism self-governance logging.

Exports:
    GovernanceLogger: EventBus subscriber that records governance events
    to an append-only JSONL file.
"""

from .governance_logger import GovernanceLogger

__all__ = ["GovernanceLogger"]
