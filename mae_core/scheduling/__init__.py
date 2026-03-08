"""Scheduling package — organism-aware wall-clock task dispatch.

Exports:
    InhabitantScheduler: daemon thread that dispatches registered systems
    on their own wall-clock cadences.
"""

from .inhabitant_scheduler import InhabitantScheduler

__all__ = ["InhabitantScheduler"]
