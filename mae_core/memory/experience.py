"""Experience dataclass - shared data structure for all memory subsystems."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Experience:
    """Single agent experience for storage and replay.

    Used by episodic memory, generative replay, and consolidation.
    State/next_state are numpy arrays (74-128 dims for Mae agents).
    """

    state: np.ndarray
    action: Any  # int for discrete, np.ndarray for continuous
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: Optional[float] = None
