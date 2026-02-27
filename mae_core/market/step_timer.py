"""Per-operation step timing for MIDGE market hooks.

Law 8 applied to performance: MIDGE watches her own metabolism.
Tracks percentile latencies per operation so bottlenecks surface
from real data rather than theoretical analysis.

Wired into bootstrap/market.py. Registered with SomaticMap + HolonProxy.
"""

import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Dict


class StepTimer:
    """Lightweight per-operation step latency tracker.

    Usage::

        timer = StepTimer()
        with timer.track("convergence_check"):
            alerts = alerter.check_convergence()
        print(timer.get_statistics())
    """

    def __init__(self, max_samples: int = 1000):
        self._timings: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._max_samples = max_samples

    @contextmanager
    def track(self, operation: str):
        """Time a block and record the duration under *operation*."""
        t0 = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - t0
            self._timings[operation].append(elapsed)

    def get_statistics(self) -> dict:
        """Return p50/p95/max/count for each tracked operation."""
        stats: Dict[str, dict] = {}
        for op, times in self._timings.items():
            if not times:
                continue
            arr = sorted(times)
            n = len(arr)
            stats[op] = {
                "p50_ms": arr[n // 2] * 1000,
                "p95_ms": arr[int(n * 0.95)] * 1000,
                "max_ms": arr[-1] * 1000,
                "count": n,
            }
        return stats

    def reset(self):
        """Clear all recorded timings."""
        self._timings.clear()
