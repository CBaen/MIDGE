"""Per-operation step timing for MIDGE market hooks.

Law 8 applied to performance: MIDGE watches her own metabolism.
Tracks percentile latencies per operation so bottlenecks surface
from real data rather than theoretical analysis.

Wired into bootstrap/market.py. Registered with SomaticMap + HolonProxy.
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


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

    def save_session_stats(self, path) -> None:
        """Write current timing statistics to a JSON file with a timestamp.

        Non-critical — all exceptions are caught silently so a failing write
        never disrupts the step loop.
        """
        try:
            out_path = path if hasattr(path, "write_text") else __import__("pathlib").Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            payload = {
                "saved_at": datetime.now().isoformat(),
                "stats": self.get_statistics(),
            }
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, out_path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass
