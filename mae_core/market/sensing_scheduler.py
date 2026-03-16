"""Sensing scheduler mixin — Thompson-guided and round-robin fetch scheduling.

Provides _launch_next_fetch, _launch_thompson_guided, _launch_round_robin.
Mixed into MarketSensingHook via SensingSchedulerMixin.
"""

from __future__ import annotations

import logging
import random
import time

logger = logging.getLogger("midge.market.sensing")


class SensingSchedulerMixin:
    """Mixin providing Thompson-guided and round-robin fetch scheduling.

    Requires the host class to have:
      self._priority_requests, self._pending_futures, self._market_clock,
      self._thompson_sampler, self._fetch_queue, self._total_fetches,
      self._executor, self._fetch_source
    Plus module-level constants from sensing_hook:
      SOURCE_ROTATION, _ROTATION_TO_THOMPSON, _DOMAIN_TO_SOURCES
    """

    def _launch_next_fetch(self):
        """Fill up to 8 concurrent fetch slots using Thompson-guided selection.

        When a Thompson sampler is available, sources are selected
        probabilistically: each source draws from its Beta distribution,
        then the highest-scoring source is picked. This naturally
        balances exploration (new sources with wide posteriors) and
        exploitation (proven sources with high means). Falls back to
        round-robin when no sampler is available.
        """
        from mae_core.market.sensing_constants import SOURCE_ROTATION, _DOMAIN_TO_SOURCES

        # Clean up expired priority requests so stale entries don't accumulate.
        try:
            now = time.time()
            expired = [t for t, entry in self._priority_requests.items()
                       if now > entry.get("expires", 0)]
            for ticker in expired:
                del self._priority_requests[ticker]
        except Exception:
            logger.debug("Priority request cleanup failed", exc_info=True)

        # Fill available slots — scaled by circadian activity (default 8)
        _max_concurrent = max(3, int(12 * getattr(self, "_circadian_scale", 1.0)))
        slots = _max_concurrent - len(self._pending_futures)
        if slots <= 0:
            return

        # Identify eligible sources (not already in-flight)
        # Filter by market clock availability (if available)
        if self._market_clock is not None:
            available = set(self._market_clock.get_available_sources())
            eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures and s in available]
            if not eligible:
                # Fall back to always-available sources if nothing matches
                eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures]
        else:
            eligible = [s for s in SOURCE_ROTATION if s not in self._pending_futures]
        if not eligible:
            return

        if self._thompson_sampler is not None:
            self._launch_thompson_guided(eligible, slots)
        else:
            self._launch_round_robin(slots)

    def _launch_thompson_guided(self, eligible: list, slots: int):
        """Select sources via Thompson sampling draws.

        Priority polling: if _priority_requests is non-empty (and within the
        50-entry cap), sources that serve the needed domains receive a 2x boost
        on their sampled score (capped at 1.0).  This steers sensing toward
        the missing evidence for a developing convergence without blocking any
        other source entirely — it is a soft, probabilistic nudge.
        """
        from mae_core.market.sensing_constants import _ROTATION_TO_THOMPSON, _DOMAIN_TO_SOURCES

        # Build a set of boosted source names from active priority requests.
        # Only apply boost when the queue is within the 50-entry cap.
        boosted_sources: set = set()
        try:
            if 0 < len(self._priority_requests) <= 50:
                for entry in self._priority_requests.values():
                    for domain in entry.get("domains_needed", []):
                        for src in _DOMAIN_TO_SOURCES.get(domain, []):
                            boosted_sources.add(src)
        except Exception:
            logger.debug("Priority boost computation failed", exc_info=True)

        # Forced exploration: every 10 cycles, guarantee 2 slots for sources
        # that haven't been fetched recently (starvation prevention for new sources)
        _forced: list = []
        _fetch_counts = getattr(self, "_source_fetch_counts", {})
        if not hasattr(self, "_source_fetch_counts"):
            self._source_fetch_counts = {}
            _fetch_counts = self._source_fetch_counts
        _min_fetches = min(_fetch_counts.get(s, 0) for s in eligible) if eligible else 0
        _starved = [s for s in eligible if _fetch_counts.get(s, 0) <= _min_fetches]
        if _starved and slots >= 4:
            import random as _rng
            _rng.shuffle(_starved)
            _forced = _starved[:2]  # Reserve 2 slots for least-fetched sources
            eligible = [s for s in eligible if s not in _forced]
            slots -= len(_forced)

        # Draw from each source's Beta distribution
        scored = []
        for source in eligible:
            thompson_key = _ROTATION_TO_THOMPSON.get(source)
            if thompson_key is None:
                # Unknown source — use uniform prior (maximally uncertain)
                score = random.betavariate(1.0, 1.0)
            else:
                try:
                    dist = self._thompson_sampler.get_distribution(thompson_key)
                    if dist is not None and hasattr(dist, "alpha") and hasattr(dist, "beta"):
                        score = random.betavariate(
                            max(dist.alpha, 0.01), max(dist.beta, 0.01)
                        )
                    else:
                        # No distribution yet — use wide prior (encourages exploration)
                        score = random.betavariate(1.0, 1.0)
                except Exception:
                    score = random.betavariate(1.0, 1.0)

            # Apply priority boost: multiply by 2, cap at 1.0
            if source in boosted_sources:
                score = min(1.0, score * 2.0)

            # Weekend/overnight: crypto is the only live market — boost 3x
            try:
                _mc = getattr(self, "_market_clock", None)
                if _mc is not None:
                    _session = _mc.get_session()
                    if _session in ("weekend", "overnight", "pre_market"):
                        _crypto_sources = {"crypto_prices", "crypto_exchange", "binance_funding", "crypto_fear_greed"}
                        if source in _crypto_sources:
                            score = min(1.0, score * 3.0)
            except Exception:
                pass

            scored.append((score, source))

        # Sort descending by score, pick top N
        scored.sort(reverse=True)
        # Prepend forced-exploration sources so they always get dispatched
        _selected = [(1.0, s) for s in _forced] + scored[:slots]
        _cb = getattr(self, "_circuit_breaker", None)
        for _, source in _selected:
            if _cb is not None and not _cb.can_call(source):
                continue  # Circuit breaker blocking this source
            self._total_fetches += 1
            _fetch_counts[source] = _fetch_counts.get(source, 0) + 1
            logger.info(
                "Market sensing: Thompson-guided fetch [%s] (cycle %d)",
                source, self._total_fetches,
            )
            self._pending_futures[source] = self._executor.submit(
                self._fetch_source, source
            )

    def _launch_round_robin(self, slots: int):
        """Fallback: simple round-robin selection."""
        from mae_core.market.sensing_constants import SOURCE_ROTATION

        attempts = 0
        launched = 0
        while launched < slots and attempts < len(SOURCE_ROTATION):
            source = self._fetch_queue[0]
            self._fetch_queue.rotate(-1)
            attempts += 1

            if source in self._pending_futures:
                continue

            _cb = getattr(self, "_circuit_breaker", None)
            if _cb is not None and not _cb.can_call(source):
                continue  # Circuit breaker blocking this source

            self._total_fetches += 1
            logger.info(
                "Market sensing: launching async fetch [%s] (cycle %d)",
                source, self._total_fetches,
            )
            self._pending_futures[source] = self._executor.submit(
                self._fetch_source, source
            )
            launched += 1
