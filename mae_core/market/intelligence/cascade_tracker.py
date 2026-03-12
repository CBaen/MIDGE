"""Cascade Tracker — watches causal chains unfold as dominoes confirm.

When a convergence alert fires with ripple_effects (from WorldModel),
this tracker registers the predicted cascade. As subsequent signals
arrive on predicted tickers in the predicted direction, links are
confirmed — strengthening the thesis that remaining dominoes will fall.

This is the core of MIDGE's inevitability tracking. Not just predicting
cascades, but watching them unfold and strengthening confidence as each
domino confirms the chain.

Stage-gating: links are grouped into temporal stages by predicted lag.
Stage 0 (shortest lag) is always watchable. Stage N only opens when
stage N-1 has at least one confirmed link. This enforces causal ordering —
energy can't skip dominoes.

Biological analogy: predictive coding. The brain generates predictions
about what sensory input should arrive. When it does arrive, the model
is reinforced. When it doesn't, error signals update the model.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class CascadeTracker:
    """Track active causal chains and confirm links as dominoes fall.

    Lifecycle:
    1. Convergence alert fires with ripple_effects → register_cascade()
    2. Signal arrives on predicted ticker → check_signal()
       - If match: mark link confirmed, call world_model.record_outcome(True)
       - Emit CH_CASCADE_CONFIRMED with remaining dominoes
    3. Chain ages past max_age_days → expire_stale()
       - Unconfirmed links: world_model.record_outcome(False)
    """

    def __init__(
        self,
        world_model=None,
        event_bus=None,
        max_chains: int = 50,
        stage_tolerance_days: float = 2.0,
    ):
        self._world_model = world_model
        self._bus = event_bus
        self._max_chains = max_chains
        self._stage_tolerance_days = stage_tolerance_days
        self._active_chains: Dict[str, dict] = {}
        self._chains_lock = threading.Lock()

    def register_cascade(
        self,
        alert_id: str,
        trigger: str,
        ripple_effects: List[dict],
        direction: str,
    ) -> bool:
        """Register a predicted cascade from a convergence alert.

        Args:
            alert_id: Convergence alert ID (dedup key)
            trigger: WorldModel trigger node (e.g. 'crude_price_spike')
            ripple_effects: List of {ticker, direction, strength, lag_days, ...}
            direction: Alert direction (bullish/bearish)

        Returns:
            True if registered, False if duplicate or no effects.
        """
        with self._chains_lock:
            return self._register_cascade_locked(alert_id, trigger, ripple_effects, direction)

    def _register_cascade_locked(
        self,
        alert_id: str,
        trigger: str,
        ripple_effects: List[dict],
        direction: str,
    ) -> bool:
        """Inner — caller must hold self._chains_lock."""
        if not ripple_effects or alert_id in self._active_chains:
            return False

        # Build links sorted by predicted lag (temporal ordering)
        raw_links = [{
            "ticker": r.get("ticker", ""),
            "predicted_direction": r.get("direction", "neutral"),
            "predicted_lag_days": r.get("lag_days", 0),
            "strength": r.get("strength", 0),
            "status": "pending",
            "confirmed_at": None,
            "stage": 0,  # assigned below
        } for r in ripple_effects if r.get("ticker")]

        if not raw_links:
            return False

        # Sort by predicted lag and assign stages
        raw_links.sort(key=lambda l: l["predicted_lag_days"])
        self._assign_stages(raw_links)

        self._active_chains[alert_id] = {
            "alert_id": alert_id,
            "trigger": trigger,
            "direction": direction,
            "registered_at": time.time(),
            "links": raw_links,
        }

        # Evict oldest if over capacity
        while len(self._active_chains) > self._max_chains:
            oldest = min(
                self._active_chains,
                key=lambda k: self._active_chains[k]["registered_at"],
            )
            del self._active_chains[oldest]

        logger.info(
            "Cascade registered: %s → %d downstream links from trigger '%s'",
            alert_id, len(self._active_chains[alert_id]["links"]), trigger,
        )
        return True

    def _assign_stages(self, links: list) -> None:
        """Assign stage numbers to sorted links based on lag proximity.

        Links within stage_tolerance_days of each other share a stage.
        This groups parallel effects (e.g., crude → XOM and CVX at similar
        lags) while separating sequential ones (crude → XOM at 1d vs
        airline stocks at 7d).
        """
        if not links:
            return

        stage = 0
        stage_anchor = links[0]["predicted_lag_days"]
        links[0]["stage"] = 0

        for i in range(1, len(links)):
            lag = links[i]["predicted_lag_days"]
            if lag - stage_anchor > self._stage_tolerance_days:
                stage += 1
                stage_anchor = lag
            links[i]["stage"] = stage

    def _get_watchable_stage(self, chain: dict) -> int:
        """Return the highest stage currently watchable for a chain.

        Stage 0 is always watchable. Stage N opens when stage N-1 has
        at least one confirmed link. Returns the max stage where all
        prior stages have confirmations.
        """
        if not chain["links"]:
            return 0

        max_stage = max(l["stage"] for l in chain["links"])
        watchable = 0

        for stage in range(max_stage + 1):
            if stage == 0:
                watchable = 0
                continue
            # Stage N watchable only if stage N-1 has a confirmation
            prev_confirmed = any(
                l["status"] == "confirmed"
                for l in chain["links"]
                if l["stage"] == stage - 1
            )
            if prev_confirmed:
                watchable = stage
            else:
                break

        return watchable

    def check_signal(self, ticker: str, direction: str) -> List[dict]:
        """Check if a signal confirms any active cascade link.

        Called on every signal ingestion. When a ticker+direction matches
        a predicted ripple in a watchable stage, that domino is confirmed.
        Links in future stages (where prior stages lack confirmations) are
        skipped — energy can't jump ahead in the causal chain.

        Returns:
            List of confirmation events (one per chain that matched).
        """
        confirmations = []

        for chain_id, chain in self._active_chains.items():
            watchable = self._get_watchable_stage(chain)

            for link in chain["links"]:
                if (link["status"] == "pending"
                        and link["stage"] <= watchable
                        and link["ticker"] == ticker
                        and link["predicted_direction"] == direction):
                    # DOMINO CONFIRMED
                    link["status"] = "confirmed"
                    link["confirmed_at"] = time.time()

                    # Temporal energy: actual lag vs predicted lag
                    # energy_ratio > 1.0 → energy moving faster than predicted
                    # energy_ratio < 1.0 → energy decaying (slower than predicted)
                    actual_lag_days = (link["confirmed_at"] - chain["registered_at"]) / 86400
                    link["actual_lag_days"] = actual_lag_days
                    predicted_lag = max(link.get("predicted_lag_days", 0), 0.0)
                    energy_ratio = predicted_lag / max(actual_lag_days, 0.1)
                    link["energy_ratio"] = round(energy_ratio, 4)

                    # Feed back to WorldModel — strengthen the causal edge
                    if self._world_model is not None:
                        try:
                            self._world_model.record_outcome(
                                chain["trigger"], ticker, was_correct=True,
                            )
                        except Exception:
                            logger.debug("WorldModel outcome failed", exc_info=True)

                    confirmed = sum(1 for l in chain["links"] if l["status"] == "confirmed")
                    total = len(chain["links"])
                    remaining = [
                        l for l in chain["links"] if l["status"] == "pending"
                    ]

                    # Recompute watchable stage after this confirmation
                    new_watchable = self._get_watchable_stage(chain)
                    unlocked_stage = (
                        new_watchable > watchable
                    )

                    confirmation = {
                        "chain_id": chain_id,
                        "trigger": chain["trigger"],
                        "confirmed_ticker": ticker,
                        "confirmed_direction": direction,
                        "confirmed_count": confirmed,
                        "total_links": total,
                        "energy_ratio": link["energy_ratio"],
                        "actual_lag_days": round(actual_lag_days, 4),
                        "confirmed_stage": link["stage"],
                        "watchable_stage": new_watchable,
                        "unlocked_next_stage": unlocked_stage,
                        "remaining": [{
                            "ticker": l["ticker"],
                            "direction": l["predicted_direction"],
                            "lag_days": l["predicted_lag_days"],
                            "strength": l["strength"],
                            "stage": l["stage"],
                            "watchable": l["stage"] <= new_watchable,
                        } for l in remaining],
                    }
                    confirmations.append(confirmation)

                    # Publish confirmation event
                    if self._bus is not None:
                        try:
                            self._bus.publish(
                                "market.intel.cascade_confirmed", confirmation,
                            )
                        except Exception:
                            logger.debug("Cascade publish failed", exc_info=True)

                    logger.info(
                        "CASCADE CONFIRMED: %s %s (%d/%d links, trigger=%s, %d remaining)",
                        ticker, direction, confirmed, total,
                        chain["trigger"], len(remaining),
                    )

        return confirmations

    def expire_stale(self, max_age_days: float = 30.0) -> List[str]:
        """Expire chains older than max_age_days.

        Unconfirmed links are treated as misses — WorldModel edge
        strength is reduced. This prevents stale predictions from
        accumulating indefinitely.

        Returns:
            List of expired chain IDs.
        """
        now = time.time()
        expired = []

        for chain_id in list(self._active_chains):
            chain = self._active_chains[chain_id]
            age_days = (now - chain["registered_at"]) / 86400

            if age_days > max_age_days:
                # Mark unconfirmed links as misses
                for link in chain["links"]:
                    if link["status"] == "pending":
                        link["status"] = "expired"
                        if self._world_model is not None:
                            try:
                                self._world_model.record_outcome(
                                    chain["trigger"], link["ticker"],
                                    was_correct=False,
                                )
                            except Exception:
                                pass

                expired.append(chain_id)
                del self._active_chains[chain_id]

        if expired:
            logger.info("Expired %d stale cascade chains", len(expired))

        return expired

    def get_active_chains(self) -> Dict[str, dict]:
        """Return all active cascade chains."""
        return dict(self._active_chains)

    def get_statistics(self) -> dict:
        """Return tracker statistics.

        Includes mean_energy_ratio: average energy_ratio across all confirmed
        links in all active chains.  Values > 1.0 indicate that cascades are
        moving faster than the WorldModel predicted; < 1.0 means slower.
        """
        total_links = 0
        confirmed = 0
        pending = 0
        gated = 0
        energy_ratios = []

        for chain in self._active_chains.values():
            watchable = self._get_watchable_stage(chain)
            for link in chain["links"]:
                total_links += 1
                if link["status"] == "confirmed":
                    confirmed += 1
                    if "energy_ratio" in link:
                        energy_ratios.append(link["energy_ratio"])
                elif link["status"] == "pending":
                    pending += 1
                    if link["stage"] > watchable:
                        gated += 1

        mean_energy_ratio = (
            round(sum(energy_ratios) / len(energy_ratios), 4)
            if energy_ratios else None
        )

        return {
            "active_chains": len(self._active_chains),
            "total_links": total_links,
            "confirmed_links": confirmed,
            "pending_links": pending,
            "gated_links": gated,
            "confirmation_rate": round(confirmed / max(total_links, 1), 3),
            "mean_energy_ratio": mean_energy_ratio,
        }
