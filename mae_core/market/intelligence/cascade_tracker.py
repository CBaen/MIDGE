"""Cascade Tracker — watches causal chains unfold as dominoes confirm.

When a convergence alert fires with ripple_effects (from WorldModel),
this tracker registers the predicted cascade. As subsequent signals
arrive on predicted tickers in the predicted direction, links are
confirmed — strengthening the thesis that remaining dominoes will fall.

This is the core of MIDGE's inevitability tracking. Not just predicting
cascades, but watching them unfold and strengthening confidence as each
domino confirms the chain.

Biological analogy: predictive coding. The brain generates predictions
about what sensory input should arrive. When it does arrive, the model
is reinforced. When it doesn't, error signals update the model.
"""

import logging
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

    def __init__(self, world_model=None, event_bus=None, max_chains: int = 50):
        self._world_model = world_model
        self._bus = event_bus
        self._max_chains = max_chains
        self._active_chains: Dict[str, dict] = {}

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
        if not ripple_effects or alert_id in self._active_chains:
            return False

        self._active_chains[alert_id] = {
            "alert_id": alert_id,
            "trigger": trigger,
            "direction": direction,
            "registered_at": time.time(),
            "links": [{
                "ticker": r.get("ticker", ""),
                "predicted_direction": r.get("direction", "neutral"),
                "predicted_lag_days": r.get("lag_days", 0),
                "strength": r.get("strength", 0),
                "status": "pending",
                "confirmed_at": None,
            } for r in ripple_effects if r.get("ticker")],
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

    def check_signal(self, ticker: str, direction: str) -> List[dict]:
        """Check if a signal confirms any active cascade link.

        Called on every signal ingestion. When a ticker+direction matches
        a predicted ripple, that domino is confirmed.

        Returns:
            List of confirmation events (one per chain that matched).
        """
        confirmations = []

        for chain_id, chain in self._active_chains.items():
            for link in chain["links"]:
                if (link["status"] == "pending"
                        and link["ticker"] == ticker
                        and link["predicted_direction"] == direction):
                    # DOMINO CONFIRMED
                    link["status"] = "confirmed"
                    link["confirmed_at"] = time.time()

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

                    confirmation = {
                        "chain_id": chain_id,
                        "trigger": chain["trigger"],
                        "confirmed_ticker": ticker,
                        "confirmed_direction": direction,
                        "confirmed_count": confirmed,
                        "total_links": total,
                        "remaining": [{
                            "ticker": l["ticker"],
                            "direction": l["predicted_direction"],
                            "lag_days": l["predicted_lag_days"],
                            "strength": l["strength"],
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
        """Return tracker statistics."""
        total_links = 0
        confirmed = 0
        pending = 0

        for chain in self._active_chains.values():
            for link in chain["links"]:
                total_links += 1
                if link["status"] == "confirmed":
                    confirmed += 1
                elif link["status"] == "pending":
                    pending += 1

        return {
            "active_chains": len(self._active_chains),
            "total_links": total_links,
            "confirmed_links": confirmed,
            "pending_links": pending,
            "confirmation_rate": round(confirmed / max(total_links, 1), 3),
        }
