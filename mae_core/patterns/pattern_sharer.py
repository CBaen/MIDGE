"""PatternSharer - Triadic pattern communication.

Agents share pattern discoveries with their triad-mates via GNN
KNOWLEDGE_SHARE messages. When 2 of 3 agents report the same pattern
domain, it becomes a triadic consensus: a tissue-level truth.

Biological analogy: Cells in a tissue communicate via gap junctions
and local signaling. When enough cells agree on a stimulus, the
tissue responds as a unit.

The 2/3 agreement rule: the same Rule of Three that governs Mae's
fractal structure also governs consensus. Two witnesses out of three
are enough to elevate a signal from REACTIVE to CORRELATED.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)

logger = logging.getLogger(__name__)

# Minimum agreement ratio for triadic consensus
CONSENSUS_RATIO = 2 / 3  # 2 of 3 agents


class PatternSharer:
    """Shares PatternSense signals with triad-mates, detects consensus.

    Lifecycle (called from MycelialAgent._communicate):
    1. share(signals) -- send own signals to triad-mates
    2. receive_and_correlate() -- check inbox for peer signals, detect consensus
    """

    def __init__(
        self,
        agent_id: str,
        holon_registry: Any,
        gnn_communicator: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._agent_id = agent_id
        self._holon_registry = holon_registry
        self._gnn_communicator = gnn_communicator
        self._event_bus = event_bus
        self._inbox: dict[str, list[dict]] = defaultdict(list)  # peer_id -> signals
        self._total_shares = 0
        self._total_consensus = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def get_triad_mates(self) -> list[str]:
        """Return the IDs of this agent's triad-mates."""
        if self._holon_registry is None:
            return []
        get_peers = getattr(self._holon_registry, "get_peers", None)
        if get_peers is None:
            return []
        return get_peers(self._agent_id)

    def share(self, signals: list[PatternSignal]) -> None:
        """Send own PatternSense signals to triad-mates via GNN.

        Only sends when there are signals worth sharing. Uses
        KNOWLEDGE_SHARE message type with pattern data.
        """
        if not signals:
            return

        peers = self.get_triad_mates()
        if not peers:
            return

        # Serialize signal data for transmission
        payload = {
            "type": "pattern_share",
            "sender": self._agent_id,
            "domains": [s.domain.value for s in signals],
            "signals": [
                {
                    "domain": s.domain.value,
                    "salience": s.salience,
                    "confidence": s.confidence,
                    "description": s.description,
                }
                for s in signals
            ],
        }

        # Send via GNN if available
        if self._gnn_communicator is not None:
            send_fn = getattr(self._gnn_communicator, "send_message", None)
            if send_fn is not None:
                try:
                    send_fn(
                        sender_id=self._agent_id,
                        content=payload,
                        message_type="KNOWLEDGE_SHARE",
                        target_ids=peers,
                    )
                    self._total_shares += 1
                except Exception:
                    logger.debug(
                        "PatternSharer %s: GNN share failed",
                        self._agent_id,
                        exc_info=True,
                    )

    def receive_peer_signal(self, peer_id: str, signal_data: dict) -> None:
        """Receive a pattern share from a peer (called from GNN message handler)."""
        self._inbox[peer_id].append(signal_data)

    def receive_and_correlate(
        self, own_signals: list[PatternSignal],
    ) -> list[PatternSignal]:
        """Check for triadic consensus: 2/3 agents reporting same domain.

        Compares own signals with peer signals in inbox. When 2+ agents
        (including self) report the same domain, produces a CORRELATED
        signal representing tissue-level agreement.

        Returns list of consensus signals. Clears inbox after processing.
        """
        peers = self.get_triad_mates()
        triad_size = len(peers) + 1  # Include self

        if triad_size < 2:
            self._clear_inbox()
            return []

        # Count domain votes: self + peers
        domain_votes: dict[str, set[str]] = defaultdict(set)  # domain -> voter_ids

        # Self votes
        for sig in own_signals:
            domain_votes[sig.domain.value].add(self._agent_id)

        # Peer votes from inbox
        for peer_id, signal_list in self._inbox.items():
            for sig_data in signal_list:
                domains = sig_data.get("domains", [])
                for d in domains:
                    domain_votes[d].add(peer_id)

        # Detect consensus
        consensus_signals: list[PatternSignal] = []
        min_votes = max(2, int(triad_size * CONSENSUS_RATIO))

        for domain_str, voters in domain_votes.items():
            if len(voters) >= min_votes:
                try:
                    domain = PatternDomain(domain_str)
                except ValueError:
                    continue

                # Determine triad parent ID for source_system
                triad_id = self._get_triad_id()

                consensus_sig = PatternSignal(
                    source_system=f"triad:{triad_id}",
                    domain=domain,
                    form=PatternForm.CORRELATED,
                    confidence=min(1.0, 0.5 + 0.1 * len(voters)),
                    salience=min(0.7, 0.3 + 0.1 * len(voters)),
                    description=(
                        f"Triadic consensus: {domain_str} confirmed by "
                        f"{len(voters)}/{triad_size} agents"
                    ),
                    evidence={
                        "domain": domain_str,
                        "voters": sorted(voters),
                        "triad_size": triad_size,
                        "agreement_ratio": len(voters) / triad_size,
                    },
                )
                consensus_signals.append(consensus_sig)
                self._total_consensus += 1

                # Publish to EventBus for upward propagation
                if self._event_bus is not None:
                    publish = getattr(self._event_bus, "publish", None)
                    if publish is not None:
                        try:
                            publish(
                                "pattern.triadic_correlation",
                                consensus_sig.evidence,
                            )
                        except Exception:
                            pass

        self._clear_inbox()
        return consensus_signals

    def _get_triad_id(self) -> str:
        """Get the parent holon ID for this agent's triad."""
        if self._holon_registry is None:
            return "unknown"
        get_parent = getattr(self._holon_registry, "get_parent", None)
        if get_parent is None:
            return "unknown"
        parent = get_parent(self._agent_id)
        return parent if parent is not None else "unknown"

    def _clear_inbox(self) -> None:
        self._inbox.clear()

    def get_statistics(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "total_shares": self._total_shares,
            "total_consensus": self._total_consensus,
            "inbox_peers": len(self._inbox),
        }

    def __repr__(self) -> str:
        return (
            f"PatternSharer(agent={self._agent_id}, "
            f"shares={self._total_shares}, consensus={self._total_consensus})"
        )
