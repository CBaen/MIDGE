"""ConvergenceLagScoringMixin — lag findings, domain sequence, sequence scoring.

Used by ConvergenceConfidenceMixin as a mixin. Expects the following instance
attributes:
    self._lag_findings, self._DOMAIN_SOURCES
    (_DOMAIN_SOURCES provided as class attribute on ConvergenceAlerter)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"


class ConvergenceLagScoringMixin:
    """Lag findings loading, domain sequence building, and sequence score computation."""

    def set_lag_findings(self, findings: List) -> None:
        """Accept lag findings from LagCorrelationAnalyzer.

        Called by the step hook after every lag analysis run (every 500 steps).
        Findings shape the sequence_score on future alerts.

        Args:
            findings: List of LagFinding objects (source_a leads source_b by lag_days)
        """
        self._lag_findings = findings if findings else []
        logger.info(
            "ConvergenceAlerter: received %d lag findings for sequence scoring",
            len(self._lag_findings),
        )

    def _load_lag_findings(self) -> None:
        """Load persisted lag findings on startup so sequence scoring works immediately."""
        lag_path = _DATA_DIR / "lag_correlations.json"
        if not lag_path.exists():
            return
        try:
            from mae_core.market.intelligence.lag_correlation_analyzer import LagFinding
            raw = json.loads(lag_path.read_text(encoding="utf-8"))
            self._lag_findings = [LagFinding(**r) for r in raw]
            logger.debug(
                "ConvergenceAlerter: loaded %d lag findings from disk",
                len(self._lag_findings),
            )
        except Exception:
            logger.debug("ConvergenceAlerter: failed to load lag findings", exc_info=True)
            self._lag_findings = []

    def _build_domain_sequence(self, signals: list) -> List[str]:
        """Sort contributing signals by timestamp (earliest first) and return domain order.

        Returns a list of domain names ordered from first-to-fire to last-to-fire.
        Domains with multiple signals use the timestamp of the earliest signal.
        """
        try:
            domain_first_ts: Dict[str, datetime] = {}
            for sig in signals:
                ts = getattr(sig, "timestamp", None)
                if ts is None:
                    continue
                domain = getattr(sig, "domain", "")
                if domain and (domain not in domain_first_ts or ts < domain_first_ts[domain]):
                    domain_first_ts[domain] = ts
            if not domain_first_ts:
                return []
            return [d for d, _ in sorted(domain_first_ts.items(), key=lambda x: x[1])]
        except Exception:
            logger.debug("domain sequence build failed", exc_info=True)
            return []

    def _compute_sequence_score(self, domain_sequence: List[str]) -> float:
        """Score how well the observed domain firing order matches known lag relationships.

        For each pair (A, B) in domain_sequence where A fired before B:
        - If LagFinding says A leads B → matches → contributes a boost (+).
        - If LagFinding says B leads A → reversed → contributes a discount (-).
        - No data → neutral.

        Score formula:
            score = 1.0 + sum(matched_boosts) - sum(reversed_penalties)
        Clamped to [0.5, 1.3] so it never destroys or artificially doubles confidence.

        Returns:
            float in [0.5, 1.3]. 1.0 = neutral (no data or no matches).
        """
        if len(domain_sequence) < 2 or not self._lag_findings:
            return 1.0

        try:
            # Build a domain-level lag lookup: domain_a leads domain_b → max |correlation|
            # Uses _DOMAIN_SOURCES to map sources → domains.
            domain_lead: Dict = {}  # (lead_domain, lag_domain) → strength

            for finding in self._lag_findings:
                src_a = finding.source_a
                src_b = finding.source_b
                corr = abs(finding.correlation)
                dom_a = self._source_to_domain(src_a)
                dom_b = self._source_to_domain(src_b)
                if dom_a and dom_b and dom_a != dom_b:
                    key = (dom_a, dom_b)
                    if corr > domain_lead.get(key, 0.0):
                        domain_lead[key] = corr

            score_delta = 0.0
            pair_count = 0

            for i, dom_a in enumerate(domain_sequence):
                for dom_b in domain_sequence[i + 1:]:
                    pair_count += 1
                    forward_key = (dom_a, dom_b)
                    reverse_key = (dom_b, dom_a)

                    forward_strength = domain_lead.get(forward_key, 0.0)
                    reverse_strength = domain_lead.get(reverse_key, 0.0)

                    if forward_strength > 0.3:
                        score_delta += 0.05 * forward_strength
                    elif reverse_strength > 0.3:
                        score_delta -= 0.05 * reverse_strength

            if pair_count == 0:
                return 1.0

            return max(0.5, min(1.3, 1.0 + score_delta))

        except Exception:
            logger.debug("sequence score computation failed", exc_info=True)
            return 1.0

    def _source_to_domain(self, source: str) -> str:
        """Map a source name to its domain using _DOMAIN_SOURCES (reverse lookup)."""
        for domain, sources in self._DOMAIN_SOURCES.items():
            if source in sources:
                return domain
        return ""
