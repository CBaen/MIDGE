"""RenalFilter - Metabolic waste filtration and toxin removal.

Biological analogy: Kidneys filter blood, remove metabolic waste, and
maintain electrolyte balance. Each nephron is a tiny filtering unit that
processes blood through glomerular filtration (size-based), tubular
reabsorption (reclaiming useful molecules), and tubular secretion
(actively removing toxins).

Mae's RenalFilter does the same with data flowing through the system:
- Filters items for known toxin patterns (contradictions, corrupt data)
- Classifies items as clean, suspect, toxic, or recyclable
- Tracks toxin load to detect system overload (kidney failure)
- Learns new toxin signatures over time

Connection points:
- InputValidator: pre-filtered items may come from validation
- ThreatDetector: toxic items are reported as threats
- PearlDefense: suspect items may enter pearl processing
- EventBus: publishes renal status and failure alerts

Law compliance:
- Law 6 (Autopoietic Closure): the filter maintains its own health by
  tracking toxin load and adjusting filtration. It produces the clean
  state that allows it to continue filtering.
- Law 8 (Consciousness): contributes to differentiation (property 2) --
  classifies data into clean/toxic/suspect/recyclable categories.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# EventBus channels
CH_RENAL_STATUS = "defense.renal_status"
CH_KIDNEY_FAILURE = "defense.kidney_failure"
CH_TOXIC_FILTERED = "defense.toxic_filtered"


@dataclass
class FilterResult:
    """Result of filtering a single item."""

    item_id: str
    source: str
    verdict: str  # "clean", "toxic", "suspect", "recyclable"
    toxicity_score: float
    reason: str


class RenalFilter:
    """Metabolic waste filter inspired by kidney nephrons.

    Filters data items for toxin patterns (contradictions, corruption,
    circular references) and tracks system toxin load. When toxin load
    exceeds capacity, publishes kidney failure alerts.

    The filtration rate determines how many toxins can be processed per
    step. If toxins arrive faster than they can be filtered, the system
    becomes overwhelmed.
    """

    CH_RENAL_STATUS = CH_RENAL_STATUS
    CH_KIDNEY_FAILURE = CH_KIDNEY_FAILURE
    CH_TOXIC_FILTERED = CH_TOXIC_FILTERED

    def __init__(self, event_bus: Any = None, **kwargs: Any) -> None:
        self._bus = event_bus

        # Filter history (last 100 results)
        self._filter_history: list[FilterResult] = []
        self._max_history: int = 100

        # Known toxin signatures
        self._toxin_patterns: list[str] = [
            "contradiction",
            "circular_ref",
            "corrupt",
            "overflow",
            "injection",
            "deadlock",
            "infinite_loop",
        ]

        # Filtration dynamics
        self._filtration_rate: float = 10.0  # toxins processed per step
        self._toxin_load: float = 0.0  # accumulated unfiltered toxins
        self._step_count: int = 0

        # Statistics
        self._total_filtered: int = 0
        self._clean_count: int = 0
        self._toxic_count: int = 0
        self._suspect_count: int = 0
        self._recyclable_count: int = 0

        logger.info(
            "RenalFilter initialized (filtration_rate=%.1f, patterns=%d)",
            self._filtration_rate,
            len(self._toxin_patterns),
        )

    # =========================================================================
    # Filtering
    # =========================================================================

    def filter_item(
        self, item_id: str, source: str, data: dict[str, Any]
    ) -> FilterResult:
        """Filter a single data item for toxins.

        Checks for:
        1. Known toxin pattern matches (string matching)
        2. Contradictions (value vs expected mismatch)
        3. Corruption (None values, NaN, empty required fields)

        Classifies:
        - <0.2 toxicity = clean
        - 0.2-0.5 = suspect
        - 0.5-0.8 = toxic
        - >0.5 with useful subcomponents = recyclable

        Returns a FilterResult with verdict and toxicity score.
        """
        checks_failed = 0
        total_checks = 0
        reasons: list[str] = []

        # Check 1: Known toxin patterns
        data_str = str(data).lower()
        for pattern in self._toxin_patterns:
            total_checks += 1
            if pattern.lower() in data_str:
                checks_failed += 1
                reasons.append(f"toxin_pattern:{pattern}")

        # Check 2: Contradiction detection
        if "value" in data and "expected" in data:
            total_checks += 1
            try:
                val = float(data["value"])
                exp = float(data["expected"])
                if abs(val - exp) > abs(exp) * 0.5 + 1e-9:
                    checks_failed += 1
                    reasons.append(
                        f"contradiction:value={val},expected={exp}"
                    )
            except (ValueError, TypeError):
                # Non-numeric comparison
                if data["value"] != data["expected"]:
                    checks_failed += 1
                    reasons.append("contradiction:value!=expected")

        # Check 3: Corruption detection
        for key, value in data.items():
            total_checks += 1
            if value is None:
                checks_failed += 1
                reasons.append(f"corrupt:None@{key}")
            elif isinstance(value, float) and math.isnan(value):
                checks_failed += 1
                reasons.append(f"corrupt:NaN@{key}")
            elif isinstance(value, str) and value.strip() == "" and key in (
                "id", "name", "source", "type",
            ):
                checks_failed += 1
                reasons.append(f"corrupt:empty@{key}")

        # Score toxicity (0-1)
        if total_checks > 0:
            toxicity_score = min(1.0, checks_failed / max(total_checks, 1))
        else:
            toxicity_score = 0.0

        # Classify
        has_useful = any(
            v is not None and not (isinstance(v, float) and math.isnan(v))
            for v in data.values()
        )

        if toxicity_score < 0.2:
            verdict = "clean"
            self._clean_count += 1
        elif toxicity_score < 0.5:
            verdict = "suspect"
            self._suspect_count += 1
        elif has_useful and toxicity_score < 0.8:
            verdict = "recyclable"
            self._recyclable_count += 1
        else:
            verdict = "toxic"
            self._toxic_count += 1

        # Update toxin load
        self._toxin_load += toxicity_score

        # Record result
        reason_str = "; ".join(reasons) if reasons else "clean"
        result = FilterResult(
            item_id=item_id,
            source=source,
            verdict=verdict,
            toxicity_score=toxicity_score,
            reason=reason_str,
        )

        self._filter_history.append(result)
        if len(self._filter_history) > self._max_history:
            self._filter_history = self._filter_history[-self._max_history:]

        self._total_filtered += 1

        # Publish toxic item events
        if verdict == "toxic" and self._bus is not None:
            self._bus.publish(CH_TOXIC_FILTERED, {
                "item_id": item_id,
                "source": source,
                "toxicity": toxicity_score,
                "reason": reason_str,
            })

        return result

    # =========================================================================
    # Toxin Pattern Learning
    # =========================================================================

    def add_toxin_pattern(self, pattern: str) -> None:
        """Learn a new toxin signature."""
        if pattern not in self._toxin_patterns:
            self._toxin_patterns.append(pattern)
            logger.info("Learned new toxin pattern: %s", pattern)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def get_toxin_load(self) -> float:
        """Return current accumulated toxin load."""
        return self._toxin_load

    def is_healthy(self) -> bool:
        """Return True if toxin load is manageable (< 3.0)."""
        return self._toxin_load < 3.0

    # =========================================================================
    # Step (Periodic Maintenance)
    # =========================================================================

    def step(self, current_step: int = 0) -> None:
        """Per-step maintenance: decay toxin load, check for kidney failure."""
        self._step_count = current_step if current_step > 0 else self._step_count + 1

        # Kidney failure check (before decay)
        if self._toxin_load > 5.0 and self._bus is not None:
            self._bus.publish(CH_KIDNEY_FAILURE, {
                "toxin_load": self._toxin_load,
                "filtration_rate": self._filtration_rate,
                "step": self._step_count,
                "message": "System overwhelmed: toxin load exceeds capacity",
            })
            logger.warning(
                "KIDNEY FAILURE: toxin_load=%.2f exceeds threshold",
                self._toxin_load,
            )

        # Decay toxin load by filtration rate
        self._toxin_load = max(0.0, self._toxin_load - self._filtration_rate)

        # Compute clean rate
        if self._total_filtered > 0:
            clean_rate = self._clean_count / self._total_filtered
        else:
            clean_rate = 1.0

        # Publish status
        if self._bus is not None:
            self._bus.publish(CH_RENAL_STATUS, {
                "filtered_count": self._total_filtered,
                "toxin_load": self._toxin_load,
                "clean_rate": clean_rate,
                "is_healthy": self.is_healthy(),
                "step": self._step_count,
            })

    # =========================================================================
    # Observability
    # =========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return filter statistics for observability."""
        clean_rate = (
            self._clean_count / max(self._total_filtered, 1)
        )
        return {
            "total_filtered": self._total_filtered,
            "clean_count": self._clean_count,
            "toxic_count": self._toxic_count,
            "suspect_count": self._suspect_count,
            "recyclable_count": self._recyclable_count,
            "toxin_load": self._toxin_load,
            "filtration_rate": self._filtration_rate,
            "clean_rate": clean_rate,
            "is_healthy": self.is_healthy(),
            "toxin_patterns": len(self._toxin_patterns),
            "history_size": len(self._filter_history),
            "step": self._step_count,
        }

    # =========================================================================
    # Persistence (Tier 2)
    # =========================================================================

    def serialize(self) -> dict[str, Any]:
        """Serialize filter state for persistence."""
        history = [
            {
                "item_id": r.item_id,
                "source": r.source,
                "verdict": r.verdict,
                "toxicity_score": r.toxicity_score,
                "reason": r.reason,
            }
            for r in self._filter_history
        ]
        return {
            "filter_history": history,
            "toxin_patterns": self._toxin_patterns,
            "filtration_rate": self._filtration_rate,
            "toxin_load": self._toxin_load,
            "step_count": self._step_count,
            "total_filtered": self._total_filtered,
            "clean_count": self._clean_count,
            "toxic_count": self._toxic_count,
            "suspect_count": self._suspect_count,
            "recyclable_count": self._recyclable_count,
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore filter state from serialized data."""
        self._toxin_patterns = data.get("toxin_patterns", self._toxin_patterns)
        self._filtration_rate = data.get("filtration_rate", 10.0)
        self._toxin_load = data.get("toxin_load", 0.0)
        self._step_count = data.get("step_count", 0)
        self._total_filtered = data.get("total_filtered", 0)
        self._clean_count = data.get("clean_count", 0)
        self._toxic_count = data.get("toxic_count", 0)
        self._suspect_count = data.get("suspect_count", 0)
        self._recyclable_count = data.get("recyclable_count", 0)

        self._filter_history = []
        for item in data.get("filter_history", []):
            self._filter_history.append(
                FilterResult(
                    item_id=item["item_id"],
                    source=item["source"],
                    verdict=item["verdict"],
                    toxicity_score=item["toxicity_score"],
                    reason=item["reason"],
                )
            )

        logger.info(
            "RenalFilter restored (toxin_load=%.2f, patterns=%d)",
            self._toxin_load,
            len(self._toxin_patterns),
        )
