"""Attentional Gate - Thalamic Reticular Nucleus (TRN) analog.

Top-down attentional filtering for Mae's pattern recognition ecosystem.

Biological analogy: The TRN wraps the thalamus like a thin shell. It
receives copies of BOTH ascending (sensory) and descending (cortical /
attentional) signals. When cortex says "attend to threats", the TRN
amplifies threat-related thalamic relay neurons and suppresses others.
You see what you expect to see -- unless something screams loud enough
to break through anyway.

Mathematical identity:
- Law 8, Property 8: prediction/error-correction (top-down predictions
  gate bottom-up signals)
- Law 4: Fractal Self-Similarity (gating operates at signal level; the
  same principle applies at every scale)
- Transfractal Compromise: TRN is WITHIN-module filtering (fractal
  substrate), not between-module shortcuts (EventBus)

Three gating mechanisms (triadic, per Law 2):
1. Attention vector: domain-based gain control (cortical top-down)
2. Surprise override: high-salience signals bypass the gate (pop-out)
3. Prediction error modulation: high uncertainty widens attention (global)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from mae_core.patterns.pattern_signal import PatternDomain, PatternSignal

logger = logging.getLogger(__name__)

# ── Configuration Constants ──────────────────────────────────────────
# Neutral attention: neither amplifying nor suppressing
NEUTRAL_ATTENTION = 0.5

# How much the dominant domain gets boosted per advisory
BOOST_PER_ADVISORY = 0.2

# How fast non-dominant domains decay toward neutral per step
DECAY_RATE = 0.1

# Threat level above which THREAT domain is forced to maximum
SURVIVAL_OVERRIDE_THRESHOLD = 0.7

# Signal salience above which the gate is bypassed entirely (pop-out)
SURPRISE_BYPASS_THRESHOLD = 0.9

# Prediction error above which attention widens globally
PREDICTION_ERROR_THRESHOLD = 0.5

# Target attention level when prediction error is high (widened attention)
WIDENED_ATTENTION_TARGET = 0.7

# Rate at which attention moves toward widened target under high pred error
WIDENING_RATE = 0.3


@dataclass
class GateStatistics:
    """Diagnostic snapshot of the attentional gate's state."""

    attention_vector: dict[str, float]
    signals_gated: int = 0
    signals_bypassed: int = 0
    total_signals_processed: int = 0
    last_dominant_domain: str | None = None
    prediction_error: float = 0.0
    is_widened: bool = False


class AttentionalGate:
    """Thalamic Reticular Nucleus analog -- top-down attentional filtering.

    The gate sits between raw signal collection (inbox drain) and signal
    processing (grouping/correlation) in the PatternBus. It modulates
    signal salience based on what the organism is currently attending to.

    The attention vector is a dict mapping PatternDomain -> float [0.0, 1.0].
    High values amplify signals of that domain; low values suppress them.

    Gating formula:
        gated_salience = raw_salience * (0.5 + 0.5 * attention[domain])

        At neutral (0.5): gated = raw * 0.75 (slight suppression)
        At full (1.0):    gated = raw * 1.0  (no suppression)
        At zero (0.0):    gated = raw * 0.5  (half suppression)

    The gate NEVER fully blocks a signal. Even unattended signals can
    break through if their raw salience is high enough. This mirrors
    biology: you notice a gunshot regardless of what you're reading.
    """

    def __init__(self) -> None:
        # Attention vector: all domains start at neutral
        self._attention: dict[PatternDomain, float] = {
            domain: NEUTRAL_ATTENTION for domain in PatternDomain
        }

        # Tracking
        self._signals_gated = 0
        self._signals_bypassed = 0
        self._total_processed = 0
        self._last_dominant_domain: PatternDomain | None = None
        self._prediction_error: float = 0.0

    @property
    def attention_vector(self) -> dict[PatternDomain, float]:
        """Current attention levels per domain (read-only copy)."""
        return dict(self._attention)

    @property
    def prediction_error(self) -> float:
        """Current prediction error level."""
        return self._prediction_error

    def gate_signals(self, signals: list[PatternSignal]) -> list[PatternSignal]:
        """Apply attentional gating to a list of signals.

        Modifies signal salience IN PLACE based on the attention vector.
        Signals above the surprise threshold bypass the gate entirely.

        This is called by PatternBus.process_step() after draining the
        inbox but before grouping and correlation.

        Args:
            signals: List of PatternSignals to gate.

        Returns:
            The same list (modified in place) for chaining convenience.
        """
        for sig in signals:
            self._total_processed += 1

            # Surprise override: very salient signals bypass the gate
            # (biological: pop-out effect -- a loud noise or sudden motion
            # captures attention regardless of what you're focused on)
            if sig.salience > SURPRISE_BYPASS_THRESHOLD:
                self._signals_bypassed += 1
                continue

            # Look up attention for this signal's domain
            attention = self._attention.get(sig.domain, NEUTRAL_ATTENTION)

            # Gating formula: modulate salience
            # gain ranges from 0.5 (zero attention) to 1.0 (full attention)
            gain = 0.5 + 0.5 * attention
            sig.salience = max(0.0, min(1.0, sig.salience * gain))
            self._signals_gated += 1

        return signals

    def update_attention(self, advisory: Any) -> None:
        """Update attention vector from a PatternAdvisory (top-down signal).

        After the PatternCortex generates an advisory, this method updates
        what the organism is attending to. The dominant domain in the
        advisory gets boosted; other domains decay toward neutral.

        This creates the recurrent loop:
            sense -> cortex -> advisory -> gate -> sense (next step)

        Args:
            advisory: A PatternAdvisory (or any object with the expected
                      attributes). Uses getattr for backward compatibility.
        """
        if advisory is None:
            return

        # Extract dominant domain from the advisory
        dominant_pattern = getattr(advisory, "dominant_pattern", None)
        dominant_domain = None
        if dominant_pattern is not None:
            dominant_domain = getattr(dominant_pattern, "domain", None)

        # ── Step 1: Boost the dominant domain ─────────────────────────
        if dominant_domain is not None and isinstance(dominant_domain, PatternDomain):
            self._last_dominant_domain = dominant_domain
            current = self._attention.get(dominant_domain, NEUTRAL_ATTENTION)
            self._attention[dominant_domain] = min(1.0, current + BOOST_PER_ADVISORY)

        # ── Step 2: Decay all non-dominant domains toward neutral ─────
        for domain in PatternDomain:
            if domain == dominant_domain:
                continue
            current = self._attention[domain]
            if current > NEUTRAL_ATTENTION:
                self._attention[domain] = max(
                    NEUTRAL_ATTENTION,
                    current - DECAY_RATE,
                )
            elif current < NEUTRAL_ATTENTION:
                self._attention[domain] = min(
                    NEUTRAL_ATTENTION,
                    current + DECAY_RATE,
                )

        # ── Step 3: Survival override -- threats force attention ──────
        # (biological: amygdala hijack -- high threat overrides cortical
        # attention and forces the TRN to relay threat signals at max gain)
        threat_level = getattr(advisory, "threat_level", 0.0)
        if threat_level > SURVIVAL_OVERRIDE_THRESHOLD:
            self._attention[PatternDomain.THREAT] = 1.0

        # ── Step 4: Prediction error modulation ──────────────────────
        # When prediction error is high, the organism is uncertain about
        # what's happening. Response: widen attention globally so MORE
        # information gets through. (biological: norepinephrine release
        # from locus coeruleus during surprise broadens attentional scope)
        self._apply_prediction_error_modulation()

    def set_prediction_error(self, error: float) -> None:
        """Set the current prediction error from the agent layer.

        Prediction error comes from the FEP comparison step in the agent
        lifecycle: PREDICT -> OBSERVE -> COMPARE. High error means the
        world is surprising, which should widen attention.

        Args:
            error: Prediction error value (0.0 = perfect prediction,
                   higher = more surprise). Will be clamped to [0.0, 1.0].
        """
        self._prediction_error = max(0.0, min(1.0, error))

    def _apply_prediction_error_modulation(self) -> None:
        """Widen attention when prediction error is high.

        When the organism's predictions are wrong (high error), it needs
        to cast a wider sensory net to figure out what changed. All
        domain attention levels move toward WIDENED_ATTENTION_TARGET.

        This is NOT the same as setting everything to max -- widened
        attention (0.7) is above neutral (0.5) but below full (1.0).
        The organism becomes more globally attentive but not maximally
        focused on everything (which would be no focus at all).
        """
        if self._prediction_error <= PREDICTION_ERROR_THRESHOLD:
            return

        # Strength of widening scales with prediction error above threshold
        excess = self._prediction_error - PREDICTION_ERROR_THRESHOLD
        strength = min(1.0, excess * 2.0) * WIDENING_RATE  # 0-0.3 range

        for domain in PatternDomain:
            current = self._attention[domain]
            if current < WIDENED_ATTENTION_TARGET:
                self._attention[domain] = min(
                    WIDENED_ATTENTION_TARGET,
                    current + strength,
                )

    def reset(self) -> None:
        """Reset to neutral attention (useful for testing)."""
        for domain in PatternDomain:
            self._attention[domain] = NEUTRAL_ATTENTION
        self._prediction_error = 0.0
        self._signals_gated = 0
        self._signals_bypassed = 0
        self._total_processed = 0
        self._last_dominant_domain = None

    def get_statistics(self) -> dict[str, Any]:
        """Return diagnostic statistics for monitoring."""
        return {
            "attention_vector": {
                d.value: round(v, 3) for d, v in self._attention.items()
            },
            "signals_gated": self._signals_gated,
            "signals_bypassed": self._signals_bypassed,
            "total_signals_processed": self._total_processed,
            "last_dominant_domain": (
                self._last_dominant_domain.value
                if self._last_dominant_domain is not None
                else None
            ),
            "prediction_error": round(self._prediction_error, 3),
            "is_widened": self._prediction_error > PREDICTION_ERROR_THRESHOLD,
        }

    def __repr__(self) -> str:
        attended = [
            d.value for d, v in self._attention.items()
            if v > NEUTRAL_ATTENTION + 0.05
        ]
        return (
            f"AttentionalGate(attended={attended}, "
            f"pred_error={self._prediction_error:.2f}, "
            f"gated={self._signals_gated}, bypassed={self._signals_bypassed})"
        )
