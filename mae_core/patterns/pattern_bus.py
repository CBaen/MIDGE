"""Pattern Bus - The thalamic relay of Mae's pattern recognition ecosystem.

Collects PatternSignals from all registered translators, groups them by
domain, detects cross-source correlations, and produces a per-step
PatternDigest summarizing all detected patterns.

Biological analogy: The thalamus receives raw signals from all senses
and routes them to the appropriate cortical area. Before the cortex
sees anything, the thalamus has already organized, filtered, and
prioritized the incoming information.

The PatternBus does the same: it subscribes translators to the EventBus,
collects their PatternSignals, correlates multi-source agreements, and
packages everything into a clean digest for the PatternCortex.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from typing import Any

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)
from mae_core.patterns.translators.base import PatternTranslator

logger = logging.getLogger(__name__)


class GainModulator:
    """Endocrine gain modulation of pattern signal processing.

    Biological basis: Neuromodulators (norepinephrine, dopamine, serotonin,
    cortisol, oxytocin) don't create neural signals -- they multiplicatively
    modulate HOW existing signals are processed. Gain modulation changes the
    slope of the input-output function, not the baseline.

    Key neuroscience:
    - Norepinephrine (adrenaline analog) increases signal-to-noise ratio:
      amplifies strong signals, suppresses weak ones. LC-NE activation
      facilitates sensory representation by inhibiting spontaneous activity
      more than evoked responses (Berridge & Waterhouse 2003).
    - Dopamine enhances signal-to-noise for salient stimuli and encodes
      reward prediction error (Schultz 1998). Amplifies novelty/capability.
    - Cortisol (stress) narrows attentional bandwidth to threat-relevant
      stimuli, suppressing non-threat processing.
    - Serotonin modulates patience and temporal discounting -- smooths
      extreme salience values, reducing impulsivity (Doya 2002).
    - Oxytocin amplifies social/cooperative signals (Shamay-Tsoory &
      Abu-Akel 2016).

    Implementation: Each hormone produces a multiplicative gain factor
    per signal. The composite gain is the product of all individual
    gains. Applied to signal salience before domain grouping.

    Backward compatible: default hormone levels produce gain = 1.0
    everywhere (no modulation).
    """

    # Threat-relevant domains for cortisol narrowing
    _THREAT_DOMAINS = frozenset({PatternDomain.THREAT, PatternDomain.FAILURE})
    # Reward/novelty domains for dopamine amplification
    _REWARD_DOMAINS = frozenset({PatternDomain.NOVELTY, PatternDomain.CAPABILITY})
    # Social domains for oxytocin amplification
    _SOCIAL_DOMAINS = frozenset({PatternDomain.BEHAVIORAL})

    def __init__(self) -> None:
        # Hormone levels: 0.0-1.0 (default = 0.0, no modulation)
        # When no EndocrineSystem is wired, all gains are 1.0 (backward
        # compatible). Once wired, EndocrineSystem pushes real levels.
        self._hormones: dict[str, float] = {
            "adrenaline": 0.0,
            "cortisol": 0.0,
            "dopamine": 0.0,
            "serotonin": 0.0,
            "oxytocin": 0.0,
        }
        self._total_modulations = 0

    def set_hormone_levels(self, hormones: dict[str, float]) -> None:
        """Update hormone levels from EndocrineSystem state.

        Args:
            hormones: Mapping of hormone name to level (0.0-1.0).
                      Only updates known hormones; ignores unknown keys.
        """
        for name in self._hormones:
            if name in hormones:
                self._hormones[name] = max(0.0, min(1.0, hormones[name]))

    def modulate(self, signals: list[PatternSignal]) -> list[PatternSignal]:
        """Apply endocrine gain modulation to a list of signals.

        Each signal's salience is multiplied by a composite gain factor
        derived from all active hormones. Uses dataclasses.replace() to
        avoid mutating the original signals.

        Returns:
            New list of signals with gain-modulated salience values.
        """
        if not signals:
            return signals

        modulated: list[PatternSignal] = []
        adrenaline = self._hormones["adrenaline"]
        cortisol = self._hormones["cortisol"]
        dopamine = self._hormones["dopamine"]
        serotonin = self._hormones["serotonin"]
        oxytocin = self._hormones["oxytocin"]

        for sig in signals:
            gain = 1.0

            # ── Adrenaline: signal-to-noise ratio enhancement ────────
            # High-salience signals amplified, low-salience suppressed.
            # Biological: LC-NE potentiates strong synaptic responses
            # and reduces weak ones (multiplicative gain control).
            if sig.salience > 0.5:
                gain *= 1.0 + adrenaline * 0.5
            elif sig.salience < 0.3:
                gain *= 1.0 - adrenaline * 0.3

            # ── Cortisol: attentional narrowing to threats ───────────
            # Threat/failure domains amplified; all others suppressed.
            # Biological: stress hormones narrow attentional bandwidth
            # to threat-relevant stimuli (Arnsten 2009).
            if sig.domain in self._THREAT_DOMAINS:
                gain *= 1.0 + cortisol * 0.4
            else:
                gain *= 1.0 - cortisol * 0.2

            # ── Dopamine: reward and novelty amplification ───────────
            # Novelty/capability domains amplified.
            # Biological: dopamine enhances signal-to-noise ratio for
            # salient stimuli in mPFC (Seamans & Yang 2004).
            if sig.domain in self._REWARD_DOMAINS:
                gain *= 1.0 + dopamine * 0.3

            # ── Serotonin: smoothing / patience modulation ───────────
            # Reduces extreme salience values toward 0.5.
            # Biological: serotonin modulates temporal discounting and
            # reduces impulsive responses (Miyazaki et al. 2014).
            gain *= 1.0 - serotonin * 0.2 * abs(sig.salience - 0.5)

            # ── Oxytocin: social signal amplification ────────────────
            # Behavioral/social domain signals amplified.
            # Biological: oxytocin enhances salience of social cues
            # (Shamay-Tsoory & Abu-Akel 2016).
            if sig.domain in self._SOCIAL_DOMAINS:
                gain *= 1.0 + oxytocin * 0.3

            # Apply composite gain, clamping salience to [0.0, 1.0]
            new_salience = max(0.0, min(1.0, sig.salience * gain))

            if abs(new_salience - sig.salience) > 0.001:
                modulated.append(replace(sig, salience=new_salience))
                self._total_modulations += 1
            else:
                modulated.append(sig)

        return modulated

    def get_statistics(self) -> dict[str, Any]:
        """Return gain modulator statistics."""
        return {
            "hormone_levels": dict(self._hormones),
            "total_modulations": self._total_modulations,
        }


@dataclass
class PatternDigest:
    """Per-step summary of all detected patterns.

    Produced by PatternBus.process_step() -- consumed by PatternCortex.
    """

    step: int
    signals: list[PatternSignal] = field(default_factory=list)
    by_domain: dict[PatternDomain, list[PatternSignal]] = field(default_factory=dict)
    by_form: dict[PatternForm, list[PatternSignal]] = field(default_factory=dict)
    correlated_groups: list[list[PatternSignal]] = field(default_factory=list)
    cross_domain_groups: list[list[PatternSignal]] = field(default_factory=list)
    dominant_domain: PatternDomain | None = None
    aggregate_salience: float = 0.0
    signal_count: int = 0


class PatternBus:
    """Collects PatternSignals from translators, produces per-step digests.

    Thread-safe: all inbox operations are append-only on a deque.
    """

    # Maximum signals processed per step (prevents flooding)
    MAX_SIGNALS_PER_STEP = 50

    # High-value cross-domain pairs (biological resonance)
    CROSS_DOMAIN_PAIRS: list[tuple[PatternDomain, PatternDomain, str]] = [
        (PatternDomain.THREAT, PatternDomain.NOVELTY, "novel threat"),
        (PatternDomain.NOVELTY, PatternDomain.CAUSATION, "learning opportunity"),
        (PatternDomain.CAPABILITY, PatternDomain.PREDICTION, "new ability changes expectations"),
        (PatternDomain.FAILURE, PatternDomain.THREAT, "system failure under attack"),
        (PatternDomain.BEHAVIORAL, PatternDomain.CAUSATION, "action-consequence link"),
    ]

    # Minimum salience for cross-domain correlation (prevents noise)
    CROSS_DOMAIN_MIN_SALIENCE = 0.3

    def __init__(self, event_bus: Any, attentional_gate: Any = None) -> None:
        self._event_bus = event_bus
        self._translators: list[PatternTranslator] = []
        self._inbox: deque[PatternSignal] = deque(maxlen=200)
        self._recent_digests: deque[PatternDigest] = deque(maxlen=21)  # Fibonacci
        self._total_signals = 0
        self._total_correlations = 0
        # TRN analog: top-down attentional gating (Law 8, Property 8)
        # If None, all signals pass through unmodified (backward compatible).
        self._attentional_gate = attentional_gate
        # Endocrine gain modulation: hormones modulate signal salience
        # via multiplicative gain (biological neuromodulation analog).
        self._gain_modulator = GainModulator()

    @property
    def translator_count(self) -> int:
        return len(self._translators)

    def register_translator(self, translator: PatternTranslator) -> None:
        """Register a translator and subscribe it to its EventBus channels."""
        self._translators.append(translator)

        for channel in translator.channels:
            self._event_bus.register_callback(
                channel,
                lambda ch, msg, t=translator: self._on_event(t, ch, msg),
            )

        logger.debug(
            "PatternBus: registered %s translator for channels %s",
            translator.source_name,
            translator.channels,
        )

    def set_hormone_levels(self, hormones: dict[str, float]) -> None:
        """Update endocrine hormone levels for gain modulation.

        Called from main.py's endocrine EventBus callback so that the
        PatternBus receives updated hormone concentrations each step.
        The GainModulator uses these to compute multiplicative gain
        on each signal's salience.

        Args:
            hormones: Mapping of hormone name to level (0.0-1.0).
        """
        self._gain_modulator.set_hormone_levels(hormones)

    def _on_event(
        self, translator: PatternTranslator, channel: str, message: Any,
    ) -> None:
        """EventBus callback -- translate and enqueue."""
        try:
            # Parse JSON if needed
            parsed = message
            if isinstance(message, str):
                try:
                    parsed = json.loads(message)
                except (json.JSONDecodeError, ValueError):
                    parsed = message

            signal = translator.translate(channel, parsed)
            if signal is not None:
                self._inbox.append(signal)
        except Exception:
            logger.debug(
                "PatternBus: translator %s failed on %s",
                translator.source_name,
                channel,
                exc_info=True,
            )

    def process_step(self, step_number: int) -> PatternDigest:
        """Drain inbox, correlate, group, and produce a digest for this step."""
        # Drain inbox, sorted by salience (highest first) so the most
        # important signals are processed when the budget is tight.
        pending = list(self._inbox)
        self._inbox.clear()
        pending.sort(key=lambda s: s.salience, reverse=True)

        signals: list[PatternSignal] = pending[: self.MAX_SIGNALS_PER_STEP]

        # Return any overflow back to the inbox (already salience-ordered)
        for leftover in pending[self.MAX_SIGNALS_PER_STEP :]:
            self._inbox.append(leftover)

        # ── Attentional gating (TRN analog) ─────────────────────────
        # Apply top-down attentional filtering BEFORE grouping/correlation.
        # Modulates salience based on what the organism is attending to.
        # Signals may be re-ordered after gating (salience changed), but
        # the budget cut already happened, so no signals are lost.
        gate = getattr(self, "_attentional_gate", None)
        if gate is not None:
            gate.gate_signals(signals)

        # ── Endocrine gain modulation ────────────────────────────────
        # Hormones multiplicatively modulate signal salience AFTER
        # attentional gating and BEFORE domain grouping/correlation.
        # Biological: neuromodulators (NE, DA, 5-HT, cortisol, OT)
        # change the gain of neural populations, affecting which
        # signals reach awareness. This is the reticular activating
        # system modulating thalamic relay.
        signals = self._gain_modulator.modulate(signals)

        self._total_signals += len(signals)

        # Group by domain
        by_domain: dict[PatternDomain, list[PatternSignal]] = defaultdict(list)
        for sig in signals:
            by_domain[sig.domain].append(sig)

        # Group by form
        by_form: dict[PatternForm, list[PatternSignal]] = defaultdict(list)
        for sig in signals:
            by_form[sig.form].append(sig)

        # Detect correlations: same domain, different source systems
        correlated_groups = self._detect_correlations(by_domain)

        # Detect cross-domain correlations: high-value domain pairs
        cross_domain_groups = self._detect_cross_domain_correlations(by_domain)

        # Find dominant domain (highest aggregate salience)
        dominant_domain = None
        max_salience = 0.0
        for domain, domain_signals in by_domain.items():
            domain_salience = sum(s.salience for s in domain_signals)
            if domain_salience > max_salience:
                max_salience = domain_salience
                dominant_domain = domain

        aggregate_salience = sum(s.salience for s in signals) if signals else 0.0

        digest = PatternDigest(
            step=step_number,
            signals=signals,
            by_domain=dict(by_domain),
            by_form=dict(by_form),
            correlated_groups=correlated_groups,
            cross_domain_groups=cross_domain_groups,
            dominant_domain=dominant_domain,
            aggregate_salience=aggregate_salience,
            signal_count=len(signals),
        )

        self._recent_digests.append(digest)
        return digest

    def _detect_correlations(
        self, by_domain: dict[PatternDomain, list[PatternSignal]],
    ) -> list[list[PatternSignal]]:
        """Find cross-source agreements within the same domain.

        When 2+ signals in the same domain come from different source systems,
        they form a correlated group. Each signal in the group gets elevated
        to CORRELATED form with boosted confidence.

        Confidence boost: min(1.0, max_confidence + 0.1 * log(num_sources))
        Same coalescing math as SignalPriorityResolver.
        """
        groups: list[list[PatternSignal]] = []

        for domain, signals in by_domain.items():
            # Group by source system
            by_source: dict[str, list[PatternSignal]] = defaultdict(list)
            for sig in signals:
                by_source[sig.source_system].append(sig)

            # Need 2+ different sources for correlation
            if len(by_source) < 2:
                continue

            # Take the highest-salience signal from each source
            group: list[PatternSignal] = []
            for source_signals in by_source.values():
                best = max(source_signals, key=lambda s: s.salience)
                group.append(best)

            # Boost confidence for correlated signals
            max_conf = max(s.confidence for s in group)
            boosted_conf = min(1.0, max_conf + 0.1 * math.log(len(group)))

            # Create copies so original signals are not mutated in-place
            # (other consumers may still hold references to the originals).
            group = [
                replace(sig, form=PatternForm.CORRELATED, confidence=boosted_conf)
                for sig in group
            ]

            groups.append(group)
            self._total_correlations += 1

        return groups

    def _detect_cross_domain_correlations(
        self, by_domain: dict[PatternDomain, list[PatternSignal]],
    ) -> list[list[PatternSignal]]:
        """Detect co-occurring signals across different domains.

        When two domains from the predefined high-value pairs both appear
        in the same step with sufficient salience, a cross-domain group
        is created. Unlike same-domain correlation (which elevates existing
        signals), this creates a synthetic CORRELATED signal that captures
        the cross-domain insight.

        Biological analogy: Seeing fire (THREAT) and smelling something
        new (NOVELTY) simultaneously produces the unified percept
        "novel danger" which is more than either signal alone.
        """
        groups: list[list[PatternSignal]] = []

        for domain_a, domain_b, label in self.CROSS_DOMAIN_PAIRS:
            signals_a = by_domain.get(domain_a, [])
            signals_b = by_domain.get(domain_b, [])

            if not signals_a or not signals_b:
                continue

            # Best signal from each domain
            best_a = max(signals_a, key=lambda s: s.salience)
            best_b = max(signals_b, key=lambda s: s.salience)

            # Both must have sufficient salience
            if best_a.salience < self.CROSS_DOMAIN_MIN_SALIENCE:
                continue
            if best_b.salience < self.CROSS_DOMAIN_MIN_SALIENCE:
                continue

            # Geometric mean of confidences
            conf = math.sqrt(best_a.confidence * best_b.confidence)
            # Max salience + cross-domain bonus
            sal = min(1.0, max(best_a.salience, best_b.salience) + 0.1)

            # Higher-salience domain becomes the synthetic signal's domain
            primary_domain = domain_a if best_a.salience >= best_b.salience else domain_b

            synthetic = PatternSignal(
                source_system="cross_domain",
                domain=primary_domain,
                form=PatternForm.CORRELATED,
                confidence=conf,
                salience=sal,
                description=f"Cross-domain: {label} ({domain_a.value} + {domain_b.value})",
                evidence={
                    "domain_a": domain_a.value,
                    "domain_b": domain_b.value,
                    "label": label,
                },
                related_signals=[best_a.signal_id, best_b.signal_id],
            )

            groups.append([best_a, best_b, synthetic])
            self._total_correlations += 1

        return groups

    def get_recent_digests(self, n: int = 5) -> list[PatternDigest]:
        """Return the last N digests for trend analysis."""
        return list(self._recent_digests)[-n:]

    def get_statistics(self) -> dict[str, Any]:
        stats = {
            "translators": self.translator_count,
            "total_signals": self._total_signals,
            "total_correlations": self._total_correlations,
            "inbox_size": len(self._inbox),
            "recent_digests": len(self._recent_digests),
        }
        gate = getattr(self, "_attentional_gate", None)
        if gate is not None:
            stats["attentional_gate"] = gate.get_statistics()
        stats["gain_modulator"] = self._gain_modulator.get_statistics()
        return stats

    def __repr__(self) -> str:
        return (
            f"PatternBus(translators={self.translator_count}, "
            f"total_signals={self._total_signals})"
        )
