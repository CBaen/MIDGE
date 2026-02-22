"""PatternSense - Per-agent pattern membrane.

Each agent gets a lightweight pattern detector that watches its own
experience. Not a brain -- a membrane that notices when things keep
being the same or keep changing.

Biological analogy: Cell membrane receptors. Each cell can sense its
immediate environment and signal when conditions change.

Three detectors, each under 0.2ms:
- Reward Trend: monotonically increasing/decreasing rewards
- Action Repetition: same action taken 3+ times in last 5 steps
- Reward Surprise: reward differs from mean by >2 standard deviations
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from mae_core.patterns.pattern_signal import (
    PatternDomain,
    PatternForm,
    PatternSignal,
)

# Minimum steps before trend detection activates (Rule of Three)
MIN_TREND_LENGTH = 3

# Window for action repetition and reward surprise
WINDOW_SIZE = 8

# How many recent actions to check for repetition
ACTION_WINDOW = 5

# Minimum repetitions to trigger (Rule of Three)
REPETITION_THRESHOLD = 3

# Standard deviations for surprise detection
SURPRISE_THRESHOLD = 2.0

# Habituation decay factor per step (biological: ~15% salience loss per
# sustained identical stimulus, like auditory adaptation to a constant tone)
HABITUATION_DECAY = 0.85

# Minimum salience floor for habituated signals.  Biological: even fully
# habituated stimuli retain a baseline salience — you still flinch at a
# loud noise you've heard a hundred times.  Prevents persistent threats
# from decaying to zero and becoming invisible to the agent.
HABITUATION_FLOOR = 0.1

# Synthetic z-score when baseline has zero variance but current value
# differs -- represents maximal surprise from a perfectly stable baseline
_ZERO_VARIANCE_ZSCORE = 10.0


@dataclass
class SenseResult:
    """Output of PatternSense.sense() -- zero or more signals detected."""

    signals: list[PatternSignal] = field(default_factory=list)
    step: int = 0


class PatternSense:
    """Per-agent pattern membrane.

    Lightweight: only arithmetic on small deques. Target < 0.5ms per call.
    """

    def __init__(self, agent_id: str, window_size: int = WINDOW_SIZE) -> None:
        self._agent_id = agent_id
        self._rewards: deque[float] = deque(maxlen=window_size)
        self._actions: deque = deque(maxlen=window_size)
        self._step = 0
        # Habituation / sensory adaptation: tracks consecutive steps each
        # signal pattern has been active. Like biological neural adaptation --
        # sustained identical stimuli lose salience (you stop hearing the
        # fridge hum). Keyed by "domain:form".
        self._salience_decay: dict[str, float] = {}

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def sense(self, reward: float, action: object, step: int) -> SenseResult:
        """Detect patterns in this agent's own experience.

        Called from _learn(action, reward). Returns a SenseResult with
        any PatternSignals detected this step.
        """
        self._step = step
        self._rewards.append(reward)
        self._actions.append(action)

        signals: list[PatternSignal] = []

        trend_sig = self._detect_reward_trend()
        if trend_sig is not None:
            signals.append(trend_sig)

        repetition_sig = self._detect_action_repetition()
        if repetition_sig is not None:
            signals.append(repetition_sig)

        surprise_sig = self._detect_reward_surprise()
        if surprise_sig is not None:
            signals.append(surprise_sig)

        # ── Habituation / sensory adaptation ──────────────────────────
        # Biological analog: sensory neurons reduce firing rate when the
        # same stimulus persists unchanged. You stop noticing a constant
        # background hum. Repeated identical signals decay in salience;
        # novel signals reset the decay counter.
        active_keys: set[str] = set()
        for sig in signals:
            key = f"{sig.domain.value}:{sig.form.value}"
            active_keys.add(key)
            if key in self._salience_decay:
                # Same pattern persists -- decay its salience, but never
                # below HABITUATION_FLOOR (persistent threats stay visible)
                self._salience_decay[key] = max(
                    HABITUATION_FLOOR,
                    self._salience_decay[key] * HABITUATION_DECAY,
                )
            else:
                # First step this pattern fires -- full salience
                self._salience_decay[key] = 1.0
            sig.salience = max(
                0.0, min(1.0, sig.salience * self._salience_decay[key])
            )

        # Reset decay for patterns that stopped firing (stimulus removed)
        stale_keys = [k for k in self._salience_decay if k not in active_keys]
        for k in stale_keys:
            del self._salience_decay[k]

        return SenseResult(signals=signals, step=step)

    def _detect_reward_trend(self) -> PatternSignal | None:
        """Detect monotonically increasing or decreasing reward runs.

        Needs 3+ consecutive steps trending in one direction.
        """
        rewards = list(self._rewards)
        if len(rewards) < MIN_TREND_LENGTH:
            return None

        # Check from the end: how many consecutive steps are monotonic?
        increasing = 0
        decreasing = 0

        for i in range(len(rewards) - 1, 0, -1):
            if rewards[i] > rewards[i - 1]:
                increasing += 1
            else:
                break

        for i in range(len(rewards) - 1, 0, -1):
            if rewards[i] < rewards[i - 1]:
                decreasing += 1
            else:
                break

        if increasing >= MIN_TREND_LENGTH:
            delta = rewards[-1] - rewards[-1 - increasing]
            return PatternSignal(
                source_system=f"agent:{self._agent_id}",
                domain=PatternDomain.OPPORTUNITY,
                form=PatternForm.REACTIVE,
                confidence=min(1.0, 0.4 + 0.1 * increasing),
                salience=min(0.7, 0.3 + abs(delta)),
                description=(
                    f"Reward trend UP for {increasing} steps "
                    f"(delta={delta:+.3f})"
                ),
                evidence={
                    "agent_id": self._agent_id,
                    "direction": "increasing",
                    "run_length": increasing,
                    "delta": delta,
                },
            )

        if decreasing >= MIN_TREND_LENGTH:
            delta = rewards[-1] - rewards[-1 - decreasing]
            return PatternSignal(
                source_system=f"agent:{self._agent_id}",
                domain=PatternDomain.THREAT,
                form=PatternForm.REACTIVE,
                confidence=min(1.0, 0.4 + 0.1 * decreasing),
                salience=min(0.7, 0.3 + abs(delta)),
                description=(
                    f"Reward trend DOWN for {decreasing} steps "
                    f"(delta={delta:+.3f})"
                ),
                evidence={
                    "agent_id": self._agent_id,
                    "direction": "decreasing",
                    "run_length": decreasing,
                    "delta": delta,
                },
            )

        return None

    def _detect_action_repetition(self) -> PatternSignal | None:
        """Detect the same action repeated 3+ times in last 5 steps.

        Signals a potential loop or stuck agent.
        """
        actions = list(self._actions)
        recent = actions[-ACTION_WINDOW:]
        if len(recent) < REPETITION_THRESHOLD:
            return None

        # Count most recent action
        last_action = recent[-1]
        count = sum(1 for a in recent if a == last_action)

        if count >= REPETITION_THRESHOLD:
            return PatternSignal(
                source_system=f"agent:{self._agent_id}",
                domain=PatternDomain.BEHAVIORAL,
                form=PatternForm.REACTIVE,
                confidence=min(1.0, 0.3 + 0.15 * count),
                salience=min(0.6, 0.2 + 0.1 * count),
                description=(
                    f"Action repetition: '{last_action}' repeated "
                    f"{count}/{len(recent)} times"
                ),
                evidence={
                    "agent_id": self._agent_id,
                    "action": str(last_action),
                    "count": count,
                    "window": len(recent),
                },
            )

        return None

    def _detect_reward_surprise(self) -> PatternSignal | None:
        """Detect when current reward is >2 std deviations from baseline mean.

        BUG-09 fix: z-score is now computed against the PRIOR observations
        (excluding the current value) to avoid self-inclusion bias that
        systematically pulls the z-score toward zero.

        Biological analog: surprise is measured against what the organism
        has already experienced, not a distribution that includes the
        novel stimulus itself.

        Positive surprise -> OPPORTUNITY, negative -> THREAT.
        """
        rewards = list(self._rewards)
        # Need at least 2 prior values + 1 current to compute meaningful
        # baseline statistics (MIN_TREND_LENGTH == 3 satisfies this)
        if len(rewards) < MIN_TREND_LENGTH:
            return None

        current = rewards[-1]
        baseline = rewards[:-1]  # Exclude current observation from baseline

        mean = sum(baseline) / len(baseline)
        variance = sum((r - mean) ** 2 for r in baseline) / len(baseline)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std < 1e-9:
            # Zero-variance baseline: all prior values were identical.
            # If the current value also matches, there is no surprise.
            # If it differs, this is maximally surprising -- the organism
            # has NEVER seen anything different (like a sudden sound in
            # perfect silence). Use a synthetic z-score.
            if abs(current - mean) < 1e-9:
                return None
            z_score = (
                _ZERO_VARIANCE_ZSCORE
                if current > mean
                else -_ZERO_VARIANCE_ZSCORE
            )
        else:
            z_score = (current - mean) / std

        if abs(z_score) < SURPRISE_THRESHOLD:
            return None

        if z_score > 0:
            domain = PatternDomain.OPPORTUNITY
            label = "positive"
        else:
            domain = PatternDomain.THREAT
            label = "negative"

        return PatternSignal(
            source_system=f"agent:{self._agent_id}",
            domain=domain,
            form=PatternForm.REACTIVE,
            confidence=min(1.0, 0.5 + 0.1 * abs(z_score)),
            salience=min(0.8, 0.4 + 0.1 * abs(z_score)),
            description=(
                f"Reward surprise ({label}): z={z_score:.2f}, "
                f"reward={current:.3f}, mean={mean:.3f}"
            ),
            evidence={
                "agent_id": self._agent_id,
                "z_score": z_score,
                "reward": current,
                "mean": mean,
                "std": std,
            },
        )

    def get_statistics(self) -> dict:
        return {
            "agent_id": self._agent_id,
            "rewards_buffered": len(self._rewards),
            "actions_buffered": len(self._actions),
            "last_step": self._step,
        }

    def __repr__(self) -> str:
        return f"PatternSense(agent={self._agent_id}, step={self._step})"
