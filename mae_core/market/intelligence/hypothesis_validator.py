"""Hypothesis Validator — Adversarial validation with Deflated Sharpe Ratio.

The validator's job is to DISPROVE hypotheses. It runs retrospective event
studies on signal archives, computes performance metrics, and recommends
promotion or retirement.

Key metric: Deflated Sharpe Ratio (DSR) from Bailey & Lopez de Prado (2014).
DSR adjusts the Sharpe ratio for the number of hypotheses tested on the same
data — the more hypotheses you test, the higher the bar each one must clear.
This is THE anti-overfitting gate that prevents the system from promoting
spurious patterns.

Promotion bars (all must hold):
  - observations >= 20
  - win_rate > 0.52
  - DSR > 0.5
  - causal_story present and not "REQUIRES MANUAL REVIEW"

Retirement triggers (any one):
  - observations >= 20 AND win_rate < 0.45
  - DSR < 0 after 20+ observations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
)
from mae_core.market.intelligence.hypothesis_dsr import (  # noqa: F401
    compute_sharpe,
    compute_dsr,
    load_dsr_trials,
    save_dsr_trials,
)
from mae_core.market.intelligence.hypothesis_event_search import (  # noqa: F401
    find_trigger_events,
    find_composite_trigger_events,
    check_event_outcome,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"

# ── Promotion / retirement thresholds ───────────────────────────────
# Hardcoded fallbacks (used if LEARNING_CONFIG is unavailable)
_GATE_FALLBACKS = {
    "min_observations": 20,
    "promote_win_rate": 0.52,
    "promote_dsr": 0.5,
    "retire_win_rate": 0.45,
    "retire_dsr": 0.0,
}

# Legacy module-level aliases for backward compatibility with tests
MIN_OBSERVATIONS = _GATE_FALLBACKS["min_observations"]
PROMOTE_WIN_RATE = _GATE_FALLBACKS["promote_win_rate"]
PROMOTE_DSR = _GATE_FALLBACKS["promote_dsr"]
RETIRE_WIN_RATE = _GATE_FALLBACKS["retire_win_rate"]
RETIRE_DSR = _GATE_FALLBACKS["retire_dsr"]


def _get_gate(key: str, regime: str = "default") -> float:
    """Read a hypothesis gate from learning_config, with regime delta and fallback.

    Graceful degradation: if learning_config import fails or key is missing,
    returns the hardcoded fallback value. This means the system works identically
    to before if the config sections are absent.
    """
    try:
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        gates = LEARNING_CONFIG.get("hypothesis_gates", {})
    except ImportError:
        return float(_GATE_FALLBACKS[key])

    base = gates.get(key, _GATE_FALLBACKS.get(key, 0.0))
    delta = gates.get("_regime_deltas", {}).get(regime, {}).get(key, 0.0)
    return float(base + delta)


@dataclass
class ValidationResult:
    """Result of validating a hypothesis against historical data."""
    hypothesis_id: str
    wins: int = 0
    losses: int = 0
    total_observations: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    deflated_sharpe_ratio: float = 0.0
    recommend_promote: bool = False
    recommend_retire: bool = False
    retire_reason: str = ""


class HypothesisValidator:
    """Adversarial hypothesis validator using event studies and DSR.

    The validator tracks a global counter of hypotheses tested
    (dsr_trials_tracked). This counter is the input to the DSR
    formula — more hypotheses tested = higher bar for each one.
    """

    def __init__(
        self,
        signals_dir: Path = None,
        outcomes_path: Path = None,
        data_dir: Path = None,
        causal_engine=None,
    ):
        self._data_dir = data_dir or DATA_DIR
        self._signals_dir = signals_dir or (
            Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"
        )
        self._outcomes_path = outcomes_path or (self._data_dir / "outcomes.jsonl")

        self._causal_engine = causal_engine

        # Global DSR counter — tracks total hypotheses ever tested
        # Persisted so it survives restarts
        self._dsr_state_path = self._data_dir / "dsr_state.json"
        self._dsr_trials_tracked = load_dsr_trials(self._dsr_state_path)

    def validate(
        self,
        hypothesis: Hypothesis,
        lookback_days: int = 180,
    ) -> ValidationResult:
        """Validate a hypothesis against historical signal archives.

        Finds all historical instances where the trigger pattern fired,
        checks outcomes, computes performance metrics and DSR.

        Args:
            hypothesis: The hypothesis to validate.
            lookback_days: How far back to search for trigger events.

        Returns:
            ValidationResult with promote/retire recommendations.
        """
        # Increment global trial counter
        self._dsr_trials_tracked += 1
        save_dsr_trials(self._dsr_state_path, self._data_dir, self._dsr_trials_tracked)

        # BACKTEST_DERIVED hypotheses have pre-computed stats from
        # historical trades. Use them directly — the backtest IS the evidence.
        if hypothesis.source_type == SourceType.BACKTEST_DERIVED:
            return self._validate_from_precomputed(hypothesis)

        # Find historical trigger events — composite hypotheses require both sources
        is_composite = bool(hypothesis.trigger.conjunct_source)
        if is_composite:
            trigger_events = find_composite_trigger_events(
                hypothesis, self._signals_dir, lookback_days
            )
        else:
            trigger_events = find_trigger_events(
                hypothesis, self._signals_dir, lookback_days
            )

        if not trigger_events:
            return ValidationResult(
                hypothesis_id=hypothesis.hypothesis_id,
            )

        # Check outcomes for each trigger event
        wins = 0
        losses = 0
        returns = []

        for event in trigger_events:
            outcome = check_event_outcome(event, hypothesis, self._outcomes_path)
            if outcome is None:
                continue

            pct_return, success = outcome
            returns.append(pct_return)
            if success:
                wins += 1
            else:
                losses += 1

        total = wins + losses
        if total == 0:
            return ValidationResult(
                hypothesis_id=hypothesis.hypothesis_id,
            )

        win_rate = wins / total
        sharpe = self._compute_sharpe(returns)
        dsr = self._compute_dsr(sharpe, total)

        # Promotion/retirement decisions
        has_real_causal_story = (
            hypothesis.causal_story
            and "REQUIRES MANUAL REVIEW" not in hypothesis.causal_story
        )

        # Read live gates from config (with regime delta + fallback)
        _min_obs = _get_gate("min_observations")
        _pwr = _get_gate("promote_win_rate")
        _pdsr = _get_gate("promote_dsr")
        _rwr = _get_gate("retire_win_rate")
        _rdsr = _get_gate("retire_dsr")

        # Composite hypotheses are more specific — need more evidence
        is_composite = bool(hypothesis.trigger.conjunct_source)
        if is_composite:
            _min_obs += 10
            _pwr += 0.02

        # Auto-generated causal stories get a slightly tighter win rate bar
        if hypothesis.causal_story.startswith("[AUTO]"):
            _pwr += 0.01

        # Causal engine confounding check — if the correlation is likely
        # driven by a hidden confounder, tighten the promotion gate.
        _confounded = False
        if (self._causal_engine is not None
                and hypothesis.trigger.source_a
                and hypothesis.trigger.source_b):
            try:
                result = self._causal_engine.query_causation(
                    hypothesis.trigger.source_a,
                    hypothesis.trigger.source_b,
                )
                if hasattr(result, "relation_type"):
                    from mae_core.cognition.causal_reasoning import CausalRelationType
                    if result.relation_type == CausalRelationType.CONFOUNDED:
                        _pwr += 0.03
                        _confounded = True
                        logger.info(
                            "Hypothesis %s: causal engine flags confounded (%s → %s), "
                            "tightening promote_win_rate to %.3f",
                            hypothesis.name, hypothesis.trigger.source_a,
                            hypothesis.trigger.source_b, _pwr,
                        )
            except Exception:
                logger.debug("Causal engine query failed for %s", hypothesis.name, exc_info=True)

        recommend_promote = (
            total >= _min_obs
            and win_rate > _pwr
            and dsr > _pdsr
            and has_real_causal_story
        )

        recommend_retire = False
        retire_reason = ""
        if total >= _min_obs:
            if win_rate < _rwr:
                recommend_retire = True
                retire_reason = f"Win rate {win_rate:.3f} < {_rwr} after {total} obs"
            elif dsr < _rdsr:
                recommend_retire = True
                retire_reason = (
                    f"DSR {dsr:.3f} < {_rdsr} after {total} obs "
                    f"(multiple testing penalty)"
                )

        result = ValidationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            wins=wins,
            losses=losses,
            total_observations=total,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            deflated_sharpe_ratio=dsr,
            recommend_promote=recommend_promote,
            recommend_retire=recommend_retire,
            retire_reason=retire_reason,
        )

        logger.info(
            "Validated %s: %d/%d wins (%.1f%%), SR=%.3f, DSR=%.3f → %s",
            hypothesis.name, wins, total, win_rate * 100,
            sharpe, dsr,
            "PROMOTE" if recommend_promote else
            ("RETIRE" if recommend_retire else "HOLD"),
        )

        return result

    def _validate_from_precomputed(
        self,
        hypothesis: Hypothesis,
    ) -> ValidationResult:
        """Validate using pre-populated stats (BACKTEST_DERIVED only).

        The backtest already computed wins, losses, and Sharpe from real
        historical trades. We trust these numbers and only compute DSR
        (which requires the global trial counter) here.

        No archive scanning. No outcome matching. The data is already there.
        """
        wins = hypothesis.stats.wins
        losses = hypothesis.stats.losses
        total = hypothesis.stats.total_observations

        if total == 0:
            return ValidationResult(
                hypothesis_id=hypothesis.hypothesis_id,
            )

        win_rate = wins / total
        sharpe = hypothesis.stats.sharpe_ratio
        dsr = compute_dsr(sharpe, total, self._dsr_trials_tracked)

        has_real_causal_story = (
            hypothesis.causal_story
            and "REQUIRES MANUAL REVIEW" not in hypothesis.causal_story
        )

        # Read live gates from config (with regime delta + fallback)
        _min_obs = _get_gate("min_observations")
        _pwr = _get_gate("promote_win_rate")
        _pdsr = _get_gate("promote_dsr")
        _rwr = _get_gate("retire_win_rate")
        _rdsr = _get_gate("retire_dsr")

        recommend_promote = (
            total >= _min_obs
            and win_rate > _pwr
            and dsr > _pdsr
            and has_real_causal_story
        )

        recommend_retire = False
        retire_reason = ""
        if total >= _min_obs:
            if win_rate < _rwr:
                recommend_retire = True
                retire_reason = (
                    f"Backtest win rate {win_rate:.3f} < {_rwr} "
                    f"over {total} historical trades"
                )
            elif dsr < _rdsr:
                recommend_retire = True
                retire_reason = (
                    f"DSR {dsr:.3f} < {_rdsr} after multiple-testing correction "
                    f"({self._dsr_trials_tracked} trials tracked)"
                )

        result = ValidationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            wins=wins,
            losses=losses,
            total_observations=total,
            win_rate=win_rate,
            sharpe_ratio=sharpe,
            deflated_sharpe_ratio=dsr,
            recommend_promote=recommend_promote,
            recommend_retire=recommend_retire,
            retire_reason=retire_reason,
        )

        logger.info(
            "Validated (backtest) %s: %d/%d wins (%.1f%%), SR=%.3f, DSR=%.3f → %s",
            hypothesis.name, wins, total, win_rate * 100,
            sharpe, dsr,
            "PROMOTE" if recommend_promote else
            ("RETIRE" if recommend_retire else "HOLD"),
        )

        return result

    # ── Private method aliases for backward compatibility ──────────────
    # Some tests may call these as instance methods via the validator object.

    def _find_trigger_events(self, hypothesis, lookback_days: int) -> list:
        return find_trigger_events(hypothesis, self._signals_dir, lookback_days)

    def _find_composite_trigger_events(self, hypothesis, lookback_days: int) -> list:
        return find_composite_trigger_events(hypothesis, self._signals_dir, lookback_days)

    def _check_event_outcome(self, trigger_event: dict, hypothesis) -> Optional[tuple]:
        return check_event_outcome(trigger_event, hypothesis, self._outcomes_path)

    def _compute_sharpe(self, returns: List[float]) -> float:
        return compute_sharpe(returns)

    def _compute_dsr(self, sharpe: float, n_obs: int) -> float:
        return compute_dsr(sharpe, n_obs, self._dsr_trials_tracked)

    def _load_dsr_trials(self) -> int:
        return load_dsr_trials(self._dsr_state_path)

    def _save_dsr_trials(self) -> None:
        save_dsr_trials(self._dsr_state_path, self._data_dir, self._dsr_trials_tracked)

    def get_statistics(self) -> dict:
        """Summary stats for monitoring."""
        return {
            "dsr_trials_tracked": self._dsr_trials_tracked,
        }
