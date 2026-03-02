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

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from mae_core.market.intelligence.hypothesis import (
    Hypothesis,
    HypothesisStats,
    HypothesisStatus,
    SourceType,
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

        # Global DSR counter — tracks total hypotheses ever tested
        # Persisted so it survives restarts
        self._dsr_state_path = self._data_dir / "dsr_state.json"
        self._dsr_trials_tracked = self._load_dsr_trials()

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
        self._save_dsr_trials()

        # BACKTEST_DERIVED hypotheses have pre-computed stats from
        # historical trades. Use them directly — the backtest IS the evidence.
        if hypothesis.source_type == SourceType.BACKTEST_DERIVED:
            return self._validate_from_precomputed(hypothesis)

        # Find historical trigger events — composite hypotheses require both sources
        is_composite = bool(hypothesis.trigger.conjunct_source)
        if is_composite:
            trigger_events = self._find_composite_trigger_events(
                hypothesis, lookback_days
            )
        else:
            trigger_events = self._find_trigger_events(
                hypothesis, lookback_days
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
            outcome = self._check_event_outcome(event, hypothesis)
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
                retire_reason = f"DSR {dsr:.3f} < {_rdsr} after {total} obs (multiple testing penalty)"

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
        dsr = self._compute_dsr(sharpe, total)

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

    def _find_trigger_events(
        self,
        hypothesis: Hypothesis,
        lookback_days: int,
    ) -> List[dict]:
        """Find historical instances where source_a fired.

        Scans signal archive JSONL files for signals matching
        hypothesis.trigger.source_a within lookback window.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)
        events = []

        if not self._signals_dir.exists():
            return events

        for jsonl_file in sorted(self._signals_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        source = record.get("source", "")
                        if source != hypothesis.trigger.source_a:
                            continue

                        ts_str = record.get("timestamp", "")
                        if not ts_str:
                            continue
                        try:
                            ts = datetime.fromisoformat(ts_str)
                        except ValueError:
                            continue

                        if ts < cutoff:
                            continue

                        events.append(record)
            except Exception:
                continue

        return events

    def _find_composite_trigger_events(
        self,
        hypothesis: Hypothesis,
        lookback_days: int,
    ) -> List[dict]:
        """Find historical instances where BOTH source_a AND conjunct_source fired.

        For each source_a event, checks whether conjunct_source also fired
        within ± lag_days/2 of the source_a event. Only events where both
        conditions are met are returned.

        This ensures composite hypotheses are validated against co-firing
        patterns, not just single-source events.
        """
        cutoff = datetime.now() - timedelta(days=lookback_days)

        # First pass: collect all source_a events
        source_a_events = self._find_trigger_events(hypothesis, lookback_days)
        if not source_a_events:
            return []

        # Second pass: collect all conjunct_source events for window matching
        conjunct_source = hypothesis.trigger.conjunct_source
        lag = hypothesis.trigger.lag_days
        half_window = timedelta(days=max(1, lag // 2))

        conjunct_events: List[dict] = []
        if self._signals_dir.exists():
            for jsonl_file in sorted(self._signals_dir.glob("*.jsonl")):
                try:
                    with open(jsonl_file) as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if record.get("source", "") != conjunct_source:
                                continue

                            ts_str = record.get("timestamp", "")
                            if not ts_str:
                                continue
                            try:
                                ts = datetime.fromisoformat(ts_str)
                            except ValueError:
                                continue

                            if ts < cutoff:
                                continue

                            conjunct_events.append(record)
                except Exception:
                    continue

        if not conjunct_events:
            return []

        # Third pass: keep only source_a events that have a matching conjunct event
        matched_events = []
        for event_a in source_a_events:
            ts_a_str = event_a.get("timestamp", "")
            if not ts_a_str:
                continue
            try:
                ts_a = datetime.fromisoformat(ts_a_str)
            except ValueError:
                continue

            symbol_a = event_a.get("symbol", "")
            window_start = ts_a - half_window
            window_end = ts_a + half_window

            for event_c in conjunct_events:
                # Symbol must match if source_a had a symbol
                if symbol_a and event_c.get("symbol", "") != symbol_a:
                    continue

                ts_c_str = event_c.get("timestamp", "")
                if not ts_c_str:
                    continue
                try:
                    ts_c = datetime.fromisoformat(ts_c_str)
                except ValueError:
                    continue

                if window_start <= ts_c <= window_end:
                    matched_events.append(event_a)
                    break  # One conjunct match per source_a event is enough

        return matched_events

    def _check_event_outcome(
        self,
        trigger_event: dict,
        hypothesis: Hypothesis,
    ) -> Optional[tuple]:
        """Check if a trigger event resulted in a successful prediction.

        Looks for outcome records in outcomes.jsonl where the source matches
        hypothesis.trigger.source_b and the timing aligns with the lag window.

        Returns (pct_return, success) or None if no matching outcome found.
        """
        trigger_ts_str = trigger_event.get("timestamp", "")
        if not trigger_ts_str:
            return None
        try:
            trigger_ts = datetime.fromisoformat(trigger_ts_str)
        except ValueError:
            return None

        trigger_symbol = trigger_event.get("symbol", "")
        lag = hypothesis.trigger.lag_days
        expected_outcome_start = trigger_ts + timedelta(days=max(1, lag - 10))
        expected_outcome_end = trigger_ts + timedelta(days=lag + 10)

        if not self._outcomes_path.exists():
            return None

        try:
            with open(self._outcomes_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        outcome = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    outcome_source = outcome.get("source", "")
                    if outcome_source != hypothesis.trigger.source_b:
                        continue

                    # Symbol match (if trigger had a symbol)
                    if trigger_symbol and outcome.get("symbol", "") != trigger_symbol:
                        continue

                    predicted_at_str = outcome.get("predicted_at", "")
                    if not predicted_at_str:
                        continue
                    try:
                        predicted_at = datetime.fromisoformat(predicted_at_str)
                    except ValueError:
                        continue

                    if expected_outcome_start <= predicted_at <= expected_outcome_end:
                        pct = outcome.get("price_change_pct", 0.0)
                        success = outcome.get("success", False)

                        # Adjust for hypothesis direction
                        if hypothesis.trigger.direction == "negative":
                            # Negative correlation: source_a up → source_b down
                            direction_match = pct < 0
                        elif hypothesis.trigger.direction == "positive":
                            direction_match = pct > 0
                        else:
                            direction_match = True

                        return (pct, success and direction_match)
        except Exception:
            pass

        return None

    def _compute_sharpe(self, returns: List[float]) -> float:
        """Compute annualized Sharpe ratio from returns series."""
        if len(returns) < 2:
            return 0.0

        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 1e-10

        # Annualize: assume ~12 observations per year (monthly-ish cadence)
        return (mean_r / std_r) * math.sqrt(12)

    def _compute_dsr(self, sharpe: float, n_obs: int) -> float:
        """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

        DSR adjusts the observed Sharpe ratio for the number of independent
        trials (hypotheses tested). The more you test, the higher the bar.

        DSR = SR * sqrt(n) / sqrt(1 + (SR^2/2) * ((gamma_3 * SR / 3) + ((gamma_4 - 3) / 4)))
              adjusted by E[max(SR)] over M trials

        Simplified approximation used here:
            E[max(SR)] ≈ sqrt(2 * log(M)) * (1 - gamma / (2 * log(M)))
            where M = dsr_trials_tracked, gamma = Euler-Mascheroni constant

        The DSR penalizes the observed Sharpe by the expected maximum Sharpe
        you'd get from M random strategies.
        """
        if n_obs < 2 or self._dsr_trials_tracked < 1:
            return 0.0

        M = max(1, self._dsr_trials_tracked)

        # Expected maximum Sharpe from M random trials
        # (Bonferroni-style approximation)
        euler_gamma = 0.5772156649
        log_M = math.log(max(2, M))
        e_max_sr = math.sqrt(2 * log_M) * (1 - euler_gamma / (2 * log_M))

        # Standard error of the Sharpe ratio
        se_sr = math.sqrt((1 + 0.5 * sharpe ** 2) / max(1, n_obs))

        # DSR = P(SR > E[max(SR)]) approximated as z-score
        if se_sr < 1e-10:
            return 0.0

        dsr = (sharpe - e_max_sr) / se_sr
        return dsr

    # ── Persistence ─────────────────────────────────────────────────

    def _load_dsr_trials(self) -> int:
        """Load global DSR trial counter."""
        if self._dsr_state_path.exists():
            try:
                data = json.loads(self._dsr_state_path.read_text())
                return data.get("trials_tracked", 0)
            except (json.JSONDecodeError, Exception):
                pass
        return 0

    def _save_dsr_trials(self) -> None:
        """Persist global DSR trial counter."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._dsr_state_path.write_text(json.dumps({
                "trials_tracked": self._dsr_trials_tracked,
                "last_updated": datetime.now().isoformat(),
            }))
        except Exception as e:
            logger.warning("Failed to persist DSR state: %s", e)

    def get_statistics(self) -> dict:
        """Summary stats for monitoring."""
        return {
            "dsr_trials_tracked": self._dsr_trials_tracked,
        }
