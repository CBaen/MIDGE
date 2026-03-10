"""DSR/Sharpe computation and persistence for HypothesisValidator."""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def compute_sharpe(returns: List[float]) -> float:
    """Compute annualized Sharpe ratio from returns series."""
    if len(returns) < 2:
        return 0.0

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std_r = math.sqrt(variance) if variance > 0 else 1e-10

    # Annualize: assume ~12 observations per year (monthly-ish cadence)
    return (mean_r / std_r) * math.sqrt(12)


def compute_dsr(sharpe: float, n_obs: int, dsr_trials_tracked: int) -> float:
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    DSR adjusts the observed Sharpe ratio for the number of independent
    trials (hypotheses tested). The more you test, the higher the bar.

    DSR = SR * sqrt(n) / sqrt(1 + (SR^2/2) * ((gamma_3 * SR / 3) + ((gamma_4 - 3) / 4)))
          adjusted by E[max(SR)] over M trials

    Simplified approximation used here:
        E[max(SR)] ~ sqrt(2 * log(M)) * (1 - gamma / (2 * log(M)))
        where M = dsr_trials_tracked, gamma = Euler-Mascheroni constant

    The DSR penalizes the observed Sharpe by the expected maximum Sharpe
    you'd get from M random strategies.
    """
    if n_obs < 2 or dsr_trials_tracked < 1:
        return 0.0

    M = max(1, dsr_trials_tracked)

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


def load_dsr_trials(dsr_state_path: Path) -> int:
    """Load global DSR trial counter."""
    if dsr_state_path.exists():
        try:
            data = json.loads(dsr_state_path.read_text())
            return data.get("trials_tracked", 0)
        except (json.JSONDecodeError, Exception):
            pass
    return 0


def save_dsr_trials(dsr_state_path: Path, data_dir: Path, dsr_trials_tracked: int) -> None:
    """Persist global DSR trial counter."""
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        dsr_state_path.write_text(json.dumps({
            "trials_tracked": dsr_trials_tracked,
            "last_updated": datetime.now().isoformat(),
        }))
    except Exception as e:
        logger.warning("Failed to persist DSR state: %s", e)
