"""
learning_config.py - Self-modifiable learning parameters

This file CAN be modified by the meta-learner.
All changes are logged to config_history.jsonl.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "market"
_HISTORY_PATH = _DATA_DIR / "config_history.jsonl"

LEARNING_CONFIG = {
    # Metadata
    "version": 1,
    "last_modified": "2026-01-14T00:00:00",
    "modified_by": "initial_setup",

    # Paper trading account — denominator for Kelly fraction position sizing
    "paper_account_value": 50000,

    # Learning parameters (can be adjusted by meta-learner)
    "learning_rate": 0.1,
    "min_reliability": 0.1,
    "max_reliability": 0.95,

    # Signal decay rates (per day)
    # Can be adjusted based on observed signal effectiveness
    "decay_rates": {
        "news": 0.5,           # Half-life: ~1.4 days
        "sentiment": 0.3,      # Half-life: ~2.3 days
        "technical": 0.1,      # Half-life: ~7 days
        "insider": 0.05,       # Half-life: ~14 days
        "institutional": 0.03, # Half-life: ~23 days
        "politician": 0.04,    # Half-life: ~17 days
        "contract": 0.02,      # Half-life: ~35 days
        "research": 0.01,      # Half-life: ~69 days
        "price": 0.15,         # Half-life: ~5 days
        # New domains (Layer 6)
        "positioning": 0.03,   # Half-life: ~23 days (weekly COT data)
        "volatility": 0.30,    # Half-life: ~2.3 days (VIX changes daily)
        # Absence signals (the dog that didn't bark)
        "absence": 0.15,       # Half-life: ~4.6 days (medium-duration)
        # Ten Gifts (Wave 1-2)
        "order_flow": 0.30,    # Half-life: ~2.3 days (intraday signal, fast decay)
        "archetype": 0.03,     # Half-life: ~23 days (archetypes unfold over weeks)
        "crypto": 0.50,        # Half-life: ~1.4 days (crypto moves fast)
    },

    # Source reliability defaults
    # Can be adjusted based on observed accuracy
    "source_reliability": {
        # Legacy keys (retained for backward compatibility with existing distributions)
        "sec_edgar": 0.59,
        "13f_filing": 0.90,
        "form_4": 0.90,
        "capitol_trades": 0.85,
        "unusual_whales": 0.80,
        "polygon": 0.95,
        "twitter_verified": 0.60,
        "reddit": 0.30,
        "stocktwits": 0.50,
        "gemini": 0.75,
        "deepseek": 0.70,
        "unknown": 0.50,
        # Signal-source-level keys (match MarketSignal.source values from signal.py)
        "sec_form4": 0.36,
        "sec_form8k": 0.65,
        "sec_efts": 0.70,
        "congressional": 0.20,
        "senate": 0.75,
        "insider_cluster": 0.85,
        "contract_award": 0.15,
        "contract_prediction": 0.60,
        "hiring_tracker": 0.60,
        "sam_gov": 0.40,
        "social_sentiment": 0.30,
        "finra_short": 0.36,
        "finnhub_news": 0.60,
        "finnhub_earnings": 0.27,
        "fred_macro": 0.70,
        "session_sweep": 0.55,
        "session_sweep_ifvg": 0.58,
        "yfinance_price": 0.23,
        "ta_rsi": 0.60,
        "ta_macd": 0.53,
        "ta_bollinger": 0.55,
        "ta_structure": 0.55,
        "ta_candle": 0.55,
        # New sources (Layer 6)
        "cot_positioning": 0.55,
        "stocktwits_sentiment": 0.50,
        "vix_term_structure": 0.60,
        "google_trends": 0.45,
        "finnhub_economic": 0.55,
        "finnhub_analyst": 0.50,
        "finnhub_earnings_calendar": 0.55,
        # Contextual Thompson priors (sub-source discrimination — Capability 2)
        # These are keyed by {source}:{role}, {source}:{role}:{sector}, etc.
        # The cascade falls through to the base source key if no contextual key
        # has enough data (>= 5 samples in Thompson).
        "sec_form4:officer": 0.45,          # Officers have more info than directors
        "sec_form4:officer:large": 0.55,    # Large officer buys = high conviction
        "sec_form4:director": 0.30,         # Directors have less predictive power
        "sec_form4:10pct_owner": 0.35,      # Major shareholders, mixed signal
        "congressional:senate": 0.25,       # Senate committee trades
        "congressional:house": 0.18,        # House trades, less informed
        "finra_short:large_change": 0.45,   # Large short interest changes
        "finra_short:small_change": 0.30,   # Small changes are noise
        "finnhub_earnings:beat": 0.40,      # Earnings beats
        "finnhub_earnings:miss": 0.35,      # Earnings misses
        "contract_award:defense": 0.20,     # Defense contract awards
        "contract_award:tech": 0.18,        # Tech contract awards
        "ta_rsi:oversold": 0.65,            # RSI oversold more reliable
        "ta_rsi:overbought": 0.58,          # RSI overbought slightly less
        "ta_macd:crossover": 0.55,          # MACD crossover signals
        # Absence detection (the dog that didn't bark)
        "absence_signal": 0.50,             # Neutral prior — let the data teach
        # Ten Gifts (Wave 1-2)
        "order_flow": 0.50,                 # Neutral prior — intraday volume imbalance
        "fractal_resonance": 0.55,          # Slightly optimistic (multi-timeframe = robust)
        "archetype_match": 0.55,            # Slightly optimistic (historical backing)
        # Wave 2: Real-Time + Crypto (Always-On)
        "finnhub_realtime": 0.50,       # Neutral prior — real-time signals are new
        "crypto_coingecko": 0.50,       # Neutral prior — crypto source
        "crypto_coincap": 0.50,         # Neutral prior — second crypto source
        # Wave 3: Data Enrichment
        "openinsider_purchase": 0.55,   # Pre-filtered insider buys = higher signal
        "institutional_13f": 0.50,      # 45-day lag tempers initial confidence
        "activist_13d": 0.60,           # Activist filings are very high signal
        "finviz_unusual_volume": 0.50,  # Neutral prior — volume spikes
        "finviz_short_squeeze": 0.50,   # Neutral prior — squeeze candidates
        "economic_calendar": 0.55,      # Scheduled events are reliable signals
    },

    # Confidence calibration
    "confidence_calibration": {
        "underconfident_threshold": 0.6,  # If accuracy > 0.6 when confidence < 0.5
        "overconfident_threshold": 0.4,   # If accuracy < 0.4 when confidence > 0.7
        "adjustment_rate": 0.05
    },

    # Anti-overfitting parameters
    "overfitting_protection": {
        "holdout_days": 14,            # Days to hold out for validation
        "min_samples": 50,             # Min predictions before adjusting
        "max_adjustment": 0.1,         # Max change per adjustment
        "validation_threshold": 0.05,  # Reject if holdout accuracy drops more than this
    },

    # Meta-learning parameters
    "meta_learning": {
        "meta_learning_rate": 0.05,
        "evaluation_window_days": 30,
        "min_samples_for_adjustment": 20,
    },

    # Exploration vs exploitation
    "exploration": {
        "base_exploration_rate": 0.2,
        "novelty_decay_threshold": 0.8,  # Force exploration if novelty decays
        "performance_threshold": 0.6,    # Explore more if doing well
    },

    # Curiosity parameters
    "curiosity": {
        "novelty_threshold": 0.7,        # Log patterns with novelty > this
        "curiosity_weight": 0.3,         # Weight for curiosity in total reward
        "exploration_bonus": 0.1,        # Bonus for exploring new states
    },

    # Bridge 4: Dynamic hypothesis promotion/retirement gates
    # These replace the hardcoded constants in hypothesis_validator.py.
    # Adjustable at runtime via update_config(). _bounds prevent runaway drift.
    "hypothesis_gates": {
        "min_observations":  20,
        "promote_win_rate":  0.52,
        "promote_dsr":       0.5,
        "retire_win_rate":   0.45,
        "retire_dsr":        0.0,
        "_bounds": {
            "min_observations":  [10, 50],
            "promote_win_rate":  [0.50, 0.65],
            "promote_dsr":       [0.2, 0.8],
            "retire_win_rate":   [0.35, 0.50],
            "retire_dsr":        [-0.2, 0.1],
        },
        "_regime_deltas": {
            "volatile": {"promote_win_rate": 0.03, "promote_dsr": 0.1},
            "bear":     {"promote_win_rate": 0.02},
            "bull":     {"promote_win_rate": -0.01},
            "sideways": {},
            "default":  {},
        },
    },

    # Bridge 5: Generator thresholds for hypothesis creation
    # Adjustable by meta-learning based on retirement rate.
    "generator_thresholds": {
        "min_correlation": 0.6,
        "min_pairs":       10,
        "_bounds": {
            "min_correlation": [0.4, 0.85],
            "min_pairs":       [5, 30],
        },
    },
}


def get_config():
    """Return current config."""
    return LEARNING_CONFIG.copy()


def update_config(key_path: str, new_value, modified_by: str = "meta_learner"):
    """
    Update a config value.

    Args:
        key_path: Dot-separated path (e.g., "decay_rates.news")
        new_value: New value to set
        modified_by: Who made this change

    Returns:
        dict with old_value, new_value, success
    """
    import json

    keys = key_path.split(".")
    current = LEARNING_CONFIG

    # Navigate to parent
    for key in keys[:-1]:
        current = current[key]

    # Get old value
    old_value = current[keys[-1]]

    # Set new value
    current[keys[-1]] = new_value

    # Update metadata
    LEARNING_CONFIG["version"] += 1
    LEARNING_CONFIG["last_modified"] = datetime.now().isoformat()
    LEARNING_CONFIG["modified_by"] = modified_by

    # Log to history
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "key_path": key_path,
            "old_value": old_value,
            "new_value": new_value,
            "modified_by": modified_by,
            "version": LEARNING_CONFIG["version"],
        }) + "\n")

    save_snapshot()

    return {
        "old_value": old_value,
        "new_value": new_value,
        "success": True,
    }


def save_snapshot(path=None) -> None:
    """Persist the full LEARNING_CONFIG dict to disk.

    Writes to data/market/config_snapshot.json by default. Non-critical —
    failures are logged but do not raise. Called automatically by update_config().
    """
    snapshot_path = Path(path) if path else _DATA_DIR / "config_snapshot.json"
    tmp = snapshot_path.with_suffix(".tmp")
    try:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(LEARNING_CONFIG, indent=2))
        os.replace(tmp, snapshot_path)
    except Exception as e:
        logger.debug("Failed to save config snapshot: %s", e)
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def load_snapshot(path=None) -> bool:
    """Load a previously saved config snapshot and deep-merge into live LEARNING_CONFIG.

    Modifies LEARNING_CONFIG in-place (other modules hold references — do not
    replace the dict). Only updates keys that already exist in the base dict
    for forward compatibility with newer code that has added new keys.

    Returns True if a snapshot was found and merged successfully.
    """
    snapshot_path = Path(path) if path else _DATA_DIR / "config_snapshot.json"
    if not snapshot_path.exists():
        return False
    try:
        data = json.loads(snapshot_path.read_text())
        _deep_merge(LEARNING_CONFIG, data)
        logger.info("Loaded config snapshot from %s (version=%s)", snapshot_path, LEARNING_CONFIG.get("version"))
        return True
    except Exception as e:
        logger.warning("Failed to load config snapshot: %s", e)
        return False


def _deep_merge(base: dict, update: dict) -> None:
    """Recursive in-place dict merge. Overwrites leaf values in base with those
    from update, but only for keys that already exist in base. Nested dicts are
    merged recursively. Keys in update that are absent from base are ignored
    (forward-compatibility: new keys added in code take precedence).
    """
    for key, value in update.items():
        if key not in base:
            continue  # Ignore keys not in base (forward compatibility)
        if isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
