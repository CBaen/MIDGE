"""
learning_config.py - Self-modifiable learning parameters

This file CAN be modified by the meta-learner.
All changes are logged to config_history.jsonl.
"""

from datetime import datetime

LEARNING_CONFIG = {
    # Metadata
    "version": 1,
    "last_modified": "2026-01-14T00:00:00",
    "modified_by": "initial_setup",

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
    },

    # Source reliability defaults
    # Can be adjusted based on observed accuracy
    "source_reliability": {
        "sec_edgar": 0.95,
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
    from pathlib import Path

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
    history_path = Path(__file__).parent / "config_history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "key_path": key_path,
            "old_value": old_value,
            "new_value": new_value,
            "modified_by": modified_by,
            "version": LEARNING_CONFIG["version"],
        }) + "\n")

    return {
        "old_value": old_value,
        "new_value": new_value,
        "success": True,
    }
