"""Global test fixtures for MIDGE.

The autouse fixture below prevents ALL tests from touching production
data files.  Every test that instantiates ThompsonSampler(), LearningConfig,
or HypothesisGenerator without explicit paths will use throw-away temp
directories instead of data/market/.

The HistoricalDataFetcher guard prevents tests from loading the 130 MB
production signal archive (911+ JSONL files in data/midge/signals/).
"""

import copy
import gc
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_market_state(tmp_path, monkeypatch):
    """Redirect all mutable module-level market state to temp directories."""
    import mae_core.market.intelligence.thompson_sampler as ts_mod

    monkeypatch.setattr(ts_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ts_mod, "DISTRIBUTIONS_FILE", tmp_path / "thompson_distributions.json")
    monkeypatch.setattr(ts_mod, "HISTORY_FILE", tmp_path / "thompson_history.jsonl")

    # Also isolate ThompsonCalibrator's DATA_DIR
    try:
        import mae_core.market.intelligence.thompson_calibrator as tc_mod
        monkeypatch.setattr(tc_mod, "DATA_DIR", tmp_path)
        monkeypatch.setattr(tc_mod, "CALIBRATION_PATH", tmp_path / "calibration_report.json")
    except (ImportError, AttributeError):
        pass

    # Also isolate OutcomeCollector's default data paths
    try:
        import mae_core.market.intelligence.outcome_collector as oc_mod
        if hasattr(oc_mod, "DATA_DIR"):
            monkeypatch.setattr(oc_mod, "DATA_DIR", tmp_path)
    except (ImportError, AttributeError):
        pass

    # Prevent tests from loading the 130 MB production signal archive.
    # Tests that need HistoricalDataFetcher get an empty temp directory.
    try:
        import mae_core.market.archaeology.historical_fetcher as hf_mod
        monkeypatch.setattr(hf_mod, "SIGNAL_ARCHIVE_DIR", tmp_path / "signals")
    except (ImportError, AttributeError):
        pass

    # Isolate LEARNING_CONFIG: snapshot before test, restore after.
    # create_mae() and meta-learning mutate this module-level dict in place,
    # which pollutes subsequent tests (e.g. min_correlation 0.6 → 0.85).
    try:
        import mae_core.market.intelligence.learning_config as lc_mod
        original_config = copy.deepcopy(lc_mod.LEARNING_CONFIG)
        monkeypatch.setattr(lc_mod, "_DATA_DIR", tmp_path)
        monkeypatch.setattr(lc_mod, "_HISTORY_PATH", tmp_path / "config_history.jsonl")
        yield
        lc_mod.LEARNING_CONFIG.clear()
        lc_mod.LEARNING_CONFIG.update(original_config)
    except (ImportError, AttributeError):
        yield

    # Force garbage collection after each test to break circular references
    # in large objects (ConvergenceAlerter, MarketSensingHook, etc.)
    gc.collect()
