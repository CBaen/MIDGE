"""Global test fixtures for MIDGE.

The autouse fixture below prevents ALL tests from touching production
Thompson data files.  Every test that instantiates ThompsonSampler()
without an explicit persistence_path will write to a throw-away temp
directory instead of data/market/.
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_thompson(tmp_path, monkeypatch):
    """Redirect Thompson module-level paths to a temp directory."""
    import mae_core.market.intelligence.thompson_sampler as ts_mod

    monkeypatch.setattr(ts_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ts_mod, "DISTRIBUTIONS_FILE", tmp_path / "thompson_distributions.json")
    monkeypatch.setattr(ts_mod, "HISTORY_FILE", tmp_path / "thompson_history.jsonl")
