# Crucible Round 2 Build Report

**Role:** Crucible — wiring tests + minor fixes
**Date:** 2026-02-28

---

## Part 1: Wiring Tests

**File created:** `tests/test_decomposition_wiring.py`

### What it tests

62 tests across 5 classes, all verifying that the monolith decompositions introduced by Forge and Anvil preserve backward compatibility and enforce the 500-line cap.

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestSignalReexports` | 26 | All 25 from_* adapters importable from mae_core.market.signal (+ count assertion) |
| `TestSignalAdapters` | 13 | All 6 signal_adapters sub-modules importable + each function callable |
| `TestBootstrapDecomposition` | 11 | All 5 bootstrap sub-modules importable + all orchestrator/function exports |
| `TestSensingDecomposition` | 6 | sensing_hook, sensing_fetchers, sensing_lifecycle importable and functional |
| `TestNoMonoliths` | 6 | 500-line cap enforced across all decomposed files |

### Test results

**60 passed, 2 xfailed** — both xfails are expected mid-decomposition states:

1. `test_sensing_hook_under_500` — `sensing_hook.py` is 576 lines. Anvil created `sensing_fetchers.py` (468 lines) and `sensing_lifecycle.py` (124 lines) but has not yet refactored sensing_hook.py to delegate to them. The sub-modules exist and import correctly. xfail is `strict=True` — it will become an unexpected pass once Anvil completes the delegation refactor.

2. `test_market_hooks_under_500` — `market_hooks.py` is 551 lines. Forge's decomposition of `bootstrap/market.py` (original: ~1400 lines) produced 5 clean sub-modules, but market_hooks.py absorbed the EventBus callbacks + step hooks + sensing hook wiring in one pass, landing 51 lines over the cap. xfail is `strict=True` — will pass once market_hooks.py is trimmed.

### Adapter verification

All 25 from_* converters confirmed present and callable at `mae_core.market.signal`:
- `regulatory` (5): from_insider_trade, from_form8k_event, from_filing_keyword, from_cluster_signal, from_correlation_signal
- `political` (2): from_congressional_trade, from_senate_trade
- `market_data` (6): from_short_interest, from_news_sentiment, from_earnings_event, from_macro_indicator, from_price_data, from_social_sentiment
- `technical` (2): from_ta_signal, from_session_sweep
- `contracts` (4): from_government_contract, from_contract_opportunity, from_contract_prediction, from_hiring_signal
- `layer6` (6): from_cot_positioning, from_stocktwits_sentiment, from_vix_structure, from_trends_signal, from_economic_event, from_analyst_recommendation

---

## Part 2: hypothesis_activity.jsonl Bug Investigation

### Observation

Every entry in `data/midge/hypothesis_activity.jsonl` shows:
- `"step": 100` — always exactly 100
- `"agent": "42"` — always exactly 42
- `"role"` cycles between `"HYPOTHESIS_EXPLORER"` and `"HYPOTHESIS_VALIDATOR"` — but same agent

This was suspicious. 100 steps, agent 42, same pattern repeated across 12+ separate sessions.

### Root cause

**Tests were writing to production data files.**

Traced the call path:
1. `test_validate_promoted` in `tests/test_market_actions.py` calls `act_market(agent, "exploit")`
2. `act_market` dispatches to `_hypothesis_validate(agent)`
3. `_hypothesis_validate` calls `_log_hypothesis_activity(agent, "promoted")`
4. `_log_hypothesis_activity` opens `_OUTPUT_DIR / "hypothesis_activity.jsonl"` in append mode
5. `_OUTPUT_DIR = Path("data/midge")` — a **relative path**, which resolves to the project root during test runs

The mock agent in `_make_agent()` uses hardcoded defaults `step_count=100` and `unique_id=42`. Three tests (`test_generate_success`, `test_validate_promoted`, `test_validate_retired`) triggered hypothesis logging without patching `_OUTPUT_DIR` to a temp path. The `test_broadcast_with_alert` test correctly patched `_OUTPUT_DIR` for `agent_activity.jsonl` but the hypothesis tests were missing the same protection.

**Why step=100 always:** Every test run creates a fresh mock agent with `step_count=100`. After a `--steps 100` real run, each agent would also land at step_count=100. Either way, the value 100 appeared in every session's log, creating the illusion of a frozen counter.

**Why agent=42 always:** Mesa assigns sequential unique_ids. With a fixed agent population, the same agent always gets unique_id=42. In tests, `_make_agent()` hardcodes `unique_id=42`.

**Why both roles appear:** The HYPOTHESIS_EXPLORER test (`test_generate_success`) and HYPOTHESIS_VALIDATOR tests (`test_validate_promoted`, `test_validate_retired`) each write separate JSONL entries with different roles but the same fake agent attributes.

### Fix applied

Added `tmp_path` parameter and `patch("mae_core.market.market_actions._OUTPUT_DIR", tmp_path)` to the three tests that trigger hypothesis logging:

- `TestHypothesisExplorer.test_generate_success`
- `TestHypothesisValidator.test_validate_promoted`
- `TestHypothesisValidator.test_validate_retired`

Also added assertions in each test verifying the log was written to `tmp_path` (not the production path) with the correct event type and data.

**Result:** All 41 tests in `test_market_actions.py` pass. Future test runs will no longer append mock data to `data/midge/hypothesis_activity.jsonl`.

### Note on existing JSONL data

The existing `data/midge/hypothesis_activity.jsonl` is contaminated with test data. It should be cleared or noted as unreliable. The data is gitignored (data/ directory), so no repository contamination — but any future analysis of this file for real production runs should start fresh after this fix is in.

---

## Remaining xfail work items for Forge and Anvil

These are not Crucible's to fix — documenting for the review cycle:

| File | Current lines | Over by | Owner | Action needed |
|------|--------------|---------|-------|---------------|
| `mae_core/market/sensing_hook.py` | 576 | 76 | Anvil | Refactor to delegate fetch methods to sensing_fetchers.py |
| `mae_core/bootstrap/market_hooks.py` | 551 | 51 | Forge | Extract EventBus or step hook logic into a sub-module |

Both xfail markers are `strict=True` — they will automatically break the test suite if those files are accidentally reduced to under 500 without removing the xfail, preventing silent regression.

---

## Files changed

| File | Change |
|------|--------|
| `tests/test_decomposition_wiring.py` | Created — 62 wiring tests |
| `tests/test_market_actions.py` | Fixed — patched _OUTPUT_DIR in 3 hypothesis tests + added log assertions |
