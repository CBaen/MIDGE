# Attention Builder Build Report — Round 1
## Date: 2026-03-09
## Builder: Attention Builder — Sensing Pipeline Priority Polling

---

### Tasks Completed

- **Task:** Implement priority queue reading + Thompson score boosting in sensing_hook.py
- **Files Changed:**
  - `mae_core/market/sensing_hook.py` — imports, module-level constant, `__init__`, `_launch_next_fetch`, `_launch_thompson_guided`
  - `tests/test_priority_polling.py` — new file, 23 tests

**Approach:**

Built `_DOMAIN_TO_SOURCES` as a module-level constant computed once at import time by `_build_domain_to_sources()`. The function iterates every SOURCE_ROTATION name and resolves its domain via two lookup paths: (1) direct match in `_ABSENCE_SOURCE_DOMAINS` (for rotation names that are also signal-source keys, like "congressional"), (2) via `_ROTATION_TO_THOMPSON` to the Thompson key, then into `_ABSENCE_SOURCE_DOMAINS` (for names like "sec_form4" whose Thompson key matches the absence key). Each rotation name is assigned to its primary domain. The result is a dict like `{"insider": ["sec_form4", "sec_form8k", "sec_efts", "openinsider"], ...}`.

Added `self._priority_requests: Dict[str, dict] = {}` to `__init__` between `_recent_domains` and the stats block, with a full docstring explaining the expected entry format and constraints.

Added expired-entry cleanup at the start of `_launch_next_fetch()` (wrapped in try/except per project constraint 1 — never block the step loop). This is the correct location: the Build Brief referenced "`_schedule_fetch()`" which does not exist by that name; `_launch_next_fetch` at line ~655 is the method that fits the described behavior.

Modified `_launch_thompson_guided()` to: (1) build `boosted_sources` set from all active priority request domains, guarded by the `0 < len(...) <= 50` cap check, (2) after drawing the Beta variate for each source, apply `score = min(1.0, score * 2.0)` if the source is in `boosted_sources`. The boost is applied after the random draw, ensuring it is additive on top of Thompson learning rather than overriding it.

**New Interfaces:**

`_DOMAIN_TO_SOURCES: Dict[str, List[str]]` — module-level constant, importable. Maps domain name to list of SOURCE_ROTATION names that serve it. Read-only. Used by market_hooks.py (or any other module) to validate which domains have coverage before writing priority requests.

`sensing_hook._priority_requests: Dict[str, dict]` — instance attribute on `MarketSensingHook`. Direct dict access (no setter method). Callers write entries in this format:
```python
hook._priority_requests[ticker] = {
    "ticker": str,           # ticker symbol (also the dict key)
    "domains_needed": list,  # list of domain strings, e.g. ["insider", "macro"]
    "priority": str,         # "high", "medium", "low" (informational only)
    "expires": float,        # time.time() + seconds_until_expiry (use 3600 for 1 hour)
    "source": str,           # who wrote this entry, e.g. "partial_convergence"
}
```
Entries are cleaned up automatically when they expire (on each `_launch_next_fetch()` call). Entries over 50 total disable boosting until count drops back to ≤50.

---

### Tasks Not Completed

None. All assigned tasks are complete.

---

### Cross-Domain Dependencies

**What market_hooks.py (another builder's domain) needs from mine:**

The priority polling system is a one-way write interface. `market_hooks.py` needs to:
1. Import or hold a reference to the `sensing_hook` instance (it already has `ctx`, and the sensing hook is available as `ctx.sensing_hook` or similar — confirm the exact attribute name from `mae_core/bootstrap/market_systems.py`)
2. Write a priority request entry to `sensing_hook._priority_requests[ticker]` when partial convergence fires (`CH_PARTIAL_CONVERGENCE` channel or equivalent)
3. Use `time.time() + 3600` as the `expires` value

The `_DOMAIN_TO_SOURCES` constant can be imported to validate that the needed domain has at least one mapped source before writing the request:
```python
from mae_core.market.sensing_hook import _DOMAIN_TO_SOURCES
```

**Nothing I consume from other builders:** My changes are self-contained within `sensing_hook.py`.

---

### Decisions Made

**Decision: `_build_domain_to_sources()` as module-level function vs inline dict**
Chose to compute it programmatically from existing constants rather than hardcoding a manual mapping. This ensures it stays in sync with `_ABSENCE_SOURCE_DOMAINS` and `SOURCE_ROTATION` as they evolve — no dual-maintenance. The trade-off is a small one-time computation at import time (one pass over ~33 items), which is negligible.

**Decision: Primary domain assignment (one domain per rotation source)**
Each rotation source is assigned to exactly one domain (its primary/dominant one). "finviz" maps to "technical" (via `finviz_unusual_volume` as primary Thompson key) even though it also produces insider and institutional signals. This is intentional — the boost is a heuristic nudge, not a precision tool. Perfect multi-domain coverage would require restructuring `_ROTATION_TO_THOMPSON` to support multi-key mappings, which is out of scope.

**Decision: Cleanup in `_launch_next_fetch()` not as a separate method**
The cleanup is a 4-line try/except block integrated directly into `_launch_next_fetch()`. Making it a separate `_schedule_fetch()` method (as the Build Brief named it) would add a method that's only called from one place. Kept it inline to match the existing pattern in the codebase (other periodic cleanup is similarly inlined).

**Decision: Boost multiplier 2.0, cap 1.0**
Exactly as specified. The 2x multiplier doubles a Beta variate that's already in [0,1], then caps at 1.0. This means a source with a raw draw of 0.6 becomes 1.0 (maximum), while a source with 0.3 becomes 0.6 — still competitive but not guaranteed selection. This is a soft nudge, not a hard override.

**Decision: `0 < len(...) <= 50` not `len(...) > 50`**
The Build Brief says "if over 50, don't boost." The condition `0 < len() <= 50` handles both edges: empty queue (0 entries, no boost needed) and over-cap (>50 entries, boost disabled). The zero check prevents the outer loop from running unnecessarily on an empty dict.

---

### Test Results

```
============================= test session starts =============================
collected 23 items

tests/test_priority_polling.py::TestDomainToSources::test_all_values_are_source_rotation_names PASSED
tests/test_priority_polling.py::TestDomainToSources::test_insider_domain_mapped PASSED
tests/test_priority_polling.py::TestDomainToSources::test_government_domain_mapped PASSED
tests/test_priority_polling.py::TestDomainToSources::test_macro_domain_mapped PASSED
tests/test_priority_polling.py::TestDomainToSources::test_no_duplicates_per_domain PASSED
tests/test_priority_polling.py::TestDomainToSources::test_all_absence_domains_covered PASSED
tests/test_priority_polling.py::TestDomainToSources::test_inverted_from_absence_source_domains PASSED
tests/test_priority_polling.py::TestThompsonBoost::test_boosted_source_scores_higher PASSED
tests/test_priority_polling.py::TestThompsonBoost::test_boost_capped_at_1_0 PASSED
tests/test_priority_polling.py::TestThompsonBoost::test_no_boost_when_empty_priority_queue PASSED
tests/test_priority_polling.py::TestThompsonBoost::test_multi_domain_priority_boosts_multiple_source_groups PASSED
tests/test_priority_polling.py::TestExpiredCleanup::test_expired_entries_removed_on_next_fetch PASSED
tests/test_priority_polling.py::TestExpiredCleanup::test_valid_entries_not_removed PASSED
tests/test_priority_polling.py::TestExpiredCleanup::test_mixed_expiry_partial_cleanup PASSED
tests/test_priority_polling.py::TestCapAt50::test_over_50_entries_disables_boost PASSED
tests/test_priority_polling.py::TestCapAt50::test_exactly_50_entries_applies_boost PASSED
tests/test_priority_polling.py::TestCapAt50::test_zero_entries_no_boost PASSED
tests/test_priority_polling.py::TestDomainSourceIdentification::test_known_domain_returns_nonempty_sources PASSED
tests/test_priority_polling.py::TestDomainSourceIdentification::test_unknown_domain_returns_empty PASSED
tests/test_priority_polling.py::TestDomainSourceIdentification::test_priority_request_format_accepted PASSED
tests/test_priority_polling.py::TestDomainSourceIdentification::test_entry_missing_domains_key_is_safe PASSED
tests/test_priority_polling.py::TestHookInitialization::test_priority_requests_initialized_as_empty_dict PASSED
tests/test_priority_polling.py::TestHookInitialization::test_priority_requests_survives_multiple_fetch_cycles PASSED

23 passed in 0.69s
```

**Full suite:** 961 passed, 1 failed (transient). The single failure is `tests/test_congress_gov_client.py::TestCongressGovClient::test_request_fails_without_key` — this test makes a live HTTP call to congress.gov and received a 429 Too Many Requests during the suite run (rate-limited by the external API). Re-running it in isolation immediately after passes (1 passed in 0.55s). This is a pre-existing environmental issue unrelated to my changes — confirmed by running it against the pre-change codebase, which also occasionally fails under concurrent suite load.

---

### Self-Review Concerns

**1. `_build_domain_to_sources()` missing some coverage for multi-signal sources.**
"finviz" fetches signals for `finviz_unusual_volume` (technical), `finviz_short_squeeze` (institutional), and `finviz_insider` (insider). The current logic assigns "finviz" only to "technical" (its primary Thompson key). If a priority request asks for "insider" or "institutional", the "finviz" source won't be boosted. The reviewer should assess whether this is acceptable or whether `_build_domain_to_sources` needs a multi-domain mapping. A fix would require changing `_ROTATION_TO_THOMPSON` to support multi-key mappings or adding a separate `_ROTATION_MULTI_DOMAINS` dict.

**2. Cleanup timing: cleanup happens at `_launch_next_fetch()` not `step()`.**
Expired entries survive until the next cadence tick (default: every 25 steps). In a slow step loop, entries could linger for minutes after their 1-hour expiry. This is acceptable given entries expire after 3600 seconds (well past any realistic cadence delay), but reviewers should note it.

**3. Priority requests dict is written by external code with no setter method.**
The interface is `hook._priority_requests[ticker] = {...}`. There's no `add_priority_request()` method with validation. A malformed entry (e.g., `expires` is a string instead of float) would be silently ignored at cleanup time (the `entry.get("expires", 0)` would fail the comparison). This is protected by try/except but worth flagging as a potential source of silent bugs.

**4. No persistence of priority requests across restarts.**
Priority requests live in memory only. If MIDGE restarts, all in-flight partial convergence contexts are lost. This is by design (they're short-lived, 1-hour expiry), but the reviewer should confirm this matches the intended behavior.
