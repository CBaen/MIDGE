## Crucible Round 1 Build Report

### Item 9: Auto-healer starvation nodes fix

**Files changed:**
- `C:\Users\baenb\projects\MIDGE\mae_core\emergent\auto_healer.py`

**Fix 1 (node_id → nodes list):**

MIDGE's `_on_starvation` was calling `message.get("node_id", "unknown")` and filing a single FailureReport for a node named `"unknown"`. The substrate actually publishes `{"nodes": [list_of_starving_nodes], "step": N}` (confirmed in `mycelial_substrate.py` line 347). There is no `"node_id"` key in that payload — it was silently defaulting to `"unknown"` on every starvation event. The fix iterates the `nodes` list and files one FailureReport per node, with an early return if the list is empty.

Lines modified: `_on_starvation` method — replaced 4 lines (single node_id get + single report) with 8 lines (nodes list iteration with empty guard). Updated docstring to document the payload contract.

**Fix 2 (cooldown system — the "silent failures" root cause):**

The task brief described a `query_causation` keyword mismatch, but inspection of the MIDGE file reveals the `_phase_assess` call uses positional args identical to mae-core. The actual substantive bug causing repeated/silent failures is the missing cooldown system: MIDGE had no `_healing_cooldowns` dict, no self-heal guard in `step()`, no cooldown check before filing proactive reports, and no cooldown stamping in `_execute_healing`. Without cooldowns, every scan interval (every 10 steps) would re-file healing reports for the same still-recovering systems. The mae-core diff shows this is the companion fix.

Three coordinated changes applied:
1. Added `self._healing_cooldowns: dict[str, int] = {}` in `__init__` (after statistics block, with explanation comment)
2. Added self-heal exclusion guard (`if system_id == "auto_healer": continue`) and cooldown check in `step()` scan loop (5 lines added)
3. Added cooldown stamping in `_execute_healing` finally block: iterates `record.failure.affected_agents` and stamps `_healing_cooldowns[agent_id] = self._step_count` (3 lines added)

**Market-specific code preserved:** Yes — no market-specific code exists in auto_healer.py. The file is pure shared infrastructure. All existing MIDGE code (meta-healing triad, `_self_monitor`, endocrine modulation, statistics) is fully intact.

---

### Item 10: Phi forced measurement

**Files changed:**
- `C:\Users\baenb\projects\MIDGE\mae_core\backbone\growth_tracker.py`

**Lines added/modified:** `_read_phi` method — 2 lines added (guard + force call), 1 line changed (key lookup).

MIDGE's `_read_phi` had two bugs:

Bug A (the ported fix): No `_last_report` guard. If the `integration_meter` cadence had not yet been reached at run end, `_last_report` is `None` and `get_statistics()` returns a `last_measurement` key of `None`. The fix adds: `if getattr(meter, "_last_report", None) is None: meter._compute_and_publish()`. This forces a measurement before reading, so the first growth report has actual Phi data.

Bug B (found during inspection): MIDGE was reading `stats.get("current_phi", 0.0)` — a key that does NOT exist in `integration_meter.get_statistics()`. Confirmed by reading `integration_meter.py`: the stats dict has `last_measurement.organism_mean_phi`, not `current_phi`. MIDGE's Phi column in the growth tracker has been reporting 0.000 on every run since creation. Fixed to match mae-core: `lm = stats.get("last_measurement") or {}; return float(lm.get("organism_mean_phi", 0.0) or 0.0)`.

**Market-specific code preserved:** Yes. MIDGE's `_deep_store` call is synchronous (not backgrounded in a threading.Thread like mae-core). This is an intentional MIDGE divergence (simpler, no daemon thread) and was preserved. The header comment, logger name (`midge.growth`), output dir default (`data/midge`), collection name (`midge_meta`/`midge_narrative`), and file header text (`# MIDGE Growth Tracker`) are all preserved.

---

### Adversarial Notes

**What should the other agents watch for:**

1. **`inject_nutrient` is a phantom method.** `auto_healer._register_defaults()` registers `_inject_nutrients` as a recovery callback for STARVATION and RESOURCE_EXHAUSTION. It calls `self._substrate.inject_nutrient(agent_id, 1.0)`. Search `mae_core/substrate/` — `inject_nutrient` is never defined on `MycelialSubstrate`. Every STARVATION healing will silently fail in Phase 3 with `AttributeError` caught by the exception handler. The healer will record `success=False` for the `_inject_nutrients` action but continue. This is pre-existing debt, not introduced by this port.

2. **Race condition in `_on_starvation` failure_id.** Multiple starving nodes in the same message all get `failure_id = f"starve-{node_id}-{int(time.time())}"`. If two nodes arrive in the same second, IDs are unique (different `node_id`). But if the same node appears in two consecutive starvation events within the same second, the second filing will find its failure_id already in `_active_healings` and skip it. Acceptable behavior — the first healing is still running. But if it hits `_max_concurrent`, the second node in the same message gets dropped silently. This was also true before the fix (just now it can actually happen with real node IDs).

3. **Cooldown uses step count, not wall time.** `_healing_cooldowns` stores step numbers. The 50-step cooldown window means "50 model ticks." If step intervals vary (parallel senses, background tasks), the cooldown could be shorter or longer in real time than intended. Not a bug, but reviewers should know this is tick-relative, not time-relative.

4. **Growth tracker deep_store synchronous vs async.** Mae-core uses `threading.Thread(target=_store_async, daemon=True).start()` for Qdrant writes. MIDGE calls `self._deep_store.store_point(...)` synchronously. If Qdrant is slow or unavailable at run-end, MIDGE's `record_run` will block. This is pre-existing MIDGE design — preserved intentionally — but Forge or Anvil should note it as a known performance risk.

5. **The `query_causation` "keyword mismatch" described in the task brief is not present in the file.** Both MIDGE and mae-core call `self._causal.query_causation(failure.failure_type.value, "system_degradation")` with positional args. The `CausalQueryResult` attributes (`is_causal`, `cause`, `causal_path`) all exist. Either this was fixed before MIDGE diverged from mae-core, or the brief was describing a different bug. The cooldown system was the actual substantive missing piece. Reviewers should confirm this interpretation is correct.

6. **`_compute_and_publish` is a private method.** The growth tracker now calls `meter._compute_and_publish()` directly. This works because GrowthTracker is infrastructure that runs at the end of a run (not during), and integration_meter is internal. But it's a private method call across module boundaries. If mae-core renames or removes this method, MIDGE's growth tracker silently falls back (the `hasattr` guard catches it). Low risk, but worth noting for future maintainers.

7. **Phi was always 0.000 in MIDGE.** The `current_phi` key bug means every row in `data/midge/growth-tracker.md` shows `0.000` in the Phi column. After this fix, future runs will show real Phi values — which may be jarring if anyone has been using the tracker to baseline performance. Not a code problem, but a data interpretation note.
