## Review of Anvil's Round 1 Work

---

### Item 5: EpisodicMemory stats fix (lifecycle_learning.py)

**Spec Compliance: PASS with one divergence worth flagging**

The primary fix is correct. MIDGE line 127 now reads:
```python
_stats = self.get_episodic_memory_statistics() if hasattr(self, "get_episodic_memory_statistics") else {}
```
This matches mae-core line 131 exactly. The mixin's safe accessor is used instead of direct `.get_statistics()`. Correct.

The `update_meta_memory` background-thread block (lines 123-134) matches mae-core.

**Bug found: Variable shadowing in generative replay block**

MIDGE line 97 uses `metacog_dream` as the variable name for the dream-cadence metacognition check. Mae-core line 97 reuses `metacog` for this same assignment. This is not a functional bug — both load `getattr(self, "_metacognition", None)` and use it immediately in the next line. However there is a subtle semantic consequence: in mae-core, the `metacog` variable from line 41 is *rebound* at line 97 to the same value. In MIDGE, `metacog_dream` is a separate name. This means in MIDGE, if the dream block were ever refactored to reference `metacog` (instead of `metacog_dream`), it would use the value assigned at line 41, which could be from a different scope. Minor, but the Anvil build report claims to match mae-core style and this is actually a divergence. It is harmless in the current code because `metacog_dream` is only used on line 98.

**Unported divergence (flagged by Anvil, confirmed):**

`store_ancestral_pattern` is called synchronously in MIDGE (lines 114-121) but in a background thread in mae-core (lines 114-125). This is outside the stated scope of Item 5, but it means every high-reward step with `_memory_bridge` available blocks the step loop for an Ollama embed + Qdrant write — potentially 100-500ms. Flagged for future work, not a blocker.

**Integration issues: None.** Bootstrap injects `_memory_bridge` via `organs.py`. `get_episodic_memory_statistics` is provided by EpisodicMemoryMixin which is composed before LearningLifecycleMixin in the MRO. No ordering dependency issues.

---

### Item 6: Six missing EventBus channel registrations (connection_registrations.py)

**Spec Compliance: PASS — matches mae-core exactly**

Verified by reading MIDGE lines 662-675 and 1129-1145 against mae-core lines 662-675 and 1129-1145. Both files are byte-for-byte identical in these sections. All six registrations are present, in the correct locations, with correct witnesses and descriptions.

**Integration issues: None.** The `signal_bus` source used in `signal.COLLABORATION_REQUEST` is already registered as a system in wiring.py. The `gnn_communicator` source used in `agent.shared` is already registered. All witnesses (`gnn_communicator`, `pattern_bus`, `metacognition`, `somatic_map`, `connection_registry`, `stem_cell_registry`, `enforcer`, `reproductive_system`, `morph_coordinator`) are registered systems. No orphaned references.

**Note on Item 6 + 7 pairing:** Anvil correctly identifies that Items 6 and 7 are coupled. `genome_reader` and `genome_sandbox` appear as sources in the Item 6 registrations, and they must be registered as abstract names in SomaticMap for `verify_all()` to report them as healthy. Item 7 does this. Verified below.

---

### Item 7: SomaticMap abstract names (wiring.py)

**Spec Compliance: PASS — matches mae-core exactly**

MIDGE `wiring.py` line 489-492:
```python
for abstract_name in (
    "agent", "decision_router", "defense", "frl",
    "genome_reader", "genome_sandbox",
    "healing", "improvement", "memory", "morphogenesis", "triad_audit",
):
```

Mae-core `wiring.py` line 538-542 (verified by reading mae-core's bootstrap/wiring.py):
```python
for abstract_name in (
    "agent", "decision_router", "defense", "frl",
    "genome_reader", "genome_sandbox",
    "healing", "improvement", "memory", "morphogenesis", "triad_audit",
):
```

Identical. All three new abstract names (`"agent"`, `"genome_reader"`, `"genome_sandbox"`) are present.

**Integration issues: None.** This runs before connection registration at Layer 18. SomaticMap.register_system is idempotent by design. Adding abstract registrations before connections are verified is the correct ordering.

**The hardcoded `41` log count (line 505) is now stale.** Adding 3 abstract names (agent, genome_reader, genome_sandbox) to what was 11 brings the abstract total to 13, plus the concrete systems registered before this loop. Anvil correctly flagged this as out of scope, but the stale count will emit a misleading log line on every run. This is a documentation debt, not a functional bug.

---

### Item 8: Agent.shared channel normalization (lifecycle_decision.py)

**Spec Compliance: PASS with one important divergence to flag**

The `agent.shared` publish is correctly added in `_act_communicate()` at lines 558-566. The channel name is normalized (`"agent.shared"` not `f"agent.{id}.shared"`). The payload matches mae-core. The comment explaining why is preserved.

**Structural divergence confirmed — not a bug, but a risk:**

MIDGE's `_act_communicate()` publishes `agent.shared` only inside the `if task.state == "completed"` branch (line 544). Mae-core's `_act_communicate()` publishes `agent.shared` unconditionally every time the method is called (mae-core lines 524-530).

This is because MIDGE's `_act_communicate` is fundamentally different from mae-core's: MIDGE's version is TaskPool-coupled (it works on tasks, broadcasts solutions, has a `pool` parameter). Mae-core's version is task-pool-free (pure intrinsic reward). The `agent.shared` publish belongs at different logical points in each version. Anvil's placement — inside the completed-task broadcast branch — is the correct analog for MIDGE's architecture. The channel still fires; it just fires less frequently than mae-core (only on task completion, not every communicate action).

**Integration issues: None.** `_event_bus` is injected in bootstrap organs.py. The `bus.publish("agent.shared", ...)` call is guarded with `if bus is not None`. The channel is now registered in connection_registrations.py (Item 6). The subscriber pattern (metacognition + pattern_bus) is intact.

**One potential issue: early return bypasses the channel publish.**

At MIDGE line 567: `return reward`. This `return` is INSIDE the completed-task branch, AFTER the `agent.shared` publish at lines 560-566. That is correct — the publish happens before the return. No issue here. The Anvil build report described this correctly.

---

## Review of Crucible's Round 1 Work

---

### Item 9: Auto-healer starvation nodes fix (auto_healer.py)

**Spec Compliance: PASS on starvation fix; DIVERGENCE on `_inject_nutrients`**

**Fix 1 (node_id → nodes list): CORRECT**

MIDGE `_on_starvation` at lines 505-528 now iterates `message.get("nodes", [])` and files one `FailureReport` per node. This matches mae-core exactly. The empty-guard early return is present. Correct.

**Fix 2 (cooldown system): CORRECT**

`self._healing_cooldowns: dict[str, int] = {}` at line 159 is present in `__init__`. The self-heal exclusion guard (`if system_id == "auto_healer": continue`) and cooldown check are in the `step()` scan loop at lines 236-241. Cooldown stamping in the `_execute_healing` finally block at lines 563-565. All three coordinated changes are present. This matches mae-core exactly.

**`query_causation` mismatch: CONFIRMED ABSENT**

Both MIDGE and mae-core call `self._causal.query_causation(failure.failure_type.value, "system_degradation")` with positional arguments on lines 612-614. The `CausalQueryResult` attributes used (`is_causal`, `cause`, `causal_path`) exist in both codebases. Crucible's assessment is correct: the brief described a bug that does not exist in the current files. This was either a false alarm in the brief or was fixed before the divergence point.

**REAL BUG: `_inject_nutrients` implementation divergence (not flagged in build report)**

This is a genuine discrepancy between MIDGE and mae-core that Crucible did NOT port and did NOT document:

MIDGE `_inject_nutrients` (lines 747-753):
```python
def _inject_nutrients(record: HealingRecord) -> str:
    if not self._substrate:
        return "no_substrate"
    for agent_id in record.failure.affected_agents:
        self._substrate.inject_nutrient(agent_id, 1.0)
    return f"nutrients_injected_for_{len(record.failure.affected_agents)}_agents"
```

Mae-core `_inject_nutrients` (lines 747-758):
```python
def _inject_nutrients(record: HealingRecord) -> str:
    if not self._substrate:
        return "no_substrate"
    flow = getattr(self._substrate, "nutrient_flow", None)
    if flow is None or not hasattr(flow, "inject_resources"):
        return "no_nutrient_flow"
    injected = 0
    for node_id in record.failure.affected_agents:
        if flow.inject_resources(str(node_id), 0.5):
            injected += 1
    return f"nutrients_injected_for_{injected}_nodes"
```

MIDGE calls `self._substrate.inject_nutrient(agent_id, 1.0)` directly. Mae-core routes through `self._substrate.nutrient_flow.inject_resources(str(node_id), 0.5)`. Crucible's build report (adversarial note 1) correctly identified that `inject_nutrient` is a phantom method — it does not exist on `MycelialSubstrate`. Every STARVATION healing will silently fail with `AttributeError` caught in `_phase_restore`'s exception handler and recorded as `success=False`.

This is pre-existing debt, but it means the fix Crucible applied to `_on_starvation` (correctly routing multiple starvation nodes) will now correctly identify multiple starving nodes and correctly file multiple FailureReports — each of which will then run `_inject_nutrients` and silently fail. The starvation fix works at the detection layer but the recovery layer remains broken.

Mae-core's version routes through `nutrient_flow.inject_resources()`. Whether `MycelialSubstrate` in MIDGE has a `nutrient_flow` attribute needs verification, but this is the correct path to check. This should be ported to MIDGE.

**Cooldown tick-relativity (adversarial note 3): Accepted.**

The 50-step cooldown window is tick-relative. With parallel senses running in ThreadPoolExecutor(3), step loop cadence is not wall-clock constant. This is a known property of the system, not a bug.

---

### Item 10: Phi forced measurement (growth_tracker.py)

**Spec Compliance: PASS — matches mae-core `_read_phi` exactly**

MIDGE `growth_tracker.py` lines 165-178:
```python
def _read_phi(self, systems: dict) -> float:
    meter = systems.get("integration_meter")
    if meter:
        try:
            if getattr(meter, "_last_report", None) is None:
                if hasattr(meter, "_compute_and_publish"):
                    meter._compute_and_publish()
            stats = meter.get_statistics()
            lm = stats.get("last_measurement") or {}
            return float(lm.get("organism_mean_phi", 0.0) or 0.0)
        except Exception:
            pass
    return 0.0
```

Mae-core `growth_tracker.py` lines 169-182 (same logic, verified by reading mae-core's backbone/growth_tracker.py). The `hasattr` guard on `_compute_and_publish` means the fix degrades gracefully if that private method is renamed. The correct key (`organism_mean_phi` inside `last_measurement`) is now used. Bug B (the `current_phi` key that doesn't exist) is fixed.

**MIDGE-specific preservation: PASS**

Logger name `midge.growth`, output dir default `data/midge`, collection names `midge_meta`/`midge_narrative`, file header `# MIDGE Growth Tracker`, and synchronous `_deep_store.store_point()` (vs mae-core's background thread) are all preserved. Correct.

**Integration issues: None.** `growth_tracker.py` is called from `main.py` at run-end only. The `integration_meter` key in the `systems` dict is populated in bootstrap layer 13 (in `wiring.py`). The growth tracker receives the systems dict as a parameter — no coupling risk.

---

## Summary Table

| Item | Agent | Finding | Severity |
|------|-------|---------|----------|
| 5 | Anvil | Variable renamed `metacog` → `metacog_dream` in dream block (harmless divergence from mae-core style) | Low |
| 5 | Anvil | `store_ancestral_pattern` still synchronous (I/O blocks step loop on high-reward steps) | Low (pre-existing, out of scope) |
| 6 | Anvil | All 6 registrations correct and complete | None |
| 7 | Anvil | Hardcoded `41` log count is stale by 3 | Low (documentation debt) |
| 8 | Anvil | `agent.shared` publishes less frequently than mae-core (task-completion only vs every communicate) | Acceptable (architecture difference) |
| 9 | Crucible | `_inject_nutrients` not ported from mae-core — calls phantom `inject_nutrient` method instead of `nutrient_flow.inject_resources` | Medium (starvation recovery silently fails) |
| 9 | Crucible | `query_causation` bug confirmed absent — correct assessment | None |
| 10 | Crucible | `_read_phi` fix complete and correct | None |

**Required fix before Round 2:**
- Item 9: Port `_inject_nutrients` to use `nutrient_flow.inject_resources()` pattern from mae-core. Starvation detection is now working (after the nodes-list fix) but recovery is silently broken.
