## Anvil Round 1 Build Report

### Item 5: EpisodicMemory stats fix
- **Files changed:** `mae_core/agents/lifecycle_learning.py`
- **What changed:** Lines 123-130 (MIDGE original) replaced with lines 123-134 (new).
  - Old: Direct call `self.episodic_memory.get_statistics()` — crashes if episodic_memory is None or if get_statistics() doesn't exist on this subclass.
  - New: Safe accessor `self.get_episodic_memory_statistics() if hasattr(self, "get_episodic_memory_statistics") else {}` — uses the mixin's own safe getter which handles None internally.
  - Also ported the background-thread pattern from mae-core: meta-memory update now fires in a daemon thread instead of blocking the step loop. Comment updated to match.
- **Market-specific code preserved:** Yes — no market-specific code in this file.
- **Root cause:** MIDGE was calling `.get_statistics()` directly on the episodic_memory object, bypassing the EpisodicMemoryMixin's safe wrapper. Mae-core fixed this with a `hasattr` guard and the mixin's own accessor.

### Item 6: Six missing EventBus channel registrations
- **Files changed:** `mae_core/backbone/connection_registrations.py`
- **Lines added:** 18 lines total across two insertion points.
- **Insertion point 1** (after `stem_cell.redifferentiated`, line 661):
  - `stem_cell.auto_redifferentiated` — witnessed by `reproductive_system` + `morph_coordinator`
  - `genome.snapshot_taken` — witnessed by `stem_cell_registry` + `enforcer`
  - `genome.sandbox_result` — witnessed by `stem_cell_registry` + `enforcer`
- **Insertion point 2** (after `pattern.consolidation`, line 1113 in original, now line 1128):
  - `signal.COLLABORATION_REQUEST` — witnessed by `gnn_communicator` + `pattern_bus`
  - `agent.shared` — witnessed by `pattern_bus` + `metacognition`
  - `topology.analysis` — witnessed by `somatic_map` + `connection_registry`
- **Market-specific code preserved:** Yes — MIDGE market channels (Group 17, Group 18) untouched.
- **Note:** These channels were being published by existing code (redifferentiation triggers, lifecycle_decision, topology_analyzer) but had no registry entry, causing advisory warnings from verify_all().

### Item 7: SomaticMap abstract names
- **Files changed:** `mae_core/bootstrap/wiring.py`
- **Lines modified:** 1 line (the tuple in the `for abstract_name in (...)` loop).
  - Added `"agent"` — used as witness in connection registrations for agent-level channels.
  - Added `"genome_reader"` — used as source in `genome.snapshot_taken` registration.
  - Added `"genome_sandbox"` — used as source in `genome.sandbox_result` registration.
  - Tuple reformatted across 3 lines for readability (matches mae-core style).
- **Market-specific code preserved:** Yes — no market-specific code in this section.
- **Note:** Without these entries, `verify_all()` marks the 3 genome connections added in Item 6 as unhealthy (unknown source system). Items 6 and 7 are paired.

### Item 8: Agent.shared channel normalization
- **Files changed:** `mae_core/agents/lifecycle_decision.py`
- **What changed:** Added `agent.shared` EventBus publish call inside `_act_communicate()`.
  - Location: inside the `if task.state == "completed":` branch, after the `emit_fn("COLLABORATION_REQUEST")` call.
  - Publishes `{"agent_id": str(self.unique_id), "step": self.step_count, "prediction_error": float(...)}` to the normalized channel `"agent.shared"`.
  - The comment explicitly documents WHY: "not per-agent channel `agent.{id}.shared` which breaks group subscriptions."
- **Market-specific code preserved:** Yes — all market-specific code in `_act_communicate` (TaskPool mechanics, `broadcast_solution`, market-role dispatch in `_act`) untouched.
- **Context on the "f-string form" in the brief:** The brief says to change `f"agent.{self.unique_id}.shared"` → `"agent.shared"`. That per-agent form did not exist in MIDGE's current code — MIDGE had simply never added this publish at all. The fix adds it in its correct normalized form from the start.
- **Conflict with Forge (Item 1):** Forge owns the VDN epsilon-greedy section in `_decide()`. My edit is in `_act_communicate()` — a different method, no overlap.

### Verification
- `python -m pytest tests/test_integration.py::TestBootstrap` — 7/7 passed (82s)
- `python -m pytest tests/test_integration.py::TestAgentLifecycle` — 4/4 passed (60s)
- All imports clean: `from mae_core.agents.lifecycle_learning import LearningLifecycleMixin` etc.

### Notes for Reviewers
1. **Item 5 background thread:** Mae-core also uses a background thread for the `store_ancestral_pattern` call (MIDGE calls it blocking). That difference was NOT ported — it was outside the stated scope of Item 5. Reviewers may want to flag this for a future pass.
2. **Item 7 count comment:** `wiring.py` line 505 logs `"Layer 17 - SomaticMap: %d systems registered for body awareness", 41`. Adding 3 abstract names increases the count. The hardcoded `41` is not updated here — this is an existing imprecision in the codebase and changing it is outside scope.
3. **Item 8 scope:** Only the task-completion branch of `_act_communicate` publishes `agent.shared`. The "share task" completion path at the end of the method does NOT emit it. This matches the mae-core semantics where the publish accompanies an explicit insight share, not routine task work.
