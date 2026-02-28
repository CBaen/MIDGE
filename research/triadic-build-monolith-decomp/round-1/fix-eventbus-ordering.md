# Crucible Finding: EventBus Injection Before restore_state()

**Reviewer:** Crucible (adversarial)
**Finding ID:** eventbus-ordering
**Date:** 2026-02-28
**Files examined:**
- `C:\Users\baenb\projects\MIDGE\mae_core\bootstrap\agents.py`
- `C:\Users\baenb\projects\mae-core\mae_core\bootstrap\agents.py`
- `C:\Users\baenb\projects\MIDGE\mae_core\agents\base_agent.py`
- `C:\Users\baenb\projects\MIDGE\mae_core\agents\mycelial_agent.py`
- `C:\Users\baenb\projects\MIDGE\mae_core\backbone\holon_mixin.py`
- All 10 files in `C:\Users\baenb\projects\MIDGE\mae_core\agents\mixins\`

---

## Finding

Crucible flagged that in `agents.py`, post-creation injections happen before `restore_state()`:

```python
# Lines 112-122 — injections
agent._holon_parent_id = "colony"
agent.curiosity_drive = ctx.curiosity
agent._event_bus = ctx.bus
agent._validated_imagination = ctx.validated_imagination

# Lines 135-138 — restore (happens AFTER injections)
prior_state = ctx.model.load_agent_state(agent.unique_id)
if prior_state:
    agent.restore_state(prior_state)
```

The concern: if any mixin serializes `_event_bus` during shutdown, `restore_state()` would overwrite the live bus with a stale or None value from the persisted state.

---

## Investigation Results

### 1. Are MIDGE and mae-core identical here?

**Yes. Both files are byte-for-byte identical** in the agent creation loop (lines 104-148). The ordering issue exists in both codebases at the same commit points. This is inherited shared code, not a MIDGE-specific regression.

### 2. Does any mixin serialize `_event_bus`?

**No. Zero mixins serialize `_event_bus`.**

Search results across all relevant files:

| File | `_event_bus` present? | Serialized? |
|------|----------------------|-------------|
| `mixins/gamification.py` | No | No |
| `mixins/collective_consensus.py` | No | No |
| `mixins/convergence.py` | No | No |
| `mixins/transfer_learning.py` | No | No |
| `mixins/gnn_communication.py` | No | No |
| `mixins/signal_processing.py` | No | No |
| `mixins/stigmergy.py` | No | No |
| `mixins/advanced_features.py` | No | No |
| `mixins/episodic_memory.py` | No | No |
| `backbone/holon_mixin.py` (`_serialize_holon`) | No | No |

The `_serialize_holon()` method (the 10th mixin, wired via `state["holon"]`) serializes only `holon_id`, `parent_id`, and `holon_memory`. No bus reference.

### 3. What does `restore_state()` actually restore?

**`base_agent.restore_state()`** restores only:
- `step_count`
- `cumulative_reward`
- `last_reward`
- `risk_score`
- `performance_history`
- `reward_history`

**`mycelial_agent.restore_state()`** adds mixin restores for:
convergence, gamification, signal_processing, stigmergy, gnn_communication, transfer_learning, episodic_memory, collective_consensus, advanced_features, holon, signal_priority

None of these restore paths touch `_event_bus`, `curiosity_drive`, `_validated_imagination`, or `_holon_parent_id`.

---

## Verdict: No Fix Needed

The ordering concern is structurally valid — injections before restore is a pattern that *could* be dangerous. But the precondition for danger does not exist:

**`_event_bus` is never serialized by any mixin, and `restore_state()` never writes to it.**

The overwrite risk Crucible identified requires both:
1. A mixin storing `_event_bus` in its `_serialize_*()` output, AND
2. The corresponding `_restore_*()` writing it back onto the agent

Neither condition holds. The restoring code only touches numerical scalars, deques, and domain-specific data structures (holon memory, GNN message history, stigmergy trails, etc.).

### What about the ordering in principle?

The injections that precede `restore_state()` are:
- `_holon_parent_id` — set to `"colony"`. `_restore_holon()` *does* restore `parent_id` from saved state, meaning a saved holon parent_id could overwrite the fresh assignment. However, the saved parent_id would also be `"colony"` (from the prior run), so this is idempotent. Not a real bug.
- `curiosity_drive` — injected object reference. Not serialized anywhere.
- `_event_bus` — injected object reference. **Not serialized anywhere.** The subject of this investigation.
- `_validated_imagination` — injected object reference. Not serialized anywhere.

The only field that `restore_state()` touches which was also set before it is `_holon_parent_id`, and that case is idempotent. All injected object references (`_event_bus`, `curiosity_drive`, `_validated_imagination`) survive `restore_state()` untouched because no serializer ever captured them.

---

## Recommendation

No code change required.

**If future work adds a mixin that serializes `_event_bus`** (e.g., a mixin that tracks bus subscription state), the injections must be moved to after `restore_state()` at that time. The safe ordering would be:

```python
# 1. Create agent
# 2. restore_state(prior_state)        ← restores learned numeric/structural state
# 3. Inject live object references      ← always overwrites with fresh objects
```

This is the correct principle. The current code happens to be safe only because no mixin serializes object references. Document this as a latent ordering risk for future contributors.

---

## Action

**NONE.** Finding closed as false positive given current serialization coverage.

Add a comment in `agents.py` near the injection block to make the invariant explicit for future contributors. This is optional but would protect siblings who add new serialized fields.
