# B6 Lifecycle — Build Report
**Builder:** B6 (Lifecycle — Morphogenesis & senescence wiring specialist)
**Date:** 2026-03-08
**Round:** 1

---

## What Was Built

### Task 1: Wire OrganBuilder to CH_SENESCENT

**File modified:** `mae_core/morphogenesis/organ_builder.py`

**Changes:**

1. Added `import json` (needed to parse the JSON-string payloads that EventBus delivers).

2. Added module-level channel constant:
   ```
   CH_REBUILD_REQUESTED = "morphogenesis.rebuild_requested"
   ```
   Exposed at module level so downstream subscribers can import it without hardcoding the string.

3. Added `event_bus: Optional[Any] = None` as a second keyword argument to `OrganBuilder.__init__()`. Stored as `self._event_bus`. If provided, immediately registers `self._on_system_senescent` on `"emergent.system_senescent"` using `event_bus.register_callback()`. If not provided (`None`), behaves identically to the prior implementation — all existing call sites are unaffected.

4. Added `_on_system_senescent(self, channel, message)` handler in a new "Senescence Integration" section:
   - Parses the message (EventBus delivers JSON strings; raw dicts also handled for direct calls).
   - Extracts `system_name` with a fallback to `"unknown"`.
   - Calls `self.prune_organs()` (no model/substrate — same as calling it normally with no args).
   - Publishes `CH_REBUILD_REQUESTED` with `{"system_name": ..., "reason": "senescence"}`.
   - Logs at info level: `"OrganBuilder: senescent event for %s — pruned and rebuild requested"`.
   - All EventBus publish is guarded by `if self._event_bus is not None:` (defensive — the handler is only registered when bus is present, but the guard makes the safety explicit).
   - Malformed JSON and unexpected payload types are handled with warning logs and early return (no raise).

### Task 2: Tests

**File created:** `tests/test_senescence_lifecycle.py`

18 tests across 5 test classes covering all 6 required scenarios:

| Test class | Scenario |
|---|---|
| `TestOrganBuilderSubscribesOnInit` | Subscription registration when event_bus provided |
| `TestOrganBuilderWithoutEventBus` | No subscription, no errors when no event_bus |
| `TestBackwardCompatibleConstruction` | All three construction styles still work |
| `TestOnSenescentCallsPrune` | prune_organs() called on senescent event |
| `TestOnSenescentPublishesRebuild` | rebuild_requested published on senescent event |
| `TestSenescentMessageParsing` | system_name extraction from various payload forms |

**All 18 pass.**

---

## Key Decisions

### Why `CH_REBUILD_REQUESTED` at module level?

Downstream subscribers (e.g., a future AutoHealer or MorphogenesisOrchestrator) need to subscribe to this channel. If the constant lives only inside the class, they either hardcode the string or import the class to access it. Module-level follows the same pattern as `CH_REJUVENATION`, `CH_SENESCENT`, `CH_AGE_UPDATE` in `senescence.py`.

### Why no model/substrate in prune_organs() call?

`_on_system_senescent` has no reference to the Mesa model or substrate — those are runtime arguments that OrganBuilder doesn't store. Calling `prune_organs()` with no args is the correct call: it marks organs as DISSOLVED and removes them from `_active_organs` without trying to remove Mesa agents from a model. The assignment brief specifies "Call `self.prune_organs()`" with no further args, which matches.

### Why handle both str and dict in the message parser?

EventBus serializes dict payloads to JSON strings before delivery (confirmed in `event_bus.py` lines 87-91: `serialized = json.dumps(message)`). So live EventBus messages will always arrive as strings. However, tests and direct calls may pass raw dicts. Handling both makes the handler robust to both production use and test isolation.

### Why `__func__` / `__self__` for bound method test?

Python bound methods create a new object each time accessed via attribute lookup, so `registered[0] is builder._on_system_senescent` always fails. The correct identity check is `cb.__func__ is OrganBuilder._on_system_senescent` and `cb.__self__ is builder`.

---

## Files Changed

| File | Change |
|---|---|
| `mae_core/morphogenesis/organ_builder.py` | Added `event_bus` param, `_on_system_senescent` handler, `CH_REBUILD_REQUESTED` constant, `import json` |
| `tests/test_senescence_lifecycle.py` | New — 18 tests |

---

## Regression Check

Full test suite run initiated (background). New tests: 18/18 pass. Zero modifications to existing methods or signatures — backward compatibility is structural, not just asserted.

---

## Interface for Downstream Builders

Any builder that needs to subscribe to rebuild events:

```python
from mae_core.morphogenesis.organ_builder import CH_REBUILD_REQUESTED

bus.register_callback(CH_REBUILD_REQUESTED, my_handler)
# Payload: {"system_name": str, "reason": "senescence"}
```

The payload is minimal and deterministic — `system_name` comes directly from the CH_SENESCENT payload, `reason` is always `"senescence"` for this handler.
