# Fingerprint Offload Build Report

**Date:** 2026-03-09
**Objective:** Remove 223K fingerprint objects from RAM at startup. Keep templates.

---

## Problem

`PatternLibrary._load()` read every line of `data/market/pattern_library.jsonl` at startup and deserialised each into a `MoveFingerprint` object stored in `self._fingerprints: dict[str, MoveFingerprint]`. With 223K entries, this consumed roughly 100MB of RAM every time MIDGE boots — for data that the live system never touches.

`PatternWatcher.check()` calls `library.query_similar()`, which iterates only `self._templates`. Fingerprints are the raw evidence; templates are the generalised patterns. The live pattern-detection path touches zero fingerprints.

Fingerprints are needed in two infrequent operations:
- `rebuild_templates()` — runs when templates file is empty/corrupted
- `update_outcome(fingerprint_id=...)` — outcome grading cadence

---

## Design

**Keep in RAM always:**
- `_templates: dict[str, PatternTemplate]` — 39 entries, PatternWatcher depends on them live
- `_template_key_index: dict[str, str]` — O(1) lookup for dedup during store
- `_fingerprint_ids: set[str]` — lightweight ID set for dedup (just strings, ~15 bytes each × 223K = ~3.3MB vs ~100MB full objects)
- `_fingerprint_count: int` — for the `size` property

**Keep on disk, load lazily:**
- All `MoveFingerprint` objects — never in RAM except during infrequent operations

**New methods:**
- `_load_fingerprints()` — loads all fingerprints from disk into a temporary dict. Caller discards when done. Used by `rebuild_templates()` and `update_outcome(fingerprint_id=...)`.
- `_persist_fingerprints_dict(fingerprints)` — renamed from `_persist_fingerprints()`, now takes an explicit dict parameter (caller owns it).

**`get(fingerprint_id)`** — changed from dict lookup to targeted JSONL scan. Checks `_fingerprint_ids` first (fast reject), then scans file line-by-line until match found. Called infrequently (outcome grading, tests).

---

## What Changed

`mae_core/market/archaeology/pattern_library.py`:

| Old | New |
|-----|-----|
| `self._fingerprints: dict[str, MoveFingerprint]` loaded at startup | Removed — replaced with `_fingerprint_ids: set[str]` + `_fingerprint_count: int` |
| `_load()` parsed all 223K fingerprint objects | `_load()` only reads `fingerprint_id` field per line (fast scan) |
| `store()` checked `self._fingerprints` | Checks `self._fingerprint_ids` |
| `store_batch()` populated `self._fingerprints` | Updates IDs set + count only |
| `get(id)` did dict lookup | Targeted JSONL scan (via `_fingerprint_ids` fast-reject) |
| `size` returned `len(self._fingerprints)` | Returns `self._fingerprint_count` |
| `rebuild_templates()` iterated `self._fingerprints.values()` | Calls `_load_fingerprints()`, iterates, discards |
| `update_outcome(fingerprint_id=...)` mutated in-RAM object | Calls `_load_fingerprints()`, mutates, persists via `_persist_fingerprints_dict()` |
| `_persist_fingerprints()` wrote from `self._fingerprints` | Renamed to `_persist_fingerprints_dict(fingerprints: dict)` |
| `get_statistics()` used `len(self._fingerprints)` | Uses `self._fingerprint_count` |
| `clopper_pearson_ci(fingerprint_id=...)` used `self._fingerprints.get()` | Uses `self.get()` (targeted scan) |

---

## Memory Savings

| Before | After |
|--------|-------|
| ~100MB of MoveFingerprint objects in RAM at startup | ~3.3MB of fingerprint IDs (set of strings) |
| Full parse of 223K JSONL lines at startup (deserialise all) | Scan of 223K JSONL lines at startup (read `fingerprint_id` field only, ~3x faster) |
| Fingerprints in RAM for lifetime of process | Fingerprints loaded on demand, discarded after use |

---

## Tests

22 new tests in `tests/test_fingerprint_offload.py`:

- `TestLazyFingerprints` — verifies `_fingerprints` dict absent/empty, count accurate, IDs set populated, templates always in RAM
- `TestStoreWithoutRam` — store/store_batch work without in-RAM dict; disk persistence; dedup via ID set; dedup across reload
- `TestGetFingerprint` — targeted scan returns correct object; missing returns None; no RAM pollution after get()
- `TestRebuildTemplates` — loads fingerprints on demand, builds templates, releases; no `_fingerprints` after rebuild
- `TestUpdateOutcome` — template outcome update; fingerprint outcome update via load-mutate-persist
- `TestQuerySimilar` — query works from templates alone; empty when no templates file
- `TestMemorySavings` — 1000 fingerprint pre-existing library: size accurate, IDs correct, no full objects; new stores still work after large preload

**Results:** 22/22 passing. Full suite: 970+ passed, 1 pre-existing flaky failure in `test_congress_gov_client.py` (passes in isolation, fails under test-ordering pollution — unrelated to this change).

---

## Constraints Satisfied

- Zero regressions in archaeology test suite (90 tests across pattern_library, excavator, pattern_watcher)
- Templates always in RAM — PatternWatcher unaffected
- Fingerprint count remains accurate — `size` property correct
- `store()` / `store_batch()` work correctly — dedup via ID set
- `add_fingerprint()` (store) appends to JSONL without loading existing
- ExcavationDaemon still works — it calls `store_batch()` which remains functional
- `rebuild_templates()` still works — loads fingerprints on demand

---

## Files Modified

- `mae_core/market/archaeology/pattern_library.py` — lazy loading implementation
- `tests/test_fingerprint_offload.py` — new: 22 tests verifying lazy behaviour
