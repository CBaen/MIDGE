# Build Report: Channels & Translators (Builder 1 — Round 1)

**Date:** 2026-03-08
**Builder:** Builder 1 — Channels & Translators
**Status:** COMPLETE — 15/15 tests passing

---

## What Was Built

### Task 1: Channel constants (`mae_core/market/channels.py`)
Added two new constants after line 55 (after `CH_PATTERN_COMPLETED`):
- `CH_PARTIAL_CONVERGENCE = "market.intel.partial_convergence"` — emitted when domains are accumulating but haven't crossed the full convergence threshold. Feeds `MarketPartialTranslator`.
- `CH_OCTOPUS_INVESTIGATION = "market.intel.octopus_investigation"` — reserved for OctopusColony to publish investigation results. Not yet consumed; present so downstream builders can subscribe without a merge conflict.

### Task 2: Package marker (`mae_core/market/translators/__init__.py`)
Minimal package marker with a one-line docstring. The directory did not exist — created fresh.

### Task 3: Translator implementations (`mae_core/market/translators/market_signal_translator.py`)
Two translator classes, each satisfying the `PatternTranslator` Protocol (source_name, channels, translate).

**`MarketConvergenceTranslator`**
- Subscribes to `CH_CONVERGENCE`
- Bullish → `OPPORTUNITY / CORRELATED`, bearish → `THREAT / CORRELATED`, anything else → `None`
- Salience: `min(1.0, confidence * 0.6 + (domain_count / 12) * 0.4)`
- TTL: 10 steps
- Accepts both dict and JSON-string messages via `_parse()` helper

**`MarketPartialTranslator`**
- Subscribes to `CH_PARTIAL_CONVERGENCE`
- Any direction → `NOVELTY / REACTIVE`
- Salience: `min(0.6, len(domains_seen) * 0.2)` — hard cap at 0.6 keeps partials below any full convergence
- TTL: 15 steps (longer than full convergences — partials are investigative hints that age slowly)
- Same `_parse()` helper for robustness

**`_parse()` internal helper**
Accepts `dict | str | Any`. Returns `dict | None`. Handles JSON string messages in addition to plain dicts — EventBus messages in this codebase arrive as JSON strings from some publishers.

### Task 4: Tests (`tests/test_market_signal_translator.py`)
15 tests across two test classes. Covers:
- Domain/form mapping for both directions
- Neutral/missing direction → None
- Salience scaling with domain count (both translators)
- Salience cap enforcement (1.0 for convergence, 0.6 for partial)
- Empty domains_seen → 0.0 salience
- JSON string message acceptance
- Non-dict/non-string → None
- source_name and channels properties

---

## Decisions Made

**Why `_parse()` accepts JSON strings:** The `emergent.py` reference only handled dicts, but examining EventBus usage in `sensing_hook.py` and `market_hooks.py` confirms messages arrive as both dicts (internal) and JSON strings (cross-module). A shared helper at module bottom keeps both classes clean.

**Why TTL 15 > 10 for partials:** Counterintuitive but intentional. Full convergences are action-ready and should expire quickly if not acted on. Partials are investigative seeds — they should persist long enough for other domains to arrive and potentially upgrade them to a full convergence.

**Why domain_count from `domain_count` field with fallback to `len(domains)`:** The `ConvergenceAlerter` publishes both; the fallback handles cases where only `domains` list is present (e.g., replay harness payloads).

---

## Zero Regressions
Only files touched:
- `mae_core/market/channels.py` — additive only (2 new constants appended)
- `mae_core/market/translators/__init__.py` — new file
- `mae_core/market/translators/market_signal_translator.py` — new file
- `tests/test_market_signal_translator.py` — new file

No existing files modified except channels.py (append only). No existing tests affected.
