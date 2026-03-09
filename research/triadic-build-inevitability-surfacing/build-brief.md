# Build Brief: Inevitability Surfacing — Focused Attention + Chain Boost + Backward Discovery

## Date: 2026-03-09
## Project: MIDGE
## Source: Guiding Light directive + evolution blueprint "Focused Attention" gap + Guiding Light feedback on backward temporal curiosity

### Goal

Three connected features that make MIDGE a true inevitability investigator:

1. **Focused Attention** — When MIDGE notices something developing (partial convergence), she leans in and actively seeks the missing evidence.
2. **Sequential Chain Boost** — When a causal chain confirms a domino, inject that confirmation into the convergence engine so it naturally stacks with independent signals.
3. **Backward Cascade Discovery** — When MIDGE spots a signal that matches a known downstream effect, she traces BACKWARD through the causal graph to find what started the chain. If she's mid-pattern, she discovers it and tracks the rest.

Together: MIDGE detects she's witnessing a pattern unfolding, investigates where it started, predicts what comes next, and actively seeks evidence for both. Curiosity in action.

### Build Tasks

**Round 1 (parallel — no file overlap):**

1. **Focused Attention (Priority Polling)** — When partial convergence fires, populate a priority queue on ctx. The sensing hook checks this queue during source selection and boosts sources that serve the missing domains. Entries expire after 1 hour. When backward discovery identifies genesis events to investigate, it also populates this same queue.

2. **Sequential Chain Boost + Backward Discovery** — Two connected capabilities in the intelligence/hooks layer:
   - Subscribe to CH_CASCADE_CONFIRMED in market_hooks.py. For each remaining domino, call convergence_alerter.record_signal() with domain="cascade" and metadata={"cascade_boosted": True}. Add "cascade" to domain_categories.
   - Add `find_root_causes()` to WorldModel (reverse BFS using predecessors). When CascadeTracker.check_signal() confirms a mid-chain domino, trace backward to find genesis triggers. Check signal buffer for evidence of earlier events. If found: register a late-joining cascade with remaining forward dominoes. If not found: populate priority queue with genesis domains.

### Team Size: 2 builders + 2 reviewers

Two file domains. Clean separation between sensing pipeline and intelligence/hooks layer. The backward discovery lives naturally in world_model.py + market_hooks.py (Chain Builder's domain).

### Builder Assignments

| Builder | Domain | Files Owned | Round Tasks |
|---------|--------|-------------|-------------|
| Attention Builder | Sensing pipeline | `mae_core/market/sensing_hook.py`, `tests/test_priority_polling.py` (new) | Task 1: priority queue + source boost + expiry |
| Chain Builder | Intelligence + hooks | `mae_core/bootstrap/market_hooks.py`, `mae_core/market/intelligence/convergence_alerter.py`, `mae_core/market/intelligence/world_model.py`, `tests/test_chain_boost.py` (new) | Task 2: cascade subscriber + synthetic signals + backward discovery |

### Round Structure

**Single round** — both builders' files don't overlap. The priority queue interface is simple: Chain Builder writes `ctx._priority_requests[ticker] = {...}` in market_hooks.py. Attention Builder reads it in sensing_hook.py. Dict on ctx — no import coupling.

### Project Constraints (from CLAUDE.md)

1. **Never block the step loop** — all handlers wrapped in try/except
2. **Use existing EventBus patterns** — register_callback, publish
3. **No unbounded growth** — priority queue entries MUST expire (1 hour TTL), cap at 50 entries. Cascade-boosted signals MUST be tagged in metadata.
4. **Law 2 (triadic)** — "cascade" domain added to domain_categories for convergence counting
5. **Zero regressions** — `python -m pytest tests/ -v` must pass
6. **No monoliths** — don't bloat files beyond their current scope
7. **Follow established patterns** — match existing naming, error handling, imports
8. **Tests required** — both features need dedicated test coverage

### Verification Plan

1. `python -m pytest tests/test_priority_polling.py tests/test_chain_boost.py -v` — new tests pass
2. `python -m pytest tests/ -v` — zero regressions (892+ tests)
3. `python main.py --agents 3 --steps 30` — smoke test

---

## Technical Context for Builders

### Feature 1 — Focused Attention (Attention Builder)

**What exists:**
- `_on_partial_convergence` at market_hooks.py:390 already fires on `"market.intel.partial_convergence"`
- Partial convergence data carries: `domains_seen`, `missing_domains`, `causal_predictions`, `ticker`
- Source selection at sensing_hook.py:620 (`_launch_fetch`) picks sources via Thompson sampling from `eligible` list
- `SOURCE_ROTATION` (line 144) lists all 31 source names
- `_launch_thompson_guided()` (line 651) scores sources via Beta distributions, sorts descending, picks top N

**What to build:**
- A `_DOMAIN_TO_SOURCES` mapping: given a domain name (e.g., "insider"), which SOURCE_ROTATION entries serve it. Inverse of the existing `_SOURCE_DOMAIN_MAP` in excavator.py. Example: `"insider" -> ["sec_form4", "openinsider", "finviz"]`, `"technical" -> ["ta_indicators", "session_sweep"]`.
- In `_launch_thompson_guided()`: before scoring, check `ctx._priority_requests` (dict on ctx, populated by market_hooks.py). For any non-expired entries, identify which sources serve the needed domains. Boost those sources' Thompson scores (e.g., multiply by 2.0 or add 0.3 to the sampled score, capped at 1.0).
- Priority queue cleanup: on every `_launch_fetch()` call, remove expired entries from `ctx._priority_requests` (entries where `time.time() > entry["expires"]`).
- The priority queue is a simple dict: `ctx._priority_requests = {}` keyed by ticker. Value: `{"ticker": str, "domains_needed": List[str], "priority": str, "expires": float, "source": str}`. The "source" field indicates who added it ("partial_convergence" or "backward_discovery").
- **IMPORTANT:** Attention Builder does NOT write to `ctx._priority_requests`. That's done by market_hooks.py (Chain Builder's domain). Attention Builder only READS it and boosts sources accordingly.
- Cap: if `_priority_requests` has >50 entries when read, skip the oldest ones.

**Tests:**
- Priority boost changes source selection order
- Expired entries are cleaned up
- Sources correctly mapped to domains
- No boost when queue is empty (baseline unchanged)
- Cap prevents unbounded growth

### Feature 2 — Sequential Chain Boost + Backward Discovery (Chain Builder)

**Part A: Forward Chain Boost**

**What exists:**
- CH_CASCADE_CONFIRMED is published by CascadeTracker.check_signal() (cascade_tracker.py:146)
- Confirmation payload: `{chain_id, trigger, confirmed_ticker, confirmed_direction, confirmed_count, total_links, remaining: [{ticker, direction, lag_days, strength}]}`
- CH_CASCADE_CONFIRMED has NO listener yet (line ~505 of market_hooks.py — dead wire)
- `convergence_alerter.record_signal()` (line 636) accepts: signal_id, strength, domain, direction, confidence, velocity, timestamp, metadata, source

**What to build:**
- Subscribe to CH_CASCADE_CONFIRMED in market_hooks.py (after the existing cascade tracker block, ~line 505)
- Handler: `_on_cascade_confirmed(channel, data)` — for each remaining domino in the confirmation, call:
  ```
  convergence_alerter.record_signal(
      signal_id=f"cascade_{chain_id}_{ticker}",
      strength=confirmed_ratio * domino_strength,
      domain="cascade",
      direction=domino_direction,
      confidence=confirmed_ratio,
      metadata={"cascade_boosted": True, "chain_id": chain_id, "trigger": trigger, "remaining_lag_days": lag_days}
  )
  ```
  Where `confirmed_ratio = confirmed_count / total_links`
- Add `"cascade": "causal"` to `convergence_alerter.domain_categories` (line 294)
- Wrap in try/except — never block

**Part B: Backward Cascade Discovery**

**What exists:**
- WorldModel has `find_ripple_effects()` (forward BFS via `self._graph.successors()`)
- WorldModel has `map_signal_to_trigger()` for signal→trigger mapping
- DiGraph supports `self._graph.predecessors(node)` for reverse traversal
- Signal buffer on convergence_alerter: `self._signal_buffer` has recent signals

**What to build:**
- Add `find_root_causes(ticker_or_event, min_strength=0.3, max_depth=4) -> List[RootCause]` to WorldModel
  - New dataclass: `RootCause(trigger: str, direction: str, strength: float, path: List[str], total_lag_days: float, confidence: float)`
  - Reverse BFS using `self._graph.predecessors()` — walk backward from a known downstream node
  - Accumulate strength/lag/direction the same way `find_ripple_effects` does
  - Return sorted by strength (strongest root cause first)
- In the `_on_signal_causal_watch` handler (market_hooks.py:433), enhance the logic:
  - After checking forward ripple effects, also check: does this ticker appear as a DOWNSTREAM node in the world model? If so, call `find_root_causes(ticker)`.
  - For each root cause: check if a cascade for that trigger is already registered in cascade_tracker. If NOT, look for recent signals matching the root cause trigger (check signal_buffer on convergence_alerter or ctx).
  - If evidence found: register a late-joining cascade (the genesis already happened, track remaining dominoes going forward).
  - If no evidence: populate `ctx._priority_requests` with the genesis domain, so Focused Attention goes looking for the trigger event.
- This is the "curiosity" mechanism — MIDGE sees a downstream effect and asks "who pushed the first domino?"

**Tests (test_chain_boost.py):**
- Forward: cascade confirmation injects synthetic signals into convergence_alerter
- Forward: synthetic signals tagged with cascade_boosted=True in metadata
- Forward: strength proportional to confirmed_count/total_links
- Forward: "cascade" domain recognized in domain_categories
- Backward: find_root_causes returns correct predecessors
- Backward: mid-chain detection triggers backward investigation
- Backward: late-joining cascade registered when genesis evidence found
- Backward: priority queue populated when genesis evidence missing
- Integration: full chain (backward discovery → priority request → OR → late cascade registration)
