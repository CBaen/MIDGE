# Build Brief: Inevitability Surfacing — Focused Attention + Sequential Chain Boost

## Date: 2026-03-09
## Project: MIDGE
## Source: Guiding Light directive + evolution blueprint "Focused Attention" gap

### Goal

When MIDGE notices something developing (partial convergence — 2 domains but not 3), she should lean in and actively seek the missing evidence. When a causal chain confirms a domino (CascadeTracker), she should inject that confirmation into the convergence engine so it naturally stacks with independent signals. These two features close the loop between MIDGE's causal reasoning and her attention/sensing systems.

### Build Tasks

**Round 1 (parallel — no file overlap):**

1. **Focused Attention (Priority Polling)** — When partial convergence fires, populate a priority queue on ctx. The sensing hook checks this queue during source selection and boosts sources that serve the missing domains. Entries expire after 1 hour.

2. **Sequential Chain Boost (Cascade → Synthetic Signals)** — Subscribe to CH_CASCADE_CONFIRMED in market_hooks.py. For each remaining domino, call convergence_alerter.record_signal() with domain="cascade" and metadata={"cascade_boosted": True}. Add "cascade" to domain_categories in convergence_alerter.py.

### Team Size: 2 builders + 2 reviewers

Two features, two distinct file domains (sensing vs. intelligence/hooks), clean separation. Minimal build — 2 builders is right.

### Builder Assignments

| Builder | Domain | Files Owned | Round Tasks |
|---------|--------|-------------|-------------|
| Attention Builder | Sensing pipeline | `mae_core/market/sensing_hook.py`, `tests/test_priority_queue.py` (new) | Task 1: priority queue + source boost |
| Chain Builder | Intelligence hooks | `mae_core/bootstrap/market_hooks.py`, `mae_core/market/intelligence/convergence_alerter.py`, `tests/test_chain_boost.py` (new) | Task 2: cascade subscriber + synthetic signals |

### Round Structure

**Single round** — both features are independent. No file overlap. Builder domains are clean.

- Attention Builder edits sensing_hook.py (source selection) + writes tests
- Chain Builder edits market_hooks.py (new subscriber) + convergence_alerter.py (add "cascade" domain) + writes tests

### Project Constraints (from CLAUDE.md)

1. **Never block the step loop** — all handlers wrapped in try/except
2. **Use existing EventBus patterns** — register_callback, publish
3. **No unbounded growth** — priority queue entries MUST expire, cascade-boosted signals MUST be tagged
4. **Law 2 (triadic)** — "cascade" domain added to domain_categories for convergence counting
5. **Zero regressions** — `python -m pytest tests/ -v` must pass
6. **No monoliths** — don't bloat files beyond their current scope
7. **Follow established patterns** — match existing naming, error handling, imports
8. **Tests required** — both features need dedicated test coverage

### Verification Plan

1. `python -m pytest tests/test_priority_queue.py tests/test_chain_boost.py -v` — new tests pass
2. `python -m pytest tests/ -v` — zero regressions (892+ tests)
3. `python main.py --agents 3 --steps 30` — smoke test

### Technical Context for Builders

**Feature 1 — Focused Attention:**
- `_on_partial_convergence` at market_hooks.py:390 already fires on "market.intel.partial_convergence"
- Partial convergence data carries: `domains_seen`, `missing_domains`, `causal_predictions`, `ticker`
- Source selection at sensing_hook.py:620 (`_launch_fetch`) picks sources via Thompson sampling from `eligible` list
- Need a `_DOMAIN_TO_SOURCES` reverse map: given a domain name (e.g., "insider"), which SOURCE_ROTATION entries serve it
- Priority queue goes on `ctx._priority_requests` (dict keyed by ticker, with domains_needed, priority, expires timestamp)
- In `_launch_fetch` or `_launch_thompson_guided`: if any priority_requests are active and not expired, boost scores for sources serving those domains (e.g., multiply Thompson score by 2.0)
- Expire entries past their timestamp on every check (simple cleanup)

**Feature 2 — Sequential Chain Boost:**
- CH_CASCADE_CONFIRMED is published by CascadeTracker.check_signal() (cascade_tracker.py:146)
- Confirmation payload: `{chain_id, trigger, confirmed_ticker, confirmed_direction, confirmed_count, total_links, remaining: [{ticker, direction, lag_days, strength}]}`
- For each `remaining` domino, call: `convergence_alerter.record_signal(signal_id=f"cascade_{chain_id}_{ticker}", strength=confirmed_ratio * domino_strength, domain="cascade", direction=domino_direction, metadata={"cascade_boosted": True, "chain_id": chain_id, "trigger": trigger})`
- `confirmed_ratio = confirmed_count / total_links` — more confirmed dominoes = stronger synthetic signal
- Add `"cascade": "causal"` to `convergence_alerter.domain_categories` (line 294)
- New subscriber in market_hooks.py after the existing cascade_tracker block (~line 505)
