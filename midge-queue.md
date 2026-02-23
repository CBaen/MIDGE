# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.

> MIDGE diverged from mae-core at commit `2d1ed7d` (2026-02-22, "Fix 3 runtime bugs: closure connections, reproductive spam, Groq quarantine"). All mae-core completed tasks prior to this point are inherited.

---

## Completed

All Phase 1 and Phase 2 work is complete. See HANDOFF.md for full history.

- Fork & port (2026-02-22)
- Bootstrap integration: Layer 33 with 16 systems, 23 connections, 20 holons (2026-02-22)
- Phase 2: CorrelationTracker persistence, discovery_log reader, KNOWN_POLITICIANS, TickerResolver, MarketDataProvider, ContractPredictor evaluation (2026-02-22)
- Regime-aware Thompson Sampling (2026-02-22)
- Client migration: all 6 API clients route through MarketDataProvider/ApiGateway (2026-02-22)
- SEC user agent email updated (2026-02-22)

---

## Active

No active tasks. MIDGE's integration work is complete.

**Potential next directions (not yet decided):**
- Live data testing with real API keys
- Dashboard/UI for convergence alerts
- Upstream mae-core genome sync
- Additional edge detectors or signal sources

---

**When completing a task:**
1. Mark as `[x]` with completion date
2. Git commit preserves history -- no separate history file needed
