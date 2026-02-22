# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.

> MIDGE diverged from mae-core at commit `2d1ed7d` (2026-02-22, "Fix 3 runtime bugs: closure connections, reproductive spam, Groq quarantine"). All mae-core completed tasks prior to this point are inherited.

---

## Completed (2026-02-22) — Fork & Port

- [x] **Fork mae-core into MIDGE** (2026-02-22)
      What: Copied full mae-core (85 systems, 2425 tests, 313 connections) into MIDGE directory. Verified all tests pass.
      Result: MIDGE is a standalone Mae organism with its own git history.

- [x] **Port MIDGE market intelligence** (2026-02-22)
      What: 16 market modules ported into `mae_core/market/` (7 APIs, 4 edge detectors, 5 intelligence). Fixed all imports (`trading.apis.xxx` -> `mae_core.market.apis.xxx`). Fixed hardcoded data paths. Created `__init__.py` for all 6 subpackages. Moved learned data to `data/market/`.
      Result: All 16 modules import cleanly. 2425 tests pass. Smoke test clean.

- [x] **Write MIDGE identity docs** (2026-02-22)
      What: CLAUDE.md, README.md, HANDOFF.md, midge-index.md, pyproject.toml, MAES_BIOLOGY.md, SYSTEMS.md updated for MIDGE identity. Divergence point documented. Project memory created.
      Result: All project files reflect MIDGE as trading-purposed Mae fork.

---

## Active

### Priority 1: Bootstrap Integration

- [ ] **Create market bootstrap layer (Layer 33)**
      What: Instantiate ThompsonSampler, ConvergenceAlerter, VelocityDetector, CorrelationTracker, edge detectors in bootstrap. Register on ctx namespace.

- [ ] **Register market connections with ConnectionRegistry**
      What: All market system connections need triadic witnesses (Law 1). Edge detectors -> EventBus, intelligence -> learning loop.

- [ ] **Register market systems with HolonRegistry**
      What: Market systems as holons with 10 capabilities (Law 3). Place in fractal hierarchy.

- [ ] **Wire market signals to EventBus**
      What: Edge detectors publish signals, convergence alerter subscribes, agent decision cascade receives.

### Priority 2: Stem Cell Specialization

- [ ] **Create market-specific role profiles**
      What: Add to stem_cell.py: sec_watcher, contract_tracker, market_analyst roles.

### Priority 3: API Gateway Integration

- [ ] **Route market APIs through BoundaryMembrane**
      What: SEC EDGAR (trusted), RapidAPI (keyed), yfinance (trusted), USASpending (trusted), SAM.gov (keyed).

### Infrastructure

- [ ] **Clean up staging directory**
      What: Remove `_midge_staging/` after confirming all code ported. Delete recovery clone from AppData/Local/Temp.

---

**When completing a task:**
1. Mark as `[x]` with completion date
2. Git commit preserves history -- no separate history file needed
