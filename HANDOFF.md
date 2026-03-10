# MIDGE Handoff

**Last updated:** 2026-03-09
**For session history:** `git log --oneline`

---

## What Is MIDGE

MIDGE is Mae differentiated for financial markets. She's an inevitability surfacer — a living organism that observes patterns across 32 data sources, finds where converging forces make outcomes structurally inevitable, and trades on them.

Guiding Light's vision: MIDGE as personal autonomous trader across ALL markets — stocks (Alpaca), futures/forex (FTMO), crypto (exchanges), prediction markets (Kalshi). Not one venue — all of them.

---

## What Works Right Now

### The Brain
- **32 data sources** feeding signals through 12 concurrent workers, 25-step rotation cadence
- **Convergence engine** (crown jewel) — fires when 3+ independent domains agree on direction
- **Thompson Bayesian learning** — 35/37 distributions rebuilt from 13,190 historical updates. Brain is learning.
- **Signal translator** — ConvergenceAlert → ExecutableSignal with ATR-based stop-loss/take-profit
- **Pattern archaeology** — 223K fingerprints, 39 templates, live matching via PatternWatcher
- **WorldModel causal graph** — 114 nodes, 102 edges, forward/backward cascade tracking

### The Body
- **149 systems** (92 core + 57 market), 33-layer bootstrap, 157 holons, 428 connections
- **29/30 biological systems** wired to market channels (only GenerativeReplayMemory unwired)
- **OctopusColony** bootstrapped with 3-7 auto-scaling octopuses, market task handlers, investigation pipeline
- **Two pipelines bridged** — market signals reach core attention via PatternTranslator protocol

### Execution
- **Alpaca paper trading: WIRED.** Keys in `.env`. Convergence alerts auto-submit bracket orders (entry + SL + TP) for US equities. DrawdownMonitor circuit breaker + SelfMonitor anomaly detection gate all trades. Forex/futures/crypto tickers filtered out (Alpaca = equities only).
- **FTMO backtester: PORTED.** `ftmo_engine.py` + `ftmo_config.py` in `mae_core/market/execution/`. Simulates challenge constraints (daily loss, total DD, profit target). Not yet validated against MIDGE signals.
- **Kalshi SDK installed.** Not yet verified or wired.

### Risk
- **DrawdownMonitor** — 40% max DD circuit breaker, halts all paper trades
- **SystemHealthMonitor** — 8 subsystems tracked, tier-based health (Green→Red)
- **SelfMonitor** — behavioral anomaly detection (runaway rate, direction bias, ticker flooding)

---

## What To Do Next

See `midge-queue.md` for the full prioritized list. Top items:

1. **Start the daemon** — `python main.py --daemon --agents 12 --steps 500 --pace 2.0` — MIDGE will sense, converge, and paper-trade on Alpaca automatically
2. **Kalshi integration** — verify SDK, review ToS, prototype MarketSelector
3. **FTMO validation** — run historical convergence alerts through ftmo_engine.py
4. **Write missing tests** — USDA client, FRED yield curves
5. **New data sources** — BDI logistics, central bank speeches, crypto depth

---

## Key Technical Notes

**Files that matter:**
| File | Purpose |
|------|---------|
| `main.py` | 33-layer bootstrap orchestrator |
| `mae_core/bootstrap/market_hooks.py` | Step hooks, EventBus wiring, paper trading, Alpaca submission |
| `mae_core/bootstrap/market_systems.py` | System instantiation (444 lines) |
| `mae_core/market/intelligence/convergence_alerter.py` | Crown jewel — multi-domain synthesis |
| `mae_core/market/intelligence/thompson_sampler.py` | Bayesian learning with replay |
| `mae_core/market/execution/signal_translator.py` | ConvergenceAlert → ExecutableSignal |
| `mae_core/market/execution/ftmo_engine.py` | FTMO challenge backtester |
| `mae_core/market/sensing_hook.py` | MarketSensingHook — data fetching orchestrator |
| `data/midge/watchlist.json` | Tickers + keywords MIDGE watches |

**Bootstrap sub-modules** (market_systems.py delegates to):
- `market_infrastructure.py` — OctopusColony, risk monitors, pattern discovery, scheduling
- `market_intelligence.py` — hypothesis engine, archaeology
- `market_gifts.py` — ten gifts (portfolio, order flow, etc.)
- `market_hooks.py` — EventBus channels, step hooks
- `market_registration.py` — holon + fractal registration
- `market_connections.py` — triadic connections
- `market_agents.py` — agent differentiation

**Paper trade pipeline:**
1. Convergence alert fires (3+ domains agree)
2. DrawdownMonitor checks — blocked if halted
3. SelfMonitor checks — blocked if behavioral anomaly
4. `_write_paper_trade()` — logs to `data/midge/paper_trades.jsonl`
5. `_translate_and_log_executable_signal()` — ATR-based SL/TP → `data/midge/executable_signals.jsonl`
6. `_submit_to_alpaca()` — bracket order to Alpaca (US equities only)

**500-line cap enforced** on all bootstrap sub-modules. Two xfails: `sensing_hook.py` (576 lines) and `market_hooks.py` (551 lines) — both have sub-modules created but delegation not yet complete.

**Pre-existing flaky test:** `test_congress_gov_client::test_request_fails_without_key` — passes in isolation, fails in full suite due to env var pollution from another test. Not a real bug.

---

## Guiding Light's Vision

> "MIDGE needs to be an entire functioning ecosystem. She's more of a planet than a singular biological organism. Everything inside her should be active, not passive."

> "The goal is for MIDGE to become my personal trader using inevitabilities, temporal knowledge, and aggregate factors on when to buy/sell/hold — stocks, crypto, futures, ANYTHING that MIDGE can make money off of."

> "$1,000 gate: Deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy — inevitability, not prediction."

---

## Research

| Expedition | Location | Key Finding |
|------------|----------|-------------|
| FTMO Viability | `research/expedition-ftmo-viability/` | "Right destination, wrong next step" — fix Thompson first, expand senses, then FTMO |
| Autonomous Trading | `research/expedition-autonomous-trading/` | Kalshi as first venue, Alpaca for equities |
| Competitive Edge | `research/expedition-competitive-edge/` | Cross-domain convergence is MIDGE's structural moat |
| Evolution Blueprint | `research/evolution-blueprint/` | 10-team architectural roadmap |
| Phase 0 Measurements | `research/phase0-measurements.md` | 3.34:1 payoff ratio, 19.9% convergence WR |

---

## Verification

```bash
python -m pytest tests/ -q                    # ~970+ pass
python -m pytest tests/test_decomposition_wiring.py -v  # 61 pass, 2 xfail
python main.py --agents 3 --steps 30          # Smoke test
```

## Stats

- **149 systems** (92 core + 57 market), **4,536+ tests**, **157 holons**, **428 connections**
- **122 market files** (33 API + 12 edge + 36 intelligence + 8 signal_adapters + 10 archaeology + 6 execution + 17 root)
- **32 sources**, **12 domains**, **12 concurrent fetches**, **25-step cadence**
- **33-layer bootstrap**, **14 mixins** on MycelialAgent
