# MIDGE Handoff

**Last updated:** 2026-03-17 (Session 13 — Pipeline Unblock + Crypto Pivot)
**Start ecosystem:** `python -m mae_core.market.ecosystem.supervisor`
**Check status:** `python -m mae_core.market.ecosystem.supervisor --status`
**For session history:** `git log --oneline`
**Current mode:** SPRINT (ecosystem + crypto education sprint running overnight)

---

## Session 13 (2026-03-17): PIPELINE UNBLOCK + CRYPTO PIVOT

### Guiding Light's Directives (updated)

1. **Income every 1-2 weeks.** Method doesn't matter. Whatever works.
2. **Crypto is the focus.** 24/7 market, no PDT rule, no minimum balance.
3. **MIDGE trades autonomously.** No consulting Guiding Light. Act and report.
4. **Historical mining is the foundation.** SPRINT overnight, always.
5. **Letters must be SPECIFIC.** "$54.7M insider selling" not "people inside the company."
6. **Social media matters for crypto.** Reddit, Twitter — where sentiment forms first.
7. **Math and chaos patterns.** Study massive history, find mathematical regularities.

### What Was Fixed (13 fixes across 15 files)

**Pipeline unblock:** Absence signals excluded from convergence, ticker dedup added, only TCKR- alerts trade, SelfMonitor auto-recovers, validator gate=1 for paper trading.

**Crypto execution:** Symbol conversion (BTC→BTC-USD→BTC/USD), fractional order fix, no bracket for crypto, crypto domain windows (4-24h instead of 72h).

**Crypto sources (5 new):** Kraken Futures (derivatives), mempool.space (on_chain), CoinDesk/Cointelegraph RSS (news), Reddit crypto RSS (sentiment), DefiLlama (defi, was dead-wired).

**Signal fan-out:** Fear & Greed and BTC dominance now emit to BTC/ETH/SOL/XRP/ADA.

**Narrative:** Signal detail extractor pulls actual values from metadata.

### Current Stats

- **48 sources, 17 domains** | BTC has **7 independent domains** for convergence
- **Paper trades flowing** for equities (AKAM, ADBE). Crypto pending domain population.
- **616 tests pass**, 0 failures

### Critical Issues (from Research Council Devil's Advocate)

1. **No exit management for crypto** — positions are naked (no SL/TP on Alpaca crypto)
2. **Can't short crypto** on Alpaca — bearish signals can't execute
3. **All crypto domains correlate with BTC** — "independence" is weaker than in equities
4. **TA indicators locked to CME hours** — no TA on weekends
5. **Thompson cold start** — new sources have zero history

### Running NOW (SPRINT)

- Ecosystem supervisor (10 processes)
- Crypto education sprint (8 workers, 100 coins, 3 timeframes)

### Next Priorities

1. Verify crypto paper trades land on Alpaca
2. Build crypto exit manager (time-based trailing stop)
3. Fix TA for 24/7 crypto (remove CME filter)
4. Complete Research Council synthesis + tension analysis
5. Add CryptoPanic API (free with signup) for social sentiment
6. Measure paper trade P&L after 1 week → gate for live trading
