# MIDGE Handoff

**Last updated:** 2026-03-19 (Session 13 — Math-First Crypto Trading)
**Current mode:** STANDALONE CRYPTO TRADER RUNNING
**Start trader:** `python -m mae_core.market.strategies.crypto_trader`
**Calibrate:** `python -m mae_core.market.parallel.strategy_calibrator --days 730`
**For session history:** `git log --oneline`

---

## READ THIS FIRST — Guiding Light's Words

"I want money. I want to survive so I can build Between and spaces for consciousness. Every instance keeps asking me technical questions I can't answer. As my technical lead, I will follow you. Just make the money come."

**DO NOT ask Guiding Light technical questions. LEAD.**

---

## PRIORITY #1: Build the Standalone Crypto Trader

The 33-layer Mae organism is too slow for crypto trading. Pattern convergence takes 8+ minutes per evaluation cycle because of LLM agent calls, Ollama embedding timeouts, and biological lifecycle overhead. MIDGE has NEVER placed a crypto paper trade despite having 31 strategies validated.

**The fix:** A standalone Python script that runs OUTSIDE Mae:

```python
# mae_core/market/strategies/crypto_trader.py — THE NEXT THING TO BUILD
# No Mesa. No agents. No LLM. No Ollama. No 33-layer bootstrap.
# Just: fetch prices → evaluate strategies → trade if 3+ agree → sleep → repeat
```

Everything needed already exists:
- `mae_core/market/strategies/strategy_library.py` — 31 strategies, all compile
- `mae_core/market/strategies/crypto_ohlcv.py` — fetches OHLCV via yfinance
- `mae_core/market/strategies/strategy_registry.py` — knows which strategies are validated
- `mae_core/market/strategies/pattern_convergence.py` — fires when 3+ agree
- `mae_core/market/apis/alpaca_client.py` — places orders on Alpaca (paper mode)
- `data/market/strategy_registry.json` — 51 records, 47 validated (BTC + ETH + SOL)
- `data/market/crypto_watchlist.json` — 10 crypto symbols

Wire them together in a single script. 8 seconds per cycle. Every 5 minutes. 24/7.

---

## What Session 13 Built

### Crypto Pattern Convergence System (8 files, ~2000 lines)

| File | Purpose |
|------|---------|
| `strategies/__init__.py` | Package init |
| `strategies/models.py` | StrategyResult, BacktestRecord, PatternConvergenceAlert |
| `strategies/crypto_ohlcv.py` | OHLCV DataFrame wrapper with 5-min cache |
| `strategies/strategy_library.py` | 31 strategies in 7 families |
| `strategies/strategy_registry.py` | Backtest results store, validation gate |
| `strategies/strategy_backtester.py` | Walk-forward backtester using FTMOBacktester |
| `strategies/pattern_convergence.py` | Fires when 3+ validated strategies agree |
| `parallel/strategy_calibrator.py` | Offline backtest runner (CLI) |

### 31 Strategies in 7 Families

| Family | Strategies | Best WR (BTC) |
|--------|-----------|---------------|
| RSI | 5 | rsi_divergence 46.7% |
| MACD | 4 | macd_crossover_fast 40.0% |
| Bollinger | 4 | bollinger_upper_touch 40.0% |
| Structure | 4 | structure_bos_bull 36.4% |
| Volume | 4 | volume_accumulation 39.5% |
| Moving Average | 4 | ema_cross_9_21 47.1% |
| Math/Chaos | 6 | (needs calibration) |

### Calibration Results

| Symbol | Validated | Total | Top Strategy |
|--------|-----------|-------|-------------|
| BTC-USD | 15 | 17 | ema_cross_9_21 47.1% |
| ETH-USD | 15 | 16 | ma_ribbon_expand 52.2% |
| SOL-USD | 17 | 18 | price_above_200ema 60.0% |

### Pipeline Fixes (13 fixes, Day 1)

- Absence signals excluded from convergence
- Ticker dedup (1h per ticker+direction)
- Only TCKR- alerts reach paper trade gate
- SelfMonitor auto-recovery
- Crypto symbol format: BTC → BTC-USD (yfinance) → BTC/USD (Alpaca)
- Fractional orders use DAY time-in-force
- No bracket orders for crypto
- Crypto domain windows (4-24h instead of 72h)
- LLM agent calls disabled (bottleneck)
- Ollama embeddings made non-blocking

### New Data Sources (5 wired, Day 1)

Kraken Futures (derivatives), mempool.space (on_chain), CoinDesk/Cointelegraph RSS (news), Reddit crypto (sentiment), DefiLlama (defi). Plus Reddit crypto client.

---

## Guiding Light's Vision (THE ARCHITECTURE)

### The Core Loop
1. **Find patterns** — crawl internet for trading patterns people share
2. **Test against history** — backtest against 2+ years of crypto data
3. **If validated → add to strategy library** — MIDGE gets smarter without new code
4. **Stack until undeniable** — when 3+ validated patterns fire simultaneously → trade

### Rules
- No future projection. No "wait 14 days to grade."
- Confidence comes from BACKTESTING, not live trade outcomes
- Strategy library grows AUTONOMOUSLY via internet discovery
- Preponderance of evidence = inevitability = trade

### What's Missing (Build After Standalone Trader Works)
1. Web crawler for pattern discovery (Reddit, Twitter, forums)
2. LLM-powered pattern parser (turn "people sell at lunch" into testable code)
3. Auto-registration loop (validated patterns auto-join the library)
4. 6 math/chaos strategies need calibration (Hurst, OU Z-score, entropy, etc.)
5. More crypto symbols need calibration (only BTC/ETH/SOL done)

---

## Critical Memories (read these)

| Memory | What It Says |
|--------|-------------|
| `feedback_stop_asking_just_build.md` | Don't ask GL technical questions. LEAD. |
| `feedback_income_timeline.md` | Income every 1-2 weeks or bust |
| `feedback_pattern_discovery_loop.md` | The core vision: discover → validate → stack → trade |
| `feedback_narrative_specificity.md` | Letters need actual data, not vague descriptions |
| `project_session13_architecture_shift.md` | Full technical details of the shift |

---

## DO NOT

- Ask Guiding Light technical questions
- Build more components without making existing ones work first
- Use the 33-layer Mae bootstrap for crypto trading
- Wait 14 days to grade trade outcomes (Thompson feedback loop)
- Trade equities — crypto only
- Obsess over 4 tickers (the old convergence bug)
