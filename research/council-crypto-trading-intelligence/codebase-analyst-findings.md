# Codebase Analyst Findings — Crypto Trading Intelligence
**Date:** 2026-03-17
**Analyst Role:** Codebase Analyst
**Project:** MIDGE

---

## Executive Summary

MIDGE's architecture is crypto-capable but crypto-unoptimized. The plumbing is correct — signals reach convergence, execution reaches Alpaca — but four structural gaps prevent reliable crypto profits: (1) new crypto sources are invisible to MarketClock during off-hours, (2) convergence windows are equity-length for crypto signals that expire in hours, (3) outcome tracking for crypto trades uses wrong success thresholds and decay rates, and (4) Thompson has no crypto-specific priors for the five new sources added today. None of these require architectural changes — they are all configuration or small code additions.

---

## Section 1: 24/7 Sensing Architecture

### What the code actually does

**The step loop runs continuously.** There is no "daemon sleep" at the signal-collection level. When run as `--daemon`, Mesa executes steps indefinitely at the configured pace. At `--pace 2.0`, each step takes ~2 seconds of wall time; at actual LLM overhead (~24s), steps take ~24s. The sensing hook fires every `fetch_cadence` steps (default 10 steps = ~4 min at 24s/step actual).

**MarketClock filters by source name, not domain.** The `source_available()` method checks three sets: `ALWAYS_AVAILABLE`, `FUTURES_HOURS`, and `MARKET_HOURS`. If a source isn't in any of these sets, it defaults to `True` (fail-open policy, line 298 of `market_clock.py`).

**Critical finding — three new crypto sources are NOT registered in MarketClock:**

| Source Name | Should Be | Currently In |
|-------------|-----------|--------------|
| `kraken_futures` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |
| `mempool_btc` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |
| `crypto_news` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |
| `defillama` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |
| `crypto_rotation` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |
| `polymarket` | ALWAYS_AVAILABLE | Not in any set → defaults True (OK by accident) |

The fail-open behavior means these sources currently work 24/7 by accident. However, this is fragile — any future code that tightens the fail-open behavior would silently drop all crypto sources on weekends. The fix is to explicitly add them to `ALWAYS_AVAILABLE`.

**Sources that already explicitly support 24/7 crypto:**
- `crypto_prices` (CoinGecko) — in ALWAYS_AVAILABLE
- `crypto_exchange` (CoinCap) — in ALWAYS_AVAILABLE
- `binance_funding` — in ALWAYS_AVAILABLE
- `crypto_fear_greed` — in ALWAYS_AVAILABLE

**Weekend behavior for TA indicators:** `ta_indicators` is in `FUTURES_HOURS` (not equity hours). This means RSI/MACD/Bollinger runs Sunday 6PM through Friday 5PM — it does NOT run Saturday or early Sunday. For crypto, which moves on weekends, this is a real gap. Weekend TA signals cannot contribute to convergence.

**The sensing cadence:** At 24s/step actual pace with `fetch_cadence=10`, each source gets fetched roughly every 4 minutes. With 48 sources in `SOURCE_ROTATION` and 8 concurrent workers, each source completes one full rotation cycle every ~24 minutes. For crypto day trading where signals can expire in 1-2 hours, a 24-minute cycle is marginal but acceptable given the signal buffer persists across cycles.

---

## Section 2: Convergence Tuning for Crypto

### Threshold Analysis

**Current thresholds (global, equity-derived):**
- `min_domains = 3` (set in `convergence_alerter.py` line 188)
- `min_confidence = 0.45` (in `learning_config.py` line 29, used in `market_hooks_sensing.py` line 171)
- `min_strength = 0.65` (in `learning_config.py` line 30)

**Are these appropriate for crypto?** The `min_domains=3` requirement is a concern. For a crypto ticker like ETH, the available domain inventory is:
- `crypto` (CoinGecko price change)
- `sentiment` (Fear & Greed index, fanned out to BTC/ETH/SOL/XRP/ADA)
- `derivatives` (Kraken funding rates — NEW today)
- `defi` (DefiLlama TVL — for ETH specifically)
- `on_chain` (mempool — BTC only, not ETH/SOL)
- `news` (CoinDesk/Cointelegraph RSS — NEW today)
- `technical` (TA indicators — if futures hours)
- `crypto_structure` (BTC dominance via `crypto_rotation`)

An ETH trade could plausibly get 4-5 domains. A SOL trade might only get 3 (crypto price + sentiment + news). The threshold is appropriate IF all sources are wired and firing. The concern is coverage gaps for non-BTC assets.

**The enrichment_group deduplication (critical finding):** In `convergence_ticker.py` lines 85-100, signals from the same `enrichment_group` only count as one domain contribution. The `fetch_crypto_fear_greed` function in `fetchers_crypto.py` (lines 177-178) fans out the same Fear & Greed reading to all 5 crypto tickers with `enrichment_group = f"crypto_fg_{date}"`. This is correct — it prevents one reading from artificially inflating convergence. However, this means for any crypto ticker, the Fear & Greed index contributes exactly ONE domain (sentiment), not five separate domain contributions.

**Domain window problem for crypto:** `convergence_alerter.py` lines 210-215 define `_domain_windows`:
```python
self._domain_windows = {
    "positioning": timedelta(hours=14 * 24),  # 14 days
    "government": timedelta(hours=7 * 24),    # 7 days
    "contracts": timedelta(hours=7 * 24),     # 7 days
    "energy": timedelta(hours=7 * 24),        # 7 days
}
```
Domains not listed fall back to `self.convergence_window` = **72 hours** (3 days). This includes `crypto`, `sentiment`, `derivatives`, `defi`, `on_chain`, `news`.

For crypto day trading, a funding rate signal from 72 hours ago is stale — funding rates update every 8 hours and mean something right now, not 3 days ago. A crypto news signal from 48 hours ago may describe events that already resolved. The 72-hour default window was designed for equity signals (insider trades, contract awards) that have multi-day predictive value. Crypto signals need a 2-6 hour window.

**No crypto-specific domain window is defined.** This is the highest-priority structural gap for crypto day trading.

---

## Section 3: Execution Gaps

### The Execution Pipeline

The full crypto execution chain is:
1. `check_ticker_convergence()` → `ConvergenceAlert` (per-ticker, domain-filtered)
2. `_run_paper_trading_gate()` in `market_hooks_sensing.py` — confidence + strength + combo gates
3. `_write_paper_trade()` in `market_hooks_trades.py` — writes to `paper_trades.jsonl`
4. `_translate_and_log_executable_signal()` — fetches price, computes ATR, builds `ExecutableSignal`
5. `_submit_to_alpaca()` — converts symbol, places market order

### Gap 1: Crypto SL/TP is disabled — no bracket orders

In `_submit_to_alpaca()` (line 469):
```python
_tp = None if is_crypto else round(signal.take_profit, 2)
_sl = None if is_crypto else round(signal.stop_loss, 2)
```
Alpaca does not support bracket orders for crypto. Crypto orders are placed as simple market orders with no stop-loss and no take-profit. The signal translator computes ATR-based SL/TP but those values are discarded. **This means MIDGE can enter a crypto trade but has no automatic exit mechanism.** For swing trading (multi-day holds) this may be acceptable. For day trading, it is a critical risk gap — an adverse move has no floor.

### Gap 2: Crypto positions are never closed

The Alpaca client has a `close_position()` method (line 276). The portfolio tracker runs every 50 steps (`sensing_hook.py` line 283). However, there is no code path that calls `close_position()` on a crypto trade when the thesis fails or a target is hit. The equity path uses ATR-based TP/SL embedded in bracket orders (which auto-close). Crypto has no equivalent.

**Blast radius if we add crypto exits:** `portfolio_tracker` (in `mae_core/market/`) handles mark-to-market and exit signal checks. Adding crypto exit logic there would be contained to that module and `market_hooks_sensing.py` where `_run_step_portfolio()` is wired.

### Gap 3: ATR computation may fail for crypto

`_translate_and_log_executable_signal()` (line 368) calls `price_fetcher.get_daily_history(_yf_ticker, days=30)`. The yfinance ticker for BTC is `BTC-USD` — this is handled by the `_CRYPTO_BASES` set (line 348). For DOGE, AVAX, LINK, LTC, SHIB, BCH, AAVE, DOT — also in `_ALPACA_CRYPTO_BASES` (line 428) — the yfinance suffix is correctly applied. ATR computation requires 15+ daily bars (line 372), which yfinance provides for major crypto. This gap is likely small but worth verifying for thin-liquidity assets.

### Gap 4: TimeInForce.DAY for crypto market orders

In `alpaca_client.py` lines 204-209, simple market orders use `TimeInForce.DAY`. Crypto markets trade 24/7 but Alpaca paper trading for crypto may still require DAY time-in-force. This is API constraint behavior that should be verified against Alpaca's crypto documentation. If a DAY order is placed outside Alpaca's processing hours it may not fill.

---

## Section 4: Signal Quality — Which Sources Actually Produce Crypto Signals

### Source inventory for crypto tickers (BTC, ETH, SOL, XRP, ADA, DOGE, AVAX, LINK, LTC)

| Source | Crypto Tickers | Domain | Status |
|--------|----------------|--------|--------|
| `crypto_prices` (CoinGecko) | BTC/ETH/SOL/XRP/ADA (5) | `crypto` | Active, 24/7 |
| `crypto_exchange` (CoinCap) | Top 10 by market cap | `crypto` | Active, 24/7 |
| `crypto_fear_greed` | BTC/ETH/SOL/XRP/ADA (5, same reading, enrichment_group) | `sentiment` | Active, 24/7 |
| `crypto_rotation` (BTC dominance) | BTC + ETH/SOL/XRP/ADA (fan-out, enrichment_group) | `crypto_structure` | Active, 24/7 |
| `kraken_futures` | BTC/ETH/SOL/XRP/ADA/DOGE/LINK/LTC/AVAX (9) | `derivatives` | NEW today, wired |
| `mempool_btc` | BTC ONLY | `on_chain` | NEW today, BTC-only |
| `crypto_news` | BTC/ETH/SOL/XRP/ADA/DOGE (6) | `news` | NEW today, wired |
| `defillama` | ETH/SOL/BTC/AVAX/ADA (chain-mapped) | `defi` | NEW today, inline code |
| `ta_indicators` | All watchlist tickers | `technical` | Futures hours only |
| `finra_short` | Equity only | — | No crypto |
| `stocktwits` | Watchlist | `sentiment` | Limited crypto coverage |
| `google_trends` | Watchlist keywords | `sentiment` | Keyword-dependent |

**Dead wire finding:** The `defillama` source is implemented entirely as inline code in `_fetch_source()` in `sensing_reactive.py` (lines 329-356). It instantiates `DefiLlamaClient()` directly on every fetch call (no client instance stored in `self`), which means it cannot receive a pre-configured client from bootstrap, is not visible to the health monitoring system, and its call rate is not tracked. All other sources use injected clients via `self._xxx_client`. This is an inconsistency in the pattern.

**Thompson key gaps:** The `_SOURCE_TO_THOMPSON_KEY` map in `convergence_alerter.py` (lines 82-142) does NOT include entries for:
- `kraken_futures` — will not get Thompson weighting in confidence computation
- `mempool_btc` — will not get Thompson weighting
- `crypto_news` — will not get Thompson weighting (source key is `crypto_news_rss` in the signal dict, but Thompson key lookup uses `signal.source`)
- `defillama` — source is `defillama` in the signal dict, no Thompson key
- `crypto_rotation` — source is `crypto_rotation`, no Thompson key in `_SOURCE_TO_THOMPSON_KEY`

When a source is missing from `_SOURCE_TO_THOMPSON_KEY`, it falls back to using the raw signal's `confidence` value rather than Thompson-weighted confidence. This means new crypto sources contribute raw confidence (typically 0.50-0.65 as hardcoded in the client) rather than learned reliability. Confidence computation degrades but does not break.

**`learning_config.py` priors missing for new sources.** The `source_reliability` dict in `learning_config.py` has `crypto_coingecko: 0.50` and `crypto_coincap: 0.50` but nothing for `kraken_futures`, `mempool_btc`, `crypto_news`, `defillama`, `crypto_rotation`, or `crypto_fear_greed`. These will start at the uninformative `Beta(1,1)` prior (mean 0.50).

**`crypto_news_client` source name mismatch:** The `CryptoNewsClient.get_news_signals()` returns dicts with `"signal_source": "crypto_news_rss"` (line 174) but the outer key is `"symbol"`, not `"source"`. When these dicts flow into `enrich_signal()` and then `record_signal()`, they need a `"source"` key. Looking at `fetchers_crypto.py` line 353-361, the function returns `crypto_news_client.get_news_signals()` directly without normalizing the source field. The signal adapter layer may or may not handle this — requires verification.

---

## Section 5: Thompson Learning for Crypto

### Are crypto outcomes being graded?

**The feedback loop is wired but uses equity-inappropriate windows.** `outcome_collector.py` lines 43-56 defines `OUTCOME_WINDOWS`. There is no entry for:
- `crypto_coingecko`
- `crypto_coincap`
- `crypto_fear_greed`
- `kraken_futures`
- `mempool_btc`
- `crypto_news`
- `derivatives`
- `on_chain`

Any signal from these sources falls back to the default 14-day window (`OUTCOME_WINDOWS.get(source, 14)`). For crypto day/swing trading, a 14-day outcome window is problematic — it evaluates a day-trading signal against a 2-week price move, which is dominated by macro noise, not the day-trade thesis. Crypto outcome windows should be 1-3 days for day trading, 5-7 days for swing.

**`SUCCESS_THRESHOLD_PCT = 5.0`** (line 59) — for equities, 5% is a meaningful move. For crypto, 5% can happen in hours and is almost table stakes volatility. A 5% threshold may be too low (catches noise moves) or fine depending on the instrument. For BTC, 5% is moderate. For SOL or ADA, 5% is a routine daily fluctuation. This means Thompson will receive too many "success" updates for crypto signals, inflating their perceived reliability.

**`RegimeClassifier` uses SPY as reference** (line 31 of `regime_classifier.py`). The `_REFERENCE_SYMBOL = "SPY"` is hardcoded. Crypto regime (bull/bear/volatile/sideways) is completely uncorrelated with SPY regime during certain periods (crypto has its own cycles). Thompson regime-aware learning for crypto uses the equity regime, which may mismatch the actual crypto market context. For example, crypto may be in a violent bull run while equities are sideways — Thompson applies the "sideways" decay rate to crypto distributions.

**The forgetting cadence.** `apply_forgetting()` applies a uniform decay to ALL distributions. This means crypto-specific distributions (which learn faster because crypto is more active) decay at the same rate as equity distributions. No crypto-specific decay rate exists in `REGIME_DECAY_RATES`.

---

## Section 6: Blast Radius of Required Changes

### Change 1: Add crypto windows to `_domain_windows` in `convergence_alerter.py`
- **Files:** `convergence_alerter.py` only (lines 210-215)
- **Risk:** Low. Adding new keys to `_domain_windows` only affects the pruning cutoff for new domains. Existing equity domains are unchanged.
- **Pattern:** Same pattern as existing dict, just add `"crypto"`, `"derivatives"`, `"on_chain"`, `"defi"`, `"news"` with 2-6 hour windows.
- **Tests:** `tests/market/test_convergence_alerter.py` — domain window tests would need to cover new keys.

### Change 2: Add new sources to `MarketClock.ALWAYS_AVAILABLE`
- **Files:** `market_clock.py` only (lines 37-67)
- **Risk:** Very low. Sources not in any set already default to True. Adding them to ALWAYS_AVAILABLE makes the behavior explicit and stable.
- **No tests break.** MarketClock tests check specific sources.

### Change 3: Add Thompson keys for new crypto sources
- **Files:** `convergence_alerter.py` (`_SOURCE_TO_THOMPSON_KEY`, lines 82-142), `convergence_alerter.py` (`_DOMAIN_SOURCES`, lines 144-166), `learning_config.py` (`source_reliability`)
- **Risk:** Low. Thompson falls back gracefully when a source is missing. Adding keys improves accuracy but doesn't change control flow.
- **Tests:** Thompson tests in `tests/market/test_thompson_sampler.py`.

### Change 4: Add crypto outcome windows to `OUTCOME_WINDOWS`
- **Files:** `outcome_collector.py` (lines 43-56)
- **Risk:** Low. New keys don't affect existing predictions. Old predictions use their stored window from registration, not the current `OUTCOME_WINDOWS` dict.
- **Tests:** `tests/market/test_outcome_collector.py`.

### Change 5: Add crypto exit mechanism (TP/SL for Alpaca crypto)
- **Files:** `market_hooks_trades.py` (small), `portfolio_tracker.py` (main work), possibly `alpaca_client.py`
- **Risk:** Medium. Alpaca crypto doesn't support bracket orders. Alternative: submit a separate limit order for TP and a stop-limit order for SL after entry. Requires Alpaca API investigation. Could also use manual close logic in portfolio_tracker via `close_position()`.
- **Tests:** Would require new tests.

### Change 6: Fix DefiLlama inline instantiation
- **Files:** `sensing_reactive.py` (lines 329-356), `sensing_hook.py` (constructor), `bootstrap/market_hooks_sensing_setup.py`
- **Risk:** Low. Pattern is identical to every other client. Inject via constructor, wire in bootstrap.
- **Tests:** No new tests required (existing sensing tests cover the pattern).

### Change 7: Fix `crypto_news_rss` source key mismatch
- **Files:** `mae_core/market/apis/crypto_news_client.py` (line 174 — change `signal_source` to `source`)
- **Risk:** Very low. The key rename is purely cosmetic for the convergence pipeline.
- **Tests:** Existing signal adapter tests.

---

## Section 7: Pattern Inventory

**How TA indicators work for crypto:** `ta_indicators` source in `FUTURES_HOURS` means RSI/MACD run on BTC/ETH/SOL etc. during CME futures hours. These indicators are calculated using `price_fetcher.get_daily_history()` with yfinance. The `_CRYPTO_BASES` set in `market_hooks_trades.py` handles the `-USD` suffix for yfinance. This is fully functional — just unavailable on weekends.

**How enrichment groups prevent double-counting:** A signal with `enrichment_group` set (line 93 in `convergence_ticker.py`) will not count twice in a single convergence evaluation. The Fear & Greed fan-out and BTC dominance fan-out both use this mechanism correctly — each contributes one domain regardless of how many tickers were fanned to.

**How the dedup window interacts with crypto:** The 1-hour dedup window (`_TICKER_DEDUP_HOURS = 1.0` in `convergence_ticker.py` line 25) means the same BTC signal won't alert twice within an hour. For crypto day trading where momentum moves fast, this is appropriate. It would prevent signal spam on volatile crypto days.

**How the paper trading gate interacts with crypto:** Only `TCKR-` prefixed alerts generate trades (line 179 in `market_hooks_sensing.py`). Global convergence alerts are explicitly skipped. This means all crypto trades will be ticker-specific convergences — correct behavior.

---

## Scorecard

### Role-Specific Dimensions

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Feasibility** | 8 | All required changes are small and localized — config additions and two-line fixes, not architectural rewrites |
| **Blast Radius** | 8 | Changes are confined to 5-6 files; equity pipeline is completely isolated from crypto-domain additions |
| **Pattern Consistency** | 9 | All proposed changes follow existing patterns exactly — dict extensions, set additions, constructor injections |
| **Dependency Risk** | 8 | No new external libraries; Alpaca crypto exit mechanism is the only change touching an external API contract |

### Shared Dimensions

| Dimension | Score | Justification |
|-----------|-------|---------------|
| **Overall Risk** | 8 | The five most impactful fixes (windows, clock, Thompson keys, outcome windows, source key) are all low-risk configuration changes with no breaking behavior |
| **Reversibility** | 9 | Every fix is either a dict key addition (trivially removed) or a constant change (one-line revert); the only non-trivial change is the crypto exit mechanism |
| **Evidence Confidence** | 9 | Findings are based on direct code reading of all specified files plus dependency-chased supporting files; conclusions are grounded in specific line numbers |

---

## Priority Recommendations

**P0 — Do these today before any live trading:**

1. **Add crypto domain windows** to `convergence_alerter.py._domain_windows`. Suggested: `"crypto": timedelta(hours=4)`, `"derivatives": timedelta(hours=4)`, `"on_chain": timedelta(hours=2)`, `"defi": timedelta(hours=6)`, `"news": timedelta(hours=3)`. Without this, 48-hour-old crypto signals are being treated as current signals.

2. **Add crypto outcome windows** to `outcome_collector.py.OUTCOME_WINDOWS`. Suggested: `"crypto_coingecko": 3`, `"crypto_coincap": 3`, `"kraken_futures": 2`, `"mempool_btc": 1`, `"crypto_news": 2`, `"defillama": 5`. Without this, Thompson is learning from 14-day evaluations of 2-hour signals.

3. **Fix `crypto_news` source field name** in `crypto_news_client.py` — `signal_source` → `source`.

**P1 — Do these before collecting meaningful Thompson data:**

4. **Add Thompson keys** for the 5 new crypto sources to `_SOURCE_TO_THOMPSON_KEY` and `learning_config.py source_reliability`.

5. **Add new sources to `MarketClock.ALWAYS_AVAILABLE`** to eliminate fragility.

6. **Add TA indicators to ALWAYS_AVAILABLE** (or at minimum add a crypto-hours set). Weekend crypto moves will miss TA domain signals.

**P2 — Required before actual money:**

7. **Crypto exit mechanism** — portfolio_tracker needs to close positions when price hits TP/SL levels by calling `alpaca_client.close_position()`. The signal translator already computes the levels; they just need to be tracked and acted on.

8. **Crypto RegimeClassifier** — add a `CryptoRegimeClassifier` using BTC price data instead of SPY, so Thompson regime-aware learning uses the right reference.

---

## File Reference Map

| File | Key Finding |
|------|-------------|
| `mae_core/market/intelligence/convergence_alerter.py` | `_domain_windows` has no crypto entries; 72h default is too long |
| `mae_core/market/intelligence/convergence_alerter.py` | `_SOURCE_TO_THOMPSON_KEY` missing 5 new crypto sources |
| `mae_core/market/intelligence/convergence_ticker.py` | Enrichment group dedup is correct; 1h dedup is appropriate |
| `mae_core/market/sensing_hook.py` | 24 concurrent workers, 10-step cadence, no sleep gaps |
| `mae_core/market/sensing_constants.py` | 48 sources in rotation; all new crypto sources present |
| `mae_core/market/sensing_reactive.py` | DefiLlama is inline (not injected client) — inconsistent pattern |
| `mae_core/market/market_clock.py` | New crypto sources not in ALWAYS_AVAILABLE; fail-open saves them |
| `mae_core/market/market_clock.py` | `ta_indicators` in FUTURES_HOURS → no TA on weekends |
| `mae_core/bootstrap/market_hooks_trades.py` | Crypto SL/TP explicitly disabled (`_tp = None if is_crypto`) |
| `mae_core/market/intelligence/thompson_sampler.py` | No crypto-specific decay rate; SPY regime contaminates crypto |
| `mae_core/market/intelligence/learning_config.py` | Only `crypto_coingecko`/`crypto_coincap` have priors; 5 new sources missing |
| `mae_core/market/intelligence/outcome_collector.py` | `SUCCESS_THRESHOLD_PCT=5.0` may overreport crypto success; 14d default too long |
| `mae_core/market/intelligence/regime_classifier.py` | `_REFERENCE_SYMBOL = "SPY"` hardcoded — wrong for crypto regime detection |
| `mae_core/market/apis/alpaca_client.py` | `close_position()` exists but never called for crypto |
| `mae_core/market/apis/kraken_futures_client.py` | 9 symbols, 30s cache, correct contrarian direction logic |
| `mae_core/market/apis/mempool_client.py` | BTC-only, 60s cache, two signal types (fee spike + congestion) |
| `mae_core/market/apis/crypto_news_client.py` | 6 tickers, 5min cache, `signal_source` key instead of `source` |
