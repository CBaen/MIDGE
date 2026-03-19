# Crypto Strategy Library Research

**Date:** 2026-03-19
**Purpose:** Foundation for MIDGE strategy library — mathematically validated trading strategies for cryptocurrency swing/day trading.
**Scope:** 4 categories: Classic TA, Crypto-Specific, Pattern-Based, Mathematical/Chaos

---

## How To Read This Document

Each strategy entry follows this format:
- **What it measures** — the underlying signal
- **Parameters** — specific numbers to implement
- **Documented win rate** — from backtests where available (source noted)
- **Best timeframe** — tested evidence, not marketing
- **OHLCV-only?** — whether it can be computed from price/volume alone, or requires external data

Win rates without a source should be treated as community consensus, not rigorous backtest. Where sources exist, they are noted in brackets.

---

## Category 1: Classic TA Strategies

### 1.1 RSI (Relative Strength Index)

**What it measures:** Momentum oscillator comparing magnitude of recent gains vs losses. Values 0–100.

**Parameters (crypto-validated):**
- Period: **2-day RSI** (short-period outperforms in mean-reverting conditions), or **14-day** (standard), or **21-day** (smoother for swing)
- Oversold: **15–20** for aggressive entries, **30** for conservative
- Overbought: **80–85** for aggressive exits, **70** for conservative
- For crypto specifically: wider bands (20/80) are recommended due to crypto's larger volatility swings — standard 30/70 generates excessive signals in trending markets
- Divergence mode: RSI makes higher low while price makes lower low (bullish divergence), or RSI makes lower high while price makes higher high (bearish divergence)

**Documented win rates:**
- 2-day RSI mean-reversion on S&P 500: **91% win rate** (QuantifiedStrategies, $100K → $861K, 1993–2020, 42% market exposure, 33% max drawdown) [Source: quantifiedstrategies.com]
- RSI momentum on crypto (daily): **60%+ win rates** documented on BTC/USD 1H
- RSI divergences on crypto: ~**65% win rate** (community consensus, multiple sources)
- RSI alone without filters: **~50%**, close to random — always combine

**Best timeframe:** Daily is statistically strongest for RSI momentum strategies. 1H works for RSI divergence in crypto. 2-day period on daily bars is the highest-performing configuration in backtests.

**OHLCV-only?** Yes — computed entirely from close prices.

**MIDGE relevance:** Already implemented (`ta_indicators.py`). RSI parameters should use 14 or 21 period for swing, 2-day for mean-reversion mode. Oversold threshold should be 20 for crypto alerts, not 30.

---

### 1.2 MACD (Moving Average Convergence Divergence)

**What it measures:** Difference between two EMAs (trend + momentum). Three signals: signal line cross, zero line cross, histogram divergence.

**Parameters:**
- Standard: **12/26/9** (12-period EMA minus 26-period EMA, 9-period signal line)
- Fast/day-trading crypto: **5/13/5** or **8/17/9**
- The histogram is the MACD line minus the signal line — its slope matters more than its position

**Three signal types ranked by effectiveness (backtested):**
1. **MACD Histogram mean-reversion** — strongest: Profit Factor 4.22, CAGR 4.79%, max drawdown 16% [QuantifiedStrategies, S&P 500]
2. **MACD + 2-day RSI combination** — best combined: Profit Factor 2.45, CAGR 6.36%, max drawdown 16%
3. **Signal line crossover** — weakest: Profit Factor 1.65, CAGR 4.19%, max drawdown 30%
4. **Zero line crossover (aligned with 200 SMA)** — 72% win rate when trend-confirmed [multiple sources]

**On crypto specifically:**
- MACD alone on BTC/ETH: ~50–55% accuracy
- MACD + RSI filters on crypto: significantly improved (exact number not isolated, but consensus is 65–73%)
- Simple MACD crossover on Bitcoin standalone backtest: 49.39% annual return but 51.85% drawdown — acceptable only with position sizing
- 24/7 crypto markets make standard 12/26/9 suboptimal; faster settings (5/13/5) provide more timely signals but more noise

**Best timeframe:** Daily for swing trading. 4H for intraday trend confirmation. 5/13/5 on 1H for day trading.

**OHLCV-only?** Yes — computed entirely from close prices.

**MIDGE relevance:** Already implemented. The histogram mean-reversion mode is the highest-signal configuration. Prioritize histogram divergence signals over simple crossovers.

---

### 1.3 Bollinger Bands

**What it measures:** Volatility envelope around a moving average. Band width measures volatility compression/expansion.

**Parameters:**
- Standard: **20-period SMA, ±2 standard deviations**
- Scalping: **9-period MA, ±2 SD**
- Crypto-adjusted: **±2.5 or ±3 SD** (higher volatility = bands too tight at 2 SD)
- BandWidth (BBW) = (upper - lower) / middle band × 100
- Squeeze threshold for BTC: BBW < 5% = squeeze; BBW > 15% = extreme volatility

**Three strategy modes:**

**A. Mean Reversion (highest documented win rate)**
- Buy when price closes below lower band + RSI/MACD confirmation
- Exit at middle band or upper band
- Works in ranging/consolidating crypto markets
- Bitcoin BB mean-reversion backtest: ~**50% CAGR** while in market only **34% of the time** [QuantifiedStrategies, 2026]

**B. Squeeze Breakout**
- Identify when BBW < 4–5% (compression)
- Enter on candle close OUTSIDE the band
- Confirm with volume ≥ 1.5× 20-period average
- Win rate: positive expectancy documented across 5–10 bar holding period, but lower win rate than mean-reversion (breakout nature — many false starts)
- False breakout ("head fake") risk is high in crypto; volume confirmation is non-optional

**C. Trend Breakout**
- Price closing above upper band in trending market = momentum continuation
- Price closing below lower band in downtrend = momentum continuation
- Works in trending markets, fails badly in ranges

**Best timeframe:** Daily for mean-reversion. 4H for squeeze detection. Not reliable for scalping without volume confirmation.

**OHLCV-only?** Yes — computed from close prices and volume (for confirmation).

**MIDGE relevance:** Already implemented. BBW squeeze + breakout with volume confirmation is the most actionable for MIDGE's convergence engine (BB squeeze = a signal domain, volume spike = confirmation).

---

### 1.4 Fibonacci Retracement/Extension

**What it measures:** Key price levels derived from the Fibonacci sequence where reversals or continuations statistically occur more often.

**Parameters:**
- Key retracement levels: **23.6%, 38.2%, 50%, 61.8% (Golden Ratio), 78.6%**
- Key extension levels (targets): **100%, 127.2%, 161.8%, 261.8%**
- Most watched: **0.618** (61.8%) — the Golden Ratio. Most traders buy/sell here, making it self-fulfilling
- Optimal pullback zone: **0.382–0.618** (between 38.2% and 61.8%)
- Deep pullback / "last stand" zone: **0.786** (78.6%) — if this fails, trend is broken

**Strategy rules (consensus):**
- Swing low to swing high (bullish): retrace entry zone at 38.2–61.8%, stop below 78.6%, targets at 100%, 127.2%, 161.8%
- Confluence (strongly recommended): Fib level + BB lower band + RSI oversold + volume spike = high-quality entry

**Documented win rates:** No isolated backtest exists with clean win rate numbers — effectiveness is highly context-dependent and requires confluence. Without confluence, fib levels alone perform near random. With 3+ confluence factors at a fib level, community consensus puts probability at 65–75%, but no clean academic backtest.

**Best timeframe:** Works on all timeframes, but daily/4H levels carry more weight (more participants watching them). Crypto respects 61.8% more consistently than equities due to retail-dominant market structure.

**OHLCV-only?** Yes — computed from price extremes.

**MIDGE relevance:** Not currently implemented as a signal. Add as a signal layer: when price retraces to 61.8% fib level coinciding with BB lower band + RSI < 30 → strong convergence signal.

---

### 1.5 VWAP (Volume Weighted Average Price)

**What it measures:** Average price weighted by volume — institutional "fair value" benchmark for the session. Price above VWAP = bullish bias; below = bearish.

**Parameters:**
- Standard: resets daily (24h for crypto)
- Anchored VWAP (AVWAP): anchored from a significant event (breakout high, earnings, macro catalyst) — more useful for swing
- Standard deviation bands: VWAP ±1 SD, ±2 SD (institutional entry/exit zones)

**Strategy rules:**
- VWAP pullback: price above 50 EMA → pulls back to VWAP → RSI > 50 → enters bullish candle above VWAP
- VWAP rejection: price below VWAP, rallies to test VWAP from below → short entry
- VWAP + SD bands: VWAP +2 SD = overbought short-term; VWAP -2 SD = oversold short-term

**Win rate:** No clean isolated backtest found. Widely used by institutional desks as execution benchmark. Community consensus: VWAP pullback strategy in trending conditions achieves 60–65%.

**Crypto-specific caveats:**
- Crypto trades 24/7 — VWAP resets at midnight UTC cause artificial discontinuities
- Low-liquidity altcoins: a few large trades skew VWAP significantly — less reliable
- AVWAP anchored from major swing lows/highs is more robust than daily-reset VWAP for crypto swing trading

**Best timeframe:** Day trading: 1–30 min. Swing trading: use AVWAP on 4H/daily. Not useful for position trading.

**OHLCV-only?** Yes — requires price and volume.

**MIDGE relevance:** Not currently implemented. Useful as an intraday signal filter — if price is below daily VWAP, down-signals carry more weight; above VWAP, up-signals carry more weight.

---

### 1.6 Moving Average Crossovers

**What it measures:** Short-term trend crossing long-term trend — momentum shifts.

**Parameters (crypto-validated from backtests):**

| Pair | Use Case | Notes |
|------|----------|-------|
| 9 EMA / 21 EMA | Short-term swing, 4H–daily | Fast, many signals, crypto standard |
| 13 EMA / 48 EMA | Medium-term, daily | Backtested as best EMA combo for crossovers [Medium/Superalgos] |
| 15 MA / 150 MA | Medium–long term | Found optimal in one BTC backtest |
| 50 SMA / 200 SMA | "Golden/Death Cross," macro | Slow, few signals, macro trend confirmation |
| 20 SMA / 50 SMA | Classic swing | Equities-derived, works in crypto on daily |

**Backtest findings:**
- No single pair is universally optimal — depends on market phase
- Short EMA pairs (5–20): earlier entry, high false signal rate, lower win rate
- Long EMA pairs (50–200): later entry, fewer signals, higher win rate, less upside captured
- Best documented result: 13/48 EMA cross — "proven best EMA combination for crossover strategies" [Superalgos quantitative study]
- Golden Cross (50/200) + MACD confirmation on crypto: ~65% win rate (community consensus)

**Best timeframe:** Daily for Golden/Death Cross (macro signal). 4H for 9/21 swing. 1H for 9/21 day trading. Avoid using EMA crosses on sub-1H for crypto (too noisy).

**OHLCV-only?** Yes — computed from close prices only.

**MIDGE relevance:** Already partially implemented (TA indicators). Should add explicit 9/21 EMA cross signal and 50/200 SMA Golden/Death Cross signal as distinct signal types in the convergence engine.

---

## Category 2: Crypto-Specific Strategies

### 2.1 Funding Rate Mean Reversion

**What it measures:** In perpetual futures markets, funding rate is the periodic payment between longs and shorts to keep the futures price anchored to spot. Extreme positive funding = overcrowded longs (contrarian short signal). Extreme negative funding = overcrowded shorts (contrarian long signal).

**Parameters:**
- Data required: perpetual swap funding rate (8-hour periods on most exchanges)
- Extreme positive threshold: funding > **+0.1% per 8h** (annualizes to ~109%) = overleveraged longs
- Extreme negative threshold: funding < **-0.05% per 8h** = overleveraged shorts
- Spread strategy: Binance vs OKX funding spread mean reversion — spread typically ~2%, extremes at >5% or negative [rho.trading]

**Strategy types:**

**A. Directional reversion:** When funding hits extreme positive, price tends to revert downward as longs get squeezed. Buy extreme negative, sell/short extreme positive.

**B. Spread convergence:** Long funding on one exchange vs short on another — market-neutral, profits from convergence. Example: Binance/OKX spread returned 21% over 22 days in 2024 illustration [rho.trading].

**Research findings:**
- Slight **negative correlation** between BTC returns and BTC funding rates — alpha in mean-reversion confirmed [Fulgur Ventures via Medium]
- Mean reversion strategies on funding: **60–80% win rate** but small average profit per trade [QuantifiedStrategies]
- Fear-weighted DCA (buying more in extreme fear/negative funding) outperformed standard DCA by ~38% (2020–2025 backtest, The Block)

**Best timeframe:** 4H–Daily (funding settles every 8 hours, so minimum 8H holding for clean signal). Signal leads price reversal by 1–3 sessions on average.

**OHLCV-only?** No — requires dedicated funding rate data feed (Binance, Bybit, OKX APIs provide this for free).

**MIDGE relevance:** Not currently implemented. MIDGE already has CoinGecko/CoinCap. Funding rate data is available free from Bybit/Binance REST API. This is a high-signal crypto-native strategy that requires minimal data beyond the rate itself.

---

### 2.2 BTC Dominance Rotation

**What it measures:** Bitcoin's percentage of total crypto market cap. Falling dominance = capital rotating into altcoins (altcoin season). Rising dominance = capital fleeing to BTC (risk-off within crypto).

**Parameters:**
- BTC.D available on TradingView and CoinMarketCap
- Key thresholds (2024–2025 data):
  - BTC.D > 60%: BTC dominance phase, avoid altcoin longs
  - BTC.D < 50% and falling: altcoin season, altcoin longs favored
  - BTC.D breakdown from rising trend: rotation signal, buy ETH/altcoins relative to BTC
  - BTC.D at 62% (July 2025): signs of structural breakdown mirroring 2021 pre-altseason pattern [CCN]

**Strategy rules:**
- When BTC.D drops below a recent swing low and ETH/BTC pair breaks out: rotate into high-cap altcoins (ETH, SOL, BNB)
- When BTC.D rises sharply while BTC price falls: broad sell signal (market stress, alts will fall harder)
- Altcoin Season Index (CoinMarketCap) > 75: majority of top 50 outperforming BTC = confirmed altseason

**Documented performance:** No clean backtest with win rate. Historical observation: 2021 altseason was preceded by BTC.D dropping from 70% → 40%. The 2024–2025 BTC.D pattern mirrors 2021 setup [Millionero/CCN].

**Best timeframe:** Weekly/Monthly for macro positioning. Daily for rotation timing.

**OHLCV-only?** Requires market cap data (not OHLCV alone) — but CoinMarketCap and CoinGecko provide BTC.D as a chartable metric.

**MIDGE relevance:** MIDGE has CoinGecko already. BTC.D could be polled as a cross-market signal in the convergence engine. When BTC.D trend reverses + BTC price stable = altcoin rotation signal.

---

### 2.3 Fear & Greed Index (Contrarian)

**What it measures:** Composite sentiment index (0–100) aggregating: price volatility, market momentum/volume, social media sentiment, BTC dominance, Google Trends. 0 = extreme fear, 100 = extreme greed.

**Parameters:**
- Source: alternative.me (free API, updates daily)
- Extreme fear zone: **< 20** = buy signal (contrarian)
- Extreme greed zone: **> 80** = reduce/sell signal (contrarian)
- Confluence rule: index < 15 + RSI < 25 + on-chain accumulation = strongest buy signal

**Documented performance:**
- Buying BTC when index < 20: **+62% average 90-day return** historically — but 1 in 4 entries saw >25% further drawdown before recovering [multiple sources, The Block]
- Fear-weighted DCA (1%/day allocation when index < 20, sell 1% when > 80): **significantly outperformed buy-and-hold** [various sources citing The Block analysis]
- Fear-weighted DCA 2020–2025: **+38% better** than standard DCA [The Block backtest]
- Index alone is unreliable for short-term trading — markets can stay in extreme fear/greed for months

**Best timeframe:** Macro positioning (weekly, monthly). Poor signal quality on daily timeframes for active trading. Best used as position sizing modifier, not entry trigger.

**OHLCV-only?** No — requires sentiment aggregation (but alternative.me API is free).

**MIDGE relevance:** Already integrated (Crypto Fear & Greed client in Session 12). The 60-day average should be computed to separate macro regime from local extremes. The threshold for signal is < 20, not just "fear."

---

### 2.4 Stablecoin Supply Growth (Dry Powder Signal)

**What it measures:** Total supply of USDT/USDC on exchanges. Large stablecoin inflows to exchanges = traders prepared to buy = bullish dry powder. Stablecoin supply growth overall = more capital entering crypto ecosystem.

**Parameters:**
- Data: on-chain stablecoin supply from Glassnode, CryptoQuant, or stablecoin issuer reports
- Signal: large spike in exchange stablecoin inflows (not price moves) → bullish 1–3 weeks forward
- Signal: stablecoin total market cap growing >10% month-over-month = bullish macro crypto environment

**Documented performance:** No clean isolated backtest with win rate. Research confirms stablecoin inflows provide "early indications" of bullish moves [Mudrex]. CNN-LSTM models incorporating on-chain signals achieved 82.03% accuracy for next-day BTC direction [ScienceDirect 2025].

**Best timeframe:** Macro/weekly — this is a leading indicator, not a timing tool. Signals lead by 1–4 weeks.

**OHLCV-only?** No — requires on-chain or aggregated stablecoin supply data. Glassnode provides free tier. CryptoQuant has limited free access.

**MIDGE relevance:** Not currently implemented. Lower priority than funding rates (harder data access). Could be approximated by monitoring USDT market cap via CoinGecko.

---

### 2.5 Exchange Flow (Supply Shock Signal)

**What it measures:** Net Bitcoin flowing out of exchanges to cold storage. Sustained outflows = supply shock = bullish. Large inflows = selling intent = bearish.

**Parameters:**
- Data: Glassnode, CryptoQuant exchange netflow (requires subscription for real-time)
- Bullish signal: exchange reserves declining steadily over 30+ days
- Bearish signal: large single-day inflow spike (whale preparing to sell)
- Example: BTC exchange reserves declined from 3.4M (2022) to <2.5M (April 2025) = sustained bullish supply pressure [search results]

**Documented performance:** Directional accuracy without clean win rates in published backtests. CNN-LSTM models using exchange flow as a feature: 82.03% directional accuracy [ScienceDirect 2025].

**Best timeframe:** Macro (weekly/monthly). A single day of inflow does not override the macro trend.

**OHLCV-only?** No — requires dedicated on-chain data feed. Free tiers on Glassnode are limited.

**MIDGE relevance:** Not currently implemented. Lower priority without a clean free data source. Note: CryptoQuant has a free tier that includes some exchange flow data.

---

## Category 3: Pattern-Based Strategies

### Source Data Note

Win rates below come from a professional trader study of 10 patterns across 5 markets (forex, futures, equities, crypto, bonds) over 22 months on the **daily timeframe** [TradingView analysis by TheGoldDoctor]. Results vary significantly on shorter timeframes. These are among the most cited numbers in the literature.

### 3.1 Reversal Patterns

| Pattern | Win Rate | Direction | Key Confirmation |
|---------|----------|-----------|-----------------|
| Inverted Head & Shoulders | **83.44%** | Bullish | Neckline break + volume spike |
| Head & Shoulders | **83.04%** | Bearish | Neckline break + volume confirms |
| Triple Bottom | **79.33%** | Bullish | Third bounce + volume on breakout |
| Triple Top | **77.59%** | Bearish | Triple rejection + declining volume at tops |
| Double Bottom | **78.55%** | Bullish | Second bounce higher than first or equal |
| Double Top | **75.01%** | Bearish | Second top lower or equal, neckline break |

**Key caveats:**
- These rates are on DAILY timeframe across multiple markets — crypto alone may differ
- False breakouts are more frequent in crypto due to retail-driven liquidity and stop hunting
- Head & Shoulders has a 33% failure rate (pattern appears but doesn't confirm) — only count it after neckline breaks
- Volume confirmation is mandatory for all reversal patterns in crypto

**Parameters for all reversal patterns:**
- Minimum pattern formation: 15–30 candles (daily) for clean setup
- Volume should decline during pattern formation and spike on breakout
- Entry: candle close beyond neckline/breakout level (not on wick)
- Stop: below the lowest point of the pattern (H&S: below head; double bottom: below second bottom)
- Target: measured move = height of the pattern projected from the breakout level

**OHLCV-only?** Yes — price and volume only.

**Best timeframe:** Daily for highest documented win rates. 4H viable. Below 1H: significantly degraded.

---

### 3.2 Continuation Patterns

| Pattern | Win Rate | Direction | Key Confirmation |
|---------|----------|-----------|-----------------|
| Bearish Rectangle | **79.51%** | Bearish continuation | Volume on breakdown |
| Bullish Rectangle | **78.23%** | Bullish continuation | Volume on breakout |
| Ascending Channel | **73.03%** | Bullish bias (channel ride) | Price bouncing off lower trendline |
| Descending Triangle | **72.93%** | Bearish continuation | Flat bottom + declining highs |
| Ascending Triangle | **72.77%** | Bullish continuation | Flat top + rising lows |

**Parameters:**
- Minimum 2 touches on each trendline to draw valid pattern
- Triangle apex = breakout expected within the last 75% of the triangle
- Volume confirmation: breakout volume ≥ 1.5× average
- False breakout risk in crypto: high for triangles — always wait for candle close outside the triangle, not just wick break

---

### 3.3 Flag/Pennant Patterns

| Pattern | Win Rate | Direction | Key Confirmation |
|---------|----------|-----------|-----------------|
| Bear Flag | **67.72%** | Bearish continuation | Breakdown on volume |
| Bull Flag | **67.13%** | Bullish continuation | Breakout on volume |
| Bearish Pennant | **55.19%** | Bearish continuation | Similar to flag but converging trendlines |
| Bullish Pennant | **54.87%** | Bullish continuation | Higher volume on initial pole |

**Parameters:**
- Pole: strong momentum move (3–5 candles minimum)
- Flag: 3–7 candles of consolidation, slight counter-trend angle
- Entry: break above flag upper boundary (bull flag) or below lower boundary (bear flag)
- Target: measured move = pole height from flag breakout
- Pennants have lower win rates — the converging structure is noisier than flags

**Crypto note:** Bull flags are extremely common in crypto during bull runs. The 67% win rate drops if entered too early (before confirmed breakout).

---

### 3.4 Candlestick Patterns

**Backtested across 56,680 trades [LiberatedStockTrader]:**

| Pattern | Win Rate | Type |
|---------|----------|------|
| Inverted Hammer | **60%** | Bullish reversal |
| Gravestone Doji | **57%** | Bearish reversal |
| Bearish Marubozu | **56.1%** | Bearish continuation |
| Bearish Engulfing | **57%** | Bearish reversal (also acts bullish — context matters) |

**Additional findings:**
- Bearish Engulfing with volume confirmation on ES futures: **75.76% win rate** [TradesViz backtest]
- Three Outside Up (bullish reversal): confirmed on BTC in 2024 [Morpher]
- Bullish/Bearish Engulfing: most reliable candlestick signals — clear sentiment reversal
- Doji alone: near-random (indicates indecision, not direction)

**Critical caveats:**
- Candlestick patterns alone have slim profit margins — effective for maximum 10 days forward
- Require higher-timeframe trend context to filter out low-quality setups
- Crypto's volatility means individual candles are often manipulated — wicks hunt stops before actual reversal
- Best used as confirmation of a larger pattern, not as standalone entry

**OHLCV-only?** Yes — pure price data.

**Best timeframe:** Daily for highest accuracy. On hourly charts, noise degrades reliability significantly.

---

## Category 4: Mathematical / Chaos Approaches

### 4.1 Hurst Exponent

**What it measures:** A single number (0–1) describing the long-term memory of a time series:
- H < 0.5: anti-persistent (mean-reverting) — use mean-reversion strategies
- H = 0.5: random walk — no edge from trend or reversion
- H > 0.5: persistent (trending) — use trend-following strategies

**Key findings for crypto:**
- Bitcoin and Ethereum: Hurst exponent ≈ **0.32** (anti-persistent / mean-reverting) [2025 research, fractal Brownian motion studies]
- Fractal dimension ≈ **1.68**, Lévy index ≈ **1.22** — BTC does NOT conform to standard Brownian motion
- AI tokens (2025 study): Hurst > **0.58** (persistent, trending) — trend-following works better on newer AI-sector tokens
- BTC is gradually becoming MORE efficient (H approaching 0.5) as institutional adoption grows

**Practical trading rules:**
- Compute rolling Hurst over 60–120 periods using rescaled range (R/S) method or DFA (Detrended Fluctuation Analysis)
- H < 0.4: use mean-reversion strategies (RSI oversold/overbought, BB mean reversion)
- H > 0.6: use trend-following strategies (MA crossovers, breakouts)
- H between 0.4–0.6: no clear regime — reduce position size or avoid

**Documented performance:**
- Moving Hurst (MH) indicator on crypto: gains of ~**10%** by identifying optimal micro-trend positions [2025 research, MDPI]
- Fractal-based arbitrage on CSI 300 futures: **12.71% return vs 7.06% traditional**, Sharpe 0.32 vs -0.61 [harbourfronts.com 2025]

**OHLCV-only?** Yes — computed from price series alone (close prices).

**Best timeframe:** Rolling 60-period Hurst on daily bars for regime detection. Update every 5–10 days.

**MIDGE relevance:** Not currently implemented. High value for MIDGE's regime classifier. Would complement the existing `regime_classifier.py`. When Hurst signals mean-reversion regime, weight RSI/BB signals higher; when trending, weight MA crossover/breakout signals higher. The Hurst could be a dynamic weight modifier in the Thompson sampler's domain weights.

---

### 4.2 Fractal Dimension Analysis

**What it measures:** Related to Hurst (D = 2 - H). Measures the "roughness" or complexity of the price series. High fractal dimension = choppy/noisy. Low fractal dimension = smooth/trending.

**Parameters:**
- BTC fractal dimension: ~**1.68** (confirmed 2025 research) — rough, anti-persistent
- Trending smooth market: D ≈ 1.2–1.4
- Choppy noisy market: D ≈ 1.6–1.9

**Trading application:**
- When D < 1.4 (smooth trend): enter momentum/breakout strategies
- When D > 1.6 (rough/choppy): enter mean-reversion strategies, reduce breakout trades
- Fractal Market Hypothesis: different investor horizons create fractal price structure — liquidity comes from temporal diversity of participants

**OHLCV-only?** Yes — computed from price series.

**MIDGE relevance:** Directly useful as a regime detection signal feeding into domain weight adjustments. Can be computed alongside Hurst since D = 2 - H.

---

### 4.3 Power Law Distribution Fitting (Bitcoin)

**What it measures:** Bitcoin's long-term price follows a log-log power law: `Price ≈ 10^(-17) × (days since genesis)^5.8`. Deviations from the power law fair value identify macro over/undervaluation.

**Parameters:**
- Creator: Giovanni Santostasi (astrophysicist)
- Formula: `log(Price) = 5.8 × log(days_since_genesis) - 17`
- Upper band (97.5th percentile): macro overvalued — cycle top likely
- Lower band (2.5th percentile): macro undervalued — cycle bottom likely
- 67% of observations fall within ±1 standard deviation of the log-linear trend

**Documented performance:** Model has described every BTC cycle since genesis. The 2022 bear market bottom touched the lower power law band. However:
- The model gives NO timing signal — only macro over/undervaluation
- Bitcoin can break outside model bands (model is a framework, not a law)
- Best used as a sizing guide: large positions near lower band, reduce near upper band

**OHLCV-only?** Yes — requires only price and date since genesis block (2009-01-03).

**Best timeframe:** Monthly/quarterly. Not a trading signal — a macro positioning framework.

**MIDGE relevance:** Useful for MIDGE's macro context layer. Could provide a "power law deviation score" (how far current price is from the fair value band) to modulate risk appetite in the convergence engine.

---

### 4.4 Entropy-Based Regime Detection

**What it measures:** Shannon entropy or permutation entropy applied to price return series. High entropy = random/unpredictable market. Low entropy = structured/predictable regime.

**Parameters:**
- Permutation entropy (PE): computed over sliding window of 5–7 elements
- High PE (> 0.9): market is near-random — reduce position sizes
- Low PE (< 0.7): market shows structure — strategies have edge
- Approximate entropy (ApEn): higher values = more random; lower = more predictable

**Research findings:**
- Entropy-based regime detection: **87% accuracy** in identifying regime shifts (trending vs mean-reverting) [Preprints.org 2025]
- Entropy-enhanced LSTM vs standard LSTM: reduced max drawdown by **21%** and maintained more stable equity curve [multiple sources]
- Applied to 176 cryptocurrencies 2015–2024: entropy successfully differentiates randomness levels across different coins [MDPI 2025]

**OHLCV-only?** Yes — computed from price returns alone.

**Best timeframe:** Rolling 30–60 period calculation on daily bars for regime detection.

**MIDGE relevance:** High value. Entropy could serve as a "confidence gate" in MIDGE's convergence engine — when entropy is high (near-random market), reduce confidence scores across all technical signals by 20–30%. When entropy is low, boost technical signal confidence. A direct complement to Hurst exponent.

---

### 4.5 Correlation Breakdown Detection

**What it measures:** When historically correlated assets decouple (correlation drops sharply), it often precedes a large move in one or both assets. Breakdown in BTC/altcoin correlations, or BTC/S&P500 correlations, signals structural market change.

**Parameters:**
- Rolling correlation window: 20–30 days
- Correlation breakdown threshold: correlation drops from > 0.7 to < 0.3 within 5 days
- Cross-market: BTC vs S&P500, BTC vs Gold, BTC vs DXY (US Dollar Index)
- DXY strongly negatively correlated with BTC historically — DXY rally = BTC headwind

**Documented performance:** No isolated win rate backtest found. Academic research confirms correlation instability in crypto as a distinctive characteristic. CNN-LSTM incorporating correlation features: 82.03% accuracy [ScienceDirect 2025].

**OHLCV-only?** Requires multi-asset price data — but all from OHLCV sources (yfinance covers SPY, GLD, DXY).

**MIDGE relevance:** Partially implemented via `correlation_tracker.py`. The "breakdown" event (sharp drop in rolling correlation) is not currently tracked as a distinct signal. Add: when BTC/SPY 30-day correlation drops from > 0.6 to < 0.2 within 10 days → signal that crypto is decoupling (could be diverging up or down — combine with direction indicator).

---

## Summary: MIDGE Implementation Priority

### Already Implemented (verify parameters match research)
| Strategy | File | Recommended Parameter Check |
|----------|------|----------------------------|
| RSI | `ta_indicators.py` | Use 20/80 thresholds for crypto alerts, not 30/70 |
| MACD | `ta_indicators.py` | Prioritize histogram divergence over signal line cross |
| Bollinger Bands | `ta_indicators.py` | Add ±2.5 SD variant for crypto; add BBW squeeze metric |
| Regime Classifier | `regime_classifier.py` | Could be augmented with Hurst + entropy inputs |
| Correlation Tracker | `correlation_tracker.py` | Add correlation breakdown event as distinct signal |
| Fear & Greed | `convergence_alerter.py` | Extreme threshold must be < 20 for buy signal, not just "fear" |

### High Priority Gaps (high signal value, feasible data access)
| Strategy | Signal Type | Data Source | Estimated Implementation Size |
|----------|-------------|-------------|-------------------------------|
| Funding Rate Mean Reversion | crypto-specific | Bybit/Binance free API | ~200 lines |
| Hurst Exponent Regime | mathematical | OHLCV only | ~150 lines |
| Permutation Entropy Regime | mathematical | OHLCV only | ~100 lines |
| EMA 9/21 crossover + 50/200 Golden Cross | TA | OHLCV only | ~50 lines (extend existing) |
| Fibonacci 61.8% confluence zones | TA | OHLCV only | ~150 lines |
| BTC Dominance rotation | crypto-specific | CoinGecko (already integrated) | ~100 lines |

### Lower Priority (valuable but harder data access)
| Strategy | Blocker |
|----------|---------|
| Exchange Flow (BTC leaving exchanges) | Requires Glassnode paid tier or CryptoQuant |
| Stablecoin Supply Growth | Approximable via CoinGecko market cap but imprecise |
| VWAP (intraday) | Requires tick/minute data; less useful for swing |
| Power Law deviation score | Useful for macro positioning only — low trading frequency |

---

## Key Cross-Strategy Findings for MIDGE

1. **Combination dramatically outperforms solo signals.** MACD + RSI: 65–77% vs either alone at 50–55%. BB + RSI: similar improvement. MIDGE's convergence engine is architecturally aligned with this finding — the stacking of signals is the correct approach.

2. **Daily timeframe has the most validated backtest data.** Most pattern win rates (75–83%) are measured on daily candles. Sub-1H signals are significantly noisier. MIDGE should weight daily-timeframe signals higher than hourly in convergence scoring.

3. **Volume confirmation is the most universal filter.** Across patterns, indicators, and strategies — when volume confirms a signal, win rate increases significantly. Signals without volume confirmation should receive a confidence penalty.

4. **Crypto needs wider parameters than equities.** RSI 20/80 instead of 30/70. BB ±2.5 SD instead of ±2 SD. EMA periods 9/21 instead of 10/20. Standard equity parameters generate too many false signals in crypto's high-volatility environment.

5. **Hurst exponent ≈ 0.32 for BTC means mean-reversion strategies have a structural edge.** BTC's anti-persistent nature (H < 0.5) is well-documented across multiple 2025 studies. RSI and Bollinger Band mean-reversion strategies should theoretically outperform trend-following in BTC on shorter timeframes.

6. **Pattern win rates decline sharply below daily timeframe.** The 75–83% documented pattern win rates apply ONLY to daily charts. On 4H, expect 60–70%. On 1H, 50–60%. Below 1H, near-random.

7. **Funding rates are the most crypto-native high-signal metric.** They measure actual market positioning (leveraged longs vs shorts) in real-time. Extreme funding rates predict short-term price reversals with documented edge. This is the highest-priority gap in MIDGE's current signal library.

---

## Sources

- [RSI Trading Strategy (91% Win Rate) — QuantifiedStrategies.com](https://www.quantifiedstrategies.com/rsi-trading-strategy/)
- [MACD Trading Strategy: Statistics, Facts And Historical Backtests — QuantifiedStrategies.com](https://www.quantifiedstrategies.com/macd-trading-strategy/)
- [Bollinger Bands Trading Strategies: Backtest and Performance — QuantifiedStrategies.com](https://www.quantifiedstrategies.com/bollinger-bands-trading-strategy/)
- [Bitcoin Bollinger Bands Trading Strategy — QuantifiedStrategies.com](https://www.quantifiedstrategies.com/bitcoin-bollinger-bands-trading-strategy-performance-backtest/)
- [Success Rate of Popular Patterns for BTC/USD — TradingView/TheGoldDoctor](https://www.tradingview.com/chart/BTCUSD/bdDsKTi4-Success-Rate-of-Popular-Patterns/)
- [Mean Reversion Strategy in Crypto Rate Trading — rho.trading](https://www.rho.trading/blog/mean-reversion-strategy-in-crypto-rate-trading)
- [Optimisation of Cryptocurrency Trading Using the Fractal Market Hypothesis — MDPI 2025](https://www.mdpi.com/2813-2432/4/4/22)
- [Fractal Market Hypothesis: From Theory to Practice — harbourfronts.com 2025](https://blog.harbourfronts.com/2025/11/17/fractal-market-hypothesis-from-theory-to-practice/)
- [Fractional and fractal processes applied to cryptocurrencies — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8408330/)
- [Bitcoin price direction prediction using on-chain data — ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S266682702500057X)
- [Key On-Chain Indicators Every Crypto Trader Should Know — Mudrex](https://mudrex.com/learn/top-on-chain-indicators/)
- [How A Bitcoin Fear And Greed Index Trading Strategy Beats Buy And Hold — Nasdaq/BitcoinMagazine](https://www.nasdaq.com/articles/how-bitcoin-fear-and-greed-index-trading-strategy-beats-buy-and-hold-investing)
- [Crypto Fear and Greed Index Trading Strategy — spotedcrypto.com](https://www.spotedcrypto.com/crypto-fear-greed-index-guide/)
- [The 10 Best Candle Patterns Proven With 56,680 Trades — LiberatedStockTrader](https://www.liberatedstocktrader.com/candle-patterns-reliable-profitable/)
- [Altcoin Season 2025: How to Time BTC Dominance — Millionero](https://blog.millionero.com/blog/altcoin-season-2025-how-to-time-btc-dominance-for-max-gains/)
- [Optimizing Algorithmic Trading with Entropy-Based Decision Making — Preprints.org 2025](https://www.preprints.org/manuscript/202502.1717)
- [Entropy-Based Portfolio Optimization in Cryptocurrency Markets — MDPI](https://www.mdpi.com/1099-4300/28/3/285)
- [EMA Crossover Strategy Quantitative Study — Medium/Superalgos](https://medium.com/superalgos/quantitative-study-of-the-ema-cross-trading-strategy-29d5ed655a4)
- [Bitcoin Power Law — Bitcoin Magazine Pro](https://www.bitcoinmagazinepro.com/charts/bitcoin-power-law/)
- [Fibonacci Retracement in Crypto: 2026 Complete Trading Guide — cryptoprofitcalc.com](https://cryptoprofitcalc.com/fibonacci-retracement-in-crypto-levels-strategies-and-risk-management/)
- [MACD vs RSI for Crypto — altrady.com](https://www.altrady.com/blog/crypto-trading-strategies/macd-trading-strategy-macd-vs-rsi)
- [VWAP in Crypto 2025 — Mudrex](https://mudrex.com/learn/vwap-in-crypto/)
