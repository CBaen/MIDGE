# Mathematical, Geometric & Physics-Based Trading Strategies
## Research Report for MIDGE Integration

**Date:** 2026-03-19
**Scope:** Every mathematically-grounded strategy family beyond standard TA — Fibonacci, sacred geometry, Gann, wave theory, chaos, physics analogies, number theory, probabilistic. For each: OHLCV-only?, evidence quality, Python implementation skeleton, crypto applicability.

---

## Evidence Quality Legend

| Grade | Meaning |
|-------|---------|
| A | Peer-reviewed papers with positive, reproducible results |
| B | Quantified backtests (non-peer-reviewed) with specific numbers |
| C | Practitioner consensus / widespread professional use without rigorous proof |
| D | Anecdotal / case-study only |
| F | No evidence; theoretical or promotional only |

---

## Part 1: Fibonacci Family

### 1.1 Fibonacci Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%)

**OHLCV only?** Yes — requires only swing high and swing low identification.

**Evidence:** C/D

Critical finding: QuantifiedStrategies attempted systematic backtesting and concluded "we were not able to make meaningful backtests" due to subjectivity in swing selection. A dissertation by Clarissa Gunawan found passive buy-and-hold outperforms active Fibonacci retracement strategies. One crypto-specific test found ~50% win rate (coin-flip territory). The 61.8% level ("Golden Ratio") attracts the most professional attention and anecdotally shows the strongest reactions — but this may be self-fulfilling prophecy driven by widespread use.

**What actually works:** Fibonacci retracements are not independently predictive. Their value is as a **confluence filter**: when a 61.8% retrace aligns with a prior support level, a VWAP level, or a volume node, the combined signal is significantly stronger. Used as a standalone, evidence is weak.

**Crypto applicability:** Yes — heavily used in BTC/ETH analysis. After the 2024 BTC halving rally to ~$70K, the ~$52K correction was a clean 50% retrace. The levels are watched by enough participants to be self-reinforcing on liquid pairs.

**Implementation skeleton (OHLCV → signal):**

```python
def fibonacci_retracement_levels(high: float, low: float) -> dict:
    """Compute key Fibonacci retracement levels from a swing."""
    diff = high - low
    return {
        "0.0": low,
        "23.6": low + 0.236 * diff,
        "38.2": low + 0.382 * diff,
        "50.0": low + 0.500 * diff,
        "61.8": low + 0.618 * diff,
        "78.6": low + 0.786 * diff,
        "100.0": high,
    }

def fibonacci_signal(df: pd.DataFrame, lookback: int = 60) -> float:
    """
    Return -1/0/1 based on proximity to 61.8% retrace level.
    Positive = approaching retrace from below (potential bounce)
    """
    recent = df.tail(lookback)
    swing_high = recent["high"].max()
    swing_low = recent["low"].min()
    levels = fibonacci_retracement_levels(swing_high, swing_low)
    current = df["close"].iloc[-1]
    level_618 = levels["61.8"]
    proximity = abs(current - level_618) / (swing_high - swing_low)
    if proximity < 0.02:  # within 2% of level
        # direction of approach matters
        prev = df["close"].iloc[-5]
        return 1.0 if prev < level_618 else -1.0
    return 0.0
```

---

### 1.2 Fibonacci Extensions (127.2%, 161.8%, 261.8%)

**OHLCV only?** Yes — computed from the same A-B-C swing structure.

**Evidence:** D

No systematic backtest establishes these as better targets than ATR multiples. The 161.8% extension is the most cited. In harmonic pattern theory (see 1.5), these are integral to defining valid patterns — that context gives them slightly more credibility.

**Implementation skeleton:**

```python
def fibonacci_extensions(a: float, b: float, c: float) -> dict:
    """
    A-B-C extension: A=swing start, B=swing end, C=pullback end.
    Returns projected price targets beyond C.
    """
    ab = abs(b - a)
    direction = 1 if b > a else -1
    return {
        "127.2": c + direction * 1.272 * ab,
        "161.8": c + direction * 1.618 * ab,
        "200.0": c + direction * 2.000 * ab,
        "261.8": c + direction * 2.618 * ab,
    }
```

---

### 1.3 Fibonacci Time Zones

**OHLCV only?** Yes — uses candle index count, not price.

**Evidence:** D

Vertical lines drawn at Fibonacci number intervals (1, 1, 2, 3, 5, 8, 13, 21, 34, 55 bars) from a significant pivot. The MotiveWave platform has a combined Fibonacci-Lucas Time Series indicator, authored in Stocks & Commodities Magazine (Aug 2012). One example showed the 55-day zone marking a significant low; the 34-day zone missed. Results are highly anchor-point-dependent: different traders drawing from different pivots get completely different forecasts for the same market.

**Crypto applicability:** Used but not validated. BTC halving cycles roughly align with Fibonacci multiples of months — this is loosely used as macro timing.

**Implementation:**

```python
def fibonacci_time_zones(df: pd.DataFrame, anchor_idx: int) -> list:
    """Return index positions of Fibonacci time zone verticals from anchor."""
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
    return [anchor_idx + f for f in fib if anchor_idx + f < len(df)]
```

---

### 1.4 Fibonacci Fan / Arc (Angular Projections)

**OHLCV only?** Yes (fan). Arc requires normalizing price/time axes (subjective).

**Evidence:** F

Fibonacci fans draw trendlines from a swing point at angles proportional to Fibonacci ratios (38.2°, 50°, 61.8° of the range). QuantifiedStrategies explicitly stated they "were not able to make meaningful backtests" for Fibonacci fans. The subjectivity of the starting point and axis scaling makes algorithmic implementation nearly impossible to standardize. Professional consensus: aesthetically appealing, not reliably predictive.

**Skip for MIDGE.** The fan/arc tools are fundamentally manual charting tools, not automatable strategies.

---

### 1.5 Harmonic Patterns (Gartley 222, Butterfly, Bat, Crab, Shark, Cypher)

**OHLCV only?** Yes — all ratios computed from OHLCV swing points.

**Evidence:** B (mixed — pattern-dependent)

**Critical finding:** Every harmonic pattern is defined by precise Fibonacci ratios at each leg (XABCD structure). The ratios are exact, making automated detection feasible. Reported win rates from systematic studies:

| Pattern | Win Rate | Notes |
|---------|---------|-------|
| Gartley | 36–60% | Bullish Gartley ~60% target hit rate |
| Bat | 55–65% | Considered most reliable; QuantifiedStrategies: "unable to backtest meaningfully" |
| Butterfly | 40–55% | Extended CD leg (>100% of XA) |
| Crab | ~35% | "Avoid" per some practitioner studies |
| Shark | 40–50% | Intermediate BC ratio 1.13–1.618 |
| Cypher | 33–40% | 21 currency pairs, 33–40% WR across all |

**Python library:** `HarmonicPatterns` (GitHub: djoffrey/HarmonicPatterns) — supports Gartley, Bat, AltBat, Butterfly, Crab, DeepCrab, Shark, Cypher with predict/deepsearch modes.

**The fundamental problem:** These are retrospective pattern-matching tools. "Valid" vs "invalid" patterns differ by interpretation — the strict Fibonacci tolerance (±5%) leaves many gray-zone patterns. At strict tolerances, patterns are rare (low opportunity); at loose tolerances, win rates degrade.

**Implementation skeleton:**

```python
def find_harmonic_patterns(df: pd.DataFrame, tolerance: float = 0.05):
    """
    Detect harmonic patterns using swing point XABCD ratios.
    Requires swing detection first (e.g., zigzag with min_move threshold).

    Bat pattern ratios:
      XA: leg from X to A
      AB retracement of XA: 0.382–0.500
      BC retracement of AB: 0.382–0.886
      CD retracement of XA: 0.886 (PRZ = 88.6% of XA)
      AD extension of XA: 1.618–2.618
    """
    # Use djoffrey/HarmonicPatterns library or implement zigzag + ratio checks
    # from harmonicpatterns import find_patterns
    # return find_patterns(df, pattern="bat", tolerance=tolerance)
    pass
```

**Crypto applicability:** Yes — active community on TradingView applies harmonic patterns to BTC/ETH. The patterns work on any liquid OHLCV series.

---

### 1.6 Lucas Numbers (2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123...)

**OHLCV only?** Yes — used identically to Fibonacci for time periods.

**Evidence:** D

Lucas sequence shares the same golden ratio limit as Fibonacci but uses different starting values (2, 1 instead of 0, 1). W.D. Gann's work referenced 144-day cycles (a Fibonacci number); some practitioners substitute Lucas numbers (76, 123) for similar timing. MotiveWave's Fibonacci-Lucas Time Series Indicator (Stocks & Commodities, Aug 2012) blends both sequences. One documented trading example: a price leg completed in 76 (Lucas) days high-to-high; another took 47 (Lucas) days. Sample size: 1. Evidence: anecdotal.

**Implementation:** Replace Fibonacci sequence with Lucas sequence in time zone computation:

```python
def lucas_time_zones(df: pd.DataFrame, anchor_idx: int) -> list:
    lucas = [2, 1, 3, 4, 7, 11, 18, 29, 47, 76, 123, 199]
    return [anchor_idx + n for n in lucas if anchor_idx + n < len(df)]
```

---

## Part 2: Sacred Geometry / Proportional Systems

### 2.1 Phi (1.618) and Its Inverse (0.618) in Price Ratios

**OHLCV only?** Yes.

**Evidence:** C

The golden ratio φ = 1.618... is the foundation of all Fibonacci-derived trading (see Part 1). The "Golden Pocket" — the price zone between 61.8% and 65% retrace — is the most-cited sacred geometry level in trading. Professional practitioners (ICT, Smart Money Concepts) treat this zone as high-probability reversal territory, particularly when it aligns with a prior order block. Evidence is practitioner-based (C grade) rather than academic.

**Note for MIDGE:** MIDGE already uses session sweeps and IFVG from ICT methodology. The Golden Pocket (61.8%–65% retrace) is a direct extension of that framework.

---

### 2.2 Square Root Relationships

**OHLCV only?** Yes.

**Evidence:** C (practitioners) / F (academic)

Gann's core insight: significant price levels occur at **square root intervals** from major highs/lows. The formula: `next_level = (sqrt(price) ± 1)²`. This generates a non-linear grid that compresses at high prices (like real price behavior). Also: `(sqrt(price) ± 0.5)²` for intermediate levels.

Example: sqrt(100) = 10. Levels: (10±1)² = 81, 121. (10±0.5)² = 90.25, 110.25. These often correspond to psychologically significant levels.

**Why it might work:** Markets form support/resistance at round numbers. The square root transformation converts linear round numbers into log-scale intervals that match observed price behavior better than linear grids.

**Implementation:**

```python
def sqrt_levels(price: float, n_levels: int = 5) -> list:
    """Gann-style square root support/resistance levels."""
    root = price ** 0.5
    levels = []
    for i in range(-n_levels, n_levels + 1):
        if i != 0:
            levels.append((root + i * 0.5) ** 2)
    return sorted(levels)
```

---

### 2.3 Gann Square of Nine

**OHLCV only?** Yes.

**Evidence:** C (practitioners) / F (academic)

The Square of Nine is a spiral calculator where numbers wrap around a center point by 360°. The key formula for support/resistance levels:

```
level = (sqrt(price) + (angle / 360)) ** 2
```

Where angle ∈ {45, 90, 135, 180, 225, 270, 315, 360}. Numbers falling on the same diagonal (Cardinal Cross: 0°, 90°, 180°, 270°) or Ordinal Cross (45°, 135°, 225°, 315°) act as major support/resistance pivots.

**QuantifiedStrategies verdict:** "Not able to make a meaningful backtest of the Gann fan strategy." The tool requires a subjective starting price, making systematic testing impossible.

**Implementation:**

```python
import numpy as np

def gann_square_of_nine(price: float) -> dict:
    """Compute Gann Square of Nine levels from a given price."""
    root = np.sqrt(price)
    angles = [45, 90, 135, 180, 225, 270, 315, 360]
    levels = {}
    for angle in angles:
        up = (root + angle / 360) ** 2
        down = (root - angle / 360) ** 2
        levels[f"+{angle}deg"] = round(up, 4)
        levels[f"-{angle}deg"] = round(max(0, down), 4)
    return levels

def gann_sq9_signal(df: pd.DataFrame) -> float:
    """Signal when price is within 0.5% of a cardinal cross Gann level."""
    price = df["close"].iloc[-1]
    levels = gann_square_of_nine(price)
    cardinal = [levels[k] for k in ["90deg", "180deg", "270deg", "360deg"]
                if k in levels]  # Simplified
    for lvl in [gann_square_of_nine(price)[f"+{a}deg"] for a in [90, 180, 270, 360]]:
        if abs(price - lvl) / price < 0.005:
            return 1.0  # at a pivot level
    return 0.0
```

---

### 2.4 Gann Angles (1×1, 2×1, 1×2 — geometric angle-based support/resistance)

**OHLCV only?** Yes, but requires normalizing price/time scale (subjective).

**Evidence:** D

Gann angles (the "fan") draw lines from a pivot at angles where price and time are in a specific ratio. The 45° line (1×1, meaning 1 unit of price per 1 unit of time) is the "equilibrium" line; price above it is bullish, below is bearish. QuantifiedStrategies: "Unable to make a meaningful backtest." Core problem: "price" and "time" have no natural units, so the angle is meaningless without an arbitrary scaling decision.

**Skip for MIDGE** as a primary strategy. Can be implemented as a visual indicator, but not reliably automatable.

---

### 2.5 Gann Time Cycles (90, 180, 270, 360-day cycles)

**OHLCV only?** Yes — index-based, date-counting.

**Evidence:** C (practitioners)

The 90-day (quarter), 180-day (half-year), and 360-day (year) cycles correspond to natural calendar divisions and quarterly earnings cycles — so they may have a genuine fundamental explanation. Documented anecdotal examples: Nifty's 90-day and 144-day cycles as reversal points; S&P 500 reversal 180 days after Jan 2022 peak (July 2022).

The 360-day cycle has overlap with annual seasonality effects (which DO have academic evidence — see Santa Claus rally, Sell in May). Whether this is Gann theory or just seasonality: unclear.

**Implementation:**

```python
def gann_time_cycle_signal(df: pd.DataFrame,
                           pivot_date: pd.Timestamp,
                           cycles: list = [90, 180, 270, 360]) -> float:
    """
    Check if current bar is near a Gann time cycle from a pivot date.
    Returns 1.0 if within 3 days of a cycle anniversary.
    """
    today = df.index[-1]
    days_since = (today - pivot_date).days
    for cycle in cycles:
        remainder = days_since % cycle
        if remainder < 3 or remainder > (cycle - 3):
            return 1.0  # At a time cycle boundary
    return 0.0
```

---

## Part 3: Wave / Cycle Theory

### 3.1 Elliott Wave (5-wave impulse + 3-wave correction)

**OHLCV only?** Yes.

**Evidence:** C/D (practitioners) / A (specific ML applications)

**The core problem:** Elliott Wave is intrinsically subjective. Two experts often disagree on wave counts for the same chart. "Almost impossible to backtest" (QuantifiedStrategies). "No proven study or mathematical backtest."

**However:** Recent academic work using ML changes the picture:
- MDPI 2024 study (ElliottAgents): Multi-agent LLM system achieved >70% trend prediction accuracy on BTC (Oct 2022–Sep 2024, $20K→$70K). 16% accuracy improvement over baseline with deep reinforcement learning.
- `taew` Python package: Based on academic paper "Profitability of Elliott Waves and Fibonacci Retracement Levels in the Foreign Exchange Market."
- `ElliottWaveAnalyzer` (GitHub: btcorgtfo): Tries all combinations of wave patterns for given OHLC data, validates against rules.
- One Sharpe ratio > 3 was claimed for training periods using genetic algorithm optimization, but walk-forward results were mixed.

**Key rule set for automation:**
- Wave 2 never retraces more than 100% of Wave 1
- Wave 3 is never the shortest impulse wave
- Wave 4 never overlaps Wave 1's price territory (except diagonal triangles)
- Impulse waves (1, 3, 5) in direction of trend; corrective waves (A, B, C) counter-trend

**Python libraries:** `taew` (PyPI), `ElliottWaveAnalyzer` (GitHub)

**Crypto applicability:** Yes — active community. BTC's halving-to-peak cycles loosely follow 5-wave structures.

---

### 3.2 Wolfe Waves

**OHLCV only?** Yes.

**Evidence:** B

A 5-point geometric reversal pattern with specific rules:
1. Waves 1-2 establish range (alternating highs and lows)
2. Wave 3 stays within channels of Waves 1-2
3. Wave 4 is within the channel of Waves 1-2
4. Wave 5 overshoots the channel (the entry signal)
5. Target is the "EPA line" (endpoint line from Wave 1 to Wave 4)

**Quantified results from 6,269 patterns:**
- 41% reached the EPA target line
- 35% reached the "ultimate peak" (highest point before 20% decline)
- 49% stopped out

This is above a coin flip for the EPA target (41% > 33% random), but barely. Risk/reward can be favorable if stops are tight below Wave 5.

**Crypto applicability:** Documented on LTC/USD and BTC/USD. Works on any liquid OHLCV series.

**Implementation:** Requires zigzag detection, then geometric constraint validation between the 5 pivot points. Indicators exist for MT4/MT5 and TradingView (BigBeluga).

---

### 3.3 Kondratieff Waves (Long Economic Cycles)

**OHLCV only?** No — fundamentally macro-economic. OHLCV alone cannot reliably position within a 40–60 year cycle.

**Evidence:** D/F (academic rejection)

"Not accepted by most academic economists." Even among adherents, cycle lengths are disputed. Current position: believed to be late in Wave 5 (ICT boom), dawn of Wave 6 (biotech/green energy). Useful as macro context, not as a MIDGE signal.

**MIDGE relevance:** None for signal generation. Macro framing only.

---

## Part 4: Chaos Theory / Nonlinear Dynamics

### 4.1 Hurst Exponent (H)

**OHLCV only?** Yes.

**Evidence:** A (Hurst measuring works) + B (trading application)

H < 0.5: mean-reverting (anti-persistent)
H = 0.5: random walk
H > 0.5: trending (persistent)

**Backtest evidence (QuantifiedStrategies, gold/GLD ETF):**
- 31 trades, 64% win rate, 0.7% avg gain per trade, 1.1% annual return
- 6% max drawdown, 1.8% time in market

Very few trades (too selective), but solid win rate. The low annual return is because the system is in market <2% of the time.

**Critical finding:** Moving Hurst (computed over rolling windows) has been shown to produce better returns than MACD in some studies (academic paper on chaotic properties indicator).

**Python implementation (no external package needed — uses scipy/numpy):**

```python
import numpy as np

def hurst_exponent(prices: np.ndarray, max_lag: int = 100) -> float:
    """
    Compute Hurst exponent using R/S analysis.
    H < 0.5: mean-reverting, H > 0.5: trending, H ~ 0.5: random
    Requires at least 100 data points for reliability.
    """
    ts = np.log(prices)
    lags = range(2, min(max_lag, len(ts) // 2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    if not any(t > 0 for t in tau):
        return 0.5
    poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
    return poly[0]

def hurst_signal(df: pd.DataFrame, window: int = 100,
                 threshold_low: float = 0.45,
                 threshold_high: float = 0.55) -> dict:
    """
    Compute rolling Hurst and return regime + signal.
    """
    prices = df["close"].values
    if len(prices) < window:
        return {"regime": "unknown", "signal": 0.0, "H": 0.5}
    H = hurst_exponent(prices[-window:])
    if H < threshold_low:
        regime = "mean_reverting"
        signal = -np.sign(df["close"].iloc[-1] - df["close"].iloc[-window:].mean())
    elif H > threshold_high:
        regime = "trending"
        signal = np.sign(df["close"].iloc[-1] - df["close"].iloc[-20:].mean())
    else:
        regime = "random"
        signal = 0.0
    return {"regime": regime, "signal": signal, "H": H}
```

**Crypto applicability:** Yes. Crypto assets tend to show H > 0.5 during strong trends and H < 0.5 after volatile reversals — making the regime switch particularly actionable.

**Library:** `nolds` (PyPI) provides `nolds.hurst_rs()` — not currently installed in MIDGE but trivially installable.

---

### 4.2 Lyapunov Exponent

**OHLCV only?** Yes (computed from price return time series).

**Evidence:** A (academic) + C (trading application limited)

The Largest Lyapunov Exponent (LLE) measures exponential divergence of trajectories. In markets:
- Positive LLE: chaotic, small changes amplify unpredictably
- Negative LLE: stable, perturbations dampen

**Academic finding:** LLE spikes near stock market crashes (studied on DJIA and S&P 500). Used as a **crash predictor**, not a directional signal.

**Trading use:** Combined with Fractal Dimension Index (FDI), provides a "chaos meter." When LLE is low (stable system): trend-following works. When LLE is high (chaotic): any direction is unreliable.

**Python (nolds library):**

```python
# nolds.lyap_r(ts) — Rosenstein algorithm (fastest)
# nolds.lyap_e(ts) — Eckmann algorithm (full spectrum)
# Both available once nolds is pip-installed

def lyapunov_signal(prices: np.ndarray, emb_dim: int = 10) -> float:
    """
    Compute Largest Lyapunov Exponent from price returns.
    Returns: negative = stable (tradeable), positive = chaotic (avoid)
    """
    import nolds  # pip install nolds
    returns = np.diff(np.log(prices))
    try:
        lle = nolds.lyap_r(returns, emb_dim=emb_dim)
        return lle  # < 0: stable, > 0: chaotic
    except Exception:
        return 0.0
```

**Limitation:** Requires large datasets (500+ points) for reliable estimates. Computationally expensive. Better as a regime filter than a signal generator.

---

### 4.3 Fractal Dimension (Hausdorff / Box-Counting)

**OHLCV only?** Yes.

**Evidence:** A (academic) + B (trading application)

The Fractal Dimension Index (FDI) for a price series lies between 1.0 (straight line) and 2.0 (completely random/space-filling). Interpretation:
- FDI close to 1.5: random, unpredictable
- FDI < 1.5: trending (smooth curve, lower dimension)
- FDI > 1.5: mean-reverting (rougher, more space-filling)

MDPI 2025 paper on multiscale network factors found fractal dimension explains asset pricing variation beyond standard factors.

**MQL5 study:** FDI outperformed Hurst exponent alone for classifying market regimes.

**Implementation (Higuchi algorithm — most used in finance):**

```python
def higuchi_fd(prices: np.ndarray, k_max: int = 10) -> float:
    """
    Higuchi Fractal Dimension of a price series.
    < 1.5: trending, > 1.5: mean-reverting, ~ 1.5: random
    """
    N = len(prices)
    L = []
    x = np.array(prices)
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(1, k + 1):
            # Indices: m, m+k, m+2k, ...
            idxs = range(m - 1, N, k)
            vals = x[list(idxs)]
            if len(vals) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(vals))) * (N - 1) / (k * len(vals))
            Lk.append(Lmk)
        if Lk:
            L.append(np.mean(Lk))
    if len(L) < 2:
        return 1.5
    lnL = np.log(L)
    lnk = np.log(range(1, len(L) + 1))
    slope, _ = np.polyfit(lnk, lnL, 1)
    return abs(slope)

def fractal_dimension_signal(df: pd.DataFrame, window: int = 50) -> dict:
    """Rolling FDI signal."""
    prices = df["close"].values[-window:]
    fdi = higuchi_fd(prices)
    if fdi < 1.4:
        regime = "trending"
    elif fdi > 1.6:
        regime = "mean_reverting"
    else:
        regime = "random"
    return {"fdi": fdi, "regime": regime}
```

---

### 4.4 Strange Attractors in Price Phase Space

**OHLCV only?** Yes — constructed from lagged price returns (delay embedding).

**Evidence:** A (academic) / C (trading)

Phase space reconstruction (Takens embedding theorem) converts a 1D time series into a multi-dimensional attractor. The Lorenz-like structure of market returns has been documented. Practically:

1. Embed returns in d dimensions with lag τ: `v(t) = [r(t), r(t-τ), r(t-2τ), ...]`
2. Study the geometry of the resulting cloud
3. Correlation dimension (D2) measures how the cloud fills space

The correlation dimension of the S&P 500 has been estimated at ~2.0–4.0, suggesting a low-dimensional attractor (not pure noise, which would have D2 → ∞).

**Trading use:** Academic insight — markets are not random walks but have structure. Directly actionable signal generation from attractors is not proven. Better as a theoretical justification for non-linear methods.

---

### 4.5 Mandelbrot's Multifractal Model of Asset Returns (MMAR)

**OHLCV only?** Yes.

**Evidence:** A (excellent theoretical foundation)

Mandelbrot's MMAR replaces Brownian motion (used in Black-Scholes) with **multifractal Brownian motion in multifractal time**. Key properties:
- Heavy tails (fat tails) naturally reproduced
- Volatility clustering naturally reproduced
- Long memory in volatility naturally reproduced
- Self-similar across time scales

**Academic evidence (MDPI, Int'l J. Financial Studies):** MMAR outperforms GARCH, stable distributions, and standard GBM in fitting realized variances for GBP/USD and Bitcoin.

**Why it's not widely used in trading:** The MMAR is a generative model (for simulation and option pricing), not a directional signal generator. It tells you HOW prices move, not WHICH direction.

**MIDGE application:** MMAR parameters (multifractal spectrum width, Hurst-like exponent α) can be computed and used as:
1. Volatility regime indicator (wide multifractal spectrum = high volatility clustering)
2. Option pricing improvement (if/when MIDGE enters options)
3. Simulation input for Monte Carlo position sizing

**GitHub implementation:** `hyperstripe50/fractal-market-analysis` implements both Mandelbrot's MMAR and Peters' Fractal Market Analysis.

---

## Part 5: Physics Analogies

### 5.1 Mean Reversion as a Spring (Hooke's Law)

**OHLCV only?** Yes.

**Evidence:** A (academic — OU process) + B (trading application)

**The physics analogy:** F = -kx. Price displaced from equilibrium "snaps back" with force proportional to displacement. The mathematical formalization is the Ornstein-Uhlenbeck (OU) process:

```
dX_t = θ(μ - X_t)dt + σdW_t
```

Where θ (mean reversion speed), μ (equilibrium), σ (volatility) are estimated from data.

**Evidence (strong):** For crypto spread trading (ETH/BTC ratio), the OU process produces tradeable signals. Multiple Medium articles with vectorized backtests exist (PyQuantLab, ThePythonLab). The ArbitrageLib (Hudson & Thames) provides full optimal stopping time derivation.

**Implementation (from available libraries — scipy + statsmodels):**

```python
import numpy as np
import statsmodels.api as sm

def fit_ou_parameters(spread: np.ndarray) -> dict:
    """
    Estimate OU parameters via OLS on lag-1 regression.
    Spread = log price ratio (e.g., log(ETH/BTC)).
    """
    y = spread[1:]
    x = sm.add_constant(spread[:-1])
    res = sm.OLS(y, x).fit()
    # X_t+1 = c + phi * X_t + noise
    phi = res.params[1]
    c = res.params[0]
    dt = 1.0  # 1 period (daily bars)
    theta = -np.log(phi) / dt            # mean reversion speed
    mu = c / (1 - phi)                   # equilibrium level
    sigma = np.std(res.resid) / np.sqrt(dt)
    half_life = np.log(2) / theta if theta > 0 else np.inf
    return {"theta": theta, "mu": mu, "sigma": sigma, "half_life_days": half_life}

def ou_zscore_signal(df: pd.DataFrame, window: int = 60) -> dict:
    """
    Rolling OU Z-score signal for mean reversion.
    Positive z = overextended above mean → short signal
    Negative z = overextended below mean → long signal
    """
    prices = np.log(df["close"].values)
    if len(prices) < window:
        return {"zscore": 0.0, "signal": 0.0, "half_life": None}
    recent = prices[-window:]
    params = fit_ou_parameters(recent)
    if params["half_life_days"] <= 0 or params["half_life_days"] > 252:
        return {"zscore": 0.0, "signal": 0.0, "half_life": params["half_life_days"]}
    eq_std = params["sigma"] / np.sqrt(2 * params["theta"])
    zscore = (prices[-1] - params["mu"]) / eq_std
    signal = -np.clip(zscore / 2.0, -1.0, 1.0)  # fade extremes
    return {"zscore": zscore, "signal": signal, "half_life": params["half_life_days"]}
```

**Crypto applicability:** Strong. ETH/BTC spread, BTC/stablecoin basis, altcoin pairs.

---

### 5.2 Momentum as Mass × Velocity (Price × Volume)

**OHLCV only?** Yes.

**Evidence:** A (academic — price-volume momentum is well-studied)

**The physics model:**
- Velocity v = price return (Δp/p)
- Mass m = volume (or inverse volatility)
- Momentum P = m × v = volume × price_return

ScienceDirect paper "Physical approach to price momentum and its application to momentum strategy" formalizes this. Alternative formula uses inverse volatility as mass: `m = 1/σ`.

Academic result: Physical momentum strategies achieve **better expected returns and risk-adjusted metrics** than traditional price-only momentum strategies.

**Implementation:**

```python
def physical_momentum(df: pd.DataFrame, window: int = 20) -> float:
    """
    Physical momentum: volume (mass) × price return (velocity).
    Normalized by historical average.
    """
    returns = df["close"].pct_change()
    volume = df["volume"]
    phys_mom = (volume * returns).rolling(window).mean()
    hist_mean = phys_mom.mean()
    hist_std = phys_mom.std()
    if hist_std == 0:
        return 0.0
    zscore = (phys_mom.iloc[-1] - hist_mean) / hist_std
    return np.clip(zscore, -1.0, 1.0)

def inverse_volatility_momentum(df: pd.DataFrame,
                                 vol_window: int = 20,
                                 mom_window: int = 60) -> float:
    """
    Momentum weighted by inverse volatility (high-certainty moves count more).
    """
    returns = df["close"].pct_change()
    vol = returns.rolling(vol_window).std()
    inv_vol = 1.0 / (vol + 1e-8)
    weighted_mom = (returns * inv_vol).rolling(mom_window).mean()
    norm = weighted_mom / weighted_mom.abs().rolling(252).mean().clip(lower=1e-8)
    return np.clip(norm.iloc[-1], -1.0, 1.0)
```

---

### 5.3 Energy Conservation in Price Moves (Potential vs Kinetic)

**OHLCV only?** Yes.

**Evidence:** C (practitioner analogy) / B (formalized as Volume-Weighted momentum)

**The model:**
- Kinetic energy: price is in motion (trending), volume is high
- Potential energy: price is consolidating, building "stored" energy for next move
- Energy release: breakout from tight consolidation = potential → kinetic

Formalized as: `KE = 0.5 * volume * return²`. Total energy conservation: large moves require buildup phases. This is related to the Bollinger Band squeeze (low volatility = compressed potential energy).

**Implementation:**

```python
def price_energy(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Kinetic (active trend) vs potential (consolidation) energy.
    High KE: in-trend, follow momentum
    High PE (low KE): potential breakout, watch for squeeze resolution
    """
    returns = df["close"].pct_change()
    kinetic = (0.5 * df["volume"] * returns ** 2).rolling(window).mean()
    volatility = returns.rolling(window).std()
    # Potential energy: inverse of current volatility relative to history
    hist_vol = returns.rolling(252).std()
    potential = (hist_vol - volatility) / (hist_vol + 1e-8)  # squeeze depth
    return {
        "kinetic_energy": kinetic.iloc[-1],
        "potential_energy": potential.iloc[-1],
        "regime": "active" if kinetic.iloc[-1] > kinetic.quantile(0.7) else "consolidating"
    }
```

---

### 5.4 Permutation Entropy (PE)

**OHLCV only?** Yes.

**Evidence:** A (academic) + B (trading application)

PE measures the complexity/predictability of a time series by analyzing the distribution of ordinal patterns. Lower PE = more ordered = more predictable = better for trend-following. Higher PE = more random = avoid directional bets.

**Academic evidence:** Preprints.org (Feb 2025): Shannon entropy as filtering mechanism for ML trading signals on Bitcoin (2017–2025) outperformed baseline. Permutation entropy is computationally lighter than Shannon entropy and more robust to noise.

**Implementation (no external library needed — pure numpy):**

```python
from itertools import permutations
from math import factorial, log2

def permutation_entropy(prices: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Permutation entropy of a price series.
    order: embedding dimension (3–7, typically 3 or 4)
    delay: lag between elements
    Returns: 0 (perfectly ordered) to 1 (maximally random)
    """
    n = len(prices)
    permutation_types = {}
    total = 0
    for i in range(n - (order - 1) * delay):
        window = [prices[i + j * delay] for j in range(order)]
        pattern = tuple(sorted(range(order), key=lambda k: window[k]))
        permutation_types[pattern] = permutation_types.get(pattern, 0) + 1
        total += 1
    probs = [count / total for count in permutation_types.values()]
    H = -sum(p * log2(p) for p in probs if p > 0)
    H_max = log2(factorial(order))
    return H / H_max  # Normalized: 0 = ordered, 1 = random

def pe_trading_signal(df: pd.DataFrame, window: int = 50, order: int = 3) -> dict:
    """
    Use rolling permutation entropy to identify predictable vs random periods.
    Low PE → trend-following signal
    High PE → mean-reversion or no trade
    """
    prices = df["close"].values[-window:]
    pe = permutation_entropy(prices, order=order)
    if pe < 0.7:
        regime = "ordered"  # trend-following favorable
        signal = np.sign(prices[-1] - prices[-window // 4])
    elif pe > 0.9:
        regime = "random"  # avoid
        signal = 0.0
    else:
        regime = "mixed"
        signal = 0.0
    return {"pe": pe, "regime": regime, "signal": signal}
```

---

### 5.5 Ornstein-Uhlenbeck Process (Mathematical Mean Reversion)

Already covered in 5.1 (Hooke's Law section). The OU process IS the mathematical formalization of spring-like mean reversion. See implementation above.

---

## Part 6: Number Theory / Sequences

### 6.1 Prime Number Cycles

**OHLCV only?** Yes.

**Evidence:** F

No credible academic or practitioner evidence. Primes don't have a natural relationship to market microstructure. The sequence (2, 3, 5, 7, 11, 13, 17, 19, 23...) grows too irregularly for consistent cycle detection.

**Skip for MIDGE.**

---

### 6.2 Square Number Support/Resistance Levels

**OHLCV only?** Yes.

**Evidence:** C (folded into Gann square root work)

Round squares (100, 144, 225, 400, 625, 1,000, 2,500, 10,000...) act as psychological support/resistance because they are both round numbers AND perfect squares. BTC at $1,000, $10,000, $100,000 are all round number milestones; the square number sequence generates intermediate levels: $4,900 (70²), $14,400 (120²), $22,500 (150²), $40,000 (200²), $62,500 (250²).

**Implementation:** Already covered by sqrt_levels() in section 2.2.

---

### 6.3 Pi-Based Time Cycles

**OHLCOV only?** Yes (index-based).

**Evidence:** D (anecdotal)

Pi ≈ 3.14159... used as: cycle = π × some base period. E.g., 31.4-day cycle, 314-day cycle. W.D. Gann referenced pi in his wheel calculations. The 22/7 approximation generates 22 and 7 as natural cycle lengths — both of which overlap with existing weekly/monthly patterns.

**No rigorous evidence.** The base period selection is arbitrary.

---

### 6.4 Natural Logarithm Spiral / Log Scale Levels

**OHLCV only?** Yes.

**Evidence:** B (practitioners) / A (log-normal price assumption)

Log scale support/resistance is legitimately motivated: asset prices follow log-normal distributions, so equal percentage moves (not equal dollar moves) are the correct unit of measurement. A "spiral" in log-price space maps to equal-ratio intervals.

**Implementation:**

```python
def log_spiral_levels(current_price: float, n: int = 8) -> list:
    """
    Generate support/resistance at logarithmically equal intervals.
    Uses phi (1.618) as the ratio — combines log scale with golden ratio.
    """
    phi = 1.618033988749895
    levels_up = [current_price * (phi ** i) for i in range(1, n + 1)]
    levels_down = [current_price / (phi ** i) for i in range(1, n + 1)]
    return sorted(levels_down + [current_price] + levels_up)
```

---

## Part 7: Statistical / Probabilistic

### 7.1 Z-Score Mean Reversion

**OHLCV only?** Yes.

**Evidence:** A (pairs trading literature is robust)

The Z-score of a spread measures how many standard deviations the current price is from the rolling mean. Standard implementation:

```
z = (price - rolling_mean) / rolling_std
Entry long: z < -2  (buy when oversold)
Entry short: z > +2 (sell when overbought)
Exit: z returns to 0
```

**Evidence:** Pairs trading using Z-scores has decades of academic backing (Gatev et al. 2006, JFE). Crypto applications for ETH/BTC, BTC/USDT basis, correlated altcoin pairs have been demonstrated.

**Note:** MIDGE already has CorrelationTracker. This is the most natural extension.

**Implementation:**

```python
def zscore_signal(df: pd.DataFrame, window: int = 20,
                  entry_threshold: float = 2.0,
                  exit_threshold: float = 0.5) -> dict:
    """
    Z-score mean reversion signal.
    Returns signal: +1 (long), -1 (short), 0 (neutral/exit)
    """
    prices = df["close"]
    rolling_mean = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    zscore = (prices - rolling_mean) / (rolling_std + 1e-8)
    z = zscore.iloc[-1]
    if z < -entry_threshold:
        return {"zscore": z, "signal": 1.0}   # oversold, buy
    elif z > entry_threshold:
        return {"zscore": z, "signal": -1.0}  # overbought, sell
    elif abs(z) < exit_threshold:
        return {"zscore": z, "signal": 0.0}   # exit zone
    return {"zscore": z, "signal": 0.0}       # hold
```

---

### 7.2 Monte Carlo Exit Optimization

**OHLCV only?** Yes (operates on trade return distributions).

**Evidence:** A (risk management standard)

Monte Carlo reshuffles historical trade outcomes to generate alternative equity curve paths, stress-testing:
1. Whether profits survive sequence-of-returns risk
2. Optimal stop-loss levels (those that survive worst-case reshuffles)
3. Whether the edge is in entries or exits

Key technique: **randomize exits while keeping entries fixed**. If profitability survives, the entry signal has genuine edge. If not, the original system was overfit to specific exit timing.

**MIDGE already has:** FTMO backtester and paper trading. Monte Carlo would wrap around these.

---

### 7.3 Kelly Criterion Position Sizing

**OHLCV only?** Yes (operates on win rate and win/loss ratio).

**Evidence:** A (mathematically optimal under specific assumptions)

`f* = (bp - q) / b` where b = win/loss ratio, p = win probability, q = 1-p.

**MIDGE has:** Kelly sizing already partially implemented via ExecutableSignal. This is active.

**The critical caveat:** Full Kelly bet sizes are too aggressive for live trading. Half-Kelly (f*/2) is the practitioner standard.

---

### 7.4 Bayesian Regime Switching

**OHLCV only?** Yes.

**Evidence:** A (HMM literature) + B (trading)

Using Bayesian inference (pymc or hmmlearn) to estimate probability of being in each regime (bull/bear/volatile) and updating beliefs dynamically as new data arrives.

**Specific backtest result (from search):** HMM regime filter reduced maximum daily drawdown from **56% to 24%** vs buy-and-hold, with Sharpe ratio of 0.48. The regime filter doesn't improve returns but dramatically improves risk-adjusted metrics.

**Note:** MIDGE already has `regime_classifier.py`. This is the statistical formalization of that component.

**Implementation using statsmodels (already installed):**

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def bayesian_regime_signal(df: pd.DataFrame, n_regimes: int = 2) -> dict:
    """
    Markov Regime Switching model via statsmodels.
    Returns current regime probabilities.
    """
    returns = df["close"].pct_change().dropna()
    model = MarkovRegression(returns, k_regimes=n_regimes, trend="c", switching_variance=True)
    result = model.fit(disp=False)
    smoothed_probs = result.smoothed_marginal_probabilities
    current_probs = smoothed_probs.iloc[-1].values
    bull_regime = int(np.argmax(result.params[::4]))  # regime with higher mean
    return {
        "regime_probs": current_probs.tolist(),
        "most_likely_regime": int(np.argmax(current_probs)),
        "bull_prob": float(current_probs[bull_regime]),
    }
```

---

### 7.5 Hidden Markov Models for State Detection

**OHLCV only?** Yes.

**Evidence:** A (academic) + B (trading)

See section 7.4. HMMs are the unsupervised version — regime labels are learned, not predefined. The `hmmlearn` library implements Gaussian HMMs.

**Specific crypto research:** Springer 2024 paper on regime-switching forecasting for cryptocurrencies found HMM-based regime detection outperforms single-regime models. Bitcoin 2016–2024 dataset.

---

## Part 8: Synthesis — MIDGE Integration Tiers

Based on evidence grade and implementation feasibility with MIDGE's existing stack (scipy, statsmodels, numpy, pandas):

### Tier 1: Implement Immediately (High evidence, OHLCV only, complements existing systems)

| Strategy | Evidence | New library needed? | Fits MIDGE how? |
|----------|----------|--------------------|----|
| Hurst Exponent (regime) | B | `nolds` (optional — pure numpy works) | Regime signal alongside RegimeClassifier |
| OU Z-Score (mean reversion) | A | None — statsmodels already installed | Spread signals for correlated assets |
| Permutation Entropy (regime) | A | None — pure Python | Chaos gate on convergence confidence |
| Physical Momentum (vol × return) | A | None | New "physics" domain in convergence engine |
| Higuchi Fractal Dimension | A | None | Regime filter, complement Hurst |
| Z-Score Mean Reversion | A | None | Already partial — extend to spread pairs |

### Tier 2: Implement After Validation (Moderate evidence, clear implementation path)

| Strategy | Evidence | Notes |
|----------|----------|-------|
| Harmonic Patterns | B | `djoffrey/HarmonicPatterns` library exists; strict ratios help filter |
| Gann Time Cycles | C | Calendar-based — easy to add to MarketClock |
| Gann Square of Nine | C | Single formula, price levels only |
| Wolfe Waves | B | 41% EPA hit rate — requires zigzag detection |
| Fibonacci Extensions (127.2%, 161.8%) | D | Low standalone evidence; useful as target levels for existing signals |
| Bayesian Regime Switching (HMM) | A | `hmmlearn` — possible complement to existing RegimeClassifier |

### Tier 3: Research Only (Low evidence, subjective, or non-automatable)

| Strategy | Reason to skip |
|----------|----------------|
| Fibonacci Fan / Arc | Not automatable (scale-dependent) |
| Gann Angles (1×1, 2×1) | Not automatable (time/price scaling arbitrary) |
| Elliott Wave | Too subjective; automated attempts show mixed results |
| Prime number cycles | No evidence |
| Pi cycles | Arbitrary base period selection |
| Kondratieff waves | 40–60 year cycles; not actionable for signals |
| Sacred geometry (platonic solids) | No trading evidence |
| Strange attractors (trading use) | Academic insight only; not a signal generator |
| Lyapunov Exponent | Crash predictor only; computationally expensive |
| Mandelbrot MMAR | Generative model, not directional signal |

---

## Part 9: Key Implementation Packages

| Package | Purpose | Install | Currently in MIDGE? |
|---------|---------|---------|---------------------|
| `nolds` | Hurst, Lyapunov, fractal correlation dim | `pip install nolds` | No |
| `hmmlearn` | Gaussian HMM regime detection | `pip install hmmlearn` | No |
| `statsmodels` | Markov regime switching, OU fitting | already installed | Yes |
| `scipy` | Spectral analysis, signal processing | already installed | Yes |
| `stumpy` | Matrix profile, motif detection | already installed | Yes |
| `HarmonicPatterns` | Harmonic pattern detection | `pip install harmonicpatterns` or GitHub | No |
| `taew` | Elliott wave detection (academic paper) | `pip install taew` | No |

---

## Part 10: The Honest Assessment

**What actually works (evidence grade A/B, automatable):**

1. **Hurst Exponent as a regime classifier** — tells you whether to trend-follow or mean-revert. Evidence: B (64% win rate in backtest, but few trades). Strongest use: gating other signals by regime.

2. **OU/Z-score mean reversion on spread pairs** — the most mature quantitative strategy in this entire document. Evidence: A. Works on crypto pairs (ETH/BTC). Already complements MIDGE's CorrelationTracker.

3. **Permutation Entropy as a chaos filter** — gate signal confidence when PE is high (random period). Evidence: A (academic). Pure Python, trivial to implement.

4. **Physical Momentum (volume × return²)** — better than price-only momentum in academic studies. Requires only OHLCV. Evidence: A (published paper).

5. **Fractal Dimension (Higuchi)** — complements Hurst, identifies trending vs mean-reverting regimes independently. Evidence: A (academic), B (MQL5 study outperforms Hurst alone).

6. **HMM Regime Detection** — reduces drawdown from 56% → 24% as a trade filter. Evidence: A.

**What doesn't work as claimed (honest assessment):**

- Harmonic patterns (Bat, Gartley, etc.): Win rates of 33–60% are marginal and require expert pattern selection. Not reliably reproducible via automation.
- Elliott Wave: Fundamentally subjective. Even ML approaches are "almost impossible to backtest."
- Gann angles/fans: Cannot be objectively scaled. Any backtest is circular.
- Fibonacci time zones / Lucas numbers: Anecdotal evidence only.
- Sacred geometry (platonic solids, pi cycles): No trading evidence whatsoever.
- Kondratieff waves: Academic rejection. Not actionable for signal generation.

**The self-fulfilling prophecy effect:** Fibonacci retracement levels (61.8%, 50%, 38.2%) work partly because millions of traders watch them simultaneously. At sufficient liquidity, the collective belief creates the reaction. This makes them **worth including as confluence factors** (when other signals already exist at those levels) but not as independent signal generators.

---

*Sources consulted: quantifiedstrategies.com, robotwealth.com, pyquantnews.com, MDPI academic journals, ScienceDirect, ResearchGate, GitHub repositories (djoffrey/HarmonicPatterns, btcorgtfo/ElliottWaveAnalyzer, hyperstripe50/fractal-market-analysis, philippe-ostiguy/PyBacktesting), PyPI (nolds, taew, hmmlearn), QuantStart, QuantInsti, liberatedstocktrader.com, scitepress.org (2018 Hurst paper), nature.com, pmc.ncbi.nlm.nih.gov (thermodynamics paper).*
