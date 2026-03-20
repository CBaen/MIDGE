# MIDGE Queue

**Purpose:** Active tasks only. Git history preserves completed work.
**Last updated:** 2026-03-19 (Session 13)

## Critical — Do These First

1. **EXIT MANAGEMENT** — MIDGE buys but never sells. Positions sit forever with no stop-loss. Need: ATR trailing stop checked every cycle, strategy reversal exit, and time-based exit (close after N hours if no movement).

2. **POSITION SIZING LIMITS** — MIDGE allocated 80%+ of account, went negative on cash. Need: max 5% per trade, max 30% total exposure, check available cash before ordering.

3. **CLOSE THE LEARNING LOOP** — Forensic scorer finds winning combos but doesn't feed back into strategy registry. When a combo wins 5+ times at >60% WR in forensic data → auto-update its registry confidence. When a combo loses at <30% WR → auto-downweight.

## Important — Income Path

4. **EQUITY SWING TRADING** — $0 Alpaca fees. AKAM +3.42% is proving the equity strategies work. Build a parallel equity trader using the same standalone pattern. Daily bars, market hours only.

5. **KALSHI RESEARCH** — Guiding Light asked about making money on Kalshi prediction markets. Research: what's available, how does it work, can MIDGE's convergence approach apply?

6. **FUTURES RESEARCH** — Evaluate FTMO or direct futures access. Fees, leverage, strategy fit.

7. **LOWER-FEE CRYPTO EXCHANGE** — Move from Alpaca (0.25%/side) to Coinbase Advanced (0.05% maker) via CCXT library (already installed conceptually).

## Growth — Make MIDGE Smarter

8. **WEB CRAWLER FOR PATTERN DISCOVERY** — Scrape Reddit, Twitter, trading forums for pattern claims → test against history → auto-add validated patterns to strategy library.

9. **MULTI-TIMEFRAME CONFLUENCE** — Run strategies on 1m + 5m + 15m + 1h simultaneously. Agreement across timeframes = strongest possible signal.

10. **HURST + ENTROPY REGIME GATES** — Use Hurst exponent and permutation entropy to detect whether the market is trending or mean-reverting RIGHT NOW, then gate which strategy families are active.

11. **COMPLETE RESEARCH COUNCIL** — Crypto trading intelligence council has 3 agent findings complete. Need challenge round, synthesis, and tension analysis.
