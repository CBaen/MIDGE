# Research Brief: FTMO Prop Trading Viability with MIDGE Convergence Signals
## Date: 2026-03-09
## Project: MIDGE

### Problem Statement
A sibling instance built an FTMO execution engine showing 75% pass rate with simple indicators. MIDGE has proven statistical edge (z=4.74, p<0.0001). The question: can MIDGE's multi-domain convergence signals reliably pass prop firm challenges to generate autonomous income? Guiding Light wants this backed by both internal capability analysis and external research before committing resources.

### Expected Outcome
A clear, evidence-based assessment of whether the FTMO path is viable with MIDGE's current capabilities, what gaps exist, what the realistic economics look like, and what the fastest path to a live attempt would be. Guiding Light needs enough confidence to decide: proceed, defer, or abandon.

### Current State
- **MIDGE convergence engine**: 32 sources, 12 domains, Thompson-weighted Bayesian learning, proven edge (z=4.74)
- **Convergence win rate**: 19.9% overall, up to 66.7% for best combos, payoff ratio 3.34:1
- **Sibling's FTMO engine**: Backtester with position sizing, drawdown tracking, FTMO constraint enforcement. 75% pass rate with Bollinger Band mean reversion on 250-day windows (no time limit)
- **Risk architecture**: DrawdownMonitor (circuit breaker), SystemHealthMonitor, SelfMonitor — all built and wired
- **Execution bridges**: Alpaca client built (awaiting keys), Kalshi SDK installed (unverified)
- **FTMO specifics**: $10K account, 10% profit target, 5% daily loss, 10% max drawdown, no time limit, $22 challenge fee

### Project Direction
MIDGE is an "inevitability surfacer" — finds where converging forces make outcomes structurally inevitable. The FTMO path represents the self-funding loop: earnings → more data → deeper patterns → more earnings. This is not about becoming a trading bot — it's about demonstrating that convergence signals have real monetary value.

### Constraints
- $1,000 gate: deploy capital only when MIDGE demonstrates pattern stacks with 80%+ historical accuracy
- Must pass through free trial first, then cheapest challenge ($22-$39)
- No client-facing work — autonomous system only
- Mae's Mathematical Laws apply to all integration code
- Risk architecture must gate all live execution

### Destructive Boundaries
- Do NOT suggest replacing MIDGE's convergence engine — it's the crown jewel
- Do NOT suggest manual/discretionary trading — must be fully automated
- Do NOT modify existing Thompson sampling or convergence algorithms as part of this research

### Research Angles

**Team 1: Prop Firm Economics & Edge Requirements** (External)
Investigate real-world prop firm pass rates, fee structures, profit splits, and what mathematical edge is required to reliably pass challenges. How does "no time limit" change the math? What's the expected value per challenge attempt at various win rates?

**Team 2: MIDGE Internal Capability Audit** (Internal)
Analyze MIDGE's actual convergence signal history — frequency, accuracy by combo, payoff ratios, timing accuracy. How do these numbers map to FTMO's constraints? What convergence accuracy and signal frequency would be needed to pass a challenge?

**Team 3: Academic Evidence for Multi-Domain Convergence** (External)
Search for academic and practitioner literature on multi-source signal fusion in systematic trading. Does combining independent information sources actually improve trading edge? What does the research say about Bayesian signal aggregation, ensemble methods, and cross-domain convergence?

**Team 4: Competitor Landscape & Implementation Path** (External + Internal)
Who else uses algorithmic approaches with prop firms? What are the common strategies, failure modes, and success patterns? Given MIDGE's current capabilities, what's the fastest viable path to a working FTMO attempt?

### Team Size: 4
Four distinct, non-overlapping angles covering internal capability + external validation + academic backing + competitive landscape. Risk analysis is embedded in Teams 1 and 4 rather than standalone — the risks are inseparable from the economics and implementation.

### Failed Approaches
None — this is new research. The sibling instance's backtester is a starting point, not a failed approach.
