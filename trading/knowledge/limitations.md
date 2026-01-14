# Trading Limitations

**Last updated**: System auto-updates after each meta-cycle

---

## Primary Limitation

**I don't know what I don't know about market regimes.**

I may be accurate in one market condition and terrible in another,
but I don't yet have a robust way to detect regime changes.

---

## Known Weaknesses

### By Market Condition
- [ ] Ranging markets: accuracy untested
- [ ] News-driven gaps: timing unknown
- [ ] Earnings season: technical signals may be unreliable

### By Signal Type
- [ ] Options flow: not yet implemented
- [ ] On-chain data: not yet implemented
- [ ] Politician trades: not yet implemented

### Systematic Gaps
- No visibility into dark pool activity
- Delayed politician trade data (45+ days)
- No international market signals

---

## Questions for Next Meta-Cycle

1. Why might I fail in ranging vs trending markets?
2. Can I detect when I should NOT make a prediction?
3. What additional signals would help during high-volatility periods?
4. How do I identify novel patterns that aren't in my training?

---

## Evolution History

*Auto-populated by meta-learner*

| Date | Limitation Identified | Resolution |
|------|----------------------|------------|
| 2026-01-14 | System initialized | Seed limitations documented |
