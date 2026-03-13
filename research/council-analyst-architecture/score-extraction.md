# Score Extraction: Multi-Analyst Architecture
## Date: 2026-03-13

## Shared Dimensions (Cross-Agent Comparison)

| Dimension | Codebase Analyst | External Researcher | Devil's Advocate | Spread |
|-----------|-----------------|---------------------|------------------|--------|
| Overall Risk (10=safe) | 7/10 (revised from 8) | 7/10 (implied) | 6/10 | 1 |
| Reversibility (10=trivial) | 10/10 | 8/10 | 9/10 | 2 |
| Evidence Confidence (10=solid) | 9/10 | 8/10 | 8/10 | 1 |

## Role-Specific Dimensions

### Codebase Analyst
| Dimension | Phase 1 | Post-Challenge |
|-----------|---------|----------------|
| Feasibility | 9/10 | 9/10 (build), 6/10 (outcome) per DA challenge |
| Blast Radius | 8/10 | 8/10 |
| Pattern Consistency | 9/10 | 7/10 per ER challenge (missing cycle-id) |
| Dependency Risk | 8/10 | 6/10 per DA challenge (sparse inputs) |

### External Researcher
| Dimension | Phase 1 | Post-Challenge |
|-----------|---------|----------------|
| Blackboard Relevance | 10/10 | 10/10 |
| Temporal Analyst Relevance | 9/10 | 6/10 (revised by self, near-term) |
| Integration Effort | 7/10 | 2/10 per DA challenge (impossible output) |
| FinCon Maturity | 9/10 | 9/10 |

### Devil's Advocate
| Dimension | Phase 1 | Post-Challenge |
|-----------|---------|----------------|
| Failure Probability | 5/10 | 5/10 (3/10 if sequenced, per ER) |
| Failure Severity | 8/10 | 8/10 |
| Assumption Fragility | 4/10 | 4/10 |
| Hidden Complexity | 5/10 | 5/10 |

## Unanimous Agreements (Independent Convergence)
1. `ctx.inevitabilities` is orphaned — needs a consumer before analysts add value
2. SituationBoard (typed class replacing `ctx._market_advisory`) is valuable independent of analyst count
3. Tiered alerter failure = warning case (signal starvation, not architecture flaw)
4. Temporal Analyst is correct long-term, premature at current data maturity
5. Performance is safe only if analysts read pre-computed ctx data
6. market_systems.py is over 500-line cap — needs extraction before additions

## Key Divergences
1. **Build timing:** CA says "build board now, analysts after data matures (50+ combos)." DA says "wire orphan, fix operations, wait." ER says "board + wiring first, analysts can be Gen 1 (sequential) immediately."
2. **Feasibility scope:** CA scores build feasibility (9/10). DA scores outcome feasibility (5/10). Different questions, different answers.
3. **Analyst vs enhanced DeepAnalyst:** DA argues DeepAnalyst already does all six components. CA/ER argue specialization enables depth impossible in a generalist.
