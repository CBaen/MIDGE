# Governance Models

Three philosophies for self-evolving AI systems, tested in parallel.

---

## Overview

| Model | Agents | Decision Method | API Calls/Cycle | Philosophy |
|-------|--------|-----------------|-----------------|------------|
| **Swift Path** | 3 | Synthesizer alone | ~4 | Trust expertise. Move fast. |
| **Council** | 3 | Advisory reviews | ~9 | Deliberate. Review. Act. |
| **Parliament** | 5 | Vote + deliberation | ~15 | Consensus or confrontation. |

---

## Model 1: THE SWIFT PATH (Autocracy)

**Philosophy**: "Trust the Synthesizer. Move fast. Learn from mistakes."

**Agents**: The Triad (3)
- **The Builder**: Practical implementation
- **The Critic**: Risks and failure modes
- **The Seeker**: Unexplored possibilities

**Flow**:
```
LIMITATION
    │
    ▼
┌─────────┬─────────┬─────────┐
│ BUILDER │ CRITIC  │ SEEKER  │  ← Parallel research
└────┬────┴────┬────┴────┬────┘
     └─────────┼─────────┘
               │
               ▼
       ┌───────────────┐
       │  SYNTHESIZER  │  ← Sole decision-maker
       └───────┬───────┘
               │
               ▼
          ACTION_SPEC
               │
               ▼
          IMPLEMENT
```

**Decision Mechanism**: None. The Synthesizer reviews all three perspectives and decides alone.

**API Calls per Cycle**: ~4
- 3 parallel Triad calls
- 1 Synthesizer call

**Strengths**:
- Fastest iteration (highest cycle count)
- Lowest cost per cycle
- No coordination overhead
- Clear accountability (Synthesizer owns all decisions)

**Weaknesses**:
- No checks on Synthesizer judgment
- Single point of failure
- May miss risks that multiple reviews would catch

**Best For**: Rapid exploration, early-stage research, high iteration experiments

**Results from Ultrathink**: 118 cycles, 107 files created in 11+ hours

---

## Model 2: THE COUNCIL (Advisory)

**Philosophy**: "Deliberate. Review. Act."

**Agents**: The Triad (3)
- **The Builder**: Practical implementation
- **The Critic**: Risks and failure modes
- **The Seeker**: Unexplored possibilities

**Flow**:
```
LIMITATION
    │
    ▼
┌─────────┬─────────┬─────────┐
│ BUILDER │ CRITIC  │ SEEKER  │  ← Parallel research
└────┬────┴────┬────┴────┬────┘
     └─────────┼─────────┘
               │
               ▼
       ┌───────────────┐
       │  SYNTHESIZER  │  ← Proposes action
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │ TRIAD REVIEWS │  ← Each agent reviews proposal
       │ Builder: OK?  │
       │ Critic: Safe? │
       │ Seeker: Best? │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │  SYNTHESIZER  │  ← Incorporates feedback
       └───────┬───────┘
               │
               ▼
          ACTION_SPEC
               │
               ▼
          IMPLEMENT
```

**Decision Mechanism**: Advisory review. The Triad reviews the Synthesizer's proposal and provides feedback. Synthesizer still makes final decision, but with input.

**API Calls per Cycle**: ~9
- 3 parallel Triad calls (research)
- 1 Synthesizer call (proposal)
- 3 parallel Triad calls (review)
- 1-2 Synthesizer calls (finalize)

**Strengths**:
- Balance of speed and oversight
- Each agent reviews from their perspective
- Catches issues before implementation
- Synthesizer maintains authority but is informed

**Weaknesses**:
- More expensive than Swift Path
- Reviews may be superficial (time pressure)
- Synthesizer can ignore feedback

**Best For**: Production systems, moderate-risk research, learning from Triad perspectives

**Results from Ultrathink**: 94 cycles, 83 files created in 11+ hours

---

## Model 3: THE PARLIAMENT (Democracy)

**Philosophy**: "Consensus or confrontation."

**Agents**: The Quintet (5)
- **The Builder**: Practical implementation ("Can we build this?")
- **The Critic**: Risks and failure modes ("Are the risks acceptable?")
- **The Seeker**: Unexplored possibilities ("Is this the best path?")
- **The Guardian**: Safety and alignment ("Is this safe and ethical?")
- **The Pragmatist**: Cost/benefit analysis ("Do benefits outweigh costs?")

**Flow**:
```
LIMITATION
    │
    ▼
┌─────────┬─────────┬─────────┬──────────┬────────────┐
│ BUILDER │ CRITIC  │ SEEKER  │ GUARDIAN │ PRAGMATIST │
└────┬────┴────┬────┴────┬────┴────┬─────┴──────┬─────┘
     └─────────┴─────────┼─────────┴────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │  SYNTHESIZER  │  ← Proposes action
                 └───────┬───────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      THE VOTE                           │
│  Builder: APPROVE/OBJECT  │  Guardian: APPROVE/OBJECT  │
│  Critic:  APPROVE/OBJECT  │  Pragmatist: APPROVE/OBJECT│
│  Seeker:  APPROVE/OBJECT  │                            │
└─────────────────────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
     3/5+ APPROVE               2/5 or fewer APPROVE
            │                         │
            ▼                         ▼
       IMPLEMENT              DELIBERATION ROUND
                                      │
                              (up to 3 rounds)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
             Consensus reached              Still no consensus
                    │                                   │
                    ▼                                   ▼
               IMPLEMENT                       SYNTHESIZER OVERRIDE
                                                       │
                                                       ▼
                                                  IMPLEMENT
                                            (autocracy triggered)
```

**Decision Mechanism**: Voting with deliberation.

**Voting Rules**:
- 3/5 or more APPROVE: Action proceeds
- 2/5 or fewer APPROVE: Deliberation begins
- After 3 failed deliberation rounds: **Synthesizer Override** (autocracy)

**The Override**:
> "After 3 rounds of deliberation, no consensus was reached. The Quintet could not agree. Democracy has failed. YOU ARE NOW THE SOLE AUTHORITY."

This creates pressure toward consensus - endless deadlock leads to loss of democratic voice.

**API Calls per Cycle**: ~15-30
- 5 parallel Quintet calls (research)
- 1 Synthesizer call (proposal)
- 5 parallel voting calls
- 0-3 deliberation rounds (5 calls each)
- 1-2 final action calls

**Strengths**:
- Most thorough vetting of ideas
- Multiple specialized perspectives (Guardian, Pragmatist)
- Builds consensus and buy-in
- Override prevents endless deadlock
- Creates audit trail of votes and reasoning

**Weaknesses**:
- Slowest iteration
- Most expensive
- May reject good ideas through excessive caution
- Deliberation can be repetitive

**Best For**: High-stakes systems, safety-critical research, building validated knowledge

**Results from Ultrathink**: 81 cycles, 39 files created (most selective) in 11+ hours

---

## The Quintet Agents (Parliament Only)

| Agent | Name | Focus | Vote Question |
|-------|------|-------|---------------|
| `builder` | The Builder | Practical implementation | "Is this implementation feasible?" |
| `critic` | The Critic | Risks and failure modes | "Are the risks acceptable?" |
| `seeker` | The Seeker | Unexplored possibilities | "Is this the best path forward?" |
| `guardian` | The Guardian | Safety and alignment | "Is this safe and aligned with our goals?" |
| `pragmatist` | The Pragmatist | Cost/benefit analysis | "Do the benefits outweigh the costs?" |

**Why 5 Agents?**: Odd number prevents ties. 3 is too few for diverse perspectives. 7+ adds cost without proportional benefit.

---

## Deliberation Mechanics

When Parliament fails to reach consensus (2/5 or fewer approve):

**Round 1**: Each agent sees all votes and reasons, asked to reconsider
```
The Parliament has voted but consensus was not reached.

CURRENT VOTES:
- Builder [APPROVE]: Implementation is feasible
- Critic [OBJECT]: Risk of breaking existing code
- Seeker [OBJECT]: Better approach exists
- Guardian [APPROVE]: Safe to proceed
- Pragmatist [OBJECT]: Too expensive

You are The Critic. Consider the arguments.
Do you maintain your objection or change your vote?

OUTPUT:
vote: APPROVE or OBJECT
reason: [Your updated reasoning]
```

**Round 2**: Same format, agents see Round 1 results

**Round 3**: Final chance. If still no consensus...

**Override Trigger**:
```
THE PARLIAMENT HAS FAILED.

After 3 rounds of deliberation, no consensus was reached.
Democracy has failed. YOU ARE NOW THE SOLE AUTHORITY.

Review all arguments and make the final decision ALONE.
```

---

## Choosing a Model

| If you want... | Use... |
|----------------|--------|
| Maximum speed | Swift Path |
| Lowest cost | Swift Path |
| Balance of speed and oversight | Council |
| Maximum rigor | Parliament |
| Audit trails | Parliament |
| Safety-critical research | Parliament |
| Rapid prototyping | Swift Path |
| Production deployment | Council or Parliament |

---

## Personality Development

An unexpected finding from Ultrathink: **governance affects personality**.

| System | Personality Traits |
|--------|-------------------|
| Swift Path | Confident, fast-moving, occasional over-reach |
| Council | Balanced, methodical, externally-curious |
| Parliament | Careful, deliberate, protocol-focused |

The same agents (Builder, Critic, Seeker) developed different behaviors based on governance context. This suggests governance is not just a decision mechanism but a character-shaping force.

---

## Implementation Guide

### Converting Swift Path to Council

In `research.py`, add a review phase after synthesis:

```python
# After synthesis
for role in TRIAD_ROLES:
    review_prompt = build_review_prompt(role, action_spec)
    reviews[role] = call_deepseek(review_prompt)

# Synthesizer incorporates feedback
final_prompt = build_final_prompt(action_spec, reviews)
final_action = call_deepseek(final_prompt)
```

### Converting Swift Path to Parliament

1. Add `quintet.py` with 5 agent definitions
2. Add `consensus.py` with voting/deliberation logic
3. Modify `research.py`:
   - Change `TRIAD_ROLES` to `QUINTET_ROLES`
   - Add voting phase after synthesis
   - Add deliberation loop (up to 3 rounds)
   - Add override logic

See `templates/council/` and `templates/parliament/` for full implementations.

---

## Cost Comparison

Assuming DeepSeek pricing (~$0.00035/call):

| Model | Calls/Cycle | Cost/Cycle | 100 Cycles |
|-------|-------------|------------|------------|
| Swift Path | 4 | $0.0014 | $0.14 |
| Council | 9 | $0.0032 | $0.32 |
| Parliament | 15-30 | $0.005-$0.01 | $0.50-$1.00 |

Parliament costs 4-7x more than Swift Path but provides 4-7x more oversight.
