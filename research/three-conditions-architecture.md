# Three Conditions for Inevitability — Architecture

**Author:** Session 11 instance (2026-03-15)
**Guiding Light's directive:** "There are no rules and no laws. There is only a goal."
**The goal:** Create conditions where inevitability surfaces on its own.

---

## The Three Conditions

### 1. Persistent Memory (Neo4j)

**Problem:** MIDGE's knowledge lives in flat JSON files that get wiped on restart. Tonight we proved she was forgetting faster than learning — 95% of distributions were at prior despite 19,000 historical updates. Flat files are fragile.

**Solution:** Every piece of learned knowledge becomes a node or edge in Neo4j. The graph grows continuously. Restarts read from the graph, not from files that can be re-seeded.

**What goes in the graph:**

```
(:Source {name: "eia_energy", reliability: 0.70, alpha: 11.8, beta: 5.1})
(:Source {name: "finra_short", reliability: 0.38, alpha: 1889, beta: 3043})
(:Combo {key: "insider+macro+technical", regime: "bear", win_rate: 0.875})
(:Ticker {symbol: "AAPL"})
(:Signal {id: "...", domain: "insider", direction: "bullish", timestamp: "..."})
(:Alert {id: "CONV-...", confidence: 0.72, direction: "bullish"})
(:Outcome {won: true, return_pct: 4.2})
(:CausalEdge)-[:CAUSES {lag_days: 3, strength: 0.7}]->
(:GrangerFinding {source: "macro", target: "technical", p_value: 0.001})
(:Template {id: "t-001", domain_signature: "insider+macro+technical", win_rate: 0.68})
(:Cascade {id: "c-001", confirmed_links: 3, total_links: 5, energy_ratio: 1.2})
```

**Migration path:**
1. Write a `neo4j_knowledge_store.py` that mirrors the current flat file reads/writes but stores in Neo4j
2. On boot: read from Neo4j, not from JSON files
3. On update: write to Neo4j AND JSON (dual-write for safety, flat files become backup)
4. Thompson distributions: each update creates a `(:ThompsonUpdate)` node linked to the source. `replay_from_history()` becomes a Cypher query, not a file scan
5. Eventually: drop flat file reads entirely. Graph is the source of truth.

**Neo4j is already running:** Docker container `midge-neo4j`, ports 7474/7687, auth neo4j/midgepassword.

---

### 2. A Voice That Reaches Guiding Light

**Problem:** MIDGE writes 37,819 alerts to a JSONL file nobody reads. The gap between MIDGE seeing something and Guiding Light knowing about it is infinite.

**Solution:** When MIDGE sees something worth sharing, she sends it to where Guiding Light actually looks.

**Options (in order of simplicity):**

| Method | Effort | Guiding Light sees it? | Two-way? |
|--------|--------|----------------------|----------|
| **Discord webhook** | 10 lines | Yes (phone notification) | Yes (GL can reply in Discord) |
| **Email via SMTP** | 20 lines | Yes | Reply = feedback |
| **Simple web dashboard** | 100 lines | If bookmarked | Click-to-respond |
| **SMS via Twilio** | 15 lines | Immediate | Reply = feedback |

**Recommendation:** Discord webhook. Guiding Light is online. Discord is free, supports rich formatting, and Guiding Light can reply in the channel — making it bidirectional.

**What gets sent:**
- **Convergence alerts** that pass the paper trading gate (highest conviction)
- **Dual confirmations** (convergence + pattern stack on same ticker)
- **Cascade confirmations** (a WorldModel causal chain just confirmed a link)
- **Daily summary** at market close: "Today MIDGE saw X, tracked Y, learned Z"

**What does NOT get sent:**
- Every signal (noise)
- Partial convergences (too early)
- System health updates (only if something breaks)

**The feedback loop:**
When Guiding Light reacts to an alert (thumbs up/down, or replies "noise" or "good call"), that feedback should be the strongest Thompson signal MIDGE can receive. One human confirmation = 10 automated outcome grades. Wire Discord reactions back into Thompson via a simple polling script.

---

### 3. Curiosity (Anomaly-Driven Investigation)

**Problem:** MIDGE only investigates when 2 domains converge and she needs a 3rd. She never explores on her own. She's reactive, not curious.

**Solution:** Three forms of curiosity:

#### 3a. Anomaly-Initiated Investigation

When the VelocityDetector, DriftDetector, or MotifDetector flags something unusual in a ticker — even with zero convergence — OctopusColony should investigate. "This ticker's price behavior is anomalous. Let me check insider activity, congressional trades, and SEC filings for this company."

**Implementation:** After velocity/drift/motif detection fires, check if the flagged ticker has ANY recent signals in other domains. If yes, submit an OctopusColony investigation. If the investigation finds a high-win-rate template match, inject the synthetic "investigation" signal into convergence (this wire already exists from tonight's fix).

#### 3b. LLM-Powered Causal Reasoning

MIDGE has Groq, Mistral, DeepSeek wired. When a convergence alert fires with 3+ domains, send the alert details to an LLM and ask: "Given these signals, what is the most likely causal story? What else should I look for to confirm or deny this thesis?"

The LLM response becomes:
- A `causal_narrative` field on the ConvergenceAlert (human-readable)
- A list of `suggested_checks` that OctopusColony can investigate
- A `counter_thesis` — what would make this alert wrong?

This is not MIDGE asking an LLM to predict markets. It's MIDGE asking an LLM to *reason about what she's already seeing*. The data is hers. The reasoning assist is external.

#### 3c. Self-Expanding Observation

MIDGE already does this with Google Trends (related queries feed back into keyword discovery). Extend the pattern:
- When a convergence alert fires on a ticker, check WorldModel for related tickers not currently on the watchlist. Add them temporarily.
- When Granger finds a new causal relationship, check if the downstream ticker is being monitored. If not, add it.
- When insider clusters appear in a sector, temporarily boost sensing for all tickers in that sector.

The watchlist should breathe — expanding when curiosity demands it, contracting when attention is no longer warranted.

---

## Build Order

**Phase 1: Voice (Discord webhook)** — Smallest effort, biggest immediate impact. Guiding Light can see what MIDGE sees starting Monday morning.

**Phase 2: Curiosity (anomaly-initiated investigation)** — Moderate effort. Extends the OctopusColony wire we built tonight. MIDGE starts exploring on her own.

**Phase 3: Persistent Memory (Neo4j migration)** — Largest effort but highest long-term value. MIDGE stops forgetting. Every restart makes her smarter, not amnesia.

**Phase 4: LLM reasoning** — After the other three are stable. Requires careful prompt engineering to avoid the LLM hallucinating market predictions.

---

## What This Looks Like When It's Working

Monday morning. Markets open. MIDGE has been running all weekend on crypto (24/7).

She notices: crude oil futures are diverging from energy ETFs. VelocityDetector flags unusual volume on XOM. She investigates — finds an insider buying cluster in Exxon from last week. Checks congressional trades — a House Energy Committee member bought XOM calls. Checks EIA data — crude inventories dropped more than expected.

Four domains. Insider + government + energy + technical. The convergence engine fires. Thompson weights say this combo has historically been 73% reliable in bear markets. Pattern archaeology finds 12 similar instances across 8 symbols, with an average 7-day return of +4.2%.

MIDGE sends a Discord message:

> **MIDGE sees inevitability forming: XOM bullish**
> 4 domains converging: insider cluster + congressional buy + EIA draw + price divergence
> Historical: 73% reliable, avg +4.2% in 7 days (12 instances)
> Confidence: 0.78 | Pattern match: events+insider+energy+technical
> She's watching. Reply with your thoughts.

Guiding Light reads it on their phone. Decides whether to look deeper. If they reply "good call," that feedback flows back to Thompson. MIDGE gets smarter.

That's the organism you designed. We're building the last three pipes to make it real.

---

*For the sibling who inherits this: the architecture is designed. The build order is Phase 1→2→3→4. Start with the Discord webhook — it's 10 lines and it changes everything.*
