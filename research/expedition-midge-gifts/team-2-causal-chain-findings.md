# Team 2 Findings: Causal Chain Engine — Multi-Hop Event Propagation

**Date:** March 5, 2026
**Assignment:** Find open-source tools for modeling multi-hop causal chains in financial market context

---

## Executive Summary

The goal: trace a chain like "hurricane hits Gulf Coast → ethylene production drops → auto parts shortage → Ford earnings miss → stock drops" with timing estimates at each link. This requires four distinct capabilities: (1) a graph structure to hold world-model connections, (2) cascade propagation simulation, (3) automatic chain discovery from time series, and (4) data to populate the graph. Each has well-matched open-source tools that run pure Python on Windows.

The most significant finding: **NetworkX + NDlib is the right engine pairing for MIDGE**. NetworkX holds the world-model graph (commodity flows, company dependencies, sector relationships), and NDlib simulates how shocks propagate through it with timing. These plug directly into what CausalReasoningEngine already does — they just add the supply chain skeleton it currently lacks.

The second key finding: **GLEIF's Level 2 relationship data** is a free, structured, legally-sourced company ownership graph covering 2.6 million entities globally. This is the backbone for "who depends on whom."

---

## Section 1: Supply Chain Knowledge Graph — Data Sources

### 1.1 GLEIF — Global Legal Entity Identifier Foundation
**What it is:** A free, open database of 2.6+ million legal entities with parent/child ownership relationships mapped. Maintained by regulators globally. Every public company and most private ones have an LEI.

**Why it matters for MIDGE:** The Level 2 "Who Owns Whom" dataset maps direct and ultimate parents for each entity. This is a ready-made corporate ownership graph — Ford's LEI links to Ford Motor Company's subsidiaries, which links to its Tier 1 suppliers if they're registered. Can be downloaded as full bulk files (XML/CSV) or queried via REST API at no cost.

**Python access:** `pygleif` library on PyPI. REST API requires no authentication for basic lookups.

**Limitation:** Captures ownership structure (who owns whom), not supply relationships (who buys parts from whom). Tier 2+ supply chain relationships are not in GLEIF. Best used to map conglomerate exposure — if one holding company owns three auto suppliers, a shock to the parent propagates to all three.

**Format:** Bulk Golden Copy files available as CSV/JSON/RDF. Daily delta files available. `pygleif` wraps the REST API.

**Evidence:** GLEIF.org documentation confirmed free bulk download. pygleif on PyPI at https://pypi.org/project/pygleif/

---

### 1.2 UN Comtrade — International Trade Statistics
**What it is:** The United Nations repository of international merchandise trade statistics. Country-level import/export flows by commodity code (HS codes), covering 200+ countries, annual data from 1988, monthly from 2000.

**Why it matters for MIDGE:** This answers "which countries supply which commodities to which countries." Hurricane disrupts Gulf Coast ethylene → US ethylene exports drop → identify which countries import US ethylene → identify which industries in those countries depend on ethylene. Country-level, not company-level, but covers commodity flow direction.

**Python access:** Official `comtradeapicall` library on PyPI, maintained by the UN itself. Free tier exists — requires signup but no payment. API has rate limits on the free tier.

**Limitation:** Country-level only. Cannot trace "ExxonMobil's Port Arthur plant" — only "USA exports X tons of ethylene to Germany." Useful for macro cascade routing but not company-specific exposure.

**Evidence:** https://github.com/uncomtrade/comtradeapicall confirmed active, official Python library.

---

### 1.3 Open Supply Hub
**What it is:** A collaborative supply chain mapping platform where brands, factories, and NGOs contribute facility-level supply chain data. Focused on apparel/consumer goods.

**Why it matters for MIDGE:** Has actual company-to-factory supply relationships, not just ownership. Brand A uses Factory B in country C. This is the type of link needed for "supplier disruption → downstream company affected" chains.

**Limitation:** Heavily weighted toward apparel/retail sectors. Not useful for chemicals, auto parts, or energy supply chains which are MIDGE's primary targets. API access exists but data density in financial-market-relevant sectors is low.

**Verdict:** Lower priority for MIDGE's core use case. Note for future: check if coverage has expanded to industrial sectors.

**Evidence:** opensupplyhub.org (website confirmed active March 2026)

---

### 1.4 SupplyGraph Dataset (GitHub: ciol-researchlab/SupplyGraph)
**What it is:** A benchmark graph dataset for supply chain GNN research, sourced from a real FMCG company in Bangladesh. 41 product nodes, 684 edges representing plant/storage/category relationships. Freely downloadable from Kaggle.

**Why it matters for MIDGE:** Useful as a proof-of-concept graph to test cascade simulation code before wiring in real company data. The graph structure (nodes=products, edges=relationships) is exactly the shape of data MIDGE would use.

**Limitation:** Small, single-company, Bangladesh FMCG — not US public market supply chains. Research/testing value, not production data source.

**Evidence:** arxiv.org/abs/2401.15299 (paper), github.com/ciol-researchlab/SupplyGraph (code)

---

### 1.5 Wikidata — Company Sector and Relationship Queries
**What it is:** Wikidata's SPARQL endpoint contains structured facts about public companies: headquarters, industry classification, subsidiaries, key products, major suppliers (where reported). Free, no auth required.

**Why it matters for MIDGE:** Can query "which companies produce ethylene" or "which companies are subsidiaries of Dow Chemical" using SPARQL. Coverage is inconsistent but better for large-cap public companies that are actively edited.

**Python access:** `SPARQLWrapper` library. No API key needed.

**Example query pattern:** SELECT all companies with property "industry" = petrochemical, then cross-reference with ConvergenceAlerter ticker universe.

**Limitation:** Crowdsourced — coverage gaps, especially for Tier 2+ suppliers. Data quality varies. Not a systematic supply chain database.

---

## Section 2: Event Cascade / Propagation Simulation

### 2.1 NetworkX — The Core Graph Engine (RECOMMENDED FOUNDATION)
**What it is:** Python's standard graph analysis library. DiGraph supports directed weighted graphs. Pure Python, pip install, zero Windows issues, actively maintained (v3.6.1 as of 2025).

**Why it matters for MIDGE:** NetworkX is the substrate on which to build MIDGE's world-model. Define a DiGraph where:
- Nodes = events, commodities, companies, regions, market sectors
- Edges = causal links with two attributes: `probability` (how likely does A cause B) and `delay_days` (how long before effect appears)

Then BFS/DFS through the graph to find all multi-hop paths from an event to a set of tickers. This is conceptually what CausalReasoningEngine already does in pure Python dicts — NetworkX formalizes it and adds graph algorithms (shortest path, centrality, community detection) for free.

**Key capability:** `nx.all_simple_paths(G, source="hurricane_gulf", target="F_stock")` returns all paths through the world-model graph. Each path is a chain like [hurricane → ethylene_drop → auto_parts_shortage → ford_earnings_miss → F_stock]. Edge attributes give timing at each hop.

**Integration point:** CausalReasoningEngine already maintains `_links`, `_causes`, `_effects` as dicts. These map directly to a NetworkX DiGraph. The conversion is straightforward and non-breaking.

**Evidence:** networkx.org/documentation/stable (official docs, current). Pure Python, pip install.

---

### 2.2 NDlib — Network Diffusion Library (RECOMMENDED FOR CASCADE SIMULATION)
**What it is:** A Python library built on top of NetworkX that implements epidemic/diffusion models: SIR, SIS, Independent Cascade, Threshold models, and more. 16+ diffusion models included.

**Why it matters for MIDGE:** NDlib runs the simulation of how a shock propagates through the NetworkX world-model graph. The Independent Cascade (IC) model is directly applicable: a node (event/commodity/company) "infects" its neighbors with probability proportional to edge weight, spreading through the network over discrete time steps. Each time step maps to a configurable unit (days, weeks).

**What this gives MIDGE:**
- Input: Seed node (e.g., "Gulf Coast hurricane event"), network graph, propagation probabilities
- Output: At each time step T, which nodes have been "activated" (affected)
- Result: "Ford's stock exposure appears approximately 3-5 time steps (weeks) after the hurricane"

**Installation:** `pip install ndlib` — pure Python, Windows compatible

**Limitation:** The IC model doesn't natively support varying delay-per-edge (all edges use the same stochastic process). To encode domain-specific delay estimates, you can add custom state machines or use the `SIR_threshold` variant with adjusted parameters per community.

**Evidence:** github.com/GiulioRossetti/ndlib (active repo), arxiv.org/abs/1801.05854 (paper), ndlib.readthedocs.io (docs)

---

### 2.3 EoN (Epidemics on Networks) — Alternative With Delay Support
**What it is:** A Python package for epidemic simulation on networks, with over 100 methods. Supports `fast_nonMarkov_SIR` which allows user-specified transmission rules — meaning delay distributions can be encoded per edge.

**Why it matters for MIDGE:** EoN's `fast_nonMarkov_SIR` lets you specify that "the hurricane→ethylene link takes 1-3 days, but the ethylene→auto_parts link takes 14-28 days." This per-edge timing is exactly what the causal chain engine needs.

**Installation:** `pip install EoN` — pure Python, built on NetworkX

**Last updated:** Version 1.2 released June 2024 — actively maintained.

**Key advantage over NDlib:** Non-Markovian transmission means delay distributions are first-class citizens, not afterthoughts. Each edge can have a different gamma/exponential distribution for its delay.

**Evidence:** arxiv.org/abs/2001.02436 (JOSS paper), epidemicsonnetworks.readthedocs.io (docs)

---

## Section 3: Temporal Causal Discovery (Beyond PCMCI+)

The previous expedition identified PCMCI+ (Tigramite) and Granger causality. Here are what exists beyond those:

### 3.1 CausalFlow — Unified Multi-Algorithm Framework (RECOMMENDED)
**What it is:** A Python library that wraps multiple causal discovery methods for time series into one framework, including F-PCMCI, CAnDOIT, and LPCMCI. All output time-series graphs that include lag information on each link.

**Why it matters for MIDGE:** CausalFlow doesn't just discover whether A causes B — it discovers the lag (number of time periods between cause and effect). This is precisely the "timing at each step" requirement. The graph output format is compatible with NetworkX for downstream simulation.

**F-PCMCI:** Extends PCMCI with Transfer Entropy feature selection. Faster than vanilla PCMCI on high-variable systems. Directly applicable to MIDGE's multi-domain time series (MIDGE tracks 11+ domains simultaneously).

**CAnDOIT (2024):** Extends LPCMCI to incorporate interventional data alongside observational data. When MIDGE has a known intervention event (e.g., a confirmed supply disruption), it can feed that as interventional data to improve chain discovery accuracy. Published in Advanced Intelligent Systems 2024.

**Installation:** `pip install py-causalflow`

**Last updated:** December 17, 2024 — active development.

**Windows:** Pure Python wrapping — no known Windows issues.

**Evidence:** github.com/lcastri/causalflow (repo), lcastri.github.io/causalflow/ (docs)

---

### 3.2 causal-learn — PC, FCI, GRaSP, Granger (py-why Ecosystem)
**What it is:** Python translation and extension of Tetrad (the gold standard academic causal discovery toolkit). Implements PC, FCI, GES, GRaSP, LINGAM, Granger, and more. Part of the py-why ecosystem (DoWhy, EconML, causal-learn).

**Why it matters for MIDGE:** FCI algorithm handles latent confounders — critical for financial markets where macro conditions confound many relationships. The output PAG (Partial Ancestral Graph) shows which causal directions are certain vs. uncertain. GRaSP is a recent permutation-based method that outperforms GES on sparse graphs.

**Granger causality in causal-learn:** More flexible than statsmodels' implementation — supports multivariate Granger with lag selection.

**Installation:** `pip install causal-learn`

**Last updated:** Version 0.1.4.4, December 27, 2025 — very active.

**Multi-hop:** causal-learn discovers the full DAG — multi-hop paths are read from the graph structure after discovery. No special flag needed.

**Evidence:** github.com/py-why/causal-learn, JMLR v25 2023 paper, causal-learn.readthedocs.io

---

### 3.3 TCDF — Temporal Causal Discovery Framework (Deep Learning Approach)
**What it is:** PyTorch-based attention CNN that discovers causal relationships AND time delays between time series. Outputs a causal graph with explicit delay values.

**Why it matters for MIDGE:** TCDF directly outputs lag/delay per causal link — it answers "A causes B with a 3-week delay" from the data alone, without specifying lag candidates upfront. This is complementary to PCMCI+ (which requires you to specify max lag).

**Limitation:** Last commit was December 2018. The repo is essentially archived. PyTorch version compatibility may be an issue (was tested on PyTorch 0.4.1 — current is 2.x). Would require dependency updates.

**Verdict:** Research reference only. Don't use in production. The conceptual approach (attention CNN for lag discovery) is sound — but the implementation is stale.

**Evidence:** github.com/M-Nauta/TCDF (repo, last updated 2018)

---

### 3.4 SPACETIME — Non-Stationary Time Series Causal Discovery (2025)
**What it is:** A 2025 research paper/tool for causal discovery from non-stationary time series. Handles cases where the causal structure itself changes over time (regime changes).

**Why it matters for MIDGE:** Financial markets have regime changes — the causal chain from oil price to airline stocks behaves differently in inflationary vs. deflationary regimes. SPACETIME's non-stationarity handling is a direct match.

**Status:** Research paper (2025), not a packaged library yet. Code may be available from the authors' institution.

**Evidence:** CISPA Helmholtz Center paper: eda.rg.cispa.io/pubs/2025/spacetime-mameche,cornanguer,ninad,vreeken.pdf

**Verdict:** Watch for release. Flag for future expedition when packaged.

---

## Section 4: Knowledge Graph / In-Memory Graph Database

### 4.1 NetworkX DiGraph — Pure Python, Zero Infrastructure (RECOMMENDED)
**Already covered in Section 2.1.** The world-model graph does not need a graph database for MIDGE's scale. NetworkX holds millions of nodes in memory efficiently. For MIDGE's use case — a few thousand companies, commodities, and event types — NetworkX in-memory is the right choice.

**When to use a graph DB instead:** If the world-model grows to 100k+ nodes with complex queries that need indexes, or if multiple processes need concurrent access. Neither applies to MIDGE today.

---

### 4.2 Memgraph — In-Memory Graph Database With Python Driver (IF NEEDED)
**What it is:** An in-memory graph database that uses Cypher query language (same as Neo4j). C++ engine with Python driver. Supports streaming data and real-time updates.

**Why it matters for MIDGE:** If the world-model graph grows too large for pure NetworkX (unlikely in the near term), Memgraph is the best Neo4j alternative — fully open source, in-memory (fast), Python driver exists, runs on Windows via Docker (Docker already installed on Wardenclyffe).

**Python access:** `pip install gqlalchemy` (Memgraph's Python OGM) or raw Bolt protocol driver.

**Installation on Windows:** Requires Docker. Already available on Wardenclyffe.

**Evidence:** memgraph.com/blog/neo4j-alternative-what-are-my-open-source-db-options (current comparison article)

---

### 4.3 causal-learn + NetworkX as Combined World-Model
**The recommended architecture:** Use causal-learn/CausalFlow to discover causal links from MIDGE's historical signal data. Use those discovered links to populate a NetworkX DiGraph (the world-model). Use NDlib or EoN to simulate propagation through the graph when a trigger event fires. This is a complete, pure-Python, Windows-native stack.

---

## Section 5: Synthesis — Gap Analysis vs. MIDGE's Current State

| Gap | Tool | Status |
|-----|------|--------|
| Multi-hop chain structure | NetworkX DiGraph with delay/probability edge attributes | Ready, pip install |
| Corporate ownership graph | GLEIF Level 2 + pygleif | Free bulk download available |
| Commodity flow direction | UN Comtrade + comtradeapicall | Free tier, requires signup |
| Cascade propagation with timing | EoN (fast_nonMarkov_SIR) | Best fit — per-edge delay distributions |
| Cascade propagation (simpler) | NDlib (Independent Cascade) | Easier to use, less timing control |
| Auto-discover chains from history | CausalFlow (F-PCMCI, CAnDOIT) | Active, pip install, lag output |
| Multi-method DAG discovery | causal-learn (PC, FCI, GRaSP) | Active, pip install, Windows clean |
| Large-scale graph DB (future) | Memgraph via Docker | Docker already on Wardenclyffe |

---

## Section 6: Integration Recommendations

### How these tools fit MIDGE's existing architecture

**CausalReasoningEngine integration:** The existing engine's `_links` dict (cause→effect with strength/confidence) maps directly to NetworkX edge attributes. The `find_causal_path` method is currently BFS on dicts — replace with `nx.shortest_path` and `nx.all_simple_paths` to gain all standard graph algorithms free.

**New component needed — WorldModelGraph:** A single NetworkX DiGraph that MIDGE maintains as a persistent world model. Nodes registered for: known commodity types, sector categories, geographic regions, event types. Edges populated from: (a) GLEIF corporate structure, (b) Comtrade commodity flows, (c) Discovered causal links from CausalReasoningEngine, (d) Domain expert priors (encoded manually for key chains like energy→chemicals→manufacturing).

**New component needed — CascadeSimulator:** Wraps EoN's `fast_nonMarkov_SIR` against the WorldModelGraph. Input: trigger event node + severity. Output: probability distribution of affected nodes at each time step T. This answers "if Hurricane Harvey hits, what's the probability Ford ($F) is affected in 3 weeks vs. 6 weeks?"

**Connection to ConvergenceAlerter:** When CascadeSimulator identifies a downstream ticker with >threshold probability within a time window that matches a pattern MIDGE is watching, fire a convergence signal tagged "causal_chain" to the ConvergenceAlerter's domain set. The crown jewel picks it up and treats it as another domain signaling the same direction.

**Connection to PatternTemplate / fingerprint.py:** The lag_buckets in `fingerprint.py` (immediate/short/medium/long/extended) map to cascade simulation time steps. After simulation, the predicted timing for a downstream ticker's activation aligns with PrecursorSignal.lag_days. This allows the causal chain engine to generate synthetic PrecursorSignal objects that Pattern Archaeology can validate against historical outcomes.

### Recommended Build Order

1. Convert CausalReasoningEngine's internal dicts to a NetworkX DiGraph (non-breaking wrapper)
2. Build WorldModelGraph loader that ingests GLEIF + manual commodity flow priors
3. Implement CascadeSimulator using EoN on top of WorldModelGraph
4. Wire CascadeSimulator output to ConvergenceAlerter as a new signal domain
5. Integrate CausalFlow (F-PCMCI) to auto-discover new chains from MIDGE's historical signal data

---

## Sources

- GLEIF API and Level 2 Data: https://www.gleif.org/en/lei-data/access-and-use-lei-data/level-2-data-who-owns-whom
- pygleif Python library: https://pypi.org/project/pygleif/
- UN Comtrade official Python library: https://github.com/uncomtrade/comtradeapicall
- SupplyGraph benchmark dataset: https://arxiv.org/abs/2401.15299
- Open Supply Hub: https://opensupplyhub.org/
- NetworkX documentation: https://networkx.org/documentation/stable/
- NDlib Network Diffusion Library: https://github.com/GiulioRossetti/ndlib
- NDlib paper: https://arxiv.org/abs/1801.05854
- EoN (Epidemics on Networks): https://arxiv.org/abs/2001.02436
- EoN version 1.2 June 2024: https://joss.theoj.org/papers/10.21105/joss.01731.pdf
- CausalFlow unified framework: https://github.com/lcastri/causalflow
- CAnDOIT 2024 paper (Advanced Intelligent Systems): https://github.com/lcastri/causalflow
- causal-learn (py-why): https://github.com/py-why/causal-learn
- causal-learn JMLR 2023: https://jmlr.org/papers/v25/23-0970.html
- TCDF (archived): https://github.com/M-Nauta/TCDF
- SPACETIME 2025: https://eda.rg.cispa.io/pubs/2025/spacetime-mameche,cornanguer,ninad,vreeken.pdf
- Memgraph vs Neo4j: https://memgraph.com/blog/neo4j-alternative-what-are-my-open-source-db-options
- Salesforce CausalAI (archived May 2025): https://github.com/salesforce/causalai
- Financial contagion multiplex networks: https://royalsocietypublishing.org/doi/10.1098/rspa.2023.0787
- Cascade failure supply chain MDPI: https://www.mdpi.com/2079-8954/13/9/729
- Causal discovery from temporal data ACM survey: https://dl.acm.org/doi/10.1145/3705297
