# MIDGE Research Index

**Purpose:** Pointers to stored knowledge about MIDGE's architecture, market intelligence, and biological inspirations.

> MIDGE is a trading-specialized fork of mae-core. Market intelligence lives in `mae_core/market/`. Everything else is Mae's universal genome.

---

## Qdrant (universal_vault_v2)

Query with: `python C:/Users/baenb/.claude/scripts/qdrant-peek.py peek -c universal_vault_v2 -q "QUERY" -l 5`

| Query | What You'll Find |
|-------|-----------------|
| `mae-core architecture` | Architecture overview, module structure, tech stack |
| `mae-core design decisions` | Key decisions: pure Python, Mesa 3.4, mixin composition, Rule of 3 |
| `mae-core module inventory` | All 14 modules with file counts and purposes |
| `mae-core biological mapping` | How biological systems map to code modules |
| `MIDGE market intelligence` | Market APIs, edge detectors, Thompson Sampling |

---

## In-Repository Documents

| Document | What It Contains |
|----------|-----------------|
| README.md | MIDGE overview — mae-core foundation + market intelligence |
| CLAUDE.md | Project rules, market package structure, key files |
| HANDOFF.md | Session context — what happened, what's next, integration status |
| SYSTEMS.md | System classification — 85 systems across 14 modules |
| MAES_BIOLOGY.md | Anatomist's map — every organ system, status, connections |
| mae_core/CONNECTIONS.md | Connection index (per-module maps in `mae_core/*/CONNECTIONS.md`) |
| midge-queue.md | Active tasks and their blocking relationships |
| midge-decisions.md | Append-only decision log |
| main.py | Bootstrap entry point — `python main.py` runs Mae |

---

## Market Intelligence (MIDGE-specific)

| Module | Location | What It Does |
|--------|----------|-------------|
| SEC EDGAR | `mae_core/market/apis/sec_edgar/` | Form 4 insider trades, Form 8-K material events |
| Price Fetcher | `mae_core/market/apis/price_fetcher.py` | yfinance + Alpha Vantage fallback |
| Congressional Trades | `mae_core/market/apis/house_stock_watcher.py` | STOCK Act disclosures |
| Job Tracker | `mae_core/market/apis/job_tracker.py` | RapidAPI hiring blitz detection |
| USA Spending | `mae_core/market/apis/usa_spending.py` | Government contract search |
| SAM.gov | `mae_core/market/apis/sam_gov.py` | Federal contracting opportunities |
| Cluster Detector | `mae_core/market/edge/cluster_detector.py` | Insider buying clusters |
| Politician Tracker | `mae_core/market/edge/politician_tracker.py` | Congress trade + contract correlation |
| Filing Time Analyzer | `mae_core/market/edge/filing_time_analyzer.py` | SEC filing behavioral signals |
| Contract Predictor | `mae_core/market/edge/contract_predictor.py` | Pre-announcement winner prediction |
| Thompson Sampler | `mae_core/market/intelligence/thompson_sampler.py` | Bayesian explore/exploit |
| Velocity Detector | `mae_core/market/intelligence/velocity_detector.py` | Rate-of-change anomalies |
| Correlation Tracker | `mae_core/market/intelligence/correlation_tracker.py` | Cross-domain signal correlation |
| Convergence Alerter | `mae_core/market/intelligence/convergence_alerter.py` | Multi-domain synthesis (crown jewel) |
| Learning Config | `mae_core/market/intelligence/learning_config.py` | Self-modifiable parameters |

---

## Learned Data (data/market/)

| File | What It Contains |
|------|-----------------|
| `thompson_distributions.json` | 30+ signal Beta(alpha, beta) parameters |
| `predictions.jsonl` | Historical predictions made |
| `outcomes.jsonl` | Ground truth outcomes |
| `discovery_log.jsonl` | Novel pattern discoveries |
| `config_history.jsonl` | Learning config evolution |

---

## Biological Inspirations (Research Sources)

These are the biological mechanisms that drive Mae's architecture. Papers cited in codebase:

| Mechanism | Code Module | Key Papers |
|-----------|------------|------------|
| Prioritized Experience Replay | memory/prioritized_replay_buffer.py | Schaul et al. 2016 |
| Generative Replay (anti-forgetting) | memory/generative_replay.py | Shin et al. 2017, Kingma & Welling 2014 |
| Causal Reasoning | cognition/causal_reasoning.py | Pearl's causal hierarchy |
| Meta-Learning (MAML) | learning/maml.py | Finn et al. 2017 |
| Working Memory capacity | memory/working_memory.py | Miller 1956 (7 +/- 2) |
| Synaptic Homeostasis | memory/memory_consolidator.py | Tononi & Cirelli |
| Quorum Sensing | communication/quorum_*.py | Vibrio fischeri bioluminescence |
| Stigmergy | communication/stigmergy.py | Ant colony pheromone trails |
| Morphogenesis | morphogenesis/ | Morphogen gradients (Sonic Hedgehog) |
| Distributed Cognition | network/octopus_*.py | Octopus neurobiology (2/3 neurons in arms) |
| Circadian Rhythm | coordination/circadian_rhythm.py | Sleep-wake cycle biology |
| Endocrine System | coordination/endocrine_system.py | Hormonal signaling (cortisol, dopamine, serotonin) |
| Thompson Sampling | market/intelligence/thompson_sampler.py | Thompson 1933, Bayesian bandits |

---

**Last updated:** 2026-02-22
