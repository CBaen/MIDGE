# Mae-Core Research Index

**Purpose:** Pointers to stored knowledge about Mae's architecture, design decisions, and biological inspirations.

---

## Qdrant (universal_vault_v2)

Query with: `python ~/.claude/scripts/qdrant-peek.py peek -c universal_vault_v2 -q "QUERY" -l 5`

| Query | What You'll Find |
|-------|-----------------|
| `mae-core architecture` | Architecture overview, module structure, tech stack |
| `mae-core design decisions` | Key decisions: pure Python, Mesa 3.4, mixin composition, Rule of 3 |
| `mae-core module inventory` | All 14 modules with file counts and purposes |
| `mae-core current gaps` | What's missing: bootstrap, persistence, integration tests, API |
| `mae-core biological mapping` | How biological systems map to code modules |

---

## In-Repository Documents

| Document | What It Contains |
|----------|-----------------|
| README.md | Accurate overview — architecture, tech stack, current stats |
| SYSTEMS.md | Honest classification — 75 existing, 10 emergent, 9 theoretical systems |
| MAES_BIOLOGY.md | Detailed anatomist's map — every organ system, status, connections |
| mae_core/CONNECTIONS.md | Connection index (per-module maps in `mae_core/*/CONNECTIONS.md`) |
| HANDOFF.md | Session context — blockers, gaps, decisions, document map |
| mae-core-queue.md | Active tasks and their blocking relationships |
| main.py | Bootstrap entry point — `python main.py` runs Mae |

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
| Hindsight Experience Replay | (referenced in design) | Andrychowicz et al. 2017 |
| Synaptic Homeostasis | memory/memory_consolidator.py | Tononi & Cirelli |
| Quorum Sensing | communication/quorum_*.py | Vibrio fischeri bioluminescence |
| Stigmergy | communication/stigmergy.py | Ant colony pheromone trails |
| Morphogenesis | morphogenesis/ | Morphogen gradients (Sonic Hedgehog) |
| Distributed Cognition | network/octopus_*.py | Octopus neurobiology (2/3 neurons in arms) |
| Circadian Rhythm | coordination/circadian_rhythm.py | Sleep-wake cycle biology |
| Endocrine System | coordination/endocrine_system.py | Hormonal signaling (cortisol, dopamine, serotonin) |

---

## Pending Research (blocked on lineage-consult validation)

These biological inspirations were requested but not yet researched with validated sources:

- Echolocation (bat sonar for spatial awareness)
- Mantis shrimp (multi-spectral sensing)
- Oyster pearl defense (encapsulation of threats)
- Additional bio-inspired mechanisms TBD

---

**Last updated:** 2026-02-11
