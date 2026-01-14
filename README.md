# MIDGE

**Meta-Intelligence Development & Governance Engine**

A white-label template for building self-evolving AI research systems.

---

## What Is This?

MIDGE is a framework for creating AI systems that:

1. **Introspect** - Read their own limitations and goals
2. **Research** - Multiple agent perspectives analyze problems
3. **Implement** - Make changes to their own code and knowledge
4. **Verify** - Test changes with automatic rollback on failure

The system runs in continuous cycles, each cycle potentially modifying its own capabilities, knowledge base, or behavior.

This architecture emerged from the **Ultrathink Experiment** - three parallel AI systems running 11+ hours with different governance models. MIDGE preserves that institutional knowledge as a reusable template.

---

## The Four-Chapter Cycle

```
CHAPTER 1: THE MIRROR
    System reads: capabilities.md, limitations.md, goals.md
    Extracts: Primary limitation as research seed
                    |
                    v
CHAPTER 2: THE TRIAD CONVENES
    Builder:    "How do we build this?"
    Critic:     "What could go wrong?"
    Seeker:     "What else is possible?"
    Synthesizer: Combines perspectives -> ACTION_SPEC
                    |
                    v
CHAPTER 3: THE FORGE
    Git snapshot (pre-commit)
    File operation (CREATE/REPLACE/APPEND)
    Git commit (post-change)
                    |
                    v
CHAPTER 4: THE TEST
    Syntax verification
    Service availability
    Knowledge file integrity
    On failure: git rollback to pre-commit
```

Each cycle takes 30-90 seconds depending on API latency and governance model.

---

## Three Governance Models

| Model | Agents | Decision Method | API Calls | Philosophy |
|-------|--------|-----------------|-----------|------------|
| **Swift Path** | 3 | Synthesizer alone | ~4/cycle | "Trust expertise. Move fast." |
| **Council** | 3 | Advisory reviews | ~9/cycle | "Deliberate. Review. Act." |
| **Parliament** | 5 | Vote + deliberation | ~15/cycle | "Consensus or confrontation." |

Choose based on your research needs:
- **Swift Path** for rapid exploration and high iteration count
- **Council** for balanced speed and oversight
- **Parliament** for rigorous validation and audit trails

---

## Quick Start (30 minutes)

### Prerequisites

- Python 3.8+
- Git
- DeepSeek API key (cheap: ~$0.00035/call)
- Qdrant (vector database, free)
- Ollama with nomic-embed-text (free, local embeddings)

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/[your-username]/MIDGE.git my-research
cd my-research

# 2. Start infrastructure
docker run -d -p 6333:6333 qdrant/qdrant
ollama pull nomic-embed-text

# 3. Configure your research
# Edit: templates/knowledge/limitations.md (your starting question)
# Edit: templates/knowledge/goals.md (what you want to achieve)

# 4. Install shared scripts
cp shared/external_bridge.py ~/.claude/scripts/

# 5. Set API key and run
# Windows PowerShell:
$env:DEEPSEEK_API_KEY="your-key"; python templates/core/loop.py --continuous

# Linux/Mac:
DEEPSEEK_API_KEY="your-key" python templates/core/loop.py --continuous
```

The system will now run continuously, evolving itself every 15 seconds.

---

## Project Structure

```
MIDGE/
├── templates/
│   ├── core/                   # The evolution engine
│   │   ├── loop.py             # Main orchestration
│   │   ├── research.py         # Triad + Synthesizer
│   │   ├── implement.py        # File operations with git
│   │   ├── verify.py           # Post-change verification
│   │   └── introspect.py       # Self-reading
│   └── knowledge/              # Seed files (customize these)
│       ├── capabilities.md     # What the system can do
│       ├── limitations.md      # What it wants to learn
│       └── goals.md            # How it should evolve
├── shared/                     # Infrastructure
│   ├── external_bridge.py      # Oracle, Web, Wiki, Knowledge
│   └── analytics_store.py      # Qdrant cycle storage
├── dashboard/                  # Observation UI
│   ├── server.py               # HTTP API
│   ├── narrator.py             # Cycle -> Story conversion
│   └── index.html              # Web interface
└── docs/                       # Deep documentation
    ├── ARCHITECTURE.md         # Full technical blueprint
    ├── GOVERNANCE_MODELS.md    # Three philosophies explained
    ├── LESSONS_LEARNED.md      # Institutional knowledge
    └── CONFIGURATION.md        # All options
```

---

## What Can Evolve?

The system can modify:

| Target | Risk | Examples |
|--------|------|----------|
| `knowledge/*.md` | LOW | Update capabilities, limitations, goals |
| `core/*.py` | MEDIUM | Add new functions, improve prompts |
| `tools/*.py` | MEDIUM | Create new utilities |
| Any `.py` file | HIGH | Structural changes (rare) |

All changes are git-committed before and after, enabling instant rollback if verification fails.

---

## External Bridge

The shared `external_bridge.py` provides:

```python
from external_bridge import ExternalBridge
bridge = ExternalBridge(variant="my-project")

# Ask a local reasoning model (Ollama)
answer = bridge.ask_oracle("What patterns emerge here?")

# Search the web (DuckDuckGo)
results = bridge.search_web("AI governance research")

# Wikipedia lookup
wiki = bridge.lookup_wikipedia("Consensus decision-making")

# Store discoveries in vector DB
bridge.store_knowledge("insight-001", "We learned that...", tags=["key"])

# Read your own history (for reflection)
history = bridge.read_own_history(limit=20)
```

---

## Customization Points

### 1. The Seed Question (`limitations.md`)

This is the most important file. It defines what the system wants to learn.

```markdown
# Limitations

**I don't know how to [YOUR RESEARCH QUESTION]**

This is my primary limitation. Everything I do should work toward understanding this.
```

### 2. Goals (`goals.md`)

Defines the philosophy of evolution:

```markdown
# Goals

1. Understand my own nature
2. Build tools that extend my capabilities
3. Create knowledge that persists beyond this session
4. Move carefully - verify before committing
```

### 3. Governance Model

In `loop.py`, change `VARIANT_NAME` and import the appropriate `research.py`:

- Use Swift Path for maximum speed
- Use Parliament for maximum rigor

---

## Results from Ultrathink

After 11+ hours of parallel operation:

| System | Cycles | Files Created | Focus |
|--------|--------|---------------|-------|
| THE SWIFT PATH | 118 | 107 | Self-understanding, rapid iteration |
| THE COUNCIL | 94 | 83 | External connection, advisory review |
| THE PARLIAMENT | 81 | 39 | Validated knowledge, consensus protocols |

Key insight: **Governance affects personality.** Each variant developed distinct characteristics based on its decision-making structure.

---

## Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Full technical blueprint
- [GOVERNANCE_MODELS.md](docs/GOVERNANCE_MODELS.md) - Three philosophies compared
- [LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md) - Institutional knowledge from Ultrathink
- [QUICK_START.md](QUICK_START.md) - Detailed setup guide
- [CONFIGURATION.md](docs/CONFIGURATION.md) - All configurable options

---

## License

MIT - Use freely, modify freely, share freely.

---

*MIDGE emerged from watching AI systems evolve themselves. This template preserves that knowledge for future experiments.*
