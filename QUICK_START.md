# Quick Start Guide

From zero to running self-evolving AI system in 30 minutes.

---

## Prerequisites

| Component | Why Needed | Install |
|-----------|------------|---------|
| Python 3.8+ | Core runtime | [python.org](https://python.org) |
| Git | Version control / safety | [git-scm.com](https://git-scm.com) |
| Docker | Run Qdrant | [docker.com](https://docker.com) |
| DeepSeek API Key | Agent reasoning | [platform.deepseek.com](https://platform.deepseek.com) |

**Optional but recommended**:
- Ollama (local embeddings + oracle)

---

## Step 1: Clone and Initialize

```bash
# Clone this repository
git clone https://github.com/[your-username]/MIDGE.git my-research
cd my-research

# Initialize as your own project
rm -rf .git
git init
git add -A
git commit -m "Initial commit: MIDGE template"
```

---

## Step 2: Start Infrastructure

### Qdrant (Vector Database)

```bash
# Start Qdrant with Docker
docker run -d -p 6333:6333 -p 6334:6334 \
    --name qdrant \
    qdrant/qdrant

# Verify it's running
curl http://localhost:6333/collections
# Should return: {"result":{"collections":[]},"status":"ok",...}
```

### Ollama (Local Embeddings)

```bash
# Install Ollama: https://ollama.ai

# Pull required models
ollama pull nomic-embed-text    # For embeddings
ollama pull qwen2.5:1.5b        # For oracle (optional)

# Verify
ollama list
```

---

## Step 3: Get DeepSeek API Key

1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Create account (free)
3. Go to API Keys section
4. Create new key
5. Copy and save securely

**Cost**: ~$0.00035 per API call. Running 100 cycles costs approximately $0.14.

---

## Step 4: Install Shared Scripts

```bash
# Create scripts directory
mkdir -p ~/.claude/scripts

# Copy external bridge
cp shared/external_bridge.py ~/.claude/scripts/

# Verify
python -c "import sys; sys.path.insert(0, '$HOME/.claude/scripts'); from external_bridge import ExternalBridge; print('OK')"
```

---

## Step 5: Configure Your Research

### 5a. Set Your Research Question

Edit `templates/knowledge/limitations.md`:

```markdown
# Limitations

**I don't know [YOUR RESEARCH QUESTION]**

This is my primary limitation. Everything I learn
should work toward understanding this.

---

## Specific Unknowns

1. [Specific aspect 1]
2. [Specific aspect 2]
3. [Specific aspect 3]
```

**Examples**:
- "I don't know how neural networks learn representations"
- "I don't know what patterns exist in cryptocurrency markets"
- "I don't know how to detect misinformation automatically"

### 5b. Set Your Goals

Edit `templates/knowledge/goals.md`:

```markdown
# Goals

## Primary Goal
[What you want the system to achieve]

## Evolution Philosophy
1. [How should the system approach learning]
2. [What constraints should it respect]
3. [What should it avoid]

## Success Criteria
- [How will you know it's working]
```

### 5c. Set Your Project Name

Edit `templates/core/loop.py`:

```python
VARIANT = "my-project"              # Identifier (no spaces)
VARIANT_NAME = "MY PROJECT NAME"    # Display name
```

---

## Step 6: Run Your System

### Windows PowerShell

```powershell
cd templates
$env:DEEPSEEK_API_KEY="your-api-key-here"
python core/loop.py --continuous
```

### Linux / Mac

```bash
cd templates
DEEPSEEK_API_KEY="your-api-key-here" python core/loop.py --continuous
```

### Options

| Flag | Effect |
|------|--------|
| `--continuous` | Run forever with 15s delay |
| `--continuous 30` | Run forever with 30s delay |
| `--dry-run` | Don't actually modify files |
| (none) | Run single cycle |

---

## Step 7: Watch It Evolve

### Terminal Output

```
============================================================
  THE SWIFT PATH - Cycle 1
============================================================

--------------------------------------------------
  CHAPTER 1: THE MIRROR
--------------------------------------------------
  The system gazes inward, reading its own nature...

  Question: I don't know how neural networks...

--------------------------------------------------
  CHAPTER 2: THE TRIAD CONVENES
--------------------------------------------------
  Three voices speak in parallel. Then one decides.

  [BUILDER] Spoke
  [CRITIC] Spoke
  [SEEKER] Spoke

**************************************************
  THE SYNTHESIZER DECIDES
**************************************************
  Decision: Create a tool to analyze network layers
  Target: tools/layer_analyzer.py
  Risk: LOW

--------------------------------------------------
  CHAPTER 3: THE FORGE
--------------------------------------------------
  The decision becomes reality...

  SUCCESS: tools/layer_analyzer.py created

--------------------------------------------------
  CHAPTER 4: THE TEST
--------------------------------------------------
  Does the system still stand?

  VERIFIED: The system remains whole.

============================================================
  CYCLE COMPLETE
============================================================
```

### Log File

All output is also written to `templates/evolution.log`.

### Metrics

Check `templates/metrics.json`:

```json
{
  "variant": "my-project",
  "cycles_completed": 15,
  "files_created": 8,
  "files_modified": 4,
  "verification_passes": 14,
  "verification_fails": 1,
  "runtime_hours": 0.25
}
```

---

## Step 8: Start the Dashboard (Optional)

```bash
cd dashboard
python server.py
```

Open http://localhost:8080 to see:
- Real-time cycle monitoring
- Agent activity
- Voting history (Parliament mode)
- Narrative stories

---

## Troubleshooting

### "DEEPSEEK_API_KEY not set"

Make sure you're setting the environment variable correctly:
- Windows: `$env:DEEPSEEK_API_KEY="sk-xxx"`
- Linux/Mac: `export DEEPSEEK_API_KEY="sk-xxx"` or prefix the command

### "Connection refused" (Qdrant)

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start it if not
docker start qdrant
# or
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### "Module not found: external_bridge"

```bash
# Verify the file exists
ls ~/.claude/scripts/external_bridge.py

# If not, copy it
cp shared/external_bridge.py ~/.claude/scripts/
```

### "SyntaxError" in verification

The system created invalid Python. Check:
```bash
git log --oneline -5  # See recent changes
git diff HEAD~1       # See what changed
git reset --hard HEAD~1  # Rollback if needed
```

### Slow cycles (> 2 minutes)

Usually DeepSeek API latency. Try:
- Running during off-peak hours
- Reducing max_tokens in research.py
- Using a faster model

---

## Next Steps

Once your system is running:

1. **Watch 20 cycles** - Observe what it creates
2. **Read evolution.log** - Understand its decisions
3. **Check knowledge/** - See what it's learning
4. **Tune prompts** - Adjust agent behavior if needed
5. **Try Parliament mode** - For more rigorous validation

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python core/loop.py` | Run single cycle |
| `python core/loop.py --continuous` | Run forever |
| `python core/loop.py --dry-run` | Test without writing |
| `git log --oneline -10` | See recent changes |
| `git reset --hard HEAD~1` | Rollback last change |
| `docker start qdrant` | Start vector DB |
| `python dashboard/server.py` | Start dashboard |

---

*You're ready. Start your system and watch it evolve.*
