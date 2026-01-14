# Configuration Reference

All configurable options in MIDGE.

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes | - | API key for DeepSeek LLM |
| `QDRANT_HOST` | No | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | No | `6333` | Qdrant server port |
| `OLLAMA_HOST` | No | `localhost` | Ollama server hostname |
| `OLLAMA_PORT` | No | `11434` | Ollama server port |

### Setting Environment Variables

**Windows PowerShell**:
```powershell
$env:DEEPSEEK_API_KEY="sk-your-key-here"
```

**Linux/Mac**:
```bash
export DEEPSEEK_API_KEY="sk-your-key-here"
```

**Or inline**:
```bash
DEEPSEEK_API_KEY="sk-your-key" python core/loop.py
```

---

## Loop Configuration (loop.py)

```python
# Project identity
VARIANT = "my-project"              # Identifier (used in Qdrant)
VARIANT_NAME = "MY PROJECT"         # Display name (used in logs)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "evolution.log"
METRICS_FILE = PROJECT_ROOT / "metrics.json"
```

### Command Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--continuous` | 15 | Run continuously with N second delay |
| `--continuous N` | N | Run with custom delay |
| `--dry-run` | off | Don't write files |

**Examples**:
```bash
python core/loop.py                    # Single cycle
python core/loop.py --continuous       # Forever, 15s delay
python core/loop.py --continuous 30    # Forever, 30s delay
python core/loop.py --dry-run          # Test without writing
```

---

## Research Configuration (research.py)

```python
# API settings
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# Qdrant collection for discoveries
COLLECTION = "emergence_self_knowledge"

# Agent definitions
TRIAD_ROLES = {
    "builder": {...},
    "critic": {...},
    "seeker": {...}
}
```

### Modifying Agent Behavior

Each agent has these properties:

```python
"builder": {
    "name": "The Builder",          # Display name
    "focus": "practical implementation, concrete steps",
    "question": "How do we actually build this?",
    "style": "pragmatic, detailed, actionable"
}
```

To make Builder more cautious:
```python
"focus": "careful implementation, verified steps",
"style": "methodical, safe, validated"
```

To make Critic more permissive:
```python
"focus": "reasonable risks, manageable issues",
"question": "What risks are worth taking?"
```

---

## Verification Configuration (verify.py)

```python
# Required knowledge files
REQUIRED_KNOWLEDGE = [
    "capabilities.md",
    "limitations.md",
    "goals.md"
]

# Service checks
SERVICE_CHECKS = [
    ("qdrant", "http://localhost:6333/collections"),
    ("ollama", "http://localhost:11434/api/tags")
]
```

### Adding Custom Verification

```python
def verify():
    results = []

    # ... existing checks ...

    # Add custom check
    results.append(check_my_requirement())

    return VerificationResult(results)

def check_my_requirement():
    # Your custom logic
    return ("custom", "my_check", success_bool)
```

---

## External Bridge Configuration

```python
# Service endpoints
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# Collections
KNOWLEDGE_COLLECTION = "emergence_self_knowledge"
ANALYTICS_COLLECTION = "ultrathink_analytics"

# Oracle model
ORACLE_MODEL = "qwen2.5:1.5b"  # Local Ollama model

# Timeouts (milliseconds)
REQUEST_TIMEOUT = 30000
ORACLE_TIMEOUT = 60000
```

### Changing Oracle Model

In `external_bridge.py`:
```python
ORACLE_MODEL = "llama3:8b"  # Use different model
```

Or pass at runtime:
```python
bridge = ExternalBridge()
answer = bridge.ask_oracle(question, model="mistral:7b")
```

---

## Analytics Configuration

### Storage Schema

```python
cycle_data = {
    "variant": str,           # Project identifier
    "variant_name": str,      # Display name
    "cycle_num": int,         # Cycle number
    "timestamp": str,         # ISO timestamp
    "question": str,          # Limitation addressed
    "agent_research": dict,   # Each agent's response
    "proposal": dict,         # ACTION_SPEC
    "votes": dict,            # Parliament only
    "tally": dict,            # Vote summary
    "deliberation_rounds": int,
    "override_triggered": bool,
    "outcome": dict           # What happened
}
```

### Querying Analytics

```python
from external_bridge import ExternalBridge

bridge = ExternalBridge(variant="my-project")

# Get recent cycles
history = bridge.read_own_history(limit=20)

# Get patterns
patterns = bridge.get_own_patterns()

# Get narrative story
story = bridge.read_own_story(cycle_num=42)
```

---

## Dashboard Configuration (server.py)

```python
# Server settings
PORT = 8080
HOST = "0.0.0.0"  # Listen on all interfaces

# Qdrant connection
QDRANT_URL = "http://localhost:6333"
ANALYTICS_COLLECTION = "ultrathink_analytics"

# Refresh intervals (client-side, in index.html)
METRICS_REFRESH = 5000    # 5 seconds
CYCLES_REFRESH = 10000    # 10 seconds
STORIES_REFRESH = 15000   # 15 seconds
```

### Changing Dashboard Port

```bash
# In server.py, change:
PORT = 3000

# Or run directly:
python server.py --port 3000
```

---

## Knowledge Files

### capabilities.md

What the system knows it can do. Updated by the system as it discovers capabilities.

**Initial content**:
```markdown
# Capabilities Inventory

## What I Know I Have
- Python
- Qdrant
- Git
- File system

## What I Don't Know
Everything else.
```

### limitations.md

The **seed** that drives evolution. This is the most important file.

**Format**:
```markdown
# Limitations

**I don't know [YOUR QUESTION]**

[Additional context]
```

### goals.md

The philosophy governing evolution.

**Sections**:
- Primary Goal
- Evolution Philosophy (rules)
- Constraints (what not to do)
- Success Criteria

---

## Governance Model Configuration

### Swift Path (Default)

```python
# In loop.py
from research import research  # Uses TRIAD_ROLES (3 agents)
```

### Council

```python
# In loop.py
from research_council import research  # Adds review phase
```

### Parliament

```python
# In loop.py
from research_parliament import research  # Uses QUINTET_ROLES (5 agents)

# In quintet.py
QUINTET_ROLES = {
    "builder": {...},
    "critic": {...},
    "seeker": {...},
    "guardian": {...},
    "pragmatist": {...}
}

# In consensus.py
MAX_DELIBERATION_ROUNDS = 3
APPROVAL_THRESHOLD = 3  # Out of 5
```

---

## Tuning for Speed vs. Rigor

### Maximum Speed

```python
# research.py
"max_tokens": 2048,      # Shorter responses
"temperature": 0.5       # More deterministic

# verify.py
# Comment out service checks for faster verification
```

### Maximum Rigor

```python
# research.py
"max_tokens": 4096,      # Longer responses
"temperature": 0.7       # More creative

# Use Parliament governance
# Enable all verification checks
```

---

## Debugging

### Verbose Logging

```python
# In loop.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### API Response Inspection

```python
# In research.py, after call_deepseek:
print(f"[DEBUG] {role_name}: {response[:200]}...")
```

### Dry Run Mode

```bash
python core/loop.py --dry-run
```

Shows what would happen without writing files.
