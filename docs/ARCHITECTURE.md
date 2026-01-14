# Architecture

The complete technical blueprint for self-evolving AI research systems.

---

## System Overview

```
                    ┌─────────────────────────────────────────────┐
                    │                MIDGE SYSTEM                  │
                    │                                              │
   ┌────────────────┤  knowledge/     core/        tools/          │
   │                │  ├─capabilities ├─loop.py    ├─(generated)   │
   │ External       │  ├─limitations  ├─research   └─...           │
   │ Bridge         │  └─goals        ├─implement                  │
   │                │                 └─verify                     │
   │ ├─Oracle       │                                              │
   │ ├─Web          └──────────────────────────────────────────────┘
   │ ├─Wikipedia                          │
   │ └─Knowledge                          │ reads/writes
   │                                      ▼
   │                ┌─────────────────────────────────────────────┐
   └───────────────►│              Qdrant Vector DB               │
                    │  emergence_self_knowledge (discoveries)      │
                    │  ultrathink_analytics (cycle history)        │
                    └─────────────────────────────────────────────┘
```

---

## The Four Chapters

### Chapter 1: THE MIRROR (Introspection)

**Purpose**: Read the system's current state to identify what needs to change.

**Input**: Nothing - begins fresh each cycle

**Process**:
```python
def introspect():
    capabilities = read("knowledge/capabilities.md")[:500]
    limitations = read("knowledge/limitations.md")[:500]
    goals = read("knowledge/goals.md")[:500]
    return {capabilities, limitations, goals}
```

**Output**: A `state` dict + extracted `limitation` string

**Key Design Decision**: Truncation to 500 chars prevents context bloat while preserving essential information.

---

### Chapter 2: THE TRIAD CONVENES (Research)

**Purpose**: Three perspectives analyze the limitation, then synthesize into action.

**Agents**:

| Agent | Name | Focus | Question |
|-------|------|-------|----------|
| `builder` | The Builder | Practical implementation | "How do we actually build this?" |
| `critic` | The Critic | Risks and failure modes | "What could go wrong?" |
| `seeker` | The Seeker | Unexplored possibilities | "What else is possible?" |

**Execution**: Parallel API calls to DeepSeek
```python
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        executor.submit(call_deepseek, builder_prompt): "builder",
        executor.submit(call_deepseek, critic_prompt): "critic",
        executor.submit(call_deepseek, seeker_prompt): "seeker"
    }
```

**Synthesis**: Fourth API call combines all three perspectives
```
THE SYNTHESIZER reviews:
├── Builder's concrete steps
├── Critic's risk assessment
└── Seeker's alternatives
          │
          ▼
    ACTION_SPEC (single atomic action)
```

**Output**:
```python
{
    "triad": {"builder": "...", "critic": "...", "seeker": "..."},
    "synthesis": "...",
    "action_spec": {
        "decision": "Create a tool for X",
        "why": "Builder suggested, Critic approved, Seeker enhanced",
        "risk": "LOW",
        "file_path": "tools/x_tool.py",
        "action": "CREATE",
        "content": "# The actual file content..."
    }
}
```

---

### Chapter 3: THE FORGE (Implementation)

**Purpose**: Execute the ACTION_SPEC with git safety.

**Git Safety Protocol**:
```
1. git add -A && git commit -m "Pre-cycle snapshot"
2. Execute file operation
3. git add -A && git commit -m "Cycle N: {decision}"
```

**File Operations**:

| Action | Behavior |
|--------|----------|
| `CREATE` | Write new file (fails if exists) |
| `REPLACE` | Overwrite entire file |
| `APPEND` | Add content to end of file |

**Code**:
```python
def implement(action_spec, dry_run=False):
    # Pre-commit
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", "Pre-cycle snapshot"])

    # Execute
    path = PROJECT_ROOT / action_spec["file_path"]
    if action_spec["action"] == "CREATE":
        if path.exists():
            return {"success": False, "error": "File exists"}
        path.write_text(action_spec["content"])
    elif action_spec["action"] == "REPLACE":
        path.write_text(action_spec["content"])
    elif action_spec["action"] == "APPEND":
        existing = path.read_text() if path.exists() else ""
        path.write_text(existing + "\n" + action_spec["content"])

    # Post-commit
    subprocess.run(["git", "add", "-A"])
    subprocess.run(["git", "commit", "-m", f"Cycle: {action_spec['decision']}"])

    return {"success": True}
```

---

### Chapter 4: THE TEST (Verification)

**Purpose**: Ensure the system still works after changes.

**Checks**:
1. **Syntax**: All `.py` files parse without errors
2. **Knowledge**: Required `.md` files exist
3. **Services**: Qdrant and Ollama respond

**On Failure**: Automatic rollback
```python
if not verify_result["success"]:
    subprocess.run(["git", "reset", "--hard", "HEAD~1"])
```

**Code**:
```python
def verify():
    results = []

    # Syntax check
    for py_file in PROJECT_ROOT.glob("**/*.py"):
        try:
            compile(py_file.read_text(), py_file, "exec")
            results.append(("syntax", py_file.name, True))
        except SyntaxError as e:
            results.append(("syntax", py_file.name, False, str(e)))

    # Knowledge files
    for required in ["capabilities.md", "limitations.md", "goals.md"]:
        path = PROJECT_ROOT / "knowledge" / required
        results.append(("knowledge", required, path.exists()))

    # Services
    try:
        requests.get("http://localhost:6333/collections", timeout=5)
        results.append(("service", "qdrant", True))
    except:
        results.append(("service", "qdrant", False))

    return VerificationResult(results)
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ CYCLE START                                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ INTROSPECT                    │
                    │ Read: capabilities.md         │
                    │ Read: limitations.md          │
                    │ Read: goals.md                │
                    └───────────────────────────────┘
                                    │
                              limitation
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ RESEARCH (parallel)                                │
        │ ┌─────────┐ ┌─────────┐ ┌─────────┐               │
        │ │ BUILDER │ │ CRITIC  │ │ SEEKER  │  → DeepSeek   │
        │ └────┬────┘ └────┬────┘ └────┬────┘               │
        │      └───────────┼───────────┘                     │
        │                  ▼                                 │
        │          ┌─────────────┐                           │
        │          │ SYNTHESIZER │  → DeepSeek               │
        │          └──────┬──────┘                           │
        └─────────────────┼─────────────────────────────────┘
                          │
                    ACTION_SPEC
                          │
                          ▼
                    ┌───────────────────────────────┐
                    │ IMPLEMENT                     │
                    │ 1. Git pre-commit             │
                    │ 2. File operation             │
                    │ 3. Git post-commit            │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ VERIFY                        │
                    │ Check: syntax, files, services│
                    │ On fail: git reset --hard     │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ ANALYTICS                     │
                    │ Store cycle data to Qdrant   │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ CYCLE END - Wait 15 seconds - Repeat                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Prompt Structure

### Triad Agent Prompt Template

```
You are {ROLE_NAME} in {VARIANT_NAME}, a self-evolving AI system.

YOUR ROLE: {role_name}
YOUR FOCUS: {role_focus}
YOUR GUIDING QUESTION: {role_question}

THE TRIAD: Two others research this simultaneously:
- {other_role_1}: {focus_1}
- {other_role_2}: {focus_2}

Focus on YOUR angle. Trust your partners.

---
LIMITATION TO ADDRESS:
{limitation}

SYSTEM GOALS:
{goals}

CURRENT CAPABILITIES:
{capabilities}
---

OUTPUT FORMAT:
---{ROLE_KEY}---
## Analysis
[Your perspective]

## Key Insights
[What others might miss]

## Recommendations
[Concrete suggestions]
---END {ROLE_KEY}---
```

### Synthesizer Prompt Template

```
You are THE SYNTHESIZER in {VARIANT_NAME}.

Your job: Review all three perspectives and produce ONE clear ACTION_SPEC.
This ACTION_SPEC goes DIRECTLY to implementation. No more discussion.

THE TRIAD HAS SPOKEN:

=== THE BUILDER ===
{builder_response}

=== THE CRITIC ===
{critic_response}

=== THE SEEKER ===
{seeker_response}

=== THE QUESTION ===
{limitation}

YOUR TASK:
1. Identify the SINGLE most important action
2. Produce an ACTION_SPEC that can be executed directly
3. Be SPECIFIC - file path, action type, complete content

OUTPUT FORMAT (follow EXACTLY):

---SYNTHESIS---
## Agreement Points
[What they agree on]

## Conflicts Resolved
[Where they disagreed, your decision]

## Chosen Action
[Why this specific action]
---END SYNTHESIS---

---ACTION_SPEC---
decision: [One sentence - what we're doing]
why: [Why this is the right next step]
risk: LOW|MEDIUM|HIGH
file_path: [relative path]
action: CREATE|REPLACE|APPEND
---END ACTION_SPEC---

---CONTENT---
[Complete file content to write]
---END CONTENT---
```

---

## Analytics Schema

Cycles are stored in Qdrant with this structure:

```python
{
    "variant": "simple",                    # Project identifier
    "variant_name": "THE SWIFT PATH",       # Display name
    "cycle_num": 42,                        # Cycle number
    "timestamp": "2026-01-13T15:30:00",     # ISO timestamp

    "question": "How can I understand X?",  # The limitation addressed

    "agent_research": {
        "builder": {"spoke": True, "summary": "..."},
        "critic": {"spoke": True, "summary": "..."},
        "seeker": {"spoke": True, "summary": "..."}
    },

    "proposal": {
        "decision": "Create X tool",
        "target_file": "tools/x.py",
        "action": "CREATE",
        "risk": "LOW"
    },

    "votes": {                              # Parliament only
        "marcus": "approve",
        "elena": "object",
        "quinn": "approve",
        "gideon": "approve",
        "petra": "approve"
    },

    "tally": {
        "approve": 4,
        "object": 1,
        "proceed": True
    },

    "deliberation_rounds": 0,               # Parliament deliberation
    "override_triggered": False,            # Sophia override

    "outcome": {
        "file_created": "tools/x.py",
        "action_taken": True,
        "verification_passed": True
    }
}
```

---

## External Bridge API

The `ExternalBridge` class provides controlled external access:

```python
class ExternalBridge:
    def __init__(self, variant: str = None, verbose: bool = False):
        """Initialize with optional variant for analytics access."""

    def status(self) -> dict:
        """Check availability of all services."""

    # Knowledge Operations
    def ask_oracle(self, question: str) -> str:
        """Ask local Ollama model (qwen2.5:1.5b)."""

    def search_web(self, query: str) -> list[dict]:
        """Search DuckDuckGo, returns [{title, url, snippet}]."""

    def lookup_wikipedia(self, topic: str) -> str:
        """Get Wikipedia article summary."""

    def store_knowledge(self, topic: str, content: str, tags: list = None):
        """Store to Qdrant vector DB."""

    def search_knowledge(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search of stored knowledge."""

    # Self-Analytics (read-only)
    def read_own_history(self, limit: int = 20) -> list[dict]:
        """Read your variant's cycle history."""

    def read_own_story(self, cycle_num: int = None) -> str:
        """Get narrative story of a cycle."""

    def get_own_patterns(self) -> dict:
        """Analyze patterns in your governance history."""
```

---

## Service Dependencies

| Service | Port | Purpose | Required |
|---------|------|---------|----------|
| **Qdrant** | 6333 | Vector database | Yes |
| **Ollama** | 11434 | Local embeddings + Oracle | Yes |
| **DeepSeek API** | - | Agent reasoning | Yes |

### Qdrant Collections

| Collection | Purpose |
|------------|---------|
| `emergence_self_knowledge` | Shared discoveries |
| `ultrathink_analytics` | Cycle history (per-variant) |

### Ollama Models

| Model | Purpose |
|-------|---------|
| `nomic-embed-text` | Embedding generation |
| `qwen2.5:1.5b` | Oracle (optional) |

---

## Configuration

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DEEPSEEK_API_KEY` | Yes | - | LLM API access |
| `QDRANT_HOST` | No | localhost | Qdrant server |
| `QDRANT_PORT` | No | 6333 | Qdrant port |
| `OLLAMA_HOST` | No | localhost | Ollama server |
| `OLLAMA_PORT` | No | 11434 | Ollama port |

### Loop Configuration

In `loop.py`:

```python
VARIANT = "my-project"              # Project identifier
VARIANT_NAME = "MY PROJECT"         # Display name
# --continuous [N]                  # Delay between cycles (default 15s)
# --dry-run                         # Don't write files
```

---

## Extension Points

### Adding Agents

In `research.py`, add to `TRIAD_ROLES`:

```python
TRIAD_ROLES = {
    "builder": {...},
    "critic": {...},
    "seeker": {...},
    "your_agent": {
        "name": "The Your Agent",
        "focus": "specific focus area",
        "question": "What question drives this agent?"
    }
}
```

### Adding Verification Checks

In `verify.py`, add to the checks list:

```python
def verify():
    checks = [
        check_syntax(),
        check_knowledge(),
        check_services(),
        check_your_custom_thing(),  # Add here
    ]
```

### Adding External Capabilities

In `external_bridge.py`:

```python
def your_new_capability(self, param):
    """Your new external access method."""
    # Implementation
```

---

## Safety Guarantees

1. **Git Rollback**: Every change can be undone with `git reset --hard HEAD~1`
2. **Verification**: Syntax errors are caught before deployment
3. **Isolation**: Each variant operates in its own project directory
4. **Read-Only Analytics**: Systems can read their history but not modify it
5. **Rate Limiting**: External bridge has built-in timeouts and limits
