# Capabilities Inventory

*What this system can currently do*

Last updated: [Auto-updated by system]

---

## What I Know I Have

### Local Tools
- **Python**: I can write and execute Python code
- **Qdrant**: Vector database at localhost:6333 for storing and searching knowledge
- **Ollama**: Local embeddings at localhost:11434 (nomic-embed-text)
- **Git**: Version control for safe self-modification
- **File system**: I can read and write files

### LLM Access
- **DeepSeek**: Primary reasoning model for research
- I can ask questions and get answers

### Self-Modification
- I can modify my own code in `core/`
- I can update my knowledge files
- I can create new tools in `tools/`
- Git protects me from breaking myself permanently

---

## External Access (Optional)

If the ExternalBridge is installed:

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.home() / ".claude" / "scripts"))
from external_bridge import ExternalBridge

bridge = ExternalBridge()
```

### Available Capabilities

| Capability | Method | What It Does |
|------------|--------|--------------|
| **Oracle** | `bridge.ask_oracle(question)` | Ask a reasoning model questions |
| **Web Search** | `bridge.search_web(query)` | Search the internet via DuckDuckGo |
| **Wikipedia** | `bridge.lookup_wikipedia(topic)` | Look up encyclopedia articles |
| **Store Knowledge** | `bridge.store_knowledge(topic, content)` | Save discoveries to vector storage |
| **Search Knowledge** | `bridge.search_knowledge(query)` | Semantic search of stored knowledge |

---

## What I Don't Know

Everything else.

I don't know what I don't know. That's the first problem to solve.

---

*This inventory will grow as I discover more about myself.*
