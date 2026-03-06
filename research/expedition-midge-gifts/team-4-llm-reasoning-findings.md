# Team 4 Findings: LLM-Powered Reasoning via Ollama
## Expedition: Gifts for Midge
## Date: 2026-03-05
## Angle: Giving Midge Internal Dialogue

---

## Executive Summary

Ollama integration into Midge is achievable with minimal new infrastructure. The `OpenAIProvider` class is the integration point — Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1`, which means a new `OllamaProvider` is one parameter swap away from the existing pattern. The recommended model for Midge's use case is **Qwen3-14B** (Q4_K_M quantized, ~10-12GB VRAM) as the primary reasoning engine, falling back to **Qwen3-8B** for speed-critical passes. For structured output extraction, **Instructor + Pydantic** is the clear winner for reliability at local scale.

The critical architectural insight: LLM reasoning should hook into three existing Midge systems — the HypothesisEngine (narrative generation at `has_causal_story` creation time), the `CausalReasoningEngine` (enriching `explain_causation()` with LLM-generated language), and the `plain_language.py` formatter (adding a "WHY" section to convergence alerts).

---

## Research Area 1: Model Selection for Financial Reasoning

### Recommended Primary: Qwen3-14B (Q4_K_M)

**Why Qwen3-14B over the alternatives:**
- MMLU score: 74.2+ for 7B variant; 14B significantly higher — competitive with models twice its size
- Qwen family is the most quantization-tolerant tested: accuracy difference between BF16 and Q4_K_M is under 1% on reasoning benchmarks (Ionio.ai benchmark study, 2025)
- VRAM requirement: ~10-12GB at Q4_K_M — fits on common consumer GPUs (RTX 4070, 4080, 4090)
- Strong multilingual grounding makes it better at parsing noisy financial text (SEC filings, earnings transcripts often have unusual formatting)
- Qwen3 235B MoE variant (22B active per token) exists for Wardenclyffe if the GPU supports it, but 14B is the practical daily driver
- Native tool calling support in Ollama confirmed

**Ollama pull command:** `ollama pull qwen3:14b`

### Recommended Secondary/Fast Path: Qwen3-8B (Q4_K_M)

- ~6-8GB VRAM — fits on RTX 4070 or below
- ~40+ tokens/second generation speed
- Use for: quick causal label generation, low-confidence signals where reasoning overhead must be minimal
- Ollama pull: `ollama pull qwen3:8b`

### DeepSeek-R1-Distill (7B or 14B) — For Reasoning Chains Only

- DeepSeek-R1 achieves 90.8% on MMLU, 84.0% on MMLU-Pro — exceptional reasoning
- The distilled variants (7B, 14B) run comfortably on consumer hardware
- Key characteristic: DeepSeek-R1 generates explicit chain-of-thought reasoning traces — these traces ARE the internal dialogue Midge needs
- Strong quantization resilience: MATH/GPQA/MMLU accuracy difference under 1% between BF16 and Q4_K_M (MLCommons, 2025)
- Tradeoff: slower than Qwen3 for simple tasks due to chain-of-thought overhead; better for high-stakes convergence analysis
- Ollama pull: `ollama pull deepseek-r1:14b`

### Models NOT Recommended

- **Mistral-7B / Mistral-Small-24B**: Decent general reasoning but lower financial domain benchmark scores than Qwen3 or DeepSeek-R1 distills. Mistral's MoE architecture (Mixtral) is more efficient for large scale but adds complexity on single-GPU consumer hardware.
- **Llama 3.3-70B**: Excellent reasoning but requires 48-64GB VRAM at Q4 — exceeds likely Wardenclyffe specs. Not viable unless machine has 3090/4090 x2 or similar.
- **Phi-3/4**: Smaller, faster, but weaker at multi-step financial reasoning chains. Good for classification tasks only.

### Hardware Reality Check

Wardenclyffe is a Win11 desktop with Ollama already installed. Without knowing the exact GPU, the safe assumption is 12-24GB VRAM (RTX 4070 Ti through 4090 range). Qwen3-14B at Q4_K_M (~10-12GB) fits with room for context. Qwen3-32B requires 22-24GB — borderline on 24GB VRAM, tight.

Recommendation: Start with Qwen3-14B. If generation is too slow (<10 tok/sec), drop to 8B. Both fit comfortably within the 30-second latency budget.

---

## Research Area 2: Structured LLM Output for Agent Systems

### The Problem

Midge's pipeline is fully structured (Python objects, EventBus messages, typed dicts). Raw LLM text output is incompatible. Feeding narrative text back into a typed system requires reliable parsing.

### Recommended Solution: Instructor + Pydantic

**Instructor** (github.com/567-labs/instructor) is the production standard for structured LLM output with Ollama as of 2025-2026:

- Wraps the Ollama OpenAI-compatible client with automatic validation and retry
- Uses Pydantic models to define output schemas — the same validation library Midge's hypothesis engine likely uses
- On validation failure: automatically reasks the model with the error, up to N retries
- Solves >95% of JSON hallucination issues per benchmark (Markaicode, 2025)
- Works identically whether the backend is Ollama, Groq, or Claude — same code, different base_url

**How it plugs into Midge's architecture:**

```
Convergence detected
  -> ConvergenceAlerter publishes to EventBus
  -> New OllamaReasoningAgent subscribes
  -> Builds structured prompt with convergence data
  -> Calls OllamaProvider via ApiGateway (existing infrastructure)
  -> Instructor validates JSON response against ReasoningResult Pydantic model
  -> ReasoningResult published to EventBus as "cognition.llm_reasoning_complete"
  -> plain_language.py formatter adds WHY section to alert
```

**Pydantic model for Midge's reasoning output:**

```python
class CausalNarrative(BaseModel):
    ticker: str
    domains_analyzed: list[str]
    causal_story: str          # 2-3 sentence plain English explanation
    story_strength: float      # 0.0-1.0, LLM's confidence in the narrative
    bull_case: str             # Why signal might be real
    bear_case: str             # What could invalidate it
    hidden_risks: list[str]    # Risks the statistical engine can't see
    comparable_pattern: str    # "Similar to X in YYYY when..." or ""
    reasoning_confidence: str  # "high" | "medium" | "low"
```

**Alternative: Ollama's native structured output**

Ollama directly supports JSON schema via the `format` field since late 2024. This is simpler than Instructor but lacks automatic retry on validation failure. For Midge's production pipeline, Instructor's retry loop is worth the extra dependency.

**Alternative: Outlines / Guidance**

Grammar-constrained decoding — guarantees valid JSON by constraining the token sampler. More reliable than retry-based approaches for complex nested structures. Tradeoff: requires running the model through a different server (not standard Ollama API), complicating integration. Not recommended for Midge's current architecture.

---

## Research Area 3: Multi-Perspective Reasoning Frameworks

### The Vision: Bull/Bear/Risk Debate

Midge's vision includes "different reasoning perspectives debate the signal." Research as of 2025 has clarified what actually works vs. what's theoretically appealing.

### What Research Shows

**Multi-agent debate** (multiple LLM instances arguing): Improves factual accuracy 4-6% and reduces errors 30%+ over single-agent generation (Springer Nature, 2025). BUT: consistently fails to outperform well-designed single-agent approaches on many benchmarks. The computational cost of multiple model calls is significant.

**iMAD** (Intelligent Multi-Agent Debate): Selectively triggers debate only when the initial answer is uncertain. Reduces token usage 92% while improving accuracy 13.5%. This is the practical version of debate.

**Self-critique / Reflexion**: Single model generates answer, then critiques it. Cheaper, often comparable to multi-agent debate for reasoning tasks. Most practical for Midge's 30-second latency budget.

### Recommended Approach for Midge: Sequential Perspective Prompting

Rather than multiple model instances, use a single model call with a structured multi-perspective prompt. This is cheaper, faster, and well within the latency budget.

**Three-stage prompt structure:**

```
Stage 1 (BULL): You are a quantitative analyst. Given these signals, build the strongest possible case for [TICKER] moving [DIRECTION] in the next [WINDOW] days. Cite specific signal combinations.

Stage 2 (BEAR): Now play devil's advocate. What are the three strongest arguments that this signal is a false positive or that the move won't materialize?

Stage 3 (SYNTHESIS): Given both perspectives, what is the probability-weighted conclusion? What would need to be true for the bull case to be correct? What hidden risks should the statistical engine be told about?
```

This produces Midge's desired "internal dialogue" in a single inference pass. The model holds both perspectives in its context window simultaneously, which research shows is nearly as effective as separate agents for this type of analysis.

**When to escalate to two-pass reasoning:**

If Qwen3-14B's `story_strength` field comes back below 0.5, or `reasoning_confidence` is "low", trigger a second pass with DeepSeek-R1 (chain-of-thought mode) for that specific convergence. This implements iMAD's selective escalation pattern.

---

## Research Area 4: Financial NLP — Earnings Calls, SEC Filings, News

### Current State of the Art (2026)

**What local LLMs can do reliably:**
- Extract named entities from SEC filings (executives, financial figures, dates, regulatory references)
- Classify sentiment of earnings call tone (confident/cautious, specific vs. vague guidance)
- Summarize key risks from 10-K filings into bullet points
- Identify "language drift" — when a company's communication style shifts between quarters (often a leading indicator)

**What requires caution:**
- Precise numerical extraction from tables in PDFs (LLMs invent numbers; always validate against raw data)
- Cross-document synthesis across many filings (hallucination risk increases with document count)
- Causality attribution ("the reason earnings missed was X") — often speculative

### MarketSenseAI 2.0 as Reference Architecture (Feb 2025)

A peer-reviewed research system using LLM agents on S&P 100 stocks (2023-2024) achieved 125.9% cumulative returns vs. 73.5% for the index. Its architecture is directly applicable to Midge:

- Five specialized LLM agents: news analysis, SEC filing analysis, macro analysis, price history, fundamentals
- RAG (Retrieval-Augmented Generation) for grounding responses in actual document content
- Multi-agent coordination to synthesize across data streams

**For Midge's use case (local, no cloud):** The relevant lesson from MarketSenseAI is specialization. Rather than one general-purpose "analyze this convergence" prompt, specialized prompts per domain outperform.

### Practical Financial NLP Patterns for Midge

**Pattern 1: Convergence Narrative Generator**
Input: Convergence alert data (ticker, domains, signal timestamps, confidence scores)
Output: `CausalNarrative` object (see above Pydantic model)
Model: Qwen3-14B
Latency target: 10-20 seconds
When triggered: Every convergence alert with confidence > 0.6

**Pattern 2: Hidden Risk Scanner**
Input: Ticker symbol + convergence domains
Context injected: Recent news headlines (from Midge's existing news sources), known litigations, recent SEC 8-K filings
Output: List of risk factors the statistical engine cannot see
Model: Qwen3-14B
Latency target: 15-25 seconds
When triggered: High-tier signals only (strong confidence)

**Pattern 3: Historical Pattern Matcher**
Input: Current signal combination (4 domains, direction, ticker sector)
System prompt includes: Summary of past hypotheses that were promoted and their trigger patterns
Output: "Similar patterns in the past" narrative + outcome reference
Model: Qwen3-8B (simpler task)
Latency target: 5-10 seconds

---

## Research Area 5: ReAct / Tool-Use Patterns

### What's Available for Ollama

**Native Ollama tool calling** (confirmed 2025): Available in Qwen3, Llama 3.1/3.2, Mistral-nemo, and others. Define tools as JSON schemas in the API call; model can invoke them and incorporate results. Streaming tool calls also supported.

**Practical tool use for Midge:**
Midge already has data via her existing sources. The useful tool-use pattern is NOT web search (Midge already has data pipelines) — it is querying Midge's own systems:

```
Tool: query_hypothesis_registry(ticker) -> List of active hypotheses for this ticker
Tool: get_causal_graph_neighbors(signal_type) -> Known causal relationships
Tool: check_insider_cluster(ticker, days=30) -> Insider trading cluster summary
```

These tools let the LLM "look up" internal Midge state during reasoning, producing grounded narratives rather than hallucinated ones.

**Framework recommendation: Direct Ollama API (no framework overhead)**

CrewAI and LangChain both work with Ollama, but they add framework overhead and abstraction layers that complicate integration with Midge's custom EventBus architecture. The pattern used by the existing `OpenAIProvider` (direct httpx calls) is cleaner and more controllable.

For tool use specifically: use Ollama's native tool calling via the OpenAI-compatible API, defining Midge's query functions as tools in the API payload. No external framework required.

---

## Integration Architecture: The Shortest Path

### What Already Exists

1. `OpenAIProvider` — already handles any OpenAI-compatible endpoint via httpx
2. `ApiGateway` — already routes requests to registered providers via EventBus
3. `HypothesisEngine` — already has a `causal_story` field on hypotheses (currently filled with "REQUIRES MANUAL REVIEW" for many)
4. `CausalReasoningEngine` — already generates `explain_causation()` text, but mechanically
5. `plain_language.py` — already has a section architecture (WHAT, HISTORY, TIMING, ACTION, TRACKING) — WHY is missing
6. `ApiGateway.ThreadPoolExecutor` — already handles async calls, so LLM reasoning won't block the organism

### What Needs to Be Built

**Step 1: OllamaProvider** (1 new file, ~50 lines)
`OpenAIProvider` with `base_url="http://localhost:11434/v1"` and `api_key="ollama"`. Add `format="json"` to requests for structured output. This is a zero-risk addition — same pattern as existing Groq/Mistral/DeepSeek providers.

**Step 2: ReasoningPayloadBuilder** (new function in existing hypothesis or new small module)
Takes a convergence alert or hypothesis and builds the structured prompt for the LLM. Output: a `question` string + `context` dict matching the format `ApiGateway` already expects.

**Step 3: OllamaReasoningSubscriber** (EventBus subscriber)
Subscribes to `CH_HYPOTHESIS_DISCOVERED` and convergence alert channels. For qualifying signals (above confidence threshold), submits reasoning request to ApiGateway targeting "ollama" provider. Receives response on `CH_EXTERNAL_RESPONSE` and stores/forwards the `CausalNarrative`.

**Step 4: WHY section in plain_language.py**
Add `_section_why(narrative: CausalNarrative) -> str` to the existing section functions. Insert between HISTORY and TIMING. Falls back gracefully (returns empty string) if no LLM narrative is available.

### What NOT to Build

- Do not build a separate LLM daemon or service — the existing ApiGateway ThreadPoolExecutor is the async runtime
- Do not install CrewAI or LangChain — adds framework debt with no benefit over direct API calls
- Do not use Instructor as a mandatory dependency in the hot path — use Ollama's native `format` parameter first; add Instructor retry wrapper only if JSON parse failures become a problem in practice
- Do not route all hypotheses through LLM reasoning — gate on confidence score to prevent runaway inference load

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Ollama not running when Midge needs it | Medium | OllamaProvider.available returns False if connection fails; ApiGateway routes to fallback or skips |
| LLM generation exceeds 30-second budget | Low-Medium | Set httpx timeout=25s; Qwen3-14B is typically 10-20s for 200-token responses |
| Hallucinated causal narratives mislead Guiding Light | Medium | Clearly label all LLM-generated content as "LLM interpretation, not statistical fact" in output; `story_strength` field quantifies confidence |
| JSON parse failures from local model | Low-Medium | Use Ollama's native `format` parameter; add Instructor retry if failures exceed 5% |
| VRAM contention with other Ollama workloads | Low | Midge's reasoning requests are infrequent (per convergence alert, not per step); low collision risk |

---

## Decision: What to Recommend

**Primary recommendation:** Build `OllamaProvider` as a thin wrapper on `OpenAIProvider`. Register it in the bootstrap as provider "ollama". Wire `OllamaReasoningSubscriber` to convergence alerts. Add WHY section to plain_language.py. Total new code: ~200-300 lines.

**Model to pull first:** `ollama pull qwen3:14b` — best balance of capability and hardware fit.

**Secondary model:** `ollama pull deepseek-r1:14b` — for high-stakes convergence analysis where chain-of-thought traces are valuable.

**Structured output:** Ollama native `format` parameter (JSON schema mode) first. Install Instructor (`pip install instructor`) as a fallback layer only after testing reveals parse failures.

**Multi-perspective reasoning:** Single-pass three-stage prompt (bull/bear/synthesis). No external debate framework needed.

**Tool use:** Native Ollama tool calling, defining Midge's own query functions as tools. No additional framework.

---

## Sources

- [Ollama Model Library](https://ollama.com/library)
- [DeepSeek-R1 on Ollama](https://ollama.com/library/deepseek-r1)
- [Ollama Structured Outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama Tool Calling](https://docs.ollama.com/capabilities/tool-calling)
- [Ollama OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Instructor + Ollama Guide](https://python.useinstructor.com/integrations/ollama/)
- [Instructor Examples: Ollama](https://python.useinstructor.com/examples/ollama/)
- [Reliable Structured Output Pipeline (Markaicode, 2025)](https://markaicode.com/ollama-structured-output-pipeline/)
- [Top 10 Open Source LLMs 2026 (o-mega.ai)](https://o-mega.ai/articles/top-10-open-source-llms-the-deepseek-revolution-2026)
- [Best Open Source LLMs 2026 (Contabo)](https://contabo.com/blog/open-source-llms/)
- [Qwen3-14B VRAM Requirements](https://apxml.com/models/qwen3-14b)
- [Qwen3 on Consumer Hardware (Bored Consultant)](https://boredconsultant.com/2025/06/26/Qwen3-and-Gemma3-Performance-on-Consumer-Hardware/)
- [DeepSeek-R1 Technical Report](https://arxiv.org/html/2501.12948v1)
- [MarketSenseAI 2.0 (Feb 2025)](https://arxiv.org/html/2502.00415v2)
- [Multi-LLM Debate ICLR 2025](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-mad-159/blog/mad/)
- [iMAD Framework](https://arxiv.org/html/2511.11306v1)
- [Multi-Agent Debate Factuality (Du et al.)](https://composable-models.github.io/llm_debate/)
- [CrewAI + Ollama Integration](https://www.analyticsvidhya.com/blog/2024/09/build-multi-agent-system/)
- [LiteLLM + Ollama Guide](https://apidog.com/blog/litellm-ollama/)
- [LLMs for Financial Document Analysis (IntuitionLabs)](https://intuitionlabs.ai/articles/llm-financial-document-analysis)
- [LLM Quantization Benchmark (Ionio.ai)](https://www.ionio.ai/blog/llm-quantize-analysis)
- [PRBench Finance Leaderboard (Scale AI)](https://scale.com/leaderboard/prbench-finance)
- [Ollama Tool Support Blog](https://ollama.com/blog/tool-support)
- [Best Ollama Models for Function Calling 2025 (Collabnix)](https://collabnix.com/best-ollama-models-for-function-calling-tools-complete-guide-2025/)
