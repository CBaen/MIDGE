# Lead Findings — Convergence Engine Integrity Review
**Lens:** Convergence Engine Integrity — will these changes make the core intelligence better or accidentally corrupt it?

---

## FILE 1: Signal Enricher

### Files Reviewed
- `mae_core/market/intelligence/signal_enricher.py`
- `mae_core/market/sensing_collector.py`

---

### Finding 1.1 — FALSE CONVERGENCE IS THE PRIMARY RISK (CRITICAL)

**The 15% haircut does not prevent multi-copy inflation.** A single macro FRED signal with `domain="macro"` fans out to 5 tickers: `SPY`, `ES=F`, `NQ=F`, `QQQ`, `DIA`. Each copy enters the signal buffer under a different `symbol`. The convergence alerter groups signals by ticker. If those 5 copies are the only signals, no ticker reaches the convergence threshold (3 domains minimum). However, the danger is compounding: if a real-ticker signal for SPY arrives from a second domain (say `technical`), and separately an `energy` signal fans out to `ES=F` and `SPY`, then SPY now has three "domain" entries: `macro` (from enrichment), `technical` (direct), and `energy` (from enrichment). Two of those three are inflated copies of ticker-less signals. The convergence check on SPY sees 3 domains and fires.

**This is structurally identical to fabricating convergence.** The 15% haircut only reduces confidence magnitude — it does not prevent a ticker from meeting the minimum-domains threshold using enriched signals.

The haircut math: `0.85 × original_confidence`. At `original_confidence = 0.5` (the default), enriched signals arrive at `0.425`. This is below the alerter's min-confidence gate if one exists, but the domain membership check — which is what triggers convergence — is not confidence-gated. A 0.01 confidence enriched signal still counts as "domain X has a signal for ticker Y."

**Concrete scenario:** EIA crude inventory (domain=`energy`, symbol="") fans out to `["CL=F", "XLE", "XOM", "CVX"]`. An economic calendar event (domain=`economic_calendar`, symbol="") fans out to `["SPY", "ES=F", "NQ=F", "TLT"]`. SPY appears in both `institutional` enrichment AND `economic_calendar`. If a third direct signal arrives for SPY (say a technical RSI signal), that is 3 domains — convergence fires. But two of the three are inferred, not observed. The alert says SPY has multi-domain convergence when SPY was never actually mentioned by either macro data source.

**Line reference:** `signal_enricher.py:32-64` (DOMAIN_TICKER_MAP). `sensing_collector.py:82-98` (enrich_ticker called before all alerter feeds).

---

### Finding 1.2 — ENRICHMENT HAPPENS BEFORE DEDUPLICATION (ARCHITECTURAL GAP)

Looking at `sensing_collector.py:64-98`: the comment at line 64 says "Signals arrive pre-enriched from background thread (_fetch_source)." This means enrichment may be happening BOTH in the background fetch thread AND in `_collect_one`. The comment at line 82 says "Expand ticker-less signals; direct-ticker signals pass through unchanged." Both paths are live simultaneously.

If enrichment runs twice (once in background fetch, once in collector), a single ticker-less signal produces 2x the number of copies. The `::TICKER` suffix on `signal_id` would prevent dedup collision between enriched copies for the same signal. However, if the same signal is enriched in two different code paths, the resulting copies would be identical — same `signal_id::TICKER` — and would collide with each other (overwriting in the buffer, not doubling). This needs verification but is a latent double-enrichment risk.

**The deduplication order issue is real regardless:** Enrichment at line 82 happens before the signals are fed to the convergence alerter. The convergence alerter's own deduplication (if any) would see the already-enriched, already-suffixed IDs. The original ticker-less signal is silently dropped (per `signal_enricher.py:96-100`). If the convergence alerter has domain-level signal expiry (sliding window), the enriched copy's expiry is tied to the original signal's timestamp, which is correct. But the buffer would show `domain=macro, symbol=SPY` as a live signal — indistinguishable from a real SPY macro signal.

**Line reference:** `signal_enricher.py:96-100`, `sensing_collector.py:64,82`.

---

### Finding 1.3 — DOUBLE-COUNTING WHEN TWO DIFFERENT TICKER-LESS SIGNALS SHARE A DOMAIN

`DOMAIN_TICKER_MAP["macro"]` = `["SPY", "ES=F", "NQ=F", "QQQ", "DIA"]`. `DOMAIN_TICKER_MAP["fred_macro"]` = `["SPY", "ES=F", "NQ=F"]`.

If a FRED GDP signal (`domain="macro"`) and a FRED yield curve signal (`domain="fred_macro"`) both arrive within the convergence window, they both produce enriched copies for SPY. The convergence alerter groups by domain first: if the alerter treats `macro` and `fred_macro` as the same domain (via its own `_SOURCE_DOMAIN_MAP`), the two copies collapse into one. If the alerter treats them as different domains, SPY now has two "independent" macro domain entries from two ticker-less signals — appearing to provide two domains of evidence when both are the same information stream.

The domain normalization in `failure_explainer.py:443-462` (`_SIGNAL_TO_DOMAIN`) maps both `fred` and `macro` to the canonical domain `"macro"`. Whether the convergence alerter uses the same normalization determines whether this is a real risk. If it does not, the enricher silently inflates domain counts.

**Line reference:** `signal_enricher.py:32-36` (overlapping macro entries), `failure_explainer.py:446` (fred→macro mapping).

---

### Finding 1.4 — 15% HAIRCUT MAGNITUDE: TECHNICALLY DEFENSIBLE, PRACTICALLY WEAK

A 15% multiplicative haircut (`confidence × 0.85`) is conservative enough to preserve signal ordering but not large enough to segregate enriched from direct signals in practice. If an enriched signal arrives at confidence=0.70 (from a strong FRED data point), after haircut it's 0.595 — still higher than many direct signals from noisy sources like StockTwits. The enriched signal is effectively indistinguishable from a moderate-confidence direct signal.

A more principled approach would be to tag enriched signals with a distinct flag in metadata so the convergence alerter can weight them separately (e.g., count them at 0.5 weight toward domain membership). The current approach cannot be inspected downstream — once a signal enters the buffer, its "enriched" origin is only visible in metadata `enriched_from=domain_ticker_map`, and there is no evidence the convergence alerter reads that field.

**Line reference:** `signal_enricher.py:79-80` (_CONFIDENCE_HAIRCUT), `signal_enricher.py:125-127` (enriched_from tag written but likely never read).

---

### Finding 1.5 — `::TICKER` SUFFIX IS SUFFICIENT FOR DEDUP, BUT CREATES A SUBTLE WINDOW PROBLEM

The suffix `f"{signal_id}::{ticker}"` guarantees uniqueness per ticker per original signal. This is correct for preventing duplicate buffer entries. However, the original signal_id before the suffix is typically a UUID or timestamp-based ID. If the same source API call returns the same event twice (idempotent calls, retry scenarios), two separate enrichment passes would produce identical `signal_id::TICKER` strings — meaning the second batch overwrites the first silently. This is a dedup strength, not a weakness. The concern is the inverse: if enrichment never runs again (e.g., daemon restart), the enriched signals from the previous run persist in the buffer while no new enriched copies are generated. The buffer then contains enriched copies from a previous session mixed with new direct signals. This is probably acceptable given the sliding-window expiry, but it means buffer state at restart is briefly "stale-enriched."

---

## FILE 2: Daily Narrative Restructure

### Files Reviewed
- `mae_core/market/intelligence/daily_narrative.py` (focus on `_build_llm_prompt` and `_template_narrative`)
- `mae_core/market/intelligence/narrative_style.md`

---

### Finding 2.1 — STRUCTURE MISMATCH BETWEEN STYLE GUIDE AND IMPLEMENTATION

The style guide (`narrative_style.md:68-72`) specifies: "Watching section: 3 situations max" / "Confirmed section: 1-2 items" / "Total: Under 400 words. One page."

The `_SYSTEM_PROMPT` in `daily_narrative.py:168` says "Under 600 words total." The `_MAX_TOKENS` is set to 1100. There is a direct contradiction between the style guide (400 words) and the system prompt (600 words). The LLM follows the system prompt, not the markdown style guide — the style guide is not injected into the model. The model will generate up to 600 words. Guiding Light gets a longer letter than the style guide intends.

Additionally, the style guide has no explicit layered-structure requirement (it does not use the word "layers"). The `_SYSTEM_PROMPT` introduces layers (Big Picture → Crypto → Commodities → Stocks → Learned → Wrong), which is a restructure relative to the original style guide format. The style guide does not have a dedicated Crypto or Commodities section — it has a general "What MIDGE sees" structure. The restructure adds sections the style guide never specified.

**Line reference:** `narrative_style.md:72`, `daily_narrative.py:168` (_MAX_TOKENS=1100), `daily_narrative.py:168` ("Under 600 words").

---

### Finding 2.2 — STOCK ACTIONABLE INFORMATION IS BURIED — RISK CONFIRMED

The prompt structure (`_build_llm_prompt`, line 856+) presents data in this order: Big Picture → Crypto → Commodities → Stocks. The letter structure mirrors this. For a user with ADHD (per style guide line 1: "Has ADHD"), the most actionable items — paper trades, convergence alerts, specific tickers — appear in section 4 of 6.

When the daily letter is 600 words and sections 1-3 consume 300+ words, the stocks section gets the last third of reading attention. The style guide explicitly states the top priority is "Wild connections across unrelated domains" — which in MIDGE's case most often involves stocks (insider + contract + congressional correlations). By layering macro/crypto/commodities FIRST, the most interesting cross-domain stories that typically involve stocks are structurally downgraded.

The `_template_narrative` correctly implements the layered structure at line 1422. The LLM is instructed to use the same order. The risk is not hypothetical — it is baked into the template implementation.

**This is a design choice that conflicts with the stated priority order in the style guide.** The style guide priority 1 is "wild connections" (usually stocks). The letter structure priority 1 is "macro regime." These are inverted.

**Line reference:** `narrative_style.md:21-25` (priority order), `daily_narrative.py:1422-1585` (template order), `daily_narrative.py:869` (LLM instruction).

---

### Finding 2.3 — LLM INSTRUCTIONS ARE SPECIFIC ENOUGH FOR A 70B MODEL, WITH ONE WEAKNESS

The `_SYSTEM_PROMPT` is well-structured: explicit section headers, concrete forbidden phrases, worked confidence language examples, and a quoted "gold standard" outcome. A 70B model will follow this level of specificity reliably in most cases.

The weakness: the instruction at line 168 says "Never hallucinate. Only write what the data below actually shows." This is not mechanically enforceable. The model sees the summary data (translated to plain English) and is asked to "tell a story." Story-telling pressure combined with sparse data (many empty sections) will produce filler — phrases like "markets remain uncertain" or "I'll be watching this carefully." The system prompt says "If a section has nothing real to say, be honest: 'Nothing notable here today.'" But this instruction conflicts with the story-telling tone and the model will hedge rather than omit.

There is no mechanism to detect or reject a hallucinated narrative. The template fallback (`_template_narrative`) is strictly data-driven and cannot hallucinate, but the LLM path has no validation layer.

**Line reference:** `daily_narrative.py:166` ("Never hallucinate"), `daily_narrative.py:171` (gold standard instruction).

---

### Finding 2.4 — `_template_narrative` CORRECTLY FOLLOWS LAYERED STRUCTURE

The template implementation at lines 1404-1600 correctly implements: Big Picture → Crypto → Commodities & Futures → Stocks (in that order). The section at line 1582 is "STOCKS — THE INTERESTING ONES." This matches the new layered design intent.

However, the template leads with a Granger discovery hook (`lines 1416-1420`), which is data-driven and not domain-specific. This is the strongest element of the template — it surfaces the weirdest pattern first regardless of domain, consistent with the style guide priority 1. The LLM is instructed similarly ("Then a 1-sentence hook — the single strangest or most striking thing"). This hook mechanism is the saving grace that partially mitigates Finding 2.2 — weird stock stories CAN appear in the hook even if they're buried in section 4.

---

### Finding 2.5 — COMMODITIES & FUTURES SECTION IS CONDITIONALLY SUPPRESSED IN TEMPLATE

At `_template_narrative:1560-1580`, the section is only emitted if `futures_activity` is non-empty OR COT positioning is non-"mixed." This means quiet days produce a 5-section letter (no Commodities section). The LLM does not have this suppression logic — its section headers are fixed. This creates a structural inconsistency between the LLM and template outputs. The LLM will generate an empty Commodities section on quiet days; the template will skip it cleanly.

**Line reference:** `daily_narrative.py:1560` (conditional section).

---

## FILE 3: Episodic Memory + Failure Explainer

### Files Reviewed
- `mae_core/market/intelligence/episodic_memory.py`
- `mae_core/market/intelligence/failure_explainer.py`
- `mae_core/bootstrap/market_hooks_sensing.py` (episodic_memory wiring)
- `mae_core/bootstrap/market_hooks_steps.py` (failure_explainer wiring)

---

### Finding 3.1 — EPISODIC MEMORY IS NOT WIRED (CRITICAL — TODOS STILL IN PLACE)

`episodic_memory.py:399-446` contains a large comment block with `# TODO` integration instructions. These are not implemented.

Looking at `market_hooks_sensing.py:544-562`, there IS a real call to `_em.record_episode()` for convergence alerts. However:

1. The `Episode` constructor is called with only 9 of 17 required fields. The dataclass has no defaults for `timestamp`, `resolved_at`, `concurrent_events`, `macro_context`, `action_taken`, `outcome`, `outcome_details`, `price_at_signal`, `price_at_resolution`, `move_pct`, `lessons`, `similar_episodes`. The current call at line 551-562 provides: `episode_id`, `event_type`, `tickers`, `domains`, `direction`, `confidence`, `regime`, `action_taken="alert_generated"`. Missing required fields: `timestamp`, `resolved_at`, `concurrent_events`, `macro_context`, `outcome`, `outcome_details`, `price_at_signal`, `price_at_resolution`, `move_pct`, `lessons`, `similar_episodes`.

**This will throw a `TypeError` at runtime.** The `try/except` at line 546 silently swallows the error. Episodic memory records nothing, silently.

2. The `episode_id` is `f"conv_{_ticker}_{step}"`. If the same ticker fires twice in 100 steps, the second episode gets a different ID and a new record. But if the ticker fires on step 100 and then step 100 again (two alerts in the same step), both get the same `episode_id` and the second silently overwrites the first in `_index`. This is acceptable given the `_append` write semantics (JSONL append, index latest-wins).

3. There is NO `resolve_episode()` call anywhere in the codebase. Episodes are recorded at signal time but never resolved. The `query_failures()` method at line 274 queries for `outcome == "wrong"` — this will always return empty because every episode has `outcome="pending"` forever. The failure explainer reads from `outcomes.jsonl`, not from episodic memory, so this is an independent failure.

**The episodic memory system records episodes (when the constructor call doesn't crash) but can never grade them. It is permanently a one-way write system with no resolution path.**

**Line reference:** `market_hooks_sensing.py:551-562` (broken Episode constructor), `episodic_memory.py:399` (TODO comments still present), `episodic_memory.py:274` (query_failures always empty).

---

### Finding 3.2 — EPISODE RECORDING HAPPENS AT THE WRONG POINT IN THE PIPELINE

Recording happens at `market_hooks_sensing.py:544`, AFTER the paper trading gate at line 542. This means episodes are created for alerts that PASSED the advisory step, not for all convergence events. Alerts that were blocked by the drawdown circuit breaker, the Law 7 validator check, or the bio caution penalty are NOT recorded.

This matters for learning: if MIDGE is consistently blocking high-confidence alerts on a certain ticker due to bio caution, those non-trades are invisible to episodic memory. The system cannot learn that "when bio caution is high, good opportunities are being suppressed."

The correct recording point is immediately when alerts appear (`ctx._cached_alerts[0]`), before any gating logic.

**Line reference:** `market_hooks_sensing.py:542` (paper trade gate), `market_hooks_sensing.py:544` (episode recording after gate).

---

### Finding 3.3 — UNBOUNDED DISK AND RAM GROWTH FOR EPISODIC MEMORY (CONFIRMED RISK)

`episodic_memory.py:150-157` (`_append`): every write is a new line to the JSONL file. There is no eviction, rotation, or size cap.

`episodic_memory.py:126-148` (`_load`): the full JSONL is loaded into `_index` at startup. The index holds all unique episode_ids → latest Episode. Episodes are never deleted from `_index`.

At the daemon's current pace with 25-step cadence, a convergence alert fires potentially hundreds of times per day. Each produces an episode record (when not crashing). After 30 days of daemon operation: potentially thousands of unresolved "pending" episodes in both RAM and on disk. At 500-byte average episode JSON, 10,000 episodes = 5MB on disk (manageable). In RAM, 10,000 Episode dataclass objects with 17 fields each is approximately 10-20MB. Not catastrophic but growing without bound.

The "latest-wins" JSONL scheme means resolved episodes write a second line (the resolution update). Without resolution, only the initial record is ever written — one line per episode. This is the only natural size bound currently in place (no resolution = no second write).

**However, there is no TTL or cleanup policy.** A running daemon will accumulate episodic memory indefinitely.

**Line reference:** `episodic_memory.py:150-157` (unbounded append), `episodic_memory.py:126-148` (full load on startup), `episodic_memory.py:115-119` (no cap in __init__).

---

### Finding 3.4 — JACCARD SIMILARITY IS A POOR MEASURE FOR MARKET SITUATIONS

`episodic_memory.py:228-262` (`query_similar`): similarity is computed as `|A ∩ B| / |A ∪ B|` over domain sets.

The problem: Jaccard treats all domains as equally important. A past episode with `domains=["insider", "macro"]` and a current query with `domains=["insider", "macro", "technical"]` gets score `2/3 = 0.67`. A past episode with `domains=["macro", "technical", "sentiment"]` gets score `1/3 = 0.33`. The first is correctly ranked higher.

But consider: a past episode with `domains=["technical", "sentiment"]` vs the current query `["insider", "macro", "technical"]`. Score = `1/3 ≈ 0.33`. That past episode is ranked equal to the wrong one above. Yet "technical" as a shared domain in both cases is completely different information — one situation is about price momentum, the other about institutional activity.

**The core flaw:** Jaccard has no concept of domain weight. In MIDGE's architecture, `insider` + `institutional` domains carry far higher signal weight than `technical` + `sentiment`. Two "similar" episodes scored equally by Jaccard could represent completely different market conditions (insider-driven vs retail-sentiment-driven). Using these episodes as historical precedents for confidence adjustment would degrade the convergence engine.

Additionally, `query_similar` is called inside `resolve_episode` at line 214, which is never called (Finding 3.1). So this flaw has no live impact today — but it is a time bomb for when resolution is wired.

**A better approach:** weight domain overlap by the domain's Thompson-learned reliability score. Domains that have historically been more predictive should count more toward "similar situation."

**Line reference:** `episodic_memory.py:251-254` (Jaccard calculation), `episodic_memory.py:214` (called in resolve_episode), `episodic_memory.py:403-444` (TODO for using historical context in convergence).

---

### Finding 3.5 — FAILURE EXPLAINER IS READING THE WRONG DATA FORMAT

`market_hooks_steps.py:758-770`:
```python
_out_path = _Pfe("data/market/outcomes.jsonl")
_failed_pairs: list = []
if _out_path.exists():
    with open(_out_path, "r") as _f:
        for _ln in _f:
            _o = _jfe.loads(_ln)
            if not _o.get("success", True):
                _failed_pairs.append((_o, _o))  # ← same dict for both pred and outcome
```

The `batch_explain()` method signature is `predictions_with_outcomes: list[tuple[dict, dict]]` where the first element is the prediction and the second is the outcome. The call above passes `(_o, _o)` — the same outcome record as both the prediction AND the outcome.

This means `explain_failure` receives the outcome record in the `prediction` parameter slot. The prediction extraction at `failure_explainer.py:108-123` looks for `prediction.get("contributing_signals")` and `prediction.get("confidence")`. The outcomes.jsonl schema likely has `predicted_confidence` and `contributing_signals` at the outcome level, but `confidence` (without the `predicted_` prefix) is unlikely to exist. This means `confidence` defaults to 0.0 in the explainer.

**Effect:** Every failure explanation that checks `_check_insufficient_evidence` (line 345-358) receives `confidence = 0.0`, which is below `_WEAK_CONFIDENCE_CEILING = 0.50`. ALL failures get tagged as "insufficient_evidence" regardless of whether confidence was actually the issue. The failure category distribution is corrupted.

The correct fix is to read predictions from `data/market/predictions.jsonl` and match against outcomes by `prediction_id`, passing the actual prediction dict as the first element.

**Line reference:** `market_hooks_steps.py:766` (`_failed_pairs.append((_o, _o))`), `failure_explainer.py:102-123` (extraction logic assuming two different dicts), `failure_explainer.py:350` (confidence=0.0 always fails this check).

---

### Finding 3.6 — FAILURE EXPLAINER `_check_regime_shift` HAS A LOGIC BUG

`failure_explainer.py:203-228` (`_check_regime_shift`):

The check reads the CURRENT regime from `convergence_state.json` (written every 100 steps). It then checks if the CURRENT regime is "bear" or "volatile" — and if so, flags the failure as a regime shift.

This is wrong. The failure explainer should detect whether the regime CHANGED between when the prediction was made and when the outcome was evaluated. Instead, it detects the regime AT GRADING TIME. A prediction made during a bear market, evaluated during a bear market, would incorrectly trigger `regime_shift` — the regime did not shift, it was consistently bearish. The explanation "This failed because the market regime was 'bear' when the outcome was evaluated" is misleading when the prediction was also made in a bear regime.

Additionally, the comment at line 214 acknowledges: "We don't have full regime history, but can use deception_state timestamp as a proxy." The code doesn't actually use the deception_state timestamp at all — that comment appears to be a stale placeholder. The method reads from `convergence_state.json` which carries the current regime, not a historical regime.

**The result:** regime_shift is over-fired for bear/volatile regimes regardless of actual regime change. This inflates the "regime_shift" category in the failure summary.

**Line reference:** `failure_explainer.py:210-228` (reads current regime, not regime at prediction time), `failure_explainer.py:214` (stale comment about using deception_state timestamp).

---

### Finding 3.7 — FAILURE EXPLAINER `_check_stale_signal` OPERATES ON WRONG DATA

`failure_explainer.py:230-251` (`_check_stale_signal`):

The `signals` parameter comes from `prediction.get("contributing_signals")`. Because `_failed_pairs` passes `(_o, _o)` (Finding 3.5), `contributing_signals` is read from the outcome record. If the outcome record doesn't carry `contributing_signals` (it may not — this field is typically in predictions.jsonl), `signals` defaults to `[]` and the stale check always returns null.

Even if the stale check received the correct signals list, it only checks for keyword presence (`"congressional"`, `"senate"`, etc. in the signal string). If the actual contributing signal IDs are formatted differently — e.g., as domain names like `"government"` rather than source names — the keyword check would miss them entirely.

**Line reference:** `failure_explainer.py:111` (signals extraction), `failure_explainer.py:234-236` (keyword check on signal strings).

---

## Summary: Ranked Issues by Severity

| # | Finding | Severity | Impact on Convergence |
|---|---------|----------|----------------------|
| 1.1 | Enriched multi-copies enable false convergence on tickers never directly mentioned | CRITICAL | Directly corrupts convergence |
| 3.1 | Episodic memory Episode constructor crashes silently; never resolves | HIGH | Memory system non-functional |
| 3.5 | FailureExplainer reads same dict as pred+outcome; categories all wrong | HIGH | Learning signal corrupted |
| 3.6 | regime_shift check fires on current regime, not regime delta | HIGH | Failure taxonomy corrupted |
| 1.2 | Possible double-enrichment across two code paths | MEDIUM | Amplifies Finding 1.1 |
| 1.3 | FRED domain duplication may inflate domain count for SPY/ES=F | MEDIUM | Convergence count inflation |
| 2.2 | Stocks buried under macro/crypto; actionable info structurally deprioritized | MEDIUM | UX degradation |
| 3.2 | Episodes recorded after gating, not before | MEDIUM | Incomplete learning record |
| 3.3 | Unbounded episodic memory growth | LOW-MEDIUM | Operational, not correctness |
| 3.4 | Jaccard treats all domains as equal weight | LOW | Deferred (resolve never called) |
| 2.1 | Style guide says 400 words, system prompt says 600 words | LOW | UX inconsistency |
| 3.7 | Stale signal check operates on wrong data structure | LOW | Masked by Finding 3.5 |

---

*Generated by lead reviewer — session 12 triadic review. Independent analysis, no cross-contamination with other reviewer findings.*
