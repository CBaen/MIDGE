# Beta Review: Adversarial Findings — Session 12 New Systems

**Reviewer role:** Adversarial (find breaks, not praise)
**Systems reviewed:** Signal Enricher, Narrative Feedback Loop, Episodic Memory + Failure Explainer
**Date:** 2026-03-15

---

## SYSTEM 1: Signal Enricher

### Attack 1: Phantom Independence — The 9-Signal Problem

**Verdict: REAL. Severity: HIGH.**

A single FRED yield-curve observation fires one signal with `domain="macro"`. `enrich_ticker` in `signal_enricher.py` expands this to 5 copies: SPY, ES=F, NQ=F, QQQ, DIA (line 34). A concurrent VIX signal with `domain="volatility"` expands to 4 copies: SPY, QQQ, ES=F, VXX (line 44).

Now look at what happens in `convergence_ticker.py` `_check_ticker_convergence_locked()` (lines 60–65):

```
for domain, signals in self.signals.items():
    for sig in signals:
        symbol = sig.metadata.get("symbol", "")
        if symbol:
            by_ticker[symbol][domain].append(sig)
```

SPY receives: 1 macro-domain signal (FRED copy) + 1 volatility-domain signal (VIX copy). That is 2 domains on SPY. With `min_domains=2`, this **passes the per-ticker convergence gate**. The convergence engine treats these as 2 independent voices saying "SPY is bearish." They are not — they are 2 copies of the same macro observation (yield curve inversion → VIX spike → both bearish on SPY are causally linked, not independent).

The `_compute_effective_domain_count` in `convergence_confidence.py` (lines 261–285) is supposed to handle this via correlation lookup. But it calls `self._correlation_tracker.get_correlation(src_a, src_b)` for source-level correlation. `fred_macro` and `vix_term_structure` are different sources. The CorrelationTracker would only discount them if it has observed them co-moving historically. On a fresh install or after a reset, **it has no data** — both sources return `max_corr = 0.0` (line 295 returns early if `sources_a` is empty) and the effective count stays at 2.0. The diversity bonus fires at full strength.

**Blast radius:** Every bearish macro + volatility convergence alert on SPY/QQQ/ES=F is potentially a false positive from a single co-occurring event. At 100 alerts/day, this could be the cause behind a significant fraction of the 80.1% loss rate.

**Fix:** `enrich_ticker` should stamp each enriched copy with `"enriched_from_event_id"` equal to the original signal's `signal_id` (before the `::ticker` suffix). `_compute_effective_domain_count` should check if two signals share an `enriched_from_event_id` in their metadata and, if so, **collapse them to 0.5 effective domains regardless of CorrelationTracker state** — they are definitionally the same event.

---

### Attack 2: Cascading Enrichment — Can a Signal Enrich Itself Twice?

**Verdict: NOT REAL in the current code path. Risk contained.**

The enrichment gate in `signal_enricher.py` line 107 is: `if symbol: return [signal_dict]`. An enriched copy has `metadata["symbol"] = ticker` injected (line 125). When it is fed back through `enrich_ticker` via `sensing_collector.py` line 82, the gate fires and it passes through unchanged.

However, there is a subtler risk: enriched copies are fed to `self._convergence_alerter.record_signal(**sig_kwargs)` (line 88) AND to all tiered alerters (lines 94–98). The tiered alerters (`tactical/strategic/thematic`) each maintain their own signal buffer. If those alerters also call `check_convergence` and if that convergence ever emits a signal back into the sensing pipeline — it could re-enrich. I did not find evidence of this loop closing in the current code. **Contained for now, but the pattern is a latent risk if tiered alerters ever publish back to the bus.**

---

### Attack 3: Thompson Pollution From Enriched Copies

**Verdict: REAL. Severity: MEDIUM.**

Enriched copies pass through with their original `source` intact (only `metadata["symbol"]` and `confidence` are modified in `signal_enricher.py` lines 125–131). The `_get_thompson_weight` in `convergence_confidence.py` keys on `signal.source` to look up distributions. A FRED macro signal enriched to 5 tickers creates 5 signals all keyed to `source="fred_macro"`.

When the `outcome_collector` later grades 5 predictions associated with the same FRED event, it fires 5 Thompson updates on the `fred_macro` distribution. If 3 of the 5 enriched predictions succeed (SPY, ES=F, QQQ) and 2 fail (NQ=F, DIA), Thompson records 3 wins and 2 losses for `fred_macro` from what was **one FRED observation**. The distribution drifts based on which specific enriched tickers happened to move, not on whether the underlying FRED signal was informative.

**Blast radius:** Thompson distributions for macro/volatility/energy sources will be corrupted over time — they will reflect enrichment-ticker selection luck rather than source reliability. This undermines the entire Bayesian learning premise for these domains.

**Fix:** The `outcome_collector.register_signals()` call in `sensing_collector.py` line 204 receives the original `signals` (before expansion). The enriched copies are fed only to the convergence alerter, not to the outcome collector. **Verify this is true.** If it is, Thompson pollution may be less severe than described — the learning happens on the original (ticker-less) signals, not the enriched copies. If the outcome_collector also receives enriched copies anywhere, the pollution is live.

*Checking:* Line 199–208 of `sensing_collector.py` passes `signals` (the original list from `future.result()`) to `store_signals` and `outcome_collector.register_signals`. The `expanded` copies from `enrich_ticker` are only used in the convergence alerter loop (lines 84–98). **So Thompson pollution via outcome_collector is NOT live today.** However, Thompson weights ARE used inside `_compute_confidence` called during convergence checking — and the `_get_thompson_weight` in convergence_confidence keys on `sig.source`. Enriched copies carry the original source, so 5 copies all pull the same `fred_macro` weight. They don't pollute the distribution, but they **amplify the weight 5x** inside the geometric mean calculation. This is still an inflation problem.

---

### Attack 4: Domain Inflation — Inferred vs. Direct Domain Membership

**Verdict: REAL. Severity: MEDIUM.**

`enrich_ticker` assigns domain from the original signal to each copy (the `domain` key is not modified, only `metadata["symbol"]`). So a `domain="macro"` FRED signal enriched to SPY creates a signal in the `macro` domain with `symbol=SPY`.

If a direct technical signal for SPY fires (`domain="technical"`, `symbol=SPY`), the per-ticker convergence on SPY sees `macro + technical`. That counts as 2 independently sourced domains.

The macro signal is saying: "FRED yield curve inverted, economy-wide." The technical signal is saying: "SPY RSI crossed below 30." These ARE independently sourced in the empirical sense — one is macroeconomic data, one is price action. The convergence is conceptually valid.

**But the problem is representation:** the enriched macro signal implies MIDGE has a direct SPY-specific macro observation. It does not — it has a broad economy observation attributed to SPY by a lookup table. If the economy is fine but SPY drops for company-specific reasons, the macro "SPY signal" is noise. The 15% confidence haircut on line 131 acknowledges this but does not change the domain count or the domain diversity bonus.

**Blast radius:** Two domains (macro + technical) on SPY can now fire from: (a) one FRED observation + one RSI reading. Three domains (macro + volatility + technical) can fire from one FRED + one VIX + one RSI — all three being SPY-attributed but only one being directly SPY-specific. The min_domains=3 gate passes on what is effectively 1.15 independent observations (macro inference + VIX inference + actual technical).

**Fix:** Enriched signals should carry a flag `"inferred_ticker": True` in their metadata. The per-ticker convergence grouper (`convergence_ticker.py` lines 60–65) should require at least one direct (non-inferred) signal per ticker before counting inferred signals toward the domain threshold. A ticker should not achieve convergence on inferred signals alone.

---

## SYSTEM 2: Narrative Feedback Loop

### Attack 5: Hallucination Becomes Permanent WorldModel Edge

**Verdict: REAL. Severity: HIGH.**

The LLM (`llama-3.3-70b-versatile`) generates the daily letter. NarrativeFeedback.extract_insights() in `narrative_feedback.py` applies regex to that text (lines 68–82 define `_CAUSAL_PATTERNS`). If the model writes:

> "I've noticed that institutional moves lead insider buying by 4 days"

`_CAUSAL_PATTERNS[0]` matches. The cause string extracted is `"institutional moves"`, effect is `"insider buying"`, lag is `4`. With `insight.confidence > 0.60`, `_push_to_world_model` is called (line 316–317).

`world_model.add_discovered_edge()` at `world_model.py` line 266 adds this edge permanently. If the edge already exists, it applies an EMA update: `edge["strength"] = min(1.0, edge["strength"] * 0.8 + strength * 0.2)` (line 282). **A hallucinated edge, added once per day, strengthens at 20% per iteration.** After 10 days of the same hallucination (plausible if the LLM is calibrated to repeat itself), the edge strength converges toward the hallucinated strength regardless of evidence.

There is no ground truth check. There is no `evidence="narrative_synthesis"` gating that WorldModel treats skeptically — `add_discovered_edge` applies the same EMA update regardless of `evidence` parameter value (line 289 shows `evidence` is stored but line 282 ignores it during the EMA merge).

**Blast radius:** Hallucinated causal edges feed into `_compute_ripple_effects()` in `convergence_detection.py` (lines 384–413), which uses WorldModel to generate `ripple_effects` on every convergence alert. A hallucinated `institutional_moves → insider_buying (lag=4d)` edge would cause every institutional signal to generate a ripple alert predicting insider buying 4 days later. This poisons the cascade tracker and confidence modifiers.

**Fix:** Edges from `evidence="narrative_synthesis"` should (a) not strengthen existing curated/granger edges — only update other discovered/narrative edges, and (b) have a max strength ceiling of 0.4 until corroborated by Granger analysis. WorldModel should track `evidence_type` per edge and apply different EMA coefficients.

---

### Attack 6: Echo Chamber — NVDA Monopolizes Attention Forever

**Verdict: REAL. Severity: MEDIUM.**

The loop:

1. MIDGE detects NVDA signals → writes about NVDA in daily letter
2. `NarrativeFeedback.extract_insights()` extracts NVDA as `ticker_call` or `watching`
3. `_push_to_shared_attention()` calls `self._sa.update_hot_ticker(ticker="NVDA", ...)` (line 335)
4. `shared_attention.py` line 90–102: `hot["NVDA"] = {confidence: X, updated: now}` — **this is an overwrite with no decay**
5. Next day's data gathering reads `SharedAttention.get_hot_tickers()` to prioritize fetching
6. More NVDA data → more NVDA signals → more letter content about NVDA

The `hot_tickers` dict in `SharedAttention` has no TTL. Once NVDA enters `hot_tickers`, it stays forever unless something explicitly removes it. The `update_hot_ticker` method (lines 90–102) only writes — it never expires stale entries.

There is no competing eviction mechanism. `update_cascade()` has a `cascades[:] = cascades[-50:]` eviction (line 123), and `add_causal_discovery` has `discoveries[:] = discoveries[-100:]` (line 158). But `hot_tickers` is an unbounded dict with no equivalent cleanup.

**Blast radius:** MIDGE becomes increasingly NVDA-biased over time (or whatever stock the LLM fixates on). Other tickers that should generate attention get crowded out. This is compounded because the NarrativeFeedback confidence defaults to 0.55 (`_detect_confidence` line 139) when no language marker matches — so most ticker_calls go in at 0.55, and `get_hot_tickers(min_confidence=0.5)` returns all of them.

**Fix:** Add a TTL to `hot_tickers` entries. `get_hot_tickers()` should filter out entries where `updated` is older than N days (suggest 3 days). Or add a `max_hot_tickers` limit with LRU eviction, so new entries push out old ones.

---

### Attack 7: Phantom Tickers From English Words

**Verdict: REAL. Severity: LOW-MEDIUM.**

`_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")` in `narrative_feedback.py` line 65 matches any sequence of 1–5 uppercase letters. The `_COMMON_WORDS` set (lines 55–63) filters the obvious ones.

**Words NOT in `_COMMON_WORDS` that are real tickers and also common English words:**

- `"US"` (not in list) → could match U.S. Steel (X) or just "US markets" where the regex grabs "US" — wait, U.S. has a period so regex won't match that. But "US equities" → "US" would match and "US" is not in `_COMMON_WORDS`. US is not a real ticker but `U` is (Unity Software). Sentences like "I noticed US markets rising" → matches `US` → not in common words → treated as ticker.
- `"AI"` → not in `_COMMON_WORDS`. Every sentence containing "AI trends" would extract `AI` as ticker C3.ai.
- `"OIL"` → not in `_COMMON_WORDS`. "Oil prices are rising" → `OIL` extracted as a ticker.
- `"RISK"` → not in `_COMMON_WORDS`. "High risk environment" → `RISK` extracted.
- `"MOVE"` → not in `_COMMON_WORDS`. "The move happened quickly" → `MOVE`.
- `"CASH"` → not in `_COMMON_WORDS`. Common word but also ticker.
- `"GAS"` → not in `_COMMON_WORDS`. "Natural gas prices" → `GAS`.

The prompt instructs the LLM to avoid jargon and use plain English, which means the letter will contain ordinary sentences like "The oil move was sharp" → tickers `OIL`, `MOVE` both extracted → both added to `SharedAttention.hot_tickers` at 0.55 confidence.

**However, severity is mitigated:** The `ticker_call` type requires both a ticker AND a direction word (line 275: `if tickers and direction`). Phantom tickers only cause harm if they also appear near bullish/bearish language. "The OIL move was sharp" has no directional word → no ticker_call. The `watching` type (line 219) does add tickers without requiring direction, so `OIL`/`AI`/`RISK` could reach `SharedAttention` via that path.

**Fix:** Add `"AI", "OIL", "GAS", "RISK", "MOVE", "CASH", "GOLD", "BOND", "FUND", "BANK", "TECH", "BULL", "BEAR"` to `_COMMON_WORDS`. These are high-frequency plain-English words that also happen to be tickers and will appear constantly in MIDGE's letters.

---

## SYSTEM 3: Episodic Memory + Failure Explainer

### Attack 8: Unbounded JSONL Growth Into RAM

**Verdict: REAL. Severity: HIGH (long-term).**

`EpisodicMemory._load()` in `episodic_memory.py` lines 132–146:

```python
with open(self.path, "r", encoding="utf-8") as f:
    for line in f:
        ...
        self._index[ep.episode_id] = ep
        loaded += 1
```

Every line is parsed and stored in `self._index` (a dict). The memory model is "latest-wins" — duplicate episode_ids overwrite. This is used to handle resolve updates: an episode is written twice (initial + resolution).

**Growth math:** 100 convergence alerts/day → 100 episode records/day (from `market_hooks_sensing.py` lines 544–562). Each resolution adds another line. Call it 150 lines/day. Over 1 year: 54,750 lines. Each `Episode` has `concurrent_events: List[str]` (up to 10 episode_ids) and `similar_episodes: List[str]` (up to 5 episode_ids). The Episode dataclass has ~18 fields. Estimated RAM per episode: ~500 bytes. At 54K unique episodes (latest-wins deduplication reduces raw line count): ~27MB. Not catastrophic at year 1.

**But the JSONL file itself is append-only and never pruned.** At 150 lines/day, year 1 = 54,750 lines. The `_load()` function reads ALL lines on every startup (not just on first load). Every daemon restart causes a full-file parse. At year 3: 164K lines, each parsed from JSON into a Python object and discarded if it's a stale version — this parsing overhead is wasted on every restart.

**More immediately:** the `resolve_episode()` call at line 213–216 calls `self.query_similar()` before writing, which iterates ALL episodes in `self._index` with Jaccard scoring. At 5K episodes this is noticeable; at 50K it becomes a blocking call on every outcome resolution.

**Fix:** Add a max_episodes_in_memory cap (e.g. 10,000 latest). During `_load()`, collect all lines, deduplicate by episode_id (latest wins), then only keep the most recent N by timestamp. Or: rotate to a new file every 90 days and keep the index as a separate JSON for O(1) lookup without full parse.

---

### Attack 9: Episode ID Collision and Overwriting

**Verdict: REAL. Severity: MEDIUM.**

Episode IDs are constructed in `market_hooks_sensing.py` line 552:

```python
episode_id=f"conv_{_ticker}_{step}",
```

`step` is the current step counter. Two convergence alerts on the same ticker in the same step will have the **same episode_id** — e.g., `conv_SPY_1042`. The second call to `_em.record_episode(Episode(...))` would overwrite the first in `self._index` (line 155: `self._index[episode.episode_id] = episode`) after both are appended to the JSONL file.

More problematically: if the daemon restarts and resumes at step 1042 (because step counter resets to 0, or is restored from checkpoint at a different value), new episodes will collide with historical episodes from the same step. The `latest-wins` resolution in `_load()` will silently discard the older one.

The `uuid4().hex[:12]` generator is available in `episodic_memory.py` line 169 for cases where `episode_id` is empty, but the hook supplies a non-empty ID so that branch is never taken.

**Fix:** Use `alert.alert_id` (which is already a unique CONV-YYYYMMDD-XXXX format) as the episode_id instead of the synthetic `conv_{ticker}_{step}` construction. The alert_id is unique per alert. See the TODO comment in episodic_memory.py line 411: `episode_id=alert.alert_id` — this TODO has the right answer and wasn't implemented.

---

### Attack 10: Failure Misattribution — Reading Current Regime, Not Prediction-Time Regime

**Verdict: REAL. Severity: MEDIUM.**

`FailureExplainer._check_regime_shift()` in `failure_explainer.py` lines 203–228:

```python
state = _load_json(_CONVERGENCE_STATE_PATH)
regime_at_grading = state.get("regime", "")
```

`_CONVERGENCE_STATE_PATH = Path("data/midge/convergence_state.json")`. This file is written by the convergence alerter at its current state — meaning it reflects the **regime at the time the explainer runs**, not the regime at prediction time or even at outcome time.

The failure explainer runs every 200 steps (from `market_hooks_steps.py` lines 752–773). If a prediction was made 500 steps ago in a bull regime, then the regime shifted to bear, and 200 steps later the regime shifted back to bull — the explainer runs during the bull recovery and reads `regime=bull`. It will return `_null_check("regime_shift")` because bull regime doesn't trigger the check. The real cause (regime was bear during the holding period) is silently missed.

**Blast radius:** Regime shift will be systematically underdiagnosed as a failure cause. The explainer will attribute failures to `correlated_domains` or `insufficient_evidence` instead, because those checks don't have temporal dependency. The `failure_summary.json` will undercount `regime_shift` and overcount other categories, corrupting the signal to Guiding Light about what's actually going wrong.

**Fix:** `EpisodicMemory` stores `regime` at signal time (line 558 in market_hooks_sensing.py: `regime=getattr(ctx, "_cached_regime", ...)`). The failure explainer should look up the episode for this prediction_id and compare `episode.regime` (prediction-time regime) against the `regime_at_grading`. If they differ → regime shift. This requires connecting FailureExplainer to EpisodicMemory, which currently it is not.

---

### Attack 11: Double Counting — Same Failure Re-Explained Every 200 Steps

**Verdict: REAL. Severity: MEDIUM.**

The failure explainer loop in `market_hooks_steps.py` lines 758–770:

```python
for _ln in _f:
    _o = _jfe.loads(_ln)
    if not _o.get("success", True):
        _failed_pairs.append((_o, _o))
if _failed_pairs:
    _fe.batch_explain(_failed_pairs[-50:])
```

This reads ALL lines from `outcomes.jsonl`, collects ALL failures, takes the last 50, and explains them. Every 200 steps. `outcomes.jsonl` is append-only and never pruned. A failure from week 1 will be in the last-50 forever if fewer than 50 new failures have occurred since.

`_fe.batch_explain()` calls `self._persist(expl)` for each explanation. `_persist()` calls `_update_summary()` which reads `failure_explanations.jsonl` and increments counts using `last_line_processed` as a cursor. The cursor prevents double-counting **in the summary JSON**, but `failure_explanations.jsonl` itself gets a new line appended every 200 steps for the same prediction. A prediction from week 1 will have hundreds of identical explanation records in the JSONL.

**Blast radius:** `failure_explanations.jsonl` grows at `50 × (200-step cadence)` records per cycle. At 200 steps/hour daemon speed, that's 50 new (possibly duplicate) explanations per hour for every old failure that stays in the last-50. File growth is O(days_running × failures). At day 30 with 100 total failures: ~360,000 explanation records for 100 actual failures.

**Fix:** Before calling `batch_explain`, check if a prediction_id already has an entry in `failure_explanations.jsonl`. The simplest approach: maintain a `_explained_ids: set[str]` in FailureExplainer that is populated during `_load_caches()` by reading existing explanation IDs. Skip predictions already in the set.

---

### Attack 12: Stale Signal Check Is Too Coarse

**Verdict: REAL. Severity: LOW.**

`_check_stale_signal()` in `failure_explainer.py` lines 230–251 flags any signal containing `"congressional"`, `"politician"`, `"congress"`, `"senate"`, or `"house"` as stale, with **0.80 confidence** — a very high score that will win against most other checks.

The explanation says "the market has already priced this." But:

1. `OpenInsider` and `finviz_insider` signals can also carry congressional sources — and those systems filter by disclosure recency. The source name check doesn't distinguish fresh-disclosure vs. old-disclosure.
2. A congressional trade disclosed yesterday (within 2 days of the 45-day window — meaning the trade occurred ~43 days ago) is priced in. But a congressional trade disclosed the same week it was filed (rare but possible — some members file within days) is NOT stale.
3. The check ignores `predicted_at` vs. signal timestamp entirely — the actual `_STALE_SIGNAL_DAYS = 7` constant on line 51 is **never used** in `_check_stale_signal()`.

The 0.80 confidence score means this check will be chosen as the "best explanation" in most cases where congressional sources appear, suppressing the real cause (regime shift, correlation, etc.).

**Fix:** Actually compute signal age: compare the trade date in the signal metadata against `predicted_at`. If the signal timestamp is within `_STALE_SIGNAL_DAYS` days of `predicted_at`, it's fresh — return `_null_check`. The `_STALE_SIGNAL_DAYS` constant exists for this purpose but is orphaned.

---

## Summary Table

| # | Attack | Real? | Severity | File | Lines |
|---|--------|-------|----------|------|-------|
| 1 | 9 signals from 2 events pass independence gate | YES | HIGH | signal_enricher.py, convergence_ticker.py | 34, 60–65 |
| 2 | Cascading enrichment loop | Contained | LOW | sensing_collector.py | 82–98 |
| 3 | Thompson weight amplification (5x) from enrichment | YES | MEDIUM | sensing_collector.py, convergence_confidence.py | 84–90, 240–246 |
| 4 | Inferred domain membership inflates convergence | YES | MEDIUM | signal_enricher.py, convergence_ticker.py | 83–142, 60–86 |
| 5 | LLM hallucination → permanent WorldModel edge | YES | HIGH | narrative_feedback.py, world_model.py | 347–401, 266–292 |
| 6 | Echo chamber — hot_tickers has no TTL | YES | MEDIUM | shared_attention.py, narrative_feedback.py | 90–102, 329–345 |
| 7 | Phantom tickers from common English words | YES | LOW-MED | narrative_feedback.py | 55–65 |
| 8 | EpisodicMemory full-file load + query_similar O(N) | YES | HIGH (long-term) | episodic_memory.py | 126–148, 228–262 |
| 9 | Episode ID collision via conv_{ticker}_{step} | YES | MEDIUM | market_hooks_sensing.py | 552 |
| 10 | Regime misattribution — reads current, not prediction-time | YES | MEDIUM | failure_explainer.py | 203–228 |
| 11 | Same failure re-explained every 200 steps | YES | MEDIUM | market_hooks_steps.py, failure_explainer.py | 758–770, 373–380 |
| 12 | Stale signal check uses keyword match, ignores timestamp | YES | LOW | failure_explainer.py | 230–251 |

**Most critical issues to fix first, ranked by impact on active signals:**

1. **Attack 1** — Phantom independence from enrichment is causing false convergence alerts RIGHT NOW. Every FRED + VIX co-occurrence on SPY is a false convergence.
2. **Attack 5** — Hallucinated WorldModel edges corrupt ripple effects on every future alert. Accumulates silently.
3. **Attack 11** — failure_explanations.jsonl is growing unboundedly every 200 steps. Will degrade disk and startup time.
4. **Attack 9** — Episode IDs collide on restart. Episodic memory is unreliable.
