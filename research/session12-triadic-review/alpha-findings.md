# Alpha Review — Session 12 Triadic Review
**Lens: User Experience — will these changes help or confuse the human?**
**Reviewer role: UX / Human-facing output**
**Date: 2026-03-15**

---

## Area 1: Signal Enricher's Effect on What the Human Sees

### What the code does

`signal_enricher.py` takes a ticker-less macro signal (e.g., a FRED yield-curve reading with `symbol=""`) and copies it into multiple per-ticker signals. A single FRED reading about yield spreads becomes one signal for SPY, one for ES=F, and one for NQ=F. The original ticker-less signal is dropped and replaced by these three copies.

The confidence haircut (15%) is applied to each copy — so a FRED signal at 0.60 confidence becomes three signals each at 0.51 confidence.

### UX Problem 1: Alert Multiplication — "Why is MIDGE shouting about the same thing three times?"

**The risk is real.** If one FRED yield-curve inversion reaches the convergence threshold for SPY, ES=F, AND NQ=F simultaneously, the human could receive three separate convergence alerts in their email/notification feed that all say essentially the same thing in plain language:

- "I think you should look at selling SPY because economic data is pointing down"
- "I think you should look at selling ES=F because economic data is pointing down"
- "I think you should look at selling NQ=F because economic data is pointing down"

From the human's perspective, this looks like MIDGE is "very sure" about three different things when she is actually only sure about one macro condition that affects all three. The style guide says alerts should feel like "I noticed something" — three near-identical alerts feel like system noise, not curiosity.

**The macro domain is the highest-risk offender.** The DOMAIN_TICKER_MAP for "macro" maps to five tickers: SPY, ES=F, NQ=F, QQQ, DIA. A single FRED macro signal could generate five enriched copies. If two or three of those independently hit the convergence threshold, the human could receive multiple alerts that feel indistinguishable.

**The economic_calendar domain** maps to four tickers (SPY, ES=F, NQ=F, TLT). FOMC announcement day could trigger a cascade of alerts about bond behavior and equity behavior that are all downstream of the same single event.

### UX Problem 2: Unfamiliar Tickers Appearing Without Explanation

The human has never been told what "ES=F" or "NQ=F" means. The plain-language formatter (`plain_language.py`) translates directions and domains into readable English, but it does NOT translate ticker symbols. The style guide states: "Include the MARKET: 'This is a US stock on the NYSE' or 'This is a futures contract.'"

An alert about ES=F will say "I think you should look at selling ES=F because economic data is pointing down." Without context, "ES=F" is meaningless to a designer. The `signal_enricher.py` injects these futures tickers silently — there is no mechanism to flag enriched tickers as "this is an index futures contract" vs. a regular stock.

The somatic state data gathering in `daily_narrative.py` partially addresses this: it has a `_futures_symbols` dict that maps `ES=F → "S&P 500 futures"` and surfaces this in the Commodities section. But this friendly-name mapping lives only in the narrative layer, not in the plain-language alert formatter. Real-time convergence alerts from enriched signals will show the raw ticker.

### UX Problem 3: "I think you should look at selling SPY" — Is That Actionable?

The style guide asks MIDGE to be actionable: "Include the MARKET: 'This is a US stock on the NYSE.'" For a designer, the question is: "What do I actually do with this?" SPY, ES=F, NQ=F, QQQ, and DIA are index funds and futures — broad market instruments. A convergence alert suggesting "look at selling SPY" because of a yield curve inversion is not the same kind of actionable call as "look at buying LMT because three independent signals about defense contracts are converging."

The macro enrichment creates alerts that sound specific (a ticker is named) but are actually just "the market might go down" — the same information available from any news feed. This risks making MIDGE sound like a ticker-screener parroting macro news rather than the "magic machine" that spots non-obvious connections.

### What IS working

The energy domain enrichment is better-scoped. `eia_energy` maps to `[CL=F, XLE, XOM, CVX]` — four tickers with real causal relationships to oil inventory data. A crude inventory surprise affecting CL=F futures AND XLE energy ETF AND XOM is genuinely useful information because those are different things that happen to share a cause. The human might want to know that energy sector stocks and oil futures are both affected.

The 15% confidence haircut is good signal hygiene — it protects convergence quality.

### Specific Finding

The single highest risk is the `macro` domain mapping to 5 tickers. If MIDGE uses this configuration and FRED emits a strong signal on a single day, the deduplication in `_gather_data()` (which deduplicates `top_alerts` by ticker) will catch it at the *daily narrative level*, but real-time email alerts sent by `_submit_to_alpaca()` and `plain_language.py` do NOT deduplicate across tickers. The human could receive 3-5 near-identical real-time emails within seconds of each other.

---

## Area 2: Daily Narrative Restructure

### What changed

The prompt was restructured from a simpler format (implied by the old letter structure with sections: WHAT I'M WATCHING / WHAT CONFIRMED / WHAT I LEARNED / WHAT I GOT WRONG) to a 5-layer structure: THE BIG PICTURE → CRYPTO → COMMODITIES & FUTURES → STOCKS — THE INTERESTING ONES → WHAT I LEARNED / WHAT I GOT WRONG.

### Comparing Yesterday's Letter to the New Structure

Yesterday's letter (2026-03-15) was produced by an older/simpler structure. It had:
- A strong hook (the institutional→insider lag finding)
- WHAT I'M WATCHING (3 stock situations with plain-language reasons)
- WHAT CONFIRMED (1 cascade item)
- WHAT I LEARNED (3 bullets)
- WHAT I'M UNCERTAIN ABOUT (2 items)
- WHAT I GOT WRONG (empty)

The total letter was **under 250 words** — well inside the style guide's "under 400 words" limit. It felt like a partner's note, not a report.

The new structure has **6 required sections** (Big Picture, Crypto, Commodities, Stocks, Learned, Wrong) plus an optional 7th (What I Think You Should Look At). Each section has data to populate it even when the market is quiet.

### Finding: The New Structure Risks Bloat

The style guide says: **"Total: Under 400 words. One page. Coffee-length."** The new structure, with 5-6 mandatory sections each containing 2-4 bullets, could easily produce 500-700 words even when the model follows the instructions. The system prompt says "Under 600 words" — already 50% over the style guide's stated limit. This is a direct conflict.

The style guide is explicit: **"30-second sections. If a section takes longer than 30 seconds to read, split it."** Six sections × 30 seconds = 3 minutes minimum. That is not a coffee-length letter.

### Finding: The "Weird Part" Is Now Third-Layer Deep

The style guide says: **"Lead with the weird part. 'Here's what's strange:' not 'Based on our analysis of...'"**

In the old structure, the hook was the first sentence. In yesterday's letter, the first sentence was: "Here's something strange I noticed: when big institutions make moves, insiders start buying or selling the same stocks about 4 days later."

In the new structure, the letter opens with THE BIG PICTURE — regime, macro alignment, economic readings. This is the least interesting content for someone with ADHD. The weird cross-domain connection is supposed to appear as "the 1-sentence hook at the very top" per the system prompt, but then THE BIG PICTURE section forces 2-4 bullets of regime/macro content before the reader reaches anything surprising.

The prompt tries to fix this with: "Then a 1-sentence hook — the single strangest or most striking thing across ALL markets today." But this instruction is immediately followed by six section headers that push macro content first. An LLM following the structure will drift toward the structure, not the hook-first principle.

### Finding: The COT Positioning Data Is a Jargon Risk

The COMMODITIES & FUTURES section explicitly includes: "COT positioning shifts (are big traders piling in or bailing out?)." The system prompt translates this as: "The big professional traders are heavily positioned [direction] right now — which is either a smart bet or a crowded trade that could snap back."

The style guide says: **"No financial jargon. Not ever."** "Crowded trade that could snap back" is financial jargon. A designer reading this would not know what it means. The style guide specifically says "overcrowded" is a failure category — and yet the COT instruction uses the concept directly in user-facing language.

The COMMODITIES & FUTURES section as a whole assumes the reader knows what futures, forex, and COT data are. "Big professional traders shifted their positions" (from the DOMAIN_PLAIN map) is borderline acceptable, but the section instructions to the LLM include raw terminology like "COT positioning" and "index futures" that could leak through.

### Finding: The New Structure Is Better for DATA, Worse for ADHD

The 5-layer data gathering in `_build_llm_prompt` is technically excellent. The Python layer pre-translates everything — FRED series IDs become plain English descriptions, macro indicators get direction language, Granger findings get "leads by N days" framing. This is a genuine improvement.

But the new structure optimizes for comprehensiveness over attention. For someone with ADHD, the letter should answer ONE question first: "Is there something I should care about today?" Under the old structure, the answer was in the first two sentences. Under the new structure, the answer is buried in layer 4 (STOCKS — THE INTERESTING ONES) after three prior sections.

The style guide's priority list is: (1) wild cross-domain connections, (2) things building slowly, (3) what MIDGE learned from being wrong. The new structure puts regime and macro (low interest) before cross-domain weirdness (high interest). This is inverted from the priority list.

### What IS working in the new structure

The "INSTRUCTION:" inline hints to the LLM are very good. Phrases like "Only mention if something is interesting — skip the section if flat" and "Lead with the WEIRDEST convergence, not the highest confidence" give the model permission to be brief. The _domain_plain() and _regime_plain() translation functions ensure jargon is pre-translated. The template fallback in `_template_narrative` follows the correct hook-first order.

---

## Area 3: Failure Explanations in the Letter

### What the failure taxonomy contains (raw)

The `failure_explainer.py` taxonomy has 8 categories: `regime_shift`, `stale_signal`, `correlated_domains`, `timing_error`, `deception`, `overcrowded`, `external_shock`, `insufficient_evidence`.

### Finding: The Taxonomy Speaks in Technical Language the Letter Doesn't Translate

The taxonomy produces internal category names that feed into `failure_summary.json`. The `daily_narrative.py` reads this file in Layer 5 via `pm.get("timing_insight")` — a single string field. The full category breakdown (how many were `correlated_domains` vs. `regime_shift`) does NOT appear to be plumbed into the prompt.

The result: the failure information that reaches the letter is the high-level postmortem win rate and timing insight, not the specific failure category analysis. This is actually good for the user — "I thought X, I was wrong because the timing was off" is readable. "I failed because of correlated_domains" is not.

However, there is a gap. The `failure_explainer.py` produces plain-English `explanation` strings (e.g., "This failed because the signal included congressional trade data... the market has already priced this information"). These explanations are stored in `failure_explanations.jsonl` but are NOT plumbed into the daily narrative. Only the category counts reach the letter, not the actual explanations.

### Finding: The "I Keep Being Wrong Because..." Risk Is Real But Contained

The style guide says to be honest about mistakes: **"Humble about mistakes. 'I was wrong about X and here's what I think happened'"**

The specific risk asked about — does failure explanation undermine trust? — depends on the framing. There are two failure modes:

1. **Good framing**: "I thought NVDA would fall. I was wrong. What I think happened: the move was real but my timing was off — it reached the target 3 days after my window closed." This is trustworthy. MIDGE shows self-awareness.

2. **Bad framing**: "I keep being wrong because my signals are correlated, and the market was in bear regime, and the data was stale." This sounds like a system making excuses and will erode trust quickly.

The current data pipeline avoids the bad framing because the detailed failure categories don't reach the letter — only summary win rates do. This is protective. But it also means MIDGE never uses the rich explanations the FailureExplainer produces. Those explanations are generated and persisted and then never communicated to the human.

Yesterday's letter says: "What I Got Wrong: Nothing notable to report today." That is the right tone when there's nothing to report. But the section is empty in a template way — it exists as a structural placeholder, not because MIDGE is genuinely reflecting.

### Finding: Narrative Feedback Extraction Would Fail on Yesterday's Letter

The `NarrativeFeedback` extractor uses regex patterns to find causal claims matching "X leads Y by N days." Yesterday's letter contained:
- "Institutional moves lead macro trends by about 5 days" — this WOULD match `_CAUSAL_PATTERNS[0]`
- "Macro trends lead insider buying/selling by about 3 days" — this WOULD match

So those two lines would be extracted as `causal_claim` insights and fed into WorldModel as discovered edges. That is the correct behavior.

However, the old letter used section headers that DON'T match the new structure. The old letter has "## WHAT I LEARNED" but the new structure expects "## WHAT I LEARNED" — same name, so this is fine. But the old letter also has "## WHAT I'M WATCHING" and "## WHAT CONFIRMED" and "## WHAT I'M UNCERTAIN ABOUT" — three sections with no equivalent in the new structure. The `_SECTION_RE` regex will capture these headings for attribution, but NarrativeFeedback will attribute insights to section names like "WHAT I'M UNCERTAIN ABOUT" which the new letter structure will never produce. This is a minor attribution drift issue, not a functional failure.

The extraction would also fire on the line "I'm sensing something with GOOGL" — matching `_WATCHING_MARKERS` on "not ready to call" — and on "My attention keeps going back to AMZN" — matching "my attention keeps going back to." Both of these would correctly create `watching` type insights and push GOOGL and AMZN to SharedAttention. This is working correctly.

### Finding: No Path From Failure Summary to Human-Readable Letter

The `daily_narrative.py` gathers `failure_summary.json` indirectly via the postmortem path, but I did not find direct integration of `failure_summary.json` into `_build_llm_prompt`. The Layer 5 data includes postmortem win rates and timing insights. The failure taxonomy categories are not surfaced. This means the system knows "35% of failures were timing errors" but never tells the human. The letter could be significantly more useful if the top failure category were translated into a single sentence: "My most common mistake lately: right direction, wrong timing — the move happens, but after my window closes."

---

## Summary: Risk Table

| Risk | Severity | Likelihood | Protective Mechanism Exists? |
|------|----------|-----------|------------------------------|
| Triple-alert for same macro event (SPY + ES=F + NQ=F) | High | High | Only at narrative level (dedup by ticker); not at real-time alert level |
| Unfamiliar tickers (ES=F, NQ=F) appearing in alerts without explanation | Medium | High | Only in narrative layer; plain-language alerts use raw ticker |
| ADHD reader loses interest before reaching the interesting content (new 6-section structure) | Medium | Medium | "INSTRUCTION" hints give model permission to skip flat sections |
| COT jargon ("crowded trade", "snap back") leaking into letter | Low-Medium | Low | System prompt says "no jargon" but instruction text uses jargon concepts |
| Letter length blowing past 400-word guide (600-word cap in code vs 400 in style guide) | Medium | High | No enforcement mechanism; LLM will fill sections |
| Failure taxonomy terms reaching the letter ("correlated_domains") | Low | Very Low | Taxonomy categories not plumbed into letter prompt |
| NarrativeFeedback misattributing insights to old section names | Low | Low | Functional; only affects internal attribution metadata |

## Single Most Important Fix

The real-time alert deduplication gap is the most urgent. If a macro signal enriches to SPY + ES=F + NQ=F and all three cross the convergence threshold on the same day, the human could receive multiple nearly-identical emails within minutes. The daily narrative deduplicates by ticker, but the email notifier fires on each alert independently. A simple "has this ticker+direction already been alerted in the last 2 hours" deduplication check in the email path would prevent this entirely.

The second priority is the letter length vs. ADHD focus: the style guide says 400 words, the code targets 600. The new 6-section structure makes 400 words structurally unlikely. The solution is to collapse the structure back to "hook first, then the interesting stuff, then the rest if there is any."
