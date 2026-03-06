# Team 1 Findings: Competitive Landscape
## Date: 2026-03-05
## Researcher: Team Member 1

---

### Executive Context

MIDGE's proven statistical edge (z=4.74, p<0.0001, 19.9% win rate vs 9% random) is built on cross-domain convergence — stacking 4-5 independent domain signals rather than optimizing individual signal accuracy. The research question: what are the major AI trading platforms doing, and where are their blind spots that MIDGE can exploit?

The short answer: every major platform treats domains as separate inputs or filters. None of them systematically discover and exploit multi-domain pattern *stacking* the way MIDGE does. This is a genuine structural gap.

---

### Battle-Tested Approaches

#### 1. Kensho (S&P Global) — Enterprise NLP + Knowledge Graph

- **What:** Kensho is S&P Global's AI hub. Products: Scribe (transcription), NERD (entity extraction), Extract (PDF-to-structured), Link (entity matching), Datasets (ML training data). They acquired AIS shipping data from ORBCOMM in April 2025 and data modeling firm TeraHelix in June 2025. Core asset: the S&P Global dataset moat — Capital IQ Financials, Compustat, Key Developments from 20,000 news outlets.
- **Evidence:** S&P Global is a $140B+ enterprise. Kensho's products are used by major investment banks and institutional investors. Acquired by S&P for $550M in 2018. The AIS acquisition (April 2025) signals serious intent toward maritime/supply chain data.
- **Source:** [S&P Global Kensho LLM-ready API launch](https://www.prnewswire.com/news-releases/sp-global-launches-kensho-llm-ready-api-beta-making-its-structured-data-accessible-for-generative-ai-302303392.html) (accessed 2026-03-05); [Klover.ai S&P Global AI strategy analysis](https://www.klover.ai/sp-global-ai-strategy-analysis-of-dominance-in-financial-intelligence/) (accessed 2026-03-05)
- **Fits our case because:** Kensho's AIS maritime acquisition tells us they believe in cross-domain correlation (shipping + commodity + financial) — but their products are enterprise-only NLP tools, not signal synthesis engines. They build data pipelines, not convergence engines.
- **Tradeoffs:** Completely inaccessible to retail or small-scale operators. Pricing is institutional. Their architecture is about data structuring and entity linking, not cross-domain pattern stacking. They normalize data INTO silos (financial, news, shipping) but don't systematically converge them.

**MIDGE Gap Exploited:** Kensho gives institutions clean data in organized buckets. No product exists to find the *convergence* across those buckets. That's MIDGE's job.

---

#### 2. QuantConnect (LEAN Engine) — Democratized Backtesting

- **What:** Cloud-based quantitative backtesting and live trading platform. Open-source LEAN engine. Hundreds of terabytes of data. Alternative data marketplace with vendors including Quiver Quantitative (congressional trades, insider trading, lobbying, government contracts, social media), Brain (ML sentiment, Wikipedia metrics), FRED, SEC filings, VIX Central, USDA agricultural data, ExtractAlpha, RegAlytics (regulatory alerts), Smart Insider (buybacks), EOD Historical (macro indicators, earnings, IPOs).
- **Evidence:** Active platform with thousands of users. Dataset market documented as of 2025. Quiver Quantitative congressional trading data covers 1,800 US equities back to January 2016. USDA and FRED integrations confirm macro/agricultural coverage.
- **Source:** [QuantConnect Dataset Overview](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/overview) (accessed 2026-03-05); [QuantConnect Alternative Data](https://www.quantconnect.com/docs/v2//cloud-platform/datasets/quantconnect/alternative-data) (accessed 2026-03-05)
- **Fits our case because:** QuantConnect has assembled many of the same raw data types MIDGE uses. The key difference: QuantConnect presents each dataset as a separate input that the user's algorithm must manually combine. There is no convergence engine. There is no Bayesian learning layer. There is no pattern archaeology. Users get ingredients; MIDGE gets a recipe that writes itself.
- **Tradeoffs:** QuantConnect requires Python or C# coding skill and significant learning curve. Local deployment gives up access to the full Dataset Market (local platform uses on-premise data only). Cloud-first architecture means no single-machine equivalent. L-MICRO node has 512MB RAM limit.

**MIDGE Gap Exploited:** QuantConnect users who want cross-domain signal synthesis have to build it themselves from scratch. MIDGE has built the synthesis layer that QuantConnect treats as the user's problem.

**Documented QuantConnect Gaps (as of 2026):** No weather/climate data, no shipping/AIS data (though S&P/Kensho is acquiring it for institutional clients), no real estate signals, no earnings call sentiment synthesis. Source: [QuantConnect Dataset Overview](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/overview) (accessed 2026-03-05)

---

#### 3. Alpaca — Developer-First Execution Infrastructure

- **What:** Developer-focused brokerage API for stocks, options, crypto. In 2025, launched an official MCP Server enabling AI interfaces (Claude, ChatGPT, Cursor) to execute trades directly. Market data comes from Polygon.io (real-time, historical). News from Google Search. No proprietary signal generation.
- **Evidence:** Alpaca's 2025 review documents MCP Server launch, multi-leg options, 24/5 trading, FPSL, Fixed Income access. Alpaca processes execution; signal generation is entirely the user's responsibility.
- **Source:** [Alpaca 2025 in Review](https://alpaca.markets/blog/alpacas-2025-in-review/) (accessed 2026-03-05); [Alpaca MCP Server](https://alpaca.markets/mcp-server) (accessed 2026-03-05)
- **Fits our case because:** Alpaca is pure infrastructure — the rails, not the train. MIDGE could use Alpaca as its execution layer without any architectural conflict. Alpaca has zero cross-domain signal synthesis capability. Their MCP Server enables AI agents to trade, but the AI must provide its own signals.
- **Tradeoffs:** Alpaca is US-equities-and-crypto focused. No futures. No options flow data. No signal intelligence whatsoever.

**MIDGE Gap Exploited:** MIDGE produces signals; Alpaca executes them. This is complementary, not competitive. The gap to exploit: no platform in the retail space combines MIDGE-style multi-domain convergence WITH Alpaca-style automated execution.

---

#### 4. Man Group — Agentic AI Signal Discovery (AlphaGPT)

- **What:** Man Group's quant arm (Man Numeric) built AlphaGPT — an internal agentic AI system that mimics quant researcher workflow: ideation, implementation (code writing), evaluation (backtesting). Produces trading signals autonomously. Has already produced "several dozen" signals approved for live trading. Partnered with Anthropic (Claude) in 2025.
- **Evidence:** Reported in Bloomberg (July 2025), confirmed by Man Group technical writeups. Architecture: orchestrator "Alpha Assistant" agent with sub-agents for specialized tasks. System-prompted with core principles, conventions, and access to internal proprietary codebase.
- **Source:** [Man Group AlphaGPT details](https://www.ai-street.co/p/man-group-s-alphagpt) (accessed 2026-03-05); [Man Group AI capabilities](https://www.man.com/insights/what-ai-can-do-for-alpha) (accessed 2026-03-05); [Hedgeweek Man Group article](https://www.hedgeweek.com/man-group-deploys-agentic-ai-for-quant-signal-discovery/) (accessed 2026-03-05)
- **Fits our case because:** Man Group's AlphaGPT is the closest architectural analog to MIDGE's RSI (Recursive Self-Improvement) loop — it generates, tests, and learns from signals. But critically: "most successful" in systematic equity research, with no mention of cross-domain stacking. Focus is on finding signals within financial data, not across domains. Human oversight remains mandatory ("We can't leave it unsupervised just yet").
- **Tradeoffs:** Institutional-only. Requires proprietary data access. No Bayesian learning layer reported. No pattern archaeology equivalent described. Focused on signal discovery within single-domain financial data, not cross-domain convergence.

**MIDGE Gap Exploited:** Man Group automates the quant researcher's workflow but within existing domain silos. MIDGE's convergence architecture systematically seeks the *intersection* of multiple independent domains — a category AlphaGPT doesn't appear to target.

---

### Novel Approaches

#### 5. Numerai — Crowdsourced Obfuscated Quant Model

- **What:** Crowdsourced machine learning competition that combines participant predictions into a "Stake-Weighted Meta Model" for an actual hedge fund. Obfuscated financial data (fundamentals + technical signals + market data) is provided free. Participants build ML models; Numerai trades the meta-model. $500M valuation as of November 2025. $30M Series C.
- **Why it's interesting:** Numerai solved the ensemble meta-learning problem at scale — they get thousands of independent models, each uncorrelated, then stake-weight combine them. This is the world's largest implementation of the principle MIDGE applies at domain level (independent signals = stronger combined signal). Numerai applies it at model level.
- **Evidence:** Documented tournament structure, weekly rounds, staking mechanism. JPMorgan invested $500M through Numerai. $500M valuation. Raised $30M Series C November 2025.
- **Source:** [Numerai overview](https://docs.numer.ai) (accessed 2026-03-05); [Numerai December 2025 update](https://blog.numer.ai/numerai-december-2025-update/) (accessed 2026-03-05); [Numerai $500M valuation](https://www.ainvest.com/news/numerai-ai-crowdsourced-hedge-fund-scales-500m-valuation-2511/) (accessed 2026-03-05)
- **Fits our case because:** Numerai demonstrates that combining *independent* predictive signals produces robust ensemble performance — this is the theoretical validation for MIDGE's domain-stacking approach. But Numerai's model is fundamentally closed and non-transferable: data is obfuscated, models can't be used externally, participants get NMR tokens not trading knowledge.
- **Risks:** Numerai's approach is directionally similar but architecturally inverted — they aggregate many narrowly-focused models. MIDGE is a single organism that self-specializes across domains. The Numerai architecture isn't applicable to MIDGE's single-machine, self-directed mission.

**Critical Numerai Blindspot:** Numerai's data is exclusively equity-focused, traditional financial features (P/E, RSI, short interest, analyst ratings). No weather, no shipping, no legislative, no government contracts, no options flow, no congressional trades — none of the alternative/cross-domain sources that constitute MIDGE's 11-domain model. Numerai is a sophisticated single-domain (equities) ensemble machine.

---

#### 6. Two Sigma's Venn (now Solovis/Insight Partners) — Factor Analytics

- **What:** Portfolio risk analytics platform originally built by Two Sigma, spun off and acquired by Insight Partners in January 2026, now merged with Solovis. Institutional-only. Two Sigma Factor Model used for factor decomposition. Used satellite imagery and supply chain metrics as alternative data inputs. Institutional investors only — explicitly not for retail.
- **Why it's interesting:** Two Sigma's public writings confirm they use satellite imagery, credit card flows, Fed meeting minute NLP analysis, and cross-asset multimodal AI. Their multimodal approach improved signal quality by 18% (satellite + traditional factors combined). Citadel discovered cross-market correlations (Korean consumer discretionary → European luxury goods) using multimodal AI across decades of data.
- **Evidence:** Venn sold to Insight Partners January 2026. Two Sigma confirmed satellite + credit card + NLP use. Multimodal AI results (18% improvement) cited from investment analysis publication.
- **Source:** [Insight Partners purchases Venn](https://www.themiddlemarket.com/latest-news/insight-partners-purchases-venn-from-two-sigma) (accessed 2026-03-05); [Multimodal AI in market analysis](https://investmentists.com/multimodal-ai-systems-for-market-analysis-the-future-of-trading/) (accessed 2026-03-05)
- **Fits our case because:** Two Sigma proves cross-domain correlation works at institutional scale. Their 18% improvement from satellite + traditional factors is empirical validation of domain stacking. But their architecture requires institutional infrastructure, cloud compute, and proprietary data licenses.
- **Risks:** Venn sold and restructured. Two Sigma's approach is institutionally inaccessible. Their multimodal methods require GPU processing at scale.

**MIDGE Gap Exploited:** Two Sigma and Citadel find cross-domain correlations through brute-force compute and proprietary data. MIDGE finds them through systematic domain independence testing and Thompson-weighted convergence — a methodology that scales down to a single machine.

---

### Emerging Approaches

#### 7. Probability Stacking in Retail Quant Communities

- **What:** Growing retail quant practice of combining multiple independent high-probability setups to multiply edge. Platforms like Edgeful promote "probability stacking" — finding confluence between independently high-probability setups (e.g., gap fills at 89% + opening range breakout at 65% = compounded edge).
- **Momentum:** Growing community practice, documented in 2025 retail quant blogs. Validates MIDGE's theoretical foundation from a practitioner direction.
- **Source:** [Edgeful probability stacking article](https://www.edgeful.com/blog/posts/trade-backtesting-2025-best-practices) (accessed 2026-03-05); [QuantLabs retail quants](https://www.quantlabsnet.com/post/retail-quants-the-next-stabilizing-force-in-financial-marketsintroduction) (accessed 2026-03-05)
- **Fits our case because:** Retail practitioners are discovering the same principle MIDGE implements systematically — stacking independent edges multiplies probability. The difference: retail probability stackers work manually with technical setups; MIDGE stacks 11 independent *domain* signals automatically with Bayesian learning.
- **Maturity risk:** Most retail probability stacking is applied to intraday technical setups (gap fills, ORBs), not cross-domain fundamental/alternative data synthesis. The jump from "stack two technical setups" to "stack insider+macro+government+technical+sentiment" is not yet established in retail quant practice.

---

#### 8. TenderAlpha — Government Contracts as Trading Signal

- **What:** Provider of global government procurement data (100M+ contract awards from 50+ countries) for investment analysis. In March 2025, launched Daily Macro Government Contract Spending Data Feed covering $1.8 trillion in annual public procurement from 40+ countries. Published evidence: "unexpected government receivables" (UGR) signal generates 5.4%-7.1% alpha per year.
- **Momentum:** Launched macro daily feed March 2025. Listed on FactSet marketplace. Used by institutional investors, credit rating agencies, hedge funds.
- **Source:** [TenderAlpha Daily Macro Feed launch](https://www.globenewswire.com/news-release/2025/03/13/3042078/0/en/TenderAlpha-Launches-Daily-Macro-Government-Contracting-Data-Feed-for-Real-Time-Public-Spending-Insights.html) (accessed 2026-03-05); [TenderAlpha quantitative strategies](https://www.tenderalpha.com/blog/post/quantitative-analysis/3-quantitative-strategies-based-on-alternative-data) (accessed 2026-03-05)
- **Fits our case because:** MIDGE already tracks government contracts via USASpending.gov and SAM.gov. TenderAlpha proves this is a real alpha source (5.4-7.1% annual). The key difference: TenderAlpha sells the raw data; MIDGE's ContractPredictor correlates contracts with insider trades and hiring blitz signals — which is the cross-domain stacking layer TenderAlpha doesn't provide.
- **Maturity risk:** TenderAlpha is institutional pricing. But the underlying FPDS data (fpds.gov) is free and public — MIDGE already accesses an equivalent via USASpending.gov.

---

#### 9. Quiver Quantitative — Congressional and Alternative Data Aggregator

- **What:** Retail-accessible alternative data platform aggregating congressional trades, insider trades, lobbying data, government contracts, social sentiment. API starts at $10/month. "Congressional Alpha" metric showing per-politician performance. Available as a QuantConnect dataset (1,800 equities, back to January 2016).
- **Momentum:** Active 2025 development — added Insider Confidence Index (seniority + transaction size + timing). Available via QuantConnect. Low-cost entry point.
- **Source:** [Quiver Quantitative API](https://www.quiverquant.com/) (accessed 2026-03-05); [QuantConnect Quiver integration](https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quiver-quantitative/us-congress-trading) (accessed 2026-03-05)
- **Fits our case because:** MIDGE already tracks congressional trades via house_stock_watcher.py. Quiver's Congressional Alpha metric (per-politician performance history) and Insider Confidence Index are features MIDGE doesn't yet have — filtering by committee membership + trade size + timing proximity to legislation.
- **Maturity risk:** Quiver sells aggregated data points. The alpha comes from correlating those points with other domains — which is exactly MIDGE's convergence layer. Quiver is an ingredient supplier, not a convergence engine.

---

### Gaps and Unknowns

**What research did NOT answer:**

1. **Man Group AlphaGPT cross-domain scope:** Whether AlphaGPT is constrained to single-domain financial signals or attempts multi-domain synthesis is not publicly documented. The articles are architecturally vague. This is unknowable from public sources.

2. **Exact QuantConnect alternative data pricing:** The Dataset Market pricing for individual vendors is not publicly listed in documentation. Quiver starts at $10/month for their direct API; QuantConnect's resale pricing is unclear.

3. **Institutional implementations of cross-domain stacking:** How Two Sigma and Renaissance specifically implement cross-domain correlation (their actual architecture) is proprietary. Evidence of their results (18% improvement from multimodal) comes from secondary reports, not primary technical disclosures.

4. **Congressional trades alpha decay:** Now that ETFs like NANC and GOP directly track congressional trades, and Quiver makes the data widely available, the alpha from congressional trades alone may be decaying. MIDGE's moat is not congressional trades in isolation — it's congressional trades COMBINED with contract awards COMBINED with hiring blitzes.

5. **Weather data as trading signal:** Research confirms weather data is used for agricultural futures and commodity trading, but evidence for weather as a stock-market signal in MIDGE's cross-domain stacking context is thin. Needs prototyping.

6. **Satellite data cost floor:** While prices have decreased with commercial satellite proliferation, retail-accessible satellite data with actionable time resolution is still expensive. The cheapest retail-accessible proxy (Google Trends for search interest as attention signal) is already in MIDGE.

**Where evidence was contradictory:**

- Satellite imagery: some sources cite 18% improvement in earnings estimates; others note it's now widely adopted by hedge funds and alpha is decaying. The differentiation now comes from *combining* satellite with other signals — not satellite alone.
- Congressional trade alpha: Some sources show 47% outperformance for committee leaders vs. market; others note 32% of congressional trades don't beat the market. The signal is real but concentrated in committee members with direct legislative authority over sectors they're trading.

---

### Synthesis

#### What's the strongest finding?

**The structural gap is real and large.** Every major platform (Kensho, QuantConnect, Alpaca, Numerai, Man Group's AlphaGPT) treats domains as separate layers that a human analyst or the user's own algorithm must combine. No platform systematically:

1. Ingests signals from 11+ independent domains simultaneously
2. Maintains Bayesian reliability distributions per signal type
3. Detects when 3+ independent domains converge on the same ticker/direction
4. Reverse-engineers historical moves to find the domain patterns that preceded them
5. Learns from outcomes to improve future convergence detection

This is MIDGE's structural advantage. It's not that MIDGE has better data — QuantConnect has more data sources by count. It's that MIDGE has the *synthesis layer* that no competitor provides.

#### What makes competitors successful (and what MIDGE can learn):

1. **Numerai's meta-model principle:** Stake-weighted combination of *independent* signals produces robust ensemble. MIDGE should ensure domain independence — when domains become correlated (e.g., insider buys and government contracts for the same company overlap), treat them as partial correlation, not full independence.

2. **Man Group's validation discipline:** AlphaGPT has human Investment Committee review and technology team code audits before signals go live. MIDGE's DSR (Deflated Sharpe Ratio) anti-overfitting and hypothesis probation system serve this function — but the discipline should be documented as a strength.

3. **Two Sigma's multimodal approach:** The 18% improvement from combining satellite + traditional factors proves cross-domain works at institutional scale. MIDGE's convergence engine is the retail-scale implementation of this principle.

4. **TenderAlpha's government contract signal:** 5.4-7.1% annual alpha from "unexpected government receivables" is empirically documented. MIDGE already has this domain. The question is whether MIDGE's ContractPredictor is correctly weighting government contract signals through Thompson Sampling.

#### Where are the specific competitive blind spots to exploit?

1. **Cross-domain synthesis at the signal level:** No retail platform synthesizes across all 11 domains automatically. This is MIDGE's primary differentiator.

2. **Congressional + Sector Committee + Contract correlation:** The documented alpha is concentrated in committee members (defense, tech, healthcare) who vote on legislation affecting their trading targets. MIDGE's politician_tracker.py already tracks this — but cross-referencing committee membership with the specific contracts and legislation is not yet explicit.

3. **Pattern archaeology (reverse-engineering):** No competitor platform reverse-engineers historical moves to find what domain patterns preceded them. QuantConnect enables backtesting of user-defined signals, but not autonomous pattern excavation from multi-domain signal stacks.

4. **Autonomous operation on single machine:** QuantConnect is cloud-dependent. Man Group's AlphaGPT requires human oversight. Alpaca is execution-only. MIDGE's daemon mode runs continuously on a single desktop — this is a constraint but also a differentiator (no cloud costs, no external dependencies, runs 24/7).

5. **Uncrowded data domains:** Research confirms that credit card data, satellite imagery, and social media sentiment are now "table stakes" at institutional level — highly crowded, alpha decaying. Less crowded as of 2026: government procurement data (TenderAlpha is still early), congressional committee-specific trade correlation (ETFs track raw trades but not committee-sector correlation), job tracker data combined with insider signals, AIS/shipping data (just acquired by S&P, not yet deployed in retail products).

#### What combination of approaches would work best?

The research points to a specific weakness to address: MIDGE's 19.9% win rate needs to increase. The key levers from competitive landscape research:

1. **Domain independence enforcement:** Treat highly correlated domains as partial (not full) independent signals. If insider buys and government contract wins appear for the same company in the same week, that may be one signal expressed twice, not two independent signals. The Numerai meta-model principle applies here.

2. **Committee-specific congressional filter:** Filter congressional trades by committee membership in sectors relevant to the stock. Committee members outperform rank-and-file by 40-50 percentage points annually. MIDGE's current $50K minimum filter is a blunt tool; a committee + sector filter would be more precise.

3. **Outcome window calibration:** MIDGE's outcome tracking window for pattern stacks should be calibrated against the actual lead-time of each domain. Congressional trade signals may have 7-45 day disclosure lag. Government contract signals may have 30-90 day lead time before stock reaction. These are domain-specific, not uniform.

4. **The uncrowded data edge:** The research confirms alpha decay in crowded alternative data. MIDGE should prioritize data sources that are less crowded: government procurement (TenderAlpha shows only $1.8T tracked vs. much larger universe), congressional committee-specific correlation, job tracker as pre-announcement signal. These are inherently harder to arbitrage because they require cross-domain synthesis to extract signal.

#### What the orchestrator needs to know:

- **Alpaca as execution layer is a natural fit.** MIDGE generates signals; Alpaca executes them. No architectural conflict. The MCP Server Alpaca launched in 2025 would enable MIDGE to trigger trades autonomously without a human at the keyboard.
- **Alternative data alpha decay is a real threat.** The largest hedge funds now all use the same credit card data, satellite imagery, and news sentiment — alpha from these sources alone is decaying. MIDGE's edge must come from *combination*, not any single source.
- **Man Group is the closest institutional analog.** Their AlphaGPT does autonomous signal discovery within financial data. If they extend to cross-domain stacking, they become the closest institutional competitor to MIDGE's convergence architecture. Worth monitoring.
- **Congressional trade data is being commoditized.** NANC and GOP ETFs now make raw congressional trade following mainstream. MIDGE's edge must be the *correlation* layer (congressional trade + same company's government contract award + insider cluster + hiring blitz), not congressional trades in isolation.
- **The Numerai principle validates MIDGE's design.** Numerai reached $500M valuation by aggregating independent signals. MIDGE applies the same principle (independence = strength) but at the domain level rather than model level. This is theoretically sound and empirically supported.
