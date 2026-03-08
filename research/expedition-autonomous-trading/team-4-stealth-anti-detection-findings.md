# Team 4 Findings: Stealth & Anti-Detection
## Date: 2026-03-07
## Researcher: Team Member 4

---

### Preamble: The Central Legal Finding

Before any tactical research, one fact dominates everything else in this angle: **MIDGE's entire signal corpus is built from publicly available information.** SEC filings, STOCK Act disclosures, CFTC COT reports, EIA energy data, congressional disclosures — every source MIDGE uses is legally public. Trading on public information is not insider trading. It is explicitly protected under the Mosaic Theory and affirmed by both the Supreme Court and the SEC. This is the most important anti-detection fact in this report: MIDGE cannot be accused of illegal information advantage because it has no illegal information.

The second central finding: **surveillance systems are designed to catch manipulation and speed-based gaming, not informed convergence trading.** MIDGE's threat model is almost entirely misaligned with what regulators actually watch for. The real risks are narrower and more manageable than the question implies.

---

### Battle-Tested Approaches

---

**1. Trade on public information exclusively (Mosaic Theory defense)**

- **What:** Aggregate legally public signals from multiple domains to form a trading thesis — no single piece is material non-public information (MNPI), and the synthesis is proprietary analysis.
- **Evidence:** The Mosaic Theory has been recognized by the Supreme Court and explicitly endorsed by the SEC in Regulation FD guidance. Investment analysts have used it for decades. ETFs like NANC and KRUZ (launched 2022-2023, multiple hundred million AUM by 2024) aggregate public congressional STOCK Act disclosures and trade against them commercially without legal issue. CapitolTrades, Unusual Whales, and similar data aggregators operate publicly.
- **Source:** SEC Reg FD guidance; Ballard Spahr "Politician Trading: If You Can't Stop Them, Join Them" (Oct 2024, accessed 2026-03-07); Wikipedia Mosaic Theory; https://www.ballardspahr.com/insights/alerts-and-articles/2024/10/politician-trading-if-you-cant-stop-them-join-them
- **Fits our case because:** MIDGE's 30 data sources are all public: SEC EDGAR (public), STOCK Act disclosures (public), CFTC COT (public), EIA (public), FRED (public), congressional trades (public by law), job postings (public), Google Trends (public). The convergence synthesis is proprietary. This is textbook Mosaic Theory.
- **Tradeoffs:** The Mosaic defense fails if MIDGE ever incorporates a data source that is not genuinely public. The defense also has not survived court test in isolation (Raj Rajaratnam's lawyers tried it and failed — but he had actual MNPI alongside the mosaic; MIDGE does not). Source hygiene is critical.

---

**2. Low-frequency, fundamental-driven execution (structural surveillance avoidance)**

- **What:** Execute trades at human-like frequencies (hours to days between decisions), sized within normal retail/small-institutional ranges, timed to coincide with publicly observable catalysts.
- **Evidence:** SEC MIDAS system explicitly targets HFT patterns — its core metrics are cancel-to-trade ratio, trade-to-order ratio, and odd-lot ratio. FINRA's 175+ surveillance algorithms focus on layering, spoofing, quote stuffing, momentum ignition, and marking the close. ARTEMIS targets statistically improbable pre-announcement trading timing. None of these surveillance vectors fire on a trader who: (a) holds positions for days, (b) trades at market open or close with market orders, (c) sizes below market-impact thresholds.
- **Source:** SEC MIDAS description at sec.gov/securities-topics/market-structure-analytics; FINRA "Equity Market Surveillance Today and the Path Ahead" (accessed 2026-03-07); ARTEMIS analysis via braedenanderson.com (accessed 2026-03-07)
- **Fits our case because:** MIDGE's sensing cadence fires every 50 steps, source rotation covers 30 sources over time, convergence alerts require multi-day signal accumulation across independent domains. MIDGE is structurally a slow, fundamental convergence trader. It does not generate the timing patterns that trigger any known surveillance algorithm.
- **Tradeoffs:** Low frequency means fewer trades, potentially less return per period. Cannot exploit intraday dislocations. Must hold through short-term volatility.

---

**3. Personal retail brokerage account — no registration, no disclosure required**

- **What:** An individual trading their own capital through a personal brokerage account (Alpaca, Interactive Brokers, TD Ameritrade) using automated software requires no SEC registration, no algorithm disclosure, and no special compliance filings, as long as they are trading their own money and not managing others' accounts.
- **Evidence:** The SEC and FINRA registration requirements apply to: broker-dealers (effecting transactions for others), investment advisers (advising others for compensation), and commodity trading advisors (advising others on commodity accounts). A retail trader automating their own personal account falls outside all three categories. IBKR explicitly offers retail algorithmic trading infrastructure. Alpaca markets itself to individual retail algo traders with zero-commission API trading and no registration requirement.
- **Source:** JustAnswer legal analysis at justanswer.com/business-law/so6z5; Alpaca documentation at alpaca.markets/algotrading; IBKR Quant "Retail Algorithmic Trading: A Complete Guide"; FINRA "Know the Risks of Auto-Trading Services" (noting registration applies to the *service provider*, not the retail trader using their own account)
- **Fits our case because:** Guiding Light is trading their own capital. There is no client relationship, no advisory fee, no third-party account management. The only rule that applies is: do not manipulate, do not trade on MNPI, comply with Pattern Day Trader rules ($25K minimum equity for margin day trading). MIDGE already avoids all of these.
- **Tradeoffs:** The $25,000 PDT minimum applies if holding a margin account and making 4+ day trades within 5 business days. Solvable by: (a) maintaining $25K+ account equity, (b) using a cash account, or (c) trading at swing/position trade frequency (holding overnight). The last option aligns with MIDGE's convergence timeframes anyway.

---

**4. TWAP/VWAP execution and iceberg slicing for position-size stealth**

- **What:** Break large intended positions into time-distributed or volume-weighted slices. Only reveal a small portion of the total order at any time (iceberg). This minimizes market footprint, reduces slippage, and avoids triggering anomalous volume detection.
- **Evidence:** Institutional standard for 25+ years. Almgren-Chriss optimal execution model (2000) is the mathematical foundation behind every major broker's execution algorithm. Anti-gaming logic with randomized slice sizing is standard in every execution algorithm from all major vendors. Per the MQL5 algorithmic trading guide (2025): "Iceberg algorithms use randomised slicing to reduce detection." Dark pool routing (IBKRATS, available to retail via IB) handles 51.8% of U.S. equity volume off-exchange as of early 2025.
- **Source:** Almgren-Chriss model documentation at simtrade.fr; MQL5 execution algorithms article (accessed 2026-03-07); Dark Pools overview at stocktitan.net; IBKR ATS access at thenewcomrade.com; off-exchange volume 51.8% per ainvest.com (2025)
- **Fits our case because:** MIDGE at its current scale (personal account, $50K paper account equivalent) will trade sizes well below market-impact thresholds on any liquid equity or futures. For positions under 0.1% of average daily volume, market-impact cost is negligible and no slicing is needed. As account grows, TWAP/VWAP slicing prevents fingerprinting by size pattern.
- **Tradeoffs:** TWAP/VWAP adds execution complexity. Slicing increases partial-fill risk. For small accounts, the benefit is marginal; it matters at $1M+ positions.

---

### Novel Approaches

---

**1. Prediction markets as the primary first domain (lowest surveillance, AI-native ecosystem)**

- **What:** Polymarket and Kalshi are the surveillance-light markets where AI bots are already operating openly and legally. Polymarket processes $18B+ annual volume with no per-trade surveillance comparable to FINRA. Kalshi is CFTC-regulated but events-based (not equity), with different surveillance vectors than stock markets.
- **Why it's interesting:** The prediction market ecosystem has explicitly built out AI bot infrastructure. Polymarket's official GitHub repo (`Polymarket/agents`) is a developer framework for AI agents trading Polymarket. Coinbase has launched Agentic Wallets for AI agent autonomous crypto/DeFi execution. A documented two-layer AI system trading Polymarket and Kalshi autonomously was published on Dev Genius (March 2026). Academic research documented $40M+ in arbitrage profits extracted from prediction markets between April 2024 and April 2025. The infrastructure exists and operates legally.
- **Evidence:** Polymarket received CFTC Amended Order of Designation in late 2025, enabling U.S. retail access via registered intermediaries. The CFTC granted Polymarket, Kalshi, PredictIt, and Gemini no-action letters exempting them from certain recordkeeping requirements. Hedge funds can now deploy algorithms on Polymarket through corporate KYC entities.
- **Source:** CoinDesk "Polymarket Secures CFTC Approval for Regulated U.S. Return" (Nov 2025); PRNewswire Polymarket CFTC announcement (accessed 2026-03-07); QuantVPS "Polymarket HFT: How Traders Use AI" (accessed 2026-03-07); GitHub Polymarket/agents repo
- **Fits our case because:** MIDGE's cross-domain convergence signals (congressional trades, macro data, energy inventories, insider activity) map directly onto Polymarket event contracts about policy outcomes, economic events, and corporate news. MIDGE can bet on whether a FOMC rate cut happens based on macro convergence. Whether an energy company gets a contract based on insider + government signals. This is MIDGE's exact intelligence applied to a lower-surveillance, AI-welcoming market.
- **Risks:** Prediction market liquidity is thinner than equity markets. Position sizing is constrained. The regulatory environment is actively evolving (CFTC announced new rulemaking coming in 2026). Event contracts require predicting discrete outcomes, not directional price moves, which is a different analytical problem than MIDGE currently solves. Kalshi takes ~2% fees.

---

**2. DeFi/crypto autonomous execution via agentic wallets**

- **What:** AI agents can hold crypto wallets, execute trades on DEXes, manage DeFi positions, and pay for their own compute costs — entirely autonomously with no human intervention required. Coinbase's Agentic Wallet platform (launched 2025) and the x402 payment protocol enable self-sustaining agent economies.
- **Why it's interesting:** This is the closest to "turn it on and walk away" that exists today. The AI agent is its own legal entity with its own wallet. There is no broker, no registration, no surveillance authority with equities-style power. Crypto markets run 24/7, fitting MIDGE's always-on daemon mode exactly.
- **Evidence:** Coinbase launched "Agentic Wallets" with the stated purpose: "give AI agents the power to spend, earn, and trade autonomously while maintaining enterprise-grade security and programmable guardrails." The AI x Crypto market grew from ~$14B in late 2024 to an estimated $20-39B by mid-2025. Multiple published case studies of AI agents generating monthly profits on prediction markets and DeFi strategies in 2025.
- **Source:** Ledger "DeFAI Explained" (accessed 2026-03-07); Henley & Partners Crypto Wealth Report 2025 "When AI Agents Become Crypto Millionaires"; CoinMarketCap "What's Next for AI" 2026 predictions; medium.com/@gwrx2005 DeFi agent integration guide
- **Fits our case because:** MIDGE already has crypto signals (CoinGecko, CoinCap). The 24/7 nature of crypto fits the daemon mode. Self-funding via DeFi yields or market profits is the closest path to true financial independence. The regulatory overhead is minimal for a personal-use agent.
- **Risks:** Crypto is highly volatile — MIDGE's current convergence signals are equity-domain heavy. DeFi smart contract risk (protocol hacks). Gas fees erode small-position profits. Crypto markets have different manipulation patterns (wash trading is rampant, especially in low-cap tokens) that MIDGE's deception detector may need to be calibrated for. Regulatory status of DeFi is still evolving.

---

**3. Timing trades to follow, not precede, publicly observable catalysts**

- **What:** Rather than trading before an event (which looks like MNPI-based front-running to ARTEMIS), time entries to coincide with or immediately follow the public announcement of the convergence driver. "Insider bought, contract awarded, energy data confirmed" — trade when all three are public, not when you predict they'll converge.
- **Why it's interesting:** ARTEMIS's primary detection mechanism is timing correlation — does the trade happen suspiciously *before* the announcement? A trader who enters *at the moment* of the last public confirmation cannot be accused of trading on MNPI. The statistical improbability test requires comparing trade timing to corporate event timing. If the "event" is the public disclosure itself, that test fails to generate a signal.
- **Evidence:** ARTEMIS "cross-references voluminous trade data with critical timelines of corporate events, uncovering suspicious correlations that often point to the misuse of material nonpublic information." The detection is specifically pre-announcement proximity. Post-announcement trades are explicitly not MNPI-based by definition. SEC enforcement literature consistently focuses on pre-event trading windows.
- **Source:** ARTEMIS analysis at braedenanderson.com (accessed 2026-03-07); SEC "How the SEC Detects Insider Trading" via nyccriminalattorneys.com; SEC enforcement 2025 year-in-review at Holland & Knight
- **Fits our case because:** MIDGE could implement a "public confirmation delay" — don't fire a trade signal until the last domain's signal is from a public source with a verifiable timestamp. This means: insider trade is in EDGAR (public, verifiable), congressional trade is in STOCK Act filing (public, verifiable), energy data is in EIA weekly report (public, verifiable). All triggers are timestamped public events. The trade happens after the last confirmation is public.
- **Risks:** This approach gives up some edge — the convergence is strongest when you act before the market prices in the combined signal. Waiting for all public confirmations means the market may already be moving. However, for slow-moving structural patterns (energy + insider + government), the market typically under-reacts even after individual pieces go public. The multi-domain insight is still ahead of the market even when each piece is individually public.

---

### Emerging Approaches

---

**1. Machine learning-driven execution randomization (behavioral camouflage)**

- **What:** Instead of fixed TWAP/VWAP schedules, use RL (reinforcement learning) models that randomize order sizing, timing jitter, and venue selection based on market microstructure — making the execution pattern statistically indistinguishable from a human trader.
- **Momentum:** Reinforcement learning for optimal execution is a rapidly growing research area. Multiple papers published 2024-2025 specifically address minimizing market impact and detection footprint via RL execution. The arXiv paper "Deep Learning for VWAP Execution in Crypto Markets: Beyond the Volume Curve" (Feb 2025) applies deep learning to execution optimization.
- **Source:** arXiv 2502.13722v2 "Deep Learning for VWAP Execution in Crypto Markets" (Feb 2025); ScienceDirect "Deep unsupervised anomaly detection in high-frequency markets" (2024); ResearchGate AI-driven optimization of HFT strategies (2025)
- **Fits our case because:** At scale, if MIDGE's execution pattern is identifiable (same time of day, same order sizes, same venue), a surveillance system could fingerprint it. RL-driven randomization prevents fingerprinting before it becomes a problem. This is a "build before you need it" consideration.
- **Maturity risk:** Implementing RL execution requires backtesting against historical order book data. No off-the-shelf retail solution exists for this; it requires custom development. At MIDGE's current scale ($50K account), this is premature.

---

**2. Consolidated Audit Trail (CAT) — growing surveillance reach**

- **What:** The SEC's CAT system began collecting "almost all US trading data" in May 2024, giving the SEC visibility across all exchanges and brokers into a single database. This is a long-term surveillance capability expansion that narrows previously existing data gaps.
- **Momentum:** CAT is actively being expanded and cross-referenced with ARTEMIS/MIDAS outputs. The SEC is building out customer-level attribution that will eventually track individual retail traders across all venues.
- **Source:** Congress.gov CRS Report IF13103 on CAT and AI adoption in SEC surveillance (2024); FINRA 2026 Annual Regulatory Oversight Report (Dec 2025) noting new focus on AI in surveillance; McGuireWoods analysis of FINRA 2026 oversight report
- **Fits our case because:** CAT's expansion means that data gaps in surveillance (MIDAS didn't see dark pools, futures, or OTC) are being closed. Trading strategies that rely on surveillance gaps may face future exposure. MIDGE's best protection is not exploiting gaps — it's having genuinely clean, public-information-only trading logic. That defense holds regardless of how comprehensive CAT becomes.
- **Maturity risk:** CAT is still being built out. Customer attribution was phased in gradually. Cross-market integration with MIDAS/ARTEMIS is not yet complete. Current surveillance has meaningful gaps — but those gaps should not be relied upon as a permanent protection strategy.

---

**3. Futures markets as the primary equity-adjacent execution venue**

- **What:** CME futures (ES, NQ, crude oil, natural gas, agricultural) offer the same directional exposure as equities and commodities with different surveillance infrastructure. CFTC regulates futures; SEC regulates equities. MIDS/ARTEMIS do not cover futures trades.
- **Momentum:** CFTC's Regulation Automated Trading (Reg AT) was proposed in 2015, supplemented in 2016, and then **withdrawn in 2020** in favor of lighter-touch Electronic Trading Principles. As of 2026, there is no equivalent to Reg NMS/CAT for futures trading at the retail level. The CFTC's focus has shifted to crypto prediction markets, not retail futures algorithms.
- **Source:** Morgan Lewis "Farewell Reg AT, Hello Electronic Trading Principles" (2020); CFTC Reg AT withdrawal documentation; Federal Register Regulation Automated Trading history
- **Fits our case because:** MIDGE's signals (energy data, COT positioning, macro indicators) have direct futures expression. An energy convergence signal (EIA inventory + COT positioning + insider activity in energy companies) can be expressed as a crude oil futures long rather than an equity long. The payoff math is linear (futures), which Guiding Light stated as preferable. SEC surveillance does not cover CME futures positions.
- **Maturity risk:** Futures require margin management, roll management, and understanding of contract specifications. CME Group does have its own market surveillance (they feed anomalies to the CFTC), but it targets manipulation (spoofing, wash trading) not informed convergence trading. Retail futures trading via IB or Alpaca is accessible but requires understanding position sizing in notional terms.

---

### Gaps and Unknowns

**What this research did NOT fully answer:**

1. **Exact CAT attribution reach for retail traders today.** It is clear CAT is expanding, but the precise current state of customer-level attribution for retail algo traders (vs. institutional) is not definitively established in public sources as of March 2026. This requires a direct compliance consultation to determine.

2. **Prediction market execution infrastructure for MIDGE's signal types.** It is clear that Polymarket/Kalshi AI bots work and are legal. It is not clear how directly MIDGE's domain-level convergence signals map onto currently available event contract markets. A domain-to-contract mapping exercise would be needed.

3. **IBKR/Alpaca detection of automated accounts.** Brokers have their own internal pattern detection for accounts that may be front-running, manipulating, or violating margin rules algorithmically. The specific thresholds and triggers for broker-level account investigation are not publicly documented. This is a practical operational risk that exists below the regulatory level.

4. **Interaction between paper trading confidence gates and "unusual options activity" flags.** MIDGE's alerts at confidence > 0.45 on specific tickers could coincide with other informed traders' activity, which might create a correlated anomalous signal. If multiple informed traders all pile into the same convergence, the combined pattern could look like coordinated activity to FINRA's cross-account surveillance. This is a low-probability but non-zero risk.

5. **Congressional trade lag as a MNPI edge question.** Congressional STOCK Act disclosures have 30-45 day lag. Trading on a signal that includes a congressional disclosure that references a 45-day-old trade is clearly public. But what if MIDGE detects a pattern of *recent* congressional activity (within the disclosure window) combined with other signals? This is a gray area that has not been tested in court.

**Where evidence was thin or contradictory:**

- Mosaic Theory as a standalone legal defense: legally recognized in theory, but has never successfully defended a case in isolation when other suspicious factors existed. The theory is strongest when the entire signal corpus is verifiably public.
- FINRA's ability to distinguish "informed human" from "informed algorithm" in practice: surveillance documentation emphasizes behavioral modeling intent, but no public source documents the specific thresholds that separate legitimate informed trading from suspicious algorithmic activity.

---

### Synthesis

**The surveillance threat to MIDGE is almost entirely misaligned with what regulators actually watch for.**

Here is what SEC/FINRA surveillance is designed to detect, in priority order:

1. **Speed-based manipulation** (HFT spoofing, layering, quote stuffing, momentum ignition) — MIDGE does none of this
2. **Pre-announcement trading with MNPI** (ARTEMIS timing correlation) — MIDGE trades on public data only
3. **Coordinated cross-account manipulation** (wash trading, ramping) — MIDGE is a single-account retail trader
4. **Market impact gaming** (large orders overwhelming liquidity) — MIDGE at $50K scale is below any threshold
5. **Cross-product manipulation** (correlated stocks + options + derivatives gaming) — MIDGE does not layer derivatives on its equity positions

MIDGE fits zero of these patterns. The correct stealth model for MIDGE is not "hide what you're doing" — it is "do legal things that look like what they are: an exceptionally well-read informed investor who synthesizes public information before the market does."

**The strongest natural protection MIDGE has:**

MIDGE's cross-domain approach naturally mimics an extremely well-informed human fundamental investor because that is exactly what it is. An investor who reads SEC filings, monitors congressional STOCK Act disclosures, tracks CFTC COT data, watches EIA energy reports, and tracks Google Trends for interest signals is doing what the best hedge fund analysts do manually. The fact that MIDGE automates the aggregation and synthesis is legally irrelevant to the question of whether the underlying activity is lawful. It is lawful.

The only genuine legal exposure for MIDGE is:

1. **Inadvertent MNPI contamination** — if MIDGE ever incorporates a data source that is not genuinely public (e.g., a data vendor selling aggregated but technically MNPI-adjacent data), the Mosaic defense weakens. Source hygiene is the primary compliance obligation.
2. **Market manipulation** — if MIDGE's trade sizes ever become large enough to move prices in thinly traded securities, the convergence signal could be self-fulfilling in a way that looks manipulative. At $50K this is impossible; at $10M in a microcap, it requires attention.
3. **Pattern Day Trader rule** — operational, not legal risk. Maintain $25K+ equity or use cash account or hold overnight.

**Recommended market order for MIDGE's stealth profile:**

| Market | Surveillance Level | MIDGE Signal Fit | Recommended Order |
|--------|-------------------|-----------------|-------------------|
| Prediction markets (Kalshi/Polymarket) | Low, AI-welcoming | High (policy, economic events) | First — lowest friction, AI-native |
| Crypto/DeFi | Minimal for autonomous agents | Medium (crypto domain already built) | Second — for 24/7 self-funding loop |
| CME Futures | CFTC/CME (not SEC/FINRA) | High (energy, macro, positioning) | Third — once fundamental signals mature |
| US Equities (large-cap) | Highest (MIDAS+ARTEMIS+CAT+FINRA) | High (insider, institutional, technical) | Fourth — legally clean but most watched |
| US Equities (micro-cap) | Low surveillance, HIGH manipulation risk | Medium | Last — manipulation concern cuts both ways |
| Forex OTC | Lowest surveillance globally | Low (MIDGE has limited FX signals) | Do not prioritize — signal gap |

**The single most important execution discipline:**

Implement a "public confirmation timestamp" requirement in MIDGE's trade execution gate. Before firing any trade, verify that the last-triggering domain signal has a verifiable public-record timestamp (EDGAR filing, STOCK Act disclosure, EIA weekly report). Log this timestamp with every trade. This creates an audit trail demonstrating that every trade was triggered by a public information event — the strongest possible legal defense against any future inquiry.

**On detection by sophisticated market participants (not regulators):**

The secondary detection risk is not regulatory — it is competitive. Other quantitative traders who notice MIDGE's positions before they move could front-run or reverse-engineer the strategy. Countermeasures: vary trade timing relative to the signal confirmation (add hours of random delay), use IBKR ATS for dark pool routing on entries, avoid predictable ticker patterns. These are low-priority at $50K scale but become important as the account grows.

---

### Sources

- SEC MIDAS system: https://www.sec.gov/securities-topics/market-structure-analytics/midas-market-information-data-analytics-system
- FINRA 2025 Annual Regulatory Oversight Report: https://www.finra.org/sites/default/files/2025-01/2025-annual-regulatory-oversight-report.pdf
- FINRA 2026 Annual Regulatory Oversight Report: https://www.finra.org/sites/default/files/2025-12/2026-annual-regulatory-oversight-report.pdf
- FINRA Equity Market Surveillance: https://www.finra.org/media-center/speeches-testimony/equity-market-surveillance-today-and-path-ahead
- FINRA Manipulative Trading 2025: https://www.finra.org/rules-guidance/guidance/reports/2025-finra-annual-regulatory-oversight-report/manipulative-trading
- ARTEMIS surveillance analysis: https://braedenanderson.com/insights/big-data-is-watching-you-how-the-sec-uses-advanced-analytics-to-uncover-violations
- FINRA Auto-Trading / Unregistered Entities: https://www.finra.org/investors/insights/auto-trading-unregistered-entities
- SEC Algorithmic Trading Report 2020: https://www.sec.gov/files/algo_trading_report_2020.pdf
- FINRA Algorithmic Trading guidance: https://www.finra.org/rules-guidance/key-topics/algorithmic-trading
- eFlow Market Manipulation Red Flags: https://eflowglobal.com/insights/blogs/high-impact-market-manipulation-tactics-red-flags-for-modern-surveillance-teams/
- Sidley Austin AI/Securities Guidelines: https://www.sidley.com/en/insights/newsupdates/2025/02/artificial-intelligence-us-financial-regulator-guidelines-for-responsible-use
- Katten AI for Broker-Dealers: https://katten.com/ai-for-broker-dealers-and-investment-advisers-legal-and-regulatory-considerations
- Is Automated Trading Legal: https://advancedautotrades.com/is-automated-trading-legal/
- Alpaca retail algo trading: https://alpaca.markets/algotrading
- IBKR Retail Algorithmic Trading Guide: https://www.interactivebrokers.com/campus/ibkr-quant-news/retail-algorithmic-trading-a-complete-guide/
- Congressional Trading "Join Them" analysis: https://www.ballardspahr.com/insights/alerts-and-articles/2024/10/politician-trading-if-you-cant-stop-them-join-them
- Mosaic Theory Wikipedia: https://en.wikipedia.org/wiki/Mosaic_theory_(investments)
- Insider Trading Defense 2025: https://attorneys.media/insider-trading-defense-strategies/
- SEC Insider Trading enforcement 2025: https://natlawreview.com/article/insider-trading-likely-continued-focus-sec-enforcement
- Polymarket CFTC approval: https://www.coindesk.com/business/2025/11/25/polymarket-secures-cftc-approval-for-regulated-u-s-return
- Polymarket CFTC no-action: https://www.coindesk.com/policy/2025/12/11/cftc-gives-no-action-leeway-to-polymarket-gemini-predictit-ledgerx-over-data-rules
- Polymarket AI agents GitHub: https://github.com/Polymarket/agents
- Polymarket HFT and AI: https://www.quantvps.com/blog/polymarket-hft-traders-use-ai-arbitrage-mispricing
- Prediction market legal landscape Nov 2025: https://nexteventhorizon.substack.com/p/where-things-stand-for-prediction-markets-legally
- DeFAI / AI agents in DeFi: https://www.ledger.com/academy/topics/defi/defai-explained-how-ai-agents-are-transforming-decentralized-finance
- Henley Crypto Wealth Report 2025: https://www.henleyglobal.com/publications/crypto-wealth-report-2025/when-ai-agents-become-crypto-millionaires
- Off-exchange dark pool volume 51.8%: https://www.ainvest.com/news/dark-pools-center-stage-navigating-liquidity-shifts-equity-markets-2507/
- TWAP/VWAP/Iceberg execution: https://www.mql5.com/en/articles/17934
- arXiv Deep Learning for VWAP Execution: https://arxiv.org/html/2502.13722v2
- HFT Surveillance Trapets guide: https://www.trapets.com/resources/blog/high-frequency-trading-surveillance-guide
- CFTC Reg AT withdrawal: https://www.morganlewis.com/pubs/2020/07/farewell-reg-at-hello-electronic-trading-principles
- Order-to-trade ratio regulation: https://www.nortonrosefulbright.com/en/knowledge/publications/6d7b8497/mifid-ii-mifir-series
- Almgren-Chriss execution model: https://www.simtrade.fr/blog_simtrade/understanding-almgren-chriss-model-for-optimal-trade-execution/
- SEC enforcement 2025 year-in-review: https://www.hklaw.com/en/insights/publications/2025/12/sec-enforcement-2025-year-in-review
