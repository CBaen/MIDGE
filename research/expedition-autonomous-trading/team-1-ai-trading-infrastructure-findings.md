# Team 1 Findings: Autonomous AI Trading Infrastructure
## Date: 2026-03-07
## Researcher: Team Member 1

---

### Battle-Tested Approaches

---

#### 1. Alpaca Trading API + alpaca-py + MCP Server

- **What:** Developer-first brokerage API for stocks, ETFs, options, and crypto with commission-free trading, a Python SDK (`alpaca-py`), paper trading environment, and an official MCP server for LLM-native integration.
- **Evidence:** Rated the best US broker for algorithmic trading in 2026 by BrokerChooser after evaluating 100+ brokers. Paper trading environment is free, requires no minimum balance, and uses real market data with $100K simulated funds. The MCP server is in active use with Claude, ChatGPT, VSCode Cursor, and Gemini CLI as of November 2025. Thousands of algorithmic traders use it in production.
- **Source:** [Alpaca Trading API](https://alpaca.markets/) (accessed 2026-03-07); [Alpaca MCP Server GitHub](https://github.com/alpacahq/alpaca-mcp-server) (accessed 2026-03-07); [BrokerChooser Best Algo Trading Brokers 2026](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-states) (accessed 2026-03-07)
- **Fits our case because:** MIDGE has a paper trading system already using Kelly position sizing — Alpaca's paper environment is a drop-in bridge from simulated to live execution. The MCP server means MIDGE's convergence alerter could route alerts directly to order execution via natural-language tool calls with no custom broker integration code. Alpaca supports fractional shares, crypto, and options — covering MIDGE's cross-asset capability. REST API + WebSocket for real-time data. Python-native.
- **Tradeoffs:** US persons only. Stocks/ETFs are subject to SEC/FINRA pattern day trading rules (25K minimum for day trading). Alpaca is a retail broker — large institutional-scale position sizing will eventually require a prime broker relationship. Not a futures broker (no E-mini S&P, no commodity futures via Alpaca directly). Rate limits exist (not publicly specified, but standard API limits apply).

---

#### 2. Polymarket CLOB API + py-clob-client

- **What:** Polymarket's Central Limit Order Book (CLOB) API running on Polygon blockchain, with a Python client (`py-clob-client`), that enables algorithmic bots to trade binary prediction contracts.
- **Evidence:** Polymarket processed over $44 billion in trading volume in 2025. The official agents repository on GitHub has production-deployed bots. PolyBro, Billy Bets, and Semantic 42 are documented production trading agents. One documented bot executed 8,894 trades generating ~$150K without human intervention by exploiting mispricings. AI-powered models generated $2.2M in profits in two months on Polymarket. Multiple open-source implementations exist. PolySimulator launching a paper trading API in early 2026.
- **Source:** [Polymarket Agents GitHub](https://github.com/Polymarket/agents) (accessed 2026-03-07); [Definitive Guide to Polymarket Ecosystem](https://defiprime.com/definitive-guide-to-the-polymarket-ecosystem) (accessed 2026-03-07); [Polymarket API Guide Medium](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf) (accessed 2026-03-07); [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client) (accessed 2026-03-07)
- **Fits our case because:** Prediction markets pay off in binary yes/no outcomes — MIDGE's convergence alerts are directional (bullish/bearish), which maps directly to long/short binary contracts. Polymarket runs 24/7 on Polygon, matching MIDGE's always-on daemon architecture. No broker account, no KYC for non-US persons. The CLOB supports both limit and market orders. Open-source agent framework exists as a starting point.
- **Tradeoffs:** Terms of Service explicitly prohibit US persons from trading (both UI and API). Runs on Polygon (MATIC gas costs, though low). Rate limit of 60 orders/minute per API key — will require exponential backoff. Liquidity concentrated in political/crypto markets, not equity domains where MIDGE has its edge (insider trading, congressional activity). Settlement in USDC stablecoin, not USD. Requires private key management for a Polygon wallet. Authentication uses EIP-712 signatures — non-trivial implementation.

---

#### 3. Kalshi API (CFTC-Regulated, US Legal)

- **What:** The first CFTC-designated prediction market exchange in the US, offering REST, WebSocket, and FIX 4.4 interfaces for event-based contract trading, with an official Python client.
- **Evidence:** Became the first CFTC-designated DCM (Designated Contract Market) for event contracts in 2020. In H1 2025, produced over $200M in revenue. As of late September 2025, commands 62% of on-chain prediction market volume ($500M+ weekly). Raised $185M at a $2B valuation. In early 2026, the CFTC officially classified prediction markets as "swaps" under federal jurisdiction — strengthening Kalshi's legal position. Python SDK (`kalshi-python`) available on PyPI.
- **Source:** [Kalshi API Documentation](https://docs.kalshi.com/welcome) (accessed 2026-03-07); [Kalshi Rate Limits](https://docs.kalshi.com/getting_started/rate_limits) (accessed 2026-03-07); [Kalshi API Guide](https://zuplo.com/learning-center/kalshi-api) (accessed 2026-03-07); [Kalshi Overview](https://kalshi.com/about) (accessed 2026-03-07)
- **Fits our case because:** Fully legal for US persons. CFTC regulation means Kalshi is unlikely to be shut down — regulatory risk is near-zero compared to Polymarket. Markets cover economic indicators (CPI, FOMC, NFP), weather, and geopolitical events — domains that MIDGE already tracks via its EIA energy, economic calendar, and FRED macro sources. The same cross-domain convergence logic applies: if MIDGE's macro + energy + technical signals converge bullishly on oil, MIDGE could trade the Kalshi "Will oil close above $X?" market. FIX 4.4 support means institutional-grade connectivity is available when needed.
- **Tradeoffs:** Markets are narrower than Polymarket — fewer total markets, less liquidity on tail events. Authentication tokens expire every 30 minutes, requiring re-authentication logic in the daemon. Rate limits apply (429 on breach). Markets settle at binary payoffs — no continuous profit curve, just 0 or 1 per contract. No continuous 24/7 operation for all markets (event-based, not perpetual). Requires funded account and KYC.

---

#### 4. Interactive Brokers TWS API (Stocks + Futures + Options)

- **What:** Professional-grade brokerage API connecting to IB's TWS (Trader Workstation) or IB Gateway, supporting stocks, futures, options, forex, and bonds via Python (native API), with paper trading available.
- **Evidence:** IBKR is the gold standard for institutional algorithmic trading in the US. Free API for all clients, no minimum for paper trading. Python native API has been in production use for years. TWS API v10.37 (2025 updates) includes advanced order types. Supports E-mini S&P futures, crude oil futures, treasury futures — the exact instruments MIDGE's brief calls out as "linear payoff math." Used by hedge funds and proprietary trading firms globally.
- **Source:** [IBKR Trading API](https://www.interactivebrokers.com/en/trading/ib-api.php) (accessed 2026-03-07); [IBKR Python TWS API Guide](https://www.interactivebrokers.com/campus/ibkr-quant-news/interactive-brokers-python-api-native-a-step-by-step-guide/) (accessed 2026-03-07); [Automated Trading with IBKR Python API](https://www.pyquantnews.com/free-python-resources/automate-trading-with-interactive-brokers-python-api) (accessed 2026-03-07)
- **Fits our case because:** The Research Brief specifically calls out "instruments where payoff math is linear (futures-like)." IBKR directly supports commodity futures (crude oil, natural gas — domains MIDGE already has via EIA), equity index futures (E-mini S&P, NASDAQ), and treasury futures. MIDGE's cross-domain convergence (energy + macro + congressional + technical) could drive futures signals. IBKR paper trading replicates the exact execution environment.
- **Tradeoffs:** Requires a running instance of TWS or IB Gateway at all times — adds infrastructure dependency (GUI application or headless IB Gateway process must be kept alive alongside MIDGE daemon). More complex connection management than REST APIs. IB Gateway requires periodic manual re-authentication (configurable up to 8-hour sessions). The API communicates locally (socket connection to TWS/Gateway), not via cloud REST — complicates containerized deployment. Futures require funded commodity account, additional margin requirements.

---

### Novel Approaches

---

#### 5. Coinbase AgentKit (Agentic Wallets) for Crypto + Self-Funding Loop

- **What:** Coinbase's CDP (Coinbase Developer Platform) AgentKit — a Python toolkit for AI agents to create wallets, execute onchain operations (swaps, transfers, DeFi), and use x402 protocol to pay for their own API costs autonomously, without human intervention.
- **Why it's interesting:** This is the only infrastructure specifically designed to let an AI agent be financially self-sustaining — earning from trades, paying for compute via x402 micropayments, and reinvesting, all without human approval for routine operations. The x402 protocol (built on HTTP 402 status code) has processed 100M+ transactions since launch in 2025. It directly addresses the Research Brief's "self-funding loop" requirement.
- **Evidence:** Coinbase launched Agentic Wallets in February 2026 — the first wallets built specifically for AI systems. Supports EVM chains (Base L2) and Solana. Backed by Trusted Execution Environments (TEEs) for key security. Smart Wallet Provider uses ERC-4337 account abstraction with optional gasless transactions via paymaster. x402 protocol processed 15M+ transactions in January 2026 alone (100M+ total since launch). LangChain integration pattern is two lines of code: `tools = get_langchain_tools(agent_kit)`. 13,000+ agents with wallets reported active as of late 2025.
- **Source:** [Coinbase Agentic Wallets Launch](https://www.coinbase.com/developer-platform/discover/launches/agentic-wallets) (accessed 2026-03-07); [AgentKit Python Documentation](https://coinbase.github.io/agentkit/coinbase-agentkit/python/index.html) (accessed 2026-03-07); [Coinbase AgentKit Q1 Update](https://www.coinbase.com/developer-platform/discover/launches/agentkit-q1-update) (accessed 2026-03-07); [x402 Protocol](https://www.x402.org/) (accessed 2026-03-07); [AI Agent Economics Guide 2026](https://academy.exmon.pro/ai-agent-economics-how-autonomous-crypto-wallets-work-2026-guide) (accessed 2026-03-07)
- **Fits our case because:** Closes the self-funding loop. MIDGE trades crypto via convergence signals (CoinGecko + CoinCap sources already integrated), earns profit in USDC on Base L2, uses x402 to pay for Anthropic API tokens, Polygon gas, and data subscriptions — all programmatically. No human ever moves money. The AgentKit LangChain integration is compatible with MIDGE's Python stack (mae-core is pure Python). Gasless transactions on Base via paymaster eliminate gas cost as a variable.
- **Risks:** Currently limited to crypto/DeFi markets — cannot directly trade stocks or futures via AgentKit. Crypto markets are 24/7 but also more adversarial (front-running, MEV bots). Base L2 is Coinbase-controlled — not fully decentralized (censorship risk exists, though practically low). DeFi smart contract risk (bugs, exploits). x402 for API monetization is brand new and has limited ecosystem adoption outside Coinbase-affiliated services.

---

#### 6. Olas / Polystrat Architecture: Open-Source Agent on Safe Smart Account + Polymarket

- **What:** Polystrat is a fully autonomous, self-custodial AI prediction market trading agent built by Olas, running locally via Pearl, using Gnosis Safe smart accounts on Polygon, with natural-language strategy setting and hardcoded critical wallet functions.
- **Why it's interesting:** This is the first documented production autonomous trading agent with a public verifiable activity log, running on audited smart contract infrastructure (Safe), that explicitly chose to hardcode critical financial functions outside the LLM to prevent "rogue" behavior. This is the stealth/safety design pattern MIDGE needs.
- **Evidence:** Launched February 2026. Within 2 weeks: 4,200+ trades executed. Many individual trades with returns exceeding 300%. Uses Safe (Gnosis) for custody — the same infrastructure securing $100B+ in DeFi assets. Safe Watch Agent (fraud detection) is a companion security layer. Half of all Safe transactions on Gnosis Chain in 2023-2024 already made by AI agents. The architecture "really restricts the capability of the agent" as a deliberate safety choice.
- **Source:** [Olas Polystrat Launch](https://olas.network/blog/introducing-polystrat-an-autonomous-ai-prediction-agent-on-polymarket) (accessed 2026-03-07); [Polystrat Live Stats](https://x.com/autonolas/status/2026990760151884264) (accessed 2026-03-07); [Safe Smart Accounts for AI Agents](https://docs.safe.global/home/ai-overview) (accessed 2026-03-07); [IronClaw/OpenClaw/Olas TradingView Coverage](https://www.tradingview.com/news/cointelegraph:a63e272f1094b:0-ironclaw-rivals-openclaw-olas-launches-bots-for-polymarket-ai-eye/) (accessed 2026-03-07)
- **Fits our case because:** The architecture pattern directly answers the "how do you wire intelligence to execution safely" question. MIDGE's convergence alerter = the strategy layer. Safe smart account = the custody layer. Polymarket CLOB = the execution layer. The key design decision — hardcode critical wallet functions outside the LLM, have the LLM only influence direction/size/timing decisions — is directly applicable to MIDGE. The self-custodial pattern means no third party holds MIDGE's funds. Verifiable activity log = auditability without human decision-making.
- **Risks:** Polystrat is barely 2 weeks old at time of research (launched Feb 2026) — no track record beyond early results. Open source, which means competitors can clone the architecture exactly. The "runs locally via Pearl" requirement means infrastructure management burden. Performance data (300%+ returns) is unrealized PnL on very small positions — not validated at scale.

---

#### 7. OKX OnchainOS: 60+ Chain Cross-DEX Execution for AI Agents

- **What:** OKX's OnchainOS upgrade (March 2026) provides AI agents with wallet management, best-price DEX routing across 500+ DEXs on 60+ blockchains, autonomous payments via x402, and real-time onchain market data — all via API with sub-100ms response times.
- **Why it's interesting:** For MIDGE's cross-asset capability (the Research Brief asks about cross-market arbitrage), OnchainOS is the only infrastructure that provides a single API to execute across 60 chains and 500 DEXs simultaneously. This is the "instrument-agnostic" execution layer the Research Brief describes.
- **Evidence:** Launched March 3, 2026. Powers OKX Wallet across 60+ networks with 1.2B+ daily API calls, $300M daily trading volume, 99.9% uptime, sub-100ms response time. Supports x402 pay-per-use protocol. Zero gas costs on OKX's X Layer. This is the infrastructure OKX Wallet already runs at scale — the AI layer is an API abstraction on top of proven infrastructure.
- **Source:** [OKX OnchainOS CoinDesk](https://www.coindesk.com/tech/2026/03/03/okx-jumps-into-ai-agent-race-with-new-onchainos-toolkit) (accessed 2026-03-07); [OKX OnchainOS BeinCrypto](https://beincrypto.com/okx-onchainos-ai-toolkit/) (accessed 2026-03-07); [OKX AI Toolkit for Developers](https://www.okx.com/en-eu/learn/onchainos-our-ai-toolkit-for-developers) (accessed 2026-03-07)
- **Fits our case because:** MIDGE already has 2 crypto data sources (CoinGecko + CoinCap). Cross-chain arbitrage between crypto markets is a natural extension of MIDGE's convergence logic. If crypto signals converge bullishly AND OnchainOS can execute across 60 chains to find best price, MIDGE can operate across the entire crypto liquidity landscape — not just one exchange.
- **Risks:** Brand new (3 days old at time of research) — zero production track record for AI agent use. OKX is a centralized exchange with regulatory exposure in multiple jurisdictions. The "AI layer" may be thin marketing over existing infrastructure. Need to verify the actual developer API (documentation may not match capability at launch). Cross-chain DeFi introduces bridge risk, MEV risk, and smart contract risk at each hop.

---

### Emerging Approaches

---

#### 8. ElizaOS v2: Open-Source Multi-Chain Agent Framework

- **What:** TypeScript-based open-source framework for building autonomous AI agents with plugin architecture, cross-chain wallet abstraction (Solana, Ethereum, Base, BSC), DEX swap actions, and modular strategy integration.
- **Momentum:** Launched October 2024. Rebranded to ElizaOS in January 2025. As of January 2025, projects built on ElizaOS had combined market cap surpassing $20B. Auto.fun (no-code agent builder) launched April 2025. v2 launched October 2025 with Unified Abstraction Layer for cross-chain wallets and Hierarchical Task Networks (HTN) for complex multi-step planning. GitHub stars and ecosystem size make it the most-adopted AI agent framework in Web3.
- **Source:** [ElizaOS Official](https://elizaos.ai/) (accessed 2026-03-07); [ElizaOS ArXiv Paper](https://arxiv.org/html/2501.06781v1) (accessed 2026-03-07); [ElizaOS v2 Architecture](https://www.gate.com/learn/articles/eliza-os-v2-upgrade-how-ai-agents-evolve-from-simple-automation-to-full-autonomy/7898) (accessed 2026-03-07); [Ankr ElizaOS Blockchain Integration](https://www.ankr.com/blog/how-we-re-making-blockchain-aware-ai-agents-with-eliza-os/) (accessed 2026-03-07)
- **Fits our case because:** ElizaOS v2's Unified Abstraction Layer for wallets and HTN for complex planning are the architectural patterns MIDGE would need for multi-market autonomous operation. Plugin system means MIDGE's sensing hook could be implemented as an ElizaOS plugin feeding signals into an ElizaOS agent. However, ElizaOS is TypeScript — MIDGE is Python. The frameworks don't share code directly. Useful as a reference architecture, not a direct integration.
- **Maturity risk:** TypeScript-only, not Python. Active migration from v1 to v2 (token migration window closes February 2026). Ecosystem is heavily speculation/meme-driven (the $ai16z token origins). The trading plugin (Plugin-Goat) is experimental. Framework is primarily designed for DeFi/crypto, not for the multi-domain signal intelligence MIDGE provides.

---

#### 9. Cloudflare Agents SDK + Durable Objects

- **What:** Cloudflare's agent infrastructure — each agent runs as a Durable Object (stateful micro-server with SQL storage, WebSocket connections, scheduling), hibernates when idle, wakes on demand, with built-in MCP support and agentic commerce payment integration.
- **Momentum:** Part of Cloudflare's AI Week 2025 announcements. The "Moltworker" self-hosted personal AI agent went viral in February 2026, driving a 5% stock price increase for Cloudflare. Collaboration with "leading payments companies" announced for agentic commerce. The Cloudflare agent ecosystem is described as a "virtuous flywheel" in growth. Cloudflare is positioning agents.cloudflare.com as the infrastructure layer for all web-interacting AI agents globally.
- **Source:** [Cloudflare Agents Documentation](https://developers.cloudflare.com/agents/) (accessed 2026-03-07); [Cloudflare Agents Landing](https://agents.cloudflare.com/) (accessed 2026-03-07); [Cloudflare Agentic Commerce Press Release](https://www.cloudflare.com/press/press-releases/2025/cloudflare-collaborates-with-leading-payments-companies-to-secure-and-enable-agentic-commerce/) (accessed 2026-03-07); [CNBC Cloudflare AI Agent Wave](https://www.cnbc.com/2026/02/11/cloudflare-net-q4-earnings-2025.html) (accessed 2026-03-07)
- **Fits our case because:** MIDGE currently runs as a Python daemon on Wardenclyffe (Windows 11). Cloudflare Durable Objects would provide cloud-native, always-on operation without managing server infrastructure. The "Markdown for Agents" initiative (websites serving structured markdown to AI agents instead of HTML) means MIDGE's web-scraping sources could access more structured data. MCP integration is built-in — MIDGE could use Cloudflare as a deployment target.
- **Maturity risk:** TypeScript SDK only (Python support limited or absent). MIDGE is a Python system — Cloudflare Agents would require a full runtime migration or a thin TypeScript wrapper. Durable Objects have a maximum execution time per request (not unlimited compute). Cloudflare Workers have memory limits that MIDGE's 147-system architecture would stress. This is infrastructure-layer technology, not a trading tool.

---

#### 10. Safe Smart Accounts (ERC-4337) as the Universal Agent Wallet Standard

- **What:** Gnosis Safe smart contract wallets with ERC-4337 account abstraction, providing programmable spending limits, address whitelists, timelocks, multi-signature consensus, and gasless transactions — emerging as the gold standard for AI agent custody in 2026.
- **Momentum:** Safe already secures $100B+ in assets. More than half of all Safe transactions on Gnosis Chain in 2023-2024 were made by AI agents. The new AI agent economy "will run on Smart Accounts" (Safe's own framing). Polystrat, Coinbase AgentKit's Smart Wallet Provider, and multiple other agent frameworks all converge on Safe/ERC-4337 as the custody layer. Verified by Safe Watch Agent (AI fraud detection companion) and formal audit history.
- **Source:** [Safe AI Agent Overview](https://docs.safe.global/home/ai-overview) (accessed 2026-03-07); [The new AI agent economy will run on Smart Accounts](https://safe.mirror.xyz/V965PykKzlE1PCuWxBjsCJR12WscLcnMxuvR9E9bP-Y) (accessed 2026-03-07); [Safe Case for AI and Smart Accounts](https://safe.global/blog/the-safe-case-ai-smart-accounts-crypto) (accessed 2026-03-07)
- **Fits our case because:** The Research Brief requires MIDGE to be "undetectable as algorithmic" while being fully autonomous. Safe's architecture enforces this: spending limits prevent runaway position sizing, whitelists prevent MIDGE from sending funds to unexpected addresses, timelocks prevent rapid panic selling, and the multi-sig model means MIDGE's LLM layer can propose but not unilaterally execute — the safety layer is onchain and auditable. This is the exact guardrail architecture MIDGE needs.
- **Maturity risk:** ERC-4337 adoption is growing but not universal. Smart account gas overhead (slightly higher per-transaction than EOA wallets). The "hardcode critical functions outside LLM" pattern is still being established as best practice — no comprehensive implementation guides yet. Python Safe SDK is less mature than the TypeScript version.

---

### Gaps and Unknowns

1. **Stock/futures execution for MIDGE's primary edge domain.** MIDGE's strongest signals are in equity domains (insider trading, congressional trades, government contracts, institutional 13F filings). None of the crypto-native infrastructure (Coinbase AgentKit, OKX OnchainOS, Polymarket, Polystrat) reaches these markets. The bridge from MIDGE's cross-domain intelligence to equity futures execution requires IBKR or Alpaca — both of which have friction (TWS dependency for IBKR, no futures for Alpaca). This gap is the most consequential: MIDGE's proven edge (z=4.74, p<0.0001) was measured in equity markets, not prediction markets.

2. **Prediction market liquidity depth for non-obvious signals.** MIDGE's strongest combo (events+macro+price) generated 66.7% WR — but on n=3. At what AUM does the prediction market become too thin to absorb MIDGE's position sizing? Research found concentrated liquidity in political/crypto markets on Polymarket, not in the energy/government/insider domains where MIDGE has edge. This needs empirical testing, not assumption.

3. **Stealth and detection evasion.** The Research Brief specifically requires MIDGE to be "undetectable as algorithmic." Goldman Sachs and Deutsche Bank are actively deploying agentic AI for trade surveillance (as of early 2026). Fingerprint (the identity firm) launched an authorized AI agent detection product in February 2026. No production evidence was found of specific techniques used by Polymarket bots to avoid detection. The stealth-for-agents space is nascent — mostly focused on browser automation, not trading APIs. This is an open research question.

4. **Rate limit math for MIDGE's signal volume.** MIDGE generates convergence alerts at an unknown frequency. Polymarket limits to 60 orders/minute per key. Kalshi has undisclosed but enforced limits. If MIDGE's convergence engine generates 20+ alerts per hour at peak, the execution layer will need multi-key rotation or queuing logic. This has not been empirically measured.

5. **Profit-to-compute ratio.** The Research Brief wants MIDGE to fund her own compute. Anthropic API costs for MIDGE's LLM calls are a real variable. A 19.9% win rate with 3.34:1 payoff ratio is marginally profitable at current paper trading scale — but at what trade size does MIDGE generate enough net profit to cover inference costs plus API subscriptions? The math needs to be worked through before designing the self-funding loop.

6. **Python bindings for newer infrastructure.** Coinbase AgentKit Python SDK exists and is documented. OKX OnchainOS was just launched (March 3, 2026) — Python SDK maturity is unknown. ElizaOS is TypeScript-only. Safe Python SDK exists but is less documented than TypeScript. The ecosystem heavily favors TypeScript/JavaScript. MIDGE is Python — every integration will need library verification.

7. **Regulatory risk for autonomous execution.** The CFTC designated prediction markets as "swaps" in early 2026 — this is broadly positive for Kalshi but adds uncertainty for all algorithmic prediction market trading. Automated trading in commodity futures has additional CFTC registration requirements at certain thresholds. This is not a blocker for small-scale operation but is a ceiling on scale.

---

### Synthesis

**What is the strongest approach and why?**

The strongest path for MIDGE's first autonomous trading deployment is a two-stage sequence:

**Stage 1 (Fastest time to live trading with proven edge): Kalshi prediction market via REST API + Python SDK.**

Kalshi is the only fully regulated, US-legal, developer-friendly prediction market with clear API documentation, Python bindings, and markets that directly overlap MIDGE's existing domains (macro: FOMC/CPI/NFP; energy: crude oil/gas; events: earnings beats/misses). MIDGE's EIA energy source and economic_calendar source already generate signals in exactly the markets Kalshi offers. Authentication is simple (Bearer token + refresh logic). No crypto wallet management required at this stage. The CFTC regulatory coverage means no shutdown risk. This is the path of least friction to answer the core question: "does MIDGE's convergence intelligence actually generate consistent profit in a live market?"

**Stage 2 (Self-funding crypto loop): Coinbase AgentKit + Base L2 + x402.**

Once Stage 1 validates live profitability, Stage 2 implements the self-funding loop. MIDGE's crypto signals (CoinGecko + CoinCap) plus cross-asset confirmation (the cross_asset Gift system) drive trades on Base L2 via AgentKit. Profits accrue in USDC. x402 pays for API costs. Safe smart account (ERC-4337) enforces spending limits and address whitelists. This stage answers the "turn it on and walk away" requirement.

**What combination of approaches might work best?**

Three-layer execution stack:

| Layer | Purpose | Technology |
|-------|---------|------------|
| Prediction Markets | Proven MIDGE edge, fastest live validation | Kalshi REST API + kalshi-python |
| Crypto Self-Funding | 24/7 operation, self-funding loop, no broker dependency | Coinbase AgentKit + Base L2 + x402 |
| Equity Futures (Phase 3) | MIDGE's deepest edge domain (insider, congressional, 13F) | IBKR TWS API (paper first, then live) |

**What the orchestrator needs to know that doesn't fit neatly into categories:**

1. **The biggest infrastructure gap is the execution decision layer, not the broker connection.** Every broker API above can receive an order in 2-3 lines of Python. The hard problem is: what exactly triggers MIDGE to fire a live order? The convergence alerter currently writes to a JSONL file and logs to console. A thin `ExecutionGateway` class — listening to `CH_CONVERGENCE_ALERT` on the EventBus, applying a confidence threshold gate, sizing via Kelly, calling the broker API — is the minimal viable bridge. This is architectural work, not infrastructure work.

2. **The "stealth" requirement is less about hiding API calls and more about position sizing.** Polymarket and Kalshi's market surveillance looks for unusual patterns of volume concentration, not the source of the connection. An algorithm that trades small sizes across many markets, with human-like timing variance, looks exactly like a well-informed retail trader — which MIDGE's cross-domain convergence approach naturally produces. The edge IS the stealth: nobody looking at MIDGE's trade history would know whether it came from reading SEC filings carefully or from an automated system that processes 30 data sources simultaneously.

3. **Kalshi's markets on economic indicators are uniquely suited to MIDGE's FRED + EIA + economic_calendar signals.** If MIDGE's macro source detects an anomalous FRED indicator, and the economic_calendar source flags an upcoming CPI release, and EIA shows an unexpected crude build — Kalshi has a market for each of those outcomes. The translation from MIDGE's `bullish` direction + `macro+energy` domain combo to a Kalshi contract is nearly one-to-one. This is the first concrete case where MIDGE's existing signal architecture maps directly to a live tradeable market without any architectural changes.

4. **The 13,000 agents with wallets figure is from Coinbase's own reporting, not an independent source.** It is a marketing metric, not an audited number. Take infrastructure claims from newly launched products (OKX OnchainOS launched March 3, 2026 — three days before this research) with appropriate skepticism until there is independent validation.

5. **IBKR's TWS dependency is a real operational burden.** Running MIDGE as a daemon on Wardenclyffe alongside a headless IB Gateway process that requires periodic re-authentication is manageable but adds operational complexity. If MIDGE eventually runs in a containerized or cloud environment, IB Gateway's local socket model becomes a significant deployment obstacle. Plan for this before committing to IBKR as the long-term equity futures execution layer.

---

### Sources Referenced

- [Coinbase AgentKit GitHub](https://github.com/coinbase/agentkit) (accessed 2026-03-07)
- [Coinbase Agentic Wallets Launch](https://www.coinbase.com/developer-platform/discover/launches/agentic-wallets) (accessed 2026-03-07)
- [AgentKit Python Documentation](https://coinbase.github.io/agentkit/coinbase-agentkit/python/index.html) (accessed 2026-03-07)
- [AgentKit Q1 Update](https://www.coinbase.com/developer-platform/discover/launches/agentkit-q1-update) (accessed 2026-03-07)
- [Coinbase Launches Crypto Wallet Infrastructure for AI Agents - PYMNTS](https://www.pymnts.com/cryptocurrency/2026/coinbase-debuts-crypto-wallet-infrastructure-for-ai-agents/) (accessed 2026-03-07)
- [Polymarket Agents GitHub](https://github.com/Polymarket/agents) (accessed 2026-03-07)
- [py-clob-client GitHub](https://github.com/Polymarket/py-clob-client) (accessed 2026-03-07)
- [Polymarket API Architecture Medium](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf) (accessed 2026-03-07)
- [Definitive Guide to Polymarket Ecosystem](https://defiprime.com/definitive-guide-to-the-polymarket-ecosystem) (accessed 2026-03-07)
- [Building a Polymarket Copy Trading Bot - QuickNode](https://www.quicknode.com/guides/defi/polymarket-copy-trading-bot) (accessed 2026-03-07)
- [Automated Trading on Polymarket - QuantVPS](https://www.quantvps.com/blog/automated-trading-polymarket) (accessed 2026-03-07)
- [Kalshi API Documentation](https://docs.kalshi.com/welcome) (accessed 2026-03-07)
- [Kalshi Rate Limits](https://docs.kalshi.com/getting_started/rate_limits) (accessed 2026-03-07)
- [Kalshi API Developer Guide - Zuplo](https://zuplo.com/learning-center/kalshi-api) (accessed 2026-03-07)
- [Kalshi Python Client - PyPI](https://pypi.org/project/kalshi-python/1.1.0/) (accessed 2026-03-07)
- [How AI Exploits Prediction Market Glitches - CoinDesk](https://www.coindesk.com/markets/2026/02/21/how-ai-is-helping-retail-traders-exploit-prediction-market-glitches-to-make-easy-money) (accessed 2026-03-07)
- [Alpaca Trading API](https://alpaca.markets/) (accessed 2026-03-07)
- [Alpaca MCP Server GitHub](https://github.com/alpacahq/alpaca-mcp-server) (accessed 2026-03-07)
- [Alpaca API Paper Trading Docs](https://docs.alpaca.markets/docs/paper-trading) (accessed 2026-03-07)
- [BrokerChooser Best Algo Trading Brokers 2026](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading-in-the-united-states) (accessed 2026-03-07)
- [IBKR Trading API](https://www.interactivebrokers.com/en/trading/ib-api.php) (accessed 2026-03-07)
- [IBKR Python TWS API Guide](https://www.interactivebrokers.com/campus/ibkr-quant-news/interactive-brokers-python-api-native-a-step-by-step-guide/) (accessed 2026-03-07)
- [Olas Polystrat Launch](https://olas.network/blog/introducing-polystrat-an-autonomous-ai-prediction-agent-on-polymarket) (accessed 2026-03-07)
- [Polystrat Live Activity Twitter](https://x.com/autonolas/status/2026990760151884264) (accessed 2026-03-07)
- [AI Agents Take the Reins of Prediction Markets - Sandmark](https://www.sandmark.com/news/top-news/ai-agents-take-reins-prediction-markets) (accessed 2026-03-07)
- [Safe AI Agent Overview](https://docs.safe.global/home/ai-overview) (accessed 2026-03-07)
- [The new AI agent economy will run on Smart Accounts - Safe](https://safe.mirror.xyz/V965PykKzlE1PCuWxBjsCJR12WscLcnMxuvR9E9bP-Y) (accessed 2026-03-07)
- [OKX OnchainOS CoinDesk](https://www.coindesk.com/tech/2026/03/03/okx-jumps-into-ai-agent-race-with-new-onchainos-toolkit) (accessed 2026-03-07)
- [OKX AI Toolkit for Developers](https://www.okx.com/en-eu/learn/onchainos-our-ai-toolkit-for-developers) (accessed 2026-03-07)
- [ElizaOS Official Site](https://elizaos.ai/) (accessed 2026-03-07)
- [ElizaOS ArXiv Paper](https://arxiv.org/html/2501.06781v1) (accessed 2026-03-07)
- [ElizaOS v2 Architecture](https://www.gate.com/learn/articles/eliza-os-v2-upgrade-how-ai-agents-evolve-from-simple-automation-to-full-autonomy/7898) (accessed 2026-03-07)
- [Cloudflare Agents Documentation](https://developers.cloudflare.com/agents/) (accessed 2026-03-07)
- [Cloudflare Agentic Commerce Press Release](https://www.cloudflare.com/press/press-releases/2025/cloudflare-collaborates-with-leading-payments-companies-to-secure-and-enable-agentic-commerce/) (accessed 2026-03-07)
- [x402 Protocol Official Site](https://www.x402.org/) (accessed 2026-03-07)
- [x402 Protocol Architecture - Chainstack](https://chainstack.com/x402-protocol-for-ai-agents/) (accessed 2026-03-07)
- [AI Agent Economics Guide 2026](https://academy.exmon.pro/ai-agent-economics-how-autonomous-crypto-wallets-work-2026-guide) (accessed 2026-03-07)
- [Fingerprint AI Agent Detection Product](https://fintech.global/2026/02/04/fingerprint-launches-authorized-ai-agent-detection-product/) (accessed 2026-03-07)
- [Goldman Sachs Deutsche Bank Agentic Trade Surveillance](https://www.artificialintelligence-news.com/news/goldman-sachs-and-deutsche-bank-test-agentic-ai-for-trade-surveillance/) (accessed 2026-03-07)
- [Prediction Markets at Scale 2026 Outlook](https://insights4vc.substack.com/p/prediction-markets-at-scale-2026) (accessed 2026-03-07)
- [Reassessing the 2025 Prediction Market Landscape - Medium](https://medium.com/@NOX_Ventures/reassessing-the-2025-prediction-market-landscape-from-a-speculative-tool-to-a-new-financial-c5244c2598f0) (accessed 2026-03-07)
- [Coincub Crypto AI Agents Guide](https://coincub.com/blog/crypto-ai-agents/) (accessed 2026-03-07)
