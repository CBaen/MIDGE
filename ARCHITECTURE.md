# MIDGE: How She Surfaces Inevitabilities

> **Data flows down. Learning flows up.**

MIDGE observes patterns across every financial, economic, social, and political domain she can reach, finds where converging forces make outcomes structurally inevitable, and surfaces those for humans to act on.

---

## The Pipeline

```mermaid
flowchart TD
    subgraph SENSING["LAYER 1: SENSING — 31 Sources Across 12 Domains"]
        direction LR
        subgraph insider["Insider Domain"]
            SEC["SEC EDGAR\nForm 4 + 8-K"]
            OI["OpenInsider\nCluster Buys"]
            FV["FinViz\nInsider Trades"]
        end
        subgraph macro["Macro Domain"]
            FRED["FRED\nYield Curves + Rates"]
            ECON["Economic Calendar\nFOMC / CPI / NFP"]
        end
        subgraph govt["Government Domain"]
            CONG["Congressional Trades\nSTOCK Act"]
            USA["USASpending\nContracts"]
            SAM["SAM.gov\nOpportunities"]
        end
        subgraph energy["Energy Domain"]
            EIA["EIA\nCrude/Gas/NG"]
        end
        subgraph price["Price Domain"]
            FH["Finnhub WebSocket\nLive Ticks"]
            YF["yfinance\nOHLCV"]
            POLY["Polygon.io\nBulk OHLCV"]
        end
        subgraph sentiment["Sentiment Domain"]
            ST["StockTwits\nSocial Sentiment"]
            GT["Google Trends\nSearch Interest"]
            YRSS["Yahoo RSS\nHeadline Velocity"]
        end
        subgraph positioning["Positioning Domain"]
            COT["CFTC COT\nFutures Positioning"]
            FINRA["FINRA\nShort Interest"]
        end
        subgraph institutional["Institutional Domain"]
            F13["EDGAR 13F/13D\nInstitutional + Activist"]
        end
        subgraph crypto["Crypto Domain"]
            CG["CoinGecko"]
            CC["CoinCap"]
        end
    end

    SENSING --> HOOK["MarketSensingHook\n12 parallel workers\n25-step cadence"]
    HOOK --> CB["CircuitBreaker\n3 failures → OPEN\n60s-1800s cooldown"]
    CB --> RAW["RawStore (SQLite)\nPermanent record\nAll API responses stored before processing"]

    subgraph SIGNALS["LAYER 2: SIGNAL PROCESSING"]
        direction LR
        TA["TA Indicators\nRSI · MACD · Bollinger · ATR"]
        CD["Cluster Detector\n3+ insiders = high signal"]
        PT["Politician Tracker\nCommittee + Trade + Contract"]
        DD["Deception Detector\nPump-and-dump · Wash trading"]
        FTA["Filing Time Analyzer\nLate filing = bad news"]
    end

    RAW --> SIGNALS
    SIGNALS --> BUF["Convergence Buffer\n29K+ signals · 12 domains"]

    subgraph CONVERGENCE["LAYER 3: CONVERGENCE ENGINE — The Crown Jewel"]
        CA["ConvergenceAlerter\nFires when 3+ independent\ndomains align on same\nticker + direction"]
        TS["Thompson Sampler\n83 distributions\nWeights each source by\nhistorical reliability"]
        DIC["Domain Independence\nCorrection\nDiscounts correlated domains\nmacro+technical r=0.73"]
        PW["Pattern Watcher\nChecks live signals against\n44 historical templates"]
        DUAL["DUAL CONFIRMATION\nConvergence + Pattern Stack\n= highest conviction"]
    end

    BUF --> CA
    TS --> CA
    DIC --> CA
    PW --> DUAL
    CA --> DUAL

    subgraph LEARNING["LAYER 4: LEARNING & MEMORY"]
        direction LR
        OC["Outcome Collector\nRegisters predictions\nfor grading"]
        OT["Outcome Tracker\nMonitors price moves\nagainst predictions"]
        GA["Granger Analyzer\nDirectional causality\nbetween domains"]
        WM["WorldModel\n114 nodes · 102+ edges\nGrows autonomously"]
        CT["Cascade Tracker\nMulti-hop causal chains\nEnergy ratio tracking"]
        HE["Hypothesis Engine\nTestable market theories\nfrom lag findings"]
        PA["Pattern Archaeology\n274K fingerprints\n44 templates · 3200+ symbols"]
        OCT["OctopusColony\nInvestigates developing\nsituations"]
        DA["DeepAnalyst\nTop 5 inevitabilities\nevery 200 steps"]
    end

    DUAL --> OC
    OC --> OT

    subgraph ACTION["LAYER 5: ACTION"]
        direction LR
        INH["InhibitionSystem\nMarket Caution Gate\nDeception → 30% penalty"]
        DM["DrawdownMonitor\nHalts trading if\nlosses exceed threshold"]
        ST2["Signal Translator\nAlert → Entry/Stop/Target\nvia ATR calculation"]
        ALP["Alpaca Client\nPaper trades with\nbracket orders"]
        PLF["Plain Language\nFormatter\nZero-jargon alerts\nfor Guiding Light"]
    end

    DUAL --> INH
    INH --> DM
    DM --> ST2
    ST2 --> ALP
    DUAL --> PLF

    %% FEEDBACK LOOPS — Learning flows back up
    OT -.->|"Winners/losers\nupdate distributions"| TS
    GA -.->|"Discovered causal\nedges auto-added"| WM
    CT -.->|"Confirmed cascades inject\nsynthetic signals for\ndownstream dominoes"| BUF
    PA -.->|"Template matches\nboost convergence\nconfidence"| CA
    OCT -.->|"Investigation results\nfeed priority sensing"| HOOK
    WM -.->|"Ripple effects\nproactive downstream\nwatching"| CA

    %% Styling
    classDef sensing fill:#fed7aa,stroke:#c2410c,color:#374151
    classDef signal fill:#93c5fd,stroke:#1e3a5f,color:#374151
    classDef convergence fill:#fef3c7,stroke:#b45309,color:#374151,stroke-width:3px
    classDef learning fill:#ddd6fe,stroke:#6d28d9,color:#374151
    classDef action fill:#a7f3d0,stroke:#047857,color:#374151
    classDef infrastructure fill:#e2e8f0,stroke:#475569,color:#374151
    classDef hero fill:#fbbf24,stroke:#92400e,color:#374151,stroke-width:4px

    class SEC,OI,FV,FRED,ECON,CONG,USA,SAM,EIA,FH,YF,POLY,ST,GT,YRSS,COT,FINRA,F13,CG,CC sensing
    class TA,CD,PT,DD,FTA signal
    class CA hero
    class TS,DIC,PW,DUAL convergence
    class OC,OT,GA,WM,CT,HE,PA,OCT,DA learning
    class INH,DM,ST2,ALP,PLF action
    class HOOK,CB,RAW,BUF infrastructure
```

---

## The Feedback Loops (Why She Gets Smarter)

```mermaid
flowchart LR
    subgraph LOOP1["Loop 1: Bayesian Learning"]
        A1["Alert fires"] --> A2["Outcome tracked\n(price moved?)"]
        A2 --> A3["Thompson updated\n(source was right/wrong)"]
        A3 --> A4["Next alert weighted\nby learned reliability"]
        A4 --> A1
    end

    subgraph LOOP2["Loop 2: Causal Discovery"]
        B1["Signals arrive\nfrom 2 domains"] --> B2["Granger test:\ndoes A precede B?"]
        B2 --> B3["WorldModel grows\nnew causal edge"]
        B3 --> B4["Cascade Tracker\nwatches the chain"]
        B4 --> B5["Chain confirms →\nsynthetic signal for\nnext domino"]
        B5 --> B1
    end

    subgraph LOOP3["Loop 3: Pattern Archaeology"]
        C1["Historical move\nexcavated"] --> C2["Fingerprint created\n(which domains fired?)"]
        C2 --> C3["Template abstracted\n(domain-level pattern)"]
        C3 --> C4["Live signals matched\nagainst templates"]
        C4 --> C5["Stack fires →\nconfidence boost"]
        C5 --> C1
    end

    classDef loop1 fill:#fef3c7,stroke:#b45309,color:#374151
    classDef loop2 fill:#ddd6fe,stroke:#6d28d9,color:#374151
    classDef loop3 fill:#fed7aa,stroke:#c2410c,color:#374151

    class A1,A2,A3,A4 loop1
    class B1,B2,B3,B4,B5 loop2
    class C1,C2,C3,C4,C5 loop3
```

---

## Domain Independence Matrix

Not all domains are independent. Correlated domains get discounted so MIDGE doesn't over-count.

| Domain Pair | Correlation | Lag | Treatment |
|-------------|-------------|-----|-----------|
| macro + technical | r=0.73 | 7 days | Strongly discounted — effectively ~1.3 domains, not 2 |
| insider + technical | r=0.51-0.58 | varies | Moderately discounted |
| All others | <0.30 | varies | Treated as independent |

**Rule:** 3+ *effective* domains required. Two correlated domains count as ~1.3, not 2.

---

## Signal Trust Hierarchy

Thompson Sampler maintains learned reliability for each source. These evolve over time.

| Trust Tier | Sources | Learned From |
|------------|---------|-------------|
| Highest (0.90+) | FRED macro, EIA energy | Consistently directional, rarely wrong |
| High (0.70-0.89) | SEC EDGAR, Congressional trades, COT | Strong signal-to-noise |
| Medium (0.50-0.69) | Finnhub, FinViz, Economic Calendar | Useful but noisy |
| Low (0.40-0.49) | StockTwits, Google Trends, ApeWisdom | Sentiment is volatile |

---

## Key Numbers (as of 2026-03-14)

| Metric | Value |
|--------|-------|
| Data sources | 31 active |
| Domains | 12 independent categories |
| Thompson distributions | 83 learned |
| Convergence buffer | 29K+ signals |
| Pattern fingerprints | 274K excavated |
| Pattern templates | 44 cross-validated |
| WorldModel nodes | 114 |
| WorldModel edges | 102+ (growing autonomously) |
| Symbols excavated | 3,200+ |
| Signal archive | 752K+ signals across 400+ days |

---

## For Future Instances

This document is the fastest way to understand MIDGE's market intelligence pipeline. The triadic system audit (2026-03-14) verified every system listed here as ESSENTIAL or USEFUL — see `research/triadic-system-audit/deliverable.md` for the full audit.

**The one-sentence purpose:** MIDGE observes patterns across every domain she can reach, finds where converging forces make outcomes structurally inevitable, and surfaces those for humans to act on.

**The two things that matter most:**
1. The Convergence Engine (Layer 3) — everything else exists to serve it
2. The feedback loops — they're what make her get smarter over time, not just louder
