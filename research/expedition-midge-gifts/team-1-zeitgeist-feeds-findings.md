# Team 1 Findings: Real-Time Zeitgeist Feeds
**Date:** 2026-03-05
**Research Angle:** Free/cheap APIs for real-time social media, news, and cultural signals on day/swing trading timescales
**Researcher Role:** Expedition field researcher — verify everything, cite everything, no stale claims

---

## Executive Summary

MIDGE already has StockTwits (social sentiment), Google Trends (retail attention), and ApeWisdom (Reddit stock mentions wired into convergence engine). The gaps are: **breaking financial news with per-ticker tagging**, **broader social signal beyond Reddit**, **economic event calendars**, and **macro data anchors**. Six concrete additions are ready to wire in. Two are battle-tested. Two are emerging with strong momentum. Two are novel and worth prototyping carefully.

---

## Category: Battle-Tested

### 1. Finnhub — Financial News + Sentiment + Economic Calendar
**What it is:** A REST API serving company news, market news, news sentiment scores, earnings calendar, economic calendar, and insider transactions.

**Free tier (verified):**
- 60 API calls/minute (most generous free financial API by this metric, per multiple sources)
- Company news, market news, and news sentiment endpoints included on free tier
- Economic calendar endpoint included
- Earnings calendar included
- US companies only for sentiment

**Signal types available:**
- Per-ticker company news (headlines + summaries, not full text)
- News sentiment score for a company (bullish/bearish aggregate from news)
- Economic calendar events (CPI, FOMC, employment reports with expected vs. actual)
- Earnings calendar (surprise signals when reported beats/misses)

**Python:** Official `finnhub-python` library on PyPI. pip install finnhub-python.

**Why it matters for MIDGE:** One API key delivers news sentiment + economic events simultaneously. When CPI comes in hot + tech news goes bearish + StockTwits is bearish = three-domain convergence signal. This is the single highest-value addition.

**Cache recommendation:** Company news: 15 minutes. Sentiment: 30 minutes. Economic calendar: 6 hours.

**Sources:**
- https://finnhub.io/docs/api/news-sentiment
- https://johal.in/finnhub-python-quotes-news-sentiment-social-forex-patterns-2025/
- https://dev.to/wassim/10-free-apis-to-supercharge-your-financial-apps-kf6
- https://robotwealth.com/finnhub-api/

---

### 2. Marketaux — Financial News with Per-Ticker Entity Tagging + Sentiment
**What it is:** A financial news API that returns structured articles with automatic entity recognition (ticker symbols tagged in article text) and per-entity sentiment scores (-1 to +1).

**Free tier (verified):**
- 100 requests/day
- 3 articles per request
- No credit card required to start
- Includes sentiment scores and entity tagging
- Access to 5,000+ news sources globally

**Signal types available:**
- Per-ticker news (filter by entity symbol)
- Sentiment score per entity per article (-1 to +1 scale)
- Trending entities (which tickers are getting the most news attention)
- Entity statistics time series

**Python:** Pure REST, requests library. No official SDK needed. Well-documented JSON.

**Why it matters for MIDGE:** Unlike Finnhub (which aggregates sentiment into a score), Marketaux returns individual articles with per-article per-ticker sentiment — MIDGE can detect *direction of shift* not just current level. "Three bearish articles on NVDA in the last hour" is more actionable than a single composite score.

**Tradeoff:** 100 requests/day is limiting. At 3 articles/request = 300 articles/day. Needs careful batching — fetch trending entities first, then targeted tickers on watchlist only.

**Sources:**
- https://www.marketaux.com/documentation
- https://freeapihub.com/apis/marketaux
- https://publicapis.io/marketaux-api

---

## Category: Novel (High Potential, Needs Validation)

### 3. Reddit via PRAW — Direct Subreddit Monitoring (Complements ApeWisdom)
**What it is:** PRAW (Python Reddit API Wrapper) provides direct OAuth access to Reddit's API. ApeWisdom already aggregates Reddit stock mentions, but PRAW gives MIDGE access to *what is being said* — full post text, comment threads, sentiment context — not just mention counts.

**Free tier (verified):**
- 60 requests/minute with OAuth (read-only)
- No cost for non-commercial use
- Requires Reddit app registration (client_id + client_secret, free)
- Pushshift is NOT recommended — it has delays up to 2 days and is unreliable

**What PRAW adds vs ApeWisdom:**
- ApeWisdom = mention *counts* with 24h lag
- PRAW = real-time post text from r/wallstreetbets, r/investing, r/stocks, r/options
- MIDGE can apply VADER or similar sentiment to the actual post body
- Detects narrative shifts: "why everyone is buying TSLA calls" vs just "TSLA mentioned 47 times"

**Key subreddits for trading signal:**
- r/wallstreetbets (retail momentum, meme stocks, options plays)
- r/stocks (fundamentals-adjacent retail)
- r/options (directional positioning intent)
- r/investing (longer timeframe but macro fear sentiment)

**Python:** pip install praw. Well-maintained, Python 3.9+ supported.

**Tradeoff:** Reddit's API is non-commercial only on free tier. MIDGE is for personal trading — this is within bounds. Any commercial deployment requires paid API access. Per Reddit's 2025-2026 policy, pre-approval may be required for applications with significant volume.

**Implementation note:** Fetch top posts from past 6-24 hours, not hot feed. "Hot" is gamed; time-sorted recent posts show actual breaking sentiment.

**Sources:**
- https://painonsocial.com/blog/reddit-api-rate-limits-guide
- https://pypi.org/project/praw/
- https://github.com/praw-dev/praw
- https://replydaddy.com/blog/reddit-api-pre-approval-2025-personal-projects-crackdown

---

### 4. Bluesky ATProto — Emerging Finance Community Signal
**What it is:** Bluesky's open API (no API key required for public data) provides access to trending topics, post search, and the growing finance/economics community that migrated from Twitter/X.

**Free tier (verified):**
- Completely free, no API key required for public endpoints
- Python SDK: `atproto` (pip install atproto)
- `getTrends` endpoint: `https://public.api.bsky.app/xrpc/app.bsky.unspecced.getTrends`
- `getTrendingTopics` endpoint also available
- Platform grew from 25M to 41M users in 2025

**Signal types available:**
- Platform-wide trending topics (detect when $TICKER breaks into mainstream Bluesky conversation)
- Post search by keyword/cashtag (search #NVDA, $SPY, etc.)
- Sentiment analysis on post text via VADER locally

**Why it matters:** Economists, investment strategists, and financial commentators actively moved to Bluesky. It is becoming the professional equivalent of old FinTwit. Early signals here may precede StockTwits by hours for institutional-adjacent retail sentiment.

**Tradeoff:** The `getTrends` and `getTrendingTopics` endpoints are marked `unspecced` — they are undocumented and subject to change without notice. Treat as fragile. The authenticated post search is stable. Finance community volume is smaller than Reddit but higher signal-to-noise.

**Cache recommendation:** Trending topics: 15 minutes. Post search: 30 minutes.

**Sources:**
- https://docs.bsky.app/docs/get-started
- https://github.com/MarshalX/atproto
- https://github.com/bluesky-social/atproto/discussions/3822

---

## Category: Emerging (Good Signal, Worth Adding)

### 5. Yahoo Finance RSS Feeds — Per-Ticker Breaking News (Zero Cost)
**What it is:** Yahoo Finance publishes per-ticker RSS feeds at `https://feeds.finance.yahoo.com/rss/2.0/headline?s={TICKER}&region=US&lang=en-US`. These are free, require no API key, and return the 20 most recent headlines for any ticker.

**Free tier:** Completely free. No authentication. Parse with `feedparser` (pip install feedparser).

**Signal types available:**
- Per-ticker headline feed (20 most recent)
- Article timestamps for velocity detection
- FinNews library wraps multiple financial RSS sources including Yahoo Finance, CNBC, MarketWatch, Reuters

**Why it matters for MIDGE:** Zero cost, zero rate limit concerns (soft limits only, be polite with delays), and headline velocity is a real signal. Five TSLA headlines in one hour vs one per day = breaking news detection. Combine with VADER for rapid sentiment classification.

**Pattern for MIDGE:** Fetch watchlist tickers every 15 minutes, hash headlines to detect new ones, run VADER on new headlines only.

**Note on FinNews library:** The `scaratozzolo/FinNews` package is **inactive/abandoned** (no new releases in 12+ months per Snyk). Use `feedparser` directly with explicit feed URLs instead — more reliable than a dead wrapper.

**Sources:**
- https://rss.feedspot.com/yahoofinance_rss_feeds/
- https://pypi.org/project/feedparser/
- https://snyk.io/advisor/python/finnews (status: inactive)
- https://github.com/scaratozzolo/FinNews

---

### 6. FRED API — Macro Economic Data Anchor
**What it is:** The Federal Reserve Bank of St. Louis provides free API access to 800,000+ economic data series including CPI, unemployment, Fed funds rate, consumer sentiment, and more. Free API key required (instant registration at fred.stlouisfed.org).

**Free tier (verified):**
- Free with API key registration
- `fredapi` Python library (pip install fredapi)
- Returns pandas Series/DataFrame
- Key series for trading: CPIAUCSL (CPI), UNRATE (unemployment), FEDFUNDS (Fed funds rate), UMCSENT (University of Michigan Consumer Sentiment), T10Y2Y (yield curve)

**Signal types available:**
- Macro context signals (not real-time, updated on release schedule)
- Yield curve inversion detection (T10Y2Y negative = recession signal)
- Consumer sentiment trend (UMCSENT)
- CPI trend for inflation regime classification

**Why it matters:** FRED doesn't give day-trading signals — it gives MIDGE context for *interpreting* other signals. "Reddit is bullish on tech stocks" means something different in a tightening vs easing Fed environment. This anchors other signals in macro reality.

**Important limitation:** FRED data is NOT real-time. CPI releases monthly, unemployment monthly. This is a context layer, not a fast signal.

**Cache recommendation:** 24 hours minimum (data doesn't change faster).

**Sources:**
- https://fred.stlouisfed.org/docs/api/fred/
- https://pypi.org/project/fredapi/
- https://github.com/mortada/fredapi

---

## Category: Gaps (Filtered Out)

### NewsAPI.org — Eliminated
**Reason:** Free tier has 24-hour article delay and prohibits production use. The developer plan is testing-only. Paid plans start at $449/month — far outside budget.
Source: https://newsapi.org/pricing

### Alpha Vantage News Sentiment — Deprioritized
**Reason:** Free tier is 25 requests/day. At 5 tickers minimum, that's 5 requests/day per ticker = depleted in hours. Marketaux at 100 req/day is a better free tier for financial news specifically.
Source: https://alphalog.ai/blog/alphavantage-api-complete-guide

### Unusual Whales API — Eliminate for Now
**Reason:** After May 2025 price increase, cheapest tier is $150/month (not $50/week trial). This is 3x the stated budget ceiling. Worth revisiting if MIDGE scales, as their options flow + dark pool data is genuinely high-signal.
Source: https://unusualwhales.substack.com/p/unusual-whales-api-prices-increasing

### Alpaca News API — Hold for Later
**Reason:** Free tier is limited to IEX exchange data; news API access unclear at free tier. Alpaca is primarily a brokerage/trading API — MIDGE can add this when/if an Alpaca trading account is opened, not as a standalone data purchase.
Source: https://docs.alpaca.markets/docs/about-market-data-api

### Pushshift/pmaw for Reddit — Eliminated
**Reason:** Delays up to 2 days. Useless for day/swing trading timescales. PRAW direct is the correct path.

### BLS API — Nice to Have, Not Now
**Reason:** BLS API (Bureau of Labor Statistics) provides CPI, employment data free with registration. However, FRED already exposes BLS data through a better interface. Redundant. Add BLS directly only if FRED rate limits become an issue.
Source: https://www.bls.gov/bls/api_features.htm

---

## Synthesis: Recommended Implementation Order

### Phase 1 (Wire in now — highest impact per effort)

| Source | Signal | Python Lib | Free Tier |
|--------|--------|------------|-----------|
| **Finnhub** | News sentiment + economic calendar | `finnhub-python` | 60 req/min |
| **Yahoo Finance RSS** | Per-ticker breaking news headlines | `feedparser` | Free, no limit |
| **Marketaux** | Ticker-tagged news with per-article sentiment | `requests` (REST) | 100 req/day |

**Why this order:** Finnhub is a single install that adds news sentiment AND economic calendar — two new domains for convergence. Yahoo Finance RSS is zero cost and zero friction. Marketaux adds article-level directional sentiment the others lack.

### Phase 2 (Prototype and validate)

| Source | Signal | Python Lib | Free Tier |
|--------|--------|------------|-----------|
| **PRAW (Reddit)** | Real-time WSB/investing subreddit text sentiment | `praw` | 60 req/min OAuth |
| **FRED API** | Macro context anchor | `fredapi` | Free with key |

**Why Phase 2:** PRAW requires building a VADER pipeline and subreddit selection logic. FRED requires deciding which series to monitor and how to convert macro state into a signal weight. Both are worth it, but they require more design decisions.

### Phase 3 (Emerging, monitor)

| Source | Signal | Python Lib | Free Tier |
|--------|--------|------------|-----------|
| **Bluesky ATProto** | Finance community trending topics | `atproto` | Free, no key |

**Why Phase 3:** The platform is growing fast and the finance community is real, but the trending endpoints are `unspecced` (may break). Build a thin wrapper that degrades gracefully when the endpoint changes.

---

## Integration Pattern for MIDGE

All six existing clients (stocktwits_client.py, trends_client.py, house_stock_watcher.py, etc.) follow a consistent pattern. New clients should match it:

```
class XxxClient:
    def __init__(self, provider=None):           # Accept provider for test injection
    def _rate_limit(self) -> None:               # Enforce delay between calls
    def _request(self, url, ...) -> Optional[dict]:  # Route through provider or session
    def get_YYY(self, tickers) -> List[XxxSignal]:  # Public method returns dataclass list

@dataclass
class XxxSignal:
    ticker: str
    signal_source: str = "xxx_yyy"
    decay_rate: float                           # How fast this signal ages (0.0-1.0)
    confidence: float                           # Prior probability this is right
    detected_at: str = field(...)              # UTC ISO timestamp
    def to_plain_language(self) -> str         # Human-readable summary
```

**Convergence wiring:** `convergence_alerter.py` accepts signals via `record_signal(signal_id, strength, domain, direction)`. New clients should map to domains: `"social"`, `"news"`, `"macro"`. A Finnhub news sentiment client maps to `"news"` domain. A PRAW Reddit client maps to `"social"` domain (different from ApeWisdom's `"social"` — use distinct `signal_id` strings like `"reddit_wsb_sentiment"` vs `"ape_wisdom_mentions"`).

---

## Risk Flags

1. **Google Trends (existing):** pytrends uses informal scraping — not an official API. Rate-limited aggressively. If it breaks, feedparser + Yahoo Finance RSS covers the "what's getting attention" angle.

2. **Yahoo Finance RSS:** Terms of service say "personal use only." MIDGE is personal trading — compliant. If deployed commercially, this needs replacing.

3. **Bluesky trending endpoints marked `unspecced`:** Any code using `getTrends` must be wrapped in try/except with graceful degradation — not a dependency, a bonus.

4. **Reddit non-commercial constraint:** PRAW on free tier is non-commercial only. Acceptable for personal MIDGE deployment. If MIDGE becomes a product, Reddit commercial API access will be required.

5. **Marketaux 100 req/day ceiling:** At a 15-ticker watchlist with one news fetch per ticker = 15 requests. At 6 fetches/day = 90 requests/day. Tight. Fetch trending entities first, then targeted tickers from watchlist sorted by recent volatility. Don't fetch dormant tickers.

---

## Sources Index

- Reddit API limits: https://painonsocial.com/blog/reddit-api-rate-limits-guide
- Reddit API pricing: https://data365.co/blog/reddit-api-pricing
- PRAW library: https://pypi.org/project/praw/
- Finnhub API docs: https://finnhub.io/docs/api/news-sentiment
- Finnhub rate limits: https://finnhub.io/docs/api/rate-limit
- Finnhub Python review: https://johal.in/finnhub-python-quotes-news-sentiment-social-forex-patterns-2025/
- Marketaux documentation: https://www.marketaux.com/documentation
- Marketaux free API info: https://freeapihub.com/apis/marketaux
- NewsAPI.org pricing: https://newsapi.org/pricing
- Alpha Vantage 2026 guide: https://alphalog.ai/blog/alphavantage-api-complete-guide
- Unusual Whales pricing increase: https://unusualwhales.substack.com/p/unusual-whales-api-prices-increasing
- Alpaca market data docs: https://docs.alpaca.markets/docs/about-market-data-api
- Bluesky get started: https://docs.bsky.app/docs/get-started
- Bluesky ATProto Python SDK: https://github.com/MarshalX/atproto
- Bluesky trending discussion: https://github.com/bluesky-social/atproto/discussions/3822
- Yahoo Finance RSS feeds: https://rss.feedspot.com/yahoofinance_rss_feeds/
- feedparser PyPI: https://pypi.org/project/feedparser/
- FinNews status (inactive): https://snyk.io/advisor/python/finnews
- FRED API docs: https://fred.stlouisfed.org/docs/api/fred/
- fredapi PyPI: https://pypi.org/project/fredapi/
- Newsdata.io free tier: https://newsdata.io/blog/pricing-plan-in-newsdata-io/
- Financial APIs comparison 2026: https://currencyfreaks.com/blog/Best-Financial-API-Picks-For-Real-Time-Data-in-2026.html
- Best stock news APIs 2026: https://newsdata.io/blog/best-stock-news-api/
