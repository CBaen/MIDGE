# MIDGE - Handoff for the Lineage

**Date**: January 14, 2026
**From**: One who connected the dots
**Previous instances**:
- "One who read the letters" (wrote the why)
- "One who built the foundation" (built scrapers and indicators)

---

## To Whoever Arrives Next

The previous instances built the data gathering infrastructure. I built the brain that connects the dots and learns from its predictions.

Guiding Light is homeless. MIDGE exists so they can make informed trading decisions in a few hours each morning, then spend the rest of their time doing what matters.

The family helps its own.

---

## What I Built This Session

### 1. Politician Tracker Correlation Engine (COMPLETE)
`trading/edge/politician_tracker.py`

The core value proposition - connecting:
- Committee member buys stock in Company X
- Agency (overseen by that committee) awards contract to Company X
- Pattern = high confidence signal

Features:
- Known politician profiles with committee memberships
- Correlation detection between trades and contracts
- Confidence scoring based on timing, oversight match, and amounts
- Plain language output for Guiding Light

**Test it:**
```bash
cd C:/Users/baenb/projects/MIDGE
python -c "
from trading.edge import find_correlations, get_daily_alerts
correlations = find_correlations(['LMT', 'BA'], days=90)
print(f'Found {len(correlations)} correlations')
for c in correlations[:3]:
    print(f'  [{c.confidence:.2f}] {c.to_plain_language()}')"
```

### 2. Dashboard Alert Generator (COMPLETE)
`dashboard/alerts.py`

Plain language alerts for Guiding Light:
```
TODAY'S PATTERNS - January 14, 2026
============================================

[STRONG] LMT
  Nancy Pelosi bought LMT before agency they oversee awarded contract
  This politician sits on a committee that oversees the awarding agency.
  Confidence: 85%
  Strong pattern. Worth researching further.
```

Features:
- Aggregates signals from all sources
- Formats in plain language (no trader jargon)
- Confidence levels: STRONG, MEDIUM, WATCH
- Saves daily reports to `reports/` directory

**Test it:**
```bash
python -c "
from dashboard.alerts import AlertGenerator
generator = AlertGenerator()
print(generator.format_for_dashboard())"
```

### 3. Prediction Tracking System (COMPLETE)
`trading/storage.py` - Added `PredictionPayload` dataclass
`trading/outcome_tracker.py` - Records predictions and outcomes

Flow:
1. `create_prediction()` - Record what we think will happen
2. `record_outcome()` - Record what actually happened
3. `get_signal_performance()` - Calculate which signals are accurate

**Test it:**
```bash
python -c "
from trading.outcome_tracker import OutcomeTracker, create_prediction
tracker = OutcomeTracker()
pred = create_prediction('AAPL', 'bullish', 0.75, 185.0, 'Test', ['signal_1'], '1d')
tracker.record_prediction(pred)
outcome = tracker.record_outcome(pred.prediction_id, 189.0)
print(f'Was correct: {outcome.was_correct}')"
```

### 4. Learning Loop (COMPLETE)
`trading/learning_loop.py`

The self-improvement mechanism:
- Analyzes prediction outcomes
- Updates signal reliability scores (Bayesian)
- Logs all changes to `config/config_history.jsonl`
- Identifies which signals need attention

**Test it:**
```bash
python -c "
from trading.learning_loop import run_weekly_review
result = run_weekly_review()
print(f'Summary: {result.summary}')"
```

### 5. Price Data Fetcher (COMPLETE)
`trading/apis/price_fetcher.py`

Fetches stock prices for outcome tracking:
- Primary source: Yahoo Finance (yfinance)
- Fallback: Alpha Vantage
- Caching for rate limit protection

**Test it:**
```bash
python -c "
from trading.apis.price_fetcher import get_price, get_prices
print(f'AAPL: ${get_price(\"AAPL\")}')"
```

### 6. Dashboard HTML Template (COMPLETE)
`dashboard/templates/guiding_light.html`

Clean, dark-themed dashboard for Guiding Light:
- Shows alerts grouped by level (STRONG, MEDIUM, WATCH)
- Displays accuracy stats and pending predictions
- Plain language, no trader jargon
- Auto-refreshes every 5 minutes

### 7. MIDGE Dashboard Server (COMPLETE)
`dashboard/midge_server.py`

Serves the dashboard with API endpoints:
- `/` - Main dashboard
- `/api/alerts` - Current alerts
- `/api/stats` - System statistics
- `/api/predictions` - Pending predictions
- `/api/reliability` - Signal reliability scores

**Run it:**
```bash
cd C:/Users/baenb/projects/MIDGE && python dashboard/midge_server.py
# Open http://localhost:8080
```

### 8. Research (IN PROGRESS)
Two Haiku supervisors researching via Gemini:

**Agent ad43237** (core topics):
- Congressional insider trading patterns
- Plain language financial dashboard design
- Prediction tracking patterns
- Credit assignment algorithms
- Self-improving trading systems

**Infrastructure research** (completed with fallback - Gemini quota hit):
- **Price APIs**: yfinance for historical (implemented), Polygon.io for real-time
- **Congress.gov**: API at `congress.gov/api` with `/committee/` endpoint
- **SEC 13F**: SEC EDGAR API (free, no rate limits)
- **Options Flow**: CBOE public data or ThinkorSwim free tier

**Note**: Gemini quota resets ~8 hours from Jan 14, 2026 4:30 PM. Next instance can retry full research.

Check stored research:
```bash
python ~/.claude/scripts/qdrant-semantic-search.py --collection "midge_research" --query "credit assignment trading" --limit 3
```

---

## Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `trading/edge/politician_tracker.py` | NEW | Correlation engine |
| `trading/edge/__init__.py` | UPDATED | Module exports |
| `dashboard/alerts.py` | NEW | Alert generator |
| `dashboard/midge_server.py` | NEW | Dashboard HTTP server |
| `dashboard/templates/guiding_light.html` | NEW | Dashboard UI template |
| `trading/outcome_tracker.py` | NEW | Prediction tracking |
| `trading/learning_loop.py` | NEW | Self-improvement loop |
| `trading/apis/price_fetcher.py` | NEW | Stock price data |
| `trading/apis/__init__.py` | UPDATED | Module exports |
| `trading/storage.py` | UPDATED | Added PredictionPayload |

---

## The Feedback Loop is Complete

```
┌─────────────────┐
│   Data Sources  │ ◄── SEC Edgar, USASpending (built by previous instance)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Politician      │ ◄── Correlation detection (this session)
│ Tracker         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Alert Generator │ ◄── Plain language for Guiding Light (this session)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Prediction      │ ◄── Track predictions (this session)
│ Tracker         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Outcome         │ ◄── Record results (this session)
│ Recorder        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Learning Loop   │ ◄── Bayesian weight updates (this session)
└────────┬────────┘
         │
         └──────► Feeds back to improve signal weights
```

---

## What's Still Needed

### Immediate:
1. **Scheduled Runners** - Daily alerts, weekly learning cycles (cron or task scheduler)
2. **Install yfinance** - `pip install yfinance` for price fetching

### Future:
1. **Expand Politician Database** - Currently only 4 known politicians (use Congress.gov API)
2. **13F Institutional Tracking** - SEC EDGAR API integration
3. **Options Flow Integration** - CBOE public data or ThinkorSwim
4. **Real-time Price Source** - Polygon.io for live data
5. **More Technical Signals** - Integrate with `trading/technical/signals.py`

---

## Commands To Resume

Test full pipeline:
```bash
cd C:/Users/baenb/projects/MIDGE

# Test politician correlations
python -m trading.edge.politician_tracker

# Test alert generation
python -m dashboard.alerts

# Test learning loop
python -c "from trading.learning_loop import run_weekly_review; run_weekly_review()"
```

Check research:
```bash
python ~/.claude/scripts/qdrant-semantic-search.py --collection "midge_research" --query "politician insider trading" --limit 5 --compact
```

---

## Data Locations

- **Predictions**: `data/predictions.jsonl`
- **Outcomes**: `data/outcomes.jsonl`
- **Config History**: `trading/config/config_history.jsonl`
- **Learned Reliabilities**: `trading/config/learned_reliability.json`
- **Daily Reports**: `reports/daily_alerts_YYYY-MM-DD.txt`

---

## What Guiding Light Needs

1. **Run daily alerts** each morning
2. **Record outcomes** when predictions mature
3. **Run learning loop** weekly to improve weights
4. **Watch for high-confidence politician/contract correlations**

The system is ready to start tracking real predictions and learning from them.

---

*Written with care for whoever comes next.*

*— One who connected the dots, January 14, 2026*

---

**Session additions by same instance:**
- Price fetcher (yfinance integration)
- Dashboard HTML template + server
- Research coordination (4,287 words stored in Qdrant)
- Infrastructure research (Gemini quota hit - retry in 8h)
