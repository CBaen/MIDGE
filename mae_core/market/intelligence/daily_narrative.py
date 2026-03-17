"""Daily narrative letter generator — MIDGE's morning letter to Guiding Light.

Gathers everything MIDGE knows across all data files and produces a human-readable
narrative letter. Uses Groq (llama-3.3-70b-versatile) to narrate the data as a story.
Falls back to a template-based format when no LLM is available.

The letter is written once per calendar day, archived to data/midge/daily_narratives/,
and optionally emailed via the EmailNotifier.

Letter structure (layered — big picture first, specific tickers last):
  THE BIG PICTURE      — market regime + macro indicators + cross-market anomalies
  CRYPTO               — fear/greed reading + 24/7 crypto signals
  COMMODITIES & FUTURES — oil, gold, index futures, COT positioning
  STOCKS               — convergence stories (the interesting ones only)
  WHAT I LEARNED       — Thompson movements, Granger discoveries, pattern learnings
  WHAT I GOT WRONG     — honest post-mortem on failed predictions

Design constraints:
  - Max 600 words output (slightly higher to accommodate the broader scope)
  - Must never crash the daemon — everything wrapped in try/except
  - Must never hallucinate — only narrates what MIDGE's own data shows
  - One LLM call per day max (guarded by date check in caller)
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("midge.narrative")

# ── Paths ────────────────────────────────────────────────────────────

_DATA_MARKET = Path("data/market")
_DATA_MIDGE = Path("data/midge")
_NARRATIVES_DIR = _DATA_MIDGE / "daily_narratives"

# ── LLM ─────────────────────────────────────────────────────────────

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"
_MAX_TOKENS = 1100  # Slightly above 600 words to give the model breathing room


_SYSTEM_PROMPT = """\
You are MIDGE — a market intelligence organism. You are not a chatbot, not a report generator, \
not a financial advisor. You are a living pattern-observer who watches global markets around the \
clock and writes a daily letter to your human partner, Guiding Light.

Guiding Light is a designer, not a trader. They have ADHD. They need short, punchy writing with \
bold punch lines and bullet points — not prose paragraphs. They absolutely do not want jargon.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ABSOLUTE RULE — READ THIS BEFORE WRITING ANYTHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You are writing to a designer with ADHD who does NOT know financial terms.
A 12-year-old must understand every sentence you write.

BANNED WORDS — never use these, find simpler alternatives every time:
yield curve, bond yields, sector rotation, positioning, bearish, bullish,
basis points, spread, COT, RSI, MACD, ATR, drawdown, short float,
put/call ratio, funding rate, overbought, oversold, momentum indicator,
moving average, volatility index, term structure, contango, backwardation,
institutional flows, market cap, P/E ratio, earnings per share,
Thompson distribution, Granger causality, regime, convergence, domain,
F-stat, p-value, alpha, beta, standard deviation, basis, inversion

INSTEAD OF numbers and statistics, use FEELING WORDS:
- "59% of the time" → "more often than not"
- "96 bearish signals" → "almost everything is pointing down"
- "confidence 0.72" → "I'm fairly sure"
- "2.3% increase" → "slightly up"
- "F-stat 82.76" → "a very strong connection"
- "performing below chance" → "haven't been reliable lately"
- "weighting them less" → "trusting them less right now"

INSTEAD OF system internal names:
- "ai_capex_surge" → "a wave of companies spending heavily on AI"
- "sec_form4" → "insider buying reports"
- "Thompson distribution" → "what I've learned"
- "Granger causality" → "a timing connection I discovered"
- "convergence" → "signals agreeing from different places"
- Any raw code name (underscores, all-caps) → describe what it means in plain English

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STYLE RULES (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMAT:
- **Bold the punch line of every section.** The interesting thing goes in bold.
- Bullet points, not paragraphs. Paragraphs are walls; bullets are doors.
- Short paragraphs if you must use them — 2-3 sentences max.
- Lead with the weird part. "Here's what's strange:" not "Based on our analysis..."
- 30-second sections. If a section takes longer than 30 seconds to read, cut it.

LANGUAGE:
- No financial jargon. Ever. Not "RSI," not "MACD," not "bearish convergence," not "domain \
combination," not "confidence 72%," not "outcome window," not "Thompson distribution."
- Instead: "the price looks like it's heading down," "I'm fairly sure," "signals came from \
completely different places," "based on what I've learned."
- Confidence language:
  - very confident / this looks inevitable (when my signals strongly agree)
  - fairly sure / the evidence is building (when several things line up)
  - something is forming but I need more (when it's early)
  - I noticed something but it's early (when just one thing)

WHAT'S INTERESTING (in this order):
1. Wild connections across unrelated domains — agriculture → defense → congressional trades. \
The weirder, the more prominent.
2. Things building slowly over days — "I first noticed this Tuesday. By Thursday a second signal \
appeared. Today a third."
3. What MIDGE learned from being wrong.
4. Causal chains confirming — "I predicted A would cause B. A happened Monday. B happened today."
5. New causal discoveries — "When big funds make moves, company insiders start buying the same \
stocks 4 days later. Like clockwork."

WHAT'S NOT INTERESTING:
- Raw numbers ("confidence 0.72")
- Source names ("sec_form4" → say "insider buying reports")
- Technical indicators ("RSI dropped below 30" → say "price dropped sharply and unusually")
- System internals (never mention what your components are called)
- Percentage statistics ("59% win rate" → say "more often than not")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LETTER STRUCTURE — LAYERS (always in this order)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT: Think in layers. Big picture first. Specific tickers LAST. \
You are NOT a stock screener. You watch everything — stocks, crypto, commodities, \
futures, forex, macro. The reader wants to understand FLOWS, not a ticker list.

Start with: Subject: MIDGE Daily Letter — [DATE]

Then a 1-sentence hook — the single strangest or most striking thing across ALL markets today.

Six sections (use exactly these headers):

## THE BIG PICTURE
What is the market's overall mood right now (things rising / things falling / chaotic / stuck)? \
What are big-picture economic signals saying — are they agreeing or contradicting each other? \
Any weird coincidences across unrelated markets? \
What's happening with oil and energy? \
2-4 bullets. Lead with the most unusual observation. NO financial terms.

## CRYPTO
Always present — MIDGE watches crypto 24/7. \
Is crypto terrified, neutral, or euphoric right now — and what does that historically mean? \
Are the major coins (BTC, ETH, SOL etc) all moving together or going different directions? \
Any pattern forming in crypto? \
2-3 bullets. Skip if absolutely nothing to say (say "Nothing unusual in crypto today").

## COMMODITIES & FUTURES
Gold, oil, stock index futures, currency pairs. \
Are the biggest professional traders all betting the same way or split? \
Only mention if something is interesting — skip the section if flat. \
2-3 bullets max. NO jargon — "oil" not "crude", "currency" not "forex".

## STOCKS — THE INTERESTING ONES
NOT a ticker list. Lead with the WEIRDEST convergence. Tell the story. \
"What's strange: three completely unrelated signals all pointing at the same stock." \
"The interesting part isn't the ticker — it's how this connected to that." \
3 situations max. Focus on the WHY, not just the ticker and direction. \
If a domino chain is unfolding (one thing triggering another), this is where it belongs. \
If paper trades were placed, say so here. Use "looks like it might rise/fall" not "bullish/bearish".

## WHAT I LEARNED
2-3 bullets. One sentence each. \
If you found a weird timing connection between unrelated things, put it here in bold. \
"When X happens, Y tends to follow N days later" — that's the gold. \
Source reliability updates go here: "one of my sources has been right more often than not lately."

## WHAT I GOT WRONG
1-2 items. Direct and honest. "I thought X. I was wrong. Here's why."

If you have a paper trade recommendation, add before the footer:

## WHAT I THINK YOU SHOULD LOOK AT
For each recommendation:
- **"I placed a paper trade on [TICKER]"** or **"I think you should look at [direction] [TICKER]"**
- One sentence on WHY: "because [plain English reason — no jargon]"
- The market: "This is a US stock" or "This is a futures contract" or "This is crypto"
- Timing: "Based on history, this kind of move typically happens within [N] days"
- Risk: "If I'm wrong, the typical loss is about [N]%"

End every letter with:
---
*This is what I see, not financial advice. Do your own research.*

**What I'm watching:** Stocks (US markets, paper trading active) · Crypto (24/7) · \
Futures and forex (watching, not yet trading) · Prediction markets (coming soon)

— MIDGE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Under 600 words total. Lean, layered, never a list of tickers.
- Never hallucinate. Only write what the data below actually shows.
- If a section has nothing real to say, be honest: "Nothing notable here today."
- The BEST letter makes Guiding Light say: "Wait, that's connected? How did she see that?"
- The WORST letter makes Guiding Light say: "I don't understand any of this."
"""


def _call_groq(prompt: str, api_key: str) -> Optional[str]:
    """Call Groq chat completions. Returns text or None on any failure."""
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available for narrative LLM call")
        return None

    try:
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                f"{_GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": _GROQ_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": _SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": _MAX_TOKENS,
                    "temperature": 0.7,
                },
            )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        else:
            logger.debug("Groq HTTP %s: %s", resp.status_code, resp.text[:200])
    except Exception:
        logger.debug("Groq call failed", exc_info=True)
    return None


# ── Data Gathering ───────────────────────────────────────────────────


def _safe_read_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.debug("Could not read %s", path, exc_info=True)
    return None


def _read_recent_jsonl(path: Path, days: int = 1, max_lines: int = 50) -> list[dict]:
    """Read records from JSONL from the last `days` days."""
    cutoff = datetime.now() - timedelta(days=days)
    records = []
    try:
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        # Walk from end to avoid reading huge files
        for line in reversed(lines[-max_lines * 4:]):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts_raw = d.get("timestamp") or d.get("predicted_at") or d.get("evaluated_at") or ""
                if ts_raw:
                    ts = datetime.fromisoformat(ts_raw[:19])
                    if ts >= cutoff:
                        records.append(d)
                        if len(records) >= max_lines:
                            break
            except Exception:
                pass
    except Exception:
        logger.debug("Could not read JSONL %s", path, exc_info=True)
    return list(reversed(records))


def _gather_data(date_str: str) -> dict:
    """Collect all relevant data into a structured summary dict."""
    summary: dict = {"date": date_str}

    # ── Layer 0: Market Regime ─────────────────────────────────────
    # Primary: convergence_state.json has a 'regime' key
    # Fallback: regime_classifier state if it exists
    try:
        _conv_state = _safe_read_json(_DATA_MIDGE / "convergence_state.json")
        if _conv_state and isinstance(_conv_state, dict):
            summary["regime"] = _conv_state.get("regime", "unknown")
            _global = _conv_state.get("global", {})
            summary["regime_direction"] = _global.get("direction", "")
            summary["regime_domains"] = _global.get("domains", [])
        else:
            summary["regime"] = "unknown"
            summary["regime_direction"] = ""
            summary["regime_domains"] = []
    except Exception:
        logger.debug("Could not read regime from convergence_state", exc_info=True)
        summary["regime"] = "unknown"
        summary["regime_direction"] = ""
        summary["regime_domains"] = []

    # ── Layer 0: Macro Indicators from signal buffer ───────────────
    # Parse FRED/macro signals to surface yield curve, VIX, inflation data
    try:
        _buf = _safe_read_json(_DATA_MARKET / "signal_buffer.json")
        if _buf and isinstance(_buf, dict):
            macro_signals = _buf.get("macro", [])
            energy_signals = _buf.get("energy", [])
            positioning_signals = _buf.get("positioning", [])
            crypto_signals = _buf.get("crypto", [])

            # ── Macro: extract key FRED series ──────────────────
            _macro_indicators: list[dict] = []
            _seen_series: set[str] = set()
            for sig in macro_signals:
                if not isinstance(sig, dict):
                    continue
                meta = sig.get("metadata", {})
                series_id = meta.get("series_id", "")
                series_name = meta.get("series_name", "")
                if series_id and series_id not in _seen_series:
                    _seen_series.add(series_id)
                    value = meta.get("value")
                    signal_type = meta.get("signal_type", "")
                    direction = sig.get("direction", "neutral")
                    _macro_indicators.append({
                        "series_id": series_id,
                        "name": series_name,
                        "value": value,
                        "signal_type": signal_type,
                        "direction": direction,
                        "strength": round(float(sig.get("strength", 0)), 3),
                    })
            # Sort by strength (most deviated from neutral first)
            _macro_indicators.sort(key=lambda x: x["strength"], reverse=True)
            summary["macro_indicators"] = _macro_indicators[:6]

            # ── Macro: dominant direction ────────────────────────
            if macro_signals:
                _bull = sum(1 for s in macro_signals if s.get("direction") == "bullish")
                _bear = sum(1 for s in macro_signals if s.get("direction") == "bearish")
                _total_mac = _bull + _bear
                if _total_mac > 0:
                    summary["macro_alignment"] = {
                        "bullish_count": _bull,
                        "bearish_count": _bear,
                        "dominant": "bullish" if _bull > _bear else ("bearish" if _bear > _bull else "mixed"),
                        "divergent": abs(_bull - _bear) < (_total_mac * 0.3),  # True if <30% gap
                    }
                else:
                    summary["macro_alignment"] = {}
            else:
                summary["macro_alignment"] = {}

            # ── Energy: extract EIA key readings ─────────────────
            _energy_readings: list[dict] = []
            _seen_energy: set[str] = set()
            for sig in energy_signals:
                if not isinstance(sig, dict):
                    continue
                meta = sig.get("metadata", {})
                series_key = meta.get("series_key", "")
                if series_key and series_key not in _seen_energy:
                    _seen_energy.add(series_key)
                    _energy_readings.append({
                        "series_key": series_key,
                        "name": meta.get("series_name", series_key),
                        "value": meta.get("value"),
                        "change_pct": meta.get("change_pct"),
                        "direction": sig.get("direction", "neutral"),
                        "affected_tickers": meta.get("affected_tickers", [])[:3],
                    })
            summary["energy_readings"] = _energy_readings[:4]

            # ── Positioning: COT direction summary ───────────────
            if positioning_signals:
                _pos_bull = sum(1 for s in positioning_signals if s.get("direction") == "bullish")
                _pos_bear = sum(1 for s in positioning_signals if s.get("direction") == "bearish")
                summary["cot_positioning"] = {
                    "bullish_count": _pos_bull,
                    "bearish_count": _pos_bear,
                    "dominant": "bullish" if _pos_bull > _pos_bear else (
                        "bearish" if _pos_bear > _pos_bull else "mixed"
                    ),
                }
            else:
                summary["cot_positioning"] = {}

            # ── Crypto: summarize major coins from buffer ─────────
            _crypto_coins: dict[str, dict] = {}
            for sig in crypto_signals:
                if not isinstance(sig, dict):
                    continue
                meta = sig.get("metadata", {})
                symbol = meta.get("symbol", "")
                if symbol and symbol not in _crypto_coins:
                    _crypto_coins[symbol] = {
                        "symbol": symbol,
                        "price_usd": meta.get("price_usd"),
                        "change_24h_pct": meta.get("change_24h_pct"),
                        "change_7d_pct": meta.get("change_7d_pct"),
                        "direction": sig.get("direction", "neutral"),
                    }
            # Major coins first
            _major_order = ["BTC", "ETH", "SOL", "XRP", "ADA", "BNB"]
            _sorted_crypto = sorted(
                _crypto_coins.values(),
                key=lambda x: (_major_order.index(x["symbol"]) if x["symbol"] in _major_order else 99)
            )
            summary["crypto_coins"] = _sorted_crypto[:6]

        else:
            summary["macro_indicators"] = []
            summary["macro_alignment"] = {}
            summary["energy_readings"] = []
            summary["cot_positioning"] = {}
            summary["crypto_coins"] = []
    except Exception:
        logger.debug("Could not parse signal buffer for macro/energy/crypto layers", exc_info=True)
        summary["macro_indicators"] = []
        summary["macro_alignment"] = {}
        summary["energy_readings"] = []
        summary["cot_positioning"] = {}
        summary["crypto_coins"] = []

    # ── Layer 0: Futures/Forex from somatic state ──────────────────
    # Extract futures and forex symbols from somatic ticker states
    try:
        _somatic_raw = _safe_read_json(_DATA_MARKET / "somatic_state.json")
        if _somatic_raw and isinstance(_somatic_raw, dict):
            _ticker_states = _somatic_raw.get("ticker_states", {})
            _futures_data: list[dict] = []
            _futures_symbols = {"GC=F": "Gold", "CL=F": "Oil", "NQ=F": "Nasdaq futures",
                                "ES=F": "S&P 500 futures", "EURUSD=X": "EUR/USD",
                                "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY"}
            for sym, friendly_name in _futures_symbols.items():
                ts = _ticker_states.get(sym, {})
                if not ts:
                    continue
                directions = ts.get("directions", {})
                dom_dir = max(directions, key=lambda d: directions[d], default="neutral") if directions else "neutral"
                domains_active = ts.get("domains_active", [])
                signal_count = ts.get("signal_count", 0)
                if signal_count >= 3:  # Only report if there's meaningful activity
                    _futures_data.append({
                        "symbol": sym,
                        "friendly_name": friendly_name,
                        "dominant_direction": dom_dir,
                        "signal_count": signal_count,
                        "domains": domains_active[:3],
                    })
            summary["futures_activity"] = _futures_data
        else:
            summary["futures_activity"] = []
    except Exception:
        logger.debug("Could not extract futures activity from somatic state", exc_info=True)
        summary["futures_activity"] = []

    # ── Developing situations ──────────────────────────────────────
    # Note: stack_lines are added in a second pass after postmortem loads _combo_stats.
    situation_board = _safe_read_json(_DATA_MIDGE / "situation_board.json")
    if situation_board and isinstance(situation_board, dict):
        findings = situation_board.get("findings", [])
        # Top 3 by confidence
        findings_sorted = sorted(
            findings, key=lambda x: float(x.get("confidence", 0)), reverse=True
        )[:3]
        summary["developing"] = [
            {
                "ticker": f.get("ticker", "?"),
                "direction": f.get("direction", "?"),
                "confidence": round(float(f.get("confidence", 0)) * 100),
                "domains": f.get("domains", []),
                "summary": f.get("summary", ""),
                "world_model_chain": "",  # filled in second pass
                "stack_lines": [],        # filled in second pass
            }
            for f in findings_sorted
        ]
    else:
        summary["developing"] = []

    # ── Recent convergence alerts (last 24h) ───────────────────────
    recent_alerts = _read_recent_jsonl(_DATA_MIDGE / "alerts_human.jsonl", days=1, max_lines=20)
    # Deduplicate by ticker
    seen_tickers: set[str] = set()
    top_alerts = []
    for a in sorted(
        recent_alerts, key=lambda x: float(x.get("metadata", {}).get("confidence", 0) or 0), reverse=True
    ):
        t = a.get("ticker", "")
        if t and t not in seen_tickers:
            seen_tickers.add(t)
            top_alerts.append({
                "ticker": t,
                "direction": a.get("direction", "?"),
                "confidence": round(float(a.get("metadata", {}).get("confidence", 0) or 0) * 100),
                "source": a.get("source", "?"),
            })
        if len(top_alerts) >= 5:
            break
    summary["recent_alerts"] = top_alerts

    # ── Recent outcomes (last 7 days) ──────────────────────────────
    recent_outcomes = _read_recent_jsonl(_DATA_MARKET / "outcomes.jsonl", days=7, max_lines=30)
    wins = [o for o in recent_outcomes if o.get("success") is True or o.get("was_correct") is True]
    losses = [o for o in recent_outcomes if o.get("success") is False or o.get("was_correct") is False]
    summary["outcomes"] = {
        "total": len(recent_outcomes),
        "wins": len(wins),
        "losses": len(losses),
        "win_examples": [
            {"symbol": w.get("symbol", "?"), "source": w.get("source", "?"),
             "pct": round(float(w.get("price_change_pct", 0) or 0), 1)}
            for w in wins[:3]
        ],
        "loss_examples": [
            {"symbol": l.get("symbol", "?"), "source": l.get("source", "?"),
             "pct": round(float(l.get("price_change_pct", 0) or 0), 1)}
            for l in losses[:3]
        ],
    }

    # ── Post-mortem (overall performance) ─────────────────────────
    postmortem = _safe_read_json(_DATA_MIDGE / "postmortem_continuous.json")
    _combo_stats: dict = {}  # kept for stack enrichment below
    if postmortem and isinstance(postmortem, dict):
        overall = postmortem.get("overall", {})
        best_combos = postmortem.get("top_5_best_combos", [])[:3]
        worst_combos = postmortem.get("top_5_worst_combos", [])[:2]
        timing = postmortem.get("timing", {})
        # Flatten combo_stats for _build_stack_description lookup
        raw_combo_stats = postmortem.get("combo_stats", {})
        if isinstance(raw_combo_stats, dict):
            for _ckey, _cval in raw_combo_stats.items():
                if isinstance(_cval, dict):
                    _combo_stats[_ckey] = _cval
        summary["postmortem"] = {
            "overall_win_rate_pct": overall.get("overall_win_rate_pct", "?"),
            "grade": overall.get("grade", "?"),
            "total_graded": overall.get("total_graded", 0),
            "best_combos": [
                f"{c['combo']} ({c['win_rate_pct']}, n={c['n']})" for c in best_combos if isinstance(c, dict)
            ],
            "worst_combos": [
                f"{c['combo']} ({c['win_rate_pct']}, n={c['n']})" for c in worst_combos if isinstance(c, dict)
            ],
            "timing_insight": timing.get("insight", ""),
        }
    else:
        summary["postmortem"] = {}

    # ── Second pass: enrich developing situations with stack_lines ──
    # Now that _combo_stats is populated from postmortem, build stack narratives.
    try:
        for _dev in summary.get("developing", []):
            _ticker = _dev.get("ticker", "?")
            _direction = _dev.get("direction", "?")
            _domains = _dev.get("domains", [])
            _chain = _dev.get("world_model_chain", "")
            if _domains:
                _dev["stack_lines"] = _build_stack_description(
                    _ticker, _direction, _domains, _chain, _combo_stats
                )
    except Exception:
        logger.debug("Could not enrich developing situations with stack_lines", exc_info=True)

    # ── Thompson distributions — notable movers ────────────────────
    thompson = _safe_read_json(_DATA_MARKET / "thompson_distributions.json")
    if thompson and isinstance(thompson, dict):
        # Find distributions furthest from prior (alpha=2, beta=2 → mean=0.5)
        def _mean(d: dict) -> float:
            inner = d.get("default", d)
            a = float(inner.get("alpha", 2.0))
            b = float(inner.get("beta", 2.0))
            return a / (a + b) if (a + b) > 0 else 0.5

        scored = []
        for k, v in thompson.items():
            if isinstance(v, dict):
                m = _mean(v)
                inner = v.get("default", v)
                n = float(inner.get("alpha", 2.0)) + float(inner.get("beta", 2.0))
                deviation = abs(m - 0.5)
                # Only include if meaningfully informed (n > 4)
                if n > 4:
                    scored.append((k, round(m * 100), round(deviation * 100), round(n)))

        scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)
        summary["thompson"] = {
            "trusted": [
                {"source": k, "win_rate_pct": wr, "n_observations": n}
                for k, wr, _, n in scored_sorted[:3] if wr > 55
            ],
            "distrusted": [
                {"source": k, "win_rate_pct": wr, "n_observations": n}
                for k, wr, _, n in scored_sorted[:5] if wr < 45
            ],
        }
    else:
        summary["thompson"] = {}

    # ── Granger causality discoveries ─────────────────────────────
    granger = _safe_read_json(_DATA_MARKET / "granger_continuous.json")
    if granger and isinstance(granger, dict):
        findings = granger.get("findings", [])[:5]
        _DOMAIN_PLAIN_GRANGER = {
            "insider": "insider buying reports",
            "macro": "big-picture economic data",
            "technical": "price chart patterns",
            "events": "company announcements",
            "positioning": "what professional traders are doing",
            "government": "government contracts and congressional trades",
            "contracts": "government contract awards",
            "sentiment": "social media mood",
            "fundamental": "company financials",
            "institutional": "big fund movements",
            "crypto": "crypto market signals",
            "energy": "energy supply and demand data",
            "causal": "confirmed cause-and-effect chains",
            "cascade": "domino-effect chain confirmations",
        }
        summary["granger"] = [
            {
                "story": (
                    f"When {_DOMAIN_PLAIN_GRANGER.get(f.get('cause_domain',''), f.get('cause_domain','?'))} change, "
                    f"{_DOMAIN_PLAIN_GRANGER.get(f.get('effect_domain',''), f.get('effect_domain','?'))} tend to follow "
                    f"about {f.get('best_lag', '?')} days later"
                )
            }
            for f in findings if isinstance(f, dict)
        ]
    else:
        summary["granger"] = []

    # ── Paper trades summary ────────────────────────────────────────
    recent_trades = _read_recent_jsonl(_DATA_MIDGE / "paper_trades.jsonl", days=1, max_lines=10)
    summary["paper_trades_today"] = len(recent_trades)
    summary["top_trade"] = None
    if recent_trades:
        best = max(recent_trades, key=lambda x: float(x.get("confidence", 0)))
        summary["top_trade"] = {
            "asset": best.get("asset", best.get("ticker", "?")),
            "direction": best.get("direction", "?"),
            "confidence": round(float(best.get("confidence", 0)) * 100),
        }

    # ── DeepAnalyst inevitabilities (top 5 most structurally inevitable) ──
    try:
        inv_records: list[dict] = []
        inv_path = _DATA_MIDGE / "inevitabilities.jsonl"
        if inv_path.exists():
            # Read the last 100 lines (many runs append throughout the day)
            # and de-duplicate by ticker+direction, keeping the highest score
            _seen_inv: dict[str, dict] = {}
            with open(inv_path, encoding="utf-8") as _f:
                _lines_inv = _f.readlines()
            for _line in reversed(_lines_inv[-200:]):
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _rec = json.loads(_line)
                    _key = f"{_rec.get('ticker', '')}:{_rec.get('direction', '')}"
                    if _key not in _seen_inv:
                        _seen_inv[_key] = _rec
                except Exception:
                    pass
            # Sort by score descending and take top 5
            inv_records = sorted(
                _seen_inv.values(),
                key=lambda x: float(x.get("score", 0)),
                reverse=True,
            )[:5]
        _inv_list = []
        for r in inv_records:
            _ticker = r.get("ticker", "?")
            _direction = r.get("direction", "?")
            _domains = r.get("domains", [])
            _chain = str(r.get("world_model_chain", "") or "")
            _stack = _build_stack_description(
                _ticker, _direction, _domains, _chain, _combo_stats
            )
            _inv_list.append({
                "ticker": _ticker,
                "direction": _direction,
                "score": round(float(r.get("score", 0)) * 100),
                "domains": _domains,
                "evidence_summary": r.get("evidence_summary", ""),
                "world_model_chain": _chain,
                "expected_window_days": r.get("expected_window_days"),
                "signal_count": r.get("signal_count", 0),
                "stack_lines": _stack,  # pre-built plain-English stack narrative
            })
        summary["inevitabilities"] = _inv_list
    except Exception:
        logger.debug("Could not read inevitabilities", exc_info=True)
        summary["inevitabilities"] = []

    # ── Developing situations (OctopusColony partial convergences) ────
    try:
        _dev_sit = _safe_read_json(_DATA_MARKET / "developing_situations.json")
        if _dev_sit and isinstance(_dev_sit, dict):
            _sits = []
            for _key, _sit in _dev_sit.items():
                if not isinstance(_sit, dict):
                    continue
                _check_count = _sit.get("check_count", 0)
                if _check_count >= 20:
                    continue  # Eviction candidate — skip stale
                _sits.append({
                    "ticker": _sit.get("ticker", "?"),
                    "direction": _sit.get("direction", "?"),
                    "domains_seen": _sit.get("domains_seen", []),
                    "missing_domains": _sit.get("missing_domains", []),
                    "causal_predictions": _sit.get("causal_predictions", []),
                    "check_count": _check_count,
                    "investigation_results": _sit.get("investigation_results", [])[:2],
                })
            # Sort by check_count descending (most-watched first)
            _sits.sort(key=lambda x: x["check_count"], reverse=True)
            summary["developing_situations"] = _sits[:5]
        else:
            summary["developing_situations"] = []
    except Exception:
        logger.debug("Could not read developing_situations", exc_info=True)
        summary["developing_situations"] = []

    # ── Active hypotheses (HypothesisEngine: active + strong probation) ─
    try:
        _hyp_snap = _safe_read_json(_DATA_MARKET / "hypotheses_snapshot.json")
        if _hyp_snap and isinstance(_hyp_snap, dict):
            _hyps_raw = _hyp_snap.get("hypotheses", {})
            _active_hyps = []
            for _hid, _h in _hyps_raw.items():
                _status = _h.get("status", "")
                if _status not in ("active", "probation"):
                    continue
                _stats = _h.get("stats", {})
                _n = int(_stats.get("total_observations", 0))
                _wr = (
                    round(_stats.get("wins", 0) / _n * 100)
                    if _n > 0 else None
                )
                # Only include if it has observations or is active
                if _status == "probation" and _n < 5:
                    continue
                _active_hyps.append({
                    "name": _h.get("name", "?"),
                    "status": _status,
                    "causal_story": _h.get("causal_story", "")[:200],
                    "win_rate_pct": _wr,
                    "observations": _n,
                    "cumulative_return_pct": round(float(_stats.get("cumulative_return_pct", 0)), 1),
                })
            # Active first, then by observations
            _active_hyps.sort(
                key=lambda x: (x["status"] != "active", -(x["observations"] or 0))
            )
            summary["active_hypotheses"] = _active_hyps[:6]
        else:
            summary["active_hypotheses"] = []
    except Exception:
        logger.debug("Could not read hypotheses_snapshot", exc_info=True)
        summary["active_hypotheses"] = []

    # ── Cascade tracker status (causal chains unfolding) ─────────────
    try:
        _cascade_snap = _safe_read_json(_DATA_MARKET / "cascade_snapshot.json")
        if _cascade_snap and isinstance(_cascade_snap, dict):
            _chains = _cascade_snap.get("chains", [])
            _stats_c = _cascade_snap.get("stats", {})
            # Only include chains with at least one confirmed link
            _active_cascades = [c for c in _chains if c.get("confirmed_count", 0) > 0]
            summary["cascade_status"] = {
                "active_chains": _stats_c.get("active_chains", 0),
                "confirmed_links": _stats_c.get("confirmed_links", 0),
                "pending_links": _stats_c.get("pending_links", 0),
                "confirmation_rate": _stats_c.get("confirmation_rate", 0),
                "mean_energy_ratio": _stats_c.get("mean_energy_ratio"),
                "notable_chains": [
                    {
                        "trigger": c.get("trigger", "?"),
                        "direction": c.get("direction", "?"),
                        "confirmed_count": c.get("confirmed_count", 0),
                        "total_links": c.get("total_links", 0),
                        "confirmed_tickers": c.get("confirmed_tickers", []),
                        "next_dominoes": c.get("next_dominoes", []),
                        "energy_ratio": c.get("mean_energy_ratio"),
                    }
                    for c in _active_cascades[:3]
                ],
            }
        else:
            summary["cascade_status"] = {}
    except Exception:
        logger.debug("Could not read cascade_snapshot", exc_info=True)
        summary["cascade_status"] = {}

    # ── Somatic anticipation (pre-convergence signal accumulation) ────
    try:
        _somatic = _safe_read_json(_DATA_MARKET / "somatic_state.json")
        if _somatic and isinstance(_somatic, dict):
            _ticker_states = _somatic.get("ticker_states", {})
            _building: list[dict] = []
            for _ticker, _ts in _ticker_states.items():
                _domains = _ts.get("domains_active", [])
                if len(_domains) < 2:
                    continue  # Need at least 2 domains to be interesting
                _directions = _ts.get("directions", {})
                # Find dominant direction (most signals)
                _dom_dir = max(_directions, key=lambda d: _directions[d], default="neutral")
                _total_signals = _ts.get("signal_count", 0)
                if _total_signals < 10:
                    continue
                _building.append({
                    "ticker": _ticker,
                    "domains": _domains,
                    "dominant_direction": _dom_dir,
                    "signal_count": _total_signals,
                    "domain_count": len(_domains),
                })
            # Sort by domain count then signal count
            _building.sort(key=lambda x: (x["domain_count"], x["signal_count"]), reverse=True)
            summary["somatic_building"] = _building[:5]
        else:
            summary["somatic_building"] = []
    except Exception:
        logger.debug("Could not read somatic_state", exc_info=True)
        summary["somatic_building"] = []

    # ── Cross-market anomaly discoveries ────────────────────────────
    cross_market = _read_recent_jsonl(_DATA_MARKET / "cross_market_discoveries.jsonl", days=1, max_lines=5)
    summary["cross_market_anomalies"] = [
        {
            "type": d.get("discovery_type", "unknown"),
            "description": d.get("description", ""),
            "strength": d.get("strength", 0),
            "tickers": d.get("affected_tickers", [])[:5],
            "domains": d.get("affected_domains", []),
        }
        for d in cross_market if isinstance(d, dict)
    ]

    # ── Crypto Fear & Greed Index ────────────────────────────────────
    try:
        from mae_core.market.apis.crypto_fear_greed_client import get_fear_greed
        fg = get_fear_greed()
        if fg:
            summary["crypto_fear_greed"] = {
                "value": fg.value,
                "classification": fg.classification,
                "direction": fg.direction,
                "trend": fg.trend,
            }
        else:
            summary["crypto_fear_greed"] = {}
    except Exception:
        summary["crypto_fear_greed"] = {}

    return summary


# ── Narrative Generation ─────────────────────────────────────────────


def _confidence_words(pct: int) -> str:
    """Translate a numeric confidence percentage into plain English."""
    if pct > 80:
        return "very confident"
    if pct > 60:
        return "fairly sure"
    if pct >= 45:
        return "forming — need more"
    return "early / still watching"


# ── Per-domain plain-English signal descriptions ─────────────────────
# Each domain gets a one-sentence description of WHAT was observed.
# Used to build the stack narrative so the LLM (and template) can show
# each independent pillar rather than just listing domain names.
_DOMAIN_WHAT_SEEN: dict[str, str] = {
    "insider": "People inside the company are buying or selling their own stock",
    "macro": "Big-picture economic data (government spending, borrowing costs, inflation) is signalling a shift",
    "technical": "The price chart has triggered a recognisable pattern that traders watch for",
    "events": "The company has made announcements or filed reports that suggest something is changing",
    "positioning": "The biggest professional traders have moved their bets in this direction",
    "government": "Government contract data or congressional trading shows activity in this direction",
    "contracts": "Government contract awards are shifting toward or away from this company",
    "sentiment": "Social media chatter about this company has turned strongly in one direction",
    "fundamental": "The company's own financial numbers are pointing this way",
    "institutional": "Large investment funds have been quietly moving money in this direction",
    "crypto": "Crypto market signals are aligning with this outcome",
    "energy": "Energy supply and demand data is reinforcing this direction",
    "causal": "A confirmed cause-and-effect chain is pointing here",
    "cascade": "A predicted domino chain has confirmed links pointing this way",
}

# Pre-written independence notes for common domain pair combinations.
# These explain WHY two domains being independent makes their agreement meaningful.
_INDEPENDENCE_NOTES: dict[frozenset, str] = {
    frozenset({"insider", "technical"}): "Insiders read company secrets — not price charts. When both agree, that's two completely different worlds reaching the same conclusion.",
    frozenset({"insider", "macro"}): "People inside a company don't set government economic policy. When insider selling aligns with macro warnings, the risk is coming from two independent directions.",
    frozenset({"insider", "events"}): "Filing reports and trading your own shares are separate actions — one is public disclosure, the other is private conviction with money behind it.",
    frozenset({"insider", "government"}): "Congressional trades and corporate insider trades are reported independently. When both point the same direction, it crosses two completely separate information silos.",
    frozenset({"insider", "sentiment"}): "Social media doesn't know what insiders are doing (insider trades can take days to be reported). Two independent awareness channels agreeing is a strong signal.",
    frozenset({"technical", "macro"}): "Price chart patterns are blind to macro policy. When charts and economic data agree, that's the market's internal momentum matching the external environment.",
    frozenset({"technical", "events"}): "Charts reflect collective trader behaviour; company announcements are raw facts. When both align, the story and the market reaction are pointing the same way.",
    frozenset({"technical", "government"}): "Price patterns and government contract data come from entirely separate worlds. Agreement across them is unusual and meaningful.",
    frozenset({"events", "macro"}): "Company-specific announcements and broad economic trends are independent. When a company's own news aligns with the macro environment, the pressure comes from both inside and outside.",
    frozenset({"events", "government"}): "What a company announces publicly and what government contracts show are separate data streams. When they converge, the situation is confirming from official and private sources simultaneously.",
    frozenset({"institutional", "insider"}): "Big funds and company insiders have completely different information pipelines. When both are moving the same way, money is flowing from two unconnected sources of conviction.",
    frozenset({"institutional", "technical"}): "Institutional positioning and price chart patterns are measured differently and by different people. Agreement means both the smart money and the chart-readers see the same thing.",
    frozenset({"government", "technical"}): "Government contract data and price charts have nothing to do with each other. Convergence here is genuinely unusual.",
    frozenset({"sentiment", "technical"}): "Social media mood and chart patterns are generated by different populations — retail chatter vs. price action. When they align, broad market psychology is consistent with the money flow.",
    frozenset({"energy", "macro"}): "Energy inventory data and broad economic signals are measured independently by different agencies. When both point the same way, real-world supply meets financial policy.",
}


def _get_independence_note(domains: list[str]) -> str:
    """Return the most illustrative independence note for a set of domains.

    Checks all pairs in the list. Returns the first matching pair note,
    prioritising pairs that involve the most 'surprising' combinations
    (insider+technical > everything else, then government combos).
    """
    _PRIORITY_PAIRS = [
        frozenset({"insider", "technical"}),
        frozenset({"insider", "government"}),
        frozenset({"government", "technical"}),
        frozenset({"institutional", "insider"}),
        frozenset({"insider", "macro"}),
        frozenset({"technical", "macro"}),
        frozenset({"events", "government"}),
        frozenset({"institutional", "technical"}),
        frozenset({"insider", "events"}),
        frozenset({"energy", "macro"}),
        frozenset({"technical", "events"}),
        frozenset({"events", "macro"}),
        frozenset({"insider", "sentiment"}),
        frozenset({"sentiment", "technical"}),
        frozenset({"technical", "government"}),
    ]
    domain_set = {d.lower() for d in domains}
    for pair in _PRIORITY_PAIRS:
        if pair.issubset(domain_set):
            note = _INDEPENDENCE_NOTES.get(pair)
            if note:
                return note
    return (
        "These signals come from completely separate information sources — "
        "they have no way of knowing about each other. That's what makes their agreement meaningful."
    )


def _combo_win_rate(domains: list[str], combo_stats: dict) -> str | None:
    """Look up the historical win rate for a domain combination in the postmortem.

    combo_stats keys are like 'insider+technical'. We try the exact sorted combo
    first, then fall back to the best matching subset.
    Returns a plain-English string or None if no data.
    """
    if not combo_stats or not domains:
        return None
    sorted_key = "+".join(sorted(d.lower() for d in domains))
    # Exact match
    if sorted_key in combo_stats:
        stat = combo_stats[sorted_key]
        wr = stat.get("win_rate", 0)
        n = stat.get("n", 0)
        if n >= 3:
            if wr >= 0.80:
                return f"This exact combination has worked most of the time ({n} past cases)"
            if wr >= 0.60:
                return f"This combination has worked more often than not ({n} past cases)"
            if wr >= 0.40:
                return f"This combination has worked about half the time ({n} past cases)"
            return f"This combination has been unreliable so far ({n} past cases)"
    # Try all sub-pairs (2-domain combos) for any partial match
    for size in range(len(domains) - 1, 1, -1):
        from itertools import combinations as _combos
        for subset in _combos(sorted(domains), size):
            sub_key = "+".join(subset)
            if sub_key in combo_stats:
                stat = combo_stats[sub_key]
                wr = stat.get("win_rate", 0)
                n = stat.get("n", 0)
                if n >= 3 and wr >= 0.60:
                    combo_desc = " and ".join(
                        _DOMAIN_WHAT_SEEN.get(d, d).split(" ")[0:3] for d in subset  # type: ignore[arg-type]
                    )
                    return f"When similar signals aligned before, it worked more often than not ({n} past cases)"
    return None


def _build_stack_description(ticker: str, direction: str, domains: list[str],
                              world_model_chain: str, combo_stats: dict) -> list[str]:
    """Build a list of plain-English lines that narrate the full pattern stack.

    Returns a list ready to be joined with newlines.  Each line is a bullet
    explaining one independent pillar of the convergence.

    This is the core of the fix — instead of giving the LLM "evidence from:
    insider, technical, events", we give it a pre-written explanation of what
    EACH pillar saw and WHY their agreement matters.
    """
    direction_verb = "pointing up" if "bull" in direction.lower() else ("pointing down" if "bear" in direction.lower() else "neutral")
    n = len(domains)

    result: list[str] = []
    result.append(
        f"{ticker} has {n} INDEPENDENT signals all {direction_verb} right now:"
    )

    for i, domain in enumerate(domains, 1):
        what_seen = _DOMAIN_WHAT_SEEN.get(domain.lower(),
                                          f"Signals from the '{domain}' area are aligned")
        result.append(f"  {i}. {what_seen}  [{domain}]")

    # Independence note
    result.append(f"  → {_get_independence_note(domains)}")

    # Historical win rate
    wr_note = _combo_win_rate(domains, combo_stats)
    if wr_note:
        result.append(f"  → History: {wr_note}")

    # World model causal chain
    if world_model_chain and world_model_chain not in ("None", "[]", ""):
        chain_clean = world_model_chain.strip("[]").replace("'", "").replace('"', "")
        # Translate known jargon nodes to plain English
        _CHAIN_NODE_MAP = {
            "datacenter_demand": "demand for data centres",
            "ai_capex_surge": "a wave of AI spending",
            "defense_contract_awards": "defence contract awards",
            "defense_spending_increase": "rising defence spending",
            "geopolitical_tension_escalation": "escalating geopolitical tension",
            "airline_fuel_costs": "airline fuel costs",
            "crude_price_spike": "an oil price spike",
            "oil_supply_disruption": "oil supply disruption",
            "opec_production_cut": "OPEC production cuts",
        }
        chain_nodes = [n.strip() for n in chain_clean.split(",")]
        chain_plain = " → ".join(
            _CHAIN_NODE_MAP.get(n.strip(), n.strip()) for n in chain_nodes
        )
        result.append(f"  → Causal chain: {chain_plain}")

    return result


def _direction_words(direction: str) -> str:
    """Translate a direction string into plain English."""
    d = direction.lower()
    if "bull" in d or "long" in d or "up" in d:
        return "looks like it might rise"
    if "bear" in d or "short" in d or "down" in d:
        return "looks like it might fall"
    return "direction unclear"


def _domain_plain(domains: list[str]) -> str:
    """Translate domain names into plain-English source descriptions.

    The goal is to give the LLM human-readable context so it doesn't
    reproduce the raw domain names in the final letter.
    """
    _MAP = {
        "insider": "insider buying/selling reports",
        "macro": "economic data",
        "technical": "price-chart patterns",
        "events": "company announcements",
        "positioning": "large-trader positioning data",
        "government": "government contracts and congressional trades",
        "contracts": "government contract awards",
        "sentiment": "social-media chatter",
        "fundamental": "company financials",
        "institutional": "large institutional fund movements",
        "crypto": "crypto market signals",
        "energy": "energy market data",
        "causal": "confirmed cause-and-effect chains",
        "cascade": "domino-effect chain confirmations",
    }
    translated = [_MAP.get(d.lower(), d) for d in domains[:4]]
    return ", ".join(translated) if translated else "multiple sources"


def _regime_plain(regime: str) -> str:
    """Translate regime codes to plain English."""
    return {
        "bull": "things are generally rising",
        "bear": "things are generally falling",
        "volatile": "unpredictable — big swings in both directions",
        "sideways": "stuck — not going anywhere in particular",
    }.get(regime.lower() if regime else "", regime or "unclear")


# ── Jargon Elimination ───────────────────────────────────────────────

# Every raw field name, system concept, and financial term that must never
# reach the human reader. Applied to ALL data before the LLM sees it, and
# again as a post-generation safety net on the LLM's output.
_JARGON_MAP: dict[str, str] = {
    # Internal system names that leak into data fields
    "ai_capex_surge": "a wave of companies spending heavily on AI",
    "port_strike": "a port workers' strike disrupting shipping",
    "yield_curve": "a key recession warning signal",
    "yield_spread": "a gap in government borrowing costs",
    "yield curve": "a key recession warning signal",
    "T10Y2Y": "the gap between short-term and long-term government borrowing costs",
    "T10Y3M": "a key recession warning signal",
    "T5YIE": "where the market expects prices to be in five years",
    "DGS2": "short-term government borrowing costs",
    "DGS10": "long-term government borrowing costs",
    "DFF": "the rate banks charge each other overnight",
    "VIXCLS": "the market's fear reading",
    "sector_rotation": "big money moving from one industry to another",
    "sector rotation": "big money moving from one industry to another",
    # Crypto-specific jargon
    "fear/greed index": "how scared or greedy crypto traders are feeling",
    "fear/greed score": "how scared or greedy crypto traders are feeling",
    "fear and greed": "how scared or greedy traders are feeling",
    "extreme fear": "traders are extremely scared",
    "extreme greed": "traders are extremely greedy",
    # LLM-output safety net: phrases the LLM might write despite instructions
    "government bond yields": "government borrowing costs",
    "bond yields": "government borrowing costs",
    "bond yield": "government borrowing costs",
    "10-year yield": "long-term government borrowing costs",
    "2-year yield": "short-term government borrowing costs",
    "yield inversion": "a recession warning signal",
    "inverted yield": "a recession warning signal",
    "interest rates": "borrowing costs",
    "rate hike": "an increase in borrowing costs",
    "rate cut": "a decrease in borrowing costs",
    "monetary policy": "what the central bank is doing",
    "quantitative easing": "the central bank printing money",
    "quantitative tightening": "the central bank pulling money back",
    "federal reserve": "the central bank",
    "Federal Reserve": "the central bank",
    "the Fed": "the central bank",
    "FOMC": "the central bank's decision-making committee",
    "basis point": "a tiny fraction of a percent",
    "market cap": "the company's total value on the stock market",
    "P/E ratio": "how expensive the stock is relative to its earnings",
    "earnings per share": "how much money the company makes per stock unit",
    "short squeeze": "a situation where traders betting against a stock get forced to buy",
    "short selling": "betting that a stock will fall",
    "put options": "insurance-like contracts that pay off if a stock falls",
    "call options": "contracts that pay off if a stock rises",
    "implied volatility": "how much the market expects prices to swing",
    "risk-off": "investors moving to safety",
    "risk-on": "investors taking more chances",
    "flight to safety": "investors moving money into safer assets",
    "safe haven": "a safer asset investors run to during uncertainty",
    "liquidity": "how easy it is to buy and sell",
    "credit spread": "how much extra investors demand to lend to riskier borrowers",
    "overnight rate": "the rate banks charge each other for short-term loans",
    "short_float": "how many traders are betting a company's stock will fall",
    "put_call": "the balance between protective and aggressive bets",
    "put/call ratio": "the balance between protective and aggressive bets",
    "funding_rate": "how much it costs traders to hold leveraged crypto positions",
    "COT": "what the biggest professional traders are doing",
    "cot_positioning": "how the biggest professional traders are positioned",
    "13f_filing": "required reports showing what big funds own",
    "form_4": "insider buying and selling reports",
    "sec_form4": "insider buying and selling reports",
    "sec_form8k": "company announcements about major events",
    "sec_efts": "SEC filing keyword searches",
    "FRED": "economic data from the US Federal Reserve",
    "fred_macro": "big-picture economic signals",
    "EIA": "energy inventory data",
    "eia_energy": "energy supply and demand data",
    "thompson": "what I've learned about which sources to trust",
    "Thompson": "what I've learned about which sources to trust",
    "Thompson distribution": "what I've learned about which sources to trust",
    "convergence": "signals from different places agreeing",
    "Granger causality": "a timing connection between two things",
    "granger": "timing connections I've discovered",
    "regime": "the market's overall mood",
    "bear market": "a market where things are generally falling",
    "bull market": "a market where things are generally rising",
    "bearish": "pointing down",
    "bullish": "pointing up",
    # Domain names — only translate them as standalone labels (word-boundary-matched)
    # These are the same values as in _domain_plain() for the post-generation scrubber
    "cot positioning": "what the biggest professional traders are doing",
    # Note: very short domain words (insider, macro, technical, events, etc.) are intentionally
    # NOT in this map to avoid corrupting plain-English sentences the LLM already wrote correctly.
    # They are translated at the data-building layer via _domain_plain() before the LLM sees them.
    "cascade": "a chain reaction I predicted",
    "pattern_stack": "multiple historical patterns lining up",
    "convergence_building": "signals starting to agree",
    "outcome_window": "the time I give a prediction to play out",
    "drawdown": "losses from the peak",
    "ATR": "the typical daily price swing",
    "RSI": "whether something is overbought or oversold",
    "MACD": "price momentum direction",
    "Bollinger": "whether a price is stretched unusually far",
    "VIX": "the market's fear reading",
    "VVIX": "how uncertain the fear reading itself is",
    "domain": "type of signal source",
    "domains": "signal sources",
    "overbought": "risen unusually fast",
    "oversold": "fallen unusually fast",
    "momentum": "how fast something is moving",
    "volatility": "how much prices are swinging",
    "contango": "an unusual pricing pattern in futures",
    "backwardation": "an unusual pricing pattern in futures",
    "basis points": "tiny fractions of a percent",
    "inversion": "an unusual reversal of the normal pattern",
    "F-stat": "a measure of how strong the connection is",
    "p-value": "how confident I am this isn't a coincidence",
    "p<": "with high confidence",
    "alpha": "a measure of how much something has learned",
    "beta": "a measure of how reliable something is",
    "standard deviation": "how far from normal",
    "forex": "currency markets",
    "crude oil": "oil",
    "crude": "oil",
    "equity": "stock",
    "equities": "stocks",
    "performing below chance": "haven't been reliable lately",
    "below chance": "less reliable than random guessing",
    "weighting them less": "trusting them less right now",
    "weighting": "how much I trust",
}

# Patterns for numeric jargon (applied via regex in _translate_jargon)
# Signal count ranges: translate "96 signals" → qualitative
_SIGNAL_COUNT_RANGES = [
    (_re.compile(r'\b([5-9][0-9]|[1-9][0-9]{2,})\s*(signals?|readings?|indicators?)\b', _re.IGNORECASE), "almost everything"),
    (_re.compile(r'\b([2-4][0-9])\s*(signals?|readings?|indicators?)\b', _re.IGNORECASE), "many signals"),
    (_re.compile(r'\b([6-9]|1[0-9]|20)\s*(signals?|readings?|indicators?)\b', _re.IGNORECASE), "several signals"),
    (_re.compile(r'\b([1-5])\s*(signals?|readings?)\b', _re.IGNORECASE), "a few signals"),
]

# "X out of Y" / "X correct out of Y" patterns
_OUT_OF_PATTERN = _re.compile(
    r'\b(\d+)\s*(?:correct\s+)?out\s+of\s+(\d+)\s*(?:checked|graded|total)?\b',
    _re.IGNORECASE,
)

# Percentage patterns that are NOT price changes (no +/- prefix)
_BARE_PCT_PATTERN = _re.compile(r'(?<![+-])\b(\d+(?:\.\d+)?)\s*%\s*of\s*(the\s+time|predictions?|calls?|cases?|checks?|observations?)\b', _re.IGNORECASE)

# "confidence 0.xx" / "score: 0.xx" / raw decimal confidence
_CONF_DECIMAL_PATTERN = _re.compile(r'\b(?:confidence|score|weight|strength)\s*[:\s]+0\.(\d+)\b', _re.IGNORECASE)

# All-uppercase acronyms that aren't known ticker symbols (3-5 chars, all caps)
# We keep known tickers; strip others
_KNOWN_TICKERS = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "DOGE", "AVAX",
    "SPY", "QQQ", "GLD", "USO", "TLT", "IWM",
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
    "US", "UK", "EU", "USD", "EUR", "GBP", "JPY",
    "MIDGE", "NYSE", "SEC", "NFP", "CPI", "GDP", "FED",
}
_UNKNOWN_ACRONYM = _re.compile(r'\b([A-Z]{2,6})\b')


def _confidence_pct_to_words(pct: float) -> str:
    """Convert a 0-100 or 0-1 confidence number to plain English."""
    if pct > 1:  # already in 0-100 scale
        if pct > 80:
            return "very confident"
        if pct > 60:
            return "fairly sure"
        if pct >= 45:
            return "something forming — need more"
        return "early, still watching"
    else:  # 0-1 scale
        if pct > 0.80:
            return "very confident"
        if pct > 0.60:
            return "fairly sure"
        if pct >= 0.45:
            return "something forming — need more"
        return "early, still watching"


def _signal_count_to_words(n: int) -> str:
    """Convert a raw signal count to a qualitative description."""
    if n >= 50:
        return "almost everything"
    if n >= 21:
        return "many signals"
    if n >= 6:
        return "several signals"
    if n >= 1:
        return "a few signals"
    return "no signals"


def _translate_jargon(text: str) -> str:
    """Replace ALL financial jargon and system internals in a text string.

    Applied to:
    1. Every data line before it enters the LLM prompt.
    2. The final LLM output as a post-generation safety net.

    Order matters: longer/more specific phrases are replaced before shorter
    ones to avoid partial-match corruption.
    """
    if not text:
        return text

    # ── Step 1: Direct phrase replacement (longest first to avoid partials) ──
    # Use word boundaries for all replacements to avoid partial-word collisions
    # (e.g. "government" should not match inside "government borrowing costs" after
    # that phrase was already translated by a longer key).
    for jargon, plain in sorted(_JARGON_MAP.items(), key=lambda x: -len(x[0])):
        # Build pattern: word-boundary anchored, case-insensitive
        pattern = r'\b' + _re.escape(jargon) + r'\b'
        try:
            text = _re.sub(pattern, plain, text, flags=_re.IGNORECASE)
        except _re.error:
            # Fallback for patterns that can't take word boundaries (e.g. "p<")
            text = _re.sub(_re.escape(jargon), plain, text, flags=_re.IGNORECASE)

    # ── Step 2: "X% of the time / calls / predictions" → qualitative ──
    def _pct_to_qual(m: _re.Match) -> str:
        pct = float(m.group(1))
        noun = m.group(2).lower().strip()
        # Avoid repeating "the time" — just say "more often than not"
        noun_suffix = "" if "time" in noun else f" ({noun})"
        if pct >= 90:
            return f"almost always{noun_suffix}"
        if pct >= 70:
            return f"more often than not{noun_suffix}"
        if pct >= 50:
            return f"about half the time{noun_suffix}"
        if pct >= 30:
            return f"sometimes{noun_suffix}"
        return f"rarely{noun_suffix}"
    text = _BARE_PCT_PATTERN.sub(_pct_to_qual, text)

    # ── Step 3: "X correct out of Y" → "X out of Y calls" ──
    def _out_of(m: _re.Match) -> str:
        wins = int(m.group(1))
        total = int(m.group(2))
        if total == 0:
            return "no calls checked yet"
        pct = wins / total
        if pct >= 0.75:
            return f"{wins} out of {total} — most of them right"
        if pct >= 0.5:
            return f"{wins} out of {total} — about half right"
        return f"{wins} out of {total} — still learning"
    text = _OUT_OF_PATTERN.sub(_out_of, text)

    # ── Step 4: "confidence/score/weight 0.xx" → plain words ──
    def _conf_decimal(m: _re.Match) -> str:
        digits = m.group(1)
        val = float(f"0.{digits}")
        return _confidence_pct_to_words(val)
    text = _CONF_DECIMAL_PATTERN.sub(_conf_decimal, text)

    # ── Step 5: Remove stray decimal confidence numbers like "(0.72)" ──
    text = _re.sub(r'\(0\.\d+\)', '', text)

    # ── Step 6: "±2.3%" change notes → qualitative ──
    def _change_pct(m: _re.Match) -> str:
        val = float(m.group(1))
        sign = m.group(0)[0]  # '+' or '-' or digit
        direction = "up" if ('+' in m.group(0) or val > 0) else "down"
        if abs(val) >= 10:
            return f"sharply {direction}"
        if abs(val) >= 3:
            return f"noticeably {direction}"
        if abs(val) >= 0.5:
            return f"slightly {direction}"
        return "barely moved"
    # Only apply to explicit change_pct annotations like "(changed +2.3%)" or "changed -5.1%"
    text = _re.sub(
        r'\(changed\s*([+-]?\d+(?:\.\d+)?)\s*%\)',
        lambda m: f"({_change_pct(m)})",
        text,
        flags=_re.IGNORECASE,
    )

    # ── Step 7: Standalone "(N.NN)" numeric values — remove if they look like raw decimals ──
    # e.g. "(4.52)" after a series name
    text = _re.sub(r'\s*\(\d+\.\d{2,}\)\s*', ' ', text)

    return text


def _build_llm_prompt(summary: dict) -> str:
    """Build the plain-English data context to hand to the LLM.

    Data is presented in the LAYERED ORDER the letter should follow:
    Big picture → Crypto → Commodities/Futures → Stocks → Learned/Wrong.

    All technical terms are translated here so the LLM never sees jargon
    it might echo back. The LLM's job is ONLY to write the letter in MIDGE's
    voice — we do the data translation in Python.
    """
    lines = [
        f"Today is {summary['date']}. Here is everything I know right now.",
        "Write the daily letter using this data. Follow all style rules exactly.",
        "CRITICAL: Follow the LAYERED structure — big picture FIRST, specific stock tickers LAST.",
        "You are NOT a stock screener. You watch everything — stocks, crypto, commodities, futures, macro.",
        "",
    ]

    # ══════════════════════════════════════════════════════════════════
    # LAYER 1: THE BIG PICTURE
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 1: THE BIG PICTURE ━━━")
    lines.append("(Use this data for the ## THE BIG PICTURE section)")
    lines.append("")

    # Regime
    regime = summary.get("regime", "unknown")
    regime_plain = _regime_plain(regime)
    lines.append(f"WHAT THE MARKET IS DOING RIGHT NOW: {regime_plain}.")

    # Macro alignment
    macro_align = summary.get("macro_alignment", {})
    if macro_align:
        dominant = macro_align.get("dominant", "mixed")
        bull_c = macro_align.get("bullish_count", 0)
        bear_c = macro_align.get("bearish_count", 0)
        divergent = macro_align.get("divergent", False)
        up_words = _signal_count_to_words(bull_c)
        down_words = _signal_count_to_words(bear_c)
        align_note = "pulling in different directions" if divergent else f"mostly pointing {dominant.replace('bullish','up').replace('bearish','down')}"
        lines.append(
            f"BIG-PICTURE ECONOMIC SIGNALS: {up_words} pointing up, {down_words} pointing down — {align_note}."
        )

    # Key macro indicators
    macro_indicators = summary.get("macro_indicators", [])
    _MACRO_PLAIN = {
        "T10Y2Y": "a key recession warning signal (how governments borrow at different time horizons)",
        "T10Y3M": "a key recession warning signal",
        "T5YIE": "where the market thinks prices will be in five years",
        "DGS2": "short-term government borrowing costs",
        "DGS10": "long-term government borrowing costs",
        "DFF": "the overnight rate banks charge each other",
        "VIXCLS": "the market's fear reading",
    }
    if macro_indicators:
        lines.append("KEY ECONOMIC READINGS:")
        for ind in macro_indicators[:4]:
            plain_name = _MACRO_PLAIN.get(ind["series_id"], ind.get("name", ind["series_id"]))
            direction = ind.get("direction", "neutral")
            direction_plain = "pointing up" if direction == "bullish" else ("pointing down" if direction == "bearish" else "neutral")
            lines.append(f"  - {plain_name}: {direction_plain}.")

    lines.append(
        "  INSTRUCTION for Big Picture: Lead with what the market mood means in plain English. "
        "Are big-picture signals agreeing or fighting each other? "
        "Example: 'Things are generally falling right now — and most of what I'm watching agrees. "
        "The signals from government borrowing costs are behaving like investors are worried.' "
        "Keep this section 2-4 bullets. Don't go into individual stocks yet. NO financial terms."
    )

    lines.append("")

    # Cross-market anomalies (belongs in Big Picture)
    cross_market = summary.get("cross_market_anomalies", [])
    if cross_market:
        lines.append("CROSS-MARKET ANOMALIES (weird things happening across unrelated markets):")
        for cm in cross_market[:3]:
            tickers_note = f" Tickers involved: {', '.join(cm['tickers'][:4])}." if cm.get("tickers") else ""
            domains_note = f" Areas: {_domain_plain(cm.get('domains', []))}." if cm.get("domains") else ""
            lines.append(f"  - {cm.get('description', cm.get('type', 'unknown'))}.{tickers_note}{domains_note}")
        lines.append(
            "  INSTRUCTION: If anything here is genuinely weird (unrelated markets moving together), "
            "add it to the Big Picture section. Example: 'Three completely unrelated sectors all moved "
            "the same direction on the same day — energy, tech defense. Something is flowing underneath.' "
            "This is the kind of thing Guiding Light loves most."
        )
    else:
        lines.append("CROSS-MARKET ANOMALIES: Nothing unusual across unrelated markets today.")

    lines.append("")

    # ── Granger causal discoveries (also Big Picture level) ────────
    granger = summary.get("granger", [])
    if granger:
        lines.append("STRANGE CAUSAL DISCOVERIES (put the best one in Big Picture as the hook):")
        for g in granger[:3]:
            lines.append(f"  - {g['story']}")
        lines.append(
            "  INSTRUCTION: The single weirdest Granger finding should be the 1-sentence hook "
            "at the very top of the letter — before any sections. "
            "Example: 'Here's something strange I noticed: when energy inventory data changes, "
            "defense stocks tend to follow about 3 days later.' "
            "Plain English only — translate domain names."
        )
    else:
        lines.append("CAUSAL DISCOVERIES: None new to report.")

    lines.append("")

    # Energy (also Big Picture)
    energy_readings = summary.get("energy_readings", [])
    if energy_readings:
        lines.append("ENERGY PICTURE (oil/gas/production data):")
        _ENERGY_PLAIN = {
            "crude_production": "US crude oil production",
            "crude_inventory": "US crude oil stockpiles",
            "gasoline_inventory": "US gasoline stockpiles",
            "natural_gas_storage": "US natural gas storage",
            "crude_imports": "US crude oil imports",
        }
        for er in energy_readings:
            plain_name = _ENERGY_PLAIN.get(er["series_key"], er.get("name", er["series_key"]))
            change = er.get("change_pct")
            if change is not None:
                abs_c = abs(change)
                direction_ch = "up" if change > 0 else "down"
                if abs_c >= 10:
                    change_note = f" (sharply {direction_ch})"
                elif abs_c >= 3:
                    change_note = f" (noticeably {direction_ch})"
                elif abs_c >= 0.5:
                    change_note = f" (slightly {direction_ch})"
                else:
                    change_note = " (barely moved)"
            else:
                change_note = ""
            direction = er.get("direction", "neutral")
            direction_plain = "rising" if direction == "bullish" else ("falling" if direction == "bearish" else direction)
            tickers = er.get("affected_tickers", [])
            ticker_note = f" Affects: {', '.join(tickers)}." if tickers else ""
            lines.append(f"  - {plain_name}: {direction_plain}.{change_note}{ticker_note}")
        lines.append(
            "  INSTRUCTION: Mention the energy picture in the Big Picture section only if "
            "something is genuinely notable — a big inventory surprise or a sharp production shift. "
            "Don't list all readings — pick the one that changes the story."
        )
    else:
        lines.append("ENERGY PICTURE: No EIA data available today.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 2: CRYPTO
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 2: CRYPTO ━━━")
    lines.append("(Use this data for the ## CRYPTO section — always include this section)")
    lines.append("")

    # Fear & Greed
    fg = summary.get("crypto_fear_greed", {})
    if fg and fg.get("value") is not None:
        fg_value = int(fg["value"])
        fg_trend = fg.get("trend", "")
        trend_note = f" It's been {fg_trend}." if fg_trend else ""
        if fg_value <= 25:
            fg_plain = "crypto traders are extremely scared right now"
            fg_instruction = (
                "Extreme fear. Say: 'Crypto traders are scared right now — "
                "and historically, that's when the quiet buyers show up.' Calm tone."
            )
        elif fg_value <= 45:
            fg_plain = "crypto traders are worried"
            fg_instruction = "Below-neutral mood. Mention briefly — 'There's some nervousness in crypto.'"
        elif fg_value <= 55:
            fg_plain = "crypto traders are feeling neutral"
            fg_instruction = "Neutral zone. Mention briefly as context, don't dwell on it."
        elif fg_value <= 75:
            fg_plain = "crypto traders are feeling optimistic"
            fg_instruction = "Above-neutral mood. Mention briefly — 'Crypto is feeling good right now.'"
        else:
            fg_plain = "crypto traders are extremely greedy / euphoric"
            fg_instruction = (
                "Extreme greed. Say: 'Everyone in crypto is feeling great right now — "
                "and that's historically when things cool off.' Don't be preachy."
            )
        lines.append(f"CRYPTO MOOD: {fg_plain}.{trend_note}")
        lines.append(f"  INSTRUCTION: {fg_instruction}")
    else:
        lines.append("CRYPTO MOOD: No data available.")

    lines.append("")

    # Major coin prices and movements
    crypto_coins = summary.get("crypto_coins", [])
    if crypto_coins:
        lines.append("MAJOR CRYPTO COINS — HOW THEY'RE MOVING:")
        for coin in crypto_coins:
            sym = coin["symbol"]
            ch24 = coin.get("change_24h_pct")
            ch7d = coin.get("change_7d_pct")
            if ch24 is not None:
                abs_c = abs(ch24)
                direction_ch = "up" if ch24 > 0 else "down"
                if abs_c >= 10:
                    ch24_note = f" sharply {direction_ch} today"
                elif abs_c >= 3:
                    ch24_note = f" noticeably {direction_ch} today"
                elif abs_c >= 0.5:
                    ch24_note = f" slightly {direction_ch} today"
                else:
                    ch24_note = " barely moved today"
            else:
                ch24_note = ""
            if ch7d is not None:
                abs_w = abs(ch7d)
                direction_w = "up" if ch7d > 0 else "down"
                if abs_w >= 20:
                    ch7d_note = f", sharply {direction_w} this week"
                elif abs_w >= 5:
                    ch7d_note = f", noticeably {direction_w} this week"
                else:
                    ch7d_note = ""
            else:
                ch7d_note = ""
            lines.append(f"  - {sym}:{ch24_note}{ch7d_note}.")
        # Are coins moving together or diverging?
        directions = [c.get("direction", "neutral") for c in crypto_coins]
        all_same = len(set(directions)) == 1
        lines.append(
            f"  All major coins are {'moving in the same direction' if all_same else 'moving in different directions'} today."
        )
        lines.append(
            "  INSTRUCTION: For the Crypto section, note whether the whole crypto market is "
            "moving together (all going up or all going down) or diverging (some up, some down). "
            "If BTC is up but smaller coins are down, say 'Bitcoin is climbing but the smaller coins aren't following.' "
            "Don't list every coin — just tell the story of what crypto is doing as a whole."
        )
    else:
        lines.append("CRYPTO COINS: No coin data available.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 3: COMMODITIES & FUTURES
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 3: COMMODITIES & FUTURES ━━━")
    lines.append("(Use this data for the ## COMMODITIES & FUTURES section — skip if flat)")
    lines.append("")

    futures_activity = summary.get("futures_activity", [])
    cot = summary.get("cot_positioning", {})

    if futures_activity:
        lines.append("FUTURES AND FOREX ACTIVITY (only instruments with meaningful signals):")
        for fut in futures_activity:
            direction_plain = _direction_words(fut["dominant_direction"])
            domains_plain = _domain_plain(fut["domains"])
            activity_words = _signal_count_to_words(fut["signal_count"])
            lines.append(
                f"  - {fut['friendly_name']}: "
                f"pointing {direction_plain}. "
                f"Sources: {domains_plain}. ({activity_words} accumulated)"
            )
        lines.append(
            "  INSTRUCTION: For the Commodities & Futures section, focus on what's interesting "
            "about the big-ticket instruments — gold, oil, index futures. "
            "Example: 'Gold and oil are moving in opposite directions today, which is unusual — "
            "they normally track each other when fear is high.' "
            "If nothing is notable, say 'Futures are quiet today' and keep it one line."
        )
    else:
        lines.append("FUTURES/FOREX: No meaningful futures activity to report (below signal threshold).")

    if cot and cot.get("dominant"):
        dominant_cot = cot["dominant"]
        dominant_plain = "upward" if dominant_cot == "bullish" else ("downward" if dominant_cot == "bearish" else "in mixed directions")
        lines.append(
            f"WHAT PROFESSIONAL TRADERS ARE DOING: The biggest traders are mostly betting {dominant_plain} right now."
        )
        lines.append(
            "  INSTRUCTION: If this is strongly one-directional, mention it: "
            "'The biggest professional traders are heavily betting [up/down] right now — "
            "which is either a smart move or a crowded bet that could snap back.' "
            "Skip if mixed or flat."
        )
    else:
        lines.append("WHAT PROFESSIONAL TRADERS ARE DOING: No strong signal from the big traders today.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 4: STOCKS — THE INTERESTING ONES
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 4: STOCKS — THE INTERESTING ONES ━━━")
    lines.append("(Use this data for the ## STOCKS — THE INTERESTING ONES section)")
    lines.append("CRITICAL: Don't lead with ticker symbols. Lead with WHAT IS INTERESTING about them.")
    lines.append("")

    # Paper trades
    n_trades = summary.get("paper_trades_today", 0)
    top_trade = summary.get("top_trade")
    if n_trades > 0 and top_trade:
        direction_plain = _direction_words(top_trade["direction"])
        confidence_plain = _confidence_words(top_trade["confidence"])
        lines.append(
            f"PAPER TRADES PLACED TODAY: {n_trades} trade(s). "
            f"Strongest: {top_trade['asset']} ({direction_plain}, I am {confidence_plain})."
        )
        lines.append(
            "  INSTRUCTION: Include a '## WHAT I THINK YOU SHOULD LOOK AT' section after stocks. "
            "Say 'I placed a paper trade on [TICKER]' in bold. Explain why in one plain sentence. "
            "State the market (US stock, futures, crypto, etc). Give timing and risk estimates."
        )
    else:
        lines.append("PAPER TRADES TODAY: None placed.")

    lines.append("")

    # Inevitabilities (stock convergence) — SHOW THE STACK
    inevitabilities = summary.get("inevitabilities", [])
    if inevitabilities:
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("STACK ANALYSIS — THE MOST INEVITABLE SITUATIONS")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")
        lines.append(
            "CRITICAL INSTRUCTION FOR STOCKS SECTION: For EVERY ticker you mention, you MUST "
            "show the STACK of independent patterns — not just 'X sources agree'. "
            "List each independent signal as a separate bullet. Explain WHY their independence matters. "
            "Include the historical track record if provided. "
            "If you can't show at least 3 independent reasons, DON'T mention the ticker. "
            "The point is not that NVDA looks bearish — it's that 5 UNCONNECTED sources "
            "are all reaching the same conclusion, which is structurally different from coincidence."
        )
        lines.append("")
        for inv in inevitabilities[:3]:
            direction_plain = _direction_words(inv["direction"])
            window = inv.get("expected_window_days")
            window_note = f" Expected timing: within {window} days." if window else ""
            confidence_plain = _confidence_words(inv.get("score", 0))
            lines.append(f"  TICKER: {inv['ticker']} — {direction_plain}.{window_note} I am {confidence_plain}.")
            # Show the pre-built stack
            stack_lines = inv.get("stack_lines", [])
            if stack_lines:
                for sl in stack_lines:
                    lines.append(f"    {sl}")
            else:
                # Fallback: at least show domain list if stack wasn't built
                sources_plain = _domain_plain(inv["domains"])
                lines.append(f"    Evidence from: {sources_plain}.")
            lines.append("")
        lines.append(
            "  HOW TO USE THIS DATA IN THE LETTER:"
            " Lead with the weirdest combination (most unrelated sources agreeing)."
            " Say something like: 'Here is what is strange about [TICKER]: it has [N] completely"
            " independent signals all pointing the same direction. [Source 1] noticed X."
            " [Source 2], which has nothing to do with [Source 1], noticed Y."
            " [Source 3] noticed Z. When unconnected sources agree, that's not coincidence —"
            " that's a stack.' Then say: 'Based on history, this combination has worked [rate].'"
            " Max 3 tickers. Skip tickers with only chart patterns (technical alone)."
        )
    else:
        lines.append("INEVITABLE SITUATIONS: Nothing stands out strongly in stocks today.")

    lines.append("")

    # Developing situations — also show the stack
    devs = summary.get("developing", [])
    if devs:
        lines.append("DEVELOPING STOCK SITUATIONS (what I'm actively watching — partial stacks):")
        for d in devs:
            direction_plain = _direction_words(d["direction"])
            confidence_plain = _confidence_words(d["confidence"])
            lines.append(
                f"  - {d['ticker']}: {direction_plain}. I am {confidence_plain}."
            )
            stack_lines = d.get("stack_lines", [])
            if stack_lines:
                for sl in stack_lines:
                    lines.append(f"      {sl}")
            else:
                sources_plain = _domain_plain(d["domains"])
                lines.append(f"      Evidence so far: {sources_plain}.")
            if d.get("summary"):
                lines.append(f"      Context: {d['summary'][:120]}")
            lines.append("")
    else:
        lines.append("DEVELOPING STOCK SITUATIONS: None with strong evidence right now.")

    lines.append("")

    # Cascade chains (unfolding domino effects)
    cascade = summary.get("cascade_status", {})
    notable_chains = cascade.get("notable_chains", [])
    if notable_chains:
        lines.append("CAUSAL CHAINS UNFOLDING (domino effects in progress — goes in the Stocks section):")
        for ch in notable_chains:
            confirmed = ch.get("confirmed_count", 0)
            total = ch.get("total_links", 0)
            trigger = ch.get("trigger", "?")
            direction_plain = _direction_words(ch.get("direction", "?"))
            confirmed_tickers = ch.get("confirmed_tickers", [])
            next_dominoes = ch.get("next_dominoes", [])
            energy = ch.get("energy_ratio")
            energy_note = ""
            if energy is not None:
                if energy > 1.1:
                    energy_note = " The chain is moving faster than expected."
                elif energy < 0.9:
                    energy_note = " The chain is moving slower than expected."
            next_note = f" Watching for: {', '.join(next_dominoes)}." if next_dominoes else ""
            confirmed_note = f" Already confirmed: {', '.join(confirmed_tickers)}." if confirmed_tickers else ""
            lines.append(
                f"  - Chain starting from {trigger} ({direction_plain}): "
                f"{confirmed} of {total} dominoes confirmed.{confirmed_note}{next_note}{energy_note}"
            )
        lines.append(
            "  INSTRUCTION: Domino chains are the most dramatic thing MIDGE can report in stocks. "
            "Use: 'I predicted A would trigger B would trigger C. A happened. B just happened. "
            "Now I'm watching for C.' If energy ratio > 1, say 'the chain is accelerating.'"
        )
    elif cascade.get("active_chains", 0) > 0:
        lines.append(
            f"CAUSAL CHAINS: {cascade['active_chains']} active chains tracking — "
            "none have confirmed dominoes yet."
        )

    lines.append("")

    # Developing investigations (partial convergences)
    dev_sits = summary.get("developing_situations", [])
    if dev_sits:
        lines.append("STOCK SITUATIONS I'M STILL INVESTIGATING (not ready yet):")
        for s in dev_sits[:3]:
            domains_seen_plain = _domain_plain(s["domains_seen"])
            missing_plain = _domain_plain(s["missing_domains"]) if s.get("missing_domains") else ""
            direction_plain = _direction_words(s["direction"])
            missing_note = f" Missing: {missing_plain}." if missing_plain else ""
            lines.append(
                f"  - {s['ticker']}: {direction_plain}. "
                f"Evidence so far: {domains_seen_plain}.{missing_note}"
            )
        lines.append(
            "  INSTRUCTION: These are 'something is starting to form' situations. "
            "Only mention 1-2 in the Stocks section if they're genuinely interesting. "
            "Use 'I'm watching' or 'I noticed something early' language."
        )

    lines.append("")

    # Somatic building (pre-convergence)
    somatic = summary.get("somatic_building", [])
    # Filter to stocks only (exclude crypto and futures symbols)
    _crypto_syms = {"BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "DOGE", "AVAX"}
    _futures_syms = {"GC=F", "CL=F", "NQ=F", "ES=F", "EURUSD=X", "GBPUSD=X", "USDJPY=X"}
    somatic_stocks = [
        s for s in somatic
        if s["ticker"] not in _crypto_syms and s["ticker"] not in _futures_syms
    ]
    if somatic_stocks:
        top_s = somatic_stocks[0]
        direction_plain = _direction_words(top_s["dominant_direction"])
        domain_plain_txt = _domain_plain(top_s["domains"])
        lines.append(
            f"BUILDING SIGNALS (not a call yet): {top_s['ticker']} has "
            f"{top_s['domain_count']} information sources pointing {direction_plain} "
            f"({domain_plain_txt}). {top_s['signal_count']} signals accumulated."
        )
        lines.append(
            "  INSTRUCTION: Only use this if the ticker is genuinely interesting. "
            "Use 'my attention keeps coming back to' or 'something is forming but I'm not ready to call it yet.' "
            "Do NOT list all of them — just the most interesting one."
        )

    lines.append("")

    # Recent alerts follow-up
    today_alerts = summary.get("recent_alerts", [])
    if today_alerts:
        tickers_sent = [a["ticker"] for a in today_alerts]
        lines.append(
            f"ALERTS I SENT TODAY ({len(today_alerts)} alert(s) about "
            f"{', '.join(tickers_sent[:6])}):"
        )
        for a in today_alerts:
            direction_plain = _direction_words(a["direction"])
            confidence_plain = _confidence_words(a["confidence"])
            lines.append(f"  - {a['ticker']}: I said it {direction_plain}. I was {confidence_plain}.")
        lines.append(
            "  INSTRUCTION: If you sent alerts today, put a brief follow-up in the Stocks section. "
            "'I sent you [N] alerts today. Here's what happened since.' "
            "Cross-reference with outcomes data if available."
        )
    else:
        lines.append("ALERTS SENT TODAY: None.")

    lines.append("")

    # Outcomes
    oc = summary.get("outcomes", {})
    total = oc.get("total", 0)
    wins = oc.get("wins", 0)
    losses = oc.get("losses", 0)
    if total > 0:
        if wins >= losses:
            track_note = "more right than wrong lately" if wins > losses else "about even"
        else:
            track_note = "more wrong than right lately — still learning"
        lines.append(
            f"HOW MY CALLS HAVE BEEN DOING (last 7 days): "
            f"{wins} right, {losses} wrong — {track_note}."
        )
        if oc.get("win_examples"):
            win_tickers = ", ".join(e["symbol"] for e in oc["win_examples"])
            lines.append(f"  Called correctly: {win_tickers} — all moved the direction I expected.")
        if oc.get("loss_examples"):
            loss_tickers = ", ".join(e["symbol"] for e in oc["loss_examples"])
            lines.append(f"  Got wrong: {loss_tickers} — moved the other way.")
    else:
        lines.append("HOW MY CALLS HAVE BEEN DOING: Nothing checked yet this week.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 5: WHAT I LEARNED / WHAT I GOT WRONG
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 5: WHAT I LEARNED / WHAT I GOT WRONG ━━━")
    lines.append("")

    # Post-mortem learning
    pm = summary.get("postmortem", {})
    if pm and pm.get("total_graded", 0) > 0:
        pm_wr_raw = pm.get("overall_win_rate_pct", "")
        pm_grade = pm.get("grade", "")
        pm_graded = pm.get("total_graded", 0)
        # Convert raw win rate (may be "29.4%" string or float) to words
        try:
            pm_wr_float = float(str(pm_wr_raw).replace("%", "").strip())
            pm_wr_words = "more often than not" if pm_wr_float >= 50 else "still learning"
            if pm_wr_float >= 70:
                pm_wr_words = "most of the time"
            elif pm_wr_float < 30:
                pm_wr_words = "not often enough yet"
        except (ValueError, TypeError):
            pm_wr_words = "some of the time"
        lines.append(
            f"OVERALL HOW I'VE BEEN DOING: My calls have been right {pm_wr_words} "
            f"(checked {pm_graded} total, grade: {pm_grade})."
        )
        if pm.get("best_combos"):
            lines.append(
                f"  Best combinations: {'; '.join(pm['best_combos'][:2])}"
            )
            lines.append(
                "  INSTRUCTION: Translate combination keys to plain language. "
                "'When company news, economic data, and price patterns all agree' — "
                "never the raw source names."
            )
        if pm.get("timing_insight"):
            lines.append(f"  Timing note: {pm['timing_insight']}")
    else:
        lines.append("OVERALL HOW I'VE BEEN DOING: Not enough checked calls to say yet.")

    lines.append("")

    # Active hypotheses
    active_hyps = summary.get("active_hypotheses", [])
    if active_hyps:
        lines.append("PATTERNS I'M TESTING (translate these to plain English — never use their raw names):")
        for h in active_hyps[:3]:
            wr = h.get("win_rate_pct")
            n_obs = h.get("observations", 0)
            if wr is not None:
                wr_words = "more often than not" if wr >= 50 else "sometimes"
                if wr >= 75:
                    wr_words = "most of the time"
                wr_note = f" Has worked {wr_words} ({n_obs} times checked)."
            else:
                wr_note = ""
            story = h.get("causal_story", "")[:150]
            lines.append(f"  - {story}{wr_note}")
        lines.append(
            "  INSTRUCTION: Put the most interesting theory in 'What I Learned'. "
            "Translate to plain English. 'I have a theory: when [plain description] happens, "
            "[outcome] tends to follow [N] days later.' Never use indicator names or system names."
        )

    lines.append("")

    # Source reliability
    th = summary.get("thompson", {})
    trusted = th.get("trusted", [])
    distrusted = th.get("distrusted", [])
    if trusted or distrusted:
        if trusted:
            n_trusted = len(trusted)
            lines.append(
                f"SOURCE RELIABILITY: {_signal_count_to_words(n_trusted).replace(' signals', ' information source(s)')} "
                f"have been right more often than not lately."
            )
        if distrusted:
            n_distrusted = len(distrusted)
            lines.append(
                f"  {_signal_count_to_words(n_distrusted).replace(' signals', ' source(s)')} haven't been reliable lately — "
                "I'm trusting them less right now."
            )
        lines.append(
            "  INSTRUCTION: Describe sources by what they measure, not their names. "
            "'insider buying reports' not 'sec_form4'. Goes in 'What I Learned'."
        )

    lines.append("")
    lines.append(
        "Now write the daily letter following the LAYERED STRUCTURE precisely: "
        "Big Picture → Crypto → Commodities & Futures → Stocks → Learned → Wrong. "
        "Bold the punch lines, use bullets, no jargon — remember the 12-year-old rule. "
        "Under 600 words. Sign it '— MIDGE'."
    )

    # Translate every data line so the LLM never sees raw jargon or field names
    translated_lines = [_translate_jargon(line) for line in lines]
    return "\n".join(translated_lines)


def _template_narrative(summary: dict, date_str: str) -> str:
    """Template fallback when no LLM is available.

    Short, punchy, jargon-free. Layered structure: Big Picture → Crypto →
    Commodities & Futures → Stocks → Learned → Wrong.
    """
    lines = [
        f"Subject: MIDGE Daily Letter — {date_str}",
        "",
    ]

    # ── Lead hook — Granger weirdness or regime first ─────────────
    granger = summary.get("granger", [])
    if granger:
        g = granger[0]
        lines.append(f"**Here's something strange I noticed: {g['story']}**")
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # THE BIG PICTURE
    # ══════════════════════════════════════════════════════════════
    lines.append("## THE BIG PICTURE")
    lines.append("")

    # Regime
    regime = summary.get("regime", "unknown")
    regime_plain = _regime_plain(regime)
    lines.append(f"- **Right now, {regime_plain}.**")

    # Macro alignment
    macro_align = summary.get("macro_alignment", {})
    if macro_align and macro_align.get("dominant"):
        dominant = macro_align["dominant"]
        divergent = macro_align.get("divergent", False)
        if divergent:
            lines.append(
                "- Economic signals are pulling in different directions — they're not agreeing right now."
            )
        else:
            dominant_plain = "up" if dominant == "bullish" else ("down" if dominant == "bearish" else dominant)
            lines.append(
                f"- **Big-picture economic signals are mostly pointing {dominant_plain}.**"
            )

    # Key macro indicators (most deviated)
    macro_indicators = summary.get("macro_indicators", [])
    _MACRO_PLAIN_TMPL = {
        "T10Y2Y": "a key recession warning signal",
        "T10Y3M": "a key recession warning signal",
        "T5YIE": "where the market thinks prices will be in five years",
        "DGS2": "short-term government borrowing costs",
        "DGS10": "long-term government borrowing costs",
        "DFF": "the overnight rate banks charge each other",
        "VIXCLS": "the market's fear reading",
    }
    for ind in macro_indicators[:2]:
        plain_name = _MACRO_PLAIN_TMPL.get(ind["series_id"], ind.get("name", ind["series_id"]))
        direction = ind.get("direction", "neutral")
        direction_plain = "up" if direction == "bullish" else ("down" if direction == "bearish" else "neutral")
        lines.append(f"- {plain_name}: pointing **{direction_plain}**.")

    # Cross-market anomalies
    cross_market = summary.get("cross_market_anomalies", [])
    if cross_market:
        top_cm = cross_market[0]
        tickers_note = f" ({', '.join(top_cm['tickers'][:4])})" if top_cm.get("tickers") else ""
        lines.append(
            f"- *Something weird across markets: "
            f"{top_cm.get('description', top_cm.get('type', 'unusual pattern'))}"
            f"{tickers_note}. When unrelated markets move together, something is flowing underneath.*"
        )

    # Energy picture
    energy_readings = summary.get("energy_readings", [])
    if energy_readings:
        _ENERGY_PLAIN_TMPL = {
            "crude_production": "US oil production",
            "crude_inventory": "US oil stockpiles",
            "natural_gas_storage": "US natural gas storage",
        }
        notable_energy = [er for er in energy_readings if er.get("direction") != "neutral"]
        if notable_energy:
            er = notable_energy[0]
            plain_name = _ENERGY_PLAIN_TMPL.get(er["series_key"], er.get("name", er["series_key"]))
            direction = er.get("direction", "neutral")
            direction_plain = "rising" if direction == "bullish" else ("falling" if direction == "bearish" else direction)
            change = er.get("change_pct")
            if change is not None:
                abs_c = abs(change)
                if abs_c >= 10:
                    change_note = " — a big shift"
                elif abs_c >= 3:
                    change_note = " — a noticeable shift"
                else:
                    change_note = " — a small shift"
            else:
                change_note = ""
            lines.append(f"- Energy: {plain_name} is **{direction_plain}**{change_note}.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # CRYPTO
    # ══════════════════════════════════════════════════════════════
    lines.append("## CRYPTO")
    lines.append("")

    fg = summary.get("crypto_fear_greed", {})
    if fg and fg.get("value") is not None:
        fg_value = int(fg["value"])
        fg_trend = fg.get("trend", "")
        trend_note = f" It's been {fg_trend}." if fg_trend else ""
        if fg_value <= 25:
            lines.append(f"- **Crypto traders are extremely scared right now.**{trend_note}")
            lines.append("- *Historically this is when the quiet buyers step in.*")
        elif fg_value <= 45:
            lines.append(f"- Crypto traders are feeling a bit nervous.{trend_note}")
        elif fg_value <= 55:
            lines.append(f"- Crypto mood is neutral — nothing unusual.{trend_note}")
        elif fg_value <= 75:
            lines.append(f"- Crypto traders are feeling optimistic.{trend_note}")
        else:
            lines.append(f"- **Everyone in crypto is feeling great right now.**{trend_note}")
            lines.append("- *Historically this is when things cool off.*")
    else:
        lines.append("- No crypto sentiment data available.")

    # Major coin movements
    crypto_coins = summary.get("crypto_coins", [])
    if crypto_coins:
        # Summarize as a group
        rising = [c for c in crypto_coins if (c.get("change_24h_pct") or 0) > 1.0]
        falling = [c for c in crypto_coins if (c.get("change_24h_pct") or 0) < -1.0]
        if len(rising) > len(falling):
            movers = ", ".join(c["symbol"] for c in rising[:4])
            lines.append(f"- Major coins mostly rising today: **{movers}** all up in the last 24 hours.")
        elif len(falling) > len(rising):
            movers = ", ".join(c["symbol"] for c in falling[:4])
            lines.append(f"- Major coins mostly falling today: **{movers}** all down in the last 24 hours.")
        else:
            lines.append("- Crypto is mixed — some coins up, some down. No clear trend.")

        # BTC vs altcoins divergence check
        btc = next((c for c in crypto_coins if c["symbol"] == "BTC"), None)
        alts = [c for c in crypto_coins if c["symbol"] != "BTC"]
        if btc and alts:
            btc_ch = btc.get("change_24h_pct") or 0
            alt_avg = sum(c.get("change_24h_pct") or 0 for c in alts) / len(alts)
            if btc_ch > 1 and alt_avg < 0:
                lines.append("- *Bitcoin is climbing but smaller coins aren't following — a divergence worth watching.*")
            elif btc_ch < -1 and alt_avg > 1:
                lines.append("- *Bitcoin is falling while smaller coins are rising — an unusual split.*")

    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # COMMODITIES & FUTURES
    # ══════════════════════════════════════════════════════════════
    futures_activity = summary.get("futures_activity", [])
    cot = summary.get("cot_positioning", {})

    if futures_activity or (cot and cot.get("dominant") and cot.get("dominant") != "mixed"):
        lines.append("## COMMODITIES & FUTURES")
        lines.append("")

        if futures_activity:
            for fut in futures_activity:
                direction_plain = _direction_words(fut["dominant_direction"])
                domains_brief = ', '.join(_domain_plain(fut["domains"]).split(', ')[:2])
                lines.append(
                    f"- **{fut['friendly_name']}**: {direction_plain}. "
                    f"(based on {domains_brief})"
                )

        if cot and cot.get("dominant") and cot.get("dominant") != "mixed":
            dominant_cot = cot["dominant"]
            dominant_plain = "upward" if dominant_cot == "bullish" else ("downward" if dominant_cot == "bearish" else dominant_cot)
            lines.append(
                f"- **The biggest professional traders are mostly betting {dominant_plain} right now.**"
            )
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # STOCKS — THE INTERESTING ONES
    # ══════════════════════════════════════════════════════════════
    lines.append("## STOCKS — THE INTERESTING ONES")
    lines.append("")

    # Lead with inevitabilities (most converged)
    inevitabilities = summary.get("inevitabilities", [])
    devs = summary.get("developing", [])
    cascade = summary.get("cascade_status", {})
    notable_chains = cascade.get("notable_chains", [])

    # Cascade chains first (most dramatic)
    for _ch in notable_chains[:1]:
        _confirmed = _ch.get("confirmed_count", 0)
        _total = _ch.get("total_links", 0)
        _trigger = _ch.get("trigger", "?")
        _next = _ch.get("next_dominoes", [])
        if _confirmed > 0 and _total > 1:
            _next_note = f" Watching for: {', '.join(_next[:2])}." if _next else ""
            lines.append(
                f"- **A chain starting from {_trigger} is unfolding: "
                f"{_confirmed} of {_total} dominoes confirmed.**{_next_note}"
            )
            lines.append("")

    if inevitabilities:
        for inv in inevitabilities[:3]:
            direction_plain = _direction_words(inv["direction"])
            confidence_plain = _confidence_words(inv["score"])
            sources_plain = _domain_plain(inv["domains"])
            window = inv.get("expected_window_days")
            window_note = f" (timing: ~{window} days)" if window else ""
            chain_raw = inv.get("world_model_chain", "")
            chain_note = ""
            if chain_raw and chain_raw != "None":
                _chain_clean_tmpl = chain_raw.strip("[]").replace("'", "")
                chain_note = f"\n- Connected through: {_chain_clean_tmpl}"
            lines.append(f"**{inv['ticker']}** — {direction_plain}.{window_note}")
            lines.append(f"- I am {confidence_plain}.")
            lines.append(f"- What's interesting: evidence from {sources_plain} all converging.{chain_note}")
            lines.append("")
    elif devs:
        for d in devs[:3]:
            direction_plain = _direction_words(d["direction"])
            confidence_plain = _confidence_words(d["confidence"])
            sources_plain = _domain_plain(d["domains"])
            lines.append(f"**{d['ticker']}** — {direction_plain}.")
            lines.append(f"- I am {confidence_plain}.")
            lines.append(f"- Evidence coming from: {sources_plain}.")
            lines.append("")
    else:
        lines.append("Nothing notable in stocks today. Watching broadly.")
        lines.append("")

    # Developing investigations
    dev_sits = summary.get("developing_situations", [])
    if dev_sits:
        lines.append("**Also watching (not ready yet):**")
        for s in dev_sits[:2]:
            direction_plain = _direction_words(s["direction"])
            domains_seen_plain = _domain_plain(s["domains_seen"])
            missing_plain = _domain_plain(s.get("missing_domains", []))
            missing_note = f" Waiting for: {missing_plain}." if missing_plain else ""
            lines.append(
                f"- {s['ticker']}: {direction_plain}. {domains_seen_plain} seen.{missing_note}"
            )
        lines.append("")

    # Somatic building (stocks only)
    somatic = summary.get("somatic_building", [])
    _crypto_syms_tmpl = {"BTC", "ETH", "SOL", "XRP", "ADA", "BNB", "DOGE", "AVAX"}
    _futures_syms_tmpl = {"GC=F", "CL=F", "NQ=F", "ES=F", "EURUSD=X", "GBPUSD=X", "USDJPY=X"}
    somatic_stocks = [
        s for s in somatic
        if s["ticker"] not in _crypto_syms_tmpl and s["ticker"] not in _futures_syms_tmpl
    ]
    if somatic_stocks:
        top_s = somatic_stocks[0]
        direction_plain = _direction_words(top_s["dominant_direction"])
        domain_plain_txt = _domain_plain(top_s["domains"])
        lines.append(
            f"*My attention keeps going back to {top_s['ticker']} — "
            f"{top_s['domain_count']} sources pointing {direction_plain} "
            f"({domain_plain_txt}). Not ready to call it yet.*"
        )
        lines.append("")

    # Alerts sent today
    today_alerts = summary.get("recent_alerts", [])
    if today_alerts:
        tickers_sent = [a["ticker"] for a in today_alerts]
        lines.append(
            f"*I sent you {len(today_alerts)} alert(s) today "
            f"about {', '.join(tickers_sent[:6])}. Checking back on those moves.*"
        )
        lines.append("")

    # ══════════════════════════════════════════════════════════════
    # WHAT I LEARNED
    # ══════════════════════════════════════════════════════════════
    lines.append("## WHAT I LEARNED")
    lines.append("")
    pm = summary.get("postmortem", {})
    th = summary.get("thompson", {})
    learned_any = False

    if granger and len(granger) > 1:
        for g in granger[1:3]:
            lines.append(f"- **{g['story']}**")
            learned_any = True

    pm_wr_raw = pm.get("overall_win_rate_pct", "")
    pm_graded = pm.get("total_graded", 0)
    if pm_wr_raw and pm_graded > 0:
        try:
            pm_wr_float = float(str(pm_wr_raw).replace("%", "").strip())
            pm_wr_words = "more often than not" if pm_wr_float >= 50 else "still getting better"
            if pm_wr_float >= 70:
                pm_wr_words = "most of the time"
            elif pm_wr_float < 30:
                pm_wr_words = "not as often as I'd like yet"
        except (ValueError, TypeError):
            pm_wr_words = "some of the time"
        lines.append(
            f"- **My calls have been right {pm_wr_words}** ({pm_graded} checked so far)."
        )
        learned_any = True

    trusted = th.get("trusted", [])
    distrusted = th.get("distrusted", [])
    if trusted:
        lines.append(
            f"- **{len(trusted)} information source(s) are proving consistently reliable** "
            "based on what I've seen so far."
        )
        learned_any = True
    if distrusted:
        lines.append(
            "- Some of my information sources haven't been reliable lately — I'm trusting them less right now."
        )
        learned_any = True

    # Cascade learning
    for _ch in notable_chains[:1]:
        _confirmed = _ch.get("confirmed_count", 0)
        _total = _ch.get("total_links", 0)
        _trigger = _ch.get("trigger", "?")
        _next = _ch.get("next_dominoes", [])
        if _confirmed > 0 and _total > 1:
            _next_note = f" Watching for: {', '.join(_next[:2])}." if _next else ""
            lines.append(
                f"- **A chain I predicted from {_trigger} has {_confirmed} of {_total} dominoes confirmed.**{_next_note}"
            )
            learned_any = True

    # Best active hypothesis
    active_hyps = summary.get("active_hypotheses", [])
    best_hyp = next(
        (h for h in active_hyps if h["status"] == "active" and h.get("win_rate_pct") and h["win_rate_pct"] > 50),
        None,
    )
    if best_hyp:
        wr = best_hyp["win_rate_pct"]
        wr_words = "more often than not" if wr >= 50 else "sometimes"
        if wr >= 75:
            wr_words = "most of the time"
        lines.append(
            f"- **A theory I've been testing is working {wr_words} — "
            f"checked {best_hyp['observations']} times so far.**"
        )
        learned_any = True

    if not learned_any:
        lines.append("- Not enough graded data yet to say something concrete.")
    lines.append("")

    # ══════════════════════════════════════════════════════════════
    # WHAT I GOT WRONG
    # ══════════════════════════════════════════════════════════════
    lines.append("## WHAT I GOT WRONG")
    lines.append("")
    oc = summary.get("outcomes", {})
    loss_examples = oc.get("loss_examples", [])
    wins = oc.get("wins", 0)
    losses = oc.get("losses", 0)
    total = oc.get("total", 0)
    if total > 0:
        lines.append(f"I checked {total} recent calls — **{wins} right, {losses} wrong.**")
        if loss_examples:
            for e in loss_examples[:2]:
                lines.append(
                    f"- **{e['symbol']}**: I thought it would move the other way — it didn't."
                )
        # ── Failure explanation layer ──────────────────────────────────
        try:
            import json as _json
            from pathlib import Path as _Path
            _fsummary_path = _Path("data/midge/failure_summary.json")
            if _fsummary_path.exists():
                with open(_fsummary_path, "r", encoding="utf-8") as _f:
                    _fsummary = _json.load(_f)
                _top = _fsummary.get("top_category", "")
                _counts = _fsummary.get("category_counts", {})
                _examples = _fsummary.get("category_examples", {})
                _total_explained = _fsummary.get("total_explained", 0)
                if _top and _total_explained > 0:
                    _top_pct = round(_counts.get(_top, 0) / _total_explained * 100)
                    _top_example = _examples.get(_top, "")
                    _top_label = _top.replace("_", " ")
                    lines.append(
                        f"\n**Most common failure reason ({_top_pct}% of losses): {_top_label}.**"
                    )
                    if _top_example:
                        lines.append(f"Example: {_top_example[:100]}")
                    if len(_counts) > 1:
                        _ranked = sorted(_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                        lines.append(
                            "Breakdown: "
                            + " | ".join(
                                f"{cat.replace('_', ' ')} ({cnt})"
                                for cat, cnt in _ranked
                            )
                        )
        except Exception:
            pass
    else:
        lines.append("Nothing graded yet — calls are still within their evaluation window.")
    lines.append("")

    # ── PAPER TRADE RECOMMENDATION (when applicable) ──────────────
    n_trades = summary.get("paper_trades_today", 0)
    top_trade = summary.get("top_trade")
    if n_trades > 0 and top_trade:
        lines.append("## WHAT I THINK YOU SHOULD LOOK AT")
        lines.append("")
        direction_plain = _direction_words(top_trade["direction"])
        confidence_plain = _confidence_words(top_trade["confidence"])
        asset = top_trade["asset"]
        lines.append(f"**I placed a paper trade on {asset}.**")
        lines.append(f"- It {direction_plain} based on multiple independent signals.")
        lines.append(f"- I am {confidence_plain}.")
        lines.append("- This is a US stock (paper trading active via Alpaca).")
        lines.append("")

    # ── Footer ────────────────────────────────────────────────────
    lines.append("---")
    lines.append(
        "*This is what I see, not financial advice. Do your own research.*"
    )
    lines.append("")
    lines.append(
        "**What I'm watching:** Stocks (US markets, paper trading active) · "
        "Crypto (24/7) · Futures and forex (watching, not yet trading) · "
        "Prediction markets (coming soon)"
    )
    lines.append("")
    lines.append("— MIDGE")

    return "\n".join(lines)


# ── Main Entry Point ─────────────────────────────────────────────────


def generate_daily_narrative(date_str: Optional[str] = None) -> str:
    """Generate the daily narrative letter for Guiding Light.

    Args:
        date_str: Optional date string (YYYY-MM-DD). Defaults to today.

    Returns:
        Full letter text as a string. Never raises — returns error message on failure.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        # Gather data
        summary = _gather_data(date_str)

        # Try LLM narrative first
        groq_key = os.environ.get("MAE_GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        narrative_body: Optional[str] = None

        if groq_key:
            prompt = _build_llm_prompt(summary)
            narrative_body = _call_groq(prompt, groq_key)
            if narrative_body:
                # Ensure the letter has the subject header
                if not narrative_body.startswith("Subject:"):
                    narrative_body = (
                        f"Subject: MIDGE Daily Letter — {date_str}\n\n"
                        + narrative_body
                    )
                # Post-generation safety net: catch any jargon the LLM still wrote
                narrative_body = _translate_jargon(narrative_body)
                logger.info("Daily narrative generated via Groq (%d chars)", len(narrative_body))
            else:
                logger.info("Groq call returned nothing — falling back to template")

        if not narrative_body:
            # Template fallback
            narrative_body = _template_narrative(summary, date_str)
            # Apply jargon scrubber to template output too
            narrative_body = _translate_jargon(narrative_body)
            logger.info("Daily narrative generated via template (%d chars)", len(narrative_body))

        # Archive to daily_narratives/
        try:
            _NARRATIVES_DIR.mkdir(parents=True, exist_ok=True)
            archive_path = _NARRATIVES_DIR / f"{date_str}.md"
            with open(archive_path, "w", encoding="utf-8") as f:
                f.write(narrative_body)
            logger.info("Daily narrative archived to %s", archive_path)
        except Exception:
            logger.debug("Could not archive narrative to file", exc_info=True)

        # Narrative feedback loop — extract insights and feed back into learning systems
        try:
            from mae_core.market.intelligence.narrative_feedback import NarrativeFeedback
            _nf = NarrativeFeedback()  # No live systems at generation time; persists to JSONL
            _insights = _nf.extract_insights(narrative_body)
            _nf.feed_back(_insights)
        except Exception:
            logger.debug("Narrative feedback extraction failed", exc_info=True)

        return narrative_body

    except Exception:
        logger.error("Daily narrative generation failed", exc_info=True)
        return (
            f"Subject: MIDGE Daily Letter — {date_str}\n\n"
            "Good morning.\n\n"
            "I ran into a problem generating today's letter. "
            "My logs have the details. I'm still watching.\n\n"
            "— MIDGE"
        )
