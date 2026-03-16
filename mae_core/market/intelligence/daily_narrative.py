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
  - >80% → "I'm very confident" or "This looks inevitable"
  - 60-80% → "I'm fairly sure" or "The evidence is building"
  - 45-60% → "Something is forming but I need more"
  - <45% → "I noticed something but it's early"

WHAT'S INTERESTING (in this order):
1. Wild connections across unrelated domains — agriculture → defense → congressional trades. \
The weirder, the more prominent.
2. Things building slowly over days — "I first noticed this Tuesday. By Thursday a second signal \
appeared. Today a third."
3. What MIDGE learned from being wrong.
4. Causal chains confirming — "I predicted A would cause B. A happened Monday. B happened today."
5. New causal discoveries — "When big institutions make moves, insiders start buying the same \
stocks 4 days later. Like clockwork."

WHAT'S NOT INTERESTING:
- Raw numbers ("confidence 0.72")
- Source names ("sec_form4" → say "insider buying reports")
- Technical indicators ("RSI dropped below 30" → say "price dropped sharply and unusually")
- System internals (never mention what your components are called)

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
What regime are we in (bull / bear / volatile / sideways)? \
What are macro signals saying — are they aligned or contradicting each other? \
Any cross-market anomalies (unrelated markets moving together)? \
What's the energy picture (oil inventories, production)? \
2-4 bullets. Lead with the most unusual macro observation.

## CRYPTO
Always present — MIDGE watches crypto 24/7. \
What's the fear/greed reading and what does it mean from a contrarian angle? \
Are the major coins (BTC, ETH, SOL etc) moving in the same direction or diverging? \
Any crypto-specific pattern forming? \
2-3 bullets. Skip if absolutely nothing to say (say "Nothing unusual in crypto today").

## COMMODITIES & FUTURES
Oil, gold, index futures (S&P, Nasdaq), forex pairs. \
COT positioning shifts (are big traders piling in or bailing out?). \
Only mention if something is interesting — skip the section if flat. \
2-3 bullets max.

## STOCKS — THE INTERESTING ONES
NOT a ticker list. Lead with the WEIRDEST convergence. Tell the story. \
"What's strange: three completely unrelated signals all pointing at the same stock." \
"The interesting part isn't the ticker — it's how this connected to that." \
3 situations max. Focus on the WHY, not just the ticker and direction. \
If a causal chain is unfolding (domino effects), this is where it belongs. \
If paper trades were placed, say so here.

## WHAT I LEARNED
2-3 bullets. One sentence each. \
If you found a weird causal relationship between domains, put it here in bold. \
Granger findings go here. Source reliability updates go here. \
"When X happens, Y tends to follow N days later" — that's the gold.

## WHAT I GOT WRONG
1-2 items. Direct and honest. "I thought X. I was wrong. Here's why."

If you have a paper trade recommendation, add before the footer:

## WHAT I THINK YOU SHOULD LOOK AT
For each recommendation:
- **"I placed a paper trade on [TICKER]"** or **"I think you should look at [direction] [TICKER]"**
- One sentence on WHY: "because [plain English reason]"
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
- The WORST letter makes Guiding Light say: "Why is this just a list of stock tickers?"
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
    if postmortem and isinstance(postmortem, dict):
        overall = postmortem.get("overall", {})
        best_combos = postmortem.get("top_5_best_combos", [])[:3]
        worst_combos = postmortem.get("top_5_worst_combos", [])[:2]
        timing = postmortem.get("timing", {})
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
        summary["granger"] = [
            {
                "story": (
                    f"{f['cause_domain']} leads {f['effect_domain']} "
                    f"by ~{f.get('best_lag', '?')} days "
                    f"(p<{f.get('p_value', 1):.3f})"
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
        summary["inevitabilities"] = [
            {
                "ticker": r.get("ticker", "?"),
                "direction": r.get("direction", "?"),
                "score": round(float(r.get("score", 0)) * 100),
                "domains": r.get("domains", []),
                "evidence_summary": r.get("evidence_summary", ""),
                "world_model_chain": r.get("world_model_chain", ""),
                "expected_window_days": r.get("expected_window_days"),
                "signal_count": r.get("signal_count", 0),
            }
            for r in inv_records
        ]
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
        "bull": "bullish (things are generally rising)",
        "bear": "bearish (things are generally falling)",
        "volatile": "volatile (big swings, hard to read)",
        "sideways": "sideways (not much happening)",
    }.get(regime.lower() if regime else "", regime or "unclear")


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
    lines.append(f"CURRENT MARKET REGIME: {regime_plain}.")

    # Macro alignment
    macro_align = summary.get("macro_alignment", {})
    if macro_align:
        dominant = macro_align.get("dominant", "mixed")
        bull_c = macro_align.get("bullish_count", 0)
        bear_c = macro_align.get("bearish_count", 0)
        divergent = macro_align.get("divergent", False)
        align_note = "diverging in different directions" if divergent else f"mostly pointing {dominant}"
        lines.append(
            f"MACRO SIGNALS: {bull_c} signals pointing up, {bear_c} pointing down — {align_note}."
        )

    # Key macro indicators
    macro_indicators = summary.get("macro_indicators", [])
    _MACRO_PLAIN = {
        "T10Y2Y": "the gap between 10-year and 2-year US government bond yields",
        "T10Y3M": "the gap between 10-year and 3-month US bond yields (recession signal)",
        "T5YIE": "what the market expects inflation to be over the next 5 years",
        "DGS2": "2-year US government bond interest rate",
        "DGS10": "10-year US government bond interest rate",
        "DFF": "the US Federal Reserve's overnight interest rate",
        "VIXCLS": "the stock market's fear gauge (VIX)",
    }
    if macro_indicators:
        lines.append("KEY ECONOMIC READINGS:")
        for ind in macro_indicators[:4]:
            plain_name = _MACRO_PLAIN.get(ind["series_id"], ind.get("name", ind["series_id"]))
            value = ind.get("value")
            direction = ind.get("direction", "neutral")
            value_note = f" Currently at {value:.2f}." if value is not None else ""
            lines.append(f"  - {plain_name}: pointing {direction}.{value_note}")

    lines.append(
        "  INSTRUCTION for Big Picture: Lead with what the regime means in plain English. "
        "Are macro signals agreeing with the regime or fighting it? "
        "Example: 'We're in a bear market right now — and most of my economic signals agree. "
        "Bond yields are behaving like investors are worried.' "
        "Keep this section 2-4 bullets. Don't go into individual stocks yet."
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
            change_note = f" (changed {change:+.1f}%)" if change is not None else ""
            direction = er.get("direction", "neutral")
            tickers = er.get("affected_tickers", [])
            ticker_note = f" Affects: {', '.join(tickers)}." if tickers else ""
            lines.append(f"  - {plain_name}: {direction}.{change_note}{ticker_note}")
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
        fg_class = fg.get("classification", "")
        fg_trend = fg.get("trend", "")
        trend_note = f" It's been {fg_trend}." if fg_trend else ""
        lines.append(
            f"CRYPTO FEAR/GREED: Score is {fg_value} out of 100 ({fg_class}).{trend_note}"
        )
        if fg_value <= 25:
            lines.append(
                "  INSTRUCTION: Extreme fear. Contrarian note: "
                "'Crypto is terrified right now — which historically is when the quiet buyers step in.' "
                "Calm observer tone, not alarmist."
            )
        elif fg_value >= 75:
            lines.append(
                "  INSTRUCTION: Extreme greed. Contrarian note: "
                "'Everyone in crypto is euphoric — which is exactly when corrections tend to hit.' "
                "Don't be preachy — just note the historical pattern."
            )
        else:
            lines.append(
                "  INSTRUCTION: Neutral zone. Mention it briefly as context, don't dwell on it."
            )
    else:
        lines.append("CRYPTO FEAR/GREED: No data available.")

    lines.append("")

    # Major coin prices and movements
    crypto_coins = summary.get("crypto_coins", [])
    if crypto_coins:
        lines.append("MAJOR CRYPTO COINS RIGHT NOW:")
        for coin in crypto_coins:
            sym = coin["symbol"]
            price = coin.get("price_usd")
            ch24 = coin.get("change_24h_pct")
            ch7d = coin.get("change_7d_pct")
            price_note = f" Price: ${price:,.0f}." if price is not None else ""
            ch24_note = f" 24h: {ch24:+.1f}%." if ch24 is not None else ""
            ch7d_note = f" 7 days: {ch7d:+.1f}%." if ch7d is not None else ""
            lines.append(f"  - {sym}:{price_note}{ch24_note}{ch7d_note}")
        # Are coins moving together or diverging?
        directions = [c.get("direction", "neutral") for c in crypto_coins]
        all_same = len(set(directions)) == 1
        lines.append(
            f"  All major coins are {'moving in the same direction' if all_same else 'moving in different directions'} today."
        )
        lines.append(
            "  INSTRUCTION: For the Crypto section, note whether the whole crypto market is "
            "moving together (all going up or all going down) or diverging (some up, some down). "
            "If BTC is up but altcoins are down, that's a specific and interesting signal — "
            "say 'Bitcoin is climbing but the smaller coins aren't following.' "
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
            lines.append(
                f"  - {fut['friendly_name']} ({fut['symbol']}): "
                f"signals pointing {direction_plain}. "
                f"Sources: {domains_plain}. ({fut['signal_count']} signals)"
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
        bull_c = cot.get("bullish_count", 0)
        bear_c = cot.get("bearish_count", 0)
        lines.append(
            f"LARGE TRADER POSITIONING: The big professional traders are positioned "
            f"{dominant_cot} overall ({bull_c} bullish signals vs {bear_c} bearish)."
        )
        lines.append(
            "  INSTRUCTION: If COT data is notable (strongly one-directional), mention it: "
            "'The big professional traders are heavily positioned [direction] right now — "
            "which is either a smart bet or a crowded trade that could snap back.' "
            "Skip if mixed or flat."
        )
    else:
        lines.append("LARGE TRADER POSITIONING: No strong positioning signal today.")

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

    # Inevitabilities (stock convergence)
    inevitabilities = summary.get("inevitabilities", [])
    if inevitabilities:
        lines.append("SITUATIONS I THINK ARE MOST INEVITABLE (stocks/instruments with converging signals):")
        for inv in inevitabilities:
            direction_plain = _direction_words(inv["direction"])
            sources_plain = _domain_plain(inv["domains"])
            window = inv.get("expected_window_days")
            chain_raw = inv.get("world_model_chain", "")
            chain_note = ""
            if chain_raw and chain_raw != "None":
                _chain_clean = chain_raw.strip("[]").replace("'", "")
                chain_note = f" Causal chain: {_chain_clean}"
            window_note = f" Expected timing: within {window} days." if window else ""
            lines.append(
                f"  - {inv['ticker']}: {direction_plain}. "
                f"Evidence from: {sources_plain}.{window_note}{chain_note}"
            )
        lines.append(
            "  INSTRUCTION: Lead with the WEIRDEST convergence, not the highest confidence. "
            "Tell the story of WHY it's interesting. "
            "Example: 'Here's what's strange about [TICKER]: three completely unrelated signals "
            "all arrived at the same conclusion — insider buying reports, economic data, AND "
            "government contract data. That combination is unusual.' "
            "If a causal chain exists, mention it: 'What's interesting is this connects through "
            "[X] all the way to [Y].' Max 3 stocks. Skip tickers that are just technical signals."
        )
    else:
        lines.append("INEVITABLE SITUATIONS: Nothing stands out strongly in stocks today.")

    lines.append("")

    # Developing situations
    devs = summary.get("developing", [])
    if devs:
        lines.append("DEVELOPING STOCK SITUATIONS (what I'm actively watching):")
        for d in devs:
            direction_plain = _direction_words(d["direction"])
            confidence_plain = _confidence_words(d["confidence"])
            sources_plain = _domain_plain(d["domains"])
            lines.append(
                f"  - {d['ticker']}: {direction_plain}. "
                f"I am {confidence_plain}. "
                f"Evidence comes from: {sources_plain}. "
                f"Note: {d['summary'][:120]}" if d.get("summary") else
                f"  - {d['ticker']}: {direction_plain}. "
                f"I am {confidence_plain}. "
                f"Evidence comes from: {sources_plain}."
            )
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
        lines.append(
            f"RECENT PREDICTION RESULTS (last 7 days): "
            f"{wins} correct out of {total} checked."
        )
        if oc.get("win_examples"):
            lines.append(
                "  Right: "
                + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in oc["win_examples"])
            )
        if oc.get("loss_examples"):
            lines.append(
                "  Wrong: "
                + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in oc["loss_examples"])
            )
    else:
        lines.append("RECENT PREDICTION RESULTS: Nothing graded yet this week.")

    lines.append("")

    # ══════════════════════════════════════════════════════════════════
    # LAYER 5: WHAT I LEARNED / WHAT I GOT WRONG
    # ══════════════════════════════════════════════════════════════════
    lines.append("━━━ LAYER 5: WHAT I LEARNED / WHAT I GOT WRONG ━━━")
    lines.append("")

    # Post-mortem learning
    pm = summary.get("postmortem", {})
    if pm and pm.get("total_graded", 0) > 0:
        pm_wr = pm.get("overall_win_rate_pct", "?")
        pm_grade = pm.get("grade", "")
        lines.append(
            f"OVERALL TRACK RECORD: {pm_wr} of predictions correct "
            f"({pm.get('total_graded', 0)} total checked, grade: {pm_grade})."
        )
        if pm.get("best_combos"):
            lines.append(
                f"  Best signal combinations: {'; '.join(pm['best_combos'][:2])}"
            )
            lines.append(
                "  INSTRUCTION: Translate combo keys to plain language. "
                "'When company news, economic data, and price patterns all agree' — "
                "not the raw source names."
            )
        if pm.get("timing_insight"):
            lines.append(f"  Timing note: {pm['timing_insight']}")
    else:
        lines.append("OVERALL TRACK RECORD: Not enough graded data yet.")

    lines.append("")

    # Active hypotheses
    active_hyps = summary.get("active_hypotheses", [])
    if active_hyps:
        lines.append("THEORIES I'M TESTING:")
        for h in active_hyps[:3]:
            wr = h.get("win_rate_pct")
            wr_note = f" Works {wr}% of the time ({h['observations']} checks)." if wr is not None else ""
            story = h.get("causal_story", "")[:150]
            lines.append(f"  - {h['name']}: {story}{wr_note}")
        lines.append(
            "  INSTRUCTION: Put the most interesting theory in the 'What I Learned' section. "
            "Translate to plain English. 'I have a theory: when [plain description] happens, "
            "[outcome] tends to follow.' Don't use technical indicator names."
        )

    lines.append("")

    # Source reliability
    th = summary.get("thompson", {})
    trusted = th.get("trusted", [])
    distrusted = th.get("distrusted", [])
    if trusted or distrusted:
        if trusted:
            lines.append(
                f"SOURCE RELIABILITY: {len(trusted)} information source(s) are consistently reliable "
                f"(correct more than 55% of the time)."
            )
        if distrusted:
            lines.append(
                f"  {len(distrusted)} source(s) are performing below chance — I'm weighting them less."
            )
        lines.append(
            "  INSTRUCTION: Describe sources by what they measure, not their technical names. "
            "'insider buying reports' not 'sec_form4'. Goes in 'What I Learned'."
        )

    lines.append("")
    lines.append(
        "Now write the daily letter following the LAYERED STRUCTURE precisely: "
        "Big Picture → Crypto → Commodities & Futures → Stocks → Learned → Wrong. "
        "Bold the punch lines, use bullets, no jargon. "
        "Under 600 words. Sign it '— MIDGE'."
    )

    return "\n".join(lines)


def _template_narrative(summary: dict, date_str: str) -> str:
    """Template fallback when no LLM is available.

    Short, punchy, jargon-free. Matches the style guide: bold punch lines,
    bullets, plain English, lead with the weird thing.
    """
    lines = [
        f"Subject: MIDGE Daily Letter — {date_str}",
        "",
    ]

    # ── Lead hook — Granger weirdness first ───────────────────────
    granger = summary.get("granger", [])
    if granger:
        g = granger[0]
        lines.append(f"**Here's something strange I noticed: {g['story']}**")
        lines.append("")

    # ── WHAT I'M WATCHING ─────────────────────────────────────────
    lines.append("## WHAT I'M WATCHING")
    lines.append("")
    devs = summary.get("developing", [])
    if devs:
        for d in devs[:3]:
            direction_plain = _direction_words(d["direction"])
            confidence_plain = _confidence_words(d["confidence"])
            sources_plain = _domain_plain(d["domains"])
            lines.append(f"**{d['ticker']}** — {direction_plain}.")
            lines.append(f"- I am {confidence_plain}.")
            lines.append(f"- Evidence is coming from: {sources_plain}.")
            lines.append("")
    else:
        lines.append("Nothing with strong evidence right now. Watching broadly.")
        lines.append("")

    # ── Inevitabilities (inject into WHAT I'M WATCHING if present) ──
    inevitabilities = summary.get("inevitabilities", [])
    if inevitabilities and not devs:
        # Only use inevitabilities as fallback if situation_board is empty
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
                chain_note = f"\n- Causal chain: {_chain_clean_tmpl}"
            lines.append(f"**{inv['ticker']}** — {direction_plain}.{window_note}")
            lines.append(f"- I am {confidence_plain}.")
            lines.append(f"- Evidence from: {sources_plain}.{chain_note}")
            lines.append("")

    # ── Developing situations ─────────────────────────────────────
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

    # ── Somatic building ──────────────────────────────────────────
    somatic = summary.get("somatic_building", [])
    if somatic:
        top_s = somatic[0]
        direction_plain = _direction_words(top_s["dominant_direction"])
        domain_plain = _domain_plain(top_s["domains"])
        lines.append(
            f"*Sensing something in {top_s['ticker']} — "
            f"{top_s['domain_count']} sources pointing {direction_plain} "
            f"({domain_plain}). Not ready to call it yet.*"
        )
        lines.append("")

    # ── Cross-market anomalies ─────────────────────────────────────
    cross_market = summary.get("cross_market_anomalies", [])
    if cross_market:
        top_cm = cross_market[0]
        tickers_note = f" ({', '.join(top_cm['tickers'][:4])})" if top_cm.get("tickers") else ""
        lines.append(
            f"*Here's something weird across markets: "
            f"{top_cm.get('description', top_cm.get('type', 'unusual pattern'))}"
            f"{tickers_note}. When unrelated markets move together like this, something is flowing underneath.*"
        )
        lines.append("")

    # ── Crypto Fear & Greed ───────────────────────────────────────
    fg = summary.get("crypto_fear_greed", {})
    if fg and fg.get("value") is not None:
        fg_value = int(fg["value"])
        fg_class = fg.get("classification", "")
        fg_trend = fg.get("trend", "")
        trend_note = f" It's been {fg_trend}." if fg_trend else ""
        if fg_value <= 25:
            lines.append(
                f"*Crypto markets are terrified right now (sentiment score: {fg_value}, {fg_class}).{trend_note} "
                "Historically this is when smart money quietly buys.*"
            )
            lines.append("")
        elif fg_value >= 75:
            lines.append(
                f"*Everyone in crypto is euphoric right now (sentiment score: {fg_value}, {fg_class}).{trend_note} "
                "This is exactly when corrections tend to hit.*"
            )
            lines.append("")

    # ── Alerts sent today (follow-up) ─────────────────────────────
    today_alerts = summary.get("recent_alerts", [])
    if today_alerts:
        tickers_sent = [a["ticker"] for a in today_alerts]
        lines.append(
            f"*I sent you {len(today_alerts)} alert(s) today "
            f"about {', '.join(tickers_sent[:6])}. "
            "Checking back on those moves.*"
        )
        lines.append("")

    # ── WHAT CONFIRMED ────────────────────────────────────────────
    lines.append("## WHAT CONFIRMED")
    lines.append("")
    oc = summary.get("outcomes", {})
    total = oc.get("total", 0)
    wins = oc.get("wins", 0)
    losses = oc.get("losses", 0)
    if total > 0:
        win_examples = oc.get("win_examples", [])
        loss_examples = oc.get("loss_examples", [])
        lines.append(
            f"I checked {total} of my recent calls — **{wins} were right, {losses} were wrong.**"
        )
        if win_examples:
            lines.append(
                "- Right: " + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in win_examples)
            )
        if loss_examples:
            lines.append(
                "- Wrong: " + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in loss_examples)
            )
    else:
        lines.append("No recent calls have been checked yet.")
    lines.append("")

    # ── WHAT I LEARNED ────────────────────────────────────────────
    lines.append("## WHAT I LEARNED")
    lines.append("")
    pm = summary.get("postmortem", {})
    th = summary.get("thompson", {})
    learned_any = False

    if granger and len(granger) > 1:
        for g in granger[1:3]:
            lines.append(f"- **{g['story']}**")
            learned_any = True

    pm_wr = pm.get("overall_win_rate_pct", "")
    pm_graded = pm.get("total_graded", 0)
    if pm_wr and pm_graded > 0:
        lines.append(
            f"- Overall, **{pm_wr} of my calls have been correct** ({pm_graded} checked so far)."
        )
        learned_any = True

    trusted = th.get("trusted", [])
    distrusted = th.get("distrusted", [])
    if trusted:
        lines.append(
            f"- **{len(trusted)} of my information source(s) are proving consistently reliable** "
            "based on what I've seen so far."
        )
        learned_any = True
    if distrusted:
        lines.append(
            f"- {len(distrusted)} source(s) are not doing well — I'm trusting them less."
        )
        learned_any = True

    # ── Cascade learning ─────────────────────────────────────────
    cascade = summary.get("cascade_status", {})
    notable_chains = cascade.get("notable_chains", [])
    for _ch in notable_chains[:1]:
        _confirmed = _ch.get("confirmed_count", 0)
        _total = _ch.get("total_links", 0)
        _trigger = _ch.get("trigger", "?")
        _next = _ch.get("next_dominoes", [])
        if _confirmed > 0 and _total > 1:
            _next_note = f" Watching for: {', '.join(_next[:2])}." if _next else ""
            lines.append(
                f"- **A chain I predicted from {_trigger} is unfolding: "
                f"{_confirmed} of {_total} dominoes confirmed.**{_next_note}"
            )
            learned_any = True

    # ── Active hypothesis insight ─────────────────────────────────
    active_hyps = summary.get("active_hypotheses", [])
    best_hyp = next(
        (h for h in active_hyps if h["status"] == "active" and h.get("win_rate_pct") and h["win_rate_pct"] > 50),
        None,
    )
    if best_hyp:
        lines.append(
            f"- **My theory '{best_hyp['name']}' is holding up: "
            f"{best_hyp['win_rate_pct']}% accuracy over {best_hyp['observations']} checks.**"
        )
        learned_any = True

    if not learned_any:
        lines.append("- Not enough graded data yet to say something concrete.")
    lines.append("")

    # ── WHAT I'M UNCERTAIN ABOUT ─────────────────────────────────
    lines.append("## WHAT I'M UNCERTAIN ABOUT")
    lines.append("")
    pm_grade = pm.get("grade", "")
    if pm_graded < 30:
        lines.append(
            "Most of my calls are still within their evaluation window. "
            "I don't have enough checked results yet to know which patterns are truly reliable."
        )
    elif pm_grade in ("WEAK", "POOR"):
        lines.append(
            "**My hit rate is lower than I'd like.** "
            "The signal combinations that work are not consistent enough yet."
        )
    else:
        lines.append("My timing is often off — moves are happening, but outside the windows I expected.")

    pm_timing = pm.get("timing_insight", "")
    if pm_timing:
        lines.append(f"- {pm_timing}")
    lines.append("")

    # ── WHAT I GOT WRONG ─────────────────────────────────────────
    lines.append("## WHAT I GOT WRONG")
    lines.append("")
    loss_examples = oc.get("loss_examples", [])
    if loss_examples:
        for e in loss_examples[:2]:
            lines.append(
                f"- **{e['symbol']}**: I thought it would move the other way. "
                f"It moved {e['pct']}%."
            )
    else:
        lines.append("Nothing specific to report yet — not enough graded calls.")
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
                logger.info("Daily narrative generated via Groq (%d chars)", len(narrative_body))
            else:
                logger.info("Groq call returned nothing — falling back to template")

        if not narrative_body:
            # Template fallback
            narrative_body = _template_narrative(summary, date_str)
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
