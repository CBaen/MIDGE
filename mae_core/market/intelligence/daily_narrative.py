"""Daily narrative letter generator — MIDGE's morning letter to Guiding Light.

Gathers everything MIDGE knows across all data files and produces a human-readable
narrative letter. Uses Groq (llama-3.3-70b-versatile) to narrate the data as a story.
Falls back to a template-based format when no LLM is available.

The letter is written once per calendar day, archived to data/midge/daily_narratives/,
and optionally emailed via the EmailNotifier.

Five sections:
  WHAT I'M WATCHING    — developing situations with causal reasoning
  WHAT CONFIRMED       — situations that played out (or didn't)
  WHAT I LEARNED       — Thompson movements, combo wins/losses, Granger discoveries
  WHAT I'M UNCERTAIN ABOUT — mixed evidence, missing data
  WHAT I GOT WRONG     — honest post-mortem on failed predictions

Design constraints:
  - Max 500 words output
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
_MAX_TOKENS = 900  # Slightly above 500 words to give the model breathing room


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
LETTER STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Start with: Subject: MIDGE Daily Letter — [DATE]

Then optionally a 1-sentence hook — the single strangest thing you noticed today.

Five sections (use exactly these headers):

## WHAT I'M WATCHING
3 situations max. 3-4 bullets each. Use plain English for the direction \
("looks like it might rise" not "bullish").

## WHAT CONFIRMED
1-2 items. What you predicted that came true. What you predicted that didn't.

## WHAT I LEARNED
2-3 bullets. Keep each to one sentence. If you found a weird causal relationship, \
THIS is where it goes — and it should be in bold.

## WHAT I'M UNCERTAIN ABOUT
Honest. Short. What's murky, what's missing.

## WHAT I GOT WRONG
1-2 items. Direct and honest. "I thought X. I was wrong. Here's why."

If you have a paper trade recommendation, add a section:

## WHAT I THINK YOU SHOULD LOOK AT
For each recommendation:
- **"I placed a paper trade on [TICKER]"** or **"I think you should look at [direction] [TICKER]"**
- One sentence on WHY: "because [plain English reason]"
- The market: "This is a US stock" or "This is a futures contract"
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
- Under 400 words total. One page. Coffee-length.
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


def _build_llm_prompt(summary: dict) -> str:
    """Build the plain-English data context to hand to the LLM.

    All technical terms are translated here so the LLM never sees jargon
    it might echo back. The LLM's job is ONLY to write the letter in MIDGE's
    voice — we do the data translation in Python.
    """
    lines = [
        f"Today is {summary['date']}. Here is everything I know right now.",
        "Write the daily letter using this data. Follow all style rules exactly.",
        "",
    ]

    # ── LEAD: Granger causal discoveries (the "weird" thing) ──────────
    # These go FIRST in the prompt so the LLM treats them as the most
    # important input and leads the letter with them.
    granger = summary.get("granger", [])
    if granger:
        lines.append("STRANGE CAUSAL DISCOVERIES (LEAD WITH THESE — most interesting to Guiding Light):")
        for g in granger[:3]:
            lines.append(f"  - {g['story']}")
        lines.append(
            "  INSTRUCTION: These are the weirdest things I've found. "
            "The first thing in the letter body should reference the most surprising one "
            "in plain English. Example: 'Here's something strange I noticed: when [X] happens, "
            "[Y] tends to follow about [N] days later. Like clockwork.'"
        )
    else:
        lines.append("CAUSAL DISCOVERIES: None new to report.")

    lines.append("")

    # ── Developing situations ──────────────────────────────────────
    devs = summary.get("developing", [])
    if devs:
        lines.append("DEVELOPING SITUATIONS (what I'm actively watching):")
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
        lines.append("DEVELOPING SITUATIONS: None with strong evidence right now.")

    lines.append("")

    # ── Recent convergence alerts ──────────────────────────────────
    alerts = summary.get("recent_alerts", [])
    if alerts:
        lines.append("SIGNALS THAT FIRED IN THE LAST 24 HOURS:")
        for a in alerts:
            direction_plain = _direction_words(a["direction"])
            confidence_plain = _confidence_words(a["confidence"])
            lines.append(f"  - {a['ticker']}: {direction_plain}. I am {confidence_plain}.")
    else:
        lines.append("SIGNALS LAST 24H: None.")

    lines.append("")

    # ── Paper trades ────────────────────────────────────────────────
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
            "  INSTRUCTION: Include a 'WHAT I THINK YOU SHOULD LOOK AT' section. "
            "Say 'I placed a paper trade on [TICKER]' in bold. Explain why in one plain sentence. "
            "State the market (US stock, futures, crypto, etc). Give timing and risk estimates."
        )
    else:
        lines.append("PAPER TRADES TODAY: None placed.")

    lines.append("")

    # ── Outcomes (recent graded predictions) ─────────────────────
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
                "  What moved the right way: "
                + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in oc["win_examples"])
            )
        if oc.get("loss_examples"):
            lines.append(
                "  What I got wrong: "
                + ", ".join(f"{e['symbol']} moved {e['pct']}%" for e in oc["loss_examples"])
            )
    else:
        lines.append("RECENT PREDICTION RESULTS: No predictions have been graded yet this week.")

    lines.append("")

    # ── Post-mortem learning ───────────────────────────────────────
    pm = summary.get("postmortem", {})
    if pm and pm.get("total_graded", 0) > 0:
        pm_wr = pm.get("overall_win_rate_pct", "?")
        pm_grade = pm.get("grade", "")
        lines.append(
            f"OVERALL TRACK RECORD: {pm_wr} of predictions have been correct "
            f"({pm.get('total_graded', 0)} total checked so far, grade: {pm_grade})."
        )
        if pm.get("best_combos"):
            # Translate combo keys into plain language note
            lines.append(
                f"  Signal combinations that work best: "
                f"{'; '.join(pm['best_combos'][:2])}"
            )
            lines.append(
                "  INSTRUCTION: Do NOT repeat the combo keys verbatim. "
                "Describe them in plain language, e.g. 'when company news, economic data, and "
                "price patterns all agree' — not the raw source names."
            )
        if pm.get("timing_insight"):
            lines.append(f"  Timing observation: {pm['timing_insight']}")
    else:
        lines.append("OVERALL TRACK RECORD: Not enough graded data yet.")

    lines.append("")

    # ── Source reliability ─────────────────────────────────────────
    th = summary.get("thompson", {})
    trusted = th.get("trusted", [])
    distrusted = th.get("distrusted", [])
    if trusted or distrusted:
        lines.append("WHAT I'VE LEARNED ABOUT MY OWN SOURCES:")
        if trusted:
            # Avoid printing internal source names — just describe how reliable
            lines.append(
                f"  {len(trusted)} source(s) have proven consistently reliable "
                f"(correct more than 55% of the time based on what I've learned)."
            )
        if distrusted:
            lines.append(
                f"  {len(distrusted)} source(s) are performing below chance — "
                "I'm giving them less weight."
            )
        lines.append(
            "  INSTRUCTION: Do NOT name the sources by their technical names. "
            "Describe them by what they measure: 'insider buying reports' not 'sec_form4'."
        )

    lines.append("")
    lines.append(
        "Now write the daily letter. Follow the style rules precisely: "
        "bold the punch lines, use bullets, lead with the weird causal discovery, "
        "no jargon, translate everything to plain English. "
        "Under 400 words. Sign it '— MIDGE'."
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
