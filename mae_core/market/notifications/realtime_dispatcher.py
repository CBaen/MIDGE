"""Real-time alert dispatcher — MIDGE's immediate voice.

Fires email alerts immediately for high-confidence events, without waiting
for the daily letter. Three trigger types:
  - Convergence alerts: confidence >= 0.65 AND 4+ domains
  - Causal cascade confirmations: any confirmed domino
  - Cross-market anomalies: strength >= 0.70

Deduplication per ticker+direction: 4-hour window, persisted across restarts
to data/midge/realtime_dispatch_state.json (last 100 entries).
Rate limit: 5 real-time alerts per hour (separate budget from daily emails).
All sends are wrapped in try/except — never crashes the daemon.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("midge.realtime_dispatcher")

_CONVERGENCE_MIN_CONFIDENCE = 0.65
_CONVERGENCE_MIN_DOMAINS    = 4
_CROSS_MARKET_MIN_STRENGTH  = 0.70
_DEDUP_WINDOW_HOURS         = 4
_MAX_ALERTS_PER_HOUR        = 5
_STATE_MAX_ENTRIES          = 100

_ROOT       = Path(__file__).resolve().parents[4]
_STATE_FILE = _ROOT / "data" / "midge" / "realtime_dispatch_state.json"
_LOG_FILE   = _ROOT / "data" / "midge" / "realtime_alerts.jsonl"

# ── Plain-language helpers ──────────────────────────────────────────

def _confidence_phrase(c: float) -> str:
    if c > 0.80: return "I'm very confident"
    if c >= 0.60: return "I'm fairly sure"
    if c >= 0.45: return "something is forming but I need more"
    return "I noticed something but it's early"

def _direction_word(d: str) -> str:
    return {"bullish":"up","up":"up","buy":"up","bearish":"down","down":"down","sell":"down"}.get(d.lower(), d)

def _domain_plain(d: str) -> str:
    return {
        "insider":"insider trades","macro":"economic shifts","technical":"price chart patterns",
        "events":"regulatory filings","positioning":"large-trader positioning",
        "government":"congressional trades","congress":"congressional trades",
        "contracts":"government contracts","sentiment":"social media discussion",
        "fundamental":"financial data","institutional":"institutional holdings",
        "crypto":"crypto market activity","price":"price history patterns",
        "energy":"energy inventory data","cascade":"confirmed causal chain",
        "cross_market":"cross-market structure",
    }.get(d.lower(), d)

def _resolve_ticker(alert: dict) -> str:
    t = alert.get("ticker") or alert.get("primary_ticker") or ""
    if not t:
        for s in alert.get("signals", []):
            sym = s.get("metadata", {}).get("symbol", "") if isinstance(s, dict) else ""
            if sym:
                return sym
    return t or "UNKNOWN"

# ── Email body formatters ───────────────────────────────────────────

def _fmt_convergence(alert: dict, plain_text: Optional[str], confidence: float) -> str:
    ticker    = _resolve_ticker(alert)
    direction = alert.get("direction", "neutral")
    domains   = alert.get("domains_converging", [])
    timing    = alert.get("expected_move_window_days") or alert.get("window_days")
    win_rate  = alert.get("template_win_rate") or alert.get("win_rate")
    ripples   = alert.get("ripple_effects", [])
    stop_pct  = alert.get("stop_loss_pct")
    domain_list = ", ".join(_domain_plain(d) for d in domains[:5]) if domains else "multiple areas"

    lines: List[str] = []

    # What I see
    if plain_text:
        pl_lines = [l for l in plain_text.splitlines()
                    if l.strip() and not any(l.startswith(s) for s in ("HISTORY","TIMING","ACTION","TRACKING"))]
        lines.append(f"What I see: {pl_lines[0].strip()}" if pl_lines
                     else f"What I see: {ticker} looks like it's going {_direction_word(direction)}.")
    else:
        lines.append(f"What I see: {ticker} looks like it's going {_direction_word(direction)}.")
    lines.append("")

    # Why confident
    lines.append(f"Why I'm confident: {_confidence_phrase(confidence)} — {domain_list} are all pointing the same way.")
    if win_rate and float(win_rate) > 0:
        lines.append(f"When this combination appeared before, it worked {int(round(float(win_rate)*100))}% of the time.")
    if len(domains) >= 5:
        lines.append(f"That's {len(domains)} independent areas of evidence, which is unusually high.")
    lines.append("")

    # What I'd do
    if confidence >= 0.75:
        action = "buying" if direction in ("bullish","up","buy") else "selling"
        lines.append(f"What I'd do: I think you should look at {action} {ticker}.")

    if timing:
        lines.append(f"Timing: Based on history, this move typically happens within {int(timing)} days.")
    if stop_pct:
        lines.append(f"Risk: If I'm wrong, the typical loss is about {stop_pct:.1f}%.")

    # Ripple preview
    watchable_ripples = [r for r in ripples[:2] if isinstance(r, dict)]
    if watchable_ripples:
        parts = [f"{r.get('ticker','?')} (~{r.get('lag_days',0):.0f}d later)" for r in watchable_ripples]
        lines += ["", f"If this plays out: watch for {' and '.join(parts)} to follow."]

    lines += ["", "---", "This is what I see, not financial advice. Do your own research."]
    return "\n".join(lines)


def _fmt_cascade(cascade_info: dict) -> str:
    trigger      = cascade_info.get("trigger", "an earlier prediction")
    confirmed    = cascade_info.get("confirmed_ticker", "unknown")
    conf_count   = cascade_info.get("confirmed_count", 1)
    total        = cascade_info.get("total_links", "?")
    energy_ratio = cascade_info.get("energy_ratio", 1.0)
    remaining    = cascade_info.get("remaining", [])

    energy_desc = ""
    if isinstance(energy_ratio, (int, float)):
        if energy_ratio > 1.1: energy_desc = " The chain is moving faster than I predicted."
        elif energy_ratio < 0.9: energy_desc = " The chain is moving slower than I predicted."

    lines = [
        f"What happened: Remember the causal chain I was tracking from {trigger}? "
        f"The next domino just fell — {confirmed} moved as predicted.",
        "",
        f"Progress: {conf_count} of {total} links in the chain have now confirmed.{energy_desc}",
    ]
    watchable = [r for r in remaining if r.get("watchable", True)][:3]
    if watchable:
        next_list = ", ".join(f"{r.get('ticker','?')} (~{r.get('lag_days',0):.0f}d)" for r in watchable)
        lines.append(f"Still watching: {next_list}.")

    lines += ["", "Why this matters: each confirmed domino makes the rest more likely. The chain is real.",
              "", "---", "This is what I see, not financial advice. Do your own research."]
    return "\n".join(lines)


def _fmt_cross_market(discovery: dict) -> str:
    d_type   = discovery.get("discovery_type", "anomaly")
    domains  = discovery.get("affected_domains", [])
    tickers  = discovery.get("affected_tickers", [])
    strength = float(discovery.get("strength", 0.0))
    desc     = discovery.get("description", "")

    type_plain = {
        "correlation_breakdown":  "two normally-related areas are now moving independently",
        "volume_cluster":         "multiple unrelated assets are seeing unusually heavy trading at the same time",
        "cross_asset_divergence": "fear signals and risk signals are both firing simultaneously — the market is contradicting itself",
        "domain_silence":         "an area that's normally busy has gone quiet",
    }.get(d_type, d_type.replace("_", " "))

    domain_str    = " and ".join(_domain_plain(d) for d in domains[:3]) if domains else "several areas"
    strength_word = "very strong" if strength >= 0.85 else "strong"

    lines = [f"Something weird is happening: {type_plain}.", "", f"Where I see it: {domain_str}."]
    if tickers:
        lines.append(f"Assets involved: {', '.join(tickers[:5])}.")
    if desc:
        lines += ["", f"Details: {desc}"]
    lines += [
        "", f"How unusual: {strength_word} ({int(round(strength*100))}% anomaly score).",
        "", "Why I'm telling you now: cross-market structural breaks often precede larger moves. This is worth watching.",
        "", "---", "This is what I see, not financial advice. Do your own research.",
    ]
    return "\n".join(lines)


# ── Main class ──────────────────────────────────────────────────────

class RealtimeDispatcher:
    """MIDGE's immediate voice — fires email alerts for high-confidence events.

    Usage:
        dispatcher = RealtimeDispatcher(email_notifier=ctx.email_notifier)
        dispatcher.dispatch_convergence(alert_dict)
    """

    def __init__(self, email_notifier=None) -> None:
        self._notifier = email_notifier
        self._dedup: Dict[str, float] = {}
        self._send_timestamps: List[float] = []
        self._load_state()

    # ── Public API ──────────────────────────────────────────────────

    def dispatch_convergence(self, alert: dict, plain_language_text: Optional[str] = None) -> bool:
        """Send immediately when convergence fires at confidence >= 0.65 AND 4+ domains."""
        try:
            confidence = float(alert.get("confidence", 0.0))
            domains    = alert.get("domains_converging", [])
            if confidence < _CONVERGENCE_MIN_CONFIDENCE or len(domains) < _CONVERGENCE_MIN_DOMAINS:
                return False

            ticker    = _resolve_ticker(alert)
            direction = alert.get("direction", "neutral")
            dedup_key = f"{ticker}:{direction}"
            if not self._gate(dedup_key):
                return False

            subject = f"MIDGE: {ticker} looks {_direction_word(direction)} — {int(round(confidence*100))}% sure"
            body    = _fmt_convergence(alert, plain_language_text, confidence)
            sent    = self._send(subject, body)
            if sent:
                self._record(dedup_key, "convergence", ticker, direction, confidence)
            return sent
        except Exception:
            logger.error("dispatch_convergence failed", exc_info=True)
            return False

    def dispatch_cascade_confirmed(self, cascade_info: dict) -> bool:
        """Send when a causal chain domino confirms."""
        try:
            chain_id  = cascade_info.get("chain_id", "")
            confirmed = cascade_info.get("confirmed_ticker", "")
            direction = cascade_info.get("confirmed_direction", "neutral")
            dedup_key = f"cascade:{chain_id}:{confirmed}"
            if not self._gate(dedup_key):
                return False

            conf_count = cascade_info.get("confirmed_count", 1)
            total      = cascade_info.get("total_links", "?")
            subject    = f"MIDGE: Causal chain confirming — {conf_count}/{total} dominoes down"
            body       = _fmt_cascade(cascade_info)
            sent       = self._send(subject, body)
            if sent:
                self._record(dedup_key, "cascade", confirmed, direction, 1.0)
            return sent
        except Exception:
            logger.error("dispatch_cascade_confirmed failed", exc_info=True)
            return False

    def dispatch_cross_market_anomaly(self, discovery) -> bool:
        """Send when cross-market hunter finds an anomaly with strength >= 0.70."""
        try:
            if hasattr(discovery, "__dict__"):
                discovery = discovery.__dict__
            elif hasattr(discovery, "_asdict"):
                discovery = discovery._asdict()

            strength = float(discovery.get("strength", 0.0))
            if strength < _CROSS_MARKET_MIN_STRENGTH:
                return False

            d_id      = discovery.get("discovery_id", "")
            d_type    = discovery.get("discovery_type", "anomaly")
            domains   = discovery.get("affected_domains", [])
            dedup_key = f"cross:{d_type}:{d_id[:12] if d_id else 'unknown'}"
            if not self._gate(dedup_key):
                return False

            domain_str = "+".join(domains[:2]) if domains else "cross-market"
            subject    = f"MIDGE: Something weird across {domain_str} — worth watching"
            body       = _fmt_cross_market(discovery)
            sent       = self._send(subject, body)
            if sent:
                self._record(dedup_key, "cross_market", "", "neutral", strength)
            return sent
        except Exception:
            logger.error("dispatch_cross_market_anomaly failed", exc_info=True)
            return False

    # ── Internal ────────────────────────────────────────────────────

    def _send(self, subject: str, body: str) -> bool:
        if self._notifier is None:
            logger.warning("RealtimeDispatcher: no EmailNotifier — skipping: %s", subject)
            return False
        try:
            return self._notifier.send(subject, body)
        except Exception:
            logger.error("RealtimeDispatcher: notifier.send raised", exc_info=True)
            return False

    def _gate(self, key: str) -> bool:
        """Return True if dedup AND rate limit both pass."""
        now = time.time()
        last = self._dedup.get(key)
        if last is not None and (now - last) / 3600.0 < _DEDUP_WINDOW_HOURS:
            logger.debug("RealtimeDispatcher dedup suppressed: %s", key)
            return False
        cutoff = now - 3600.0
        self._send_timestamps = [t for t in self._send_timestamps if t > cutoff]
        if len(self._send_timestamps) >= _MAX_ALERTS_PER_HOUR:
            logger.warning("RealtimeDispatcher: rate limit %d/hr reached", _MAX_ALERTS_PER_HOUR)
            return False
        return True

    def _record(self, dedup_key: str, event_type: str, ticker: str, direction: str, confidence: float) -> None:
        now = time.time()
        self._dedup[dedup_key] = now
        self._send_timestamps.append(now)
        # Evict stale dedup entries
        cutoff = now - (_DEDUP_WINDOW_HOURS * 3600)
        self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}
        self._save_state()
        entry = {
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type, "ticker": ticker,
            "direction": direction, "confidence": round(confidence, 4),
            "dedup_key": dedup_key,
        }
        try:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception:
            logger.warning("RealtimeDispatcher: could not write log", exc_info=True)

    def _load_state(self) -> None:
        try:
            if _STATE_FILE.exists():
                data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
                self._dedup = {
                    e["dedup_key"]: e["sent_at"]
                    for e in data.get("recent_dispatches", [])
                    if "dedup_key" in e and "sent_at" in e
                }
        except Exception:
            logger.warning("RealtimeDispatcher: could not load state", exc_info=True)
            self._dedup = {}

    def _save_state(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            entries = sorted(self._dedup.items(), key=lambda kv: kv[1], reverse=True)
            recent  = [{"dedup_key": k, "sent_at": v} for k, v in entries[:_STATE_MAX_ENTRIES]]
            _STATE_FILE.write_text(json.dumps({"recent_dispatches": recent}, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("RealtimeDispatcher: could not save state", exc_info=True)
