"""Email notification system for MIDGE.

Sends plain-text emails to Guiding Light when MIDGE detects high-confidence
convergence alerts or wants to share a daily summary.

Configuration (environment variables):
    MIDGE_SMTP_HOST    SMTP server hostname (default: smtp.gmail.com)
    MIDGE_SMTP_PORT    SMTP port (default: 587)
    MIDGE_SMTP_USER    Sender email address
    MIDGE_SMTP_PASS    App password (not account password for Gmail)
    MIDGE_NOTIFY_EMAIL Recipient email (Guiding Light)

All sends are wrapped in try/except — MIDGE never crashes the daemon
because of a failed notification.
"""

from __future__ import annotations

import logging
import os
import smtplib
import time
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, Optional

logger = logging.getLogger("midge.notifications")

# Deduplication window: same ticker+direction within 4 hours is suppressed
_DEDUP_WINDOW_HOURS = 4

# Rate limit: max 10 emails per hour across all send calls
_MAX_EMAILS_PER_HOUR = 10


class EmailNotifier:
    """Sends email notifications from MIDGE to Guiding Light.

    Lifecycle:
        - Constructed once at bootstrap (market_systems.py)
        - Stored on ctx.email_notifier
        - Called by market_hooks_sensing.py after paper trade gate fires
        - If SMTP is not configured, all sends silently no-op after one boot warning
    """

    def __init__(self) -> None:
        self._host: str = os.environ.get("MIDGE_SMTP_HOST", "smtp.gmail.com")
        self._port: int = int(os.environ.get("MIDGE_SMTP_PORT", "587"))
        self._user: Optional[str] = os.environ.get("MIDGE_SMTP_USER") or None
        self._pass: Optional[str] = os.environ.get("MIDGE_SMTP_PASS") or None
        self._recipient: Optional[str] = os.environ.get("MIDGE_NOTIFY_EMAIL") or None

        # Deduplication: {"{ticker}:{direction}": last_sent_timestamp_float}
        self._dedup: Dict[str, float] = {}

        # Rate limiting: ring buffer of send timestamps (epoch seconds)
        self._send_timestamps: list[float] = []

        # Log a single warning at boot if not configured, then stay silent
        self._configured_warned = False
        if not self.is_configured():
            logger.warning(
                "EmailNotifier: SMTP not configured "
                "(set MIDGE_SMTP_USER, MIDGE_SMTP_PASS, MIDGE_NOTIFY_EMAIL). "
                "All notifications will be silently skipped."
            )
            self._configured_warned = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True if all required SMTP credentials are present in env."""
        return bool(self._user and self._pass and self._recipient)

    def send(self, subject: str, body: str) -> bool:
        """Send a plain-text email.

        Returns True on success, False on any failure. Never raises.
        """
        if not self.is_configured():
            return False

        if not self._check_rate_limit():
            logger.warning(
                "EmailNotifier: rate limit reached (%d/hr) — suppressing send",
                _MAX_EMAILS_PER_HOUR,
            )
            return False

        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self._user
            msg["To"] = self._recipient

            with smtplib.SMTP(self._host, self._port, timeout=15) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(self._user, self._pass)
                smtp.sendmail(self._user, [self._recipient], msg.as_string())

            self._record_send()
            logger.info("Email sent: %s", subject)
            return True

        except Exception:
            logger.error("EmailNotifier: send failed for subject=%r", subject, exc_info=True)
            return False

    def send_convergence_alert(self, alert_dict: dict) -> bool:
        """Format and send a convergence alert email.

        Applies 4-hour deduplication per ticker+direction pair.
        Only fires if confidence > 0.60.

        Args:
            alert_dict: ConvergenceAlert.to_dict() output

        Returns:
            True if email was sent, False if skipped or failed.
        """
        if not self.is_configured():
            return False

        confidence = float(alert_dict.get("confidence", 0.0))
        if confidence < 0.60:
            return False

        # Resolve ticker from alert dict
        ticker = alert_dict.get("ticker") or alert_dict.get("primary_ticker") or ""
        if not ticker:
            # Try to pull from signals list
            for sig in alert_dict.get("signals", []):
                sym = (
                    sig.get("metadata", {}).get("symbol", "")
                    if isinstance(sig, dict) else ""
                )
                if sym:
                    ticker = sym
                    break
        if not ticker:
            ticker = "UNKNOWN"

        direction = alert_dict.get("direction", "neutral")

        # Deduplication gate: same ticker+direction within 4 hours → skip
        dedup_key = f"{ticker}:{direction}"
        now_ts = time.time()
        last_sent = self._dedup.get(dedup_key)
        if last_sent is not None:
            elapsed_hours = (now_ts - last_sent) / 3600.0
            if elapsed_hours < _DEDUP_WINDOW_HOURS:
                logger.debug(
                    "Email dedup suppressed: %s (%.1fh ago)",
                    dedup_key, elapsed_hours,
                )
                return False

        # Build subject
        domains = alert_dict.get("domains_converging", [])
        n_domains = len(domains)
        direction_label = direction.upper() if direction != "neutral" else "NEUTRAL"
        subject = f"MIDGE: {ticker} {direction_label} — {n_domains} domains converging"

        # Build body
        summary = alert_dict.get("summary", "")
        strength = float(alert_dict.get("strength", 0.0))
        sequence_score = alert_dict.get("sequence_score")
        ripple_effects = alert_dict.get("ripple_effects", [])

        body_lines = [
            f"WHAT: {summary or 'Multi-domain convergence detected'}",
            "",
            f"CONFIDENCE: {confidence * 100:.1f}%",
            f"STRENGTH:   {strength * 100:.1f}%",
            f"DOMAINS:    {', '.join(domains) if domains else 'unknown'}",
        ]

        if sequence_score is not None:
            body_lines.append(f"SEQUENCE:   {sequence_score:.2f} (temporal ordering quality)")

        # Pattern match info if available
        pattern_match = alert_dict.get("pattern_match") or alert_dict.get("template_match")
        if pattern_match:
            win_rate = alert_dict.get("template_win_rate") or alert_dict.get("win_rate", 0)
            body_lines += [
                "",
                f"HISTORICAL: Template match found — {win_rate * 100:.0f}% win rate on similar setups",
            ]

        # Ripple effects (WorldModel downstream predictions)
        if ripple_effects:
            top_ripples = ripple_effects[:3]
            ripple_strs = [
                f"  {r.get('ticker', '?')} {r.get('direction', '?')} "
                f"(strength={r.get('strength', 0):.2f}, lag={r.get('lag_days', 0):.0f}d)"
                for r in top_ripples
                if isinstance(r, dict)
            ]
            if ripple_strs:
                body_lines += ["", "CASCADE:    Predicted downstream moves:"] + ripple_strs

        body_lines += [
            "",
            "---",
            "This is not financial advice. MIDGE surfaces patterns for your evaluation.",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        body = "\n".join(body_lines)
        sent = self.send(subject, body)

        if sent:
            # Record dedup only on successful send
            self._dedup[dedup_key] = now_ts
            # Evict stale dedup entries to prevent unbounded growth
            cutoff = now_ts - (_DEDUP_WINDOW_HOURS * 3600)
            self._dedup = {k: v for k, v in self._dedup.items() if v > cutoff}

        return sent

    def send_daily_narrative(self, narrative_text: str) -> bool:
        """Send the daily narrative letter to Guiding Light.

        The full text is passed as-is. Extracts subject from first line if it
        begins with "Subject:", otherwise uses a default subject.

        Args:
            narrative_text: Full letter text from daily_narrative.generate_daily_narrative()

        Returns:
            True if sent, False if skipped or failed.
        """
        if not self.is_configured():
            return False

        lines = narrative_text.strip().splitlines()
        subject = f"MIDGE Daily Letter — {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}"
        body = narrative_text

        # Extract subject from first line if present
        if lines and lines[0].lower().startswith("subject:"):
            subject = lines[0][len("subject:"):].strip()
            # Body starts after the subject line (skip blank line if present)
            body_start = 1
            if len(lines) > 1 and lines[1].strip() == "":
                body_start = 2
            body = "\n".join(lines[body_start:])

        return self.send(subject, body)

    def send_daily_summary(self, summary_dict: dict) -> bool:
        """Format and send a daily summary email.

        Args:
            summary_dict: Dict with keys:
                date (str), total_alerts (int), top_alerts (list[dict]),
                win_rate (float, optional), total_paper_trades (int, optional),
                domains_active (list[str], optional), regime (str, optional)

        Returns:
            True if sent, False if skipped or failed.
        """
        if not self.is_configured():
            return False

        date_str = summary_dict.get("date", datetime.now().strftime("%Y-%m-%d"))
        total_alerts = summary_dict.get("total_alerts", 0)
        top_alerts = summary_dict.get("top_alerts", [])
        win_rate = summary_dict.get("win_rate")
        total_trades = summary_dict.get("total_paper_trades", 0)
        domains_active = summary_dict.get("domains_active", [])
        regime = summary_dict.get("regime", "unknown")

        subject = f"MIDGE Daily Summary — {date_str}"

        body_lines = [
            f"MIDGE Daily Summary: {date_str}",
            "=" * 40,
            "",
            f"Market Regime: {regime.upper()}",
            f"Convergence Alerts:  {total_alerts}",
            f"Paper Trades Filed:  {total_trades}",
        ]

        if win_rate is not None:
            body_lines.append(f"Historical Win Rate: {win_rate * 100:.1f}%")

        if domains_active:
            body_lines += ["", f"Active Domains: {', '.join(domains_active)}"]

        if top_alerts:
            body_lines += ["", "TOP ALERTS TODAY:", ""]
            for i, alert in enumerate(top_alerts[:5], 1):
                ticker = alert.get("ticker", "?")
                direction = alert.get("direction", "?").upper()
                conf = float(alert.get("confidence", 0)) * 100
                domains = alert.get("domains_converging", [])
                body_lines.append(
                    f"  {i}. {ticker} {direction} — {conf:.0f}% confidence "
                    f"({len(domains)} domains: {', '.join(domains[:3])})"
                )

        body_lines += [
            "",
            "---",
            "This is not financial advice. MIDGE surfaces patterns for your evaluation.",
        ]

        return self.send(subject, "\n".join(body_lines))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_rate_limit(self) -> bool:
        """Return True if we are below the hourly send cap."""
        now_ts = time.time()
        cutoff = now_ts - 3600.0
        # Evict sends older than 1 hour
        self._send_timestamps = [t for t in self._send_timestamps if t > cutoff]
        return len(self._send_timestamps) < _MAX_EMAILS_PER_HOUR

    def _record_send(self) -> None:
        """Record a successful send for rate limiting."""
        self._send_timestamps.append(time.time())
