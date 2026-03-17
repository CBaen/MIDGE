"""Startup warmup — fills the convergence buffer from the signal archive.

MIDGE starts with 297K archived signals but wakes up blind every restart,
waiting for live signals to drip in before convergence can fire. This module
reads the last N days of the archive and injects them into the convergence
alerter's buffer so she starts from knowledge, not zero.

Key design decisions:
- Uses record_signal() directly — the exact same path live signals use.
- Respects domain-specific windows (positioning=14d, government/contracts=7d).
- Deduplicates against whatever the load_signal_buffer() call already restored.
- Does NOT call check_convergence() — the first live sensing cycle does that.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

logger = logging.getLogger("midge.market.warmup")

# Mirror of ConvergenceAlerter._domain_windows. Defined here so warmup can
# apply the same cutoffs without importing the alerter (avoids circular risk).
_DOMAIN_WINDOWS: dict[str, timedelta] = {
    "positioning": timedelta(days=14),
    "government":  timedelta(days=7),
    "contracts":   timedelta(days=7),
    "energy":      timedelta(days=7),
    # Crypto — short windows for 24/7 fast-moving market
    "crypto":           timedelta(hours=6),
    "derivatives":      timedelta(hours=4),
    "on_chain":         timedelta(hours=6),
    "defi":             timedelta(hours=12),
    "crypto_structure": timedelta(hours=24),
    "news":             timedelta(hours=6),
}
_DEFAULT_WINDOW = timedelta(hours=72)


def warm_up_from_archive(
    convergence_alerter: "ConvergenceAlerter",
    signals_dir: str | Path,
    days: int = 7,
) -> int:
    """Inject recent archived signals into the convergence buffer.

    Reads the last `days` of JSONL signal files from `signals_dir`, skips
    signals that have expired under their domain window, skips signal_ids
    already in the buffer (dedup against load_signal_buffer restore), and
    injects the rest via record_signal().

    Returns the number of signals actually injected.
    """
    signals_dir = Path(signals_dir)
    if not signals_dir.exists():
        logger.warning("Warmup: signals directory not found: %s", signals_dir)
        return 0

    now = datetime.now()
    archive_cutoff = now - timedelta(days=days)

    # Build set of signal_ids already in the buffer (from load_signal_buffer).
    existing_ids: set[str] = {
        s.signal_id
        for domain_sigs in convergence_alerter.signals.values()
        for s in domain_sigs
    }

    # Collect candidate files by date — only files within our archive window.
    candidates: list[Path] = []
    for f in signals_dir.glob("*.jsonl"):
        try:
            file_date = datetime.strptime(f.stem, "%Y-%m-%d")
            if file_date >= archive_cutoff:
                candidates.append(f)
        except ValueError:
            continue  # non-date filenames (e.g. debug dumps) — skip

    if not candidates:
        logger.info("Warmup: no archive files found in last %d days", days)
        return 0

    injected = 0
    domains_seen: set[str] = set()

    for filepath in sorted(candidates):
        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.debug("Warmup: could not read %s", filepath)
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            signal_id = rec.get("signal_id", "")
            if not signal_id or signal_id in existing_ids:
                continue

            domain = rec.get("domain", "unknown")
            window = _DOMAIN_WINDOWS.get(domain, _DEFAULT_WINDOW)
            try:
                ts = datetime.fromisoformat(rec["timestamp"])
            except (KeyError, ValueError):
                continue

            if ts < now - window:
                continue  # expired under domain window — skip

            metadata = rec.get("metadata") or {}
            symbol = rec.get("symbol", "")
            if symbol:
                metadata = {**metadata, "symbol": symbol}

            convergence_alerter.record_signal(
                signal_id=signal_id,
                strength=float(rec.get("strength", 0.5)),
                domain=domain,
                direction=rec.get("direction", "neutral"),
                confidence=float(rec.get("confidence", 0.5)),
                velocity=float(rec.get("velocity", 0.0)),
                timestamp=ts,
                metadata=metadata,
                source=rec.get("source", ""),
            )
            existing_ids.add(signal_id)
            injected += 1
            domains_seen.add(domain)

    logger.info(
        "Warmup: loaded %d signals from %d archive days across %d domains: %s",
        injected,
        days,
        len(domains_seen),
        sorted(domains_seen),
    )
    return injected
