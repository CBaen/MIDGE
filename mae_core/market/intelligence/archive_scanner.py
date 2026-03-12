"""Archive scanner — answers "what does MIDGE know from her signal archive?"

Called at startup to log the current knowledge state before warmup runs.
Single public function: scan_archive_state(signals_dir, days) -> ArchiveState.
Under 80 lines.
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("midge.market.archive_scanner")


@dataclass
class ArchiveState:
    """Summary of what MIDGE knows from her signal archive."""
    total_signals: int
    date_range: tuple[str, str]           # (oldest_date, newest_date)
    signals_by_domain: dict[str, int]
    signals_by_ticker: dict[str, int]     # top 50 tickers
    tickers_with_multi_domain: list[str]  # tickers appearing in 3+ domains
    domain_coverage: list[str]            # which domains have data


def scan_archive_state(signals_dir: str | Path, days: int = 30) -> ArchiveState:
    """Scan the last `days` of signal archive and return a knowledge summary."""
    signals_dir = Path(signals_dir)
    now = datetime.now()
    cutoff = now - timedelta(days=days)

    by_domain: Counter = Counter()
    by_ticker: Counter = Counter()
    ticker_domains: dict[str, set] = defaultdict(set)
    oldest: str = ""
    newest: str = ""

    if not signals_dir.exists():
        logger.warning("Archive scanner: signals directory not found: %s", signals_dir)
        return ArchiveState(0, ("", ""), {}, {}, [], [])

    files_scanned = 0
    for filepath in sorted(signals_dir.glob("*.jsonl")):
        try:
            file_date = datetime.strptime(filepath.stem, "%Y-%m-%d")
        except ValueError:
            continue
        if file_date < cutoff:
            continue

        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        date_str = filepath.stem
        if not oldest or date_str < oldest:
            oldest = date_str
        if not newest or date_str > newest:
            newest = date_str
        files_scanned += 1

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            domain = rec.get("domain", "unknown")
            symbol = rec.get("symbol") or rec.get("metadata", {}).get("symbol", "")
            by_domain[domain] += 1
            if symbol:
                by_ticker[symbol] += 1
                ticker_domains[symbol].add(domain)

    multi_domain = sorted(
        sym for sym, domains in ticker_domains.items() if len(domains) >= 3
    )
    top_tickers = dict(by_ticker.most_common(50))

    state = ArchiveState(
        total_signals=sum(by_domain.values()),
        date_range=(oldest, newest),
        signals_by_domain=dict(by_domain),
        signals_by_ticker=top_tickers,
        tickers_with_multi_domain=multi_domain,
        domain_coverage=sorted(by_domain.keys()),
    )
    logger.info(
        "Archive scanner: %d signals across %d domains from %s files (%d tickers with 3+ domains)",
        state.total_signals,
        len(state.domain_coverage),
        files_scanned,
        len(multi_domain),
    )
    return state
