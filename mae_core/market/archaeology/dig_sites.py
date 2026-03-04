"""Dig Sites — automatic identification of significant historical price moves.

Takes a symbol, fetches price history, and identifies "dig sites" — dates
where the price made a significant move. These become inputs for the
Excavator, which looks backward through the signal archive to find what
preceded each move.

Multi-day compound moves are detected: 3 consecutive days each moving 2%+
in the same direction = a single dig site with the compound magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DigSite:
    """A significant historical price move worth excavating."""
    symbol: str
    direction: str       # "bullish" or "bearish"
    move_date: str       # ISO date (YYYY-MM-DD) — the END of the move
    move_start: str      # ISO date — the START of the move (for multi-day)
    move_pct: float      # Total percentage change
    move_days: int = 1   # Number of trading days the move spanned
    entry_price: float = 0.0
    exit_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "move_date": self.move_date,
            "move_start": self.move_start,
            "move_pct": self.move_pct,
            "move_days": self.move_days,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
        }


def find_dig_sites(
    price_history: list,
    symbol: str,
    min_single_day_pct: float = 5.0,
    min_compound_pct: float = 6.0,
    min_compound_daily_pct: float = 2.0,
    min_gap_days: int = 5,
    max_compound_days: int = 5,
) -> list[DigSite]:
    """Find significant price moves in a price history.

    Args:
        price_history: List of PriceData objects (oldest first), must have .price and .timestamp.
        symbol: Ticker symbol.
        min_single_day_pct: Minimum single-day move to qualify (%).
        min_compound_pct: Minimum total compound move to qualify (%).
        min_compound_daily_pct: Minimum daily move to extend a compound run (%).
        min_gap_days: Minimum trading days between dig sites (avoid overlapping windows).
        max_compound_days: Maximum days to extend a compound move.

    Returns:
        List of DigSite objects sorted by move_date.
    """
    if len(price_history) < 2:
        return []

    sites: list[DigSite] = []
    last_site_idx = -min_gap_days  # Allow first site immediately

    # Compute daily returns
    daily_returns: list[tuple[int, float, str]] = []  # (index, pct_change, date_str)
    for i in range(1, len(price_history)):
        prev_price = price_history[i - 1].price
        curr_price = price_history[i].price
        if prev_price <= 0:
            continue
        pct = (curr_price - prev_price) / prev_price * 100
        ts = getattr(price_history[i], "timestamp", "")
        daily_returns.append((i, pct, ts))

    i = 0
    while i < len(daily_returns):
        idx, pct, ts = daily_returns[i]

        # Check gap from last site
        if idx - last_site_idx < min_gap_days:
            i += 1
            continue

        abs_pct = abs(pct)

        # Single-day significant move
        if abs_pct >= min_single_day_pct:
            direction = "bullish" if pct > 0 else "bearish"
            site = DigSite(
                symbol=symbol,
                direction=direction,
                move_date=_extract_date(ts),
                move_start=_extract_date(ts),
                move_pct=round(pct, 2),
                move_days=1,
                entry_price=price_history[idx - 1].price,
                exit_price=price_history[idx].price,
            )
            sites.append(site)
            last_site_idx = idx
            i += 1
            continue

        # Try to build a compound move
        if abs_pct >= min_compound_daily_pct:
            compound_direction = 1 if pct > 0 else -1
            compound_start_idx = idx
            compound_start_ts = ts
            compound_end_idx = idx
            compound_end_ts = ts
            compound_total = pct
            j = i + 1

            while j < len(daily_returns) and (j - i) < max_compound_days:
                next_idx, next_pct, next_ts = daily_returns[j]
                next_dir = 1 if next_pct > 0 else -1
                if next_dir == compound_direction and abs(next_pct) >= min_compound_daily_pct:
                    compound_total += next_pct
                    compound_end_idx = next_idx
                    compound_end_ts = next_ts
                    j += 1
                else:
                    break

            if abs(compound_total) >= min_compound_pct and (j - i) >= 2:
                direction = "bullish" if compound_total > 0 else "bearish"
                site = DigSite(
                    symbol=symbol,
                    direction=direction,
                    move_date=_extract_date(compound_end_ts),
                    move_start=_extract_date(compound_start_ts),
                    move_pct=round(compound_total, 2),
                    move_days=j - i,
                    entry_price=price_history[compound_start_idx - 1].price,
                    exit_price=price_history[compound_end_idx].price,
                )
                sites.append(site)
                last_site_idx = compound_end_idx
                i = j
                continue

        i += 1

    logger.info("Found %d dig sites for %s (from %d price points)", len(sites), symbol, len(price_history))
    return sites


def _extract_date(timestamp_str: str) -> str:
    """Extract YYYY-MM-DD from a timestamp string."""
    if not timestamp_str:
        return ""
    return timestamp_str[:10]
