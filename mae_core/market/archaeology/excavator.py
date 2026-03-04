"""Excavator — reverse-engineer historical price moves into pattern fingerprints.

Given a dig site (a known significant price move), the Excavator:
1. Loads signal archive files for the lookback window
2. Filters to signals matching the symbol (or symbol-agnostic signals like macro)
3. Groups signals by lag bucket
4. Builds a MoveFingerprint encoding the precursor signal configuration

Statistical rigor:
  - Look-ahead bias guard: only signals with received_at < move_date
  - Small sample filter: fingerprints with < 3 precursor signals discarded
  - Survivorship bias: only mines symbols present in the archive
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from mae_core.market.archaeology.dig_sites import DigSite
from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PrecursorSignal,
    lag_bucket_for_days,
)

logger = logging.getLogger(__name__)

SIGNAL_ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"
MIN_PRECURSOR_SIGNALS = 3  # Minimum signals needed to form a valid fingerprint

# Domains that are symbol-agnostic (macro conditions apply to all symbols)
SYMBOL_AGNOSTIC_DOMAINS = {"macro", "events"}


class Excavator:
    """Reverse-engineer historical price moves into pattern fingerprints.

    Takes a list of dig sites and produces MoveFingerprints by reading
    the signal archive for precursor signals.
    """

    def __init__(
        self,
        signal_dir: Optional[Path] = None,
        min_precursors: int = MIN_PRECURSOR_SIGNALS,
        regime_classifier=None,
    ):
        self._signal_dir = signal_dir or SIGNAL_ARCHIVE_DIR
        self._min_precursors = min_precursors
        self._regime_classifier = regime_classifier
        self._signal_cache: dict[str, list[dict]] = {}  # date_str -> list of signal records

    def excavate(self, site: DigSite, lookback_days: int = 30) -> Optional[MoveFingerprint]:
        """Excavate a single dig site into a fingerprint.

        Args:
            site: The dig site (known price move) to excavate.
            lookback_days: How far back to search for precursor signals.

        Returns:
            MoveFingerprint if enough precursor signals found, None otherwise.
        """
        move_date = datetime.strptime(site.move_date, "%Y-%m-%d").date()
        lookback_start = move_date - timedelta(days=lookback_days)

        # Collect signals from the lookback window
        precursors: list[PrecursorSignal] = []
        for day_offset in range(lookback_days + 1):
            check_date = lookback_start + timedelta(days=day_offset)
            if check_date >= move_date:
                break
            signals = self._load_signals_for_date(check_date.isoformat())
            for sig in signals:
                precursor = self._signal_to_precursor(sig, site, move_date, check_date)
                if precursor is not None:
                    precursors.append(precursor)

        if len(precursors) < self._min_precursors:
            logger.debug(
                "Discarded %s %s on %s — only %d precursors (need %d)",
                site.symbol, site.direction, site.move_date, len(precursors), self._min_precursors,
            )
            return None

        regime = "default"
        if self._regime_classifier is not None:
            try:
                regime = self._regime_classifier.classify()
            except Exception:
                pass

        fp = MoveFingerprint(
            fingerprint_id="",  # auto-computed in __post_init__
            symbol=site.symbol,
            direction=site.direction,
            move_date=site.move_date,
            move_pct=site.move_pct,
            regime=regime,
            lookback_days=lookback_days,
            precursor_signals=precursors,
        )

        logger.info(
            "Excavated %s %s %s: %.1f%% move, %d precursors, domains=%s",
            site.symbol, site.direction, site.move_date,
            site.move_pct, len(precursors), fp.domain_signature,
        )
        return fp

    def excavate_batch(
        self,
        sites: list[DigSite],
        lookback_days: int = 30,
    ) -> list[MoveFingerprint]:
        """Excavate multiple dig sites. Returns only valid fingerprints."""
        fingerprints = []
        for site in sites:
            fp = self.excavate(site, lookback_days)
            if fp is not None:
                fingerprints.append(fp)
        logger.info(
            "Batch excavation: %d/%d sites produced valid fingerprints",
            len(fingerprints), len(sites),
        )
        return fingerprints

    def _signal_to_precursor(
        self,
        sig: dict,
        site: DigSite,
        move_date,
        signal_date,
    ) -> Optional[PrecursorSignal]:
        """Convert an archive signal record to a PrecursorSignal, if relevant."""
        sig_symbol = sig.get("symbol", "")
        sig_domain = sig.get("domain", "")

        # Filter: must match symbol OR be symbol-agnostic domain
        if sig_symbol != site.symbol and sig_domain not in SYMBOL_AGNOSTIC_DOMAINS:
            return None

        # Look-ahead bias guard: use received_at if available, else timestamp
        received_str = sig.get("received_at") or sig.get("timestamp", "")
        if received_str:
            try:
                received = datetime.fromisoformat(received_str).date()
                if received >= move_date:
                    return None  # Signal arrived after the move — look-ahead bias
            except (ValueError, TypeError):
                pass

        lag_days = (move_date - signal_date).days
        if lag_days < 0:
            return None

        return PrecursorSignal(
            source=sig.get("source", "unknown"),
            domain=sig_domain,
            direction=sig.get("direction", ""),
            strength=sig.get("strength", 0.0),
            lag_days=lag_days,
            lag_bucket=lag_bucket_for_days(lag_days),
            signal_id=sig.get("signal_id", ""),
        )

    def _load_signals_for_date(self, date_str: str) -> list[dict]:
        """Load all signals from the archive for a given date."""
        if date_str in self._signal_cache:
            return self._signal_cache[date_str]

        file_path = self._signal_dir / f"{date_str}.jsonl"
        signals: list[dict] = []
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                signals.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                logger.debug("Could not read signal file: %s", file_path)

        self._signal_cache[date_str] = signals
        return signals

    def clear_cache(self):
        """Clear the signal cache to free memory."""
        self._signal_cache.clear()
