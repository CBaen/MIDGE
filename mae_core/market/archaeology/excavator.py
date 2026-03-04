"""Excavator — reverse-engineer historical price moves into pattern fingerprints.

The Excavator is MIDGE's archaeologist. Given a dig site (a known significant
price move), it reaches back through EVERY available data source to find what
signals preceded the move. It doesn't just read the signal archive — it
actively fetches historical data from SEC EDGAR, FRED, COT, Congressional
trades, and computes retroactive TA indicators from price history.

The result: MoveFingerprints (symbol-specific observations) that get grouped
into PatternTemplates (symbol-agnostic transferable patterns). A pattern
discovered on NVDA that also appears on AAPL, AMD, and MSFT is cross-
validated — the market is telling us this pattern is real.

Statistical rigor:
  - Look-ahead bias guard: only signals with received_at < move_date
  - Small sample filter: fingerprints with < 3 precursor signals discarded
  - Cross-symbol validation: templates need 3+ symbols for confidence boost
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from mae_core.market.archaeology.dig_sites import DigSite
from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PatternTemplate,
    PrecursorSignal,
    lag_bucket_for_days,
)
from mae_core.market.archaeology.historical_fetcher import HistoricalDataFetcher

logger = logging.getLogger(__name__)

MIN_PRECURSOR_SIGNALS = 3
# Domains that apply regardless of symbol
SYMBOL_AGNOSTIC_DOMAINS = {"macro", "events", "positioning", "volatility"}


class Excavator:
    """Reverse-engineer historical price moves into pattern fingerprints and templates.

    Unlike the original version that only read signal archive files, this
    excavator actively fetches historical data from all available sources
    and computes retroactive TA indicators from price history.
    """

    def __init__(
        self,
        fetcher: Optional[HistoricalDataFetcher] = None,
        signal_dir: Optional[Path] = None,
        min_precursors: int = MIN_PRECURSOR_SIGNALS,
        regime_classifier: Any = None,
        # Data source clients (passed through to HistoricalDataFetcher)
        sec_client: Any = None,
        fred_client: Any = None,
        cot_client: Any = None,
        congress_client: Any = None,
        senate_client: Any = None,
    ):
        if fetcher is not None:
            self._fetcher = fetcher
        else:
            self._fetcher = HistoricalDataFetcher(
                signal_dir=signal_dir,
                sec_client=sec_client,
                fred_client=fred_client,
                cot_client=cot_client,
                congress_client=congress_client,
                senate_client=senate_client,
            )
        self._min_precursors = min_precursors
        self._regime_classifier = regime_classifier

        # Template accumulator: template_key -> PatternTemplate
        self._templates: dict[str, PatternTemplate] = {}
        # Stats
        self._sites_processed = 0
        self._fingerprints_produced = 0

    def excavate(
        self,
        site: DigSite,
        price_history: list,
        lookback_days: int = 30,
    ) -> Optional[MoveFingerprint]:
        """Excavate a single dig site into a fingerprint.

        Uses the HistoricalDataFetcher to gather signals from ALL sources
        (TA computation, API fetches, archive), then builds a fingerprint.

        Args:
            site: The dig site (known price move) to excavate.
            price_history: Full PriceData list for the symbol (for TA computation).
            lookback_days: How far back to search for precursor signals.

        Returns:
            MoveFingerprint if enough precursor signals found, None otherwise.
        """
        self._sites_processed += 1
        move_date = date.fromisoformat(site.move_date)
        lookback_start = move_date - timedelta(days=lookback_days)

        # Fetch ALL available historical signals for this window
        raw_signals = self._fetcher.fetch_all(
            symbol=site.symbol,
            price_history=price_history,
            start_date=lookback_start,
            end_date=move_date,
        )

        # Convert to PrecursorSignals
        precursors: list[PrecursorSignal] = []
        for sig in raw_signals:
            precursor = self._signal_to_precursor(sig, site, move_date)
            if precursor is not None:
                precursors.append(precursor)

        if len(precursors) < self._min_precursors:
            logger.debug(
                "Discarded %s %s on %s — only %d precursors (need %d)",
                site.symbol, site.direction, site.move_date,
                len(precursors), self._min_precursors,
            )
            return None

        regime = "default"
        if self._regime_classifier is not None:
            try:
                regime = self._regime_classifier.classify()
            except Exception:
                pass

        fp = MoveFingerprint(
            fingerprint_id="",  # auto-computed
            symbol=site.symbol,
            direction=site.direction,
            move_date=site.move_date,
            move_pct=site.move_pct,
            regime=regime,
            lookback_days=lookback_days,
            precursor_signals=precursors,
        )

        self._fingerprints_produced += 1

        # Group into template
        self._register_template(fp)

        logger.info(
            "Excavated %s %s %s: %.1f%% move, %d precursors, domains=%s",
            site.symbol, site.direction, site.move_date,
            site.move_pct, len(precursors), fp.domain_signature,
        )
        return fp

    def excavate_batch(
        self,
        sites: list[DigSite],
        price_history: list,
        lookback_days: int = 30,
    ) -> list[MoveFingerprint]:
        """Excavate multiple dig sites for a single symbol.

        Args:
            sites: List of dig sites (all should be for the same symbol).
            price_history: Full PriceData list for the symbol.
            lookback_days: How far back to search for precursor signals.

        Returns:
            List of valid MoveFingerprints produced.
        """
        fingerprints = []
        for site in sites:
            fp = self.excavate(site, price_history, lookback_days)
            if fp is not None:
                fingerprints.append(fp)
        logger.info(
            "Batch excavation: %d/%d sites produced valid fingerprints",
            len(fingerprints), len(sites),
        )
        return fingerprints

    def excavate_symbol(
        self,
        symbol: str,
        price_history: list,
        sites: list[DigSite],
        lookback_days: int = 30,
    ) -> list[MoveFingerprint]:
        """Excavate all dig sites for a symbol, then clear per-symbol caches.

        This is the primary entry point for batch excavation. Call once per
        symbol with the full price history and all identified dig sites.
        """
        fingerprints = self.excavate_batch(sites, price_history, lookback_days)
        self._fetcher.clear_cache()  # Free per-symbol memory
        return fingerprints

    def _signal_to_precursor(
        self,
        sig: dict,
        site: DigSite,
        move_date: date,
    ) -> Optional[PrecursorSignal]:
        """Convert a signal dict to a PrecursorSignal, computing lag."""
        sig_symbol = sig.get("symbol", "")
        sig_domain = sig.get("domain", "")

        # Must match symbol OR be symbol-agnostic domain
        if sig_symbol and sig_symbol != site.symbol and sig_domain not in SYMBOL_AGNOSTIC_DOMAINS:
            return None

        # Extract signal date
        ts = sig.get("timestamp", "") or sig.get("received_at", "")
        if not ts:
            return None
        try:
            signal_date = date.fromisoformat(ts[:10])
        except ValueError:
            return None

        # Look-ahead bias guard
        if signal_date >= move_date:
            return None

        lag_days = (move_date - signal_date).days
        if lag_days < 0 or lag_days > 60:  # Cap at 60 days (beyond extended bucket)
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

    def _register_template(self, fp: MoveFingerprint) -> None:
        """Register a fingerprint with the appropriate PatternTemplate."""
        key = fp.template_key
        if key in self._templates:
            self._templates[key].add_instance(fp)
        else:
            self._templates[key] = PatternTemplate.from_fingerprint(fp)

    @property
    def templates(self) -> dict[str, PatternTemplate]:
        """Current accumulated templates."""
        return dict(self._templates)

    def get_cross_validated_templates(self, min_symbols: int = 3) -> list[PatternTemplate]:
        """Return templates seen on at least min_symbols different stocks."""
        return [
            t for t in self._templates.values()
            if len(t.unique_symbols) >= min_symbols
        ]

    def get_statistics(self) -> dict:
        """Return excavation statistics."""
        templates = list(self._templates.values())
        cross_validated = [t for t in templates if t.cross_validated]
        return {
            "sites_processed": self._sites_processed,
            "fingerprints_produced": self._fingerprints_produced,
            "templates_discovered": len(templates),
            "cross_validated_templates": len(cross_validated),
            "avg_symbols_per_template": (
                sum(len(t.unique_symbols) for t in templates) / len(templates)
                if templates else 0.0
            ),
            "top_templates": [
                {
                    "domain_signature": t.domain_signature,
                    "direction": t.direction,
                    "n_symbols": len(t.unique_symbols),
                    "n_instances": t.n_instances,
                    "avg_move_pct": round(t.avg_move_pct, 1),
                }
                for t in sorted(templates, key=lambda x: len(x.unique_symbols), reverse=True)[:10]
            ],
        }

    def clear_cache(self):
        """Clear fetcher caches to free memory."""
        self._fetcher.clear_cache()

    def clear_all(self):
        """Clear all caches including bulk data."""
        self._fetcher.clear_all_caches()
