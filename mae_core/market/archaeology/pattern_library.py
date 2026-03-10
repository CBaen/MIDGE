"""Pattern Library — storage, querying, and management of fingerprints AND templates.

Two levels of storage:
  1. MoveFingerprints: symbol-specific observations in pattern_library.jsonl
  2. PatternTemplates: symbol-agnostic patterns in pattern_templates.jsonl

The PatternWatcher queries TEMPLATES, not fingerprints. Templates are the
transferable truth — a pattern that works across 47 symbols is real.
Fingerprints are the evidence that backs a template.

Memory model (lazy fingerprints):
  Templates (39 entries, ~100KB) — always in RAM. PatternWatcher needs them.
  Fingerprints (223K entries, ~100MB) — NOT loaded at startup. Kept on disk.
  A lightweight ID set (_fingerprint_ids) and count (_fingerprint_count) are
  maintained so dedup and size queries work without loading the full dataset.
  Fingerprints are loaded into RAM only when rebuild_templates() or
  update_outcome(fingerprint_id=...) is called (both are infrequent operations).

Stats tracking:
  - Win rate with Clopper-Pearson 95% CI
  - Cross-symbol validation count
  - Regime-conditional performance

Disk I/O is delegated to pattern_library_io — see that module for load/persist
function implementations.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PatternTemplate,
)
from mae_core.market.archaeology.pattern_library_io import (
    load_library,
    load_fingerprints_from_disk,
    persist_fingerprints_dict,
    persist_templates,
)

logger = logging.getLogger(__name__)

LIBRARY_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pattern_library.jsonl"
TEMPLATES_PATH = Path(__file__).resolve().parents[3] / "data" / "market" / "pattern_templates.jsonl"
DEFAULT_MATCH_THRESHOLD = 0.60


@dataclass
class PatternMatch:
    """Result of matching live signals against a template."""
    template: PatternTemplate
    match_score: float          # 0-1, fraction of template domains matched
    matched_domains: list[str]  # Which template domains matched
    missing_domains: list[str]  # Which template domains are absent
    matched_sources: list[str]  # Specific sources that matched
    missing_sources: list[str]  # For backward compatibility

    @property
    def fingerprint(self) -> Optional[MoveFingerprint]:
        """Backward compatibility: return None (templates don't have a single fingerprint)."""
        return None


class PatternLibrary:
    """Stores and queries pattern fingerprints and templates.

    Fingerprints are the raw evidence. Templates are the generalized patterns.
    The PatternWatcher queries templates (symbol-agnostic matching).

    Memory model: templates always in RAM; fingerprints kept on disk and
    loaded lazily only when rebuild_templates() or update_outcome(fingerprint_id=...)
    is called. A lightweight ID set + count are maintained for dedup and size.
    """

    def __init__(
        self,
        library_path: Optional[Path] = None,
        templates_path: Optional[Path] = None,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    ):
        self._path = library_path or LIBRARY_PATH
        self._templates_path = templates_path or TEMPLATES_PATH
        self._match_threshold = match_threshold
        # Lightweight fingerprint tracking — no full objects in RAM at startup.
        self._fingerprint_ids: set[str] = set()
        self._fingerprint_count: int = 0
        # Templates stay in RAM — they are tiny (39 entries) and PatternWatcher needs them.
        self._templates: dict[str, PatternTemplate] = {}
        # Secondary index: "direction:domain_signature" -> template_id for O(1) lookup
        self._template_key_index: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load templates from disk; scan fingerprint file for IDs and count only.

        Delegates to pattern_library_io.load_library.
        """
        fp_count, templates, template_key_index = load_library(
            self._path, self._templates_path, self._fingerprint_ids,
        )
        self._fingerprint_count = fp_count
        self._templates.update(templates)
        self._template_key_index.update(template_key_index)

    def _load_fingerprints(self) -> dict[str, MoveFingerprint]:
        """Load all fingerprints from disk into a temporary dict.

        Called only by rebuild_templates() and update_outcome(fingerprint_id=...).
        Delegates to pattern_library_io.load_fingerprints_from_disk.
        """
        return load_fingerprints_from_disk(self._path)

    def store(self, fingerprint: MoveFingerprint) -> bool:
        """Store a single fingerprint. Returns False if duplicate.

        For bulk operations, use store_batch() which batches template persistence.
        Fingerprints are appended to disk only — not kept in RAM.
        """
        if fingerprint.fingerprint_id in self._fingerprint_ids:
            return False

        self._fingerprint_ids.add(fingerprint.fingerprint_id)
        self._fingerprint_count += 1

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(fingerprint.to_json() + "\n")
        except OSError as e:
            logger.warning("Could not persist fingerprint: %s", e)

        # Update or create template and persist immediately (single-store path)
        self._update_template(fingerprint)
        self._persist_templates()
        return True

    def store_batch(self, fingerprints: list[MoveFingerprint]) -> int:
        """Store multiple fingerprints. Persists templates ONCE at the end.

        Fingerprints are appended to disk only — not kept in RAM.
        """
        stored = 0
        for fp in fingerprints:
            if fp.fingerprint_id in self._fingerprint_ids:
                continue
            self._fingerprint_ids.add(fp.fingerprint_id)
            self._fingerprint_count += 1
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a") as f:
                    f.write(fp.to_json() + "\n")
            except OSError as e:
                logger.warning("Could not persist fingerprint: %s", e)
            self._update_template(fp)
            stored += 1
        if stored > 0:
            self._persist_templates()
        return stored

    def store_template(self, template: PatternTemplate) -> bool:
        """Store or update a template. Returns True if new."""
        is_new = template.template_id not in self._templates
        self._templates[template.template_id] = template
        self._template_key_index[f"{template.direction}:{template.domain_signature}"] = template.template_id
        self._persist_templates()
        return is_new

    def store_templates(self, templates: dict[str, PatternTemplate]) -> int:
        """Store multiple templates. Returns count of new entries."""
        new_count = 0
        for t in templates.values():
            if t.template_id not in self._templates:
                new_count += 1
            self._templates[t.template_id] = t
            self._template_key_index[f"{t.direction}:{t.domain_signature}"] = t.template_id
        self._persist_templates()
        return new_count

    def _update_template(self, fingerprint: MoveFingerprint) -> None:
        """Update or create the template for a fingerprint.

        Does NOT persist — caller must call _persist_templates() when ready.
        This prevents 216K+ full rewrites during batch operations.
        Uses O(1) key index instead of linear scan over all templates.
        """
        key = f"{fingerprint.direction}:{fingerprint.domain_signature}"
        tid = self._template_key_index.get(key)
        if tid and tid in self._templates:
            self._templates[tid].add_instance(fingerprint)
            return

        # Create new template
        template = PatternTemplate.from_fingerprint(fingerprint)
        self._templates[template.template_id] = template
        self._template_key_index[key] = template.template_id

    def get(self, fingerprint_id: str) -> Optional[MoveFingerprint]:
        """Fetch a single fingerprint by ID via targeted disk scan.

        Called infrequently (outcome grading, tests). Scans the JSONL file
        rather than keeping all 223K entries in RAM.
        """
        if fingerprint_id not in self._fingerprint_ids:
            return None
        if not self._path.exists():
            return None
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("fingerprint_id") == fingerprint_id:
                            return MoveFingerprint.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as e:
            logger.warning("Could not scan fingerprint file: %s", e)
        return None

    def get_template(self, template_id: str) -> Optional[PatternTemplate]:
        return self._templates.get(template_id)

    @property
    def size(self) -> int:
        """Return fingerprint count without loading fingerprints into RAM."""
        return self._fingerprint_count

    @property
    def template_count(self) -> int:
        return len(self._templates)

    def query_similar(
        self,
        live_sources: set[str],
        symbol: str = "",
        direction: str = "",
        regime: str = "default",
    ) -> list[PatternMatch]:
        """Find templates whose domain profile matches live signal state.

        This is the key change from v1: we match against TEMPLATES (symbol-
        agnostic) instead of fingerprints (symbol-specific). A pattern
        discovered on NVDA can now activate on ANY symbol.

        Args:
            live_sources: Set of source names currently active
                          (e.g. {"sec_form4", "fred_macro", "ta_rsi"})
            symbol: Ticker symbol (for logging, NOT for filtering).
            direction: "bullish" or "bearish".
            regime: Current market regime (optional filter).

        Returns:
            List of PatternMatch objects sorted by match_score descending.
        """
        # Build domain set from live sources
        live_domains = self._sources_to_domains(live_sources)

        matches: list[PatternMatch] = []

        for template in self._templates.values():
            # Direction must match
            if template.direction != direction:
                continue

            template_domains = set(template.domains)
            if not template_domains:
                continue

            # Domain overlap matching
            matched_domains = template_domains & live_domains
            missing_domains = template_domains - live_domains

            if not matched_domains:
                continue

            # Base: fraction of domains matched
            domain_score = len(matched_domains) / len(template_domains)

            # Boost for cross-validated templates
            cross_boost = template.confidence_multiplier

            score = domain_score * cross_boost

            if score >= self._match_threshold:
                # Find which specific sources matched for reporting
                matched_sources = sorted(s for s in live_sources if self._source_domain(s) in matched_domains)
                missing_source_domains = sorted(missing_domains)

                matches.append(PatternMatch(
                    template=template,
                    match_score=round(min(score, 1.0), 3),
                    matched_domains=sorted(matched_domains),
                    missing_domains=sorted(missing_domains),
                    matched_sources=matched_sources,
                    missing_sources=missing_source_domains,
                ))

        matches.sort(key=lambda m: m.match_score, reverse=True)
        return matches

    def update_outcome(
        self,
        fingerprint_id: str = "",
        template_id: str = "",
        won: bool = False,
        return_pct: float = 0.0,
    ) -> None:
        """Update stats after an outcome is graded.

        When fingerprint_id is provided, loads ALL fingerprints from disk,
        updates the target entry, and rewrites the file. This is an infrequent
        operation (outcome grading cadence), so the full-load cost is acceptable.
        """
        if fingerprint_id:
            fingerprints = self._load_fingerprints()
            fp = fingerprints.get(fingerprint_id)
            if fp is not None:
                fp.total_activations += 1
                if won:
                    fp.wins += 1
                else:
                    fp.losses += 1
                fp.last_activation = datetime.now().isoformat()
                self._persist_fingerprints_dict(fingerprints)

        if template_id:
            t = self._templates.get(template_id)
            if t is not None:
                if won:
                    t.wins += 1
                else:
                    t.losses += 1
                self._persist_templates()

    def _persist_fingerprints_dict(self, fingerprints: dict[str, MoveFingerprint]) -> None:
        """Rewrite fingerprints file atomically from a provided dict."""
        if not fingerprints:
            return  # Never overwrite with empty
        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for fp in fingerprints.values():
                    f.write(fp.to_json() + "\n")
            try:
                tmp.replace(self._path)
            except OSError:
                with open(self._path, "w") as f:
                    with open(tmp, "r") as src:
                        f.write(src.read())
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except OSError as e:
            logger.warning("Could not persist fingerprints: %s", e)

    def _persist_templates(self) -> None:
        """Rewrite templates file atomically (write .tmp then rename).

        On Windows, rename can fail if another process holds the target file.
        Falls back to direct overwrite (still safe — we have the empty guard).
        """
        if not self._templates:
            return  # Never overwrite with empty — protects against crash-induced data loss
        try:
            self._templates_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._templates_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for t in self._templates.values():
                    f.write(t.to_json() + "\n")
            try:
                tmp.replace(self._templates_path)
            except OSError:
                # Windows: target file locked by another process — direct write fallback
                with open(self._templates_path, "w") as f:
                    with open(tmp, "r") as src:
                        f.write(src.read())
                try:
                    tmp.unlink()
                except OSError:
                    pass
        except OSError as e:
            logger.warning("Could not persist templates: %s", e)

    def rebuild_templates(self) -> int:
        """Rebuild all templates from fingerprints.

        Useful when templates file is empty/corrupted but fingerprints
        are intact. Groups fingerprints by direction+domain_signature,
        builds PatternTemplate objects, and persists.

        Loads fingerprints from disk for this operation, then releases them.

        Returns:
            Number of templates rebuilt.
        """
        fingerprints = self._load_fingerprints()
        self._templates.clear()
        self._template_key_index.clear()
        for fp in fingerprints.values():
            self._update_template(fp)
        self._persist_templates()
        logger.info("Rebuilt %d templates from %d fingerprints",
                     len(self._templates), len(fingerprints))
        return len(self._templates)

    def get_statistics(self) -> dict:
        """Return summary statistics about the library."""
        templates = list(self._templates.values())
        cross_validated = [t for t in templates if t.cross_validated]
        with_outcomes = [t for t in templates if t.wins + t.losses > 0]

        return {
            "total_fingerprints": self._fingerprint_count,
            "total_templates": len(templates),
            "cross_validated_templates": len(cross_validated),
            "templates_with_outcomes": len(with_outcomes),
            "top_templates": [
                {
                    "domain_signature": t.domain_signature,
                    "direction": t.direction,
                    "symbols": len(t.unique_symbols),
                    "instances": t.n_instances,
                    "win_rate": round(t.win_rate, 3) if t.wins + t.losses > 0 else None,
                }
                for t in sorted(templates, key=lambda x: x.n_instances, reverse=True)[:10]
            ],
        }

    def clopper_pearson_ci(self, template_id: str = "", fingerprint_id: str = "", alpha: float = 0.05) -> tuple[float, float]:
        """Compute Clopper-Pearson CI for win rate."""
        if template_id:
            t = self._templates.get(template_id)
            k = t.wins if t else 0
            n = (t.wins + t.losses) if t else 0
        elif fingerprint_id:
            fp = self.get(fingerprint_id)
            k = fp.wins if fp else 0
            n = (fp.wins + fp.losses) if fp else 0
        else:
            return (0.0, 1.0)

        if n == 0:
            return (0.0, 1.0)

        try:
            from scipy.stats import beta as beta_dist
            lower = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
            upper = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
            return (round(lower, 4), round(upper, 4))
        except ImportError:
            p = k / n
            z = 1.96
            denom = 1 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
            return (round(max(0, center - spread), 4), round(min(1, center + spread), 4))

    # ── Domain mapping ────────────────────────────────────────────────────

    # Source → domain mapping (covers all 30 known sources)
    _SOURCE_DOMAIN_MAP = {
        "sec_form4": "insider", "openinsider_purchase": "insider",
        "sec_form8k": "events", "sec_efts": "events",
        "finnhub_earnings": "events", "finnhub_news": "events",
        "finnhub_realtime": "events", "finnhub_earnings_calendar": "events",
        "massive_snapshot": "events", "hiring_tracker": "events",
        "fred_macro": "macro", "economic_calendar": "macro",
        "ta_rsi": "technical", "ta_macd": "technical", "ta_bollinger": "technical",
        "ta_structure": "technical", "ta_candle": "technical",
        "session_sweep": "technical", "session_sweep_ifvg": "technical",
        "fractal_resonance": "technical", "order_flow": "technical",
        "finviz_unusual_volume": "technical", "finviz_short_squeeze": "technical",
        "social_sentiment": "sentiment", "google_trends": "sentiment",
        "stocktwits_sentiment": "sentiment",
        "congressional": "government", "congress_legislation": "government",
        "contract_award": "contracts",
        "sam_gov": "contracts",
        "finnhub_analyst": "fundamentals",
        "cot_positioning": "positioning",
        "vix_term_structure": "volatility",
        "crypto_coingecko": "crypto", "crypto_coincap": "crypto",
        "activist_13d": "institutional", "institutional_13f": "institutional",
        "finra_short": "institutional",
        "yfinance_price": "technical",
        "absence_signal": "meta",
        "archetype_match": "meta",
        "convergence_alert": "convergence",
        # Real-economy
        "eia_energy": "energy",
    }

    @classmethod
    def _source_domain(cls, source: str) -> str:
        """Map a source name to its domain."""
        return cls._SOURCE_DOMAIN_MAP.get(source, "unknown")

    @classmethod
    def _sources_to_domains(cls, sources: set[str]) -> set[str]:
        """Convert a set of source names to their domains."""
        return {cls._source_domain(s) for s in sources} - {"unknown"}
