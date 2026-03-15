"""Granger-to-WorldModel bridge — translates continuous Granger findings to daemon-readable edges.

The continuous Granger process (continuous_granger.py) runs as a separate OS process and
writes domain-level causal findings to data/market/granger_continuous.json.  The MIDGE daemon
cannot import that process directly; it reads a shared file instead.

This bridge script:
  1. Reads granger_continuous.json (the parallel process's output)
  2. Filters to significant findings (p < P_VALUE_THRESHOLD) that are new or stronger
     than what is already in granger_bridge.json
  3. Converts each finding to a WorldModel-compatible edge record (cause, effect,
     strength, lag_days, evidence="granger_continuous")
  4. Writes granger_bridge.json atomically — the daemon reads this every N steps

Strength mapping:
  F-statistic is a better proxy for effect size than p-value alone (a very large
  sample can produce p≈0 with a tiny F).  We cap at 1.0 using a soft sigmoid on
  F, combined with an evidence count factor from significant_lags / all_lags_tested.

      raw_strength = min(1.0, F / (F + 10))   # asymptotic at F→∞ → 1.0
                                               # F=10 → 0.50, F=82 → 0.89
      lag_coverage = significant_lags / max(1, all_lags_tested)
      strength = 0.4 * raw_strength + 0.6 * lag_coverage  # weight coverage more

Daemon-side consumer: _ingest_granger_bridge() in market_hooks_steps.py, called
every 200 steps inside _run_slow_cadence_ops().

Cross-process file safety:
  - Both writer (this script) and reader (daemon) use atomic rename (write to .tmp,
    then replace).  This prevents the daemon from reading a half-written file.
  - granger_bridge.json contains a "last_written" ISO timestamp; the daemon skips
    the file if the timestamp hasn't changed since last read (tracked in ctx).

Usage:
    # Run once (e.g., after continuous_granger completes a cycle):
    python -m mae_core.market.parallel.feed_granger_to_worldmodel

    # Or import and call programmatically:
    from mae_core.market.parallel.feed_granger_to_worldmodel import run_bridge
    run_bridge()

Note: This script has zero imports from the MIDGE organism — it only reads/writes
JSON.  It is safe to run while the daemon is live.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _PROJECT_ROOT / "data" / "market"
_GRANGER_CONTINUOUS_PATH = _DATA_DIR / "granger_continuous.json"
_BRIDGE_PATH = _DATA_DIR / "granger_bridge.json"

# ── Configuration ──────────────────────────────────────────────────────────────
P_VALUE_THRESHOLD = 0.05     # Only include findings below this corrected p-value
MIN_F_STATISTIC = 4.0        # Minimum F-statistic (filters noise near significance boundary)
MIN_SIGNIFICANT_LAGS = 2     # At least 2 lags must be independently significant

logger = logging.getLogger(__name__)


# ── Strength conversion ────────────────────────────────────────────────────────

def _compute_strength(f_stat: float, significant_lags: int, all_lags_tested: int) -> float:
    """Convert Granger F-statistic + lag coverage to a WorldModel edge strength [0, 1].

    Uses a soft sigmoid on F (so huge F-values don't dominate) combined with a
    lag-coverage fraction (how many tested lags were independently significant).

    Examples:
        F=10, sig=3/7  → raw=0.50, cov=0.43 → strength≈0.46
        F=30, sig=5/7  → raw=0.75, cov=0.71 → strength≈0.73
        F=82, sig=6/7  → raw=0.89, cov=0.86 → strength≈0.87
    """
    raw_strength = f_stat / (f_stat + 10.0)  # [0,1), asymptotes to 1
    lag_coverage = significant_lags / max(1, all_lags_tested)
    return min(1.0, 0.4 * raw_strength + 0.6 * lag_coverage)


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _load_continuous_findings() -> Tuple[List[dict], str]:
    """Load granger_continuous.json findings.

    Returns (findings_list, last_updated_iso) where findings_list is a list of
    raw dicts from the file.  Returns ([], "") if the file is missing or corrupt.
    """
    if not _GRANGER_CONTINUOUS_PATH.exists():
        return [], ""
    try:
        raw = json.loads(_GRANGER_CONTINUOUS_PATH.read_text(encoding="utf-8"))
        last_updated = raw.get("meta", {}).get("last_updated", "")
        return raw.get("findings", []), last_updated
    except Exception as exc:
        logger.warning("Could not read %s: %s", _GRANGER_CONTINUOUS_PATH.name, exc)
        return [], ""


def _load_existing_bridge() -> Tuple[Dict[str, dict], str]:
    """Load existing granger_bridge.json.

    Returns (edges_by_key, last_written_iso) where edges_by_key maps
    "{cause}→{effect}" to the edge dict.
    """
    if not _BRIDGE_PATH.exists():
        return {}, ""
    try:
        raw = json.loads(_BRIDGE_PATH.read_text(encoding="utf-8"))
        edges = {e["key"]: e for e in raw.get("edges", []) if "key" in e}
        last_written = raw.get("last_written", "")
        return edges, last_written
    except Exception as exc:
        logger.warning("Could not read %s: %s", _BRIDGE_PATH.name, exc)
        return {}, ""


def _save_bridge(edges: Dict[str, dict], source_last_updated: str) -> None:
    """Write granger_bridge.json atomically."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        output = {
            "last_written": datetime.utcnow().isoformat(),
            "source_last_updated": source_last_updated,
            "edge_count": len(edges),
            "description": (
                "Domain-level Granger causal edges for WorldModel injection. "
                "Written by feed_granger_to_worldmodel.py. "
                "Read by daemon every 200 steps via _ingest_granger_bridge()."
            ),
            "edges": sorted(edges.values(), key=lambda e: e.get("strength", 0), reverse=True),
        }
        tmp = _BRIDGE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
        tmp.replace(_BRIDGE_PATH)
        logger.info(
            "granger_bridge.json written: %d edges (source cycle: %s)",
            len(edges), source_last_updated,
        )
    except Exception as exc:
        logger.error("Failed to write granger_bridge.json: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def run_bridge(
    p_threshold: float = P_VALUE_THRESHOLD,
    min_f: float = MIN_F_STATISTIC,
    min_sig_lags: int = MIN_SIGNIFICANT_LAGS,
) -> int:
    """Read granger_continuous.json and write granger_bridge.json.

    Filters findings, merges with any previously written bridge edges (keeping
    the stronger strength per pair), and writes atomically.

    Returns the number of edges written to the bridge file.
    """
    findings, source_ts = _load_continuous_findings()
    if not findings:
        logger.info("No findings in granger_continuous.json — bridge unchanged")
        return 0

    existing_edges, _ = _load_existing_bridge()
    merged: Dict[str, dict] = dict(existing_edges)

    added = 0
    updated = 0
    skipped = 0

    for f in findings:
        cause = f.get("cause_domain", "")
        effect = f.get("effect_domain", "")
        p_val = f.get("p_value", 1.0)
        f_stat = f.get("f_statistic", 0.0)
        sig_lags = f.get("significant_lags", 0)
        all_lags = f.get("all_lags_tested", 1)
        lag_days = float(f.get("best_lag", 3))

        if not cause or not effect:
            skipped += 1
            continue

        # Quality gate
        if p_val >= p_threshold or f_stat < min_f or sig_lags < min_sig_lags:
            skipped += 1
            continue

        strength = _compute_strength(f_stat, sig_lags, all_lags)
        key = f"{cause}→{effect}"

        edge = {
            "key": key,
            "cause": cause,
            "effect": effect,
            "strength": round(strength, 4),
            "lag_days": lag_days,
            "evidence": "granger_continuous",
            "p_value": p_val,
            "f_statistic": f_stat,
            "significant_lags": sig_lags,
            "all_lags_tested": all_lags,
            "first_detected": f.get("first_detected", ""),
            "last_updated": f.get("last_updated", ""),
        }

        if key not in merged:
            merged[key] = edge
            added += 1
        else:
            # Keep the version with higher strength (stronger causal signal)
            if strength > merged[key].get("strength", 0.0):
                edge["first_detected"] = merged[key].get("first_detected", edge["first_detected"])
                merged[key] = edge
                updated += 1

    if added > 0 or updated > 0 or not existing_edges:
        _save_bridge(merged, source_ts)
    else:
        logger.info(
            "granger_bridge.json unchanged — no new/stronger findings "
            "(skipped=%d, existing=%d)", skipped, len(merged)
        )

    logger.info(
        "Bridge run complete: added=%d updated=%d skipped=%d total=%d",
        added, updated, skipped, len(merged),
    )
    return len(merged)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [granger_bridge] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    count = run_bridge()
    print(f"[granger_bridge] {count} edges written to granger_bridge.json")
