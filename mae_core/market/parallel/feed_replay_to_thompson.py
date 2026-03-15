"""feed_replay_to_thompson.py — Continuous Replay → Thompson bridge.

Reads ``data/midge/continuous_replay_results.jsonl`` (written by
continuous_replay.py) and aggregates graded outcomes into a compact
bridge file at ``data/market/replay_bridge.json`` that the live daemon
can ingest every 200 steps to update Thompson combo distributions.

Why a bridge file instead of direct Thompson writes:
  - continuous_replay.py runs as a SEPARATE OS process (no shared memory).
  - Writing directly to thompson_distributions.json from two processes
    would corrupt the file.  The bridge file is an atomic intermediary:
    this script writes it; the daemon reads it.
  - Same pattern as the Granger bridge (granger_bridge.json /
    feed_granger_to_worldmodel.py / _ingest_granger_bridge).

Bridge file schema (data/market/replay_bridge.json):
  {
    "last_updated": "<ISO timestamp>",
    "source_line_count": <int>,   # lines read from results JSONL
    "outcome_count": <int>,       # total success+failure records
    "replay_outcomes": [
      {
        "combo": "insider+technical",   # sorted domains joined by +
        "regime": "default",            # always "default" — replay has no regime
        "wins":   6,
        "losses": 3
      },
      ...
    ]
  }

Daemon-side consumer: _ingest_replay_bridge() in market_hooks_steps.py,
called every 200 steps inside _run_slow_cadence_ops().

Cross-process file safety:
  - Atomic write (write to .tmp, then replace) so the daemon never reads
    a half-written file.
  - last_updated ISO timestamp: daemon skips the file if unchanged since
    last read (tracked in ctx._replay_bridge_ts).

Minimum evidence gate:
  An entry is only fed to Thompson if wins + losses >= MIN_OUTCOMES.
  This prevents noise from single observations dominating weak priors.

Usage:
    # Run once after a replay cycle completes:
    python -m mae_core.market.parallel.feed_replay_to_thompson

    # Import and call programmatically (e.g., from continuous_replay.py
    # at end of each cycle):
    from mae_core.market.parallel.feed_replay_to_thompson import run_bridge
    run_bridge()

Note: This script has zero imports from the MIDGE organism — it only
reads/writes JSON/JSONL.  Safe to run while the daemon is live.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_JSONL  = _PROJECT_ROOT / "data" / "midge" / "continuous_replay_results.jsonl"
_BRIDGE_PATH    = _PROJECT_ROOT / "data" / "market" / "replay_bridge.json"
_DATA_DIR       = _PROJECT_ROOT / "data" / "market"

# ── Configuration ──────────────────────────────────────────────────────────────
# Only feed combos with at least this many graded outcomes to Thompson.
# Below this threshold the evidence is too thin to be reliable.
MIN_OUTCOMES: int = 5

logger = logging.getLogger(__name__)


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _load_existing_bridge() -> Tuple[str, int]:
    """Return (last_updated_iso, source_line_count) from the current bridge file.

    Used to detect whether the results JSONL has grown since the last run.
    Returns ("", 0) if the bridge file is missing or corrupt.
    """
    if not _BRIDGE_PATH.exists():
        return "", 0
    try:
        raw = json.loads(_BRIDGE_PATH.read_text(encoding="utf-8"))
        return raw.get("last_updated", ""), int(raw.get("source_line_count", 0))
    except Exception:
        return "", 0


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file without loading it all at once."""
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
    except Exception:
        pass
    return count


def _read_graded_records(path: Path) -> List[dict]:
    """Read all success/failure records from continuous_replay_results.jsonl.

    Skips:
      - Blank lines
      - JSON parse errors
      - cycle_summary meta-records (type == "cycle_summary")
      - Records with grade not in {success, failure}
      - Records with no usable domains list
    """
    if not path.exists():
        return []
    records: List[dict] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Skip meta-records
                if rec.get("type") == "cycle_summary":
                    continue
                grade = rec.get("grade", "")
                if grade not in ("success", "failure"):
                    continue
                domains = rec.get("domains")
                if not domains or not isinstance(domains, list):
                    continue
                records.append(rec)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path.name, exc)
    return records


def _build_combo_key(domains: list) -> str:
    """Return a stable combo key: sorted domain names joined by '+'."""
    return "+".join(sorted(str(d) for d in domains if d))


def _aggregate(records: List[dict]) -> Dict[str, Dict[str, int]]:
    """Group graded records by (combo, regime) → {wins, losses}.

    The replay process does not write a regime field, so we default to
    "default".  If a future version adds regime info, it will be picked up
    automatically here.

    Returns a dict keyed by "<combo>|<regime>" with {"wins": int, "losses": int}.
    """
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "losses": 0})
    for rec in records:
        combo = _build_combo_key(rec["domains"])
        if not combo:
            continue
        regime = rec.get("regime", "default") or "default"
        key = f"{combo}|{regime}"
        if rec["grade"] == "success":
            buckets[key]["wins"] += 1
        else:
            buckets[key]["losses"] += 1
    return dict(buckets)


def _save_bridge(
    buckets: Dict[str, Dict[str, int]],
    source_line_count: int,
) -> None:
    """Write replay_bridge.json atomically."""
    outcomes: List[dict] = []
    for key, counts in buckets.items():
        combo, _, regime = key.partition("|")
        outcomes.append({
            "combo":  combo,
            "regime": regime or "default",
            "wins":   counts["wins"],
            "losses": counts["losses"],
        })

    # Sort by total evidence descending so the daemon processes the richest
    # combos first (useful for logging / debugging).
    outcomes.sort(key=lambda o: o["wins"] + o["losses"], reverse=True)

    output = {
        "last_updated":    datetime.utcnow().isoformat(),
        "source_line_count": source_line_count,
        "outcome_count":   sum(o["wins"] + o["losses"] for o in outcomes),
        "combo_count":     len(outcomes),
        "min_outcomes_gate": MIN_OUTCOMES,
        "description": (
            "Replay outcome aggregates for Thompson combo distribution injection. "
            "Written by feed_replay_to_thompson.py. "
            "Read by daemon every 200 steps via _ingest_replay_bridge()."
        ),
        "replay_outcomes": outcomes,
    }
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _BRIDGE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(output, indent=2), encoding="utf-8")
        tmp.replace(_BRIDGE_PATH)
        logger.info(
            "replay_bridge.json written: %d combos, %d outcomes (source: %d lines)",
            len(outcomes), output["outcome_count"], source_line_count,
        )
    except Exception as exc:
        logger.error("Failed to write replay_bridge.json: %s", exc)


# ── Core logic ────────────────────────────────────────────────────────────────

def run_bridge() -> int:
    """Read continuous_replay_results.jsonl and write replay_bridge.json.

    Aggregates success/failure records by (combo, regime), then writes the
    bridge file atomically so the live daemon can ingest it.

    Incremental-safe: if the results JSONL has not grown since the last run
    (same line count), the bridge file is left unchanged.

    Returns the number of distinct combo entries written to the bridge.
    """
    if not _RESULTS_JSONL.exists():
        logger.info("continuous_replay_results.jsonl not found — bridge unchanged")
        return 0

    current_lines = _count_jsonl_lines(_RESULTS_JSONL)
    _, prev_lines = _load_existing_bridge()

    if current_lines == prev_lines and _BRIDGE_PATH.exists():
        logger.info(
            "replay_bridge.json up-to-date (%d source lines) — no change",
            current_lines,
        )
        return 0

    records = _read_graded_records(_RESULTS_JSONL)
    if not records:
        logger.info(
            "No graded records found in %s (%d total lines) — bridge unchanged",
            _RESULTS_JSONL.name, current_lines,
        )
        return 0

    buckets = _aggregate(records)
    _save_bridge(buckets, current_lines)

    eligible = sum(
        1 for v in buckets.values() if v["wins"] + v["losses"] >= MIN_OUTCOMES
    )
    logger.info(
        "Bridge run complete: %d combos total, %d meet MIN_OUTCOMES=%d threshold",
        len(buckets), eligible, MIN_OUTCOMES,
    )
    return len(buckets)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [replay_bridge] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    count = run_bridge()
    print(f"[replay_bridge] {count} combo entries written to replay_bridge.json")
