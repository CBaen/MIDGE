"""Daily stats dashboard generator — Mermaid charts for Guiding Light.

Produces data/midge/daily_stats/YYYY-MM-DD.md with:
  - Combo Win Rate Bar Chart (xychart-beta)
  - Source Reliability Bar Chart (xychart-beta)
  - Domain Signal Distribution Pie Chart
  - Causal Discovery Flowchart (Granger findings)
  - Key Numbers Table

All data comes from real files — no hardcoded sample values.
Empty/missing data sources are skipped gracefully.
Optionally renders to PNG via mmdc if available.

Design constraints:
  - Must NEVER crash the daemon — everything wrapped in try/except
  - All charts built only when source data exists and has content
  - mmdc rendering is best-effort, not required
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("midge.daily_stats")

# ── Paths ─────────────────────────────────────────────────────────────

_DATA_MARKET = Path("data/market")
_DATA_MIDGE = Path("data/midge")
_STATS_DIR = _DATA_MIDGE / "daily_stats"
_SIGNALS_DIR = _DATA_MIDGE / "signals"

# ── Helpers ───────────────────────────────────────────────────────────


def _safe_read_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.debug("Could not read %s", path, exc_info=True)
    return None


def _thompson_mean(entry: dict) -> float:
    """Extract mean reliability from a Thompson distribution entry."""
    inner = entry.get("default", entry)
    a = float(inner.get("alpha", 2.0))
    b = float(inner.get("beta", 2.0))
    total = a + b
    return a / total if total > 0 else 0.5


def _thompson_n(entry: dict) -> float:
    """Total observations (alpha + beta) for a Thompson entry."""
    inner = entry.get("default", entry)
    return float(inner.get("alpha", 2.0)) + float(inner.get("beta", 2.0))


# ── Data gathering ────────────────────────────────────────────────────


def _gather_combo_win_rates() -> list[dict]:
    """Return list of {combo, win_rate_pct, n} sorted by win_rate descending.
    Only includes combos with n >= 3 for statistical credibility.
    """
    postmortem = _safe_read_json(_DATA_MIDGE / "postmortem_continuous.json")
    if not postmortem or not isinstance(postmortem, dict):
        return []
    combo_stats = postmortem.get("combo_stats", {})
    rows = []
    for combo, stats in combo_stats.items():
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0)
        if n < 3:
            continue
        rows.append({
            "combo": combo,
            "win_rate": round(stats.get("win_rate", 0) * 100, 1),
            "n": n,
            "avg_return_pct": stats.get("avg_return_pct"),
            "grade": stats.get("grade", ""),
        })
    return sorted(rows, key=lambda x: x["win_rate"], reverse=True)


def _gather_source_reliability(top_n: int = 10) -> list[dict]:
    """Return top N sources by deviation from prior (most-learned), excluding combo keys.
    Only includes sources with n > 4 observations.
    """
    thompson = _safe_read_json(_DATA_MARKET / "thompson_distributions.json")
    if not thompson or not isinstance(thompson, dict):
        return []

    rows = []
    for key, entry in thompson.items():
        if not isinstance(entry, dict):
            continue
        # Skip combo distributions — those are in postmortem
        if key.startswith("combo:"):
            continue
        n = _thompson_n(entry)
        if n <= 4:
            continue
        mean = _thompson_mean(entry)
        deviation = abs(mean - 0.5)
        rows.append({
            "source": key,
            "reliability_pct": round(mean * 100, 1),
            "n": round(n),
            "deviation": deviation,
        })

    # Sort by deviation (most different from coin-flip = most learned)
    rows.sort(key=lambda x: x["deviation"], reverse=True)
    return rows[:top_n]


def _gather_domain_signals_7d() -> dict[str, int]:
    """Count signals by domain from signal_buffer.json (in-memory buffer).
    Falls back to scanning the archive if buffer is empty.
    """
    # Primary: signal_buffer.json (already domain-bucketed)
    buffer = _safe_read_json(_DATA_MARKET / "signal_buffer.json")
    if buffer and isinstance(buffer, dict):
        counts: dict[str, int] = {}
        for domain, signals in buffer.items():
            if isinstance(signals, list) and len(signals) > 0:
                counts[domain] = len(signals)
        if counts:
            return counts

    # Fallback: scan archive files from last 7 days
    cutoff = datetime.now() - timedelta(days=7)
    counts = {}
    try:
        for f in sorted(_SIGNALS_DIR.glob("*.jsonl")):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d")
                if file_date < cutoff:
                    continue
            except ValueError:
                continue
            try:
                with open(f, encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sig = json.loads(line)
                            dom = sig.get("domain", "unknown")
                            counts[dom] = counts.get(dom, 0) + 1
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        logger.debug("Could not scan signal archive", exc_info=True)

    return counts


def _gather_granger_findings() -> list[dict]:
    """Return Granger causality findings from granger_continuous.json."""
    granger = _safe_read_json(_DATA_MARKET / "granger_continuous.json")
    if not granger or not isinstance(granger, dict):
        return []
    return granger.get("findings", [])


def _gather_key_numbers() -> dict:
    """Collect key system metrics for the numbers table."""
    numbers: dict = {}

    # Archive signal count
    try:
        files = list(_SIGNALS_DIR.glob("*.jsonl"))
        total = 0
        for f in files:
            try:
                with open(f, encoding="utf-8") as fp:
                    total += sum(1 for line in fp if line.strip())
            except Exception:
                pass
        numbers["archive_signals"] = total
        numbers["archive_files"] = len(files)
    except Exception:
        numbers["archive_signals"] = "?"
        numbers["archive_files"] = "?"

    # Thompson distributions
    thompson = _safe_read_json(_DATA_MARKET / "thompson_distributions.json")
    if thompson and isinstance(thompson, dict):
        total_keys = len(thompson)
        # "learned" = n > 4 (moved beyond prior)
        learned = sum(
            1 for v in thompson.values()
            if isinstance(v, dict) and _thompson_n(v) > 4
        )
        numbers["thompson_total"] = total_keys
        numbers["thompson_learned"] = learned
    else:
        numbers["thompson_total"] = "?"
        numbers["thompson_learned"] = "?"

    # Granger discoveries
    granger = _safe_read_json(_DATA_MARKET / "granger_continuous.json")
    if granger and isinstance(granger, dict):
        numbers["granger_discoveries"] = len(granger.get("findings", []))
        numbers["granger_cycle"] = granger.get("meta", {}).get("cycle", "?")
    else:
        numbers["granger_discoveries"] = 0
        numbers["granger_cycle"] = "?"

    # Pattern templates
    try:
        templates_path = _DATA_MARKET / "pattern_templates.jsonl"
        if templates_path.exists():
            with open(templates_path, encoding="utf-8") as f:
                numbers["pattern_templates"] = sum(
                    1 for line in f if line.strip()
                )
        else:
            numbers["pattern_templates"] = 0
    except Exception:
        numbers["pattern_templates"] = "?"

    # Daemon uptime (steps)
    heartbeat = _safe_read_json(_DATA_MIDGE / "heartbeat.json")
    if heartbeat and isinstance(heartbeat, dict):
        numbers["daemon_steps"] = heartbeat.get("total_daemon_steps", heartbeat.get("current_step", "?"))
        numbers["daemon_round"] = heartbeat.get("daemon_round", "?")
        started_at = heartbeat.get("started_at", "")
        heartbeat_at = heartbeat.get("heartbeat_at", "")
        if started_at and heartbeat_at:
            try:
                # Use heartbeat_at - started_at to avoid local clock skew (both are daemon-reported)
                hb = datetime.fromisoformat(heartbeat_at.replace("Z", "").split("+")[0])
                st = datetime.fromisoformat(started_at.replace("Z", "").split("+")[0])
                uptime_hrs = (hb - st).total_seconds() / 3600
                numbers["daemon_uptime_hours"] = round(abs(uptime_hrs), 1)
            except Exception:
                numbers["daemon_uptime_hours"] = "?"
        else:
            numbers["daemon_uptime_hours"] = "?"
    else:
        numbers["daemon_steps"] = "?"
        numbers["daemon_round"] = "?"
        numbers["daemon_uptime_hours"] = "?"

    # Paper trades
    try:
        trades_path = _DATA_MIDGE / "paper_trades.jsonl"
        if trades_path.exists():
            with open(trades_path, encoding="utf-8") as f:
                numbers["paper_trades_total"] = sum(1 for line in f if line.strip())
        else:
            numbers["paper_trades_total"] = 0
    except Exception:
        numbers["paper_trades_total"] = "?"

    # Hypotheses
    hyp_snapshot = _safe_read_json(_DATA_MARKET / "hypotheses_snapshot.json")
    if hyp_snapshot and isinstance(hyp_snapshot, dict):
        hyps = hyp_snapshot.get("hypotheses", {})
        if isinstance(hyps, dict):
            statuses: dict[str, int] = {}
            for v in hyps.values():
                if isinstance(v, dict):
                    s = v.get("status", "unknown")
                    statuses[s] = statuses.get(s, 0) + 1
            numbers["hypotheses_active"] = statuses.get("active", 0)
            numbers["hypotheses_probation"] = statuses.get("probation", 0)
            numbers["hypotheses_retired"] = statuses.get("retired", 0)
            numbers["hypotheses_total"] = len(hyps)
        else:
            numbers["hypotheses_active"] = "?"
            numbers["hypotheses_total"] = "?"
    else:
        numbers["hypotheses_active"] = "?"
        numbers["hypotheses_total"] = "?"

    # Best combo from postmortem
    postmortem = _safe_read_json(_DATA_MIDGE / "postmortem_continuous.json")
    if postmortem and isinstance(postmortem, dict):
        best = postmortem.get("top_5_best_combos", [])
        if best and isinstance(best[0], dict):
            top = best[0]
            numbers["best_combo"] = f"{top.get('combo', '?')}: {top.get('win_rate_pct', '?')} (n={top.get('n', '?')})"
            numbers["overall_win_rate"] = postmortem.get("overall", {}).get("overall_win_rate_pct", "?")
            numbers["overall_grade"] = postmortem.get("overall", {}).get("grade", "?")
            numbers["total_graded"] = postmortem.get("overall", {}).get("total_graded", "?")
        else:
            numbers["best_combo"] = "?"
            numbers["overall_win_rate"] = "?"
    else:
        numbers["best_combo"] = "?"
        numbers["overall_win_rate"] = "?"

    return numbers


# ── Chart builders ────────────────────────────────────────────────────


def _build_combo_bar_chart(combos: list[dict]) -> Optional[str]:
    """Build xychart-beta bar chart for combo win rates.
    Returns None if no data.
    """
    if not combos:
        return None

    # Mermaid xychart-beta: label strings cannot contain + or special chars easily
    # We shorten long combos and use safe labels
    def _shorten(combo: str) -> str:
        parts = combo.split("+")
        if len(parts) <= 3:
            return combo
        # Abbreviate: first 2 + "+" count
        return "+".join(parts[:2]) + f"+{len(parts)-2}more"

    labels = [f'"{_shorten(c["combo"])}"' for c in combos[:8]]
    values = [c["win_rate"] for c in combos[:8]]

    x_axis = "[" + ", ".join(labels) + "]"
    y_values = "[" + ", ".join(str(v) for v in values) + "]"

    return f"""```mermaid
xychart-beta
    title "Win Rate by Domain Combo (n>=3)"
    x-axis {x_axis}
    y-axis "Win Rate %" 0 --> 100
    bar {y_values}
```"""


def _build_reliability_bar_chart(sources: list[dict]) -> Optional[str]:
    """Build xychart-beta bar chart for source reliability.
    Returns None if no data.
    """
    if not sources:
        return None

    # Clean up source names for display
    def _clean(name: str) -> str:
        return name.replace("_", " ").title()

    labels = [f'"{_clean(s["source"])}"' for s in sources[:10]]
    values = [s["reliability_pct"] for s in sources[:10]]

    x_axis = "[" + ", ".join(labels) + "]"
    y_values = "[" + ", ".join(str(v) for v in values) + "]"

    return f"""```mermaid
xychart-beta
    title "Source Reliability (Thompson Mean, n>4)"
    x-axis {x_axis}
    y-axis "Reliability %" 0 --> 100
    bar {y_values}
```"""


def _build_domain_pie_chart(domain_counts: dict[str, int]) -> Optional[str]:
    """Build Mermaid pie chart for domain signal distribution.
    Returns None if no data.
    """
    if not domain_counts:
        return None

    # Filter out noise domains and combine small ones
    _SKIP = {"unknown", "institutional_synthesis", "cascade"}
    counts = {
        k: v for k, v in domain_counts.items()
        if k not in _SKIP and v > 0
    }
    if not counts:
        return None

    # Sort by count, keep top 8, rest into "Other"
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top = sorted_counts[:8]
    other_total = sum(v for _, v in sorted_counts[8:])

    lines = []
    for domain, count in top:
        label = domain.replace("_", " ").title()
        lines.append(f'    "{label}" : {count}')
    if other_total > 0:
        lines.append(f'    "Other" : {other_total}')

    pie_content = "\n".join(lines)

    return f"""```mermaid
pie title "Signals by Domain (Buffer)"
{pie_content}
```"""


def _build_causal_flowchart(findings: list[dict]) -> Optional[str]:
    """Build Mermaid flowchart from Granger causality findings.
    Returns None if no findings.
    """
    if not findings:
        return None

    # Build node ID map (safe identifiers)
    node_ids: dict[str, str] = {}
    node_counter = [0]

    def _node_id(domain: str) -> str:
        if domain not in node_ids:
            safe = domain.upper().replace("_", "")
            node_ids[domain] = safe
        return node_ids[domain]

    # Build the arrow lines
    lines = []
    for f in findings[:10]:  # Cap at 10 to keep readable
        cause = f.get("cause_domain", "")
        effect = f.get("effect_domain", "")
        lag = f.get("best_lag", "?")
        if not cause or not effect:
            continue
        cause_id = _node_id(cause)
        effect_id = _node_id(effect)
        cause_label = cause.replace("_", " ").title()
        effect_label = effect.replace("_", " ").title()
        lines.append(f'    {cause_id}["{cause_label}"] -->|"{lag}d lag"| {effect_id}["{effect_label}"]')

    if not lines:
        return None

    flow_content = "\n".join(lines)

    return f"""```mermaid
flowchart LR
{flow_content}
```"""


def _build_confidence_band_chart(postmortem: dict) -> Optional[str]:
    """Build win rate by confidence band bar chart.
    Returns None if no data.
    """
    if not postmortem:
        return None
    overall = postmortem.get("overall", {})
    by_confidence = overall.get("by_confidence", {})
    if not by_confidence:
        return None

    # Sort by confidence band (low to high)
    _BAND_ORDER = ["0.45-0.50", "0.50-0.65", "0.65-0.80", "0.80+"]
    bands = []
    for band in _BAND_ORDER:
        if band in by_confidence:
            stats = by_confidence[band]
            n = stats.get("n", 0)
            if n > 0:
                wr_str = stats.get("win_rate_pct", "0%").replace("%", "")
                try:
                    wr = float(wr_str)
                except ValueError:
                    continue
                bands.append((band, wr, n))

    # Also try any bands not in the predefined order
    for band, stats in by_confidence.items():
        if band not in _BAND_ORDER and isinstance(stats, dict):
            n = stats.get("n", 0)
            if n > 0:
                wr_str = stats.get("win_rate_pct", "0%").replace("%", "")
                try:
                    wr = float(wr_str)
                    bands.append((band, wr, n))
                except ValueError:
                    pass

    if not bands:
        return None

    labels = [f'"{b} (n={n})"' for b, _, n in bands]
    values = [wr for _, wr, _ in bands]

    x_axis = "[" + ", ".join(labels) + "]"
    y_values = "[" + ", ".join(str(v) for v in values) + "]"

    return f"""```mermaid
xychart-beta
    title "Win Rate by Confidence Band"
    x-axis {x_axis}
    y-axis "Win Rate %" 0 --> 100
    bar {y_values}
```"""


# ── Key numbers table ─────────────────────────────────────────────────


def _build_key_numbers_table(numbers: dict) -> str:
    """Build the markdown key numbers table."""
    rows = [
        ("Metric", "Value"),
        ("---", "---"),
        ("Signals in archive", f"{numbers.get('archive_signals', '?'):,}" if isinstance(numbers.get('archive_signals'), int) else "?"),
        ("Archive files (days)", str(numbers.get("archive_files", "?"))),
        ("Thompson distributions total", str(numbers.get("thompson_total", "?"))),
        ("Thompson distributions learned (n>4)", str(numbers.get("thompson_learned", "?"))),
        ("Granger causal discoveries", str(numbers.get("granger_discoveries", "?"))),
        ("Granger analysis cycles", str(numbers.get("granger_cycle", "?"))),
        ("Pattern templates", str(numbers.get("pattern_templates", "?"))),
        ("Hypotheses (active)", str(numbers.get("hypotheses_active", "?"))),
        ("Hypotheses (probation)", str(numbers.get("hypotheses_probation", "?"))),
        ("Hypotheses (retired)", str(numbers.get("hypotheses_retired", "?"))),
        ("Paper trades logged", f"{numbers.get('paper_trades_total', '?'):,}" if isinstance(numbers.get('paper_trades_total'), int) else "?"),
        ("Daemon total steps", f"{numbers.get('daemon_steps', '?'):,}" if isinstance(numbers.get('daemon_steps'), int) else str(numbers.get('daemon_steps', '?'))),
        ("Daemon round", str(numbers.get("daemon_round", "?"))),
        ("Daemon uptime (hours)", str(numbers.get("daemon_uptime_hours", "?"))),
        ("Overall win rate", str(numbers.get("overall_win_rate", "?"))),
        ("Overall grade", str(numbers.get("overall_grade", "?"))),
        ("Total graded predictions", str(numbers.get("total_graded", "?"))),
        ("Best combo", str(numbers.get("best_combo", "?"))),
    ]

    table_lines = []
    for label, value in rows:
        table_lines.append(f"| {label} | {value} |")
    return "\n".join(table_lines)


# ── Document assembly ─────────────────────────────────────────────────


def _assemble_document(date_str: str) -> str:
    """Gather all data and assemble the full markdown document."""
    sections: list[str] = []

    sections.append(f"# MIDGE Daily Stats — {date_str}")
    sections.append(f"> Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sections.append("")

    # ── 1. Combo Win Rate Bar Chart ────────────────────────────────
    try:
        combos = _gather_combo_win_rates()
        chart = _build_combo_bar_chart(combos)
        if chart:
            sections.append("## Win Rate by Domain Combo")
            sections.append("")
            sections.append(chart)
            sections.append("")
            # Add data table beneath chart
            sections.append("| Combo | Win Rate | Trades | Avg Return |")
            sections.append("|-------|----------|--------|------------|")
            for c in combos[:8]:
                ret = f"{c['avg_return_pct']:.1f}%" if c.get("avg_return_pct") is not None else "—"
                sections.append(f"| {c['combo']} | {c['win_rate']}% | {c['n']} | {ret} |")
            sections.append("")
    except Exception:
        logger.debug("Combo chart failed", exc_info=True)

    # ── 2. Win Rate by Confidence Band ────────────────────────────
    try:
        postmortem = _safe_read_json(_DATA_MIDGE / "postmortem_continuous.json")
        if postmortem:
            conf_chart = _build_confidence_band_chart(postmortem)
            if conf_chart:
                sections.append("## Win Rate by Confidence Band")
                sections.append("")
                sections.append(conf_chart)
                sections.append("")
    except Exception:
        logger.debug("Confidence band chart failed", exc_info=True)

    # ── 3. Thompson Source Reliability Bar Chart ───────────────────
    try:
        sources = _gather_source_reliability(top_n=10)
        chart = _build_reliability_bar_chart(sources)
        if chart:
            sections.append("## Source Reliability (What MIDGE Has Learned)")
            sections.append("")
            sections.append(chart)
            sections.append("")
            # Add data table
            sections.append("| Source | Reliability | Observations |")
            sections.append("|--------|-------------|--------------|")
            for s in sources:
                bar = "+" * int(s["reliability_pct"] / 10) + "-" * (10 - int(s["reliability_pct"] / 10))
                sections.append(f"| {s['source']} | {s['reliability_pct']}% `{bar}` | {s['n']:,} |")
            sections.append("")
    except Exception:
        logger.debug("Source reliability chart failed", exc_info=True)

    # ── 4. Domain Signal Distribution Pie Chart ───────────────────
    try:
        domain_counts = _gather_domain_signals_7d()
        chart = _build_domain_pie_chart(domain_counts)
        if chart:
            total_sigs = sum(domain_counts.values())
            sections.append("## Signal Domain Distribution (Buffer)")
            sections.append("")
            sections.append(chart)
            sections.append("")
            # Add sorted table
            sections.append(f"Total signals in buffer: **{total_sigs:,}**")
            sections.append("")
            sections.append("| Domain | Count | Share |")
            sections.append("|--------|-------|-------|")
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
                share = round(count / total_sigs * 100, 1) if total_sigs > 0 else 0
                sections.append(f"| {domain} | {count:,} | {share}% |")
            sections.append("")
    except Exception:
        logger.debug("Domain pie chart failed", exc_info=True)

    # ── 5. Causal Discovery Flowchart ─────────────────────────────
    try:
        findings = _gather_granger_findings()
        chart = _build_causal_flowchart(findings)
        if chart:
            sections.append("## Causal Discovery Map (Granger Causality)")
            sections.append("")
            sections.append("> Arrows show statistically significant temporal causation. Lag = days before effect appears.")
            sections.append("")
            sections.append(chart)
            sections.append("")
            # Add findings table
            sections.append("| Cause | Effect | Lag (days) | F-stat | Significant Lags |")
            sections.append("|-------|--------|------------|--------|-----------------|")
            for f in findings:
                sections.append(
                    f"| {f.get('cause_domain', '?')} | {f.get('effect_domain', '?')} "
                    f"| {f.get('best_lag', '?')} | {f.get('f_statistic', '?'):.2f} "
                    f"| {f.get('significant_lags', '?')}/{f.get('all_lags_tested', '?')} |"
                )
            sections.append("")
    except Exception:
        logger.debug("Causal flowchart failed", exc_info=True)

    # ── 6. Key Numbers Table ───────────────────────────────────────
    try:
        numbers = _gather_key_numbers()
        sections.append("## Key Numbers")
        sections.append("")
        sections.append(_build_key_numbers_table(numbers))
        sections.append("")
    except Exception:
        logger.debug("Key numbers table failed", exc_info=True)

    # ── Footer ─────────────────────────────────────────────────────
    sections.append("---")
    sections.append("*MIDGE Daily Stats — all data from live files, no estimates.*")

    return "\n".join(sections)


# ── PNG rendering ─────────────────────────────────────────────────────


def _render_png(md_path: Path) -> Optional[Path]:
    """Try to render the markdown to PNG via mmdc.
    Returns PNG path on success, None on failure.
    """
    mmdc = shutil.which("mmdc")
    if not mmdc:
        logger.debug("mmdc not found — skipping PNG render")
        return None

    png_path = md_path.with_suffix(".png")
    try:
        result = subprocess.run(
            [mmdc, "-i", str(md_path), "-o", str(png_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0 and png_path.exists():
            logger.info("Daily stats PNG rendered: %s", png_path)
            return png_path
        else:
            logger.debug(
                "mmdc render failed (code %d): %s",
                result.returncode,
                result.stderr[:200],
            )
    except Exception:
        logger.debug("mmdc render error", exc_info=True)
    return None


# ── Public API ────────────────────────────────────────────────────────


def generate_daily_stats(date_str: Optional[str] = None) -> str:
    """Generate the daily stats dashboard.

    Args:
        date_str: Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        Path to the generated markdown file as a string.
        Returns empty string on total failure.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        _STATS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.warning("Could not create stats directory", exc_info=True)
        return ""

    output_path = _STATS_DIR / f"{date_str}.md"

    try:
        content = _assemble_document(date_str)
    except Exception:
        logger.warning("Daily stats assembly failed", exc_info=True)
        return ""

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Daily stats written: %s", output_path)
    except Exception:
        logger.warning("Could not write daily stats file", exc_info=True)
        return ""

    # Best-effort PNG render
    try:
        _render_png(output_path)
    except Exception:
        logger.debug("PNG render step raised unexpectedly", exc_info=True)

    return str(output_path)
