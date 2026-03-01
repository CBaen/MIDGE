"""Marathon Post-Mortem Report Generator.

Reads persisted MIDGE state and generates a structured field report
showing what MIDGE learned during the last marathon run.

Usage:
    python marathon_report.py                    # Generate from current data
    python marathon_report.py --output path.md   # Custom output path

Also callable from main.py:
    from marathon_report import generate_report
    generate_report(output_path="data/midge/marathon-report.md")
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

DATA_MARKET = Path(__file__).parent / "data" / "market"
DATA_MIDGE = Path(__file__).parent / "data" / "midge"

# Gate defaults for comparison in Section 4
_GATE_DEFAULTS = {
    "min_observations": 20,
    "promote_win_rate": 0.52,
    "promote_dsr": 0.5,
    "retire_win_rate": 0.45,
    "retire_dsr": 0.0,
}


# ---------------------------------------------------------------------------
# Section 1: Run Vitals
# ---------------------------------------------------------------------------

def _render_vitals(round_times: list[float] | None) -> str:
    """Parse last row of growth-tracker.md and show run vitals."""
    lines = ["## 1. Run Vitals\n"]

    try:
        tracker = DATA_MIDGE / "growth-tracker.md"
        text = tracker.read_text(encoding="utf-8")
        rows = [ln for ln in text.splitlines() if ln.startswith("|") and "---" not in ln]
        # Skip the header row (first row)
        data_rows = [r for r in rows if "When" not in r]
        if not data_rows:
            lines.append("*No growth-tracker rows found.*\n")
            return "\n".join(lines)

        last = data_rows[-1]
        # Pipe-separated: | When | Steps | Agents | Run Reward | Lifetime Reward | Memories | Policies | Oracle Calls | Self-Sufficient | Phi | Actions |
        cols = [c.strip() for c in last.split("|")]
        cols = [c for c in cols if c]  # remove empty from leading/trailing |

        when        = cols[0]  if len(cols) > 0 else "?"
        steps       = cols[1]  if len(cols) > 1 else "?"
        agents      = cols[2]  if len(cols) > 2 else "?"
        run_reward  = cols[3]  if len(cols) > 3 else "?"
        lifetime    = cols[4]  if len(cols) > 4 else "?"
        memories    = cols[5]  if len(cols) > 5 else "?"
        self_suf    = cols[8]  if len(cols) > 8 else "?"
        phi         = cols[9]  if len(cols) > 9 else "?"
        actions     = cols[10] if len(cols) > 10 else "?"

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Last run | {when} |")
        lines.append(f"| Steps | {steps} |")
        lines.append(f"| Agents | {agents} |")
        lines.append(f"| Run reward | {run_reward} |")
        lines.append(f"| Lifetime reward | {lifetime} |")
        lines.append(f"| Episodic memories | {memories} |")
        lines.append(f"| Self-sufficient | {self_suf} |")
        lines.append(f"| Phi (integration) | {phi} |")
        lines.append(f"| Action distribution | {actions} |")
        lines.append("")

    except Exception:
        lines.append("*Data unavailable.*\n")
        return "\n".join(lines)

    if round_times:
        total = sum(round_times)
        avg = total / len(round_times)
        lines.append("| Timing | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Rounds completed | {len(round_times)} |")
        lines.append(f"| Total duration | {total:.1f}s ({total/3600:.2f}h) |")
        lines.append(f"| Avg per round | {avg:.1f}s |")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 2: Market Intelligence
# ---------------------------------------------------------------------------

def _render_market_intelligence() -> str:
    """Show convergence state and recent discovery counts."""
    lines = ["## 2. Market Intelligence\n"]

    try:
        cs_path = DATA_MIDGE / "convergence_state.json"
        cs = json.loads(cs_path.read_text(encoding="utf-8"))

        regime = cs.get("regime", "?")
        g = cs.get("global", {})
        direction = g.get("direction", "?")
        strength = g.get("strength", 0.0)
        domains = g.get("domains", 0)

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Regime | {regime} |")
        lines.append(f"| Global direction | {direction} |")
        lines.append(f"| Global strength | {strength:.3f} |")
        lines.append(f"| Domains converging | {domains} |")

        tactical = cs.get("tactical")
        if tactical:
            lines.append(f"| Tactical alert | {tactical.get('summary', '?')} |")
            lines.append(f"| Tactical urgency | {tactical.get('urgency', '?')} |")

        ticker_alerts = cs.get("ticker_alerts", {})
        if ticker_alerts:
            tickers_str = ", ".join(f"{k}={v}" for k, v in ticker_alerts.items())
            lines.append(f"| Per-ticker alerts | {tickers_str} |")

        hyp = cs.get("hypotheses", {})
        if hyp:
            lines.append(f"| Hypotheses active | {hyp.get('active', 0)} |")
            lines.append(f"| Hypotheses generated | {hyp.get('generated', 0)} |")
            lines.append(f"| Hypotheses promoted | {hyp.get('promoted', 0)} |")

        lines.append("")

    except Exception:
        lines.append("*Convergence state unavailable.*\n")

    # Recent discovery count (last 72h)
    try:
        discovery_path = DATA_MARKET / "discovery_log.jsonl"
        cutoff = datetime.now() - timedelta(hours=72)
        recent_count = 0
        total_count = 0
        with open(discovery_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                total_count += 1
                try:
                    rec = json.loads(raw)
                    ts_str = rec.get("timestamp", "")
                    if ts_str:
                        ts = datetime.fromisoformat(ts_str)
                        if ts >= cutoff:
                            recent_count += 1
                except Exception:
                    pass
        lines.append(f"Discovery log: **{recent_count}** alerts in last 72h, **{total_count}** total\n")
    except Exception:
        lines.append("*Discovery log unavailable.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 3: Bayesian Learning
# ---------------------------------------------------------------------------

def _render_bayesian_learning() -> str:
    """Show Thompson distribution win rates and calibration status."""
    lines = ["## 3. Bayesian Learning\n"]

    try:
        td_path = DATA_MARKET / "thompson_distributions.json"
        distributions = json.loads(td_path.read_text(encoding="utf-8"))

        # Parse each key; each has at least a "default" regime bucket.
        # Use the "default" bucket as the canonical value for ranking.
        entries: list[dict] = []
        for key, regimes in distributions.items():
            default_bucket = regimes.get("default", {})
            if not default_bucket:
                # some keys only have regime-specific buckets
                # pick the first available bucket
                default_bucket = next(iter(regimes.values()), {})
            alpha = default_bucket.get("alpha", 1.0)
            beta_val = default_bucket.get("beta", 1.0)
            samples = (alpha + beta_val) - 2  # subtract priors
            if samples < 0:
                samples = 0
            mean = alpha / (alpha + beta_val) if (alpha + beta_val) > 0 else 0.5
            entries.append({
                "key": key,
                "mean": mean,
                "samples": samples,
                "alpha": alpha,
                "beta": beta_val,
            })

        # Filter for minimum 10 samples
        qualified = [e for e in entries if e["samples"] >= 10]
        qualified.sort(key=lambda x: x["mean"], reverse=True)

        top5 = qualified[:5]
        bottom5 = list(reversed(qualified))[:5]

        lines.append("### Top 5 Sources by Win Rate (min 10 samples)")
        lines.append("")
        if top5:
            lines.append("| Source | Win Rate | Samples |")
            lines.append("|--------|----------|---------|")
            for e in top5:
                lines.append(f"| {e['key']} | {e['mean']:.1%} | {e['samples']:.0f} |")
        else:
            lines.append("*No sources with 10+ samples yet.*")
        lines.append("")

        lines.append("### Bottom 5 Sources by Win Rate (min 10 samples)")
        lines.append("")
        if bottom5:
            lines.append("| Source | Win Rate | Samples |")
            lines.append("|--------|----------|---------|")
            for e in bottom5:
                lines.append(f"| {e['key']} | {e['mean']:.1%} | {e['samples']:.0f} |")
        else:
            lines.append("*No sources with 10+ samples yet.*")
        lines.append("")

        total_dist = len(entries)
        total_samples = sum(e["samples"] for e in entries)
        lines.append(f"Total: **{total_dist}** distributions, **{total_samples:.0f}** accumulated samples\n")

    except Exception:
        lines.append("*Thompson distributions unavailable.*\n")

    # Calibration report
    try:
        cal_path = DATA_MARKET / "calibration_report.json"
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        sources = cal.get("sources", [])
        overconfident = [s for s in sources if s.get("is_overconfident")]
        underconfident = [s for s in sources if s.get("is_underconfident")]

        lines.append("### Calibration Status")
        lines.append("")
        if not sources:
            lines.append("*No calibration data yet.*")
        else:
            if overconfident:
                lines.append(f"**Overconfident ({len(overconfident)}):** " +
                              ", ".join(s["source"] for s in overconfident))
            if underconfident:
                lines.append(f"**Underconfident ({len(underconfident)}):** " +
                              ", ".join(s["source"] for s in underconfident))
            if not overconfident and not underconfident:
                lines.append("All calibrated sources are within tolerance.")
            lines.append("")
            lines.append(f"*Generated: {cal.get('generated_at', '?')}*")
        lines.append("")

    except Exception:
        lines.append("*Calibration report unavailable.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 4: Hypothesis Pipeline
# ---------------------------------------------------------------------------

def _render_hypothesis_pipeline() -> str:
    """Replay hypothesis events to count statuses; show gate values."""
    lines = ["## 4. Hypothesis Pipeline\n"]

    # Replay hypothesis events
    try:
        hyp_path = DATA_MARKET / "hypotheses.jsonl"
        # Track current status per hypothesis_id
        statuses: dict[str, str] = {}
        with open(hyp_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    event = rec.get("event", "")
                    hyp = rec.get("hypothesis", {})
                    hid = hyp.get("hypothesis_id", "")
                    status = hyp.get("status", "")
                    if hid:
                        statuses[hid] = status
                except Exception:
                    pass

        counts: dict[str, int] = {}
        for s in statuses.values():
            counts[s] = counts.get(s, 0) + 1

        total = sum(counts.values())
        lines.append(f"Total hypotheses tracked: **{total}**\n")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        for status in ("active", "probation", "hibernated", "retired"):
            lines.append(f"| {status.capitalize()} | {counts.get(status, 0)} |")
        lines.append("")

    except Exception:
        lines.append("*Hypothesis JSONL unavailable.*\n")

    # DSR state
    try:
        dsr_path = DATA_MARKET / "dsr_state.json"
        dsr = json.loads(dsr_path.read_text(encoding="utf-8"))
        trials = dsr.get("trials_tracked", 0)
        updated = dsr.get("last_updated", "?")
        lines.append(f"Global DSR trial counter: **{trials}** | Last updated: {updated}\n")
    except Exception:
        lines.append("*DSR state unavailable.*\n")

    # Config snapshot — hypothesis gates
    try:
        snap_path = DATA_MARKET / "config_snapshot.json"
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        gates = snap.get("hypothesis_gates", {})

        if gates:
            lines.append("### Hypothesis Gates")
            lines.append("")
            lines.append("| Gate | Current | Default | Changed |")
            lines.append("|------|---------|---------|---------|")
            for gate_key, default_val in _GATE_DEFAULTS.items():
                current_val = gates.get(gate_key, default_val)
                changed = "YES" if abs(float(current_val) - float(default_val)) > 1e-9 else "no"
                lines.append(f"| {gate_key} | {current_val} | {default_val} | {changed} |")
            lines.append("")
        else:
            lines.append("*No hypothesis_gates found in config snapshot.*\n")

    except FileNotFoundError:
        lines.append("*Config snapshot not yet written (runs after first marathon).*\n")
    except Exception:
        lines.append("*Config snapshot unavailable.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 5: Position Sizing
# ---------------------------------------------------------------------------

def _render_position_sizing() -> str:
    """Show last 20 position sizing decisions and tier counts."""
    lines = ["## 5. Position Sizing\n"]

    try:
        ps_path = DATA_MARKET / "position_sizing_log.jsonl"
        records: list[dict] = []
        with open(ps_path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except Exception:
                    pass

        last20 = records[-20:]

        if not last20:
            lines.append("*No position sizing records yet.*\n")
            return "\n".join(lines)

        lines.append("| Source | Symbol | Kelly Capped | Confidence |")
        lines.append("|--------|--------|-------------|------------|")
        for rec in last20:
            source = rec.get("signal_source", "?")
            symbol = rec.get("symbol", "?")
            kelly = rec.get("kelly_capped", 0.0)
            tier = rec.get("confidence_in_sizing", "?")
            lines.append(f"| {source} | {symbol} | {kelly:.4f} | {tier} |")
        lines.append("")

        # Count by confidence tier across all records
        tier_counts: dict[str, int] = {}
        for rec in records:
            tier = rec.get("confidence_in_sizing", "unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        lines.append(f"Total sizing decisions: **{len(records)}**\n")
        lines.append("| Confidence Tier | Count |")
        lines.append("|----------------|-------|")
        for tier in ("high", "medium", "low"):
            lines.append(f"| {tier.capitalize()} | {tier_counts.get(tier, 0)} |")
        lines.append("")

    except Exception:
        lines.append("*Data unavailable.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section 6: Performance Health
# ---------------------------------------------------------------------------

def _render_performance_health() -> str:
    """Show step timer percentiles and prediction resolution rate."""
    lines = ["## 6. Performance Health\n"]

    # Step timer snapshot
    try:
        timer_path = DATA_MIDGE / "step_timer_snapshot.json"
        snap = json.loads(timer_path.read_text(encoding="utf-8"))
        saved_at = snap.get("saved_at", "?")
        stats = snap.get("stats", {})

        lines.append(f"*Timer snapshot: {saved_at}*\n")
        if stats:
            lines.append("| Operation | p50 (ms) | p95 (ms) | Max (ms) | Count |")
            lines.append("|-----------|----------|----------|----------|-------|")
            for op, s in sorted(stats.items()):
                p50 = s.get("p50_ms", 0.0)
                p95 = s.get("p95_ms", 0.0)
                mx = s.get("max_ms", 0.0)
                cnt = s.get("count", 0)
                lines.append(f"| {op} | {p50:.1f} | {p95:.1f} | {mx:.1f} | {cnt} |")
            lines.append("")
        else:
            lines.append("*No operations timed yet.*\n")

    except FileNotFoundError:
        lines.append("*Step timer snapshot not yet written (generated at end of run).*\n")
    except Exception:
        lines.append("*Step timer data unavailable.*\n")

    # Prediction resolution rate
    try:
        pred_path = DATA_MARKET / "predictions.jsonl"
        out_path = DATA_MARKET / "outcomes.jsonl"

        pred_count = sum(1 for ln in open(pred_path, encoding="utf-8") if ln.strip())
        out_count = sum(1 for ln in open(out_path, encoding="utf-8") if ln.strip())

        if pred_count > 0:
            rate = out_count / pred_count
            lines.append(f"Prediction resolution: **{out_count}** outcomes / "
                         f"**{pred_count}** predictions = **{rate:.1%}**\n")
        else:
            lines.append("*No predictions recorded yet.*\n")

    except Exception:
        lines.append("*Prediction/outcome counts unavailable.*\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    output_path: str | None = None,
    round_times: list[float] | None = None,
) -> None:
    """Generate and append a marathon post-mortem report.

    Args:
        output_path: Destination markdown file. Defaults to
            ``data/midge/marathon-report.md`` relative to this file.
        round_times: Per-round elapsed seconds list from main.py's run loop.
            When provided, Section 1 includes timing totals.
    """
    if output_path is None:
        out = DATA_MIDGE / "marathon-report.md"
    else:
        out = Path(output_path)

    out.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        f"\n---\n",
        f"# MIDGE Marathon Post-Mortem — {now}\n",
        "",
        _render_vitals(round_times),
        _render_market_intelligence(),
        _render_bayesian_learning(),
        _render_hypothesis_pipeline(),
        _render_position_sizing(),
        _render_performance_health(),
    ]

    report_text = "\n".join(sections)

    with open(out, "a", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIDGE Marathon Post-Mortem Report")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (default: data/midge/marathon-report.md)",
    )
    args = parser.parse_args()

    generate_report(output_path=args.output)
    resolved = args.output or str(DATA_MIDGE / "marathon-report.md")
    print(f"Report generated: {resolved}")
