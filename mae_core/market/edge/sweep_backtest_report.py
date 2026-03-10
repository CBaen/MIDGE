"""
sweep_backtest_report.py - Reporting for ICT Session Sweep + IFVG Backtest

The report() function generates comprehensive statistics from Trade results.
"""

from typing import List
from mae_core.market.edge.sweep_backtest_models import Trade


def report(trades: List[Trade]) -> str:
    """Generate comprehensive backtest statistics."""
    if not trades:
        return "\nNo trades found. Strategy produced zero setups in this data."

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  ICT SESSION SWEEP + IFVG BACKTEST RESULTS")
    lines.append("=" * 70)

    total = len(trades)
    wins = [t for t in trades if t.result == "win_2r"]
    losses = [t for t in trades if t.result == "loss"]
    timeouts = [t for t in trades if t.result == "timeout"]
    hit_1r = [t for t in trades if t.hit_1r]

    win_rate = len(wins) / total * 100
    avg_r = sum(t.r_captured for t in trades) / total

    avg_win_r = sum(t.r_captured for t in wins) / len(wins) if wins else 0
    avg_loss_r = (
        abs(sum(t.r_captured for t in losses)) / len(losses)
        if losses else 0
    )
    expectancy = (
        avg_win_r * (len(wins) / total)
        - avg_loss_r * (len(losses) / total)
    )

    would_win_1r = len(hit_1r)
    win_rate_1r = would_win_1r / total * 100
    expectancy_1r = (
        1.0 * (would_win_1r / total)
        - 1.0 * ((total - would_win_1r) / total)
    )

    lines.append(f"\n--- Overall (targeting 2R) ---")
    lines.append(f"  Total trades:     {total}")
    lines.append(f"  Wins (2R):        {len(wins)} ({win_rate:.1f}%)")
    lines.append(f"  Losses:           {len(losses)} ({len(losses)/total*100:.1f}%)")
    lines.append(f"  Timeouts:         {len(timeouts)} ({len(timeouts)/total*100:.1f}%)")
    lines.append(f"  Avg R captured:   {avg_r:+.3f}R")
    lines.append(f"  Expectancy:       {expectancy:+.3f}R per trade")

    lines.append(f"\n--- What if targeting 1R instead? ---")
    lines.append(f"  Reached 1R:       {would_win_1r} ({win_rate_1r:.1f}%)")
    lines.append(f"  1R expectancy:    {expectancy_1r:+.3f}R per trade")

    gross_profit = sum(t.r_captured for t in trades if t.r_captured > 0)
    gross_loss = abs(sum(t.r_captured for t in trades if t.r_captured < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    lines.append(f"\n--- Risk metrics ---")
    lines.append(f"  Profit factor:    {profit_factor:.2f}")
    lines.append(f"  Gross profit:     {gross_profit:+.1f}R")
    lines.append(f"  Gross loss:       {-gross_loss:.1f}R")
    lines.append(f"  Net R:            {gross_profit - gross_loss:+.1f}R")

    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if t.result == "loss":
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0
    lines.append(f"  Max consec. loss: {max_consec_loss}")

    lines.append(f"\n--- By direction ---")
    for dir_name in ("bullish", "bearish"):
        dir_trades = [t for t in trades if t.direction == dir_name]
        if not dir_trades:
            continue
        dir_wins = [t for t in dir_trades if t.result == "win_2r"]
        dir_wr = len(dir_wins) / len(dir_trades) * 100
        dir_avg = sum(t.r_captured for t in dir_trades) / len(dir_trades)
        lines.append(
            f"  {dir_name:10s}: {len(dir_trades):3d} trades, "
            f"{dir_wr:.1f}% WR, {dir_avg:+.3f}R avg"
        )

    lines.append(f"\n--- By session swept ---")
    sessions = sorted(set(t.session_swept for t in trades))
    for sess in sessions:
        s_trades = [t for t in trades if t.session_swept == sess]
        s_wins = [t for t in s_trades if t.result == "win_2r"]
        s_wr = len(s_wins) / len(s_trades) * 100
        s_avg = sum(t.r_captured for t in s_trades) / len(s_trades)
        lines.append(
            f"  {sess:10s}: {len(s_trades):3d} trades, "
            f"{s_wr:.1f}% WR, {s_avg:+.3f}R avg"
        )

    has_quality = any(t.quality_score > 0 for t in trades)
    if has_quality:
        lines.append(f"\n--- By quality tier ---")
        tiers = [
            ("Elite (>0.60)", lambda t: t.quality_score > 0.60),
            ("Good (0.40-0.60)", lambda t: 0.40 <= t.quality_score <= 0.60),
            ("Marginal (0.20-0.39)", lambda t: 0.20 <= t.quality_score < 0.40),
            ("Low (<0.20)", lambda t: t.quality_score < 0.20),
        ]
        for tier_name, tier_fn in tiers:
            tier_trades = [t for t in trades if tier_fn(t)]
            if not tier_trades:
                lines.append(f"  {tier_name:22s}:   0 trades")
                continue
            tier_wins = [t for t in tier_trades if t.result == "win_2r"]
            tier_wr = len(tier_wins) / len(tier_trades) * 100
            tier_avg = sum(t.r_captured for t in tier_trades) / len(tier_trades)
            tier_net = sum(t.r_captured for t in tier_trades)
            lines.append(
                f"  {tier_name:22s}: {len(tier_trades):3d} trades, "
                f"{tier_wr:.1f}% WR, {tier_avg:+.3f}R avg, {tier_net:+.1f}R net"
            )

        avg_disp = sum(t.displacement_score for t in trades) / len(trades)
        avg_fvg_atr = sum(t.fvg_atr_ratio for t in trades) / len(trades)
        avg_kz = sum(t.kill_zone_score for t in trades) / len(trades)
        avg_quality = sum(t.quality_score for t in trades) / len(trades)
        lines.append(f"\n  Score averages:")
        lines.append(f"    Displacement:  {avg_disp:.3f}  (body ratio of reversal candles)")
        lines.append(f"    FVG/ATR ratio: {avg_fvg_atr:.3f}  (FVG size / 14-period ATR)")
        lines.append(f"    Kill zone:     {avg_kz:.3f}  (1.0=NY, 0.85=London, 0.70=Asia)")
        lines.append(f"    Composite:     {avg_quality:.3f}  (40% disp + 35% fvg/atr + 25% kz)")

    lines.append(f"\n--- By symbol ---")
    symbols = sorted(set(t.symbol for t in trades))
    for sym in symbols:
        sym_trades = [t for t in trades if t.symbol == sym]
        sym_wins = [t for t in sym_trades if t.result == "win_2r"]
        sym_wr = len(sym_wins) / len(sym_trades) * 100
        sym_avg = sum(t.r_captured for t in sym_trades) / len(sym_trades)
        sym_net = sum(t.r_captured for t in sym_trades)
        lines.append(
            f"  {sym:8s}: {len(sym_trades):3d} trades, "
            f"{sym_wr:.1f}% WR, {sym_avg:+.3f}R avg, {sym_net:+.1f}R net"
        )

    lines.append(f"\n--- Recent trades (last 30) ---")
    header = (
        f"  {'Symbol':8s} {'Dir':8s} {'Sess':8s} "
        f"{'Entry':>9s} {'Stop':>9s} {'Target':>9s} "
        f"{'Result':8s} {'R':>6s} {'1R?':>4s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for t in trades[-30:]:
        lines.append(
            f"  {t.symbol:8s} {t.direction:8s} {t.session_swept:8s} "
            f"{t.entry_price:9.2f} {t.stop_price:9.2f} {t.target_2r:9.2f} "
            f"{t.result:8s} {t.r_captured:+5.1f}R "
            f"{'Y' if t.hit_1r else 'N':>3s}"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
