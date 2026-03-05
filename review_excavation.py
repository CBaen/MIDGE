#!/usr/bin/env python3
"""Review Excavation Results — human-readable summary of the Pattern Library.

Produces a clear, non-technical report of what MIDGE discovered by
reverse-engineering historical market moves. Designed for Guiding Light
to review excavation results efficiently.

Usage:
    python review_excavation.py              # Full report
    python review_excavation.py --top 20     # Show top 20 templates
    python review_excavation.py --domain     # Domain-focused analysis
    python review_excavation.py --json       # Machine-readable output
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mae_core.market.archaeology.pattern_library import PatternLibrary


def main():
    parser = argparse.ArgumentParser(description="Review MIDGE Pattern Library")
    parser.add_argument("--top", type=int, default=15,
                        help="Number of top templates to show (default: 15)")
    parser.add_argument("--domain", action="store_true",
                        help="Show domain-focused analysis")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of human-readable")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild templates from fingerprints")
    args = parser.parse_args()

    lib = PatternLibrary()

    # Always rebuild templates from fingerprints (the source of truth).
    # The templates file can get corrupted by concurrent writes from
    # the running excavation daemon. Fingerprints are append-only and safe.
    if lib.size > 0:
        import logging
        logging.getLogger("mae_core.market.archaeology.pattern_library").setLevel(logging.WARNING)
        lib.rebuild_templates()

    if args.json:
        print_json_report(lib)
    else:
        print_human_report(lib, args.top, args.domain)


def print_human_report(lib: PatternLibrary, top_n: int, show_domain: bool):
    """Print a human-readable report for Guiding Light."""

    fingerprints = list(lib._fingerprints.values())
    templates = list(lib._templates.values())

    if not fingerprints:
        print("No excavation data found. Run populate_library.py first.")
        return

    # ── Overall Summary ─────────────────────────────────────────────
    symbols = set(fp.symbol for fp in fingerprints)
    directions = Counter(fp.direction for fp in fingerprints)
    domain_sigs = Counter(fp.domain_signature for fp in fingerprints)
    cross_validated = [t for t in templates if t.cross_validated]

    print("=" * 70)
    print("  MIDGE PATTERN ARCHAEOLOGIST — EXCAVATION REPORT")
    print("=" * 70)
    print()
    print(f"  Fingerprints discovered:  {len(fingerprints):,}")
    print(f"  Pattern templates:        {len(templates):,}")
    print(f"  Cross-validated (3+ sym): {len(cross_validated):,}")
    print(f"  Symbols excavated:        {len(symbols):,}")
    print(f"  Bullish fingerprints:     {directions.get('bullish', 0):,}")
    print(f"  Bearish fingerprints:     {directions.get('bearish', 0):,}")
    print(f"  Unique domain combos:     {len(domain_sigs):,}")
    print()

    # ── Top Templates by Cross-Symbol Validation ─────────────────────
    print("-" * 70)
    print("  TOP PATTERNS BY CROSS-SYMBOL VALIDATION")
    print("  (Patterns seen across the most different stocks = most real)")
    print("-" * 70)
    print()

    sorted_templates = sorted(templates,
                               key=lambda t: (len(t.unique_symbols), t.n_instances),
                               reverse=True)

    for i, t in enumerate(sorted_templates[:top_n], 1):
        n_syms = len(t.unique_symbols)
        confidence = "MARKET-WIDE" if n_syms >= 10 else (
            "STRONG" if n_syms >= 5 else (
                "VALIDATED" if n_syms >= 3 else "SINGLE"))
        dir_icon = "UP" if t.direction == "bullish" else "DN"

        print(f"  {i:>2}. [{dir_icon}] {t.domain_signature}")
        print(f"      {n_syms} symbols | {t.n_instances} observations | "
              f"avg move: {t.avg_move_pct:.1f}% | [{confidence}]")

        # Show sample symbols
        sample_syms = sorted(t.unique_symbols)[:8]
        suffix = f" +{n_syms - 8} more" if n_syms > 8 else ""
        print(f"      Symbols: {', '.join(sample_syms)}{suffix}")

        # Show which specific data sources feed this pattern
        if t.source_examples:
            sources_flat = []
            for domain, srcs in sorted(t.source_examples.items()):
                sources_flat.extend(srcs[:2])  # Max 2 per domain
            print(f"      Sources: {', '.join(sources_flat[:6])}")

        # Lag profile
        if t.lag_profile_normalized:
            lag_parts = []
            for bucket in ["immediate", "short", "medium", "long", "extended"]:
                pct = t.lag_profile_normalized.get(bucket, 0)
                if pct > 0.05:
                    lag_parts.append(f"{bucket}: {pct:.0%}")
            if lag_parts:
                print(f"      Timing: {', '.join(lag_parts)}")

        print()

    # ── Domain Distribution ───────────────────────────────────────────
    if show_domain:
        _print_domain_analysis(fingerprints, templates)

    # ── Stacking Potential ────────────────────────────────────────────
    print("-" * 70)
    print("  STACKING POTENTIAL")
    print("  (Multi-domain templates = the real edge)")
    print("-" * 70)
    print()

    # Use domain_signature (always reliable) rather than domains list
    multi_domain = [t for t in templates if "+" in t.domain_signature]
    single_domain = [t for t in templates if "+" not in t.domain_signature]

    print(f"  Single-domain templates: {len(single_domain):,}")
    print(f"  Multi-domain templates:  {len(multi_domain):,}")
    print()

    if multi_domain:
        print("  Multi-domain patterns (where MIDGE's edge lives):")
        print()
        multi_sorted = sorted(multi_domain,
                               key=lambda t: (len(t.domain_signature.split("+")),
                                              len(t.unique_symbols)),
                               reverse=True)
        for t in multi_sorted[:10]:
            n_syms = len(t.unique_symbols)
            dir_icon = "UP" if t.direction == "bullish" else "DN"
            print(f"    [{dir_icon}] {t.domain_signature}")
            print(f"        {len(t.domains)} domains | {n_syms} symbols | "
                  f"{t.n_instances} obs | avg {t.avg_move_pct:.1f}%")
        print()

    # ── Coverage Stats ────────────────────────────────────────────────
    print("-" * 70)
    print("  SYMBOL COVERAGE")
    print("-" * 70)
    print()

    # Symbols with most fingerprints
    sym_counts = Counter(fp.symbol for fp in fingerprints)
    top_syms = sym_counts.most_common(15)
    print("  Most data-rich symbols (most historical patterns found):")
    for sym, count in top_syms:
        print(f"    {sym:>8}: {count} fingerprints")
    print()

    # Move magnitude distribution
    moves = [abs(fp.move_pct) for fp in fingerprints]
    if moves:
        moves.sort()
        median_idx = len(moves) // 2
        print("  Move magnitude distribution:")
        print(f"    Smallest: {min(moves):.1f}%")
        print(f"    Median:   {moves[median_idx]:.1f}%")
        print(f"    Largest:  {max(moves):.1f}%")
        print(f"    >= 10%:   {sum(1 for m in moves if m >= 10):,} moves")
        print(f"    >= 20%:   {sum(1 for m in moves if m >= 20):,} moves")
        print()

    # ── What This Means ──────────────────────────────────────────────
    print("=" * 70)
    print("  WHAT THIS MEANS FOR MIDGE")
    print("=" * 70)
    print()

    if cross_validated:
        print(f"  {len(cross_validated)} patterns have been independently validated")
        print(f"  across 3+ different stocks. These are REAL market patterns,")
        print(f"  not coincidences on a single ticker.")
        print()

    if multi_domain:
        print(f"  {len(multi_domain)} patterns combine signals from multiple")
        print(f"  independent data domains. When MIDGE sees these combos")
        print(f"  stack in real-time, that's the high-confidence signal.")
        print()

    print(f"  Next: Run MIDGE's daemon and the PatternWatcher will")
    print(f"  automatically match incoming signals against these")
    print(f"  {len(templates)} templates. When patterns stack, MIDGE alerts.")
    print()


def _print_domain_analysis(fingerprints, templates):
    """Print domain-focused analysis."""
    print("-" * 70)
    print("  DOMAIN ANALYSIS")
    print("  (Which types of information precede big moves)")
    print("-" * 70)
    print()

    # Count domain appearances across all fingerprints
    domain_counts = Counter()
    domain_direction = defaultdict(lambda: Counter())
    for fp in fingerprints:
        domains = set(s.domain for s in fp.precursor_signals)
        for d in domains:
            domain_counts[d] += 1
            domain_direction[d][fp.direction] += 1

    total = len(fingerprints)
    print("  How often each data type appears before a big move:")
    print()
    for domain, count in domain_counts.most_common():
        pct = count / total * 100
        bull = domain_direction[domain].get("bullish", 0)
        bear = domain_direction[domain].get("bearish", 0)
        bar_len = int(pct / 2)
        bar = "#" * bar_len
        print(f"    {domain:>15}: {count:>6,} ({pct:5.1f}%) {bar}")
        print(f"    {'':>15}  bullish: {bull:,} | bearish: {bear:,}")
    print()

    # Domain co-occurrence
    print("  Domain pairs that appear together most often:")
    print()
    pair_counts = Counter()
    for fp in fingerprints:
        domains = sorted(set(s.domain for s in fp.precursor_signals))
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                pair_counts[(domains[i], domains[j])] += 1

    for (d1, d2), count in pair_counts.most_common(10):
        pct = count / total * 100
        print(f"    {d1} + {d2}: {count:,} ({pct:.1f}%)")
    print()


def print_json_report(lib: PatternLibrary):
    """Print machine-readable JSON report."""
    fingerprints = list(lib._fingerprints.values())
    templates = list(lib._templates.values())

    report = {
        "summary": {
            "total_fingerprints": len(fingerprints),
            "total_templates": len(templates),
            "cross_validated": sum(1 for t in templates if t.cross_validated),
            "symbols_excavated": len(set(fp.symbol for fp in fingerprints)),
            "multi_domain_templates": sum(1 for t in templates if len(t.domains) >= 2),
        },
        "templates": [
            {
                "domain_signature": t.domain_signature,
                "direction": t.direction,
                "n_symbols": len(t.unique_symbols),
                "n_instances": t.n_instances,
                "avg_move_pct": round(t.avg_move_pct, 2),
                "cross_validated": t.cross_validated,
                "domains": t.domains,
                "sample_symbols": sorted(t.unique_symbols)[:10],
            }
            for t in sorted(templates,
                            key=lambda t: len(t.unique_symbols),
                            reverse=True)
        ],
        "domain_distribution": dict(Counter(
            fp.domain_signature for fp in fingerprints
        ).most_common()),
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
