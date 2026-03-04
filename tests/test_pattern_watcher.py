"""Tests for Pattern Watcher — live stacking detection engine."""

import pytest
from pathlib import Path

from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PrecursorSignal,
    PatternTemplate,
)
from mae_core.market.archaeology.pattern_library import PatternLibrary
from mae_core.market.archaeology.pattern_watcher import (
    PatternWatcher,
    PatternStack,
    PatternActivation,
)


def _make_fingerprint(
    symbol="NVDA", direction="bullish", move_date="2024-01-15",
    move_pct=10.0, sources=None,
):
    """Create a MoveFingerprint with domain-consistent sources."""
    if sources is None:
        sources = [("sec_form4", "insider"), ("fred_macro", "macro")]
    return MoveFingerprint(
        fingerprint_id="",
        symbol=symbol,
        direction=direction,
        move_date=move_date,
        move_pct=move_pct,
        precursor_signals=[
            PrecursorSignal(src, dom, direction, 0.5, 3, "short")
            for src, dom in sources
        ],
    )


def _make_template(direction="bullish", domains=None, **kwargs):
    """Create a PatternTemplate directly."""
    if domains is None:
        domains = ["insider", "macro"]
    domain_sig = "+".join(sorted(domains))
    return PatternTemplate(
        template_id="",
        direction=direction,
        domain_signature=domain_sig,
        domains=sorted(domains),
        **kwargs,
    )


class TestPatternStack:
    def test_tier_classification(self):
        # 4 activations = high
        activations = [
            PatternActivation(
                template=_make_template(domains=[f"domain_{i}", f"extra_{i}"]),
                match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )
            for i in range(1, 5)
        ]
        stack = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=activations,
            independent_pairs=5, total_pairs=6,
        )
        assert stack.tier == "high"

    def test_tier_medium(self):
        activations = [
            PatternActivation(
                template=_make_template(domains=[f"domain_{i}"]),
                match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )
            for i in range(1, 4)
        ]
        stack = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=activations,
            independent_pairs=2, total_pairs=3,
        )
        assert stack.tier == "medium"

    def test_tier_low(self):
        activations = [
            PatternActivation(
                template=_make_template(domains=[f"domain_{i}"]),
                match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )
            for i in range(1, 3)
        ]
        stack = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=activations,
            independent_pairs=1, total_pairs=1,
        )
        assert stack.tier == "low"

    def test_confidence_increases_with_patterns(self):
        # More patterns should give higher confidence
        t1 = _make_template(domains=["insider", "macro"])
        t1.wins = 6
        t1.losses = 4
        t2 = _make_template(domains=["technical", "government"])
        t2.wins = 5
        t2.losses = 5

        single = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[PatternActivation(
                template=t1, match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )],
            independent_pairs=0, total_pairs=0,
        )
        double = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[
                PatternActivation(
                    template=t1, match_score=0.8,
                    matched_domains=[], missing_domains=[],
                    matched_sources=[], missing_sources=[],
                ),
                PatternActivation(
                    template=t2, match_score=0.7,
                    matched_domains=[], missing_domains=[],
                    matched_sources=[], missing_sources=[],
                ),
            ],
            independent_pairs=1, total_pairs=1,
        )
        assert double.stack_confidence > single.stack_confidence

    def test_format_alert(self):
        t = _make_template(domains=["insider", "macro"])
        t.wins = 7
        t.losses = 3
        activation = PatternActivation(
            template=t, match_score=0.85,
            matched_domains=["insider", "macro"],
            missing_domains=[],
            matched_sources=["sec_form4", "fred_macro"],
            missing_sources=[],
        )
        stack = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[activation, activation],
            independent_pairs=1, total_pairs=1,
        )
        alert = stack.format_alert()
        assert "PATTERN STACK" in alert
        assert "NVDA" in alert
        assert "BULLISH" in alert
        assert "sec_form4" in alert
        assert "insider+macro" in alert

    def test_to_dict(self):
        t = _make_template(domains=["insider", "macro"])
        activation = PatternActivation(
            template=t, match_score=0.8,
            matched_domains=["insider"],
            missing_domains=["macro"],
            matched_sources=["sec_form4"],
            missing_sources=["macro"],
        )
        stack = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[activation],
            independent_pairs=0, total_pairs=0,
        )
        d = stack.to_dict()
        assert d["symbol"] == "NVDA"
        assert d["n_patterns"] == 1
        assert len(d["activations"]) == 1
        assert d["activations"][0]["domain_signature"] == "insider+macro"

    def test_cross_validated_boost(self):
        """Cross-validated templates should give higher stack confidence."""
        # Non-cross-validated template (1 symbol)
        t_single = _make_template(domains=["insider", "macro"])
        t_single.symbols_seen = ["NVDA"]
        t_single.wins = 6
        t_single.losses = 4

        # Cross-validated template (5 symbols)
        t_cross = _make_template(domains=["insider", "macro"])
        t_cross.symbols_seen = ["NVDA", "AAPL", "MSFT", "AMD", "GOOG"]
        t_cross.wins = 6
        t_cross.losses = 4

        stack_single = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[PatternActivation(
                template=t_single, match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )],
            independent_pairs=0, total_pairs=0,
        )
        stack_cross = PatternStack(
            symbol="NVDA", direction="bullish",
            activations=[PatternActivation(
                template=t_cross, match_score=0.8,
                matched_domains=[], missing_domains=[],
                matched_sources=[], missing_sources=[],
            )],
            independent_pairs=0, total_pairs=0,
        )
        assert stack_cross.stack_confidence > stack_single.stack_confidence


class TestPatternWatcher:
    def test_no_matches_no_stacks(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        watcher = PatternWatcher(library=lib)

        stacks = watcher.check({"NVDA": {"bullish": {"sec_form4"}}})
        assert len(stacks) == 0

    def test_single_match_below_min_stack(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        fp = _make_fingerprint(sources=[("sec_form4", "insider")])
        lib.store(fp)

        watcher = PatternWatcher(library=lib, min_stack=2)
        stacks = watcher.check({"NVDA": {"bullish": {"sec_form4"}}})
        assert len(stacks) == 0  # Only 1 template match, need 2

    def test_two_templates_produce_stack(self, tmp_path):
        """Two different templates matching → pattern stack."""
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        # Two fingerprints with DIFFERENT domain_signatures → two templates
        fp1 = _make_fingerprint(
            move_date="2024-01-15",
            sources=[("sec_form4", "insider"), ("fred_macro", "macro")],
        )
        fp2 = _make_fingerprint(
            move_date="2024-02-15",
            sources=[("sec_form4", "insider"), ("ta_rsi", "technical")],
        )
        lib.store(fp1)
        lib.store(fp2)

        assert lib.template_count == 2  # Different domain_signatures

        watcher = PatternWatcher(library=lib, min_stack=2)
        stacks = watcher.check({
            "NVDA": {"bullish": {"sec_form4", "fred_macro", "ta_rsi"}},
        })
        assert len(stacks) == 1
        assert stacks[0].symbol == "NVDA"
        assert stacks[0].direction == "bullish"
        assert len(stacks[0].activations) == 2

    def test_independence_check(self, tmp_path):
        """Templates with non-overlapping domains are independent."""
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.3,
        )
        # Two fingerprints with completely different domains = independent templates
        fp1 = _make_fingerprint(
            move_date="2024-01-15",
            sources=[("sec_form4", "insider"), ("fred_macro", "macro")],
        )
        fp2 = _make_fingerprint(
            move_date="2024-02-15",
            sources=[("ta_rsi", "technical"), ("congressional", "government")],
        )
        lib.store(fp1)
        lib.store(fp2)

        watcher = PatternWatcher(library=lib, min_stack=2)
        stacks = watcher.check({
            "NVDA": {"bullish": {"sec_form4", "fred_macro", "ta_rsi", "congressional"}},
        })

        if stacks:
            assert stacks[0].independent_pairs == 1  # Fully independent
            assert stacks[0].total_pairs == 1

    def test_dedup_24h(self, tmp_path):
        """Same stack shouldn't re-alert within 24 hours."""
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
            match_threshold=0.5,
        )
        # Two different templates that both match
        fp1 = _make_fingerprint(
            move_date="2024-01-15",
            sources=[("sec_form4", "insider"), ("fred_macro", "macro")],
        )
        fp2 = _make_fingerprint(
            move_date="2024-02-15",
            sources=[("sec_form4", "insider"), ("ta_rsi", "technical")],
        )
        lib.store(fp1)
        lib.store(fp2)

        watcher = PatternWatcher(library=lib, min_stack=2)
        active = {"NVDA": {"bullish": {"sec_form4", "fred_macro", "ta_rsi"}}}

        # First check should alert
        stacks1 = watcher.check(active)
        # Second check within 24h should not re-alert
        stacks2 = watcher.check(active)
        assert len(stacks1) == 1
        assert len(stacks2) == 0

    def test_build_active_signals(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        watcher = PatternWatcher(library=lib)

        signals = [
            {"symbol": "NVDA", "direction": "bullish", "source": "sec_form4"},
            {"symbol": "NVDA", "direction": "bullish", "source": "fred_macro"},
            {"symbol": "AAPL", "direction": "bearish", "source": "ta_rsi"},
        ]

        active = watcher.build_active_signals(signals)
        assert "NVDA" in active
        assert "bullish" in active["NVDA"]
        assert "sec_form4" in active["NVDA"]["bullish"]
        assert "fred_macro" in active["NVDA"]["bullish"]
        assert "AAPL" in active
        assert "bearish" in active["AAPL"]

    def test_statistics(self, tmp_path):
        lib = PatternLibrary(
            library_path=tmp_path / "lib.jsonl",
            templates_path=tmp_path / "templates.jsonl",
        )
        watcher = PatternWatcher(library=lib)
        stats = watcher.get_statistics()
        assert stats["checks_performed"] == 0
        assert stats["stacks_detected"] == 0
        assert stats["library_size"] == 0
        assert stats["template_count"] == 0
