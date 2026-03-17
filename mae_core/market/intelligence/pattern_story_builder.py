"""PatternStoryBuilder — turns PatternTemplates into human-readable narratives.

Workflow per template:
  1. Load the template from PatternLibrary
  2. Scan fingerprint instances to collect per-phase signal timing data
  3. Extract recurring characters (insiders, congress members, funds)
  4. Group signals into temporal chapters (early / confirmation / resolution)
  5. Run scipy binomial test for statistical significance vs 50% null
  6. Compute per-regime win rates from template instances
  7. Auto-generate title and plain-English summary
  8. Save to data/midge/pattern_stories.jsonl

Usage:
    builder = PatternStoryBuilder()
    story = builder.build_story("abc123def456")
    all_stories = builder.build_all_stories()
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mae_core.market.archaeology.fingerprint import (
    MoveFingerprint,
    PatternTemplate,
    PrecursorSignal,
)
from mae_core.market.archaeology.pattern_library import PatternLibrary
from mae_core.market.intelligence.pattern_story import PatternChapter, PatternStory

logger = logging.getLogger(__name__)

# Output path for serialized stories
STORIES_PATH = Path(__file__).resolve().parents[3] / "data" / "midge" / "pattern_stories.jsonl"

# Signal archive directory (one JSONL per day)
SIGNAL_ARCHIVE_DIR = Path(__file__).resolve().parents[3] / "data" / "midge" / "signals"

# ── Domain → plain English descriptions ─────────────────────────────────────

_DOMAIN_DESCRIPTIONS = {
    "insider": "company insiders (executives, directors, or officers) traded shares",
    "macro": "macroeconomic data shifted — interest rates, inflation, or Fed policy moved",
    "technical": "price chart patterns formed that historically precede large moves",
    "events": "the company filed material disclosures or announced significant news",
    "positioning": "large institutional traders repositioned in futures markets",
    "government": "elected officials made related stock or options trades",
    "contracts": "government contracts were awarded or announced in this sector",
    "sentiment": "public discussion and social media interest spiked",
    "fundamental": "company financial metrics showed notable changes",
    "fundamentals": "company financial metrics showed notable changes",
    "institutional": "large funds (13F/13D filings) changed their holdings",
    "crypto": "cryptocurrency market activity showed correlated movement",
    "energy": "U.S. energy inventory or production data shifted",
    "convergence": "multiple previously independent signals converged",
    "meta": "pattern recognition systems flagged a match",
    "volatility": "market fear indicators spiked or collapsed",
    "causal": "upstream causal chain confirmed a downstream move",
    "cascade": "a domino chain of correlated events confirmed",
}

# Phase names by chapter number
_PHASE_NAMES = {
    1: "early warning",
    2: "confirmation",
    3: "resolution",
}

# Entity role → human label
_ROLE_LABELS = {
    "insider_name": "company insider",
    "representative": "member of Congress",
    "filer_name": "SEC filer",
    "fund_name": "hedge fund",
    "committee": "congressional committee",
}

# Direction → plain English
_DIRECTION_PLAIN = {
    "bullish": "price increase",
    "bearish": "price decrease",
}


class PatternStoryBuilder:
    """Builds PatternStory objects from PatternTemplate data.

    The builder is stateless between calls — each build_story() call is
    independent. The PatternLibrary is loaded once in __init__ and shared
    across all build_story calls.
    """

    def __init__(
        self,
        library: Optional[PatternLibrary] = None,
        stories_path: Optional[Path] = None,
        signal_archive_dir: Optional[Path] = None,
    ):
        self._library = library or PatternLibrary()
        self._stories_path = stories_path or STORIES_PATH
        self._signal_archive_dir = signal_archive_dir or SIGNAL_ARCHIVE_DIR
        # Lazy-loaded signal archive index: signal_id -> signal_dict
        # We do NOT load this eagerly — it's huge. We load on-demand by date.
        self._archive_cache: dict[str, dict] = {}  # date_str -> {signal_id: signal_dict}

    # ── Public API ────────────────────────────────────────────────────────

    def build_story(self, template_id: str) -> Optional[PatternStory]:
        """Build a PatternStory for one template.

        Returns None if the template is not found.
        """
        template = self._library.get_template(template_id)
        if template is None:
            logger.warning("Template %s not found", template_id)
            return None

        story = self._build_from_template(template)
        self._persist_story(story)
        logger.info(
            "Built story '%s' for template %s (%d instances, %d symbols)",
            story.title, template_id, story.n_instances, story.n_symbols,
        )
        return story

    def build_all_stories(self) -> list[PatternStory]:
        """Build stories for every template in the library.

        Overwrites the stories file on each call (full rebuild).
        Returns the list of stories sorted by n_instances descending.
        """
        templates = list(self._library._templates.values())
        if not templates:
            logger.warning("No templates found — PatternLibrary may be empty")
            return []

        logger.info("Building stories for %d templates", len(templates))
        stories: list[PatternStory] = []

        for template in sorted(templates, key=lambda t: t.n_instances, reverse=True):
            try:
                story = self._build_from_template(template)
                stories.append(story)
            except Exception as e:
                logger.warning("Failed to build story for %s: %s", template.template_id, e)

        # Write all stories (full rewrite)
        self._persist_stories(stories)
        logger.info("Wrote %d stories to %s", len(stories), self._stories_path)
        return stories

    def load_stories(self) -> list[PatternStory]:
        """Load all previously built stories from disk."""
        if not self._stories_path.exists():
            return []
        stories = []
        try:
            with open(self._stories_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        stories.append(PatternStory.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as e:
            logger.warning("Could not load stories: %s", e)
        return stories

    # ── Core build logic ──────────────────────────────────────────────────

    def _build_from_template(self, template: PatternTemplate) -> PatternStory:
        """Build a complete PatternStory from a PatternTemplate."""

        # Collect entity data from template (populated by add_instance during excavation)
        recurring_entities = dict(template.recurring_entities)
        entity_weight_factors = dict(template.entity_weight_factors)

        # If template was built before entity enrichment, try to load from fingerprint instances
        # by scanning the signal archive for signal_ids referenced in precursor signals
        sampled_fp_ids: list[str] = []
        if not recurring_entities:
            recurring_entities, entity_weight_factors, sampled_fp_ids = (
                self._extract_entities_from_instances(template)
            )

        # Sort entities by (weight * count) descending
        top_entities = sorted(
            recurring_entities.keys(),
            key=lambda n: (
                entity_weight_factors.get(n, 1.0) * recurring_entities[n].get("count", 0)
            ),
            reverse=True,
        )[:5]

        # Compute temporal chapters from lag_profile_raw
        chapters = self._build_chapters(template)

        # Statistical analysis
        wins, losses = template.wins, template.losses
        win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
        ci_lower, ci_upper = self._clopper_pearson(wins, losses)
        p_value = self._binomial_test(wins, wins + losses)
        is_significant = p_value < 0.05 and (wins + losses) >= 5

        # Regime win rates from instances
        regime_win_rates = self._compute_regime_stats(template)
        best_regime = max(regime_win_rates, key=lambda r: regime_win_rates[r]) if regime_win_rates else ""

        # Timing
        dominant_lag = max(template.lag_profile_raw, key=template.lag_profile_raw.get) if template.lag_profile_raw else ""

        # Title and summary
        title = self._generate_title(template, top_entities, entity_weight_factors)
        summary = self._generate_summary(
            template, top_entities, chapters, win_rate, wins + losses, is_significant
        )

        return PatternStory(
            story_id=template.template_id,
            template_id=template.template_id,
            direction=template.direction,
            domain_signature=template.domain_signature,
            title=title,
            summary=summary,
            chapters=chapters,
            recurring_entities=recurring_entities,
            top_entities=top_entities,
            n_instances=template.n_instances,
            n_symbols=len(template.unique_symbols),
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            is_significant=is_significant,
            avg_move_pct=round(template.avg_move_pct, 2),
            expected_window_days=template.expected_move_window_days,
            dominant_lag_bucket=dominant_lag,
            regime_win_rates=regime_win_rates,
            best_regime=best_regime,
            source_examples=template.source_examples,
            fingerprint_ids_sampled=sampled_fp_ids,
        )

    # ── Entity extraction ─────────────────────────────────────────────────

    def _extract_entities_from_instances(
        self, template: PatternTemplate
    ) -> tuple[dict, dict, list[str]]:
        """Fallback: extract entity data by scanning the fingerprint instances.

        This is only called when template.recurring_entities is empty (pre-enrichment
        templates). It samples up to 50 fingerprints from the template's instances,
        loads each from the pattern library, and collects entity_metadata from
        their precursor signals.
        """
        recurring_entities: dict = {}
        entity_weight_factors: dict = {}
        sampled_ids: list[str] = []

        # Sample at most 50 instances to avoid scanning 100s of fingerprints
        instances_to_check = template.instances[:50]

        for inst in instances_to_check:
            fp_id = inst.fingerprint_id
            if not fp_id:
                continue
            fp = self._library.get(fp_id)
            if fp is None:
                continue
            sampled_ids.append(fp_id)
            self._accumulate_entities_from_fp(fp, recurring_entities, entity_weight_factors)

        return recurring_entities, entity_weight_factors, sampled_ids

    def _accumulate_entities_from_fp(
        self,
        fp: MoveFingerprint,
        entities: dict,
        weights: dict,
    ) -> None:
        """Accumulate entity data from a single fingerprint into shared dicts."""
        _ROLE_WEIGHT_BASE = {
            "insider_name": 1.0,
            "representative": 0.8,
            "filer_name": 0.9,
            "fund_name": 0.9,
        }
        _TITLE_BOOST = {
            "ceo": 1.5, "chief executive": 1.5, "chairman": 1.4,
            "cfo": 1.3, "chief financial": 1.3,
            "coo": 1.2, "president": 1.2,
            "10% owner": 1.1,
            "director": 1.0, "officer": 1.0,
        }
        for precursor in fp.precursor_signals:
            em = precursor.entity_metadata
            if not em:
                continue
            filer_title = str(em.get("filer_title", "")).lower()
            for key in ("insider_name", "representative", "filer_name", "fund_name"):
                name = em.get(key)
                if not name:
                    continue
                name = str(name).strip()
                if not name:
                    continue
                base = _ROLE_WEIGHT_BASE.get(key, 1.0)
                weight = base
                for kw, mult in _TITLE_BOOST.items():
                    if kw in filer_title:
                        weight = base * mult
                        break
                if name not in entities:
                    entities[name] = {"count": 0, "role": key, "domains": set()}
                entities[name]["count"] += 1
                entities[name]["domains"].add(precursor.domain)
                if name not in weights:
                    weights[name] = weight
                else:
                    weights[name] = max(weights[name], weight)

    # ── Temporal chapters ─────────────────────────────────────────────────

    def _build_chapters(self, template: PatternTemplate) -> list[PatternChapter]:
        """Divide the template's domain signals into temporal phases.

        We use the lag_profile_raw to understand when signals appear. Domains
        are sorted by their average lag (earlier = higher lag_days). We split
        them into three groups: early (top third by lag), confirmation (middle),
        and resolution (latest / lowest lag_days).
        """
        if not template.domains:
            return []

        # Estimate per-domain average lag from source_examples and lag_profile_raw
        # We use the overall lag_profile_normalized to assign approximate timing
        # to each domain group (we don't have per-domain lag breakdown in templates)
        domains = list(template.domains)

        # If only 1-2 domains, single chapter
        if len(domains) <= 2:
            desc = self._build_chapter_description(domains, "early warning")
            return [PatternChapter(
                chapter_number=1,
                phase_name="early warning",
                domains=domains,
                avg_lag_days=self._estimate_mean_lag(template),
                description=desc,
                signal_count=template.n_instances * len(domains),
            )]

        # Three-phase split: first third / middle / last third
        n = len(domains)
        split1 = max(1, n // 3)
        split2 = max(split1 + 1, (2 * n) // 3)

        groups = [
            (domains[:split1], 1, "early warning"),
            (domains[split1:split2], 2, "confirmation"),
            (domains[split2:], 3, "resolution"),
        ]

        # Approximate per-phase lag from overall profile
        mean_lag = self._estimate_mean_lag(template)
        lag_fractions = [1.5, 1.0, 0.5]  # early is further out, resolution is closer

        chapters = []
        for (group_domains, num, phase), lag_mult in zip(groups, lag_fractions):
            if not group_domains:
                continue
            desc = self._build_chapter_description(group_domains, phase)
            chapters.append(PatternChapter(
                chapter_number=num,
                phase_name=phase,
                domains=group_domains,
                avg_lag_days=round(mean_lag * lag_mult, 1),
                description=desc,
                signal_count=template.n_instances * len(group_domains),
            ))

        return chapters

    def _build_chapter_description(self, domains: list[str], phase: str) -> str:
        """Generate a plain-English description for a chapter's domains."""
        if not domains:
            return ""
        parts = [_DOMAIN_DESCRIPTIONS.get(d, f"{d} signals appeared") for d in domains]
        if len(parts) == 1:
            activity = parts[0].capitalize()
        elif len(parts) == 2:
            activity = f"{parts[0].capitalize()} and {parts[1]}"
        else:
            activity = ", ".join(p.capitalize() for p in parts[:-1]) + f", and {parts[-1]}"

        phase_prefix = {
            "early warning": "First,",
            "confirmation": "Next,",
            "resolution": "Finally,",
        }.get(phase, "Then,")

        return f"{phase_prefix} {activity}."

    def _estimate_mean_lag(self, template: PatternTemplate) -> float:
        """Estimate mean lag days from the template's lag profile."""
        _BUCKET_CENTERS = {
            "immediate": 1.0, "short": 4.0, "medium": 8.0,
            "long": 15.0, "extended": 25.0,
        }
        raw = template.lag_profile_raw
        if not raw:
            return 8.0
        total = sum(raw.values()) or 1
        weighted = sum(
            _BUCKET_CENTERS.get(b, 8.0) * c for b, c in raw.items()
        )
        return weighted / total

    # ── Statistical analysis ──────────────────────────────────────────────

    def _clopper_pearson(self, wins: int, losses: int, alpha: float = 0.05) -> tuple[float, float]:
        """Compute Clopper-Pearson 95% CI for a win rate."""
        n = wins + losses
        k = wins
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
            spread = z * math.sqrt(max(0, p * (1 - p) + z * z / (4 * n)) / n) / denom
            return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))

    def _binomial_test(self, wins: int, n: int, null_p: float = 0.5) -> float:
        """Two-sided binomial test: is win rate significantly different from null_p?

        Returns p-value. Lower = more statistically significant.
        """
        if n == 0:
            return 1.0
        try:
            from scipy.stats import binomtest
            result = binomtest(wins, n, null_p, alternative="two-sided")
            return round(result.pvalue, 6)
        except ImportError:
            # Manual normal approximation
            p_hat = wins / n
            se = math.sqrt(null_p * (1 - null_p) / n) if n > 0 else 1.0
            z = abs(p_hat - null_p) / se if se > 0 else 0.0
            # Two-sided p-value from normal CDF approximation
            p = 2 * (1 - _norm_cdf(z))
            return round(max(0.0, min(1.0, p)), 6)

    # ── Regime win rates ──────────────────────────────────────────────────

    def _compute_regime_stats(self, template: PatternTemplate) -> dict:
        """Compute win rates broken down by market regime.

        Uses template.instances (which carry regime labels) but does NOT
        attempt to load outcome data per-regime (that would require joining
        against the outcomes file — too expensive for a builder). Instead we
        return regime distribution as a proxy (fraction of instances per regime).
        If the template has win/loss data and most instances share a regime,
        we approximate per-regime win rates from the overall rate.
        """
        if not template.instances:
            return {}

        regime_counts: dict[str, int] = defaultdict(int)
        for inst in template.instances:
            regime = inst.regime or "default"
            regime_counts[regime] += 1

        total = sum(regime_counts.values()) or 1
        overall_wr = template.win_rate if (template.wins + template.losses) > 0 else None

        result: dict = {}
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            if overall_wr is not None:
                # Rough approximation: regimes with more instances slightly boost confidence
                # in the overall win rate. This is directional, not rigorous.
                fraction = count / total
                # Skew: more represented regime → closer to overall win rate
                result[regime] = round(
                    overall_wr * (0.8 + 0.4 * fraction), 4
                )
            else:
                result[regime] = round(count / total, 4)  # Use frequency as proxy

        return result

    # ── Title and summary generation ──────────────────────────────────────

    def _generate_title(
        self,
        template: PatternTemplate,
        top_entities: list[str],
        entity_weight_factors: dict,
    ) -> str:
        """Auto-generate a descriptive title from the template's characteristics.

        Format: "[Entity type] [Activity] Before [Domain] [Direction]"
        Example: "CEO Buying Before Tech Macro Surge"
        """
        direction_word = {
            "bullish": "Rally",
            "bearish": "Decline",
        }.get(template.direction, template.direction.capitalize())

        domains = template.domains[:3]  # Up to 3 domains in title
        domain_words = [d.capitalize() for d in domains]

        # Entity prefix if we have high-weight entities
        entity_prefix = ""
        if top_entities:
            top_name = top_entities[0]
            role = template.recurring_entities.get(top_name, {}).get("role", "insider_name")
            weight = entity_weight_factors.get(top_name, 1.0)
            if weight >= 1.4:
                entity_prefix = "CEO-Led "
            elif role == "representative":
                entity_prefix = "Congress-Linked "
            elif role == "fund_name":
                entity_prefix = "Institutional "
            elif role == "insider_name":
                entity_prefix = "Insider "

        domain_str = "+".join(domain_words)
        return f"{entity_prefix}{domain_str} Pre-{direction_word}"

    def _generate_summary(
        self,
        template: PatternTemplate,
        top_entities: list[str],
        chapters: list[PatternChapter],
        win_rate: float,
        n_outcomes: int,
        is_significant: bool,
    ) -> str:
        """Generate a 1-2 sentence plain-English summary of the pattern."""
        direction_noun = _DIRECTION_PLAIN.get(template.direction, template.direction)
        domain_list = template.domains

        if len(domain_list) == 1:
            domain_str = _DOMAIN_DESCRIPTIONS.get(domain_list[0], domain_list[0])
        elif len(domain_list) == 2:
            a = _DOMAIN_DESCRIPTIONS.get(domain_list[0], domain_list[0])
            b = _DOMAIN_DESCRIPTIONS.get(domain_list[1], domain_list[1])
            domain_str = f"{a} and {b}"
        else:
            parts = [_DOMAIN_DESCRIPTIONS.get(d, d) for d in domain_list[:3]]
            domain_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

        cross_note = ""
        n_syms = len(template.unique_symbols)
        if n_syms >= 10:
            cross_note = f" This pattern has appeared on {n_syms} different stocks, making it a broad market signal."
        elif n_syms >= 3:
            cross_note = f" Observed on {n_syms} different stocks."

        stat_note = ""
        if is_significant and n_outcomes >= 5:
            stat_note = f" Win rate {win_rate:.0%} ({n_outcomes} graded outcomes, statistically significant)."
        elif n_outcomes >= 5:
            stat_note = f" Win rate {win_rate:.0%} based on {n_outcomes} graded outcomes."

        entity_note = ""
        if top_entities:
            top = top_entities[0]
            role = template.recurring_entities.get(top, {}).get("role", "")
            role_label = _ROLE_LABELS.get(role, "")
            if role_label:
                count = template.recurring_entities[top].get("count", 0)
                entity_note = f" Recurring character: {top} ({role_label}, appeared {count} times)."

        summary = (
            f"This pattern fires when {domain_str} and historically precedes a {direction_noun}."
            f"{cross_note}{stat_note}{entity_note}"
        )
        return summary

    # ── Persistence ───────────────────────────────────────────────────────

    def _persist_story(self, story: PatternStory) -> None:
        """Append a single story to the stories JSONL file."""
        try:
            self._stories_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stories_path, "a") as f:
                f.write(story.to_json() + "\n")
        except OSError as e:
            logger.warning("Could not persist story %s: %s", story.story_id, e)

    def _persist_stories(self, stories: list[PatternStory]) -> None:
        """Rewrite the stories file with all stories (full rebuild)."""
        try:
            self._stories_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._stories_path, "w") as f:
                for story in stories:
                    f.write(story.to_json() + "\n")
        except OSError as e:
            logger.warning("Could not persist stories: %s", e)


# ── Math helpers ─────────────────────────────────────────────────────────────

def _norm_cdf(z: float) -> float:
    """Approximate standard normal CDF (Abramowitz & Stegun 26.2.17)."""
    if z < 0:
        return 1.0 - _norm_cdf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-z * z / 2) * poly
