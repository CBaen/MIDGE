"""Causal story templates and auto-generation for HypothesisGenerator."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Minimum thresholds for hypothesis generation ────────────────────
# Fallback values if learning_config is unavailable
_GEN_FALLBACKS = {"min_correlation": 0.6, "min_pairs": 10}


def _get_gen_threshold(key: str) -> float:
    """Read a generator threshold from learning_config, with fallback.

    Graceful degradation: if config import fails or key is missing,
    returns the hardcoded fallback value.
    """
    try:
        from mae_core.market.intelligence.learning_config import LEARNING_CONFIG
        cfg = LEARNING_CONFIG.get("generator_thresholds", {})
        return float(cfg.get(key, _GEN_FALLBACKS.get(key, 0.0)))
    except ImportError:
        return float(_GEN_FALLBACKS.get(key, 0.0))


# ── Domain role classification ───────────────────────────────────────
# Maps each source to (domain_category, activity_description).
# Used by _auto_generate_causal_story() to produce stories for unknown pairs.

_DOMAIN_ROLES = {
    "sec_form4": ("insider", "equity_trading"),
    "sec_form8k": ("insider", "material_disclosure"),
    "sec_efts": ("regulatory", "filing_activity"),
    "congressional": ("political", "equity_trading"),
    "senate": ("political", "equity_trading"),
    "insider_cluster": ("insider", "coordinated_buying"),
    "contract_award": ("government", "procurement"),
    "contract_prediction": ("government", "procurement_forecast"),
    "hiring_tracker": ("corporate", "workforce_expansion"),
    "sam_gov": ("government", "opportunity_posting"),
    "social_sentiment": ("social", "retail_sentiment"),
    "finra_short": ("institutional", "short_positioning"),
    "finnhub_news": ("information", "news_flow"),
    "finnhub_earnings": ("fundamental", "earnings_reporting"),
    "fred_macro": ("macro", "economic_indicator"),
    "session_sweep": ("technical", "liquidity_sweep"),
    "session_sweep_ifvg": ("technical", "liquidity_sweep_fvg"),
    "ta_rsi": ("technical", "momentum_indicator"),
    "ta_macd": ("technical", "trend_indicator"),
    "ta_bollinger": ("technical", "volatility_band"),
    "ta_structure": ("technical", "market_structure"),
    "ta_candle": ("technical", "candlestick_pattern"),
    "cot_positioning": ("institutional", "futures_positioning"),
    "stocktwits_sentiment": ("social", "retail_sentiment"),
    "vix_term_structure": ("volatility", "fear_gauge"),
    "google_trends": ("social", "search_attention"),
    "finnhub_economic": ("macro", "economic_calendar"),
    "finnhub_analyst": ("fundamental", "analyst_consensus"),
    "finnhub_earnings_calendar": ("fundamental", "earnings_schedule"),
    "yfinance_price": ("technical", "price_action"),
    "openinsider_purchase": ("insider", "insider_buying"),
    "openinsider_cluster": ("insider", "coordinated_buying"),
    "massive_snapshot": ("technical", "volume_anomaly"),
    "eia_energy": ("macro", "energy_supply"),
    "congress_legislation": ("political", "legislative_activity"),
    "economic_calendar": ("macro", "economic_event"),
    "activist_13d": ("institutional", "activist_position"),
    "institutional_13f": ("institutional", "portfolio_rebalancing"),
    "crypto_coingecko": ("crypto", "price_action"),
    "crypto_coincap": ("crypto", "exchange_volume"),
    "yahoo_rss": ("information", "headline_velocity"),
    "finviz_unusual_volume": ("technical", "volume_anomaly"),
    "finviz_short_squeeze": ("institutional", "short_positioning"),
}


def _auto_generate_causal_story(source_a: str, source_b: str) -> str:
    """Generate a causal story from domain roles when no template exists.

    Looks up both sources in _DOMAIN_ROLES, then applies a role-pair matrix
    to produce a plausible story. Returns empty string if either source is unknown.

    Stories are prefixed with [AUTO] to distinguish from human-written ones.
    """
    role_a = _DOMAIN_ROLES.get(source_a)
    role_b = _DOMAIN_ROLES.get(source_b)

    if role_a is None or role_b is None:
        return ""

    domain_a, activity_a = role_a
    domain_b, activity_b = role_b

    # Role-pair matrix: ordered by specificity (most specific first)

    # Insider/political → anything fundamental or reporting
    if (domain_a in ("insider", "political")
            and domain_b in ("fundamental", "regulatory", "information")):
        return (
            f"[AUTO] Information advantage: {domain_a} participants ({activity_a}) "
            f"may have advance knowledge of {activity_b} events, creating a lead-lag "
            f"relationship where informed actors move before public disclosure."
        )

    # Technical → technical (independent confirmation)
    if domain_a == "technical" and domain_b == "technical":
        return (
            f"[AUTO] Technical confirmation: {activity_a} and {activity_b} are independent "
            f"indicators that, when aligned, strengthen the directional signal by reducing "
            f"the probability of false positives through multi-indicator consensus."
        )

    # Macro → institutional (cascade effect)
    if domain_a == "macro" and domain_b in ("institutional", "technical"):
        return (
            f"[AUTO] Macro-institutional cascade: {activity_a} changes drive {activity_b} "
            f"positioning as institutions respond to economic signals, creating measurable "
            f"lead-lag dynamics in market structure."
        )

    # Social → fundamental (attention/front-running)
    if domain_a == "social" and domain_b in ("fundamental", "information"):
        return (
            f"[AUTO] Attention-fundamental link: {activity_a} may front-run or react to "
            f"{activity_b} events, as retail attention often anticipates or amplifies "
            f"fundamental disclosures."
        )

    # Institutional → fundamental (positioning before events)
    if domain_a == "institutional" and domain_b == "fundamental":
        return (
            f"[AUTO] Institutional positioning: {activity_a} changes reflect informed "
            f"expectations about {activity_b} outcomes, as large participants position "
            f"before scheduled events based on proprietary research."
        )

    # Volatility → any (fear/regime signal)
    if domain_a == "volatility":
        return (
            f"[AUTO] Volatility regime signal: {activity_a} levels precede changes in "
            f"{activity_b} as fear/complacency cycles drive participant behavior across "
            f"asset classes and trading styles."
        )

    # Government → corporate (procurement signal chain)
    if domain_a == "government" and domain_b == "corporate":
        return (
            f"[AUTO] Government-corporate signal chain: {activity_a} activity leads "
            f"{activity_b} responses as companies prepare for or react to public sector "
            f"procurement cycles."
        )

    # Insider → government (regulatory foresight)
    if domain_a == "insider" and domain_b == "government":
        return (
            f"[AUTO] Regulatory foresight: {activity_a} may reflect insider awareness of "
            f"upcoming {activity_b} decisions, particularly for defense, healthcare, and "
            f"infrastructure sectors."
        )

    # Generic fallback for any remaining known pairs
    return (
        f"[AUTO] Cross-domain correlation: {domain_a} {activity_a} leads {domain_b} "
        f"{activity_b}. Mechanism requires further research, but statistical signal "
        f"justifies monitoring while causal story develops."
    )


CAUSAL_STORY_TEMPLATES = {
    ("sec_form4", "finnhub_earnings"): (
        "Insider trading activity (Form 4) precedes earnings announcements. "
        "Insiders have material non-public information about upcoming results. "
        "Lakonishok & Lee (2001) demonstrated insider alpha persists 3-12 months."
    ),
    ("congressional", "finra_short"): (
        "Congressional trades precede short interest changes. Members with "
        "committee access may trade ahead of regulatory or legislative actions "
        "that affect short sellers. STOCK Act disclosure lag creates "
        "information asymmetry."
    ),
    ("finra_short", "finnhub_earnings"): (
        "Short interest changes precede earnings surprises. Short sellers "
        "conduct deep fundamental analysis; rising short interest before "
        "earnings correlates with negative surprises (Desai et al. 2002)."
    ),
    ("sec_form4", "sec_efts"): (
        "Insider trades precede SEC filing activity. Insiders may trade "
        "before major filings are submitted. SEC EFTS keyword matches "
        "catch the filing; Form 4 catches the trade."
    ),
    ("sec_efts", "congressional"): (
        "SEC filing text mentions precede congressional trades. Material "
        "events disclosed in SEC filings (contracts, M&A, regulatory) may "
        "inform committee members' trading decisions."
    ),
    ("finra_short", "congressional"): (
        "Short interest changes precede congressional trading. Short sellers' "
        "public positions signal market sentiment that members observe "
        "before trading."
    ),
    ("insider_cluster", "finnhub_earnings"): (
        "Insider buying clusters (3+ officers) precede earnings outcomes. "
        "Alldredge (2019) showed cluster alpha peaks 40-80 days. Coordinated "
        "insider buying signals management conviction about upcoming results."
    ),
    ("sec_form8k", "sec_form4"): (
        "Material events (8-K) trigger subsequent insider trades. After a "
        "material event disclosure, insiders trade on their assessment of "
        "the market's reaction efficiency."
    ),
    ("hiring_tracker", "contract_award"): (
        "Hiring surges precede government contract awards. Companies hire "
        "before contract execution begins. Hiring lead time is 60-120 days "
        "before contract announcement."
    ),
    ("sam_gov", "contract_award"): (
        "SAM.gov opportunity postings lead to contract awards. The "
        "solicitation → proposal → evaluation → award pipeline has "
        "predictable timing by agency and contract type."
    ),
    ("fred_macro", "finra_short"): (
        "Macroeconomic indicator changes precede short interest shifts. "
        "FRED data releases (GDP, CPI, unemployment) signal regime changes "
        "that drive short positioning."
    ),
}


def _get_causal_story(source_a: str, source_b: str) -> str:
    """Look up causal story for a source pair (order-independent).

    Priority:
    1. Hardcoded CAUSAL_STORY_TEMPLATES (human-written, highest quality)
    2. _auto_generate_causal_story() using _DOMAIN_ROLES (auto, slightly tighter gates)
    3. "REQUIRES MANUAL REVIEW" only when both sources are completely unknown
    """
    story = CAUSAL_STORY_TEMPLATES.get((source_a, source_b))
    if story:
        return story
    story = CAUSAL_STORY_TEMPLATES.get((source_b, source_a))
    if story:
        return story

    # Attempt auto-generation from domain roles
    auto_story = _auto_generate_causal_story(source_a, source_b)
    if auto_story:
        return auto_story

    return (
        f"REQUIRES MANUAL REVIEW: Statistical correlation found between "
        f"{source_a} and {source_b} but no known causal mechanism. "
        f"Do not promote until a causal story is established."
    )
