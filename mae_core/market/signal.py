"""Canonical signal types for MIDGE market intelligence.

MarketSignal is the normalized format for all market data flowing through MIDGE.
TradeSignal is the actionable output — what to trade and why.

Adapter functions convert raw source types into MarketSignal.
Each adapter is a standalone function named `from_{source_type}`.
"""

from __future__ import annotations

import math
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from mae_core.market.apis.sec_edgar.models import InsiderTrade, Form8KEvent
from mae_core.market.apis.house_stock_watcher import CongressionalTrade
from mae_core.market.apis.job_tracker import HiringSignal
from mae_core.market.apis.usa_spending import GovernmentContract
from mae_core.market.apis.sam_gov import ContractOpportunity
from mae_core.market.edge.cluster_detector import ClusterSignal
from mae_core.market.edge.contract_predictor import ContractPrediction
from mae_core.market.edge.politician_tracker import CorrelationSignal
from mae_core.market.apis import ticker_resolver

# Lazy imports for new sources (avoid circular imports / missing deps)
# SocialSentiment, ShortInterestData, FilingKeywordHit, NewsSentiment,
# EarningsEvent, MacroIndicator are imported inside their adapter functions.

logger = logging.getLogger(__name__)


def _ensure_datetime(value) -> datetime:
    """Coerce a string or datetime to datetime. Returns now() on failure."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                    "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(value.split("T")[0] if "T" in value else value, fmt)
            except ValueError:
                continue
        logger.debug("Could not parse date string %r, using now()", value)
    return datetime.now()


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MarketSignal:
    """Canonical normalized signal format for all market data flowing through MIDGE."""

    # Identity
    signal_id: str          # UUID or "{source}:{symbol}:{timestamp}"
    source: str             # "sec_form4", "congressional", "hiring_tracker", etc.
    symbol: str             # Ticker. Empty string ("") for macro/pre-ticker signals.
    asset_class: str        # "stock", "crypto", "futures", "commodities", "macro"

    # Classification
    domain: str             # "insider", "congress", "contracts", "government",
                            # "technical", "sentiment", "news", "institutional"
    direction: str          # "bullish", "bearish", "neutral"
    strength: float         # 0.0–1.0 normalized

    # Reliability
    confidence: float       # Source reliability estimate 0.0–1.0
    decay_rate: float       # Per-day decay

    # Time
    timestamp: datetime     # When the UNDERLYING EVENT occurred
    received_at: datetime   # When MIDGE received/detected the signal

    # Velocity — populated by VelocityDetector
    velocity: float = 0.0  # Per-DAY velocity. Default 0.0 before wiring.

    # Feedback loop
    outcome_symbol: str = ""         # Ticker to check for price outcome
    outcome_window_days: int = 14    # Default 14 days forward

    # Audit trail
    raw_id: str = ""        # Identifier of original record in source system
    raw_type: str = ""      # "InsiderTrade", "Form8KEvent", etc.

    # Pattern discovery context
    metadata: dict = field(default_factory=dict)


@dataclass
class TradeSignal:
    """Actionable output: what to trade, when, and why."""
    signal_id: str
    asset: str              # Ticker or asset name
    asset_class: str        # "stock", "crypto", "futures", "commodities"
    direction: str          # "buy" or "sell"
    confidence: float       # 0.0–1.0
    timeframe_days: int     # Expected holding period
    catalyst: str           # Human-readable reason
    contributing_signals: list = field(default_factory=list)  # MarketSignal IDs
    hit_rate: float = 0.0   # Historical accuracy for this signal type
    generated_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def from_insider_trade(trade: InsiderTrade) -> MarketSignal:
    """Convert a Form 4 InsiderTrade to a MarketSignal.

    Filters out noise from scheduled compensation transactions:
    - Transaction code "D" (disposition/delivery for RSU vesting): strength * 0.25
    - Transaction code "A" (award/grant): strength * 0.25
    - Transaction code "F" (tax withholding on vesting): strength * 0.25
    - Suspected 10b5-1 plan sales (non-buy with plan indicators): strength * 0.25
    """
    is_buy = trade.is_purchase
    direction = "bullish" if is_buy else "bearish"

    # Log-linear scale: preserves differentiation at high values
    # $100k -> 0.42, $500k -> 0.73, $1M -> 0.83, $5M -> 0.95, $10M -> 1.0
    if is_buy:
        strength = min(1.0, math.log1p(trade.total_value / 100_000) / math.log1p(10))
    else:
        strength = min(1.0, math.log1p(trade.total_value / 50_000) / math.log1p(10))

    confidence = 0.70

    # Detect compensation/plan transactions (noise, not informed trading)
    compensation_codes = {"D", "A", "F", "G", "M"}  # disposition, award, tax, gift, option exercise
    tc = trade.transaction_code.upper().strip() if trade.transaction_code else ""
    is_compensation = tc in compensation_codes
    # 10b5-1 plan sales: scheduled, not informed (Pichai, Kress, etc.)
    is_plan = trade.is_plan_sale and not is_buy

    if is_compensation or is_plan:
        strength *= 0.25
        confidence = 0.40

    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.filing_date) if trade.filing_date else datetime.now()

    symbol = trade.ticker_symbol or ""
    signal_id = f"sec_form4:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_form4",
        symbol=symbol,
        asset_class="stock",
        domain="insider",
        direction=direction,
        strength=strength,
        confidence=confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.accession_number or "",
        raw_type="InsiderTrade",
        metadata={
            "filer_name": trade.filer_name,
            "filer_title": trade.filer_title,
            "transaction_type": trade.transaction_type,
            "shares": trade.shares,
            "price_per_share": trade.price_per_share,
            "total_value": trade.total_value,
            "company_name": trade.company_name,
        },
    )


def from_form8k_event(event: Form8KEvent) -> MarketSignal:
    """Convert a Form 8-K event to a MarketSignal."""
    # Direction from the item code's known impact
    _, impact = Form8KEvent.get_item_info(event.item_code)
    direction = impact if impact in ("bullish", "bearish") else "neutral"

    event_dt = _ensure_datetime(event.event_date)
    received_dt = _ensure_datetime(event.filing_date) if event.filing_date else datetime.now()

    symbol = event.ticker_symbol or ""
    signal_id = f"sec_form8k:{symbol}:{event.event_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_form8k",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=event.confidence,  # 0.50–0.70 range per source model
        confidence=event.confidence,
        decay_rate=event.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=event.accession_number or "",
        raw_type="Form8KEvent",
        metadata={
            "item_code": event.item_code,
            "item_description": event.item_description,
            "event_summary": event.event_summary,
            "material_impact": event.material_impact,
            "company_name": event.company_name,
        },
    )


def from_congressional_trade(trade: CongressionalTrade) -> MarketSignal:
    """Convert a CongressionalTrade to a MarketSignal.

    Critical: timestamp=transaction_date (when the trade occurred, NOT when disclosed).
    received_at=disclosure_date (when MIDGE/public learned of it).
    """
    is_buy = "purchase" in trade.transaction_type.lower()
    direction = "bullish" if is_buy else "bearish"

    # Strength from upper bound of trade amount range
    strength = min(1.0, trade.amount_high / 500_000)

    # transaction_date is the event; disclosure_date is when MIDGE received it
    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.disclosure_date) if trade.disclosure_date else datetime.now()

    symbol = trade.ticker or ""
    signal_id = f"congressional:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="congressional",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=strength,
        confidence=trade.confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.disclosure_url or "",
        raw_type="CongressionalTrade",
        metadata={
            "representative": trade.representative,
            "party": trade.party,
            "district": trade.district,
            "transaction_type": trade.transaction_type,
            "amount_range": trade.amount_range,
            "amount_low": trade.amount_low,
            "amount_high": trade.amount_high,
            "owner": trade.owner,
            "asset_description": trade.asset_description,
        },
    )


def from_hiring_signal(signal: HiringSignal) -> MarketSignal:
    """Convert a HiringSignal to a MarketSignal."""
    # Hiring blitzes are bullish pre-announcement indicators
    direction = "bullish" if signal.is_spike else "neutral"

    strength = min(1.0, signal.spike_ratio / 5.0)

    event_dt = _ensure_datetime(signal.detected_at)

    symbol = signal.ticker or ""
    signal_id = f"hiring_tracker:{symbol or signal.company_name}:{signal.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="hiring_tracker",
        symbol=symbol,
        asset_class="stock",
        domain="institutional",
        direction=direction,
        strength=strength,
        confidence=signal.confidence,
        decay_rate=signal.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol,
        raw_id="",
        raw_type="HiringSignal",
        metadata={
            "company_name": signal.company_name,
            "jobs_24h": signal.jobs_24h,
            "jobs_7d": signal.jobs_7d,
            "jobs_30d": signal.jobs_30d,
            "spike_ratio": signal.spike_ratio,
            "is_spike": signal.is_spike,
            "engineering_jobs": signal.engineering_jobs,
            "cleared_jobs": signal.cleared_jobs,
            "contract_related_jobs": signal.contract_related_jobs,
        },
    )


def from_government_contract(contract: GovernmentContract) -> MarketSignal:
    """Convert a GovernmentContract award to a MarketSignal."""
    # A contract award is bullish for the recipient
    direction = "bullish"

    strength = min(1.0, contract.award_amount / 100_000_000)

    event_dt = _ensure_datetime(contract.award_date)

    # Resolve company name to ticker via TickerResolver
    resolved_ticker = ticker_resolver.resolve(contract.recipient_name) or ""

    signal_id = f"contract_award:{contract.award_id}:{contract.award_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="contract_award",
        symbol=resolved_ticker,
        asset_class="stock",
        domain="contracts",
        direction=direction,
        strength=strength,
        confidence=0.75,
        decay_rate=contract.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=resolved_ticker,
        raw_id=contract.award_id or "",
        raw_type="GovernmentContract",
        metadata={
            "recipient_name": contract.recipient_name,
            "award_amount": contract.award_amount,
            "award_type": contract.award_type,
            "awarding_agency": contract.awarding_agency,
            "description": contract.description,
            "naics_code": contract.naics_code,
            "start_date": contract.start_date,
            "end_date": contract.end_date,
        },
    )


def from_contract_opportunity(opportunity: ContractOpportunity) -> MarketSignal:
    """Convert a SAM.gov ContractOpportunity to a MarketSignal.

    Opportunities are weaker than awards — strength is fixed at 0.3.
    No ticker is known at this stage.
    """
    event_dt = _ensure_datetime(opportunity.posted_date)

    signal_id = f"sam_gov:{opportunity.notice_id}:{opportunity.posted_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sam_gov",
        symbol="",
        asset_class="stock",
        domain="contracts",
        direction="neutral",
        strength=0.3,
        confidence=0.40,
        decay_rate=0.008,  # ~87 day half-life (competition periods last months)
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="",
        raw_id=opportunity.notice_id or "",
        raw_type="ContractOpportunity",
        metadata={
            "title": opportunity.title,
            "department": opportunity.department,
            "agency": opportunity.agency,
            "naics_code": opportunity.naics_code,
            "estimated_value": opportunity.estimated_value,
            "response_deadline": opportunity.response_deadline,
            "contract_type": opportunity.contract_type,
            "url": opportunity.url,
        },
    )


def from_cluster_signal(cluster: ClusterSignal) -> MarketSignal:
    """Convert an insider ClusterSignal to a MarketSignal."""
    # Clusters are always bullish (cluster_detector only tracks buys)
    direction = "bullish"

    event_dt = _ensure_datetime(cluster.detected_at)

    signal_id = f"insider_cluster:{cluster.symbol}:{cluster.cluster_id}"

    return MarketSignal(
        signal_id=signal_id,
        source="insider_cluster",
        symbol=cluster.symbol,
        asset_class="stock",
        domain="insider",
        direction=direction,
        strength=cluster.confidence,
        confidence=cluster.confidence,
        decay_rate=cluster.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=cluster.symbol,
        raw_id=cluster.cluster_id or "",
        raw_type="ClusterSignal",
        metadata={
            "insider_count": cluster.insider_count,
            "total_value": cluster.total_value,
            "weighted_score": cluster.weighted_score,
            "avg_conviction": cluster.avg_conviction,
            "has_csuite": cluster.has_csuite,
            "window_days": cluster.window_days,
        },
    )


def from_contract_prediction(prediction: ContractPrediction) -> MarketSignal:
    """Convert a ContractPrediction to a MarketSignal.

    Domain is "institutional_synthesis" to prevent double-counting with the
    individual hiring and insider signals that fed into this prediction.
    """
    direction = "bullish"

    event_dt = _ensure_datetime(prediction.predicted_at)

    ticker = prediction.predicted_ticker or ""
    signal_id = f"contract_prediction:{ticker or prediction.predicted_winner}:{prediction.predicted_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="contract_prediction",
        symbol=ticker,
        asset_class="stock",
        domain="institutional_synthesis",
        direction=direction,
        strength=prediction.confidence,
        confidence=prediction.confidence,
        decay_rate=prediction.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=ticker,
        raw_id="",
        raw_type="ContractPrediction",
        metadata={
            "predicted_winner": prediction.predicted_winner,
            "contract_title": prediction.contract_title,
            "contract_value": prediction.contract_value,
            "hiring_blitz_detected": prediction.hiring_blitz_detected,
            "hiring_spike_ratio": prediction.hiring_spike_ratio,
            "insider_buying_detected": prediction.insider_buying_detected,
            "insider_buy_value": prediction.insider_buy_value,
            "confidence_breakdown": prediction.confidence_breakdown,
            "contract_deadline": prediction.contract_deadline,
            "expected_award_date": prediction.expected_award_date,
        },
    )


def from_correlation_signal(correlation: CorrelationSignal) -> MarketSignal:
    """Convert a politician/insider CorrelationSignal to a MarketSignal."""
    is_buy = correlation.trade_type == "buy" if hasattr(correlation, "trade_type") else True
    direction = "bullish" if is_buy else "bearish"

    event_dt = _ensure_datetime(correlation.trade_date)

    symbol = correlation.symbol or ""
    signal_id = f"politician_correlation:{symbol}:{correlation.trade_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="politician_correlation",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=correlation.confidence,
        confidence=correlation.confidence,
        decay_rate=0.04,  # ~17 day half-life (combination of trade staleness + political info)
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="CorrelationSignal",
        metadata={
            "trader_name": correlation.trader_name,
            "trade_value": correlation.value,
            "correlation_type": correlation.correlation_type,
            "contract_value": correlation.contract_value,
            "awarding_agency": correlation.awarding_agency,
            "committee": correlation.committee,
            "oversight_match": correlation.oversight_match,
            "days_between": correlation.days_between,
        },
    )


# ---------------------------------------------------------------------------
# Price-action adapters (Layer 5 Phase 3 expansion)
# ---------------------------------------------------------------------------


def from_session_sweep(sweep) -> MarketSignal:
    """Convert a SessionSweepSignal to a MarketSignal.

    ICT session liquidity sweep → tactical signal. Decay is hourly-scale
    (~18h half-life). Outcome checked by next session (1 day).
    """
    event_dt = _ensure_datetime(sweep.detected_at)

    signal_id = (
        f"session_sweep:{sweep.symbol}:{sweep.session_swept}:"
        f"{sweep.sweep_type}:{sweep.detected_at}"
    )

    return MarketSignal(
        signal_id=signal_id,
        source="session_sweep",
        symbol=sweep.symbol,
        asset_class="futures",
        domain="technical",
        direction=sweep.direction,
        strength=sweep.strength,
        confidence=sweep.confidence,
        decay_rate=sweep.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=sweep.symbol,
        outcome_window_days=1,
        raw_id=sweep.sweep_id or "",
        raw_type="SessionSweepSignal",
        metadata={
            "sweep_type": sweep.sweep_type,
            "session_swept": sweep.session_swept,
            "sweep_level": sweep.sweep_level,
            "fvg_top": sweep.fvg_top,
            "fvg_bottom": sweep.fvg_bottom,
            "entry_zone_top": sweep.entry_zone_top,
            "entry_zone_bottom": sweep.entry_zone_bottom,
            "stop_level": sweep.stop_level,
            "target_level": sweep.target_level,
            "rr_ratio": sweep.rr_ratio,
            "kill_zone": sweep.kill_zone,
        },
    )


# ---------------------------------------------------------------------------
# New source adapters (Phase 2 expansion)
# ---------------------------------------------------------------------------

def from_senate_trade(trade: "CongressionalTrade") -> MarketSignal:
    """Convert a Senate CongressionalTrade to a MarketSignal.

    Same logic as from_congressional_trade but tagged as senate source.
    Uses same CongressionalTrade dataclass — Senate Stock Watcher produces
    the same shape.
    """
    is_buy = "purchase" in trade.transaction_type.lower()
    direction = "bullish" if is_buy else "bearish"

    strength = min(1.0, trade.amount_high / 500_000)

    event_dt = _ensure_datetime(trade.transaction_date)
    received_dt = _ensure_datetime(trade.disclosure_date) if trade.disclosure_date else datetime.now()

    symbol = trade.ticker or ""
    signal_id = f"senate:{symbol}:{trade.transaction_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="senate",
        symbol=symbol,
        asset_class="stock",
        domain="congress",
        direction=direction,
        strength=strength,
        confidence=trade.confidence,
        decay_rate=trade.decay_rate,
        timestamp=event_dt,
        received_at=received_dt,
        outcome_symbol=symbol,
        raw_id=trade.disclosure_url or "",
        raw_type="CongressionalTrade",
        metadata={
            "representative": trade.representative,
            "party": trade.party,
            "chamber": "Senate",
            "district": trade.district,
            "transaction_type": trade.transaction_type,
            "amount_range": trade.amount_range,
            "amount_low": trade.amount_low,
            "amount_high": trade.amount_high,
            "owner": trade.owner,
        },
    )


def from_social_sentiment(sentiment) -> MarketSignal:
    """Convert an ApeWisdom SocialSentiment to a MarketSignal.

    The signal is in the mention velocity (mention_change), not the
    absolute count. A ticker going from 10 to 100 mentions is more
    actionable than one steady at 500.
    """
    # Direction from velocity: accelerating mentions = bullish retail momentum
    if sentiment.mention_change >= 3.0:
        direction = "bullish"
    elif sentiment.mention_change <= 0.3:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from mention velocity (log scale)
    strength = min(1.0, math.log1p(sentiment.mention_change) / math.log1p(10))

    event_dt = _ensure_datetime(sentiment.detected_at)

    symbol = sentiment.ticker or ""
    signal_id = f"social:{symbol}:{sentiment.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="social_sentiment",
        symbol=symbol,
        asset_class="stock",
        domain="sentiment",
        direction=direction,
        strength=strength,
        confidence=sentiment.confidence,
        decay_rate=sentiment.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol,
        raw_id="",
        raw_type="SocialSentiment",
        metadata={
            "mentions_24h": sentiment.mentions_24h,
            "mentions_prior_24h": sentiment.mentions_prior_24h,
            "mention_change": sentiment.mention_change,
            "upvotes": sentiment.upvotes,
            "rank": sentiment.rank,
            "source_subreddit": sentiment.source_subreddit,
        },
    )


def from_short_interest(data) -> MarketSignal:
    """Convert a FINRA ShortInterestData to a MarketSignal.

    High short ratio (>50% of daily volume) is a potential squeeze signal.
    When combined with insider buying clusters, this becomes very actionable.
    """
    # Direction: high short ratio is bearish pressure, but also squeeze potential
    if data.short_ratio >= 0.6:
        direction = "bearish"  # Heavy shorting pressure
    elif data.short_ratio >= 0.4:
        direction = "neutral"  # Elevated but not extreme
    else:
        direction = "neutral"

    # Strength from short ratio
    strength = min(1.0, data.short_ratio / 0.7)

    event_dt = _ensure_datetime(data.date)

    symbol = data.symbol or ""
    signal_id = f"finra_short:{symbol}:{data.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="finra_short",
        symbol=symbol,
        asset_class="stock",
        domain="institutional",
        direction=direction,
        strength=strength,
        confidence=data.confidence,
        decay_rate=data.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="ShortInterestData",
        metadata={
            "short_volume": data.short_volume,
            "total_volume": data.total_volume,
            "short_ratio": data.short_ratio,
        },
    )


def from_filing_keyword(hit) -> MarketSignal:
    """Convert a SEC EFTS FilingKeywordHit to a MarketSignal.

    Keyword matches in 8-K filings are event-driven signals.
    "Tender offer" = M&A bullish. "Going concern" = distress bearish.
    """
    from mae_core.market.apis.sec_edgar.efts import SECFullTextSearchClient

    # Direction from keyword category
    keyword_lower = hit.keyword_matched.lower()
    if keyword_lower in [k.lower() for k in SECFullTextSearchClient.BULLISH_KEYWORDS]:
        direction = "bullish"
    elif keyword_lower in [k.lower() for k in SECFullTextSearchClient.BEARISH_KEYWORDS]:
        direction = "bearish"
    else:
        direction = "neutral"

    # M&A keywords are higher strength than general distress
    if "tender offer" in keyword_lower or "merger" in keyword_lower:
        strength = 0.90
    elif "going concern" in keyword_lower or "restatement" in keyword_lower:
        strength = 0.85
    else:
        strength = 0.65

    event_dt = _ensure_datetime(hit.filing_date)

    symbol = hit.ticker or ""
    signal_id = f"sec_efts:{symbol}:{hit.keyword_matched}:{hit.filing_date}"

    return MarketSignal(
        signal_id=signal_id,
        source="sec_efts",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=hit.confidence,
        decay_rate=hit.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="FilingKeywordHit",
        metadata={
            "keyword_matched": hit.keyword_matched,
            "form_type": hit.form_type,
            "description": hit.description,
            "company_name": hit.company_name,
        },
    )


def from_news_sentiment(sentiment) -> MarketSignal:
    """Convert a Finnhub NewsSentiment to a MarketSignal.

    The buzz score (articles this week vs weekly average) is the velocity.
    Bullish/bearish from Finnhub's own sentiment analysis.
    """
    if sentiment.bullish_pct >= 0.6:
        direction = "bullish"
    elif sentiment.bearish_pct >= 0.6:
        direction = "bearish"
    else:
        direction = "neutral"

    # Strength from buzz score (how much more attention than normal)
    strength = min(1.0, sentiment.buzz_score / 3.0)

    event_dt = _ensure_datetime(sentiment.detected_at)

    symbol = sentiment.ticker or ""
    signal_id = f"finnhub_news:{symbol}:{sentiment.detected_at}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_news",
        symbol=symbol,
        asset_class="stock",
        domain="news",
        direction=direction,
        strength=strength,
        confidence=sentiment.confidence,
        decay_rate=sentiment.decay_rate,
        timestamp=event_dt,
        received_at=event_dt,
        outcome_symbol=symbol,
        raw_id="",
        raw_type="NewsSentiment",
        metadata={
            "bullish_pct": sentiment.bullish_pct,
            "bearish_pct": sentiment.bearish_pct,
            "news_score": sentiment.news_score,
            "buzz_score": sentiment.buzz_score,
        },
    )


def from_earnings_event(event) -> MarketSignal:
    """Convert a Finnhub EarningsEvent to a MarketSignal.

    Pre-earnings: neutral signal (catalyst approaching).
    Post-earnings with surprise: strong directional signal.
    """
    if event.eps_actual is not None and event.eps_estimate is not None:
        # Post-earnings: direction from surprise
        surprise = event.eps_actual - event.eps_estimate
        if surprise > 0:
            direction = "bullish"
            strength = min(1.0, abs(surprise) / max(0.01, abs(event.eps_estimate)) * 2)
        elif surprise < 0:
            direction = "bearish"
            strength = min(1.0, abs(surprise) / max(0.01, abs(event.eps_estimate)) * 2)
        else:
            direction = "neutral"
            strength = 0.3
    else:
        # Pre-earnings: catalyst approaching
        direction = "neutral"
        strength = 0.4

    event_dt = _ensure_datetime(event.date)

    symbol = event.symbol or ""
    signal_id = f"finnhub_earnings:{symbol}:{event.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="finnhub_earnings",
        symbol=symbol,
        asset_class="stock",
        domain="events",
        direction=direction,
        strength=strength,
        confidence=event.confidence,
        decay_rate=event.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="EarningsEvent",
        metadata={
            "eps_estimate": event.eps_estimate,
            "eps_actual": event.eps_actual,
            "revenue_estimate": event.revenue_estimate,
            "revenue_actual": event.revenue_actual,
            "hour": event.hour,
        },
    )


def from_macro_indicator(indicator) -> MarketSignal:
    """Convert a FRED MacroIndicator to a MarketSignal.

    Macro indicators modulate all other signals — they're regime context.
    A bullish insider signal during yield curve inversion is less reliable.
    """
    event_dt = _ensure_datetime(indicator.date)

    signal_id = f"fred:{indicator.series_id}:{indicator.date}"

    return MarketSignal(
        signal_id=signal_id,
        source="fred_macro",
        symbol="",  # Macro signals have no ticker
        asset_class="macro",
        domain="macro",
        direction=indicator.direction,
        strength=min(1.0, abs(indicator.value) / 10.0) if indicator.signal_type != "yield_curve"
                 else min(1.0, abs(indicator.value) / 3.0),
        confidence=indicator.confidence,
        decay_rate=indicator.decay_rate,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol="",
        raw_id=indicator.series_id,
        raw_type="MacroIndicator",
        metadata={
            "series_id": indicator.series_id,
            "series_name": indicator.series_name,
            "value": indicator.value,
            "signal_type": indicator.signal_type,
        },
    )


def from_price_data(data, move_threshold: float = 1.5) -> Optional[MarketSignal]:
    """Convert a PriceData object to a MarketSignal.

    Only produces a signal when the intraday move exceeds move_threshold%.
    Returns None for sub-threshold moves to avoid flooding with noise.

    Price signals are supporting context — they tell the lag-correlation
    analyzer and convergence alerter what the market actually DID, so it
    can learn which other signals predicted the move.

    Args:
        data: PriceData from price_fetcher.py
        move_threshold: Minimum |change_pct| to emit a signal (default 1.5%)

    Returns:
        MarketSignal or None if the move is below threshold
    """
    if abs(data.change_pct) < move_threshold:
        return None

    direction = "bullish" if data.change_pct > 0 else "bearish"
    # 5% move = full strength. 1.5% = 0.30. 3% = 0.60.
    strength = min(1.0, abs(data.change_pct) / 5.0)

    event_dt = _ensure_datetime(data.timestamp)

    symbol = data.symbol or ""
    signal_id = f"price_move:{symbol}:{data.timestamp}"

    return MarketSignal(
        signal_id=signal_id,
        source="yfinance_price",
        symbol=symbol,
        asset_class="stock",
        domain="price",
        direction=direction,
        strength=strength,
        confidence=0.50,
        decay_rate=0.15,
        timestamp=event_dt,
        received_at=datetime.now(),
        outcome_symbol=symbol,
        raw_id="",
        raw_type="PriceData",
        metadata={
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "close": data.price,
            "volume": data.volume,
            "change_pct": data.change_pct,
        },
    )
