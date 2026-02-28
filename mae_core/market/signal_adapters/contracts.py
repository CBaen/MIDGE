"""Contract signal adapters — government contracts, SAM.gov opportunities, predictions, hiring.

Converts raw contract and hiring data into normalized MarketSignal objects.
"""

from __future__ import annotations

from datetime import datetime

from mae_core.market.signal import MarketSignal, _ensure_datetime
from mae_core.market.apis.job_tracker import HiringSignal
from mae_core.market.apis.usa_spending import GovernmentContract
from mae_core.market.apis.sam_gov import ContractOpportunity
from mae_core.market.edge.contract_predictor import ContractPrediction
from mae_core.market.apis import ticker_resolver


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
