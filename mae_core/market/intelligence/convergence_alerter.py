#!/usr/bin/env python3
"""
Convergence Alerter - Generates alerts when patterns converge across domains.

CORE OUTPUT for trading decisions:
- Detects when multiple signals from different domains align
- "crypto whale + congress trade + hiring surge = actionable signal"
- Combines velocity, correlation, and confidence into single alert

Usage:
    from mae_core.market.intelligence.convergence_alerter import ConvergenceAlerter

    alerter = ConvergenceAlerter()
    alerter.record_signal("insider_buys", 0.8, "insider", direction="bullish")
    alerter.record_signal("crypto_whales", 0.7, "crypto", direction="bullish")

    alerts = alerter.check_convergence()
    for alert in alerts:
        print(alert.summary)

Implementation split (all under 500-line cap):
  convergence_models.py      — Signal, ConvergenceAlert dataclasses + read_discoveries
  convergence_confidence.py  — ConvergenceConfidenceMixin: Thompson weighting, coherence,
                               signal buffer persistence, record_signal, lag findings
  convergence_detection.py   — ConvergenceDetectionMixin: check_convergence,
                               ticker convergence, reporting, save/log
  convergence_alerter.py     — ConvergenceAlerter coordinator (this file, ~145 lines)
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from mae_core.market.intelligence.convergence_models import (  # noqa: F401
    Signal,
    ConvergenceAlert,
    read_discoveries,
)
from mae_core.market.intelligence.convergence_confidence import (
    ConvergenceConfidenceMixin,
    STRONG_CORRELATION_THRESHOLD,   # noqa: F401
    MODERATE_CORRELATION_THRESHOLD, # noqa: F401
)
from mae_core.market.intelligence.convergence_detection import ConvergenceDetectionMixin

# ConvergenceDetectionMixin inherits ConvergenceTickerMixin
# ConvergenceConfidenceMixin inherits ConvergenceLagScoringMixin + ConvergenceBufferMixin
# Full MRO: ConvergenceAlerter → ConvergenceConfidenceMixin → ConvergenceDetectionMixin
#         → ConvergenceLagScoringMixin → ConvergenceBufferMixin → ConvergenceTickerMixin

logger = logging.getLogger(__name__)


class ConvergenceAlerter(ConvergenceConfidenceMixin, ConvergenceDetectionMixin):
    """
    Detects convergence across domains and generates actionable alerts.

    Key insight: Single-domain signals are noisy. When 3+ domains
    from different categories all point the same direction, that's
    a much stronger signal.

    Example convergence:
    - Insider domain: Executives buying (bullish)
    - Crypto domain: Whales accumulating (bullish)
    - Government domain: Contract awarded (bullish)
    → CONVERGENCE ALERT: 3 domains bullish = high confidence

    Confidence is computed via Thompson-weighted geometric mean when
    a ThompsonSampler is provided. This replaces the original additive
    formula (Alpha's standing dissent — see deliverable.md).
    """

    # Sweep sources eligible for granular Thompson cascade (Bridge 2).
    _SWEEP_SOURCES = {"session_sweep", "session_sweep_ifvg"}

    # Maps MarketSignal.source values -> ThompsonSampler distribution keys.
    # signal.py uses descriptive source names, Thompson uses shorter keys
    # from learning_config.py source_reliability dict.
    _SOURCE_TO_THOMPSON_KEY = {
        "sec_form4": "sec_form4",
        "sec_form8k": "sec_form8k",
        "sec_efts": "sec_efts",
        "congressional": "congressional",
        "senate": "senate",
        "insider_cluster": "insider_cluster",
        "politician_correlation": "congressional",
        "contract_award": "contract_award",
        "contract_prediction": "contract_prediction",
        "hiring_tracker": "hiring_tracker",
        "sam_gov": "sam_gov",
        "social_sentiment": "social_sentiment",
        "finra_short": "finra_short",
        "finnhub_news": "finnhub_news",
        "finnhub_earnings": "finnhub_earnings",
        "fred_macro": "fred_macro",
        "session_sweep": "session_sweep",
        "session_sweep_ifvg": "session_sweep_ifvg",
        "ta_rsi": "ta_rsi",
        "ta_macd": "ta_macd",
        "ta_bollinger": "ta_bollinger",
        "ta_structure": "ta_structure",
        "ta_candle": "ta_candle",
        # Ten Gifts: Wave 1
        "order_flow": "order_flow",
        # Ten Gifts: Wave 2
        "fractal_resonance": "fractal_resonance",
        "archetype_match": "archetype_match",
        # New sources (Layer 6)
        "cot_positioning": "cot_positioning",
        "stocktwits_sentiment": "stocktwits_sentiment",
        "vix_term_structure": "vix_term_structure",
        "google_trends": "google_trends",
        "finnhub_economic": "finnhub_economic",
        "finnhub_analyst": "finnhub_analyst",
        "finnhub_earnings_calendar": "finnhub_earnings_calendar",
        # Real-economy: Energy
        "eia_energy": "eia_energy",
        # Real-economy: Legislative
        "congress_legislation": "congress_legislation",
        # Always-On Wave 2: Real-Time + Crypto
        "finnhub_realtime": "finnhub_realtime",
        "crypto_coingecko": "crypto_coingecko",
        "crypto_coincap": "crypto_coincap",
        # Always-On Wave 3: Data Enrichment
        "openinsider_purchase": "openinsider_purchase",
        "institutional_13f": "institutional_13f",
        "activist_13d": "activist_13d",
        "finviz_unusual_volume": "finviz_unusual_volume",
        "finviz_short_squeeze": "finviz_short_squeeze",
        "economic_calendar": "economic_calendar",
        # Massive/Polygon.io
        "massive_snapshot": "massive_snapshot",
        # Price data
        "yfinance_price": "yfinance_price",
        # Pattern discovery (motif + anomaly detectors)
        "motif_match": "motif_match",
        "price_discord": "price_discord",
        "streaming_anomaly": "streaming_anomaly",
    }

    # Domain → list of source names used in lag_correlations.json.
    # Enables domain-level correlation lookup from source-level correlation data.
    _DOMAIN_SOURCES: Dict[str, List[str]] = {
        "insider": ["sec_form4", "openinsider_purchase", "insider_cluster"],
        "events": ["sec_form8k", "sec_efts", "finnhub_earnings", "finnhub_news",
                   "finnhub_realtime", "finnhub_earnings_calendar", "massive_snapshot",
                   "hiring_tracker"],
        "macro": ["fred_macro", "economic_calendar"],
        "technical": ["ta_rsi", "ta_macd", "ta_bollinger", "ta_structure", "ta_candle",
                      "session_sweep", "session_sweep_ifvg", "fractal_resonance",
                      "order_flow", "finviz_unusual_volume", "finviz_short_squeeze",
                      "yfinance_price", "motif_match", "price_discord",
                      "streaming_anomaly"],
        "sentiment": ["social_sentiment", "google_trends", "stocktwits_sentiment"],
        "government": ["congressional", "senate", "congress_legislation"],
        "contracts": ["contract_award", "contract_prediction", "sam_gov"],
        "fundamentals": ["finnhub_analyst"],
        "positioning": ["cot_positioning"],
        "volatility": ["vix_term_structure"],
        "crypto": ["crypto_coingecko", "crypto_coincap"],
        "institutional": ["activist_13d", "institutional_13f", "finra_short"],
        "energy": ["eia_energy"],
    }

    def __init__(
        self,
        min_domains: int = 3,
        min_strength: float = 0.6,
        convergence_window_hours: int = 72,
        persistence_path: str = None,
        thompson_sampler=None,
        regime_classifier=None,
        causal_engine=None,
        event_bus=None,
        catalyst_calendar=None,
        cross_asset_confirmer=None,
        deception_detector=None,
        pattern_archetype_engine=None,
        economic_calendar=None,
        correlation_tracker=None,
        world_model=None,
        quorum_space=None,
    ):
        self.min_domains = min_domains
        self.min_strength = min_strength
        self.convergence_window = timedelta(hours=convergence_window_hours)
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self._thompson = thompson_sampler
        self._regime_classifier = regime_classifier
        self._causal_engine = causal_engine
        self._world_model = world_model
        self._bus = event_bus
        self._catalyst_calendar = catalyst_calendar
        self._cross_asset_confirmer = cross_asset_confirmer
        self._deception_detector = deception_detector
        self._pattern_archetype_engine = pattern_archetype_engine
        self._economic_calendar = economic_calendar
        self._correlation_tracker = correlation_tracker
        self._quorum_space = quorum_space
        self._cached_regime = ("default", 0.0)  # (regime_str, timestamp)

        # Per-domain convergence windows — slow-moving data sources need longer lookback.
        # Domains NOT listed here fall back to self.convergence_window (72h global default).
        self._domain_windows: Dict[str, timedelta] = {
            "positioning": timedelta(hours=14 * 24),  # 14 days — COT is weekly data
            "government": timedelta(hours=7 * 24),    # 7 days — congressional trades
            "contracts": timedelta(hours=7 * 24),     # 7 days — contract awards
            "energy": timedelta(hours=7 * 24),        # 7 days — weekly EIA reports
        }

        # Recent signals by domain
        self.signals: Dict[str, list] = defaultdict(list)

        # Alert history
        self.alerts: List[ConvergenceAlert] = []

        # Domain categories for cross-domain verification
        self.domain_categories = {
            "insider": "behavioral",
            "congress": "behavioral",
            "crypto": "market",
            "technical": "market",
            "volume": "market",
            "sentiment": "social",
            "reddit": "social",
            "news": "information",
            "events": "information",
            "fundamentals": "financial",
            "macro": "financial",
            "government": "institutional",
            "contracts": "institutional",
            "institutional_synthesis": "institutional",
            # New domains (Layer 6)
            "positioning": "institutional",
            "volatility": "market",
            # Real-economy
            "energy": "fundamental",
            # Causal chain — synthetic signals from confirmed cascade dominoes (Law 2)
            "cascade": "causal",
        }

        self._alert_counter = 0

        # Alert deduplication (defense in depth — step hook also deduplicates)
        # Per-direction tracking: both bullish and bearish can be suppressed
        # independently, preventing the cross-evasion bug where each direction
        # kept overwriting the other's suppression record.
        self._last_alert_times: Dict[str, datetime] = {}
        self._alert_lock = threading.RLock()
        self._min_alert_interval_hours = 4.0

        # Lag findings for temporal ordering (Task 1 + Task 3).
        # Set via set_lag_findings() after LagCorrelationAnalyzer runs.
        # Loaded from disk on startup so alerter has data from prior runs.
        self._lag_findings: List = []
        self._load_lag_findings()
